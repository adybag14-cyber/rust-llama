//! Bounded queues and replay rings. Replay offsets count raw UTF-8 bytes.
use std::collections::{BTreeMap, VecDeque};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

#[derive(Clone, Default, Debug)]
pub struct CancellationToken(Arc<AtomicBool>);
impl CancellationToken {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn cancel(&self) {
        self.0.store(true, Ordering::Release);
    }
    pub fn is_cancelled(&self) -> bool {
        self.0.load(Ordering::Acquire)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum QueueError {
    Cancelled,
    Closed,
    Timeout,
    ChunkTooLarge,
}
#[derive(Debug, PartialEq, Eq)]
pub enum QueueRead {
    Data(Vec<u8>),
    Timeout,
    Closed,
}

#[derive(Default)]
struct QueueState {
    chunks: VecDeque<Vec<u8>>,
    bytes: usize,
    high_water: usize,
    closed: bool,
}
struct QueueInner {
    state: Mutex<QueueState>,
    changed: Condvar,
    capacity: usize,
}

/// A byte-bounded MPMC queue with cooperative cancellation and bounded waits.
#[derive(Clone)]
pub struct ByteQueue(Arc<QueueInner>);
impl ByteQueue {
    pub fn new(capacity: usize) -> Result<Self, &'static str> {
        if capacity == 0 {
            return Err("Queue capacity must be positive");
        }
        Ok(Self(Arc::new(QueueInner {
            state: Mutex::new(QueueState::default()),
            changed: Condvar::new(),
            capacity,
        })))
    }
    pub fn send(
        &self,
        chunk: &[u8],
        cancel: &CancellationToken,
        timeout: Duration,
    ) -> Result<(), QueueError> {
        if chunk.len() > self.0.capacity {
            return Err(QueueError::ChunkTooLarge);
        }
        if chunk.is_empty() {
            return Ok(());
        }
        let started = Instant::now();
        let mut state = self.0.state.lock().unwrap();
        loop {
            if cancel.is_cancelled() {
                return Err(QueueError::Cancelled);
            }
            if state.closed {
                return Err(QueueError::Closed);
            }
            if chunk.len() <= self.0.capacity - state.bytes {
                state.bytes += chunk.len();
                state.high_water = state.high_water.max(state.bytes);
                state.chunks.push_back(chunk.to_vec());
                self.0.changed.notify_all();
                return Ok(());
            }
            let remaining = timeout.saturating_sub(started.elapsed());
            if remaining.is_zero() {
                return Err(QueueError::Timeout);
            }
            (state, _) = self
                .0
                .changed
                .wait_timeout(state, remaining.min(Duration::from_millis(25)))
                .unwrap();
        }
    }
    pub fn recv(&self, timeout: Duration) -> QueueRead {
        let started = Instant::now();
        let mut state = self.0.state.lock().unwrap();
        loop {
            if let Some(chunk) = state.chunks.pop_front() {
                state.bytes -= chunk.len();
                self.0.changed.notify_all();
                return QueueRead::Data(chunk);
            }
            if state.closed {
                return QueueRead::Closed;
            }
            let remaining = timeout.saturating_sub(started.elapsed());
            if remaining.is_zero() {
                return QueueRead::Timeout;
            }
            (state, _) = self.0.changed.wait_timeout(state, remaining).unwrap();
        }
    }
    /// Close after draining already queued data.
    pub fn finish(&self) {
        self.0.state.lock().unwrap().closed = true;
        self.0.changed.notify_all();
    }
    /// Discard queued bytes and wake blocked producers on client disconnect.
    pub fn abort(&self) {
        let mut state = self.0.state.lock().unwrap();
        state.closed = true;
        state.bytes = 0;
        state.chunks.clear();
        self.0.changed.notify_all();
    }
    pub fn high_water_bytes(&self) -> usize {
        self.0.state.lock().unwrap().high_water
    }
    pub fn capacity(&self) -> usize {
        self.0.capacity
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ReplayError {
    OffsetLost { oldest: u64 },
    Closed,
    Full,
    InvalidId,
    OffsetOverflow,
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ReplayRead {
    Data(Vec<u8>),
    Timeout,
    Done,
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ReplayInfo {
    pub conversation_id: String,
    pub is_done: bool,
    pub total_bytes: u64,
    pub oldest_byte: u64,
    pub started_at: u64,
    pub completed_at: Option<u64>,
}
struct ReplayState {
    bytes: VecDeque<u8>,
    oldest: u64,
    total: u64,
    done: bool,
    completed_at: Option<u64>,
    completed: Option<Instant>,
}
pub struct ReplaySession {
    id: String,
    capacity: usize,
    started_at: u64,
    state: Mutex<ReplayState>,
    changed: Condvar,
    cancel: CancellationToken,
}
fn epoch_seconds() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}
impl ReplaySession {
    fn new(id: String, capacity: usize) -> Self {
        Self {
            id,
            capacity,
            started_at: epoch_seconds(),
            state: Mutex::new(ReplayState {
                bytes: VecDeque::new(),
                oldest: 0,
                total: 0,
                done: false,
                completed_at: None,
                completed: None,
            }),
            changed: Condvar::new(),
            cancel: CancellationToken::new(),
        }
    }
    pub fn cancellation(&self) -> CancellationToken {
        self.cancel.clone()
    }
    pub fn append(&self, bytes: &[u8]) -> Result<(), ReplayError> {
        let mut state = self.state.lock().unwrap();
        if state.done || self.cancel.is_cancelled() {
            return Err(ReplayError::Closed);
        }
        let total = state
            .total
            .checked_add(bytes.len() as u64)
            .ok_or(ReplayError::OffsetOverflow)?;
        if bytes.len() >= self.capacity {
            state.bytes.clear();
            state.bytes.extend(&bytes[bytes.len() - self.capacity..]);
        } else {
            let evict = state
                .bytes
                .len()
                .saturating_add(bytes.len())
                .saturating_sub(self.capacity);
            state.bytes.drain(..evict);
            state.bytes.extend(bytes);
        }
        state.total = total;
        state.oldest = total - state.bytes.len() as u64;
        self.changed.notify_all();
        Ok(())
    }
    pub fn read_from(
        &self,
        offset: u64,
        max_bytes: usize,
        timeout: Duration,
    ) -> Result<ReplayRead, ReplayError> {
        if max_bytes == 0 {
            return Ok(ReplayRead::Timeout);
        }
        let started = Instant::now();
        let mut state = self.state.lock().unwrap();
        loop {
            if offset < state.oldest {
                return Err(ReplayError::OffsetLost {
                    oldest: state.oldest,
                });
            }
            if offset < state.total {
                let start = (offset - state.oldest) as usize;
                let bytes = state
                    .bytes
                    .iter()
                    .skip(start)
                    .take(max_bytes)
                    .copied()
                    .collect();
                return Ok(ReplayRead::Data(bytes));
            }
            if state.done {
                return Ok(ReplayRead::Done);
            }
            let remaining = timeout.saturating_sub(started.elapsed());
            if remaining.is_zero() {
                return Ok(ReplayRead::Timeout);
            }
            (state, _) = self.changed.wait_timeout(state, remaining).unwrap();
        }
    }
    pub fn finish(&self) {
        let mut state = self.state.lock().unwrap();
        if !state.done {
            state.done = true;
            state.completed_at = Some(epoch_seconds());
            state.completed = Some(Instant::now());
        }
        self.changed.notify_all();
    }
    pub fn cancel(&self) {
        self.cancel.cancel();
        self.finish();
    }
    pub fn info(&self) -> ReplayInfo {
        let state = self.state.lock().unwrap();
        ReplayInfo {
            conversation_id: self.id.clone(),
            is_done: state.done,
            total_bytes: state.total,
            oldest_byte: state.oldest,
            started_at: self.started_at,
            completed_at: state.completed_at,
        }
    }
    fn expired(&self, ttl: Duration) -> bool {
        self.state
            .lock()
            .unwrap()
            .completed
            .is_some_and(|completed| completed.elapsed() >= ttl)
    }
}

pub struct ReplayStore {
    sessions: Mutex<BTreeMap<String, Arc<ReplaySession>>>,
    capacity: usize,
    max_sessions: usize,
    ttl: Duration,
}
impl ReplayStore {
    pub fn new(capacity: usize, max_sessions: usize, ttl: Duration) -> Result<Self, &'static str> {
        if capacity == 0 || max_sessions == 0 {
            return Err("Replay limits must be positive");
        }
        Ok(Self {
            sessions: Mutex::new(BTreeMap::new()),
            capacity,
            max_sessions,
            ttl,
        })
    }
    pub fn create(&self, id: &str) -> Result<Arc<ReplaySession>, ReplayError> {
        if id.is_empty() || id.len() > 256 || id.chars().any(char::is_control) {
            return Err(ReplayError::InvalidId);
        }
        let mut sessions = self.sessions.lock().unwrap();
        sessions.retain(|_, session| !session.expired(self.ttl));
        if !sessions.contains_key(id) && sessions.len() >= self.max_sessions {
            return Err(ReplayError::Full);
        }
        if let Some(old) = sessions.remove(id) {
            old.cancel();
        }
        let session = Arc::new(ReplaySession::new(id.to_owned(), self.capacity));
        sessions.insert(id.to_owned(), session.clone());
        Ok(session)
    }
    pub fn get(&self, id: &str) -> Option<Arc<ReplaySession>> {
        let mut sessions = self.sessions.lock().unwrap();
        sessions.retain(|_, session| !session.expired(self.ttl));
        sessions.get(id).cloned()
    }
    pub fn lookup(&self, ids: &[String]) -> Vec<ReplayInfo> {
        let mut sessions = self.sessions.lock().unwrap();
        sessions.retain(|_, session| !session.expired(self.ttl));
        sessions
            .iter()
            .filter(|(key, _)| {
                ids.iter().any(|id| {
                    !id.is_empty()
                        && (*key == id
                            || key
                                .strip_prefix(id)
                                .is_some_and(|suffix| suffix.starts_with("::")))
                })
            })
            .map(|(_, session)| session.info())
            .collect()
    }
    pub fn delete(&self, id: &str) {
        if let Some(session) = self.sessions.lock().unwrap().remove(id) {
            session.cancel();
        }
    }
    pub fn cancel_all(&self) {
        let mut sessions = self.sessions.lock().unwrap();
        for session in sessions.values() {
            session.cancel();
        }
        sessions.clear();
    }
}

/// An SSE data event. Newlines are individually prefixed as required by SSE.
pub fn sse_data(data: &str) -> Vec<u8> {
    let mut result = String::new();
    let normalized = data.replace("\r\n", "\n").replace('\r', "\n");
    for line in normalized.split('\n') {
        result.push_str("data: ");
        result.push_str(line);
        result.push('\n');
    }
    result.push('\n');
    result.into_bytes()
}

#[derive(Default, Debug)]
pub struct Utf8Decoder {
    pending: Vec<u8>,
    failed: bool,
}
impl Utf8Decoder {
    pub fn push(&mut self, bytes: &[u8]) -> Result<String, &'static str> {
        if self.failed {
            return Err("UTF-8 decoder is poisoned after invalid input");
        }
        let mut all = std::mem::take(&mut self.pending);
        all.extend_from_slice(bytes);
        match std::str::from_utf8(&all) {
            Ok(text) => Ok(text.to_owned()),
            Err(error) if error.error_len().is_none() => {
                let valid = error.valid_up_to();
                self.pending.extend_from_slice(&all[valid..]);
                Ok(std::str::from_utf8(&all[..valid]).unwrap().to_owned())
            }
            Err(_) => {
                self.failed = true;
                Err("Stream contains invalid UTF-8")
            }
        }
    }
    pub fn finish(&self) -> Result<(), &'static str> {
        if self.failed {
            Err("UTF-8 decoder is poisoned after invalid input")
        } else if self.pending.is_empty() {
            Ok(())
        } else {
            Err("Stream ended inside a UTF-8 code point")
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;
    #[test]
    fn utf8_decoder_handles_every_byte_boundary() {
        let text = "a€🦀 日本語\n";
        for split in 0..=text.len() {
            let mut decoder = Utf8Decoder::default();
            let mut output = decoder.push(&text.as_bytes()[..split]).unwrap();
            output += &decoder.push(&text.as_bytes()[split..]).unwrap();
            decoder.finish().unwrap();
            assert_eq!(output, text);
        }
        let mut decoder = Utf8Decoder::default();
        decoder.push(&[0xF0]).unwrap();
        assert!(decoder.finish().is_err());
        assert!(Utf8Decoder::default().push(&[0xFF]).is_err());
        let mut decoder = Utf8Decoder::default();
        decoder.push(&[0xE2]).unwrap();
        assert!(decoder.push(b"X").is_err());
        assert!(decoder.finish().is_err());
        assert!(decoder.push(b"valid").is_err());
    }
    #[test]
    fn concurrent_queue_never_exceeds_byte_budget() {
        let queue = ByteQueue::new(31).unwrap();
        let cancel = CancellationToken::new();
        let mut workers = Vec::new();
        for _ in 0..4 {
            let queue = queue.clone();
            let cancel = cancel.clone();
            workers.push(thread::spawn(move || {
                for _ in 0..40 {
                    queue
                        .send(&[1; 7], &cancel, Duration::from_secs(2))
                        .unwrap();
                }
            }));
        }
        let mut received = 0;
        for _ in 0..160 {
            match queue.recv(Duration::from_secs(2)) {
                QueueRead::Data(data) => received += data.len(),
                other => panic!("{other:?}"),
            }
        }
        for worker in workers {
            worker.join().unwrap();
        }
        assert_eq!(received, 1120);
        assert!(queue.high_water_bytes() <= 31);
    }
    #[test]
    fn blocked_sender_observes_cancel_and_disconnect() {
        let queue = ByteQueue::new(1).unwrap();
        let cancel = CancellationToken::new();
        queue.send(b"a", &cancel, Duration::ZERO).unwrap();
        let (worker_queue, worker_cancel) = (queue.clone(), cancel.clone());
        let worker =
            thread::spawn(move || worker_queue.send(b"b", &worker_cancel, Duration::from_secs(2)));
        cancel.cancel();
        assert_eq!(worker.join().unwrap(), Err(QueueError::Cancelled));
        queue.abort();
        assert_eq!(queue.recv(Duration::ZERO), QueueRead::Closed);
        assert_eq!(
            queue.send(b"a", &CancellationToken::new(), Duration::ZERO),
            Err(QueueError::Closed)
        );
    }
    #[test]
    fn queue_timeout_and_oversize_are_explicit() {
        let queue = ByteQueue::new(1).unwrap();
        let cancel = CancellationToken::new();
        assert_eq!(
            queue.send(b"ab", &cancel, Duration::ZERO),
            Err(QueueError::ChunkTooLarge)
        );
        queue.send(b"a", &cancel, Duration::ZERO).unwrap();
        assert_eq!(
            queue.send(b"b", &cancel, Duration::ZERO),
            Err(QueueError::Timeout)
        );
        queue.finish();
        assert_eq!(queue.recv(Duration::ZERO), QueueRead::Data(b"a".to_vec()));
        assert_eq!(queue.recv(Duration::ZERO), QueueRead::Closed);
    }
    #[test]
    fn replay_overflow_preserves_raw_suffix_and_absolute_offsets() {
        let store = ReplayStore::new(5, 2, Duration::from_secs(300)).unwrap();
        let session = store.create("chat").unwrap();
        session.append("a€xy".as_bytes()).unwrap();
        assert_eq!(session.info().oldest_byte, 1);
        assert_eq!(
            session.read_from(0, 10, Duration::ZERO),
            Err(ReplayError::OffsetLost { oldest: 1 })
        );
        assert_eq!(
            session.read_from(2, 10, Duration::ZERO).unwrap(),
            ReplayRead::Data(vec![0x82, 0xAC, b'x', b'y'])
        );
        session.append(b"0123456789").unwrap();
        assert_eq!(session.info().total_bytes, 16);
        assert_eq!(
            session.read_from(11, 9, Duration::ZERO).unwrap(),
            ReplayRead::Data(b"56789".to_vec())
        );
    }
    #[test]
    fn replay_replacement_lookup_cancel_and_future_offset() {
        let store = ReplayStore::new(32, 2, Duration::from_secs(300)).unwrap();
        let old = store.create("chat::1").unwrap();
        let new = store.create("chat::1").unwrap();
        assert!(old.cancellation().is_cancelled());
        assert!(old.info().is_done);
        let second = store.create("chat::2").unwrap();
        assert!(matches!(store.create("third"), Err(ReplayError::Full)));
        assert_eq!(store.lookup(&["chat".into(), "chat::1".into()]).len(), 2);
        assert_eq!(
            new.read_from(100, 4, Duration::ZERO).unwrap(),
            ReplayRead::Timeout
        );
        new.finish();
        assert_eq!(
            new.read_from(100, 4, Duration::ZERO).unwrap(),
            ReplayRead::Done
        );
        store.delete("chat::2");
        assert!(second.cancellation().is_cancelled());
        assert!(store.get("chat::2").is_none());
    }
    #[test]
    fn completed_replay_expires_but_active_does_not() {
        let store = ReplayStore::new(8, 2, Duration::ZERO).unwrap();
        let active = store.create("active").unwrap();
        let finished = store.create("done").unwrap();
        finished.finish();
        assert!(store.get("done").is_none());
        assert!(store.get("active").is_some());
        active.append(b"a").unwrap();
    }
    #[test]
    fn sse_multiline_framing() {
        assert_eq!(sse_data("a\nb"), b"data: a\ndata: b\n\n");
        assert_eq!(
            sse_data("a\revent: evil\r\nb"),
            b"data: a\ndata: event: evil\ndata: b\n\n"
        );
    }
}
