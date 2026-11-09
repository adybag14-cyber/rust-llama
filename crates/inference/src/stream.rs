use crate::{Error, Result};

pub(crate) struct StopFilter {
    stops: Vec<Vec<u8>>,
    pending: Vec<u8>,
}
impl StopFilter {
    pub fn new(stops: &[String]) -> Self {
        Self {
            stops: stops.iter().map(|s| s.as_bytes().to_vec()).collect(),
            pending: Vec::new(),
        }
    }
    pub fn push(&mut self, bytes: &[u8]) -> (Vec<u8>, bool) {
        self.pending.extend_from_slice(bytes);
        let found = self
            .stops
            .iter()
            .filter_map(|s| self.pending.windows(s.len()).position(|w| w == s))
            .min();
        if let Some(i) = found {
            let safe = self.pending[..i].to_vec();
            self.pending.clear();
            return (safe, true);
        }
        let retain = self
            .stops
            .iter()
            .map(|s| {
                let max = self.pending.len().min(s.len().saturating_sub(1));
                (1..=max)
                    .rev()
                    .find(|&n| self.pending.ends_with(&s[..n]))
                    .unwrap_or(0)
            })
            .max()
            .unwrap_or(0);
        let safe = self.pending.drain(..self.pending.len() - retain).collect();
        (safe, false)
    }
    pub fn finish(&mut self) -> Vec<u8> {
        std::mem::take(&mut self.pending)
    }
}
#[derive(Default)]
pub(crate) struct Utf8Stream {
    pending: Vec<u8>,
}
impl Utf8Stream {
    pub fn push(&mut self, bytes: &[u8], emit: &mut dyn FnMut(&str) -> bool) -> Result<()> {
        self.pending.extend_from_slice(bytes);
        loop {
            match std::str::from_utf8(&self.pending) {
                Ok(s) => {
                    if !s.is_empty() && !emit(s) {
                        return Err(Error::Cancelled);
                    }
                    self.pending.clear();
                    return Ok(());
                }
                Err(e) => {
                    let valid = e.valid_up_to();
                    let length = e.error_len();
                    if valid > 0 {
                        let text = std::str::from_utf8(&self.pending[..valid])
                            .map_err(|_| Error::Internal("UTF-8 valid prefix".into()))?;
                        if !emit(text) {
                            return Err(Error::Cancelled);
                        }
                        self.pending.drain(..valid);
                    }
                    match length {
                        Some(n) => {
                            if !emit("�") {
                                return Err(Error::Cancelled);
                            }
                            self.pending.drain(..n);
                        }
                        None => return Ok(()),
                    }
                }
            }
        }
    }
    pub fn finish(&mut self, emit: &mut dyn FnMut(&str) -> bool) -> Result<()> {
        if !self.pending.is_empty() {
            if !emit("�") {
                return Err(Error::Cancelled);
            }
            self.pending.clear();
        }
        Ok(())
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn stop_spans_tokens_without_leaking_prefix() {
        let mut f = StopFilter::new(&["END".into(), "ENDING".into()]);
        assert_eq!(f.push(b"hello E"), (b"hello ".to_vec(), false));
        assert_eq!(f.push(b"N"), (vec![], false));
        assert_eq!(f.push(b"D rest"), (vec![], true));
    }
    #[test]
    fn unmatched_stop_prefix_is_flushed() {
        let mut f = StopFilter::new(&["END".into()]);
        assert_eq!(f.push(b"aEN"), (b"a".to_vec(), false));
        assert_eq!(f.finish(), b"EN");
    }
    #[test]
    fn utf8_is_preserved_across_every_byte_boundary() {
        let mut u = Utf8Stream::default();
        let mut out = String::new();
        for b in "A£🦀界".as_bytes() {
            u.push(&[*b], &mut |s| {
                out.push_str(s);
                true
            })
            .unwrap()
        }
        u.finish(&mut |s| {
            out.push_str(s);
            true
        })
        .unwrap();
        assert_eq!(out, "A£🦀界");
    }
    #[test]
    fn invalid_utf8_and_tail_are_replaced_once() {
        let mut u = Utf8Stream::default();
        let mut out = String::new();
        u.push(&[b'a', 0xff, 0xe2, 0x82], &mut |s| {
            out.push_str(s);
            true
        })
        .unwrap();
        u.finish(&mut |s| {
            out.push_str(s);
            true
        })
        .unwrap();
        assert_eq!(out, "a��");
    }
    #[test]
    fn sink_failure_propagates() {
        let mut u = Utf8Stream::default();
        assert!(matches!(
            u.push(b"hello", &mut |_| false),
            Err(Error::Cancelled)
        ));
    }
}
