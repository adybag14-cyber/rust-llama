//! Sampler-side speculative transactions and greedy prefix verification.
//! Model KV/recurrent-state rollback belongs to T05 and is not emulated here.

use crate::sampling::{Chain, TokenId};
use crate::{Error, Result};

pub const MAX_DRAFT_TOKENS: usize = 4096;
pub const MAX_SNAPSHOT_BYTES: usize = 64 * 1024 * 1024;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Acceptance {
    pub accepted_draft: usize,
    pub rejected_draft: usize,
    /// Accepted draft prefix followed by the mismatch replacement or bonus.
    pub emitted: Vec<TokenId>,
}

/// `target[i]` must be evaluated after the same prefix as `draft[i]`; the
/// final target entry is the bonus prediction after all draft tokens.
pub fn verify_greedy(draft: &[TokenId], target: &[TokenId]) -> Result<Acceptance> {
    if draft.len() > MAX_DRAFT_TOKENS {
        return Err(Error::new("speculative draft exceeds limit"));
    }
    if target.len() != draft.len() + 1 {
        return Err(Error::new(
            "target predictions must include one entry per draft token plus the bonus",
        ));
    }
    if draft.iter().chain(target).any(|t| *t < 0) {
        return Err(Error::new("negative token in speculative verification"));
    }
    let accepted = draft.iter().zip(target).take_while(|(a, b)| a == b).count();
    let mut emitted = draft[..accepted].to_vec();
    emitted.push(target[accepted]);
    Ok(Acceptance {
        accepted_draft: accepted,
        rejected_draft: draft.len() - accepted,
        emitted,
    })
}

/// A draft consumes only its private cloned sampler. Dropping it cancels all
/// proposals; committing a prefix adopts the exact sampler/RNG snapshot after
/// that prefix, so rejected proposals cannot advance the live RNG.
pub struct Transaction<'a> {
    live: &'a mut Chain,
    working: Chain,
    snapshots: Vec<Chain>,
    tokens: Vec<TokenId>,
    limit: usize,
    snapshot_bytes: usize,
    byte_budget: usize,
}
impl<'a> Transaction<'a> {
    pub fn new(live: &'a mut Chain, limit: usize) -> Result<Self> {
        Self::with_byte_budget(live, limit, MAX_SNAPSHOT_BYTES)
    }
    /// A smaller budget is useful for constrained runtimes. The conservative
    /// budget covers stored sampler states and headroom for transient clones;
    /// caller-owned logits and sampler application scratch are separate.
    pub fn with_byte_budget(live: &'a mut Chain, limit: usize, byte_budget: usize) -> Result<Self> {
        if limit == 0 || limit > MAX_DRAFT_TOKENS {
            return Err(Error::new("invalid speculative transaction limit"));
        }
        if byte_budget == 0 || byte_budget > MAX_SNAPSHOT_BYTES {
            return Err(Error::new(
                "speculative state budget must be within 1..=64 MiB",
            ));
        }
        let initial = live.snapshot_size_bound()?;
        if initial
            .checked_mul(4)
            .and_then(|n| n.checked_add(256))
            .is_none_or(|n| n > byte_budget)
        {
            return Err(Error::new(
                "sampler state exceeds speculative byte budget before cloning",
            ));
        }
        let working = live.clone();
        Ok(Self {
            live,
            working,
            snapshots: Vec::new(),
            tokens: Vec::new(),
            limit,
            snapshot_bytes: 0,
            byte_budget,
        })
    }
    pub fn tokens(&self) -> &[TokenId] {
        &self.tokens
    }
    pub fn propose(&mut self, logits: &[f32]) -> Result<TokenId> {
        if self.tokens.len() == self.limit {
            return Err(Error::new("speculative transaction is full"));
        }
        // One accept can grow a history deque and add one frequency-map key
        // per stage. Doubling the current bound plus 1024 bytes/stage covers
        // that growth; five copies cover nested transactional apply clones.
        let current = self.working.snapshot_size_bound()?;
        let next_bound = current
            .checked_mul(2)
            .and_then(|n| n.checked_add(self.working.samplers().len() * 1024))
            .ok_or_else(|| Error::new("speculative state size overflow"))?;
        let peak = next_bound
            .checked_mul(5)
            .and_then(|n| n.checked_add(current))
            .and_then(|n| n.checked_add(self.snapshot_bytes))
            .and_then(|n| n.checked_add(128))
            .ok_or_else(|| Error::new("speculative state size overflow"))?;
        if peak > self.byte_budget {
            return Err(Error::new(
                "speculative snapshot byte budget exhausted before cloning",
            ));
        }
        let mut staged = self.working.clone();
        let token = staged.sample_and_accept(logits)?;
        let actual = staged.snapshot_size_bound()?;
        if actual > next_bound {
            return Err(Error::new(
                "sampler state grew beyond conservative snapshot bound",
            ));
        }
        self.snapshot_bytes = self
            .snapshot_bytes
            .checked_add(actual)
            .and_then(|n| n.checked_add(128))
            .ok_or_else(|| Error::new("speculative state size overflow"))?;
        self.tokens.push(token);
        self.working = staged.clone();
        self.snapshots.push(staged);
        Ok(token)
    }
    pub fn commit_prefix(mut self, accepted: usize) -> Result<Vec<TokenId>> {
        if accepted > self.tokens.len() {
            return Err(Error::new("accepted prefix exceeds proposed tokens"));
        }
        if accepted > 0 {
            *self.live = self.snapshots.swap_remove(accepted - 1);
        }
        self.tokens.truncate(accepted);
        Ok(self.tokens)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sampling::Config;
    #[test]
    fn greedy_rejection_and_bonus() {
        assert_eq!(
            verify_greedy(&[1, 2, 3], &[1, 2, 9, 4]).unwrap(),
            Acceptance {
                accepted_draft: 2,
                rejected_draft: 1,
                emitted: vec![1, 2, 9]
            }
        );
        assert_eq!(
            verify_greedy(&[1, 2], &[1, 2, 3]).unwrap().emitted,
            vec![1, 2, 3]
        );
        assert_eq!(verify_greedy(&[], &[7]).unwrap().emitted, vec![7]);
        assert!(verify_greedy(&[1], &[1]).is_err());
    }
    #[test]
    fn partial_commit_restores_rng_and_history_exactly() {
        let config = [
            Config::Penalties {
                last_n: 8,
                repeat: 1.1,
                frequency: 0.2,
                presence: 0.1,
            },
            Config::Dist { seed: 42 },
        ];
        let mut live = Chain::new(config.clone()).unwrap();
        let mut serial = Chain::new(config).unwrap();
        let logits = [0.0, 1.0, 2.0, 3.0];
        let mut txn = Transaction::new(&mut live, 4).unwrap();
        for _ in 0..4 {
            txn.propose(&logits).unwrap();
        }
        let committed = txn.commit_prefix(2).unwrap();
        assert_eq!(
            committed,
            vec![
                serial.sample_and_accept(&logits).unwrap(),
                serial.sample_and_accept(&logits).unwrap()
            ]
        );
        assert_eq!(live, serial);
        for _ in 0..20 {
            assert_eq!(
                live.sample_and_accept(&logits).unwrap(),
                serial.sample_and_accept(&logits).unwrap()
            );
        }
    }
    #[test]
    fn dropping_or_invalid_commit_leaves_live_state_untouched() {
        let mut live = Chain::new([Config::Dist { seed: 5 }]).unwrap();
        let before = live.clone();
        {
            let mut txn = Transaction::new(&mut live, 2).unwrap();
            txn.propose(&[1.0, 2.0]).unwrap();
        }
        assert_eq!(live, before);
        let txn = Transaction::new(&mut live, 2).unwrap();
        assert!(txn.commit_prefix(1).is_err());
        assert_eq!(live, before);
    }
    #[test]
    fn snapshot_budget_rejects_before_unbounded_chain_copying() {
        let mut live = Chain::new([Config::Dist { seed: 1 }, Config::Dist { seed: 2 }]).unwrap();
        let before = live.clone();
        assert!(Transaction::with_byte_budget(&mut live, 4096, 1024).is_err());
        {
            let mut txn = Transaction::with_byte_budget(&mut live, 4096, 256 * 1024).unwrap();
            let mut rejected = false;
            for _ in 0..100 {
                let old = txn.tokens().to_vec();
                if txn.propose(&[0.0, 1.0]).is_err() {
                    assert_eq!(txn.tokens(), old);
                    rejected = true;
                    break;
                }
            }
            assert!(
                rejected,
                "snapshot budget should stop well before the token cap"
            );
        }
        assert_eq!(live, before);
    }
}
