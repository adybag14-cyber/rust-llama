use crate::{Error, Result};
#[derive(Debug, Clone, Default)]
struct Node {
    edges: Vec<(u8, usize)>,
    value: Option<i32>,
}
#[derive(Debug, Clone)]
pub(super) struct Trie {
    nodes: Vec<Node>,
}
impl Trie {
    pub(super) fn new() -> Self {
        Self {
            nodes: vec![Node::default()],
        }
    }
    pub(super) fn insert(&mut self, bytes: &[u8], value: i32) -> Result<()> {
        let mut index = 0;
        for &byte in bytes {
            match self.nodes[index]
                .edges
                .binary_search_by_key(&byte, |edge| edge.0)
            {
                Ok(slot) => index = self.nodes[index].edges[slot].1,
                Err(slot) => {
                    if self.nodes.len() >= 8 * 1024 * 1024 {
                        return Err(Error::Limit("token trie exceeds 8 million nodes".into()));
                    }
                    let next = self.nodes.len();
                    self.nodes.push(Node::default());
                    self.nodes[index].edges.insert(slot, (byte, next));
                    index = next;
                }
            }
        }
        self.nodes[index].value = Some(value);
        Ok(())
    }
    pub(super) fn next(&self, index: usize, byte: u8) -> Option<usize> {
        let edges = &self.nodes[index].edges;
        edges
            .binary_search_by_key(&byte, |edge| edge.0)
            .ok()
            .map(|slot| edges[slot].1)
    }
    pub(super) fn value(&self, index: usize) -> Option<i32> {
        self.nodes[index].value
    }
    pub(super) fn longest_path(&self, bytes: &[u8]) -> usize {
        let mut index = 0;
        for (offset, &byte) in bytes.iter().enumerate() {
            match self.next(index, byte) {
                Some(next) => index = next,
                None => return offset,
            }
        }
        bytes.len()
    }
}
