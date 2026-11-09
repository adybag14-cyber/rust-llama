//! Bounded byte ECMAScript matching for the frozen C grammar trigger contract.
//! Byte classes, capture order, lookahead and identity escapes follow the frozen
//! MSVC C-locale target; complexity/allocation limits report checked errors.
use crate::{Error, Result};
use std::ops::Range;
use std::sync::Arc;
const MAX_PATTERN: usize = 64 * 1024;
const MAX_NODES: usize = 16 * 1024;
const MAX_GROUPS: usize = 128;
const MAX_DEPTH: usize = 128;
const MAX_INPUT: usize = 1024 * 1024;
const MAX_STEPS: usize = 5_000_000;
const MAX_STACK_BYTES: usize = 64 * 1024 * 1024;
#[derive(Debug, Clone)]
enum Kind {
    Empty,
    Class(Box<[bool; 256]>),
    Seq(Vec<usize>),
    Alt(Vec<usize>),
    Group(usize, usize),
    Repeat {
        child: usize,
        min: usize,
        max: Option<usize>,
        greedy: bool,
    },
    Start,
    End,
    Boundary(bool),
    Backref(usize),
    Look(usize, bool),
}
#[derive(Debug, Clone)]
struct Node {
    kind: Kind,
    captures: Range<usize>,
}
#[derive(Debug)]
struct Program {
    nodes: Vec<Node>,
    root: usize,
    groups: usize,
}
#[derive(Debug, Clone)]
pub struct NativeRegex {
    program: Arc<Program>,
}
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NativeMatch {
    pub range: Range<usize>,
    /// Capture one is at index zero, matching the frozen trigger's scan order.
    pub captures: Vec<Option<Range<usize>>>,
}
struct Parser<'a> {
    pattern: &'a [u8],
    pos: usize,
    nodes: Vec<Node>,
    finished: Vec<bool>,
}
enum Atom {
    Byte(u8),
    Set(Box<[bool; 256]>),
}
fn error(message: &str) -> Error {
    Error::new(format!("native trigger regex: {message}"))
}
fn one(byte: u8) -> Box<[bool; 256]> {
    let mut c = Box::new([false; 256]);
    c[byte as usize] = true;
    c
}
fn named(name: &[u8]) -> Result<Box<[bool; 256]>> {
    let mut c = Box::new([false; 256]);
    for n in 0u16..256 {
        let b = n as u8;
        c[n as usize] = match name {
            b"d" | b"digit" => b.is_ascii_digit(),
            b"s" | b"space" => matches!(b, b' ' | b'\t' | b'\n' | b'\r' | 0x0b | 0x0c),
            b"w" => b.is_ascii_alphanumeric() || b == b'_',
            b"alnum" => b.is_ascii_alphanumeric(),
            b"alpha" => b.is_ascii_alphabetic(),
            b"blank" => matches!(b, b' ' | b'\t' | b'\n' | b'\r' | 0x0b | 0x0c),
            b"cntrl" => b < 32 || b == 127,
            b"graph" => b.is_ascii_graphic(),
            b"lower" => b.is_ascii_lowercase(),
            b"print" => b.is_ascii_graphic() || b == b' ',
            b"punct" => b.is_ascii_punctuation(),
            b"upper" => b.is_ascii_uppercase(),
            b"xdigit" => b.is_ascii_hexdigit(),
            _ => return Err(error("unknown named character class")),
        };
    }
    Ok(c)
}
fn collating(name: &[u8]) -> Result<u8> {
    if name.len() == 1 {
        Ok(name[0])
    } else {
        Err(error("unknown C-locale collating element"))
    }
}

impl Parser<'_> {
    fn peek(&self) -> Option<u8> {
        self.pattern.get(self.pos).copied()
    }
    fn take(&mut self) -> Result<u8> {
        let b = self
            .peek()
            .ok_or_else(|| error("unexpected end of pattern"))?;
        self.pos += 1;
        Ok(b)
    }
    fn eat(&mut self, b: u8) -> bool {
        if self.peek() == Some(b) {
            self.pos += 1;
            true
        } else {
            false
        }
    }
    fn push(&mut self, kind: Kind, captures: Range<usize>) -> Result<usize> {
        if self.nodes.len() >= MAX_NODES {
            return Err(error("node budget exceeded"));
        }
        let n = self.nodes.len();
        self.nodes.push(Node { kind, captures });
        Ok(n)
    }
    fn extent(&self, items: &[usize]) -> Range<usize> {
        let mut start = usize::MAX;
        let mut end = 0;
        for &i in items {
            let r = &self.nodes[i].captures;
            if !r.is_empty() {
                start = start.min(r.start);
                end = end.max(r.end);
            }
        }
        if start == usize::MAX {
            0..0
        } else {
            start..end
        }
    }
    fn number(&mut self) -> Result<usize> {
        let start = self.pos;
        let mut value = 0usize;
        while let Some(b'0'..=b'9') = self.peek() {
            value = value
                .checked_mul(10)
                .and_then(|v| v.checked_add((self.pattern[self.pos] - b'0') as usize))
                .ok_or_else(|| error("integer overflow"))?;
            self.pos += 1;
            if value > i32::MAX as usize {
                return Err(error("integer exceeds source range"));
            }
        }
        if self.pos == start {
            return Err(error("expected decimal number"));
        }
        Ok(value)
    }
    fn hex(&mut self, n: usize) -> Result<u8> {
        let mut value = 0u32;
        for _ in 0..n {
            let b = self.take()?;
            let x = match b {
                b'0'..=b'9' => b - b'0',
                b'a'..=b'f' => b - b'a' + 10,
                b'A'..=b'F' => b - b'A' + 10,
                _ => return Err(error("invalid hex escape")),
            };
            value = (value << 4) + u32::from(x);
        }
        u8::try_from(value).map_err(|_| error("escape exceeds byte character range"))
    }
    fn escaped_atom(&mut self, in_class: bool) -> Result<Atom> {
        let b = self.take()?;
        let byte = match b {
            b'f' => 12,
            b'n' => b'\n',
            b'r' => b'\r',
            b't' => b'\t',
            b'v' => 11,
            b'b' if in_class => 8,
            b'0' => {
                if self.peek().is_some_and(|v| v.is_ascii_digit()) {
                    return Err(error("octal escape is not ECMAScript"));
                }
                0
            }
            b'1'..=b'9' if in_class => return Err(error("backreference inside character class")),
            b'x' => self.hex(2)?,
            b'u' => self.hex(4)?,
            b'c' => {
                let x = self.take()?;
                if !x.is_ascii_alphabetic() {
                    return Err(error("invalid control escape"));
                }
                x % 32
            }
            b'd' | b'D' | b's' | b'S' | b'w' | b'W' => {
                let mut set = named(&[b.to_ascii_lowercase()])?;
                if b.is_ascii_uppercase() {
                    for value in set.iter_mut() {
                        *value = !*value;
                    }
                }
                return Ok(Atom::Set(set));
            }
            other => other,
        };
        Ok(Atom::Byte(byte))
    }
    fn class_atom(&mut self) -> Result<Atom> {
        if self.eat(b'\\') {
            return self.escaped_atom(true);
        }
        if self.peek() == Some(b'[')
            && self
                .pattern
                .get(self.pos + 1)
                .is_some_and(|b| matches!(*b, b':' | b'.' | b'='))
        {
            self.pos += 1;
            let marker = self.take()?;
            let start = self.pos;
            while self
                .peek()
                .is_some_and(|b| !matches!(b, b':' | b'.' | b'='))
            {
                self.pos += 1;
            }
            let name = self.pattern[start..self.pos].to_vec();
            if !self.eat(marker) || !self.eat(b']') {
                return Err(error("invalid delimited character class"));
            }
            return if marker == b':' {
                Ok(Atom::Set(named(&name)?))
            } else {
                Ok(Atom::Byte(collating(&name)?))
            };
        }
        Ok(Atom::Byte(self.take()?))
    }
    fn class(&mut self) -> Result<usize> {
        let inverse = self.eat(b'^');
        let mut set = Box::new([false; 256]);
        while self.peek() != Some(b']') {
            if self.peek().is_none() {
                return Err(error("unclosed character class"));
            }
            let first = self.class_atom()?;
            if self.peek() == Some(b'-') && self.pattern.get(self.pos + 1) != Some(&b']') {
                self.pos += 1;
                let last = self.class_atom()?;
                let (Atom::Byte(a), Atom::Byte(b)) = (first, last) else {
                    return Err(error("character class set used as range bound"));
                };
                if a > b {
                    return Err(error("reversed character range"));
                }
                for byte in a..=b {
                    set[byte as usize] = true;
                }
            } else {
                match first {
                    Atom::Byte(b) => set[b as usize] = true,
                    Atom::Set(other) => {
                        for i in 0..256 {
                            set[i] |= other[i];
                        }
                    }
                }
            }
        }
        self.pos += 1;
        if inverse {
            for v in set.iter_mut() {
                *v = !*v;
            }
        }
        self.push(Kind::Class(set), 0..0)
    }
    fn alternative(&mut self, depth: usize) -> Result<usize> {
        let mut items = Vec::new();
        while !matches!(self.peek(), None | Some(b'|' | b')')) {
            let b = self.take()?;
            let mut quantifiable = true;
            let mut node = match b {
                b'[' => self.class()?,
                b'.' => {
                    let mut c = Box::new([true; 256]);
                    c[b'\r' as usize] = false;
                    c[b'\n' as usize] = false;
                    self.push(Kind::Class(c), 0..0)?
                }
                b'^' => {
                    quantifiable = false;
                    self.push(Kind::Start, 0..0)?
                }
                b'$' => {
                    quantifiable = false;
                    self.push(Kind::End, 0..0)?
                }
                b'\\' => match self.peek() {
                    Some(b'b' | b'B') => {
                        let positive = self.take()? == b'b';
                        quantifiable = false;
                        self.push(Kind::Boundary(positive), 0..0)?
                    }
                    Some(b'1'..=b'9') => {
                        let index = self.number()?;
                        if !self.finished.get(index - 1).copied().unwrap_or(false) {
                            return Err(error("backreference does not name a completed group"));
                        }
                        self.push(Kind::Backref(index - 1), 0..0)?
                    }
                    _ => {
                        let c = match self.escaped_atom(false)? {
                            Atom::Byte(v) => one(v),
                            Atom::Set(c) => c,
                        };
                        self.push(Kind::Class(c), 0..0)?
                    }
                },
                b'(' => {
                    if depth >= MAX_DEPTH {
                        return Err(error("group nesting exceeds limit"));
                    }
                    let kind = if self.eat(b'?') { self.take()? } else { b'(' };
                    if !matches!(kind, b'(' | b':' | b'=' | b'!') {
                        return Err(error("unsupported or invalid group syntax"));
                    }
                    let group = if kind == b'(' {
                        if self.finished.len() >= MAX_GROUPS {
                            return Err(error("capture group budget exceeded"));
                        }
                        let i = self.finished.len();
                        self.finished.push(false);
                        Some(i)
                    } else {
                        None
                    };
                    let child = self.disjunction(depth + 1)?;
                    if !self.eat(b')') {
                        return Err(error("unclosed group"));
                    }
                    if let Some(group) = group {
                        self.finished[group] = true;
                        self.push(Kind::Group(group, child), group..self.finished.len())?
                    } else if kind == b':' {
                        child
                    } else {
                        quantifiable = false;
                        let captures = self.nodes[child].captures.clone();
                        self.push(Kind::Look(child, kind == b'='), captures)?
                    }
                }
                b'*' | b'+' | b'?' | b'{' => return Err(error("quantifier without atom")),
                b'}' | b']' => return Err(error("unmatched closing delimiter")),
                other => self.push(Kind::Class(one(other)), 0..0)?,
            };
            if quantifiable {
                let repeat = match self.peek() {
                    Some(b'*') => {
                        self.pos += 1;
                        Some((0, None))
                    }
                    Some(b'+') => {
                        self.pos += 1;
                        Some((1, None))
                    }
                    Some(b'?') => {
                        self.pos += 1;
                        Some((0, Some(1)))
                    }
                    Some(b'{') => {
                        self.pos += 1;
                        let min = self.number()?;
                        let max = if self.eat(b',') {
                            if self.peek() == Some(b'}') {
                                None
                            } else {
                                Some(self.number()?)
                            }
                        } else {
                            Some(min)
                        };
                        if !self.eat(b'}') || max.is_some_and(|n| n < min) {
                            return Err(error("invalid repetition bounds"));
                        }
                        Some((min, max))
                    }
                    _ => None,
                };
                if let Some((min, max)) = repeat {
                    let greedy = !self.eat(b'?');
                    let captures = self.nodes[node].captures.clone();
                    node = self.push(
                        Kind::Repeat {
                            child: node,
                            min,
                            max,
                            greedy,
                        },
                        captures,
                    )?;
                }
            }
            items.push(node);
        }
        if items.is_empty() {
            self.push(Kind::Empty, 0..0)
        } else if items.len() == 1 {
            Ok(items[0])
        } else {
            let captures = self.extent(&items);
            self.push(Kind::Seq(items), captures)
        }
    }
    fn disjunction(&mut self, depth: usize) -> Result<usize> {
        let mut choices = vec![self.alternative(depth)?];
        while self.eat(b'|') {
            choices.push(self.alternative(depth)?);
        }
        if choices.len() == 1 {
            Ok(choices[0])
        } else {
            let captures = self.extent(&choices);
            self.push(Kind::Alt(choices), captures)
        }
    }
}
#[derive(Debug, Clone)]
enum Task {
    Node(usize),
    EndGroup(usize, usize),
    Repeat {
        node: usize,
        count: usize,
        previous: Option<usize>,
    },
}
#[derive(Clone)]
struct State {
    pos: usize,
    todo: Vec<Task>,
    captures: Vec<Option<Range<usize>>>,
}
fn word(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b == b'_'
}
fn allocation_bytes(todo: usize, captures: usize) -> Result<usize> {
    todo.checked_mul(std::mem::size_of::<Task>())
        .and_then(|n| {
            captures
                .checked_mul(std::mem::size_of::<Option<Range<usize>>>())
                .and_then(|c| n.checked_add(c))
        })
        .and_then(|n| n.checked_add(128))
        .ok_or_else(|| error("matching allocation overflow"))
}
fn state_bytes(s: &State) -> Result<usize> {
    allocation_bytes(s.todo.capacity(), s.captures.capacity())
}
struct Budget {
    steps: usize,
    stack_bytes: usize,
}
impl Budget {
    fn charge(&mut self, work: usize) -> Result<()> {
        self.steps = self
            .steps
            .checked_sub(work)
            .ok_or_else(|| error("matching complexity budget exceeded"))?;
        Ok(())
    }
    fn step(&mut self) -> Result<()> {
        self.charge(1)
    }
    fn check_allocation(&self, bytes: usize) -> Result<usize> {
        let next = self
            .stack_bytes
            .checked_add(bytes)
            .ok_or_else(|| error("matching allocation overflow"))?;
        if next > MAX_STACK_BYTES {
            return Err(error("matching allocation budget exceeded"));
        }
        Ok(next)
    }
    fn clone_state(&self, state: &State, extra: usize) -> Result<State> {
        let capacity = state
            .todo
            .len()
            .checked_add(extra)
            .ok_or_else(|| error("matching allocation overflow"))?;
        self.check_allocation(allocation_bytes(capacity, state.captures.len())?)?;
        // Reserve the appended tasks before copying so growth cannot double a
        // large pending continuation after the budget preflight.
        let mut todo = Vec::with_capacity(capacity);
        todo.extend_from_slice(&state.todo);
        Ok(State {
            pos: state.pos,
            todo,
            captures: state.captures.clone(),
        })
    }
    fn push(&mut self, stack: &mut Vec<State>, state: State) -> Result<()> {
        let next = self.check_allocation(state_bytes(&state)?)?;
        stack.push(state);
        self.stack_bytes = next;
        Ok(())
    }
}
impl NativeRegex {
    pub fn parse(pattern: &[u8]) -> Result<Self> {
        if pattern.len() > MAX_PATTERN {
            return Err(error("pattern size exceeds limit"));
        }
        let mut parser = Parser {
            pattern,
            pos: 0,
            nodes: Vec::new(),
            finished: Vec::new(),
        };
        let root = parser.disjunction(0)?;
        if parser.pos != pattern.len() {
            return Err(error("unmatched closing parenthesis"));
        }
        Ok(Self {
            program: Arc::new(Program {
                nodes: parser.nodes,
                root,
                groups: parser.finished.len(),
            }),
        })
    }
    fn execute(
        &self,
        input: &[u8],
        mut state: State,
        full: bool,
        budget: &mut Budget,
        depth: usize,
    ) -> Result<Option<State>> {
        if depth > MAX_DEPTH {
            return Err(error("lookahead nesting exceeds limit"));
        }
        let mut alternatives = Vec::new();
        loop {
            budget.step()?;
            let Some(task) = state.todo.pop() else {
                if !full || state.pos == input.len() {
                    for pending in &alternatives {
                        budget.stack_bytes -= state_bytes(pending)?;
                    }
                    return Ok(Some(state));
                }
                let Some(next) = alternatives.pop() else {
                    return Ok(None);
                };
                budget.stack_bytes -= state_bytes(&next)?;
                state = next;
                continue;
            };
            let mut failed = false;
            match task {
                Task::EndGroup(group, start) => state.captures[group] = Some(start..state.pos),
                Task::Repeat {
                    node,
                    count,
                    previous,
                } => {
                    let Kind::Repeat {
                        child,
                        min,
                        max,
                        greedy,
                    } = self.program.nodes[node].kind
                    else {
                        unreachable!()
                    };
                    if count > min && previous == Some(state.pos) {
                        failed = true;
                    } else {
                        let can_exit = count >= min;
                        let can_repeat = max.is_none_or(|m| count < m)
                            && !(can_exit && previous == Some(state.pos));
                        if can_repeat {
                            let mut repeat = budget.clone_state(&state, 2)?;
                            for slot in self.program.nodes[child].captures.clone() {
                                repeat.captures[slot] = None;
                            }
                            repeat.todo.push(Task::Repeat {
                                node,
                                count: count + 1,
                                previous: Some(state.pos),
                            });
                            repeat.todo.push(Task::Node(child));
                            if can_exit {
                                if greedy {
                                    budget.push(&mut alternatives, state)?;
                                    state = repeat;
                                } else {
                                    budget.push(&mut alternatives, repeat)?;
                                }
                            } else {
                                state = repeat;
                            }
                        } else if !can_exit {
                            failed = true;
                        }
                    }
                }
                Task::Node(index) => match &self.program.nodes[index].kind {
                    Kind::Empty => {}
                    Kind::Class(set) => {
                        if input.get(state.pos).is_some_and(|b| set[*b as usize]) {
                            state.pos += 1;
                        } else {
                            failed = true;
                        }
                    }
                    Kind::Seq(nodes) => {
                        for &node in nodes.iter().rev() {
                            state.todo.push(Task::Node(node));
                        }
                    }
                    Kind::Alt(nodes) => {
                        for &node in nodes.iter().skip(1).rev() {
                            let mut other = budget.clone_state(&state, 1)?;
                            other.todo.push(Task::Node(node));
                            budget.push(&mut alternatives, other)?;
                        }
                        state.todo.push(Task::Node(nodes[0]));
                    }
                    Kind::Group(group, child) => {
                        state.todo.push(Task::EndGroup(*group, state.pos));
                        state.todo.push(Task::Node(*child));
                    }
                    Kind::Repeat { .. } => state.todo.push(Task::Repeat {
                        node: index,
                        count: 0,
                        previous: None,
                    }),
                    Kind::Start => failed = state.pos != 0,
                    Kind::End => failed = state.pos != input.len(),
                    Kind::Boundary(positive) => {
                        let before = state.pos > 0 && word(input[state.pos - 1]);
                        let after = input.get(state.pos).is_some_and(|b| word(*b));
                        failed = (before != after) != *positive;
                    }
                    Kind::Backref(group) => {
                        if let Some(range) = &state.captures[*group] {
                            let length = range.len();
                            budget.charge(length)?;
                            if input.get(state.pos..state.pos.saturating_add(length))
                                == Some(&input[range.clone()])
                            {
                                state.pos += length;
                            } else {
                                failed = true;
                            }
                        }
                    }
                    Kind::Look(child, positive) => {
                        let found = self.execute(
                            input,
                            State {
                                pos: state.pos,
                                todo: vec![Task::Node(*child)],
                                captures: state.captures.clone(),
                            },
                            false,
                            budget,
                            depth + 1,
                        )?;
                        if *positive {
                            if let Some(found) = found {
                                state.captures = found.captures;
                            } else {
                                failed = true;
                            }
                        } else if found.is_some() {
                            failed = true;
                        }
                    }
                },
            }
            if failed {
                let Some(next) = alternatives.pop() else {
                    return Ok(None);
                };
                budget.stack_bytes -= state_bytes(&next)?;
                state = next;
            }
        }
    }
    /// First leftmost match; offsets are bytes, including positions within UTF-8.
    pub fn find(&self, input: &[u8]) -> Result<Option<NativeMatch>> {
        self.find_mode(input, false)
    }
    pub fn full_match(&self, input: &[u8]) -> Result<Option<NativeMatch>> {
        self.find_mode(input, true)
    }
    fn find_mode(&self, input: &[u8], full: bool) -> Result<Option<NativeMatch>> {
        if input.len() > MAX_INPUT {
            return Err(error("input size exceeds limit"));
        }
        let mut budget = Budget {
            steps: MAX_STEPS,
            stack_bytes: 0,
        };
        let end = if full { 0 } else { input.len() };
        for start in 0..=end {
            if let Some(state) = self.execute(
                input,
                State {
                    pos: start,
                    todo: vec![Task::Node(self.program.root)],
                    captures: vec![None; self.program.groups],
                },
                full,
                &mut budget,
                0,
            )? {
                return Ok(Some(NativeMatch {
                    range: start..state.pos,
                    captures: state.captures,
                }));
            }
        }
        Ok(None)
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn raw_byte_classes_and_captures() {
        let r = NativeRegex::parse(br"(\w+)\s*(!)").unwrap();
        let m = r.find(b"\xff abc !").unwrap().unwrap();
        assert_eq!(m.range, 2..7);
        assert_eq!(m.captures, vec![Some(2..5), Some(6..7)]);
        assert!(
            NativeRegex::parse(br"\s")
                .unwrap()
                .find(&[0xa0])
                .unwrap()
                .is_none()
        );
        assert_eq!(
            NativeRegex::parse(b".")
                .unwrap()
                .find(&[0xff])
                .unwrap()
                .unwrap()
                .range,
            0..1
        );
    }
    #[test]
    fn lookahead_backreferences_and_lazy_order() {
        let r = NativeRegex::parse(br"(?=(a+))a*b\1").unwrap();
        assert_eq!(r.find(b"baaabac").unwrap().unwrap().range, 3..6);
        let r = NativeRegex::parse(br"a+?").unwrap();
        assert_eq!(r.find(b"aaaa").unwrap().unwrap().range, 0..1);
        assert_eq!(r.full_match(b"aaaa").unwrap().unwrap().range, 0..4);
        assert!(
            NativeRegex::parse(br"(a)?b\1")
                .unwrap()
                .full_match(b"b")
                .unwrap()
                .is_some()
        );
    }
    #[test]
    fn reserved_capacity_and_backreference_bytes_consume_budget() {
        let state = State {
            pos: 0,
            todo: Vec::with_capacity(4096),
            captures: Vec::new(),
        };
        assert!(state_bytes(&state).unwrap() >= 4096 * std::mem::size_of::<Task>());
        let budget = Budget {
            steps: MAX_STEPS,
            stack_bytes: MAX_STACK_BYTES - 64,
        };
        assert!(budget.clone_state(&state, 16).is_err());
        let regex = NativeRegex::parse(br"(?=(a{256}))(?:(?=\1)){32768}").unwrap();
        assert!(regex.full_match(&[b'a'; 256]).is_err());
    }
    #[test]
    fn syntax_errors_and_bounded_backtracking() {
        for p in [
            br"\1".as_slice(),
            br"(\1)",
            br"(?<=x)y",
            br"\u0100",
            br"a{2,1}",
            br"[\1]",
            br"(?=a)*",
        ] {
            assert!(NativeRegex::parse(p).is_err(), "{p:?}");
        }
        let r = NativeRegex::parse(br"^(a+)+$").unwrap();
        assert!(r.find(format!("{}!", "a".repeat(24)).as_bytes()).is_err());
    }
}

#[cfg(test)]
mod reference_tests {
    use super::*;
    struct Reader {
        bytes: &'static [u8],
        position: usize,
    }
    impl Reader {
        fn take(&mut self, count: usize) -> &'static [u8] {
            let result = &self.bytes[self.position..self.position + count];
            self.position += count;
            result
        }
        fn u32(&mut self) -> u32 {
            u32::from_le_bytes(self.take(4).try_into().unwrap())
        }
        fn blob(&mut self) -> &'static [u8] {
            let size = self.u32() as usize;
            self.take(size)
        }
        fn expected(&mut self) -> Option<Vec<Option<Range<usize>>>> {
            let count = self.u32();
            if count == u32::MAX {
                return None;
            }
            Some(
                (0..count)
                    .map(|_| {
                        let start = self.u32() as i32;
                        let size = self.u32() as usize;
                        (start >= 0).then_some(start as usize..(start as usize).wrapping_add(size))
                    })
                    .collect(),
            )
        }
    }
    #[test]
    fn frozen_msvc_regex_search_and_capture_corpus() {
        // Independently captured from MSVC std::regex with ECMAScript + C locale.
        // The fixture stores byte inputs and complete capture vectors, not only
        // booleans, so lazy trigger replay positions are part of the assertion.
        let bytes = include_bytes!("../../../fixtures/behavior/native-regex-reference.bin");
        let mut r = Reader { bytes, position: 0 };
        assert_eq!(r.take(8), b"NREGEX01");
        let count = r.u32();
        let mut checks = 0;
        for case in 0..count {
            let pattern = r.blob();
            let compiled = r.take(1)[0] != 0;
            let regex = NativeRegex::parse(pattern);
            assert_eq!(regex.is_ok(), compiled, "compile case {case}: {pattern:?}");
            checks += 1;
            let inputs = r.u32();
            for input in 0..inputs {
                let text = r.blob();
                for full in [false, true] {
                    let expected = r.expected();
                    let regex = regex.as_ref().unwrap();
                    let found = if full {
                        regex.full_match(text)
                    } else {
                        regex.find(text)
                    }
                    .unwrap();
                    let actual = found.map(|m| {
                        let mut groups = vec![Some(m.range)];
                        groups.extend(m.captures);
                        groups
                    });
                    assert_eq!(actual, expected, "case {case}, input {input}, full {full}");
                    checks += 1;
                }
            }
        }
        assert_eq!(checks, 12_349);
        assert_eq!(r.position, bytes.len());
    }
}
