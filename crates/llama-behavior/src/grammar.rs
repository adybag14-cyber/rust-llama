//! Bounded character GBNF compiler and incremental UTF-8 recognizer.
//!
//! Supports the character-language syntax used by the eight bundled `.gbnf`
//! examples: rules, alternatives, literals, character classes/ranges, inverse
//! classes, `.`, grouping, `?`, `*`, `+`, `{n}`, `{n,}` and `{n,m}`. Token-level
//! `<[id]>`/`!<[id]>` terminals and lazy regex triggers remain explicit T04/T10
//! integration work; they are rejected rather than ignored.

use crate::{Error, Result};
use std::collections::{BTreeMap, BTreeSet, HashSet, VecDeque};
use std::sync::Arc;

#[derive(Debug, Clone, Copy)]
pub struct Limits {
    pub max_source_bytes: usize,
    pub max_rules: usize,
    pub max_productions: usize,
    pub max_symbols: usize,
    pub max_parse_depth: usize,
    pub max_repeat: usize,
    pub max_states_per_position: usize,
    pub max_total_states: usize,
    pub max_codepoints: usize,
}
impl Default for Limits {
    fn default() -> Self {
        Self {
            max_source_bytes: 1024 * 1024,
            max_rules: 4096,
            max_productions: 16384,
            max_symbols: 65536,
            max_parse_depth: 128,
            max_repeat: 2000,
            max_states_per_position: 65536,
            max_total_states: 1_000_000,
            max_codepoints: 65536,
        }
    }
}

#[derive(Debug, Clone)]
struct Class {
    inverted: bool,
    ranges: Vec<(u32, u32)>,
}
impl Class {
    fn single(c: char) -> Self {
        Self {
            inverted: false,
            ranges: vec![(c as u32, c as u32)],
        }
    }
    fn matches(&self, c: char) -> bool {
        self.ranges
            .iter()
            .any(|&(a, b)| a <= c as u32 && c as u32 <= b)
            != self.inverted
    }
    fn may_match(&self, lo: u32, hi: u32) -> bool {
        if lo > hi {
            return false;
        }
        if !self.inverted {
            return self.ranges.iter().any(|&(a, b)| a <= hi && b >= lo);
        }
        let mut cursor = lo;
        for &(a, b) in &self.ranges {
            if b < cursor {
                continue;
            }
            if a > cursor {
                return true;
            }
            if b >= hi {
                return false;
            }
            cursor = b + 1;
        }
        cursor <= hi
    }
    fn normalize(&mut self) {
        self.ranges.sort_unstable();
        let mut merged: Vec<(u32, u32)> = Vec::new();
        for &(a, b) in &self.ranges {
            if let Some(last) = merged.last_mut()
                && a <= last.1.saturating_add(1)
            {
                last.1 = last.1.max(b);
                continue;
            }
            merged.push((a, b));
        }
        self.ranges = merged;
    }
}

#[derive(Debug, Clone)]
enum Expr {
    Terminal(Class),
    Ref(String),
    Sequence(Vec<Expr>),
    Alternatives(Vec<Expr>),
    Repeat(Box<Expr>, usize, Option<usize>),
}

fn expression_depth(expression: &Expr) -> usize {
    let mut stack = vec![(expression, 1usize)];
    let mut deepest = 0;
    while let Some((node, depth)) = stack.pop() {
        deepest = deepest.max(depth);
        match node {
            Expr::Sequence(values) | Expr::Alternatives(values) => {
                for child in values {
                    stack.push((child, depth + 1));
                }
            }
            Expr::Repeat(child, _, _) => stack.push((child, depth + 1)),
            _ => {}
        }
    }
    deepest
}

struct Parser<'a> {
    text: &'a str,
    pos: usize,
    limits: Limits,
    items: usize,
}
impl Parser<'_> {
    fn error(&self, message: &str) -> Error {
        Error::new(format!("GBNF byte {}: {message}", self.pos))
    }
    fn peek(&self) -> Option<char> {
        self.text[self.pos..].chars().next()
    }
    fn bump(&mut self) -> Option<char> {
        let c = self.peek()?;
        self.pos += c.len_utf8();
        Some(c)
    }
    fn eat(&mut self, c: char) -> bool {
        if self.peek() == Some(c) {
            self.bump();
            true
        } else {
            false
        }
    }
    fn space(&mut self, newlines: bool) {
        loop {
            match self.peek() {
                Some(' ' | '\t') => {
                    self.bump();
                }
                Some('\n' | '\r') if newlines => {
                    self.bump();
                }
                Some('#') => {
                    while !matches!(self.peek(), None | Some('\n' | '\r')) {
                        self.bump();
                    }
                }
                _ => break,
            }
        }
    }
    fn name(&mut self) -> Result<String> {
        let start = self.pos;
        while self
            .peek()
            .is_some_and(|c| c.is_ascii_alphanumeric() || c == '-')
        {
            self.bump();
        }
        if self.pos == start {
            return Err(self.error("expected rule name"));
        }
        Ok(self.text[start..self.pos].to_owned())
    }
    fn integer(&mut self) -> Result<usize> {
        let start = self.pos;
        while self.peek().is_some_and(|c| c.is_ascii_digit()) {
            self.bump();
        }
        if start == self.pos {
            return Err(self.error("expected repetition count"));
        }
        let value = self.text[start..self.pos]
            .parse::<usize>()
            .map_err(|_| self.error("repetition count overflow"))?;
        if value > self.limits.max_repeat {
            return Err(self.error("repetition count exceeds limit"));
        }
        Ok(value)
    }
    fn scalar(&mut self) -> Result<char> {
        let c = self
            .bump()
            .ok_or_else(|| self.error("unexpected end of character"))?;
        if c != '\\' {
            return Ok(c);
        }
        let escape = self.bump().ok_or_else(|| self.error("incomplete escape"))?;
        let digits = match escape {
            'x' => 2,
            'u' => 4,
            'U' => 8,
            _ => 0,
        };
        if digits > 0 {
            let mut value = 0u32;
            for _ in 0..digits {
                let nibble = self
                    .bump()
                    .and_then(|x| x.to_digit(16))
                    .ok_or_else(|| self.error("invalid hexadecimal escape"))?;
                value = (value << 4) | nibble;
            }
            return char::from_u32(value)
                .ok_or_else(|| self.error("escape is not a Unicode scalar"));
        }
        match escape {
            't' => Ok('\t'),
            'r' => Ok('\r'),
            'n' => Ok('\n'),
            '\\' | '"' | '[' | ']' | '-' => Ok(escape),
            _ => Err(self.error("unknown character escape")),
        }
    }
    fn atom(&mut self, depth: usize) -> Result<Expr> {
        if depth > self.limits.max_parse_depth {
            return Err(self.error("grammar nesting exceeds limit"));
        }
        self.items += 1;
        if self.items > self.limits.max_symbols {
            return Err(self.error("grammar expression count exceeds limit"));
        }
        match self.peek() {
            Some('"') => {
                self.bump();
                let mut chars = Vec::new();
                while self.peek() != Some('"') {
                    if self.peek().is_none() {
                        return Err(self.error("unterminated string literal"));
                    }
                    chars.push(Expr::Terminal(Class::single(self.scalar()?)));
                    if chars.len() > self.limits.max_symbols {
                        return Err(self.error("literal exceeds symbol limit"));
                    }
                }
                self.bump();
                Ok(Expr::Sequence(chars))
            }
            Some('[') => {
                self.bump();
                let inverted = self.eat('^');
                let mut ranges = Vec::new();
                while self.peek() != Some(']') {
                    if self.peek().is_none() {
                        return Err(self.error("unterminated character class"));
                    }
                    let first = self.scalar()? as u32;
                    let last = if self.peek() == Some('-')
                        && !self.text[self.pos + 1..].starts_with(']')
                    {
                        self.bump();
                        self.scalar()? as u32
                    } else {
                        first
                    };
                    if first > last {
                        return Err(self.error("character range is reversed"));
                    }
                    ranges.push((first, last));
                    if ranges.len() > self.limits.max_symbols {
                        return Err(self.error("character class exceeds range limit"));
                    }
                }
                self.bump();
                if ranges.is_empty() {
                    return Err(self.error("empty character class"));
                }
                let mut class = Class { inverted, ranges };
                class.normalize();
                Ok(Expr::Terminal(class))
            }
            Some('.') => {
                self.bump();
                Ok(Expr::Terminal(Class {
                    inverted: true,
                    ranges: vec![],
                }))
            }
            Some('(') => {
                self.bump();
                self.space(true);
                let expression = self.alternatives(true, depth + 1)?;
                if !self.eat(')') {
                    return Err(self.error("expected closing parenthesis"));
                }
                Ok(expression)
            }
            Some('<' | '!') => Err(self
                .error("token-level grammar terminals require the pending tokenizer integration")),
            Some(c) if c.is_ascii_alphanumeric() || c == '-' => Ok(Expr::Ref(self.name()?)),
            _ => Err(self.error("expected literal, class, rule reference or group")),
        }
    }
    fn sequence(&mut self, nested: bool, depth: usize) -> Result<Expr> {
        let mut sequence = Vec::new();
        loop {
            self.space(nested);
            if matches!(self.peek(), None | Some('|' | ')'))
                || (!nested && matches!(self.peek(), Some('\n' | '\r')))
            {
                break;
            }
            let mut atom = self.atom(depth)?;
            let mut product = 1usize;
            loop {
                self.space(nested);
                let repeat = match self.peek() {
                    Some('*') => {
                        self.bump();
                        Some((0, None))
                    }
                    Some('+') => {
                        self.bump();
                        Some((1, None))
                    }
                    Some('?') => {
                        self.bump();
                        Some((0, Some(1)))
                    }
                    Some('{') => {
                        self.bump();
                        self.space(nested);
                        let min = self.integer()?;
                        self.space(nested);
                        let max = if self.eat(',') {
                            self.space(nested);
                            if self.peek() == Some('}') {
                                None
                            } else {
                                Some(self.integer()?)
                            }
                        } else {
                            Some(min)
                        };
                        self.space(nested);
                        if !self.eat('}') {
                            return Err(self.error("expected closing repetition brace"));
                        }
                        if max.is_some_and(|m| m < min) {
                            return Err(self.error("maximum repetition is below minimum"));
                        }
                        Some((min, max))
                    }
                    _ => None,
                };
                if let Some((min, max)) = repeat {
                    product = product
                        .checked_mul(max.unwrap_or(min).max(1))
                        .ok_or_else(|| self.error("nested repetition overflow"))?;
                    if product > self.limits.max_repeat {
                        return Err(self.error("nested repetition product exceeds limit"));
                    }
                    if expression_depth(&atom) >= self.limits.max_parse_depth {
                        return Err(self.error("nested expression exceeds depth limit"));
                    }
                    atom = Expr::Repeat(Box::new(atom), min, max);
                } else {
                    break;
                }
            }
            if expression_depth(&atom) > self.limits.max_parse_depth {
                return Err(self.error("nested expression exceeds depth limit"));
            }
            sequence.push(atom);
        }
        Ok(Expr::Sequence(sequence))
    }
    fn alternatives(&mut self, nested: bool, depth: usize) -> Result<Expr> {
        let mut values = vec![self.sequence(nested, depth)?];
        while self.eat('|') {
            self.space(true);
            values.push(self.sequence(nested, depth)?);
        }
        Ok(Expr::Alternatives(values))
    }
    fn rules(&mut self) -> Result<Vec<(String, Expr)>> {
        let mut rules = Vec::new();
        let mut names = BTreeSet::new();
        self.space(true);
        while self.peek().is_some() {
            let name = self.name()?;
            if !names.insert(name.clone()) {
                return Err(self.error("duplicate rule definition"));
            }
            self.space(false);
            if !self.text[self.pos..].starts_with("::=") {
                return Err(self.error("expected ::="));
            }
            self.pos += 3;
            self.space(true);
            let expr = self.alternatives(false, 0)?;
            if !matches!(self.peek(), None | Some('\n' | '\r')) {
                return Err(self.error("expected end of rule"));
            }
            rules.push((name, expr));
            if rules.len() > self.limits.max_rules {
                return Err(self.error("rule count exceeds limit"));
            }
            self.space(true);
        }
        Ok(rules)
    }
}

#[derive(Debug, Clone)]
enum Symbol {
    Nonterminal(usize),
    Terminal(Class),
}
#[derive(Debug, Clone)]
struct Production {
    lhs: usize,
    rhs: Vec<Symbol>,
}
#[derive(Debug)]
struct Compiled {
    productions: Vec<Production>,
    by_lhs: Vec<Vec<usize>>,
    source_rules: usize,
    limits: Limits,
}
struct Compiler {
    names: BTreeMap<String, usize>,
    next: usize,
    productions: Vec<Production>,
    symbols: usize,
    limits: Limits,
}
impl Compiler {
    fn fresh(&mut self) -> Result<usize> {
        if self.next >= self.limits.max_productions {
            return Err(Error::new("generated nonterminal count exceeds limit"));
        }
        let id = self.next;
        self.next += 1;
        Ok(id)
    }
    fn add(&mut self, lhs: usize, rhs: Vec<Symbol>) -> Result<()> {
        self.symbols = self
            .symbols
            .checked_add(rhs.len())
            .ok_or_else(|| Error::new("grammar symbol overflow"))?;
        if self.symbols > self.limits.max_symbols
            || self.productions.len() >= self.limits.max_productions
        {
            return Err(Error::new(
                "expanded grammar exceeds production/symbol limit",
            ));
        }
        self.productions.push(Production { lhs, rhs });
        Ok(())
    }
    fn symbol(&mut self, expr: &Expr) -> Result<Symbol> {
        match expr {
            Expr::Terminal(class) => Ok(Symbol::Terminal(class.clone())),
            Expr::Ref(name) => self
                .names
                .get(name)
                .copied()
                .map(Symbol::Nonterminal)
                .ok_or_else(|| Error::new(format!("undefined grammar rule {name}"))),
            _ => {
                let id = self.fresh()?;
                self.rule(id, expr)?;
                Ok(Symbol::Nonterminal(id))
            }
        }
    }
    fn sequence(&mut self, expr: &Expr) -> Result<Vec<Symbol>> {
        match expr {
            Expr::Sequence(values) => {
                let mut result = Vec::new();
                for value in values {
                    result.extend(self.sequence(value)?);
                }
                if result.len() > self.limits.max_symbols {
                    return Err(Error::new("expanded sequence exceeds symbol limit"));
                }
                Ok(result)
            }
            Expr::Repeat(value, min, max) => {
                let symbol = self.symbol(value)?;
                let mut result = vec![symbol.clone(); *min];
                match max {
                    None => {
                        let id = self.fresh()?;
                        self.add(id, vec![symbol, Symbol::Nonterminal(id)])?;
                        self.add(id, vec![])?;
                        result.push(Symbol::Nonterminal(id));
                    }
                    Some(max) => {
                        let mut previous = None;
                        for _ in *min..*max {
                            let id = self.fresh()?;
                            let mut rhs = vec![symbol.clone()];
                            if let Some(previous) = previous {
                                rhs.push(Symbol::Nonterminal(previous));
                            }
                            self.add(id, rhs)?;
                            self.add(id, vec![])?;
                            previous = Some(id);
                        }
                        if let Some(id) = previous {
                            result.push(Symbol::Nonterminal(id));
                        }
                    }
                }
                Ok(result)
            }
            _ => Ok(vec![self.symbol(expr)?]),
        }
    }
    fn rule(&mut self, lhs: usize, expr: &Expr) -> Result<()> {
        if let Expr::Alternatives(values) = expr {
            for value in values {
                self.rule(lhs, value)?;
            }
        } else {
            let rhs = self.sequence(expr)?;
            self.add(lhs, rhs)?;
        }
        Ok(())
    }
    fn validate_graph(&self) -> Result<()> {
        let mut nullable = vec![false; self.next];
        let mut productive = vec![false; self.next];
        loop {
            let mut changed = false;
            for p in &self.productions {
                if !nullable[p.lhs]
                    && p.rhs
                        .iter()
                        .all(|s| matches!(s,Symbol::Nonterminal(n) if nullable[*n]))
                {
                    nullable[p.lhs] = true;
                    changed = true;
                }
                if !productive[p.lhs]
                    && p.rhs.iter().all(|s| match s {
                        Symbol::Terminal(_) => true,
                        Symbol::Nonterminal(n) => productive[*n],
                    })
                {
                    productive[p.lhs] = true;
                    changed = true;
                }
            }
            if !changed {
                break;
            }
        }
        if productive.iter().any(|x| !*x) {
            return Err(Error::new("grammar contains a nonproductive rule"));
        }
        let mut edges = vec![BTreeSet::new(); self.next];
        let mut indegree = vec![0usize; self.next];
        for p in &self.productions {
            for symbol in &p.rhs {
                if let Symbol::Nonterminal(n) = symbol {
                    if edges[p.lhs].insert(*n) {
                        indegree[*n] += 1;
                    }
                    if !nullable[*n] {
                        break;
                    }
                } else {
                    break;
                }
            }
        }
        let mut queue: VecDeque<usize> = indegree
            .iter()
            .enumerate()
            .filter(|(_, v)| **v == 0)
            .map(|(i, _)| i)
            .collect();
        let mut visited = 0;
        while let Some(n) = queue.pop_front() {
            visited += 1;
            for &m in &edges[n] {
                indegree[m] -= 1;
                if indegree[m] == 0 {
                    queue.push_back(m);
                }
            }
        }
        if visited != self.next {
            return Err(Error::new(
                "left-recursive grammar is unsupported by the source contract",
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct Grammar {
    inner: Arc<Compiled>,
}
impl Grammar {
    pub fn parse(source: &str) -> Result<Self> {
        Self::parse_with_limits(source, "root", Limits::default())
    }
    pub fn parse_with_root(source: &str, root: &str) -> Result<Self> {
        Self::parse_with_limits(source, root, Limits::default())
    }
    pub fn parse_with_limits(source: &str, root: &str, limits: Limits) -> Result<Self> {
        if source.len() > limits.max_source_bytes {
            return Err(Error::new("grammar source exceeds byte limit"));
        }
        if source.contains('\0') {
            return Err(Error::new(
                "literal NUL in grammar source is unsupported; use an escape",
            ));
        }
        if limits.max_parse_depth > 256
            || limits.max_productions > 65536
            || limits.max_rules > 65536
            || limits.max_symbols > 1_000_000
            || limits.max_repeat > 2000
        {
            return Err(Error::new("grammar limits exceed safe compiler bounds"));
        }
        let mut parser = Parser {
            text: source,
            pos: 0,
            limits,
            items: 0,
        };
        let rules = parser.rules()?;
        let names: BTreeMap<String, usize> = rules
            .iter()
            .enumerate()
            .map(|(i, (name, _))| (name.clone(), i))
            .collect();
        let root = *names
            .get(root)
            .ok_or_else(|| Error::new("grammar root rule is missing"))?;
        let mut compiler = Compiler {
            next: names.len(),
            names,
            productions: Vec::new(),
            symbols: 0,
            limits,
        };
        let start = compiler.fresh()?;
        compiler.add(start, vec![Symbol::Nonterminal(root)])?;
        for (name, expr) in &rules {
            let lhs = compiler.names[name];
            compiler.rule(lhs, expr)?;
        }
        compiler.validate_graph()?;
        let mut by_lhs = vec![Vec::new(); compiler.next];
        for (i, p) in compiler.productions.iter().enumerate() {
            by_lhs[p.lhs].push(i);
        }
        Ok(Self {
            inner: Arc::new(Compiled {
                productions: compiler.productions,
                by_lhs,
                source_rules: rules.len(),
                limits,
            }),
        })
    }
    pub fn rule_count(&self) -> usize {
        self.inner.source_rules
    }
    pub fn production_count(&self) -> usize {
        self.inner.productions.len()
    }
    pub fn recognizer(&self) -> Result<Recognizer> {
        Recognizer::new(self.clone())
    }
    pub fn matches(&self, text: &str) -> Result<bool> {
        let mut r = self.recognizer()?;
        Ok(r.accept_bytes(text.as_bytes())? && r.can_end())
    }
    pub fn allows_prefix(&self, bytes: &[u8]) -> Result<bool> {
        self.recognizer()?.accept_bytes(bytes)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct State {
    production: usize,
    dot: usize,
    origin: usize,
}
#[derive(Debug, Clone, Default)]
struct Chart {
    values: Vec<State>,
    seen: HashSet<State>,
}

#[derive(Debug, Clone)]
pub struct Recognizer {
    grammar: Grammar,
    charts: Vec<Chart>,
    pending: Vec<u8>,
    total_states: usize,
    consumed_bytes: usize,
}
impl Recognizer {
    fn new(grammar: Grammar) -> Result<Self> {
        let mut r = Self {
            grammar,
            charts: vec![Chart::default()],
            pending: Vec::new(),
            total_states: 0,
            consumed_bytes: 0,
        };
        r.insert(
            0,
            State {
                production: 0,
                dot: 0,
                origin: 0,
            },
        )?;
        r.close(0)?;
        Ok(r)
    }
    pub fn reset(&mut self) -> Result<()> {
        *self = Self::new(self.grammar.clone())?;
        Ok(())
    }
    pub fn consumed_bytes(&self) -> usize {
        self.consumed_bytes
    }
    pub fn pending_utf8_bytes(&self) -> usize {
        self.pending.len()
    }
    pub fn state_count(&self) -> usize {
        self.total_states
    }
    pub fn can_end(&self) -> bool {
        self.pending.is_empty()
            && self.charts.last().is_some_and(|c| {
                c.seen.contains(&State {
                    production: 0,
                    dot: 1,
                    origin: 0,
                })
            })
    }
    pub fn can_continue(&self) -> bool {
        self.charts.last().is_some_and(|c| {
            c.values.iter().any(|s| {
                matches!(
                    self.grammar.inner.productions[s.production].rhs.get(s.dot),
                    Some(Symbol::Terminal(_))
                )
            })
        })
    }
    /// Accepted bytes commit atomically. A rejected candidate or error leaves
    /// the current grammar/UTF-8 state unchanged, enabling speculative retries.
    pub fn accept_bytes(&mut self, bytes: &[u8]) -> Result<bool> {
        let max_bytes = self
            .grammar
            .inner
            .limits
            .max_codepoints
            .checked_mul(4)
            .ok_or_else(|| Error::new("grammar input limit overflow"))?;
        if bytes.len() > max_bytes.saturating_sub(self.consumed_bytes) {
            return Err(Error::new("grammar input exceeds byte limit"));
        }
        let mut next = self.clone();
        let mut input = std::mem::take(&mut next.pending);
        input.extend_from_slice(bytes);
        let (valid, partial) = match std::str::from_utf8(&input) {
            Ok(text) => (text.len(), false),
            Err(error) if error.error_len().is_none() => (error.valid_up_to(), true),
            Err(_) => return Err(Error::new("invalid UTF-8 token bytes")),
        };
        let text = std::str::from_utf8(&input[..valid]).expect("valid UTF-8 prefix");
        for c in text.chars() {
            if !next.accept_char(c)? {
                return Ok(false);
            }
        }
        if partial {
            next.pending.extend_from_slice(&input[valid..]);
            if !next.partial_is_possible() {
                return Ok(false);
            }
        }
        next.consumed_bytes += bytes.len();
        *self = next;
        Ok(true)
    }
    pub fn would_accept(&self, bytes: &[u8]) -> Result<bool> {
        let mut next = self.clone();
        next.accept_bytes(bytes)
    }
    /// Caller provides real T04 token pieces. End tokens are handled through
    /// `can_end`; a piece is never inferred from its numeric token ID.
    pub fn allowed_tokens(&self, pieces: &[(i32, &[u8])], end_tokens: &[i32]) -> Result<Vec<i32>> {
        if pieces.len() > 1_000_000 {
            return Err(Error::new("grammar candidate count exceeds limit"));
        }
        let mut allowed = Vec::new();
        for &(id, bytes) in pieces {
            if id < 0 {
                return Err(Error::new("negative grammar candidate ID"));
            }
            if end_tokens.contains(&id) {
                if self.can_end() {
                    allowed.push(id);
                }
            } else if !bytes.is_empty() && self.would_accept(bytes)? {
                allowed.push(id);
            }
        }
        Ok(allowed)
    }
    fn insert(&mut self, pos: usize, state: State) -> Result<()> {
        let chart = &mut self.charts[pos];
        if chart.seen.contains(&state) {
            return Ok(());
        }
        let limits = self.grammar.inner.limits;
        if chart.values.len() >= limits.max_states_per_position
            || self.total_states >= limits.max_total_states
        {
            return Err(Error::new("grammar recognition state budget exceeded"));
        }
        chart.seen.insert(state);
        chart.values.push(state);
        self.total_states += 1;
        Ok(())
    }
    fn close(&mut self, pos: usize) -> Result<()> {
        let mut index = 0;
        while index < self.charts[pos].values.len() {
            let state = self.charts[pos].values[index];
            index += 1;
            let p = &self.grammar.inner.productions[state.production];
            if let Some(Symbol::Nonterminal(next)) = p.rhs.get(state.dot) {
                let next = *next;
                let productions = self.grammar.inner.by_lhs[next].clone();
                for production in productions {
                    self.insert(
                        pos,
                        State {
                            production,
                            dot: 0,
                            origin: pos,
                        },
                    )?;
                }
                // A nullable completion can precede a newly discovered waiter.
                let empty_completed = self.charts[pos].values.iter().any(|s| {
                    s.origin == pos && {
                        let p = &self.grammar.inner.productions[s.production];
                        p.lhs == next && s.dot == p.rhs.len()
                    }
                });
                if empty_completed {
                    self.insert(
                        pos,
                        State {
                            dot: state.dot + 1,
                            ..state
                        },
                    )?;
                }
            } else if state.dot == p.rhs.len() {
                let completed = p.lhs;
                let waiters:Vec<State>=self.charts[state.origin].values.iter().copied().filter(|s|matches!(self.grammar.inner.productions[s.production].rhs.get(s.dot),Some(Symbol::Nonterminal(n)) if *n==completed)).collect();
                for waiter in waiters {
                    self.insert(
                        pos,
                        State {
                            dot: waiter.dot + 1,
                            ..waiter
                        },
                    )?;
                }
            }
        }
        Ok(())
    }
    fn accept_char(&mut self, c: char) -> Result<bool> {
        let pos = self.charts.len();
        if pos > self.grammar.inner.limits.max_codepoints {
            return Err(Error::new("grammar input exceeds codepoint limit"));
        }
        let states:Vec<State>=self.charts[pos-1].values.iter().copied().filter(|s|matches!(self.grammar.inner.productions[s.production].rhs.get(s.dot),Some(Symbol::Terminal(class)) if class.matches(c))).collect();
        if states.is_empty() {
            return Ok(false);
        }
        self.charts.push(Chart::default());
        for state in states {
            self.insert(
                pos,
                State {
                    dot: state.dot + 1,
                    ..state
                },
            )?;
        }
        self.close(pos)?;
        Ok(true)
    }
    fn partial_is_possible(&self) -> bool {
        let Some(&first) = self.pending.first() else {
            return true;
        };
        let (length, min, max, mask) = match first {
            0xc2..=0xdf => (2, 0x80u32, 0x7ffu32, 0x1f),
            0xe0..=0xef => (3, 0x800, 0xffff, 0x0f),
            0xf0..=0xf4 => (4, 0x10000, 0x10ffff, 0x07),
            _ => return false,
        };
        if self.pending.len() >= length {
            return false;
        }
        let mut value = (first & mask) as u32;
        for &byte in &self.pending[1..] {
            if byte & 0xc0 != 0x80 {
                return false;
            }
            value = (value << 6) | (byte & 0x3f) as u32;
        }
        let remaining = 6 * (length - self.pending.len());
        let lo = (value << remaining).max(min);
        let hi = (((value + 1) << remaining) - 1).min(max);
        self.charts.last().is_some_and(|chart| {
            chart.values.iter().any(|s| {
                match self.grammar.inner.productions[s.production].rhs.get(s.dot) {
                    Some(Symbol::Terminal(class)) => {
                        (lo <= 0xd7ff && class.may_match(lo, hi.min(0xd7ff)))
                            || (hi >= 0xe000 && class.may_match(lo.max(0xe000), hi))
                    }
                    _ => false,
                }
            })
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn full_gbnf_character_syntax_and_bounds() {
        let g = Grammar::parse(r#"root ::= ("hi" | [A-Z]{2,3}) [^0-9]? "."?"#).unwrap();
        for text in ["hi", "hi!", "ABC", "AB!."] {
            assert!(g.matches(text).unwrap(), "{text}");
        }
        for text in ["h", "ABCD!", "AB0"] {
            assert!(!g.matches(text).unwrap(), "{text}");
        }
        let g = Grammar::parse("root ::= \"a\"{2,} \"b\"{0,2}\n").unwrap();
        for text in ["aa", "aaaa", "aab", "aaaabb"] {
            assert!(g.matches(text).unwrap());
        }
        for text in ["a", "aabbb", "b"] {
            assert!(!g.matches(text).unwrap());
        }
    }
    #[test]
    fn recursive_nested_json_and_prefix_status() {
        let g = Grammar::parse(
            "root ::= \"[\" (value (\",\" value)*)? \"]\"\nvalue ::= root | [0-9]+\n",
        )
        .unwrap();
        for text in ["[]", "[1,2]", "[[1],[],23]"] {
            assert!(g.matches(text).unwrap());
        }
        for text in ["[1,]", "[", "[a]", "[1][2]"] {
            assert!(!g.matches(text).unwrap());
        }
        let mut r = g.recognizer().unwrap();
        assert!(r.accept_bytes(b"[[1],").unwrap());
        assert!(!r.can_end());
        let bytes = r.consumed_bytes();
        assert!(!r.accept_bytes(b"a").unwrap());
        assert_eq!(r.consumed_bytes(), bytes);
        assert!(r.accept_bytes(b"2]").unwrap());
        assert!(r.can_end());
    }
    #[test]
    fn nullable_rules_and_repeated_groups() {
        let g = Grammar::parse("root ::= blank (\"x\" blank){1,3}\nblank ::= | \" \"\n").unwrap();
        for text in ["x", " x ", "xxx", "x x "] {
            assert!(g.matches(text).unwrap(), "{text:?}");
        }
        assert!(!g.matches("").unwrap());
        assert!(!g.matches("xxxx").unwrap());
    }
    #[test]
    fn split_utf8_checks_partial_ranges_and_rolls_back() {
        let g = Grammar::parse("root ::= \"A£🦀界\"\n").unwrap();
        let mut r = g.recognizer().unwrap();
        for b in "A£🦀界".as_bytes() {
            assert!(r.accept_bytes(&[*b]).unwrap());
        }
        assert!(r.can_end());
        assert_eq!(r.consumed_bytes(), 10);
        r.reset().unwrap();
        assert!(r.accept_bytes(b"A").unwrap());
        assert!(!r.accept_bytes(&[0xe2]).unwrap());
        assert_eq!(r.consumed_bytes(), 1);
        assert!(r.accept_bytes(&[0xc2]).unwrap());
        assert_eq!(r.pending_utf8_bytes(), 1);
        assert!(r.accept_bytes(&[0xa3]).unwrap());
        assert!(r.accept_bytes(&[0xff]).is_err());
        assert_eq!(r.consumed_bytes(), 3);
    }
    #[test]
    fn rejects_invalid_grammars_and_unimplemented_token_terminals() {
        for source in [
            "",
            "root ::= missing",
            "root ::= root \"x\" | \"y\"",
            "root ::= (\"a\"?)*",
            "root ::= [z-a]",
            "root ::= \"\\uD800\"",
            "root ::= <[42]>",
            "root ::= \"x\"{2001}",
            "root ::= (\"x\"",
            "root ::= \"a\"\nroot ::= \"b\"",
        ] {
            assert!(Grammar::parse(source).is_err(), "{source}");
        }
        assert!(Grammar::parse(&("root ::= \"a\"".to_owned() + &"?".repeat(10_000))).is_err());
    }
    #[test]
    fn state_budget_failure_is_transactional() {
        let limits = Limits {
            max_total_states: 24,
            ..Limits::default()
        };
        let g = Grammar::parse_with_limits("root ::= \"a\"*", "root", limits).unwrap();
        let mut r = g.recognizer().unwrap();
        let old = r.consumed_bytes();
        assert!(r.accept_bytes(b"aaaaaaaaaaaaaaaaaaaaaaaa").is_err());
        assert_eq!(r.consumed_bytes(), old);
    }
    #[test]
    fn token_piece_filter_uses_bytes_and_end_state() {
        let g = Grammar::parse("root ::= \"ab\"").unwrap();
        let mut r = g.recognizer().unwrap();
        assert_eq!(
            r.allowed_tokens(&[(1, b"a"), (2, b"b"), (3, b"")], &[3])
                .unwrap(),
            vec![1]
        );
        assert!(r.accept_bytes(b"ab").unwrap());
        assert_eq!(
            r.allowed_tokens(&[(1, b"a"), (3, b"")], &[3]).unwrap(),
            vec![3]
        );
    }
}
