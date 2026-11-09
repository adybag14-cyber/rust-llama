//! Native-compatible GBNF parsing and token filtering over raw bytes.
//!
//! Grammar rules and transition semantics follow the frozen llama.cpp
//! llama-grammar.cpp (copyright 2023-2026 the llama.cpp authors, MIT).
//! This is deliberately separate from the Unicode-scalar grammar API. The
//! original incremental decoder accepts some non-scalar values and treats
//! within-piece and cross-piece continuation bytes differently.
//!
//! State changes are transactional on errors. Explicit resource limits bound
//! grammar construction and pushdown-state expansion; no foreign code is used.

use crate::{Error, Result};
use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

const REPEAT_LIMIT: u64 = 2_000;
const RECURSION_LIMIT: usize = 256;

/// Resource bounds in addition to the original parser's repetition and
/// recursion limits. Larger trusted grammars can supply explicit limits.
#[derive(Clone, Copy, Debug)]
pub struct NativeGrammarLimits {
    /// Input bytes and, independently, the total bytes of interned rule names.
    pub max_source_bytes: usize,
    pub max_rules: usize,
    /// Stored rule elements and total parser element construction/copy work.
    /// The latter also bounds temporary vectors from incomplete nested rules.
    pub max_elements: usize,
    pub max_stacks: usize,
    pub max_stack_depth: usize,
    pub max_state_elements: usize,
    pub max_piece_bytes: usize,
    pub max_steps: usize,
}
impl Default for NativeGrammarLimits {
    fn default() -> Self {
        Self {
            max_source_bytes: 8 * 1024 * 1024,
            max_rules: 65_536,
            max_elements: 1_048_576,
            max_stacks: 16_384,
            max_stack_depth: 4_096,
            max_state_elements: 1_048_576,
            max_piece_bytes: 1_048_576,
            max_steps: 8_388_608,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Element {
    End,
    Alternate,
    Rule(usize),
    Character(u32),
    NotCharacter(u32),
    RangeUpper(u32),
    CharacterAlternative(u32),
    AnyCharacter,
    Token(u32),
    NotToken(u32),
}
impl Element {
    fn ends_sequence(self) -> bool {
        matches!(self, Self::End | Self::Alternate)
    }
    fn token_matches(self, token: i32) -> Option<bool> {
        match self {
            Self::Token(id) => Some(id == token as u32),
            Self::NotToken(id) => Some(id != token as u32),
            _ => None,
        }
    }
    fn character_value(self) -> u32 {
        match self {
            Self::Character(v) | Self::NotCharacter(v) | Self::CharacterAlternative(v) => v,
            _ => 0,
        }
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct Position {
    rule: usize,
    index: usize,
}
type Stack = Vec<Position>;

#[derive(Clone, Debug, Default)]
struct States {
    values: BTreeSet<Stack>,
    elements: usize,
}
impl States {
    fn insert(&mut self, stack: Stack, limits: NativeGrammarLimits) -> Result<bool> {
        if self.values.contains(&stack) {
            return Ok(false);
        }
        if stack.len() > limits.max_stack_depth || self.values.len() >= limits.max_stacks {
            return Err(Error::new("native grammar stack limit exceeded"));
        }
        let elements = self
            .elements
            .checked_add(stack.len())
            .ok_or_else(|| Error::new("native grammar state size overflow"))?;
        if elements > limits.max_state_elements {
            return Err(Error::new("native grammar state memory limit exceeded"));
        }
        self.values.insert(stack);
        self.elements = elements;
        Ok(true)
    }
}
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct PartialUtf8 {
    value: u32,
    remaining: i32,
}
#[derive(Debug)]
struct Decoded {
    // The final zero is a sentinel; real decoded U+0000 values can precede it.
    points: Vec<u32>,
    partial: PartialUtf8,
}
#[derive(Debug)]
struct Compiled {
    rules: Vec<Vec<Element>>,
    alternatives: Vec<Vec<usize>>,
    initial: States,
    limits: NativeGrammarLimits,
}
#[derive(Clone, Debug)]
pub struct NativeGrammar {
    compiled: Arc<Compiled>,
    states: States,
    partial: PartialUtf8,
}
struct Budget(usize);
impl Budget {
    fn spend(&mut self) -> Result<()> {
        self.0 = self
            .0
            .checked_sub(1)
            .ok_or_else(|| Error::new("native grammar work limit exceeded"))?;
        Ok(())
    }
}

impl NativeGrammar {
    /// Parse GBNF. The resolver receives the entire spelling of a named token,
    /// including angle brackets, and returns its single token ID. Returning
    /// None rejects a spelling which does not tokenize to exactly one token.
    /// Numeric terminals do not call the resolver.
    pub fn parse(
        source: &str,
        root: &str,
        resolve_named_token: impl FnMut(&str) -> Option<i32>,
    ) -> Result<Self> {
        Self::parse_with_limits(
            source,
            root,
            resolve_named_token,
            NativeGrammarLimits::default(),
        )
    }

    pub fn parse_with_limits(
        source: &str,
        root: &str,
        resolve_named_token: impl FnMut(&str) -> Option<i32>,
        limits: NativeGrammarLimits,
    ) -> Result<Self> {
        if source.len() > limits.max_source_bytes {
            return Err(Error::new("native grammar source limit exceeded"));
        }
        let (rules, root) = Parser::new(source, resolve_named_token, limits).parse(root)?;
        check_left_recursion(&rules, limits)?;
        let alternatives = rules
            .iter()
            .map(|rule| {
                std::iter::once(0)
                    .chain(rule.iter().enumerate().filter_map(|(i, element)| {
                        (*element == Element::Alternate).then_some(i + 1)
                    }))
                    .collect()
            })
            .collect();
        let mut compiled = Compiled {
            rules,
            alternatives,
            initial: States::default(),
            limits,
        };
        let mut initial = States::default();
        let mut budget = Budget(limits.max_steps);
        for &index in &compiled.alternatives[root] {
            let mut stack = Vec::new();
            if !compiled.rules[root][index].ends_sequence() {
                stack.push(Position { rule: root, index });
            }
            compiled.expand(stack, &mut initial, &mut budget)?;
        }
        if initial.values.is_empty() {
            return Err(Error::new("native grammar has no initial states"));
        }
        compiled.initial = initial.clone();
        Ok(Self {
            compiled: Arc::new(compiled),
            states: initial,
            partial: PartialUtf8::default(),
        })
    }

    /// Whether at least one native pushdown stack can end now.
    /// This intentionally follows the source EOG rule independently of the
    /// partial-byte buffer.
    pub fn is_accepting(&self) -> bool {
        self.states.values.contains(&Vec::new())
    }

    pub fn reset(&mut self) {
        self.states = self.compiled.initial.clone();
        self.partial = PartialUtf8::default();
    }

    /// Test one candidate without changing state. The piece is raw bytes;
    /// callers performing lazy replay may pass a suffix of the original piece
    /// while retaining the original token ID.
    pub fn allows(&self, token: i32, piece: &[u8], is_eog: bool) -> Result<bool> {
        if is_eog {
            return Ok(self.is_accepting());
        }
        if piece.is_empty() || piece[0] == 0 {
            return Ok(false);
        }
        let decoded = decode_piece(piece, self.partial, self.compiled.limits)?;
        let mut current = self.states.clone();
        let mut budget = Budget(self.compiled.limits.max_steps);
        for &point in &decoded.points {
            let mut following = States::default();
            for stack in &current.values {
                budget.spend()?;
                let Some(&position) = stack.last() else {
                    if point == 0 && decoded.partial.remaining == 0 {
                        return Ok(true);
                    }
                    continue;
                };
                let element = self.compiled.element(position);
                if let Some(matches) = element.token_matches(token) {
                    // A token terminal reached after character consumption
                    // matches the original ID without examining the suffix.
                    if (point == 0 && decoded.partial.remaining == 0) || (point != 0 && matches) {
                        return Ok(true);
                    }
                    continue;
                }
                if point == 0 {
                    if decoded.partial.remaining == 0
                        || self.compiled.matches_partial(position, decoded.partial)
                    {
                        return Ok(true);
                    }
                } else {
                    let (matches, after) = self.compiled.matches_character(position, point);
                    if matches {
                        self.compiled
                            .advance_after(stack, after, &mut following, &mut budget)?;
                    }
                }
            }
            if following.values.is_empty() {
                return Ok(false);
            }
            current = following;
        }
        Ok(false)
    }

    /// Consume using the original token acceptance transition. If a piece
    /// cannot leave any live stack, the error leaves the previous state intact.
    /// Token and character acceptance intentionally remain distinct from
    /// candidate masking; the original source uses different transitions.
    pub fn accept(&mut self, token: i32, piece: &[u8], is_eog: bool) -> Result<()> {
        if is_eog {
            return if self.is_accepting() {
                Ok(())
            } else {
                Err(Error::new(
                    "native grammar does not accept an end-of-generation token",
                ))
            };
        }
        let decoded = decode_piece(piece, self.partial, self.compiled.limits)?;
        let mut budget = Budget(self.compiled.limits.max_steps);
        let mut following = States::default();
        let mut characters = States::default();
        for stack in &self.states.values {
            budget.spend()?;
            let Some(&position) = stack.last() else {
                continue;
            };
            if let Some(matches) = self.compiled.element(position).token_matches(token) {
                if matches {
                    self.compiled.advance_after(
                        stack,
                        Position {
                            index: position.index + 1,
                            ..position
                        },
                        &mut following,
                        &mut budget,
                    )?;
                }
            } else {
                characters.insert(stack.clone(), self.compiled.limits)?;
            }
        }
        for &point in &decoded.points[..decoded.points.len() - 1] {
            let mut next = States::default();
            for stack in &characters.values {
                budget.spend()?;
                let Some(&position) = stack.last() else {
                    continue;
                };
                if self
                    .compiled
                    .element(position)
                    .token_matches(token)
                    .is_some()
                {
                    continue;
                }
                let (matches, after) = self.compiled.matches_character(position, point);
                if matches {
                    self.compiled
                        .advance_after(stack, after, &mut next, &mut budget)?;
                }
            }
            characters = next;
            if characters.values.is_empty() {
                break;
            }
        }
        for stack in characters.values {
            following.insert(stack, self.compiled.limits)?;
        }
        if following.values.is_empty() {
            return Err(Error::new(
                "piece leaves no accepting native grammar prefix",
            ));
        }
        self.states = following;
        self.partial = decoded.partial;
        Ok(())
    }
}
impl Compiled {
    fn element(&self, position: Position) -> Element {
        self.rules[position.rule][position.index]
    }
    fn advance_after(
        &self,
        stack: &[Position],
        after: Position,
        output: &mut States,
        budget: &mut Budget,
    ) -> Result<()> {
        let mut next = stack[..stack.len() - 1].to_vec();
        if !self.element(after).ends_sequence() {
            next.push(after);
        }
        self.expand(next, output, budget)
    }
    fn expand(&self, stack: Stack, output: &mut States, budget: &mut Budget) -> Result<()> {
        let mut pending = vec![stack];
        let mut pending_elements = pending[0].len();
        let mut seen = States::default();
        while let Some(stack) = pending.pop() {
            budget.spend()?;
            pending_elements -= stack.len();
            if !seen.insert(stack.clone(), self.limits)? {
                continue;
            }
            let Some(&position) = stack.last() else {
                output.insert(stack, self.limits)?;
                continue;
            };
            if let Element::Rule(rule) = self.element(position) {
                for &index in &self.alternatives[rule] {
                    let mut next = stack[..stack.len() - 1].to_vec();
                    let after = Position {
                        index: position.index + 1,
                        ..position
                    };
                    if !self.element(after).ends_sequence() {
                        next.push(after);
                    }
                    if !self.rules[rule][index].ends_sequence() {
                        next.push(Position { rule, index });
                    }
                    pending_elements = pending_elements
                        .checked_add(next.len())
                        .ok_or_else(|| Error::new("native grammar pending state size overflow"))?;
                    if pending.len() >= self.limits.max_stacks
                        || next.len() > self.limits.max_stack_depth
                        || pending_elements > self.limits.max_state_elements
                    {
                        return Err(Error::new("native grammar pending state limit exceeded"));
                    }
                    pending.push(next);
                }
            } else {
                output.insert(stack, self.limits)?;
            }
        }
        Ok(())
    }
    fn matches_character(&self, position: Position, point: u32) -> (bool, Position) {
        let rule = &self.rules[position.rule];
        let positive = matches!(
            rule[position.index],
            Element::Character(_) | Element::AnyCharacter
        );
        let mut index = position.index;
        let mut found = false;
        loop {
            if let Element::RangeUpper(high) = rule[index + 1] {
                found |= rule[index].character_value() <= point && point <= high;
                index += 2;
            } else {
                found |=
                    rule[index] == Element::AnyCharacter || rule[index].character_value() == point;
                index += 1;
            }
            if !matches!(rule[index], Element::CharacterAlternative(_)) {
                break;
            }
        }
        (found == positive, Position { index, ..position })
    }
    fn matches_partial(&self, position: Position, partial: PartialUtf8) -> bool {
        if partial.remaining < 0 || (partial.remaining == 1 && partial.value < 2) {
            return false;
        }
        let shift = partial.remaining as u32 * 6;
        let mut low = partial.value << shift;
        let high = low | ((1_u32 << shift) - 1);
        if low == 0 {
            if partial.remaining == 2 {
                low = 1 << 11;
            } else if partial.remaining == 3 {
                low = 1 << 16;
            }
        }
        let rule = &self.rules[position.rule];
        let positive = matches!(
            rule[position.index],
            Element::Character(_) | Element::AnyCharacter
        );
        let mut index = position.index;
        loop {
            if let Element::RangeUpper(end) = rule[index + 1] {
                if rule[index].character_value() <= high && low <= end {
                    return positive;
                }
                index += 2;
            } else if rule[index] == Element::AnyCharacter {
                return true;
            } else {
                if low <= rule[index].character_value() && rule[index].character_value() <= high {
                    return positive;
                }
                index += 1;
            }
            if !matches!(rule[index], Element::CharacterAlternative(_)) {
                break;
            }
        }
        !positive
    }
}

fn decode_piece(piece: &[u8], start: PartialUtf8, limits: NativeGrammarLimits) -> Result<Decoded> {
    if piece.len() > limits.max_piece_bytes {
        return Err(Error::new("native grammar token piece limit exceeded"));
    }
    let piece = &piece[..piece.iter().position(|&b| b == 0).unwrap_or(piece.len())];
    let mut points = Vec::new();
    points
        .try_reserve(
            piece
                .len()
                .checked_add(1)
                .ok_or_else(|| Error::new("native grammar decoded size overflow"))?,
        )
        .map_err(|_| Error::new("native grammar decoded allocation failed"))?;
    let mut position = 0;
    let mut value = start.value;
    let mut remaining = start.remaining;
    while position < piece.len() && remaining > 0 {
        let byte = piece[position];
        if byte >> 6 != 2 {
            points.push(0);
            return Ok(Decoded {
                points,
                partial: PartialUtf8 {
                    value: 0,
                    remaining: -1,
                },
            });
        }
        value = (value << 6) + (byte & 63) as u32;
        position += 1;
        remaining -= 1;
    }
    if start.remaining > 0 && remaining == 0 {
        points.push(value);
    }
    while position < piece.len() {
        let first = piece[position];
        remaining = match first >> 4 {
            0..=7 => 0,
            8..=11 => -1,
            12..=13 => 1,
            14 => 2,
            _ => 3,
        };
        if remaining < 0 {
            points.clear();
            points.push(0);
            return Ok(Decoded {
                points,
                partial: PartialUtf8 {
                    value: 0,
                    remaining,
                },
            });
        }
        value = (first & ((1_u8 << (7 - remaining)) - 1)) as u32;
        position += 1;
        while position < piece.len() && remaining > 0 {
            // The original only checks continuation tags when finishing a
            // partial sequence from a prior piece, not inside a new piece.
            value = (value << 6) + (piece[position] & 63) as u32;
            position += 1;
            remaining -= 1;
        }
        if remaining == 0 {
            points.push(value);
        }
    }
    points.push(0);
    Ok(Decoded {
        points,
        partial: PartialUtf8 { value, remaining },
    })
}

struct Parser<'a, F> {
    bytes: &'a [u8],
    at: usize,
    symbols: BTreeMap<String, usize>,
    rules: Vec<Vec<Element>>,
    elements: usize,
    construction_elements: usize,
    symbol_name_bytes: usize,
    resolver: F,
    limits: NativeGrammarLimits,
}
impl<'a, F: FnMut(&str) -> Option<i32>> Parser<'a, F> {
    fn new(source: &'a str, resolver: F, limits: NativeGrammarLimits) -> Self {
        let bytes = source.as_bytes();
        Self {
            bytes: &bytes[..bytes.iter().position(|&v| v == 0).unwrap_or(bytes.len())],
            at: 0,
            symbols: BTreeMap::new(),
            rules: Vec::new(),
            elements: 0,
            construction_elements: 0,
            symbol_name_bytes: 0,
            resolver,
            limits,
        }
    }
    fn error(&self, message: &str) -> Error {
        Error::new(format!("{message} at grammar byte {}", self.at))
    }
    fn peek(&self) -> u8 {
        self.bytes.get(self.at).copied().unwrap_or(0)
    }
    fn space(&mut self, newline: bool) {
        loop {
            match self.peek() {
                b' ' | b'\t' => self.at += 1,
                b'\r' | b'\n' if newline => self.at += 1,
                b'#' => {
                    while !matches!(self.peek(), 0 | b'\r' | b'\n') {
                        self.at += 1;
                    }
                }
                _ => break,
            }
        }
    }
    fn name(&mut self) -> Result<String> {
        let start = self.at;
        while word(self.peek()) {
            self.at += 1;
        }
        if self.at == start {
            return Err(self.error("expected rule name"));
        }
        Ok(std::str::from_utf8(&self.bytes[start..self.at])
            .expect("ASCII name")
            .to_owned())
    }
    fn symbol(&mut self, name: &str) -> Result<usize> {
        if let Some(&id) = self.symbols.get(name) {
            return Ok(id);
        }
        if self.rules.len() >= self.limits.max_rules {
            return Err(self.error("native grammar rule limit exceeded"));
        }
        let name_bytes = self
            .symbol_name_bytes
            .checked_add(name.len())
            .ok_or_else(|| self.error("native grammar symbol name size overflow"))?;
        if name_bytes > self.limits.max_source_bytes {
            return Err(self.error("native grammar symbol name byte limit exceeded"));
        }
        let id = self.rules.len();
        self.rules
            .try_reserve(1)
            .map_err(|_| self.error("native grammar rule allocation failed"))?;
        self.rules.push(Vec::new());
        self.symbols.insert(name.to_owned(), id);
        self.symbol_name_bytes = name_bytes;
        Ok(id)
    }
    fn generated(&mut self, base: &str) -> Result<usize> {
        // Check the temporary name before formatting, as well as its interned
        // copy in symbol(). Existing source symbols may reuse a generated name.
        let digits = self.symbols.len().checked_ilog10().unwrap_or(0) as usize + 1;
        let size = base
            .len()
            .checked_add(1)
            .and_then(|n| n.checked_add(digits))
            .ok_or_else(|| self.error("native grammar generated name size overflow"))?;
        if size > self.limits.max_source_bytes {
            return Err(self.error("native grammar generated name byte limit exceeded"));
        }
        self.symbol(&format!("{base}_{}", self.symbols.len()))
    }
    fn charge_construction(&mut self, count: usize) -> Result<()> {
        let next = self
            .construction_elements
            .checked_add(count)
            .ok_or_else(|| self.error("native grammar construction size overflow"))?;
        if next > self.limits.max_elements {
            return Err(self.error("native grammar construction element limit exceeded"));
        }
        self.construction_elements = next;
        Ok(())
    }
    fn copy_elements(&mut self, elements: &[Element]) -> Result<Vec<Element>> {
        self.charge_construction(elements.len())?;
        let mut result = Vec::new();
        result
            .try_reserve_exact(elements.len())
            .map_err(|_| self.error("native grammar element copy allocation failed"))?;
        result.extend_from_slice(elements);
        Ok(result)
    }
    fn push(&mut self, rule: &mut Vec<Element>, element: Element) -> Result<()> {
        if rule.len() >= self.limits.max_elements {
            return Err(self.error("native grammar element limit exceeded"));
        }
        self.charge_construction(1)?;
        rule.try_reserve(1)
            .map_err(|_| self.error("native grammar element allocation failed"))?;
        rule.push(element);
        Ok(())
    }
    fn extend(&mut self, rule: &mut Vec<Element>, elements: &[Element]) -> Result<()> {
        if rule
            .len()
            .checked_add(elements.len())
            .is_none_or(|n| n > self.limits.max_elements)
        {
            return Err(self.error("native grammar element limit exceeded"));
        }
        self.charge_construction(elements.len())?;
        rule.try_reserve(elements.len())
            .map_err(|_| self.error("native grammar element allocation failed"))?;
        rule.extend_from_slice(elements);
        Ok(())
    }
    fn add_rule(&mut self, id: usize, rule: Vec<Element>) -> Result<()> {
        self.elements = self
            .elements
            .checked_sub(self.rules[id].len())
            .and_then(|n| n.checked_add(rule.len()))
            .ok_or_else(|| self.error("native grammar element count overflow"))?;
        if self.elements > self.limits.max_elements {
            return Err(self.error("native grammar total element limit exceeded"));
        }
        self.rules[id] = rule;
        Ok(())
    }
    fn parse(mut self, root: &str) -> Result<(Vec<Vec<Element>>, usize)> {
        self.space(true);
        while self.peek() != 0 {
            let name = self.name()?;
            self.space(false);
            let id = self.symbol(&name)?;
            if self.bytes.get(self.at..self.at + 3) != Some(b"::=") {
                return Err(self.error("expected ::="));
            }
            self.at += 3;
            self.space(true);
            self.alternates(&name, id, false, 0)?;
            match self.peek() {
                b'\r' => {
                    self.at += 1;
                    if self.peek() == b'\n' {
                        self.at += 1;
                    }
                }
                b'\n' => self.at += 1,
                0 => {}
                _ => return Err(self.error("expected newline or grammar end")),
            }
            self.space(true);
        }
        if self.rules.is_empty() {
            return Err(self.error("empty native grammar"));
        }
        for rule in &self.rules {
            if rule.is_empty() {
                return Err(self.error("undefined native grammar rule"));
            }
            for element in rule {
                if let Element::Rule(id) = element
                    && self.rules.get(*id).is_none_or(Vec::is_empty)
                {
                    return Err(self.error("undefined native grammar reference"));
                }
            }
        }
        let root = *self
            .symbols
            .get(root)
            .ok_or_else(|| self.error("missing native grammar root"))?;
        Ok((self.rules, root))
    }
    fn alternates(&mut self, name: &str, id: usize, nested: bool, depth: usize) -> Result<()> {
        let mut rule = self.sequence(name, nested, depth)?;
        while self.peek() == b'|' {
            self.push(&mut rule, Element::Alternate)?;
            self.at += 1;
            self.space(true);
            let alternative = self.sequence(name, nested, depth)?;
            self.extend(&mut rule, &alternative)?;
        }
        self.push(&mut rule, Element::End)?;
        self.add_rule(id, rule)
    }
    fn sequence(&mut self, name: &str, nested: bool, depth: usize) -> Result<Vec<Element>> {
        let mut rule = Vec::new();
        let mut last = 0;
        let mut previous_rules = 1_u64;
        loop {
            match self.peek() {
                b'"' => {
                    self.at += 1;
                    last = rule.len();
                    previous_rules = 1;
                    while self.peek() != b'"' {
                        let value = self.character()?;
                        self.push(&mut rule, Element::Character(value))?;
                    }
                    self.at += 1;
                    self.space(nested);
                }
                b'[' => {
                    self.at += 1;
                    let inverse = self.peek() == b'^';
                    if inverse {
                        self.at += 1;
                    }
                    last = rule.len();
                    previous_rules = 1;
                    while self.peek() != b']' {
                        let value = self.character()?;
                        let continuation = last < rule.len();
                        self.push(
                            &mut rule,
                            if continuation {
                                Element::CharacterAlternative(value)
                            } else if inverse {
                                Element::NotCharacter(value)
                            } else {
                                Element::Character(value)
                            },
                        )?;
                        if self.peek() == b'-' && self.bytes.get(self.at + 1).copied() != Some(b']')
                        {
                            self.at += 1;
                            let end = self.character()?;
                            self.push(&mut rule, Element::RangeUpper(end))?;
                        }
                    }
                    self.at += 1;
                    self.space(nested);
                }
                b'<' | b'!' => {
                    let inverse = self.peek() == b'!';
                    if inverse {
                        self.at += 1;
                    }
                    let id = self.token()?;
                    last = rule.len();
                    previous_rules = 1;
                    self.push(
                        &mut rule,
                        if inverse {
                            Element::NotToken(id)
                        } else {
                            Element::Token(id)
                        },
                    )?;
                    self.space(nested);
                }
                byte if word(byte) => {
                    let reference = self.name()?;
                    let id = self.symbol(&reference)?;
                    last = rule.len();
                    previous_rules = 1;
                    self.push(&mut rule, Element::Rule(id))?;
                    self.space(nested);
                }
                b'(' => {
                    if depth >= RECURSION_LIMIT {
                        return Err(self.error("native grammar nesting exceeds 256"));
                    }
                    self.at += 1;
                    self.space(true);
                    let before = self.symbols.len();
                    let id = self.generated(name)?;
                    self.alternates(name, id, true, depth + 1)?;
                    previous_rules = (self.symbols.len() - before).max(1) as u64;
                    last = rule.len();
                    self.push(&mut rule, Element::Rule(id))?;
                    if self.peek() != b')' {
                        return Err(self.error("expected )"));
                    }
                    self.at += 1;
                    self.space(nested);
                }
                b'.' => {
                    last = rule.len();
                    previous_rules = 1;
                    self.push(&mut rule, Element::AnyCharacter)?;
                    self.at += 1;
                    self.space(nested);
                }
                b'*' | b'+' | b'?' => {
                    let kind = self.peek();
                    self.at += 1;
                    self.space(nested);
                    let minimum = u64::from(kind == b'+');
                    let maximum = (kind == b'?').then_some(1);
                    self.repetitions(name, &mut rule, last, &mut previous_rules, minimum, maximum)?;
                }
                b'{' => {
                    self.at += 1;
                    self.space(nested);
                    let minimum = self.integer()?;
                    self.space(nested);
                    let mut maximum = None;
                    if self.peek() == b'}' {
                        maximum = Some(minimum);
                        self.at += 1;
                        self.space(nested);
                    } else if self.peek() == b',' {
                        self.at += 1;
                        self.space(nested);
                        if self.peek().is_ascii_digit() {
                            maximum = Some(self.integer()?);
                            self.space(nested);
                        }
                        if self.peek() != b'}' {
                            return Err(self.error("expected }"));
                        }
                        self.at += 1;
                        self.space(nested);
                    } else {
                        return Err(self.error("expected , or }"));
                    }
                    if minimum > REPEAT_LIMIT {
                        return Err(self.error("native grammar repetition exceeds 2000"));
                    }
                    if maximum.is_some_and(|value| value > REPEAT_LIMIT) {
                        maximum = None;
                    }
                    self.repetitions(name, &mut rule, last, &mut previous_rules, minimum, maximum)?;
                }
                _ => break,
            }
        }
        Ok(rule)
    }
    #[allow(clippy::too_many_arguments)] // Mirrors one bounded repetition transformation.
    fn repetitions(
        &mut self,
        name: &str,
        rule: &mut Vec<Element>,
        last: usize,
        previous_rules: &mut u64,
        minimum: u64,
        maximum: Option<u64>,
    ) -> Result<()> {
        if last == rule.len() {
            return Err(self.error("repetition has no preceding item"));
        }
        if maximum.is_some_and(|value| value < minimum) {
            return Err(self.error("repetition maximum is smaller than its minimum"));
        }
        let total = maximum.filter(|&n| n > 0).unwrap_or(minimum.max(1));
        let product = previous_rules
            .checked_mul(total)
            .ok_or_else(|| self.error("native grammar repetition complexity overflow"))?;
        if product > REPEAT_LIMIT {
            return Err(self.error("native grammar repetition complexity exceeds 2000"));
        }
        let previous = self.copy_elements(&rule[last..])?;
        if minimum == 0 {
            rule.truncate(last);
        } else {
            for _ in 1..minimum {
                self.extend(rule, &previous)?;
            }
        }
        let optional = maximum.map_or(1, |value| value - minimum);
        let mut previous_id = None;
        for i in 0..optional {
            let id = self.generated(name)?;
            let mut repeated = self.copy_elements(&previous)?;
            if i > 0 || maximum.is_none() {
                self.push(
                    &mut repeated,
                    Element::Rule(if maximum.is_none() {
                        id
                    } else {
                        previous_id.expect("prior repeat")
                    }),
                )?;
            }
            self.push(&mut repeated, Element::Alternate)?;
            self.push(&mut repeated, Element::End)?;
            self.add_rule(id, repeated)?;
            previous_id = Some(id);
        }
        if let Some(id) = previous_id {
            self.push(rule, Element::Rule(id))?;
        }
        *previous_rules = product;
        Ok(())
    }
    fn integer(&mut self) -> Result<u64> {
        if !self.peek().is_ascii_digit() {
            return Err(self.error("expected integer"));
        }
        let mut value = 0_u64;
        while self.peek().is_ascii_digit() {
            value = value
                .checked_mul(10)
                .and_then(|n| n.checked_add((self.peek() - b'0') as u64))
                .ok_or_else(|| self.error("grammar integer overflow"))?;
            self.at += 1;
        }
        Ok(value)
    }
    fn token(&mut self) -> Result<u32> {
        let start = self.at;
        if self.peek() != b'<' {
            return Err(self.error("expected <"));
        }
        self.at += 1;
        if self.peek() == b'[' {
            self.at += 1;
            let id = self.integer()?;
            let id = u32::try_from(id).map_err(|_| self.error("token ID exceeds uint32"))?;
            if self.peek() != b']' {
                return Err(self.error("expected ] in token terminal"));
            }
            self.at += 1;
            if self.peek() != b'>' {
                return Err(self.error("expected > in token terminal"));
            }
            self.at += 1;
            return Ok(id);
        }
        while !matches!(self.peek(), 0 | b'>') {
            self.at += 1;
        }
        if self.peek() != b'>' {
            return Err(self.error("expected > in named token"));
        }
        self.at += 1;
        let name = std::str::from_utf8(&self.bytes[start..self.at]).expect("source is UTF-8");
        (self.resolver)(name)
            .map(|id| id as u32)
            .ok_or_else(|| self.error("named token must resolve to exactly one token"))
    }
    fn character(&mut self) -> Result<u32> {
        if self.peek() == 0 {
            return Err(self.error("unexpected grammar end"));
        }
        if self.peek() == b'\\' {
            self.at += 1;
            let escaped = self.peek();
            self.at += 1;
            return match escaped {
                b'x' => self.hexadecimal(2),
                b'u' => self.hexadecimal(4),
                b'U' => self.hexadecimal(8),
                b't' => Ok(b'\t' as u32),
                b'r' => Ok(b'\r' as u32),
                b'n' => Ok(b'\n' as u32),
                b'\\' | b'"' | b'[' | b']' | b'-' => Ok(escaped as u32),
                _ => Err(self.error("unknown grammar escape")),
            };
        }
        let first = self.peek();
        let length = match first >> 4 {
            12..=13 => 2,
            14 => 3,
            15 => 4,
            _ => 1,
        };
        let mut value = (first & ((1_u8 << (8 - length)) - 1)) as u32;
        self.at += 1;
        for _ in 1..length {
            if self.peek() == 0 {
                break;
            }
            value = (value << 6) + (self.peek() & 63) as u32;
            self.at += 1;
        }
        Ok(value)
    }
    fn hexadecimal(&mut self, digits: usize) -> Result<u32> {
        let mut value = 0_u32;
        for _ in 0..digits {
            let digit = (self.peek() as char)
                .to_digit(16)
                .ok_or_else(|| self.error("incomplete hexadecimal grammar escape"))?;
            value = (value << 4) + digit;
            self.at += 1;
        }
        Ok(value)
    }
}
fn word(value: u8) -> bool {
    value.is_ascii_alphanumeric() || value == b'-'
}
fn check_left_recursion(rules: &[Vec<Element>], limits: NativeGrammarLimits) -> Result<()> {
    struct Visit<'a> {
        rules: &'a [Vec<Element>],
        visited: Vec<bool>,
        active: Vec<bool>,
        empty: Vec<bool>,
        budget: Budget,
    }
    impl Visit<'_> {
        fn rule(&mut self, id: usize, depth: usize) -> Result<()> {
            self.budget.spend()?;
            if self.active[id] {
                return Err(Error::new("native grammar has left recursion"));
            }
            if depth > RECURSION_LIMIT {
                return Err(Error::new("native grammar rule chain exceeds 256"));
            }
            self.active[id] = true;
            let mut at_start = true;
            for element in &self.rules[id] {
                if element.ends_sequence() {
                    if at_start {
                        self.empty[id] = true;
                        break;
                    }
                    at_start = true;
                } else {
                    at_start = false;
                }
            }
            let mut inspect_reference = true;
            for index in 0..self.rules[id].len() {
                let element = self.rules[id][index];
                if let Element::Rule(next) = element {
                    if inspect_reference {
                        self.rule(next, depth + 1)?;
                        if !self.empty[next] {
                            inspect_reference = false;
                        }
                    }
                } else {
                    inspect_reference = element.ends_sequence();
                }
            }
            self.active[id] = false;
            self.visited[id] = true;
            Ok(())
        }
    }
    let mut visit = Visit {
        rules,
        visited: vec![false; rules.len()],
        active: vec![false; rules.len()],
        empty: vec![false; rules.len()],
        budget: Budget(limits.max_steps),
    };
    for id in 0..rules.len() {
        if !visit.visited[id] {
            visit.rule(id, 0)?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    fn grammar(source: &str) -> NativeGrammar {
        NativeGrammar::parse(source, "root", |_| None).unwrap()
    }
    #[test]
    fn native_literals_alternatives_and_transactional_accept() {
        let mut g = grammar("root ::= \" \"* (\"a\" | \"ab\" | \"hello\")");
        assert!(g.allows(10, b" hello", false).unwrap());
        assert!(!g.allows(10, b"help", false).unwrap());
        assert!(g.accept(10, b"help", false).is_err());
        assert!(g.allows(10, b"hello", false).unwrap());
        g.accept(10, b"a", false).unwrap();
        assert!(g.is_accepting());
        assert!(g.allows(2, b"", true).unwrap());
        assert!(g.allows(10, b"b", false).unwrap());
        let clone = g.clone();
        g.accept(10, b"b", false).unwrap();
        assert!(!g.allows(10, b"b", false).unwrap());
        assert!(clone.allows(10, b"b", false).unwrap());
        g.reset();
        assert!(!g.is_accepting());
        assert!(g.allows(10, b"hello", false).unwrap());
    }
    #[test]
    fn numeric_named_and_inverse_token_terminals() {
        let mut g = NativeGrammar::parse("item ::= <named> !<[101]> <[4294967295]>", "item", |s| {
            (s == "<named>").then_some(100)
        })
        .unwrap();
        assert!(g.allows(100, b"anything", false).unwrap());
        assert!(!g.allows(99, b"anything", false).unwrap());
        assert!(!g.allows(100, b"", false).unwrap());
        assert!(!g.allows(100, b"\0ignored", false).unwrap());
        g.accept(100, b"anything", false).unwrap();
        assert!(!g.allows(101, b"x", false).unwrap());
        g.accept(100, b"x", false).unwrap();
        g.accept(-1, b"x", false).unwrap();
        assert!(g.is_accepting());
    }
    #[test]
    fn token_masking_and_acceptance_keep_the_source_distinction() {
        let mut g = grammar("root ::= \"a\" <[7]>");
        assert!(g.allows(7, b"ab", false).unwrap());
        assert!(g.accept(7, b"ab", false).is_err());
        assert!(g.allows(99, b"a", false).unwrap());
        g.accept(99, b"a", false).unwrap();
        g.accept(7, b"b", false).unwrap();
        assert!(g.is_accepting());
    }
    #[test]
    fn raw_utf8_prefixes_match_native_non_scalar_behavior() {
        for sequence in [vec![0xf5, 0x80, 0x80, 0x80], vec![0xed, 0xa0, 0x80]] {
            let mut g = grammar("root ::= .");
            for byte in sequence {
                assert!(g.allows(10, &[byte], false).unwrap());
                g.accept(10, &[byte], false).unwrap();
            }
            assert!(g.is_accepting());
        }
        let mut g = grammar("root ::= .");
        for piece in [&b""[..], &b"\0"[..], &[0x80], &[0xc0], &[0xc1]] {
            assert!(!g.allows(10, piece, false).unwrap(), "{piece:?}");
        }
        assert!(g.allows(10, &[0xe2, 0x28, 0xa1], false).unwrap());
        g.accept(10, &[0xe2], false).unwrap();
        assert!(!g.allows(10, &[0x28], false).unwrap());
        assert!(!g.allows(2, b"", true).unwrap());
    }
    #[test]
    fn partial_classes_and_nul_sentinel_are_source_specific() {
        let mut g = grammar("root ::= \"é\"");
        assert!(g.allows(10, &[0xc3], false).unwrap());
        assert!(!g.allows(10, &[0xc2], false).unwrap());
        g.accept(10, &[0xc3], false).unwrap();
        assert!(g.allows(10, &[0xa9], false).unwrap());
        g.accept(10, &[0xa9], false).unwrap();
        assert!(g.is_accepting());
        assert!(
            !grammar("root ::= [^\\u0800]")
                .allows(10, &[0xe0], false)
                .unwrap()
        );
        let g = grammar("root ::= .");
        assert!(g.allows(10, &[0xc0, 0x80], false).unwrap());
        let mut g = grammar("root ::= \"a\"");
        assert!(g.allows(10, b"a\0ignored", false).unwrap());
        g.accept(10, b"a\0ignored", false).unwrap();
        assert!(g.is_accepting());
    }
    #[test]
    fn bounded_repetitions_comments_and_native_limits() {
        let mut g = grammar("# test\r\nroot ::= [a-c]{1,4} # tail\r\n");
        for byte in b"abca" {
            g.accept(10, &[*byte], false).unwrap();
        }
        assert!(g.is_accepting());
        assert!(!g.allows(10, b"a", false).unwrap());
        assert!(
            grammar("root ::= \"a\"{1,3000}")
                .allows(10, &vec![b'a'; 3001], false)
                .unwrap()
        );
        for source in [
            "root ::= \"a\"{2001}",
            "root ::= \"a\"{2,1}",
            "root ::= root \"a\"",
            "root ::= missing",
            "root ::= \"\\q\"",
            "root ::= <[4294967296]>",
            "root ::= a_b",
            "",
        ] {
            assert!(
                NativeGrammar::parse(source, "root", |_| None).is_err(),
                "{source}"
            );
        }
        let source = format!("root ::= {}\"a\"{}", "(".repeat(257), ")".repeat(257));
        assert!(NativeGrammar::parse(&source, "root", |_| None).is_err());
        let limits = NativeGrammarLimits {
            max_piece_bytes: 2,
            ..NativeGrammarLimits::default()
        };
        let mut g =
            NativeGrammar::parse_with_limits("root ::= \"abc\"", "root", |_| None, limits).unwrap();
        assert!(g.allows(10, b"abc", false).is_err());
        assert!(g.accept(10, b"abc", false).is_err());
        assert!(g.allows(10, b"ab", false).unwrap());
    }
    struct Reader<'a> {
        bytes: &'a [u8],
        at: usize,
    }
    impl<'a> Reader<'a> {
        fn take(&mut self, count: usize) -> &'a [u8] {
            let result = &self.bytes[self.at..self.at + count];
            self.at += count;
            result
        }
        fn byte(&mut self) -> u8 {
            self.take(1)[0]
        }
        fn word(&mut self) -> usize {
            u32::from_le_bytes(self.take(4).try_into().unwrap()) as usize
        }
        fn blob(&mut self) -> &'a [u8] {
            let length = self.word();
            self.take(length)
        }
        fn string(&mut self) -> String {
            String::from_utf8(self.blob().to_vec()).unwrap()
        }
        fn token(&mut self) -> (i32, Vec<u8>, bool) {
            let id = i32::from_le_bytes(self.take(4).try_into().unwrap());
            let eog = self.byte() != 0;
            (id, self.blob().to_vec(), eog)
        }
    }
    fn compare_mask(
        grammar: &NativeGrammar,
        candidates: &[(i32, Vec<u8>, bool)],
        expected: &[u8],
        label: &str,
        event: usize,
    ) {
        for (i, (token, piece, eog)) in candidates.iter().enumerate() {
            let actual = grammar
                .allows(*token, piece, *eog)
                .unwrap_or_else(|e| panic!("{label}, event {event}, candidate {i}: {e}"));
            let wanted = (expected[i / 8] >> (i % 8)) & 1 != 0;
            assert_eq!(
                actual,
                wanted,
                "{label}, event {event}, candidate {i}, token {token}, eog {eog}, first bytes {:?}",
                &piece[..piece.len().min(16)]
            );
        }
    }
    #[test]
    fn native_grammar_matches_original_raw_byte_oracle_and_dll_masks() {
        let mut reader = Reader {
            bytes: include_bytes!("../../../fixtures/behavior/native-grammar-reference.bin"),
            at: 0,
        };
        assert_eq!(reader.take(8), b"NGRM0001");
        let candidate_count = reader.word();
        assert_eq!(candidate_count, 1067);
        let candidates: Vec<_> = (0..candidate_count).map(|_| reader.token()).collect();
        let case_count = reader.word();
        assert_eq!(case_count, 82);
        let mut checkpoints = 0;
        let mut accepted = 0;
        let mut rejected = 0;
        for _ in 0..case_count {
            let label = reader.string();
            let source = reader.string();
            let root = reader.string();
            let constructed = reader.byte() != 0;
            let events = reader.word();
            let result = NativeGrammar::parse(&source, &root, |name| match name {
                "<named>" => Some(123),
                "<s>" => Some(1),
                "</s>" => Some(2),
                _ => None,
            });
            assert_eq!(
                result.is_ok(),
                constructed,
                "construction {label}: {result:?}"
            );
            if !constructed {
                assert_eq!(events, 0);
                continue;
            }
            let mut grammar = result.unwrap();
            for event in 0..events {
                let kind = reader.byte();
                match kind {
                    0 | 4 => {}
                    1 | 2 => {
                        let (token, piece, eog) = reader.token();
                        if kind == 1 {
                            grammar
                                .accept(token, &piece, eog)
                                .unwrap_or_else(|e| panic!("{label}, accept event {event}: {e}"));
                            accepted += 1;
                        } else {
                            let states = grammar.states.clone();
                            let partial = grammar.partial;
                            assert!(
                                grammar.accept(token, &piece, eog).is_err(),
                                "{label}, event {event} should fail"
                            );
                            assert_eq!(grammar.states.values, states.values);
                            assert_eq!(grammar.partial, partial);
                            rejected += 1;
                            continue;
                        }
                    }
                    3 => grammar.reset(),
                    _ => panic!("unknown fixture event {kind}"),
                }
                let mask = reader.take(candidate_count.div_ceil(8));
                if kind == 4 {
                    compare_mask(&grammar.clone(), &candidates, mask, &label, event);
                } else {
                    compare_mask(&grammar, &candidates, mask, &label, event);
                }
                checkpoints += 1;
            }
        }
        assert_eq!(checkpoints, 359);
        assert_eq!(accepted, 178);
        assert_eq!(rejected, 2);
        assert_eq!(reader.at, reader.bytes.len());
    }
    #[test]
    fn parser_and_state_resource_errors_are_bounded_and_transactional() {
        let defaults = NativeGrammarLimits::default();
        for limits in [
            NativeGrammarLimits {
                max_source_bytes: 1,
                ..defaults
            },
            NativeGrammarLimits {
                max_rules: 1,
                ..defaults
            },
            NativeGrammarLimits {
                max_elements: 1,
                ..defaults
            },
            NativeGrammarLimits {
                max_stacks: 1,
                ..defaults
            },
            NativeGrammarLimits {
                max_state_elements: 1,
                ..defaults
            },
        ] {
            assert!(
                NativeGrammar::parse_with_limits(
                    "root ::= (\"a\" | \"b\")",
                    "root",
                    |_| None,
                    limits
                )
                .is_err()
            );
        }
        let limits = NativeGrammarLimits {
            max_stack_depth: 1,
            ..defaults
        };
        let mut grammar = NativeGrammar::parse_with_limits(
            "root ::= \"a\" root \"b\" | \"\"",
            "root",
            |_| None,
            limits,
        )
        .unwrap();
        let before = grammar.states.clone();
        assert!(grammar.accept(7, b"a", false).is_err());
        assert_eq!(grammar.states.values, before.values);
        assert_eq!(grammar.partial, PartialUtf8::default());
        let limits = NativeGrammarLimits {
            max_steps: 2,
            ..defaults
        };
        let mut grammar =
            NativeGrammar::parse_with_limits("root ::= \"abcdef\"", "root", |_| None, limits)
                .unwrap();
        let before = grammar.states.clone();
        assert!(grammar.accept(7, b"abcdef", false).is_err());
        assert_eq!(grammar.states.values, before.values);
        // The source's left-recursion check does not propagate nullability
        // through an intermediate reference. Bound its expansion safely.
        let limits = NativeGrammarLimits {
            max_stack_depth: 16,
            ..defaults
        };
        assert!(
            NativeGrammar::parse_with_limits(
                "root ::= a root \"b\" | \"a\"\na ::= b\nb ::= \"\"",
                "root",
                |_| None,
                limits
            )
            .is_err()
        );
    }
}

#[cfg(test)]
mod construction_limits_tests {
    use super::*;

    #[test]
    fn pending_rules_copies_and_interned_names_share_global_bounds() {
        // A discarded optional item still consumed parser construction memory.
        let limits = NativeGrammarLimits {
            max_elements: 12,
            ..NativeGrammarLimits::default()
        };
        let error =
            NativeGrammar::parse_with_limits("root ::= \"abcdefgh\"{0}", "root", |_| None, limits)
                .unwrap_err();
        assert!(
            error.to_string().contains("construction element limit"),
            "{error}"
        );
        // Each local pending sequence is below the limit; their aggregate is not.
        let limits = NativeGrammarLimits {
            max_elements: 32,
            ..NativeGrammarLimits::default()
        };
        let source = "root ::= \"abcdefgh\" (\"abcdefgh\" (\"abcdefgh\" (\"abcdefgh\")))";
        let error = NativeGrammar::parse_with_limits(source, "root", |_| None, limits).unwrap_err();
        assert!(
            error.to_string().contains("construction element limit"),
            "{error}"
        );

        let name = "a".repeat(32);
        let source = format!("{name} ::= (\"a\") (\"b\") (\"c\")");
        let limits = NativeGrammarLimits {
            max_source_bytes: source.len(),
            ..NativeGrammarLimits::default()
        };
        let error = NativeGrammar::parse_with_limits(&source, &name, |_| None, limits).unwrap_err();
        assert!(
            error.to_string().contains("symbol name byte limit"),
            "{error}"
        );

        // The same valid source remains usable with explicitly adequate bounds.
        let grammar = NativeGrammar::parse(&source, &name, |_| None).unwrap();
        assert!(grammar.allows(1, b"abc", false).unwrap());
    }
}
