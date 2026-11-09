//! Frozen pretokenizer split semantics: custom scanners and source ECMAScript syntax.
use crate::{Error, Result, unicode};
use std::collections::HashSet;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Custom {
    Gpt2,
    Llama3,
    Qwen2,
    Qwen35,
    Horizon,
    Kimi,
    RightDigits,
    Newlines,
}
#[derive(Debug, Clone, Copy)]
enum Mode {
    Collapsed,
    Wide,
}
#[derive(Debug, Clone)]
pub(crate) struct Pattern {
    implementation: Implementation,
}
#[derive(Debug, Clone)]
enum Implementation {
    Custom(Custom),
    Generic { mode: Mode, program: Program },
}

include!("regex_constants.rs");
include!("wide_classes.rs");

fn bad(message: impl Into<String>) -> Error {
    Error::Invalid(format!("pretokenizer regex: {}", message.into()))
}
impl Pattern {
    pub(crate) fn compile(expression: &str) -> Result<Self> {
        if let Some(custom) = custom_for(expression) {
            return Ok(Self {
                implementation: Implementation::Custom(custom),
            });
        }
        if expression.len() > 16384 {
            return Err(Error::Limit(
                "pretokenizer expression exceeds 16 KiB".into(),
            ));
        }
        let collapsed = [
            r"\p{N}", r"\p{L}", r"\p{P}", r"\p{M}", r"\p{S}", r"\p{Lu}", r"\p{Ll}", r"\p{Lt}",
            r"\p{Lm}", r"\p{Lo}",
        ]
        .iter()
        .any(|property| expression.contains(property));
        let mode = if collapsed {
            Mode::Collapsed
        } else {
            Mode::Wide
        };
        if collapsed && !expression.is_ascii() {
            return Err(bad(
                "source does not support Unicode categories and non-ASCII literals together",
            ));
        }
        let input = expression
            .chars()
            .map(|ch| {
                let cp = u32::from(ch);
                if matches!(mode, Mode::Wide) {
                    wide(cp)
                } else {
                    cp
                }
            })
            .collect();
        let mut parser = Parser {
            input,
            offset: 0,
            depth: 0,
            wide: matches!(mode, Mode::Wide),
        };
        let node = parser.alternation()?;
        if parser.offset != parser.input.len() {
            return Err(bad("unmatched closing parenthesis"));
        }
        Ok(Self {
            implementation: Implementation::Generic {
                mode,
                program: Program::compile(&node)?,
            },
        })
    }
    fn offsets(
        &self,
        original: &[u32],
        offsets: &[usize],
        budget: &mut Budget,
    ) -> Result<Vec<usize>> {
        match &self.implementation {
            Implementation::Custom(custom) => Ok(custom_offsets(original, offsets, *custom)),
            Implementation::Generic { mode, program } => {
                let text: Vec<u32> = original
                    .iter()
                    .copied()
                    .map(|cp| match mode {
                        Mode::Collapsed => collapse(cp),
                        Mode::Wide => {
                            let cp = wide(cp);
                            if cp > 0x7f && unicode::flags(cp).contains(unicode::Flags::WHITESPACE)
                            {
                                11
                            } else {
                                cp
                            }
                        }
                    })
                    .collect();
                let mut output = Vec::new();
                let mut begin = 0usize;
                for &len in offsets {
                    let end = begin
                        .checked_add(len)
                        .ok_or_else(|| bad("split offset overflow"))?;
                    let fragment = text
                        .get(begin..end)
                        .ok_or_else(|| bad("split offset outside input"))?;
                    program.split(fragment, &mut output, budget)?;
                    begin = end;
                }
                if begin != text.len() {
                    return Err(bad("split offsets do not cover input"));
                }
                Ok(output)
            }
        }
    }
}
fn wide(cp: u32) -> u32 {
    if cfg!(windows) {
        u32::from(cp as u16)
    } else {
        cp
    }
}
fn collapse(cp: u32) -> u32 {
    if cp < 128 {
        return cp;
    }
    let flags = unicode::flags(cp);
    if flags.contains(unicode::Flags::WHITESPACE) {
        return 11;
    }
    match flags.category() {
        unicode::Flags::NUMBER => 0xd1,
        unicode::Flags::LETTER => 0xd2,
        unicode::Flags::PUNCTUATION => 0xd3,
        unicode::Flags::ACCENT_MARK => 0xd4,
        unicode::Flags::SYMBOL => 0xd5,
        _ => 0xd0,
    }
}
pub(crate) fn split(text: &[u8], patterns: &[Pattern], byte_encode: bool) -> Result<Vec<Vec<u8>>> {
    if text.len() > crate::MAX_INPUT_BYTES * 3 + 3 {
        return Err(Error::Limit(
            "normalized pretokenizer input exceeds 12 MiB plus prefix".into(),
        ));
    }
    if text.is_empty() {
        return Ok(Vec::new());
    }
    let codepoints = unicode::from_utf8(text);
    let mut offsets = vec![codepoints.len()];
    let mut budget = Budget {
        remaining: 1_000_000usize
            .saturating_add(codepoints.len().saturating_mul(4096))
            .min(128_000_000),
    };
    for pattern in patterns {
        offsets = pattern.offsets(&codepoints, &offsets, &mut budget)?;
    }
    let mut output = Vec::with_capacity(offsets.len());
    let mut begin = 0usize;
    for len in offsets {
        let end = begin
            .checked_add(len)
            .ok_or_else(|| bad("split offset overflow"))?;
        let mut bytes = Vec::new();
        for &cp in codepoints
            .get(begin..end)
            .ok_or_else(|| bad("split outside input"))?
        {
            bytes.extend(unicode::encode_utf8(cp).map_err(|e| bad(e.to_string()))?);
        }
        if byte_encode {
            let mut encoded = Vec::with_capacity(bytes.len());
            for byte in bytes {
                encoded.extend(unicode::byte_to_utf8(byte));
            }
            output.push(encoded);
        } else {
            output.push(bytes);
        }
        begin = end;
    }
    if begin != codepoints.len() {
        return Err(bad("split does not cover input"));
    }
    Ok(output)
}

fn flag(text: &[u32], position: usize) -> unicode::Flags {
    text.get(position)
        .copied()
        .map_or(unicode::Flags(0), unicode::flags)
}
fn cpt(text: &[u32], position: usize) -> u32 {
    text.get(position).copied().unwrap_or(u32::MAX)
}
fn contraction(text: &[u32], position: usize, fold: bool, long_s: bool) -> usize {
    if cpt(text, position) != 39 {
        return 0;
    }
    let convert = |cp| if fold { unicode::lowercase(cp) } else { cp };
    let mut first = convert(cpt(text, position + 1));
    if long_s && first == 0x17f {
        first = u32::from(b's');
    }
    if [
        u32::from(b's'),
        u32::from(b't'),
        u32::from(b'm'),
        u32::from(b'd'),
    ]
    .contains(&first)
    {
        return 2;
    }
    let second = convert(cpt(text, position + 2));
    if [
        (u32::from(b'r'), u32::from(b'e')),
        (u32::from(b'v'), u32::from(b'e')),
        (u32::from(b'l'), u32::from(b'l')),
    ]
    .contains(&(first, second))
    {
        return 3;
    }
    0
}
fn custom_offsets(text: &[u32], offsets: &[usize], custom: Custom) -> Vec<usize> {
    use unicode::Flags as F;
    let mut output = Vec::new();
    let mut begin = 0;
    for &length in offsets {
        let part = &text[begin..begin + length];
        begin += length;
        if custom == Custom::Newlines {
            let mut pos = 0;
            while pos < part.len() {
                let first = pos;
                let newline = part[pos] == 10;
                while pos < part.len() && (part[pos] == 10) == newline {
                    pos += 1;
                }
                output.push(pos - first);
            }
            continue;
        }
        if custom == Custom::RightDigits {
            let mut pos = 0;
            let mut previous = 0;
            while pos < part.len() {
                if flag(part, pos).contains(F::NUMBER) {
                    let start = pos;
                    while flag(part, pos).contains(F::NUMBER) {
                        pos += 1;
                    }
                    let mut current = start;
                    let remainder = (pos - start) % 3;
                    if remainder > 0 {
                        current += remainder;
                        output.push(current - previous);
                        previous = current;
                    }
                    while current < pos {
                        current += 3;
                        output.push(current - previous);
                        previous = current;
                    }
                } else {
                    pos += 1;
                }
            }
            if previous < part.len() {
                output.push(part.len() - previous);
            }
            continue;
        }
        let mut pos = 0;
        let marks = custom == Custom::Qwen35 || custom == Custom::Horizon;
        let letter = |index| {
            let f = flag(part, index);
            f.contains(F::LETTER)
                || (marks && f.contains(F::ACCENT_MARK))
                || (custom == Custom::Horizon && matches!(cpt(part, index), 0x200c | 0x200d))
        };
        while pos < part.len() {
            let start = pos;
            let cp = part[pos];
            let flags = flag(part, pos);
            if custom == Custom::Kimi && unicode::is_han(cp) {
                while unicode::is_han(cpt(part, pos)) {
                    pos += 1;
                }
                output.push(pos - start);
                continue;
            }
            if custom != Custom::Kimi {
                let length =
                    contraction(part, pos, custom != Custom::Gpt2, custom == Custom::Horizon);
                if length > 0 {
                    output.push(length);
                    pos += length;
                    continue;
                }
            }
            if custom == Custom::Gpt2 {
                let next = pos + usize::from(cp == 32);
                let category = flag(part, next);
                if category.contains(F::LETTER) {
                    pos = next;
                    while flag(part, pos).contains(F::LETTER) {
                        pos += 1;
                    }
                    output.push(pos - start);
                    continue;
                }
                if category.contains(F::NUMBER) {
                    pos = next;
                    while flag(part, pos).contains(F::NUMBER) {
                        pos += 1;
                    }
                    output.push(pos - start);
                    continue;
                }
            } else if custom == Custom::Kimi {
                let plain_letter =
                    |i| flag(part, i).contains(F::LETTER) && !unicode::is_han(cpt(part, i));
                if plain_letter(pos)
                    || (!matches!(cp, 10 | 13)
                        && !flags.contains(F::LETTER | F::NUMBER)
                        && plain_letter(pos + 1))
                {
                    if !plain_letter(pos) {
                        pos += 1;
                    }
                    while plain_letter(pos) {
                        pos += 1;
                    }
                    pos += contraction(part, pos, true, false);
                    output.push(pos - start);
                    continue;
                }
            } else if !matches!(cp, 10 | 13)
                && !flags.contains(F::NUMBER)
                && (letter(pos) || letter(pos + 1))
            {
                pos += 1;
                while letter(pos) {
                    pos += 1;
                }
                output.push(pos - start);
                continue;
            }
            if custom != Custom::Gpt2 && flags.contains(F::NUMBER) {
                let limit = if matches!(custom, Custom::Qwen2 | Custom::Qwen35) {
                    1
                } else {
                    3
                };
                while pos - start < limit && flag(part, pos).contains(F::NUMBER) {
                    pos += 1;
                }
                output.push(pos - start);
                continue;
            }
            let punctuation = |i| {
                let f = flag(part, i);
                let excluded = F::WHITESPACE
                    | F::LETTER
                    | F::NUMBER
                    | if custom == Custom::Qwen35 {
                        F::ACCENT_MARK
                    } else {
                        0
                    };
                f.0 != 0 && !f.contains(excluded)
            };
            let next = pos + usize::from(cp == 32);
            if punctuation(next) {
                pos = next;
                while punctuation(pos) {
                    pos += 1;
                }
                if custom != Custom::Gpt2 {
                    while matches!(cpt(part, pos), 10 | 13) {
                        pos += 1;
                    }
                }
                output.push(pos - start);
                continue;
            }
            let mut end = pos;
            let mut last_newline = None;
            while flag(part, end).contains(F::WHITESPACE) {
                if matches!(cpt(part, end), 10 | 13) {
                    last_newline = Some(end + 1);
                }
                end += 1;
            }
            if custom != Custom::Gpt2
                && let Some(newline) = last_newline
            {
                pos = newline;
            } else if end - pos > 1 && end != part.len() {
                pos = end - 1;
            } else if end > pos {
                pos = end;
            } else {
                pos += 1;
            }
            output.push(pos - start);
        }
    }
    output
}

#[derive(Debug, Clone)]
enum Item {
    Range(u32, u32),
    Space(bool, bool),
    Digit(bool, bool),
    Word(bool, bool),
    Property(u16),
}
impl Item {
    fn matches(&self, cp: u32) -> bool {
        match self {
            Self::Range(first, last) => (*first..=*last).contains(&cp),
            Self::Space(positive, wide) => *positive == class(cp, 4, *wide),
            Self::Digit(positive, wide) => *positive == class(cp, 1, *wide),
            Self::Word(positive, wide) => *positive == class(cp, 2, *wide),
            Self::Property(category) => {
                let mapped = match *category {
                    unicode::Flags::NUMBER => 0xd1,
                    unicode::Flags::LETTER => 0xd2,
                    unicode::Flags::PUNCTUATION => 0xd3,
                    unicode::Flags::ACCENT_MARK => 0xd4,
                    unicode::Flags::SYMBOL => 0xd5,
                    _ => unreachable!(),
                };
                cp == mapped || (cp < 128 && unicode::flags(cp).category() == *category)
            }
        }
    }
}
fn class(cp: u32, kind: u8, wide: bool) -> bool {
    #[cfg(windows)]
    if wide {
        let table = match kind {
            1 => WIDE_DIGIT,
            2 => WIDE_WORD,
            4 => WIDE_SPACE,
            _ => unreachable!(),
        };
        let index = table.partition_point(|&(first, _)| first <= cp);
        return index > 0 && cp <= table[index - 1].1;
    }
    #[cfg(not(windows))]
    let _ = wide;
    match kind {
        1 => matches!(cp, 48..=57),
        2 => matches!(cp,48..=57|65..=90|95|97..=122),
        4 => matches!(cp, 9..=13 | 32),
        _ => unreachable!(),
    }
}
#[derive(Debug, Clone)]
struct Set {
    negated: bool,
    items: Vec<Item>,
}
impl Set {
    fn literal(cp: u32) -> Self {
        Self {
            negated: false,
            items: vec![Item::Range(cp, cp)],
        }
    }
    fn matches(&self, cp: u32) -> bool {
        self.items.iter().any(|i| i.matches(cp)) != self.negated
    }
}
#[derive(Debug, Clone, Copy)]
enum Assertion {
    Start,
    End,
    Boundary(bool, bool),
}
#[derive(Debug, Clone)]
enum Node {
    Empty,
    Character(Set),
    Sequence(Vec<Node>),
    Alternative(Vec<Node>),
    Repeat {
        child: Box<Node>,
        minimum: usize,
        maximum: Option<usize>,
        greedy: bool,
    },
    Assert(Assertion),
    Look {
        child: Box<Node>,
        positive: bool,
    },
}
struct Parser {
    input: Vec<u32>,
    offset: usize,
    depth: usize,
    wide: bool,
}
impl Parser {
    fn peek(&self) -> Option<u32> {
        self.input.get(self.offset).copied()
    }
    fn take(&mut self) -> Result<u32> {
        let cp = self
            .peek()
            .ok_or_else(|| bad("unexpected end of expression"))?;
        self.offset += 1;
        Ok(cp)
    }
    fn accept(&mut self, cp: u32) -> bool {
        if self.peek() == Some(cp) {
            self.offset += 1;
            true
        } else {
            false
        }
    }
    fn require(&mut self, cp: u32) -> Result<()> {
        if self.accept(cp) {
            Ok(())
        } else {
            Err(bad(format!("expected codepoint {cp:x}")))
        }
    }
    fn alternation(&mut self) -> Result<Node> {
        self.depth += 1;
        if self.depth > 64 {
            return Err(Error::Limit("pretokenizer regex nesting exceeds 64".into()));
        }
        let mut alternatives = vec![self.sequence()?];
        while self.accept(124) {
            alternatives.push(self.sequence()?);
        }
        self.depth -= 1;
        Ok(if alternatives.len() == 1 {
            alternatives.pop().unwrap()
        } else {
            Node::Alternative(alternatives)
        })
    }
    fn sequence(&mut self) -> Result<Node> {
        let mut nodes = Vec::new();
        while self.peek().is_some_and(|cp| cp != 41 && cp != 124) {
            let child = self.atom()?;
            let quantifier = match self.peek() {
                Some(42) => {
                    self.offset += 1;
                    Some((0, None))
                }
                Some(43) => {
                    self.offset += 1;
                    Some((1, None))
                }
                Some(63) => {
                    self.offset += 1;
                    Some((0, Some(1)))
                }
                Some(123) => {
                    self.offset += 1;
                    let minimum = self.number()?;
                    let maximum = if self.accept(44) {
                        if self.peek() == Some(125) {
                            None
                        } else {
                            Some(self.number()?)
                        }
                    } else {
                        Some(minimum)
                    };
                    self.require(125)?;
                    if maximum.is_some_and(|m| m < minimum) {
                        return Err(bad("reversed repetition bounds"));
                    }
                    Some((minimum, maximum))
                }
                _ => None,
            };
            nodes.push(if let Some((minimum, maximum)) = quantifier {
                Node::Repeat {
                    child: Box::new(child),
                    minimum,
                    maximum,
                    greedy: !self.accept(63),
                }
            } else {
                child
            });
        }
        Ok(match nodes.len() {
            0 => Node::Empty,
            1 => nodes.pop().unwrap(),
            _ => Node::Sequence(nodes),
        })
    }
    fn number(&mut self) -> Result<usize> {
        let mut value = 0usize;
        let mut digits = 0;
        while let Some(cp @ 48..=57) = self.peek() {
            self.offset += 1;
            digits += 1;
            value = value
                .checked_mul(10)
                .and_then(|v| v.checked_add((cp - 48) as usize))
                .ok_or_else(|| bad("repetition overflow"))?;
            if value > 4096 {
                return Err(Error::Limit("pretokenizer repetition exceeds 4096".into()));
            }
        }
        if digits == 0 {
            Err(bad("missing repetition bound"))
        } else {
            Ok(value)
        }
    }
    fn atom(&mut self) -> Result<Node> {
        Ok(match self.take()? {
            40 => {
                let kind = if self.accept(63) {
                    Some(self.take()?)
                } else {
                    None
                };
                if kind.is_some_and(|k| !matches!(k, 58 | 61 | 33)) {
                    return Err(bad("unsupported group extension"));
                }
                let child = self.alternation()?;
                self.require(41)?;
                match kind {
                    Some(61) => Node::Look {
                        child: Box::new(child),
                        positive: true,
                    },
                    Some(33) => Node::Look {
                        child: Box::new(child),
                        positive: false,
                    },
                    _ => child,
                }
            }
            91 => Node::Character(self.class()?),
            92 => {
                if self.accept(98) {
                    Node::Assert(Assertion::Boundary(true, self.wide))
                } else if self.accept(66) {
                    Node::Assert(Assertion::Boundary(false, self.wide))
                } else {
                    Node::Character(Set {
                        negated: false,
                        items: vec![self.escape()?],
                    })
                }
            }
            94 => Node::Assert(Assertion::Start),
            36 => Node::Assert(Assertion::End),
            46 => Node::Character(Set {
                negated: true,
                items: vec![Item::Range(10, 10), Item::Range(13, 13)],
            }),
            42 | 43 | 63 | 123 => return Err(bad("quantifier without an atom")),
            cp => Node::Character(Set::literal(cp)),
        })
    }
    fn class(&mut self) -> Result<Set> {
        let negated = self.accept(94);
        let mut items = Vec::new();
        while self.peek() != Some(93) {
            let first = self.class_item()?;
            if self.peek() == Some(45) && self.input.get(self.offset + 1).copied() != Some(93) {
                self.offset += 1;
                let last = self.class_item()?;
                match (first, last) {
                    (Item::Range(a, b), Item::Range(c, d)) if a == b && c == d && a <= c => {
                        items.push(Item::Range(a, c))
                    }
                    _ => return Err(bad("invalid character-class range")),
                }
            } else {
                items.push(first);
            }
        }
        self.require(93)?;
        Ok(Set { negated, items })
    }
    fn class_item(&mut self) -> Result<Item> {
        if self.accept(92) {
            self.escape()
        } else {
            let cp = self.take()?;
            Ok(Item::Range(cp, cp))
        }
    }
    fn escape(&mut self) -> Result<Item> {
        let cp = self.take()?;
        Ok(match cp {
            115 => Item::Space(true, self.wide),
            83 => Item::Space(false, self.wide),
            100 => Item::Digit(true, self.wide),
            68 => Item::Digit(false, self.wide),
            119 => Item::Word(true, self.wide),
            87 => Item::Word(false, self.wide),
            112 => {
                self.require(123)?;
                let start = self.offset;
                while self.peek().is_some_and(|cp| cp != 125) {
                    self.offset += 1;
                }
                let name: String = self.input[start..self.offset]
                    .iter()
                    .map(|&cp| char::from_u32(cp).unwrap_or('\u{fffd}'))
                    .collect();
                self.require(125)?;
                Item::Property(match name.as_str() {
                    "N" => unicode::Flags::NUMBER,
                    "L" | "Lu" | "Ll" | "Lt" | "Lm" | "Lo" => unicode::Flags::LETTER,
                    "P" => unicode::Flags::PUNCTUATION,
                    "M" => unicode::Flags::ACCENT_MARK,
                    "S" => unicode::Flags::SYMBOL,
                    _ => return Err(bad(format!("unsupported property {name}"))),
                })
            }
            110 => Item::Range(10, 10),
            114 => Item::Range(13, 13),
            116 => Item::Range(9, 9),
            118 => Item::Range(11, 11),
            102 => Item::Range(12, 12),
            98 => Item::Range(8, 8),
            120 | 117 => {
                let mut value = 0;
                for _ in 0..if cp == 120 { 2 } else { 4 } {
                    let digit = self.take()?;
                    let value_digit = char::from_u32(digit)
                        .and_then(|c| c.to_digit(16))
                        .ok_or_else(|| bad("invalid hexadecimal escape"))?;
                    value = value * 16 + value_digit;
                }
                Item::Range(value, value)
            }
            48 => Item::Range(0, 0),
            49..=57 => {
                return Err(bad(
                    "backreferences are outside the frozen source expression set",
                ));
            }
            cp => Item::Range(cp, cp),
        })
    }
}

#[derive(Debug, Clone)]
enum Instruction {
    Accept,
    Character(Set, usize),
    Split(usize, usize),
    Assert(Assertion, usize),
    Look(Box<Program>, bool, usize),
}
#[derive(Debug, Clone)]
struct Program {
    instructions: Vec<Instruction>,
    entry: usize,
}
struct Budget {
    remaining: usize,
}
impl Budget {
    fn step(&mut self) -> Result<()> {
        if self.remaining == 0 {
            return Err(Error::Limit(
                "pretokenizer regex work budget exhausted".into(),
            ));
        }
        self.remaining -= 1;
        Ok(())
    }
}
impl Program {
    fn compile(node: &Node) -> Result<Self> {
        let mut program = Self {
            instructions: vec![Instruction::Accept],
            entry: 0,
        };
        program.entry = program.emit(node, 0)?;
        Ok(program)
    }
    fn push(&mut self, instruction: Instruction) -> Result<usize> {
        if self.instructions.len() >= 65536 {
            return Err(Error::Limit("pretokenizer regex program too large".into()));
        }
        let index = self.instructions.len();
        self.instructions.push(instruction);
        Ok(index)
    }
    fn emit(&mut self, node: &Node, next: usize) -> Result<usize> {
        match node {
            Node::Empty => Ok(next),
            Node::Character(set) => self.push(Instruction::Character(set.clone(), next)),
            Node::Assert(assertion) => self.push(Instruction::Assert(*assertion, next)),
            Node::Look { child, positive } => self.push(Instruction::Look(
                Box::new(Self::compile(child)?),
                *positive,
                next,
            )),
            Node::Sequence(nodes) => {
                let mut entry = next;
                for node in nodes.iter().rev() {
                    entry = self.emit(node, entry)?;
                }
                Ok(entry)
            }
            Node::Alternative(nodes) => {
                let mut entries = nodes.iter().rev();
                let mut entry = self.emit(entries.next().unwrap(), next)?;
                for node in entries {
                    let first = self.emit(node, next)?;
                    entry = self.push(Instruction::Split(first, entry))?;
                }
                Ok(entry)
            }
            Node::Repeat {
                child,
                minimum,
                maximum,
                greedy,
            } => {
                let mut entry = next;
                if let Some(maximum) = maximum {
                    for _ in *minimum..*maximum {
                        let first = self.emit(child, entry)?;
                        entry = self.push(if *greedy {
                            Instruction::Split(first, entry)
                        } else {
                            Instruction::Split(entry, first)
                        })?;
                    }
                } else {
                    let repeat = self.push(Instruction::Accept)?;
                    let first = self.emit(child, repeat)?;
                    self.instructions[repeat] = if *greedy {
                        Instruction::Split(first, entry)
                    } else {
                        Instruction::Split(entry, first)
                    };
                    entry = repeat;
                }
                for _ in 0..*minimum {
                    entry = self.emit(child, entry)?;
                }
                Ok(entry)
            }
        }
    }
    fn execute(
        &self,
        text: &[u32],
        start: usize,
        nonempty: bool,
        budget: &mut Budget,
    ) -> Result<Option<usize>> {
        let mut pending = vec![(self.entry, start)];
        let mut visited = HashSet::new();
        while let Some((mut pc, mut pos)) = pending.pop() {
            loop {
                budget.step()?;
                if !visited.insert((pc, pos)) {
                    break;
                }
                match &self.instructions[pc] {
                    Instruction::Accept => {
                        if !nonempty || pos != start {
                            return Ok(Some(pos));
                        }
                        break;
                    }
                    Instruction::Character(set, next) => {
                        if text.get(pos).is_some_and(|&cp| set.matches(cp)) {
                            pc = *next;
                            pos += 1;
                        } else {
                            break;
                        }
                    }
                    Instruction::Split(first, second) => {
                        pending.push((*second, pos));
                        pc = *first;
                    }
                    Instruction::Assert(assertion, next) => {
                        let pass = match assertion {
                            Assertion::Start => pos == 0,
                            Assertion::End => pos == text.len(),
                            Assertion::Boundary(positive, wide) => {
                                let before = pos
                                    .checked_sub(1)
                                    .and_then(|i| text.get(i))
                                    .is_some_and(|&cp| class(cp, 2, *wide));
                                let after = text.get(pos).is_some_and(|&cp| class(cp, 2, *wide));
                                (before != after) == *positive
                            }
                        };
                        if pass {
                            pc = *next;
                        } else {
                            break;
                        }
                    }
                    Instruction::Look(program, positive, next) => {
                        let found = program.execute(text, pos, false, budget)?.is_some();
                        if found == *positive {
                            pc = *next;
                        } else {
                            break;
                        }
                    }
                }
            }
        }
        Ok(None)
    }
    fn split(&self, text: &[u32], output: &mut Vec<usize>, budget: &mut Budget) -> Result<()> {
        let mut position = 0;
        let mut previous_end = 0;
        while position <= text.len() {
            if let Some(end) = self.execute(text, position, false, budget)? {
                if position > previous_end {
                    output.push(position - previous_end);
                }
                output.push(end - position);
                previous_end = end;
                if end > position {
                    position = end;
                    continue;
                }
                // std::regex_iterator first tries a nonempty continuous match
                // at the same point after a zero-length match.
                if let Some(end) = self.execute(text, position, true, budget)? {
                    output.push(end - position);
                    previous_end = end;
                    position = end;
                    continue;
                }
            }
            if position == text.len() {
                break;
            }
            position += 1;
        }
        if previous_end < text.len() {
            output.push(text.len() - previous_end);
        }
        Ok(())
    }
}

#[cfg(test)]
#[path = "regex_reference.rs"]
mod native_reference;
