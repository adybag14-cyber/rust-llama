use super::*;
use std::mem::{align_of, offset_of, size_of};

struct Reader<'a> {
    data: &'a [u8],
    at: usize,
}
impl<'a> Reader<'a> {
    fn new(data: &'a [u8], magic: &[u8]) -> Self {
        assert!(data.starts_with(magic));
        Self {
            data,
            at: magic.len(),
        }
    }
    fn take(&mut self, n: usize) -> &'a [u8] {
        let end = self.at.checked_add(n).unwrap();
        let value = &self.data[self.at..end];
        self.at = end;
        value
    }
    fn byte(&mut self) -> u8 {
        self.take(1)[0]
    }
    fn word(&mut self) -> u32 {
        u32::from_le_bytes(self.take(4).try_into().unwrap())
    }
    fn int(&mut self) -> i32 {
        self.word() as i32
    }
    fn blob(&mut self) -> &'a [u8] {
        let n = self.word() as usize;
        self.take(n)
    }
    fn c_string(&mut self) -> Vec<u8> {
        let mut bytes = self.blob().to_vec();
        bytes.push(0);
        bytes
    }
    fn end(self) {
        assert_eq!(self.at, self.data.len(), "unconsumed original fixture data");
    }
}

#[test]
fn all_original_c_chat_bytes_buffers_and_builtin_layouts() {
    let mut input = Reader::new(
        include_bytes!("../../../../fixtures/ffi/chat/inputs.bin"),
        b"CCHATIN1",
    );
    let mut expected = Reader::new(
        include_bytes!("../../../../fixtures/ffi/chat/original.bin"),
        b"CCHATOU1",
    );
    let cases = input.word();
    assert_eq!(cases, 35_946);
    assert_eq!(expected.word(), cases);
    let mut compared_bytes = 0;
    let mut accepted = 0;
    for i in 0..cases {
        let flags = input.byte();
        let _requested_length = input.int();
        let template = input.c_string();
        let n = input.word() as usize;
        let mut strings = Vec::with_capacity(n);
        for _ in 0..n {
            strings.push((input.c_string(), input.c_string()));
        }
        let messages: Vec<_> = strings
            .iter()
            .map(|(role, content)| LlamaChatMessage {
                role: role.as_ptr().cast(),
                content: content.as_ptr().cast(),
            })
            .collect();
        let count = expected.int();
        let result = expected.int();
        let length = expected.int();
        let full_buffer = expected.blob();
        let mut actual = vec![0xa5u8; full_buffer.len()];
        let template = if flags & 2 == 0 {
            template.as_ptr().cast()
        } else {
            ptr::null()
        };
        let chat = if messages.is_empty() {
            ptr::null()
        } else {
            messages.as_ptr()
        };
        let output = if flags & 4 == 0 {
            unsafe { actual.as_mut_ptr().add(8).cast() }
        } else {
            ptr::null_mut()
        };
        unsafe {
            llama_rust_clear_error();
            assert_eq!(
                llama_chat_apply_template(template, chat, n, flags & 1, ptr::null_mut(), 0),
                count,
                "original count for case {i}",
            );
            if count >= 0 {
                assert!(
                    CStr::from_ptr(llama_rust_last_error())
                        .to_bytes()
                        .is_empty(),
                    "case {i}"
                );
            }
            llama_rust_clear_error();
            assert_eq!(
                llama_chat_apply_template(template, chat, n, flags & 1, output, length),
                result,
                "original return for case {i}",
            );
            if result >= 0 {
                accepted += 1;
                assert!(
                    CStr::from_ptr(llama_rust_last_error())
                        .to_bytes()
                        .is_empty(),
                    "case {i}"
                );
            }
        }
        assert_eq!(actual, full_buffer, "complete original buffer for case {i}");
        compared_bytes += actual.len();
    }
    input.end();
    assert_eq!(accepted, 34_349);
    assert_eq!(compared_bytes, 3_361_503);
    let native_layout = [
        expected.word() as usize,
        expected.word() as usize,
        expected.word() as usize,
        expected.word() as usize,
        expected.word() as usize,
    ];
    if size_of::<usize>() == 8 {
        assert_eq!(
            native_layout,
            [
                size_of::<LlamaChatMessage>(),
                align_of::<LlamaChatMessage>(),
                offset_of!(LlamaChatMessage, role),
                offset_of!(LlamaChatMessage, content),
                size_of::<bool>(),
            ]
        );
    }
    let total = expected.word() as usize;
    assert_eq!(total, 54);
    let names: Vec<_> = (0..total).map(|_| expected.blob()).collect();
    let sentinel = ptr::dangling::<c_char>();
    let mut pointers = vec![sentinel; total];
    unsafe {
        assert_eq!(
            llama_chat_builtin_templates(ptr::null_mut(), 0),
            total as i32
        );
        assert_eq!(
            llama_chat_builtin_templates(pointers.as_mut_ptr(), total),
            total as i32
        );
    }
    for (pointer, name) in pointers.iter().zip(&names) {
        assert_eq!(unsafe { CStr::from_ptr(*pointer) }.to_bytes(), *name);
    }
    let mut declared = llama_behavior::chat::BUILTIN_TEMPLATES.to_vec();
    declared.sort_unstable();
    assert_eq!(
        declared.iter().map(|s| s.as_bytes()).collect::<Vec<_>>(),
        names
    );
    let limits = [0, 1, 2, total - 1, total, total + 1, total + 8, usize::MAX];
    assert_eq!(expected.word() as usize, limits.len());
    for limit in limits {
        let result = expected.int();
        let slots = expected.word() as usize;
        let equal = expected.take(slots);
        let mut actual = vec![sentinel; slots];
        assert_eq!(
            unsafe { llama_chat_builtin_templates(actual.as_mut_ptr(), limit) },
            result
        );
        for i in 0..slots {
            let target = if i < limit.min(total) {
                pointers[i]
            } else {
                sentinel
            };
            assert_eq!(
                u8::from(actual[i] == target),
                equal[i],
                "builtin cap {limit}, slot {i}"
            );
        }
    }
    expected.end();
}

#[test]
fn checked_foreign_boundaries_preserve_output() {
    let mut output = [0x5au8; 32];
    let good = LlamaChatMessage {
        role: c"user".as_ptr(),
        content: c"hello".as_ptr(),
    };
    unsafe {
        let cases = [
            (c"chatml".as_ptr(), ptr::null(), 1, 0),
            (c"chatml".as_ptr(), &good, MAX_MESSAGES + 1, 0),
            (c"chatml".as_ptr(), &good, 1, 2),
        ];
        for (template, messages, n, flag) in cases {
            llama_rust_clear_error();
            assert_eq!(
                llama_chat_apply_template(
                    template,
                    messages,
                    n,
                    flag,
                    output.as_mut_ptr().cast(),
                    32
                ),
                -1
            );
            assert!(
                !CStr::from_ptr(llama_rust_last_error())
                    .to_bytes()
                    .is_empty()
            );
            assert_eq!(output, [0x5a; 32]);
        }
        for message in [
            LlamaChatMessage {
                role: ptr::null(),
                ..good
            },
            LlamaChatMessage {
                content: ptr::null(),
                ..good
            },
        ] {
            llama_rust_clear_error();
            assert_eq!(
                llama_chat_apply_template(
                    c"chatml".as_ptr(),
                    &message,
                    1,
                    0,
                    output.as_mut_ptr().cast(),
                    32
                ),
                -1
            );
            assert!(
                !CStr::from_ptr(llama_rust_last_error())
                    .to_bytes()
                    .is_empty()
            );
        }
        llama_rust_clear_error();
        assert_eq!(llama_chat_builtin_templates(ptr::null_mut(), 1), -1);
        assert!(
            !CStr::from_ptr(llama_rust_last_error())
                .to_bytes()
                .is_empty()
        );
    }
    assert_eq!(output, [0x5a; 32]);
}

#[test]
fn cumulative_input_and_output_limits_are_checked_before_writes() {
    let mut output = [0x39u8; 8];
    let mut huge_template = vec![b'x'; MAX_TEMPLATE_BYTES + 1];
    huge_template.push(0);
    let mut content = vec![b'x'; MAX_INPUT_BYTES + 1];
    content.push(0);
    let mut message = LlamaChatMessage {
        role: c"user".as_ptr(),
        content: content.as_ptr().cast(),
    };
    unsafe {
        assert_eq!(
            llama_chat_apply_template(
                huge_template.as_ptr().cast(),
                ptr::null(),
                0,
                0,
                output.as_mut_ptr().cast(),
                8
            ),
            -1
        );
        assert_eq!(
            llama_chat_apply_template(
                c"chatml".as_ptr(),
                &message,
                1,
                0,
                output.as_mut_ptr().cast(),
                8
            ),
            -1
        );
        // The total input now fits exactly; template expansion exceeds the separate output bound.
        content[MAX_INPUT_BYTES - 4] = 0;
        message.content = content.as_ptr().cast();
        assert_eq!(
            llama_chat_apply_template(
                c"chatml".as_ptr(),
                &message,
                1,
                0,
                output.as_mut_ptr().cast(),
                8
            ),
            -1
        );
    }
    assert_eq!(output, [0x39; 8]);
}

#[test]
fn source_undefined_signed_case_conversion_is_checked_and_safe_text_api_is_unchanged() {
    if c_char::MIN == 0 {
        return;
    }
    for template in [c"openchat", c"bailing", c"bailing-think", c"bailing2"] {
        for byte in 0x80u8..=0xfe {
            let role = [byte, 0];
            let message = LlamaChatMessage {
                role: role.as_ptr().cast(),
                content: c"x".as_ptr(),
            };
            let mut output = [0x5du8; 8];
            assert_eq!(
                unsafe {
                    llama_chat_apply_template(
                        template.as_ptr(),
                        &message,
                        1,
                        0,
                        output.as_mut_ptr().cast(),
                        8,
                    )
                },
                -1
            );
            assert_eq!(output, [0x5d; 8]);
        }
    }
    use llama_behavior::chat::{Message, render};
    assert!(
        render("openchat", &[Message::new("érole", "x")], false)
            .unwrap()
            .contains("érole")
    );
    assert!(
        render("bailing", &[Message::new("érole", "x")], false)
            .unwrap()
            .contains("éROLE")
    );
    assert!(render("openchat", &[Message::new("", "x")], false).is_err());
}
