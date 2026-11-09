use crate::*;
use std::ffi::c_char;

fn token_index(vocab: &VocabHandle, token: i32) -> Result<usize> {
    let index = usize::try_from(token).map_err(|_| "negative token id")?;
    if index >= vocab.texts.len() {
        return Err("token id outside vocabulary".into());
    }
    Ok(index)
}
ffi!(llama_vocab_type(vocab: *const VocabHandle) -> i32, 0, {
    unsafe {
        reference(vocab)?;
    }
    Ok(1)
});
ffi!(llama_vocab_n_tokens(vocab: *const VocabHandle) -> i32, 0, {
    i32::try_from(unsafe { reference(vocab)? }.texts.len())
        .map_err(|_| "vocabulary too large".into())
});
ffi!(llama_n_vocab(vocab: *const VocabHandle) -> i32, 0, {
    i32::try_from(unsafe { reference(vocab)? }.texts.len())
        .map_err(|_| "vocabulary too large".into())
});

ffi!(
    llama_vocab_get_text(vocab: *const VocabHandle, token: i32) -> *const c_char,
    ptr::null(),
    {
        let v = unsafe { reference(vocab)? };
        let i = token_index(v, token)?;
        Ok(v.texts[i].as_ptr().cast())
    }
);

ffi!(
    llama_token_get_text(vocab: *const VocabHandle, token: i32) -> *const c_char,
    ptr::null(),
    {
        let v = unsafe { reference(vocab)? };
        let i = token_index(v, token)?;
        Ok(v.texts[i].as_ptr().cast())
    }
);

ffi!(
    llama_vocab_get_score(vocab: *const VocabHandle, token: i32) -> f32,
    f32::NAN,
    {
        let v = unsafe { reference(vocab)? };
        Ok(v.scores[token_index(v, token)?])
    }
);

ffi!(
    llama_token_get_score(vocab: *const VocabHandle, token: i32) -> f32,
    f32::NAN,
    {
        let v = unsafe { reference(vocab)? };
        Ok(v.scores[token_index(v, token)?])
    }
);

ffi!(
    llama_vocab_get_attr(vocab: *const VocabHandle, token: i32) -> i32,
    0,
    {
        let v = unsafe { reference(vocab)? };
        Ok(v.attrs[token_index(v, token)?])
    }
);

ffi!(
    llama_token_get_attr(vocab: *const VocabHandle, token: i32) -> i32,
    0,
    {
        let v = unsafe { reference(vocab)? };
        Ok(v.attrs[token_index(v, token)?])
    }
);

ffi!(
    llama_vocab_is_eog(vocab: *const VocabHandle, token: i32) -> bool,
    false,
    {
        let v = unsafe { reference(vocab)? };
        token_index(v, token)?;
        Ok(v.tokenizer.is_end_of_generation(token))
    }
);

ffi!(
    llama_token_is_eog(vocab: *const VocabHandle, token: i32) -> bool,
    false,
    {
        let v = unsafe { reference(vocab)? };
        token_index(v, token)?;
        Ok(v.tokenizer.is_end_of_generation(token))
    }
);

ffi!(
    llama_vocab_is_control(vocab: *const VocabHandle, token: i32) -> bool,
    false,
    {
        let v = unsafe { reference(vocab)? };
        Ok(v.attrs[token_index(v, token)?] & 8 != 0)
    }
);

ffi!(
    llama_token_is_control(vocab: *const VocabHandle, token: i32) -> bool,
    false,
    {
        let v = unsafe { reference(vocab)? };
        Ok(v.attrs[token_index(v, token)?] & 8 != 0)
    }
);

ffi!(llama_vocab_bos(vocab: *const VocabHandle) -> i32, -1, {
    Ok(unsafe { reference(vocab)? }.special["bos"])
});

ffi!(llama_token_bos(vocab: *const VocabHandle) -> i32, -1, {
    Ok(unsafe { reference(vocab)? }.special["bos"])
});

ffi!(llama_vocab_eos(vocab: *const VocabHandle) -> i32, -1, {
    Ok(unsafe { reference(vocab)? }.special["eos"])
});

ffi!(llama_token_eos(vocab: *const VocabHandle) -> i32, -1, {
    Ok(unsafe { reference(vocab)? }.special["eos"])
});

ffi!(llama_vocab_eot(vocab: *const VocabHandle) -> i32, -1, {
    Ok(unsafe { reference(vocab)? }.special["eot"])
});

ffi!(llama_token_eot(vocab: *const VocabHandle) -> i32, -1, {
    Ok(unsafe { reference(vocab)? }.special["eot"])
});

ffi!(llama_vocab_sep(vocab: *const VocabHandle) -> i32, -1, {
    Ok(unsafe { reference(vocab)? }.special["sep"])
});

ffi!(llama_token_sep(vocab: *const VocabHandle) -> i32, -1, {
    Ok(unsafe { reference(vocab)? }.special["sep"])
});

ffi!(llama_vocab_nl(vocab: *const VocabHandle) -> i32, -1, {
    Ok(unsafe { reference(vocab)? }.special["nl"])
});

ffi!(llama_token_nl(vocab: *const VocabHandle) -> i32, -1, {
    Ok(unsafe { reference(vocab)? }.special["nl"])
});

ffi!(llama_vocab_pad(vocab: *const VocabHandle) -> i32, -1, {
    Ok(unsafe { reference(vocab)? }.special["pad"])
});

ffi!(llama_token_pad(vocab: *const VocabHandle) -> i32, -1, {
    Ok(unsafe { reference(vocab)? }.special["pad"])
});

ffi!(llama_vocab_mask(vocab: *const VocabHandle) -> i32, -1, {
    Ok(unsafe { reference(vocab)? }.special["mask"])
});

ffi!(llama_vocab_cls(vocab: *const VocabHandle) -> i32, -1, {
    Ok(unsafe { reference(vocab)? }.special["bos"])
});

ffi!(llama_token_cls(vocab: *const VocabHandle) -> i32, -1, {
    Ok(unsafe { reference(vocab)? }.special["bos"])
});

ffi!(
    llama_vocab_get_add_bos(vocab: *const VocabHandle) -> bool,
    false,
    { Ok(unsafe { reference(vocab)? }.tokenizer.add_bos) }
);

ffi!(
    llama_vocab_get_add_eos(vocab: *const VocabHandle) -> bool,
    false,
    { Ok(unsafe { reference(vocab)? }.tokenizer.add_eos) }
);

ffi!(
    llama_add_bos_token(vocab: *const VocabHandle) -> bool,
    false,
    { Ok(unsafe { reference(vocab)? }.tokenizer.add_bos) }
);

ffi!(
    llama_add_eos_token(vocab: *const VocabHandle) -> bool,
    false,
    { Ok(unsafe { reference(vocab)? }.tokenizer.add_eos) }
);

ffi!(
    llama_vocab_get_add_sep(vocab: *const VocabHandle) -> bool,
    false,
    { Ok(unsafe { reference(vocab)? }.add_sep) }
);
ffi!(
    llama_tokenize(
        vocab: *const VocabHandle,
        text: *const c_char,
        text_len: i32,
        tokens: *mut i32,
        capacity: i32,
        add_special: u8,
        parse_special: u8,
    ) -> i32,
    i32::MIN,
    {
        let v = unsafe { reference(vocab)? };
        let bytes = unsafe { input_slice(text.cast::<u8>(), count(text_len)?)? };
        let text = std::str::from_utf8(bytes).map_err(|_| "tokenizer input is not UTF-8")?;
        let result = v
            .tokenizer
            .tokenize(text, boolean(add_special)?, boolean(parse_special)?)
            .map_err(|e| e.to_string())?;
        unsafe { copy_buffer(&result, tokens, capacity) }
    }
);
ffi!(
    llama_token_to_piece(
        vocab: *const VocabHandle,
        token: i32,
        buf: *mut c_char,
        length: i32,
        lstrip: i32,
        special: u8,
    ) -> i32,
    i32::MIN,
    {
        let v = unsafe { reference(vocab)? };
        let piece = v
            .tokenizer
            .token_piece(token, boolean(special)?)
            .map_err(|e| e.to_string())?;
        let strip = count(lstrip)?;
        let skip = piece.iter().take(strip).take_while(|b| **b == b' ').count();
        unsafe { copy_buffer(&piece[skip..], buf.cast(), length) }
    }
);
ffi!(
    llama_detokenize(
        vocab: *const VocabHandle,
        tokens: *const i32,
        n_tokens: i32,
        text: *mut c_char,
        capacity: i32,
        remove_special: u8,
        unparse_special: u8,
    ) -> i32,
    i32::MIN,
    {
        let v = unsafe { reference(vocab)? };
        let tokens = unsafe { input_slice(tokens, count(n_tokens)?)? };
        let result = v
            .tokenizer
            .detokenize_bytes(tokens, boolean(remove_special)?, boolean(unparse_special)?)
            .map_err(|e| e.to_string())?;
        unsafe { copy_buffer(&result, text.cast(), capacity) }
    }
);
