//! Borrowed native memory handle; lifetime is exactly that of its context.
use crate::*;
use llama_model_runtime::NativeContext;

ffi!(
    llama_get_memory(context: *const ContextHandle) -> *mut NativeContext,
    ptr::null_mut(),
    {
        let context = unsafe { reference(context)? };
        Ok(match &context.runtime {
            crate::context::ContextRuntime::Decoder(runtime) => {
                ptr::from_ref(runtime.as_ref()).cast_mut()
            }
            crate::context::ContextRuntime::Encoder(_) => ptr::null_mut(),
        })
    }
);
ffi!(
    llama_memory_clear(memory: *mut NativeContext, data: u8) -> (),
    (),
    {
        let data = boolean(data)?;
        unsafe { reference_mut(memory)? }.clear(data);
        Ok(())
    }
);
ffi!(
    llama_memory_seq_rm(memory: *mut NativeContext, sequence: i32, p0: i32, p1: i32) -> bool,
    false,
    {
        unsafe { reference_mut(memory)? }
            .remove_range(sequence, p0, p1)
            .map_err(|e| e.to_string())
    }
);
ffi!(
    llama_memory_seq_cp(
        memory: *mut NativeContext,
        source: i32,
        destination: i32,
        p0: i32,
        p1: i32,
    ) -> (),
    (),
    {
        unsafe { reference_mut(memory)? }
            .copy_range(source, destination, p0, p1)
            .map_err(|e| e.to_string())
    }
);
ffi!(
    llama_memory_seq_keep(memory: *mut NativeContext, sequence: i32) -> (),
    (),
    {
        unsafe { reference_mut(memory)? }
            .keep_sequence(sequence)
            .map_err(|e| e.to_string())
    }
);
ffi!(
    llama_memory_seq_add(
        memory: *mut NativeContext,
        sequence: i32,
        p0: i32,
        p1: i32,
        delta: i32,
    ) -> (),
    (),
    {
        unsafe { reference_mut(memory)? }
            .shift_range(sequence, p0, p1, delta)
            .map_err(|e| e.to_string())
    }
);
ffi!(
    llama_memory_seq_div(
        memory: *mut NativeContext,
        sequence: i32,
        p0: i32,
        p1: i32,
        divisor: i32,
    ) -> (),
    (),
    {
        unsafe { reference_mut(memory)? }
            .divide_range(sequence, p0, p1, divisor)
            .map_err(|e| e.to_string())
    }
);
ffi!(
    llama_memory_seq_pos_min(memory: *mut NativeContext, sequence: i32) -> i32,
    -1,
    {
        unsafe { reference(memory)? }
            .position_min(sequence)
            .map_err(|e| e.to_string())
    }
);
ffi!(
    llama_memory_seq_pos_max(memory: *mut NativeContext, sequence: i32) -> i32,
    -1,
    {
        unsafe { reference(memory)? }
            .position_max(sequence)
            .map_err(|e| e.to_string())
    }
);
ffi!(
    llama_memory_can_shift(memory: *mut NativeContext) -> bool,
    false,
    { Ok(unsafe { reference(memory)? }.can_shift()) }
);
