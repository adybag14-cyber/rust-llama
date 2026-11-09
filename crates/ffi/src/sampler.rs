use crate::*;
use llama_behavior::sampling::{
    Candidates, Config, MAX_CANDIDATES, MAX_HISTORY, Sampler, TokenData,
};
use std::collections::BTreeMap;
use std::ffi::{CStr, CString, c_char, c_void};
use std::sync::{Mutex, OnceLock};

const CHAIN: usize = 0;
const EMPTY: usize = 1;
const BIAS: usize = 15;
const MAX_CHAIN_STAGES: usize = 128;
const MAX_CHAIN_DEPTH: usize = 64;

#[derive(Clone)]
struct Builtin {
    sampler: Sampler,
    name: CString,
    requested_seed: Option<u32>,
    effective_seed: u32,
    vocabulary: Option<usize>,
}
enum State {
    Builtin(Box<Builtin>),
    Bias(Vec<LlamaLogitBias>),
    Empty(&'static CStr),
    Chain(Vec<*mut LlamaSampler>),
}
impl State {
    fn index(&self) -> usize {
        match self {
            Self::Chain(_) => CHAIN,
            Self::Empty(_) => EMPTY,
            Self::Bias(_) => BIAS,
            Self::Builtin(b) => match b.sampler.config() {
                Config::Greedy => 2,
                Config::Dist { .. } => 3,
                Config::TopK { .. } => 4,
                Config::TopP { .. } => 5,
                Config::MinP { .. } => 6,
                Config::Typical { .. } => 7,
                Config::Temperature { .. } => 8,
                Config::DynamicTemperature { .. } => 9,
                Config::Xtc { .. } => 10,
                Config::Penalties { .. } => 11,
                Config::TopNSigma { .. } => 12,
                Config::Mirostat { .. } => 13,
                Config::MirostatV2 { .. } => 14,
                Config::LogitBias { .. } => BIAS,
                Config::AdaptiveP { .. } => 16,
                Config::Dry { .. } => 17,
            },
        }
    }
}
const INTERFACE: LlamaSamplerI = LlamaSamplerI {
    name: Some(native_name),
    accept: Some(native_accept),
    apply: Some(native_apply),
    reset: Some(native_reset),
    clone: Some(native_clone),
    free: Some(native_free_state),
    backend_init: None,
    backend_accept: None,
    backend_apply: None,
    backend_set_input: None,
    backend_reset: None,
    copy_state: None,
};
// Distinct addresses preserve sampler type identity, including chain detection.
static INTERFACES: [LlamaSamplerI; 18] = [INTERFACE; 18];
#[derive(Default)]
struct Entry {
    owner: Option<usize>,
}
fn registry() -> &'static Mutex<BTreeMap<usize, Entry>> {
    static REGISTRY: OnceLock<Mutex<BTreeMap<usize, Entry>>> = OnceLock::new();
    REGISTRY.get_or_init(|| Mutex::new(BTreeMap::new()))
}
fn native_index(iface: *mut LlamaSamplerI) -> Option<usize> {
    INTERFACES.iter().position(|item| std::ptr::eq(item, iface))
}
unsafe fn handle<'a>(sampler: *const LlamaSampler) -> Result<&'a LlamaSampler> {
    if !registry()
        .lock()
        .map_err(|_| "sampler registry poisoned")?
        .contains_key(&(sampler as usize))
    {
        return Err("sampler handle is foreign, freed, or null".into());
    }
    unsafe { reference(sampler) }
}
unsafe fn state<'a>(sampler: *const LlamaSampler) -> Result<&'a State> {
    let h = unsafe { handle(sampler)? };
    if native_index(h.iface).is_none() {
        return Err("operation requires a built-in Rust sampler".into());
    }
    unsafe { reference(h.ctx.cast()) }
}
unsafe fn state_mut<'a>(sampler: *mut LlamaSampler) -> Result<&'a mut State> {
    let h = unsafe { handle(sampler)? };
    if native_index(h.iface).is_none() {
        return Err("operation requires a built-in Rust sampler".into());
    }
    unsafe { reference_mut(h.ctx.cast()) }
}
fn allocate_shell(iface: *mut LlamaSamplerI, ctx: *mut c_void) -> Result<*mut LlamaSampler> {
    let mut r = registry().lock().map_err(|_| "sampler registry poisoned")?;
    let result = Box::into_raw(Box::new(LlamaSampler { iface, ctx }));
    r.insert(result as usize, Entry::default());
    Ok(result)
}
fn allocate(value: State) -> Result<*mut LlamaSampler> {
    let index = value.index();
    let mut state = Box::new(value);
    let context = (&mut *state as *mut State).cast();
    let result = allocate_shell(
        (&INTERFACES[index] as *const LlamaSamplerI).cast_mut(),
        context,
    )?;
    let _ = Box::into_raw(state);
    Ok(result)
}
fn empty(name: &'static CStr) -> Result<*mut LlamaSampler> {
    allocate(State::Empty(name))
}

#[cfg(windows)]
fn entropy() -> Result<u32> {
    #[link(name = "bcrypt")]
    unsafe extern "system" {
        fn BCryptGenRandom(algorithm: *mut c_void, buffer: *mut u8, length: u32, flags: u32)
        -> i32;
    }
    let mut bytes = [0u8; 4];
    // SAFETY: operating-system RNG receives a live four-byte writable buffer.
    let status = unsafe { BCryptGenRandom(ptr::null_mut(), bytes.as_mut_ptr(), 4, 2) };
    if status < 0 {
        return Err(format!("BCryptGenRandom failed: {status:#x}"));
    }
    Ok(u32::from_ne_bytes(bytes))
}
#[cfg(unix)]
fn entropy() -> Result<u32> {
    use std::io::Read;
    let mut bytes = [0u8; 4];
    std::fs::File::open("/dev/urandom")
        .and_then(|mut file| file.read_exact(&mut bytes))
        .map_err(|e| format!("system random seed unavailable: {e}"))?;
    Ok(u32::from_ne_bytes(bytes))
}
#[cfg(not(any(unix, windows)))]
fn entropy() -> Result<u32> {
    Err("default random seeds are unavailable on this target; use an explicit seed".into())
}
fn resolve_seed(seed: u32) -> Result<u32> {
    if seed != u32::MAX {
        return Ok(seed);
    }
    for _ in 0..8 {
        let value = entropy()?;
        if value != u32::MAX {
            return Ok(value);
        }
    }
    Err("system RNG repeatedly returned the reserved seed sentinel".into())
}
fn config_seed(config: &mut Config, seed: u32) {
    match config {
        Config::Dist { seed: s }
        | Config::Xtc { seed: s, .. }
        | Config::Mirostat { seed: s, .. }
        | Config::MirostatV2 { seed: s, .. }
        | Config::AdaptiveP { seed: s, .. } => *s = seed,
        _ => {}
    }
}
fn builtin(mut config: Config, requested_seed: Option<u32>) -> Result<*mut LlamaSampler> {
    let effective_seed = if let Some(seed) = requested_seed {
        resolve_seed(seed)?
    } else {
        u32::MAX
    };
    if requested_seed.is_some() {
        config_seed(&mut config, effective_seed);
    }
    let sampler = Sampler::new(config).map_err(|e| e.to_string())?;
    let name = CString::new(sampler.name()).map_err(|_| "invalid sampler name")?;
    allocate(State::Builtin(Box::new(Builtin {
        sampler,
        name,
        requested_seed,
        effective_seed,
        vocabulary: None,
    })))
}
unsafe fn name_impl(sampler: *const LlamaSampler) -> Result<*const c_char> {
    let h = unsafe { handle(sampler)? };
    if h.iface.is_null() {
        return Ok(c"(null)".as_ptr());
    }
    if native_index(h.iface).is_some() {
        return Ok(match unsafe { state(sampler)? } {
            State::Builtin(b) => b.name.as_ptr(),
            State::Bias(_) => c"logit-bias".as_ptr(),
            State::Empty(label) => label.as_ptr(),
            State::Chain(_) => c"chain".as_ptr(),
        });
    }
    let interface = *unsafe { reference(h.iface)? };
    Ok(if let Some(callback) = interface.name {
        unsafe { callback(sampler) }
    } else {
        c"(null)".as_ptr()
    })
}
unsafe fn accept_impl(sampler: *mut LlamaSampler, token: i32) -> Result<()> {
    if sampler.is_null() {
        return Ok(());
    }
    let h = unsafe { handle(sampler)? };
    if native_index(h.iface).is_some() {
        match unsafe { state_mut(sampler)? } {
            State::Builtin(b)
                if matches!(
                    b.sampler.config(),
                    Config::Penalties { .. } | Config::Dry { .. } | Config::AdaptiveP { .. }
                ) =>
            {
                if b.vocabulary
                    .is_some_and(|n| token < 0 || token as usize >= n)
                {
                    return Err("accepted token lies outside sampler vocabulary".into());
                }
                b.sampler.accept(token).map_err(|e| e.to_string())?;
            }
            State::Chain(children) => {
                let children = children.clone();
                for child in children {
                    unsafe {
                        accept_impl(child, token)?;
                    }
                }
            }
            _ => {}
        }
        return Ok(());
    }
    let interface = *unsafe { reference(h.iface)? };
    if let Some(callback) = interface.accept {
        unsafe { callback(sampler, token) }
    }
    Ok(())
}
fn validate_values(values: &[TokenData]) -> Result<()> {
    let mut candidates = Candidates {
        data: values.to_vec(),
        selected: None,
        sorted: false,
    };
    // Preserve probability payloads; native non-normalizing stages leave them untouched.
    candidates.sorted = false;
    candidates.validate().map_err(|e| e.to_string())
}
unsafe fn apply_builtin(sampler: *mut LlamaSampler, array: *mut LlamaTokenDataArray) -> Result<()> {
    let s = unsafe { state_mut(sampler)? };
    if matches!(s, State::Empty(_)) {
        return Ok(());
    }
    if let State::Chain(children) = s {
        let children = children.clone();
        for child in children {
            // The frozen chain skips custom backend-only stages with no CPU apply.
            let h = unsafe { handle(child)? };
            if !h.iface.is_null() && unsafe { reference(h.iface)? }.apply.is_none() {
                continue;
            }
            unsafe {
                apply_impl(child, array)?;
            }
        }
        return Ok(());
    }
    let output = unsafe { reference_mut(array)? };
    if output.size > MAX_CANDIDATES {
        return Err("sampler candidate count exceeds one million".into());
    }
    let sorted = boolean(output.sorted)?;
    let input = unsafe { input_slice(output.data, output.size)? };
    let mut candidates = Candidates {
        data: input
            .iter()
            .map(|v| TokenData {
                id: v.id,
                logit: v.logit,
                p: v.p,
            })
            .collect(),
        selected: if output.selected >= 0 {
            Some(
                usize::try_from(output.selected)
                    .map_err(|_| "selected index exceeds address size")?,
            )
        } else {
            None
        },
        sorted,
    };
    let old_selected = output.selected;
    let (prefix, terminal) = match s {
        State::Builtin(b) => {
            let terminal = matches!(
                b.sampler.config(),
                Config::Greedy
                    | Config::Dist { .. }
                    | Config::Mirostat { .. }
                    | Config::MirostatV2 { .. }
                    | Config::AdaptiveP { .. }
            );
            (
                b.sampler
                    .apply_native(&mut candidates)
                    .map_err(|e| e.to_string())?,
                terminal,
            )
        }
        State::Bias(biases) => {
            validate_values(&candidates.data)?;
            let mut remaining = Vec::new();
            for bias in biases {
                if bias.token >= 0
                    && (bias.token as usize) < candidates.data.len()
                    && candidates.data[bias.token as usize].id == bias.token
                {
                    candidates.data[bias.token as usize].logit += bias.bias;
                } else {
                    remaining.push(*bias);
                }
            }
            for value in &mut candidates.data {
                if let Some(bias) = remaining.iter().find(|b| b.token == value.id) {
                    value.logit += bias.bias;
                }
            }
            validate_values(&candidates.data)?;
            (Vec::new(), false)
        }
        _ => unreachable!(),
    };
    let offset = prefix.len();
    if offset + candidates.data.len() > output.size {
        return Err("sampler expanded beyond candidate allocation".into());
    }
    for (i, value) in prefix.iter().chain(&candidates.data).enumerate() {
        unsafe {
            *output.data.add(i) = LlamaTokenData {
                id: value.id,
                logit: value.logit,
                p: value.p,
            };
        }
    }
    if offset > 0 {
        output.data = unsafe { output.data.add(offset) };
    }
    output.size = candidates.data.len();
    output.sorted = u8::from(candidates.sorted);
    output.selected = candidates
        .selected
        .map_or(if terminal { -1 } else { old_selected }, |i| i as i64);
    Ok(())
}
unsafe fn apply_impl(sampler: *mut LlamaSampler, array: *mut LlamaTokenDataArray) -> Result<()> {
    if sampler.is_null() {
        return Ok(());
    }
    let h = unsafe { handle(sampler)? };
    if native_index(h.iface).is_some() {
        return unsafe { apply_builtin(sampler, array) };
    }
    let interface = *unsafe { reference(h.iface)? };
    let callback = interface
        .apply
        .ok_or("custom sampler has no CPU apply callback")?;
    unsafe {
        reference_mut(array)?;
        callback(sampler, array);
    }
    Ok(())
}
unsafe fn reset_impl(sampler: *mut LlamaSampler) -> Result<()> {
    if sampler.is_null() {
        return Ok(());
    }
    let h = unsafe { handle(sampler)? };
    if native_index(h.iface).is_some() {
        match unsafe { state_mut(sampler)? } {
            State::Builtin(b) => {
                if let Some(seed) = b.requested_seed {
                    b.effective_seed = resolve_seed(seed)?;
                    let mut config = b.sampler.config().clone();
                    config_seed(&mut config, b.effective_seed);
                    b.sampler = Sampler::new(config).map_err(|e| e.to_string())?;
                } else {
                    b.sampler.reset();
                }
            }
            State::Chain(children) => {
                let children = children.clone();
                for child in children {
                    unsafe {
                        reset_impl(child)?;
                    }
                }
            }
            _ => {}
        }
        return Ok(());
    }
    let interface = *unsafe { reference(h.iface)? };
    if let Some(callback) = interface.reset {
        unsafe { callback(sampler) }
    }
    Ok(())
}
unsafe fn free_state_impl(sampler: *mut LlamaSampler) -> Result<()> {
    let h = unsafe { reference_mut(sampler)? };
    if h.ctx.is_null() {
        return Ok(());
    }
    let state = h.ctx.cast::<State>();
    h.ctx = ptr::null_mut();
    // SAFETY: this opaque allocation originated in allocate; it is consumed once.
    let value = unsafe { Box::from_raw(state) };
    if let State::Chain(children) = *value {
        let mut first_error = None;
        for child in children {
            if let Err(e) = unsafe { free_impl(child, true) } {
                first_error.get_or_insert(e);
            }
        }
        if let Some(error) = first_error {
            return Err(error);
        }
    }
    Ok(())
}
unsafe fn free_impl(sampler: *mut LlamaSampler, owned: bool) -> Result<()> {
    if sampler.is_null() {
        return Ok(());
    }
    {
        let mut r = registry().lock().map_err(|_| "sampler registry poisoned")?;
        let entry = r
            .get(&(sampler as usize))
            .ok_or("sampler was already freed or is foreign")?;
        if entry.owner.is_some() && !owned {
            return Err("sampler is owned by a chain; remove it before freeing".into());
        }
        r.remove(&(sampler as usize));
    }
    // Own the shell before callbacks, so an error or contained Rust panic cannot leak it.
    let shell = unsafe { Box::from_raw(sampler) };
    let result = if native_index(shell.iface).is_some() {
        unsafe { free_state_impl(sampler) }
    } else if !shell.iface.is_null() {
        let interface = *unsafe { reference(shell.iface)? };
        if let Some(callback) = interface.free {
            unsafe { callback(sampler) }
        }
        Ok(())
    } else {
        Ok(())
    };
    drop(shell);
    result
}

unsafe fn chain_children(sampler: *const LlamaSampler) -> Result<Vec<*mut LlamaSampler>> {
    match unsafe { state(sampler)? } {
        State::Chain(children) => Ok(children.clone()),
        _ => Err("sampler is not a chain".into()),
    }
}
unsafe fn height(sampler: *const LlamaSampler, depth: usize) -> Result<usize> {
    if depth > MAX_CHAIN_DEPTH {
        return Err("sampler chain nesting exceeds 64".into());
    }
    let h = unsafe { handle(sampler)? };
    if native_index(h.iface) == Some(CHAIN) {
        let mut maximum = 0;
        for child in unsafe { chain_children(sampler)? } {
            maximum = maximum.max(unsafe { height(child, depth + 1)? });
        }
        Ok(1 + maximum)
    } else {
        Ok(1)
    }
}
unsafe fn chain_add_impl(chain: *mut LlamaSampler, sampler: *mut LlamaSampler) -> Result<()> {
    unsafe {
        handle(sampler)?;
    }
    if unsafe { chain_children(chain)? }.len() >= MAX_CHAIN_STAGES {
        return Err("sampler chain exceeds 128 direct stages".into());
    }
    let child_height = unsafe { height(sampler, 0)? };
    let mut r = registry().lock().map_err(|_| "sampler registry poisoned")?;
    if r.get(&(sampler as usize))
        .ok_or("foreign sampler")?
        .owner
        .is_some()
    {
        return Err("sampler already belongs to a chain".into());
    }
    let mut ancestor = Some(chain as usize);
    let mut depth = 0;
    while let Some(id) = ancestor {
        if id == sampler as usize {
            return Err("sampler chain cycle rejected".into());
        }
        depth += 1;
        ancestor = r.get(&id).ok_or("foreign chain ancestor")?.owner;
    }
    if depth + child_height > MAX_CHAIN_DEPTH {
        return Err("sampler chain nesting exceeds 64".into());
    }
    // Registry validation above establishes the live allocation; no callback occurs under the lock.
    let raw = unsafe { reference(chain)? }.ctx.cast::<State>();
    match unsafe { reference_mut(raw)? } {
        State::Chain(children) => children.push(sampler),
        _ => return Err("sampler is not a chain".into()),
    }
    r.get_mut(&(sampler as usize))
        .ok_or("missing child registry entry")?
        .owner = Some(chain as usize);
    Ok(())
}
unsafe fn clone_impl(sampler: *const LlamaSampler) -> Result<*mut LlamaSampler> {
    if sampler.is_null() {
        return Ok(ptr::null_mut());
    }
    let h = unsafe { handle(sampler)? };
    if native_index(h.iface).is_some() {
        return match unsafe { state(sampler)? } {
            State::Builtin(b) => {
                let mut result = b.clone();
                result.sampler = b.sampler.clone_native().map_err(|e| e.to_string())?;
                if b.requested_seed == Some(u32::MAX)
                    && matches!(
                        b.sampler.config(),
                        Config::Mirostat { .. } | Config::MirostatV2 { .. }
                    )
                {
                    // The frozen Mirostat clone constructs a fresh seed_cur but copies its RNG.
                    result.effective_seed = resolve_seed(u32::MAX)?;
                }
                allocate(State::Builtin(result))
            }
            State::Bias(biases) => allocate(State::Bias(biases.clone())),
            State::Empty(label) => empty(label),
            State::Chain(children) => {
                let children = children.clone();
                let result = allocate(State::Chain(Vec::new()))?;
                for child in children {
                    match unsafe { clone_impl(child) } {
                        Ok(copy) => {
                            if let Err(error) = unsafe { chain_add_impl(result, copy) } {
                                let _ = unsafe { free_impl(copy, false) };
                                let _ = unsafe { free_impl(result, false) };
                                return Err(error);
                            }
                        }
                        Err(error) => {
                            let _ = unsafe { free_impl(result, false) };
                            return Err(error);
                        }
                    }
                }
                Ok(result)
            }
        };
    }
    if h.iface.is_null() {
        return allocate_shell(ptr::null_mut(), ptr::null_mut());
    }
    let interface = *unsafe { reference(h.iface)? };
    if let Some(callback) = interface.clone {
        let result = unsafe { callback(sampler) };
        if result.is_null() {
            return Err("custom sampler clone failed".into());
        }
        unsafe {
            handle(result)?;
        }
        let r = registry().lock().map_err(|_| "sampler registry poisoned")?;
        if std::ptr::eq(result, sampler)
            || r.get(&(result as usize))
                .ok_or("missing cloned sampler")?
                .owner
                .is_some()
        {
            return Err(
                "clone must return a new unowned sampler allocated by llama_sampler_init".into(),
            );
        }
        Ok(result)
    } else if h.ctx.is_null() {
        allocate_shell(h.iface, ptr::null_mut())
    } else {
        Err("custom sampler with context does not implement cloning".into())
    }
}
unsafe fn copy_compatible(source: *const LlamaSampler, target: *const LlamaSampler) -> Result<()> {
    let src = unsafe { handle(source)? };
    let dst = unsafe { handle(target)? };
    if src.iface != dst.iface {
        return Err("sampler_copy requires the same sampler interface/type".into());
    }
    if native_index(src.iface).is_none() {
        return Ok(());
    }
    match (unsafe { state(source)? }, unsafe { state(target)? }) {
        (State::Builtin(a), State::Builtin(b)) => {
            let mut ac = a.sampler.config().clone();
            let mut bc = b.sampler.config().clone();
            if let (Some(sa), Some(sb)) = (a.requested_seed, b.requested_seed) {
                config_seed(&mut ac, sa);
                config_seed(&mut bc, sb);
            }
            if ac != bc || a.vocabulary != b.vocabulary {
                return Err("sampler_copy requires identical built-in configuration".into());
            }
        }
        (State::Chain(a), State::Chain(b)) => {
            if a.len() != b.len() {
                return Err("sampler_copy requires identical chain topology".into());
            }
            for (&aa, &bb) in a.iter().zip(b) {
                unsafe {
                    copy_compatible(aa, bb)?;
                }
            }
        }
        (State::Bias(_), State::Bias(_)) | (State::Empty(_), State::Empty(_)) => {}
        _ => return Err("sampler_copy type mismatch".into()),
    }
    Ok(())
}
unsafe fn copy_impl(source: *const LlamaSampler, target: *mut LlamaSampler) -> Result<()> {
    if source.is_null() || target.is_null() || std::ptr::eq(source, target) {
        return Ok(());
    }
    unsafe {
        copy_compatible(source, target)?;
    }
    let src = unsafe { handle(source)? };
    if src.iface.is_null() {
        return Ok(());
    }
    if native_index(src.iface) == Some(CHAIN) {
        let from = unsafe { chain_children(source)? };
        let to = unsafe { chain_children(target)? };
        for (a, b) in from.into_iter().zip(to) {
            unsafe {
                copy_impl(a, b)?;
            }
        }
        return Ok(());
    }
    if native_index(src.iface).is_some() {
        let replacement = match unsafe { state(source)? } {
            State::Builtin(value) => {
                let mut b = value.clone();
                b.sampler = value.sampler.clone_native().map_err(|e| e.to_string())?;
                State::Builtin(b)
            }
            State::Bias(value) => State::Bias(value.clone()),
            State::Empty(label) => State::Empty(label),
            State::Chain(_) => unreachable!(),
        };
        *unsafe { state_mut(target)? } = replacement;
        return Ok(());
    }
    let interface = *unsafe { reference(src.iface)? };
    if let Some(callback) = interface.copy_state {
        unsafe { callback(source, target) };
        return Ok(());
    }
    let temporary = unsafe { clone_impl(source)? };
    if temporary == target {
        return Err("custom clone returned destination handle".into());
    }
    if let Some(callback) = interface.free {
        unsafe { callback(target) }
    }
    let tmp = unsafe { reference_mut(temporary)? };
    unsafe { reference_mut(target)? }.ctx = tmp.ctx;
    tmp.ctx = ptr::null_mut();
    registry()
        .lock()
        .map_err(|_| "sampler registry poisoned")?
        .remove(&(temporary as usize));
    // Only the empty shell is deleted; context ownership moved to the destination.
    unsafe {
        drop(Box::from_raw(temporary));
    }
    Ok(())
}
unsafe fn seed_impl(sampler: *const LlamaSampler) -> Result<u32> {
    let h = unsafe { handle(sampler)? };
    if native_index(h.iface).is_none() {
        return Ok(u32::MAX);
    }
    match unsafe { state(sampler)? } {
        State::Builtin(b)
            if matches!(
                b.sampler.config(),
                Config::Dist { .. } | Config::Mirostat { .. } | Config::MirostatV2 { .. }
            ) =>
        {
            Ok(b.effective_seed)
        }
        State::Chain(children) => {
            for &child in children.iter().rev() {
                let seed = unsafe { seed_impl(child)? };
                if seed != u32::MAX {
                    return Ok(seed);
                }
            }
            Ok(u32::MAX)
        }
        _ => Ok(u32::MAX), // Frozen get_seed does not report XTC/adaptive-p seeds.
    }
}

unsafe extern "C" fn native_name(sampler: *const LlamaSampler) -> *const c_char {
    boundary(ptr::null(), || unsafe { name_impl(sampler) })
}
unsafe extern "C" fn native_accept(sampler: *mut LlamaSampler, token: i32) {
    boundary((), || unsafe { accept_impl(sampler, token) })
}
unsafe extern "C" fn native_apply(sampler: *mut LlamaSampler, array: *mut LlamaTokenDataArray) {
    boundary((), || unsafe { apply_builtin(sampler, array) })
}
unsafe extern "C" fn native_reset(sampler: *mut LlamaSampler) {
    boundary((), || unsafe { reset_impl(sampler) })
}
unsafe extern "C" fn native_clone(sampler: *const LlamaSampler) -> *mut LlamaSampler {
    boundary(ptr::null_mut(), || unsafe { clone_impl(sampler) })
}
unsafe extern "C" fn native_free_state(sampler: *mut LlamaSampler) {
    boundary((), || unsafe { free_state_impl(sampler) })
}

ffi!(
    llama_rust_sampler_abi_layout(index: usize) -> usize,
    usize::MAX,
    { Ok(abi::sampler_layout(index)) }
);
ffi!(llama_sampler_chain_default_params()->LlamaSamplerChainParams,LlamaSamplerChainParams{no_perf:1},{Ok(LlamaSamplerChainParams{no_perf:1})});
ffi!(
    llama_sampler_init(iface: *mut LlamaSamplerI, ctx: *mut c_void) -> *mut LlamaSampler,
    ptr::null_mut(),
    {
        if !iface.is_null() {
            unsafe {
                reference(iface)?;
            }
        } else if !ctx.is_null() {
            return Err("null sampler interface cannot own a context".into());
        }
        allocate_shell(iface, ctx)
    }
);
ffi!(
    llama_sampler_name(sampler: *const LlamaSampler) -> *const c_char,
    ptr::null(),
    { unsafe { name_impl(sampler) } }
);
ffi!(
    llama_sampler_accept(sampler: *mut LlamaSampler, token: i32) -> (),
    (),
    { unsafe { accept_impl(sampler, token) } }
);
ffi!(
    llama_sampler_apply(sampler: *mut LlamaSampler, array: *mut LlamaTokenDataArray) -> (),
    (),
    { unsafe { apply_impl(sampler, array) } }
);
ffi!(llama_sampler_reset(sampler: *mut LlamaSampler) -> (), (), {
    unsafe { reset_impl(sampler) }
});
ffi!(
    llama_sampler_clone(sampler: *const LlamaSampler) -> *mut LlamaSampler,
    ptr::null_mut(),
    { unsafe { clone_impl(sampler) } }
);
ffi!(
    llama_sampler_copy(source: *const LlamaSampler, target: *mut LlamaSampler) -> (),
    (),
    { unsafe { copy_impl(source, target) } }
);
ffi!(llama_sampler_free(sampler: *mut LlamaSampler) -> (), (), {
    unsafe { free_impl(sampler, false) }
});
ffi!(
    llama_sampler_get_seed(sampler: *const LlamaSampler) -> u32,
    u32::MAX,
    { unsafe { seed_impl(sampler) } }
);
ffi!(
    llama_sampler_chain_init(params: LlamaSamplerChainParams) -> *mut LlamaSampler,
    ptr::null_mut(),
    {
        if !boolean(params.no_perf)? {
            return Err("sampler performance counters are not yet implemented".into());
        }
        allocate(State::Chain(Vec::new()))
    }
);
ffi!(
    llama_sampler_chain_add(chain: *mut LlamaSampler, sampler: *mut LlamaSampler) -> (),
    (),
    { unsafe { chain_add_impl(chain, sampler) } }
);
ffi!(
    llama_sampler_chain_get(chain: *mut LlamaSampler, index: i32) -> *mut LlamaSampler,
    ptr::null_mut(),
    {
        if chain.is_null() {
            return Ok(ptr::null_mut());
        }
        let h = unsafe { handle(chain)? };
        if native_index(h.iface) != Some(CHAIN) {
            return Ok(ptr::null_mut());
        }
        if index == -1 {
            return Ok(chain);
        }
        let children = unsafe { chain_children(chain)? };
        Ok(usize::try_from(index)
            .ok()
            .and_then(|i| children.get(i))
            .copied()
            .unwrap_or(ptr::null_mut()))
    }
);
ffi!(
    llama_sampler_chain_n(chain: *const LlamaSampler) -> i32,
    -1,
    { Ok(unsafe { chain_children(chain)? }.len() as i32) }
);
ffi!(
    llama_sampler_chain_remove(chain: *mut LlamaSampler, index: i32) -> *mut LlamaSampler,
    ptr::null_mut(),
    {
        let children = unsafe { chain_children(chain)? };
        let Some(index) = usize::try_from(index).ok().filter(|i| *i < children.len()) else {
            return Ok(ptr::null_mut());
        };
        let result = children[index];
        let State::Chain(values) = (unsafe { state_mut(chain)? }) else {
            unreachable!()
        };
        values.remove(index);
        registry()
            .lock()
            .map_err(|_| "sampler registry poisoned")?
            .get_mut(&(result as usize))
            .ok_or("removed sampler registry missing")?
            .owner = None;
        Ok(result)
    }
);
ffi!(
    llama_sampler_sample(
        sampler: *mut LlamaSampler,
        context: *mut ContextHandle,
        index: i32,
    ) -> i32,
    -1,
    {
        let values = unsafe { context::sampling_values(context, index)? };
        let mut values: Vec<_> = values
            .iter()
            .enumerate()
            .map(|(i, &v)| LlamaTokenData {
                id: i as i32,
                logit: v,
                p: 0.0,
            })
            .collect();
        let mut array = LlamaTokenDataArray {
            data: values.as_mut_ptr(),
            size: values.len(),
            selected: -1,
            sorted: 0,
        };
        unsafe {
            apply_impl(sampler, &mut array)?;
        }
        if array.selected < 0 || array.selected as u64 >= array.size as u64 {
            return Err("sampler did not select a valid candidate".into());
        }
        let token = unsafe { input_slice(array.data, array.size)? }[array.selected as usize].id;
        unsafe {
            accept_impl(sampler, token)?;
        }
        Ok(token)
    }
);

fn finite(values: &[f32]) -> Result<()> {
    if values.iter().any(|v| !v.is_finite()) {
        Err("sampler parameters must be finite".into())
    } else {
        Ok(())
    }
}
fn minimum_keep(value: usize) -> Result<()> {
    if value > MAX_CANDIDATES {
        Err("min_keep exceeds candidate limit".into())
    } else {
        Ok(())
    }
}
ffi!(llama_sampler_init_greedy()->*mut LlamaSampler,ptr::null_mut(),{builtin(Config::Greedy,None)});
ffi!(
    llama_sampler_init_dist(seed: u32) -> *mut LlamaSampler,
    ptr::null_mut(),
    { builtin(Config::Dist { seed }, Some(seed)) }
);
ffi!(
    llama_sampler_init_top_k(k: i32) -> *mut LlamaSampler,
    ptr::null_mut(),
    {
        if k <= 0 {
            empty(c"?top-k")
        } else {
            builtin(Config::TopK { k }, None)
        }
    }
);
ffi!(
    llama_sampler_init_top_p(p: f32, min_keep: usize) -> *mut LlamaSampler,
    ptr::null_mut(),
    {
        finite(&[p])?;
        minimum_keep(min_keep)?;
        if p >= 1.0 {
            empty(c"?top-p")
        } else {
            builtin(Config::TopP { p, min_keep }, None)
        }
    }
);
ffi!(
    llama_sampler_init_min_p(p: f32, min_keep: usize) -> *mut LlamaSampler,
    ptr::null_mut(),
    {
        finite(&[p])?;
        minimum_keep(min_keep)?;
        if p <= 0.0 {
            empty(c"?min-p")
        } else {
            builtin(Config::MinP { p, min_keep }, None)
        }
    }
);
ffi!(
    llama_sampler_init_typical(p: f32, min_keep: usize) -> *mut LlamaSampler,
    ptr::null_mut(),
    {
        finite(&[p])?;
        minimum_keep(min_keep)?;
        if p >= 1.0 {
            empty(c"?typical")
        } else {
            builtin(Config::Typical { p, min_keep }, None)
        }
    }
);
ffi!(
    llama_sampler_init_temp(t: f32) -> *mut LlamaSampler,
    ptr::null_mut(),
    {
        finite(&[t])?;
        if t == 1.0 {
            empty(c"?temp")
        } else {
            builtin(Config::Temperature { temperature: t }, None)
        }
    }
);
ffi!(
    llama_sampler_init_temp_ext(t: f32, delta: f32, exponent: f32) -> *mut LlamaSampler,
    ptr::null_mut(),
    {
        finite(&[t, delta, exponent])?;
        if t == 1.0 && delta <= 0.0 {
            empty(c"?temp-ext")
        } else {
            builtin(
                Config::DynamicTemperature {
                    temperature: t,
                    delta,
                    exponent,
                },
                None,
            )
        }
    }
);
ffi!(
    llama_sampler_init_xtc(p: f32, t: f32, min_keep: usize, seed: u32) -> *mut LlamaSampler,
    ptr::null_mut(),
    {
        finite(&[p, t])?;
        minimum_keep(min_keep)?;
        if p <= 0.0 || t > 0.5 {
            empty(c"?xtc")
        } else {
            builtin(
                Config::Xtc {
                    probability: p,
                    threshold: t,
                    min_keep,
                    seed,
                },
                Some(seed),
            )
        }
    }
);
ffi!(
    llama_sampler_init_top_n_sigma(n: f32) -> *mut LlamaSampler,
    ptr::null_mut(),
    {
        finite(&[n])?;
        if n <= 0.0 {
            empty(c"?top-n-sigma")
        } else {
            builtin(Config::TopNSigma { n }, None)
        }
    }
);
ffi!(
    llama_sampler_init_penalties(
        n_vocab: i32,
        last_n: i32,
        repeat: f32,
        frequency: f32,
        presence: f32,
    ) -> *mut LlamaSampler,
    ptr::null_mut(),
    {
        finite(&[repeat, frequency, presence])?;
        let last_n = last_n.max(0) as usize;
        if last_n > MAX_HISTORY {
            return Err("penalty history exceeds supported limit".into());
        }
        if last_n == 0 || (repeat == 1.0 && frequency == 0.0 && presence == 0.0) {
            return empty(c"?penalties");
        }
        if n_vocab <= 0 || n_vocab as usize > MAX_CANDIDATES {
            return Err("invalid penalty vocabulary size".into());
        }
        let result = builtin(
            Config::Penalties {
                last_n,
                repeat,
                frequency,
                presence,
            },
            None,
        )?;
        if let State::Builtin(b) = unsafe { state_mut(result)? } {
            b.vocabulary = Some(n_vocab as usize);
        }
        Ok(result)
    }
);
ffi!(
    llama_sampler_init_mirostat(
        n_vocab: i32,
        seed: u32,
        tau: f32,
        eta: f32,
        m: i32,
    ) -> *mut LlamaSampler,
    ptr::null_mut(),
    {
        builtin(
            Config::Mirostat {
                n_vocab: count(n_vocab)?,
                seed,
                tau,
                eta,
                m: count(m)?,
            },
            Some(seed),
        )
    }
);
ffi!(
    llama_sampler_init_mirostat_v2(seed: u32, tau: f32, eta: f32) -> *mut LlamaSampler,
    ptr::null_mut(),
    { builtin(Config::MirostatV2 { seed, tau, eta }, Some(seed)) }
);
ffi!(
    llama_sampler_init_logit_bias(
        n_vocab: i32,
        n_bias: i32,
        bias: *const LlamaLogitBias,
    ) -> *mut LlamaSampler,
    ptr::null_mut(),
    {
        if n_bias <= 0 {
            return empty(c"?logit-bias");
        }
        if n_vocab <= 0 || n_vocab as usize > MAX_CANDIDATES || n_bias as usize > MAX_CANDIDATES {
            return Err("logit bias vocabulary/count exceeds supported limits".into());
        }
        let biases = unsafe { input_slice(bias, count(n_bias)?)? }.to_vec();
        if biases
            .iter()
            .any(|b| b.bias.is_nan() || b.bias == f32::INFINITY)
        {
            return Err("logit biases must be finite or negative infinity".into());
        }
        allocate(State::Bias(biases))
    }
);

#[cfg(test)]
mod tests {
    use super::*;
    fn candidates() -> (Vec<LlamaTokenData>, LlamaTokenDataArray) {
        let mut data: Vec<_> = [0.4f32, 0.3, 0.2, 0.1]
            .iter()
            .enumerate()
            .map(|(id, p)| LlamaTokenData {
                id: id as i32,
                logit: p.ln(),
                p: 0.0,
            })
            .collect();
        let array = LlamaTokenDataArray {
            data: data.as_mut_ptr(),
            size: data.len(),
            selected: 2,
            sorted: 0,
        };
        (data, array)
    }
    #[test]
    fn native_filter_retains_selection_and_xtc_advances_pointer() {
        unsafe {
            let (data, mut array) = candidates();
            let original = array.data;
            let xtc = llama_sampler_init_xtc(1.0, 0.19, 1, 42);
            llama_sampler_apply(xtc, &mut array);
            assert_eq!(array.data, original.add(2));
            assert_eq!(array.size, 2);
            assert_eq!(array.selected, 2);
            assert_eq!(array.sorted, 1);
            assert!((data[0].p - 0.4).abs() < 1e-6);
            assert_eq!((*array.data).id, 2);
            llama_sampler_free(xtc);
        }
    }
    #[test]
    fn chain_owns_children_rejects_cycles_and_remove_transfers_ownership() {
        unsafe {
            let parent = llama_sampler_chain_init(llama_sampler_chain_default_params());
            let nested = llama_sampler_chain_init(llama_sampler_chain_default_params());
            let greedy = llama_sampler_init_greedy();
            llama_sampler_chain_add(parent, nested);
            llama_sampler_chain_add(nested, greedy);
            llama_sampler_chain_add(nested, parent);
            assert_eq!(llama_sampler_chain_n(nested), 1);
            llama_sampler_free(greedy);
            assert_eq!(llama_sampler_chain_get(nested, 0), greedy);
            assert_eq!(llama_sampler_chain_remove(nested, 0), greedy);
            llama_sampler_free(parent);
            let (_data, mut array) = candidates();
            llama_sampler_apply(greedy, &mut array);
            assert_eq!(array.selected, 0);
            llama_sampler_free(greedy);
            llama_sampler_free(greedy);
            assert!(
                CStr::from_ptr(llama_rust_last_error())
                    .to_str()
                    .unwrap()
                    .contains("already freed")
            );
        }
    }
    #[test]
    fn sampler_layout_matches_frozen_x64_contract() {
        use std::mem::{offset_of, size_of};
        assert_eq!(size_of::<LlamaTokenData>(), 12);
        assert_eq!(size_of::<LlamaSamplerChainParams>(), 1);
        if size_of::<usize>() == 8 {
            assert_eq!(size_of::<LlamaTokenDataArray>(), 32);
            assert_eq!(offset_of!(LlamaTokenDataArray, selected), 16);
            assert_eq!(offset_of!(LlamaTokenDataArray, sorted), 24);
            assert_eq!(size_of::<LlamaSampler>(), 16);
            assert_eq!(size_of::<LlamaSamplerI>(), 96);
            assert_eq!(offset_of!(LlamaSamplerI, copy_state), 88);
        }
    }
    #[test]
    fn cloned_distribution_and_copy_preserve_rng_and_reset() {
        unsafe {
            let a = llama_sampler_init_dist(42);
            for _ in 0..3 {
                let (_v, mut c) = candidates();
                llama_sampler_apply(a, &mut c);
            }
            let b = llama_sampler_clone(a);
            let d = llama_sampler_init_dist(42);
            llama_sampler_copy(a, d);
            for _ in 0..32 {
                let (_av, mut ac) = candidates();
                let (_bv, mut bc) = candidates();
                let (_dv, mut dc) = candidates();
                llama_sampler_apply(a, &mut ac);
                llama_sampler_apply(b, &mut bc);
                llama_sampler_apply(d, &mut dc);
                assert_eq!(ac.selected, bc.selected);
                assert_eq!(ac.selected, dc.selected);
            }
            llama_sampler_reset(a);
            llama_sampler_reset(b);
            for _ in 0..8 {
                let (_av, mut ac) = candidates();
                let (_bv, mut bc) = candidates();
                llama_sampler_apply(a, &mut ac);
                llama_sampler_apply(b, &mut bc);
                assert_eq!(ac.selected, bc.selected);
            }
            llama_sampler_free(a);
            llama_sampler_free(b);
            llama_sampler_free(d);
        }
    }
    #[test]
    fn malformed_candidates_do_not_advance_builtin_rng() {
        unsafe {
            let a = llama_sampler_init_dist(7);
            let b = llama_sampler_clone(a);
            let (_values, mut c) = candidates();
            (*c.data).logit = f32::NAN;
            let previous = c.selected;
            llama_sampler_apply(a, &mut c);
            assert_eq!(c.selected, previous);
            let (_av, mut ac) = candidates();
            let (_bv, mut bc) = candidates();
            llama_sampler_apply(a, &mut ac);
            llama_sampler_apply(b, &mut bc);
            assert_eq!(ac.selected, bc.selected);
            llama_sampler_free(a);
            llama_sampler_free(b);
        }
    }

    #[cfg(target_env = "msvc")]
    #[test]
    fn frozen_windows_tied_sort_regressions() {
        // Independently captured by the frozen-header C++ oracle; probabilities
        // alone cannot catch a tied token permutation before seeded selection.
        let cases: &[(usize, usize, usize, &[i32])] = &[
            (
                4097,
                2,
                28,
                &[
                    1108, 1332, 660, 996, 1556, 548, 1220, 324, 1444, 772, 940, 1668, 492, 1052,
                    268, 1164,
                ],
            ),
            (
                129,
                0,
                32,
                &[
                    104, 20, 27, 34, 41, 48, 69, 55, 62, 6, 13, 76, 83, 90, 97, 111,
                ],
            ),
        ];
        unsafe {
            for &(n, step, expected_size, expected_ids) in cases {
                let sampler = if n > 1024 {
                    llama_sampler_init_top_p(0.03, 1)
                } else {
                    llama_sampler_init_typical(0.7, 1)
                };
                let mut values: Vec<_> = (0..n)
                    .map(|i| LlamaTokenData {
                        id: i as i32,
                        logit: ((i * 37 + step) % 7) as f32 - 3.0,
                        p: 0.0,
                    })
                    .collect();
                let mut array = LlamaTokenDataArray {
                    data: values.as_mut_ptr(),
                    size: n,
                    selected: 17,
                    sorted: 0,
                };
                llama_rust_clear_error();
                llama_sampler_apply(sampler, &mut array);
                assert_eq!(CStr::from_ptr(llama_rust_last_error()).to_bytes(), b"");
                assert_eq!(array.size, expected_size);
                assert_eq!(array.selected, 17);
                let ids: Vec<_> = values
                    .iter()
                    .take(expected_ids.len())
                    .map(|v| v.id)
                    .collect();
                assert_eq!(ids, expected_ids);
                llama_sampler_free(sampler);
            }
        }
    }

    #[test]
    fn penalty_accept_rejects_out_of_vocabulary_before_history_mutation() {
        unsafe {
            let sampler = llama_sampler_init_penalties(3, 4, 1.1, 0.2, 0.3);
            assert!(!sampler.is_null());
            llama_rust_clear_error();
            llama_sampler_accept(sampler, 3);
            assert!(
                CStr::from_ptr(llama_rust_last_error())
                    .to_str()
                    .unwrap()
                    .contains("outside sampler vocabulary")
            );
            llama_rust_clear_error();
            llama_sampler_accept(sampler, 2);
            assert_eq!(CStr::from_ptr(llama_rust_last_error()).to_bytes(), b"");
            llama_sampler_free(sampler);
        }
    }
}
