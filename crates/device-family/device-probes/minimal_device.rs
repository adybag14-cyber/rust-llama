//! Compiler availability probe only; not a GPU entry point or execution test.
#![no_std]

#[unsafe(no_mangle)]
pub extern "C" fn add(a: f32, b: f32) -> f32 {
    a + b
}
