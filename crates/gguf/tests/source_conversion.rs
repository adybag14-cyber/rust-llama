//! The Python converter wire extension must not broaden strict GGUF admission.
use llama_gguf::{
    Error, Gguf, Limits, RawGguf, RawValue, Value, write_header, write_raw_header,
    write_source_conversion_header,
};
use std::io::{self, Cursor, Write};
struct NoWrite;
impl Write for NoWrite {
    fn write(&mut self, _: &[u8]) -> io::Result<usize> {
        panic!("rejected metadata wrote output before validation")
    }
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}
fn nested() -> RawValue {
    RawValue::Array {
        element_type: 9,
        values: vec![
            RawValue::Array {
                element_type: 5,
                values: vec![RawValue::I32(1), RawValue::I32(2)],
            },
            RawValue::Array {
                element_type: 8,
                values: vec![RawValue::String(b"x".to_vec())],
            },
        ],
    }
}
#[test]
fn source_conversion_arrays_leave_strict_writers_and_readers_unchanged() {
    let value = nested();
    let values = [(b"head".as_slice(), &value)];
    assert!(write_raw_header(&mut NoWrite, &values, &[], &Limits::default()).is_err());
    let ordinary = Value::Array {
        element_type: 9,
        values: vec![Value::Array {
            element_type: 5,
            values: vec![Value::I32(1)],
        }],
    };
    assert!(
        write_header(
            &mut NoWrite,
            &[("head", &ordinary)],
            &[],
            &Limits::default()
        )
        .is_err()
    );
    let mut bytes = Vec::new();
    let layout =
        write_source_conversion_header(&mut bytes, &values, &[], &Limits::default()).unwrap();
    assert_eq!(bytes.len() as u64, layout.data_offset);
    assert!(Gguf::read(Cursor::new(&bytes), Limits::default()).is_err());
    assert!(RawGguf::read(Cursor::new(&bytes), Limits::default()).is_err());
}
#[test]
fn source_conversion_arrays_bound_nesting_and_cumulative_work_before_writes() {
    let value = nested();
    for limits in [
        Limits {
            max_total_values: 4,
            ..Limits::default()
        },
        Limits {
            max_array_values: 1,
            ..Limits::default()
        },
        Limits {
            max_string_bytes: 0,
            ..Limits::default()
        },
        Limits {
            max_metadata_bytes: 31,
            ..Limits::default()
        },
    ] {
        assert!(matches!(
            write_source_conversion_header(
                &mut NoWrite,
                &[(b"x".as_slice(), &value)],
                &[],
                &limits
            ),
            Err(Error::Limit(_))
        ));
    }
    for invalid in [
        RawValue::Array {
            element_type: 9,
            values: vec![RawValue::I32(1)],
        },
        RawValue::Array {
            element_type: 9,
            values: vec![RawValue::Array {
                element_type: 5,
                values: vec![],
            }],
        },
    ] {
        assert!(
            write_source_conversion_header(
                &mut NoWrite,
                &[(b"x".as_slice(), &invalid)],
                &[],
                &Limits::default(),
            )
            .is_err()
        );
    }
    let mut deep = RawValue::Array {
        element_type: 5,
        values: vec![RawValue::I32(1)],
    };
    for _ in 0..63 {
        deep = RawValue::Array {
            element_type: 9,
            values: vec![deep],
        };
    }
    write_source_conversion_header(
        &mut io::sink(),
        &[(b"x".as_slice(), &deep)],
        &[],
        &Limits::default(),
    )
    .unwrap();
    deep = RawValue::Array {
        element_type: 9,
        values: vec![deep],
    };
    assert!(matches!(
        write_source_conversion_header(
            &mut NoWrite,
            &[(b"x".as_slice(), &deep)],
            &[],
            &Limits::default()
        ),
        Err(Error::Limit(_))
    ));
}
