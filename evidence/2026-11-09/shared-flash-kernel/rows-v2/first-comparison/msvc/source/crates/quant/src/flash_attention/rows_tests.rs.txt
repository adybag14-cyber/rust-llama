use super::*;

fn strides(shape: [usize; 4], width: usize) -> [usize; 4] {
    [
        width,
        width * shape[0],
        width * shape[0] * shape[1],
        width * shape[0] * shape[1] * shape[2],
    ]
}
fn view(values: &[f32], shape: [usize; 4]) -> View4<'_> {
    View4::from_f32(values, shape, strides(shape, 4), 0).unwrap()
}
fn mask(values: &[u16], shape: [usize; 4]) -> View4<'_> {
    View4::from_f16(values, shape, strides(shape, 2), 0).unwrap()
}
#[test]
fn row_map_storage_preserves_signed_values_and_checks_full_spans() {
    let words = [99, 98, -1, 0, 3, 1, i32::MIN, 2];
    let typed = KvRows::from_i32(&words, [3, 2], 8).unwrap();
    let bytes = words
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect::<Vec<_>>();
    let raw = KvRows::from_bytes(&bytes, [3, 2], 8).unwrap();
    assert_eq!(typed.shape(), [3, 2]);
    assert_eq!(raw.byte_offset(), 8);
    for batch in 0..2 {
        for column in 0..3 {
            assert_eq!(raw.value(column, batch), words[2 + batch * 3 + column]);
            assert_eq!(typed.value(column, batch), raw.value(column, batch));
        }
    }
    for shape in [[0, 2], [3, 0], [3, 3], [i32::MAX as usize + 1, 1]] {
        assert!(KvRows::from_i32(&words, shape, 8).is_err());
    }
    assert!(KvRows::from_bytes(&bytes, [3, 2], 2).is_err());
    assert!(KvRows::from_bytes(&bytes[..bytes.len() - 1], [3, 2], 8).is_err());
    assert!(matches!(
        KvRows::from_bytes(&[], [1, 1], usize::MAX - 3),
        Err(Error::Overflow)
    ));
}
#[test]
fn row_map_shapes_and_positive_indices_are_checked_atomically() {
    let q = [0.0; 4];
    let k = [0.0; 3];
    let v = [10.0, 20.0, 30.0];
    let mut masks = [0_u16; 12];
    masks[2] = 0xfc00;
    masks[5] = 0xfc00;
    let rows = [2, 0, -1, 1, 1, 2];
    let inputs = Inputs {
        q: view(&q, [1, 2, 1, 2]),
        k: view(&k, [1, 3, 1, 1]),
        v: view(&v, [1, 3, 1, 1]),
        mask: Some(mask(&masks, [3, 2, 1, 2])),
        sinks: None,
    };
    let mapping = KvRows::from_i32(&rows, [3, 2], 0).unwrap();
    assert!(Plan::new(inputs, Params::default()).is_err());
    let plan = Plan::new_with_rows(inputs, mapping, Params::default()).unwrap();
    assert_eq!(plan.branch(), Branch::OneChunk);
    let output = plan.execute(Selection::All, Limits::default()).unwrap();
    assert_eq!(output.shape, [1, 1, 2, 2]);
    assert_eq!(&output.values[..2], &[20.0, 20.0]);
    let mut missing_mask = inputs;
    missing_mask.mask = None;
    assert!(Plan::new_with_rows(missing_mask, mapping, Params::default()).is_err());
    assert!(
        Plan::new_with_rows(
            inputs,
            KvRows::from_i32(&rows, [2, 2], 0).unwrap(),
            Params::default()
        )
        .is_err()
    );
    assert!(
        Plan::new_with_rows(
            inputs,
            KvRows::from_i32(&rows, [3, 1], 0).unwrap(),
            Params::default()
        )
        .is_err()
    );
    let extra_k = [0.0; 6];
    let mut unshared = inputs;
    unshared.k = view(&extra_k, [1, 3, 1, 2]);
    assert!(Plan::new_with_rows(unshared, mapping, Params::default()).is_err());
    let bad_rows = [2, 0, -1, 1, 1, i32::MAX];
    let bad = Plan::new_with_rows(
        inputs,
        KvRows::from_i32(&bad_rows, [3, 2], 0).unwrap(),
        Params::default(),
    )
    .unwrap();
    assert!(matches!(
        bad.execute(
            Selection::All,
            Limits {
                max_bytes: usize::MAX,
                max_scalar_work: 0
            }
        ),
        Err(Error::WorkLimit { .. })
    ));
    assert!(matches!(
        bad.execute(
            Selection::Row {
                query: 0,
                head: 0,
                batch: 0
            },
            Limits::default()
        ),
        Err(Error::Invalid(_))
    ));
    assert_eq!(q, [0.0; 4]);
    assert_eq!(k, [0.0; 3]);
    assert_eq!(v, [10.0, 20.0, 30.0]);
    assert_eq!(rows, [2, 0, -1, 1, 1, 2]);
}
#[test]
fn logical_row_columns_disable_split_and_are_distinct_from_physical_cache_rows() {
    let q = [0.0];
    let k = [0.0; 768];
    let v = [1.0; 768];
    let masks = [0_u16; 512];
    let rows = (0..512)
        .map(|index| if index < 264 { index + 256 } else { -1 })
        .collect::<Vec<i32>>();
    let mut masked = masks;
    masked[264..].fill(0xfc00);
    let inputs = Inputs {
        q: view(&q, [1, 1, 1, 1]),
        k: view(&k, [1, 768, 1, 1]),
        v: view(&v, [1, 768, 1, 1]),
        mask: Some(mask(&masked, [512, 1, 1, 1])),
        sinks: None,
    };
    let plan = Plan::new_with_rows(
        inputs,
        KvRows::from_i32(&rows, [512, 1], 0).unwrap(),
        Params {
            workers: 3,
            ..Params::default()
        },
    )
    .unwrap();
    assert_eq!(plan.branch(), Branch::OneChunk);
    let result = plan.execute(Selection::All, Limits::default()).unwrap();
    assert_eq!(result.values, [1.0]);
    assert_eq!(result.stats.worker_threads, 3);
}
#[test]
fn selected_row_work_budget_charges_the_complete_map_validation() {
    let data = [1.0];
    let columns = 256;
    let batches = 64;
    let masks = vec![0_u16; columns * QUERY_TILE];
    let rows = vec![0_i32; columns * batches];
    let small_rows = &rows[..columns];
    let q = View4::from_f32(&data, [1, QUERY_TILE, 1, batches], [4, 0, 0, 0], 0).unwrap();
    let kv = view(&data, [1, 1, 1, 1]);
    let inputs = Inputs {
        q,
        k: kv,
        v: kv,
        mask: Some(mask(&masks, [columns, QUERY_TILE, 1, 1])),
        sinks: None,
    };
    let large = Plan::new_with_rows(
        inputs,
        KvRows::from_i32(&rows, [columns, batches], 0).unwrap(),
        Params::default(),
    )
    .unwrap();
    let mut small_inputs = inputs;
    small_inputs.q = View4::from_f32(&data, [1, QUERY_TILE, 1, 1], [4, 0, 0, 0], 0).unwrap();
    let small = Plan::new_with_rows(
        small_inputs,
        KvRows::from_i32(small_rows, [columns, 1], 0).unwrap(),
        Params::default(),
    )
    .unwrap();
    let selection = Selection::Row {
        query: 0,
        head: 0,
        batch: 0,
    };
    let minimum_extra_visits = ((rows.len() - small_rows.len()) * 4) as u64;
    let small_work = small.requirements(selection).unwrap().scalar_work_items;
    assert!(
        large.requirements(selection).unwrap().scalar_work_items
            >= small_work + minimum_extra_visits
    );
    assert!(matches!(
        large.execute(
            selection,
            Limits {
                max_bytes: usize::MAX,
                max_scalar_work: small_work + minimum_extra_visits - 1
            }
        ),
        Err(Error::WorkLimit { .. })
    ));
}
