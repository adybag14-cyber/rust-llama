use super::*;
use llama_behavior::json::{self, Value};
use llama_conformance::digest::{Sha256, hex};

struct Reader<'a> {
    data: &'a [u8],
    position: usize,
}
impl<'a> Reader<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self { data, position: 0 }
    }
    fn take(&mut self, size: usize) -> &'a [u8] {
        let end = self.position.checked_add(size).unwrap();
        let result = &self.data[self.position..end];
        self.position = end;
        result
    }
    fn u32(&mut self) -> u32 {
        u32::from_le_bytes(self.take(4).try_into().unwrap())
    }
    fn u64(&mut self) -> u64 {
        u64::from_le_bytes(self.take(8).try_into().unwrap())
    }
    fn string(&mut self) -> &'a [u8] {
        let size = self.u32() as usize;
        self.take(size)
    }
    fn end(&self) {
        assert_eq!(self.position, self.data.len());
    }
}
fn document() -> Value {
    json::parse(include_str!(
        "../../../fixtures/ffi/model-metadata/policy-reference.json"
    ))
    .unwrap()
}
fn number(value: &Value) -> usize {
    value.as_u64().unwrap() as usize
}
fn text(value: &Value) -> &str {
    value.as_str().unwrap()
}
fn field<'a>(value: &'a Value, name: &str) -> &'a Value {
    value.get(name).unwrap()
}
fn word(hash: &mut Sha256, value: usize) {
    hash.update(&u32::try_from(value).unwrap().to_le_bytes());
}
fn string(hash: &mut Sha256, bytes: &[u8]) {
    word(hash, bytes.len());
    hash.update(bytes);
}
fn order(hash: &mut Sha256, map: &mut NativeMap<'_>) {
    let indices = map.order().unwrap();
    word(hash, indices.len());
    for index in indices {
        word(hash, index);
    }
}
#[test]
fn every_native_hash_growth_snapshot_and_iteration_index() {
    let expected = document();
    for (profile, name) in [(Profile::Msvc, "msvc"), (Profile::Gnu, "gnu")] {
        let mut input = Reader::new(include_bytes!(
            "../../../fixtures/ffi/model-metadata/policy-maps-input.bin"
        ));
        assert_eq!(input.take(8), b"MDMI0001");
        let count = input.u32() as usize;
        let maps = field(field(&expected, name), "maps").as_array().unwrap();
        assert_eq!(count, maps.len());
        let mut hashed = 0usize;
        for reference in maps {
            let label = input.string();
            assert_eq!(label, text(field(reference, "name")).as_bytes());
            let n = input.u32() as usize;
            let flags = input.u32();
            let generated = input.u32();
            assert_eq!(n, number(field(reference, "inputs")));
            let owned: Vec<Vec<u8>> = if generated == 1 {
                assert_eq!(label, b"growth-million");
                (0..n)
                    .map(|i| format!("metadata.growth.{i:07}").into_bytes())
                    .collect()
            } else {
                assert_eq!(generated, 0);
                (0..n).map(|_| input.string().to_vec()).collect()
            };
            let keys: Vec<&[u8]> = owned.iter().map(Vec::as_slice).collect();
            let limits = Limits {
                max_bytes: 128 * 1024 * 1024,
                max_work: 512 * 1024 * 1024,
            };
            let mut map = NativeMap::new(&keys, profile, limits).unwrap();
            let mut hash = Sha256::new();
            string(&mut hash, label);
            word(&mut hash, n);
            word(&mut hash, map.buckets.len());
            for i in 0..n {
                let (value, grown) = map.insert(i).unwrap();
                hash.update(&value.to_le_bytes());
                word(&mut hash, map.nodes.len());
                word(&mut hash, map.buckets.len());
                let snapshot = grown && flags & 1 != 0;
                word(&mut hash, usize::from(snapshot));
                if snapshot {
                    order(&mut hash, &mut map);
                }
            }
            word(&mut hash, map.buckets.len());
            order(&mut hash, &mut map);
            assert_eq!(
                hex(&hash.finish()),
                text(field(reference, "sha256")),
                "{name} {}",
                String::from_utf8_lossy(label)
            );
            assert_eq!(map.nodes.len(), number(field(reference, "unique")));
            // Exercise the production construction path separately on every
            // ordinary map; the million-key case already exhausts all growth.
            if generated == 0 {
                let built = default_map_order(&keys, profile, limits).unwrap();
                assert_eq!(built.indices, map.order().unwrap());
            }
            hashed += n;
        }
        input.end();
        assert_eq!((count, hashed), (282, 1_063_864));
    }
}
#[test]
fn original_float_strings_preserve_rounding_sign_payload_and_full_ieee_range() {
    let expected = document();
    for (profile, name) in [(Profile::Msvc, "msvc"), (Profile::Gnu, "gnu")] {
        let mut input = Reader::new(include_bytes!(
            "../../../fixtures/ffi/model-metadata/policy-floats-input.bin"
        ));
        assert_eq!(input.take(8), b"MDFLT001");
        let policy = field(field(&expected, name), "floats");
        for width in [32, 64] {
            let total = input.u32() as usize;
            let data = field(policy, &width.to_string());
            assert_eq!(total, number(field(data, "patterns")));
            let mut seen = 0;
            for block in field(data, "chunks").as_array().unwrap() {
                assert_eq!(seen, number(field(block, "start")));
                let mut hash = Sha256::new();
                for _ in 0..number(field(block, "count")) {
                    let formatted = if width == 32 {
                        let bits = input.u32();
                        hash.update(&bits.to_le_bytes());
                        format_f32(f32::from_bits(bits), profile).unwrap()
                    } else {
                        let bits = input.u64();
                        hash.update(&bits.to_le_bytes());
                        format_f64(f64::from_bits(bits), profile).unwrap()
                    };
                    string(&mut hash, formatted.as_bytes());
                    seen += 1;
                }
                assert_eq!(
                    hex(&hash.finish()),
                    text(field(block, "sha256")),
                    "{name} f{width} chunk ending at {seen}"
                );
            }
            assert_eq!(seen, total);
            assert_eq!(total, if width == 32 { 95_003 } else { 127_259 });
        }
        input.end();
    }
}
#[test]
fn policy_limits_reject_before_unbounded_growth_or_collision_work() {
    let owned: Vec<_> = (0..1200).map(|i| format!("metadata.key.{i}")).collect();
    let keys: Vec<_> = owned.iter().map(String::as_bytes).collect();
    for profile in [Profile::Msvc, Profile::Gnu] {
        let good = default_map_order(
            &keys,
            profile,
            Limits {
                max_bytes: 1 << 24,
                max_work: 1 << 24,
            },
        )
        .unwrap();
        assert_eq!(good.indices.len(), keys.len());
        let exact = default_map_order(
            &keys,
            profile,
            Limits {
                max_bytes: good.peak_bytes,
                max_work: good.work_units,
            },
        )
        .unwrap();
        assert_eq!(good.indices, exact.indices);
        assert!(
            default_map_order(
                &keys,
                profile,
                Limits {
                    max_bytes: good.peak_bytes - 1,
                    max_work: 1 << 24
                }
            )
            .is_err()
        );
        assert!(
            default_map_order(
                &keys,
                profile,
                Limits {
                    max_bytes: 1 << 24,
                    max_work: good.work_units - 1
                }
            )
            .is_err()
        );
        let duplicates = [b"same".as_slice(); 64];
        assert_eq!(
            default_map_order(
                &duplicates,
                profile,
                Limits {
                    max_bytes: 1 << 20,
                    max_work: 1 << 20
                }
            )
            .unwrap()
            .indices,
            vec![0]
        );
    }
    assert!(bytes(usize::MAX, 2).is_err());
    assert!(add(usize::MAX, 1).is_err());
}
