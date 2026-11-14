//! Analytical contracts for the instruction interpreter and explicit real CUDA
//! dispatch. The actual model tests use separately captured frozen C outputs.
use llama_device_family::{
    driver::CudaDriver,
    ptx::{self, DecoderKernel as D, KernelKind, KernelSpec, PtxTarget},
};
fn bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|v| v.to_le_bytes()).collect()
}
fn floats(v: &[u8]) -> Vec<f32> {
    v.as_chunks::<4>()
        .0
        .iter()
        .map(|v| f32::from_le_bytes(*v))
        .collect()
}
fn params(token: u32, position: u32) -> Vec<u8> {
    [token.to_le_bytes(), position.to_le_bytes()].concat()
}
struct Case {
    name: &'static str,
    kind: D,
    buffers: Vec<Vec<u8>>,
    expected: Vec<f32>,
    exact: bool,
}
fn case(name: &'static str, kind: D, mut inputs: Vec<Vec<u8>>, expected: Vec<f32>) -> Case {
    let generated = ptx::generate(
        &KernelSpec {
            kind: KernelKind::Decoder(kind.clone()),
        },
        PtxTarget::default(),
    )
    .unwrap();
    let output = generated.output_buffer();
    inputs.insert(output, bytes(&vec![-91.; expected.len()]));
    let exact = matches!(kind, D::ActivationQ8 { .. });
    Case {
        exact,
        name,
        kind,
        buffers: inputs,
        expected,
    }
}
fn cases() -> Vec<Case> {
    let mut all = Vec::new();
    // Independent rounding boundary: division gives F32 mean 0x3fc00002,
    // while multiplication by the rounded F64 reciprocal gives 0x3fc00001.
    // The frozen RMS row divides its F64 sum by the actual row dimension.
    let mut rms_boundary = vec![
        12.0f32,
        1.0,
        1.0,
        1.0,
        1.0 / 256.0,
        1.0 / 1024.0,
        1.0 / 1024.0,
        1.0 / 2048.0,
        1.0 / 4096.0,
        1.0 / 4096.0,
    ];
    rms_boundary.resize(98, 0.0);
    let scale = f32::from_bits(0x3f51_05bc);
    let mut scalar = case(
        "RMS division rounding boundary",
        D::RmsScale {
            width: 98,
            epsilon: 1e-5,
        },
        vec![bytes(&rms_boundary)],
        vec![scale],
    );
    scalar.exact = true;
    all.push(scalar);
    let mut headed = case(
        "head RMS division rounding boundary",
        D::HeadRms {
            width: 98,
            head_dimension: 98,
            epsilon: 1e-5,
        },
        vec![bytes(&rms_boundary), bytes(&[1.0; 98])],
        rms_boundary.iter().map(|x| x * scale).collect(),
    );
    headed.exact = true;
    all.push(headed);
    for width in [1usize, 3, 4, 7, 16] {
        let input = [
            -103.0f32,
            -90.0,
            -88.7,
            -87.0,
            -10.0,
            -1.0,
            -0.0,
            0.0,
            0.1,
            1.0,
            10.0,
            88.0,
            100.0,
            134.0,
            f32::MIN_POSITIVE,
            -134.0,
        ][..width]
            .to_vec();
        let mut expected = input.clone();
        llama_quant::cpu_math::silu_f32_in_place(&mut expected);
        all.push(case(
            "SiLU source exponent ranges and scalar tails",
            D::SiluGate {
                width: width as u32,
            },
            vec![bytes(&input), bytes(&vec![1.0; width])],
            expected,
        ));
    }
    for (heads, dimension) in [(1u32, 2u32), (3, 6), (2, 64), (8, 128)] {
        let width = heads * dimension;
        let input: Vec<f32> = (0..width)
            .map(|i| (i as i32 % 23 - 11) as f32 / 8.)
            .collect();
        let weight: Vec<f32> = (0..dimension)
            .map(|i| 0.5 + (i % 11) as f32 / 16.)
            .collect();
        let expected: Vec<f32> = input
            .chunks(dimension as usize)
            .flat_map(|head| {
                let square_sum: f64 = head.iter().map(|&v| f64::from(v * v)).sum();
                let scale = ((square_sum / f64::from(dimension)) as f32 + 1e-5)
                    .sqrt()
                    .recip();
                head.iter()
                    .zip(&weight)
                    .map(move |(&v, &w)| (v * scale) * w)
            })
            .collect();
        all.push(case(
            "independent head RMS",
            D::HeadRms {
                width,
                head_dimension: dimension,
                epsilon: 1e-5,
            },
            vec![bytes(&input), bytes(&weight)],
            expected,
        ));
        for offset in [0, width / 2, width - 1] {
            all.push(case(
                "fused projection slice",
                D::Slice {
                    source_width: width,
                    offset,
                    width: width - offset,
                },
                vec![bytes(&input)],
                input[offset as usize..].to_vec(),
            ));
        }
        let phases: Vec<f32> = (0..dimension / 2)
            .flat_map(|d| {
                let angle = (d as f32 + 1.) / 16.;
                [angle.cos(), angle.sin()]
            })
            .collect();
        let mut expected = input.clone();
        for head in 0..heads as usize {
            for pair in 0..dimension as usize / 2 {
                let first = head * dimension as usize + pair;
                let second = first + dimension as usize / 2;
                let (c, s) = (phases[2 * pair], phases[2 * pair + 1]);
                let (x, y) = (input[first], input[second]);
                expected[first] = x * c - y * s;
                expected[second] = x * s + y * c;
            }
        }
        all.push(case(
            "half-pair multihead RoPE",
            D::RopeNeox {
                width,
                head_dimension: dimension,
                context: 1,
            },
            vec![bytes(&input), bytes(&phases), params(0, 0)],
            expected,
        ));
    }
    for width in [2u32, 8, 258] {
        let input: Vec<f32> = (0..width)
            .map(|i| (i as i32 % 17 - 8) as f32 * 0.125)
            .collect();
        let weight: Vec<f32> = (0..width).map(|i| 0.75 + (i % 7) as f32 * 0.0625).collect();
        let scale = 1.0
            / ((input.iter().map(|&x| f64::from(x * x)).sum::<f64>() / f64::from(width)) as f32
                + 1e-5)
                .sqrt();
        all.push(case(
            "RMS scale",
            D::RmsScale {
                width,
                epsilon: 1e-5,
            },
            vec![bytes(&input)],
            vec![scale],
        ));
        all.push(case(
            "RMS weighted",
            D::RmsApply { width },
            vec![bytes(&input), bytes(&weight), bytes(&[scale])],
            input
                .iter()
                .zip(&weight)
                .map(|(&x, &w)| x * scale * w)
                .collect(),
        ));
        let mut activated = input.clone();
        llama_quant::cpu_math::silu_f32_in_place(&mut activated);
        all.push(case(
            "SiLU gate",
            D::SiluGate { width },
            vec![bytes(&input), bytes(&weight)],
            activated
                .iter()
                .zip(&weight)
                .map(|(&x, &w)| x * w)
                .collect(),
        ));
        let mut embedding = input.clone();
        embedding.extend(input.iter().map(|x| x + 2.0));
        all.push(case(
            "embedding second row",
            D::Embedding {
                vocabulary: 2,
                width,
            },
            vec![bytes(&embedding), params(1, 0)],
            input.iter().map(|x| x + 2.0).collect(),
        ));
    }
    let matrix = vec![1.0, -2.0, 0.5, 4.0, 0.25, -0.5, 3.0, 0.0];
    let vector = vec![2.0, -1.0, 4.0, -0.5];
    all.push(case(
        "F32 projection",
        D::MatVec {
            rows: 2,
            columns: 4,
            half: false,
        },
        vec![bytes(&matrix), bytes(&vector)],
        vec![4., 13.],
    ));
    let half: Vec<u8> = [0x3c00u16, 0xc000, 0x3800, 0x4400, 0x3400, 0xb800, 0x4200, 0]
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();
    all.push(case(
        "F16 projection",
        D::MatVec {
            rows: 2,
            columns: 4,
            half: true,
        },
        vec![half, bytes(&vector)],
        vec![4., 13.],
    ));
    let mut q8 = Vec::new();
    let mut expected_q8 = Vec::new();
    let vector_q8: Vec<f32> = (0..64).map(|j| (j as f32 - 31.0) * 0.125).collect();
    for row in 0..3 {
        let mut total = 0.0f64;
        for block in 0..2 {
            q8.extend_from_slice(&0x3400u16.to_le_bytes());
            for j in 0..32 {
                let q = (row * 53 + block * 31 + j * 7) as u8 as i8;
                q8.push(q as u8);
                total += f64::from((f32::from(q) * 0.25) * vector_q8[block * 32 + j]);
            }
        }
        expected_q8.push(total as f32);
    }
    all.push(case(
        "signed Q8 projection",
        D::Q8MatVec {
            rows: 3,
            columns: 64,
        },
        vec![q8, bytes(&vector_q8)],
        expected_q8,
    ));
    for context in [1u32, 3, 17] {
        for position in [0, context - 1] {
            let row = vec![1.0, 2.0, 3.0, 4.0];
            let mut cached = vec![-91.; context as usize * 4];
            cached[position as usize * 4..position as usize * 4 + 4].copy_from_slice(&row);
            all.push(case(
                "KV insertion",
                D::CacheWrite { width: 4, context },
                vec![bytes(&row), params(0, position)],
                cached,
            ));
            let mut phases = Vec::new();
            for p in 0..context {
                for d in 0..2 {
                    let theta = p as f32 * 0.1 * (d as f32 + 1.0);
                    phases.extend([theta.cos(), theta.sin()]);
                }
            }
            let mut rotated = Vec::new();
            for d in 0..2 {
                let (c, s) = (
                    phases[position as usize * 4 + 2 * d],
                    phases[position as usize * 4 + 2 * d + 1],
                );
                let (x, y) = (row[2 * d], row[2 * d + 1]);
                rotated.extend([x * c - y * s, x * s + y * c]);
            }
            all.push(case(
                "adjacent RoPE",
                D::Rope {
                    width: 4,
                    head_dimension: 4,
                    context,
                },
                vec![bytes(&row), bytes(&phases), params(0, position)],
                rotated,
            ));
            for (heads, kv_heads, dimension) in [(2u32, 2u32, 2u32), (4, 1, 2), (6, 2, 4)] {
                let q: Vec<f32> = (0..heads * dimension)
                    .map(|i| (i as i32 % 9 - 4) as f32 / 8.0)
                    .collect();
                let keys: Vec<f32> = (0..context * kv_heads * dimension)
                    .map(|i| (i as i32 % 13 - 6) as f32 / 8.0)
                    .collect();
                let values: Vec<f32> = (0..context * kv_heads * dimension)
                    .map(|i| (i as i32 % 19 - 9) as f32 / 4.0)
                    .collect();
                let scale = 0.5;
                let mut scores = vec![f32::NEG_INFINITY; (heads * context) as usize];
                let mut probabilities = vec![0.; scores.len()];
                let mut statistics = Vec::new();
                let mut weighted = Vec::new();
                for h in 0..heads as usize {
                    let kv = h / (heads / kv_heads) as usize;
                    for t in 0..=position as usize {
                        let sum = (0..dimension as usize)
                            .map(|d| {
                                f64::from(
                                    q[h * dimension as usize + d]
                                        * keys
                                            [(t * kv_heads as usize + kv) * dimension as usize + d],
                                )
                            })
                            .sum::<f64>() as f32;
                        scores[h * context as usize + t] = sum * scale;
                    }
                    let score = &scores[h * context as usize..(h + 1) * context as usize];
                    let max = score.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                    let mut exps = score.to_vec();
                    exps.resize(exps.len().div_ceil(256) * 256, f32::NEG_INFINITY);
                    let sum = llama_quant::cpu_math::softmax_exp_sum_f32_in_place(&mut exps, max);
                    let inverse = (1.0 / sum) as f32;
                    statistics.extend([max, inverse]);
                    for t in 0..context as usize {
                        probabilities[h * context as usize + t] = exps[t] * inverse;
                    }
                    for d in 0..dimension as usize {
                        weighted.push(
                            (0..=position as usize)
                                .map(|t| {
                                    f64::from(
                                        probabilities[h * context as usize + t]
                                            * values[(t * kv_heads as usize + kv)
                                                * dimension as usize
                                                + d],
                                    )
                                })
                                .sum::<f64>() as f32,
                        );
                    }
                }
                all.push(case(
                    "grouped attention scores",
                    D::AttentionScores {
                        heads,
                        kv_heads,
                        dimension,
                        context,
                        scale,
                    },
                    vec![bytes(&q), bytes(&keys), params(0, position)],
                    scores.clone(),
                ));
                all.push(case(
                    "stable softmax statistics",
                    D::SoftmaxStats { heads, context },
                    vec![bytes(&scores), params(0, position)],
                    statistics.clone(),
                ));
                all.push(case(
                    "masked softmax application",
                    D::SoftmaxApply { heads, context },
                    vec![bytes(&scores), bytes(&statistics), params(0, position)],
                    probabilities.clone(),
                ));
                all.push(case(
                    "grouped attention values",
                    D::AttentionValues {
                        heads,
                        kv_heads,
                        dimension,
                        context,
                    },
                    vec![bytes(&probabilities), bytes(&values), params(0, position)],
                    weighted,
                ));
            }
        }
    }

    // Q8_0's source contract uses roundf (ties away from zero), not
    // trunc(x + 0.5), which fails for adjacent binary32 values below a tie.
    let mut boundary = vec![0.0f32; 32];
    boundary[..14].copy_from_slice(&[
        127.0,
        -127.0,
        f32::from_bits(0x3effffff),
        0.5,
        f32::from_bits(0x3f000001),
        f32::from_bits(0xbeffffff),
        -0.5,
        f32::from_bits(0xbf000001),
        f32::from_bits(0x3fbfffff),
        1.5,
        f32::from_bits(0x3fc00001),
        f32::from_bits(0xbfbfffff),
        -1.5,
        f32::from_bits(0xbfc00001),
    ]);
    let mut boundary_expected = vec![0.0; 33];
    boundary_expected[0] = 1.0;
    boundary_expected[1..15].copy_from_slice(&[
        127., -127., 0., 1., 1., 0., -1., -1., 1., 2., 2., -1., -2., -2.,
    ]);
    all.push(case(
        "Q8 half-integer boundary",
        D::ActivationQ8 { columns: 32 },
        vec![bytes(&boundary)],
        boundary_expected,
    ));
    for (columns, scale) in [
        (32u32, 0.0f32),
        (64, 0.125),
        (96, 1e-6),
        (64, 0.00001),
        (288, 0.25),
    ] {
        let input: Vec<f32> = (0..columns)
            .map(|i| (i as i32 % 79 - 39) as f32 * scale)
            .collect();
        let packed = llama_quant::quantize(llama_quant::DType::Q8_0, &input).unwrap();
        let mut activation = Vec::new();
        for block in packed.as_chunks::<34>().0 {
            activation.push(llama_quant::f16_to_f32(u16::from_le_bytes([
                block[0], block[1],
            ])));
            activation.extend(block[2..].iter().map(|&x| f32::from(x as i8)));
        }
        all.push(case(
            "Q8 activation source codec",
            D::ActivationQ8 { columns },
            vec![bytes(&input)],
            activation.clone(),
        ));
        for q4 in [true, false] {
            let dtype = if q4 {
                llama_quant::DType::Q4_0
            } else {
                llama_quant::DType::Q8_0
            };
            let rows = 3;
            let values: Vec<f32> = (0..rows * columns)
                .map(|i| (i as i32 % 43 - 21) as f32 * 0.25)
                .collect();
            let weights = llama_quant::quantize(dtype, &values).unwrap();
            let expected =
                llama_quant::matvec(dtype, &weights, rows as usize, columns as usize, &input)
                    .unwrap();
            all.push(case(
                "native Q4/Q8 integer block dot",
                D::QuantizedDot { rows, columns, q4 },
                vec![weights, bytes(&activation)],
                expected,
            ));
        }
    }

    // Independent exceptional-value contracts: RMS and packed CPU activation
    // conversion reject these inputs. Propagate poison instead of hiding them
    // behind a zero scale or a max instruction's NaN suppression.
    for input in [f32::MAX, f32::INFINITY, f32::NEG_INFINITY, f32::NAN] {
        all.push(case(
            "invalid RMS scale",
            D::RmsScale {
                width: 2,
                epsilon: 1e-5,
            },
            vec![bytes(&[input, 1.])],
            vec![f32::NAN],
        ));
    }
    for input in [
        f32::MAX,
        f32::MIN_POSITIVE,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::NAN,
    ] {
        let values = vec![input; 32];
        assert!(llama_quant::quantize(llama_quant::DType::Q8_0, &values).is_err());
        all.push(case(
            "invalid Q8 activation block",
            D::ActivationQ8 { columns: 32 },
            vec![bytes(&values)],
            vec![f32::NAN; 33],
        ));
    }
    // A scale that rounds all the way to zero is allowed by the CPU codec.
    all.push(case(
        "Q8 minimum subnormal rounds to zero",
        D::ActivationQ8 { columns: 32 },
        vec![bytes(&[f32::from_bits(1); 32])],
        vec![0.; 33],
    ));
    let finite = [
        0.,
        -0.,
        f32::from_bits(1),
        -f32::from_bits(1),
        f32::MIN_POSITIVE,
        f32::MAX,
    ];
    all.push(case(
        "finite classifier preserves subnormals",
        D::Embedding {
            vocabulary: 1,
            width: 6,
        },
        vec![bytes(&finite), params(0, 0)],
        finite.to_vec(),
    ));
    for sign in [1., -1.] {
        all.push(case(
            "overflowing F32 dot",
            D::MatVec {
                rows: 1,
                columns: 2,
                half: false,
            },
            vec![bytes(&[sign * f32::MAX, 0.]), bytes(&[2., 1.])],
            vec![f32::NAN],
        ));
        all.push(case(
            "overflowing attention score is not a mask",
            D::AttentionScores {
                heads: 1,
                kv_heads: 1,
                dimension: 2,
                context: 2,
                scale: 1.,
            },
            vec![
                bytes(&[sign * 2e20, sign * 2e20]),
                bytes(&[2e20; 4]),
                params(0, 0),
            ],
            vec![f32::NAN, f32::NEG_INFINITY],
        ));
    }
    for score in [f32::INFINITY, f32::NEG_INFINITY, f32::NAN] {
        all.push(case(
            "invalid visible softmax score",
            D::SoftmaxStats {
                heads: 1,
                context: 2,
            },
            vec![bytes(&[1., score]), params(0, 1)],
            vec![f32::NAN; 2],
        ));
    }
    for q4 in [true, false] {
        let dtype = if q4 {
            llama_quant::DType::Q4_0
        } else {
            llama_quant::DType::Q8_0
        };
        let weights = llama_quant::quantize(dtype, &[0.; 32]).unwrap();
        let mut activation = vec![0.; 33];
        activation[0] = f32::NAN;
        all.push(case(
            "zero packed weights preserve poison",
            D::QuantizedDot {
                rows: 1,
                columns: 32,
                q4,
            },
            vec![weights, bytes(&activation)],
            vec![f32::NAN],
        ));
    }

    all
}
fn compare(name: &str, actual: &[u8], expected: &[f32], exact: bool) {
    let actual = floats(actual);
    assert_eq!(actual.len(), expected.len());
    for (i, (&a, &e)) in actual.iter().zip(expected).enumerate() {
        if e.is_nan() {
            assert!(
                a.is_nan(),
                "{name}[{i}]: expected numeric-failure poison, got {a}"
            );
        } else if e.is_infinite() || exact {
            assert_eq!(a.to_bits(), e.to_bits(), "{name}[{i}]");
        } else {
            assert!(
                a.is_finite() && (a - e).abs() <= 2e-6 + 2e-5 * e.abs(),
                "{name}[{i}]:{a} vs {e}"
            );
        }
    }
}
#[test]
fn interpreted_decoder_kernels_match_analytical_values() {
    let cases = cases();
    for mut c in cases {
        let k = ptx::generate(
            &KernelSpec {
                kind: KernelKind::Decoder(c.kind),
            },
            PtxTarget::default(),
        )
        .unwrap();
        k.interpret(&mut c.buffers, 10_000_000).unwrap();
        compare(c.name, &c.buffers[k.output_buffer()], &c.expected, c.exact);
    }
}
fn dynamic_index_kernels() -> [D; 8] {
    // Both rotary layouts use the same guarded runtime position uniform.
    [
        D::RopeNeox {
            width: 8,
            head_dimension: 4,
            context: 3,
        },
        D::Embedding {
            vocabulary: 2,
            width: 4,
        },
        D::Rope {
            width: 4,
            head_dimension: 4,
            context: 3,
        },
        D::CacheWrite {
            width: 4,
            context: 3,
        },
        D::AttentionScores {
            heads: 2,
            kv_heads: 1,
            dimension: 2,
            context: 3,
            scale: 1.,
        },
        D::SoftmaxStats {
            heads: 2,
            context: 3,
        },
        D::SoftmaxApply {
            heads: 2,
            context: 3,
        },
        D::AttentionValues {
            heads: 2,
            kv_heads: 1,
            dimension: 2,
            context: 3,
        },
    ]
}
#[test]
fn dynamic_indices_are_guarded_even_for_untrusted_uniform_bytes() {
    for kind in dynamic_index_kernels() {
        let k = ptx::generate(
            &KernelSpec {
                kind: KernelKind::Decoder(kind),
            },
            PtxTarget::default(),
        )
        .unwrap();
        // Uniforms are all-ones, invalid as either a token or a position.
        let mut buffers: Vec<Vec<u8>> = k.buffer_bytes().iter().map(|&n| vec![255; n]).collect();
        let before = buffers.clone();
        k.interpret(&mut buffers, 100_000).unwrap();
        assert_eq!(buffers, before);
    }
}
#[test]
fn dimensions_and_arithmetic_are_validated_before_code_generation() {
    for kind in [
        D::Slice {
            source_width: 8,
            offset: 7,
            width: 2,
        },
        D::Slice {
            source_width: 8,
            offset: u32::MAX,
            width: 2,
        },
        D::Slice {
            source_width: 8,
            offset: 0,
            width: 0,
        },
        D::HeadRms {
            width: 8,
            head_dimension: 0,
            epsilon: 1e-5,
        },
        D::HeadRms {
            width: 8,
            head_dimension: 3,
            epsilon: 1e-5,
        },
        D::HeadRms {
            width: 8,
            head_dimension: 4,
            epsilon: f32::INFINITY,
        },
        D::RopeNeox {
            width: 8,
            head_dimension: 3,
            context: 2,
        },
        D::RmsScale {
            width: 0,
            epsilon: 1e-5,
        },
        D::RmsScale {
            width: 8,
            epsilon: f32::NAN,
        },
        D::Rope {
            width: 5,
            head_dimension: 3,
            context: 2,
        },
        D::AttentionScores {
            heads: 3,
            kv_heads: 2,
            dimension: 8,
            context: 2,
            scale: 1.,
        },
        D::CacheWrite {
            width: u32::MAX,
            context: 2,
        },
        D::MatVec {
            rows: u32::MAX / 2,
            columns: 4,
            half: true,
        },
    ] {
        assert!(
            ptx::generate(
                &KernelSpec {
                    kind: KernelKind::Decoder(kind)
                },
                PtxTarget::default()
            )
            .is_err()
        );
    }
}
#[test]
#[ignore = "requires an actual NVIDIA CUDA device; absence is a failure"]
fn cuda_decoder_kernels_execute_with_persistent_buffers() {
    let context = CudaDriver::load().unwrap().context(0).unwrap();
    let mut stream = context.stream().unwrap();
    let cases = cases();
    let count = cases.len();
    for c in cases {
        let generated = ptx::generate(
            &KernelSpec {
                kind: KernelKind::Decoder(c.kind),
            },
            PtxTarget::default(),
        )
        .unwrap();
        let compiled = context.compile_kernel(&generated).unwrap();
        let mut buffers: Vec<_> = c
            .buffers
            .iter()
            .map(|v| {
                let mut b = context.allocate(v.len()).unwrap();
                b.copy_from(0, v).unwrap();
                b
            })
            .collect();
        let mut output = buffers.remove(generated.output_buffer());
        let inputs: Vec<_> = buffers.iter().collect();
        for _ in 0..2 {
            compiled.run(&mut stream, &inputs, &mut output).unwrap();
            let mut result = vec![0; output.len()];
            output.copy_to(0, &mut result).unwrap();
            compare(c.name, &result, &c.expected, c.exact);
        }
        let mut short = context.allocate(1).unwrap();
        assert!(compiled.run(&mut stream, &inputs, &mut short).is_err());
    }
    eprintln!("CUDA decoder kernel checks: {count} analytical cases, two persistent launches each");
}

#[test]
#[ignore = "requires an actual NVIDIA CUDA device; absence is a failure"]
fn cuda_dynamic_guards_and_resource_contexts_are_enforced() {
    use llama_device_family::DeviceError;
    let driver = CudaDriver::load().unwrap();
    let context = driver.context(0).unwrap();
    let mut stream = context.stream().unwrap();
    for kind in dynamic_index_kernels() {
        let generated = ptx::generate(
            &KernelSpec {
                kind: KernelKind::Decoder(kind),
            },
            PtxTarget::default(),
        )
        .unwrap();
        let compiled = context.compile_kernel(&generated).unwrap();
        let mut buffers: Vec<_> = generated
            .buffer_bytes()
            .iter()
            .map(|&n| {
                let mut b = context.allocate(n).unwrap();
                b.copy_from(0, &vec![255; n]).unwrap();
                b
            })
            .collect();
        let mut output = buffers.remove(generated.output_buffer());
        compiled
            .run(
                &mut stream,
                &buffers.iter().collect::<Vec<_>>(),
                &mut output,
            )
            .unwrap();
        for b in buffers.iter().chain(std::iter::once(&output)) {
            let mut result = vec![0; b.len()];
            b.copy_to(0, &mut result).unwrap();
            assert!(
                result.iter().all(|&byte| byte == 255),
                "invalid uniform mutated a buffer"
            );
        }
    }
    let generated = ptx::generate(
        &KernelSpec {
            kind: KernelKind::AddF32 {
                elements: 2,
                lhs_stride: 1,
                rhs_stride: 1,
                output_stride: 1,
            },
        },
        PtxTarget::default(),
    )
    .unwrap();
    let compiled = context.compile_kernel(&generated).unwrap();
    let mut lhs = context.allocate(8).unwrap();
    let mut rhs = context.allocate(8).unwrap();
    let mut output = context.allocate(8).unwrap();
    lhs.copy_from(0, &bytes(&[1., 1.])).unwrap();
    rhs.copy_from(0, &bytes(&[2., 2.])).unwrap();
    output.copy_from(0, &bytes(&[-91., -91.])).unwrap();
    let other = driver.context(0).unwrap();
    let mut foreign_stream = other.stream().unwrap();
    let foreign_input = other.allocate(8).unwrap();
    let mut foreign_output = other.allocate(8).unwrap();
    assert!(
        matches!(compiled.run(&mut foreign_stream, &[&lhs, &rhs], &mut output),
        Err(DeviceError::InvalidArgument(s)) if s.contains("context"))
    );
    assert!(
        matches!(compiled.run(&mut stream, &[&foreign_input, &rhs], &mut output),
        Err(DeviceError::InvalidArgument(s)) if s.contains("context"))
    );
    assert!(
        matches!(compiled.run(&mut stream, &[&lhs, &rhs], &mut foreign_output),
        Err(DeviceError::InvalidArgument(s)) if s.contains("context"))
    );
    let short = context.allocate(1).unwrap();
    assert!(
        matches!(compiled.run(&mut stream, &[&short, &rhs], &mut output),
        Err(DeviceError::InvalidArgument(s)) if s.contains("extent"))
    );
    let mut result = vec![0; 8];
    output.copy_to(0, &mut result).unwrap();
    assert_eq!(result, bytes(&[-91., -91.]));
    // Every retained resource keeps its originating context alive.
    drop(context);
    compiled
        .run(&mut stream, &[&lhs, &rhs], &mut output)
        .unwrap();
    output.copy_to(0, &mut result).unwrap();
    assert_eq!(result, bytes(&[3., 3.]));
    eprintln!(
        "CUDA malformed uniforms: 8 guarded kernels; foreign stream/input/output and short input rejected; owned context lifetime retained"
    );
}
