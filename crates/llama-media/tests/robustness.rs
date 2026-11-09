use llama_media::{audio::*, image::*, input::*, spectral::*, *};
use std::sync::Arc;

#[test]
fn every_codec_fixture_prefix_is_rejected_without_panicking() {
    for input in [
        include_bytes!("../../../fixtures/t09/rgb.ppm").as_slice(),
        include_bytes!("../../../fixtures/t09/gray.pgm").as_slice(),
        include_bytes!("../../../fixtures/t09/rgb24.bmp").as_slice(),
        include_bytes!("../../../fixtures/t09/rgb32.bmp").as_slice(),
    ] {
        for end in 0..input.len() {
            assert!(
                decode(&input[..end], DecodeLimits::default()).is_err(),
                "image prefix {end}"
            );
        }
    }
    for input in [
        include_bytes!("../../../fixtures/t09/pcm8.wav").as_slice(),
        include_bytes!("../../../fixtures/t09/pcm16-stereo.wav").as_slice(),
        include_bytes!("../../../fixtures/t09/pcm24.wav").as_slice(),
        include_bytes!("../../../fixtures/t09/pcm32-ext.wav").as_slice(),
        include_bytes!("../../../fixtures/t09/float32.wav").as_slice(),
        include_bytes!("../../../fixtures/t09/float64.wav").as_slice(),
    ] {
        for end in 0..input.len() {
            assert!(
                decode_wav(&input[..end], DecodeLimits::default()).is_err(),
                "WAV prefix {end}"
            );
        }
    }
}

#[test]
fn malformed_headers_and_resource_limits_fail_before_large_allocations() {
    assert!(
        decode(
            b"P6\n999999999999999999999999999999 2\n255\n",
            DecodeLimits::default()
        )
        .is_err()
    );
    assert!(decode(b"P6\n65536 65536\n255\n", DecodeLimits::default()).is_err());
    assert!(decode(b"P6\n0 2\n255\n", DecodeLimits::default()).is_err());
    let image = include_bytes!("../../../fixtures/t09/rgb24.bmp");
    assert!(matches!(
        decode(
            image,
            DecodeLimits {
                max_image_pixels: 5,
                ..Default::default()
            }
        ),
        Err(MediaError::Limit { .. })
    ));
    assert!(matches!(
        decode(
            image,
            DecodeLimits {
                max_input_bytes: 2,
                ..Default::default()
            }
        ),
        Err(MediaError::Limit { .. })
    ));
    let wave = include_bytes!("../../../fixtures/t09/pcm16-stereo.wav");
    assert!(matches!(
        decode_wav(
            wave,
            DecodeLimits {
                max_audio_samples: 7,
                ..Default::default()
            }
        ),
        Err(MediaError::Limit { .. })
    ));
    for byte in [18, 19, 20, 21, 22, 23, 24, 25, 26, 28, 30, 10, 11, 12, 13] {
        let mut bad = image.to_vec();
        bad[byte] = 255;
        assert!(
            decode(&bad, DecodeLimits::default()).is_err(),
            "BMP mutation at {byte}"
        );
    }
}

#[test]
fn malformed_wav_alignment_compression_and_nonfinite_samples() {
    let audio = AudioBuffer::new(16000, 1, vec![0.0, 0.5, -1.0]).unwrap();
    let valid = encode_wav_f32(&audio).unwrap();
    let mut nan = valid.clone();
    nan[56..60].copy_from_slice(&f32::NAN.to_le_bytes());
    assert_eq!(
        decode_wav(&nan, DecodeLimits::default()),
        Err(MediaError::NonFinite)
    );
    let mut align = valid.clone();
    align[32..34].copy_from_slice(&9u16.to_le_bytes());
    assert!(decode_wav(&align, DecodeLimits::default()).is_err());
    let mut compression = valid.clone();
    compression[20..22].copy_from_slice(&17u16.to_le_bytes());
    assert!(matches!(
        decode_wav(&compression, DecodeLimits::default()),
        Err(MediaError::Unsupported(_))
    ));
    let mut riff = valid.clone();
    riff[4..8].copy_from_slice(&u32::MAX.to_le_bytes());
    assert!(decode_wav(&riff, DecodeLimits::default()).is_err());
}

#[test]
fn wav_roundtrip_and_stereo_downmix_preserve_samples() {
    for channels in [1, 2, 6] {
        let samples: Vec<f32> = (0..20 * channels)
            .map(|i| (i as f32 - 30.0) / 32.0)
            .collect();
        let input = AudioBuffer::new(44100, channels as u16, samples).unwrap();
        assert_eq!(
            decode_wav(&encode_wav_f32(&input).unwrap(), DecodeLimits::default()).unwrap(),
            input
        );
    }
    let stereo = AudioBuffer::new(16000, 2, vec![-1.0, 1.0, 0.5, 1.0]).unwrap();
    assert_eq!(stereo.mono().unwrap().samples(), &[0.0, 0.75]);
    assert!(
        AudioBuffer::new(16000, 6, vec![0.0; 6])
            .unwrap()
            .mono()
            .is_err()
    );
    assert!(AudioBuffer::new(0, 1, vec![]).is_err());
    assert!(AudioBuffer::new(16000, 2, vec![0.0]).is_err());
}

#[test]
fn crop_composite_and_tile_positions_are_checked() {
    let source = RgbImage::new(4, 2, (0..24).collect()).unwrap();
    let tiles = source.tiles(ImageSize::new(2, 1).unwrap()).unwrap();
    assert_eq!(tiles.len(), 4);
    assert_eq!(tiles[0].data(), &[0, 1, 2, 3, 4, 5]);
    assert_eq!(tiles[3].data(), &[18, 19, 20, 21, 22, 23]);
    assert!(
        source
            .crop(usize::MAX, 0, ImageSize::new(1, 1).unwrap())
            .is_err()
    );
    assert!(source.crop(3, 0, ImageSize::new(2, 1).unwrap()).is_err());
    assert!(source.tiles(ImageSize::new(3, 1).unwrap()).is_err());
    let center = source.center_crop(ImageSize::new(2, 2).unwrap()).unwrap();
    assert_eq!(center.data(), &[3, 4, 5, 6, 7, 8, 15, 16, 17, 18, 19, 20]);
    let mut canvas = RgbImage::solid(ImageSize::new(2, 2).unwrap(), [0, 0, 0]).unwrap();
    canvas.composite(&source, -3, 1).unwrap();
    assert_eq!(canvas.pixel(0, 1), Some([9, 10, 11]));
    assert_eq!(canvas.pixel(1, 1), Some([0, 0, 0]));
    assert!(canvas.composite(&source, i64::MAX, 0).is_err());
}

#[test]
fn constant_images_and_identity_are_preserved_by_every_resize_filter() {
    for color in [[0, 0, 0], [13, 29, 47], [127, 128, 129], [255, 255, 255]] {
        for filter in [
            ResizeFilter::Bilinear,
            ResizeFilter::Bicubic,
            ResizeFilter::Lanczos,
        ] {
            for (sw, sh, dw, dh) in [(1, 1, 9, 7), (9, 7, 1, 1), (7, 3, 4, 11), (4, 4, 4, 4)] {
                let image = RgbImage::solid(ImageSize::new(sw, sh).unwrap(), color).unwrap();
                let resized = resize(
                    &image,
                    ImageSize::new(dw, dh).unwrap(),
                    filter,
                    WeightPrecision::Pillow22,
                )
                .unwrap();
                for pixel in resized.data().as_chunks::<3>().0 {
                    assert_eq!(*pixel, color);
                }
            }
        }
    }
}

#[test]
fn resize_bounds_and_normalization_are_explicit() {
    assert!(ImageSize::new(0, 1).is_err());
    assert!(ImageSize::new(usize::MAX, 1).is_err());
    let rgb = RgbImage::solid(ImageSize::new(1, 1).unwrap(), [0, 127, 255]).unwrap();
    assert!(
        resize(
            &rgb,
            ImageSize::new(2, 2).unwrap(),
            ResizeFilter::Lanczos,
            WeightPrecision::Torch16
        )
        .is_err()
    );
    assert!(rgb.normalize([0.0; 3], [0.0; 3]).is_err());
    assert!(rgb.normalize([f32::NAN; 3], [1.0; 3]).is_err());
    let tensor = RgbImage::new(2, 1, vec![0, 127, 255, 255, 127, 0])
        .unwrap()
        .normalize([0.0; 3], [1.0; 3])
        .unwrap();
    let chw = tensor.to_chw().unwrap();
    assert_eq!(chw, &[0.0, 1.0, 127.0 / 255.0, 127.0 / 255.0, 1.0, 0.0]);
}

#[test]
fn smart_resize_handles_ties_and_conflicting_bounds() {
    let size = ImageSize::new(70, 70).unwrap();
    assert_eq!(
        SmartResize {
            alignment: 28,
            ties_even: true,
            ..Default::default()
        }
        .size_for(size)
        .unwrap(),
        ImageSize::new(56, 56).unwrap()
    );
    assert_eq!(
        SmartResize {
            alignment: 28,
            ties_even: false,
            ..Default::default()
        }
        .size_for(size)
        .unwrap(),
        ImageSize::new(84, 84).unwrap()
    );
    assert!(
        SmartResize {
            min_pixels: Some(100),
            max_pixels: Some(1),
            ..Default::default()
        }
        .size_for(size)
        .is_err()
    );
    assert!(
        SmartResize {
            alignment: 0,
            ..Default::default()
        }
        .size_for(size)
        .is_err()
    );
    let opts = SmartResize {
        alignment: 2,
        longest_edge: Some(10),
        min_pixels: Some(400),
        ..Default::default()
    };
    assert_eq!(
        opts.size_for(ImageSize::new(100, 100).unwrap()).unwrap(),
        ImageSize::new(20, 20).unwrap()
    );
}

#[test]
fn fft_matches_independent_direct_dft_and_roundtrip() {
    for n in [1usize, 3, 4, 7, 8, 10, 25, 32, 400] {
        let input: Vec<f32> = (0..n)
            .map(|i| (((i * 13 + 3) % 37) as i32 - 18) as f32 / 32.0)
            .collect();
        let plan = FftPlan::new(n).unwrap();
        let freq = plan.forward_real(&input).unwrap();
        for (k, actual) in freq.iter().enumerate() {
            let (mut real, mut imag) = (0f64, 0f64);
            for (j, &x) in input.iter().enumerate() {
                let angle = 2.0 * std::f64::consts::PI * (k * j) as f64 / n as f64;
                real += x as f64 * angle.cos();
                imag -= x as f64 * angle.sin();
            }
            assert!((actual.re as f64 - real).abs() < 2e-4, "N={n} k={k} real");
            assert!((actual.im as f64 - imag).abs() < 2e-4, "N={n} k={k} imag");
        }
        let inverse = plan.inverse(&freq).unwrap();
        for (&x, y) in input.iter().zip(&inverse) {
            assert!((x - y.re).abs() < 1e-5);
            assert!(y.im.abs() < 1e-5);
        }
    }
}

#[test]
fn spectral_validation_and_zero_energy_are_finite() {
    assert!(FftPlan::new(0).is_err());
    assert!(FftPlan::new(65537).is_err());
    assert!(FftPlan::new(4093).is_err());
    assert!(
        FftPlan::new(8)
            .unwrap()
            .forward_real(&[f32::NAN; 8])
            .is_err()
    );
    assert!(hann_window(1, false).is_err());
    let bank = MelFilterbank::new(MelConfig::slaney(400, 80, 16000)).unwrap();
    assert_eq!(bank.log_mel_frame(&[0.0; 400]).unwrap(), vec![-10.0; 80]);
    assert!(
        MelFilterbank::new(MelConfig {
            min_hz: 9000.0,
            ..MelConfig::slaney(400, 80, 16000)
        })
        .is_err()
    );
    assert!(
        MelFilterbank::new(MelConfig {
            max_hz: Some(f32::NAN),
            ..MelConfig::slaney(400, 80, 16000)
        })
        .is_err()
    );
    assert!(stft(&[0.0; 4], 8, 2, StftPadding::None).is_err());
    assert!(stft(&[0.0; 4], 8, 2, StftPadding::CenterReflect).is_err());
}

#[test]
fn stft_frame_count_and_spectrogram_layout_are_consistent() {
    let wave: Vec<f32> = (0..20).map(|i| i as f32 / 20.0).collect();
    let no = stft(&wave, 8, 2, StftPadding::None).unwrap();
    let center = stft(&wave, 8, 2, StftPadding::CenterZero).unwrap();
    assert_eq!(no.len(), 7);
    assert_eq!(center.len(), 11);
    let bank = MelFilterbank::new(MelConfig::slaney(8, 3, 16000)).unwrap();
    let mel = log_mel(&wave, &bank, 2, StftPadding::None, 1e-10).unwrap();
    assert_eq!(mel.frames(), 7);
    assert_eq!(mel.bands(), 3);
    let transposed = mel.to_mel_major().unwrap();
    for t in 0..7 {
        for m in 0..3 {
            assert_eq!(transposed[m * 7 + t], mel.data_frame_major()[t * 3 + m]);
        }
    }
}

#[test]
fn linear_resampler_output_is_chunk_invariant_and_resettable() {
    let input: Vec<f32> = (0..101).map(|i| (i as f32 * 0.13).sin()).collect();
    for (src, dst) in [
        (16000, 24000),
        (44100, 16000),
        (48000, 44100),
        (1, 16),
        (16, 1),
        (16000, 16000),
    ] {
        let expected = resample_linear(&input, src, dst).unwrap();
        assert_eq!(
            expected.len(),
            (input.len() as u64 * dst as u64).div_ceil(src as u64) as usize
        );
        for chunk in [1, 2, 7, 33, 101] {
            let mut resampler = LinearResampler::new(src, dst).unwrap();
            let mut actual = Vec::new();
            for segment in input.chunks(chunk) {
                actual.extend(resampler.push(segment).unwrap());
            }
            actual.extend(resampler.finish().unwrap());
            assert_eq!(actual, expected, "{src}/{dst} chunk={chunk}");
            assert!(resampler.finish().unwrap().is_empty());
            assert!(resampler.push(&[0.0]).is_err());
            resampler.reset();
            let mut again = resampler.push(&input).unwrap();
            again.extend(resampler.finish().unwrap());
            assert_eq!(again, expected);
        }
    }
}

#[test]
fn invalid_resampler_input_does_not_change_next_output() {
    let mut stream = LinearResampler::new(2, 3).unwrap();
    let expected = stream.clone();
    assert_eq!(stream.push(&[f32::NAN]), Err(MediaError::NonFinite));
    let mut expected = expected;
    assert_eq!(
        stream.push(&[0.0, 1.0]).unwrap(),
        expected.push(&[0.0, 1.0]).unwrap()
    );
    assert!(LinearResampler::new(0, 2).is_err());
    assert_eq!(resample_linear(&[1.0], 2, 3).unwrap(), vec![1.0, 1.0]);
}

#[test]
fn istft_reset_and_flush_have_explicit_lifecycle() {
    let mut synth = StreamingIstft::new(8, 2).unwrap();
    let freq = [Complex32::new(1.0, 0.0); 5];
    let a = synth.process_frame(&freq).unwrap();
    let b = synth.process_frame(&freq).unwrap();
    let c = synth.finish().unwrap();
    assert!(a.is_empty());
    assert_eq!(b.len(), 1);
    assert_eq!(c.len(), 6);
    assert!(synth.finish().unwrap().is_empty());
    assert!(synth.process_frame(&freq).is_err());
    synth.reset();
    assert_eq!(synth.process_frame(&freq).unwrap(), a);
    assert_eq!(synth.process_frame(&freq).unwrap(), b);
    assert_eq!(synth.finish().unwrap(), c);
    let mut empty = StreamingIstft::new(8, 2).unwrap();
    assert!(empty.finish().unwrap().is_empty());
    assert!(StreamingIstft::new(7, 2).is_err());
    assert!(StreamingIstft::new(8, 9).is_err());
}

#[test]
fn waveform_padding_and_embedding_shape_do_not_fabricate_values() {
    assert_eq!(
        pad_waveform(&[1.0, 2.0, 3.0], 2, None).unwrap(),
        vec![1.0, 2.0, 3.0, 0.0]
    );
    assert_eq!(
        pad_waveform(&[1.0, 2.0, 3.0], 2, Some(1)).unwrap(),
        vec![1.0, 0.0]
    );
    assert!(pad_waveform(&[1.0], 0, None).is_err());
    let positions = vec![Position {
        token: 0,
        temporal: 0,
        y: 0,
        x: 0,
    }];
    assert!(EmbeddingBatch::new(SequenceId(1), 3, positions.clone(), vec![1.0, 2.0]).is_err());
    let batch = EmbeddingBatch::new(SequenceId(1), 2, positions.clone(), vec![1.0, 2.0]).unwrap();
    assert_eq!(batch.width(), 2);
    assert_eq!(batch.positions(), positions);
    assert_eq!(batch.embeddings(), &[1.0, 2.0]);
}

#[test]
fn sequence_cancellation_and_timestamp_order_are_isolated() {
    let cancel = Cancellation::default();
    let shared = cancel.clone();
    let separate = Cancellation::default();
    cancel.cancel();
    assert!(shared.check().is_err());
    assert!(separate.check().is_ok());
    let image = Arc::new(RgbImage::solid(ImageSize::new(1, 1).unwrap(), [0, 0, 0]).unwrap());
    let chunk = InputChunk::VideoFrame {
        sequence: SequenceId(7),
        timestamp: TimestampMicros(3),
        image,
    };
    assert_eq!(chunk.sequence(), SequenceId(7));
    let times = [
        TimestampMicros(0),
        TimestampMicros(100),
        TimestampMicros(250),
        TimestampMicros(400),
    ];
    assert_eq!(select_video_frames(&times, 200, 10).unwrap(), vec![0, 2]);
    assert!(select_video_frames(&[TimestampMicros(2), TimestampMicros(1)], 1, 10).is_err());
    assert!(select_video_frames(&[TimestampMicros(u64::MAX)], 1, 1).is_err());
}

#[test]
fn fixed_count_video_selection_retains_reference_ties_and_repetition() {
    assert_eq!(uniform_frame_indices(3, 5).unwrap(), vec![0, 0, 1, 2, 2]);
    assert_eq!(uniform_frame_indices(10, 4).unwrap(), vec![0, 3, 6, 9]);
    assert_eq!(uniform_frame_indices(8, 3).unwrap(), vec![0, 4, 7]);
    assert_eq!(uniform_frame_indices(8, 1).unwrap(), vec![0]);
    assert!(uniform_frame_indices(0, 1).is_err());
    assert!(uniform_frame_indices(1, usize::MAX).is_err());
}

#[test]
fn rejected_stream_work_preserves_next_successful_output() {
    let mut resampler = LinearResampler::new(1, 768000).unwrap();
    assert_eq!(resampler.push(&[1.0]).unwrap(), vec![1.0]);
    let mut before = resampler.clone();
    assert!(matches!(
        resampler.push(&[2.0; 100]),
        Err(MediaError::Limit { .. })
    ));
    assert_eq!(
        resampler.push(&[2.0]).unwrap(),
        before.push(&[2.0]).unwrap()
    );
    assert_eq!(resampler.finish().unwrap(), before.finish().unwrap());

    let mut synth = StreamingIstft::new(8, 2).unwrap();
    let frame = [Complex32::new(1.0, 0.0); 5];
    synth.process_frame(&frame).unwrap();
    let mut before = synth.clone();
    assert!(synth.process_frame(&frame[..4]).is_err());
    assert_eq!(
        synth.process_frame(&[Complex32::new(f32::NAN, 0.0); 5]),
        Err(MediaError::NonFinite)
    );
    assert_eq!(
        synth.process_frame(&frame).unwrap(),
        before.process_frame(&frame).unwrap()
    );
    assert_eq!(synth.finish().unwrap(), before.finish().unwrap());
}
