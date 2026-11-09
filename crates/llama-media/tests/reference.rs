//! Independent fixtures were emitted by the unmodified frozen C++ functions.
//! The Rust implementation is never used to generate the expected values.
use llama_media::{DecodeLimits, audio, image::*, spectral::*};

fn values<T: std::str::FromStr>(file: &str, name: &str) -> Vec<T>
where
    T::Err: std::fmt::Debug,
{
    let row = file
        .lines()
        .find_map(|line| line.split_once('\t').filter(|(key, _)| *key == name))
        .expect("reference fixture row");
    row.1
        .split(',')
        .map(|value| value.parse().unwrap())
        .collect()
}
const PREPROCESSING: &str = include_str!("../../../fixtures/t09/reference-preprocessing.tsv");
const CODECS: &str = include_str!("../../../fixtures/t09/reference-codecs.tsv");
fn pattern(w: usize, h: usize) -> RgbImage {
    let mut data = Vec::new();
    for y in 0..h {
        for x in 0..w {
            data.extend([
                (x * 53 + y * 29 + 7) as u8,
                (x * 17 + y * 83 + 11) as u8,
                (x * 97 + y * 13 + 19) as u8,
            ]);
        }
    }
    RgbImage::new(w, h, data).unwrap()
}
fn close(name: &str, actual: &[f32], expected: &[f32], absolute: f32, relative: f32) {
    assert_eq!(actual.len(), expected.len(), "{name} length");
    let mut max = 0f32;
    for (i, (&a, &b)) in actual.iter().zip(expected).enumerate() {
        let difference = (a - b).abs();
        assert!(
            a.is_finite() && b.is_finite() && difference <= absolute + relative * b.abs(),
            "{name}[{i}]: actual={a} expected={b} difference={difference}"
        );
        max = max.max(difference);
    }
    eprintln!("oracle {name}: {} values, max_abs={max:.9e}", actual.len());
}

#[test]
fn image_resampling_matches_original_bytes() {
    for (name, source, target, filter, precision, padding) in [
        (
            "resize_bilinear",
            (7, 5),
            (4, 3),
            ResizeFilter::Bilinear,
            WeightPrecision::Pillow22,
            Padding::None,
        ),
        (
            "resize_bicubic",
            (7, 5),
            (4, 3),
            ResizeFilter::Bicubic,
            WeightPrecision::Pillow22,
            Padding::None,
        ),
        (
            "resize_lanczos",
            (7, 5),
            (4, 3),
            ResizeFilter::Lanczos,
            WeightPrecision::Pillow22,
            Padding::None,
        ),
        (
            "resize_torch_bilinear",
            (7, 5),
            (4, 3),
            ResizeFilter::Bilinear,
            WeightPrecision::Torch16,
            Padding::None,
        ),
        (
            "resize_torch_bicubic",
            (7, 5),
            (4, 3),
            ResizeFilter::Bicubic,
            WeightPrecision::Torch16,
            Padding::None,
        ),
        (
            "upscale_bicubic",
            (3, 2),
            (8, 5),
            ResizeFilter::Bicubic,
            WeightPrecision::Pillow22,
            Padding::None,
        ),
        (
            "upscale_lanczos",
            (3, 2),
            (8, 5),
            ResizeFilter::Lanczos,
            WeightPrecision::Pillow22,
            Padding::None,
        ),
        (
            "letterbox_ceil",
            (7, 3),
            (8, 8),
            ResizeFilter::Bicubic,
            WeightPrecision::Pillow22,
            Padding::Ceil,
        ),
        (
            "letterbox_nearest",
            (7, 3),
            (8, 8),
            ResizeFilter::Bicubic,
            WeightPrecision::Pillow22,
            Padding::Nearest,
        ),
        (
            "resize_one_row",
            (7, 1),
            (3, 1),
            ResizeFilter::Lanczos,
            WeightPrecision::Pillow22,
            Padding::None,
        ),
        (
            "resize_one_column",
            (1, 7),
            (1, 3),
            ResizeFilter::Bicubic,
            WeightPrecision::Pillow22,
            Padding::None,
        ),
    ] {
        let image = pattern(source.0, source.1);
        let out = resize_padded(
            &image,
            ImageSize::new(target.0, target.1).unwrap(),
            filter,
            precision,
            padding,
            [13, 29, 47],
        )
        .unwrap();
        let expected = values::<u8>(PREPROCESSING, name);
        assert_eq!(out.data(), expected, "{name}");
        eprintln!("oracle {name}: {} exact RGB bytes", out.data().len());
    }
}

#[test]
#[allow(clippy::excessive_precision)] // Keep the frozen source's decimal parameters visible.
fn normalization_and_smart_size_match_original() {
    let rgb = RgbImage::new(2, 1, vec![0, 127, 255, 16, 128, 240]).unwrap();
    let normalized = rgb
        .normalize(
            [0.48145466, 0.4578275, 0.40821073],
            [0.26862954, 0.26130258, 0.27577711],
        )
        .unwrap();
    close(
        "normalize_clip",
        normalized.data_hwc(),
        &values::<f32>(PREPROCESSING, "normalize_clip"),
        1e-6,
        1e-5,
    );
    let options = SmartResize {
        alignment: 28,
        min_pixels: Some(28 * 28),
        max_pixels: Some(1280 * 28 * 28),
        ..Default::default()
    };
    let size = options
        .size_for(ImageSize::new(1933, 733).unwrap())
        .unwrap();
    assert_eq!(
        vec![size.width(), size.height()],
        values::<usize>(PREPROCESSING, "smart_size")
    );
}

#[test]
fn real_fft_and_windows_match_original() {
    for n in [7usize, 8, 10, 16, 400] {
        let input: Vec<f32> = (0..n)
            .map(|i| ((i * 7 % 19) as i32 - 9) as f32 / 16.0)
            .collect();
        let fft = FftPlan::new(n).unwrap().forward_real(&input).unwrap();
        let actual: Vec<f32> = fft.iter().flat_map(|x| [x.re, x.im]).collect();
        let name = format!("fft_{n}");
        close(
            &name,
            &actual,
            &values::<f32>(PREPROCESSING, &name),
            1e-6,
            1e-5,
        );
        let name = format!("hann_{n}");
        close(
            &name,
            &hann_window(n, true).unwrap(),
            &values::<f32>(PREPROCESSING, &name),
            1e-6,
            1e-5,
        );
    }
}

#[test]
fn filterbanks_and_log_mel_match_original() {
    let bank = MelFilterbank::new(MelConfig::slaney(400, 8, 16000)).unwrap();
    close(
        "mel_slaney",
        bank.weights(),
        &values::<f32>(PREPROCESSING, "mel_slaney"),
        1e-7,
        1e-5,
    );
    let frame: Vec<f32> = (0..400).map(|i| ((i * 7 % 19) - 9) as f32 / 16.0).collect();
    close(
        "log_mel_frame",
        &bank.log_mel_frame(&frame).unwrap(),
        &values::<f32>(PREPROCESSING, "log_mel_frame"),
        1e-6,
        1e-5,
    );
    let config = MelConfig {
        fft_len: 16,
        bands: 6,
        sample_rate: 16000,
        min_hz: 20.0,
        max_hz: Some(7600.0),
        scale: MelScale::Htk,
        area_normalize: false,
        multiplier: 1.0,
    };
    let bank = MelFilterbank::new(config).unwrap();
    close(
        "mel_htk",
        bank.weights(),
        &values::<f32>(PREPROCESSING, "mel_htk"),
        1e-7,
        1e-5,
    );
}

#[test]
fn streaming_synthesis_matches_original() {
    let mut synth = StreamingIstft::new(8, 2).unwrap();
    let mut actual = Vec::new();
    for frame in 0..3 {
        let spectrum: Vec<Complex32> = (0..5)
            .map(|k| {
                Complex32::new(
                    (frame + 1) as f32 * 0.125 * (k + 1) as f32,
                    if k > 0 && k < 4 {
                        (frame + 1) as f32 * -0.0625 * k as f32
                    } else {
                        0.0
                    },
                )
            })
            .collect();
        actual.extend(synth.process_frame(&spectrum).unwrap());
    }
    actual.extend(synth.finish().unwrap());
    close(
        "istft_stream",
        &actual,
        &values::<f32>(PREPROCESSING, "istft_stream"),
        1e-6,
        1e-5,
    );
}

#[test]
fn image_decoders_match_original_stb() {
    for (name, input) in [
        (
            "rgb.ppm",
            include_bytes!("../../../fixtures/t09/rgb.ppm").as_slice(),
        ),
        (
            "gray.pgm",
            include_bytes!("../../../fixtures/t09/gray.pgm").as_slice(),
        ),
        (
            "rgb24.bmp",
            include_bytes!("../../../fixtures/t09/rgb24.bmp").as_slice(),
        ),
        (
            "rgb32.bmp",
            include_bytes!("../../../fixtures/t09/rgb32.bmp").as_slice(),
        ),
    ] {
        let decoded = decode(input, DecodeLimits::default()).unwrap();
        assert_eq!(decoded.size(), ImageSize::new(3, 2).unwrap());
        assert_eq!(decoded.data(), values::<u8>(CODECS, name), "{name}");
    }
}

#[test]
fn wav_samples_match_original_miniaudio() {
    for (name, input, channels) in [
        (
            "pcm8.wav",
            include_bytes!("../../../fixtures/t09/pcm8.wav").as_slice(),
            1,
        ),
        (
            "pcm16-stereo.wav",
            include_bytes!("../../../fixtures/t09/pcm16-stereo.wav").as_slice(),
            2,
        ),
        (
            "pcm24.wav",
            include_bytes!("../../../fixtures/t09/pcm24.wav").as_slice(),
            1,
        ),
        (
            "pcm32-ext.wav",
            include_bytes!("../../../fixtures/t09/pcm32-ext.wav").as_slice(),
            1,
        ),
        (
            "float32.wav",
            include_bytes!("../../../fixtures/t09/float32.wav").as_slice(),
            1,
        ),
        (
            "float64.wav",
            include_bytes!("../../../fixtures/t09/float64.wav").as_slice(),
            1,
        ),
    ] {
        let decoded = audio::decode_wav(input, DecodeLimits::default()).unwrap();
        assert_eq!(decoded.sample_rate(), 16000);
        assert_eq!(decoded.channels(), channels);
        close(
            name,
            decoded.samples(),
            &values::<f32>(CODECS, name),
            0.0,
            0.0,
        );
    }
}
