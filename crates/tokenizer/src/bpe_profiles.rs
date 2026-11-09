// Metadata-selected profiles and expressions transcribed mechanically from the frozen source.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Pre {
    Afmoe,
    Bailingmoe,
    Bloom,
    Chameleon,
    Chatglm4,
    Codeshell,
    CommandR,
    Dbrx,
    Deepseek3Llm,
    DeepseekCoder,
    DeepseekLlm,
    Default,
    DomynSmall,
    Exaone,
    ExaoneMoe,
    Falcon,
    Flint,
    Gemma4,
    Gpt2,
    Gpt3Finnish,
    Gpt4o,
    GraniteDocling,
    GraniteEmbMulti,
    Grok2,
    Hunyuan,
    HunyuanDense,
    HyV4,
    Jais,
    Jais2,
    JoyaiLlm,
    K2Horizon,
    KimiK2,
    Laguna,
    Llama3,
    Mellum2,
    Minerva,
    Minicpm5,
    MinimaxM2,
    Mmbert,
    Mpt,
    Nemotron35Embd,
    Olmo,
    Poro,
    Qwen2,
    Qwen35,
    Refact,
    SarvamMoe,
    SeedCoder,
    Smaug,
    Smollm,
    SolarOpen,
    Soprano,
    Spark25,
    Stablelm2,
    Starcoder,
    Superbpe,
    Tekken,
    TekkenNum3,
    TinyAya,
    Trillion,
    Ufakzeka,
    Viking,
    Whitespace,
    Youtu,
}
impl Config {
    pub(super) fn profile(name: &str) -> Result<Self> {
        let mut config = Self::default();
        match name {
            "" => {
                config.pre = Pre::Default;
            }
            "flint" => {
                config.pre = Pre::Flint;
                config.clean_spaces = false;
            }
            "default" => {
                config.pre = Pre::Default;
            }
            "minicpm5" => {
                config.pre = Pre::Minicpm5;
                config.ignore_merges = true;
                config.clean_spaces = false;
            }
            "llama3" | "llama-v3" | "llama-bpe" | "falcon3" | "falcon-h1" | "midm-2.0" | "jina-v5-nano" => {
                config.pre = Pre::Llama3;
                config.ignore_merges = true;
                config.add_bos = true;
            }
            "lfm2" => {
                config.pre = Pre::Llama3;
                config.ignore_merges = true;
                config.add_bos = true;
                config.clean_spaces = false;
            }
            "deepseek-llm" => {
                config.pre = Pre::DeepseekLlm;
                config.clean_spaces = false;
            }
            "deepseek-coder" => {
                config.pre = Pre::DeepseekCoder;
                config.clean_spaces = false;
            }
            "deepseek-v3" => {
                config.pre = Pre::Deepseek3Llm;
                config.clean_spaces = false;
            }
            "spark2_5" => {
                config.pre = Pre::Spark25;
                config.clean_spaces = false;
            }
            "youtu" => {
                config.pre = Pre::Youtu;
                config.clean_spaces = false;
                config.ignore_merges = true;
            }
            "falcon" => {
                config.pre = Pre::Falcon;
            }
            "mpt" => {
                config.pre = Pre::Mpt;
            }
            "starcoder" => {
                config.pre = Pre::Starcoder;
            }
            "gpt-2" | "phi-2" | "jina-es" | "jina-de" | "gigachat" | "jina-v2-es" | "jina-v2-de" | "a.x-4.0" | "mellum" | "modern-bert" => {
                config.pre = Pre::Gpt2;
            }
            "jais-2" => {
                config.pre = Pre::Jais2;
            }
            "gemma4" | "granite-embed-multi-311m" => {
                config.pre = Pre::Gemma4;
                config.escape_whitespaces = true;
                config.clean_spaces = false;
            }
            "mmbert" => {
                config.pre = Pre::Mmbert;
                config.escape_whitespaces = true;
            }
            "sarvam-moe" => {
                config.pre = Pre::SarvamMoe;
                config.escape_whitespaces = true;
                config.clean_spaces = false;
            }
            "jina-v1-en" | "jina-v2-code" | "roberta-bpe" => {
                config.pre = Pre::Gpt2;
                config.add_sep = true;
            }
            "whitespace" => {
                config.pre = Pre::Whitespace;
                config.lowercase = false;
            }
            "refact" => {
                config.pre = Pre::Refact;
            }
            "command-r" => {
                config.pre = Pre::CommandR;
                config.clean_spaces = false;
            }
            "qwen2" | "deepseek-r1-qwen" | "kormo" | "f2llmv2" => {
                config.pre = Pre::Qwen2;
                config.clean_spaces = false;
            }
            "qwen35" => {
                config.pre = Pre::Qwen35;
                config.clean_spaces = false;
            }
            "stablelm2" => {
                config.pre = Pre::Stablelm2;
            }
            "olmo" => {
                config.pre = Pre::Olmo;
            }
            "dbrx" => {
                config.pre = Pre::Dbrx;
                config.clean_spaces = false;
            }
            "smaug-bpe" => {
                config.pre = Pre::Smaug;
            }
            "poro-chat" => {
                config.pre = Pre::Poro;
                config.clean_spaces = false;
            }
            "chatglm-bpe" => {
                config.pre = Pre::Chatglm4;
                config.clean_spaces = false;
                config.bos_null = true;
            }
            "glm4" | "glm5" => {
                config.pre = Pre::Chatglm4;
                config.ignore_merges = true;
                config.clean_spaces = false;
                config.bos_null = true;
            }
            "viking" => {
                config.pre = Pre::Viking;
                config.clean_spaces = false;
            }
            "jais" => {
                config.pre = Pre::Jais;
            }
            "tekken" | "pixtral" => {
                config.pre = Pre::Tekken;
                config.clean_spaces = false;
                config.ignore_merges = true;
                config.add_bos = true;
            }
            "tekken-num3" => {
                config.pre = Pre::TekkenNum3;
                config.clean_spaces = false;
                config.ignore_merges = true;
                config.add_bos = true;
            }
            "smollm" => {
                config.pre = Pre::Smollm;
                config.clean_spaces = false;
            }
            "codeshell" => {
                config.pre = Pre::Codeshell;
            }
            "bloom" => {
                config.pre = Pre::Bloom;
            }
            "gpt3-finnish" => {
                config.pre = Pre::Gpt3Finnish;
            }
            "exaone" => {
                config.pre = Pre::Exaone;
            }
            "exaone4" => {
                config.pre = Pre::Gpt2;
                config.clean_spaces = false;
            }
            "exaone-moe" => {
                config.pre = Pre::ExaoneMoe;
            }
            "chameleon" => {
                config.pre = Pre::Chameleon;
                config.add_bos = true;
                config.clean_spaces = false;
            }
            "minerva-7b" => {
                config.pre = Pre::Minerva;
            }
            "domyn-small" => {
                config.pre = Pre::DomynSmall;
                config.escape_whitespaces = true;
            }
            "megrez" => {
                config.pre = Pre::Qwen2;
            }
            "gpt-4o" | "llama4" | "kanana2" | "talkie" => {
                config.pre = Pre::Gpt4o;
                config.clean_spaces = false;
            }
            "granite-embed-multi-97m" => {
                config.pre = Pre::GraniteEmbMulti;
                config.clean_spaces = false;
                config.ignore_merges = true;
            }
            "tiny_aya" | "cohere2moe" => {
                config.pre = Pre::TinyAya;
                config.clean_spaces = false;
            }
            "superbpe" => {
                config.pre = Pre::Superbpe;
                config.clean_spaces = false;
            }
            "trillion" => {
                config.pre = Pre::Trillion;
                config.clean_spaces = false;
            }
            "granite-docling" => {
                config.pre = Pre::GraniteDocling;
                config.clean_spaces = false;
            }
            "bailingmoe" | "bailingmoe2" | "llada-moe" => {
                config.pre = Pre::Bailingmoe;
                config.clean_spaces = false;
            }
            "seed-coder" => {
                config.pre = Pre::SeedCoder;
                config.clean_spaces = false;
            }
            "hunyuan" => {
                config.pre = Pre::Hunyuan;
                config.clean_spaces = false;
            }
            "hunyuan-dense" => {
                config.pre = Pre::HunyuanDense;
                config.clean_spaces = false;
            }
            "hy_v4" => {
                config.pre = Pre::HyV4;
                config.clean_spaces = false;
            }
            "joyai-llm" => {
                config.pre = Pre::JoyaiLlm;
                config.clean_spaces = false;
            }
            "kimi-k2" => {
                config.pre = Pre::KimiK2;
                config.clean_spaces = false;
            }
            "ufakzeka" => {
                config.pre = Pre::Ufakzeka;
                config.clean_spaces = false;
            }
            "grok-2" => {
                config.pre = Pre::Grok2;
                config.clean_spaces = false;
            }
            "afmoe" => {
                config.pre = Pre::Afmoe;
                config.clean_spaces = false;
            }
            "laguna" => {
                config.pre = Pre::Laguna;
                config.clean_spaces = false;
            }
            "minimax-m2" => {
                config.pre = Pre::MinimaxM2;
                config.clean_spaces = false;
            }
            "solar-open" => {
                config.pre = Pre::SolarOpen;
                config.clean_spaces = false;
            }
            "mellum2" => {
                config.pre = Pre::Mellum2;
                config.clean_spaces = false;
            }
            "k2-horizon" => {
                config.pre = Pre::K2Horizon;
                config.clean_spaces = false;
            }
            "nemotron-3.5-embed" => {
                config.pre = Pre::Nemotron35Embd;
                config.ignore_merges = true;
                config.add_bos = true;
            }
            "soprano" => {
                config.pre = Pre::Soprano;
            }
            _ => return Err(Error::Unsupported(format!("BPE pretokenizer {name}"))),
        }
        Ok(config)
    }
}
impl Pre {
    fn expressions(self) -> (&'static [&'static str],bool) {
        match self {
            Self::Flint => (&[
                r###"\p{N}{1,3}"###,
                r###"[一-龥぀-ゟ゠-ヿ]+"###,
                r###"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])?|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])?| ?[\p{P}\p{S}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"###,
            ], true),
            Self::Llama3 => (&[
                r###"(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"###,
            ], true),
            Self::Jais2 => (&[
                r###"(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s{512}(?!\S)|\s{256}(?!\S)|\s{128}(?!\S)|\s{64}(?!\S)|\s{32}(?!\S)|\s{16}(?!\S)|\s{8}(?!\S)|\s{4}(?!\S)|\s{1,2}(?!\S)|\s{1}"###,
            ], true),
            Self::Dbrx | Self::Smaug => (&[
                r###"(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"###,
            ], true),
            Self::DeepseekLlm => (&[
                r###"[
]"###,
                r###"\s?[A-Za-zµÀ-ÖØ-öø-ƺƼ-ƿǄ-ʓʕ-ʯͰ-ͳͶͷͻ-ͽͿΆΈ-ΊΌΎ-ΡΣ-ϵϷ-ҁҊ-ԯԱ-ՖႠ-ჅᎠ-Ᏽᏸ-ᏽᲐ-ᲺᲽ-Ჿᴀ-ᴫᵫ-ᵷᵹ-ᶚḀ-ἕἘ-Ἕἠ-ὅὈ-Ὅὐ-ὗὙὛὝὟ-ώᾀ-ᾴᾶ-ᾼιῂ-ῄῆ-ῌῐ-ΐῖ-Ίῠ-Ῥῲ-ῴῶ-ῼℂℇℊ-ℓℕℙ-ℝℤΩℨK-ℭℯ-ℴℹℼ-ℿⅅ-ⅉⅎↃↄⰀ-ⱻⱾ-ⳤⳫ-ⳮⳲⳳꙀ-ꙭꚀ-ꚛꜢ-ꝯꝱ-ꞇꞋ-ꞎꭰ-ꮿﬀ-ﬆﬓ-ﬗＡ-Ｚａ-ｚ𐐀-𐑏𐒰-𐓓𐓘-𐓻𐲀-𐲲𐳀-𐳲𑢠-𑣟𞤀-𞥃]+"###,
                r###"\s?[!-/:-~！-／：-～‘-‟　-。]+"###,
                r###"\s+$"###,
                r###"[一-龥ࠀ-一가-퟿]+"###,
                r###"\p{N}+"###,
            ], true),
            Self::Deepseek3Llm | Self::HunyuanDense | Self::JoyaiLlm | Self::HyV4 => (&[
                r###"\p{N}{1,3}"###,
                r###"[一-龥぀-ゟ゠-ヿ]+"###,
                r###"[!"#$%&'()*+,\-./:;<=>?@\[\\\]^_`{|}~][A-Za-z]+|[^
\p{L}\p{P}\p{S}]?[\p{L}\p{M}]+| ?[\p{P}\p{S}]+[
]*|\s*[
]+|\s+(?!\S)|\s+"###,
            ], true),
            Self::Spark25 => (&[
                r###"\p{N}{1,3}"###,
                r###"[一-龥぀-ゟ゠-ヿ]+"###,
                r###"[!"#$%&'()*+,\-./:;<=>?@\[\\\]^_`{|}~][A-Za-z]+|[^
\p{L}\p{P}\p{S}]?[\p{L}\p{M}]+| ?[\p{P}\p{S}]+|[
]|\s+(?!\S)|\s+"###,
                r###"\p{N}"###,
            ], true),
            Self::Youtu => (&[
                r###"[가-힣ㄱ-ㆎ]+|[！…“”‘’—：；，、-〿︰-﹏]+|[ㄅ-ㄯ]+|[一-龥぀-ゟ゠-ヿ]+"###,
                r###"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])?|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])?|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+"###,
            ], true),
            Self::DeepseekCoder => (&[
                r###"[
]"###,
                r###"\s?\p{L}+"###,
                r###"\s?\p{P}+"###,
                r###"[一-龥ࠀ-一가-퟿]+"###,
                r###"\p{N}"###,
            ], true),
            Self::Falcon => (&[
                r###"[\p{P}\$\+<=>\^~\|`]+"###,
                r###"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)"###,
                r###"[0-9][0-9][0-9]"###,
            ], true),
            Self::Starcoder | Self::Refact | Self::CommandR | Self::Smollm | Self::Codeshell | Self::Exaone | Self::Minerva | Self::Mellum2 => (&[
                r###"\p{N}"###,
                r###"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)"###,
            ], true),
            Self::Gpt2 | Self::Mpt | Self::Olmo | Self::Jais | Self::Trillion | Self::GraniteDocling => (&[
                r###"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)"###,
            ], true),
            Self::Stablelm2 | Self::Qwen2 | Self::Hunyuan | Self::SolarOpen => (&[
                r###"(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"###,
            ], true),
            Self::Qwen35 => (&[
                r###"(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"###,
            ], true),
            Self::Poro | Self::Bloom | Self::Gpt3Finnish => (&[
                r###" ?[^(\s|.,!?…。，、।۔،)]+"###,
            ], true),
            Self::Chatglm4 => (&[
                r###"(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"###,
            ], true),
            Self::Viking => (&[
                r###" ?[^(\s|.,!?…。，、।۔،)]+"###,
                r###"\p{N}"###,
            ], true),
            Self::Tekken | Self::Nemotron35Embd => (&[
                r###"[^\r\n\p{L}\p{N}]?((?=[\p{L}\p{M}])([^a-z]))*((?=[\p{L}\p{M}])([^A-Z]))+|[^\r\n\p{L}\p{N}]?((?=[\p{L}\p{M}])([^a-z]))+((?=[\p{L}\p{M}])([^A-Z]))*|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+"###,
            ], true),
            Self::TekkenNum3 => (&[
                r###"[^\r\n\p{L}\p{N}]?((?=[\p{L}\p{M}])([^a-z]))*((?=[\p{L}\p{M}])([^A-Z]))+|[^\r\n\p{L}\p{N}]?((?=[\p{L}\p{M}])([^a-z]))+((?=[\p{L}\p{M}])([^A-Z]))*|\p{N}{1,3}(?=(?:\p{N}{3})*(?!\p{N}))|\p{N}{1,2}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+"###,
            ], true),
            Self::Chameleon => (&[
                r###"<sentinel:[0-9]+>"###,
                r###"(IMGIMG)((A|B|C|D|E|F|G|H|I){1,4})Z"###,
                r###"([\t\n]|    |  )"###,
                r###"\p{N}"###,
                r###"[\p{P}!-/:-@\[-`{-~]"###,
                r###"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)"###,
            ], true),
            Self::Gpt4o | Self::MinimaxM2 => (&[
                r###"[^\r\n\p{L}\p{N}]?((?=[\p{L}\p{M}])([^a-z]))*((?=[\p{L}\p{M}])([^A-Z]))+(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])?|[^\r\n\p{L}\p{N}]?((?=[\p{L}\p{M}])([^a-z]))+((?=[\p{L}\p{M}])([^A-Z]))*(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])?|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+"###,
            ], true),
            Self::GraniteEmbMulti => (&[
                r###"[^\r\n\p{L}\p{N}]?((?=[\p{L}\p{M}])([^a-z]))*((?=[\p{L}\p{M}])([^A-Z]))+(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])?|[^\r\n\p{L}\p{N}]?((?=[\p{L}\p{M}])([^a-z]))+((?=[\p{L}\p{M}])([^A-Z]))*(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])?|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+"###,
            ], true),
            Self::TinyAya => (&[
                r###"\d{1,3}(?=(?:\d{3})*\b)"###,
                r###"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])?|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])?|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+"###,
            ], true),
            Self::KimiK2 => (&[
                r###"\p{Han}+"###,
            ], true),
            Self::Superbpe => (&[
                r###"\p{N}+"###,
                r###"(?=(\d{3})+(?!\d))"###,
            ], true),
            Self::Bailingmoe => (&[
                r###"'(?:[sSdDmMtT]|[lL][lL]|[vV][eE]|[rR][eE])|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"###,
            ], true),
            Self::SeedCoder => (&[
                r###"(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1}| ?[^\s\p{L}\p{N}\r\n]+|\s*[\r\n]+|\s+(?!\S)|\s+"###,
            ], true),
            Self::Ufakzeka => (&[
                r###"[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"###,
            ], true),
            Self::Grok2 => (&[
                r###"(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"###,
            ], true),
            Self::Afmoe => (&[
                r###"\p{AFMoE_digits}"###,
                r###"[一-鿿㐀-䶿豈-﫿぀-ゟ゠-ヿ･-ﾟ⼀-⿟เ-๿຀-໿ក-៿က-႟ꩠ-ꩿꧠ-꧿가-힯ᄀ-ᇿ]+"###,
                r###"[!"#$%&'()*+,\-./:;<=>?@\[\\\]^_`{|}~][A-Za-z]+|[^\r\n\p{L}\p{P}\p{S}]?[\p{L}\p{M}]+| ?[\p{P}\p{S}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"###,
            ], true),
            Self::Laguna => (&[
                r###"[^\n]+|[\n]+"###,
                r###"(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"###,
            ], true),
            Self::ExaoneMoe => (&[
                r###"(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\r\n\p{L}\p{N}]?(?:\p{L}\p{M}*(?: \p{L}\p{M}*)*)+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n/]?|\s*[\r\n]|\s+(?!\S)|\s+"###,
            ], true),
            Self::Gemma4 => (&[
                r###"[^\n]+|[\n]+"###,
            ], false),
            Self::SarvamMoe | Self::DomynSmall => (&[
                r###"[^\n]+|[\n]+"###,
            ], false),
            Self::Mmbert => (&[
                r###"[^\n]+|[\n]+"###,
            ], false),
            Self::Minicpm5 => (&[
                r###"\p{N}{1,3}"###,
                r###"(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}+| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"###,
            ], true),
            Self::K2Horizon => (&[
                r###"(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\r\n\p{L}\p{N}]?(?:\p{L}|\p{M}|\u200C|\u200D)+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"###,
            ], true),
            Self::Soprano => (&[
                r###"\p{N}"###,
                r###"\s+|[\p{L}\p{N}_]+|[^\p{L}\p{N}_\s]+"###,
            ], false),
            Self::Whitespace => (&[
                r###"\S+"###,
            ], false),
            Self::Default => (&[
                r###"[\p{P}\$\+<=>\^~\|]+"###,
                r###"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)"###,
                r###"\p{N}+"###,
                r###"[0-9][0-9][0-9]"###,
            ], true),
        }
    }
}
