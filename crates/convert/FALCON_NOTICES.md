# Falcon converter source and fixture notices

The Falcon metadata and QKV behavior follow the frozen llama.cpp source archive
with SHA-256 2db1e23f619fcabfd628481683c57b1e3e90db86bd045900a04b6073e1843122.
Its MIT license is retained as tests/fixtures/falcon-conversion/LICENSE.MIT.txt.
The tensor-map records are data derived from gguf-py/gguf/tensor_mapping.py and
gguf-py/gguf/constants.py; all 253 records and 510 native lookups are independently
recorded.

Configuration defaults follow Transformers 4.57.6 FalconConfig, licensed under
Apache-2.0:
https://github.com/huggingface/transformers/blob/v4.57.6/src/transformers/models/falcon/configuration_falcon.py

Punctuation semantics and the native tokenizer oracle use Tokenizers 0.22.2,
licensed under Apache-2.0. The retained package archive has SHA-256
b238e22d44a15349529690fb07bd645cf58149a1b1e44d6cb5bd1641ff1a6223.
The punctuation membership table records independent native output for every
Unicode scalar. The complete Apache license is retained as
tests/fixtures/falcon-conversion/LICENSE.APACHE-2.0.txt.

The original Falcon 7B tokenizer and model sidecars are from Technology
Innovation Institute's Apache-2.0 model:
https://huggingface.co/tiiuae/falcon-7b/tree/ec89142b67d748a1865ea4451372db8313ada0d8

The source model card is retained in tests/fixtures/falcon-conversion/MODEL_CARD.txt.
The immutable URLs, Git blob identities, complete sizes and SHA-256 values are
in official-sidecars.json. These fixtures contain no downloaded pretrained
tensor weights. All numerical tensor inputs in pipeline.bin are synthetic.

Original Python/Torch/NumPy/Transformers/Tokenizers execution and one-off oracle
capture programs remain outside the delivered product. No foreign runtime or
new registry dependency is introduced by this path.
