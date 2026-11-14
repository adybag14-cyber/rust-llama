# StarCoder converter source and fixture notices

The StarCoder/StarCoder2 behavior and tensor aliases follow the frozen
llama.cpp archive with SHA-256
2db1e23f619fcabfd628481683c57b1e3e90db86bd045900a04b6073e1843122.
Its complete MIT license is retained in
tests/fixtures/starcoder-conversion/LICENSE.MIT.txt. The mapping data comes
from gguf-py/gguf/tensor_mapping.py and gguf-py/gguf/constants.py.

Configuration defaults and AutoConfig semantics follow Transformers 4.57.6:

- https://github.com/huggingface/transformers/blob/v4.57.6/src/transformers/models/gpt_bigcode/configuration_gpt_bigcode.py
- https://github.com/huggingface/transformers/blob/v4.57.6/src/transformers/models/starcoder2/configuration_starcoder2.py

Transformers and Tokenizers reference behavior is Apache-2.0 licensed.
The complete Apache-2.0 license is preserved in
tests/fixtures/starcoder-conversion/LICENSE.APACHE-2.0.txt.
The GPT-2 tokenizer data reused from the existing independent corpus retains
its separate GPT-2 Modified MIT notice in
tests/fixtures/starcoder-conversion/LICENSE.GPT2.txt. Its original model
revision, immutable URLs and exact base-sidecar identities are retained in
gpt2-sidecar-provenance.json.

The official tokenizer sidecars and unchanged model cards are from:

- https://huggingface.co/bigcode/tiny_starcoder_py/tree/8547527bef0bc927268c1653cce6948c5c242dd1
- https://huggingface.co/bigcode/starcoder2-3b/tree/733247c55e3f73af49ce8e9c7949bf14af205928

These data are licensed under BigCode Open RAIL-M v1, including the agreement's
use restrictions. The complete license text is retained as
tests/fixtures/starcoder-conversion/LICENSE.BIGCODE-OPENRAIL-M.txt.
license-provenance.json records the pinned official Space URL, revision,
source SHA-256, extraction rule and complete extracted-text SHA-256.
TINY_MODEL_CARD.txt and STARCODER2_MODEL_CARD.txt retain the original model
cards; acquisition manifests record immutable URLs, Git/LFS identities,
complete sizes and SHA-256 values.

The ordinary input pack contains synthetic numerical weights and tokenizer
sidecars only. The complete tiny StarCoder pretrained checkpoint and all GGUF
outputs remain external. No StarCoder2 pretrained checkpoint is bundled.

Original Python/Torch/NumPy/Transformers/Tokenizers execution and one-off
capture programs remain outside the maintained product. This extension adds
no foreign runtime or registry dependency.
