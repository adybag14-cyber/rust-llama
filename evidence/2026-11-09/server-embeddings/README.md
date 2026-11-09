# Dense BERT embedding HTTP evidence

The final integrated results are
[Windows](../native-surface/windows/revalidated/status.json) and
[Linux](../native-surface/linux/revalidated/status.json). Earlier logs and status
files in this directory are retained as individual attempts, including failures;
their presence does not replace the final revalidation record.

## Original-server fixtures

The six declared fixture/provenance files under fixtures/inference were captured
from the frozen original C++ server, built separately outside the Rust product.

| Corpus | Requests | Numeric comparisons | Base64 vectors |
| --- | ---: | ---: | ---: |
| Tiny BERT protocol and arithmetic | 226 | 1,498 | 4 |
| Numeric JSON conversion edges | 30 | 168 | 0 |
| Pretrained BGE pooling and normalization | 27 | 23,479 | 3 |

Numeric comparison counts include protocol metadata as well as embedding values.
Protocol integers, shapes, usage, error statuses and explicitly supplied model
labels remain exact. A request without a model label maps the original server's
fixture alias to the Rust provider's model identifier.

Windows embedding floats are bitwise equal to the original reference. Linux
passes the declared portable bounds. The pretrained model remains an external,
hash-pinned input; the tiny model is a structural/protocol fixture.

## Normalization and portable arithmetic

Normalization follows the original common_embd_normalize computation:
-1 preserves values, 0 scales by the maximum absolute coordinate to 32760,
1 uses the taxicab norm, 2 uses the Euclidean norm, and other integer settings use
the source p-norm expression. Unpooled token vectors remain raw.

The raw encoder budget remains absolute 2e-4 plus relative 2e-5. Before the first
Rust pretrained HTTP comparison, normalization-budget-before-comparison.json
recorded its propagation through mode 0. With original raw maximum
M = 3.1209604740142822 and E = 0.0002 + 0.00002 M, the scale-aware budget is:

- Absolute: 32760 * 0.0002 / (M - E) = 2.0995302605172497.
- Relative: (0.00002 M + E) / (M - E), plus four F32 epsilons,
  = 0.000104568426045001.

The largest Linux mode-0 difference is 0.75439453125, at this much larger output
scale. The expected fixture values were never changed. A separate 4,608-output
normalization check uses identical native inputs, requiring exact Windows
results and at most two F32 ULPs on Linux.

## Lifecycle and concrete executable checks

The complete server suite now has 48 ordinary tests: 19 unit tests, two original
HTTP fixture tests, six lifecycle tests and 21 TCP tests. The pretrained BGE HTTP
test is run explicitly after the ordinary suite.

Embedding work shares the inference admission limit, executes on bounded
workers, and handles timeout, disconnect, shutdown and capacity recovery.
A regression found that the listener could cancel a timed-out worker just before
the handler observed its own later-started clock. This could turn the intended
504 into a discarded internal cancellation response. Both embedding and existing
nonstream generation handlers now start their clock before admission and give an
expired deadline precedence over a received cancellation result. The original
failure and the focused corrected runs remain in this directory.

The final standalone release executable was also exercised on both platforms:
all six pooling/normalization configurations against 12 native HTTP responses,
the singular --embedding alias, missing/invalid options, an incompatible
generation model, graceful quit, and a live server after stdin EOF. The
11 recorded scenarios and executable hashes are in each platform's revalidated
embedding-cli.json.

Dense BERT generation, decoder embedding execution, reranking and full server
protocol parity remain open; unsupported requests fail explicitly.
