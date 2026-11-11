# Window, relative-position, triangular solve and loss operators

These seven native Rust DAG operations follow the frozen GGML constructors and CPU
kernels. Every result owns independent storage. Input buffers, shared views and
their payloads remain immutable, including on checked errors or cancellation.

| Original ID | Rust operation | Input and output contract |
| ---: | --- | --- |
| 78 | WindowPartition { window } | F32 [channels,width,height,1] to padded square windows [channels,window,window,patches]. |
| 79 | WindowUnpartition { width,height,window } | F32 window storage to [channels,width,height,1], preserving the original flat physical address formula. |
| 80 | GetRelativePosition { query,key } | Equal positive query/key widths and 2*width-1 input rows; F16 or BF16 storage copied as raw words to the original F16 output type. |
| 81 | AddRelativePosition | Contiguous F32 attention scores, width offsets and height offsets; source loop order and any unmodified fourth-axis slabs retained. |
| 85 | SolveTriangular | Contiguous F32 lower-triangular A and RHS B, equal batch axes; left solve with a non-unit diagonal, matching the only original constructor form. |
| 97 | CrossEntropyLoss | Equal-shape F32 logits and labels; source one-worker scalar log-softmax and loss reduction. Labels need not be normalized or one-hot. |
| 98 | CrossEntropyLossBack | Scalar F32 output gradient followed by contiguous F32 logits and labels; source native-profile softmax and gradient arithmetic. |

## Physical layouts

WIN_PART, WIN_UNPART and GET_REL_POS deliberately use the original kernels'
flattened physical reads, including when the input descriptor contains padding.
They do not silently gather the logical view first. GET_REL_POS does not convert
BF16 values numerically: its source kernel copies 16-bit words into a tensor that
the original constructor labels F16. All 65,536 input words are covered for both
input dtypes.

WIN_UNPART retains source behavior for additional window geometry and surplus
windows when all requested bytes lie within the declared input span. It checks
every read. Accessible bytes elsewhere in an underlying shared allocation never
make an out-of-view access admissible.

ADD_REL_POS validates equal width/height offset shapes, square key dimensions,
query-area dimensions and patch count. It copies all input scores before updating
the first source-defined slab. The nested height and width additions retain their
original order; combining both offsets before adding would change rounding.

SOLVE_TRI reads the lower half and diagonal in source row-major order. Upper
entries remain ignored. Its output keeps B's dimensions. Each F32 product and
addition rounds independently; there is no fused multiply-add transformation.

## Numerical policies and explicit boundaries

Cross entropy's forward helper uses the scalar expf loop on both original
profiles, F64 accumulation of F32 exponentials, then F32 logf. Per-class products
are summed in F64, converted to F32 per row, and accumulated in F32 over rows.
Backward loss reuses the maintained CPU row-math implementation so the original
GNU SSE2 exponential prefix and scalar tails are preserved. Forward loss's
single-worker reduction order does not establish arbitrary thread-count parity.

Loss inputs and scalar gradient must be finite. This is an explicit checked
boundary around the original debug assertion and nonfinite-loss domains.
Empty geometry, nonpositive or oversized window/query parameters, unsupported
dtypes, zero triangular diagonals, malformed shape relationships and addresses
outside the declared view return errors. Geometry fits positive signed I32,
matching the original CPU loop and parameter domain. Fallible output/scratch
allocations and checked size/address arithmetic precede unsafe-sized work.

The original MSVC and GNU kernels can select different NaN payload operands.
The Rust implementation explicitly quiets input NaNs and preserves the measured
operand order for relative-position additions and triangular arithmetic on the
two exercised x86_64 profiles. Competing-NaN policy on other targets returns an
explicit unsupported error. IEEE floating-exception flags are outside this
immutable tensor API.

## Independent reference cohorts

The main corpus contains 920 original CPU graphs per compiler profile and
1,135,798 output bytes. It includes padded layouts, extra planes, all F16/BF16
words, variable window sizes, triangular batches, cancellation-sensitive
floating reductions, saturated/tiny logits and one-hot/arbitrary label vectors.

A second corpus contains 111 original graphs per profile and 25,992 output bytes.
It covers subnormals, signed zeros, infinities, quiet/signaling NaNs and competing
NaN payloads for addition and solve. The first implementation differed in six
competing-NaN cases on each host; those failed comparisons and every original
expected byte are retained. The correction changes explicit operand selection,
without changing fixtures or introducing numerical tolerances.

A third independent cohort adds 222 graphs and 83,424 output bytes per host,
varying relative-position key sizes and triangular orders, column counts, batch
sizes and which operands carry distinct NaN payloads. Together the three corpora
contain 1,253 graphs and 1,245,214 output bytes per host. These additional controls
retain the original expected operands rather than synthesizing answers from Rust.

A fourth cohort adds 168 signaling-NaN controls over one-key query widths
1 through 17, 24, 31, 32 and 33. The original GNU compiler specializes this case
when query width exceeds one: its first addition loads height before the
accumulator, including vector and scalar-tail lanes. Width one and the general
key loop load the accumulator first. The original library disassembly and the
native boundary corpus establish that distinction. All four cohorts total
1,421 graphs and 1,271,422 exact output bytes per host.

The native generators, original libraries and reference builds remain outside
the Rust product. Fixture provenance includes baseline identity, original
source/library hashes and case-ledger hashes. The product contains only raw
reference data and Rust implementation/tests, with the existing GGML MIT
attribution.

Full graph autodifferentiation, optimizer state updates, CPU scheduling, writable
or device buffers and complete model/backend parity remain separate open work.
