# Tiny StarCoder CI configuration validation

This package records local validation of the workflow change in commit
a299c6c3c3e977c3ae020d8781a6635066bd6ad7. The validated workflow SHA-256 is
1ddfe489b2ac1ce5affde30c26f12163488c7840dc61e6b74fee8875791c3127.

These receipts do not record a hosted GitHub Actions run, a fresh public
checkpoint download, a new conversion, or a new inference comparison.
The CI commit was validated in its isolated worktree. Root workflow integration
was deliberately deferred until the independent protected integration runs
finished; packaging these receipts does not claim that integration occurred.

## Recorded checks

- Actionlint completed without diagnostics.
- The native PowerShell parser accepted all seven relevant command blocks.
- The unchanged workflow acquisition and verification body executed for both
  Windows and Linux manifest selections on Windows PowerShell 7. Only its
  acquisition boundary was replaced externally: a pinned-URL provider created
  hard links to the already acquired, unchanged official checkpoint files.
  No test hook was added to the workflow.
- Each profile verified all nine input lengths and SHA-256 hashes.
- Existing, previously compared F16, Q8_0 and auto converter outputs were
  materialized at the exact workflow paths and all complete hashes verified:
  881,027,328 output bytes per profile. No conversion was rerun.
- The two inference input identities matched their corresponding committed
  native reference metadata.
- Existing Windows MSVC and GNU release test binaries each selected exactly
  one converter test and one test for each of F16 and Q8_0 inference.
- The general ignored-test StarCoder exclusion selected zero StarCoder tests
  on each native harness.
- GNU additionally verified all nine materialized input hashes and both
  inference output paths and hashes through native POSIX paths. GNU PowerShell
  was unavailable; the Linux manifest branch rehearsal ran on Windows.

The selected tests were listed, not executed by this CI-configuration check.
The earlier complete conversion and numerical inference acceptance is retained
separately in the StarCoder conversion and inference evidence packages.

## Integrity and scope

The receipts directory contains exact byte copies of the external reports,
command logs, command-block data and materialized-path inventory. Empty log
files are retained. The copy ledger records each original path and verifies
its SHA-256 against the copied file. The manifest records all package payload
hashes and the identities of the external validation and packaging drivers.
Executable Python and PowerShell drivers remain outside the product.

Large cached checkpoints, converted outputs and hard-linked rehearsal
directories are not copied into this evidence package. Their complete
identities are recorded in the receipts.

StarCoder2 3B remains an external memory qualification. This CI configuration
package makes no hosted-runner acceptance or process-memory claim.
