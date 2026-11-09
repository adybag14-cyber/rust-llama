# GitHub CI findings

The first integration commit, 7e90479aa11f502907e062071d735bc90b1e5e73, ran as [workflow 42610404407](https://github.com/adybag14-cyber/rust-llama/actions/runs/42610404407). Ubuntu completed every gate. Windows exposed an overload-response socket lifecycle bug in `connection_admission_limit_rejects_excess_socket`: closing with unread incoming request bytes could reset the connection before the peer observed the written HTTP 503.

The fix half-closes the write side after the error response, then drains at most 8 KiB for at most 50 ms before releasing the socket. Rejection work stays bounded on the accept loop. The regression checks both a normal request and a request with unread body data and requires the complete 503 response.

The focused Windows run passed all 20 TCP tests and strict all-targets Clippy for server/inference/CLI (Hermes job job-1794225053649-3f493600-9e1). The original failed CI log is preserved here.

The committed fix, fdfeda09bbc72fa61a25218cf93a9ca5807527c1, then passed every configured gate on Windows and Ubuntu in [workflow 42612502780](https://github.com/adybag14-cyber/rust-llama/actions/runs/42612502780). The observed job and step results are preserved in `socket-fix-jobs.json`. These results apply to that exact commit; later work requires its own validation.

## Extension workflow validation

Commit 57fb7e9c1415973bbad7b4b36510eac33b0681fa failed workflow validation before creating any jobs in [run 42620236813](https://github.com/adybag14-cyber/rust-llama/actions/runs/42620236813). GitHub rejected all four uses of runner.temp in the job-level env mapping. The exact annotation and before/after workflow hashes are recorded in extension-workflow-validation.json. The configuration fix moves these environment values to the pretrained-model comparison step, where the runner context exists. Model URLs, expected hashes, numerical tolerances, and test commands are unchanged. Local Windows/Linux evidence for 57fb7e9 remains separate from CI; this failed run did not execute any tests.
