# GitHub CI findings

The first integration commit, 7e90479aa11f502907e062071d735bc90b1e5e73, ran as [workflow 42610404407](https://github.com/adybag14-cyber/rust-llama/actions/runs/42610404407). Ubuntu completed every gate. Windows exposed an overload-response socket lifecycle bug in `connection_admission_limit_rejects_excess_socket`: closing with unread incoming request bytes could reset the connection before the peer observed the written HTTP 503.

The fix half-closes the write side after the error response, then drains at most 8 KiB for at most 50 ms before releasing the socket. Rejection work stays bounded on the accept loop. The regression now checks both a normal request and a request with unread body data; it continues to require the complete 503 response rather than accepting a reset.

The focused Windows run passed all 20 TCP tests and strict all-targets Clippy for server/inference/CLI after the fix (Hermes job job-1794225053649-3f493600-9e1). The original failed CI log is preserved here. A subsequent CI run must verify the committed fix; local success alone is not recorded as remote success.
