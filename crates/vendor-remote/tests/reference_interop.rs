//! Opt-in differential tests against a separately built, frozen original.
//! No native source or library participates in the Rust product build.
use llama_vendor_remote::rpc::{Limits, RpcClient, ServerSession, exercise_client};
use std::net::{Ipv4Addr, SocketAddr, TcpListener};
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};
struct KillOnDrop(Child);
impl Drop for KillOnDrop {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}
fn oracle() -> std::ffi::OsString {
    std::env::var_os("T08_NATIVE_RPC_ORACLE")
        .expect("set T08_NATIVE_RPC_ORACLE to independently compiled frozen rpc-oracle")
}

#[test]
#[ignore = "requires separately compiled frozen C++ RPC oracle"]
fn native_rpc_layout_matches_explicit_codec() {
    let out = Command::new(oracle()).arg("--layout").output().unwrap();
    assert!(out.status.success());
    let text = String::from_utf8(out.stdout).unwrap();
    for fragment in [
        "\"tensor_size\":296",
        "\"offset_data\":220",
        "\"offset_name\":228",
        "\"offset_use_count\":292",
        "\"hello_request_size\":24",
        "\"hello_response_size\":28",
        "\"command_count\":24",
    ] {
        assert!(text.contains(fragment), "{text}");
    }
    println!("Frozen C++ layout: {}", text.trim());
}
#[test]
#[ignore = "requires separately compiled frozen C++ RPC oracle"]
fn native_rpc_server_accepts_rust_client() {
    let listener = TcpListener::bind((Ipv4Addr::LOCALHOST, 0)).unwrap();
    let address = listener.local_addr().unwrap();
    drop(listener);
    let mut child = KillOnDrop(
        Command::new(oracle())
            .arg(address.to_string())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .unwrap(),
    );
    let deadline = Instant::now() + Duration::from_secs(5);
    let mut client = loop {
        match RpcClient::connect(address, Limits::default()) {
            Ok(c) => break c,
            Err(error) => {
                assert!(
                    child.0.try_wait().unwrap().is_none(),
                    "native server exited before ready: {error}"
                );
                assert!(
                    Instant::now() < deadline,
                    "native server unavailable: {error}"
                );
                std::thread::sleep(Duration::from_millis(10));
            }
        }
    };
    let result = exercise_client(&mut client).unwrap();
    println!("Rust client -> frozen native CPU/RPC server: {result}");
}
#[test]
#[ignore = "requires separately compiled frozen C++ RPC oracle"]
fn rust_rpc_server_accepts_native_client() {
    let listener = TcpListener::bind((Ipv4Addr::LOCALHOST, 0)).unwrap();
    let address: SocketAddr = listener.local_addr().unwrap();
    let server = std::thread::spawn(move || {
        let (mut stream, _) = listener.accept().unwrap();
        stream.set_nodelay(true).unwrap();
        stream
            .set_read_timeout(Some(Duration::from_secs(5)))
            .unwrap();
        stream
            .set_write_timeout(Some(Duration::from_secs(5)))
            .unwrap();
        let mut s = ServerSession::new(Limits::default()).unwrap();
        s.serve(&mut stream).unwrap();
        assert_eq!(s.allocated_bytes(), 0);
        s.commands_processed
    });
    let output = Command::new(oracle())
        .args(["--client", &address.to_string()])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "native client status {}: {}",
        output.status,
        String::from_utf8_lossy(&output.stderr)
    );
    println!("{}", String::from_utf8_lossy(&output.stdout).trim());
    assert!(server.join().unwrap() >= 8);
}
