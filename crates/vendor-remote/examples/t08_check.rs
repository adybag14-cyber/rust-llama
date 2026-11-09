use llama_vendor_remote::rpc::{Limits, RpcClient, ServerSession, exercise_client};
use llama_vendor_remote::{Error, Result};
fn main() {
    if let Err(e) = run() {
        eprintln!("{e}");
        std::process::exit(1);
    }
}
fn run() -> Result<()> {
    let args: Vec<_> = std::env::args().collect();
    match args.get(1).map(String::as_str) {
        Some("client") => {
            let address = args
                .get(2)
                .ok_or(Error::Bounds("client requires IP:PORT"))?
                .parse()
                .map_err(|_| Error::Bounds("invalid IP:PORT"))?;
            let mut client = RpcClient::connect(address, Limits::default())?;
            println!("{}", exercise_client(&mut client)?);
        }
        Some("serve-once") => {
            let address: std::net::SocketAddr = args
                .get(2)
                .ok_or(Error::Bounds("serve-once requires loopback IP:PORT"))?
                .parse()
                .map_err(|_| Error::Bounds("invalid IP:PORT"))?;
            if !address.ip().is_loopback() {
                return Err(Error::Bounds(
                    "checkpoint example accepts loopback addresses only",
                ));
            }
            let listener = std::net::TcpListener::bind(address)?;
            println!("LISTEN {}", listener.local_addr()?);
            let (mut stream, _) = listener.accept()?;
            stream.set_nodelay(true)?;
            stream.set_read_timeout(Some(std::time::Duration::from_secs(5)))?;
            stream.set_write_timeout(Some(std::time::Duration::from_secs(5)))?;
            let mut session = ServerSession::new(Limits::default())?;
            session.serve(&mut stream)?;
            println!(
                "CLOSED commands={} retained_buffers={} retained_bytes={}",
                session.commands_processed,
                session.allocation_count(),
                session.allocated_bytes()
            );
        }
        Some("swiv") => {
            let input = args
                .get(2)
                .ok_or(Error::Bounds("swiv requires input and output"))?;
            let output = args
                .get(3)
                .ok_or(Error::Bounds("swiv requires input and output"))?;
            let bytes = std::fs::read(input)?;
            let result = llama_vendor_remote::dsp::append_swiv(&bytes)?;
            std::fs::write(output, &result)?;
            println!("SWIV generated: {} bytes", result.len());
        }
        Some("et-audit") => {
            let path = args
                .get(2)
                .ok_or(Error::Bounds("et-audit requires disassembly path"))?;
            let text = std::fs::read_to_string(path)?;
            let found = llama_vendor_remote::dsp::et_instruction_audit(&text);
            for item in &found {
                println!("line {}: {}", item.line, item.mnemonic);
            }
            if !found.is_empty() {
                return Err(Error::Unsupported(
                    "ET disassembly contains source-forbidden instructions",
                ));
            }
            println!("ET mnemonic guard PASS; binary execution not claimed");
        }
        Some(other) => {
            return Err(Error::Protocol(format!(
                "unknown command {other}; expected client, serve-once, swiv or et-audit"
            )));
        }
        None => {
            let check = llama_vendor_remote::self_check()?;
            println!("RPC {}\nBLAS {}", check.rpc, check.blas);
            for p in llama_vendor_remote::probe_all() {
                println!(
                    "{}\t{:?}\t{}\tPENDING: {}",
                    p.family, p.status, p.detail, p.remaining_gate
                );
            }
        }
    }
    Ok(())
}
