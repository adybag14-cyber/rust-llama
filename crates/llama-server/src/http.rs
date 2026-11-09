//! Strict HTTP/1.0 and HTTP/1.1 request framing. One request per connection.
use std::collections::BTreeMap;
use std::io::{self, Read, Write};

#[derive(Clone, Debug)]
pub struct HttpLimits {
    pub max_header_bytes: usize,
    pub max_body_bytes: usize,
    pub max_headers: usize,
    pub max_target_bytes: usize,
}
impl Default for HttpLimits {
    fn default() -> Self {
        Self {
            max_header_bytes: 16 * 1024,
            max_body_bytes: 1024 * 1024,
            max_headers: 64,
            max_target_bytes: 4096,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HttpError {
    pub status: u16,
    pub message: String,
}
impl HttpError {
    fn new(status: u16, message: impl Into<String>) -> Self {
        Self {
            status,
            message: message.into(),
        }
    }
}
impl std::fmt::Display for HttpError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message)
    }
}
impl std::error::Error for HttpError {}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Request {
    pub method: String,
    pub target: String,
    pub headers: BTreeMap<String, String>,
    pub body: Vec<u8>,
}
impl Request {
    pub fn header(&self, key: &str) -> Option<&str> {
        self.headers
            .get(&key.to_ascii_lowercase())
            .map(String::as_str)
    }
    pub fn path(&self) -> &str {
        self.target
            .split_once('?')
            .map_or(&self.target, |(path, _)| path)
    }
    pub fn query(&self, name: &str) -> Result<Option<String>, HttpError> {
        let mut found = None;
        if let Some((_, query)) = self.target.split_once('?') {
            for pair in query.split('&').filter(|s| !s.is_empty()) {
                let (key, value) = pair.split_once('=').unwrap_or((pair, ""));
                if percent_decode(key)? == name {
                    if found.is_some() {
                        return Err(HttpError::new(400, "Duplicate query parameter"));
                    }
                    found = Some(percent_decode(value)?);
                }
            }
        }
        Ok(found)
    }
}

pub fn percent_decode(text: &str) -> Result<String, HttpError> {
    let bytes = text.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        match bytes[i] {
            b'%' => {
                let digits = bytes
                    .get(i + 1..i + 3)
                    .ok_or_else(|| HttpError::new(400, "Invalid percent escape"))?;
                let hex = |b: u8| match b {
                    b'0'..=b'9' => Some(b - b'0'),
                    b'a'..=b'f' => Some(b - b'a' + 10),
                    b'A'..=b'F' => Some(b - b'A' + 10),
                    _ => None,
                };
                let a =
                    hex(digits[0]).ok_or_else(|| HttpError::new(400, "Invalid percent escape"))?;
                let b =
                    hex(digits[1]).ok_or_else(|| HttpError::new(400, "Invalid percent escape"))?;
                out.push(a * 16 + b);
                i += 3;
            }
            b'+' => {
                out.push(b' ');
                i += 1;
            }
            b => {
                out.push(b);
                i += 1;
            }
        }
    }
    String::from_utf8(out).map_err(|_| HttpError::new(400, "Query is not valid UTF-8"))
}

pub struct RequestParser {
    limits: HttpLimits,
    bytes: Vec<u8>,
    head: Option<(Request, usize, usize)>,
    complete: bool,
}
impl RequestParser {
    pub fn new(limits: HttpLimits) -> Self {
        Self {
            limits,
            bytes: Vec::new(),
            head: None,
            complete: false,
        }
    }
    pub fn feed(&mut self, bytes: &[u8]) -> Result<Option<Request>, HttpError> {
        if self.complete {
            return Err(HttpError::new(400, "Parser already completed"));
        }
        let ceiling = self
            .limits
            .max_header_bytes
            .saturating_add(self.limits.max_body_bytes);
        if self.bytes.len().saturating_add(bytes.len()) > ceiling {
            return Err(HttpError::new(413, "Request exceeds configured byte limit"));
        }
        let previous_len = self.bytes.len();
        self.bytes.extend_from_slice(bytes);
        if self.head.is_none() {
            let scan_start = previous_len.saturating_sub(3);
            if let Some(relative_start) = self.bytes[scan_start..]
                .windows(4)
                .position(|w| w == b"\r\n\r\n")
            {
                let start = scan_start + relative_start;
                let end = start + 4;
                if end > self.limits.max_header_bytes {
                    return Err(HttpError::new(431, "Request headers too large"));
                }
                let (request, body_len) = parse_head(&self.bytes[..end], &self.limits)?;
                self.head = Some((request, end, body_len));
            } else {
                if self.bytes.len() >= self.limits.max_header_bytes {
                    return Err(HttpError::new(431, "Request headers too large"));
                }
                // Bare LF and CR followed by a non-LF byte are never framing.
                for (i, b) in self
                    .bytes
                    .iter()
                    .enumerate()
                    .skip(previous_len.saturating_sub(1))
                {
                    if (*b == b'\n' && (i == 0 || self.bytes[i - 1] != b'\r'))
                        || (*b == b'\r' && self.bytes.get(i + 1).is_some_and(|next| *next != b'\n'))
                    {
                        return Err(HttpError::new(400, "Headers require CRLF line endings"));
                    }
                }
                return Ok(None);
            }
        }
        let (_, body_start, body_len) = self.head.as_ref().unwrap();
        let body_end = body_start
            .checked_add(*body_len)
            .ok_or_else(|| HttpError::new(413, "Request length overflows address space"))?;
        if self.bytes.len() < body_end {
            return Ok(None);
        }
        let (mut request, start, len) = self.head.take().unwrap();
        request.body = self.bytes[start..start + len].to_vec();
        self.bytes.clear();
        self.complete = true;
        Ok(Some(request))
    }
}

fn token(byte: u8) -> bool {
    byte.is_ascii_alphanumeric() || b"!#$%&'*+-.^_`|~".contains(&byte)
}

fn parse_head(bytes: &[u8], limits: &HttpLimits) -> Result<(Request, usize), HttpError> {
    if !bytes.is_ascii() {
        return Err(HttpError::new(400, "Request headers must be ASCII"));
    }
    let text = std::str::from_utf8(bytes).unwrap();
    let mut lines = text[..text.len() - 4].split("\r\n");
    let mut first = lines.next().unwrap_or("").split(' ');
    let method = first.next().unwrap_or("");
    let target = first.next().unwrap_or("");
    let version = first.next().unwrap_or("");
    if first.next().is_some() || method.is_empty() || !method.bytes().all(token) {
        return Err(HttpError::new(400, "Malformed request line"));
    }
    if !matches!(version, "HTTP/1.1" | "HTTP/1.0") {
        return Err(HttpError::new(
            505,
            "Only HTTP/1.0 and HTTP/1.1 are supported",
        ));
    }
    if !target.starts_with('/')
        || target.starts_with("//")
        || target.contains('#')
        || target.bytes().any(|b| b <= 32 || b == 127)
    {
        return Err(HttpError::new(
            400,
            "Only origin-form request targets are supported",
        ));
    }
    if target.len() > limits.max_target_bytes {
        return Err(HttpError::new(414, "Request target too long"));
    }
    let mut headers = BTreeMap::new();
    for (index, line) in lines.enumerate() {
        if index >= limits.max_headers {
            return Err(HttpError::new(431, "Too many headers"));
        }
        let (name, value) = line
            .split_once(':')
            .ok_or_else(|| HttpError::new(400, "Malformed header"))?;
        if name.is_empty()
            || !name.bytes().all(token)
            || value.bytes().any(|b| (b < 32 && b != b'\t') || b == 127)
        {
            return Err(HttpError::new(400, "Invalid header name or value"));
        }
        let name = name.to_ascii_lowercase();
        if headers
            .insert(name, value.trim_matches([' ', '\t']).to_owned())
            .is_some()
        {
            return Err(HttpError::new(400, "Duplicate headers are not accepted"));
        }
    }
    if version == "HTTP/1.1" && headers.get("host").is_none_or(|h| h.is_empty()) {
        return Err(HttpError::new(400, "HTTP/1.1 requires Host"));
    }
    if headers.contains_key("transfer-encoding") {
        return Err(HttpError::new(
            400,
            "Transfer-Encoding is not supported; send Content-Length",
        ));
    }
    if headers.contains_key("expect") {
        return Err(HttpError::new(417, "Expect is not supported"));
    }
    let body_len = if let Some(length) = headers.get("content-length") {
        if length.is_empty() || !length.bytes().all(|b| b.is_ascii_digit()) {
            return Err(HttpError::new(400, "Invalid Content-Length"));
        }
        length
            .parse::<usize>()
            .map_err(|_| HttpError::new(413, "Content-Length overflows address space"))?
    } else {
        0
    };
    if body_len > limits.max_body_bytes {
        return Err(HttpError::new(413, "Request body too large"));
    }
    Ok((
        Request {
            method: method.to_owned(),
            target: target.to_owned(),
            headers,
            body: Vec::new(),
        },
        body_len,
    ))
}

pub fn read_request(reader: &mut impl Read, limits: HttpLimits) -> Result<Request, HttpError> {
    let mut parser = RequestParser::new(limits);
    let mut buffer = [0; 4096];
    loop {
        let count = reader.read(&mut buffer).map_err(|e| {
            HttpError::new(
                if matches!(
                    e.kind(),
                    io::ErrorKind::TimedOut | io::ErrorKind::WouldBlock
                ) {
                    408
                } else {
                    400
                },
                format!("Could not read request: {e}"),
            )
        })?;
        if count == 0 {
            return Err(HttpError::new(400, "Unexpected EOF in request"));
        }
        if let Some(request) = parser.feed(&buffer[..count])? {
            return Ok(request);
        }
    }
}

pub fn reason(status: u16) -> &'static str {
    match status {
        200 => "OK",
        204 => "No Content",
        400 => "Bad Request",
        401 => "Unauthorized",
        403 => "Forbidden",
        404 => "Not Found",
        405 => "Method Not Allowed",
        408 => "Request Timeout",
        409 => "Conflict",
        413 => "Content Too Large",
        414 => "URI Too Long",
        415 => "Unsupported Media Type",
        417 => "Expectation Failed",
        429 => "Too Many Requests",
        431 => "Request Header Fields Too Large",
        499 => "Client Closed Request",
        500 => "Internal Server Error",
        501 => "Not Implemented",
        503 => "Service Unavailable",
        504 => "Gateway Timeout",
        505 => "HTTP Version Not Supported",
        _ => "Error",
    }
}

pub fn write_head(
    writer: &mut impl Write,
    status: u16,
    content_type: &str,
    length: Option<usize>,
    headers: &[(String, String)],
) -> io::Result<()> {
    let valid_value = |value: &str| value.bytes().all(|byte| byte >= 32 && byte != 127);
    if !(100..=599).contains(&status)
        || content_type.is_empty()
        || !valid_value(content_type)
        || headers
            .iter()
            .any(|(name, value)| name.is_empty() || !name.bytes().all(token) || !valid_value(value))
    {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "Invalid response header",
        ));
    }
    write!(
        writer,
        "HTTP/1.1 {status} {}\r\nContent-Type: {content_type}\r\nConnection: close\r\nX-Content-Type-Options: nosniff\r\n",
        reason(status)
    )?;
    if let Some(length) = length {
        write!(writer, "Content-Length: {length}\r\n")?;
    }
    for (name, value) in headers {
        write!(writer, "{name}: {value}\r\n")?;
    }
    writer.write_all(b"\r\n")
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn request_at_every_split_boundary() {
        let data = b"POST /x?q=%E2%82%AC HTTP/1.1\r\nHost: local\r\nContent-Length: 3\r\n\r\nabc";
        for split in 0..data.len() {
            let mut parser = RequestParser::new(HttpLimits::default());
            assert!(parser.feed(&data[..split]).unwrap().is_none());
            let request = parser.feed(&data[split..]).unwrap().unwrap();
            assert_eq!(request.body, b"abc");
            assert_eq!(request.query("q").unwrap().as_deref(), Some("€"));
        }
    }
    #[test]
    fn rejects_smuggling_and_malformed_headers() {
        let bad = [
            "POST / HTTP/1.1\r\nHost: x\r\nContent-Length: 1\r\ncontent-length: 1\r\n\r\na",
            "POST / HTTP/1.1\r\nHost: x\r\nTransfer-Encoding: chunked\r\nContent-Length: 0\r\n\r\n",
            "GET / HTTP/1.1\nHost: x\n\n",
            "GET / HTTP/1.1\r\nHost: x\r\n folded: bad\r\n\r\n",
            "GET / HTTP/1.1\r\n\r\n",
            "GET http://x/ HTTP/1.1\r\nHost: x\r\n\r\n",
            "GET / HTTP/1.1\r\nHost: x\r\nX: a\nb\r\n\r\n",
        ];
        for input in bad {
            assert!(
                RequestParser::new(HttpLimits::default())
                    .feed(input.as_bytes())
                    .is_err(),
                "{input:?}"
            );
        }
    }
    #[test]
    fn enforces_limits_before_body_allocation() {
        let limits = HttpLimits {
            max_body_bytes: 4,
            ..HttpLimits::default()
        };
        assert_eq!(
            RequestParser::new(limits)
                .feed(b"POST / HTTP/1.1\r\nHost: x\r\nContent-Length: 5\r\n\r\n")
                .unwrap_err()
                .status,
            413
        );
        let limits = HttpLimits {
            max_header_bytes: 16,
            ..HttpLimits::default()
        };
        assert_eq!(
            RequestParser::new(limits)
                .feed(b"GET / HTTP/1.1\r\nHost:")
                .unwrap_err()
                .status,
            431
        );
    }
    #[test]
    fn duplicate_query_and_bad_percent_are_errors() {
        assert!(percent_decode("%xx").is_err());
        assert!(percent_decode("%F0%80").is_err());
        let request = Request {
            method: "GET".into(),
            target: "/?from=1&from=2".into(),
            headers: BTreeMap::new(),
            body: Vec::new(),
        };
        assert!(request.query("from").is_err());
    }
    #[test]
    fn arbitrary_public_limits_cannot_overflow_and_bad_headers_write_nothing() {
        let limits = HttpLimits {
            max_body_bytes: usize::MAX,
            ..HttpLimits::default()
        };
        let data = format!(
            "POST / HTTP/1.1\r\nHost: x\r\nContent-Length: {}\r\n\r\n",
            usize::MAX
        );
        assert_eq!(
            RequestParser::new(limits)
                .feed(data.as_bytes())
                .unwrap_err()
                .status,
            413
        );
        for (status, content_type, headers) in [
            (200, "text/plain\r\nEvil: yes", vec![]),
            (200, "text/plain", vec![("".into(), "a".into())]),
            (700, "text/plain", vec![]),
            (200, "text/plain", vec![("X-Test".into(), "a\u{1}".into())]),
        ] {
            let mut output = Vec::new();
            assert!(write_head(&mut output, status, content_type, None, &headers).is_err());
            assert!(output.is_empty());
        }
    }
}
