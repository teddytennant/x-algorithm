// MIT License
//
// Copyright (c) 2019 Jasper Hugo
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
//
// Derived from tokio-postgres-rustls 0.13.0
// (https://github.com/jbg/tokio-postgres-rustls); modified by X.AI Corp.

// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 X.AI Corp.
use std::future::Future;
use std::io;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};

use aws_lc_rs::digest;
use rustls::pki_types::ServerName;
use rustls::ClientConfig;
use tokio::io::{AsyncRead, AsyncWrite, ReadBuf};
use tokio_postgres::tls::{ChannelBinding, MakeTlsConnect, TlsConnect, TlsStream};
use tokio_rustls::TlsConnector;
use x509_cert::der::oid::db::rfc5912::{
    ECDSA_WITH_SHA_256, ECDSA_WITH_SHA_384, ID_SHA_1, ID_SHA_256, ID_SHA_384, ID_SHA_512,
    SHA_1_WITH_RSA_ENCRYPTION, SHA_256_WITH_RSA_ENCRYPTION, SHA_384_WITH_RSA_ENCRYPTION,
    SHA_512_WITH_RSA_ENCRYPTION,
};
use x509_cert::der::oid::ObjectIdentifier;
use x509_cert::der::Decode;
use x509_cert::Certificate;

pub fn channel_binding_digest(
    signature_algorithm: ObjectIdentifier,
) -> Option<&'static digest::Algorithm> {
    match signature_algorithm {
        ID_SHA_1
        | ID_SHA_256
        | SHA_1_WITH_RSA_ENCRYPTION
        | SHA_256_WITH_RSA_ENCRYPTION
        | ECDSA_WITH_SHA_256 => Some(&digest::SHA256),
        ID_SHA_384 | SHA_384_WITH_RSA_ENCRYPTION | ECDSA_WITH_SHA_384 => Some(&digest::SHA384),
        ID_SHA_512 | SHA_512_WITH_RSA_ENCRYPTION => Some(&digest::SHA512),
        _ => None,
    }
}

pub fn tls_server_end_point(cert_der: &[u8]) -> Option<Vec<u8>> {
    let cert = Certificate::from_der(cert_der).ok()?;
    let algorithm = channel_binding_digest(cert.signature_algorithm.oid)?;
    Some(digest::digest(algorithm, cert_der).as_ref().to_vec())
}

#[derive(Clone)]
pub struct MakeLedgerTls {
    config: Arc<ClientConfig>,
}

impl MakeLedgerTls {
    pub fn new(config: ClientConfig) -> Self {
        Self {
            config: Arc::new(config),
        }
    }
}

impl<S> MakeTlsConnect<S> for MakeLedgerTls
where
    S: AsyncRead + AsyncWrite + Unpin + Send + 'static,
{
    type Stream = LedgerTlsStream<S>;
    type TlsConnect = LedgerTlsConnect;
    type Error = rustls::pki_types::InvalidDnsNameError;

    fn make_tls_connect(&mut self, hostname: &str) -> Result<Self::TlsConnect, Self::Error> {
        ServerName::try_from(hostname).map(|name| LedgerTlsConnect {
            hostname: name.to_owned(),
            connector: TlsConnector::from(Arc::clone(&self.config)),
        })
    }
}

pub struct LedgerTlsConnect {
    hostname: ServerName<'static>,
    connector: TlsConnector,
}

impl<S> TlsConnect<S> for LedgerTlsConnect
where
    S: AsyncRead + AsyncWrite + Unpin + Send + 'static,
{
    type Stream = LedgerTlsStream<S>;
    type Error = io::Error;
    type Future = Pin<Box<dyn Future<Output = io::Result<LedgerTlsStream<S>>> + Send>>;

    fn connect(self, stream: S) -> Self::Future {
        let handshake = self.connector.connect(self.hostname, stream);
        Box::pin(async move { handshake.await.map(LedgerTlsStream) })
    }
}

pub struct LedgerTlsStream<S>(tokio_rustls::client::TlsStream<S>);

impl<S> TlsStream for LedgerTlsStream<S>
where
    S: AsyncRead + AsyncWrite + Unpin,
{
    fn channel_binding(&self) -> ChannelBinding {
        let (_, session) = self.0.get_ref();
        session
            .peer_certificates()
            .and_then(|certs| certs.first())
            .and_then(|leaf| tls_server_end_point(leaf.as_ref()))
            .map_or_else(ChannelBinding::none, ChannelBinding::tls_server_end_point)
    }
}

impl<S> AsyncRead for LedgerTlsStream<S>
where
    S: AsyncRead + AsyncWrite + Unpin,
{
    fn poll_read(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &mut ReadBuf<'_>,
    ) -> Poll<io::Result<()>> {
        Pin::new(&mut self.0).poll_read(cx, buf)
    }
}

impl<S> AsyncWrite for LedgerTlsStream<S>
where
    S: AsyncRead + AsyncWrite + Unpin,
{
    fn poll_write(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &[u8],
    ) -> Poll<io::Result<usize>> {
        Pin::new(&mut self.0).poll_write(cx, buf)
    }

    fn poll_flush(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<io::Result<()>> {
        Pin::new(&mut self.0).poll_flush(cx)
    }

    fn poll_shutdown(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<io::Result<()>> {
        Pin::new(&mut self.0).poll_shutdown(cx)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use x509_cert::der::oid::db::rfc8410::ID_ED_25519;
    use x509_cert::TbsCertificate;

    fn hex(bytes: &[u8]) -> String {
        bytes.iter().map(|b| format!("{b:02x}")).collect()
    }

    #[test]
    fn full_certificate_der_is_not_a_tbs_certificate() {
        let tbs = TbsCertificate::from_der(TEST_CERT);
        assert!(
            tbs.is_err(),
            "tokio-postgres-rustls 0.13.0 relies on this parse, which must fail on a full \
             Certificate DER: {tbs:?}"
        );
        let cert = Certificate::from_der(TEST_CERT).expect("full Certificate parses");
        assert_eq!(cert.signature_algorithm.oid, ECDSA_WITH_SHA_384);
        assert_eq!(cert.tbs_certificate.signature.oid, ECDSA_WITH_SHA_384);
    }

    #[test]
    fn ecdsa_sha384_cert_channel_binding_is_sha384_of_the_der() {
        let binding = tls_server_end_point(TEST_CERT).expect("binding for the test cert");
        assert_eq!(binding.len(), 48);
        assert_eq!(hex(&binding), TEST_CERT_SHA384);
    }

    #[test]
    fn garbage_or_truncated_der_has_no_channel_binding() {
        assert_eq!(tls_server_end_point(&[]), None);
        assert_eq!(tls_server_end_point(b"not a certificate"), None);
        assert_eq!(tls_server_end_point(&[0x30, 0x03, 0x02, 0x01]), None);
        let cert = Certificate::from_der(TEST_CERT).unwrap();
        let tbs_only = x509_cert::der::Encode::to_der(&cert.tbs_certificate).unwrap();
        assert!(TbsCertificate::from_der(&tbs_only).is_ok());
        assert_eq!(tls_server_end_point(&tbs_only), None);
        let truncated = &TEST_CERT[..TEST_CERT.len() - 1];
        assert_eq!(tls_server_end_point(truncated), None);
    }

    #[test]
    fn digest_table_follows_rfc_5929() {
        for (oid, expected_len) in [
            (ID_SHA_1, 32),
            (SHA_1_WITH_RSA_ENCRYPTION, 32),
            (ID_SHA_256, 32),
            (SHA_256_WITH_RSA_ENCRYPTION, 32),
            (ECDSA_WITH_SHA_256, 32),
            (ID_SHA_384, 48),
            (SHA_384_WITH_RSA_ENCRYPTION, 48),
            (ECDSA_WITH_SHA_384, 48),
            (ID_SHA_512, 64),
            (SHA_512_WITH_RSA_ENCRYPTION, 64),
        ] {
            let got = channel_binding_digest(oid).unwrap_or_else(|| panic!("{oid}"));
            assert_eq!(got.output_len(), expected_len, "{oid}");
        }
        assert!(channel_binding_digest(ID_ED_25519).is_none());
        let unknown = ObjectIdentifier::new("1.2.3.4.5").unwrap();
        assert!(channel_binding_digest(unknown).is_none());
    }

    const TEST_CERT: &[u8] = include_bytes!("../tests/fixtures/tls_test_ecdsa_p384_cert.der");
    const TEST_KEY_PKCS8: &[u8] =
        include_bytes!("../tests/fixtures/tls_test_ecdsa_p384_key.pkcs8.der");
    const TEST_CERT_SHA384: &str = "251a72557887f3b2a327b9a1aeebaa334037afb991b0714026dbecf0a7d89ac2867a1d0fd6fb045554c5b4958115f004";

    fn base64_decode(s: &str) -> Vec<u8> {
        const ALPHABET: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
        let mut out = Vec::new();
        let mut acc: u32 = 0;
        let mut bits = 0;
        for c in s.bytes().filter(|&c| c != b'=') {
            let v = ALPHABET
                .iter()
                .position(|&a| a == c)
                .expect("base64 alphabet") as u32;
            acc = (acc << 6) | v;
            bits += 6;
            if bits >= 8 {
                bits -= 8;
                out.push((acc >> bits) as u8);
                acc &= (1 << bits) - 1;
            }
        }
        out
    }

    async fn read_message<R: AsyncRead + Unpin>(r: &mut R) -> (u8, Vec<u8>) {
        use tokio::io::AsyncReadExt;
        let mut head = [0u8; 5];
        r.read_exact(&mut head).await.unwrap();
        let len = u32::from_be_bytes([head[1], head[2], head[3], head[4]]) as usize;
        let mut body = vec![0u8; len - 4];
        r.read_exact(&mut body).await.unwrap();
        (head[0], body)
    }

    fn cstr(b: &[u8]) -> (&str, &[u8]) {
        let nul = b.iter().position(|&c| c == 0).unwrap();
        (std::str::from_utf8(&b[..nul]).unwrap(), &b[nul + 1..])
    }

    #[tokio::test]
    async fn client_selects_scram_plus_and_binds_to_the_presented_certificate() {
        use rustls::pki_types::{CertificateDer, PrivateKeyDer, PrivatePkcs8KeyDer};
        use tokio::io::{AsyncReadExt, AsyncWriteExt};

        let provider = Arc::new(rustls::crypto::aws_lc_rs::default_provider());
        let server_config = rustls::ServerConfig::builder_with_provider(provider)
            .with_protocol_versions(&[&rustls::version::TLS13])
            .unwrap()
            .with_no_client_auth()
            .with_single_cert(
                vec![CertificateDer::from(TEST_CERT.to_vec())],
                PrivateKeyDer::Pkcs8(PrivatePkcs8KeyDer::from(TEST_KEY_PKCS8.to_vec())),
            )
            .expect("test cert and key");
        let acceptor = tokio_rustls::TlsAcceptor::from(Arc::new(server_config));

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        let server = tokio::spawn(async move {
            let (mut tcp, _) = listener.accept().await.unwrap();
            let mut ssl_request = [0u8; 8];
            tcp.read_exact(&mut ssl_request).await.unwrap();
            assert_eq!(ssl_request, [0, 0, 0, 8, 0x04, 0xd2, 0x16, 0x2f]);
            tcp.write_all(b"S").await.unwrap();
            let mut tls = acceptor.accept(tcp).await.expect("TLS handshake");
            let mut len = [0u8; 4];
            tls.read_exact(&mut len).await.unwrap();
            let mut startup = vec![0u8; u32::from_be_bytes(len) as usize - 4];
            tls.read_exact(&mut startup).await.unwrap();
            assert_eq!(&startup[..4], &[0, 3, 0, 0]);
            let mechanisms = b"SCRAM-SHA-256-PLUS\0SCRAM-SHA-256\0\0";
            let mut msg = vec![b'R'];
            msg.extend_from_slice(&((8 + mechanisms.len()) as u32).to_be_bytes());
            msg.extend_from_slice(&10u32.to_be_bytes());
            msg.extend_from_slice(mechanisms);
            tls.write_all(&msg).await.unwrap();
            let (tag, body) = read_message(&mut tls).await;
            assert_eq!(tag, b'p');
            let (mechanism, rest) = cstr(&body);
            let client_first = std::str::from_utf8(&rest[4..]).unwrap().to_owned();
            let client_nonce = client_first
                .strip_prefix("p=tls-server-end-point,,n=,r=")
                .unwrap_or_else(|| panic!("client-first: {client_first}"))
                .to_owned();
            let server_first = format!("r={client_nonce}SERVERNONCE,s=QSXCR+Q6sek8bf92,i=4096");
            let mut msg = vec![b'R'];
            msg.extend_from_slice(&((8 + server_first.len()) as u32).to_be_bytes());
            msg.extend_from_slice(&11u32.to_be_bytes());
            msg.extend_from_slice(server_first.as_bytes());
            tls.write_all(&msg).await.unwrap();
            let (tag, body) = read_message(&mut tls).await;
            assert_eq!(tag, b'p');
            let client_final = std::str::from_utf8(&body).unwrap().to_owned();
            let c = client_final
                .split(',')
                .find_map(|attr| attr.strip_prefix("c="))
                .unwrap_or_else(|| panic!("client-final: {client_final}"))
                .to_owned();
            let fields = b"SFATAL\0C28P01\0Mtest server stops here\0\0";
            let mut msg = vec![b'E'];
            msg.extend_from_slice(&((4 + fields.len()) as u32).to_be_bytes());
            msg.extend_from_slice(fields);
            let _ = tls.write_all(&msg).await;
            let _ = tls.shutdown().await;
            (mechanism.to_owned(), c)
        });

        let cfg = crate::pg::parse_dsn(
            &format!("postgresql://u:p@127.0.0.1:{port}/db"),
            "xai-abuse-ledger-service-test",
            std::time::Duration::from_millis(150),
        )
        .unwrap();
        assert_eq!(
            cfg.get_channel_binding(),
            tokio_postgres::config::ChannelBinding::Require
        );
        let err = tokio::time::timeout(
            std::time::Duration::from_secs(10),
            cfg.connect(crate::pg::make_tls()),
        )
        .await
        .expect("fake server answers")
        .err()
        .expect("the fake server ends with an ErrorResponse");
        assert_eq!(
            err.as_db_error().map(|db| db.code().code()),
            Some("28P01"),
            "{}",
            crate::pg::render_error_chain(&err)
        );

        let (mechanism, c) = server.await.unwrap();
        assert_eq!(mechanism, "SCRAM-SHA-256-PLUS");
        let cbind_input = base64_decode(&c);
        let (gs2, cbind_data) = cbind_input.split_at("p=tls-server-end-point,,".len());
        assert_eq!(gs2, b"p=tls-server-end-point,,");
        assert_eq!(hex(cbind_data), TEST_CERT_SHA384);
        assert_eq!(cbind_data, tls_server_end_point(TEST_CERT).unwrap());
    }
}
