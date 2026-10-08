// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 X.AI Corp.
use axum::http;
use axum::response::IntoResponse;
use bytes::{BufMut, Bytes, BytesMut};
use http_body_util::{BodyExt, Full};
use lazy_static::lazy_static;
use prometheus::{HistogramVec, exponential_buckets, register_histogram_vec};
use std::convert::Infallible;
use std::future::Future;
use std::pin::Pin;
use std::task::{Context, Poll};
use std::time::Instant;
use tonic::server::NamedService;
use tower::Service;

const GRPC_FRAME_HEADER_SIZE: usize = 5;

lazy_static! {
    static ref RESPONSE_COMPRESSED_BYTES: HistogramVec = register_histogram_vec!(
        "recsys_engine_response_compressed_bytes",
        "On-wire gRPC response frame size in bytes after zstd compression, by RPC \
         method path. Recorded when GrpcCompressionService compresses the response.",
        &["method"],
        exponential_buckets(512.0, 2.0, 15).unwrap()
    )
    .unwrap();
    static ref RESPONSE_UNCOMPRESSED_BYTES: HistogramVec = register_histogram_vec!(
        "recsys_engine_response_uncompressed_bytes",
        "gRPC response frame size in bytes before zstd compression, by RPC \
         method. Recorded when GrpcCompressionService attempts to compress the \
         response.",
        &["method"],
        exponential_buckets(512.0, 2.0, 15).unwrap()
    )
    .unwrap();
    static ref RESPONSE_COMPRESS_MS: HistogramVec = register_histogram_vec!(
        "recsys_engine_response_compress_ms",
        "Outbound zstd time in ms (spawn_blocking queue + encode), by method.",
        &["method"],
        crate::request_metrics::latency_buckets_ms()
    )
    .unwrap();
}

fn method_label(path: &str) -> &str {
    path.rsplit('/')
        .next()
        .filter(|s| !s.is_empty())
        .unwrap_or("unknown")
}

#[derive(Clone)]
pub struct GrpcCompressionService<S> {
    inner: S,
}

impl<S> GrpcCompressionService<S> {
    pub fn new(inner: S) -> Self {
        Self { inner }
    }
}

impl<S: NamedService> NamedService for GrpcCompressionService<S> {
    const NAME: &'static str = S::NAME;
}

impl<S, ReqBody> Service<http::Request<ReqBody>> for GrpcCompressionService<S>
where
    S: Service<http::Request<ReqBody>, Error = Infallible> + Clone + Send + 'static,
    S::Response: axum::response::IntoResponse,
    S::Future: Send + 'static,
    ReqBody: Send + 'static,
{
    type Response = http::Response<axum::body::Body>;
    type Error = Infallible;
    type Future = Pin<Box<dyn Future<Output = Result<Self::Response, Self::Error>> + Send>>;

    fn poll_ready(&mut self, cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
        self.inner.poll_ready(cx)
    }

    fn call(&mut self, req: http::Request<ReqBody>) -> Self::Future {
        let method = method_label(req.uri().path()).to_owned();
        let client_accepts_zstd = req
            .headers()
            .get("grpc-accept-encoding")
            .and_then(|v| v.to_str().ok())
            .is_some_and(|v| v.split(',').any(|e| e.trim() == "zstd"));

        let mut inner = self.inner.clone();
        Box::pin(async move {
            let response = inner.call(req).await?;
            let response = response.into_response();

            if !client_accepts_zstd {
                return Ok(response);
            }

            let (mut parts, body) = response.into_parts();

            let collected = match body.collect().await {
                Ok(c) => c,
                Err(e) => {
                    log::warn!("gRPC compression: body collect failed: {e}");
                    parts.status = http::StatusCode::OK;
                    parts
                        .headers
                        .insert("grpc-status", http::HeaderValue::from_static("13"));
                    parts.headers.insert(
                        "grpc-message",
                        http::HeaderValue::from_static("response body collect failed"),
                    );
                    return Ok(http::Response::from_parts(parts, axum::body::Body::empty()));
                }
            };
            let trailers = collected.trailers().cloned();
            let data = collected.to_bytes();

            if data.len() <= GRPC_FRAME_HEADER_SIZE {
                return Ok(rebuild_response(parts, data, trailers));
            }

            RESPONSE_UNCOMPRESSED_BYTES
                .with_label_values(&[&method])
                .observe(data.len() as f64);
            let original = data.clone();
            let started = Instant::now();
            let result = tokio::task::spawn_blocking(move || compress_grpc_frame(&data)).await;
            RESPONSE_COMPRESS_MS
                .with_label_values(&[&method])
                .observe(started.elapsed().as_secs_f64() * 1000.0);

            match result {
                Ok(compressed) if compressed[0] == 1 => {
                    RESPONSE_COMPRESSED_BYTES
                        .with_label_values(&[&method])
                        .observe(compressed.len() as f64);
                    parts
                        .headers
                        .insert("grpc-encoding", http::HeaderValue::from_static("zstd"));
                    Ok(rebuild_response(parts, compressed, trailers))
                }
                _ => Ok(rebuild_response(parts, original, trailers)),
            }
        })
    }
}

fn rebuild_response(
    parts: http::response::Parts,
    data: Bytes,
    trailers: Option<http::HeaderMap>,
) -> http::Response<axum::body::Body> {
    let full = Full::new(data);
    let body = match trailers {
        Some(tr) => {
            axum::body::Body::new(full.with_trailers(async move { Some(Ok::<_, Infallible>(tr)) }))
        }
        None => axum::body::Body::new(full),
    };
    http::Response::from_parts(parts, body)
}

fn compress_grpc_frame(data: &Bytes) -> Bytes {
    if data.len() < GRPC_FRAME_HEADER_SIZE {
        return data.clone();
    }

    if data[0] != 0 {
        return data.clone();
    }

    let payload_len = u32::from_be_bytes([data[1], data[2], data[3], data[4]]) as usize;
    if data.len() < GRPC_FRAME_HEADER_SIZE + payload_len {
        return data.clone();
    }

    let payload = &data[GRPC_FRAME_HEADER_SIZE..GRPC_FRAME_HEADER_SIZE + payload_len];

    let compressed = match zstd::encode_all(std::io::Cursor::new(payload), 3) {
        Ok(c) => c,
        Err(_) => return data.clone(),
    };

    let mut out = BytesMut::with_capacity(GRPC_FRAME_HEADER_SIZE + compressed.len());
    out.put_u8(1);
    out.put_u32(compressed.len() as u32);
    out.put_slice(&compressed);

    let rest = GRPC_FRAME_HEADER_SIZE + payload_len;
    if rest < data.len() {
        out.put_slice(&data[rest..]);
    }

    out.freeze()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn method_label_strips_service_prefix() {
        assert_eq!(
            method_label("/xai_recsys.RecsysPredictor/PredictNextActions"),
            "PredictNextActions"
        );
        assert_eq!(method_label("/foo"), "foo");
        assert_eq!(method_label("/"), "unknown");
    }

    #[test]
    fn compress_sets_flag_and_shrinks_payload() {
        let payload = vec![0u8; 4096];
        let mut frame = BytesMut::with_capacity(GRPC_FRAME_HEADER_SIZE + payload.len());
        frame.put_u8(0);
        frame.put_u32(payload.len() as u32);
        frame.put_slice(&payload);
        let frame = frame.freeze();

        let out = compress_grpc_frame(&frame);
        assert_eq!(out[0], 1);
        let out_len = u32::from_be_bytes([out[1], out[2], out[3], out[4]]) as usize;
        assert!(out_len < payload.len());
        assert_eq!(out.len(), GRPC_FRAME_HEADER_SIZE + out_len);
    }

    #[test]
    fn compress_ms_accepts_method_label() {
        RESPONSE_COMPRESS_MS
            .with_label_values(&["PredictNextActions"])
            .observe(1.0);
    }

    #[derive(Clone)]
    struct FakeSvc {
        payload: Vec<u8>,
    }

    impl NamedService for FakeSvc {
        const NAME: &'static str = "test";
    }

    impl tower::Service<http::Request<()>> for FakeSvc {
        type Response = http::Response<axum::body::Body>;
        type Error = Infallible;
        type Future = std::future::Ready<Result<Self::Response, Self::Error>>;

        fn poll_ready(&mut self, _: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
            Poll::Ready(Ok(()))
        }

        fn call(&mut self, _: http::Request<()>) -> Self::Future {
            let mut frame = BytesMut::with_capacity(GRPC_FRAME_HEADER_SIZE + self.payload.len());
            frame.put_u8(0);
            frame.put_u32(self.payload.len() as u32);
            frame.put_slice(&self.payload);
            let full = Full::new(frame.freeze());
            let body = axum::body::Body::new(full.with_trailers(async {
                let mut t = http::HeaderMap::new();
                t.insert("grpc-status", http::HeaderValue::from_static("0"));
                Some(Ok::<_, Infallible>(t))
            }));
            std::future::ready(Ok(http::Response::new(body)))
        }
    }

    async fn call_zstd(payload: Vec<u8>) -> http::Response<axum::body::Body> {
        call_zstd_method(payload, "PredictNextActions").await
    }

    async fn call_zstd_method(payload: Vec<u8>, method: &str) -> http::Response<axum::body::Body> {
        let mut svc = GrpcCompressionService::new(FakeSvc { payload });
        let req = http::Request::builder()
            .uri(format!("/xai_recsys.RecsysPredictor/{method}"))
            .header("grpc-accept-encoding", "zstd")
            .body(())
            .unwrap();
        svc.call(req).await.unwrap()
    }

    #[tokio::test]
    async fn rebuild_keeps_grpc_status_trailers_when_compressed() {
        let resp = call_zstd(vec![0u8; 4096]).await;
        assert_eq!(
            resp.headers().get("grpc-encoding").map(|v| v.as_bytes()),
            Some(&b"zstd"[..])
        );
        let collected = resp.into_body().collect().await.unwrap();
        let trailers = collected.trailers().expect("grpc-status trailers");
        assert_eq!(
            trailers.get("grpc-status").map(|v| v.as_bytes()),
            Some(&b"0"[..])
        );
        assert_eq!(collected.to_bytes()[0], 1);
    }

    #[tokio::test]
    async fn rebuild_keeps_grpc_status_trailers_when_too_small_to_compress() {
        let resp = call_zstd(vec![]).await;
        let collected = resp.into_body().collect().await.unwrap();
        let trailers = collected.trailers().expect("grpc-status trailers");
        assert_eq!(
            trailers.get("grpc-status").map(|v| v.as_bytes()),
            Some(&b"0"[..])
        );
    }

    #[tokio::test]
    async fn records_frame_size_before_and_after_zstd() {
        let method = "SizeBeforeAfter";
        let payload = vec![0u8; 4096];
        let frame_len = (GRPC_FRAME_HEADER_SIZE + payload.len()) as f64;
        let resp = call_zstd_method(payload, method).await;
        let wire = resp.into_body().collect().await.unwrap().to_bytes();

        let before = RESPONSE_UNCOMPRESSED_BYTES.with_label_values(&[method]);
        assert_eq!(before.get_sample_count(), 1);
        assert_eq!(before.get_sample_sum(), frame_len);

        let after = RESPONSE_COMPRESSED_BYTES.with_label_values(&[method]);
        assert_eq!(after.get_sample_count(), 1);
        assert_eq!(after.get_sample_sum(), wire.len() as f64);
        assert!(after.get_sample_sum() < before.get_sample_sum());
    }

    #[tokio::test]
    async fn too_small_to_compress_records_no_sizes() {
        let method = "SizeTooSmall";
        let resp = call_zstd_method(vec![], method).await;
        resp.into_body().collect().await.unwrap();
        assert_eq!(
            RESPONSE_UNCOMPRESSED_BYTES
                .with_label_values(&[method])
                .get_sample_count(),
            0
        );
    }
}
