// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 X.AI Corp.
use axum::http;
use axum::response::IntoResponse;
use bytes::Bytes;
use http_body::{Body, Frame, SizeHint};
use lazy_static::lazy_static;
use prometheus::{HistogramVec, register_histogram_vec};
use std::convert::Infallible;
use std::future::Future;
use std::pin::Pin;
use std::task::{Context, Poll};
use std::time::Instant;
use tonic::server::NamedService;
use tower::Service;

use crate::request_metrics::{grpc_status_label, latency_buckets_ms};

lazy_static! {
    static ref GRPC_SERVER_LATENCY_MS: HistogramVec = register_histogram_vec!(
        "recsys_engine_grpc_server_latency_ms",
        "Full server-side gRPC latency in ms, from request arrival at the \
         transport to the last response frame handed to it, by RPC method and \
         terminal grpc-status ('cancelled' when the client went away before \
         the response finished, 'unknown' when no status was observed).",
        &["method", "grpc_status"],
        latency_buckets_ms()
    )
    .unwrap();
}

fn method_label(path: &str) -> &str {
    path.rsplit('/')
        .next()
        .filter(|s| !s.is_empty())
        .unwrap_or("unknown")
}

fn status_label_from_header(value: &http::HeaderValue) -> &'static str {
    let code = value
        .to_str()
        .ok()
        .and_then(|s| s.trim().parse::<i32>().ok())
        .map(tonic::Code::from_i32)
        .unwrap_or(tonic::Code::Unknown);
    grpc_status_label(code)
}

struct TimedBody {
    inner: axum::body::Body,
    start: Instant,
    method: String,
    status: Option<&'static str>,
    observed: bool,
}

impl TimedBody {
    fn finish(&mut self, status: &'static str) {
        if self.observed {
            return;
        }
        self.observed = true;
        GRPC_SERVER_LATENCY_MS
            .with_label_values(&[self.method.as_str(), status])
            .observe(self.start.elapsed().as_secs_f64() * 1000.0);
    }
}

impl Body for TimedBody {
    type Data = Bytes;
    type Error = axum::Error;

    fn poll_frame(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
    ) -> Poll<Option<Result<Frame<Self::Data>, Self::Error>>> {
        let this = &mut *self;
        let polled = Pin::new(&mut this.inner).poll_frame(cx);
        match &polled {
            Poll::Ready(Some(Ok(frame))) => {
                if let Some(trailers) = frame.trailers_ref() {
                    if let Some(v) = trailers.get("grpc-status") {
                        this.status = Some(status_label_from_header(v));
                    }
                    let status = this.status.unwrap_or("unknown");
                    this.finish(status);
                } else if this.inner.is_end_stream() {
                    let status = this.status.unwrap_or("unknown");
                    this.finish(status);
                }
            }
            Poll::Ready(None) => {
                let status = this.status.unwrap_or("unknown");
                this.finish(status);
            }
            Poll::Ready(Some(Err(_))) => this.finish("internal"),
            Poll::Pending => {}
        }
        polled
    }

    fn is_end_stream(&self) -> bool {
        self.inner.is_end_stream()
    }

    fn size_hint(&self) -> SizeHint {
        self.inner.size_hint()
    }
}

impl Drop for TimedBody {
    fn drop(&mut self) {
        self.finish("cancelled");
    }
}

#[derive(Clone)]
pub struct GrpcServerLatencyService<S> {
    inner: S,
}

impl<S> GrpcServerLatencyService<S> {
    pub fn new(inner: S) -> Self {
        Self { inner }
    }
}

impl<S: NamedService> NamedService for GrpcServerLatencyService<S> {
    const NAME: &'static str = S::NAME;
}

impl<S, ReqBody> Service<http::Request<ReqBody>> for GrpcServerLatencyService<S>
where
    S: Service<http::Request<ReqBody>, Error = Infallible> + Clone + Send + 'static,
    S::Response: IntoResponse,
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
        let start = Instant::now();
        let method = method_label(req.uri().path()).to_owned();
        let mut inner = self.inner.clone();
        Box::pin(async move {
            let response = inner.call(req).await?.into_response();
            let (parts, body) = response.into_parts();
            let status = parts
                .headers
                .get("grpc-status")
                .map(status_label_from_header);
            let mut timed = TimedBody {
                inner: body,
                start,
                method,
                status,
                observed: false,
            };
            if timed.inner.is_end_stream() {
                let status = timed.status.unwrap_or("unknown");
                timed.finish(status);
            }
            Ok(http::Response::from_parts(
                parts,
                axum::body::Body::new(timed),
            ))
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use http_body_util::BodyExt;
    use std::collections::VecDeque;

    struct ScriptedBody {
        frames: VecDeque<Frame<Bytes>>,
    }

    impl Body for ScriptedBody {
        type Data = Bytes;
        type Error = axum::Error;

        fn poll_frame(
            mut self: Pin<&mut Self>,
            _cx: &mut Context<'_>,
        ) -> Poll<Option<Result<Frame<Bytes>, Self::Error>>> {
            Poll::Ready(self.frames.pop_front().map(Ok))
        }

        fn is_end_stream(&self) -> bool {
            self.frames.is_empty()
        }
    }

    fn trailers(status: &str) -> Frame<Bytes> {
        let mut map = http::HeaderMap::new();
        map.insert("grpc-status", http::HeaderValue::from_str(status).unwrap());
        Frame::trailers(map)
    }

    #[derive(Clone)]
    struct Fixed {
        make: fn() -> http::Response<axum::body::Body>,
    }

    impl Service<http::Request<axum::body::Body>> for Fixed {
        type Response = http::Response<axum::body::Body>;
        type Error = Infallible;
        type Future = Pin<Box<dyn Future<Output = Result<Self::Response, Infallible>> + Send>>;

        fn poll_ready(&mut self, _cx: &mut Context<'_>) -> Poll<Result<(), Infallible>> {
            Poll::Ready(Ok(()))
        }

        fn call(&mut self, _req: http::Request<axum::body::Body>) -> Self::Future {
            let make = self.make;
            Box::pin(async move { Ok(make()) })
        }
    }

    fn request(method: &str) -> http::Request<axum::body::Body> {
        http::Request::builder()
            .uri(format!("/xai_recsys.RecsysPredictor/{method}"))
            .body(axum::body::Body::empty())
            .unwrap()
    }

    fn samples(method: &str, status: &str) -> u64 {
        GRPC_SERVER_LATENCY_MS
            .with_label_values(&[method, status])
            .get_sample_count()
    }

    #[test]
    fn method_label_strips_service_prefix() {
        assert_eq!(
            method_label("/xai_recsys.RecsysPredictor/PredictNextActions"),
            "PredictNextActions"
        );
        assert_eq!(method_label("/"), "unknown");
    }

    #[test]
    fn status_label_reads_tonic_codes_and_tolerates_garbage() {
        let v = |s: &str| http::HeaderValue::from_str(s).unwrap();
        assert_eq!(status_label_from_header(&v("0")), "ok");
        assert_eq!(status_label_from_header(&v("14")), "unavailable");
        assert_eq!(status_label_from_header(&v("nope")), "unknown");
    }

    #[tokio::test]
    async fn observes_once_with_the_trailer_status_when_the_body_drains() {
        let mut svc = GrpcServerLatencyService::new(Fixed {
            make: || {
                let body = ScriptedBody {
                    frames: VecDeque::from([
                        Frame::data(Bytes::from_static(b"abc")),
                        trailers("0"),
                    ]),
                };
                http::Response::new(axum::body::Body::new(body))
            },
        });
        let before = samples("DrainsOk", "ok");
        let resp = svc.call(request("DrainsOk")).await.unwrap();
        assert_eq!(samples("DrainsOk", "ok"), before);
        let collected = resp.into_body().collect().await.unwrap();
        assert_eq!(collected.to_bytes(), Bytes::from_static(b"abc"));
        assert_eq!(samples("DrainsOk", "ok"), before + 1);
        assert_eq!(samples("DrainsOk", "cancelled"), 0);
    }

    #[tokio::test]
    async fn trailers_only_response_is_observed_even_though_hyper_never_polls_it() {
        let mut svc = GrpcServerLatencyService::new(Fixed {
            make: || {
                http::Response::builder()
                    .header("grpc-status", "14")
                    .body(axum::body::Body::empty())
                    .unwrap()
            },
        });
        let before_ok = samples("Rejected", "unavailable");
        let before_cancelled = samples("Rejected", "cancelled");
        let resp = svc.call(request("Rejected")).await.unwrap();
        assert_eq!(samples("Rejected", "unavailable"), before_ok + 1);
        drop(resp);
        assert_eq!(samples("Rejected", "unavailable"), before_ok + 1);
        assert_eq!(
            samples("Rejected", "cancelled"),
            before_cancelled,
            "an unpolled finished body must not be mistaken for a cancel"
        );
    }

    #[tokio::test]
    async fn an_empty_success_body_is_not_recorded_as_cancelled_either() {
        let mut svc = GrpcServerLatencyService::new(Fixed {
            make: || {
                http::Response::builder()
                    .header("grpc-status", "0")
                    .body(axum::body::Body::empty())
                    .unwrap()
            },
        });
        let before = samples("EmptyOk", "ok");
        let resp = svc.call(request("EmptyOk")).await.unwrap();
        drop(resp);
        assert_eq!(samples("EmptyOk", "ok"), before + 1);
        assert_eq!(samples("EmptyOk", "cancelled"), 0);
    }

    #[tokio::test]
    async fn dropping_an_unfinished_body_records_it_as_cancelled() {
        let mut svc = GrpcServerLatencyService::new(Fixed {
            make: || {
                let body = ScriptedBody {
                    frames: VecDeque::from([
                        Frame::data(Bytes::from_static(b"part")),
                        Frame::data(Bytes::from_static(b"ial")),
                        trailers("0"),
                    ]),
                };
                http::Response::new(axum::body::Body::new(body))
            },
        });
        let before_cancelled = samples("GaveUp", "cancelled");
        let before_ok = samples("GaveUp", "ok");
        let resp = svc.call(request("GaveUp")).await.unwrap();
        let mut body = resp.into_body();
        let first = body.frame().await.unwrap().unwrap();
        assert_eq!(first.into_data().unwrap(), Bytes::from_static(b"part"));
        drop(body);
        assert_eq!(samples("GaveUp", "cancelled"), before_cancelled + 1);
        assert_eq!(samples("GaveUp", "ok"), before_ok);
    }

    #[tokio::test]
    async fn a_body_ending_without_any_status_reads_unknown() {
        let mut svc = GrpcServerLatencyService::new(Fixed {
            make: || {
                let body = ScriptedBody {
                    frames: VecDeque::from([Frame::data(Bytes::from_static(b"x"))]),
                };
                http::Response::new(axum::body::Body::new(body))
            },
        });
        let before = samples("NoStatus", "unknown");
        let resp = svc.call(request("NoStatus")).await.unwrap();
        resp.into_body().collect().await.unwrap();
        assert_eq!(samples("NoStatus", "unknown"), before + 1);
    }
}
