use super::lookup::LookupError;
use xai_visibility_filtering_proto as vf_pb;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, strum::IntoStaticStr)]
#[strum(serialize_all = "snake_case")]
pub(crate) enum LabelSource {
    Twemcache,
    Manhattan,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, strum::IntoStaticStr)]
#[strum(serialize_all = "snake_case")]
pub(crate) enum FallbackReason {
    Timeout,
    Backpressure,
    Decode,
    MissingResponse,
    Other,
}

#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, enum_map::Enum, strum::IntoStaticStr,
)]
#[strum(serialize_all = "snake_case")]
pub(crate) enum FailureKind {
    ManhattanFetch,
    ManhattanDecode,
}

#[derive(Debug, PartialEq)]
pub(crate) enum TwemcacheOutcome {
    Hit(vf_pb::SafetyLabelMap),
    NotFound,
    Miss,
    FallThrough(FallbackReason),
}

#[derive(Debug)]
pub(crate) enum ManhattanOutcome {
    Resolved(vf_pb::SafetyLabelMap),
    Failure(LookupError),
}
