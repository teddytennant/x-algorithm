use crate::params::LimitedActionType;
use anyhow::Context;
use serde::Deserialize;
use std::path::{Path, PathBuf};
use xai_stats_receiver::StatsReceiverExt;
use xai_stringcenter::{StringCenter, DEFAULT_LANGUAGE};
use xai_visibility_filtering::graphql_results::translate;

const BUNDLE_PATH: &str = "/config/stringcenter/visibility-ltd-actions.json";
const PINNED_METADATA_PATH: &str = "/config/stringcenter/_BUNDLE_METADATA.json";
const LIVE_METADATA_DIR: &str = "/usr/local/config/stringcenter/bundles/visibility-ltd-actions";

const KEYS_GAUGE: &str = "limited_actions_copy_keys";
const DRIFT_COUNTER: &str = "limited_actions_copy_drift_checks";

const GENERIC_SUBTEXT_NAMESPACES: [&str; 2] = ["BlockedViewer", "ReadonlyViewer"];
pub(crate) const LEARN_MORE_PLACEHOLDER: &str = "learnmore";

pub(crate) struct LimitedActionsCopy {
    bundle: StringCenter,
}

pub(crate) struct PromptCopy {
    namespace: String,
    prompt_type: PromptType,
    learn_more_url: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Prompt {
    pub(crate) headline: String,
    pub(crate) subtext: String,
    pub(crate) kind: PromptKind,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum PromptKind {
    Basic,
    SeeConversation,
    LearnMore {
        language: String,
        link_text: &'static str,
        url: String,
    },
}

#[derive(Clone, Copy)]
enum PromptType {
    Basic,
    Cta,
}

pub(crate) struct DriftCheck {
    pinned_hash: String,
    live_metadata: PathBuf,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, strum::IntoStaticStr)]
#[strum(serialize_all = "snake_case")]
pub(crate) enum Drift {
    Match,
    Mismatch,
    Unreadable,
}

#[derive(Deserialize)]
struct BundleMetadata {
    bundle_hash: String,
}

fn bundle_hash(path: &Path) -> anyhow::Result<String> {
    let bytes = std::fs::read(path).with_context(|| path.display().to_string())?;
    Ok(serde_json::from_slice::<BundleMetadata>(&bytes)?.bundle_hash)
}

impl LimitedActionsCopy {
    #[expect(
        clippy::expect_used,
        reason = "startup fail-fast: the image carries the bundle, so a load failure is a build bug"
    )]
    pub(crate) fn load_baked(
        datacenter: &str,
        stats: Option<&dyn StatsReceiverExt>,
    ) -> (Self, DriftCheck) {
        let (copy, drift) = Self::load(
            Path::new(BUNDLE_PATH),
            Path::new(PINNED_METADATA_PATH),
            Path::new(LIVE_METADATA_DIR)
                .join(datacenter)
                .join("_BUNDLE_METADATA.json"),
        )
        .expect("the image's limited-actions copy loads");
        if let Some(stats) = stats {
            stats.gauge(KEYS_GAUGE, &[], copy.bundle.len() as f64);
        }
        (copy, drift)
    }

    fn load(
        bundle: &Path,
        pinned_metadata: &Path,
        live_metadata: PathBuf,
    ) -> anyhow::Result<(Self, DriftCheck)> {
        let bundle =
            StringCenter::from_file(bundle).with_context(|| bundle.display().to_string())?;
        let drift = DriftCheck {
            pinned_hash: bundle_hash(pinned_metadata)?,
            live_metadata,
        };
        Ok((Self { bundle }, drift))
    }

    pub(crate) fn prompt(
        &self,
        action: LimitedActionType,
        copy: &PromptCopy,
        language: Option<&str>,
        country: Option<&str>,
    ) -> Option<Prompt> {
        let language = language.unwrap_or(DEFAULT_LANGUAGE);
        let namespace = copy.namespace.as_str();
        let action_key = string_center_key(action);
        let headline_key = format!("{namespace}_{action_key}_Headline");
        let subtext_key = if GENERIC_SUBTEXT_NAMESPACES.contains(&namespace) {
            format!("{namespace}_GenericSubtext")
        } else {
            format!("{namespace}_{action_key}_Subtext")
        };
        let headline = self.bundle.get(&headline_key, language, country)?;
        let subtext = self.bundle.get(&subtext_key, language, country)?;
        Some(match (copy.prompt_type, &copy.learn_more_url) {
            (PromptType::Cta, _) => Prompt {
                headline,
                subtext,
                kind: PromptKind::SeeConversation,
            },
            (PromptType::Basic, Some(url)) => Prompt {
                headline,
                subtext: format!("{subtext} {{{LEARN_MORE_PLACEHOLDER}}}"),
                kind: PromptKind::LearnMore {
                    language: language.to_string(),
                    link_text: translate("learn_more_link", language).unwrap_or("Learn more"),
                    url: url.clone(),
                },
            },
            (PromptType::Basic, None) => Prompt {
                headline,
                subtext,
                kind: PromptKind::Basic,
            },
        })
    }

    #[cfg(test)]
    pub(crate) fn from_json(json: &str) -> Self {
        Self {
            bundle: StringCenter::from_json(json.as_bytes()).unwrap(),
        }
    }
}

impl PromptCopy {
    pub(crate) fn from_switches(
        namespace: Option<&str>,
        prompt_type: Option<&str>,
        learn_more_url: Option<&str>,
    ) -> Option<Self> {
        if namespace == Some("NoCopy") {
            return None;
        }
        Some(Self {
            namespace: namespace.unwrap_or("Default").to_string(),
            prompt_type: if prompt_type == Some("cta") {
                PromptType::Cta
            } else {
                PromptType::Basic
            },
            learn_more_url: learn_more_url
                .filter(|url| !url.is_empty())
                .map(str::to_string),
        })
    }
}

impl DriftCheck {
    pub(crate) fn run(&self, stats: Option<&dyn StatsReceiverExt>) -> Drift {
        let drift = match bundle_hash(&self.live_metadata) {
            Ok(live) if live == self.pinned_hash => Drift::Match,
            Ok(_) => Drift::Mismatch,
            Err(_) => Drift::Unreadable,
        };
        if let Some(stats) = stats {
            stats.incr(DRIFT_COUNTER, &[("result", drift.into())], 1);
        }
        drift
    }
}

fn string_center_key(action: LimitedActionType) -> &'static str {
    use LimitedActionType as T;
    match action {
        T::Reply => "Reply",
        T::Retweet => "Retweet",
        T::QuoteTweet => "QuoteTweet",
        T::Like => "Like",
        T::React => "React",
        T::SendViaDm => "SendViaDm",
        T::AddToBookmarks => "AddToBookmarks",
        T::AddToMoment => "AddToMoment",
        T::PinToProfile => "PinToProfile",
        T::ViewTweetActivity => "ViewTweetActivity",
        T::ShareTweetVia => "ShareTweetVia",
        T::Follow => "Follow",
        T::ListsAddRemove => "ListsAddRemove",
        T::MuteConversation => "MuteConversation",
        T::Embed => "Embed",
        T::ViewHiddenReplies => "ViewHiddenReplies",
        T::HideCommunityTweet => "HideCommunityTweet",
        T::CopyLink => "CopyLink",
        T::VoteOnPoll => "VoteOnPoll",
        T::RemoveFromCommunity => "RemoveFromCommunity",
        T::ShowRetweetActionMenu => "ShowRetweetActionMenu",
        T::ReplyDownVote => "ReplyDownVote",
        T::Autoplay => "Autoplay",
        T::EditTweet => "EditTweet",
        T::Highlight => "Highlight",
        T::ViewPostEngagements => "ViewPostEngagements",
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    pub(crate) const BUNDLE: &str = r#"[
  {"string_key": "LimitedReplies_Reply_Headline", "current_variant": "Who can reply?",
   "instructions_v2": []},
  {"string_key": "LimitedReplies_Reply_Subtext", "current_variant": "Only some accounts can reply.",
   "instructions_v2": [{"type": "by_language", "by_language": {
     "ja": [{"type": "raw_string", "raw_string": "一部のアカウントのみが返信できます。"}]}}]},
  {"string_key": "BlockedViewer_Like_Headline", "current_variant": "Why can’t you like this?",
   "instructions_v2": [{"type": "by_language", "by_language": {
     "ja": [{"type": "raw_string", "raw_string": "いいねできない理由"}]}}]},
  {"string_key": "BlockedViewer_GenericSubtext",
   "current_variant": "This author has blocked you, so you can't perform this action",
   "instructions_v2": [{"type": "by_language", "by_language": {
     "ja": [{"type": "raw_string", "raw_string": "この作成者はあなたをブロックしているため、この操作を実行できません"}]}}]},
  {"string_key": "Default_Reply_Headline", "current_variant": "You cannot reply",
   "instructions_v2": [{"type": "by_language", "by_language": {
     "ja": [{"type": "raw_string", "raw_string": "返信できません"}]}}]},
  {"string_key": "Default_Reply_Subtext", "current_variant": "You cannot reply this post",
   "instructions_v2": [{"type": "by_language", "by_language": {
     "ja": [{"type": "raw_string", "raw_string": "このポストには返信できません"}]}}]}
]"#;

    #[test]
    fn prompts_follow_the_scala_key_scheme_and_prompt_types_in_en_and_ja() {
        use LimitedActionType as T;
        let copy = LimitedActionsCopy::from_json(BUNDLE);
        let prompt_copy = |namespace, prompt_type, url| {
            PromptCopy::from_switches(namespace, prompt_type, url).unwrap()
        };
        let limited_replies = prompt_copy(Some("LimitedReplies"), Some("cta"), Some(""));
        let blocked = prompt_copy(Some("BlockedViewer"), None, Some(""));
        let default = prompt_copy(None, Some("unknown"), None);
        let learn_more = prompt_copy(None, None, Some("https://example.com/rules"));
        let prompt = |headline: &str, subtext: &str, kind| {
            Some(Prompt {
                headline: headline.to_string(),
                subtext: subtext.to_string(),
                kind,
            })
        };
        let see_conversation =
            |headline, subtext| prompt(headline, subtext, PromptKind::SeeConversation);
        let basic = |headline, subtext| prompt(headline, subtext, PromptKind::Basic);
        let learn_more_link = |language: &str, link_text| PromptKind::LearnMore {
            language: language.to_string(),
            link_text,
            url: "https://example.com/rules".to_string(),
        };
        let cases = [
            (
                T::Reply,
                &limited_replies,
                Some("ja"),
                see_conversation("Who can reply?", "一部のアカウントのみが返信できます。"),
            ),
            (
                T::Reply,
                &limited_replies,
                None,
                see_conversation("Who can reply?", "Only some accounts can reply."),
            ),
            (
                T::Reply,
                &limited_replies,
                Some("JA-jp"),
                see_conversation("Who can reply?", "一部のアカウントのみが返信できます。"),
            ),
            (
                T::Like,
                &blocked,
                Some("ja"),
                basic(
                    "いいねできない理由",
                    "この作成者はあなたをブロックしているため、この操作を実行できません",
                ),
            ),
            (
                T::Like,
                &blocked,
                Some("en"),
                basic(
                    "Why can’t you like this?",
                    "This author has blocked you, so you can't perform this action",
                ),
            ),
            (
                T::Reply,
                &default,
                Some("ja"),
                basic("返信できません", "このポストには返信できません"),
            ),
            (
                T::Reply,
                &learn_more,
                Some("en"),
                prompt(
                    "You cannot reply",
                    "You cannot reply this post {learnmore}",
                    learn_more_link("en", "Learn more"),
                ),
            ),
            (
                T::Reply,
                &learn_more,
                Some("ja"),
                prompt(
                    "返信できません",
                    "このポストには返信できません {learnmore}",
                    learn_more_link("ja", "詳細はこちら"),
                ),
            ),
            (T::Reply, &blocked, Some("en"), None),
            (T::Like, &limited_replies, Some("en"), None),
        ];
        for (action, prompt_copy, language, expected) in cases {
            assert_eq!(
                copy.prompt(action, prompt_copy, language, None),
                expected,
                "{action:?} {} {language:?}",
                prompt_copy.namespace
            );
        }
        assert!(PromptCopy::from_switches(Some("NoCopy"), Some("cta"), None).is_none());
    }

    #[test]
    fn startup_loads_the_bundle_and_the_drift_check_compares_the_published_hash() {
        let dir = tempfile::tempdir().unwrap();
        let write = |name: &str, contents: &str| {
            let path = dir.path().join(name);
            std::fs::write(&path, contents).unwrap();
            path
        };
        let bundle = write("bundle.json", BUNDLE);
        let pinned = write("pinned.json", r#"{"bundle_hash": "aaaa"}"#);
        let live = dir.path().join("live.json");
        let (copy, drift) = LimitedActionsCopy::load(&bundle, &pinned, live.clone()).unwrap();
        assert_eq!(copy.bundle.len(), 6);
        assert_eq!(drift.run(None), Drift::Unreadable);
        write("live.json", r#"{"bundle_hash": "aaaa"}"#);
        assert_eq!(drift.run(None), Drift::Match);
        write("live.json", r#"{"bundle_hash": "bbbb"}"#);
        assert_eq!(drift.run(None), Drift::Mismatch);
        write("live.json", "{");
        assert_eq!(drift.run(None), Drift::Unreadable);

        let missing = dir.path().join("missing.json");
        assert!(LimitedActionsCopy::load(&missing, &pinned, live.clone()).is_err());
        assert!(LimitedActionsCopy::load(&bundle, &missing, live.clone()).is_err());
        let corrupt = write("corrupt.json", "[{");
        assert!(LimitedActionsCopy::load(&corrupt, &pinned, live).is_err());
    }

    #[test]
    fn the_image_layer_ships_the_paths_startup_loads() {
        let cargo = concat!(env!("CARGO_MANIFEST_DIR"), "/BUILD.bazel");
        let ws = "crates/x-product/xai-visibility-filtering-service/BUILD.bazel";
        let path = if Path::new(cargo).exists() { cargo } else { ws };
        let build = std::fs::read_to_string(path).unwrap_or_else(|e| panic!("read {path}: {e}"));
        for loaded in [BUNDLE_PATH, PINNED_METADATA_PATH] {
            assert!(build.contains(&format!("\"{loaded}\": ")), "{loaded}");
        }
    }
}
