use crate::hydration::{Hydrator, Hydrators};
use crate::models::{
    tweet_timestamp_ms, ArticleLifecycle, AuthorFeatures, AuthorLabelSet, ClientCapability,
    CommunityModeration, HydratedTweetCandidate, SafetyLabelMap, TweetFeatures, Viewer,
    ViewerFeatures, ViewerProfile,
};
use crate::params::{CountryList, CountryLists};
use xai_core_entities::entities::ConversationControl;

pub(crate) struct RuleContext<'a> {
    facts: CoreFacts<'a>,
    #[cfg(test)]
    hydrated: Hydrators,
}

#[derive(Clone, Copy)]
pub(super) struct CoreFacts<'a> {
    viewer: &'a ViewerFeatures,
    candidate: &'a HydratedTweetCandidate,
    country_lists: &'a CountryLists,
}

impl<'a> RuleContext<'a> {
    pub(super) fn new(
        viewer: &'a ViewerFeatures,
        candidate: &'a HydratedTweetCandidate,
        country_lists: &'a CountryLists,
    ) -> Self {
        Self {
            facts: CoreFacts {
                viewer,
                candidate,
                country_lists,
            },
            #[cfg(test)]
            hydrated: Hydrators::all(),
        }
    }

    #[cfg(test)]
    pub(super) fn hydrated_by(self, hydrated: Hydrators) -> Self {
        Self { hydrated, ..self }
    }

    #[inline]
    fn reads(&self, hydrator: Hydrator) -> &'a HydratedTweetCandidate {
        #[cfg(test)]
        assert!(
            self.hydrated.contains(hydrator),
            "a rule reads {hydrator:?}, which its policy does not derive"
        );
        #[cfg(not(test))]
        let _ = hydrator;
        self.facts.candidate
    }

    #[inline]
    pub(super) fn facts(&self) -> CoreFacts<'a> {
        self.facts
    }

    #[inline]
    pub(super) fn failed(&self) -> Hydrators {
        self.facts.candidate.failed
    }

    #[inline]
    pub(super) fn source_tweet_id(&self) -> Option<u64> {
        self.reads(Hydrator::PureCore).source_tweet_id
    }

    #[inline]
    pub(super) fn tweet_features(&self) -> &'a TweetFeatures {
        &self.reads(Hydrator::Tweet).tweet_features
    }

    #[inline]
    pub(super) fn tweet_safety_labels(&self) -> &'a SafetyLabelMap {
        &self.reads(Hydrator::TweetSafetyLabels).safety_labels
    }

    #[inline]
    pub(super) fn author_features(&self) -> &'a AuthorFeatures {
        &self.reads(Hydrator::AuthorSafety).author_features
    }

    #[inline]
    pub(super) fn author_labels(&self) -> AuthorLabelSet {
        self.reads(Hydrator::AuthorLabels).author_labels
    }

    #[inline]
    pub(super) fn edge(&self, node: Hydrator) -> bool {
        debug_assert!(node.is_edge(), "{node:?} is not an edge node");
        self.reads(node).edges.contains(node)
    }

    #[inline]
    pub(super) fn conversation_control(&self) -> Option<&'a ConversationControl> {
        let features = self
            .reads(Hydrator::ConversationControl)
            .conversation_control
            .as_ref();
        features.map(|features| &features.control)
    }

    #[inline]
    pub(super) fn viewer_country(&self) -> Option<&'a str> {
        let features = self
            .reads(Hydrator::ViewerCountry)
            .conversation_control
            .as_ref();
        features.and_then(|features| features.viewer_country.as_deref())
    }

    #[inline]
    pub(super) fn community_moderation(&self) -> CommunityModeration {
        self.reads(Hydrator::CommunityModeration)
            .community_moderation
    }

    #[inline]
    pub(super) fn viewer_is_community_moderator(&self) -> Option<bool> {
        self.reads(Hydrator::CommunityModerator)
            .viewer_is_community_moderator
    }

    #[inline]
    pub(super) fn viewer_is_removed_from_community(&self) -> bool {
        self.reads(Hydrator::CommunityViewerRemoved)
            .viewer_is_removed_from_community
    }

    #[inline]
    pub(super) fn article_lifecycle(&self) -> Option<ArticleLifecycle> {
        self.reads(Hydrator::ArticleLifecycle).article_lifecycle
    }

    #[inline]
    pub(super) fn viewer_has_age_verified_18_label(&self) -> bool {
        self.reads(Hydrator::ViewerLabels);
        match &self.facts.viewer.viewer {
            Viewer::LoggedIn {
                has_age_verified_18_label,
                ..
            } => *has_age_verified_18_label,
            Viewer::LoggedOut => false,
        }
    }

    #[inline]
    pub(super) fn viewer_profile(&self) -> Option<&'a ViewerProfile> {
        self.reads(Hydrator::ViewerProfile);
        match &self.facts.viewer.viewer {
            Viewer::LoggedIn { profile, .. } => Some(profile),
            Viewer::LoggedOut => None,
        }
    }
}

impl<'a> CoreFacts<'a> {
    #[inline]
    pub(super) fn tweet_id(self) -> u64 {
        self.candidate.tweet_id
    }

    #[inline]
    pub(super) fn viewer_id(self) -> Option<u64> {
        self.viewer.viewer.user_id()
    }

    #[inline]
    pub(super) fn request_country(self) -> Option<&'a str> {
        self.viewer.country_code.as_deref()
    }

    #[inline]
    pub(super) fn client_capability(self) -> ClientCapability {
        self.viewer.client_capability
    }

    #[inline]
    pub(super) fn is_author_viewer(self) -> bool {
        self.viewer_id() == Some(self.candidate.author_id)
    }

    #[inline]
    pub(super) fn created_after(self, unix_ms: u64) -> bool {
        tweet_timestamp_ms(self.candidate.tweet_id) > unix_ms
    }

    #[inline]
    pub(super) fn in_country_list(self, list: CountryList, country_code: &str) -> bool {
        self.country_lists.contains(list, country_code)
    }
}
