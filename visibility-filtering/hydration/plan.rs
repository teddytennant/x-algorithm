use crate::clients::socialgraph_client::{EdgeDirection, Graph};
use crate::hydration::{Hydrator, Hydrators};
use crate::rules::SafetyLevel;
use std::fmt;
use strum::VariantArray;
use xai_core_entities::entities::ConversationControlArm;
use xai_core_entities::gizmoduck_client::QueryFields;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, VariantArray)]
pub(crate) enum Source {
    TesPureCore,
    TesTweet,
    TesConversationControl,
    SafetyLabels,
    GizmoduckViewer,
    GizmoduckAuthor,
    Flock,
    ViewerCountry,
    Wingman,
    CommunityModeration,
    CommunityModerator,
    CommunityViewerRemoved,
    ArticleLifecycle,
    TrustedFriends,
    UserLocation,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum MissPolicy {
        Unresolves(Subject),
        FailsNode,
        ReadsNoEdge,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Subject {
    Tweet,
    Author,
}

impl Source {
    pub(super) const fn miss_policy(self) -> MissPolicy {
        match self {
            Source::TesPureCore | Source::TesTweet => MissPolicy::Unresolves(Subject::Tweet),
            Source::GizmoduckAuthor => MissPolicy::Unresolves(Subject::Author),
            Source::TesConversationControl
            | Source::SafetyLabels
            | Source::GizmoduckViewer
            | Source::Flock
            | Source::ViewerCountry
            | Source::CommunityModeration
            | Source::CommunityModerator
            | Source::CommunityViewerRemoved
            | Source::ArticleLifecycle
            | Source::UserLocation => MissPolicy::FailsNode,
            Source::Wingman | Source::TrustedFriends => MissPolicy::ReadsNoEdge,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Part {
    Column,
    Fields(&'static [QueryFields]),
    Edge(Edge),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, VariantArray)]
#[repr(u8)]
pub(super) enum Edge {
    Follows,
    Blocks,
    Mutes,
    MuteRetweets,
    BlockedBy,
    SuperFollows,
    FollowedBy,
    SecondDegree,
    TrustedFriends,
    OutsidePlace,
}

impl Edge {
    pub(super) const fn flock(self) -> Option<(Graph, EdgeDirection)> {
        use EdgeDirection::{Forward, Reverse};
        match self {
            Edge::Follows => Some((Graph::Follows, Forward)),
            Edge::Blocks => Some((Graph::Blocks, Forward)),
            Edge::Mutes => Some((Graph::Mutes, Forward)),
            Edge::MuteRetweets => Some((Graph::MuteRetweets, Forward)),
            Edge::BlockedBy => Some((Graph::Blocks, Reverse)),
            Edge::SuperFollows => Some((Graph::SuperFollows, Forward)),
            Edge::FollowedBy => Some((Graph::Follows, Reverse)),
            Edge::SecondDegree | Edge::TrustedFriends | Edge::OutsidePlace => None,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum KeyOrigin {
    RequestTweets,
    Viewer,
    PureCoreAuthor,
    PureCoreRetweeter,
    PureCoreReplyRoot,
    ExclusiveConversationAuthor,
    TweetArticle,
    ConversationRoot(&'static [ConversationControlArm]),
    ViewerForCoAllowedList,
    MyNetworkRootNotFollowingViewer,
    CommunityPost,
    ModeratedCommunity,
    TweetCommunity,
    TrustedFriendsList,
    NarrowcastPlace,
}

#[derive(Clone, Copy, Debug)]
pub(super) struct NodeSpec {
    pub(super) source: Source,
    pub(super) part: Part,
    pub(super) key: KeyOrigin,
    pub(super) label: (&'static str, &'static str),
}

impl KeyOrigin {
    const fn input(self) -> Option<Hydrator> {
        match self {
            KeyOrigin::RequestTweets | KeyOrigin::Viewer => None,
            KeyOrigin::PureCoreAuthor
            | KeyOrigin::PureCoreRetweeter
            | KeyOrigin::PureCoreReplyRoot => Some(Hydrator::PureCore),
            KeyOrigin::ExclusiveConversationAuthor
            | KeyOrigin::CommunityPost
            | KeyOrigin::TweetCommunity
            | KeyOrigin::TweetArticle
            | KeyOrigin::TrustedFriendsList
            | KeyOrigin::NarrowcastPlace => Some(Hydrator::Tweet),
            KeyOrigin::ConversationRoot(_) | KeyOrigin::ViewerForCoAllowedList => {
                Some(Hydrator::ConversationControl)
            }
            KeyOrigin::MyNetworkRootNotFollowingViewer => Some(Hydrator::RootFollowsViewer),
            KeyOrigin::ModeratedCommunity => Some(Hydrator::CommunityModeration),
        }
    }

    fn reads(self) -> Hydrators {
        let input = match self.input() {
            Some(input) => Hydrators::of(input),
            None => Hydrators::empty(),
        };
        match self {
            KeyOrigin::MyNetworkRootNotFollowingViewer => input.with(Hydrator::ConversationControl),
            KeyOrigin::CommunityPost => input.with(Hydrator::PureCore),
            KeyOrigin::RequestTweets
            | KeyOrigin::Viewer
            | KeyOrigin::PureCoreAuthor
            | KeyOrigin::PureCoreRetweeter
            | KeyOrigin::PureCoreReplyRoot
            | KeyOrigin::ExclusiveConversationAuthor
            | KeyOrigin::TweetArticle
            | KeyOrigin::ConversationRoot(_)
            | KeyOrigin::ViewerForCoAllowedList
            | KeyOrigin::ModeratedCommunity
            | KeyOrigin::TweetCommunity
            | KeyOrigin::TrustedFriendsList
            | KeyOrigin::NarrowcastPlace => input,
        }
    }
}

impl Hydrator {
    pub(super) const fn spec(self) -> NodeSpec {
        use Hydrator as H;
        use KeyOrigin as K;
        use Source as S;
        const fn node(
            source: Source,
            part: Part,
            key: KeyOrigin,
            label: (&'static str, &'static str),
        ) -> NodeSpec {
            NodeSpec {
                source,
                part,
                key,
                label,
            }
        }
        const RELATIONSHIPS: (&str, &str) = ("socialgraph", "batch_check_relationships");
        const BLOCKED_BY: (&str, &str) = ("blocked_by", "batch_check_blocked_by");
        match self {
            H::PureCore => node(
                S::TesPureCore,
                Part::Column,
                K::RequestTweets,
                ("tes", "get_tweet_core_datas"),
            ),
            H::Tweet => node(
                S::TesTweet,
                Part::Column,
                K::RequestTweets,
                ("tes", "get_tweets"),
            ),
            H::ConversationControl => node(
                S::TesConversationControl,
                Part::Column,
                K::RequestTweets,
                ("conversation_control", "get_conversation_controls"),
            ),
            H::TweetSafetyLabels => node(
                S::SafetyLabels,
                Part::Column,
                K::RequestTweets,
                ("safety_labels", "get"),
            ),
            H::ViewerProfile => node(
                S::GizmoduckViewer,
                Part::Fields(&[
                    QueryFields::ACCOUNT,
                    QueryFields::EXTENDED_PROFILE,
                    QueryFields::SAFETY,
                ]),
                K::Viewer,
                ("gizmoduck", "get_viewer_data"),
            ),
            H::ViewerLabels => node(
                S::GizmoduckViewer,
                Part::Fields(&[QueryFields::LABELS]),
                K::Viewer,
                ("gizmoduck", "get_viewer_data"),
            ),
            H::AuthorSafety => node(
                S::GizmoduckAuthor,
                Part::Fields(&[QueryFields::SAFETY]),
                K::PureCoreAuthor,
                ("gizmoduck", "get_users"),
            ),
            H::AuthorLabels => node(
                S::GizmoduckAuthor,
                Part::Fields(&[QueryFields::LABELS]),
                K::PureCoreAuthor,
                ("gizmoduck", "get_users"),
            ),
            H::Follows => node(
                S::Flock,
                Part::Edge(Edge::Follows),
                K::PureCoreAuthor,
                RELATIONSHIPS,
            ),
            H::Blocks => node(
                S::Flock,
                Part::Edge(Edge::Blocks),
                K::PureCoreAuthor,
                RELATIONSHIPS,
            ),
            H::Mutes => node(
                S::Flock,
                Part::Edge(Edge::Mutes),
                K::PureCoreAuthor,
                RELATIONSHIPS,
            ),
            H::MuteRetweets => node(
                S::Flock,
                Part::Edge(Edge::MuteRetweets),
                K::PureCoreRetweeter,
                RELATIONSHIPS,
            ),
            H::BlockedByAuthor => node(
                S::Flock,
                Part::Edge(Edge::BlockedBy),
                K::PureCoreAuthor,
                BLOCKED_BY,
            ),
            H::BlockedByReplyRoot => node(
                S::Flock,
                Part::Edge(Edge::BlockedBy),
                K::PureCoreReplyRoot,
                BLOCKED_BY,
            ),
            H::SuperFollowsExclusive => node(
                S::Flock,
                Part::Edge(Edge::SuperFollows),
                K::ExclusiveConversationAuthor,
                ("exclusive_content", "batch_check_super_follows"),
            ),
            H::RootFollowsViewer => node(
                S::Flock,
                Part::Edge(Edge::FollowedBy),
                K::ConversationRoot(&[
                    ConversationControlArm::Community,
                    ConversationControlArm::MyNetwork,
                ]),
                ("conversation_control", "batch_check_followed_by"),
            ),
            H::RootFollowsViewerSecondDegree => node(
                S::Wingman,
                Part::Edge(Edge::SecondDegree),
                K::MyNetworkRootNotFollowingViewer,
                ("conversation_control", "exists_intersect"),
            ),
            H::SuperFollowsRoot => node(
                S::Flock,
                Part::Edge(Edge::SuperFollows),
                K::ConversationRoot(&[ConversationControlArm::Subscribers]),
                ("conversation_control", "batch_check_super_follows"),
            ),
            H::ViewerCountry => node(
                S::ViewerCountry,
                Part::Column,
                K::ViewerForCoAllowedList,
                ("conversation_control", "tfe_top_country"),
            ),
            H::CommunityModeration => node(
                S::CommunityModeration,
                Part::Column,
                K::CommunityPost,
                ("communities", "moderation_state"),
            ),
            H::CommunityModerator => node(
                S::CommunityModerator,
                Part::Column,
                K::ModeratedCommunity,
                ("communities", "visibility_features"),
            ),
            H::CommunityViewerRemoved => node(
                S::CommunityViewerRemoved,
                Part::Column,
                K::TweetCommunity,
                ("communities", "is_removed"),
            ),
            H::ArticleLifecycle => node(
                S::ArticleLifecycle,
                Part::Column,
                K::TweetArticle,
                ("article", "get_lifecycles"),
            ),
            H::TrustedFriends => node(
                S::TrustedFriends,
                Part::Edge(Edge::TrustedFriends),
                K::TrustedFriendsList,
                ("trusted_friends", "is_member_or_owner"),
            ),
            H::OutsideNarrowcastPlace => node(
                S::UserLocation,
                Part::Edge(Edge::OutsidePlace),
                K::NarrowcastPlace,
                ("geoduck", "user_location"),
            ),
        }
    }

    pub(super) const fn input(self) -> Option<Hydrator> {
        self.spec().key.input()
    }

    pub(super) const fn edge(self) -> Option<Edge> {
        match self.spec().part {
            Part::Edge(edge) => Some(edge),
            Part::Column | Part::Fields(_) => None,
        }
    }

    pub(crate) const fn is_edge(self) -> bool {
        self.edge().is_some()
    }

    pub(super) const fn needs_viewer(self) -> bool {
        self.is_edge()
            || matches!(
                self.spec().key,
                KeyOrigin::Viewer
                    | KeyOrigin::ViewerForCoAllowedList
                    | KeyOrigin::ModeratedCommunity
                    | KeyOrigin::TweetCommunity
            )
    }
}

const fn inputs_precede_nodes() -> bool {
    let mut rest = Hydrator::VARIANTS;
    while let [node, tail @ ..] = rest {
        if let Some(input) = node.input()
            && input as u8 >= *node as u8
        {
            return false;
        }
        rest = tail;
    }
    true
}

const fn miss_policies_fit_their_nodes() -> bool {
    let mut rest = Hydrator::VARIANTS;
    while let [node, tail @ ..] = rest {
        let spec = node.spec();
        let fits = match spec.source.miss_policy() {
            MissPolicy::Unresolves(Subject::Tweet) => matches!(spec.key, KeyOrigin::RequestTweets),
            MissPolicy::Unresolves(Subject::Author) => {
                matches!(spec.key, KeyOrigin::PureCoreAuthor)
            }
            MissPolicy::FailsNode => true,
            MissPolicy::ReadsNoEdge => node.is_edge(),
        };
        if !fits {
            return false;
        }
        rest = tail;
    }
    true
}

const _: () = assert!(inputs_precede_nodes());
const _: () = assert!(miss_policies_fit_their_nodes());
const _: () = assert!(Hydrator::VARIANTS.len() <= u32::BITS as usize);

impl Hydrators {
    pub const fn closed(self) -> Self {
        let mut closed = self.with(Hydrator::PureCore);
        let mut rest = Hydrator::VARIANTS;
        while let [head @ .., node] = rest {
            if let Some(input) = node.input()
                && closed.contains(*node)
            {
                closed = closed.with(input);
            }
            rest = head;
        }
        closed
    }

    pub(crate) fn iter(self) -> impl Iterator<Item = Hydrator> {
        Hydrator::VARIANTS
            .iter()
            .copied()
            .filter(move |&node| self.contains(node))
    }
}

pub(crate) struct HydrationPlan {
    level: SafetyLevel,
    groups: Vec<Group>,
    nodes: Hydrators,
    logged_out_nodes: Hydrators,
}

pub(super) struct Group {
    pub(super) position: usize,
    pub(super) source: Source,
    pub(super) input: Option<Hydrator>,
    pub(super) nodes: Hydrators,
    clients: Vec<&'static str>,
    methods: Vec<&'static str>,
    edges: Vec<(Edge, Graph, EdgeDirection, Hydrators)>,
    fields: Vec<QueryFields>,
    readers: Vec<usize>,
}

impl HydrationPlan {
    pub(crate) fn new(level: SafetyLevel, hydrators: Hydrators) -> Self {
        let nodes = hydrators.closed();
        let mut groups: Vec<Group> = Vec::new();
        for node in nodes.iter() {
            let spec = node.spec();
            let (client, method) = spec.label;
            let input = node.input();
            match groups
                .iter_mut()
                .find(|group| group.source == spec.source && group.input == input)
            {
                Some(group) => {
                    group.nodes = group.nodes.with(node);
                    if !group.clients.contains(&client) {
                        group.clients.push(client);
                    }
                    if !group.methods.contains(&method) {
                        group.methods.push(method);
                    }
                }
                None => groups.push(Group {
                    position: groups.len(),
                    source: spec.source,
                    input,
                    nodes: Hydrators::of(node),
                    clients: vec![client],
                    methods: vec![method],
                    edges: Vec::new(),
                    fields: Vec::new(),
                    readers: Vec::new(),
                }),
            }
        }
        let reads: Vec<Hydrators> = groups
            .iter()
            .map(|group| {
                group.nodes.iter().fold(Hydrators::empty(), |reads, node| {
                    reads.union(node.spec().key.reads())
                })
            })
            .collect();
        let readers: Vec<Vec<usize>> = groups
            .iter()
            .map(|landed| {
                reads
                    .iter()
                    .enumerate()
                    .filter(|(_, reads)| !reads.intersection(landed.nodes).is_empty())
                    .map(|(position, _)| position)
                    .collect()
            })
            .collect();
        for (group, readers) in groups.iter_mut().zip(readers) {
            group.edges = edges(group.nodes);
            group.fields = fields(group.source, group.nodes);
            group.readers = readers;
        }
        Self {
            level,
            groups,
            nodes,
            logged_out_nodes: nodes
                .iter()
                .filter(|node| !node.needs_viewer())
                .fold(Hydrators::empty(), Hydrators::with),
        }
    }

    pub(crate) fn level(&self) -> SafetyLevel {
        self.level
    }

    pub(super) fn callable(&self, viewer_id: Option<u64>) -> Hydrators {
        match viewer_id {
            Some(_) => self.nodes,
            None => self.logged_out_nodes,
        }
    }

    pub(super) fn groups(&self) -> impl Iterator<Item = &Group> {
        self.groups.iter()
    }

    pub(super) fn readers<'a>(&'a self, landed: &'a Group) -> impl Iterator<Item = &'a Group> {
        landed
            .readers
            .iter()
            .filter_map(|&position| self.groups.get(position))
    }
}

impl Group {
    pub(super) fn label(&self) -> (String, String) {
        (self.clients.join("+"), self.methods.join("+"))
    }

    pub(super) fn fields(&self) -> &[QueryFields] {
        &self.fields
    }

    pub(super) fn edges(&self) -> &[(Edge, Graph, EdgeDirection, Hydrators)] {
        &self.edges
    }
}

fn fields(source: Source, nodes: Hydrators) -> Vec<QueryFields> {
    let nodes = match source {
        Source::GizmoduckAuthor => Hydrator::VARIANTS
            .iter()
            .copied()
            .filter(|node| node.spec().source == source)
            .fold(Hydrators::empty(), Hydrators::with),
        Source::TesPureCore
        | Source::TesTweet
        | Source::TesConversationControl
        | Source::SafetyLabels
        | Source::GizmoduckViewer
        | Source::Flock
        | Source::ViewerCountry
        | Source::Wingman
        | Source::CommunityModeration
        | Source::CommunityModerator
        | Source::CommunityViewerRemoved
        | Source::ArticleLifecycle
        | Source::TrustedFriends
        | Source::UserLocation => nodes,
    };
    let mut fields = Vec::new();
    for node in nodes.iter() {
        if let Part::Fields(node_fields) = node.spec().part {
            for field in node_fields {
                if !fields.contains(field) {
                    fields.push(*field);
                }
            }
        }
    }
    fields
}

fn edges(nodes: Hydrators) -> Vec<(Edge, Graph, EdgeDirection, Hydrators)> {
    let mut edges: Vec<(Edge, Graph, EdgeDirection, Hydrators)> = Vec::new();
    for node in nodes.iter() {
        let Some(edge) = node.edge() else {
            continue;
        };
        let Some((graph, direction)) = edge.flock() else {
            continue;
        };
        match edges.iter_mut().find(|(queried, ..)| *queried == edge) {
            Some((.., nodes)) => *nodes = nodes.with(node),
            None => edges.push((edge, graph, direction, Hydrators::of(node))),
        }
    }
    edges
}

impl fmt::Display for KeyOrigin {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            KeyOrigin::RequestTweets => f.write_str("tweets"),
            KeyOrigin::Viewer => f.write_str("viewer"),
            KeyOrigin::PureCoreAuthor => f.write_str("author"),
            KeyOrigin::PureCoreRetweeter => f.write_str("retweeter"),
            KeyOrigin::PureCoreReplyRoot => f.write_str("reply_root"),
            KeyOrigin::ExclusiveConversationAuthor => f.write_str("exclusive_author"),
            KeyOrigin::TweetArticle => f.write_str("article"),
            KeyOrigin::ConversationRoot(arms) => {
                let arms: Vec<String> = arms.iter().map(|arm| format!("{arm:?}")).collect();
                write!(f, "root:{}", arms.join("|"))
            }
            KeyOrigin::ViewerForCoAllowedList => f.write_str("viewer:co_allowed_list"),
            KeyOrigin::MyNetworkRootNotFollowingViewer => {
                f.write_str("root:MyNetwork:not_followed")
            }
            KeyOrigin::CommunityPost => f.write_str("community_post"),
            KeyOrigin::ModeratedCommunity => f.write_str("moderated_community"),
            KeyOrigin::TweetCommunity => f.write_str("tweet_community"),
            KeyOrigin::TrustedFriendsList => f.write_str("trusted_friends_list"),
            KeyOrigin::NarrowcastPlace => f.write_str("narrowcast_place"),
        }
    }
}

impl fmt::Display for HydrationPlan {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "{}: {} calls",
            <&str>::from(self.level),
            self.groups.len()
        )?;
        for group in &self.groups {
            let (client, method) = group.label();
            let after = group.input.map_or("-", <&str>::from);
            let nodes: Vec<&str> = group.nodes.iter().map(<&str>::from).collect();
            write!(
                f,
                "{client}/{method} after: {after} nodes: {}",
                nodes.join(",")
            )?;
            let fields = group.fields();
            if !fields.is_empty() {
                let fields: Vec<String> = fields.iter().map(|field| format!("{field:?}")).collect();
                write!(f, " fields: {}", fields.join("|"))?;
            }
            for &(_, graph, direction, nodes) in group.edges() {
                let direction = match direction {
                    EdgeDirection::Forward => "fwd",
                    EdgeDirection::Reverse => "rev",
                };
                let keys: Vec<String> = nodes
                    .iter()
                    .map(|node| node.spec().key.to_string())
                    .collect();
                write!(
                    f,
                    " {}-{direction}[{}]",
                    <&str>::from(graph),
                    keys.join(",")
                )?;
            }
            if group.nodes.iter().all(Hydrator::needs_viewer) {
                f.write_str(" (skipped logged out)")?;
            }
            writeln!(f)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use crate::rules::{RuleEngine, SafetyLevel};
    use strum::VariantArray;

    const PLANS: &str = "\
filter_all: 1 calls
tes/get_tweet_core_datas after: - nodes: pure_core
timeline_home: 8 calls
tes/get_tweet_core_datas after: - nodes: pure_core
tes/get_tweets after: - nodes: tweet
safety_labels/get after: - nodes: tweet_safety_labels
gizmoduck/get_viewer_data after: - nodes: viewer_profile fields: ACCOUNT|EXTENDED_PROFILE|SAFETY (skipped logged out)
gizmoduck/get_users after: pure_core nodes: author_safety fields: SAFETY|LABELS
socialgraph/batch_check_relationships after: pure_core nodes: follows,blocks,mutes,mute_retweets follows-fwd[author] blocks-fwd[author] mutes-fwd[author] mute_retweets-fwd[retweeter] (skipped logged out)
exclusive_content/batch_check_super_follows after: tweet nodes: super_follows_exclusive super_follows-fwd[exclusive_author] (skipped logged out)
trusted_friends/is_member_or_owner after: tweet nodes: trusted_friends (skipped logged out)
timeline_home_recommendations: 8 calls
tes/get_tweet_core_datas after: - nodes: pure_core
tes/get_tweets after: - nodes: tweet
safety_labels/get after: - nodes: tweet_safety_labels
gizmoduck/get_viewer_data after: - nodes: viewer_profile fields: ACCOUNT|EXTENDED_PROFILE|SAFETY (skipped logged out)
gizmoduck/get_users after: pure_core nodes: author_safety,author_labels fields: SAFETY|LABELS
socialgraph/batch_check_relationships after: pure_core nodes: follows,blocks,mutes,mute_retweets follows-fwd[author] blocks-fwd[author] mutes-fwd[author] mute_retweets-fwd[retweeter] (skipped logged out)
exclusive_content/batch_check_super_follows after: tweet nodes: super_follows_exclusive super_follows-fwd[exclusive_author] (skipped logged out)
trusted_friends/is_member_or_owner after: tweet nodes: trusted_friends (skipped logged out)
timeline_home_hydration: 17 calls
tes/get_tweet_core_datas after: - nodes: pure_core
tes/get_tweets after: - nodes: tweet
conversation_control/get_conversation_controls after: - nodes: conversation_control
safety_labels/get after: - nodes: tweet_safety_labels
gizmoduck/get_viewer_data after: - nodes: viewer_profile,viewer_labels fields: ACCOUNT|EXTENDED_PROFILE|SAFETY|LABELS (skipped logged out)
gizmoduck/get_users after: pure_core nodes: author_safety fields: SAFETY|LABELS
socialgraph+blocked_by/batch_check_relationships+batch_check_blocked_by after: pure_core nodes: follows,blocked_by_author,blocked_by_reply_root follows-fwd[author] blocks-rev[author,reply_root] (skipped logged out)
exclusive_content/batch_check_super_follows after: tweet nodes: super_follows_exclusive super_follows-fwd[exclusive_author] (skipped logged out)
conversation_control/batch_check_followed_by+batch_check_super_follows after: conversation_control nodes: root_follows_viewer,super_follows_root follows-rev[root:Community|MyNetwork] super_follows-fwd[root:Subscribers] (skipped logged out)
conversation_control/exists_intersect after: root_follows_viewer nodes: root_follows_viewer_second_degree (skipped logged out)
conversation_control/tfe_top_country after: conversation_control nodes: viewer_country (skipped logged out)
communities/moderation_state after: tweet nodes: community_moderation
communities/visibility_features after: community_moderation nodes: community_moderator (skipped logged out)
communities/is_removed after: tweet nodes: community_viewer_removed (skipped logged out)
article/get_lifecycles after: tweet nodes: article_lifecycle
trusted_friends/is_member_or_owner after: tweet nodes: trusted_friends (skipped logged out)
geoduck/user_location after: tweet nodes: outside_narrowcast_place (skipped logged out)
immersive_expanded_recommendations: 8 calls
tes/get_tweet_core_datas after: - nodes: pure_core
tes/get_tweets after: - nodes: tweet
safety_labels/get after: - nodes: tweet_safety_labels
gizmoduck/get_viewer_data after: - nodes: viewer_profile fields: ACCOUNT|EXTENDED_PROFILE|SAFETY (skipped logged out)
gizmoduck/get_users after: pure_core nodes: author_safety,author_labels fields: SAFETY|LABELS
socialgraph/batch_check_relationships after: pure_core nodes: follows,blocks,mutes,mute_retweets follows-fwd[author] blocks-fwd[author] mutes-fwd[author] mute_retweets-fwd[retweeter] (skipped logged out)
exclusive_content/batch_check_super_follows after: tweet nodes: super_follows_exclusive super_follows-fwd[exclusive_author] (skipped logged out)
trusted_friends/is_member_or_owner after: tweet nodes: trusted_friends (skipped logged out)
";

    #[test]
    fn each_level_plans_these_calls() {
        let engine = RuleEngine::for_tests();
        let plans: String = SafetyLevel::VARIANTS
            .iter()
            .map(|&level| engine.plan(level).to_string())
            .collect();
        assert_eq!(plans, PLANS);
    }
}
