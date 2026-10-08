use crate::models::AuthorLabel;
use crate::rules::rule_spec::{
    author, drop_post, everyone, except_author, not, relationship, rule, tweet, viewer,
    AuthorPredicate, Condition, RelationshipPredicate, RuleClause, RuleId, TweetPredicate,
    ViewerPredicate,
};
use xai_visibility_filtering::models::FilteredReason;

const NOT_FOLLOWER: Condition = not(relationship(RelationshipPredicate::ViewerFollowsAuthor));

fn author_drop(id: RuleId, state: AuthorPredicate, reason: FilteredReason) -> Vec<RuleClause> {
    rule(id, except_author([author(state)], drop_post(reason)))
}

fn suspended_author() -> Vec<RuleClause> {
    author_drop(
        RuleId::SuspendedAuthor,
        AuthorPredicate::IsSuspended,
        FilteredReason::AuthorIsSuspended,
    )
}

fn deactivated_author() -> Vec<RuleClause> {
    author_drop(
        RuleId::DeactivatedAuthor,
        AuthorPredicate::IsDeactivated,
        FilteredReason::AuthorIsDeactivated,
    )
}

fn erased_author() -> Vec<RuleClause> {
    author_drop(
        RuleId::ErasedAuthor,
        AuthorPredicate::IsErased,
        FilteredReason::AuthorAccountIsInactive,
    )
}

fn offboarded_author() -> Vec<RuleClause> {
    author_drop(
        RuleId::OffboardedAuthor,
        AuthorPredicate::IsOffboarded,
        FilteredReason::AuthorAccountIsInactive,
    )
}

fn protected_author() -> Vec<RuleClause> {
    rule(
        RuleId::ProtectedAuthor,
        except_author(
            [author(AuthorPredicate::IsProtected), NOT_FOLLOWER],
            drop_post(FilteredReason::AuthorIsProtected),
        ),
    )
}

pub(super) fn author_state_drops() -> Vec<RuleClause> {
    [
        suspended_author(),
        deactivated_author(),
        erased_author(),
        offboarded_author(),
        protected_author(),
    ]
    .concat()
}

pub(super) fn home_hydration_author_state_drops() -> Vec<RuleClause> {
    [
        erased_author(),
        deactivated_author(),
        suspended_author(),
        offboarded_author(),
        protected_author(),
    ]
    .concat()
}

pub(super) fn oon_nsfw_author_drops() -> Vec<RuleClause> {
    [
        author_drop(
            RuleId::NsfwUserAuthor,
            AuthorPredicate::IsNsfwUser,
            FilteredReason::ContainNsfwMedia,
        ),
        author_drop(
            RuleId::NsfwAdminAuthor,
            AuthorPredicate::IsNsfwAdmin,
            FilteredReason::ContainNsfwMedia,
        ),
    ]
    .concat()
}

fn user_label_drop(id: RuleId, label: AuthorLabel) -> Vec<RuleClause> {
    author_drop(
        id,
        AuthorPredicate::HasUserLabel(label),
        FilteredReason::UnspecifiedReason,
    )
}

fn non_follower_user_label_drop(id: RuleId, label: AuthorLabel) -> Vec<RuleClause> {
    rule(
        id,
        except_author(
            [author(AuthorPredicate::HasUserLabel(label)), NOT_FOLLOWER],
            drop_post(FilteredReason::UnspecifiedReason),
        ),
    )
}

pub(super) fn oon_nsfw_user_label_drops() -> Vec<RuleClause> {
    use AuthorLabel::{
        NsfwAvatarImage, NsfwBannerImage, NsfwHighPrecision, NsfwHighRecall, NsfwNearPerfect,
    };
    [
        user_label_drop(RuleId::NsfwHighRecallUserLabel, NsfwHighRecall),
        user_label_drop(RuleId::NsfwHighPrecisionUserLabel, NsfwHighPrecision),
        user_label_drop(RuleId::NsfwAvatarImageUserLabel, NsfwAvatarImage),
        user_label_drop(RuleId::NsfwBannerImageUserLabel, NsfwBannerImage),
        user_label_drop(RuleId::NsfwNearPerfectUserLabel, NsfwNearPerfect),
    ]
    .concat()
}

pub(super) fn oon_user_label_drops() -> Vec<RuleClause> {
    use AuthorLabel::{
        AbusiveHighRecall, Compromised, DoNotAmplify, ImpersonationHighPrecision, ReadOnly,
        SpamHighRecall,
    };
    [
        user_label_drop(RuleId::SpamHighRecallUserLabel, SpamHighRecall),
        user_label_drop(RuleId::CompromisedUserLabel, Compromised),
        user_label_drop(RuleId::ReadOnlyUserLabel, ReadOnly),
        user_label_drop(
            RuleId::ImpersonationHighPrecisionUserLabel,
            ImpersonationHighPrecision,
        ),
        non_follower_user_label_drop(RuleId::AbusiveHighRecallUserLabel, AbusiveHighRecall),
        non_follower_user_label_drop(RuleId::DoNotAmplifyUserLabel, DoNotAmplify),
    ]
    .concat()
}

pub(super) fn socialgraph_drops() -> Vec<RuleClause> {
    const NOT_LOGGED_OUT: Condition = not(viewer(ViewerPredicate::LoggedOut));
    [
        rule(
            RuleId::ViewerBlocksAuthor,
            everyone(
                [
                    NOT_LOGGED_OUT,
                    relationship(RelationshipPredicate::ViewerBlocksAuthor),
                ],
                drop_post(FilteredReason::ViewerBlocksAuthor),
            ),
        ),
        rule(
            RuleId::ViewerMutesAuthor,
            everyone(
                [
                    NOT_LOGGED_OUT,
                    relationship(RelationshipPredicate::ViewerMutesAuthor),
                ],
                drop_post(FilteredReason::ViewerMutesAuthor),
            ),
        ),
        rule(
            RuleId::ViewerMutesRetweets,
            everyone(
                [
                    NOT_LOGGED_OUT,
                    tweet(TweetPredicate::IsRetweet),
                    relationship(RelationshipPredicate::ViewerMutesRetweetsFromAuthor),
                ],
                drop_post(FilteredReason::UnspecifiedReason),
            ),
        ),
    ]
    .concat()
}
