mod author_rules;
mod context;
#[cfg(test)]
pub(crate) mod fixtures;
#[cfg(test)]
mod golden_corpus;
pub mod metrics;
pub mod registry;
mod rule_spec;
mod safety_level;
mod tweet_rules;

#[cfg(test)]
use crate::models::{HydratedTweetCandidate, ViewerFeatures};
#[cfg(test)]
use crate::params::CountryLists;
use context::RuleContext;
pub use registry::RuleEngine;
#[cfg(test)]
use rule_spec::Predicate;
pub use safety_level::SafetyLevel;

#[cfg(test)]
pub(crate) fn test_context<'a>(
    viewer: &'a ViewerFeatures,
    candidate: &'a HydratedTweetCandidate,
) -> RuleContext<'a> {
    use std::sync::LazyLock;

    static COUNTRY_LISTS: LazyLock<CountryLists> = LazyLock::new(CountryLists::starting_at_default);
    RuleContext::new(viewer, candidate, &COUNTRY_LISTS)
}

#[cfg(test)]
fn holds_narrowed(
    predicate: Predicate,
    viewer: &ViewerFeatures,
    candidate: &HydratedTweetCandidate,
) -> bool {
    predicate.holds(&test_context(viewer, candidate).hydrated_by(predicate.hydrators()))
}
