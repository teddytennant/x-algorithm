namespace java com.twitter.agatha.thriftjava

include "com/twitter/hub/agatha/agatha.thrift"

struct FlattenedBlinkScore {
  1: i64 userId
  2: string label
  3: double score
  4: optional double scoreOutbound
  5: optional double support
  6: optional double supportOutbound
}(hasPersonalData='true', persisted = 'true')

struct UserPrediction {
  1: required i64 userId(personalDataType = 'UserId')
  2: required agatha.FeatureClass featureClass
  3: required string label
  4: map<i32, double> moments
}(hasPersonalData = 'true', persisted = 'true')

struct UserQuantilePrediction {
  1: required i64 userId(personalDataType = 'UserId')
  2: required agatha.FeatureClass featureClass
  3: required string label
  4: map<double, double> quantiles
}(hasPersonalData = 'true', persisted = 'true')
