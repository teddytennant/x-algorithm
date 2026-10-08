package com.twitter.scarecrow.features;

import java.util.List;
import java.util.Set;
import java.util.stream.Collectors;

import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Lists;
import com.twitter.mediaservices.commons.media_update.thrift_java.CalibrationBucket;
import com.twitter.mediaservices.commons.media_update.thrift_java.CalibrationLevel;
import com.twitter.mediaservices.commons.media_update.thrift_java.ImageTag;
import com.twitter.mediaservices.commons.media_update.thrift_java.MediaHashInfo;
import com.twitter.mediaservices.commons.media_update.thrift_java.MediaUpdate;
import com.twitter.mediaservices.commons.media_update.thrift_java.UpdateInfo;
import com.twitter.mediaservices.commons.mediainformation.thrift_java.Md5HashInfo;
import com.twitter.mediaservices.media_util.GenericMediaKey;
import com.twitter.spam.botmaker_features.BotMakerFeatures;
import com.twitter.spam.botmaker_features.FeatureMapBuilder;
import com.twitter.spam.botmaker_features.FeatureMapExtractor;
import com.twitter.mediaservices.commons.thriftscala.MediaCategory;

import static com.twitter.botmaker.FeatureModifier.OPTIONAL;
import static com.twitter.botmaker.FeatureModifier.REQUIRED;

public class FeaturesOfMediaUpdate extends FeatureMapExtractor {

  private static final ImmutableSet<String> FILTERED_TAGS =
      ImmutableSet.of("xx", "sexy", "xxnsfw", "xxnfswv2", "xxnsfwv2");

  private static final ImmutableSet<String> FILTERED_PREDICTION_TAGS =
      ImmutableSet.of("nsfw_score");

  private static final String HIGH_PRECISION_FEATURE_KEY = "highPrecision";
  private static final String HIGH_RECALL_FEATURE_KEY = "highRecall";
  private static final String NEAR_PERFECT_FEATURE_KEY = "nearPerfect";
  private static final String PROBABILITY_FEATURE_KEY = "probability";
  private static final String PRECISION_FEATURE_KEY = "precision";
  private static final String RECALL_FEATURE_KEY = "recall";
  private static final String FALSE_POSITIVE_RATE_FEATURE_KEY = "falsePositiveRate";

  private final MediaUpdate mediaUpdate;

  public FeaturesOfMediaUpdate(MediaUpdate mediaUpdate) {
    this.mediaUpdate = mediaUpdate;
  }

  private void putTags(
      FeatureMapBuilder builder,
      List<ImageTag> tags,
      Set<String> filteredTags
  ) throws Exception {
    builder.putValue(REQUIRED, BotMakerFeatures.updateType, "tag");
    List<String> tagsList = Lists.newArrayList();

    for (ImageTag tag : tags) {
      if (filteredTags.contains(tag.getLabel().toLowerCase())) {
        double probability = tag.getProbability();
        String tagLabel = tag.getLabel();

        if ("xxnfswv2".equals(tagLabel.toLowerCase())
            || "xxnsfwv2".equals(tagLabel.toLowerCase())) {
          String featureKeySuffix = "NsfwV2";
          builder.putValue(OPTIONAL, PROBABILITY_FEATURE_KEY + featureKeySuffix, probability);
          if (tag.isSetPrecision()) {
            builder.putValue(OPTIONAL,
                PRECISION_FEATURE_KEY + featureKeySuffix, tag.getPrecision());
          }
          if (tag.isSetRecall()) {
            builder.putValue(OPTIONAL, RECALL_FEATURE_KEY + featureKeySuffix, tag.getRecall());
          }
          if (tag.isSetFpr()) {
            builder.putValue(OPTIONAL, FALSE_POSITIVE_RATE_FEATURE_KEY + featureKeySuffix,
              tag.getFpr());
          }
        } else if ("xxnsfw".equals(tagLabel.toLowerCase())) {
          String featureKeySuffix = "NewNsfw";
          builder.putValue(OPTIONAL, PROBABILITY_FEATURE_KEY + featureKeySuffix, probability);
          if (tag.isSetPrecision()) {
            builder.putValue(OPTIONAL,
                PRECISION_FEATURE_KEY + featureKeySuffix, tag.getPrecision());
          }
          if (tag.isSetRecall()) {
            builder.putValue(OPTIONAL, RECALL_FEATURE_KEY + featureKeySuffix, tag.getRecall());
          }
          if (tag.isSetFpr()) {
            builder.putValue(OPTIONAL, FALSE_POSITIVE_RATE_FEATURE_KEY + featureKeySuffix,
              tag.getFpr());
          }
        } else {
          boolean highPrecision = false;
          boolean highRecall = false;
          boolean nearPerfect = false;
          for (CalibrationBucket bucket : tag.getCalibrationBuckets()) {
            if (!bucket.isSetDescription()) {
              continue;
            }
            if (bucket.getDescription() == CalibrationLevel.HIGH_PRECISION
              && probability > bucket.getThreshold()) {
              highPrecision = true;
            } else if (bucket.getDescription() == CalibrationLevel.NEAR_PERFECT
              && probability > bucket.getThreshold()) {
              nearPerfect = true;
            } else if (bucket.getDescription() == CalibrationLevel.HIGH_RECALL
              && probability > bucket.getThreshold()) {
              highRecall = true;
            }
          }
          String featureKeySuffix = ("xx".equals(tagLabel.toLowerCase())) ?
            "Nsfw" : tagLabel;
          builder.putValue(OPTIONAL, HIGH_PRECISION_FEATURE_KEY + featureKeySuffix, highPrecision);
          builder.putValue(OPTIONAL, HIGH_RECALL_FEATURE_KEY + featureKeySuffix, highRecall);
          builder.putValue(OPTIONAL, NEAR_PERFECT_FEATURE_KEY + featureKeySuffix, nearPerfect);
          builder.putValue(OPTIONAL, PROBABILITY_FEATURE_KEY + featureKeySuffix, probability);
        }
        tagsList.add(tag.getLabel());
      }
    }
    builder.putValue(REQUIRED, BotMakerFeatures.tagLabels, tagsList);
  }

  @Override
  public void apply(FeatureMapBuilder builder) throws Exception {
    builder.putValue(REQUIRED, BotMakerFeatures.mediaId, mediaUpdate.getMedia_id());
    builder.putValue(REQUIRED, BotMakerFeatures.tweetId, mediaUpdate.getTweet_id());
    builder.putValue(REQUIRED, BotMakerFeatures.spammerId, mediaUpdate.getUser_id());
    builder.putValue(REQUIRED, BotMakerFeatures.category, mediaUpdate.getCategory().name());
    GenericMediaKey genericMediaKey = GenericMediaKey.apply(
        MediaCategory.apply(mediaUpdate.getCategory().getValue()),
        mediaUpdate.getMedia_id(),
        mediaUpdate.isSetUser_id() ? mediaUpdate.getUser_id() : null
    );
    builder.putValue(REQUIRED, BotMakerFeatures.genericMediaKey,
          genericMediaKey.toStringKey());
    if (mediaUpdate.getUpdate_info().isSet(UpdateInfo._Fields.CLUSTER)) {
      builder.putValue(REQUIRED, BotMakerFeatures.updateType, "cluster");
      builder.putValue(REQUIRED, BotMakerFeatures.clusterId,
          mediaUpdate.getUpdate_info().getCluster().getCluster_id());
      builder.putCollection(
          REQUIRED,
          BotMakerFeatures.clusterIds,
          ImmutableSet.of(
              String.valueOf(mediaUpdate.getUpdate_info().getCluster().getCluster_id())
          ),
          String.class);
      builder.putValue(REQUIRED, BotMakerFeatures.modelVersion,
          mediaUpdate.getUpdate_info().getCluster().getModel_version());
      builder.putCollection(REQUIRED, BotMakerFeatures.signature,
          mediaUpdate.getUpdate_info().getCluster().getSignature().getVector(), Double.class);
    } else if (mediaUpdate.getUpdate_info().isSet(UpdateInfo._Fields.TAGS)) {
      final List<ImageTag> tags = mediaUpdate.getUpdate_info().getTags().getTags();
      putTags(builder, tags, FILTERED_TAGS);
    } else if (mediaUpdate.getUpdate_info().isSet(UpdateInfo._Fields.PREDICTION_TAGS)) {
      final List<ImageTag> tags = mediaUpdate.getUpdate_info().getPredictionTags().getTags();
      putTags(builder, tags, FILTERED_PREDICTION_TAGS);
    } else if (mediaUpdate.getUpdate_info().isSet(UpdateInfo._Fields.MEDIA_HASH)) {
      builder.putValue(REQUIRED, BotMakerFeatures.updateType, "mediahash");
      final MediaHashInfo hashInfo = mediaUpdate.getUpdate_info().getMedia_hash();
      if (hashInfo.isSetMd5_hashes()) {
        final List<String> hashes = hashInfo.getMd5_hashes()
            .stream()
            .filter(Md5HashInfo::isSetHash)
            .map(Md5HashInfo::getHash)
            .collect(Collectors.toList());

        builder.putCollection(REQUIRED, BotMakerFeatures.md5Hashes, hashes, String.class);
      }
    } else if (mediaUpdate.getUpdate_info().isSet(UpdateInfo._Fields.VIDEO_CLUSTERS)) {
      builder.putValue(REQUIRED, BotMakerFeatures.updateType, "video_clusters");
      final Set<String> matchedVideos =
          mediaUpdate.getUpdate_info().getVideo_clusters().getMatched_videos();
      if (matchedVideos != null) {
        builder.putCollection(REQUIRED, BotMakerFeatures.clusterIds, matchedVideos, String.class);
      }
    }
    if (mediaUpdate.isSetOwnership_info() && mediaUpdate.getOwnership_info().isSetEntity_id()) {
      builder.putValue(OPTIONAL, BotMakerFeatures.entityId,
          mediaUpdate.getOwnership_info().getEntity_id());
    }
    if (mediaUpdate.isSetOwnership_info() && mediaUpdate.getOwnership_info().isSetDm_id()) {
      builder.putValue(OPTIONAL, BotMakerFeatures.dmId,
          mediaUpdate.getOwnership_info().getDm_id());
    }
  }
}
