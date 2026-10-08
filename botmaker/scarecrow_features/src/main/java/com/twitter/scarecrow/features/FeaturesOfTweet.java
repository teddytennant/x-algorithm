package com.twitter.scarecrow.features;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Set;

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Iterables;
import com.google.common.collect.Lists;
import com.google.common.collect.Sets;

import com.twitter.escherbird.thriftjava.TweetEntityAnnotation;
import com.twitter.mediaservices.commons.mediainformation.thrift_java.AdditionalMetadata;
import com.twitter.mediaservices.commons.mediainformation.thrift_java.AltText;
import com.twitter.relevance.feature_store.thrift.FeatureData;
import com.twitter.relevance.feature_store.thrift.FeatureValue;
import com.twitter.spam.botmaker_features.BotMakerFeatures;
import com.twitter.spam.botmaker_features.FeatureExtractionException;
import com.twitter.spam.botmaker_features.FeatureExtractionHelper;
import com.twitter.spam.botmaker_features.FeatureMapBuilder;
import com.twitter.spam.botmaker_features.FeatureMapExtractor;
import com.twitter.tweetypie.thriftjava.Article;
import com.twitter.tweetypie.thriftjava.CashtagEntity;
import com.twitter.tweetypie.thriftjava.DeviceSource;
import com.twitter.tweetypie.thriftjava.DirectedAtUser;
import com.twitter.tweetypie.thriftjava.HashtagEntity;
import com.twitter.tweetypie.thriftjava.MediaEntity;
import com.twitter.tweetypie.thriftjava.MediaTag;
import com.twitter.tweetypie.thriftjava.MentionEntity;
import com.twitter.tweetypie.thriftjava.NoteTweet;
import com.twitter.tweetypie.thriftjava.Reply;
import com.twitter.tweetypie.thriftjava.Share;
import com.twitter.tweetypie.thriftjava.StatusCounts;
import com.twitter.tweetypie.thriftjava.Tweet;
import com.twitter.tweetypie.thriftjava.TweetCoreData;
import com.twitter.tweetypie.thriftjava.TweetMediaTags;
import com.twitter.tweetypie.thriftjava.UrlEntity;

import static com.twitter.botmaker.FeatureModifier.OPTIONAL;
import static com.twitter.botmaker.FeatureModifier.REQUIRED;

public class FeaturesOfTweet extends FeatureMapExtractor {
  private final Tweet tweet;
  private final boolean isTweetEvent;
  private final Set<Long> victimIds;

  public FeaturesOfTweet(Tweet tweet, boolean isTweetEvent) {
    this.tweet = Preconditions.checkNotNull(tweet);
    this.isTweetEvent = isTweetEvent;
    this.victimIds = Sets.newHashSet();
  }

  private void processCoreData(FeatureMapBuilder builder, TweetCoreData coreData)
      throws FeatureExtractionException {
    builder.putValue(REQUIRED, BotMakerFeatures.actorId, coreData.getUser_id());

    if (isTweetEvent) {
      builder.putValue(REQUIRED, BotMakerFeatures.spammerId, coreData.getUser_id());
    }

    builder.putValue(REQUIRED, BotMakerFeatures.tweetText, coreData.getText());

    builder.putValue(OPTIONAL, BotMakerFeatures.applicationId,
        BotMakerFeatures.transformCreatedViaToApplicationId(coreData.getCreated_via()));

    if (coreData.isSetReply()) {
      Reply reply = coreData.getReply();
      if (reply.isSetIn_reply_to_status_id()) {
        builder.putValue(OPTIONAL, BotMakerFeatures.inReplyToStatusId,
            reply.getIn_reply_to_status_id());
      }
      if (reply.isSetIn_reply_to_user_id()) {
        builder.putValue(OPTIONAL, BotMakerFeatures.inReplyToUserId,
            reply.getIn_reply_to_user_id());
      }
      builder.putValue(OPTIONAL,
          BotMakerFeatures.inReplyToScreenName, reply.getIn_reply_to_screen_name());
    }

    if (coreData.isSetDirected_at_user()) {
      DirectedAtUser dau = coreData.getDirected_at_user();
      builder.putValue(OPTIONAL, BotMakerFeatures.directedAtUserId, dau.user_id);
      builder.putValue(OPTIONAL, BotMakerFeatures.directedAtScreenName, dau.screen_name);
    }

    if (coreData.isSetShare()) {
      Share share = coreData.getShare();
      builder.putValue(OPTIONAL, BotMakerFeatures.shareSourceStatusId, share.source_status_id);
      builder.putValue(OPTIONAL, BotMakerFeatures.shareSourceUserId, share.source_user_id);
      builder.putValue(OPTIONAL, BotMakerFeatures.parentStatusId, share.parent_status_id);
    }

    builder.putValue(REQUIRED, BotMakerFeatures.hasTakedown, coreData.has_takedown);
    builder.putValue(REQUIRED, BotMakerFeatures.nsfwUser, coreData.nsfw_user);
    builder.putValue(REQUIRED, BotMakerFeatures.nsfwAdmin, coreData.nsfw_admin);
    builder.putValue(REQUIRED, BotMakerFeatures.isNullcast, coreData.nullcast);

    if (coreData.isSetNarrowcast()) {
      builder.putValue(OPTIONAL, BotMakerFeatures.narrowcastLocations,
          coreData.getNarrowcast().getLocation());
    }

    builder.putValue(OPTIONAL, BotMakerFeatures.trackingId, coreData.tracking_id);
    builder.putValue(OPTIONAL, BotMakerFeatures.conversationId, coreData.conversation_id);
    builder.putValue(OPTIONAL, BotMakerFeatures.hasMedia, coreData.has_media);
    builder.putValue(OPTIONAL, BotMakerFeatures.placeId, coreData.place_id);

    if (tweet.isSetCommunities()) {
      builder.putValue(OPTIONAL, BotMakerFeatures.communityIds,
          tweet.communities.community_ids);
    }

    if (tweet.isSetEscherbird_entity_annotations()) {
      ImmutableList.Builder<Map<String, Long>> annotations = ImmutableList.builder();
      List<TweetEntityAnnotation> entityAnnotations =
          tweet.escherbird_entity_annotations.entity_annotations;
      for (TweetEntityAnnotation annotation : entityAnnotations) {
        ImmutableMap.Builder<String, Long> map = ImmutableMap.builder();
        map.put("domainId", annotation.domainId);
        map.put("groupId", annotation.groupId);
        map.put("entityId", annotation.entityId);
        annotations.add(map.build());
      }

      FeatureValue featureValue = new FeatureValue();
      featureValue.setStrLongMapListValue(annotations.build());
      FeatureData feature = new FeatureData();
      feature.setFeatureValue(featureValue);
      builder.put(OPTIONAL, BotMakerFeatures.tweetEntityAnnotation, feature);
    }
  }

  private void processUrlEntities(FeatureMapBuilder builder, List<UrlEntity> urlEntities)
      throws FeatureExtractionException {
    List<String> urls = Lists.newArrayListWithCapacity(urlEntities.size());
    List<String> expandedUrls = Lists.newArrayListWithCapacity(urlEntities.size());
    List<Long> clickCounts = Lists.newArrayListWithCapacity(urlEntities.size());

    for (UrlEntity url : urlEntities) {
      if (url.isSetUrl()) {
        urls.add(url.getUrl());
      }
      if (url.isSetExpanded()) {
        expandedUrls.add(url.getExpanded());
      }
    }

    builder.putCollection(OPTIONAL, BotMakerFeatures.urls, urls, String.class);
    builder.putCollection(OPTIONAL, BotMakerFeatures.expandedUrls, expandedUrls, String.class);
    builder.putCollection(OPTIONAL, BotMakerFeatures.urlClickCounts, clickCounts, Long.class);
  }

  private void processMentionEntities(FeatureMapBuilder builder, List<MentionEntity> mentions)
      throws FeatureExtractionException {
    List<String> screenNames = Lists.newArrayListWithCapacity(mentions.size());
    List<Long> ids = Lists.newArrayListWithCapacity(mentions.size());

    for (MentionEntity mention : mentions) {
      screenNames.add(mention.getScreen_name());
      ids.add(mention.getUser_id());
      if (mention.isSetUser_id()) {
        victimIds.add(mention.getUser_id());
      }
    }

    builder.putCollection(OPTIONAL, BotMakerFeatures.mentionIds, ids, Long.class);
    builder.putCollection(OPTIONAL, BotMakerFeatures.mentionScreenNames, screenNames, String.class);
  }

  private void processHashtagEntities(FeatureMapBuilder builder, List<HashtagEntity> hashtags)
      throws FeatureExtractionException {
    List<String> texts = Lists.newArrayListWithCapacity(hashtags.size());

    for (HashtagEntity hashtag : hashtags) {
      if (hashtag.isSetText()) {
        texts.add(hashtag.getText());
      }
    }

    builder.putCollection(OPTIONAL, BotMakerFeatures.hashtags, texts, String.class);
  }

  private void processCashtagEntities(FeatureMapBuilder builder, List<CashtagEntity> cashtags)
      throws FeatureExtractionException {
    List<String> texts = Lists.newArrayListWithCapacity(cashtags.size());

    for (CashtagEntity cashtag : cashtags) {
      if (cashtag.isSetText()) {
        texts.add(cashtag.getText());
      }
    }

    builder.putCollection(OPTIONAL, BotMakerFeatures.cashtags, texts, String.class);
  }

  private void processDeviceSource(FeatureMapBuilder builder, DeviceSource source)
      throws FeatureExtractionException {

    if (isTweetEvent) {
      builder.putValue(OPTIONAL, BotMakerFeatures.tweetSource, source.display);
    } else {
      Map<String, FeatureData> submap = FeatureExtractionHelper
          .toFeatureMapFunction(source).apply();
      for (Map.Entry<String, FeatureData> entry : submap.entrySet()) {
        builder.putValue(OPTIONAL, "Source." + entry.getKey(), entry.getValue());
      }
    }
  }

  private void processCounts(FeatureMapBuilder builder, StatusCounts counts)
      throws FeatureExtractionException {
    builder.putValue(OPTIONAL, BotMakerFeatures.retweetCount, counts.retweet_count);
    builder.putValue(OPTIONAL, BotMakerFeatures.replyCount, counts.reply_count);
    builder.putValue(OPTIONAL, BotMakerFeatures.favoriteCount, counts.favorite_count);
  }

  private void processMediaTags(FeatureMapBuilder builder, TweetMediaTags mediaTags)
      throws FeatureExtractionException {
    Iterable<MediaTag> tags = Iterables.concat(mediaTags.getTag_map().values());
    Iterable<MediaTag> tagsWithUserIds = Iterables.filter(tags, tag -> tag.isSetUser_id());
    Set<Long> mediaTaggedUserIds =
        Sets.newHashSet(Iterables.transform(tagsWithUserIds, tag -> tag.getUser_id()));
    builder.putCollection(
        OPTIONAL,
        BotMakerFeatures.mediaTaggedUserIds,
        mediaTaggedUserIds,
        Long.class);
    victimIds.addAll(mediaTaggedUserIds);
  }

  private static void processMedia(FeatureMapBuilder builder, List<MediaEntity> mediaList)
      throws FeatureExtractionException {

    List<String> altTexts = new ArrayList<>();
    for (MediaEntity mediaEntity : mediaList) {
      if (mediaEntity.isSetAdditional_metadata()) {
        AdditionalMetadata additionalMetadata = mediaEntity.getAdditional_metadata();
        if (additionalMetadata.isSetAlt_text()) {
          AltText altText = additionalMetadata.getAlt_text();
          if (altText.isSetText()) {
            altTexts.add(altText.getText());
          }
        }
      }
    }

    if (!altTexts.isEmpty()) {
      builder.putCollection(OPTIONAL, BotMakerFeatures.mediaAltTexts, altTexts, String.class);
    }
  }

  private void processNoteTweet(FeatureMapBuilder builder, NoteTweet noteTweet)
      throws FeatureExtractionException {
    builder.putValue(OPTIONAL, BotMakerFeatures.noteTweetId, noteTweet.id);
  }

  private void processArticle(FeatureMapBuilder builder, Article article)
      throws FeatureExtractionException {
    builder.putValue(OPTIONAL, BotMakerFeatures.articleId, article.id);
  }

  @Override
  public void apply(FeatureMapBuilder builder) throws Exception {
    builder.putValue(REQUIRED, BotMakerFeatures.tweetId, tweet.getId());

    if (tweet.isSetCore_data()) {
      processCoreData(builder, tweet.getCore_data());
    }

    if (tweet.isSetUrls()) {
      processUrlEntities(builder, tweet.getUrls());
    }

    if (tweet.isSetMentions()) {
      processMentionEntities(builder, tweet.getMentions());
    }

    if (tweet.isSetHashtags()) {
      processHashtagEntities(builder, tweet.getHashtags());
    }

    if (tweet.isSetCashtags()) {
      processCashtagEntities(builder, tweet.getCashtags());
    }

    if (tweet.isSetDevice_source()) {
      processDeviceSource(builder, tweet.getDevice_source());
    }

    if (tweet.isSetCounts()) {
      processCounts(builder, tweet.getCounts());
    }

    if (tweet.isSetMedia_tags()) {
      processMediaTags(builder, tweet.getMedia_tags());
    }

    if (tweet.isSetMedia()) {
      processMedia(builder, tweet.getMedia());
    }

    if (tweet.isSetNote_tweet()) {
      processNoteTweet(builder, tweet.getNote_tweet());
    }

    if (tweet.isSetArticle()) {
      processArticle(builder, tweet.getArticle());
    }

    if (isTweetEvent) {
      builder.putValue(REQUIRED, BotMakerFeatures.eventId, tweet.getId());
      builder.putCollection(REQUIRED, BotMakerFeatures.victimIds, victimIds, Long.class);
    }
  }
}
