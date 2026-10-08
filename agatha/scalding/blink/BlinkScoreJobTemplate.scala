package com.twitter.agatha.scalding.blink

import com.twitter.algebird._
import com.twitter.agatha.scalding.AgathaFeaturesScalaDataset
import com.twitter.agatha.scalding.data.ActiveUserFeaturesConfig
import com.twitter.agatha.scalding.labels.AgathaLabelManager
import com.twitter.agatha.scalding.labels.nsfw.NSFWLabel
import com.twitter.agatha.scalding.labels.reports_per_fav.ReportsPerFav
import com.twitter.agatha.scalding.labels.spam_reports_per_fav.AllSpamReportsPerFav
import com.twitter.agatha.scalding.labels.spam_reports_per_fav.SpamReportsPerFav
import com.twitter.agatha.scalding.labels.spam_suspended.SpamSuspendedLabel
import com.twitter.agatha.scalding.util.GetActiveUsers
import com.twitter.agatha.thriftscala.FlattenedBlinkScore
import com.twitter.agatha.thriftscala.UserPrediction
import com.twitter.dal.client.dataset.KeyValDALDataset
import com.twitter.health.platform_manipulation.calibration.PercentileCalibrationApplicator
import com.twitter.health.platform_manipulation.calibration.PercentileCalibrator
import com.twitter.hub.agatha.labels.LabelManager
import com.twitter.hub.agatha.thriftscala.FeatureClass
import com.twitter.hub.agatha.thriftscala.FeatureSource
import com.twitter.mediascience.scalding.extensions.MediaScienceBatch
import com.twitter.scalding._
import com.twitter.scalding_internal.dalv2.DAL
import com.twitter.scalding_internal.dalv2.DALWrite.D
import com.twitter.scalding_internal.dalv2.DALWrite._
import com.twitter.scalding_internal.dalv2.remote_access.ExplicitLocation
import com.twitter.scalding_internal.dalv2.remote_access.ProcAtla
import com.twitter.scalding_internal.job.TwitterExecutionApp
import com.twitter.scalding_internal.job.analytics_batch.AnalyticsBatchExecution
import com.twitter.scalding_internal.job.analytics_batch.TwitterScheduledExecutionApp
import com.twitter.scalding_internal.multiformat.format.keyval.KeyVal
import java.util.TimeZone

trait BlinkScoreExtractionConfig {
  final implicit val tz: TimeZone = DateOps.UTC
  val historyDuration: Duration
  val labelManager: LabelManager
  val labelNames: Set[String]

  val labelFilterFeatureClasses: Map[String, Set[FeatureClass]] = Map.empty

  val labelMhImportDatasets: Map[String, KeyValDALDataset[KeyVal[Long, Double]]] = Map.empty

  def getUserPredictions(dateRange: DateRange): TypedPipe[UserPrediction]

  def writeUserBlinkScores(
    userBlinkScores: TypedPipe[(Long, String, Double)],
    dateRange: DateRange
  ): Execution[Unit]

  def writeUserBlinkScoreRecords(
    userBlinkScores: TypedPipe[FlattenedBlinkScore],
    dateRange: DateRange
  ): Execution[Unit] =
    writeUserBlinkScores(BaseBlinkScoreJob.toUserAverages(userBlinkScores), dateRange)

  def writeCalibratedUserBlinkScores(
    calibratedUserBlinkScores: TypedPipe[(Long, String, Double)],
    dateRange: DateRange
  ): Execution[Unit]
}

object BaseBlinkScoreJob {
  def apply(
    config: BlinkScoreExtractionConfig,
    passedDateRange: DateRange
  )(
    implicit uniqueId: UniqueID
  ): Execution[Unit] = {
    implicit val dateRange: DateRange =
      DateRange(passedDateRange.start - config.historyDuration, passedDateRange.end)

    val labelPriors = config.labelManager
      .getLabels(config.labelNames.map((_, dateRange)).toMap)
      .map {
        case (label, labelPipe) =>
          labelPipe
            .map(uL => uL.smoothedRatio.getOrElse(uL.positiveCount / uL.totalCount.toDouble))
            .map(AveragedValue(_))
            .sum
            .map(avg => (label, avg.value))
            .toTypedPipe
      }
      .reduce(_ ++ _)

    val userPredictions = config
      .getUserPredictions(dateRange)
      .filter(userPrediction => config.labelNames.contains(userPrediction.label))
      .filter(userPrediction =>
        !config.labelFilterFeatureClasses
          .getOrElse(userPrediction.label, Set.empty).contains(userPrediction.featureClass))

    val userScores = scoreUsers(userPredictions, labelPriors).forceToDisk

    val activeUserAverages = toUserAverages(userScores)
      .groupBy { case (userId, _, _) => userId }
      .join(GetActiveUsers(DateRange(dateRange.start, RichDate.now)).asKeys)
      .mapValues(_._1)
      .values

    Execution
      .zip(
        userScores.forceToDiskExecution,
        activeUserAverages.forceToDiskExecution,
      ).flatMap {
        case (userScores, activeUserAverages) =>
          percentileCalibrateUserAverages(
            toUserAverages(userScores),
            activeUserAverages,
            config.labelNames)
            .flatMap { calibratedUserAverages =>
              Execution
                .zip(
                  config.writeUserBlinkScoreRecords(userScores, dateRange),
                  config.writeCalibratedUserBlinkScores(calibratedUserAverages, dateRange)
                ).unit
            }
      }
  }

  type MomentSums = (Double, Double, Double, Double)

  def momentSums(featureClass: FeatureClass, m0: Double, m1: Double): MomentSums =
    if (featureClass.subclass == "inbound") (m0, m0 * m1, 0.0, 0.0)
    else (m0, m0 * m1, m0, m0 * m1)

  def blinkScore(sumM0M1: Double, sumM0: Double, prior: Double): Double =
    (sumM0M1 + math.log(prior)) / (sumM0 + 1)

  def toFlattenedBlinkScore(
    userId: Long,
    label: String,
    sums: MomentSums,
    prior: Double
  ): FlattenedBlinkScore = {
    val (sumM0, sumM0M1, outboundSumM0, outboundSumM0M1) = sums
    FlattenedBlinkScore(
      userId,
      label,
      blinkScore(sumM0M1, sumM0, prior),
      scoreOutbound = Some(blinkScore(outboundSumM0M1, outboundSumM0, prior)),
      support = Some(sumM0),
      supportOutbound = Some(outboundSumM0)
    )
  }

  def toUserAverages(
    userScores: TypedPipe[FlattenedBlinkScore]
  ): TypedPipe[(Long, String, Double)] =
    userScores.map(s => (s.userId, s.label, s.score))

  def scoreUsers(
    userPredictions: TypedPipe[UserPrediction],
    labelPriors: TypedPipe[(String, Double)]
  )(
    implicit uniqueId: UniqueID
  ): TypedPipe[FlattenedBlinkScore] = {
    userPredictions
      .flatMap { uP =>
        val m0 = uP.moments(0)
        val m1 = uP.moments(1)
        if (m1.isNaN && ((uP.userId % 123456789L) == 9470603L)) {
          Stat(StatKey("UnexpectedNaNSkipped", "Debug"))(uniqueId).inc()
          None
        } else {
          require(
            !m0.isNaN && !m1.isNaN,
            f"Unexpected NaN ${uP.label} ${uP.userId} ${m0} ${m1}, see PFM-1435"
          )
          Some(((uP.userId, uP.label), momentSums(uP.featureClass, m0, m1)))
        }
      }
      .group
      .sum
      .groupBy { case ((_, label), _) => label }
      .hashJoin(labelPriors)
      .mapValues {
        case (((userId, label), sums), prior) =>
          toFlattenedBlinkScore(userId, label, sums, prior)
      }
      .values
  }

  def percentileCalibrateUserAverages(
    userAverages: TypedPipe[(Long, String, Double)],
    userAveragesForCalibration: TypedPipe[(Long, String, Double)],
    labelNames: Set[String],
  )(
    implicit uniqueId: UniqueID
  ): Execution[TypedPipe[(Long, String, Double)]] = {
    val perLabelCalibratedUserAveragesExecs: Seq[Execution[TypedPipe[(Long, String, Double)]]] =
      labelNames.toSeq
        .map { labelName =>
          def filterToLabel(
            pipe: TypedPipe[(Long, String, Double)],
            statName: String
          ): TypedPipe[(Long, Double)] = {
            val stat = Stat(StatKey(statName, f"Label${labelName}"))(uniqueId)
            pipe.collect {
              case (userId, label, rawScore) if label == labelName => {
                stat.inc()
                (userId, rawScore)
              }
            }
          }

          val rawScores = filterToLabel(userAverages, "Total")
          val rawScoresForCalibration = filterToLabel(userAveragesForCalibration, "ForCalibration")

          val distinctPercentileCalibratedScoreStat =
            Stat(StatKey("DistinctPercentileCalibratedScore", "Calibration"))(uniqueId)
          val calibrationMapping =
            PercentileCalibrator
              .calibrateDistinct(
                rawScores = rawScoresForCalibration.map { case (_, rawScore) => rawScore },
                numPartitions = 1000,
                k = 12,
                partitionedScoresReducersOpt = Some(500)
              )
              .map { percentileCalibratedScore =>
                distinctPercentileCalibratedScoreStat.inc()
                percentileCalibratedScore
              }

          PercentileCalibrationApplicator
            .fromPercentileCalibratedScorePipe(calibrationMapping)
            .toOptionExecution
            .map(_.get)
            .map { percentileCalibrationApplicator =>
              rawScores
                .map {
                  case (userId, rawScore) =>
                    val pctScore =
                      percentileCalibrationApplicator.applyCalibration(rawScore)
                    (userId, labelName, pctScore)
                }
            }
        }

    Execution
      .sequence(perLabelCalibratedUserAveragesExecs).map { labelPipes =>
        labelPipes.reduce(_ ++ _)
      }
  }
}

class BlinkScoreJobTemplate(config: BlinkScoreExtractionConfig) extends TwitterExecutionApp {
  override def job: Execution[Unit] = Execution.withId { implicit uniqueId =>
    Execution.getArgs.flatMap { args =>
      implicit val tz: TimeZone = DateOps.UTC
      implicit val dp: DateParser = DateParser.default
      val argDateRange = DateRange.parse(args.list("date"))

      BaseBlinkScoreJob(config, argDateRange)
    }
  }
}

class ScheduledBlinkScoreJobTemplate(jobConfig: BlinkScoreExtractionConfig)
    extends TwitterScheduledExecutionApp {
  override def scheduledJob: Execution[Unit] = Execution.withId { implicit uniqueId =>
    Execution.getConfig.flatMap { execConfig =>
      val userDescription = execConfig.getArgs.required("batchDesc")
      val className = execConfig.getCascadingAppName.getOrElse(getClass.getCanonicalName)
      val batchDesc = s"$className:$userDescription"
      val args = execConfig.getArgs + (("batchDesc", Iterable(batchDesc)))

      AnalyticsBatchExecution(MediaScienceBatch.batchArgs(args)) { stateBirdDateRange: DateRange =>
        BaseBlinkScoreJob(jobConfig, stateBirdDateRange)
      }
    }
  }
}

trait BlinkScoreConfig extends BlinkScoreExtractionConfig {
  val pathSuffix: String
  override val historyDuration: Duration = Days(30)
  override val labelManager: LabelManager = AgathaLabelManager
  override val labelNames: Set[String] = Set(
    ReportsPerFav,
    SpamSuspendedLabel,
    AllSpamReportsPerFav,
    SpamReportsPerFav,
    NSFWLabel,
  ).map(_.labelName)

  override val labelFilterFeatureClasses: Map[String, Set[FeatureClass]] = Map(
    ReportsPerFav.labelName -> Set(
      FeatureClass(FeatureSource.BlockGraph, "inbound"),
      FeatureClass(FeatureSource.BlockGraph, "outbound"),
      FeatureClass(FeatureSource.FavGraph, "inbound"),
      FeatureClass(FeatureSource.AbuseReportGraph, "inbound"),
      FeatureClass(FeatureSource.LegitTattleGraph, "inbound"),
      FeatureClass(FeatureSource.LegitTattleGraph, "outbound"),
      FeatureClass(FeatureSource.NonLegitTattleNoBlocksGraph, "inbound"),
      FeatureClass(FeatureSource.NonLegitTattleNoBlocksGraph, "outbound"),
    ),
    SpamSuspendedLabel.labelName -> Set.empty,
    AllSpamReportsPerFav.labelName -> Set(
      FeatureClass(FeatureSource.FavGraph, "inbound"),
      FeatureClass(FeatureSource.LegitTattleGraph, "inbound"),
      FeatureClass(FeatureSource.LegitTattleGraph, "outbound"),
      FeatureClass(FeatureSource.NonLegitTattleNoBlocksGraph, "inbound"),
      FeatureClass(FeatureSource.NonLegitTattleNoBlocksGraph, "outbound"),
    ),
    SpamReportsPerFav.labelName -> Set(
      FeatureClass(FeatureSource.LegitTattleGraph, "inbound"),
      FeatureClass(FeatureSource.LegitTattleGraph, "outbound"),
      FeatureClass(FeatureSource.NonLegitTattleNoBlocksGraph, "inbound"),
      FeatureClass(FeatureSource.NonLegitTattleNoBlocksGraph, "outbound"),
    ),
    NSFWLabel.labelName -> Set.empty,
  ).mapValues(_ ++ ActiveUserFeaturesConfig.experimentalGraphs)

  override val labelMhImportDatasets: Map[String, KeyValDALDataset[KeyVal[Long, Double]]] = Map(
    AllSpamReportsPerFav.labelName -> AgathaMhImportAllSpamReportPerFavScalaDataset,
    ReportsPerFav.labelName -> AgathaMhImportReportsPerFavScalaDataset,
    SpamSuspendedLabel.labelName -> AgathaMhImportSpamSuspendedScalaDataset,
    SpamReportsPerFav.labelName -> AgathaMhImportSpamReportsPerFavScalaDataset,
  )

  override def getUserPredictions(dateRange: DateRange): TypedPipe[UserPrediction] =
    DAL
      .readMostRecentSnapshot(AgathaFeaturesScalaDataset, dateRange)
      .withRemoteReadPolicy(ExplicitLocation(ProcAtla))
      .toTypedPipe

  override def writeUserBlinkScores(
    userBlinkScores: TypedPipe[(Long, String, Double)],
    dateRange: DateRange
  ): Execution[Unit] =
    writeUserBlinkScoreRecords(
      userBlinkScores.map {
        case (userId, label, score) => FlattenedBlinkScore(userId, label, score)
      },
      dateRange)

  override def writeUserBlinkScoreRecords(
    userBlinkScores: TypedPipe[FlattenedBlinkScore],
    dateRange: DateRange
  ): Execution[Unit] = {
    userBlinkScores
      .shard(100)
      .writeDALSnapshotExecution(
        AgathaBlinkScoresSnapshotScalaDataset,
        D.Daily,
        D.Suffix(s"predictions/user/blinkScoresSnapshotDal$pathSuffix"),
        D.Parquet,
        dateRange.end
      )
  }

  override def writeCalibratedUserBlinkScores(
    calibratedUserBlinkScores: TypedPipe[(Long, String, Double)],
    dateRange: DateRange
  ): Execution[Unit] = {
    calibratedUserBlinkScores.forceToDiskExecution.flatMap { ubs =>
      val snapshotWrite = ubs
        .map {
          case (userId, label, score) => FlattenedBlinkScore(userId, label, score)
        }
        .shard(100)
        .writeDALSnapshotExecution(
          AgathaCalibratedBlinkScoresSnapshotScalaDataset,
          D.Daily,
          D.Suffix(s"predictions/user/calibratedBlinkScoresSnapshotDal$pathSuffix"),
          D.Parquet,
          dateRange.end
        )

      val mhWrites = labelNames.filter(labelMhImportDatasets.contains).map { labelName =>
        ubs
          .filter { case (_, label, _) => label == labelName }
          .map { case (userId, _, score) => KeyVal(userId, score) }
          .shard(20)
          .writeDALVersionedKeyValExecution(
            dataset = labelMhImportDatasets(labelName),
            pathLayout = D.Suffix(s"manhattan_import/predictions/user/$labelName$pathSuffix")
          )
      }

      Execution.zip(snapshotWrite, Execution.sequence(mhWrites.toSeq)).unit
    }
  }
}

object ProdBlinkScoreConfig extends BlinkScoreConfig {
  val pathSuffix = ""
}

object AdhocBlinkScoreConfig extends BlinkScoreConfig {
  val pathSuffix = "adhoc"
}

object AdhocBlinkScoreJob extends BlinkScoreJobTemplate(AdhocBlinkScoreConfig)
object ProdNonScheduledBlinkScoreJob extends BlinkScoreJobTemplate(ProdBlinkScoreConfig)
object ScheduledBlinkScoreJob extends ScheduledBlinkScoreJobTemplate(ProdBlinkScoreConfig)
