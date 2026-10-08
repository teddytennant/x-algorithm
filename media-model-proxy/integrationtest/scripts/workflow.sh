#!/bin/bash


function usage {
  echo "Usage: $0 [--trace] [service options]"
  echo
  echo "  --prod           run against production Media Model Proxy service (default is staging)"
  echo "  --help           this help"
  echo
  exit 1
}

LOG_LEVEL="INFO"
ENV="staging"
MP_ENV="prod"
MEDIA_PATH_PREFIX=""

while [ -n "$1" ]; do
  case "$1" in
    --prod)
      ENV="prod"
      shift
      ;;
    --help)
      usage
      exit
      ;;
    *)
      break
      ;;
  esac
done

source "$(git rev-parse --show-toplevel)/media-understanding/tools/ci/scripts/common.sh"

set -o nounset
set -eu

DC="$(hostname | cut -d- -f1)"
IMAGES_DIR="${REPO_ROOT}/../../packages/integration-images"
DECIDER_OVERLAY="/usr/local/config/overlays/media-understanding/integrationtest/${ENV}/${DC}/decider_overlay.yml"

if [ -f "${DECIDER_OVERLAY}" ] &&
   grep -E 'disable_media_model_proxy_integration_tests:.*?default_availability:\s*10000[^0-9]' "${DECIDER_OVERLAY}"
then
  echo "Integration test is disabled via decider, exiting now."
  post_slack_text ":warning: Media Model Proxy ${ENV} integration test skipped via decider."
  exit 0
fi

DTAB_ADD=($(IFS=$'\n' cat << EOF
  /srv=>/$/nil
  /srv=>/srv#/staging
  /s/media-model-proxy/media-model-proxy=>/cluster/local/media-model-proxy/${ENV}/media-model-proxy
  /s/photurkey/mediainfo=>/cluster/local/photurkey/${MP_ENV}/mediainfo
  /s/image-fetcher-service/image_fetcher=>/cluster/local/image-fetcher-service/${MP_ENV}/image_fetcher
  /s/user-image-service/uis=>/cluster/local/user-image-service/${MP_ENV}/uis
  /s/media-analysis-service/mas=>/cluster/local/media-analysis-service/${MP_ENV}/mas
  /s/test-user-service/test-user-service=>/cluster/local/test-user-service/prod/test-user-service
EOF))
DTAB_ADD=$(IFS=';' ; echo "${DTAB_ADD[*]}")


JVM_OPTIONS=($(IFS=$'\n' cat << EOF
  -Dcom.twitter.finagle.util.loadServiceDenied=com.twitter.finagle.stats.OstrichStatsReceiver,com.twitter.finagle.stats.CommonsStatsReceiver
  -Dlogback.configurationFile=logback-local.xml
  -Dlog.level="${LOG_LEVEL}"
  -Dorg.apache.thrift.readLength=1572864
  -Ddtab.add=${DTAB_ADD}
  -Dmedia-model-proxy.disable=false
  -Dservice.identifier=media-model-proxy:integrationtest:${ENV}:${DC}
  -Dthrift.clientId=media-model-proxy.test
  -Dimages-dir=${IMAGES_DIR}
  -Dmedia-path-prefix=${MEDIA_PATH_PREFIX}
EOF))

BAZEL_OPTIONS=($(IFS=$'\n' cat << EOF
--test_timeout=900
--test_output=all
--test_env=RUN_LIVETESTS=1
EOF))

TARGET="media-understanding/media-model-proxy/integrationtest/src/test/scala/com/twitter/media_understanding/model_proxy/integrationtest"

CMD=(./bazel test --test_arg="--jvm_flags=${JVM_OPTIONS[*]}" "${BAZEL_OPTIONS[@]}" ${TARGET})

echo "Running:"
echo "${CMD[@]}"

"${CMD[@]}"
