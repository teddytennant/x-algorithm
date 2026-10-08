#include "comm_utils.h"

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <unistd.h>

#include <cstdint>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "absl/log/log.h"

namespace nb = nanobind;

namespace xai::kernels::common {

void checkGpuIndexConsistency(
    const std::string &ctx_name,
    const GroupInfo &info,
    int64_t local_gpu_index,
    nb::object set_kv,
    nb::object get_kv
) {
  const int TIMEOUT_MS = 300000;
  std::string key =
      "xccl_" + ctx_name + "_group_id_" + std::to_string(info.group_id) + "_device_idx";

  if (info.rank == 0) {
    char gpu_index_byte = static_cast<char>(static_cast<uint8_t>(local_gpu_index));
    set_kv(key, nb::bytes(&gpu_index_byte, 1));
  } else {
    nb::object recv_obj = get_kv(key, TIMEOUT_MS);
    nb::bytes recv_bytes = nb::cast<nb::bytes>(recv_obj);
    if (recv_bytes.size() == 1) {
      int64_t local_gpu_index_rank_0 =
          static_cast<int64_t>(static_cast<uint8_t>(recv_bytes.c_str()[0]));
      if (local_gpu_index_rank_0 != local_gpu_index) {
        LOG(WARNING) << ctx_name << " context for group_id=" << info.group_id
                     << " are not all using the same local GPU index (not the "
                        "same ib device most likely)."
                     << " rank 0: " << local_gpu_index_rank_0 << ", rank " << info.rank << ": "
                     << local_gpu_index << ". this will cause cross-NIC and poor performance.";
      }
    }
  }
}

GroupInfo findGroupInfo(
    int64_t group_size, std::vector<int64_t> &flatten_replicas, int64_t global_device_id
) {
  for (int64_t group_id = 0; group_id < flatten_replicas.size() / group_size; ++group_id) {
    for (int64_t rank = 0; rank < group_size; ++rank) {
      if (flatten_replicas[group_id * group_size + rank] == global_device_id) {
        return {group_id, group_size, rank};
      }
    }
  }
  return {-1, -1, -1};
}

CommContextRegistry &CommContextRegistry::getInstance() {
  static CommContextRegistry instance;
  return instance;
}

void CommContextRegistry::registerContext(std::shared_ptr<CommContext> ctx, uint64_t ctx_hash) {
  std::lock_guard<std::mutex> lock(mutex_);
  contexts_[ctx_hash] = ctx;
}

std::shared_ptr<CommContext> CommContextRegistry::getContext(uint64_t ctx_hash) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = contexts_.find(ctx_hash);
  return (it != contexts_.end()) ? it->second : nullptr;
}

void CommContextRegistry::removeContext(uint64_t ctx_hash) {
  std::lock_guard<std::mutex> lock(mutex_);
  contexts_.erase(ctx_hash);
}

size_t CommContextRegistry::size() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return contexts_.size();
}

std::optional<std::string> CommContext::init(GroupInfo info, int64_t local_gpu_index) {
  nb::gil_scoped_acquire guard;

  std::string ctx_name = name();

  VLOG(1) << "Initializing context " << ctx_name << " for rank=" << info.rank
          << ", group_id=" << info.group_id << ", group_size=" << info.group_size;

  auto jax_dist = nb::module_::import_("jax._src.distributed");
  auto global_state = jax_dist.attr("global_state");
  auto client = global_state.attr("client");
  auto set_kv = client.attr("key_value_set_bytes");

  if (std::getenv("CI") != nullptr && std::string(std::getenv("CI")) == "true") {
    auto get_kv = client.attr("blocking_key_value_get_bytes");
    checkGpuIndexConsistency(ctx_name, info, local_gpu_index, set_kv, get_kv);
  }

  auto data_to_send = reset(info.rank, info.group_size);
  if (static_cast<int64_t>(data_to_send.size()) != info.group_size) {
    return "CommContext::reset returned " + std::to_string(data_to_send.size()) +
           " elements, but world_size is " + std::to_string(info.group_size);
  }

  const int TIMEOUT_MS = 300000;

  for (int64_t remote_rank = 0; remote_rank < info.group_size; ++remote_rank) {
    std::string key = "xccl_" + ctx_name + "_group_id_" + std::to_string(info.group_id) +
                      "_from_rank_" + std::to_string(info.rank) + "_to_rank_" +
                      std::to_string(remote_rank);

    nb::bytes send_bytes(
        reinterpret_cast<const char *>(data_to_send[remote_rank].data()),
        data_to_send[remote_rank].size()
    );
    set_kv(key, send_bytes);
  }

  std::vector<std::vector<uint8_t>> received_data(info.group_size);
  for (int remote_rank = 0; remote_rank < info.group_size; ++remote_rank) {
    std::string key = "xccl_" + ctx_name + "_group_id_" + std::to_string(info.group_id) +
                      "_from_rank_" + std::to_string(remote_rank) + "_to_rank_" +
                      std::to_string(info.rank);

    nb::object recv_obj = client.attr("blocking_key_value_get_bytes")(key, TIMEOUT_MS);
    nb::bytes recv_bytes = nb::cast<nb::bytes>(recv_obj);
    const uint8_t *data_ptr = reinterpret_cast<const uint8_t *>(recv_bytes.c_str());
    size_t data_len = recv_bytes.size();
    received_data[remote_rank].assign(data_ptr, data_ptr + data_len);
  }

  handshake(std::move(received_data));
  return std::nullopt;
}

}
