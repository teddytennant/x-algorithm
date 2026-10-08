#pragma once

#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace xai::kernels::common {

struct GroupInfo {
  int64_t group_id;
  int64_t group_size;
  int64_t rank;
};

GroupInfo findGroupInfo(
    int64_t group_size, std::vector<int64_t> &flatten_replicas, int64_t global_device_id
);

class CommContext {
 public:
  virtual ~CommContext() = default;
  virtual std::vector<std::vector<uint8_t>> reset(int rank, int world_size) = 0;
  virtual void handshake(std::vector<std::vector<uint8_t>> data) = 0;
  virtual std::string name() const = 0;
  virtual bool isInitialized() const = 0;

  std::optional<std::string> init(GroupInfo info, int64_t local_gpu_index);
};

class CommContextRegistry {
 public:
  static CommContextRegistry &getInstance();

  void registerContext(std::shared_ptr<CommContext> ctx, uint64_t ctx_hash);

  std::shared_ptr<CommContext> getContext(uint64_t ctx_hash);

  void removeContext(uint64_t ctx_hash);

  size_t size() const;

 private:
  CommContextRegistry() = default;
  ~CommContextRegistry() = default;
  CommContextRegistry(const CommContextRegistry &) = delete;
  CommContextRegistry &operator=(const CommContextRegistry &) = delete;

  mutable std::mutex mutex_;
  std::unordered_map<uint64_t, std::shared_ptr<CommContext>> contexts_;
};

}
