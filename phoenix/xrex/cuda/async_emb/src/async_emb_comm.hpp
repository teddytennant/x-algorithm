#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <nccl.h>

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "async_emb_kernel.hpp"
#include "comm_utils.h"

namespace xai::kernels::async_emb {

struct PipelineSpec {
  int64_t tokens_per_rank = 0;
  int64_t shard_width = 0;
  int64_t num_unique = 0;
  int64_t emb_width = 0;
};

struct ArenaLayout {
  size_t block_bytes = 0;
  size_t index_bytes = 0;
  size_t accum_bytes = 0;
  size_t token_ids_all = 0;
  size_t lookup_send = 0;
  size_t grad_accum = 0;
  size_t recv = 0;
  size_t segment_ids_all = 0;
  size_t update_stage = 0;
  size_t unique_tokens = 0;
  size_t row_sq_sums = 0;
  size_t scalars = 0;
  size_t total = 0;

  static ArenaLayout build(const PipelineSpec& spec, int world_size);
};

struct LookupJob {
  const int32_t* token_ids;
  const __nv_bfloat16* table;
  int64_t vocab_rows;
};

struct UpdateJob {
  __nv_bfloat16* table;
  int64_t vocab_rows;
};

struct ReducedGradients {
  const float* row_grads;
  const float* row_sq_sums;
  const int32_t* unique_tokens;
  UpdateScalars* scalars;
  int64_t num_unique;
  int64_t shard_width;
};

using ApplyUpdateRule =
    std::function<void(const ReducedGradients&, const UpdateJob&, cudaStream_t)>;

class CudaEvent {
 public:
  CudaEvent();
  ~CudaEvent();

  CudaEvent(const CudaEvent&) = delete;
  CudaEvent& operator=(const CudaEvent&) = delete;

  operator cudaEvent_t() const { return event_; }

 private:
  cudaEvent_t event_ = nullptr;
};

class CudaStream {
 public:
  CudaStream();
  ~CudaStream();

  CudaStream(const CudaStream&) = delete;
  CudaStream& operator=(const CudaStream&) = delete;

  operator cudaStream_t() const { return stream_; }
  void release() { stream_ = nullptr; }

 private:
  cudaStream_t stream_ = nullptr;
};

class DeviceArena {
 public:
  explicit DeviceArena(size_t bytes);
  ~DeviceArena();

  DeviceArena(const DeviceArena&) = delete;
  DeviceArena& operator=(const DeviceArena&) = delete;

  int8_t* data() const { return data_; }
  void release() { data_ = nullptr; }

 private:
  int8_t* data_ = nullptr;
};

class AsyncEmbContext : public common::CommContext {
 public:
  enum class Operation { Lookup, Update };

  AsyncEmbContext(uint64_t context_id, int world_size, PipelineSpec spec);
  ~AsyncEmbContext() override;

  std::string name() const override { return name_; }
  bool isInitialized() const override { return initialized_.load(); }
  std::vector<std::vector<uint8_t>> reset(int rank, int world_size) override;
  void handshake(std::vector<std::vector<uint8_t>> data) override;

  const PipelineSpec& spec() const { return spec_; }
  const ArenaLayout& layout() const { return layout_; }
  int rank() const { return rank_; }
  int worldSize() const { return world_size_; }
  int8_t* arena() const { return arena_.data(); }

  void armLookup(LookupJob job, cudaStream_t main_stream);
  void stageUpdate(
      const __nv_bfloat16* grads,
      const int32_t* segment_ids,
      const int32_t* unique_tokens,
      const int32_t* pending,
      cudaStream_t main_stream
  );
  void armUpdate(
      UpdateJob job,
      ApplyUpdateRule apply,
      cudaStream_t main_stream,
      const int32_t* logical_step = nullptr
  );
  void warmupUpdateRule(const ApplyUpdateRule& apply);

  uint64_t armedStep(Operation operation) const;
  void streamConsumeDone(cudaStream_t stream, Operation operation);
  bool hostWaitDone(Operation operation, uint64_t step);

  std::vector<uint8_t> snapshot(size_t offset, size_t bytes) const;

  void abort();
  void ensureHealthy() const;

 private:
  enum class TablePhase { None, Lookup, Update };

  struct PipelineState {
    CudaEvent input_ready;
    CudaEvent done;
    uint64_t armed_step = 0;
    uint64_t consumed_step = 0;
    bool in_flight = false;
    std::chrono::steady_clock::time_point submitted_at;
  };

  PipelineState& pipeline(Operation operation);
  const PipelineState& pipeline(Operation operation) const;
  void bindTable(Operation operation, const void* table);
  void recordInputReady(PipelineState& pipeline, cudaStream_t main_stream);
  void recordDone(PipelineState& pipeline);
  void enqueueLookup(const LookupJob& job);
  void enqueueUpdate(const UpdateJob& job, const ApplyUpdateRule& apply);

  void submitNccl(ncclResult_t result, const char* operation);
  void waitNcclReady(const char* operation);
  void warmupCollectives();
  ncclResult_t lockedQueryAsyncError(ncclResult_t* async_result) const;
  bool commHealthy() const;

  void watchdogMain();
  bool waitSideStream(int timeout_ms);
  bool waitAbortedSideStream(int timeout_ms);

  void abortCommunicator(const std::string& reason);

  const PipelineSpec spec_;
  const std::string name_;
  const ArenaLayout layout_;
  const int world_size_;
  const std::chrono::seconds watchdog_timeout_;

  DeviceArena arena_;
  CudaStream side_stream_;
  int device_ = -1;

  ncclComm_t comm_ = nullptr;
  void* arena_registration_ = nullptr;
  mutable std::recursive_mutex comm_mu_;

  int rank_ = -1;
  std::atomic<bool> initialized_{false};
  std::atomic<bool> failed_{false};
  std::atomic<bool> shutdown_{false};
  std::thread watchdog_;
  std::string failure_;

  mutable std::mutex state_mu_;
  std::mutex arm_mu_;
  PipelineState lookup_;
  PipelineState update_;
  bool staged_ = false;

  TablePhase last_table_phase_ = TablePhase::None;
  const void* last_table_ = nullptr;
};

}
