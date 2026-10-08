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

#include "comm_utils.h"
#include "row_emb_kernel.hpp"

namespace xai::kernels::row_emb {

struct PipelineSpec {
  int64_t tokens_per_rank = 0;
  int64_t emb_width = 0;
  int64_t rows_per_rank = 0;
  int64_t recv_capacity = 0;
};

struct LookupSlot {
  int32_t* ids = nullptr;
  int32_t* unique = nullptr;
  int32_t* inverse = nullptr;
  int32_t* order = nullptr;
  int32_t* offsets = nullptr;
  int32_t* n_unique = nullptr;
  int32_t* send_counts = nullptr;
  int32_t* recv_counts = nullptr;
  int32_t* req_ids = nullptr;
  int32_t* host_counts = nullptr;
  cudaEvent_t counts_ready = nullptr;
  std::vector<int64_t> send_offsets;
  std::vector<int64_t> recv_offsets;
  int64_t received = 0;
};

struct ArenaLayout {
  size_t slot_ids[2] = {0, 0};
  size_t slot_unique[2] = {0, 0};
  size_t slot_inverse[2] = {0, 0};
  size_t slot_order[2] = {0, 0};
  size_t slot_offsets[2] = {0, 0};
  size_t slot_n_unique[2] = {0, 0};
  size_t slot_send_counts[2] = {0, 0};
  size_t slot_recv_counts[2] = {0, 0};
  size_t slot_req_ids[2] = {0, 0};
  size_t unique_ws = 0;
  size_t unique_ws_bytes = 0;
  size_t recv_rows = 0;
  size_t grads_by_owner = 0;
  size_t exchange = 0;
  size_t owner_unique = 0;
  size_t owner_inverse = 0;
  size_t owner_order = 0;
  size_t owner_offsets = 0;
  size_t owner_count = 0;
  size_t owner_row_sq_sums = 0;
  size_t scalars = 0;
  size_t total = 0;

  static ArenaLayout build(const PipelineSpec& spec, int world_size);
};

struct LookupJob {
  const int32_t* token_ids;
  const bf16* table;
  int64_t rows_local;
};

struct UpdateJob {
  bf16* table;
  int64_t rows_local;
};

struct ReducedGradients {
  const bf16* rows;
  const int32_t* order;
  const int32_t* offsets;
  const int32_t* unique;
  const int32_t* n_unique;
  int64_t capacity;
  const float* row_sq_sums;
  UpdateScalars* scalars;
  Ownership ownership;
  int rank;
  int64_t width;
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

class RowEmbContext : public common::CommContext {
 public:
  enum class Operation { Lookup, Update };

  RowEmbContext(uint64_t context_id, int world_size, PipelineSpec spec);
  ~RowEmbContext() override;

  std::string name() const override { return name_; }
  bool isInitialized() const override { return initialized_.load(); }
  std::vector<std::vector<uint8_t>> reset(int rank, int world_size) override;
  void handshake(std::vector<std::vector<uint8_t>> data) override;

  const PipelineSpec& spec() const { return spec_; }
  const ArenaLayout& layout() const { return layout_; }
  Ownership ownership() const { return Ownership{world_size_, spec_.rows_per_rank}; }
  int rank() const { return rank_; }
  int worldSize() const { return world_size_; }
  int8_t* arena() const { return arena_.data(); }

  void armLookup(LookupJob job, cudaStream_t main_stream);
  bool flushLookupTail(bool block);
  void finishLookup(cudaStream_t stream, bf16* embeddings);
  void stageUpdate(const bf16* grads, const int32_t* pending, cudaStream_t main_stream);
  void armUpdate(
      UpdateJob job,
      ApplyUpdateRule apply,
      cudaStream_t main_stream,
      const int32_t* logical_step = nullptr
  );
  void finishUpdate(cudaStream_t stream);
  void warmupUpdateRule(const ApplyUpdateRule& apply);

  uint64_t armedStep(Operation operation) const;
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
    bool tail_pending = false;
    std::chrono::steady_clock::time_point submitted_at;
  };

  PipelineState& pipeline(Operation operation);
  const PipelineState& pipeline(Operation operation) const;
  void bindTable(Operation operation, const void* table);

  void enqueueLookupTail(LookupSlot& slot, const LookupJob& job);
  void enqueueUpdate(int slot_index, const UpdateJob& job, const ApplyUpdateRule& apply);
  void exchange(
      const void* send,
      const std::vector<int64_t>& send_offsets,
      void* recv,
      const std::vector<int64_t>& recv_offsets,
      size_t elem_bytes,
      ncclDataType_t type,
      const char* operation
  );
  void waitCountsReady(LookupSlot& slot);
  bool countsReady(LookupSlot& slot);

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
  const int world_size_;
  const ArenaLayout layout_;
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
  LookupJob pending_lookup_{};
  bool staged_ = false;
  int staged_slot_ = -1;

  LookupSlot slots_[2];

  TablePhase last_table_phase_ = TablePhase::None;
  const void* last_table_ = nullptr;
};

}
