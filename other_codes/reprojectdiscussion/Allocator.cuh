#ifndef ALLOCATOR_CUH_
#define ALLOCATOR_CUH_

#include <atomic>
#include <cassert>
#include <cstdint>

class Allocator;

#define MAX_ALLOCATION_COUNT 1024
#define INVALID_ADDRESS UINT32_MAX

#define CHECK_CUDA_ERROR(call)                                                 \
  do {                                                                         \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      printf("CUDA error at %s %d: %s\n", __FILE__, __LINE__,                  \
             cudaGetErrorString(err));                                         \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

std::atomic<uint32_t> NewAllocationPoolIndex;
__constant__ uint32_t *AllocationPools[MAX_ALLOCATION_COUNT];

namespace skip_list {
namespace allocator {

using AddressT = uint32_t;

class AllocatorContext {
public:
  AllocatorContext() {}
  __host__ __device__ AllocatorContext(uint32_t WordsN, uint32_t PoolIndex,
                                       uint32_t *CurrentOffsetPtr)
      : WordsN{WordsN}, PoolIndex{PoolIndex}, CurrentOffset{CurrentOffsetPtr} {}

  __host__ __device__ AllocatorContext(const AllocatorContext &Other)
      : WordsN{Other.WordsN}, PoolIndex{Other.PoolIndex},
        CurrentOffset{Other.CurrentOffset} {}

  __host__ __device__ AllocatorContext &
  operator=(const AllocatorContext &Other) {
    WordsN = Other.WordsN;
    PoolIndex = Other.PoolIndex;
    CurrentOffset = Other.CurrentOffset;
    return (*this);
  }

  __device__ __forceinline__ AddressT Allocate(uint32_t Size) {
    uint32_t Offset = atomicAdd(CurrentOffset, Size);
    uint32_t TheCurrentOffset = *CurrentOffset;
    return (TheCurrentOffset < WordsN) ? Offset : INVALID_ADDRESS;
  }

  __device__ __forceinline__ uint32_t GetPoolIndex() { return PoolIndex; }

  __device__ __forceinline__ void *GetPtr(uint32_t Offset = 0) {
    return reinterpret_cast<void *>(AllocationPools[PoolIndex] + Offset);
  }

  template <typename T>
  __device__ __forceinline__ AddressT GetAllocatorAddr(T *Addr) {
    return (reinterpret_cast<uint32_t *>(Addr) - AllocationPools[PoolIndex]);
  }

private:
  friend class Allocator;

  uint32_t WordsN;
  uint32_t PoolIndex;
  uint32_t *CurrentOffset;
};

class Allocator {
public:
  Allocator(uint32_t WordsN) : TheContext{0, 0, nullptr}, CurrentPool{nullptr} {
    uint32_t CurrentPoolIndex = NewAllocationPoolIndex.fetch_add(1);

    assert((CurrentPoolIndex < MAX_ALLOCATION_COUNT) &&
           "Requested allocation pool exceeeds maximum available !!");

    TheContext.WordsN = WordsN;
    TheContext.PoolIndex = CurrentPoolIndex;

    CHECK_CUDA_ERROR(cudaMalloc(&CurrentPool, sizeof(uint32_t) * WordsN));
    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(
        AllocationPools, &CurrentPool, sizeof(uint32_t *),
        CurrentPoolIndex * sizeof(uint32_t *), cudaMemcpyHostToDevice));

    CHECK_CUDA_ERROR(cudaMalloc(&TheContext.CurrentOffset, sizeof(uint32_t)));
    CHECK_CUDA_ERROR(
        cudaMemset(TheContext.CurrentOffset, 0x00, sizeof(uint32_t)));
  }

  ~Allocator() {
    CHECK_CUDA_ERROR(cudaFree(CurrentPool));
    CHECK_CUDA_ERROR(cudaFree(TheContext.CurrentOffset));
  }

  const AllocatorContext &GetAllocatorCtxt() { return TheContext; }

private:
  AllocatorContext TheContext;
  uint32_t *CurrentPool;
};

} // namespace allocator

} // namespace skip_list

#endif // ALLOCATOR_CUH_
