#ifndef SKIP_LIST_CUH_
#define SKIP_LIST_CUH_

#include "Allocator.cuh"
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/scan.h>
#include <unistd.h>

#define MAX_LEVEL 20

namespace skip_list {

struct GenRand {
private:
  thrust::default_random_engine RandEng;
  thrust::uniform_int_distribution<uint32_t> UniDist;

public:
  __device__ GenRand(uint32_t MaxBound) : RandEng{}, UniDist{0, MaxBound} {}
  __device__ uint32_t operator()(int Idx) {
    RandEng.discard(Idx);
    return UniDist(RandEng);
  }
};

template <typename ValueT> class SkipListElement;

template <typename ValueT>
__device__ void JoinSkipLists(SkipListElement<ValueT> *FirstEnd,
                              SkipListElement<ValueT> *SecondStart);

template <typename ValueT>
__device__ void MakeSelfLoop(SkipListElement<ValueT> *Element,
                             allocator::AllocatorContext &TheAllocator);

template <typename ValueTy> class SkipListElement {
public:
  using ValueT = ValueTy;
  static constexpr float EXPECTATION = 0.5F;

  static __device__ __forceinline__ SkipListElement<ValueT> *
  Create(allocator::AllocatorContext &TheAllocator, uint32_t ThreadId,
         uint32_t MaxLevel_, uint32_t U, uint32_t V, const ValueT &Value) {
    SkipListElement *SLE = nullptr;
#if 0
    GenRand RNG(MaxLevel_);
    uint32_t MaxLevel = RNG(ThreadId);
#endif
    uint32_t MaxLevel = (U + V + ThreadId) % (MAX_LEVEL + 1);

    allocator::AddressT AllocAddr = TheAllocator.Allocate(
        ((sizeof(SkipListElement<ValueTy>) + (MaxLevel + 1) * (LevelInfoSize) +
          sizeof(uint32_t) - 1) /
         sizeof(uint32_t)));

    SLE = reinterpret_cast<SkipListElement<ValueT> *>(
        TheAllocator.GetPtr(AllocAddr));
    SLE->Info = (MaxLevel << 1);
    SLE->PoolIndex = TheAllocator.GetPoolIndex();
    SLE->Twin = INVALID_ADDRESS;
#if 0   
    SLE->ToUpdateLevel = (MaxLevel + 1);
#endif
    SLE->U = U;
    SLE->V = V;

    for (int I = 0; I < (MaxLevel + 1); ++I) {
      SLE->Levels[I] = {INVALID_ADDRESS, INVALID_ADDRESS};
    }

    return SLE;
  }

  __device__ SkipListElement<ValueTy> *GetPrevious(uint32_t Level = 0);
  __device__ SkipListElement<ValueTy> *GetNext(uint32_t Level = 0);
  __device__ SkipListElement<ValueTy> *FindRepresentative();

  __device__ SkipListElement<ValueTy> *Split();

  static __device__ __forceinline__ SkipListElement<ValueTy> *
  GetPtr(uint32_t PoolIndex, allocator::AddressT Addr) {
    return reinterpret_cast<SkipListElement<ValueTy> *>(
        Addr == INVALID_ADDRESS ? nullptr
                                : (AllocationPools[PoolIndex] + Addr));
  }

public:
  __device__ SkipListElement *SearchLeftParent(uint32_t Level);
  __device__ SkipListElement *SearchRightParent(uint32_t Level);

  __device__ allocator::AddressT PrevAtomicCAS(uint32_t Level,
                                               allocator::AddressT OldPrev,
                                               allocator::AddressT NewPrev);
  __device__ allocator::AddressT NextAtomicCAS(uint32_t Level,
                                               allocator::AddressT OldNext,
                                               allocator::AddressT NewNext);

public:
  __device__ __forceinline__ allocator::AddressT GetPoolAddr() {
    return reinterpret_cast<uint32_t *>(this) - AllocationPools[PoolIndex];
  }

  __device__ __forceinline__ allocator::AddressT GetTwin() { return Twin; }
  __device__ __forceinline__ void SetTwin(allocator::AddressT TwinAddr) {
    Twin = TwinAddr;
  }

  static constexpr uint32_t MaxLevelOffset = 1;

  __device__ __forceinline__ uint32_t GetU() { return U; }
  __device__ __forceinline__ uint32_t GetV() { return V; }

  /* TODO Check whether Info is correctly stored and retrieved */
  __device__ __forceinline__ uint32_t GetMaxLevel() {
    return (Info >> MaxLevelOffset);
  }
#if 0
  __device__ __forceinline__ uint32_t GetToUpdateLevel() {
    return ToUpdateLevel;
  }

  __device__ __forceinline__ void SetToUpdateLevel(uint32_t Level) {
    ToUpdateLevel = Level;
  }

  __device__ __forceinline__ void ToUpdate() {
    return ToUpdateLevel < (GetMaxLevel() + 1);
  }

  __device__ __forceinline__ void MarkToDelete() { atomicOr(&Info, 0x01u); }

  __device__ __forceinline__ bool IsMarkedToDelete() {
    return (Info & 0x01u) != 0;
  }

  __device__ __forceinline__ void UnmarkToDelete() {
    atomicAnd(&Info, ~(0x01u));
  }

  __device__ __forceinline__ ValueT GetValue(uint32_t Level) {
    return Levels[Level].Value;
  }
#endif
private:
  /* Structure of Info:
   *
   * │ 30            1 | 0              0 │
   * ┌─────────────────┬──────────────────┐
   * │ MaxLevel        │ ToDelete         │
   * ├─────────────────┼──────────────────┤
   * │ 31 bits         │ 1 bits           │
   * └─────────────────┴──────────────────┘
   */

  uint32_t Info;
  uint32_t PoolIndex;
  allocator::AddressT Twin;
#if 0
  uint32_t ToUpdateLevel;
#endif
  uint32_t U;
  uint32_t V;

public:
  struct {
    allocator::AddressT Previous;
    allocator::AddressT Next;
  } Levels[1];

  friend __device__ void JoinSkipLists<>(SkipListElement<ValueTy> *FirstEnd,
                                         SkipListElement<ValueTy> *SecondStart);
  friend __device__ void
  MakeSelfLoop<>(SkipListElement<ValueTy> *Element,
                 allocator::AllocatorContext &TheAllocator);

public:
  static constexpr size_t LevelInfoSize = sizeof(Levels[0]);
};

#include "SkipListImpl.cuh"

} // namespace skip_list

#endif // SKIP_LIST_CUH_
