#ifndef SKIP_LIST_IMPL_CUH_
#define SKIP_LIST_IMPL_CUH_

#include "Allocator.cuh"

template <typename ValueT>
__device__ __forceinline__ SkipListElement<ValueT> *
SkipListElement<ValueT>::GetPrevious(uint32_t Level) {
  return (Level <= GetMaxLevel()) ? SkipListElement<ValueT>::GetPtr(
                                        PoolIndex, Levels[Level].Previous)
                                  : nullptr;
}

template <typename ValueT>
__device__ __forceinline__ SkipListElement<ValueT> *
SkipListElement<ValueT>::GetNext(uint32_t Level) {
  return (Level <= GetMaxLevel())
             ? SkipListElement<ValueT>::GetPtr(PoolIndex, Levels[Level].Next)
             : nullptr;
}

template <typename ValueT>
__device__ __forceinline__ SkipListElement<ValueT> *
SkipListElement<ValueT>::SearchLeftParent(uint32_t Level) {
  SkipListElement<ValueT> *CurrentElement = this;
  uint32_t MaxLevel = CurrentElement->GetMaxLevel();
  /* Second Predicate: Micro-optiomization for reaching highest achievable level
   */
  if (Level > MaxLevel /* || Level == MAX_LEVEL*/)
    return nullptr;

  do {
    if (CurrentElement->GetMaxLevel() > Level)
      return CurrentElement;

    CurrentElement = SkipListElement<ValueT>::GetPtr(
        PoolIndex, CurrentElement->Levels[Level].Previous);
  } while (CurrentElement != nullptr && CurrentElement != this);

  return nullptr;
}

template <typename ValueT>
__device__ __forceinline__ SkipListElement<ValueT> *
SkipListElement<ValueT>::SearchRightParent(uint32_t Level) {
  SkipListElement<ValueT> *CurrentElement = this;
  uint32_t MaxLevel = CurrentElement->GetMaxLevel();
  /* Second Predicate: Micro-optiomization for reaching highest achievable level
   */
  if (Level > MaxLevel /* || Level == MAX_LEVEL */)
    return nullptr;

  do {
    if (CurrentElement->GetMaxLevel() > Level)
      return CurrentElement;

    CurrentElement = SkipListElement<ValueT>::GetPtr(
        PoolIndex, CurrentElement->Levels[Level].Next);
  } while (CurrentElement != nullptr && CurrentElement != this);

  return nullptr;
}

template <typename ValueT>
__device__ __forceinline__ allocator::AddressT
SkipListElement<ValueT>::PrevAtomicCAS(uint32_t Level,
                                       allocator::AddressT OldPrev,
                                       allocator::AddressT NewPrev) {
  return atomicCAS((&Levels[Level].Previous), OldPrev, NewPrev);
}

template <typename ValueT>
__device__ __forceinline__ allocator::AddressT
SkipListElement<ValueT>::NextAtomicCAS(uint32_t Level,
                                       allocator::AddressT OldNext,
                                       allocator::AddressT NewNext) {
  return atomicCAS((&Levels[Level].Next), OldNext, NewNext);
}

template <typename ValueT>
__device__ __forceinline__ SkipListElement<ValueT> *
SkipListElement<ValueT>::Split() {
  SkipListElement<ValueT> *LevelSuccessor = nullptr;
  SkipListElement<ValueT> *SLE = nullptr;
  SkipListElement<ValueT> *Current = this;

  allocator::AddressT Next = INVALID_ADDRESS;
  uint32_t CurrentLevel = 0;
#if DEGUG
  printf("[DEBUG 0]: %d %d %d %p %d\n", blockDim.x * blockIdx.x + threadIdx.x,
         Current->Levels[0].Next, Current->GetMaxLevel(),
         SkipListElement<ValueT>::GetPtr(PoolIndex, Current->Levels[0].Next),
         Current->NextAtomicCAS(CurrentLevel, Next, INVALID_ADDRESS));
#endif
  while (
      Current != nullptr &&
      (Next = Current->Levels[CurrentLevel].Next) != INVALID_ADDRESS &&
      (Next == Current->NextAtomicCAS(CurrentLevel, Next, INVALID_ADDRESS))) {
    __threadfence();
    LevelSuccessor = SkipListElement<ValueT>::GetPtr(PoolIndex, Next);
    if (CurrentLevel == 0)
      SLE = LevelSuccessor;
    LevelSuccessor->Levels[CurrentLevel].Previous = INVALID_ADDRESS;
    Current = Current->SearchLeftParent(CurrentLevel);
    ++CurrentLevel;
  }

  return SLE;
}

template <typename ValueT>
__device__ __forceinline__ SkipListElement<ValueT> *
SkipListElement<ValueT>::FindRepresentative() {
  SkipListElement<ValueT> *V = this;
  SkipListElement<ValueT> *T = nullptr;
  uint32_t CurrentLevel = GetMaxLevel();

  while ((T = V->SearchRightParent(CurrentLevel)) != nullptr) {
    V = T;
    ++CurrentLevel;
  }
  while ((T = V->SearchLeftParent(CurrentLevel)) != nullptr) {
    V = T;
    ++CurrentLevel;
  }

  SkipListElement<ValueT> *Repr = V, *Prev = nullptr, *Current = V;
  SkipListElement<ValueT> *Start = V;
  CurrentLevel = Repr->GetMaxLevel();
  uint32_t MinU = Current->GetU(), MinV = Current->GetV();

  // printf("[REPR]: this: (%d %d) Start: (%d %d) Repr: (%d %d)\n",
  // this->GetU(), this->GetV(),
  //        Start->GetU(), Start->GetV(), Repr->GetU(), Repr->GetV());

  do {
    if ((Current->GetU() < MinU) ||
        ((Current->GetU() == MinU) && (Current->GetV() < MinV))) {
      Repr = Current;
      MinU = Repr->GetU();
      MinV = Repr->GetV();
    }
    Current = Current->GetNext(CurrentLevel);
  } while (Current != Start);

  // printf("[REPR]: this: (%d %d) Start: (%d %d) Repr: (%d %d)\n",
  // this->GetU(), this->GetV(),
  //        Start->GetU(), Start->GetV(), Repr->GetU(), Repr->GetV());

  return Repr;
}

template <typename ValueT>
__device__ __forceinline__ void
JoinSkipLists(SkipListElement<ValueT> *FirstEnd,
              SkipListElement<ValueT> *SecondStart) {
  uint32_t CurrentLevel = 0;

  while (!(FirstEnd == nullptr || SecondStart == nullptr) &&
         (FirstEnd->Levels[CurrentLevel].Next == INVALID_ADDRESS) &&
         (FirstEnd->NextAtomicCAS(CurrentLevel, INVALID_ADDRESS,
                                  SecondStart->GetPoolAddr()) ==
          INVALID_ADDRESS)) {
    __threadfence();
    SecondStart->PrevAtomicCAS(CurrentLevel, INVALID_ADDRESS,
                               FirstEnd->GetPoolAddr());
    __threadfence();
    FirstEnd = FirstEnd->SearchLeftParent(CurrentLevel);
    SecondStart = SecondStart->SearchRightParent(CurrentLevel);

    ++CurrentLevel;
  }
}

template <typename ValueT>
__device__ __forceinline__ void
MakeSelfLoop(SkipListElement<ValueT> *Element,
             allocator::AllocatorContext &TheAllocator) {
  uint32_t MaxLevel = Element->GetMaxLevel();
  allocator::AddressT Addr = TheAllocator.GetAllocatorAddr(Element);
  for (uint32_t Level = 0; Level <= MaxLevel; ++Level) {
    Element->Levels[Level].Next = Element->Levels[Level].Previous = Addr;
  }
}

template <typename ValueT, typename Aggregator>
__device__ __forceinline__ ValueT AggregateValues(SkipListElement<ValueT> *L,
                                                  SkipListElement<ValueT> *R) {
  ValueT AggregateL{};
  ValueT AggregateR{R->GetValue()};
  Aggregator TheAggregator{};

  while (L != R) {
    uint32_t LMaxLevel = L->GetMaxLevel();
    uint32_t RMaxLevel = R->GetMaxLevel();
    uint32_t Min = min(LMaxLevel, RMaxLevel);

    if (LMaxLevel == Min) {
      AggregateL = TheAggregator(AggregateL, L->GetValue(LMaxLevel - 1));
      L = L->GetNext();
    } else {
      R = R->GetPrevious();
      AggregateR = TheAggregator(R->GetValue(RMaxLevel - 1), AggregateR);
    }
  }

  return TheAggregator(AggregateL, AggregateR);
}

#endif // SKIP_LIST_IMPL_CUH_
