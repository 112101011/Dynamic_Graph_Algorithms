#ifndef EULER_TOUR_TREE_IMPL_CUH_
#define EULER_TOUR_TREE_IMPL_CUH_

#include <cstdint>
#include <sys/types.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>

#include "Allocator.cuh"
#include "EulerTourTree.cuh"
#include "SkipList.cuh"
#define BLOCK_SIZE 128
#define WARP_SIZE 32

using namespace skip_list;

template <typename GraphT, typename ValueT>
__global__ static void
BatchLinkKernelPhase1(ett::EulerTourContext<GraphT, ValueT> ETContext,
                      uint32_t *Src, uint32_t *Dst, uint32_t NumberOfEdges,
                      uint64_t *SortBuffer) {
  uint32_t ThreadId = blockDim.x * blockIdx.x + threadIdx.x;
  uint32_t ThreadsN = blockDim.x * gridDim.x;
  uint32_t LaneId = threadIdx.x & 0x1F;
  uint32_t N = ((NumberOfEdges + WARP_SIZE - 1) >> 5) << 5;
  using SLAllocatorCtxtT =
      typename ett::EulerTourContext<GraphT, ValueT>::SLAllocatorCtxtT;
  using EdgeAddressStoreT =
      typename ett::EulerTourContext<GraphT, ValueT>::EdgeAddressStoreT;
  using EdgeDynAllocCtxtT = typename EdgeAddressStoreT::EdgeDynAllocCtxt;

  SLAllocatorCtxtT &Allocator = ETContext.GetSkipListAllocator();
  EdgeAddressStoreT &EAS = ETContext.GetEdgeAddressStore();
  EdgeDynAllocCtxtT &EdgeDynAllocCtxt = EAS.GetEdgeDynAllocCtxt();

  for (uint32_t I = ThreadId; I < N; I += ThreadsN) {
    SkipListElement<ValueT> *UV = nullptr;
    SkipListElement<ValueT> *VU = nullptr;
    allocator::AddressT UVAddr = INVALID_ADDRESS;
    allocator::AddressT VUAddr = INVALID_ADDRESS;
    uint32_t Source = UINT32_MAX, Destination = UINT32_MAX;

    if (I < NumberOfEdges) {
      UV = SkipListElement<ValueT>::Create(Allocator, ThreadId, MAX_LEVEL,
                                           Src[I], Dst[I], 0);
      VU = SkipListElement<ValueT>::Create(Allocator, ThreadId, MAX_LEVEL,
                                           Dst[I], Src[I], 0);
      UVAddr = Allocator.GetAllocatorAddr(UV);
      VUAddr = Allocator.GetAllocatorAddr(VU);
      UV->SetTwin(VUAddr);
      VU->SetTwin(UVAddr);

      Source = Src[I];
      Destination = Dst[I];
      SortBuffer[(I << 1)] =
          (static_cast<uint64_t>(Source) << 32) | Destination;
      SortBuffer[(I << 1) + 1] =
          (static_cast<uint64_t>(Destination) << 32) | Source;
    }

    bool ToInsert = !(UV == nullptr || VU == nullptr);
    EAS.InsertEdge(ToInsert, LaneId, Source, Destination, UVAddr,
                   EdgeDynAllocCtxt);

    EAS.InsertEdge(ToInsert, LaneId, Destination, Source, VUAddr,
                   EdgeDynAllocCtxt);
  }
}

template <typename GraphT, typename ValueT>
__global__ static void
BatchLinkKernelPhase2(ett::EulerTourContext<GraphT, ValueT> ETContext,
                      uint32_t *Src, uint32_t *Dst, uint32_t NumberOfEdges,
                      bool *VerticesToSplitFlags) {
  uint32_t ThreadId = blockDim.x * blockIdx.x + threadIdx.x;
  uint32_t ThreadsN = blockDim.x * gridDim.x;
  using SLAllocatorCtxtT =
      typename ett::EulerTourContext<GraphT, ValueT>::SLAllocatorCtxtT;
  using EdgeAddressStoreT =
      typename ett::EulerTourContext<GraphT, ValueT>::EdgeAddressStoreT;
  using EdgeDynAllocCtxtT = typename EdgeAddressStoreT::EdgeDynAllocCtxt;

  auto SplitAtSuccessor =
      [&ETContext, &VerticesToSplitFlags] __device__(uint32_t Vertex) -> void {
    allocator::AddressT VertexAddr = ETContext.GetVertexNodes()[Vertex];
    SkipListElement<ValueT> *V = reinterpret_cast<SkipListElement<ValueT> *>(
        ETContext.GetSkipListAllocator().GetPtr(VertexAddr));
    SkipListElement<ValueT> *VNext = V->GetNext();

    if (VNext != nullptr) {
      ETContext.GetSuccessors()[Vertex] = VNext;
      VerticesToSplitFlags[Vertex] = true;
    }
  };

  for (uint32_t I = ThreadId; I < NumberOfEdges; I += ThreadsN) {
    if (I < NumberOfEdges) {
      SplitAtSuccessor(Src[I]);
      SplitAtSuccessor(Dst[I]);
    }
  }
}

template <typename GraphT, typename ValueT>
__global__ static void Split(ett::EulerTourContext<GraphT, ValueT> ETContext,
                             uint32_t *VerticesToSplit, uint32_t Count) {
  uint32_t ThreadId = blockDim.x * blockIdx.x + threadIdx.x;
  uint32_t ThreadsN = blockDim.x * gridDim.x;
  using SLAllocatorCtxtT =
      typename ett::EulerTourContext<GraphT, ValueT>::SLAllocatorCtxtT;
  using EdgeAddressStoreT =
      typename ett::EulerTourContext<GraphT, ValueT>::EdgeAddressStoreT;
  using EdgeDynAllocCtxtT = typename EdgeAddressStoreT::EdgeDynAllocCtxt;

  allocator::AddressT *VertexNodes = ETContext.GetVertexNodes();
  SLAllocatorCtxtT &Allocator = ETContext.GetSkipListAllocator();

  for (uint32_t I = ThreadId; I < Count; I += ThreadsN) {
    uint32_t Vert = VerticesToSplit[I];
    allocator::AddressT VertexAddr = ETContext.GetVertexNodes()[Vert];

    SkipListElement<ValueT> *V = reinterpret_cast<SkipListElement<ValueT> *>(
        Allocator.GetPtr(VertexAddr));
    V->Split();
  }
}

template <typename GraphT, typename ValueT>
__global__ static void
BatchLinkKernelPhase3(ett::EulerTourContext<GraphT, ValueT> ETContext,
                      uint64_t *SortedEdges, uint32_t NumberOfEdges) {
  uint32_t ThreadId = blockDim.x * blockIdx.x + threadIdx.x;
  uint32_t ThreadsN = blockDim.x * gridDim.x;
  uint32_t LaneId = threadIdx.x & 0x1F;

  using SLAllocatorCtxtT =
      typename ett::EulerTourContext<GraphT, ValueT>::SLAllocatorCtxtT;
  using EdgeAddressStoreT =
      typename ett::EulerTourContext<GraphT, ValueT>::EdgeAddressStoreT;
  using AdjacencyContext = typename EdgeAddressStoreT::EdgeHashContext;
  using Iterator = typename AdjacencyContext::Iterator;

  EdgeAddressStoreT &TheEdgeAddressStore = ETContext.GetEdgeAddressStore();
  SLAllocatorCtxtT &TheAllocator = ETContext.GetSkipListAllocator();

  uint32_t N = ((NumberOfEdges + WARP_SIZE - 1) >> 5) << 5;
  uint64_t Edge;
  uint32_t Src, Dst;
  bool FirstThread, LastThread;
  uint64_t EdgePrev, EdgeNext;
  uint32_t SrcPrev, DstPrev, SrcNext, DstNext;
  bool IsSubChunkBeg, IsSubChunkEnd, NotSubChunkEnd, AlwaysSearch;

  allocator::AddressT UVAddr;
  allocator::AddressT VUAddr;
  allocator::AddressT UNextVNextAddr;
  SkipListElement<ValueT> *UV;
  SkipListElement<ValueT> *VU;
  SkipListElement<ValueT> *UNextVNext;

  for (uint32_t I = ThreadId; I < N; I += ThreadsN) {
    if (I < NumberOfEdges) {
      Edge = SortedEdges[I];
      Src = Edge >> 32;
      Dst = Edge & UINT32_MAX;

      FirstThread = (I == 0);
      EdgePrev = FirstThread ? UINT64_MAX : SortedEdges[I - 1];
      SrcPrev = EdgePrev >> 32;
      DstPrev = EdgePrev & UINT32_MAX;

      LastThread = (I == (NumberOfEdges - 1));
      EdgeNext = LastThread ? UINT64_MAX : SortedEdges[I + 1];
      SrcNext = EdgeNext >> 32;
      DstNext = EdgeNext & UINT32_MAX;

      IsSubChunkBeg = FirstThread || (Src != SrcPrev);
      IsSubChunkEnd = LastThread || (Src != SrcNext);
      NotSubChunkEnd = !IsSubChunkEnd;
      AlwaysSearch = true;
    }

    AlwaysSearch = (I < NumberOfEdges) && AlwaysSearch;
    TheEdgeAddressStore.SearchEdge(AlwaysSearch, LaneId, Dst, Src, VUAddr);
    if (I < NumberOfEdges) {
      VU = reinterpret_cast<SkipListElement<uint32_t> *>(
          TheAllocator.GetPtr(VUAddr));
      UV = reinterpret_cast<SkipListElement<uint32_t> *>(
          TheAllocator.GetPtr(VU->GetTwin()));
    }

    NotSubChunkEnd = (I < NumberOfEdges) && NotSubChunkEnd;
    TheEdgeAddressStore.SearchEdge(NotSubChunkEnd, LaneId, SrcNext, DstNext,
                                   UNextVNextAddr);

    if (I < NumberOfEdges) {
      UNextVNext = NotSubChunkEnd
                       ? reinterpret_cast<SkipListElement<uint32_t> *>(
                             TheAllocator.GetPtr(UNextVNextAddr))
                       : nullptr;

      if (IsSubChunkBeg) {
        SkipListElement<ValueT> *UU =
            reinterpret_cast<SkipListElement<uint32_t> *>(
                TheAllocator.GetPtr(ETContext.GetVertexNodes()[Src]));
        JoinSkipLists(UU, UV);
      }

      SkipListElement<ValueT> *UVNext =
          IsSubChunkEnd ? ETContext.GetSuccessors()[Src] : UNextVNext;
      JoinSkipLists(VU, UVNext);
    }
  }
}

#if 0
template <typename EdgeAddressStoreT, typename ValueT>
__global__ static void PopulateDirectedEdges(
    ett::EulerTourContext<EdgeAddressStoreT, ValueT> ETContext, uint32_t *Src,
    uint32_t *Dst, uint32_t NumberOfEdges,
    SkipListElement<ValueT> **DirectedEdges) {
  uint32_t ThreadId = blockDim.x * blockIdx.x + threadIdx.x;
  uint32_t ThreadsN = gridDim.x * blockDim.x;

  using AdjacencyContext = typename EdgeAddressStoreT::EdgeHashContext;
  using Iterator = typename AdjacencyContext::Iterator;
  using SLAllocatorCtxtT =
      typename ett::EulerTourContext<EdgeAddressStoreT,
                                     ValueT>::SLAllocatorCtxtT;

  EdgeAddressStoreT &TheEdgeAddressStore = ETContext.GetEdgeAddressStore();
  SLAllocatorCtxtT &Allocator = ETContext.GetAllocatorCtxt();

  for (uint32_t I = ThreadId;
       I < (NumberOfEdges + (NumberOfEdges & (WARP_SIZE - 1))); I += ThreadsN) {
    bool ToSearch = (I < NumberOfEdges);
    uint32_t Source = ToSearch ? Src[I] : 0xFFFFFFFFu;
    uint32_t Destination = ToSearch ? Dst[I] : 0xFFFFFFFFu;
    Iterator UVIter = TheEdgeAddressStore.Search(ToSearch, Source, Destination);
    Iterator VUIter = TheEdgeAddressStore.Search(ToSearch, Destination, Source);

    DirectedEdges[(I << 1)] = Allocator.GetPtr(*UVIter);
    DirectedEdges[(I << 1) + 1] = Allocator.GetPtr(*VUIter);
  }
}

template <typename EdgeAddressStoreT, typename ValueT>
__global__ static void RemoveFromEdgeAddressStore(
    ett::EulerTourContext<EdgeAddressStoreT, ValueT> ETContext, uint32_t *Src,
    uint32_t *Dst, uint32_t NumberOfEdges) {
  uint32_t ThreadId = blockDim.x * blockIdx.x + threadIdx.x;
  uint32_t ThreadsN = gridDim.x * blockDim.x;
  EdgeAddressStoreT &TheEdgeAddressStore = ETContext.GetEdgeAddressStore();

  for (uint32_t I = ThreadId;
       I < (NumberOfEdges + (NumberOfEdges & (WARP_SIZE - 1))); I += ThreadsN) {
    bool ToDelete = (I < NumberOfEdges);
    TheEdgeAddressStore.Delete(ToDelete, Src[I], Dst[I]);
    TheEdgeAddressStore.Delete(ToDelete, Dst[I], Src[I]);
  }
}

template <typename EdgeAddressStoreT, typename ValueT>
__global__ static void
FindJoinRightsPhase1(ett::EulerTourContext<EdgeAddressStoreT, ValueT> ETContext,
                     SkipListElement<ValueT> **DirectedEdges,
                     uint32_t NumberOfEdges) {
  uint32_t ThreadId = blockDim.x * blockIdx.x + threadIdx.x;
  uint32_t ThreadsN = gridDim.x * blockDim.x;

  using SLAllocatorCtxtT =
      typename ett::EulerTourContext<EdgeAddressStoreT,
                                     ValueT>::AllocatorContext;
  SLAllocatorCtxtT &Allocator = ETContext.GetAllocatorCtxt();

  for (uint32_t I = ThreadId; I < NumberOfEdges; I += ThreadsN) {
    SkipListElement<ValueT> *Twin =
        Allocator.GetPtr(DirectedEdges[I].GetTwin());
    SkipListElement<ValueT> *Next = Twin->GetNext();
    DirectedEdges[I]->SetNextEdge(Next->IsMarked() ? Next : nullptr);
  }
}

template <typename EdgeAddressStoreT, typename ValueT>
__global__ static void
FindJoinRightsPhase2(ett::EulerTourContext<EdgeAddressStoreT, ValueT> ETContext,
                     SkipListElement<ValueT> **DirectedEdges,
                     SkipListElement<ValueT> **LastMarked,
                     uint32_t NumberOfEdges) {
  uint32_t ThreadId = blockDim.x * blockIdx.x + threadIdx.x;
  uint32_t ThreadsN = gridDim.x * blockDim.x;

  using SLAllocatorCtxtT =
      typename ett::EulerTourContext<EdgeAddressStoreT,
                                     ValueT>::AllocatorContext;
  SLAllocatorCtxtT &Allocator = ETContext.GetAllocatorCtxt();
#if 0
  // TODO: Fix Error: Expression must have a class type 
  for (uint32_t I = ThreadId; I < NumberOfEdges; I += ThreadsN) {
    SkipListElement<ValueT> *Ptr = DirectedEdges[I];
    SkipListElement<ValueT> *T;
    /* TODO: Needs optimization */
    while ((T = Ptr.GetNext()) != NULL)
      Ptr = T;

    LastMarked[I] = Ptr;
  }
#endif
}

template <typename EdgeAddressStoreT, typename ValueT>
__global__ static void
FindJoinRightsPhase3(ett::EulerTourContext<EdgeAddressStoreT, ValueT> ETContext,
                     SkipListElement<ValueT> **LastMarked,
                     SkipListElement<ValueT> **Result, uint32_t NumberOfEdges) {
  uint32_t ThreadId = blockDim.x * blockIdx.x + threadIdx.x;
  uint32_t ThreadsN = gridDim.x * blockDim.x;

  using SLAllocatorCtxtT =
      typename ett::EulerTourContext<EdgeAddressStoreT,
                                     ValueT>::AllocatorContext;
  SLAllocatorCtxtT &Allocator = ETContext.GetAllocatorCtxt();

  for (uint32_t I = ThreadId; I < NumberOfEdges; I += ThreadsN) {
    SkipListElement<ValueT> *Twin = Allocator.GetPtr(LastMarked[I].GetTwin());
    Result[I] = Twin->GetNext();
  }
}

template <typename ValueT>
__global__ static void CutTourEdges(SkipListElement<ValueT> **DirectedEdges,
                                    uint32_t NumberOfEdges) {
  uint32_t ThreadId = blockDim.x * blockIdx.x + threadIdx.x;
  uint32_t ThreadsN = gridDim.x * blockDim.x;

  for (uint32_t I = ThreadId; I < NumberOfEdges; I += ThreadsN) {
    DirectedEdges[I]->Split();
    SkipListElement<ValueT> *Previous = DirectedEdges[I]->GetPrevious();

    if (!Previous)
      Previous->Split();
  }
}

template <typename ValueT>
__global__ static void JoinTourEdges(SkipListElement<ValueT> **JoinLefts,
                                     SkipListElement<ValueT> **JoinRights,
                                     uint32_t NumberOfEdges) {
  uint32_t ThreadId = blockDim.x * blockIdx.x + threadIdx.x;
  uint32_t ThreadsN = gridDim.x * blockDim.x;

  for (uint32_t I = ThreadId; I < ThreadsN; I += ThreadsN) {
    if (!JoinLefts[I]->IsMarkedToDelete()) {
      JoinSkipLists(JoinLefts[I], JoinRights[I]);
    }
  }
}

#endif

__global__ void FindVerticesToSplit(bool *VerticesToSplitFlags,
                                    uint32_t VertexN, uint32_t *VerticesToSplit,
                                    uint32_t *Counter) {
  uint32_t ThreadId = blockDim.x * blockIdx.x + threadIdx.x;
  uint32_t ThreadsN = blockDim.x * gridDim.x;
  uint32_t N = ((VertexN + WARP_SIZE - 1) >> 5) << 5;
  uint32_t LaneId = threadIdx.x & 0x1F;

  for (uint32_t I = ThreadId; I < N; I += ThreadsN) {
    bool HasVertex = false;
    if (I < VertexN)
      HasVertex = VerticesToSplitFlags[I];

    uint32_t BitSet = __ballot_sync(0xFFFFFFFF, HasVertex);
    uint32_t Start = 0;
    if (LaneId == 0)
      Start = atomicAdd(Counter, __popc(BitSet));

    Start = __shfl_sync(0xFFFFFFFF, Start, 0, 32);

    if (HasVertex) {
      uint32_t Offset = __popc(__brev(BitSet) & (0xFFFFFFFF << (32 - LaneId)));
      VerticesToSplit[Start + Offset] = I;
    }
  }
}

template <typename EdgeAddressStoreT, typename ValueT>
void ett::EulerTour<EdgeAddressStoreT, ValueT>::BatchLink(
    uint32_t *Src, uint32_t *Dst, uint32_t NumberOfEdges,
    uint32_t NumberOfVertices) {
  uint32_t NumberOfBlocks = (NumberOfEdges + BLOCK_SIZE - 1) / BLOCK_SIZE;
  uint64_t *SortBuffer = nullptr;
  bool *VerticesToSplitFlags = nullptr;
  uint32_t *VerticesToSplit = nullptr;
  uint32_t Count = 0;
  uint32_t *CountDev;

  CHECK_CUDA_ERROR(
      cudaMalloc(&SortBuffer, sizeof(uint64_t) * (NumberOfEdges << 1)));
  CHECK_CUDA_ERROR(
      cudaMalloc(&VerticesToSplitFlags, sizeof(bool) * NumberOfVertices));
  CHECK_CUDA_ERROR(
      cudaMemset(VerticesToSplitFlags, 0x00, sizeof(bool) * NumberOfVertices));
  CHECK_CUDA_ERROR(cudaMalloc(&CountDev, sizeof(uint32_t)));
  CHECK_CUDA_ERROR(cudaMemset(CountDev, 0x00, sizeof(uint32_t)));
  CHECK_CUDA_ERROR(
      cudaMalloc(&VerticesToSplit, sizeof(uint32_t) * NumberOfVertices));
  CHECK_CUDA_ERROR(
      cudaMemset(VerticesToSplit, 0x00, sizeof(uint32_t) * NumberOfVertices));

  BatchLinkKernelPhase1<EdgeAddressStoreT, ValueT>
      <<<NumberOfBlocks, BLOCK_SIZE>>>(GetContext(), Src, Dst, NumberOfEdges,
                                       SortBuffer);

  BatchLinkKernelPhase2<EdgeAddressStoreT, ValueT>
      <<<NumberOfBlocks, BLOCK_SIZE>>>(GetContext(), Src, Dst, NumberOfEdges,
                                       VerticesToSplitFlags);

  cudaDeviceSynchronize();

  FindVerticesToSplit<<<256, 256>>>(VerticesToSplitFlags, NumberOfVertices,
                                    VerticesToSplit, CountDev);
  cudaDeviceSynchronize();

  CHECK_CUDA_ERROR(
      cudaMemcpy(&Count, CountDev, sizeof(uint32_t), cudaMemcpyDeviceToHost));

  Split<EdgeAddressStoreT, ValueT>
      <<<256, 256>>>(GetContext(), VerticesToSplit, Count);
  cudaDeviceSynchronize();

  thrust::sort(thrust::device, SortBuffer, SortBuffer + (NumberOfEdges << 1),
               thrust::less<uint64_t>());

  BatchLinkKernelPhase3<EdgeAddressStoreT, ValueT>
      <<<(NumberOfBlocks << 1), BLOCK_SIZE>>>(GetContext(), SortBuffer,
                                              NumberOfEdges << 1);
  cudaDeviceSynchronize();

  CHECK_CUDA_ERROR(cudaFree(VerticesToSplitFlags));
  CHECK_CUDA_ERROR(cudaFree(CountDev));
  CHECK_CUDA_ERROR(cudaFree(SortBuffer));
}

#if 0
template <typename EdgeAddressStoreT, typename ValueT>
void ett::EulerTour<EdgeAddressStoreT, ValueT>::BatchCut(
    uint32_t *Src, uint32_t *Dst, uint32_t NumberOfEdges) {
  uint32_t NumberOfBlocks = (NumberOfEdges + BLOCK_SIZE - 1) / BLOCK_SIZE;
  thrust::device_vector<SkipListElement<ValueT> *> DirectedEdges{NumberOfEdges
                                                                 << 1};
  thrust::device_vector<SkipListElement<ValueT> *> JoinLefts{NumberOfEdges
                                                             << 1};
  thrust::device_vector<SkipListElement<ValueT> *> JoinRights{NumberOfEdges
                                                              << 1};
  thrust::device_vector<SkipListElement<ValueT> *> LastMarked{NumberOfEdges << 1};
  /* TODO Check LastMarked */

  PopulateDirectedEdges<EdgeAddressStoreT><<<NumberOfBlocks, BLOCK_SIZE>>>(
      GetContext(), Src, Dst, NumberOfEdges, DirectedEdges.data().get());
  cudaDeviceSynchronize();

  RemoveFromEdgeAddressStore<EdgeAddressStoreT>
      <<<NumberOfBlocks, BLOCK_SIZE>>>(GetContext(), Src, Dst, NumberOfEdges);
  cudaDeviceSynchronize();

  thrust::transform(thrust::device, DirectedEdges,
                    DirectedEdges + (NumberOfEdges << 1), JoinLefts,
                    [] __device__(SkipListElement<ValueT> * SLE) {
                      SLE->MarkToDelete();
                      return SLE->GetPrevious();
                    });

  FindJoinRightsPhase1<EdgeAddressStoreT>
      <<<(NumberOfBlocks << 1), BLOCK_SIZE>>>(
          GetContext(), DirectedEdges.data().get(), NumberOfEdges << 1);
  cudaDeviceSynchronize();

  FindJoinRightsPhase2<EdgeAddressStoreT>
      <<<(NumberOfBlocks << 1), BLOCK_SIZE>>>(
          GetContext(), DirectedEdges.data().get(), LastMarked.data().get(),
          NumberOfEdges << 1);
  cudaDeviceSynchronize();

  FindJoinRightsPhase3<EdgeAddressStoreT>
      <<<(NumberOfBlocks << 1), BLOCK_SIZE>>>(
          GetContext(), LastMarked.data().get(), JoinRights.data().get(),
          NumberOfEdges << 1);
  cudaDeviceSynchronize();

  CutTourEdges(DirectedEdges.data().get(), NumberOfEdges << 1);
  cudaDeviceSynchronize();

  JoinTourEdges(JoinLefts.data().get(), JoinRights.data().get());
  cudaDeviceSynchronize();
}

#endif

template <typename ValueT>
__global__ static void
CreateVertexNodes(allocator::AllocatorContext TheAllocator,
                  allocator::AddressT *VertexNodes, uint32_t VertexN) {
  uint32_t ThreadId = blockDim.x * blockIdx.x + threadIdx.x;
  uint32_t ThreadsN = gridDim.x * blockDim.x;

  for (uint32_t TheVertex = ThreadId; TheVertex < VertexN;
       TheVertex += ThreadsN) {
    SkipListElement<ValueT> *SLE = SkipListElement<ValueT>::Create(
        TheAllocator, ThreadId, MAX_LEVEL, TheVertex, TheVertex,
        0); /* TODO: Add Check for SLE == nullptr. Also change value. */
    VertexNodes[TheVertex] = SLE->GetPoolAddr();
    MakeSelfLoop(SLE, TheAllocator);
  }
}

__global__ static void CreateTwinEdges(uint32_t *Src, uint32_t *Dst,
                                       uint32_t EdgesN, uint32_t *TwinEdgesSrc,
                                       uint32_t *TwinEdgesDst) {
  uint32_t ThreadsId = blockDim.x * blockIdx.x + threadIdx.x;
  uint32_t ThreadsN = gridDim.x * blockDim.x;

  for (uint32_t I = ThreadsId; I < EdgesN; I += ThreadsN) {
    uint32_t U = Src[I];
    uint32_t V = Dst[I];
    uint64_t *TSrc = reinterpret_cast<uint64_t *>(TwinEdgesSrc + (I << 1));
    uint64_t *TDst = reinterpret_cast<uint64_t *>(TwinEdgesDst + (I << 1));

    *TSrc = (static_cast<uint64_t>(V) << 32) | U;
    *TDst = (static_cast<uint64_t>(U) << 32) | V;
  }
}

template <typename EdgeAddressStoreT, typename ValueT>
void ett::EulerTour<EdgeAddressStoreT, ValueT>::PopulateVertexNodes() {
  CreateVertexNodes<ValueT><<<1024, 1024>>>(
      SkipListAllocator.GetAllocatorCtxt(), VertexNodes, VertexN);
}

template <typename EdgeAddressStoreT, typename ValueT>
__global__ static void
Query(ett::EulerTourContext<EdgeAddressStoreT, ValueT> ETContext, uint32_t *Src,
      uint32_t *Dst, uint32_t EdgesN, bool *IsConnected) {
  uint32_t ThreadId = blockDim.x * blockIdx.x + threadIdx.x;
  uint32_t ThreadsN = gridDim.x * blockDim.x;

  using AllocatorCtxtT =
      typename ett::EulerTourContext<EdgeAddressStoreT,
                                     ValueT>::SLAllocatorCtxtT;
  AllocatorCtxtT &TheAllocator = ETContext.GetAllocatorCtxt();
  allocator::AddressT *VertexNodes = ETContext.GetVertexNodes();

  for (uint32_t I = ThreadId; I < EdgesN; I += ThreadsN) {
    allocator::AddressT UAddr = VertexNodes[Src[I]];
    SkipListElement<ValueT> *UU = TheAllocator.GetPtr(UAddr);

    allocator::AddressT VAddr = ETContext.GetVertexNodes()[Dst[I]];
    SkipListElement<ValueT> *VV = TheAllocator.GetPtr(VAddr);

    IsConnected = (UU->FindRepresentative() == VV->FindRepresentative());
  }
}

template <typename EdgeAddressStoreT, typename ValueT>
void ett::EulerTour<EdgeAddressStoreT, ValueT>::BatchQuery(
    uint32_t *Src, uint32_t *Dst, uint32_t NumberOfEdges, bool *IsConnected) {
  uint32_t NumberOfBlocks = (NumberOfEdges + BLOCK_SIZE - 1) / BLOCK_SIZE;
  Query<EdgeAddressStoreT><<<NumberOfBlocks, BLOCK_SIZE>>>(
      GetContext(), Src, Dst, NumberOfEdges, IsConnected);
  cudaDeviceSynchronize();
}

#endif // EULER_TOUR_TREE_IMPL_CUH_