#ifndef EULER_TOUR_TREE_CUH_
#define EULER_TOUR_TREE_CUH_

#include "SkipList.cuh"
#include "SkipListImpl.cuh"

namespace ett {

template <typename GraphT, typename ValueT> class EulerTour;

template <typename GraphT, typename ValueT> class EulerTourContext;

template <typename GraphT, typename ValueT> class EulerTourContext {
public:
  using EdgeAddressStoreT = typename GraphT::GraphContextT;
  using SLAllocatorCtxtT = skip_list::allocator::AllocatorContext;

  __device__ __forceinline__ EdgeAddressStoreT &GetEdgeAddressStore() {
    return TheEdgeAddressStore;
  }

  __device__ __forceinline__ SLAllocatorCtxtT &GetSkipListAllocator() {
    return TheSkipListAllocator;
  }

  __device__ __forceinline__ skip_list::allocator::AddressT *GetVertexNodes() {
    return VertexNodes;
  }

  __device__ __forceinline__ skip_list::SkipListElement<ValueT> **
  GetSuccessors() {
    return Successors;
  }

  __host__ __device__
  EulerTourContext(const EulerTourContext<GraphT, ValueT> &Other)
      : TheEdgeAddressStore{Other.TheEdgeAddressStore},
        TheSkipListAllocator{Other.TheSkipListAllocator},
        VertexNodes{Other.VertexNodes}, Successors{Other.Successors} {}

  __host__ __device__ EulerTourContext<GraphT, ValueT> &
  operator=(const EulerTourContext<GraphT, ValueT> &Other) {
    TheEdgeAddressStore = Other.TheEdgeAddressStore;
    TheSkipListAllocator = Other.TheSkipListAllocator;
    VertexNodes = Other.VertexNodes;
    Successors = Other.Successors;
    return *(this);
  }

  EulerTourContext() {}

private:
  friend class EulerTour<GraphT, ValueT>;

  EdgeAddressStoreT TheEdgeAddressStore;
  SLAllocatorCtxtT TheSkipListAllocator;
  skip_list::allocator::AddressT *VertexNodes;
  skip_list::SkipListElement<ValueT> **Successors;
};

template <typename GraphT, typename ValueT> class EulerTour {
public:
  void BatchLink(uint32_t *Src, uint32_t *Dst, uint32_t NumberOfEdges,
                 uint32_t NumberOfVertices);
  void BatchCut(uint32_t *Src, uint32_t *Dst, uint32_t NumberOfEdges);
  void BatchQuery(uint32_t *Src, uint32_t *Dst, uint32_t NumberOfEdges,
                  bool *ConnectedStatus);

  using ETContextT = EulerTourContext<GraphT, ValueT>;

  ETContextT &GetContext() { return TheContext; }

  ~EulerTour() {
    CHECK_CUDA_ERROR(cudaFree(Successors));
    CHECK_CUDA_ERROR(cudaFree(VertexNodes));
  }

private:
  void PopulateVertexNodes();

public:
  EulerTour(uint32_t VertexN, GraphT &EdgeAddressStore)
      : VertexN{VertexN}, VertexNodes{nullptr}, Successors{nullptr},
        SkipListAllocator{
            ((VertexN * static_cast<uint32_t>(sizeof(uint32_t)) * 10) << 2)} {
    CHECK_CUDA_ERROR(cudaMalloc(&VertexNodes, sizeof(uint32_t) * VertexN));
    CHECK_CUDA_ERROR(cudaMalloc(
        &Successors, sizeof(skip_list::SkipListElement<ValueT> *) * VertexN));
    CHECK_CUDA_ERROR(
        cudaMemset(Successors, 0x00,
                   sizeof(skip_list::SkipListElement<ValueT> *) * VertexN));

    TheContext.TheEdgeAddressStore = EdgeAddressStore.GetDynamicGraphContext();
    TheContext.TheSkipListAllocator = SkipListAllocator.GetAllocatorCtxt();
    TheContext.VertexNodes = VertexNodes;
    TheContext.Successors = Successors;

    PopulateVertexNodes();
  }

private:
  uint32_t VertexN;
  skip_list::allocator::AddressT *VertexNodes;
  skip_list::SkipListElement<ValueT> **Successors;
  skip_list::allocator::Allocator SkipListAllocator;
  EulerTourContext<GraphT, ValueT> TheContext;
};

} // namespace ett

#include "EulerTourTreeImpl.cuh"

#endif // EULER_TOUR_TREE_CUH_
