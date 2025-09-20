#ifndef __HALO_BUFFER_CUH_
#define __HALO_BUFFER_CUH_
#include "grid_buffer.cuh"
#include "particle_buffer.cuh"
#include "settings.cuh"
#include <MnBase/Meta/Polymorphism.h>
//#include <cub/device/device_scan.cuh>

namespace mn {

using HaloGridBlocksDomain = compact_domain<int, config::g_max_halo_block>;
using halo_grid_blocks_ =
    structural<structural_type::dynamic,
               decorator<structural_allocation_policy::full_allocation,
                         structural_padding_policy::sum_pow2_align>,
               HaloGridBlocksDomain, attrib_layout::soa, grid_block_>;
#if (DEBUG_COUPLED_UP)
using grid_block_ =
    structural<structural_type::dense,
               decorator<structural_allocation_policy::full_allocation,
                         structural_padding_policy::sum_pow2_align>,
               BlockDomain, attrib_layout::soa, fg_, fg_, fg_, fg_,
              fg_, fg_, fg_, 
              fg_, fg_,
              fg_, fg_>; // mass, m(vel + dt*fint) (MLS), mvel (ASFLIP), 
                         // Vol, JBar
                         // mass_water, pressure_water
#else
using grid_block_ =
    structural<structural_type::dense,
               decorator<structural_allocation_policy::full_allocation,
                         structural_padding_policy::sum_pow2_align>,
               BlockDomain, attrib_layout::soa, fg_, fg_, fg_, fg_,
              fg_, fg_, fg_, 
              fg_, fg_>; // mass, m(vel + dt*fint) (MLS), mvel (ASFLIP), Vol, JBar
#endif
/// Halo Grid-Block structure
struct HaloGridBlocks {
  // Nest a structure holding HaloGridBlocksDomain and a pointer to device mem for blockids 3D coordinates
  struct HaloBuffer {
    Instance<halo_grid_blocks_> _grid;
    ivec3 *_blockids;
  };
  // TODO : Double check use of {} and (,) to intialize - JB 
  HaloGridBlocks(int numNeighbors)
      : numTargets{numNeighbors}, h_counts(numNeighbors, 0) {
    fmt::print("HaloGridBlocks constructor.\n");
    checkCudaErrors(cudaMalloc(&_counts, sizeof(uint32_t) * numTargets));
    _buffers.resize(numTargets);
  }
  ~HaloGridBlocks() {
    fmt::print("HaloGridBlocks destructor.\n");
  } 

  template <typename Allocator>
  void deallocate(Allocator allocator) {
    if (g_log_level > 2) fmt::print("HaloGridBlocks deallocate.\n");
    // for (int did = 0; did < numTargets; ++did){
    //   //_buffers[did]._grid.deallocate(allocator);
    //   //fmt::print("Deallocated _buffers[{}]._grid\n", did);
    //   if (_buffers[did]._blockids) { 
    //     //allocator.deallocate( _buffers[did]._blockids, h_counts[did] * sizeof(ivec3));
    //     //_buffers[did]._blockids = nullptr;
    //     fmt::print("Would deallocate _buffers[{}]._blockids\n", did);
    //   }
    //   else fmt::print("Could deallocate _buffers[{}]._blockids\n", did);
    // }
    if (_counts) {
      checkCudaErrors(cudaFree(_counts)); // Free device memory
      if (_counts) { _counts = nullptr; } // Clean-up host pointer
      if (g_log_level > 2) fmt::print("Deallocated _counts\n");
    }
  }
  template <typename Allocator>
  void temp_deallocate(Allocator allocator) {
    if (g_log_level > 2) fmt::print("HaloGridBlocks temp_deallocate.\n");
    for (int did = 0; did < numTargets; ++did){
      _buffers[did]._grid.deallocate(allocator, sizeof(ivec3) * h_counts[did]);
      if (g_log_level > 2) fmt::print("Deallocated temp _buffers[{}]._grid\n", did);
      if (_buffers[did]._blockids) { 
        allocator.deallocate( _buffers[did]._blockids, h_counts[did] * sizeof(ivec3));
        if (g_log_level > 2) fmt::print("Deallocate temp _buffers[{}]._blockids\n", did);
      }
      else {
        if (g_log_level > 2) fmt::print("Already nullptr temp _buffers[{}]._blockids\n", did);
      }
    }
  }

  template <typename Allocator>
  void initBlocks(Allocator allocator, uint32_t blockCount) {
    for (int did = 0; did < numTargets; ++did) {
      _buffers[did]._blockids =
          (ivec3 *)allocator.allocate(sizeof(ivec3) * blockCount);
      if (g_log_level > 2) fmt::print("Allocated _buffers[{}]._blockids\n", did);
    }
  }

  template <typename Allocator>
  void initBuffer(Allocator allocator, std::vector<uint32_t> counts) {
    for (int did = 0; did < numTargets; ++did) {
      _buffers[did]._grid.allocate_handle(allocator, counts[did]);
      if (g_log_level > 2) fmt::print("Allocated_handle _buffers[{}]._grid\n", did);
    }
  }
  void resetCounts(cudaStream_t stream) {
    checkCudaErrors(
        cudaMemsetAsync(_counts, 0, sizeof(uint32_t) * numTargets, stream));
  }
  void resetBlocks(uint32_t blockCount, cudaStream_t stream) {
    for (int did = 0; did < numTargets; ++did) {
      checkCudaErrors(
          cudaMemsetAsync(_buffers[did]._blockids, 0, sizeof(ivec3) * blockCount, stream));//JB
    }
  }

  void retrieveCounts(cudaStream_t stream) {
    checkCudaErrors(cudaMemcpyAsync(h_counts.data(), _counts,
                                    sizeof(uint32_t) * numTargets,
                                    cudaMemcpyDefault, stream));
  }
  void send(HaloGridBlocks &other, int src, int dst, cudaStream_t stream) {
    auto cnt = other.h_counts[src] = h_counts[dst];
#if 0
    checkCudaErrors(cudaMemcpyAsync(
        &other._buffers[src].val(_1, 0), &_buffers[dst].val(_1, 0),
        sizeof(ivec3) * cnt, cudaMemcpyDefault, stream));
#else
    checkCudaErrors(
        cudaMemcpyAsync(other._buffers[src]._blockids, _buffers[dst]._blockids,
                        sizeof(ivec3) * cnt, cudaMemcpyDefault, stream));
#endif
#if 0
    checkCudaErrors(
        cudaMemcpyAsync(&other._buffers[src]._grid.ch(_0, 0).val_1d(_0, 0),
                        &_buffers[dst]._grid.ch(_0, 0).val_1d(_0, 0),
                        grid_block_::size * cnt, cudaMemcpyDefault, stream));
#else
    checkCudaErrors(
        cudaMemcpyPeerAsync(&other._buffers[src]._grid.ch(_0, 0).val_1d(_0, 0),
                            dst, &_buffers[dst]._grid.ch(_0, 0).val_1d(_0, 0),
                            src, grid_block_::size * cnt, stream));
#endif
    // printf("sending from %d to %d at %llu\n", src, dst,
    //       (unsigned long long)&other._buffers[src]._grid.ch(_0, 0).val_1d(_0,
    //       0));
  }
  const int numTargets; // Number of neighboring GPU partitions. Defaults to total number of GPUs for sim.
  uint32_t *_counts; // Number of overlapping blocks (IIRC) with each neighbor GPU for current GPU
  std::vector<uint32_t> h_counts; // Host copy of _counts, vectorized for convenience
  std::vector<HaloBuffer> _buffers; // 
};

} // namespace mn

#endif