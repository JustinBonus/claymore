#ifndef __HASH_TABLE_CUH_
#define __HASH_TABLE_CUH_
#include "mgmpm_kernels.cuh"
#include "settings.cuh"
// #include "grid_buffer.cuh"
#include "utility_funcs.hpp"
#include <MnSystem/Cuda/HostUtils.hpp>

#include <MnBase/Object/Structural.h>
#include <MnBase/Object/StructuralDeclaration.h>
#include <fmt/color.h>
#include <fmt/core.h>

namespace mn {

/// Set-up template for Halo Partitions (manages overlap and halo-tagging between GPU devices)
template <int> struct HaloPartition {
  template <typename Allocator> HaloPartition(Allocator, int) {
    std::cout << "Constructing HaloPartition.\n";
  }
  template <typename Allocator>
  void resizePartition(Allocator allocator, std::size_t prevCapacity,
                       std::size_t capacity) {}
  void copy_to(HaloPartition &other, std::size_t blockCnt,
               cudaStream_t stream) {}
};
template <> struct HaloPartition<1> {
  // Initialize with max size of maxBlockCnt using a GPU allocator
  template <typename Allocator>
  HaloPartition(Allocator allocator, int maxBlockCnt) {
    std::cout << "Constructing HaloPartition<1>.\n";
    _count = (int *)allocator.allocate(sizeof(int) ); // int or char?? with maxBlockCnt?? JB
    _haloMarks = (char *)allocator.allocate(sizeof(char) * maxBlockCnt);
    _overlapMarks = (int *)allocator.allocate(sizeof(int) * maxBlockCnt);
    _haloBlocks = (ivec3 *)allocator.allocate(sizeof(ivec3) * maxBlockCnt);
    fmt::print("Allocated _count bytes [{}].\n", sizeof(int) ); // with maxBlockCnt? JB
    fmt::print("Allocated _haloMarks bytes[{}].\n", sizeof(char) * maxBlockCnt);
    fmt::print("Allocated _overlapMarks bytes[{}].\n", sizeof(int) * maxBlockCnt);
    fmt::print("Allocated _haloBlocks bytes[{}].\n", sizeof(ivec3) * maxBlockCnt);
  }
  
  template <typename Allocator>
  void deallocate_partition(Allocator allocator, std::size_t prevCapacity) {
    allocator.deallocate(_count, sizeof(int)  ); // int or char?? with prevCapacity?
    allocator.deallocate(_haloMarks, sizeof(char) * prevCapacity);
    allocator.deallocate(_overlapMarks, sizeof(int) * prevCapacity);
    allocator.deallocate(_haloBlocks, sizeof(ivec3) * prevCapacity);
    fmt::print("Deallocated _count bytes[{}].\n", sizeof(int) ); // with prevCapacity? JB
    fmt::print("Deallocated _haloMarks bytes[{}].\n", sizeof(char) * prevCapacity);
    fmt::print("Deallocated _overlapMarks bytes[{}].\n", sizeof(int) * prevCapacity);
    fmt::print("Deallocated _haloBlocks bytes[{}].\n", sizeof(ivec3) * prevCapacity);
  }

  // Copy halo count, halo-marks, overlap-marks, and Halo-Block IDs to other GPUs
  void copy_to(HaloPartition &other, std::size_t blockCnt, cudaStream_t stream) {
    other.h_count = h_count;
    checkCudaErrors(cudaMemcpyAsync(other._haloMarks, _haloMarks,
                                    sizeof(char) * blockCnt, cudaMemcpyDefault,
                                    stream));
    checkCudaErrors(cudaMemcpyAsync(other._overlapMarks, _overlapMarks,
                                    sizeof(int) * blockCnt, cudaMemcpyDefault,
                                    stream));
    checkCudaErrors(cudaMemcpyAsync(other._haloBlocks, _haloBlocks,
                                    sizeof(ivec3) * blockCnt, cudaMemcpyDefault,
                                    stream));
  }
  // Resize Halo Partition for number of active Halo Blocks
  template <typename Allocator>
  void resizePartition(Allocator allocator, std::size_t prevCapacity,
                       std::size_t capacity) {
    allocator.deallocate(_haloMarks, sizeof(char) * prevCapacity); 
    allocator.deallocate(_overlapMarks, sizeof(int) * prevCapacity);
    allocator.deallocate(_haloBlocks, sizeof(ivec3) * prevCapacity);
    _haloMarks = (char *)allocator.allocate(sizeof(char) * capacity);
    _overlapMarks = (int *)allocator.allocate(sizeof(int) * capacity);
    _haloBlocks = (ivec3 *)allocator.allocate(sizeof(ivec3) * capacity);
    fmt::print("Resized _haloMarks, _overlapMarks, _haloBlocks. Blocks\n", sizeof(char) * capacity);
  }
  void resetHaloCount(cudaStream_t stream) {
    checkCudaErrors(cudaMemsetAsync(_count, 0, sizeof(int) , stream)); // int or char??
  }
  void resetOverlapMarks(uint32_t neighborBlockCount, cudaStream_t stream) {
    checkCudaErrors(cudaMemsetAsync(_overlapMarks, 0,
                                    sizeof(int) * neighborBlockCount, stream));
  }
  void resetHaloBlocks(uint32_t neighborBlockCount, cudaStream_t stream) {
    checkCudaErrors(cudaMemsetAsync(_haloBlocks, 0,
                                    sizeof(ivec3) * neighborBlockCount, stream)); // JB
  }
  void retrieveHaloCount(cudaStream_t stream) {
    checkCudaErrors(cudaMemcpyAsync(&h_count, _count, sizeof(int),
                                    cudaMemcpyDefault, stream));
  }
  int *_count; //< Pointer to overlpa block count in GPU memory (used for atomicAdd in halo_kerels.cuh)
  int h_count; //< Host block count, used to read _count from GPU
  char *_haloMarks; ///< Halo particle block marks
  int *_overlapMarks; //< Overlapping marks
  ivec3 *_haloBlocks; //< 3D IDs of Halo Blocks
};


// Basic data-structure for Partitions
using block_partition_ =
    structural<structural_type::hash,
               decorator<structural_allocation_policy::full_allocation,
                         structural_padding_policy::compact>,
               GridDomain, attrib_layout::aos, empty_>; // GridDomain in grid_buffer.cuh

/// @brief Template for Partitions (organizes interaction of Particles and Grids). 
/// Uses a ton of memory for large DOMAIN_BITS. Inherits from Halo Partition (organizes Multi-GPU interaction). Can hold particle buckets when only one model per GPU is used.
template <int Opt = 1>
struct Partition : Instance<block_partition_>, HaloPartition<Opt> {
  using base_t = Instance<block_partition_>;
  using halo_base_t = HaloPartition<Opt>;
  using block_partition_::key_t;
  using block_partition_::value_t;
  static_assert(sentinel_v == (value_t)(-1), "sentinel value not full 1s\n");

  template <typename Allocator>
  Partition(Allocator allocator, int maxBlockCnt)
      : halo_base_t{allocator, maxBlockCnt} {
    _runtimeExtent = domain::extent;
    allocate_table(allocator, maxBlockCnt);
    // allocate_handle(allocator);
    if (!mn::config::g_buckets_on_particle_buffer) {
      _ppcs = (int *)allocator.allocate(sizeof(int) * maxBlockCnt *
                                        config::g_blockvolume);
      _ppbs = (int *)allocator.allocate(sizeof(int) * maxBlockCnt);
      _cellbuckets = (int *)allocator.allocate(sizeof(int) * maxBlockCnt * 
                                                config::g_blockvolume * config::g_max_ppc);
      _blockbuckets = (int *)allocator.allocate(sizeof(int) * maxBlockCnt *
                                                config::g_particle_num_per_block);
      _binsts = (int *)allocator.allocate(sizeof(int) * maxBlockCnt);
      fmt::print("Allocated _ppcs bytes [{}].\n", sizeof(int) * maxBlockCnt *
                                                      config::g_blockvolume);
      fmt::print("Allocated _ppbs bytes[{}].\n", sizeof(int) * maxBlockCnt);
      fmt::print("Allocated _cellbuckets bytes[{}].\n", sizeof(int) * maxBlockCnt *
                                                          config::g_blockvolume *
                                                          config::g_max_ppc);
      fmt::print("Allocated _blockbuckets bytes[{}].\n", sizeof(int) * maxBlockCnt *
                                                            config::g_particle_num_per_block);
      fmt::print("Allocated _binsts bytes[{}].\n",  sizeof(int) * maxBlockCnt);
    }
    /// init
    reset();
  }
  template <typename Allocator>
  Partition(Allocator allocator, int maxBlockCnt, int runtimeExtent)
      : halo_base_t{allocator, maxBlockCnt} {
    _runtimeExtent = runtimeExtent;
    fmt::print("Partition compiled domain::extent[{}]\n", domain::extent);
    fmt::print("Partition _runtimeExtent[{}]\n", _runtimeExtent);
    allocate_table(allocator, maxBlockCnt, runtimeExtent);
    if (!mn::config::g_buckets_on_particle_buffer){
      _ppcs = (int *)allocator.allocate(sizeof(int) * maxBlockCnt *
                                        config::g_blockvolume);
      _ppbs = (int *)allocator.allocate(sizeof(int) * maxBlockCnt);
      _cellbuckets = (int *)allocator.allocate(sizeof(int) * maxBlockCnt * 
                                                config::g_blockvolume * config::g_max_ppc);
      _blockbuckets = (int *)allocator.allocate(sizeof(int) * maxBlockCnt *
                                                config::g_particle_num_per_block);
      _binsts = (int *)allocator.allocate(sizeof(int) * maxBlockCnt);
      fmt::print("Allocated _ppcs bytes[{}].\n", sizeof(int) * maxBlockCnt *
                                                    config::g_blockvolume);
      fmt::print("Allocated _ppbs bytes[{}].\n", sizeof(int) * maxBlockCnt);
      fmt::print("Allocated _cellbuckets bytes[{}].\n", sizeof(int) * maxBlockCnt *
                                                            config::g_blockvolume *
                                                            config::g_max_ppc);
      fmt::print("Allocated _blockbuckets bytes[{}].\n", sizeof(int) * maxBlockCnt *
                                                            config::g_particle_num_per_block);
      fmt::print("Allocated _binsts bytes[{}].\n", sizeof(int) * maxBlockCnt);
    }
    /// init
    reset();
  }
  template <typename Allocator>
  void resizePartition(Allocator allocator, std::size_t capacity) {
    halo_base_t::resizePartition(allocator, this->_capacity, capacity);
    fmt::print("Resized partitions::halo_base_t capacity from [{}] to [{}] blocks.\n", this->_capacity, capacity);
    if (!mn::config::g_buckets_on_particle_buffer){
      allocator.deallocate(_ppcs, sizeof(int) * this->_capacity * 
                                            config::g_blockvolume);
      allocator.deallocate(_ppbs, sizeof(int) * this->_capacity);
      allocator.deallocate(_cellbuckets, sizeof(int) * this->_capacity *
                                            config::g_blockvolume *
                                            config::g_max_ppc);
      allocator.deallocate(_blockbuckets, sizeof(int) * this->_capacity *                     config::g_particle_num_per_block); // Changed (JB)
      allocator.deallocate(_binsts, sizeof(int) * this->_capacity);
      fmt::print("Deallocated _ppcs bytes[{}].\n", sizeof(int) * this->_capacity *
                                                    config::g_blockvolume);
      fmt::print("Deallocated _ppbs bytes[{}].\n", sizeof(int) * this->_capacity);
      fmt::print("Deallocated _cellbuckets bytes[{}].\n", sizeof(int) * this->_capacity *
                                                            config::g_blockvolume *
                                                            config::g_max_ppc);
      fmt::print("Deallocated _blockbuckets bytes[{}].\n", sizeof(int) * this->_capacity *
                                                            config::g_particle_num_per_block);
      fmt::print("Deallocated _binsts bytes[{}].\n", sizeof(int) * this->_capacity);
      fmt::print("Allocated _ppcs bytes[{}].\n", sizeof(int) * capacity *
                                                    config::g_blockvolume);
      fmt::print("Allocated _ppbs bytes[{}].\n", sizeof(int) * capacity);
      fmt::print("Allocated _cellbuckets bytes[{}].\n", sizeof(int) * capacity *
                                                            config::g_blockvolume *
                                                            config::g_max_ppc);
      fmt::print("Allocated _blockbuckets bytes[{}].\n", sizeof(int) * capacity *
                                                            config::g_particle_num_per_block);
      fmt::print("Allocated _binsts bytes[{}].\n", sizeof(int) * capacity);
      _ppcs = (int *)allocator.allocate(sizeof(int) * capacity *
                                        config::g_blockvolume);
      _ppbs = (int *)allocator.allocate(sizeof(int) * capacity);
      _cellbuckets = (int *)allocator.allocate(sizeof(int) * capacity * 
                                        config::g_blockvolume * config::g_max_ppc);
      _blockbuckets = (int *)allocator.allocate(sizeof(int) * capacity *
                                                config::g_particle_num_per_block);
      _binsts = (int *)allocator.allocate(sizeof(int) * capacity);
    }
    resize_table(allocator, capacity);
    fmt::print("Resized partitions hash-table capacity to [{}] blocks.\n", capacity);
  }
  ~Partition() {}


  // Reset Partition's Block count and particles-per-each-cell
  void reset() {
    checkCudaErrors(cudaMemset(this->_cnt, 0, sizeof(value_t)));
    // TODO: Double-check use of 0xff - JB
    checkCudaErrors(cudaMemset(this->_indexTable, 0xff, sizeof(value_t) * 
                                                        _runtimeExtent));
    fmt::print("Reset partitions _cnt values to [{}].\n", 0);
    fmt::print("Reset partitions _indextable values to [{}].\n", 0xff);
    if (!mn::config::g_buckets_on_particle_buffer) {
      checkCudaErrors(cudaMemset( this->_ppcs, 0, sizeof(int) * this->_capacity 
                                                              * config::g_blockvolume));
      // checkCudaErrors(cudaMemset(this->_ppbs, 0, sizeof(int) * this->_capacity));
      // checkCudaErrors(cudaMemset(this->_binsts, 0, sizeof(int) * this->_capacity));
      fmt::print("Reset partitions  _ppcs values to [{}].\n", 0);
    }
  }
  // Reset Partition index-table (maps 1D - 3D Block coordinates)
  void resetTable(cudaStream_t stream) {
    checkCudaErrors(cudaMemsetAsync(this->_indexTable, 0xff,
                                    sizeof(value_t) * _runtimeExtent, stream));
    if (g_log_level >= 3) fmt::print("Reset partitions _indexTable values to [{}] over [{}] bytes\n", 0xff, sizeof(value_t) * _runtimeExtent);
  }

  template <typename Allocator>
  void deallocate_partition(Allocator allocator) {
    deallocate_buckets(allocator);
    halo_base_t::deallocate_partition(allocator, this->_capacity); // Deallocate halo_base_t
    base_t::deallocate(allocator, this->_runtimeExtent);
  }

  template <typename Allocator>
  void deallocate_buckets(Allocator allocator) {
    if (!mn::config::g_buckets_on_particle_buffer) {
      allocator.deallocate(_ppcs,
                          sizeof(int) * this->_capacity * config::g_blockvolume);
      allocator.deallocate(_ppbs, sizeof(int) * this->_capacity);
      allocator.deallocate(_cellbuckets, sizeof(int) * this->_capacity *
                                            config::g_blockvolume *
                                            config::g_max_ppc);
      allocator.deallocate(_blockbuckets,
                          sizeof(int) * this->_capacity * config::g_particle_num_per_block);
      allocator.deallocate(_binsts, sizeof(int) * this->_capacity);
      fmt::print("Deallocated _ppcs bytes[{}].\n", sizeof(int) * this->_capacity * config::g_blockvolume);
      fmt::print("Deallocated _ppbs bytes [{}].\n", sizeof(int) * this->_capacity);
      fmt::print("Deallocated _cellbuckets bytes[{}].\n", sizeof(int) * this->_capacity * config::g_blockvolume * config::g_max_ppc);
      fmt::print("Deallocated _blockbuckets bytes[{}].\n", sizeof(int) * this->_capacity * config::g_particle_num_per_block);
      fmt::print("Deallocated _binsts bytes[{}].\n", sizeof(int) * this->_capacity);
    }
  }
  // Build Particle Block Buckets 
  template <typename CudaContext>
  void buildParticleBuckets(CudaContext &&cuDev, value_t cnt) {
    if (mn::config::g_buckets_on_particle_buffer == false) {
      checkCudaErrors(cudaMemsetAsync(this->_ppbs, 0, sizeof(int) * (cnt + 1),
                                      cuDev.stream_compute()));
      fmt::print("Reset _ppbs bytes[{}] to zero.\n", sizeof(int) * (cnt + 1));
      cuDev.compute_launch({cnt, config::g_blockvolume}, cell_bucket_to_block,
                          _ppcs, _cellbuckets, _ppbs, _blockbuckets);
    }
  }
  // Copy Partition information to next time-step's Partition
  void copy_to(Partition &other, std::size_t blockCnt, cudaStream_t stream) {
    halo_base_t::copy_to(other, blockCnt, stream);
    checkCudaErrors(cudaMemcpyAsync(other._indexTable, this->_indexTable,
                                    sizeof(value_t) * _runtimeExtent,
                                    cudaMemcpyDefault, stream));
    fmt::print("Copied _indexTable bytes[{}] to other.\n", sizeof(value_t) * _runtimeExtent);
    if (mn::config::g_buckets_on_particle_buffer == false) {
      checkCudaErrors(cudaMemcpyAsync(other._ppbs, this->_ppbs,
                                      sizeof(int) * blockCnt, cudaMemcpyDefault,
                                      stream));
      checkCudaErrors(cudaMemcpyAsync(other._binsts, this->_binsts,
                                      sizeof(int) * blockCnt, cudaMemcpyDefault,
                                      stream));
      fmt::print("Copied _ppbs bytes[{}] to other.\n", sizeof(int) * blockCnt);
      fmt::print("Copied _binsts bytes[{}] to other.\n", sizeof(int) * blockCnt);
    }
  }
  /// @brief Insert new key for Block in Partition Index-Table
  /// @param key 3D coordinates of Block to insert
  /// @return New 1D index for inserted key. Returns -1 if error occurs.
  __forceinline__ __device__ value_t insert(key_t key) noexcept {
    // Set &this->index(key) = 0 if (&this->index(key) == sentinel_v), sentinel_v = -1 basically 
    // Return old value of this->index(key) as tag
    value_t tag = atomicCAS(&this->index(key), sentinel_v, 0); 
    if (tag == sentinel_v) { // If above CAS evaluated true (i.e. valid block insert)
      value_t idx = atomicAdd(this->_cnt, 1); // +1 to Partition's Block count
      this->index(key) = idx; //< Index of inserted key set by incremented Block count
      this->_activeKeys[idx] = key; ///< Created record of inserted key in Partition active keys
      return idx; //< Return new index for inserted key
    }
    return -1; //< Return -1 as error flag
  }
  /// @brief Query index of block in Partition Index-Table.
  /// @param key 3D index of block to query
  /// @return 1D index of key in partition's _indexTable.
  __forceinline__ __device__ value_t query(key_t key) const noexcept {
    return this->index(key);
  }
  /// @brief Reinsert key of Block 1D index in Partition Hash-Table
  /// @param index 1D Index of block to rehash with a key 
  __forceinline__ __device__ void reinsert(value_t index) {
    this->index(this->_activeKeys[index]) = index;
  }
  /// @brief Advect particle ID in a Block to a new Cell in Partition. Done because particles move during Grid-to-Particle-to-Grid. Writes to _cellbuckets.
  /// @param cellid Grid-cell 3D ID in block
  /// @param dirtag Direction tag of advection (0-26) to represent 3D offset
  /// @param pidib Particle-ID-in-block to advect
  __forceinline__ __device__ void add_advection(key_t cellid, int dirtag,
                                                int pidib) noexcept {
    using namespace config;
    key_t blockid = cellid / g_blocksize;
    value_t blockno = query(blockid);

    if (blockno == -1) {
      ivec3 offset{};
      dir_components(dirtag, offset);
      printf("Error in hash_table.cuh! Cell-ID(%d, %d, %d) offset_dir(%d, %d, %d) particle-ID-in-block(%d).\n",
             cellid[0], cellid[1], cellid[2], offset[0], offset[1], offset[2],
             pidib);
      printf("Possible a particle exited simulation domain or time-step is too large. Poorly implemented material laws or incorrectly set material parameters can cause this, as well as errors in the Partition data-structure allocation, deallocation, copies, etc.\n");       
      return;
    }
    value_t cellno = ((cellid[0] & g_blockmask) << (g_blockbits << 1)) |
                     ((cellid[1] & g_blockmask) << g_blockbits) |
                     (cellid[2] & g_blockmask);
    int pidic = atomicAdd(_ppcs + blockno * g_blockvolume + cellno, 1); // ++particles-in-cell count
    // TODO : Check if original bitwise OR is correct, seems like it should be addition for any g_particle_num_per_block not power of 2 (i.e. if MAX_PPC is not a power of 2)
    // _cellbuckets[blockno * g_particle_num_per_block + cellno * g_max_ppc +
    //              pidic] = (dirtag * g_particle_num_per_block) | pidib; // Update cell-bucket
    _cellbuckets[blockno * g_particle_num_per_block + cellno * g_max_ppc +
                 pidic] = (dirtag * g_particle_num_per_block) + pidib; // Update cell-bucket
  }

  int *_ppcs, *_ppbs;
  int *_cellbuckets, *_blockbuckets;
  int *_binsts;
  int _runtimeExtent;
};

} // namespace mn

#endif