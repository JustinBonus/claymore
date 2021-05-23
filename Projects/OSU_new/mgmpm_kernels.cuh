#ifndef __MULTI_GMPM_KERNELS_CUH_
#define __MULTI_GMPM_KERNELS_CUH_

#include "filters.cuh"
#include "constitutive_models.cuh"
#include "particle_buffer.cuh"
#include "settings.h"
#include "utility_funcs.hpp"
#include <MnBase/Algorithm/MappingKernels.cuh>
#include <MnBase/Math/Matrix/MatrixUtils.h>
#include <MnSystem/Cuda/DeviceUtils.cuh>

#include <MnBase/Math/Matrix/svd3_cuda.h>

#include <cooperative_groups.h>

namespace mn {

using namespace config;
using namespace placeholder;

template <typename ParticleArray, typename Partition>
__global__ void activate_blocks(uint32_t particleCount, ParticleArray parray,
                                Partition partition) {
  uint32_t parid = blockIdx.x * blockDim.x + threadIdx.x;
  if (parid >= particleCount)
    return;
  ivec3 blockid{
      int(std::lround(parray.val(_0, parid) / g_dx) - 2) / g_blocksize,
      int(std::lround(parray.val(_1, parid) / g_dx) - 2) / g_blocksize,
      int(std::lround(parray.val(_2, parid) / g_dx) - 2) / g_blocksize};
  partition.insert(blockid);
}
template <typename ParticleArray, typename ParticleBuffer, typename Partition>
__global__ void
build_particle_cell_buckets(uint32_t particleCount, ParticleArray parray,
                            ParticleBuffer pbuffer, Partition partition) {
  uint32_t parid = blockIdx.x * blockDim.x + threadIdx.x;
  if (parid >= particleCount)
    return;
  ivec3 coord{int(std::lround(parray.val(_0, parid) / g_dx) - 2),
              int(std::lround(parray.val(_1, parid) / g_dx) - 2),
              int(std::lround(parray.val(_2, parid) / g_dx) - 2)};
  int cellno = (coord[0] & g_blockmask) * g_blocksize * g_blocksize +
               (coord[1] & g_blockmask) * g_blocksize +
               (coord[2] & g_blockmask);
  coord = coord / g_blocksize;
  auto blockno = partition.query(coord);
  auto pidic = atomicAdd(pbuffer._ppcs + blockno * g_blockvolume + cellno, 1);
  pbuffer._cellbuckets[blockno * g_particle_num_per_block + cellno * g_max_ppc +
                       pidic] = parid;
}
__global__ void cell_bucket_to_block(int *_ppcs, int *_cellbuckets, int *_ppbs,
                                     int *_buckets) {
  int cellno = threadIdx.x & (g_blockvolume - 1);
  int pcnt = _ppcs[blockIdx.x * g_blockvolume + cellno];
  for (int pidic = 0; pidic < g_max_ppc; pidic++) {
    if (pidic < pcnt) {
      auto pidib = atomicAggInc<int>(_ppbs + blockIdx.x);
      _buckets[blockIdx.x * g_particle_num_per_block + pidib] =
          _cellbuckets[blockIdx.x * g_particle_num_per_block +
                       cellno * g_max_ppc + pidic];
    }
    __syncthreads();
  }
}
__global__ void compute_bin_capacity(uint32_t blockCount, int const *_ppbs,
                                     int *_bincaps) {
  uint32_t blockno = blockIdx.x * blockDim.x + threadIdx.x;
  if (blockno >= blockCount)
    return;
  _bincaps[blockno] = (_ppbs[blockno] + g_bin_capacity - 1) / g_bin_capacity;
}
__global__ void init_adv_bucket(const int *_ppbs, int *_buckets) {
  auto pcnt = _ppbs[blockIdx.x];
  auto bucket = _buckets + blockIdx.x * g_particle_num_per_block;
  for (int pidib = threadIdx.x; pidib < pcnt; pidib += blockDim.x) {
    bucket[pidib] =
        (dir_offset(ivec3{0, 0, 0}) * g_particle_num_per_block) | pidib;
  }
}
template <typename Grid> __global__ void clear_grid(Grid grid) {
  auto gridblock = grid.ch(_0, blockIdx.x);
  for (int cidib = threadIdx.x; cidib < g_blockvolume; cidib += blockDim.x) {
    gridblock.val_1d(_0, cidib) = 0.f;
    gridblock.val_1d(_1, cidib) = 0.f;
    gridblock.val_1d(_2, cidib) = 0.f;
    gridblock.val_1d(_3, cidib) = 0.f;
  }
}
template <typename Partition>
__global__ void register_neighbor_blocks(uint32_t blockCount,
                                         Partition partition) {
  uint32_t blockno = blockIdx.x * blockDim.x + threadIdx.x;
  if (blockno >= blockCount)
    return;
  auto blockid = partition._activeKeys[blockno];
  for (char i = 0; i < 2; ++i)
    for (char j = 0; j < 2; ++j)
      for (char k = 0; k < 2; ++k)
        partition.insert(ivec3{blockid[0] + i, blockid[1] + j, blockid[2] + k});
}
template <typename Partition>
__global__ void register_exterior_blocks(uint32_t blockCount,
                                         Partition partition) {
  uint32_t blockno = blockIdx.x * blockDim.x + threadIdx.x;
  if (blockno >= blockCount)
    return;
  auto blockid = partition._activeKeys[blockno];
  for (char i = -1; i < 2; ++i)
    for (char j = -1; j < 2; ++j)
      for (char k = -1; k < 2; ++k)
        partition.insert(ivec3{blockid[0] + i, blockid[1] + j, blockid[2] + k});
}
template <typename Grid, typename Partition>
__global__ void rasterize(uint32_t particleCount, const ParticleArray parray,
                          Grid grid, const Partition partition, float dt,
                          float mass, vec3 v0) {
  uint32_t parid = blockIdx.x * blockDim.x + threadIdx.x;
  if (parid >= particleCount)
    return;

  vec3 local_pos{parray.val(_0, parid), parray.val(_1, parid),
                 parray.val(_2, parid)};
  vec3 vel = v0;
  vec9 contrib, C;
  contrib.set(0.f), C.set(0.f);
  contrib = (C * mass - contrib * dt) * g_D_inv;
  ivec3 global_base_index{int(std::lround(local_pos[0] * g_dx_inv) - 1),
                          int(std::lround(local_pos[1] * g_dx_inv) - 1),
                          int(std::lround(local_pos[2] * g_dx_inv) - 1)};
  local_pos = local_pos - global_base_index * g_dx;
  vec<vec3, 3> dws;
  for (int d = 0; d < 3; ++d)
    dws[d] = bspline_weight(local_pos[d]);
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j)
      for (int k = 0; k < 3; ++k) {
        ivec3 offset{i, j, k};
        vec3 xixp = offset * g_dx - local_pos;
        float W = dws[0][i] * dws[1][j] * dws[2][k];
        ivec3 local_index = global_base_index + offset;
        float wm = mass * W;
        int blockno = partition.query(ivec3{local_index[0] >> g_blockbits,
                                            local_index[1] >> g_blockbits,
                                            local_index[2] >> g_blockbits});
        auto grid_block = grid.ch(_0, blockno);
        for (int d = 0; d < 3; ++d)
          local_index[d] &= g_blockmask;
        atomicAdd(
            &grid_block.val(_0, local_index[0], local_index[1], local_index[2]),
            wm);
        atomicAdd(
            &grid_block.val(_1, local_index[0], local_index[1], local_index[2]),
            wm * vel[0] + (contrib[0] * xixp[0] + contrib[3] * xixp[1] +
                           contrib[6] * xixp[2]) *
                              W);
        atomicAdd(
            &grid_block.val(_2, local_index[0], local_index[1], local_index[2]),
            wm * vel[1] + (contrib[1] * xixp[0] + contrib[4] * xixp[1] +
                           contrib[7] * xixp[2]) *
                              W);
        atomicAdd(
            &grid_block.val(_3, local_index[0], local_index[1], local_index[2]),
            wm * vel[2] + (contrib[2] * xixp[0] + contrib[5] * xixp[1] +
                           contrib[8] * xixp[2]) *
                              W);
      }
}

template <typename ParticleArray>
__global__ void array_to_buffer(ParticleArray parray,
                                ParticleBuffer<material_e::JFluid> pbuffer) {
  uint32_t blockno = blockIdx.x;
  int pcnt = pbuffer._ppbs[blockno];
  auto bucket = pbuffer._blockbuckets + blockno * g_particle_num_per_block;
  for (int pidib = threadIdx.x; pidib < pcnt; pidib += blockDim.x) {
    auto parid = bucket[pidib];
    auto pbin =
        pbuffer.ch(_0, pbuffer._binsts[blockno] + pidib / g_bin_capacity);
    /// pos
    pbin.val(_0, pidib % g_bin_capacity) = parray.val(_0, parid);
    pbin.val(_1, pidib % g_bin_capacity) = parray.val(_1, parid);
    pbin.val(_2, pidib % g_bin_capacity) = parray.val(_2, parid);
    /// J
    pbin.val(_3, pidib % g_bin_capacity) = 1.f;
  }
}

template <typename ParticleArray>
__global__ void
array_to_buffer(ParticleArray parray,
                ParticleBuffer<material_e::FixedCorotated> pbuffer) {
  uint32_t blockno = blockIdx.x;
  int pcnt = pbuffer._ppbs[blockno];
  auto bucket = pbuffer._blockbuckets + blockno * g_particle_num_per_block;
  for (int pidib = threadIdx.x; pidib < pcnt; pidib += blockDim.x) {
    auto parid = bucket[pidib];
    auto pbin =
        pbuffer.ch(_0, pbuffer._binsts[blockno] + pidib / g_bin_capacity);
    /// pos
    pbin.val(_0, pidib % g_bin_capacity) = parray.val(_0, parid);
    pbin.val(_1, pidib % g_bin_capacity) = parray.val(_1, parid);
    pbin.val(_2, pidib % g_bin_capacity) = parray.val(_2, parid);
    /// F
    pbin.val(_3, pidib % g_bin_capacity) = 1.f;
    pbin.val(_4, pidib % g_bin_capacity) = 0.f;
    pbin.val(_5, pidib % g_bin_capacity) = 0.f;
    pbin.val(_6, pidib % g_bin_capacity) = 0.f;
    pbin.val(_7, pidib % g_bin_capacity) = 1.f;
    pbin.val(_8, pidib % g_bin_capacity) = 0.f;
    pbin.val(_9, pidib % g_bin_capacity) = 0.f;
    pbin.val(_10, pidib % g_bin_capacity) = 0.f;
    pbin.val(_11, pidib % g_bin_capacity) = 1.f;
  }
}

template <typename ParticleArray>
__global__ void array_to_buffer(ParticleArray parray,
                                ParticleBuffer<material_e::Sand> pbuffer) {
  uint32_t blockno = blockIdx.x;
  int pcnt = pbuffer._ppbs[blockno];
  auto bucket = pbuffer._blockbuckets + blockno * g_particle_num_per_block;
  for (int pidib = threadIdx.x; pidib < pcnt; pidib += blockDim.x) {
    auto parid = bucket[pidib];
    auto pbin =
        pbuffer.ch(_0, pbuffer._binsts[blockno] + pidib / g_bin_capacity);
    /// pos
    pbin.val(_0, pidib % g_bin_capacity) = parray.val(_0, parid);
    pbin.val(_1, pidib % g_bin_capacity) = parray.val(_1, parid);
    pbin.val(_2, pidib % g_bin_capacity) = parray.val(_2, parid);
    /// F
    pbin.val(_3, pidib % g_bin_capacity) = 1.f;
    pbin.val(_4, pidib % g_bin_capacity) = 0.f;
    pbin.val(_5, pidib % g_bin_capacity) = 0.f;
    pbin.val(_6, pidib % g_bin_capacity) = 0.f;
    pbin.val(_7, pidib % g_bin_capacity) = 1.f;
    pbin.val(_8, pidib % g_bin_capacity) = 0.f;
    pbin.val(_9, pidib % g_bin_capacity) = 0.f;
    pbin.val(_10, pidib % g_bin_capacity) = 0.f;
    pbin.val(_11, pidib % g_bin_capacity) = 1.f;
    /// logJp
    pbin.val(_12, pidib % g_bin_capacity) =
        ParticleBuffer<material_e::Sand>::logJp0;
  }
}

template <typename ParticleArray>
__global__ void array_to_buffer(ParticleArray parray,
                                ParticleBuffer<material_e::NACC> pbuffer) {
  uint32_t blockno = blockIdx.x;
  int pcnt = pbuffer._ppbs[blockno];
  auto bucket = pbuffer._blockbuckets + blockno * g_particle_num_per_block;
  for (int pidib = threadIdx.x; pidib < pcnt; pidib += blockDim.x) {
    auto parid = bucket[pidib];
    auto pbin =
        pbuffer.ch(_0, pbuffer._binsts[blockno] + pidib / g_bin_capacity);
    /// pos
    pbin.val(_0, pidib % g_bin_capacity) = parray.val(_0, parid);
    pbin.val(_1, pidib % g_bin_capacity) = parray.val(_1, parid);
    pbin.val(_2, pidib % g_bin_capacity) = parray.val(_2, parid);
    /// F
    pbin.val(_3, pidib % g_bin_capacity) = 1.f;
    pbin.val(_4, pidib % g_bin_capacity) = 0.f;
    pbin.val(_5, pidib % g_bin_capacity) = 0.f;
    pbin.val(_6, pidib % g_bin_capacity) = 0.f;
    pbin.val(_7, pidib % g_bin_capacity) = 1.f;
    pbin.val(_8, pidib % g_bin_capacity) = 0.f;
    pbin.val(_9, pidib % g_bin_capacity) = 0.f;
    pbin.val(_10, pidib % g_bin_capacity) = 0.f;
    pbin.val(_11, pidib % g_bin_capacity) = 1.f;
    /// logJp
    pbin.val(_12, pidib % g_bin_capacity) =
        ParticleBuffer<material_e::NACC>::logJp0;
  }
}

template <typename ParticleArray>
__global__ void
array_to_buffer(ParticleArray parray,
                ParticleBuffer<material_e::Rigid> pbuffer) {
  uint32_t blockno = blockIdx.x;
  int pcnt = pbuffer._ppbs[blockno];
  auto bucket = pbuffer._blockbuckets + blockno * g_particle_num_per_block;
  for (int pidib = threadIdx.x; pidib < pcnt; pidib += blockDim.x) {
    auto parid = bucket[pidib];
    auto pbin =
        pbuffer.ch(_0, pbuffer._binsts[blockno] + pidib / g_bin_capacity);
    /// pos
    pbin.val(_0, pidib % g_bin_capacity) = parray.val(_0, parid);
    pbin.val(_1, pidib % g_bin_capacity) = parray.val(_1, parid);
    pbin.val(_2, pidib % g_bin_capacity) = parray.val(_2, parid);
    /// F
    pbin.val(_3, pidib % g_bin_capacity) = 1.f;
  }
}

template <typename ParticleArray>
__global__ void
array_to_buffer(ParticleArray parray,
                ParticleBuffer<material_e::Piston> pbuffer) {
  uint32_t blockno = blockIdx.x;
  int pcnt = pbuffer._ppbs[blockno];
  auto bucket = pbuffer._blockbuckets + blockno * g_particle_num_per_block;
  for (int pidib = threadIdx.x; pidib < pcnt; pidib += blockDim.x) {
    auto parid = bucket[pidib];
    auto pbin =
        pbuffer.ch(_0, pbuffer._binsts[blockno] + pidib / g_bin_capacity);
    /// pos
    pbin.val(_0, pidib % g_bin_capacity) = parray.val(_0, parid);
    pbin.val(_1, pidib % g_bin_capacity) = parray.val(_1, parid);
    pbin.val(_2, pidib % g_bin_capacity) = parray.val(_2, parid);
    /// F
    pbin.val(_3, pidib % g_bin_capacity) = 1.f;
    pbin.val(_4, pidib % g_bin_capacity) = 0.f;
    pbin.val(_5, pidib % g_bin_capacity) = 0.f;
    pbin.val(_6, pidib % g_bin_capacity) = 0.f;
    pbin.val(_7, pidib % g_bin_capacity) = 1.f;
    pbin.val(_8, pidib % g_bin_capacity) = 0.f;
    pbin.val(_9, pidib % g_bin_capacity) = 0.f;
    pbin.val(_10, pidib % g_bin_capacity) = 0.f;
    pbin.val(_11, pidib % g_bin_capacity) = 1.f;
  }
}

template <typename ParticleArray>
__global__ void array_to_buffer(ParticleArray parray,
                                ParticleBuffer<material_e::IFluid> pbuffer) {
  uint32_t blockno = blockIdx.x;
  int pcnt = pbuffer._ppbs[blockno];
  auto bucket = pbuffer._blockbuckets + blockno * g_particle_num_per_block;
  for (int pidib = threadIdx.x; pidib < pcnt; pidib += blockDim.x) {
    auto parid = bucket[pidib];
    auto pbin =
        pbuffer.ch(_0, pbuffer._binsts[blockno] + pidib / g_bin_capacity);
    /// pos
    pbin.val(_0, pidib % g_bin_capacity) = parray.val(_0, parid);
    pbin.val(_1, pidib % g_bin_capacity) = parray.val(_1, parid);
    pbin.val(_2, pidib % g_bin_capacity) = parray.val(_2, parid);
    /// J
    pbin.val(_3, pidib % g_bin_capacity) = 1.f;
  }
}

// Move values from Layer 'n' to Layer 'n-1'
template <typename Grid, typename Partition>
__global__ void update_grid_hierarchy(uint32_t blockCount, 
                                               Grid grid_from, Grid grid_to,
                                               Partition partition, float dt,
                                               float *maxVel, int layer) {

  constexpr int dx = g_dx;

  int dx_from = dx << layer;
  int dx_to   = dx >> 1;


}


// Update grid velocities, apply boundary conditions, determine max grid velocity
// Use this function to interact with the grid
template <typename Grid, typename Partition>
__global__ void update_grid_velocity_query_max(uint32_t blockCount, Grid grid,
                                               Partition partition, float dt,
                                               float *maxVel, int layer) {
  constexpr int bc = g_bc; //< Num of 'buffer' grid-blocks at domain exterior
  constexpr int numWarps = g_num_grid_blocks_per_cuda_block * 
    g_num_warps_per_grid_block; //< Warps per block
  constexpr unsigned activeMask = 0xffffffff;
  //__shared__ float sh_maxvels[g_blockvolume * g_num_grid_blocks_per_cuda_block
  /// 32];
  
  extern __shared__ float sh_maxvels[]; //< Max vel.^2s for block, shared mem
  std::size_t blockno = blockIdx.x * g_num_grid_blocks_per_cuda_block +
                        threadIdx.x / 32 / g_num_warps_per_grid_block; //< Grid-block number
  auto blockid = partition._activeKeys[blockno]; //< Grid-block ID, [i,j,k]

  /// 3 digits for grid-block presence at domain boundary
  /// Require buffers at exteriors, prevent CFL issues/particle escape
  /// e.g. 011 means y and z coords of block are at domain boundary
  int isInBound = ((blockid[0] < bc || blockid[0] >= g_grid_size_x - bc) << 2) |
                  ((blockid[1] < bc || blockid[1] >= g_grid_size_y - bc) << 1) |
                   (blockid[2] < bc || blockid[2] >= g_grid_size_z - bc);

  // One element in shared vel.^2 per warp
  if (threadIdx.x < numWarps)
    sh_maxvels[threadIdx.x] = 0.0f;
  
  // Synch threads in block, extierior boundary collision checked
  __syncthreads();

  /// Within-warp computations
  if (blockno < blockCount) {
    auto grid_block = grid.ch(_0, blockno); //< Set grid-block of buffer
    // Loop through cells in grid-block, stride by 32 to avoid thread conflicts
    for (int cidib = threadIdx.x % 32; cidib < g_blockvolume; cidib += 32) {
      float mass = grid_block.val_1d(_0, cidib); //< Mass at grid-node in grid-block
      float velSqr = 0.f;  //< Thread velocity squared (vx^2 + vy^2 + vz^2)
      vec3 vel;            //< Thread velocity vector {vx, vy, vz} (m/s)
      if (mass > 0.f) {
        mass = 1.f / mass; //< Invert mass, avoids division operator
      
        // Grid node coordinate [i,j,k] in grid-block
        int i = (cidib >> (g_blockbits << 1)) & g_blockmask;
        int j = (cidib >> g_blockbits) & g_blockmask;
        int k = cidib & g_blockmask;
        // Grid node position [x,y,z] in entire domain
        float xc = (4*blockid[0]*g_dx) + (i*g_dx); // + (g_dx/2.f);
        float yc = (4*blockid[1]*g_dx) + (j*g_dx); // + (g_dx/2.f);
        float zc = (4*blockid[2]*g_dx) + (k*g_dx); // + (g_dx/2.f);

        // Offset condition for Off-by-2 (see Xinlei & Fang et al.)
        // Note you should subtract 16 nodes from total
        // (or 4 grid blocks) to have total available length
        float offset = (8.f*g_dx);

        // Retrieve grid momentums (kg*m/s2)
        vel[0] = grid_block.val_1d(_1, cidib); //< mvx
        vel[1] = grid_block.val_1d(_2, cidib); //< mvy
        vel[2] = grid_block.val_1d(_3, cidib); //< mvz


        // WASIRF Harris Flume (Slip)
        // Acts on individual grid-cell velocities
        // https://teamer-us.org/product/university-of-washington-harris-hydraulics-wasirf/
        float flumex = 104.f / g_length; // Actually 12m, added run-in/out
        float flumey = 4.6f / g_length; // 1.22m Depth
        float flumez = 3.67f / g_length; // 0.91m Width
        int isInFlume =  ((xc < offset || xc >= flumex + offset) << 2) |
                         ((yc < offset || yc >= flumey + offset) << 1) |
                          (zc < offset || zc >= flumez + offset);
        isInBound |= isInFlume; // Update with regular boundary for efficiency


        // Add grid-cell boundary for structural block, WASIRF flume
        vec3 struct_dim; //< Dimensions of structure in [1,1,1] pseudo-dimension
        struct_dim[0] = (0.7871f) / g_length;
        struct_dim[1] = (0.3935f) / g_length;
        struct_dim[2] = (0.7871f) / g_length;
        vec3 struct_pos; //< Position of structures in [1,1,1] pseudo-dimension
        struct_pos[0] = ((46 + 12 + 36 + 48 + (10.f/12.f))*0.3048f) / g_length + offset;
        struct_pos[1] = ((69.f/12.f)*0.3048f) / g_length + offset;
        struct_pos[2] = (flumez - struct_dim[2]) / 2.f + offset;
        float t = 1.0f * g_dx;

        // Check if grid-cell is within sticky interior of structural box
        // Subtract slip-layer thickness from structural box dimension for geometry
        int isOutStruct  = ((xc >= struct_pos[0] + t && xc < struct_pos[0] + struct_dim[0] - t) << 2) | 
                           ((yc >= struct_pos[1] + t && yc < struct_pos[1] + struct_dim[1] - t) << 1) |
                            (zc >= struct_pos[2] + t && zc < struct_pos[2] + struct_dim[2] - t);
        if (isOutStruct != 7) isOutStruct = 0; // Check if 111, reset otherwise
        isInBound |= isOutStruct; // Update with regular boundary for efficiency
        
        // Check exterior slip-layer of structural block(OSU Flume)
        // One-cell depth, six-faces, order matters! (over-writes on edges, favors front) 
        int isOnStructFace[6];
        // Right (z+)
        isOnStructFace[0] = ((xc >= struct_pos[0] && xc < struct_pos[0] + struct_dim[0]) << 2) | 
                            ((yc >= struct_pos[1] && yc < struct_pos[1] + struct_dim[1]) << 1) |
                             (zc >= struct_pos[2] + struct_dim[2] - t && zc < struct_pos[2] + struct_dim[2]);
        // Left (z-)
        isOnStructFace[1] = ((xc >= struct_pos[0] && xc < struct_pos[0] + struct_dim[0]) << 2) | 
                            ((yc >= struct_pos[1] && yc < struct_pos[1] + struct_dim[1]) << 1) |
                             (zc >= struct_pos[2] && zc < struct_pos[2] + t);        
        // Top (y+)
        isOnStructFace[2] = ((xc >= struct_pos[0] && xc < struct_pos[0] + struct_dim[0]) << 2) | 
                            ((yc >= struct_pos[1] + struct_dim[1] - t && yc < struct_pos[1] + struct_dim[1]) << 1) |
                             (zc >= struct_pos[2] && zc < struct_pos[2] + struct_dim[2]);
        // Bottom (y-)
        isOnStructFace[3] = ((xc >= struct_pos[0] && xc < struct_pos[0] + struct_dim[0]) << 2) | 
                            ((yc >= struct_pos[1] && yc < struct_pos[1] + t) << 1) |
                             (zc >= struct_pos[2] && zc < struct_pos[2] + struct_dim[2]);
        // Back (x+)
        isOnStructFace[4] = ((xc >= struct_pos[0] + struct_dim[0] - t && xc < struct_pos[0] + struct_dim[1]) << 2) | 
                            ((yc >= struct_pos[1] && yc < struct_pos[1] + struct_dim[1]) << 1) |
                             (zc >= struct_pos[2] && zc < struct_pos[2] + struct_dim[2]);
        // Front (x-)
        isOnStructFace[5] = ((xc >= struct_pos[0] && xc < struct_pos[0] + t) << 2) | 
                            ((yc >= struct_pos[1] && yc < struct_pos[1] + struct_dim[1]) << 1) |
                             (zc >= struct_pos[2] && zc < struct_pos[2] + struct_dim[2]);                             
        // Reduce results from box faces to single result
        int isOnStruct = 0; // Collision reduction variable
        for (int iter=0; iter<6; iter++) {
          if (isOnStructFace[iter] != 7) {
            // Check if 111 (7), set 000 (0) otherwise
            isOnStructFace[iter] = 0;
          } else {
            // // Set 111 to 100, 010, or 001, dep. on element order
            // int halfiter = iter / 2;
            
            // // Adjust normal component collision for direction
            // if ((iter % 2) == 0 && vel[halfiter] > 0) {
            //   isOnStructFace[iter] = (1 << iter / 2);
            // } else if ((iter % 2 == 1) && vel[halfiter] < 0) {
            //   isOnStructFace[iter] = (1 << iter / 2);
            // } else {
            //   isOnStructFace[iter] = 0;
            // }
            isOnStructFace[iter] = (1 << iter / 2);
          }
          isOnStruct |= isOnStructFace[iter]; // OR operator to reduce results
        }
        if (isOnStruct == 6 || isOnStruct == 5) isOnStruct = 4; // Overlaps on front
        else if (isOnStruct == 3 || isOnStruct == 7) isOnStruct = 0; // Overlaps on sides
        isInBound |= isOnStruct; // Update with regular boundary for efficiency

#if 1
        ///< Slip contact        
        // Set cell velocity after grid-block/cell boundary check
        vel[0] = isInBound & 4 ? 0.f : vel[0] * mass; //< vx = mvx / m
        vel[1] = isInBound & 2 ? 0.f : vel[1] * mass; //< vy = mvy / m
        vel[1] += isInBound & 2 ? 0.f : (g_gravity / g_length) * dt;  //< Grav. effect
        vel[2] = isInBound & 1 ? 0.f : vel[2] * mass; //< vz = mvz / m
#endif        

#if 0
        ///< Sticky contact
        if (isInBound) ///< sticky
          vel.set(0.f);
#endif


        vec3 ns; //< Ramp boundary surface normal
        float ys;
        float xs;
        float xo;

        // Start ramp segment definition for OSU flume
        // Based on bathymetry diagram, February
        if (xc < (14.2748/g_length)+offset) {
          // Flat, 0' elev., 0' - 46'10
          ns[0] = 0.f;
          ns[1] = 1.f;
          ns[2] = 0.f;
          xo = offset;
          float yo = offset;
          xs = xc;
          ys = yo;

        } else if (xc > (14.2748/g_length)+offset && xc < (17.9324/g_length)+offset){
          // Flat (adjustable), 0' elev., 46'10 - 58'10
          ns[0] = 0.f;
          ns[1] = 1.f;
          ns[2] = 0.f;
          xo = (14.2748 / g_length) + offset;
          float yo = offset;
          xs = xc;
          ys = yo;

        } else if (xc > (17.9324/g_length)+offset && xc < (28.905/g_length)+offset) {
          // 1:12, 0' elev., 58'10 - 94'10
          ns[0] = -1.f/12.f;
          ns[1] = 1.f;
          ns[2] = 0.f;
          xo = (17.9324 / g_length) + offset;
          float yo = offset;
          xs = xc;
          ys = 1.f/12.f * (xc - xo) + yo;

        } else if (xc > (28.905/g_length)+offset && xc < (43.5356/g_length)+offset) {
          // 1:24, 3' elev., 94'10 - 142'10
          ns[0] = -1.f/24.f;
          ns[1] = 1.f;
          ns[2] = 0.f;
          xo = (28.905 / g_length) + offset;
          float yo = (0.9144 / g_length) + offset;
          xs = xc;
          ys = 1.f/24.f * (xc - xo) + yo;

        } else if (xc > (43.5356/g_length)+offset && xc < (80.1116/g_length)+offset) {
          // Flat, 5' elev., 142'10 - 262'10
          ns[0] = 0.f;
          ns[1] = 1.f;
          ns[2] = 0.f;
          xo = (43.5356 / g_length) + offset;
          float yo = (1.524 / g_length) + offset;
          xs = xc;
          ys = yo;

        } else if (xc > (80.1116/g_length)+offset && xc < (87.4268/g_length)+offset) {
          // 1:12, 5' elev., 262'10 - 286'10
          ns[0] = -1.f/12.f;
          ns[1] = 1.f;
          ns[2] = 0.f;
          xo = (80.1116 / g_length) + offset;
          float yo = (1.524 / g_length) + offset;
          xs = xc;
          ys = 1.f/12.f * (xc - xo) + yo;

        } else {
          // Flat, 7' elev., 286'10 onward
          ns[0]=0.f;
          ns[1]=1.f;
          ns[2]=0.f;
          float yo = (2.1336 / g_length) + offset;
          ys = yo;        
        }

        float ns_mag = sqrt(ns[0]*ns[0] + ns[1]*ns[1] + ns[2]*ns[2]);
        ns = ns / ns_mag;
        float vdotns = vel[0]*ns[0] + vel[1]*ns[1] + vel[2]*ns[2];

        // Boundary thickness and cell distance
        //h  = sqrt((g_dx*g_dx*ns[0]) + (g_dx*g_dx*ns[1]) + (g_dx*g_dx*ns[2]));
        //float r  = sqrt((xc - xs)*(xc - xs) + (yc - ys)*(yc - ys));

        // Decay coefficient
        float ySF;
        if (yc > ys) {
          ySF = 0.f;
        } else if (yc <= ys){
          ySF = 1.f;
        }
        // if (yc > ys + h) {
        //   ySF = 0.f;
        // } else if (yc < ys){
        //   ySF = 1.f;
        // } else {
        //   ySF = (1.f - r/h) * (1.f - r/h);
        // }

        // fbc = -fint - fext - (1/dt)*p
        // a = (1 / mass) * (fint + fext + ySf*fbc) 

        // Adjust velocity relative to surface
        if (0) {
          // Normal adjustment in decay layer, fix below
          if (ySF == 1.f) {
            vel.set(0.f);
          } else if (ySF > 0.f && ySF < 1.f) {
            vel[0] = vel[0] - ySF * (vel[0] - vdotns * ns[0]);
            vel[1] = vel[1] - ySF * (vel[1] - vdotns * ns[1]);
            vel[2] = vel[2] - ySF * (vel[2] - vdotns * ns[2]);  
          }
        }
        if (1) {
          // Free above surface, normal adjusted below
          vel[0] = vel[0] - ySF * (vdotns * ns[0]);
          vel[1] = vel[1] - ySF * (vdotns * ns[1]);
          vel[2] = vel[2] - ySF * (vdotns * ns[2]);
        }


        // Set grid buffer momentum to velocity (m/s) for G2P transfer
        grid_block.val_1d(_1, cidib) = vel[0]; //< vx
        velSqr += vel[0] * vel[0];
        grid_block.val_1d(_2, cidib) = vel[1]; //< vy
        velSqr += vel[1] * vel[1];
        grid_block.val_1d(_3, cidib) = vel[2]; //< vz
        velSqr += vel[2] * vel[2];
      }
      // Reduce velocity^2 from threads, loop
      for (int iter = 1; iter % 32; iter <<= 1) {
        float tmp = __shfl_down_sync(activeMask, velSqr, iter, 32);
        if ((threadIdx.x % 32) + iter < 32)
          velSqr = tmp > velSqr ? tmp : velSqr; //< Block max velocity^2
      }
      if (velSqr > sh_maxvels[threadIdx.x / 32] && (threadIdx.x % 32) == 0)
        sh_maxvels[threadIdx.x / 32] = velSqr; //< Block max vel^2 in shared mem
    }
  }
  __syncthreads();
  // Various assumptions
  for (int interval = numWarps >> 1; interval > 0; interval >>= 1) {
    if (threadIdx.x < interval) {
      if (sh_maxvels[threadIdx.x + interval] > sh_maxvels[threadIdx.x])
        sh_maxvels[threadIdx.x] = sh_maxvels[threadIdx.x + interval];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0)
    atomicMax(maxVel, sh_maxvels[0]); //< Leader thread sets max velocity^2 for CFL
}

template <typename Partition, typename Grid>
__global__ void g2p2g(float dt, float newDt, float curTime,
                      const ParticleBuffer<material_e::JFluid> pbuffer,
                      ParticleBuffer<material_e::JFluid> next_pbuffer,
                      const Partition prev_partition, Partition partition,
                      const Grid grid, Grid next_grid) {
  //============================================================
  // Grid-to-Particle-to Grid Kernel for JFluid material
  // Transfer Scheme:  Affine Particle-in-Cell (MLS-MPM)
  // Shape-function:   Quadratic B-Spline
  // Time Integration: Explicit
  // Material:         JFluid, Weakly incompressible fluid
  //============================================================

  //+-----------------------------------+
  //| FOR AFFINE-PIC TRANSFER
  //+-----------------------------------+
  //| mp    p s mass
  //| xp^n  p v position
  //| vp^n  p v velocity
  //| Fp    p m deformation gradient
  //| fp    p v force
  //+-----------------------------------+
  //| Bp^n  p m affine state
  //| Cp^n  p m velocity derivatives
  //| Dp^n  p m interia-like tensor
  //+-----------------------------------+
  //| mi^n    n s mass
  //| xi      n v position 
  //| vi^n    n v velocity
  //| ~vi^n+1 n v intermediate velocity
  //| fi      n v force
  //+-----------------------------------+
  //| wip^n   n+p s weights
  //| dwip^n  n+p v weight gradients
  //+-----------------------------------+
  //| N(x)    g s interpolation kernel
  //| dx      g s grid spacing
  //| v*      g m cross prod matrix of v
  //+-----------------------------------+
  //| epsilon g t permutation tensor
  //| I       g m identity matrix
  //+-----------------------------------+
  
  // Each particle-block transfers to arena of grid-blocks
  // Arena is 2x2x2 grid-blocks (G2P2G, Quad B-Spline)
  
  // Grid-to-particle buffer size set-up
  static constexpr uint64_t numViPerBlock = g_blockvolume * 3;  //< Velocities per block
  static constexpr uint64_t numViInArena  = numViPerBlock << 3; //< Velocities per arena

  // Particle-to-grid buffer size set-up
  static constexpr uint64_t numMViPerBlock = g_blockvolume * 4;   //< Mass, momentum per block
  static constexpr uint64_t numMViInArena  = numMViPerBlock << 3; //< Mass, momentum per arena

  static constexpr unsigned arenamask = (g_blocksize << 1) - 1;   //< Arena mask
  static constexpr unsigned arenabits = g_blockbits + 1;          //< Arena bits

  extern __shared__ char shmem[];
  
  // Create shared memory grid-to-particle buffer
  // Covers 8 grid blocks (2x2x2), each block 64 nodes (4x4x4)
  // Velocity (vx, vy, vz) held, f32
  using ViArena =
      float(*)[3][g_blocksize << 1][g_blocksize << 1][g_blocksize << 1];
  using ViArenaRef =
      float(&)[3][g_blocksize << 1][g_blocksize << 1][g_blocksize << 1];
  ViArenaRef __restrict__ g2pbuffer = *reinterpret_cast<ViArena>(shmem);
  
  // Create shared memory particle-to-grid buffer
  // Covers 8 grid blocks (2x2x2), each block 64 cells (4x4x4)
  // Mass (m) and momentum (mvx, mvy, mvz) held, f32
  using MViArena =
      float(*)[4][g_blocksize << 1][g_blocksize << 1][g_blocksize << 1];
  using MViArenaRef =
      float(&)[4][g_blocksize << 1][g_blocksize << 1][g_blocksize << 1];
  MViArenaRef __restrict__ p2gbuffer =
      *reinterpret_cast<MViArena>(shmem + numViInArena * sizeof(float));

  int src_blockno = blockIdx.x; //< Source block number
  int ppb = next_pbuffer._ppbs[src_blockno]; //< Particles per block
  if (ppb == 0)
    return; //< Exit of no particles in grid-block
  auto blockid = partition._activeKeys[blockIdx.x]; //< Grid-block ID

  // Thread for each element in g2pbuffer
  for (int base = threadIdx.x; base < numViInArena; base += blockDim.x) {
    char local_block_id = base / numViPerBlock;
    auto blockno = partition.query(
        ivec3{blockid[0] + ((local_block_id & 4) != 0 ? 1 : 0),
              blockid[1] + ((local_block_id & 2) != 0 ? 1 : 0),
              blockid[2] + ((local_block_id & 1) != 0 ? 1 : 0)});
    auto grid_block = grid.ch(_0, blockno);
    int channelid = base % numViPerBlock;
    char c = channelid & 0x3f;
    char cz = channelid & g_blockmask;
    char cy = (channelid >>= g_blockbits) & g_blockmask;
    char cx = (channelid >>= g_blockbits) & g_blockmask;
    channelid >>= g_blockbits;

    // Pull from grid buffer (device)
    float val; //< Set to value from grid buffer
    if (channelid == 0)
      val = grid_block.val_1d(_1, c); //< Grid-node vx (m/s)
    else if (channelid == 1)
      val = grid_block.val_1d(_2, c); //< Grid-node vy (m/s)
    else
      val = grid_block.val_1d(_3, c); //< Grid-node vz (m/s)
    
    // Set element value in g2pbuffer (device)
    g2pbuffer[channelid][cx + (local_block_id & 4 ? g_blocksize : 0)]
             [cy + (local_block_id & 2 ? g_blocksize : 0)]
             [cz + (local_block_id & 1 ? g_blocksize : 0)] = val;
  }
  __syncthreads(); // g2pbuffer is populated
  
  // Loop through p2gbuffer elements, set zero
  for (int base = threadIdx.x; base < numMViInArena; base += blockDim.x) {
    int loc = base;
    char z = loc & arenamask;
    char y = (loc >>= arenabits) & arenamask;
    char x = (loc >>= arenabits) & arenamask;
    p2gbuffer[loc >> arenabits][x][y][z] = 0.f;
  }
  __syncthreads(); // p2gbuffer is populated

  // Loop through particles in block
  for (int pidib = threadIdx.x; pidib < ppb; pidib += blockDim.x) {
    int source_blockno, source_pidib;
    ivec3 base_index;
    {
      int advect =
          next_pbuffer
              ._blockbuckets[src_blockno * g_particle_num_per_block + pidib];
      dir_components(advect / g_particle_num_per_block, base_index);
      base_index += blockid;
      source_blockno = prev_partition.query(base_index);
      source_pidib = advect & (g_particle_num_per_block - 1);
      source_blockno =
          pbuffer._binsts[source_blockno] + source_pidib / g_bin_capacity;
    }
    vec3 pos; //< Positions (x,y,z)
    float J;  //< Det. of Deformation Gradient, ||F||
    {
      auto source_particle_bin = pbuffer.ch(_0, source_blockno); //< Particle bin
      
      // Load positions (x,y,z) and J from particle bin
      pos[0] = source_particle_bin.val(_0, source_pidib % g_bin_capacity); //< x [0.,1.]
      pos[1] = source_particle_bin.val(_1, source_pidib % g_bin_capacity); //< y [0.,1.]
      pos[2] = source_particle_bin.val(_2, source_pidib % g_bin_capacity); //< z [0.,1.]
      J = source_particle_bin.val(_3, source_pidib % g_bin_capacity);      //< J
    }
    ivec3 local_base_index = (pos * g_dx_inv + 0.5f).cast<int>() - 1; //< Particle grid-cell index [0, g_domain_size-1]x3
    vec3 local_pos = pos - local_base_index * g_dx; //< Particle position in cell [0.,g_dx)x3
    base_index = local_base_index; //< Save index at time n

    /// Execute shape-function for grid-to-particle transfer
    /// Using Quad. B-Spline, 3x3x3 grid-nodes
    vec3x3 dws; //< Weight gradients G2P, matrix
    
    // Loop through x,y,z components of local particle position
#pragma unroll 3
    for (int dd = 0; dd < 3; ++dd) {
      // Assume p is already within kernel range [-1.5, 1.5]
      float d =
          (local_pos[dd] - ((int)(local_pos[dd] * g_dx_inv + 0.5) - 1) * g_dx) *
          g_dx_inv; //< Normalized offset, d = p * g_dx_inv 
      
      // Weight gradient for x direction
      dws(dd, 0) = 0.5f * (1.5 - d) * (1.5 - d);
      
      // Weight gradient for y direction
      d -= 1.0f;
      dws(dd, 1) = 0.75 - d * d;
      
      // Weight gradient for z direction
      d = 0.5f + d;
      dws(dd, 2) = 0.5 * d * d;

      // Modify grid-cell index for compatibility with shared g2pbuffer memory
      local_base_index[dd] = ((local_base_index[dd] - 1) & g_blockmask) + 1;
    }

    vec3 vel; //< Advected velocity on particle?
    vel.set(0.f);
    vec9 C; //< Used for matrices of Affine-PIC, (e.g. Bp, Cp)
    C.set(0.f);

    // Dp^n = Dp^n+1 = (1/4) * dx^2 * I (Quad.)
    // Dp^n = Dp^n+1 = (1/3) * dx^2 * I (Cubic)
    // Dp^n = Dp^n+1 = maybe singular, but...  (Trilinear)
    // Wip^n * (Dp^n)^-1 * (xi -xp^n) = dWip^n (Trilinear)
    float Dp_inv; //< Inverse Intertia-Like Tensor (1/m^2)
    float scale = g_length * g_length; //< Area scale (m^2)
    Dp_inv = g_D_inv / scale; //< Scalar 4/(dx^2) for Quad. B-Spline
    
    // Loop through 3x3x3 grid-nodes [i,j,k] for Quad. B-Spline shape-func.
#pragma unroll 3
    for (char i = 0; i < 3; i++)
#pragma unroll 3
      for (char j = 0; j < 3; j++)
#pragma unroll 3
        for (char k = 0; k < 3; k++) {
          // Perform G2P transfer for grid-node [i,j,k] and particle
          vec3 xixp = vec3{(float)i, (float)j, (float)k} * g_dx - local_pos; //< Pos. diff., (xi - xp)
          float W = dws(0, i) * dws(1, j) * dws(2, k); //< Weight value for grid node [i,j,k] 

          // Pull grid node velocity vector from shared g2pbuffer memory
          vec3 vi{g2pbuffer[0][local_base_index[0] + i][local_base_index[1] + j]
                           [local_base_index[2] + k],
                  g2pbuffer[1][local_base_index[0] + i][local_base_index[1] + j]
                           [local_base_index[2] + k],
                  g2pbuffer[2][local_base_index[0] + i][local_base_index[1] + j]
                           [local_base_index[2] + k]}; //< Grid-node velocity, ([1m]/s)
          vel += vi * W; //< Particle velocity increment from grid-node ([1m]/s)

          // Affine state (m^2 / s) increment for particle from grid-node [i,j,k]
          // Bp^n+1 = Sum_i( Wip^n * ~vi^n+1 * (xi - xp^n).T )
          C[0] += W * vi[0] * xixp[0] * scale;
          C[1] += W * vi[1] * xixp[0] * scale;
          C[2] += W * vi[2] * xixp[0] * scale;
          C[3] += W * vi[0] * xixp[1] * scale;
          C[4] += W * vi[1] * xixp[1] * scale;
          C[5] += W * vi[2] * xixp[1] * scale;
          C[6] += W * vi[0] * xixp[2] * scale;
          C[7] += W * vi[1] * xixp[2] * scale;
          C[8] += W * vi[2] * xixp[2] * scale;
        }

    // Advect particle position increment from G2P B-Spline
    pos += vel * dt; //< xp^n+1 = xp^n + (vp^n+1 * dt)

    /// Begin particle material update
    // Advance J^n (volume ratio, V/Vo, ||F^n||)
    // J^n+1 = (1 + tr(Bp^n+1) * Dp^-1 * dt) * J^n
    J = (1 + (C[0] + C[4] + C[8]) * Dp_inv * dt) * J;
    if (J < 0.1)
      J = 0.1; //< Lower-bound
    vec9 contrib;     //< Used for APIC matrix intermediates
    {
      // Update particle quantities
      // Vp^n+1 = Jp^n+1 * Vo 
      float voln = J * pbuffer.volume; //< Particle volume (m^3)
      
      // Pressure, Murnaghan-Tait state equation (JB)
      // P = (Ko/n) [(Vo/V)^(n) - 1] + Patm = (bulk/gamma) [J^(-gamma) - 1] + Patm      
       float pressure = (pbuffer.bulk / pbuffer.gamma) * 
         (powf(J, -pbuffer.gamma) - 1.f); //< Pressure (Pa)
      //float pressure = (1.5 * pbuffer.bulk) * (powf(J,-7.f/3.f) - powf(J, -5.f/3.f)) * (1.f + 0.75f * (pbuffer.gamma - 4.f) * (powf(J,-2.f/3.f) - 1.f));

      {
        // Torque matrix (N * m)
        // Tp = ((Bp + Bp.T) * Dp^-1 * visco - pressure * I) * Vp
        // ((m^2 / s) * (1 / m^2) * (N s / m^2) - (N / m^2)) * (m^3) = (N * m)
        contrib[0] = ((C[0] + C[0]) * Dp_inv * pbuffer.visco - pressure) * voln;
        contrib[1] = ((C[1] + C[3]) * Dp_inv * pbuffer.visco) * voln;
        contrib[2] = ((C[2] + C[6]) * Dp_inv * pbuffer.visco) * voln;
        contrib[3] = ((C[3] + C[1]) * Dp_inv * pbuffer.visco) * voln;
        contrib[4] = ((C[4] + C[4]) * Dp_inv * pbuffer.visco - pressure) * voln;
        contrib[5] = ((C[5] + C[7]) * Dp_inv * pbuffer.visco) * voln;
        contrib[6] = ((C[6] + C[2]) * Dp_inv * pbuffer.visco) * voln;
        contrib[7] = ((C[7] + C[5]) * Dp_inv * pbuffer.visco) * voln;
        contrib[8] = ((C[8] + C[8]) * Dp_inv * pbuffer.visco - pressure) * voln;
      }
      // Mass flow-rate matrix (kg / s)
      // mp * Cp = ((Bp * mp) - (Tp * dt)) * Dp^-1 
      // ((m^2 / s) * (kg) - (N*m) * (s)) * (1/m^2) = (kg * m^2 / s) * (1/m^2) = (kg/s)
      contrib = (C * pbuffer.mass - contrib * newDt) * Dp_inv;
      {
        // Load appropiate particle buffer bin at n+1
        auto particle_bin = next_pbuffer.ch(
            _0, next_pbuffer._binsts[src_blockno] + pidib / g_bin_capacity);

        // Set particle buffer positions (x,y,z) and attributes (J) at n+1
        particle_bin.val(_0, pidib % g_bin_capacity) = pos[0]; //< x ([1m])
        particle_bin.val(_1, pidib % g_bin_capacity) = pos[1]; //< y ([1m])
        particle_bin.val(_2, pidib % g_bin_capacity) = pos[2]; //< z ([1m])
        particle_bin.val(_3, pidib % g_bin_capacity) = J;      //< J
      }
    }

    local_base_index = (pos * g_dx_inv + 0.5f).cast<int>() - 1; //< Particle cell index in 3x3. [-1,1]
    {
      int dirtag = dir_offset((base_index - 1) / g_blocksize -
                              (local_base_index - 1) / g_blocksize); //< Particle index offset (n --> n+1)
      next_pbuffer.add_advection(partition, local_base_index - 1, dirtag,
                                 pidib); //< Update particle buffer advection tagging
    }

    // Begin Particle-to-Grid transfer
    // Loop through particle x,y,z position
#pragma unroll 3
    for (char dd = 0; dd < 3; ++dd) {
      local_pos[dd] = pos[dd] - local_base_index[dd] * g_dx;
      float d =
          (local_pos[dd] - ((int)(local_pos[dd] * g_dx_inv + 0.5) - 1) * g_dx) *
          g_dx_inv;
      dws(dd, 0) = 0.5f * (1.5 - d) * (1.5 - d);
      d -= 1.0f;
      dws(dd, 1) = 0.75 - d * d;
      d = 0.5f + d;
      dws(dd, 2) = 0.5 * d * d;

      local_base_index[dd] = (((base_index[dd] - 1) & g_blockmask) + 1) +
                             local_base_index[dd] - base_index[dd];
    }
    // Loop through 3x3 grid-nodes for B-Spline
#pragma unroll 3
    for (char i = 0; i < 3; i++)
#pragma unroll 3
      for (char j = 0; j < 3; j++)
#pragma unroll 3
        for (char k = 0; k < 3; k++) {
          pos = vec3{(float)i, (float)j, (float)k} * g_dx - local_pos; //< (xi - xp) ([1m])
          float W = dws(0, i) * dws(1, j) * dws(2, k); //< Weighting for grid node [i,j,k] 
          auto wm = pbuffer.mass * W;                  //< Weighted particle mass (kg)
          
          // Add grid node's [i,j,k] particle increment to shared p2gbuffer memory
          // mi      = Sum( Wip * mp )
          // mi * vi = Sum( Wip * mp * (vp + (Bp * Dp^-1 * (xi - xp).T) ) )
          // mi * vi = Sum((Wip * mp * vp) + (mp * Cp * (xi - xp).T * Wip))
          atomicAdd(
              &p2gbuffer[0][local_base_index[0] + i][local_base_index[1] + j]
                        [local_base_index[2] + k],
              wm); //< Mass increment (kg)
          atomicAdd(
              &p2gbuffer[1][local_base_index[0] + i][local_base_index[1] + j]
                        [local_base_index[2] + k],
              wm * vel[0] + (contrib[0] * pos[0] + contrib[3] * pos[1] +
                             contrib[6] * pos[2]) *
                                W); //< Momentum x increment (kg * m / s)
          atomicAdd(
              &p2gbuffer[2][local_base_index[0] + i][local_base_index[1] + j]
                        [local_base_index[2] + k],
              wm * vel[1] + (contrib[1] * pos[0] + contrib[4] * pos[1] +
                             contrib[7] * pos[2]) *
                                W); //< Momentum y increment (kg * m / s)
          atomicAdd(
              &p2gbuffer[3][local_base_index[0] + i][local_base_index[1] + j]
                        [local_base_index[2] + k],
              wm * vel[2] + (contrib[2] * pos[0] + contrib[5] * pos[1] +
                             contrib[8] * pos[2]) *
                                W); //< Momentum z increment (kg * m / s)
        }
  }
  __syncthreads();
  /// Finished particle --> p2gbuffer transfer
  
  /// Begin p2gbuffer   --> next grid-buffer transfer reduction
  /// arena no, channel no, cell no
  for (int base = threadIdx.x; base < numMViInArena; base += blockDim.x) {
    char local_block_id = base / numMViPerBlock;
    auto blockno = partition.query(
        ivec3{blockid[0] + ((local_block_id & 4) != 0 ? 1 : 0),
              blockid[1] + ((local_block_id & 2) != 0 ? 1 : 0),
              blockid[2] + ((local_block_id & 1) != 0 ? 1 : 0)});
    
    /// Avoid most conflicts in p2gbuffer --> next grid with following scheme
    int channelid = base & (numMViPerBlock - 1);         //< Thread channel ID, threadIdx.x % (4*64*4) --> [0,1023], reduces conflicts
    char c = channelid % g_blockvolume;                  //< Cell ID in grid buffer, [0,1023] % 64
    char cz = channelid & g_blockmask;                   //< Cell ID z in p2gbuffer, [0,1023] % 4
    char cy = (channelid >>= g_blockbits) & g_blockmask; //< Cell ID y in p2gbuffer, [0,1023] / 2 / 2 % 4
    char cx = (channelid >>= g_blockbits) & g_blockmask; //< Cell ID x in p2gbuffer, [0,1023] / 2 / 2 % 4
    channelid >>= g_blockbits;                           //< ChannelID in p2gbuffer, [0,1023] / 2 / 2

    // Reduce (m, mvx, mvy, mvz) from shared p2gbuffer to next grid buffer
    float val =
        p2gbuffer[channelid][cx + (local_block_id & 4 ? g_blocksize : 0)]
                 [cy + (local_block_id & 2 ? g_blocksize : 0)]
                 [cz + (local_block_id & 1 ? g_blocksize : 0)]; //< Pull (coalesced?) from shared p2gbuffer
    if (channelid == 0)
      atomicAdd(&next_grid.ch(_0, blockno).val_1d(_0, c), val); //< Add mass (kg)
    else if (channelid == 1)
      atomicAdd(&next_grid.ch(_0, blockno).val_1d(_1, c), val); //< Add x momentum (kg * [1m] / s)
    else if (channelid == 2)
      atomicAdd(&next_grid.ch(_0, blockno).val_1d(_2, c), val); //< Add y momentum (kg * [1m] / s)
    else
      atomicAdd(&next_grid.ch(_0, blockno).val_1d(_3, c), val); //< Add z momentum (kg * [1m] / s)
  }
}

template <typename Partition, typename Grid>
__global__ void g2p2g(float dt, float newDt, float curTime,
                      const ParticleBuffer<material_e::FixedCorotated> pbuffer,
                      ParticleBuffer<material_e::FixedCorotated> next_pbuffer,
                      const Partition prev_partition, Partition partition,
                      const Grid grid, Grid next_grid) {
  static constexpr uint64_t numViPerBlock = g_blockvolume * 3;
  static constexpr uint64_t numViInArena = numViPerBlock << 3;

  static constexpr uint64_t numMViPerBlock = g_blockvolume * 4;
  static constexpr uint64_t numMViInArena = numMViPerBlock << 3;

  static constexpr unsigned arenamask = (g_blocksize << 1) - 1;
  static constexpr unsigned arenabits = g_blockbits + 1;

  extern __shared__ char shmem[];
  using ViArena =
      float(*)[3][g_blocksize << 1][g_blocksize << 1][g_blocksize << 1];
  using ViArenaRef =
      float(&)[3][g_blocksize << 1][g_blocksize << 1][g_blocksize << 1];
  ViArenaRef __restrict__ g2pbuffer = *reinterpret_cast<ViArena>(shmem);
  using MViArena =
      float(*)[4][g_blocksize << 1][g_blocksize << 1][g_blocksize << 1];
  using MViArenaRef =
      float(&)[4][g_blocksize << 1][g_blocksize << 1][g_blocksize << 1];
  MViArenaRef __restrict__ p2gbuffer =
      *reinterpret_cast<MViArena>(shmem + numViInArena * sizeof(float));

  int src_blockno = blockIdx.x;
  int ppb = next_pbuffer._ppbs[src_blockno];
  if (ppb == 0)
    return;
  auto blockid = partition._activeKeys[blockIdx.x];

  for (int base = threadIdx.x; base < numViInArena; base += blockDim.x) {
    char local_block_id = base / numViPerBlock;
    auto blockno = partition.query(
        ivec3{blockid[0] + ((local_block_id & 4) != 0 ? 1 : 0),
              blockid[1] + ((local_block_id & 2) != 0 ? 1 : 0),
              blockid[2] + ((local_block_id & 1) != 0 ? 1 : 0)});
    auto grid_block = grid.ch(_0, blockno);
    int channelid = base % numViPerBlock;
    char c = channelid & 0x3f;
    char cz = channelid & g_blockmask;
    char cy = (channelid >>= g_blockbits) & g_blockmask;
    char cx = (channelid >>= g_blockbits) & g_blockmask;
    channelid >>= g_blockbits;

    float val;
    if (channelid == 0)
      val = grid_block.val_1d(_1, c);
    else if (channelid == 1)
      val = grid_block.val_1d(_2, c);
    else
      val = grid_block.val_1d(_3, c);
    g2pbuffer[channelid][cx + (local_block_id & 4 ? g_blocksize : 0)]
             [cy + (local_block_id & 2 ? g_blocksize : 0)]
             [cz + (local_block_id & 1 ? g_blocksize : 0)] = val;
  }
  __syncthreads();
  for (int base = threadIdx.x; base < numMViInArena; base += blockDim.x) {
    int loc = base;
    char z = loc & arenamask;
    char y = (loc >>= arenabits) & arenamask;
    char x = (loc >>= arenabits) & arenamask;
    p2gbuffer[loc >> arenabits][x][y][z] = 0.f;
  }
  __syncthreads();

  for (int pidib = threadIdx.x; pidib < ppb; pidib += blockDim.x) {
    int source_blockno, source_pidib;
    ivec3 base_index;
    {
      int advect =
          next_pbuffer
              ._blockbuckets[src_blockno * g_particle_num_per_block + pidib];
      dir_components(advect / g_particle_num_per_block, base_index);
      base_index += blockid;
      source_blockno = prev_partition.query(base_index);
      source_pidib = advect & (g_particle_num_per_block - 1);
      source_blockno =
          pbuffer._binsts[source_blockno] + source_pidib / g_bin_capacity;
    }
    vec3 pos;
    {
      auto source_particle_bin = pbuffer.ch(_0, source_blockno);
      pos[0] = source_particle_bin.val(_0, source_pidib % g_bin_capacity);
      pos[1] = source_particle_bin.val(_1, source_pidib % g_bin_capacity);
      pos[2] = source_particle_bin.val(_2, source_pidib % g_bin_capacity);
    }
    ivec3 local_base_index = (pos * g_dx_inv + 0.5f).cast<int>() - 1;
    vec3 local_pos = pos - local_base_index * g_dx;
    base_index = local_base_index;

    vec3x3 dws;
#pragma unroll 3
    for (int dd = 0; dd < 3; ++dd) {
      float d =
          (local_pos[dd] - ((int)(local_pos[dd] * g_dx_inv + 0.5) - 1) * g_dx) *
          g_dx_inv;
      dws(dd, 0) = 0.5f * (1.5 - d) * (1.5 - d);
      d -= 1.0f;
      dws(dd, 1) = 0.75 - d * d;
      d = 0.5f + d;
      dws(dd, 2) = 0.5 * d * d;
      local_base_index[dd] = ((local_base_index[dd] - 1) & g_blockmask) + 1;
    }
    vec3 vel;
    vel.set(0.f);
    vec9 C;
    C.set(0.f);
#pragma unroll 3
    for (char i = 0; i < 3; i++)
#pragma unroll 3
      for (char j = 0; j < 3; j++)
#pragma unroll 3
        for (char k = 0; k < 3; k++) {
          vec3 xixp = vec3{(float)i, (float)j, (float)k} * g_dx - local_pos;
          float W = dws(0, i) * dws(1, j) * dws(2, k);
          vec3 vi{g2pbuffer[0][local_base_index[0] + i][local_base_index[1] + j]
                           [local_base_index[2] + k],
                  g2pbuffer[1][local_base_index[0] + i][local_base_index[1] + j]
                           [local_base_index[2] + k],
                  g2pbuffer[2][local_base_index[0] + i][local_base_index[1] + j]
                           [local_base_index[2] + k]};
          vel += vi * W;
          //vel[1] = 0.f;
          //vel[2] = 0.f;
          C[0] += W * vi[0] * xixp[0];
          C[1] += W * vi[1] * xixp[0];
          C[2] += W * vi[2] * xixp[0];
          C[3] += W * vi[0] * xixp[1];
          C[4] += W * vi[1] * xixp[1];
          C[5] += W * vi[2] * xixp[1];
          C[6] += W * vi[0] * xixp[2];
          C[7] += W * vi[1] * xixp[2];
          C[8] += W * vi[2] * xixp[2];
        }
    pos += vel * dt;

#pragma unroll 9
    for (int d = 0; d < 9; ++d)
      // dWip = Bp * Dp^-1 * dt + I
      // (m^2 / s) * (1 / m^2) * (s) = ( )
      dws.val(d) = C[d] * dt * g_D_inv + ((d & 0x3) ? 0.f : 1.f);

    vec9 contrib;
    {
      vec9 F; //< Deformation Gradient, Fp^n+1
      auto source_particle_bin = pbuffer.ch(_0, source_blockno);
      // Set Deformation Gradient, Fp^n
      contrib[0] = source_particle_bin.val(_3, source_pidib % g_bin_capacity);
      contrib[1] = source_particle_bin.val(_4, source_pidib % g_bin_capacity);
      contrib[2] = source_particle_bin.val(_5, source_pidib % g_bin_capacity);
      contrib[3] = source_particle_bin.val(_6, source_pidib % g_bin_capacity);
      contrib[4] = source_particle_bin.val(_7, source_pidib % g_bin_capacity);
      contrib[5] = source_particle_bin.val(_8, source_pidib % g_bin_capacity);
      contrib[6] = source_particle_bin.val(_9, source_pidib % g_bin_capacity);
      contrib[7] = source_particle_bin.val(_10, source_pidib % g_bin_capacity);
      contrib[8] = source_particle_bin.val(_11, source_pidib % g_bin_capacity);
      // Set F = Fp^n+1 using dWip^n+1 and Fp^n
      matrixMatrixMultiplication3d(dws.data(), contrib.data(), F.data());
      {
        auto particle_bin = next_pbuffer.ch(
            _0, next_pbuffer._binsts[src_blockno] + pidib / g_bin_capacity);
        particle_bin.val(_0,  pidib % g_bin_capacity) = pos[0]; //< x (m)
        particle_bin.val(_1,  pidib % g_bin_capacity) = pos[1]; //< y (m)
        particle_bin.val(_2,  pidib % g_bin_capacity) = pos[2]; //< z (m)
        particle_bin.val(_3,  pidib % g_bin_capacity) = F[0];
        particle_bin.val(_4,  pidib % g_bin_capacity) = F[1];
        particle_bin.val(_5,  pidib % g_bin_capacity) = F[2];
        particle_bin.val(_6,  pidib % g_bin_capacity) = F[3];
        particle_bin.val(_7,  pidib % g_bin_capacity) = F[4];
        particle_bin.val(_8,  pidib % g_bin_capacity) = F[5];
        particle_bin.val(_9,  pidib % g_bin_capacity) = F[6];
        particle_bin.val(_10, pidib % g_bin_capacity) = F[7];
        particle_bin.val(_11, pidib % g_bin_capacity) = F[8];
      }
      // Torque matrix (N * m)
      // Tp = Stress * Volume
      compute_stress_fixedcorotated(pbuffer.volume, pbuffer.mu, pbuffer.lambda,
                                    F, contrib);
      // Mass-flow rate (kg / s)
      // mp * Cp^n+1 = (Bp^n * mp - Tp * dt) * Dp^n+1
      contrib = (C * pbuffer.mass - contrib * newDt) * g_D_inv;
    }

    local_base_index = (pos * g_dx_inv + 0.5f).cast<int>() - 1;
    {
      int dirtag = dir_offset((base_index - 1) / g_blocksize -
                              (local_base_index - 1) / g_blocksize);
      next_pbuffer.add_advection(partition, local_base_index - 1, dirtag,
                                 pidib);
      // partition.add_advection(local_base_index - 1, dirtag, pidib);
    }
    // dws[d] = bspline_weight(local_pos[d]);

#pragma unroll 3
    for (char dd = 0; dd < 3; ++dd) {
      local_pos[dd] = pos[dd] - local_base_index[dd] * g_dx;
      float d =
          (local_pos[dd] - ((int)(local_pos[dd] * g_dx_inv + 0.5) - 1) * g_dx) *
          g_dx_inv;
      dws(dd, 0) = 0.5f * (1.5 - d) * (1.5 - d);
      d -= 1.0f;
      dws(dd, 1) = 0.75 - d * d;
      d = 0.5f + d;
      dws(dd, 2) = 0.5 * d * d;

      local_base_index[dd] = (((base_index[dd] - 1) & g_blockmask) + 1) +
                             local_base_index[dd] - base_index[dd];
    }
#pragma unroll 3
    for (char i = 0; i < 3; i++)
#pragma unroll 3
      for (char j = 0; j < 3; j++)
#pragma unroll 3
        for (char k = 0; k < 3; k++) {
          pos = vec3{(float)i, (float)j, (float)k} * g_dx - local_pos;
          float W = dws(0, i) * dws(1, j) * dws(2, k);
          auto wm = pbuffer.mass * W;
          atomicAdd(
              &p2gbuffer[0][local_base_index[0] + i][local_base_index[1] + j]
                        [local_base_index[2] + k],
              wm);
          atomicAdd(
              &p2gbuffer[1][local_base_index[0] + i][local_base_index[1] + j]
                        [local_base_index[2] + k],
              wm * vel[0] + (contrib[0] * pos[0] + contrib[3] * pos[1] +
                             contrib[6] * pos[2]) *
                                W);
          atomicAdd(
              &p2gbuffer[2][local_base_index[0] + i][local_base_index[1] + j]
                        [local_base_index[2] + k],
              wm * vel[1] + (contrib[1] * pos[0] + contrib[4] * pos[1] +
                             contrib[7] * pos[2]) *
                                W);
          atomicAdd(
              &p2gbuffer[3][local_base_index[0] + i][local_base_index[1] + j]
                        [local_base_index[2] + k],
              wm * vel[2] + (contrib[2] * pos[0] + contrib[5] * pos[1] +
                             contrib[8] * pos[2]) *
                                W);
        }
  }
  __syncthreads();
  /// arena no, channel no, cell no
  for (int base = threadIdx.x; base < numMViInArena; base += blockDim.x) {
    char local_block_id = base / numMViPerBlock;
    auto blockno = partition.query(
        ivec3{blockid[0] + ((local_block_id & 4) != 0 ? 1 : 0),
              blockid[1] + ((local_block_id & 2) != 0 ? 1 : 0),
              blockid[2] + ((local_block_id & 1) != 0 ? 1 : 0)});
    // auto grid_block = next_grid.template ch<0>(blockno);
    int channelid = base & (numMViPerBlock - 1);
    char c = channelid % g_blockvolume;
    char cz = channelid & g_blockmask;
    char cy = (channelid >>= g_blockbits) & g_blockmask;
    char cx = (channelid >>= g_blockbits) & g_blockmask;
    channelid >>= g_blockbits;

    float val =
        p2gbuffer[channelid][cx + (local_block_id & 4 ? g_blocksize : 0)]
                 [cy + (local_block_id & 2 ? g_blocksize : 0)]
                 [cz + (local_block_id & 1 ? g_blocksize : 0)];
    if (channelid == 0)
      atomicAdd(&next_grid.ch(_0, blockno).val_1d(_0, c), val);
    else if (channelid == 1)
      atomicAdd(&next_grid.ch(_0, blockno).val_1d(_1, c), val);
    else if (channelid == 2)
      atomicAdd(&next_grid.ch(_0, blockno).val_1d(_2, c), val);
    else
      atomicAdd(&next_grid.ch(_0, blockno).val_1d(_3, c), val);
  }
}

template <typename Partition, typename Grid>
__global__ void g2p2g(float dt, float newDt, float curTime,
                      const ParticleBuffer<material_e::Sand> pbuffer,
                      ParticleBuffer<material_e::Sand> next_pbuffer,
                      const Partition prev_partition, Partition partition,
                      const Grid grid, Grid next_grid) {
  static constexpr uint64_t numViPerBlock = g_blockvolume * 3;
  static constexpr uint64_t numViInArena = numViPerBlock << 3;

  static constexpr uint64_t numMViPerBlock = g_blockvolume * 4;
  static constexpr uint64_t numMViInArena = numMViPerBlock << 3;

  static constexpr unsigned arenamask = (g_blocksize << 1) - 1;
  static constexpr unsigned arenabits = g_blockbits + 1;

  extern __shared__ char shmem[];
  using ViArena =
      float(*)[3][g_blocksize << 1][g_blocksize << 1][g_blocksize << 1];
  using ViArenaRef =
      float(&)[3][g_blocksize << 1][g_blocksize << 1][g_blocksize << 1];
  ViArenaRef __restrict__ g2pbuffer = *reinterpret_cast<ViArena>(shmem);
  using MViArena =
      float(*)[4][g_blocksize << 1][g_blocksize << 1][g_blocksize << 1];
  using MViArenaRef =
      float(&)[4][g_blocksize << 1][g_blocksize << 1][g_blocksize << 1];
  MViArenaRef __restrict__ p2gbuffer =
      *reinterpret_cast<MViArena>(shmem + numViInArena * sizeof(float));

  int src_blockno = blockIdx.x;
  auto blockid = partition._activeKeys[blockIdx.x];

  for (int base = threadIdx.x; base < numViInArena; base += blockDim.x) {
    char local_block_id = base / numViPerBlock;
    auto blockno = partition.query(
        ivec3{blockid[0] + ((local_block_id & 4) != 0 ? 1 : 0),
              blockid[1] + ((local_block_id & 2) != 0 ? 1 : 0),
              blockid[2] + ((local_block_id & 1) != 0 ? 1 : 0)});
    auto grid_block = grid.ch(_0, blockno);
    int channelid = base % numViPerBlock;
    char c = channelid & 0x3f;
    char cz = channelid & g_blockmask;
    char cy = (channelid >>= g_blockbits) & g_blockmask;
    char cx = (channelid >>= g_blockbits) & g_blockmask;
    channelid >>= g_blockbits;

    float val;
    if (channelid == 0)
      val = grid_block.val_1d(_1, c);
    else if (channelid == 1)
      val = grid_block.val_1d(_2, c);
    else
      val = grid_block.val_1d(_3, c);
    g2pbuffer[channelid][cx + (local_block_id & 4 ? g_blocksize : 0)]
             [cy + (local_block_id & 2 ? g_blocksize : 0)]
             [cz + (local_block_id & 1 ? g_blocksize : 0)] = val;
  }
  __syncthreads();
  for (int base = threadIdx.x; base < numMViInArena; base += blockDim.x) {
    int loc = base;
    char z = loc & arenamask;
    char y = (loc >>= arenabits) & arenamask;
    char x = (loc >>= arenabits) & arenamask;
    p2gbuffer[loc >> arenabits][x][y][z] = 0.f;
  }
  __syncthreads();

  for (int pidib = threadIdx.x; pidib < next_pbuffer._ppbs[src_blockno];
       pidib += blockDim.x) {
    int source_blockno, source_pidib;
    ivec3 base_index;
    {
      int advect =
          next_pbuffer
              ._blockbuckets[src_blockno * g_particle_num_per_block + pidib];
      dir_components(advect / g_particle_num_per_block, base_index);
      base_index += blockid;
      source_blockno = prev_partition.query(base_index);
      source_pidib = advect & (g_particle_num_per_block - 1);
      source_blockno =
          pbuffer._binsts[source_blockno] + source_pidib / g_bin_capacity;
    }
    vec3 pos;
    {
      auto source_particle_bin = pbuffer.ch(_0, source_blockno);
      pos[0] = source_particle_bin.val(_0, source_pidib % g_bin_capacity);
      pos[1] = source_particle_bin.val(_1, source_pidib % g_bin_capacity);
      pos[2] = source_particle_bin.val(_2, source_pidib % g_bin_capacity);
    }
    ivec3 local_base_index = (pos * g_dx_inv + 0.5f).cast<int>() - 1;
    vec3 local_pos = pos - local_base_index * g_dx;
    base_index = local_base_index;

    vec3x3 dws;
#pragma unroll 3
    for (int dd = 0; dd < 3; ++dd) {
      float d =
          (local_pos[dd] - ((int)(local_pos[dd] * g_dx_inv + 0.5) - 1) * g_dx) *
          g_dx_inv;
      dws(dd, 0) = 0.5f * (1.5 - d) * (1.5 - d);
      d -= 1.0f;
      dws(dd, 1) = 0.75 - d * d;
      d = 0.5f + d;
      dws(dd, 2) = 0.5 * d * d;
      local_base_index[dd] = ((local_base_index[dd] - 1) & g_blockmask) + 1;
    }
    vec3 vel;
    vel.set(0.f);
    vec9 C;
    C.set(0.f);
#pragma unroll 3
    for (char i = 0; i < 3; i++)
#pragma unroll 3
      for (char j = 0; j < 3; j++)
#pragma unroll 3
        for (char k = 0; k < 3; k++) {
          vec3 xixp = vec3{(float)i, (float)j, (float)k} * g_dx - local_pos;
          float W = dws(0, i) * dws(1, j) * dws(2, k);
          vec3 vi{g2pbuffer[0][local_base_index[0] + i][local_base_index[1] + j]
                           [local_base_index[2] + k],
                  g2pbuffer[1][local_base_index[0] + i][local_base_index[1] + j]
                           [local_base_index[2] + k],
                  g2pbuffer[2][local_base_index[0] + i][local_base_index[1] + j]
                           [local_base_index[2] + k]};
          vel += vi * W;
          C[0] += W * vi[0] * xixp[0];
          C[1] += W * vi[1] * xixp[0];
          C[2] += W * vi[2] * xixp[0];
          C[3] += W * vi[0] * xixp[1];
          C[4] += W * vi[1] * xixp[1];
          C[5] += W * vi[2] * xixp[1];
          C[6] += W * vi[0] * xixp[2];
          C[7] += W * vi[1] * xixp[2];
          C[8] += W * vi[2] * xixp[2];
        }
    pos += vel * dt;

#pragma unroll 9
    for (int d = 0; d < 9; ++d)
      dws.val(d) = C[d] * dt * g_D_inv + ((d & 0x3) ? 0.f : 1.f);

    vec9 contrib;
    {
      vec9 F;
      float logJp;
      auto source_particle_bin = pbuffer.ch(_0, source_blockno);
      contrib[0] = source_particle_bin.val(_3, source_pidib % g_bin_capacity);
      contrib[1] = source_particle_bin.val(_4, source_pidib % g_bin_capacity);
      contrib[2] = source_particle_bin.val(_5, source_pidib % g_bin_capacity);
      contrib[3] = source_particle_bin.val(_6, source_pidib % g_bin_capacity);
      contrib[4] = source_particle_bin.val(_7, source_pidib % g_bin_capacity);
      contrib[5] = source_particle_bin.val(_8, source_pidib % g_bin_capacity);
      contrib[6] = source_particle_bin.val(_9, source_pidib % g_bin_capacity);
      contrib[7] = source_particle_bin.val(_10, source_pidib % g_bin_capacity);
      contrib[8] = source_particle_bin.val(_11, source_pidib % g_bin_capacity);
      logJp = source_particle_bin.val(_12, source_pidib % g_bin_capacity);

      matrixMatrixMultiplication3d(dws.data(), contrib.data(), F.data());
      compute_stress_sand(pbuffer.volume, pbuffer.mu, pbuffer.lambda,
                          pbuffer.cohesion, pbuffer.beta, pbuffer.yieldSurface,
                          pbuffer.volumeCorrection, logJp, F, contrib);
      {
        auto particle_bin = next_pbuffer.ch(
            _0, next_pbuffer._binsts[src_blockno] + pidib / g_bin_capacity);
        particle_bin.val(_0, pidib % g_bin_capacity) = pos[0];
        particle_bin.val(_1, pidib % g_bin_capacity) = pos[1];
        particle_bin.val(_2, pidib % g_bin_capacity) = pos[2];
        particle_bin.val(_3, pidib % g_bin_capacity) = F[0];
        particle_bin.val(_4, pidib % g_bin_capacity) = F[1];
        particle_bin.val(_5, pidib % g_bin_capacity) = F[2];
        particle_bin.val(_6, pidib % g_bin_capacity) = F[3];
        particle_bin.val(_7, pidib % g_bin_capacity) = F[4];
        particle_bin.val(_8, pidib % g_bin_capacity) = F[5];
        particle_bin.val(_9, pidib % g_bin_capacity) = F[6];
        particle_bin.val(_10, pidib % g_bin_capacity) = F[7];
        particle_bin.val(_11, pidib % g_bin_capacity) = F[8];
        particle_bin.val(_12, pidib % g_bin_capacity) = logJp;
      }

      contrib = (C * pbuffer.mass - contrib * newDt) * g_D_inv;
    }

    local_base_index = (pos * g_dx_inv + 0.5f).cast<int>() - 1;
    {
      int dirtag = dir_offset((base_index - 1) / g_blocksize -
                              (local_base_index - 1) / g_blocksize);
      next_pbuffer.add_advection(partition, local_base_index - 1, dirtag,
                                 pidib);
      // partition.add_advection(local_base_index - 1, dirtag, pidib);
    }
    // dws[d] = bspline_weight(local_pos[d]);

#pragma unroll 3
    for (char dd = 0; dd < 3; ++dd) {
      local_pos[dd] = pos[dd] - local_base_index[dd] * g_dx;
      float d =
          (local_pos[dd] - ((int)(local_pos[dd] * g_dx_inv + 0.5) - 1) * g_dx) *
          g_dx_inv;
      dws(dd, 0) = 0.5f * (1.5 - d) * (1.5 - d);
      d -= 1.0f;
      dws(dd, 1) = 0.75 - d * d;
      d = 0.5f + d;
      dws(dd, 2) = 0.5 * d * d;

      local_base_index[dd] = (((base_index[dd] - 1) & g_blockmask) + 1) +
                             local_base_index[dd] - base_index[dd];
    }
#pragma unroll 3
    for (char i = 0; i < 3; i++)
#pragma unroll 3
      for (char j = 0; j < 3; j++)
#pragma unroll 3
        for (char k = 0; k < 3; k++) {
          pos = vec3{(float)i, (float)j, (float)k} * g_dx - local_pos;
          float W = dws(0, i) * dws(1, j) * dws(2, k);
          auto wm = pbuffer.mass * W;
          atomicAdd(
              &p2gbuffer[0][local_base_index[0] + i][local_base_index[1] + j]
                        [local_base_index[2] + k],
              wm);
          atomicAdd(
              &p2gbuffer[1][local_base_index[0] + i][local_base_index[1] + j]
                        [local_base_index[2] + k],
              wm * vel[0] + (contrib[0] * pos[0] + contrib[3] * pos[1] +
                             contrib[6] * pos[2]) *
                                W);
          atomicAdd(
              &p2gbuffer[2][local_base_index[0] + i][local_base_index[1] + j]
                        [local_base_index[2] + k],
              wm * vel[1] + (contrib[1] * pos[0] + contrib[4] * pos[1] +
                             contrib[7] * pos[2]) *
                                W);
          atomicAdd(
              &p2gbuffer[3][local_base_index[0] + i][local_base_index[1] + j]
                        [local_base_index[2] + k],
              wm * vel[2] + (contrib[2] * pos[0] + contrib[5] * pos[1] +
                             contrib[8] * pos[2]) *
                                W);
        }
  }
  __syncthreads();
  /// arena no, channel no, cell no
  for (int base = threadIdx.x; base < numMViInArena; base += blockDim.x) {
    char local_block_id = base / numMViPerBlock;
    auto blockno = partition.query(
        ivec3{blockid[0] + ((local_block_id & 4) != 0 ? 1 : 0),
              blockid[1] + ((local_block_id & 2) != 0 ? 1 : 0),
              blockid[2] + ((local_block_id & 1) != 0 ? 1 : 0)});
    // auto grid_block = next_grid.template ch<0>(blockno);
    int channelid = base & (numMViPerBlock - 1);
    char c = channelid % g_blockvolume;
    char cz = channelid & g_blockmask;
    char cy = (channelid >>= g_blockbits) & g_blockmask;
    char cx = (channelid >>= g_blockbits) & g_blockmask;
    channelid >>= g_blockbits;

    float val =
        p2gbuffer[channelid][cx + (local_block_id & 4 ? g_blocksize : 0)]
                 [cy + (local_block_id & 2 ? g_blocksize : 0)]
                 [cz + (local_block_id & 1 ? g_blocksize : 0)];
    if (channelid == 0)
      atomicAdd(&next_grid.ch(_0, blockno).val_1d(_0, c), val);
    else if (channelid == 1)
      atomicAdd(&next_grid.ch(_0, blockno).val_1d(_1, c), val);
    else if (channelid == 2)
      atomicAdd(&next_grid.ch(_0, blockno).val_1d(_2, c), val);
    else
      atomicAdd(&next_grid.ch(_0, blockno).val_1d(_3, c), val);
  }
}

template <typename Partition, typename Grid>
__global__ void g2p2g(float dt, float newDt, float curTime,
                      const ParticleBuffer<material_e::NACC> pbuffer,
                      ParticleBuffer<material_e::NACC> next_pbuffer,
                      const Partition prev_partition, Partition partition,
                      const Grid grid, Grid next_grid) {
  static constexpr uint64_t numViPerBlock = g_blockvolume * 3;
  static constexpr uint64_t numViInArena = numViPerBlock << 3;

  static constexpr uint64_t numMViPerBlock = g_blockvolume * 4;
  static constexpr uint64_t numMViInArena = numMViPerBlock << 3;

  static constexpr unsigned arenamask = (g_blocksize << 1) - 1;
  static constexpr unsigned arenabits = g_blockbits + 1;

  extern __shared__ char shmem[];
  using ViArena =
      float(*)[3][g_blocksize << 1][g_blocksize << 1][g_blocksize << 1];
  using ViArenaRef =
      float(&)[3][g_blocksize << 1][g_blocksize << 1][g_blocksize << 1];
  ViArenaRef __restrict__ g2pbuffer = *reinterpret_cast<ViArena>(shmem);
  using MViArena =
      float(*)[4][g_blocksize << 1][g_blocksize << 1][g_blocksize << 1];
  using MViArenaRef =
      float(&)[4][g_blocksize << 1][g_blocksize << 1][g_blocksize << 1];
  MViArenaRef __restrict__ p2gbuffer =
      *reinterpret_cast<MViArena>(shmem + numViInArena * sizeof(float));

  int src_blockno = blockIdx.x;
  auto blockid = partition._activeKeys[blockIdx.x];

  for (int base = threadIdx.x; base < numViInArena; base += blockDim.x) {
    char local_block_id = base / numViPerBlock;
    auto blockno = partition.query(
        ivec3{blockid[0] + ((local_block_id & 4) != 0 ? 1 : 0),
              blockid[1] + ((local_block_id & 2) != 0 ? 1 : 0),
              blockid[2] + ((local_block_id & 1) != 0 ? 1 : 0)});
    auto grid_block = grid.ch(_0, blockno);
    int channelid = base % numViPerBlock;
    char c = channelid & 0x3f;
    char cz = channelid & g_blockmask;
    char cy = (channelid >>= g_blockbits) & g_blockmask;
    char cx = (channelid >>= g_blockbits) & g_blockmask;
    channelid >>= g_blockbits;

    float val;
    if (channelid == 0)
      val = grid_block.val_1d(_1, c);
    else if (channelid == 1)
      val = grid_block.val_1d(_2, c);
    else
      val = grid_block.val_1d(_3, c);
    g2pbuffer[channelid][cx + (local_block_id & 4 ? g_blocksize : 0)]
             [cy + (local_block_id & 2 ? g_blocksize : 0)]
             [cz + (local_block_id & 1 ? g_blocksize : 0)] = val;
  }
  __syncthreads();
  for (int base = threadIdx.x; base < numMViInArena; base += blockDim.x) {
    int loc = base;
    char z = loc & arenamask;
    char y = (loc >>= arenabits) & arenamask;
    char x = (loc >>= arenabits) & arenamask;
    p2gbuffer[loc >> arenabits][x][y][z] = 0.f;
  }
  __syncthreads();

  for (int pidib = threadIdx.x; pidib < next_pbuffer._ppbs[src_blockno];
       pidib += blockDim.x) {
    int source_blockno, source_pidib;
    ivec3 base_index;
    {
      int advect =
          next_pbuffer
              ._blockbuckets[src_blockno * g_particle_num_per_block + pidib];
      dir_components(advect / g_particle_num_per_block, base_index);
      base_index += blockid;
      source_blockno = prev_partition.query(base_index);
      source_pidib = advect & (g_particle_num_per_block - 1);
      source_blockno =
          pbuffer._binsts[source_blockno] + source_pidib / g_bin_capacity;
    }
    vec3 pos;
    {
      auto source_particle_bin = pbuffer.ch(_0, source_blockno);
      pos[0] = source_particle_bin.val(_0, source_pidib % g_bin_capacity);
      pos[1] = source_particle_bin.val(_1, source_pidib % g_bin_capacity);
      pos[2] = source_particle_bin.val(_2, source_pidib % g_bin_capacity);
    }
    ivec3 local_base_index = (pos * g_dx_inv + 0.5f).cast<int>() - 1;
    vec3 local_pos = pos - local_base_index * g_dx;
    base_index = local_base_index;

    vec3x3 dws;
#pragma unroll 3
    for (int dd = 0; dd < 3; ++dd) {
      float d =
          (local_pos[dd] - ((int)(local_pos[dd] * g_dx_inv + 0.5) - 1) * g_dx) *
          g_dx_inv;
      dws(dd, 0) = 0.5f * (1.5 - d) * (1.5 - d);
      d -= 1.0f;
      dws(dd, 1) = 0.75 - d * d;
      d = 0.5f + d;
      dws(dd, 2) = 0.5 * d * d;
      local_base_index[dd] = ((local_base_index[dd] - 1) & g_blockmask) + 1;
    }
    vec3 vel;
    vel.set(0.f);
    vec9 C;
    C.set(0.f);
#pragma unroll 3
    for (char i = 0; i < 3; i++)
#pragma unroll 3
      for (char j = 0; j < 3; j++)
#pragma unroll 3
        for (char k = 0; k < 3; k++) {
          vec3 xixp = vec3{(float)i, (float)j, (float)k} * g_dx - local_pos;
          float W = dws(0, i) * dws(1, j) * dws(2, k);
          vec3 vi{g2pbuffer[0][local_base_index[0] + i][local_base_index[1] + j]
                           [local_base_index[2] + k],
                  g2pbuffer[1][local_base_index[0] + i][local_base_index[1] + j]
                           [local_base_index[2] + k],
                  g2pbuffer[2][local_base_index[0] + i][local_base_index[1] + j]
                           [local_base_index[2] + k]};
          vel += vi * W;
          C[0] += W * vi[0] * xixp[0];
          C[1] += W * vi[1] * xixp[0];
          C[2] += W * vi[2] * xixp[0];
          C[3] += W * vi[0] * xixp[1];
          C[4] += W * vi[1] * xixp[1];
          C[5] += W * vi[2] * xixp[1];
          C[6] += W * vi[0] * xixp[2];
          C[7] += W * vi[1] * xixp[2];
          C[8] += W * vi[2] * xixp[2];
        }
    pos += vel * dt;

#pragma unroll 9
    for (int d = 0; d < 9; ++d)
      dws.val(d) = C[d] * dt * g_D_inv + ((d & 0x3) ? 0.f : 1.f);

    vec9 contrib;
    {
      vec9 F;
      float logJp;
      auto source_particle_bin = pbuffer.ch(_0, source_blockno);
      contrib[0] = source_particle_bin.val(_3, source_pidib % g_bin_capacity);
      contrib[1] = source_particle_bin.val(_4, source_pidib % g_bin_capacity);
      contrib[2] = source_particle_bin.val(_5, source_pidib % g_bin_capacity);
      contrib[3] = source_particle_bin.val(_6, source_pidib % g_bin_capacity);
      contrib[4] = source_particle_bin.val(_7, source_pidib % g_bin_capacity);
      contrib[5] = source_particle_bin.val(_8, source_pidib % g_bin_capacity);
      contrib[6] = source_particle_bin.val(_9, source_pidib % g_bin_capacity);
      contrib[7] = source_particle_bin.val(_10, source_pidib % g_bin_capacity);
      contrib[8] = source_particle_bin.val(_11, source_pidib % g_bin_capacity);
      logJp = source_particle_bin.val(_12, source_pidib % g_bin_capacity);

      matrixMatrixMultiplication3d(dws.data(), contrib.data(), F.data());
      compute_stress_nacc(pbuffer.volume, pbuffer.mu, pbuffer.lambda,
                          pbuffer.bm, pbuffer.xi, pbuffer.beta, pbuffer.Msqr,
                          pbuffer.hardeningOn, logJp, F, contrib);
      {
        auto particle_bin = next_pbuffer.ch(
            _0, next_pbuffer._binsts[src_blockno] + pidib / g_bin_capacity);
        particle_bin.val(_0, pidib % g_bin_capacity) = pos[0];
        particle_bin.val(_1, pidib % g_bin_capacity) = pos[1];
        particle_bin.val(_2, pidib % g_bin_capacity) = pos[2];
        particle_bin.val(_3, pidib % g_bin_capacity) = F[0];
        particle_bin.val(_4, pidib % g_bin_capacity) = F[1];
        particle_bin.val(_5, pidib % g_bin_capacity) = F[2];
        particle_bin.val(_6, pidib % g_bin_capacity) = F[3];
        particle_bin.val(_7, pidib % g_bin_capacity) = F[4];
        particle_bin.val(_8, pidib % g_bin_capacity) = F[5];
        particle_bin.val(_9, pidib % g_bin_capacity) = F[6];
        particle_bin.val(_10, pidib % g_bin_capacity) = F[7];
        particle_bin.val(_11, pidib % g_bin_capacity) = F[8];
        particle_bin.val(_12, pidib % g_bin_capacity) = logJp;
      }

      contrib = (C * pbuffer.mass - contrib * newDt) * g_D_inv;
    }

    local_base_index = (pos * g_dx_inv + 0.5f).cast<int>() - 1;
    {
      int dirtag = dir_offset((base_index - 1) / g_blocksize -
                              (local_base_index - 1) / g_blocksize);
      next_pbuffer.add_advection(partition, local_base_index - 1, dirtag,
                                 pidib);
      // partition.add_advection(local_base_index - 1, dirtag, pidib);
    }
    // dws[d] = bspline_weight(local_pos[d]);

#pragma unroll 3
    for (char dd = 0; dd < 3; ++dd) {
      local_pos[dd] = pos[dd] - local_base_index[dd] * g_dx;
      float d =
          (local_pos[dd] - ((int)(local_pos[dd] * g_dx_inv + 0.5) - 1) * g_dx) *
          g_dx_inv;
      dws(dd, 0) = 0.5f * (1.5 - d) * (1.5 - d);
      d -= 1.0f;
      dws(dd, 1) = 0.75 - d * d;
      d = 0.5f + d;
      dws(dd, 2) = 0.5 * d * d;

      local_base_index[dd] = (((base_index[dd] - 1) & g_blockmask) + 1) +
                             local_base_index[dd] - base_index[dd];
    }
#pragma unroll 3
    for (char i = 0; i < 3; i++)
#pragma unroll 3
      for (char j = 0; j < 3; j++)
#pragma unroll 3
        for (char k = 0; k < 3; k++) {
          pos = vec3{(float)i, (float)j, (float)k} * g_dx - local_pos;
          float W = dws(0, i) * dws(1, j) * dws(2, k);
          auto wm = pbuffer.mass * W;
          atomicAdd(
              &p2gbuffer[0][local_base_index[0] + i][local_base_index[1] + j]
                        [local_base_index[2] + k],
              wm);
          atomicAdd(
              &p2gbuffer[1][local_base_index[0] + i][local_base_index[1] + j]
                        [local_base_index[2] + k],
              wm * vel[0] + (contrib[0] * pos[0] + contrib[3] * pos[1] +
                             contrib[6] * pos[2]) *
                                W);
          atomicAdd(
              &p2gbuffer[2][local_base_index[0] + i][local_base_index[1] + j]
                        [local_base_index[2] + k],
              wm * vel[1] + (contrib[1] * pos[0] + contrib[4] * pos[1] +
                             contrib[7] * pos[2]) *
                                W);
          atomicAdd(
              &p2gbuffer[3][local_base_index[0] + i][local_base_index[1] + j]
                        [local_base_index[2] + k],
              wm * vel[2] + (contrib[2] * pos[0] + contrib[5] * pos[1] +
                             contrib[8] * pos[2]) *
                                W);
        }
  }
  __syncthreads();
  /// arena no, channel no, cell no
  for (int base = threadIdx.x; base < numMViInArena; base += blockDim.x) {
    char local_block_id = base / numMViPerBlock;
    auto blockno = partition.query(
        ivec3{blockid[0] + ((local_block_id & 4) != 0 ? 1 : 0),
              blockid[1] + ((local_block_id & 2) != 0 ? 1 : 0),
              blockid[2] + ((local_block_id & 1) != 0 ? 1 : 0)});
    // auto grid_block = next_grid.template ch<0>(blockno);
    int channelid = base & (numMViPerBlock - 1);
    char c = channelid % g_blockvolume;
    char cz = channelid & g_blockmask;
    char cy = (channelid >>= g_blockbits) & g_blockmask;
    char cx = (channelid >>= g_blockbits) & g_blockmask;
    channelid >>= g_blockbits;

    float val =
        p2gbuffer[channelid][cx + (local_block_id & 4 ? g_blocksize : 0)]
                 [cy + (local_block_id & 2 ? g_blocksize : 0)]
                 [cz + (local_block_id & 1 ? g_blocksize : 0)];
    if (channelid == 0)
      atomicAdd(&next_grid.ch(_0, blockno).val_1d(_0, c), val);
    else if (channelid == 1)
      atomicAdd(&next_grid.ch(_0, blockno).val_1d(_1, c), val);
    else if (channelid == 2)
      atomicAdd(&next_grid.ch(_0, blockno).val_1d(_2, c), val);
    else
      atomicAdd(&next_grid.ch(_0, blockno).val_1d(_3, c), val);
  }
}

template <typename Partition, typename Grid>
__global__ void g2p2g(float dt, float newDt, float curTime,
                      const ParticleBuffer<material_e::Rigid> pbuffer,
                      ParticleBuffer<material_e::Rigid> next_pbuffer,
                      const Partition prev_partition, Partition partition,
                      const Grid grid, Grid next_grid) {
  static constexpr uint64_t numViPerBlock = g_blockvolume * 3;
  static constexpr uint64_t numViInArena = numViPerBlock << 3;

  static constexpr uint64_t numMViPerBlock = g_blockvolume * 4;
  static constexpr uint64_t numMViInArena = numMViPerBlock << 3;

  static constexpr unsigned arenamask = (g_blocksize << 1) - 1;
  static constexpr unsigned arenabits = g_blockbits + 1;

  extern __shared__ char shmem[];
  using ViArena =
      float(*)[3][g_blocksize << 1][g_blocksize << 1][g_blocksize << 1];
  using ViArenaRef =
      float(&)[3][g_blocksize << 1][g_blocksize << 1][g_blocksize << 1];
  ViArenaRef __restrict__ g2pbuffer = *reinterpret_cast<ViArena>(shmem);
  using MViArena =
      float(*)[4][g_blocksize << 1][g_blocksize << 1][g_blocksize << 1];
  using MViArenaRef =
      float(&)[4][g_blocksize << 1][g_blocksize << 1][g_blocksize << 1];
  MViArenaRef __restrict__ p2gbuffer =
      *reinterpret_cast<MViArena>(shmem + numViInArena * sizeof(float));

  int src_blockno = blockIdx.x;
  int ppb = next_pbuffer._ppbs[src_blockno];
  if (ppb == 0)
    return;
  auto blockid = partition._activeKeys[blockIdx.x];

  for (int base = threadIdx.x; base < numViInArena; base += blockDim.x) {
    char local_block_id = base / numViPerBlock;
    auto blockno = partition.query(
        ivec3{blockid[0] + ((local_block_id & 4) != 0 ? 1 : 0),
              blockid[1] + ((local_block_id & 2) != 0 ? 1 : 0),
              blockid[2] + ((local_block_id & 1) != 0 ? 1 : 0)});
    auto grid_block = grid.ch(_0, blockno);
    int channelid = base % numViPerBlock;
    char c = channelid & 0x3f;
    char cz = channelid & g_blockmask;
    char cy = (channelid >>= g_blockbits) & g_blockmask;
    char cx = (channelid >>= g_blockbits) & g_blockmask;
    channelid >>= g_blockbits;

    float val;
    if (channelid == 0)
      val = grid_block.val_1d(_1, c);
    else if (channelid == 1)
      val = grid_block.val_1d(_2, c);
    else
      val = grid_block.val_1d(_3, c);
    g2pbuffer[channelid][cx + (local_block_id & 4 ? g_blocksize : 0)]
             [cy + (local_block_id & 2 ? g_blocksize : 0)]
             [cz + (local_block_id & 1 ? g_blocksize : 0)] = val;
  }
  __syncthreads();
  for (int base = threadIdx.x; base < numMViInArena; base += blockDim.x) {
    int loc = base;
    char z = loc & arenamask;
    char y = (loc >>= arenabits) & arenamask;
    char x = (loc >>= arenabits) & arenamask;
    p2gbuffer[loc >> arenabits][x][y][z] = 0.f;
  }
  __syncthreads();

  for (int pidib = threadIdx.x; pidib < ppb; pidib += blockDim.x) {
    int source_blockno, source_pidib;
    ivec3 base_index;
    {
      int advect =
          next_pbuffer
              ._blockbuckets[src_blockno * g_particle_num_per_block + pidib];
      dir_components(advect / g_particle_num_per_block, base_index);
      base_index += blockid;
      source_blockno = prev_partition.query(base_index);
      source_pidib = advect & (g_particle_num_per_block - 1);
      source_blockno =
          pbuffer._binsts[source_blockno] + source_pidib / g_bin_capacity;
    }
    vec3 pos;
    vec3 static_pos;
    {
      auto source_particle_bin = pbuffer.ch(_0, source_blockno);
      pos[0] = source_particle_bin.val(_0, source_pidib % g_bin_capacity);
      pos[1] = source_particle_bin.val(_1, source_pidib % g_bin_capacity);
      pos[2] = source_particle_bin.val(_2, source_pidib % g_bin_capacity);
      static_pos[0] = pos[0];
      static_pos[1] = pos[1];
      static_pos[2] = pos[2];
    }
    ivec3 local_base_index = (pos * g_dx_inv + 0.5f).cast<int>() - 1;
    vec3 local_pos = pos - local_base_index * g_dx;
    base_index = local_base_index;

    vec3x3 dws;
#pragma unroll 3
    for (int dd = 0; dd < 3; ++dd) {
      float d =
          (local_pos[dd] - ((int)(local_pos[dd] * g_dx_inv + 0.5) - 1) * g_dx) *
          g_dx_inv;
      dws(dd, 0) = 0.5f * (1.5 - d) * (1.5 - d);
      d -= 1.0f;
      dws(dd, 1) = 0.75 - d * d;
      d = 0.5f + d;
      dws(dd, 2) = 0.5 * d * d;
      local_base_index[dd] = ((local_base_index[dd] - 1) & g_blockmask) + 1;
    }
    vec3 vel;
    vel.set(0.f);
    vec3 accel;
    accel.set(0.f);
    vec9 C;
    C.set(0.f);
#pragma unroll 3
    for (char i = 0; i < 3; i++)
#pragma unroll 3
      for (char j = 0; j < 3; j++)
#pragma unroll 3
        for (char k = 0; k < 3; k++) {
          vec3 xixp = vec3{(float)i, (float)j, (float)k} * g_dx - local_pos;
          float W = dws(0, i) * dws(1, j) * dws(2, k);
          vec3 vi{g2pbuffer[0][local_base_index[0] + i][local_base_index[1] + j]
                           [local_base_index[2] + k],
                  g2pbuffer[1][local_base_index[0] + i][local_base_index[1] + j]
                           [local_base_index[2] + k],
                  g2pbuffer[2][local_base_index[0] + i][local_base_index[1] + j]
                           [local_base_index[2] + k]};
          //vel += vi * W;
          C[0] += W * vi[0] * xixp[0];
          C[1] += W * vi[1] * xixp[0];
          C[2] += W * vi[2] * xixp[0];
          C[3] += W * vi[0] * xixp[1];
          C[4] += W * vi[1] * xixp[1];
          C[5] += W * vi[2] * xixp[1];
          C[6] += W * vi[0] * xixp[2];
          C[7] += W * vi[1] * xixp[2];
          C[8] += W * vi[2] * xixp[2];
        }
    
    //accel[0] = 0.1;
    //vel[0] = -1.f/32.f;
    //pos += vel * dt;

// #pragma unroll 9
//     for (int d = 0; d < 9; ++d)
//       // dWip = Bp * Dp^-1 * dt + I
//       // (m^2 / s) * (1 / m^2) * (s) = ( )
//       //dws.val(d) = C[d] * dt * g_D_inv + ((d & 0x3) ? 0.f : 1.f);

    vec9 contrib;
    contrib.set(0.f);
    {
      // vec9 F; //< Deformation Gradient, Fp^n+1
      auto source_particle_bin = pbuffer.ch(_0, source_blockno);
      // Set Deformation Gradient, Fp^n
      // contrib[0] = source_particle_bin.val(_3, source_pidib % g_bin_capacity);
      // contrib[1] = source_particle_bin.val(_4, source_pidib % g_bin_capacity);
      // contrib[2] = source_particle_bin.val(_5, source_pidib % g_bin_capacity);
      // contrib[3] = source_particle_bin.val(_6, source_pidib % g_bin_capacity);
      // contrib[4] = source_particle_bin.val(_7, source_pidib % g_bin_capacity);
      // contrib[5] = source_particle_bin.val(_8, source_pidib % g_bin_capacity);
      // contrib[6] = source_particle_bin.val(_9, source_pidib % g_bin_capacity);
      // contrib[7] = source_particle_bin.val(_10, source_pidib % g_bin_capacity);
      // contrib[8] = source_particle_bin.val(_11, source_pidib % g_bin_capacity);
      // Set F = Fp^n+1 using dWip^n+1 and Fp^n
      //matrixMatrixMultiplication3d(dws.data(), contrib.data(), F.data());
      {
        auto particle_bin = next_pbuffer.ch(
            _0, next_pbuffer._binsts[src_blockno] + pidib / g_bin_capacity);
        particle_bin.val(_0,  pidib % g_bin_capacity) = pos[0]; //< x (m)
        particle_bin.val(_1,  pidib % g_bin_capacity) = pos[1]; //< y (m)
        particle_bin.val(_2,  pidib % g_bin_capacity) = pos[2]; //< z (m)
        particle_bin.val(_3,  pidib % g_bin_capacity) = 1.f;
      }
      // Torque matrix (N * m)
      // Tp = Stress * Volume
      // compute_stress_fixedcorotated(pbuffer.volume, pbuffer.mu, pbuffer.lambda,
      //                               F, contrib);
      // Mass-flow rate (kg / s)
      // mp * Cp^n+1 = (Bp^n * mp - Tp * dt) * Dp^n+1
      //contrib = (C * pbuffer.mass - contrib * newDt) * g_D_inv;
      contrib = (C * pbuffer.mass) * g_D_inv;
    }

    local_base_index = (pos * g_dx_inv + 0.5f).cast<int>() - 1;
    {
      int dirtag = dir_offset((base_index - 1) / g_blocksize -
                              (local_base_index - 1) / g_blocksize);
      next_pbuffer.add_advection(partition, local_base_index - 1, dirtag,
                                 pidib);
      // partition.add_advection(local_base_index - 1, dirtag, pidib);
    }
    // dws[d] = bspline_weight(local_pos[d]);

#pragma unroll 3
    for (char dd = 0; dd < 3; ++dd) {
      local_pos[dd] = pos[dd] - local_base_index[dd] * g_dx;
      float d =
          (local_pos[dd] - ((int)(local_pos[dd] * g_dx_inv + 0.5) - 1) * g_dx) *
          g_dx_inv;
      dws(dd, 0) = 0.5f * (1.5 - d) * (1.5 - d);
      d -= 1.0f;
      dws(dd, 1) = 0.75 - d * d;
      d = 0.5f + d;
      dws(dd, 2) = 0.5 * d * d;

      local_base_index[dd] = (((base_index[dd] - 1) & g_blockmask) + 1) +
                             local_base_index[dd] - base_index[dd];
    }
#pragma unroll 3
    for (char i = 0; i < 3; i++)
#pragma unroll 3
      for (char j = 0; j < 3; j++)
#pragma unroll 3
        for (char k = 0; k < 3; k++) {
          pos = vec3{(float)i, (float)j, (float)k} * g_dx - local_pos;
          float W = dws(0, i) * dws(1, j) * dws(2, k);
          auto wm = pbuffer.mass * W;
          atomicAdd(
              &p2gbuffer[0][local_base_index[0] + i][local_base_index[1] + j]
                        [local_base_index[2] + k],
              wm);
          atomicAdd(
              &p2gbuffer[1][local_base_index[0] + i][local_base_index[1] + j]
                        [local_base_index[2] + k],
              wm * vel[0] + (contrib[0] * pos[0] + contrib[3] * pos[1] +
                             contrib[6] * pos[2]) *
                                W);
          atomicAdd(
              &p2gbuffer[2][local_base_index[0] + i][local_base_index[1] + j]
                        [local_base_index[2] + k],
              wm * vel[1] + (contrib[1] * pos[0] + contrib[4] * pos[1] +
                             contrib[7] * pos[2]) *
                                W);
          atomicAdd(
              &p2gbuffer[3][local_base_index[0] + i][local_base_index[1] + j]
                        [local_base_index[2] + k],
              wm * vel[2] + (contrib[2] * pos[0] + contrib[5] * pos[1] +
                             contrib[8] * pos[2]) *
                                W);
        }
  }
  __syncthreads();
  /// arena no, channel no, cell no
  for (int base = threadIdx.x; base < numMViInArena; base += blockDim.x) {
    char local_block_id = base / numMViPerBlock;
    auto blockno = partition.query(
        ivec3{blockid[0] + ((local_block_id & 4) != 0 ? 1 : 0),
              blockid[1] + ((local_block_id & 2) != 0 ? 1 : 0),
              blockid[2] + ((local_block_id & 1) != 0 ? 1 : 0)});
    // auto grid_block = next_grid.template ch<0>(blockno);
    int channelid = base & (numMViPerBlock - 1);
    char c = channelid % g_blockvolume;
    char cz = channelid & g_blockmask;
    char cy = (channelid >>= g_blockbits) & g_blockmask;
    char cx = (channelid >>= g_blockbits) & g_blockmask;
    channelid >>= g_blockbits;

    float val =
        p2gbuffer[channelid][cx + (local_block_id & 4 ? g_blocksize : 0)]
                 [cy + (local_block_id & 2 ? g_blocksize : 0)]
                 [cz + (local_block_id & 1 ? g_blocksize : 0)];
    if (channelid == 0)
      atomicAdd(&next_grid.ch(_0, blockno).val_1d(_0, c), val);
    else if (channelid == 1)
      atomicAdd(&next_grid.ch(_0, blockno).val_1d(_1, c), val);
    else if (channelid == 2)
      atomicAdd(&next_grid.ch(_0, blockno).val_1d(_2, c), val);
    else
      atomicAdd(&next_grid.ch(_0, blockno).val_1d(_3, c), val);
  }
}

template <typename Partition, typename Grid>
__global__ void g2p2g(float dt, float newDt, float curTime,
                      const ParticleBuffer<material_e::Piston> pbuffer,
                      ParticleBuffer<material_e::Piston> next_pbuffer,
                      const Partition prev_partition, Partition partition,
                      const Grid grid, Grid next_grid) {
  static constexpr uint64_t numViPerBlock = g_blockvolume * 3;
  static constexpr uint64_t numViInArena = numViPerBlock << 3;

  static constexpr uint64_t numMViPerBlock = g_blockvolume * 4;
  static constexpr uint64_t numMViInArena = numMViPerBlock << 3;

  static constexpr unsigned arenamask = (g_blocksize << 1) - 1;
  static constexpr unsigned arenabits = g_blockbits + 1;

  extern __shared__ char shmem[];
  using ViArena =
      float(*)[3][g_blocksize << 1][g_blocksize << 1][g_blocksize << 1];
  using ViArenaRef =
      float(&)[3][g_blocksize << 1][g_blocksize << 1][g_blocksize << 1];
  ViArenaRef __restrict__ g2pbuffer = *reinterpret_cast<ViArena>(shmem);
  using MViArena =
      float(*)[4][g_blocksize << 1][g_blocksize << 1][g_blocksize << 1];
  using MViArenaRef =
      float(&)[4][g_blocksize << 1][g_blocksize << 1][g_blocksize << 1];
  MViArenaRef __restrict__ p2gbuffer =
      *reinterpret_cast<MViArena>(shmem + numViInArena * sizeof(float));

  int src_blockno = blockIdx.x;
  int ppb = next_pbuffer._ppbs[src_blockno];
  if (ppb == 0)
    return;
  auto blockid = partition._activeKeys[blockIdx.x];

  for (int base = threadIdx.x; base < numViInArena; base += blockDim.x) {
    char local_block_id = base / numViPerBlock;
    auto blockno = partition.query(
        ivec3{blockid[0] + ((local_block_id & 4) != 0 ? 1 : 0),
              blockid[1] + ((local_block_id & 2) != 0 ? 1 : 0),
              blockid[2] + ((local_block_id & 1) != 0 ? 1 : 0)});
    auto grid_block = grid.ch(_0, blockno);
    int channelid = base % numViPerBlock;
    char c = channelid & 0x3f;
    char cz = channelid & g_blockmask;
    char cy = (channelid >>= g_blockbits) & g_blockmask;
    char cx = (channelid >>= g_blockbits) & g_blockmask;
    channelid >>= g_blockbits;

    float val;
    if (channelid == 0)
      val = grid_block.val_1d(_1, c);
    else if (channelid == 1)
      val = grid_block.val_1d(_2, c);
    else
      val = grid_block.val_1d(_3, c);
    g2pbuffer[channelid][cx + (local_block_id & 4 ? g_blocksize : 0)]
             [cy + (local_block_id & 2 ? g_blocksize : 0)]
             [cz + (local_block_id & 1 ? g_blocksize : 0)] = val;
  }
  __syncthreads();
  for (int base = threadIdx.x; base < numMViInArena; base += blockDim.x) {
    int loc = base;
    char z = loc & arenamask;
    char y = (loc >>= arenabits) & arenamask;
    char x = (loc >>= arenabits) & arenamask;
    p2gbuffer[loc >> arenabits][x][y][z] = 0.f;
  }
  __syncthreads();

  for (int pidib = threadIdx.x; pidib < ppb; pidib += blockDim.x) {
    int source_blockno, source_pidib;
    ivec3 base_index;
    {
      int advect =
          next_pbuffer
              ._blockbuckets[src_blockno * g_particle_num_per_block + pidib];
      dir_components(advect / g_particle_num_per_block, base_index);
      base_index += blockid;
      source_blockno = prev_partition.query(base_index);
      source_pidib = advect & (g_particle_num_per_block - 1);
      source_blockno =
          pbuffer._binsts[source_blockno] + source_pidib / g_bin_capacity;
    }
    vec3 pos;
    float length = mn::config::g_length;
    {
      auto source_particle_bin = pbuffer.ch(_0, source_blockno);
      pos[0] = source_particle_bin.val(_0, source_pidib % g_bin_capacity);
      pos[1] = source_particle_bin.val(_1, source_pidib % g_bin_capacity);
      pos[2] = source_particle_bin.val(_2, source_pidib % g_bin_capacity);
    }
    ivec3 local_base_index = (pos * g_dx_inv + 0.5f).cast<int>() - 1;
    vec3 local_pos = pos - local_base_index * g_dx;
    base_index = local_base_index;

    vec3x3 dws;
#pragma unroll 3
    for (int dd = 0; dd < 3; ++dd) {
      float d =
          (local_pos[dd] - ((int)(local_pos[dd] * g_dx_inv + 0.5) - 1) * g_dx) *
          g_dx_inv;
      dws(dd, 0) = 0.5f * (1.5 - d) * (1.5 - d);
      d -= 1.0f;
      dws(dd, 1) = 0.75 - d * d;
      d = 0.5f + d;
      dws(dd, 2) = 0.5 * d * d;
      local_base_index[dd] = ((local_base_index[dd] - 1) & g_blockmask) + 1;
    }
    vec3 vel;
    vel.set(0.f);
    // vec3 accel;
    // accel.set(0.f);
    vec9 C;
    C.set(0.f);
#pragma unroll 3
    for (char i = 0; i < 3; i++)
#pragma unroll 3
      for (char j = 0; j < 3; j++)
#pragma unroll 3
        for (char k = 0; k < 3; k++) {
          vec3 xixp = vec3{(float)i, (float)j, (float)k} * g_dx - local_pos;
          float W = dws(0, i) * dws(1, j) * dws(2, k);
          vec3 vi{g2pbuffer[0][local_base_index[0] + i][local_base_index[1] + j]
                           [local_base_index[2] + k],
                  g2pbuffer[1][local_base_index[0] + i][local_base_index[1] + j]
                           [local_base_index[2] + k],
                  g2pbuffer[2][local_base_index[0] + i][local_base_index[1] + j]
                           [local_base_index[2] + k]};
          //vel += vi * W;
          C[0] += W * vi[0] * xixp[0];
          C[1] += W * vi[1] * xixp[0];
          C[2] += W * vi[2] * xixp[0];
          C[3] += W * vi[0] * xixp[1];
          C[4] += W * vi[1] * xixp[1];
          C[5] += W * vi[2] * xixp[1];
          C[6] += W * vi[0] * xixp[2];
          C[7] += W * vi[1] * xixp[2];
          C[8] += W * vi[2] * xixp[2];
        }
    
    //accel[0] = 0.1;
    if (curTime < 2.f || curTime > 4.f){
      vel[0] = 0.f;
    }
    else {
      vel[0] = 2.f * (1.f/length);
    }
    pos += vel * dt;

#pragma unroll 9
    for (int d = 0; d < 9; ++d)
      // dWip = Bp * Dp^-1 * dt + I
      // (m^2 / s) * (1 / m^2) * (s) = ( )
      dws.val(d) = C[d] * dt * g_D_inv + ((d & 0x3) ? 0.f : 1.f);

    vec9 contrib;
    {
      // vec9 F; //< Deformation Gradient, Fp^n+1
      auto source_particle_bin = pbuffer.ch(_0, source_blockno);
      // Set Deformation Gradient, Fp^n
      // contrib[0] = source_particle_bin.val(_3, source_pidib % g_bin_capacity);
      // contrib[1] = source_particle_bin.val(_4, source_pidib % g_bin_capacity);
      // contrib[2] = source_particle_bin.val(_5, source_pidib % g_bin_capacity);
      // contrib[3] = source_particle_bin.val(_6, source_pidib % g_bin_capacity);
      // contrib[4] = source_particle_bin.val(_7, source_pidib % g_bin_capacity);
      // contrib[5] = source_particle_bin.val(_8, source_pidib % g_bin_capacity);
      // contrib[6] = source_particle_bin.val(_9, source_pidib % g_bin_capacity);
      // contrib[7] = source_particle_bin.val(_10, source_pidib % g_bin_capacity);
      // contrib[8] = source_particle_bin.val(_11, source_pidib % g_bin_capacity);
      // // Set F = Fp^n+1 using dWip^n+1 and Fp^n
      // matrixMatrixMultiplication3d(dws.data(), contrib.data(), F.data());
      {
        auto particle_bin = next_pbuffer.ch(
            _0, next_pbuffer._binsts[src_blockno] + pidib / g_bin_capacity);
        particle_bin.val(_0,  pidib % g_bin_capacity) = pos[0]; //< x (m)
        particle_bin.val(_1,  pidib % g_bin_capacity) = pos[1]; //< y (m)
        particle_bin.val(_2,  pidib % g_bin_capacity) = pos[2]; //< z (m)
      //   particle_bin.val(_3,  pidib % g_bin_capacity) = F[0];
      //   particle_bin.val(_4,  pidib % g_bin_capacity) = F[1];
      //   particle_bin.val(_5,  pidib % g_bin_capacity) = F[2];
      //   particle_bin.val(_6,  pidib % g_bin_capacity) = F[3];
      //   particle_bin.val(_7,  pidib % g_bin_capacity) = F[4];
      //   particle_bin.val(_8,  pidib % g_bin_capacity) = F[5];
      //   particle_bin.val(_9,  pidib % g_bin_capacity) = F[6];
      //   particle_bin.val(_10, pidib % g_bin_capacity) = F[7];
      //   particle_bin.val(_11, pidib % g_bin_capacity) = F[8];
      }
      // Torque matrix (N * m)
      // Tp = Stress * Volume
      // compute_stress_fixedcorotated(pbuffer.volume, pbuffer.mu, pbuffer.lambda,
      //                               F, contrib);
      // Mass-flow rate (kg / s)
      // mp * Cp^n+1 = (Bp^n * mp - Tp * dt) * Dp^n+1
      contrib = (C * pbuffer.mass) * g_D_inv;
    }

    local_base_index = (pos * g_dx_inv + 0.5f).cast<int>() - 1;
    {
      int dirtag = dir_offset((base_index - 1) / g_blocksize -
                              (local_base_index - 1) / g_blocksize);
      next_pbuffer.add_advection(partition, local_base_index - 1, dirtag,
                                 pidib);
      // partition.add_advection(local_base_index - 1, dirtag, pidib);
    }
    // dws[d] = bspline_weight(local_pos[d]);

#pragma unroll 3
    for (char dd = 0; dd < 3; ++dd) {
      local_pos[dd] = pos[dd] - local_base_index[dd] * g_dx;
      float d =
          (local_pos[dd] - ((int)(local_pos[dd] * g_dx_inv + 0.5) - 1) * g_dx) *
          g_dx_inv;
      dws(dd, 0) = 0.5f * (1.5 - d) * (1.5 - d);
      d -= 1.0f;
      dws(dd, 1) = 0.75 - d * d;
      d = 0.5f + d;
      dws(dd, 2) = 0.5 * d * d;

      local_base_index[dd] = (((base_index[dd] - 1) & g_blockmask) + 1) +
                             local_base_index[dd] - base_index[dd];
    }
#pragma unroll 3
    for (char i = 0; i < 3; i++)
#pragma unroll 3
      for (char j = 0; j < 3; j++)
#pragma unroll 3
        for (char k = 0; k < 3; k++) {
          pos = vec3{(float)i, (float)j, (float)k} * g_dx - local_pos;
          float W = dws(0, i) * dws(1, j) * dws(2, k);
          auto wm = pbuffer.mass * W;
          atomicAdd(
              &p2gbuffer[0][local_base_index[0] + i][local_base_index[1] + j]
                        [local_base_index[2] + k],
              wm);
          atomicAdd(
              &p2gbuffer[1][local_base_index[0] + i][local_base_index[1] + j]
                        [local_base_index[2] + k],
              wm * vel[0] + (contrib[0] * pos[0] + contrib[3] * pos[1] +
                             contrib[6] * pos[2]) *
                                W);
          atomicAdd(
              &p2gbuffer[2][local_base_index[0] + i][local_base_index[1] + j]
                        [local_base_index[2] + k],
              wm * vel[1] + (contrib[1] * pos[0] + contrib[4] * pos[1] +
                             contrib[7] * pos[2]) *
                                W);
          atomicAdd(
              &p2gbuffer[3][local_base_index[0] + i][local_base_index[1] + j]
                        [local_base_index[2] + k],
              wm * vel[2] + (contrib[2] * pos[0] + contrib[5] * pos[1] +
                             contrib[8] * pos[2]) *
                                W);
        }
  }
  __syncthreads();
  /// arena no, channel no, cell no
  for (int base = threadIdx.x; base < numMViInArena; base += blockDim.x) {
    char local_block_id = base / numMViPerBlock;
    auto blockno = partition.query(
        ivec3{blockid[0] + ((local_block_id & 4) != 0 ? 1 : 0),
              blockid[1] + ((local_block_id & 2) != 0 ? 1 : 0),
              blockid[2] + ((local_block_id & 1) != 0 ? 1 : 0)});
    // auto grid_block = next_grid.template ch<0>(blockno);
    int channelid = base & (numMViPerBlock - 1);
    char c = channelid % g_blockvolume;
    char cz = channelid & g_blockmask;
    char cy = (channelid >>= g_blockbits) & g_blockmask;
    char cx = (channelid >>= g_blockbits) & g_blockmask;
    channelid >>= g_blockbits;

    float val =
        p2gbuffer[channelid][cx + (local_block_id & 4 ? g_blocksize : 0)]
                 [cy + (local_block_id & 2 ? g_blocksize : 0)]
                 [cz + (local_block_id & 1 ? g_blocksize : 0)];
    if (channelid == 0)
      atomicAdd(&next_grid.ch(_0, blockno).val_1d(_0, c), val);
    else if (channelid == 1)
      atomicAdd(&next_grid.ch(_0, blockno).val_1d(_1, c), val);
    else if (channelid == 2)
      atomicAdd(&next_grid.ch(_0, blockno).val_1d(_2, c), val);
    else
      atomicAdd(&next_grid.ch(_0, blockno).val_1d(_3, c), val);
  }
}

template <typename Partition, typename Grid>
__global__ void g2p2g(float dt, float newDt, float curTime,
                      const ParticleBuffer<material_e::IFluid> pbuffer,
                      ParticleBuffer<material_e::IFluid> next_pbuffer,
                      const Partition prev_partition, Partition partition,
                      const Grid grid, Grid next_grid) {
  //============================================================
  // Grid-to-Particle-to Grid Kernel for IFluid material
  // Transfer Scheme:  Affine Particle-in-Cell
  // Shape-function:   Quadratic B-Spline
  // Time Integration: 
  // Material:         Imcompressible fluid
  //============================================================

  // Imcompressible fluid formulation
  // Pressures isn't tied to strain, like in equations of state
  // Two-step nodal velocity calculation
  // 1) Explicit internal force update
  // 2) Implicit Poisson solve
  // Hour-glass mode correction
  // et al. (2017)
  // https://hub.hku.hk/bitstream/10722/242215/1/Content.pdf?accept=1

  // I indice is grid-node
  // i is general?
  
  // Cauchy stress
  // c_ij = -p,i dirac_ij + s_ij,j + rho b_i
  // rho u_i^.. = -p,i + s_ij,j + rho b_i 
  
  // Divergence free condition
  // Div[v_i] = v_i,i = 0

  // Operator splitting
  // Decouple velocity and pressure fields

  // First step, compute intermediate velocity
  // Ignores pressure gradient term
  // Solve explicitly, dt depends on shear-wave speed
  // v_i^* = v_i^n + dt/rho (s_ij,j + rho b_i)

  // For Newtonian fluid
  // s_ij = 2 u  e^.'
  // visc = Dynamic viscosity
  // e^.' = tensor of deviatoric strain rate

  // Second step, correct velocity by pressure gradient
  // v_i^n+1 = v_i^* - dt/rho p,i^n+1
  
  // Solved explicitly
  // v_iI^* = v_iI^n + dt/m_I (f_iI^ext + f_iI^int,s)
  
  // External grid node force
  // f_iI^ext = Sum_p=1^n_p (N_Ip t_ip^bar h^-1 (m_p/rho_p)) + Sum_p=1^n_p(m_p S_Ip^bar b_ip)
  // h = artifical layer thickness for integral along traction boundary

  // Internal grid node force from deviatoric stress only
  // f_iI^int,s = - Sum_p=1^n_p (S_Ip,j^bar s_ijp (m_p/rho_p))
  
  // Pressure Poisson equations
  // v_i,i^* = dt/rho Laplacian[p^n+1]
  
  // Pressures stored at cell-centers (i,j,k)
  // Find approx. div. of int. velocity at cell-center (i,j,k)
  // Using grid node int. velocity 
  // v_i,i^*(X_i,j,k) = Sum(d[N_I(X_i,j,k)]]/d[x_i]) v_iI^*
  
  // Approximate sec. deriv. of pressure at cell-center
  // Use central difference method, respect to x
  // (d2p/dx2)_i,j,k = (p_i+1,j,k + p_i-1,j,k - 2p_i,j,k)/(Dx^2)
  // Dx = Dy = Dx = h

  // Approx. pressure laplacian at cell-center (i,j,k)
  // Laplacian[p]_i,j,k = 
  // (p_i+1,j,k + p_i-1,j,k + p_i,j+1,k + p_i,j-1,k + p_i,j,k+1 + p_i,j,k-1 - 6p_i,j,k) / h^2
  
  // Form system of linear equations
  // Symmetric positive semi-definite
  // Suited to preconditioned conjugate gradient (PCG) solver
  // A^> p^> = b^>
  // A^> = Coefficient matrixx
  // p^> = vector of pressures at all cell-centers
  // b^> = Vector of neg. div. of inter. velocity at cell-centers

  // Apply two boundary conditions to pressure Poisson equation
  // 1) Free-surface pressure (NOT IMPLEMENTED YET)
  // 2) Normal pressure grad. at solid-fluid interface
  //    Guarantees continutiy of normal velocity

  // Solid boundary condition
  // Assume solid is static
  // Parallel boundary to grid-line, not cell-cutting
  // Grad[p] . n^> = 0
  // n^> = Interface normal, orthonormal unit-basis here

  // Pressure gradient calculation
  // Grad[p] is required for v_i^n+1
  // v_i^n+1 is updated at grid-nodes in iMPM
  // Need to evaulate pressure grad. at grid-node (i+1/2,j+1/2) (2D case)
  // If ALL cells touching grid node are fluid-cells..
  // From pressures at cell-centers (i,j)
  
  // We find pressure grad components at grid-nodes
  // (d[p]/d[x])_i+1/2,j+1/2 = (1/2)[(d[p]/d[x])_i+1/2,j + (d[p]/d[x])_i+1/2,j+1]
  // (d[p]/d[y])_i+1/2,j+1/2 = (1/2)[(d[p]/d[y])_i,j+1/2 + (d[p]/d[y])_i+1,j+1/2]
  
  // Where pressure grad components at face-centers
  // (d[p]/d[x])_i+1/2,j   = (p_i+1,j   - p_i,j)   / Dx
  // (d[p]/d[x])_i+1/2,j+1 = (p_i+1,j+1 - p_i,j+1) / Dx
  // (d[p]/d[y])_i,j+1/2   = (p_i,j+1   - p_i,j)   / Dy
  // (d[p]/d[y])_i+1,j+1/2 = (p_i+1,j+1 - p_i+1,j) / Dy
  
  // Generalized at grid-nodes, accounting for fluid/solid/air adjacent to face
  // (d[p]/d[x])_i+1/2,j+1/2 = w_i+1/2,j (d[p]/d[x])_i+1/2,j + w_i+1/2,j+1 (d[p]/d[x])_i+1/2,j+1
  // (d[p]/d[y])_i+1/2,j+1/2 = w_i,j+1/2 (d[p]/d[y])_i,j+1/2 + w_i+1,j+1/2 (d[p]/d[y])_i+1,j+1/2]
  // w_i+1/2,j = B_i+1/2,j / (B_i+1/2,j + B_i+1/2,j+1)
  // w_i+1/2,j+1 = B_i+1/2,j+1 / (B_i+1/2,j + B_i+1/2,j+1)
  // Generalize to 3D.....

  // Spurious hourglass velocity modes
  // From Div[v] = 0 at centers of octahedral elements 
  // 12 modes total, expressed with four modes per direction
  // v^>_i = v_i - a_h Sum_k=1^4(T_k^tran v_i T_k)
  // v^>_i is velocity vector of velocity v_i of grid-nodes
  // (m,n,p), (m+1,n,p), (m,n+1,p), (m+1,n+1,p), 
  // (m,n,p+1), (m+1,n,p+1), (m,n+1,p+1), (m+1,n+1,p+1)
  // a_h is damping coefficient, often 0.05

  // Particle velocity update
  // v_ip^n+1 = coef Sum_I=1^n_g(v_iI^n+1 S_pI^bar) + (1 - coef)(v_ip^n + Dv_ip)
  // coef is taken as 0 <= coef <= 1, 0.03 used here
  // Dv_ip = Sum_I=1^n_g(v_iI^n+1 - v_iI^n) S_pI^bar


  // 1. Map the masses and momenta of particles to background grid to
  // initialize the grid nodal masses and momenta.
  // 2. Calculate the intermediate velocities v_i^∗ using Eq.(23), and impose the
  // velocity boundary conditions.
  // 3. Initialize the signed distance function and identify the fluid cell, the air
  // cell and the free surface either based on the isocontour of particle spherical
  // function or based on a user defined initial grid level set function.
  // 4. Assemble the pressure Poisson equations Eq.(32) and solve it via a
  // PCG solver.
  // 5. Update the corrected velocities v_i^n+1 via Eq.(25).
  // 6. Suppress the hourglass modes of velocities v_i^n+1 via Eq.(53).
  // 7. Update the particle velocities via Eq.(54).
  // 8. Update the particle positions using RK3-TVD method with the 
  // updated grid velocities v_i^n+1
  // Next time step continues from step 1 to step 8 again.

  //+-----------------------------------+
  //| FOR AFFINE-PIC TRANSFER
  //+-----------------------------------+
  //| mp    p s mass
  //| xp^n  p v position
  //| vp^n  p v velocity
  //| Fp    p m deformation gradient
  //| fp    p v force
  //+-----------------------------------+
  //| Bp^n  p m affine state
  //| Cp^n  p m velocity derivatives
  //| Dp^n  p m interia-like tensor
  //+-----------------------------------+
  //| mi^n    n s mass
  //| xi      n v position 
  //| vi^n    n v velocity
  //| ~vi^n+1 n v intermediate velocity
  //| fi      n v force
  //+-----------------------------------+
  //| wip^n   n+p s weights
  //| dwip^n  n+p v weight gradients
  //+-----------------------------------+
  //| N(x)    g s interpolation kernel
  //| dx      g s grid spacing
  //| v*      g m cross prod matrix of v
  //+-----------------------------------+
  //| epsilon g t permutation tensor
  //| I       g m identity matrix
  //+-----------------------------------+
  
  // Each particle-block transfers to arena of grid-blocks
  // Arena is 2x2x2 grid-blocks (G2P2G, Quad B-Spline)
  // 
  
  // Grid-to-particle buffer size set-up
  static constexpr uint64_t numViPerBlock = g_blockvolume * 3;  //< Velocities per block
  static constexpr uint64_t numViInArena  = numViPerBlock << 3; //< Velocities per arena

  // Particle-to-grid buffer size set-up
  static constexpr uint64_t numMViPerBlock = g_blockvolume * 4;   //< Mass, momentum per block
  static constexpr uint64_t numMViInArena  = numMViPerBlock << 3; //< Mass, momentum per arena

  static constexpr unsigned arenamask = (g_blocksize << 1) - 1;   //< Arena mask
  static constexpr unsigned arenabits = g_blockbits + 1;          //< Arena bits

  extern __shared__ char shmem[];
  
  // Create shared memory grid-to-particle buffer
  // Covers 8 grid blocks (2x2x2), each block 64 nodes (4x4x4)
  // Velocity (vx, vy, vz) held, f32
  using ViArena =
      float(*)[3][g_blocksize << 1][g_blocksize << 1][g_blocksize << 1];
  using ViArenaRef =
      float(&)[3][g_blocksize << 1][g_blocksize << 1][g_blocksize << 1];
  ViArenaRef __restrict__ g2pbuffer = *reinterpret_cast<ViArena>(shmem);
  
  // Create shared memory particle-to-grid buffer
  // Covers 8 grid blocks (2x2x2), each block 64 cells (4x4x4)
  // Mass (m) and momentum (mvx, mvy, mvz) held, f32
  using MViArena =
      float(*)[4][g_blocksize << 1][g_blocksize << 1][g_blocksize << 1];
  using MViArenaRef =
      float(&)[4][g_blocksize << 1][g_blocksize << 1][g_blocksize << 1];
  MViArenaRef __restrict__ p2gbuffer =
      *reinterpret_cast<MViArena>(shmem + numViInArena * sizeof(float));

  int src_blockno = blockIdx.x; //< Source block number
  int ppb = next_pbuffer._ppbs[src_blockno]; //< Particles per block
  if (ppb == 0)
    return; //< Exit of no particles in grid-block
  auto blockid = partition._activeKeys[blockIdx.x]; //< Grid-block ID

  // Thread for each element in g2pbuffer
  for (int base = threadIdx.x; base < numViInArena; base += blockDim.x) {
    char local_block_id = base / numViPerBlock;
    auto blockno = partition.query(
        ivec3{blockid[0] + ((local_block_id & 4) != 0 ? 1 : 0),
              blockid[1] + ((local_block_id & 2) != 0 ? 1 : 0),
              blockid[2] + ((local_block_id & 1) != 0 ? 1 : 0)});
    auto grid_block = grid.ch(_0, blockno);
    int channelid = base % numViPerBlock;
    char c = channelid & 0x3f;
    char cz = channelid & g_blockmask;
    char cy = (channelid >>= g_blockbits) & g_blockmask;
    char cx = (channelid >>= g_blockbits) & g_blockmask;
    channelid >>= g_blockbits;

    // Pull from grid buffer (device)
    float val; //< Set to value from grid buffer
    if (channelid == 0)
      val = grid_block.val_1d(_1, c); //< Grid-node vx (m/s)
    else if (channelid == 1)
      val = grid_block.val_1d(_2, c); //< Grid-node vy (m/s)
    else
      val = grid_block.val_1d(_3, c); //< Grid-node vz (m/s)
    
    // Set element value in g2pbuffer (device)
    g2pbuffer[channelid][cx + (local_block_id & 4 ? g_blocksize : 0)]
             [cy + (local_block_id & 2 ? g_blocksize : 0)]
             [cz + (local_block_id & 1 ? g_blocksize : 0)] = val;
  }
  __syncthreads(); // g2pbuffer is populated
  
  // Apply Wen-Chia Yang's filter to grid arena velocities
  //float [8][8][8] Out;
  // vec9 Out;
  // Out.set(0.f);
  //wenchia_filter<float(*) [3][8][8][8], float>(&g2pbuffer, Out);

  //curand_uniform(&local_state);

  // float splits[3][8][8][8];
  // for (int base = threadIdx.x; base < numViInArena; base += blockDim.x) {
  //     char local_block_id = base / numViPerBlock;
  //     int channelid = base % numViPerBlock;
  //     //char c = channelid & 0x3f;
  //     char cz = channelid & g_blockmask;
  //     char cy = (channelid >>= g_blockbits) & g_blockmask;
  //     char cx = (channelid >>= g_blockbits) & g_blockmask;
  //     char i = cx + (local_block_id & 4 ? g_blocksize : 0);
  //     char j = cy + (local_block_id & 2 ? g_blocksize : 0);
  //     char k = cz + (local_block_id & 1 ? g_blocksize : 0);
  //     channelid >>= g_blockbits;

  //     splits[channelid][i][j][k] = g2pbuffer[channelid][i][j][k] / 6.f; 
  // }
  // __syncthreads(); // splits is populated
  
//   for (int base = threadIdx.x; base < numViInArena; base += blockDim.x) {
//       char local_block_id = base / numViPerBlock;
//       int channelid = base % numViPerBlock;
//       //char c = channelid & 0x3f;
//       char cz = channelid & g_blockmask;
//       char cy = (channelid >>= g_blockbits) & g_blockmask;
//       char cx = (channelid >>= g_blockbits) & g_blockmask;
//       char i = cx + (local_block_id & 4 ? g_blocksize : 0);
//       char j = cy + (local_block_id & 2 ? g_blocksize : 0);
//       char k = cz + (local_block_id & 1 ? g_blocksize : 0);
//       channelid >>= g_blockbits;

// #pragma unroll 6
//       for (char s = 0; s < 6; s++){

//           // Skip cycle if cell [i,j,k] on boundary
//           if (i < 1 || i >7) {
//             continue;
//           }
//           else if (j < 1 || j >7) {
//             continue;
//           }
//           else if (k < 1 || k >7) {
//             continue;
//           }

//           float ql, qc, qr; //< Field at 3 adj. cells
//           float qPlus, qMinus; //< Field +/- interface
//           float osc; //< Oscillation
//           float Sc; //< Linear limit
//           float xl, xc, xr; //< Center of 3 adj. cells
//           float mMinus, mPlus; //< Mass of +/-
//           float xPlus, xMinus; //< Center of +/-
//           float xm; //< Interface coordinate
//           float eaMinus, eaPlus; //< Allow. error of +/-
//           float tune; //< Tuning coefficient
//           float dqPlus, dqMinus; //< Field inc. of +/-
          
//           switch(s){
            
//             // Dim. 0, left-sided
//             case 0:
//               qc = splits[channelid][i][j][k];
//               ql = splits[channelid][i-1][j][k];
//               qr = splits[channelid][i+1][j][k];
//               xr = 3.f;
//               xc = 2.f;
//               xl = 1.f;
//               qMinus = ql;
//               qPlus  = qc;
//               xMinus = xl;
//               xPlus  = xc;
//               mMinus  = 1.f;
//               mPlus   = 1.f; 
//               osc = (qr-qc)/(qc-ql);             
//               break;
//             case 1:
//               qc = splits[channelid][i][j][k];
//               ql = splits[channelid][i-1][j][k];
//               qr = splits[channelid][i+1][j][k];
//               xr = 3.f;
//               xc = 2.f;
//               xl = 1.f;
//               qMinus = qc;
//               qPlus  = qr;
//               xMinus = xc;
//               xPlus  = xr;
//               mMinus  = 1.f;
//               mPlus   = 1.f; 
//               osc = (qr-qc)/(qc-ql);             
//               break;
//             case 2:
//               qc = splits[channelid][i][j][k];
//               ql = splits[channelid][i][j-1][k];
//               qr = splits[channelid][i][j+1][k];
//               xr = 3.f;
//               xc = 2.f;
//               xl = 1.f;
//               qMinus = ql;
//               qPlus  = qc;
//               xMinus = xl;
//               xPlus  = xc;
//               mMinus  = 1.f;
//               mPlus   = 1.f; 
//               osc = (qr-qc)/(qc-ql);             
//               break;
//             case 3:
//               qc = splits[channelid][i][j][k];
//               ql = splits[channelid][i][j-1][k];
//               qr = splits[channelid][i][j+1][k];
//               xr = 3.f;
//               xc = 2.f;
//               xl = 1.f;
//               qMinus = qc;
//               qPlus  = qr;
//               xMinus = xc;
//               xPlus  = xr;
//               mMinus  = 1.f;
//               mPlus   = 1.f; 
//               osc = (qr-qc)/(qc-ql);             
//               break;
//             case 4:
//               qc = splits[channelid][i][j][k];
//               ql = splits[channelid][i][j][k-1];
//               qr = splits[channelid][i][j][k+1];
//               xr = 3.f;
//               xc = 2.f;
//               xl = 1.f;
//               qMinus = ql;
//               qPlus  = qc;
//               xMinus = xl;
//               xPlus  = xc;
//               mMinus  = 1.f;
//               mPlus   = 1.f;   
//               osc = (qr-qc)/(qc-ql);           
//               break;
//             case 5:
//               qc = splits[channelid][i][j][k];
//               ql = splits[channelid][i][j][k-1];
//               qr = splits[channelid][i][j][k+1];
//               xr = 3.f;
//               xc = 2.f;
//               xl = 1.f;
//               qMinus = qc;
//               qPlus  = qr;
//               xMinus = xc;
//               xPlus  = xr;
//               mMinus  = 1.f;
//               mPlus   = 1.f; 
//               osc = (qr-qc)/(qc-ql);             
//               break;
//             default:
//               osc = 1.f;
//               break;
//           }


//           if (osc < 0.f){
//             // Linear limit for cell c
//             Sc = (qr - ql) / (xr - xl);

//             // Interface coordinate and errors
//             xm = (mMinus * xPlus + mPlus * xMinus) / (mMinus + mPlus);
//             eaMinus = (xm -xMinus) * Sc;
//             eaPlus  = (xm - xPlus) * Sc;

//             // Calculate tuning coefficient
//             if (qMinus != qPlus){
//               tune = (0.1f) * min(max(fabsf((qMinus - qPlus) + (eaMinus - eaPlus) / (qMinus - qPlus)), 0.f), 1.f);
//               //tune = tune * 1.f;
//             }
//             else{
//               tune = 0.f;
//             }
//             // Determine increments for cells +/- interface
//             dqMinus = tune * ((mPlus)  / (mMinus + mPlus)) * (qPlus - qMinus);
//             dqPlus  = tune * ((mMinus) / (mMinus + mPlus)) * (qMinus - qPlus);
//           }
//           else {
//             dqMinus = 0.f;
//             dqPlus  = 0.f;
//           }
 
//           switch(s){
//             case 0:
//               atomicAdd(&g2pbuffer[channelid][i-1][j][k], dqMinus); 
//               atomicAdd(&g2pbuffer[channelid][i  ][j][k], dqPlus); 
//               break;
//             case 1:
//               atomicAdd(&g2pbuffer[channelid][i  ][j][k], dqMinus); 
//               atomicAdd(&g2pbuffer[channelid][i+1][j][k], dqPlus); 
//               break;
//             case 2:
//               atomicAdd(&g2pbuffer[channelid][i][j-1][k], dqMinus); 
//               atomicAdd(&g2pbuffer[channelid][i][j  ][k], dqPlus); 
//               break;            
//             case 3:
//               atomicAdd(&g2pbuffer[channelid][i][j  ][k], dqMinus); 
//               atomicAdd(&g2pbuffer[channelid][i][j+1][k], dqPlus); 
//               break;
//             case 4:
//               atomicAdd(&g2pbuffer[channelid][i][j][k-1], dqMinus); 
//               atomicAdd(&g2pbuffer[channelid][i][j][k  ], dqPlus); 
//               break;
//             case 5:
//               atomicAdd(&g2pbuffer[channelid][i][j][k  ], dqMinus); 
//               atomicAdd(&g2pbuffer[channelid][i][j][k+1], dqPlus); 
//               break;
//           }
//       }
//   }
//   __syncthreads(); // p2gbuffer is filtered



  // Loop through p2gbuffer elements, set zero
  for (int base = threadIdx.x; base < numMViInArena; base += blockDim.x) {
    int loc = base;
    char z = loc & arenamask;
    char y = (loc >>= arenabits) & arenamask;
    char x = (loc >>= arenabits) & arenamask;
    p2gbuffer[loc >> arenabits][x][y][z] = 0.f;
  }
  __syncthreads(); // p2gbuffer is populated

  // Loop through particles in block
  for (int pidib = threadIdx.x; pidib < ppb; pidib += blockDim.x) {
    int source_blockno, source_pidib;
    ivec3 base_index;
    {
      int advect =
          next_pbuffer
              ._blockbuckets[src_blockno * g_particle_num_per_block + pidib];
      dir_components(advect / g_particle_num_per_block, base_index);
      base_index += blockid;
      source_blockno = prev_partition.query(base_index);
      source_pidib = advect & (g_particle_num_per_block - 1);
      source_blockno =
          pbuffer._binsts[source_blockno] + source_pidib / g_bin_capacity;
    }
    vec3 pos; //< Positions (x,y,z)
    float J;  //< Det. of Deformation Gradient, ||F||
    {
      auto source_particle_bin = pbuffer.ch(_0, source_blockno); //< Particle bin
      
      // Load positions (x,y,z) and J from particle bin
      pos[0] = source_particle_bin.val(_0, source_pidib % g_bin_capacity); //< x [0.,1.]
      pos[1] = source_particle_bin.val(_1, source_pidib % g_bin_capacity); //< y [0.,1.]
      pos[2] = source_particle_bin.val(_2, source_pidib % g_bin_capacity); //< z [0.,1.]
      J = source_particle_bin.val(_3, source_pidib % g_bin_capacity);      //< J
    }
    ivec3 local_base_index = (pos * g_dx_inv + 0.5f).cast<int>() - 1; //< Particle grid-cell index [0, g_domain_size-1]x3
    vec3 local_pos = pos - local_base_index * g_dx; //< Particle position in cell [0.,g_dx)x3
    base_index = local_base_index; //< Save index at time n

    /// Execute shape-function for grid-to-particle transfer
    /// Using Quad. B-Spline, 3x3x3 grid-nodes
    vec3x3 dws; //< Weight gradients G2P, matrix
    
    // Loop through x,y,z components of local particle position
#pragma unroll 3
    for (int dd = 0; dd < 3; ++dd) {
      // Assume p is already within kernel range [-1.5, 1.5]
      float d =
          (local_pos[dd] - ((int)(local_pos[dd] * g_dx_inv + 0.5) - 1) * g_dx) *
          g_dx_inv; //< Normalized offset, d = p * g_dx_inv 
      
      // Weight gradient for x direction
      dws(dd, 0) = 0.5f * (1.5 - d) * (1.5 - d);
      
      // Weight gradient for y direction
      d -= 1.0f;
      dws(dd, 1) = 0.75 - d * d;
      
      // Weight gradient for z direction
      d = 0.5f + d;
      dws(dd, 2) = 0.5 * d * d;

      // Modify grid-cell index for compatibility with shared g2pbuffer memory
      local_base_index[dd] = ((local_base_index[dd] - 1) & g_blockmask) + 1;
    }

    vec3 vel; //< Advected velocity on particle?
    vel.set(0.f);
    vec9 C; //< Used for matrices of Affine-PIC, (e.g. Bp, Cp)
    C.set(0.f);

    // Dp^n = Dp^n+1 = (1/4) * dx^2 * I (Quad.)
    // Dp^n = Dp^n+1 = (1/3) * dx^2 * I (Cubic)
    // Dp^n = Dp^n+1 = maybe singular, but...  (Trilinear)
    // Wip^n * (Dp^n)^-1 * (xi -xp^n) = dWip^n (Trilinear)
    float Dp_inv; //< Inverse Intertia-Like Tensor (1/m^2)
    float scale = g_length_x * g_length_x; //< Area scale (m^2)
    Dp_inv = g_D_inv / scale; //< Scalar 4/(dx^2) for Quad. B-Spline
    
    // Loop through 3x3x3 grid-nodes [i,j,k] for Quad. B-Spline shape-func.
#pragma unroll 3
    for (char i = 0; i < 3; i++)
#pragma unroll 3
      for (char j = 0; j < 3; j++)
#pragma unroll 3
        for (char k = 0; k < 3; k++) {
          // Perform G2P transfer for grid-node [i,j,k] and particle
          vec3 xixp = vec3{(float)i, (float)j, (float)k} * g_dx - local_pos; //< Pos. diff., (xi - xp)
          float W = dws(0, i) * dws(1, j) * dws(2, k); //< Weight value for grid node [i,j,k] 

          // Pull grid node velocity vector from shared g2pbuffer memory
          vec3 vi{g2pbuffer[0][local_base_index[0] + i][local_base_index[1] + j]
                           [local_base_index[2] + k],
                  g2pbuffer[1][local_base_index[0] + i][local_base_index[1] + j]
                           [local_base_index[2] + k],
                  g2pbuffer[2][local_base_index[0] + i][local_base_index[1] + j]
                           [local_base_index[2] + k]}; //< Grid-node velocity, ([1m]/s)
          vel += vi * W; //< Particle velocity increment from grid-node ([1m]/s)

          // Affine state (m^2 / s) increment for particle from grid-node [i,j,k]
          // Bp^n+1 = Sum_i( Wip^n * ~vi^n+1 * (xi - xp^n).T )
          C[0] += W * vi[0] * xixp[0] * scale;
          C[1] += W * vi[1] * xixp[0] * scale;
          C[2] += W * vi[2] * xixp[0] * scale;
          C[3] += W * vi[0] * xixp[1] * scale;
          C[4] += W * vi[1] * xixp[1] * scale;
          C[5] += W * vi[2] * xixp[1] * scale;
          C[6] += W * vi[0] * xixp[2] * scale;
          C[7] += W * vi[1] * xixp[2] * scale;
          C[8] += W * vi[2] * xixp[2] * scale;
        }

    // Advect particle position increment from G2P B-Spline
    pos += vel * dt; //< xp^n+1 = xp^n + (vp^n+1 * dt)

    /// Begin particle material update
    // Advance J^n (volume ratio, V/Vo, ||F^n||)
    // J^n+1 = (1 + tr(Bp^n+1) * Dp^-1 * dt) * J^n
    J = (1 + (C[0] + C[4] + C[8]) * Dp_inv * dt) * J;
    if (J < 0.1)
      J = 0.1; //< Lower-bound
    vec9 contrib;     //< Used for APIC matrix intermediates
    {
      // Update particle quantities
      // Vp^n+1 = Jp^n+1 * Vo 
      float voln = J * pbuffer.volume; //< Particle volume (m^3)
      
      // Pressure, Murnaghan-Tait state equation (JB)
      // Add MacDonald-Tait for tangent bulk? Birch-Murnaghan? Other models?
      // P = (Ko/n) [(Vo/V)^(n) - 1] + Patm = (bulk/gamma) [J^(-gamma) - 1] + Patm
      
       //float pressure = (pbuffer.bulk / pbuffer.gamma) * 
       //  (powf(J, -pbuffer.gamma) - 1.f); //< Pressure (Pa)

      //float pressure = (1.5 * pbuffer.bulk) * (powf(J,-7.f/3.f) - powf(J, -5.f/3.f)) * (1.f + 0.75f * (pbuffer.gamma - 4.f) * (powf(J,-2.f/3.f) - 1.f));

      {
        // Torque matrix (N * m)
        // Tp = ((Bp + Bp.T) * Dp^-1 * visco - pressure * I) * Vp
        // ((m^2 / s) * (1 / m^2) * (N s / m^2) - (N / m^2)) * (m^3) = (N * m)
        contrib[0] = ((C[0] + C[0]) * Dp_inv * pbuffer.visco) * voln;
        contrib[1] = ((C[1] + C[3]) * Dp_inv * pbuffer.visco) * voln;
        contrib[2] = ((C[2] + C[6]) * Dp_inv * pbuffer.visco) * voln;
        contrib[3] = ((C[3] + C[1]) * Dp_inv * pbuffer.visco) * voln;
        contrib[4] = ((C[4] + C[4]) * Dp_inv * pbuffer.visco) * voln;
        contrib[5] = ((C[5] + C[7]) * Dp_inv * pbuffer.visco) * voln;
        contrib[6] = ((C[6] + C[2]) * Dp_inv * pbuffer.visco) * voln;
        contrib[7] = ((C[7] + C[5]) * Dp_inv * pbuffer.visco) * voln;
        contrib[8] = ((C[8] + C[8]) * Dp_inv * pbuffer.visco) * voln;
      }
      // Mass flow-rate matrix (kg / s)
      // mp * Cp = ((Bp * mp) - (Tp * dt)) * Dp^-1 
      // ((m^2 / s) * (kg) - (N*m) * (s)) * (1/m^2) = (kg * m^2 / s) * (1/m^2) = (kg/s)
      contrib = (C * pbuffer.mass - contrib * newDt) * Dp_inv;
      {
        // Load appropiate particle buffer bin at n+1
        auto particle_bin = next_pbuffer.ch(
            _0, next_pbuffer._binsts[src_blockno] + pidib / g_bin_capacity);

        // Set particle buffer positions (x,y,z) and attributes (J) at n+1
        particle_bin.val(_0, pidib % g_bin_capacity) = pos[0]; //< x ([1m])
        particle_bin.val(_1, pidib % g_bin_capacity) = pos[1]; //< y ([1m])
        particle_bin.val(_2, pidib % g_bin_capacity) = pos[2]; //< z ([1m])
        particle_bin.val(_3, pidib % g_bin_capacity) = J;      //< J
      }
    }

    local_base_index = (pos * g_dx_inv + 0.5f).cast<int>() - 1; //< Particle cell index in 3x3. [-1,1]
    {
      int dirtag = dir_offset((base_index - 1) / g_blocksize -
                              (local_base_index - 1) / g_blocksize); //< Particle index offset (n --> n+1)
      next_pbuffer.add_advection(partition, local_base_index - 1, dirtag,
                                 pidib); //< Update particle buffer advection tagging
      // partition.add_advection(local_base_index - 1, dirtag, pidib);
    }

    // Begin Particle-to-Grid transfer
    // Loop through particle x,y,z position
#pragma unroll 3
    for (char dd = 0; dd < 3; ++dd) {
      local_pos[dd] = pos[dd] - local_base_index[dd] * g_dx;
      float d =
          (local_pos[dd] - ((int)(local_pos[dd] * g_dx_inv + 0.5) - 1) * g_dx) *
          g_dx_inv;
      dws(dd, 0) = 0.5f * (1.5 - d) * (1.5 - d);
      d -= 1.0f;
      dws(dd, 1) = 0.75 - d * d;
      d = 0.5f + d;
      dws(dd, 2) = 0.5 * d * d;

      local_base_index[dd] = (((base_index[dd] - 1) & g_blockmask) + 1) +
                             local_base_index[dd] - base_index[dd];
    }
    // Loop through 3x3 grid-nodes for B-Spline
#pragma unroll 3
    for (char i = 0; i < 3; i++)
#pragma unroll 3
      for (char j = 0; j < 3; j++)
#pragma unroll 3
        for (char k = 0; k < 3; k++) {
          pos = vec3{(float)i, (float)j, (float)k} * g_dx - local_pos; //< (xi - xp) ([1m])
          float W = dws(0, i) * dws(1, j) * dws(2, k); //< Weighting for grid node [i,j,k] 
          auto wm = pbuffer.mass * W;                  //< Weighted particle mass (kg)
          
          // Add grid node's [i,j,k] particle increment to shared p2gbuffer memory
          // mi      = Sum( Wip * mp )
          // mi * vi = Sum( Wip * mp * (vp + (Bp * Dp^-1 * (xi - xp).T) ) )
          // mi * vi = Sum((Wip * mp * vp) + (mp * Cp * (xi - xp).T * Wip))
          atomicAdd(
              &p2gbuffer[0][local_base_index[0] + i][local_base_index[1] + j]
                        [local_base_index[2] + k],
              wm); //< Mass increment (kg)
          atomicAdd(
              &p2gbuffer[1][local_base_index[0] + i][local_base_index[1] + j]
                        [local_base_index[2] + k],
              wm * vel[0] + (contrib[0] * pos[0] + contrib[3] * pos[1] +
                             contrib[6] * pos[2]) *
                                W); //< Momentum x increment (kg * m / s)
          atomicAdd(
              &p2gbuffer[2][local_base_index[0] + i][local_base_index[1] + j]
                        [local_base_index[2] + k],
              wm * vel[1] + (contrib[1] * pos[0] + contrib[4] * pos[1] +
                             contrib[7] * pos[2]) *
                                W); //< Momentum y increment (kg * m / s)
          atomicAdd(
              &p2gbuffer[3][local_base_index[0] + i][local_base_index[1] + j]
                        [local_base_index[2] + k],
              wm * vel[2] + (contrib[2] * pos[0] + contrib[5] * pos[1] +
                             contrib[8] * pos[2]) *
                                W); //< Momentum z increment (kg * m / s)
        }
  }
  __syncthreads();
  /// Finished particle --> p2gbuffer transfer
  
  /// Begin p2gbuffer   --> next grid-buffer transfer reduction
  /// arena no, channel no, cell no
  for (int base = threadIdx.x; base < numMViInArena; base += blockDim.x) {
    char local_block_id = base / numMViPerBlock;
    auto blockno = partition.query(
        ivec3{blockid[0] + ((local_block_id & 4) != 0 ? 1 : 0),
              blockid[1] + ((local_block_id & 2) != 0 ? 1 : 0),
              blockid[2] + ((local_block_id & 1) != 0 ? 1 : 0)});
    // auto grid_block = next_grid.template ch<0>(blockno);
    
    /// Avoid most conflicts in p2gbuffer --> next grid with following scheme
    int channelid = base & (numMViPerBlock - 1);         //< Thread channel ID, threadIdx.x % (4*64*4) --> [0,1023], reduces conflicts
    char c = channelid % g_blockvolume;                  //< Cell ID in grid buffer, [0,1023] % 64
    char cz = channelid & g_blockmask;                   //< Cell ID z in p2gbuffer, [0,1023] % 4
    char cy = (channelid >>= g_blockbits) & g_blockmask; //< Cell ID y in p2gbuffer, [0,1023] / 2 / 2 % 4
    char cx = (channelid >>= g_blockbits) & g_blockmask; //< Cell ID x in p2gbuffer, [0,1023] / 2 / 2 % 4
    channelid >>= g_blockbits;                           //< ChannelID in p2gbuffer, [0,1023] / 2 / 2

    // Reduce (m, mvx, mvy, mvz) from shared p2gbuffer to next grid buffer
    float val =
        p2gbuffer[channelid][cx + (local_block_id & 4 ? g_blocksize : 0)]
                 [cy + (local_block_id & 2 ? g_blocksize : 0)]
                 [cz + (local_block_id & 1 ? g_blocksize : 0)]; //< Pull (coalesced?) from shared p2gbuffer
    if (channelid == 0)
      atomicAdd(&next_grid.ch(_0, blockno).val_1d(_0, c), val); //< Add mass (kg)
    else if (channelid == 1)
      atomicAdd(&next_grid.ch(_0, blockno).val_1d(_1, c), val); //< Add x momentum (kg * [1m] / s)
    else if (channelid == 2)
      atomicAdd(&next_grid.ch(_0, blockno).val_1d(_2, c), val); //< Add y momentum (kg * [1m] / s)
    else
      atomicAdd(&next_grid.ch(_0, blockno).val_1d(_3, c), val); //< Add z momentum (kg * [1m] / s)
  }
}

template <typename Grid>
__global__ void mark_active_grid_blocks(uint32_t blockCount, const Grid grid,
                                        int *_marks) {
  auto idx = blockIdx.x * blockDim.x + threadIdx.x;
  int blockno = idx / g_blockvolume, cellno = idx % g_blockvolume;
  if (blockno >= blockCount)
    return;
  if (grid.ch(_0, blockno).val_1d(_0, cellno) != 0.f)
    _marks[blockno] = 1;
}

__global__ void mark_active_particle_blocks(uint32_t blockCount,
                                            const int *__restrict__ _ppbs,
                                            int *_marks) {
  std::size_t blockno = blockIdx.x * blockDim.x + threadIdx.x;
  if (blockno >= blockCount)
    return;
  if (_ppbs[blockno] > 0)
    _marks[blockno] = 1;
}

template <typename Partition>
__global__ void
update_partition(uint32_t blockCount, const int *__restrict__ _sourceNos,
                 const Partition partition, Partition next_partition) {
  uint32_t blockno = blockIdx.x * blockDim.x + threadIdx.x;
  if (blockno >= blockCount)
    return;
  uint32_t sourceNo = _sourceNos[blockno];
  auto sourceBlockid = partition._activeKeys[sourceNo];
  next_partition._activeKeys[blockno] = sourceBlockid;
  next_partition.reinsert(blockno);
}

template <typename ParticleBuffer>
__global__ void
update_buckets(uint32_t blockCount, const int *__restrict__ _sourceNos,
               const ParticleBuffer pbuffer, ParticleBuffer next_pbuffer) {
  __shared__ std::size_t sourceNo[1];
  std::size_t blockno = blockIdx.x;
  if (blockno >= blockCount)
    return;
  if (threadIdx.x == 0) {
    sourceNo[0] = _sourceNos[blockno];
    next_pbuffer._ppbs[blockno] = pbuffer._ppbs[sourceNo[0]];
  }
  __syncthreads();

  auto pcnt = next_pbuffer._ppbs[blockno];
  for (int pidib = threadIdx.x; pidib < pcnt; pidib += blockDim.x)
    next_pbuffer._blockbuckets[blockno * g_particle_num_per_block + pidib] =
        pbuffer._blockbuckets[sourceNo[0] * g_particle_num_per_block + pidib];
}

template <typename Partition, typename Grid>
__global__ void copy_selected_grid_blocks(
    const ivec3 *__restrict__ prev_blockids, const Partition partition,
    const int *__restrict__ _marks, Grid prev_grid, Grid grid) {
  auto blockid = prev_blockids[blockIdx.x];
  if (_marks[blockIdx.x]) {
    auto blockno = partition.query(blockid);
    if (blockno == -1)
      return;
    auto sourceblock = prev_grid.ch(_0, blockIdx.x);
    auto targetblock = grid.ch(_0, blockno);
    targetblock.val_1d(_0, threadIdx.x) = sourceblock.val_1d(_0, threadIdx.x);
    targetblock.val_1d(_1, threadIdx.x) = sourceblock.val_1d(_1, threadIdx.x);
    targetblock.val_1d(_2, threadIdx.x) = sourceblock.val_1d(_2, threadIdx.x);
    targetblock.val_1d(_3, threadIdx.x) = sourceblock.val_1d(_3, threadIdx.x);
  }
}

template <typename Partition>
__global__ void check_table(uint32_t blockCount, Partition partition) {
  uint32_t blockno = blockIdx.x * blockDim.x + threadIdx.x;
  if (blockno >= blockCount)
    return;
  auto blockid = partition._activeKeys[blockno];
  if (partition.query(blockid) != blockno)
    printf("ERROR, partition table is wrong!\n");
}
template <typename Grid> __global__ void sum_grid_mass(Grid grid, float *sum) {
  atomicAdd(sum, grid.ch(_0, blockIdx.x).val_1d(_0, threadIdx.x));
}
__global__ void sum_particle_count(uint32_t count, int *__restrict__ _cnts,
                                   int *sum) {
  auto idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= count)
    return;
  atomicAdd(sum, _cnts[idx]);
}

template <typename Partition>
__global__ void check_partition(uint32_t blockCount, Partition partition) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= blockCount)
    return;
  ivec3 blockid = partition._activeKeys[idx];
  if (blockid[0] == 0 || blockid[1] == 0 || blockid[2] == 0)
    printf("\tERROR, encountered zero block record\n");
  if (partition.query(blockid) != idx) {
    int id = partition.query(blockid);
    ivec3 bid = partition._activeKeys[id];
    printf("\t\tcheck partition %d, (%d, %d, %d), feedback index %d, (%d, %d, "
           "%d)\n",
           idx, (int)blockid[0], (int)blockid[1], (int)blockid[2], id, bid[0],
           bid[1], bid[2]);
  }
}

template <typename Partition, typename Domain>
__global__ void check_partition_domain(uint32_t blockCount, int did,
                                       Domain const domain,
                                       Partition partition) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= blockCount)
    return;
  ivec3 blockid = partition._activeKeys[idx];
  if (domain.inside(blockid)) {
    printf(
        "%d-th block (%d, %d, %d) is in domain[%d] (%d, %d, %d)-(%d, %d, %d)\n",
        idx, blockid[0], blockid[1], blockid[2], did, domain._min[0],
        domain._min[1], domain._min[2], domain._max[0], domain._max[1],
        domain._max[2]);
  }
}

/// Function to retrieve particle positions (x,y,z) [0.0, 1.0] (JB)
/// Copies from particle buffer to particle array (device --> device)
template <typename Partition, typename ParticleBuffer, typename ParticleArray>
__global__ void
retrieve_particle_buffer(Partition partition, Partition prev_partition,
                         ParticleBuffer pbuffer, ParticleBuffer next_pbuffer,
                         ParticleArray parray, int *_parcnt) {
  int pcnt = next_pbuffer._ppbs[blockIdx.x];
  ivec3 blockid = partition._activeKeys[blockIdx.x];
  auto advection_bucket =
      next_pbuffer._blockbuckets + blockIdx.x * g_particle_num_per_block;
  // auto particle_offset = pbuffer._binsts[blockIdx.x];
  for (int pidib = threadIdx.x; pidib < pcnt; pidib += blockDim.x) {
    auto advect = advection_bucket[pidib];
    ivec3 source_blockid;
    dir_components(advect / g_particle_num_per_block, source_blockid);
    source_blockid += blockid;
    auto source_blockno = prev_partition.query(source_blockid);
    auto source_pidib = advect % g_particle_num_per_block;
    auto source_bin = pbuffer.ch(_0, pbuffer._binsts[source_blockno] +
                                         source_pidib / g_bin_capacity);
    auto _source_pidib = source_pidib % g_bin_capacity;

    /// Increase particle ID
    auto parid = atomicAdd(_parcnt, 1);
    
    /// Send positions (x,y,z) [0.0, 1.0] to parray (device --> device)
    parray.val(_0, parid) = source_bin.val(_0, _source_pidib);
    parray.val(_1, parid) = source_bin.val(_1, _source_pidib);
    parray.val(_2, parid) = source_bin.val(_2, _source_pidib);
  }
}


/// Functions to retrieve particle positions and attributes (JB)
/// Copies from particle buffer to two particle arrays (device --> device)
/// Depends on material model (JFluid, FixedCorotated, NACC, Sand) 
/// Copy/paste/modify function for new material
template <typename Partition, typename ParticleArray>
__global__ void
retrieve_particle_buffer_attributes(Partition partition, Partition prev_partition,
                         ParticleBuffer<material_e::JFluid> pbuffer, 
                         ParticleBuffer<material_e::JFluid> next_pbuffer,
                         ParticleArray parray, ParticleArray pattrib, 
                         int *_parcnt) {
  int pcnt = next_pbuffer._ppbs[blockIdx.x];
  ivec3 blockid = partition._activeKeys[blockIdx.x];
  auto advection_bucket =
      next_pbuffer._blockbuckets + blockIdx.x * g_particle_num_per_block;
  for (int pidib = threadIdx.x; pidib < pcnt; pidib += blockDim.x) {
    auto advect = advection_bucket[pidib];
    ivec3 source_blockid;
    dir_components(advect / g_particle_num_per_block, source_blockid);
    source_blockid += blockid;
    auto source_blockno = prev_partition.query(source_blockid);
    auto source_pidib = advect % g_particle_num_per_block;
    auto source_bin = pbuffer.ch(_0, pbuffer._binsts[source_blockno] +
                                         source_pidib / g_bin_capacity);
    auto _source_pidib = source_pidib % g_bin_capacity;

    /// Increase particle ID
    auto parid = atomicAdd(_parcnt, 1);
    
    /// Send positions (x,y,z) [0.0, 1.0] to parray (device --> device)
    parray.val(_0, parid) = source_bin.val(_0, _source_pidib);
    parray.val(_1, parid) = source_bin.val(_1, _source_pidib);
    parray.val(_2, parid) = source_bin.val(_2, _source_pidib);


    if (1) {
      /// Send attributes (J, P, P - Patm) to pattribs (device --> device)
      float J = source_bin.val(_3, _source_pidib);
      float pressure = (pbuffer.bulk / pbuffer.gamma) * 
        (powf(J, -pbuffer.gamma) - 1.f);       //< Tait-Murnaghan Pressure (Pa)
      pattrib.val(_0, parid) = J;              //< J (V/Vo)
      pattrib.val(_1, parid) = pressure;       //< Pressure (Pa)
      pattrib.val(_2, parid) = (float)pcnt;    //< Particle count for block (#)
    }

    if (0) {
      pattrib.val(_0, parid) = source_bin.val(_3, _source_pidib);
      pattrib.val(_1, parid) = source_bin.val(_3, _source_pidib);
      pattrib.val(_2, parid) = source_bin.val(_3, _source_pidib);
    }
  }
}

template <typename Partition, typename ParticleArray>
__global__ void
retrieve_particle_buffer_attributes(Partition partition, Partition prev_partition,
                         ParticleBuffer<material_e::FixedCorotated> pbuffer, 
                         ParticleBuffer<material_e::FixedCorotated> next_pbuffer,
                         ParticleArray parray, ParticleArray pattrib, 
                         int *_parcnt) {
  int pcnt = next_pbuffer._ppbs[blockIdx.x];
  ivec3 blockid = partition._activeKeys[blockIdx.x];
  auto advection_bucket =
      next_pbuffer._blockbuckets + blockIdx.x * g_particle_num_per_block;
  // auto particle_offset = pbuffer._binsts[blockIdx.x];
  for (int pidib = threadIdx.x; pidib < pcnt; pidib += blockDim.x) {
    auto advect = advection_bucket[pidib];
    ivec3 source_blockid;
    dir_components(advect / g_particle_num_per_block, source_blockid);
    source_blockid += blockid;
    auto source_blockno = prev_partition.query(source_blockid);
    auto source_pidib = advect % g_particle_num_per_block;
    auto source_bin = pbuffer.ch(_0, pbuffer._binsts[source_blockno] +
                                         source_pidib / g_bin_capacity);
    auto _source_pidib = source_pidib % g_bin_capacity;

    /// Increase particle ID
    auto parid = atomicAdd(_parcnt, 1);
    
    /// Send positions (x,y,z) [0.0, 1.0] to parray (device --> device)
    parray.val(_0, parid) = source_bin.val(_0, _source_pidib);
    parray.val(_1, parid) = source_bin.val(_1, _source_pidib);
    parray.val(_2, parid) = source_bin.val(_2, _source_pidib);

    if (1) {
      /// Send attributes (Left-Strain Invariants) to pattribs (device --> device)
      vec9 F; //< Deformation Gradient
      F[0] = source_bin.val(_3,  _source_pidib);
      F[1] = source_bin.val(_4,  _source_pidib);
      F[2] = source_bin.val(_5,  _source_pidib);
      F[3] = source_bin.val(_6,  _source_pidib);
      F[4] = source_bin.val(_7,  _source_pidib);
      F[5] = source_bin.val(_8,  _source_pidib);
      F[6] = source_bin.val(_9,  _source_pidib);
      F[7] = source_bin.val(_10, _source_pidib);
      F[8] = source_bin.val(_11, _source_pidib);
      float U[9], S[3], V[9]; //< Left, Singulars, and Right Values of Strain
      math::svd(F[0], F[3], F[6], F[1], F[4], F[7], F[2], F[5], F[8], U[0], U[3],
                U[6], U[1], U[4], U[7], U[2], U[5], U[8], S[0], S[1], S[2], V[0],
                V[3], V[6], V[1], V[4], V[7], V[2], V[5], V[8]); // SVD Operation
      float I1, I2, I3; // Principal Invariants
      I1 = U[0] + U[4] + U[8];    //< I1 = tr(C)
      I2 = U[0]*U[4] + U[4]*U[8] + 
           U[0]*U[8] - U[3]*U[3] - 
           U[6]*U[6] - U[7]*U[7]; //< I2 = 1/2((tr(C))^2 - tr(C^2))
      I3 = U[0]*U[4]*U[8] - U[0]*U[7]*U[7] - 
           U[4]*U[6]*U[6] - U[8]*U[3]*U[3] + 
           U[3]*U[6]*U[7]*2.f;      //< I3 = ||C||

      // Set pattribs for particle to Principal Strain Invariants
      pattrib.val(_0, parid) = I1;
      pattrib.val(_1, parid) = I2;
      pattrib.val(_2, parid) = I3;
    }

    if (0) {
      pattrib.val(_0, parid) = source_bin.val(_3, _source_pidib);
      pattrib.val(_1, parid) = source_bin.val(_3, _source_pidib);
      pattrib.val(_2, parid) = source_bin.val(_3, _source_pidib);
    }
  }
}

template <typename Partition, typename ParticleArray>
__global__ void
retrieve_particle_buffer_attributes(Partition partition, Partition prev_partition,
                         ParticleBuffer<material_e::NACC> pbuffer, 
                         ParticleBuffer<material_e::NACC> next_pbuffer,
                         ParticleArray parray, ParticleArray pattrib, 
                         int *_parcnt) {
  int pcnt = next_pbuffer._ppbs[blockIdx.x];
  ivec3 blockid = partition._activeKeys[blockIdx.x];
  auto advection_bucket =
      next_pbuffer._blockbuckets + blockIdx.x * g_particle_num_per_block;
  // auto particle_offset = pbuffer._binsts[blockIdx.x];
  for (int pidib = threadIdx.x; pidib < pcnt; pidib += blockDim.x) {
    auto advect = advection_bucket[pidib];
    ivec3 source_blockid;
    dir_components(advect / g_particle_num_per_block, source_blockid);
    source_blockid += blockid;
    auto source_blockno = prev_partition.query(source_blockid);
    auto source_pidib = advect % g_particle_num_per_block;
    auto source_bin = pbuffer.ch(_0, pbuffer._binsts[source_blockno] +
                                         source_pidib / g_bin_capacity);
    auto _source_pidib = source_pidib % g_bin_capacity;

    /// Increase particle ID
    auto parid = atomicAdd(_parcnt, 1);
    
    /// Send positions (x,y,z) [0.0, 1.0] to parray (device --> device)
    parray.val(_0, parid) = source_bin.val(_0, _source_pidib);
    parray.val(_1, parid) = source_bin.val(_1, _source_pidib);
    parray.val(_2, parid) = source_bin.val(_2, _source_pidib);

    if (1) {
      /// Send attributes (F_11, F_22, F_33) to pattribs (device --> device)
      pattrib.val(_0, parid) = source_bin.val(_3,  _source_pidib);
      pattrib.val(_1, parid) = source_bin.val(_7,  _source_pidib);
      pattrib.val(_2, parid) = source_bin.val(_11, _source_pidib);
    }

    if (0) {
      pattrib.val(_0, parid) = source_bin.val(_3, _source_pidib);
      pattrib.val(_1, parid) = source_bin.val(_3, _source_pidib);
      pattrib.val(_2, parid) = source_bin.val(_3, _source_pidib);
    }
  }
}

template <typename Partition, typename ParticleArray>
__global__ void
retrieve_particle_buffer_attributes(Partition partition, Partition prev_partition,
                         ParticleBuffer<material_e::Sand> pbuffer, 
                         ParticleBuffer<material_e::Sand> next_pbuffer,
                         ParticleArray parray, ParticleArray pattrib, 
                         int *_parcnt) {
  int pcnt = next_pbuffer._ppbs[blockIdx.x];
  ivec3 blockid = partition._activeKeys[blockIdx.x];
  auto advection_bucket =
      next_pbuffer._blockbuckets + blockIdx.x * g_particle_num_per_block;
  // auto particle_offset = pbuffer._binsts[blockIdx.x];
  for (int pidib = threadIdx.x; pidib < pcnt; pidib += blockDim.x) {
    auto advect = advection_bucket[pidib];
    ivec3 source_blockid;
    dir_components(advect / g_particle_num_per_block, source_blockid);
    source_blockid += blockid;
    auto source_blockno = prev_partition.query(source_blockid);
    auto source_pidib = advect % g_particle_num_per_block;
    auto source_bin = pbuffer.ch(_0, pbuffer._binsts[source_blockno] +
                                         source_pidib / g_bin_capacity);
    auto _source_pidib = source_pidib % g_bin_capacity;

    /// Increase particle ID
    auto parid = atomicAdd(_parcnt, 1);
    
    /// Send positions (x,y,z) [0.0, 1.0] to parray (device --> device)
    parray.val(_0, parid) = source_bin.val(_0, _source_pidib);
    parray.val(_1, parid) = source_bin.val(_1, _source_pidib);
    parray.val(_2, parid) = source_bin.val(_2, _source_pidib);

    if (1) {
      /// Send attributes (F_11, F_22, F_33) to pattribs (device --> device)
      pattrib.val(_0, parid) = source_bin.val(_3,  _source_pidib);
      pattrib.val(_1, parid) = source_bin.val(_7,  _source_pidib);
      pattrib.val(_2, parid) = source_bin.val(_11, _source_pidib);
    }

    if (0) {
      pattrib.val(_0, parid) = source_bin.val(_3, _source_pidib);
      pattrib.val(_1, parid) = source_bin.val(_3, _source_pidib);
      pattrib.val(_2, parid) = source_bin.val(_3, _source_pidib);
    }
  }
}

template <typename Partition, typename ParticleArray>
__global__ void
retrieve_particle_buffer_attributes(Partition partition, Partition prev_partition,
                         ParticleBuffer<material_e::Rigid> pbuffer, 
                         ParticleBuffer<material_e::Rigid> next_pbuffer,
                         ParticleArray parray, ParticleArray pattrib, 
                         int *_parcnt) {
  int pcnt = next_pbuffer._ppbs[blockIdx.x];
  ivec3 blockid = partition._activeKeys[blockIdx.x];
  auto advection_bucket =
      next_pbuffer._blockbuckets + blockIdx.x * g_particle_num_per_block;
  // auto particle_offset = pbuffer._binsts[blockIdx.x];
  for (int pidib = threadIdx.x; pidib < pcnt; pidib += blockDim.x) {
    auto advect = advection_bucket[pidib];
    ivec3 source_blockid;
    dir_components(advect / g_particle_num_per_block, source_blockid);
    source_blockid += blockid;
    auto source_blockno = prev_partition.query(source_blockid);
    auto source_pidib = advect % g_particle_num_per_block;
    auto source_bin = pbuffer.ch(_0, pbuffer._binsts[source_blockno] +
                                         source_pidib / g_bin_capacity);
    auto _source_pidib = source_pidib % g_bin_capacity;

    /// Increase particle ID
    auto parid = atomicAdd(_parcnt, 1);
    
    /// Send positions (x,y,z) [0.0, 1.0] to parray (device --> device)
    parray.val(_0, parid) = source_bin.val(_0, _source_pidib);
    parray.val(_1, parid) = source_bin.val(_1, _source_pidib);
    parray.val(_2, parid) = source_bin.val(_2, _source_pidib);

    if (1) {
      /// Send attributes (F_11, F_22, F_33) to pattribs (device --> device)
      pattrib.val(_0, parid) = source_bin.val(_3,  _source_pidib);

    }
  }
}

template <typename Partition, typename ParticleArray>
__global__ void
retrieve_particle_buffer_attributes(Partition partition, Partition prev_partition,
                         ParticleBuffer<material_e::Piston> pbuffer, 
                         ParticleBuffer<material_e::Piston> next_pbuffer,
                         ParticleArray parray, ParticleArray pattrib, 
                         int *_parcnt) {
  int pcnt = next_pbuffer._ppbs[blockIdx.x];
  ivec3 blockid = partition._activeKeys[blockIdx.x];
  auto advection_bucket =
      next_pbuffer._blockbuckets + blockIdx.x * g_particle_num_per_block;
  // auto particle_offset = pbuffer._binsts[blockIdx.x];
  for (int pidib = threadIdx.x; pidib < pcnt; pidib += blockDim.x) {
    auto advect = advection_bucket[pidib];
    ivec3 source_blockid;
    dir_components(advect / g_particle_num_per_block, source_blockid);
    source_blockid += blockid;
    auto source_blockno = prev_partition.query(source_blockid);
    auto source_pidib = advect % g_particle_num_per_block;
    auto source_bin = pbuffer.ch(_0, pbuffer._binsts[source_blockno] +
                                         source_pidib / g_bin_capacity);
    auto _source_pidib = source_pidib % g_bin_capacity;

    /// Increase particle ID
    auto parid = atomicAdd(_parcnt, 1);
    
    /// Send positions (x,y,z) [0.0, 1.0] to parray (device --> device)
    parray.val(_0, parid) = source_bin.val(_0, _source_pidib);
    parray.val(_1, parid) = source_bin.val(_1, _source_pidib);
    parray.val(_2, parid) = source_bin.val(_2, _source_pidib);

    if (1) {
      /// Send attributes (F_11, F_22, F_33) to pattribs (device --> device)
      pattrib.val(_0, parid) = source_bin.val(_3,  _source_pidib);
      pattrib.val(_1, parid) = source_bin.val(_7,  _source_pidib);
      pattrib.val(_2, parid) = source_bin.val(_11, _source_pidib);
    }

    if (0) {
      pattrib.val(_0, parid) = source_bin.val(_3, _source_pidib);
      pattrib.val(_1, parid) = source_bin.val(_3, _source_pidib);
      pattrib.val(_2, parid) = source_bin.val(_3, _source_pidib);
    }
  }
}

template <typename Partition, typename ParticleArray>
__global__ void
retrieve_particle_buffer_attributes(Partition partition, Partition prev_partition,
                         ParticleBuffer<material_e::IFluid> pbuffer, 
                         ParticleBuffer<material_e::IFluid> next_pbuffer,
                         ParticleArray parray, ParticleArray pattrib, 
                         int *_parcnt) {
  int pcnt = next_pbuffer._ppbs[blockIdx.x];
  ivec3 blockid = partition._activeKeys[blockIdx.x];
  auto advection_bucket =
      next_pbuffer._blockbuckets + blockIdx.x * g_particle_num_per_block;
  //auto atm = g_atm; //< Atmospheric pressure (Pa)
  for (int pidib = threadIdx.x; pidib < pcnt; pidib += blockDim.x) {
    auto advect = advection_bucket[pidib];
    ivec3 source_blockid;
    dir_components(advect / g_particle_num_per_block, source_blockid);
    source_blockid += blockid;
    auto source_blockno = prev_partition.query(source_blockid);
    auto source_pidib = advect % g_particle_num_per_block;
    auto source_bin = pbuffer.ch(_0, pbuffer._binsts[source_blockno] +
                                         source_pidib / g_bin_capacity);
    auto _source_pidib = source_pidib % g_bin_capacity;

    /// Increase particle ID
    auto parid = atomicAdd(_parcnt, 1);
    
    /// Send positions (x,y,z) [0.0, 1.0] to parray (device --> device)
    parray.val(_0, parid) = source_bin.val(_0, _source_pidib);
    parray.val(_1, parid) = source_bin.val(_1, _source_pidib);
    parray.val(_2, parid) = source_bin.val(_2, _source_pidib);


    if (1) {
      /// Send attributes (J, P, P - Patm) to pattribs (device --> device)
      pattrib.val(_0, parid) = 0.f;              //< J (V/Vo)
      pattrib.val(_1, parid) = 0.f; //< Pn + Patm (Pa)
      pattrib.val(_2, parid) = 0.f;       //< Pn (Pa)
    }

    if (0) {
      pattrib.val(_0, parid) = source_bin.val(_3, _source_pidib);
      pattrib.val(_1, parid) = source_bin.val(_3, _source_pidib);
      pattrib.val(_2, parid) = source_bin.val(_3, _source_pidib);
    }
  }
}

/// Retrieve selected down-sampled grid values from grid buffer to grid array (JB)
template <typename Partition, typename Grid, typename GridArray>
__global__ void retrieve_selected_grid_blocks(
    const ivec3 *__restrict__ prev_blockids, const Partition partition,
    const int *__restrict__ _marks, Grid prev_grid, GridArray garray) {
  auto blockno = blockIdx.x;
  auto blockid = prev_blockids[blockno];
  if (_marks[blockno]) {
    //auto blockno = partition.query(blockid);
    if (blockno == -1)
      return;
    auto node_id = blockIdx.x;
    auto sourceblock = prev_grid.ch(_0, node_id);

    /// Set block index via leader thread (e.g. 0)
    if (threadIdx.x == 0){
      garray.val(_0, node_id) = 4.f * blockid[0] * g_dx + (2.f * g_dx);
      garray.val(_1, node_id) = 4.f * blockid[1] * g_dx + (2.f * g_dx);
      garray.val(_2, node_id) = 4.f * blockid[2] * g_dx + (2.f * g_dx);
      garray.val(_3, node_id) = 0.f;
      garray.val(_4, node_id) = 0.f;
      garray.val(_5, node_id) = 0.f;
      garray.val(_6, node_id) = 0.f;
    }

    /// Synch threads in block
    __syncthreads();

    /// Create temp values in threads for specific cells
    auto m   = sourceblock.val_1d(_0, threadIdx.x);
    auto mvx = sourceblock.val_1d(_1, threadIdx.x);
    auto mvy = sourceblock.val_1d(_2, threadIdx.x);
    auto mvz = sourceblock.val_1d(_3, threadIdx.x);

    /// Ensure temp values are populated in all threads
    __syncthreads();

    /// Atomically add thread (cell) values to block (grid-block) value
    atomicAdd_block(&garray.val(_3, node_id), m);
    atomicAdd_block(&garray.val(_4, node_id), mvx);
    atomicAdd_block(&garray.val(_5, node_id), mvy);
    atomicAdd_block(&garray.val(_6, node_id), mvz);
    __syncthreads();  
  }
}

/// Retrieve grid-cells between points a & b from grid-buffer to gridTarget (JB)
template <typename Partition, typename Grid, typename GridTarget>
__global__ void retrieve_selected_grid_cells(
    uint32_t blockCount, const ivec3 *__restrict__ prev_blockids, const Partition partition,
    const int *__restrict__ _marks, Grid prev_grid, GridTarget garray,
    float dt, float *forceSum, vec3 point_a, vec3 point_b) {

  auto blockno = blockIdx.x;  //< Block number in partition
  if (1) {
    //auto blockid = prev_blockids[blockno]; //< 3D grid-block index
    auto blockid = partition._activeKeys[blockno];
    if (blockno < blockCount) {
      if (blockno == -1)
        return;

      auto sourceblock = prev_grid.ch(_0, blockno); //< Set grid-block by block index

      // Tolerance layer thickness around designated target space
      float tol = g_dx * 0.0f;

      // Add +1 to each? For point_b ~= point_a...
      ivec3 maxNodes_coord;
      maxNodes_coord[0] = (int)((point_b[0] + tol - point_a[0] + tol) * g_dx_inv + 0.5);
      maxNodes_coord[1] = (int)((point_b[1] + tol - point_a[1] + tol) * g_dx_inv + 0.5);
      maxNodes_coord[2] = (int)((point_b[2] + tol - point_a[2] + tol) * g_dx_inv + 0.5);
      int maxNodes = maxNodes_coord[0] * maxNodes_coord[1] * maxNodes_coord[2];
      if (maxNodes >= g_target_cells) printf("Allocate more space for gridTarget!\n");

      // Loop through cells in grid-block, stride by 32 to avoid thread conflicts
      for (int cidib = threadIdx.x % 32; cidib < g_blockvolume; cidib += 32) {

        // Grid node coordinate [i,j,k] in grid-block
        int i = (cidib >> (g_blockbits << 1)) & g_blockmask;
        int j = (cidib >> g_blockbits) & g_blockmask;
        int k = cidib & g_blockmask;

        // Grid node position [x,y,z] in entire domain 
        float xc = (4*blockid[0]*g_dx) + (i*g_dx); // + (g_dx/2.f);
        float yc = (4*blockid[1]*g_dx) + (j*g_dx); // + (g_dx/2.f);
        float zc = (4*blockid[2]*g_dx) + (k*g_dx); // + (g_dx/2.f);

        // Exit thread if cell is not inside grid-target +/- tol
        if (xc < point_a[0] - tol || xc > point_b[0] + tol) {
          continue;
        }
        if (yc < point_a[1] - tol || yc > point_b[1] + tol) {
          continue;
        }
        if (zc < point_a[2] - tol || zc > point_b[2] + tol) {
          continue;
        }

        // Unique ID by spatial position of cell in target [0 to g_target_cells-1]
        int node_id;
        // node_id = ((int)((zc - point_a[2] + tol) * g_dx_inv + 0.5f) * maxNodes_coord[1] * maxNodes_coord[0]) +
        //           ((int)((yc - point_a[1] + tol) * g_dx_inv + 0.5f) * maxNodes_coord[0]) +
        //           ((int)((xc - point_a[0] + tol) * g_dx_inv + 0.5f));
        node_id = ((int)((xc - point_a[0] + tol) * g_dx_inv + 0.5f) * maxNodes_coord[1] * maxNodes_coord[2]) +
                  ((int)((yc - point_a[1] + tol) * g_dx_inv + 0.5f) * maxNodes_coord[2]) +
                  ((int)((zc - point_a[2] + tol) * g_dx_inv + 0.5f));
        while (garray.val(_3, node_id) != 0.f) {
          node_id += 1;
          if (node_id > g_target_cells) {
            printf("node_id bigger than g_target_cells!");
            break;
          }
        }
        __syncthreads(); // Sync threads in block

        /// Set values in grid-array to specific cell from grid-buffer
        garray.val(_0, node_id) = xc;
        garray.val(_1, node_id) = yc;
        garray.val(_2, node_id) = zc;
        garray.val(_3, node_id) = sourceblock.val(_0, i, j, k);
        garray.val(_4, node_id) = sourceblock.val(_1, i, j, k);
        garray.val(_5, node_id) = sourceblock.val(_2, i, j, k);
        garray.val(_6, node_id) = sourceblock.val(_3, i, j, k);

        __syncthreads(); // Sync threads in block

        /// Set values in grid-array to specific cell from grid-buffer
        float m1  = garray.val(_3, node_id);
        float m2  = m1;
        float m = m1;
        if (m1 > 0.f) {
          m1 = 1.f / m1; //< Invert mass, avoids division operator
        }
        if (m2 > 0.f) {
          m2 = 1.f / m2; //< Invert mass, avoids division operator
        }

        float vx1 = garray.val(_4, node_id) * m1 * g_length;
        float vy1 = garray.val(_5, node_id) * m1 * g_length;
        float vz1 = garray.val(_6, node_id) * m1 * g_length;
        float vx2 = 0.f;
        float vy2 = 0.f;
        float vz2 = 0.f;

        float fx = m * (vx1 - vx2) / dt;
        float fy = m * (vy1 - vy2) / dt;
        float fz = m * (vz1 - vz2) / dt;

        garray.val(_7, node_id) = fx;
        garray.val(_8, node_id) = fy;
        garray.val(_9, node_id) = fz;
        __syncthreads(); // Sync threads in block
        atomicAdd(forceSum, fx);
        __syncthreads(); // Sync threads in block
      }
    }
  }
}


/// Retrieve grid-cells between points a & b from grid-buffer to gridTarget (JB)
template <typename Partition, typename Grid, typename GridTarget>
__global__ void process_grid_target_forces(
  const ivec3 *__restrict__ prev_blockids, const Partition partition,
  const int *__restrict__ _marks, Grid prev_grid, 
  GridTarget garray_new, GridTarget garray_old, GridTarget garray_post, int fps, 
  float *forceSum) {

  //extern __shared__ float sh_forceSums[]; //< Max vel.^2s for block, shared mem
  // std::size_t blockno = blockIdx.x; //< Grid-block number
  // //auto blockid = partition._activeKeys[blockno]; //< Grid-block ID, [i,j,k]
  // auto blockid = prev_blockids[blockno]; //< Grid-block ID, [i,j,k]


  // Reset GridTarget Array values to zero if first thread in block
  if (threadIdx.x == 0 && blockIdx.x*blockDim.x < g_target_cells) {
    for (int cidib = 0; cidib < g_blockvolume; cidib++) {
      int temp_id = cidib + blockIdx.x*g_blockvolume;
      if (temp_id >= g_target_cells) break;
      garray_post.val(_0, temp_id) = 0.f;
      garray_post.val(_1, temp_id) = 0.f;
      garray_post.val(_2, temp_id) = 0.f;
      garray_post.val(_3, temp_id) = 0.f;
      garray_post.val(_4, temp_id) = 0.f;
      garray_post.val(_5, temp_id) = 0.f;
      garray_post.val(_6, temp_id) = 0.f;
    }
  }
  __syncthreads();

  // If block is active
  if (_marks[blockIdx.x]) {
    auto blockid = prev_blockids[blockIdx.x]; //< 3D grid-block index
    auto blockno = partition.query(blockid);  //< Block number in partition
    if (blockno == -1)
      return;

    // Loop through cells in grid-block, stride by 32 to avoid thread conflicts
    for (int cidib = threadIdx.x % 32; cidib < g_blockvolume; cidib += 32) {

      int node_id = cidib + (blockno * g_blockvolume);


      // Grid node coordinate [i,j,k] in grid-block
      int i = (cidib >> (g_blockbits << 1)) & g_blockmask;
      int j = (cidib >> g_blockbits) & g_blockmask;
      int k = cidib & g_blockmask;

      // Grid node position [x,y,z] in entire domain 
      float xc = (4*blockid[0]*g_dx) + (i*g_dx); // + (g_dx/2.f);
      float yc = (4*blockid[1]*g_dx) + (j*g_dx); // + (g_dx/2.f);
      float zc = (4*blockid[2]*g_dx) + (k*g_dx); // + (g_dx/2.f);
      // float xc = garray_new.val(_0, node_id);
      // float yc = garray_new.val(_1, node_id);
      // float zc = garray_new.val(_2, node_id);

      if (garray_new.val(_0, node_id) != garray_old.val(_0, node_id)) {
          printf("ERROR Location mismatch in process_grid_target_forces!");
      };

      __syncthreads(); // Sync threads in block

      /// Set values in grid-array to specific cell from grid-buffer
      float m1  = garray_new.val(_3, node_id);
      float m2  = garray_old.val(_3, node_id);
      float m = (m1 + m2) / 2;

      if (m1 > 0.f) {
        m1 = 1.f / m1; //< Invert mass, avoids division operator
      }
      if (m2 > 0.f) {
        m2 = 1.f / m2; //< Invert mass, avoids division operator
      }

      float vx1 = garray_new.val(_4, node_id) * m1 * g_length;
      float vy1 = garray_new.val(_5, node_id) * m1 * g_length;
      float vz1 = garray_new.val(_6, node_id) * m1 * g_length;

      float vx2 = garray_old.val(_4, node_id) * m2 * g_length;
      float vy2 = garray_old.val(_5, node_id) * m2 * g_length;
      float vz2 = garray_old.val(_6, node_id) * m2 * g_length;

      float fx = m * (vx1 - vx2) * (float)fps;
      float fy = m * (vy1 - vy2) * (float)fps;
      float fz = m * (vz1 - vz2) * (float)fps;

      garray_post.val(_0, node_id) = xc;
      garray_post.val(_1, node_id) = yc;
      garray_post.val(_2, node_id) = zc;
      garray_post.val(_3, node_id) = m;
      garray_post.val(_4, node_id) = fx;
      garray_post.val(_5, node_id) = fy;
      garray_post.val(_6, node_id) = fz;

      __syncthreads(); // Sync threads in block
      atomicAdd(forceSum, fx);
      __syncthreads(); // Sync threads in block
    }
  }
  // if ((threadIdx.x + blockIdx.x*blockDim.x) == 0) {
  //   printf("Inside process_grid_target_forces!");
  // }
  // __syncthreads(); // Sync threads in block
  // atomicAdd(forceSum, forceBlockSum);
}


} // namespace mn

#endif