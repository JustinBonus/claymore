#ifndef __MULTI_GMPM_KERNELS_CUH_
#define __MULTI_GMPM_KERNELS_CUH_

#include "constitutive_models.cuh"
#include "particle_buffer.cuh"
#include "settings.h"
#include "utility_funcs.hpp"
#include <MnBase/Algorithm/MappingKernels.cuh>
#include <MnBase/Math/Matrix/MatrixUtils.h>
#include <MnSystem/Cuda/DeviceUtils.cuh>

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

template <typename Grid, typename Partition>
__global__ void update_grid_velocity_query_max(uint32_t blockCount, Grid grid,
                                               Partition partition, float dt,
                                               float *maxVel) {
  constexpr int bc = g_bc; //< Num of 'buffer' grid-blocks at domain exterior
  constexpr int numWarps = g_num_grid_blocks_per_cuda_block * 
    g_num_warps_per_grid_block; //< Warps per block
  constexpr unsigned activeMask = 0xffffffff;
  
  float gravity = grid.gravity;
  float length = grid.length;
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
  
  // Add grid-block enforced boundary for cuboid/column/columns (JB)
  int boxx = 2; //< x-width grid-blocks 
  //int boxy = 6; //< y-width grid-blocks
  int boxz = 2; //< z-width grid-blocks
  
  int buffx = ( g_grid_size_x >> 1 ); //< x grid-block offest
  //int buffy = ((g_grid_size_y >> 1) - (boxy/2)); //< y grid-block offset
  int buffz = ((g_grid_size_z >> 1) - (boxz/2)); //< z grid-block offset
  
  // 3 digits for grid-block in column
  // e.g. 011 means y and z presence, but not x 
  // Must be 111 for grid-block to be in column
  int isOutColumn  = ((blockid[0] >= buffx && blockid[0] < boxx + buffx )       << 2) | 
                     ((blockid[1] >= 0     && blockid[1] <  g_grid_size_y)      << 1) |
                      (blockid[2] >= buffz && blockid[2] < boxz + buffz);

  // Check if 3 flags are tripped, reset if not (JB)
  if (isOutColumn != 7) isOutColumn = 0;

  // Add chanel and column boundary results to box boundary results (JB)
  isInBound |= isOutColumn;

  // One element in shared vel.^2 per warp
  if (threadIdx.x < numWarps)
    sh_maxvels[threadIdx.x] = 0.0f;
  
  // Synch threads, boundary collision check finished
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
#if 0
      int i = (cidib >> (g_blockbits << 1)) & g_blockmask;
      int j = (cidib >> g_blockbits) & g_blockmask;
      int k = cidib & g_blockmask;
#endif
        // Retrieve grid momentums (kg*m/s2)
        vel[0] = grid_block.val_1d(_1, cidib); //< mvx
        vel[1] = grid_block.val_1d(_2, cidib); //< mvy
        vel[2] = grid_block.val_1d(_3, cidib); //< mvz

        ///< Slip contact
        if (1){
          // Set cell velocity (m/s) after grid-block boundary check
          vel[0] = isInBound & 4 ? 0.f : vel[0] * mass; //< vx = mvx / m
          vel[1] = isInBound & 2 ? 0.f : vel[1] * mass; //< vy = mvy / m
          vel[1] += (gravity * dt ) / length;       //< Grav (raises?)          
          vel[2] = isInBound & 1 ? 0.f : vel[2] * mass; //< vz = mvz / m
        }
        
        ///< Sticky contact
        if (0){
          if (isInBound) ///< sticky
            vel.set(0.f);
        }
        
        // Set grid buffer momentum to velocity (m/s) for G2P transfer
        grid_block.val_1d(_1, cidib) = vel[0]; //< vx
        velSqr += vel[0] * vel[0];

        grid_block.val_1d(_2, cidib) = vel[1]; //< vy
        velSqr += vel[1] * vel[1];

        grid_block.val_1d(_3, cidib) = vel[2]; //< vz
        velSqr += vel[2] * vel[2];
      }
      // Reduce velocity^2 from threads
      // Loop 
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
  /// various assumptions
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
__global__ void g2p2g(float dt, float newDt,
                      const ParticleBuffer<material_e::JFluid> pbuffer,
                      ParticleBuffer<material_e::JFluid> next_pbuffer,
                      const Partition prev_partition, Partition partition,
                      const Grid grid, Grid next_grid) {
  //============================================================
  // Grid-to-Particle-to Grid Kernel for JFluid material
  // Transfer Scheme:  Affine Particle-in-Cell
  // Shape-function:   Quadratic B-Spline
  // Time Integration: Explicit
  // Material:         JFluid, weakly incompressible fluid
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
      val = grid_block.val_1d(_1, c); //< Grid-node vx ([1m]/s)
    else if (channelid == 1)
      val = grid_block.val_1d(_2, c); //< Grid-node vy ([1m]/s)
    else
      val = grid_block.val_1d(_3, c); //< Grid-node vz ([1m]/s)
    
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
    float Dp_inv; //< Inverse Intertia-Like Tensor (m^-2)
    float scale = grid.length * grid.length; //< Area scale (m^2)
    Dp_inv = g_D_inv / scale; //< Scale to scene lenths
    
    // Loop through 3x3x3 grid-nodes [i,j,k] for Quad. B-Spline shape-func.
#pragma unroll 3
    for (char i = 0; i < 3; i++)
#pragma unroll 3
      for (char j = 0; j < 3; j++)
#pragma unroll 3
        for (char k = 0; k < 3; k++) {
          // Perform G2P transfer for grid-node [i,j,k] and particle
          vec3 xixp = vec3{(float)i, (float)j, (float)k} * g_dx - local_pos; //< (xi - xp)
          float W = dws(0, i) * dws(1, j) * dws(2, k); //< Weighs for grid node [i,j,k] 
          
          // Pull grid node velocity vector from shared g2pbuffer memory
          vec3 vi{g2pbuffer[0][local_base_index[0] + i][local_base_index[1] + j]
                           [local_base_index[2] + k],
                  g2pbuffer[1][local_base_index[0] + i][local_base_index[1] + j]
                           [local_base_index[2] + k],
                  g2pbuffer[2][local_base_index[0] + i][local_base_index[1] + j]
                           [local_base_index[2] + k]}; //< Grid-node velocity, ([1m]/s)
          vel += vi * W; //< Particle velocity increment from grid-node ([1m]/s)

          // Affine state (m^2 / s)
          // Increment for particle from grid-node [i,j,k]
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
    // Advect particle position increment
    pos += vel * dt; //< xp^n+1 = xp^n + (vp^n+1 * dt) ([1m])

    /// Begin particle material update
    // Advance J^n (volume ratio, V/Vo, ||F^n||)
    // J^n+1 = (1 + tr(Bp^n+1) * Dp^-1 * dt) * J^n
    J = (1 + (C[0] + C[4] + C[8]) * Dp_inv * dt) * J;
    if (J < 0.1)
      J = 0.1;    //< Lower-bound V/Vo
    vec9 contrib; //< Used for APIC matrix intermediates
    {
      // Update particle quantities
      // Vp^n+1 = Jp^n+1 * Vo 
      float voln = J * pbuffer.volume; //< Particle volume (m^3)
      
      // Pressure, Murnaghan-Tait state equation (JB)
      // Add MacDonald-Tait for tangent bulk? Birch-Murnaghan? Other models?
      // P = (Ko/n) [(Vo/V)^(n) - 1] + Patm = (bulk/gamma) [J^(-gamma) - 1] + Patm
      float pressure = (pbuffer.bulk / pbuffer.gamma) * 
        (powf(J, -pbuffer.gamma) - 1.f) + grid.atm; //< Pressure (Pa)
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
                                W); //< Momentum x increment (kg * [1m] / s)
          atomicAdd(
              &p2gbuffer[2][local_base_index[0] + i][local_base_index[1] + j]
                        [local_base_index[2] + k],
              wm * vel[1] + (contrib[1] * pos[0] + contrib[4] * pos[1] +
                             contrib[7] * pos[2]) *
                                W); //< Momentum y increment (kg * [1m] / s)
          atomicAdd(
              &p2gbuffer[3][local_base_index[0] + i][local_base_index[1] + j]
                        [local_base_index[2] + k],
              wm * vel[2] + (contrib[2] * pos[0] + contrib[5] * pos[1] +
                             contrib[8] * pos[2]) *
                                W); //< Momentum z increment (kg * [1m] / s)
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

template <typename Partition, typename Grid>
__global__ void g2p2g(float dt, float newDt,
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
    
    // Dp^n = Dp^n+1 = (1/4) * dx^2 * I (Quad.)
    // Dp^n = Dp^n+1 = (1/3) * dx^2 * I (Cubic)
    // Dp^n = Dp^n+1 = maybe singular, but...  (Trilinear)
    // Wip^n * (Dp^n)^-1 * (xi -xp^n) = dWip^n (Trilinear)
    float Dp_inv; //< Inverse Intertia-Like Tensor (1/m^2)
    float scale = grid.length * grid.length; //< Area scale (m^2)
    Dp_inv = g_D_inv / scale; //< Scalar 4/(dx^2) for Quad. B-Spline
    
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
    pos += vel * dt;

#pragma unroll 9
    for (int d = 0; d < 9; ++d)
      // dWip = Bp * Dp^-1 * dt + I
      // (m^2 / s) * (1 / m^2) * (s) = ( )
      dws.val(d) = C[d] * Dp_inv * dt + ((d & 0x3) ? 0.f : 1.f);

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
      contrib = (C * pbuffer.mass - contrib * newDt) * Dp_inv;
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
__global__ void g2p2g(float dt, float newDt,
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
    
    // Dp^n = Dp^n+1 = (1/4) * dx^2 * I (Quad.)
    // Dp^n = Dp^n+1 = (1/3) * dx^2 * I (Cubic)
    // Dp^n = Dp^n+1 = maybe singular, but...  (Trilinear)
    // Wip^n * (Dp^n)^-1 * (xi -xp^n) = dWip^n (Trilinear)
    float Dp_inv; //< Inverse Intertia-Like Tensor (m^-2)
    float scale = grid.length * grid.length; //< Area scale (m^2)
    Dp_inv = g_D_inv / scale; //< Scalar 4/(dx^2) for Quad. B-Spline
    
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
    pos += vel * dt;

#pragma unroll 9
    for (int d = 0; d < 9; ++d)
      dws.val(d) = C[d] * Dp_inv * dt + ((d & 0x3) ? 0.f : 1.f);

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

      contrib = (C * pbuffer.mass - contrib * newDt) * Dp_inv;
    }

    local_base_index = (pos * g_dx_inv + 0.5f).cast<int>() - 1;
    {
      int dirtag = dir_offset((base_index - 1) / g_blocksize -
                              (local_base_index - 1) / g_blocksize);
      next_pbuffer.add_advection(partition, local_base_index - 1, dirtag,
                                 pidib);
    }

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
__global__ void g2p2g(float dt, float newDt,
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
    
    // Dp^n = Dp^n+1 = (1/4) * dx^2 * I (Quad.)
    // Dp^n = Dp^n+1 = (1/3) * dx^2 * I (Cubic)
    // Dp^n = Dp^n+1 = maybe singular, but...  (Trilinear)
    // Wip^n * (Dp^n)^-1 * (xi -xp^n) = dWip^n (Trilinear)
    float Dp_inv; //< Inverse Intertia-Like Tensor (1/m^2)
    float scale = grid.length * grid.length; //< Area scale (m^2)
    Dp_inv = g_D_inv / scale; //< Scalar 4/(dx^2) for Quad. B-Spline
    
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
    pos += vel * dt;

#pragma unroll 9
    for (int d = 0; d < 9; ++d)
      dws.val(d) = C[d] * Dp_inv * dt + ((d & 0x3) ? 0.f : 1.f);

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

      contrib = (C * pbuffer.mass - contrib * newDt) * Dp_inv;
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
  auto atm = g_atm; //< Atmospheric pressure (Pa)
  auto length = g_length;
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
    
    /// Send positions (x,y,z) (m) to parray (device --> device)
    parray.val(_0, parid) = source_bin.val(_0, _source_pidib) * length;
    parray.val(_1, parid) = source_bin.val(_1, _source_pidib) * length;
    parray.val(_2, parid) = source_bin.val(_2, _source_pidib) * length;


    if (1) {
      /// Send attributes (J, P, P - Patm) to pattribs (device --> device)
      float J = source_bin.val(_3, _source_pidib);
      float pressure = (pbuffer.bulk / pbuffer.gamma) * 
        (powf(J, -pbuffer.gamma) - 1.f);       //< Tait-Murnaghan Pressure (Pa)
      pattrib.val(_0, parid) = J;              //< J (V/Vo)
      pattrib.val(_1, parid) = pressure + atm; //< Pn + Patm (Pa)
      pattrib.val(_2, parid) = pressure;       //< Pn (Pa)
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
  auto length = g_length;
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
    
    /// Send positions (x,y,z) (m) to parray (device --> device)
    parray.val(_0, parid) = source_bin.val(_0, _source_pidib) * length;
    parray.val(_1, parid) = source_bin.val(_1, _source_pidib) * length;
    parray.val(_2, parid) = source_bin.val(_2, _source_pidib) * length;

    // Deformation Gradient from particle buffer
    vec9 F;     //< Deformation Gradient
    F.set(0.f); //< Zero
    F[0] = source_bin.val(_3, _source_pidib);
    F[1] = source_bin.val(_4, _source_pidib);
    F[2] = source_bin.val(_5, _source_pidib);
    F[3] = source_bin.val(_6, _source_pidib);
    F[4] = source_bin.val(_7, _source_pidib);
    F[5] = source_bin.val(_8, _source_pidib);
    F[6] = source_bin.val(_9, _source_pidib);
    F[7] = source_bin.val(_10, _source_pidib);
    F[8] = source_bin.val(_11, _source_pidib);


    int retrieve_stretch_invariants = 0;
    int retrieve_cauchy_invariants  = 0;


    if (retrieve_stretch_invariants) {
      vec9 U;
      U.set(0.f);
      vec3 I;
      I.set(0.f);

      // Retrieve right stretch tensor U for deformation gradient F
      compute_stretch(F, U);

      /// Right stretch tensor invariants
      I[0] = U[0] + U[4] + U[8];    //< I1 = tr(U)
      I[1] = U[0]*U[4] + U[4]*U[8] + 
             U[0]*U[8] - U[3]*U[3] - 
             U[6]*U[6] - U[7]*U[7]; //< I2 = 1/2((tr(U))^2 - tr(U^2))
      I[2] = U[0]*U[4]*U[8] - U[0]*U[7]*U[7] - 
             U[4]*U[6]*U[6] - U[8]*U[3]*U[3] + 
             2*U[3]*U[6]*U[7];      //< I3 = ||U||

      /// Send attributes (I1, I2, I3) to pattribs (device --> device)
      pattrib.val(_0, parid) = I[0];
      pattrib.val(_1, parid) = I[1];
      pattrib.val(_2, parid) = I[2];
    }

    if (retrieve_cauchy_invariants) {
      vec9 C;
      C.set(0.f);
      vec3 I;
      I.set(0.f);
      
      // Retrieve Cauchy Stress tensor C for deformation gradient F
      compute_cauchy_fixedcorotated(pbuffer.volume, pbuffer.mu, pbuffer.lambda, F, C);

      /// Cauchy Stress tensor invariants
      I[0] = C[0] + C[4] + C[8];    //< I1 = tr(C)
      I[1] = C[0]*C[4] + C[4]*C[8] + 
             C[0]*C[8] - C[3]*C[3] - 
             C[6]*C[6] - C[7]*C[7]; //< I2 = 1/2((tr(C))^2 - tr(C^2))
      I[2] = C[0]*C[4]*C[8] - C[0]*C[7]*C[7] - 
             C[4]*C[6]*C[6] - C[8]*C[3]*C[3] + 
             2*C[3]*C[6]*C[7];      //< I3 = ||C||

      /// Send attributes (I1, I2, I3) to pattribs (device --> device)
      pattrib.val(_0, parid) = I[0];
      pattrib.val(_1, parid) = I[1];
      pattrib.val(_2, parid) = I[2];
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
    auto length = g_length;
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
    
    /// Send positions (x,y,z) (m) to parray (device --> device)
    parray.val(_0, parid) = source_bin.val(_0, _source_pidib) * g_length;
    parray.val(_1, parid) = source_bin.val(_1, _source_pidib) * g_length;
    parray.val(_2, parid) = source_bin.val(_2, _source_pidib) * g_length;

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
  auto length = g_length;
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
    
    /// Send positions (x,y,z) (m) to parray (device --> device)
    parray.val(_0, parid) = source_bin.val(_0, _source_pidib) * g_length;
    parray.val(_1, parid) = source_bin.val(_1, _source_pidib) * g_length;
    parray.val(_2, parid) = source_bin.val(_2, _source_pidib) * g_length;

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

/// Retrieve selected down-sampled grid values from grid buffer to grid array (JB)
template <typename Partition, typename Grid, typename GridArray>
__global__ void retrieve_selected_grid_blocks(
    const ivec3 *__restrict__ prev_blockids, const Partition partition,
    const int *__restrict__ _marks, Grid prev_grid, GridArray garray) {
  auto length = prev_grid.length;
  auto blockid = prev_blockids[blockIdx.x];
  if (_marks[blockIdx.x]) {
    auto blockno = partition.query(blockid);
    if (blockno == -1)
      return;
    auto sourceblock = prev_grid.ch(_0, blockIdx.x);
    auto node_id = blockIdx.x;
  
    /// Set block index via leader thread (e.g. 0)
    if (threadIdx.x == 0){
      garray.val(_0, node_id) = blockid[0];
      garray.val(_1, node_id) = blockid[1];
      garray.val(_2, node_id) = blockid[2];
      garray.val(_3, node_id) = 0.f;
      garray.val(_4, node_id) = 0.f;
      garray.val(_5, node_id) = 0.f;
      garray.val(_6, node_id) = 0.f;
    }

    /// Synch threads in block
    __syncthreads();

    /// Create temp values in threads for specific cells
    auto m   = sourceblock.val_1d(_0, threadIdx.x);            //< Mass (kg)
    auto mvx = sourceblock.val_1d(_1, threadIdx.x) * length; //< mvx (kg m/s)
    auto mvy = sourceblock.val_1d(_2, threadIdx.x) * length; //< mvx (kg m/s)
    auto mvz = sourceblock.val_1d(_3, threadIdx.x) * length; //< mvx (kg m/s)

    /// Ensure temp values are populated in all threads
    __syncthreads();

    /// Atomically add thread (cell) values to block (grid-block) value
    atomicAdd_block(&garray.val(_3, node_id), m);
    atomicAdd_block(&garray.val(_4, node_id), mvx);
    atomicAdd_block(&garray.val(_5, node_id), mvy);
    atomicAdd_block(&garray.val(_6, node_id), mvz);
  }
}


} // namespace mn

#endif
