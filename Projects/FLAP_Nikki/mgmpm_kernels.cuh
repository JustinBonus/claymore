#ifndef __MULTI_GMPM_KERNELS_CUH_
#define __MULTI_GMPM_KERNELS_CUH_

#include "boundary_condition.cuh"
#include "constitutive_models.cuh"
#include "particle_buffer.cuh"
#include "fem_buffer.cuh"
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
template <typename ParticleArray, typename Partition>
__global__ void build_particle_cell_buckets(uint32_t particleCount,
                                            ParticleArray parray,
                                            Partition partition) {
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
  auto pidic = atomicAdd(partition._ppcs + blockno * g_blockvolume + cellno, 1);
  partition._cellbuckets[blockno * g_particle_num_per_block +
                         cellno * g_max_ppc + pidic] = parid;
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
    gridblock.val_1d(_4, cidib) = 0.f;
    gridblock.val_1d(_5, cidib) = 0.f;
    gridblock.val_1d(_6, cidib) = 0.f;
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
                          float mass, vec3 vel0) {
  uint32_t parid = blockIdx.x * blockDim.x + threadIdx.x;
  if (parid >= particleCount)
    return;

  vec3 local_pos{parray.val(_0, parid), parray.val(_1, parid),
                 parray.val(_2, parid)};
  vec3 vel;
  vec9 contrib, C;
  vel.set(0.f), contrib.set(0.f), C.set(0.f);

  vel[0] = vel0[0]; // 58 cm/s
  vel[1] = vel0[1]; // 58 cm/s
  vel[2] = vel0[2]; // 58 cm/s

  // Dp^n = Dp^n+1 = (1/4) * dx^2 * I (Quad.)
  float Dp_inv; //< Inverse Intertia-Like Tensor (1/m^2)
  float scale = g_length * g_length; //< Area scale (m^2)
  Dp_inv = g_D_inv / scale; //< Scalar 4/(dx^2) for Quad. B-Spline
  contrib = (C * mass - contrib * dt) * Dp_inv;
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
        // ASFLIP velocity unstressed
        atomicAdd(
            &grid_block.val(_4, local_index[0], local_index[1], local_index[2]),
            wm * vel[0] + (contrib[0] * xixp[0] + contrib[3] * xixp[1] +
                           contrib[6] * xixp[2]) *
                              W);
        atomicAdd(
            &grid_block.val(_5, local_index[0], local_index[1], local_index[2]),
            wm * vel[1] + (contrib[1] * xixp[0] + contrib[4] * xixp[1] +
                           contrib[7] * xixp[2]) *
                              W);
        atomicAdd(
            &grid_block.val(_6, local_index[0], local_index[1], local_index[2]),
            wm * vel[2] + (contrib[2] * xixp[0] + contrib[5] * xixp[1] +
                           contrib[8] * xixp[2]) *
                              W);
      }
}

template <typename VerticeArray, typename ElementArray>
__global__ void fem_precompute(VerticeArray vertice_array,
                               ElementArray element_array,
                               ElementBuffer<fem_e::Tetrahedron> elementBins) {
    if (blockIdx.x >= g_max_fem_element_num) return;
    auto element = elementBins.ch(_0, blockIdx.x);
    int IDs[4];
    vec3 p[4];
    vec9 B;
    vec9 Binv;
    IDs[0] = element_array.val(_0, blockIdx.x);
    IDs[1] = element_array.val(_1, blockIdx.x);
    IDs[2] = element_array.val(_2, blockIdx.x);
    IDs[3] = element_array.val(_3, blockIdx.x);

    for (int v = 0; v < 4; v++) {
      int ID = IDs[v] - 1;
      p[v][0] = vertice_array.val(_0, ID) * g_length;
      p[v][1] = vertice_array.val(_1, ID) * g_length;
      p[v][2] = vertice_array.val(_2, ID) * g_length;
    }

    Binv.set(0.f);
    Binv[0] = p[1][0] - p[0][0];
    Binv[1] = p[1][1] - p[0][1];
    Binv[2] = p[1][2] - p[0][2];
    Binv[3] = p[2][0] - p[0][0];
    Binv[4] = p[2][1] - p[0][1];
    Binv[5] = p[2][2] - p[0][2];
    Binv[6] = p[3][0] - p[0][0];
    Binv[7] = p[3][1] - p[0][1];
    Binv[8] = p[3][2] - p[0][2];
    
    //float invCheck = matrixDeterminant3d(Binv.data());
    float restVolume = (matrixDeterminant3d(Binv.data())) / 6.f;
    
    B.set(0.f);
    matrixInverse(Binv.data(), B.data());

    {
      element.val(_0, 0) = IDs[0];
      element.val(_1, 0) = IDs[1];
      element.val(_2, 0) = IDs[2];
      element.val(_3, 0) = IDs[3];
      element.val(_4, 0) = B[0];
      element.val(_5, 0) = B[1];
      element.val(_6, 0) = B[2];
      element.val(_7, 0) = B[3];
      element.val(_8, 0) = B[4];
      element.val(_9, 0) = B[5];
      element.val(_10, 0) = B[6];
      element.val(_11, 0) = B[7];
      element.val(_12, 0) = B[8];
      element.val(_13, 0) = restVolume;
    }

}


template <typename VerticeArray, typename ElementArray>
__global__ void fem_precompute(VerticeArray vertice_array,
                               ElementArray element_array,
                               ElementBuffer<fem_e::Brick> elementBins) {
                                 return;
}

template <typename ParticleArray, typename Partition>
__global__ void array_to_buffer(ParticleArray parray,
                                ParticleBuffer<material_e::JFluid> pbuffer,
                                Partition partition, vec3 vel) {
  uint32_t blockno = blockIdx.x;
  int pcnt = partition._ppbs[blockno];
  auto bucket = partition._blockbuckets + blockno * g_particle_num_per_block;
  for (int pidib = threadIdx.x; pidib < pcnt; pidib += blockDim.x) {
    auto parid = bucket[pidib];
    auto pbin =
        pbuffer.ch(_0, partition._binsts[blockno] + pidib / g_bin_capacity);
    /// pos
    pbin.val(_0, pidib % g_bin_capacity) = parray.val(_0, parid);
    pbin.val(_1, pidib % g_bin_capacity) = parray.val(_1, parid);
    pbin.val(_2, pidib % g_bin_capacity) = parray.val(_2, parid);
    /// J
    pbin.val(_3, pidib % g_bin_capacity) = 1.f;
  }
}

template <typename ParticleArray, typename Partition>
__global__ void array_to_buffer(ParticleArray parray,
                                ParticleBuffer<material_e::JFluid_ASFLIP> pbuffer,
                                Partition partition, vec3 vel) {
  uint32_t blockno = blockIdx.x;
  int pcnt = partition._ppbs[blockno];
  auto bucket = partition._blockbuckets + blockno * g_particle_num_per_block;
  for (int pidib = threadIdx.x; pidib < pcnt; pidib += blockDim.x) {
    auto parid = bucket[pidib];
    auto pbin =
        pbuffer.ch(_0, partition._binsts[blockno] + pidib / g_bin_capacity);
    /// pos
    pbin.val(_0, pidib % g_bin_capacity) = parray.val(_0, parid);
    pbin.val(_1, pidib % g_bin_capacity) = parray.val(_1, parid);
    pbin.val(_2, pidib % g_bin_capacity) = parray.val(_2, parid);
    /// J
    pbin.val(_3, pidib % g_bin_capacity) = 1.f;
    /// vel
    pbin.val(_4, pidib % g_bin_capacity) = vel[0]; //< Vel_x m/s
    pbin.val(_5, pidib % g_bin_capacity) = vel[1]; //< Vel_y m/s
    pbin.val(_6, pidib % g_bin_capacity) = vel[2]; //< Vel_z m/s

  }
}

template <typename ParticleArray, typename Partition>
__global__ void
array_to_buffer(ParticleArray parray,
                ParticleBuffer<material_e::FixedCorotated> pbuffer,
                Partition partition, vec3 vel) {
  uint32_t blockno = blockIdx.x;
  int pcnt = partition._ppbs[blockno];
  auto bucket = partition._blockbuckets + blockno * g_particle_num_per_block;
  for (int pidib = threadIdx.x; pidib < pcnt; pidib += blockDim.x) {
    auto parid = bucket[pidib];
    auto pbin =
        pbuffer.ch(_0, partition._binsts[blockno] + pidib / g_bin_capacity);
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

template <typename ParticleArray, typename Partition>
__global__ void
array_to_buffer(ParticleArray parray,
                ParticleBuffer<material_e::FixedCorotated_ASFLIP> pbuffer,
                Partition partition, vec3 vel) {
  uint32_t blockno = blockIdx.x;
  int pcnt = partition._ppbs[blockno];
  auto bucket = partition._blockbuckets + blockno * g_particle_num_per_block;
  for (int pidib = threadIdx.x; pidib < pcnt; pidib += blockDim.x) {
    auto parid = bucket[pidib];
    auto pbin =
        pbuffer.ch(_0, partition._binsts[blockno] + pidib / g_bin_capacity);
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
    /// vel
    pbin.val(_12, pidib % g_bin_capacity) = vel[0]; //< Vel_x m/s
    pbin.val(_13, pidib % g_bin_capacity) = vel[1]; //< Vel_y m/s
    pbin.val(_14, pidib % g_bin_capacity) = vel[2]; //< Vel_z m/s
  }
}

template <typename ParticleArray, typename Partition>
__global__ void array_to_buffer(ParticleArray parray,
                                ParticleBuffer<material_e::Sand> pbuffer,
                                Partition partition, vec3 vel) {
  uint32_t blockno = blockIdx.x;
  int pcnt = partition._ppbs[blockno];
  auto bucket = partition._blockbuckets + blockno * g_particle_num_per_block;
  for (int pidib = threadIdx.x; pidib < pcnt; pidib += blockDim.x) {
    auto parid = bucket[pidib];
    auto pbin =
        pbuffer.ch(_0, partition._binsts[blockno] + pidib / g_bin_capacity);
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

template <typename ParticleArray, typename Partition>
__global__ void array_to_buffer(ParticleArray parray,
                                ParticleBuffer<material_e::NACC> pbuffer,
                                Partition partition, vec3 vel) {
  uint32_t blockno = blockIdx.x;
  int pcnt = partition._ppbs[blockno];
  auto bucket = partition._blockbuckets + blockno * g_particle_num_per_block;
  for (int pidib = threadIdx.x; pidib < pcnt; pidib += blockDim.x) {
    auto parid = bucket[pidib];
    auto pbin =
        pbuffer.ch(_0, partition._binsts[blockno] + pidib / g_bin_capacity);
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

template <typename ParticleArray, typename Partition>
__global__ void array_to_buffer(ParticleArray parray,
                                ParticleBuffer<material_e::Meshed> pbuffer,
                                Partition partition, vec3 vel) {
  uint32_t blockno = blockIdx.x;
  int pcnt = partition._ppbs[blockno];
  auto bucket = partition._blockbuckets + blockno * g_particle_num_per_block;
  for (int pidib = threadIdx.x; pidib < pcnt; pidib += blockDim.x) {
    auto parid = bucket[pidib];
    if (parid >= g_max_fem_vertice_num) {
      printf("Particle %d incorrect ID!\n", parid);
      break;
    }
    auto pbin =
        pbuffer.ch(_0, partition._binsts[blockno] + pidib / g_bin_capacity);
    /// pos
    pbin.val(_0, pidib % g_bin_capacity) = parray.val(_0, parid);
    pbin.val(_1, pidib % g_bin_capacity) = parray.val(_1, parid);
    pbin.val(_2, pidib % g_bin_capacity) = parray.val(_2, parid);
    /// ID
    pbin.val(_3, pidib % g_bin_capacity) = parid;
    /// vel
    pbin.val(_4, pidib % g_bin_capacity) = vel[0]; //< Vel_x m/s
    pbin.val(_5, pidib % g_bin_capacity) = vel[1]; //< Vel_y m/s
    pbin.val(_6, pidib % g_bin_capacity) = vel[2]; //< Vel_z m/s
    pbin.val(_7, pidib % g_bin_capacity) = 0.f; //< J
    pbin.val(_8, pidib % g_bin_capacity)  = 0.f; //< b_x   
    pbin.val(_9, pidib % g_bin_capacity)  = 0.f; //< b_y
    pbin.val(_10, pidib % g_bin_capacity) = 0.f; //< b_z
  }
}


template <typename Grid, typename Partition>
__global__ void update_grid_velocity_query_max(uint32_t blockCount, Grid grid,
                                               Partition partition, float dt,
                                               float *maxVel, float curTime) {
  constexpr int bc = g_bc;
  constexpr int numWarps =
      g_num_grid_blocks_per_cuda_block * g_num_warps_per_grid_block;
  constexpr unsigned activeMask = 0xffffffff;
  //__shared__ float sh_maxvels[g_blockvolume * g_num_grid_blocks_per_cuda_block
  /// 32];
  extern __shared__ float sh_maxvels[];
  std::size_t blockno = blockIdx.x * g_num_grid_blocks_per_cuda_block +
                        threadIdx.x / 32 / g_num_warps_per_grid_block;
  auto blockid = partition._activeKeys[blockno];
  int isInBound = ((blockid[0] < bc || blockid[0] >= g_grid_size_x - bc) << 2) |
                  ((blockid[1] < bc || blockid[1] >= g_grid_size_y - bc) << 1) |
                   (blockid[2] < bc || blockid[2] >= g_grid_size_z - bc);

  if (threadIdx.x < numWarps)
    sh_maxvels[threadIdx.x] = 0.0f;
  __syncthreads();

  /// within-warp computations
  if (blockno < blockCount) {
    auto grid_block = grid.ch(_0, blockno);
    for (int cidib = threadIdx.x % 32; cidib < g_blockvolume; cidib += 32) {
      float mass = grid_block.val_1d(_0, cidib), velSqr = 0.f, vel[3], vel_n[3];
      if (mass > 0.f) {
        mass = 1.f / mass;

        // Grid node coordinate [i,j,k] in grid-block
        int i = (cidib >> (g_blockbits << 1)) & g_blockmask;
        int j = (cidib >> g_blockbits) & g_blockmask;
        int k = cidib & g_blockmask;
        // Grid node position [x,y,z] in entire domain
        float xc = (4*blockid[0]*g_dx) + (i*g_dx); // + (g_dx/2.f);
        float yc = (4*blockid[1]*g_dx) + (j*g_dx); // + (g_dx/2.f);
        float zc = (4*blockid[2]*g_dx) + (k*g_dx); // + (g_dx/2.f);

        // Offset condition for Off-by-2 (see Xinlei & Fang et al.)
        // Note, subtract 16 nodes from total
        // (4 grid blocks) to have available scene length
        float offset = (8.f*g_dx);

        // Retrieve grid momentums (kg*m/s2)
        vel[0] = grid_block.val_1d(_1, cidib); //< mvx
        vel[1] = grid_block.val_1d(_2, cidib); //< mvy
        vel[2] = grid_block.val_1d(_3, cidib); //< mvz
        vel_n[0] = grid_block.val_1d(_4, cidib); //< mvx
        vel_n[1] = grid_block.val_1d(_5, cidib); //< mvy
        vel_n[2] = grid_block.val_1d(_6, cidib); //< mvz

        isInBound = (((blockid[0] < bc && vel[0] < 0.f) || (blockid[0] >= g_grid_size_x - bc && vel[0] > 0.f)) << 2) |
                    (((blockid[1] < bc && vel[1] < 0.f) || (blockid[1] >= g_grid_size_y - bc && vel[1] > 0.f)) << 1) |
                   ((blockid[2] < bc && vel[2] < 0.f) || (blockid[2] >= g_grid_size_z - bc  && vel[2] > 0.f));

        // Slip boundaries of simulation
        // Acts on individual grid-cell velocities
        float flumex = 0.584f / g_length; // Length
        float flumey = 0.372f / g_length; // Depth
        float flumez = 0.008f / g_length; // Width
        int isInFlume =  (((xc <= offset && vel[0] < 0.f) || (xc >= flumex + offset && vel[0] > 0.f)) << 2) |
                         (((yc <= offset && vel[1] < 0.f) || (yc >= flumey + offset && vel[1] > 0.f)) << 1) |
                          ((zc <= offset && vel[2] < 0.f) || (zc >= flumez + offset && vel[2] > 0.f));                          
        isInBound |= isInFlume; // Update with regular boundary for efficiency

#if 1
        ///< Slip contact        
        // Set cell velocity after grid-block/cell boundary check
        vel[0] = isInBound & 4 ? 0.f : vel[0] * mass; //< vx = mvx / m
        vel[1] = isInBound & 2 ? 0.f : vel[1] * mass; //< vy = mvy / m
        vel[1] += isInBound & 2 ? 0.f : (g_gravity / g_length) * dt;  //< Grav. effect
        vel[2] = isInBound & 1 ? 0.f : vel[2] * mass; //< vz = mvz / m
        vel_n[0] = isInBound & 4 ? 0.f : vel_n[0] * mass; //< vx = mvx / m
        vel_n[1] = isInBound & 2 ? 0.f : vel_n[1] * mass; //< vy = mvy / m
        //vel_n[1] += isInBound & 2 ? 0.f : (g_gravity / g_length) * dt;  //< Grav. effect
        vel_n[2] = isInBound & 1 ? 0.f : vel_n[2] * mass; //< vz = mvz / m
#endif        

#if 0
        ///< Sticky contact
        if (isInBound) ///< sticky
          vel.set(0.f);
#endif

        float o = offset;
        float l = g_length;
        float bt = 0.004f; // Beam-fix, meters
        if ((xc >= 0.288f / l + o  && yc <= bt / l + o) || (xc <= 0.300f / l + o  && yc <= bt / l + o)){
          vel[0] = 0.f;
          vel[1] = 0.f;
          vel[2] = 0.f;
        } else if ((xc < 0.288f / l + o && yc <= bt / l + o) && vel[1] < 0.f){
          vel[1] = 0.f;
        } else if ((xc > 0.300f / l + o && yc <= bt / l + o) && vel[1] < 0.f){
          vel[1] = 0.f;
        }


        grid_block.val_1d(_1, cidib) = vel[0];
        velSqr += vel[0] * vel[0];
        grid_block.val_1d(_2, cidib) = vel[1];
        velSqr += vel[1] * vel[1];
        grid_block.val_1d(_3, cidib) = vel[2];
        velSqr += vel[2] * vel[2];
        grid_block.val_1d(_4, cidib) = vel_n[0];
        grid_block.val_1d(_5, cidib) = vel_n[1];
        grid_block.val_1d(_6, cidib) = vel_n[2];
      }
      // unsigned activeMask = __ballot_sync(0xffffffff, mv[0] != 0.0f);
      for (int iter = 1; iter % 32; iter <<= 1) {
        float tmp = __shfl_down_sync(activeMask, velSqr, iter, 32);
        if ((threadIdx.x % 32) + iter < 32)
          velSqr = tmp > velSqr ? tmp : velSqr;
      }
      if (velSqr > sh_maxvels[threadIdx.x / 32] && (threadIdx.x % 32) == 0)
        sh_maxvels[threadIdx.x / 32] = velSqr;
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
    atomicMax(maxVel, sh_maxvels[0]);
}

template <typename Grid, typename Partition, typename Boundary>
__global__ void update_grid_velocity_query_max(uint32_t blockCount, Grid grid,
                                               Partition partition, float dt,
                                               Boundary boundary,
                                               float *maxVel, float curTime) {
  constexpr int bc = 2;
  constexpr int numWarps =
      g_num_grid_blocks_per_cuda_block * g_num_warps_per_grid_block;
  constexpr unsigned activeMask = 0xffffffff;
  //__shared__ float sh_maxvels[g_blockvolume * g_num_grid_blocks_per_cuda_block
  /// 32];
  extern __shared__ float sh_maxvels[];
  std::size_t blockno = blockIdx.x * g_num_grid_blocks_per_cuda_block +
                        threadIdx.x / 32 / g_num_warps_per_grid_block;
  auto blockid = partition._activeKeys[blockno];
  int isInBound = ((blockid[0] < bc || blockid[0] >= g_grid_size_x - bc) << 2) |
                  ((blockid[1] < bc || blockid[1] >= g_grid_size_y - bc) << 1) |
                  (blockid[2] < bc || blockid[2] >= g_grid_size_z - bc);
  if (threadIdx.x < numWarps)
    sh_maxvels[threadIdx.x] = 0.0f;
  __syncthreads();

  /// within-warp computations
  if (blockno < blockCount) {
    auto grid_block = grid.ch(_0, blockno);
    for (int cidib = threadIdx.x % 32; cidib < g_blockvolume; cidib += 32) {
      float mass = grid_block.val_1d(_0, cidib), velSqr = 0.f;
      vec3 vel;
      if (mass > 0.f) {
        mass = 1.f / mass;

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


        // Tank Dimensions
        // See Wen-Chia Yang Dissertation, Page 82 Figure 5.1
        // Acts on individual grid-cell velocities
        float flumex = 3.2f / g_length; // Actually 12m, added run-in/out
        float flumey = 6.4f / g_length; // 1.22m Depth
        float flumez = 0.4f / g_length; // 0.91m Width
        int isInFlume =  ((xc < offset || xc >= flumex + offset) << 2) |
                         ((yc < offset || yc >= flumey + offset) << 1) |
                          (zc < offset || zc >= flumez + offset);
        isInBound |= isInFlume; // Update with regular boundary for efficiency

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


        ivec3 cellid{(cidib & 0x30) >> 4, (cidib & 0xc) >> 2, cidib & 0x3};
        boundary.detect_and_resolve_collision(blockid, cellid, 0.f, vel);
        velSqr = vel.dot(vel);
        grid_block.val_1d(_1, cidib) = vel[0];
        grid_block.val_1d(_2, cidib) = vel[1];
        grid_block.val_1d(_3, cidib) = vel[2];
      }
      // unsigned activeMask = __ballot_sync(0xffffffff, mv[0] != 0.0f);
      for (int iter = 1; iter % 32; iter <<= 1) {
        float tmp = __shfl_down_sync(activeMask, velSqr, iter, 32);
        if ((threadIdx.x % 32) + iter < 32)
          velSqr = tmp > velSqr ? tmp : velSqr;
      }
      if (velSqr > sh_maxvels[threadIdx.x / 32] && (threadIdx.x % 32) == 0)
        sh_maxvels[threadIdx.x / 32] = velSqr;
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
    atomicMax(maxVel, sh_maxvels[0]);
}

template <typename Partition, typename Grid, typename VerticeArray>
__global__ void g2p2g(float dt, float newDt, const ivec3 *__restrict__ blocks,
                      const ParticleBuffer<material_e::JFluid> pbuffer,
                      ParticleBuffer<material_e::JFluid> next_pbuffer,
                      const Partition prev_partition, Partition partition,
                      const Grid grid, Grid next_grid,
                      VerticeArray vertice_array) {
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

  ivec3 blockid;
  int src_blockno;
  if (blocks != nullptr) {
    blockid = blocks[blockIdx.x];
    src_blockno = partition.query(blockid);
  } else {
    if (partition._haloMarks[blockIdx.x])
      return;
    blockid = partition._activeKeys[blockIdx.x];
    src_blockno = blockIdx.x;
  }
  // auto blockid = partition._activeKeys[blockIdx.x];

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

  for (int pidib = threadIdx.x; pidib < partition._ppbs[src_blockno];
       pidib += blockDim.x) {
    int source_blockno, source_pidib;
    ivec3 base_index;
    {
      int advect =
          partition
              ._blockbuckets[src_blockno * g_particle_num_per_block + pidib];
      dir_components(advect / g_particle_num_per_block, base_index);
      base_index += blockid;
      source_blockno = prev_partition.query(base_index);
      source_pidib = advect & (g_particle_num_per_block - 1);
      source_blockno = prev_partition._binsts[source_blockno] +
                       source_pidib / g_bin_capacity;
    }
    vec3 pos;
    float J;
    {
      auto source_particle_bin = pbuffer.ch(_0, source_blockno);
      pos[0] = source_particle_bin.val(_0, source_pidib % g_bin_capacity);
      pos[1] = source_particle_bin.val(_1, source_pidib % g_bin_capacity);
      pos[2] = source_particle_bin.val(_2, source_pidib % g_bin_capacity);
      J = source_particle_bin.val(_3, source_pidib % g_bin_capacity);
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
    float Dp_inv; //< Inverse Intertia-Like Tensor (1/m^2)
    float scale = g_length * g_length; //< Area scale (m^2)
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

    J = (1 + (C[0] + C[4] + C[8]) * dt * Dp_inv) * J;
    if (J > 1.0f)
      J = 1.0f;
    else if (J < 0.1f)
      J = 0.1;
    vec9 contrib;
    {
      float voln = J * pbuffer.volume;
      float pressure = (pbuffer.bulk / pbuffer.gamma) * (powf(J, -pbuffer.gamma) - 1.f);
      {
        contrib[0] =
            ((C[0] + C[0]) * Dp_inv * pbuffer.visco - pressure) * voln;
        contrib[1] = (C[1] + C[3]) * Dp_inv * pbuffer.visco * voln;
        contrib[2] = (C[2] + C[6]) * Dp_inv * pbuffer.visco * voln;

        contrib[3] = (C[3] + C[1]) * Dp_inv * pbuffer.visco * voln;
        contrib[4] =
            ((C[4] + C[4]) * Dp_inv * pbuffer.visco - pressure) * voln;
        contrib[5] = (C[5] + C[7]) * Dp_inv * pbuffer.visco * voln;

        contrib[6] = (C[6] + C[2]) * Dp_inv * pbuffer.visco * voln;
        contrib[7] = (C[7] + C[5]) * Dp_inv * pbuffer.visco * voln;
        contrib[8] =
            ((C[8] + C[8]) * Dp_inv * pbuffer.visco - pressure) * voln;
      }
      contrib = (C * pbuffer.mass - contrib * newDt) * Dp_inv;
      {
        auto particle_bin = next_pbuffer.ch(_0, partition._binsts[src_blockno] +
                                                    pidib / g_bin_capacity);
        particle_bin.val(_0, pidib % g_bin_capacity) = pos[0];
        particle_bin.val(_1, pidib % g_bin_capacity) = pos[1];
        particle_bin.val(_2, pidib % g_bin_capacity) = pos[2];
        particle_bin.val(_3, pidib % g_bin_capacity) = J;
      }
    }

    local_base_index = (pos * g_dx_inv + 0.5f).cast<int>() - 1;
    {
      int dirtag = dir_offset((base_index - 1) / g_blocksize -
                              (local_base_index - 1) / g_blocksize);
      partition.add_advection(local_base_index - 1, dirtag, pidib);
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

// Grid-to-Particle-to-Grid - Weakly-Incompressible Fluid - ASFLIP Transfer
template <typename Partition, typename Grid, typename VerticeArray>
__global__ void g2p2g(float dt, float newDt, const ivec3 *__restrict__ blocks,
                      const ParticleBuffer<material_e::JFluid_ASFLIP> pbuffer,
                      ParticleBuffer<material_e::JFluid_ASFLIP> next_pbuffer,
                      const Partition prev_partition, Partition partition,
                      const Grid grid, Grid next_grid,
                      VerticeArray vertice_array) {
  static constexpr uint64_t numViPerBlock = g_blockvolume * 3;
  static constexpr uint64_t numViInArena = numViPerBlock << 3;
  static constexpr uint64_t numViInArena_ASFLIP = (g_blockvolume * 6) << 3;

  static constexpr uint64_t numMViPerBlock = g_blockvolume * 4;
  static constexpr uint64_t numMViInArena = numMViPerBlock << 3;

  static constexpr unsigned arenamask = (g_blocksize << 1) - 1;
  static constexpr unsigned arenabits = g_blockbits + 1;

  extern __shared__ char shmem[];
  using ViArena =
      float(*)[6][g_blocksize << 1][g_blocksize << 1][g_blocksize << 1];
  using ViArenaRef =
      float(&)[6][g_blocksize << 1][g_blocksize << 1][g_blocksize << 1];
  ViArenaRef __restrict__ g2pbuffer = *reinterpret_cast<ViArena>(shmem);
  using MViArena =
      float(*)[7][g_blocksize << 1][g_blocksize << 1][g_blocksize << 1];
  using MViArenaRef =
      float(&)[7][g_blocksize << 1][g_blocksize << 1][g_blocksize << 1];
  MViArenaRef __restrict__ p2gbuffer =
      *reinterpret_cast<MViArena>(shmem + numViInArena_ASFLIP * sizeof(float));

  ivec3 blockid;
  int src_blockno;
  if (blocks != nullptr) {
    blockid = blocks[blockIdx.x];
    src_blockno = partition.query(blockid);
  } else {
    if (partition._haloMarks[blockIdx.x])
      return;
    blockid = partition._activeKeys[blockIdx.x];
    src_blockno = blockIdx.x;
  }
  // auto blockid = partition._activeKeys[blockIdx.x];

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
    else if (channelid == 2) 
      val = grid_block.val_1d(_3, c);
    g2pbuffer[channelid][cx + (local_block_id & 4 ? g_blocksize : 0)]
             [cy + (local_block_id & 2 ? g_blocksize : 0)]
             [cz + (local_block_id & 1 ? g_blocksize : 0)] = val;
  }
  __syncthreads();
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
    channelid += 3;
    float val;
    if (channelid == 3) {
      val = grid_block.val_1d(_4, c);
    } else if (channelid == 4) {
      val = grid_block.val_1d(_5, c);
    } else if (channelid == 5) {
      val = grid_block.val_1d(_6, c);
    }
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
  for (int base = threadIdx.x; base < numMViInArena; base += blockDim.x) {
    int loc = base;
    char z = loc & arenamask;
    char y = (loc >>= arenabits) & arenamask;
    char x = (loc >>= arenabits) & arenamask;
    int channelid = loc >> arenabits;
    if (channelid != 0) {
      channelid += 3;
      p2gbuffer[channelid][x][y][z] = 0.f;
    }
  }
  __syncthreads();
  for (int pidib = threadIdx.x; pidib < partition._ppbs[src_blockno];
       pidib += blockDim.x) {
    int source_blockno, source_pidib;
    ivec3 base_index;
    {
      int advect =
          partition
              ._blockbuckets[src_blockno * g_particle_num_per_block + pidib];
      dir_components(advect / g_particle_num_per_block, base_index);
      base_index += blockid;
      source_blockno = prev_partition.query(base_index);
      source_pidib = advect & (g_particle_num_per_block - 1);
      source_blockno = prev_partition._binsts[source_blockno] +
                       source_pidib / g_bin_capacity;
    }
    vec3 pos;  //< Particle position at n
    vec3 vp_n; //< Particle vel. at n
    float J;   //< Particle volume ratio at n
    {
      auto source_particle_bin = pbuffer.ch(_0, source_blockno);
      pos[0] = source_particle_bin.val(_0, source_pidib % g_bin_capacity);  //< x
      pos[1] = source_particle_bin.val(_1, source_pidib % g_bin_capacity);  //< y
      pos[2] = source_particle_bin.val(_2, source_pidib % g_bin_capacity);  //< z
      J = source_particle_bin.val(_3, source_pidib % g_bin_capacity);       //< Vo/V
      vp_n[0] = source_particle_bin.val(_4, source_pidib % g_bin_capacity); //< vx
      vp_n[1] = source_particle_bin.val(_5, source_pidib % g_bin_capacity); //< vy
      vp_n[2] = source_particle_bin.val(_6, source_pidib % g_bin_capacity); //< vz
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
    vec3 vel;   //< Stressed, collided grid velocity
    vec3 vel_n; //< Unstressed, uncollided grid velocity
    vec9 C;     //< APIC affine matrix, used a few times
    vel.set(0.f); 
    vel_n.set(0.f);
    C.set(0.f);

    // Dp^n = Dp^n+1 = (1/4) * dx^2 * I (Quad.)
    float Dp_inv; //< Inverse Intertia-Like Tensor (1/m^2)
    float scale = g_length * g_length; //< Area scale (m^2)
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
#pragma unroll 3
    for (char i = 0; i < 3; i++)
#pragma unroll 3
      for (char j = 0; j < 3; j++)
#pragma unroll 3
        for (char k = 0; k < 3; k++) {
          vec3 xixp = vec3{(float)i, (float)j, (float)k} * g_dx - local_pos;
          float W = dws(0, i) * dws(1, j) * dws(2, k);
          vec3 vi_n{g2pbuffer[3][local_base_index[0] + i][local_base_index[1] + j]
                            [local_base_index[2] + k],
                    g2pbuffer[4][local_base_index[0] + i][local_base_index[1] + j]
                            [local_base_index[2] + k],
                    g2pbuffer[5][local_base_index[0] + i][local_base_index[1] + j]
                            [local_base_index[2] + k]};
          vel_n += vi_n * W; 
        }
    J = (1 + (C[0] + C[4] + C[8]) * dt * Dp_inv) * J;

    float beta; //< Position correction factor (ASFLIP)
    float Jc = 1.f; // Critical J for weak-comp fluid
    if (J >= Jc) {
      J = Jc;       // No vol. expansion, Tamp. 2017
      beta = pbuffer.beta_max;  // beta max
    } else {
      beta = pbuffer.beta_min; // beta min
    }

    pos += dt * (vel + beta * pbuffer.alpha * (vp_n - vel_n));
    vel += pbuffer.alpha * (vp_n - vel_n);

    vec9 contrib;
    {
      float voln = J * pbuffer.volume;
      float pressure = (pbuffer.bulk / pbuffer.gamma) * (powf(J, -pbuffer.gamma) - 1.f);
      {
        contrib[0] =
            ((C[0] + C[0]) * Dp_inv * pbuffer.visco - pressure) * voln;
        contrib[1] = (C[1] + C[3]) * Dp_inv * pbuffer.visco * voln;
        contrib[2] = (C[2] + C[6]) * Dp_inv * pbuffer.visco * voln;

        contrib[3] = (C[3] + C[1]) * Dp_inv * pbuffer.visco * voln;
        contrib[4] =
            ((C[4] + C[4]) * Dp_inv * pbuffer.visco - pressure) * voln;
        contrib[5] = (C[5] + C[7]) * Dp_inv * pbuffer.visco * voln;

        contrib[6] = (C[6] + C[2]) * Dp_inv * pbuffer.visco * voln;
        contrib[7] = (C[7] + C[5]) * Dp_inv * pbuffer.visco * voln;
        contrib[8] =
            ((C[8] + C[8]) * Dp_inv * pbuffer.visco - pressure) * voln;
      }
      // Merged affine matrix and stress contribution for MLS-MPM P2G
      contrib = (C * pbuffer.mass - contrib * newDt) * Dp_inv;
      {
        auto particle_bin = next_pbuffer.ch(_0, partition._binsts[src_blockno] +
                                                    pidib / g_bin_capacity);
        particle_bin.val(_0, pidib % g_bin_capacity) = pos[0]; //< x
        particle_bin.val(_1, pidib % g_bin_capacity) = pos[1]; //< y
        particle_bin.val(_2, pidib % g_bin_capacity) = pos[2]; //< z
        particle_bin.val(_3, pidib % g_bin_capacity) = J;      //< V/Vo
        particle_bin.val(_4, pidib % g_bin_capacity) = vel[0]; //< vx
        particle_bin.val(_5, pidib % g_bin_capacity) = vel[1]; //< vy
        particle_bin.val(_6, pidib % g_bin_capacity) = vel[2]; //< vz
      }
    }

    local_base_index = (pos * g_dx_inv + 0.5f).cast<int>() - 1;
    {
      int dirtag = dir_offset((base_index - 1) / g_blocksize -
                              (local_base_index - 1) / g_blocksize);
      partition.add_advection(local_base_index - 1, dirtag, pidib);
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
          // ASFLIP unstressed velocity
          atomicAdd(
              &p2gbuffer[4][local_base_index[0] + i][local_base_index[1] + j]
                        [local_base_index[2] + k],
              wm * (vel[0] + Dp_inv * (C[0] * pos[0] + C[3] * pos[1] +
                             C[6] * pos[2])));
          atomicAdd(
              &p2gbuffer[5][local_base_index[0] + i][local_base_index[1] + j]
                        [local_base_index[2] + k],
              wm * (vel[1] + Dp_inv * (C[1] * pos[0] + C[4] * pos[1] +
                             C[7] * pos[2])));
          atomicAdd(
              &p2gbuffer[6][local_base_index[0] + i][local_base_index[1] + j]
                        [local_base_index[2] + k],
              wm * (vel[2] + Dp_inv * (C[2] * pos[0] + C[5] * pos[1] +
                             C[8] * pos[2])));
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
    if (channelid == 0) {
      atomicAdd(&next_grid.ch(_0, blockno).val_1d(_0, c), val);
    } else if (channelid == 1) {
      atomicAdd(&next_grid.ch(_0, blockno).val_1d(_1, c), val);
    } else if (channelid == 2) {
      atomicAdd(&next_grid.ch(_0, blockno).val_1d(_2, c), val);
    } else if (channelid == 3) {
      atomicAdd(&next_grid.ch(_0, blockno).val_1d(_3, c), val);
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
    int channelid = base & (numMViPerBlock - 1);
    char c = channelid % g_blockvolume;
    char cz = channelid & g_blockmask;
    char cy = (channelid >>= g_blockbits) & g_blockmask;
    char cx = (channelid >>= g_blockbits) & g_blockmask;
    channelid >>= g_blockbits;
    channelid += 3;
    float val =
        p2gbuffer[channelid][cx + (local_block_id & 4 ? g_blocksize : 0)]
                 [cy + (local_block_id & 2 ? g_blocksize : 0)]
                 [cz + (local_block_id & 1 ? g_blocksize : 0)];
    if (channelid == 4) {
      atomicAdd(&next_grid.ch(_0, blockno).val_1d(_4, c), val);
    } else if (channelid == 5) {
      atomicAdd(&next_grid.ch(_0, blockno).val_1d(_5, c), val);
    } else if (channelid == 6) {
      atomicAdd(&next_grid.ch(_0, blockno).val_1d(_6, c), val);
    }
  }
}


template <typename Partition, typename Grid, typename VerticeArray>
__global__ void g2p2g(float dt, float newDt, const ivec3 *__restrict__ blocks,
                      const ParticleBuffer<material_e::FixedCorotated> pbuffer,
                      ParticleBuffer<material_e::FixedCorotated> next_pbuffer,
                      const Partition prev_partition, Partition partition,
                      const Grid grid, Grid next_grid,
                      VerticeArray vertice_array) {
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

  ivec3 blockid;
  int src_blockno;
  if (blocks != nullptr) {
    blockid = blocks[blockIdx.x];
    src_blockno = partition.query(blockid);
  } else {
    if (partition._haloMarks[blockIdx.x])
      return;
    blockid = partition._activeKeys[blockIdx.x];
    src_blockno = blockIdx.x;
  }
  // auto blockid = partition._activeKeys[blockIdx.x];

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

  for (int pidib = threadIdx.x; pidib < partition._ppbs[src_blockno];
       pidib += blockDim.x) {
    int source_blockno, source_pidib;
    ivec3 base_index;
    {
      int advect =
          partition
              ._blockbuckets[src_blockno * g_particle_num_per_block + pidib];
      dir_components(advect / g_particle_num_per_block, base_index);
      base_index += blockid;
      source_blockno = prev_partition.query(base_index);
      source_pidib = advect & (g_particle_num_per_block - 1);
      source_blockno = prev_partition._binsts[source_blockno] +
                       source_pidib / g_bin_capacity;
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
    float Dp_inv; //< Inverse Intertia-Like Tensor (1/m^2)
    float scale = g_length * g_length; //< Area scale (m^2)
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
      dws.val(d) = C[d] * dt * Dp_inv + ((d & 0x3) ? 0.f : 1.f);

    vec9 contrib;
    {
      vec9 F;
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
      matrixMatrixMultiplication3d(dws.data(), contrib.data(), F.data());
      {
        auto particle_bin = next_pbuffer.ch(_0, partition._binsts[src_blockno] +
                                                    pidib / g_bin_capacity);
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
      }
      compute_stress_fixedcorotated(pbuffer.volume, pbuffer.mu, pbuffer.lambda,
                                    F, contrib);
      contrib = (C * pbuffer.mass - contrib * newDt) * Dp_inv;
    }

    local_base_index = (pos * g_dx_inv + 0.5f).cast<int>() - 1;
    {
      int dirtag = dir_offset((base_index - 1) / g_blocksize -
                              (local_base_index - 1) / g_blocksize);
      partition.add_advection(local_base_index - 1, dirtag, pidib);
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

// Grid-to-Particle-to-Grid - Fixed-Corotated - ASFLIP transfer
template <typename Partition, typename Grid, typename VerticeArray>
__global__ void g2p2g(float dt, float newDt, const ivec3 *__restrict__ blocks,
                      const ParticleBuffer<material_e::FixedCorotated_ASFLIP> pbuffer,
                      ParticleBuffer<material_e::FixedCorotated_ASFLIP> next_pbuffer,
                      const Partition prev_partition, Partition partition,
                      const Grid grid, Grid next_grid,
                      VerticeArray vertice_array) {
  static constexpr uint64_t numViPerBlock = g_blockvolume * 3;
  static constexpr uint64_t numViInArena = numViPerBlock << 3;
  static constexpr uint64_t numViInArena_ASFLIP = (g_blockvolume * 6) << 3;

  static constexpr uint64_t numMViPerBlock = g_blockvolume * 4;
  static constexpr uint64_t numMViInArena = numMViPerBlock << 3;

  static constexpr unsigned arenamask = (g_blocksize << 1) - 1;
  static constexpr unsigned arenabits = g_blockbits + 1;

  extern __shared__ char shmem[];
  using ViArena =
      float(*)[6][g_blocksize << 1][g_blocksize << 1][g_blocksize << 1];
  using ViArenaRef =
      float(&)[6][g_blocksize << 1][g_blocksize << 1][g_blocksize << 1];
  ViArenaRef __restrict__ g2pbuffer = *reinterpret_cast<ViArena>(shmem);
  using MViArena =
      float(*)[7][g_blocksize << 1][g_blocksize << 1][g_blocksize << 1];
  using MViArenaRef =
      float(&)[7][g_blocksize << 1][g_blocksize << 1][g_blocksize << 1];
  MViArenaRef __restrict__ p2gbuffer =
      *reinterpret_cast<MViArena>(shmem + numViInArena_ASFLIP * sizeof(float));

  ivec3 blockid;
  int src_blockno;
  if (blocks != nullptr) {
    blockid = blocks[blockIdx.x];
    src_blockno = partition.query(blockid);
  } else {
    if (partition._haloMarks[blockIdx.x])
      return;
    blockid = partition._activeKeys[blockIdx.x];
    src_blockno = blockIdx.x;
  }
  // auto blockid = partition._activeKeys[blockIdx.x];

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
    channelid += 3;
    float val;
    if (channelid == 3)
      val = grid_block.val_1d(_4, c);
    else if (channelid == 4)
      val = grid_block.val_1d(_5, c);
    else if (channelid == 5)
      val = grid_block.val_1d(_6, c);
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
  for (int base = threadIdx.x; base < numMViInArena; base += blockDim.x) {
    int loc = base;
    char z = loc & arenamask;
    char y = (loc >>= arenabits) & arenamask;
    char x = (loc >>= arenabits) & arenamask;
    int channelid = loc >> arenabits;
    if (channelid != 0) {
      channelid += 3;
      p2gbuffer[channelid][x][y][z] = 0.f;
    }
  }
  __syncthreads();

  for (int pidib = threadIdx.x; pidib < partition._ppbs[src_blockno];
       pidib += blockDim.x) {
    int source_blockno, source_pidib;
    ivec3 base_index;
    {
      int advect =
          partition
              ._blockbuckets[src_blockno * g_particle_num_per_block + pidib];
      dir_components(advect / g_particle_num_per_block, base_index);
      base_index += blockid;
      source_blockno = prev_partition.query(base_index);
      source_pidib = advect & (g_particle_num_per_block - 1);
      source_blockno = prev_partition._binsts[source_blockno] +
                       source_pidib / g_bin_capacity;
    }
    vec3 pos;
    vec3 vp_n;
    {
      auto source_particle_bin = pbuffer.ch(_0, source_blockno);
      pos[0] = source_particle_bin.val(_0, source_pidib % g_bin_capacity);
      pos[1] = source_particle_bin.val(_1, source_pidib % g_bin_capacity);
      pos[2] = source_particle_bin.val(_2, source_pidib % g_bin_capacity);
      vp_n[0] = source_particle_bin.val(_12, source_pidib % g_bin_capacity);
      vp_n[1] = source_particle_bin.val(_13, source_pidib % g_bin_capacity);
      vp_n[2] = source_particle_bin.val(_14, source_pidib % g_bin_capacity);
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
    vec3 vel_n;
    vel.set(0.f);
    vel_n.set(0.f);
    vec9 C;
    C.set(0.f);

    // Dp^n = Dp^n+1 = (1/4) * dx^2 * I (Quad.)
    float Dp_inv; //< Inverse Intertia-Like Tensor (1/m^2)
    float scale = g_length * g_length; //< Area scale (m^2)
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
#pragma unroll 3
    for (char i = 0; i < 3; i++)
#pragma unroll 3
      for (char j = 0; j < 3; j++)
#pragma unroll 3
        for (char k = 0; k < 3; k++) {
          vec3 xixp = vec3{(float)i, (float)j, (float)k} * g_dx - local_pos;
          float W = dws(0, i) * dws(1, j) * dws(2, k);
          vec3 vi_n{g2pbuffer[3][local_base_index[0] + i][local_base_index[1] + j]
                           [local_base_index[2] + k],
                  g2pbuffer[4][local_base_index[0] + i][local_base_index[1] + j]
                           [local_base_index[2] + k],
                  g2pbuffer[5][local_base_index[0] + i][local_base_index[1] + j]
                           [local_base_index[2] + k]};
          vel_n += vi_n * W;
        }

#pragma unroll 9
    for (int d = 0; d < 9; ++d)
      dws.val(d) = C[d] * dt * Dp_inv + ((d & 0x3) ? 0.f : 1.f);
    vec9 contrib;
    {
      vec9 F;
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
      matrixMatrixMultiplication3d(dws.data(), contrib.data(), F.data());
      float Jc = 1.f; //< Critical vol. ratio for Fixed-Corotated
      float J = F[0]*F[4]*F[8] + F[3]*F[7]*F[2] + 
                F[6]*F[1]*F[5] - F[6]*F[4]*F[2] - 
                F[3]*F[1]*F[8]; //< J = V/Vo = ||F||
      float beta;
      if (J >= Jc) beta = pbuffer.beta_max; //< beta max
      else beta = pbuffer.beta_min;          //< beta min
      pos += dt * (vel + beta * pbuffer.alpha * (vp_n - vel_n)); //< pos update
      vel += pbuffer.alpha * (vp_n - vel_n); //< vel update
      {
        auto particle_bin = next_pbuffer.ch(_0, partition._binsts[src_blockno] +
                                                    pidib / g_bin_capacity);
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
        particle_bin.val(_12, pidib % g_bin_capacity) = vel[0];
        particle_bin.val(_13, pidib % g_bin_capacity) = vel[1];
        particle_bin.val(_14, pidib % g_bin_capacity) = vel[2];
      }
      compute_stress_fixedcorotated(pbuffer.volume, pbuffer.mu, pbuffer.lambda,
                                    F, contrib);
      contrib = (C * pbuffer.mass - contrib * newDt) * Dp_inv;
    }

    local_base_index = (pos * g_dx_inv + 0.5f).cast<int>() - 1;
    {
      int dirtag = dir_offset((base_index - 1) / g_blocksize -
                              (local_base_index - 1) / g_blocksize);
      partition.add_advection(local_base_index - 1, dirtag, pidib);
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
          pos = vec3{(float)i, (float)j, (float)k} * g_dx - local_pos; //< (xi-xp)
          float W = dws(0, i) * dws(1, j) * dws(2, k); //< Weight (2nd B-Spline)
          auto wm = pbuffer.mass * W; //< Weighted mass
          atomicAdd(
              &p2gbuffer[0][local_base_index[0] + i][local_base_index[1] + j]
                        [local_base_index[2] + k],
              wm); //< m
          atomicAdd(
              &p2gbuffer[1][local_base_index[0] + i][local_base_index[1] + j]
                        [local_base_index[2] + k],
              wm * vel[0] + (contrib[0] * pos[0] + contrib[3] * pos[1] +
                             contrib[6] * pos[2]) *
                                W); //< mvi_star x
          atomicAdd(
              &p2gbuffer[2][local_base_index[0] + i][local_base_index[1] + j]
                        [local_base_index[2] + k],
              wm * vel[1] + (contrib[1] * pos[0] + contrib[4] * pos[1] +
                             contrib[7] * pos[2]) *
                                W); //< mvi_star y
          atomicAdd(
              &p2gbuffer[3][local_base_index[0] + i][local_base_index[1] + j]
                        [local_base_index[2] + k],
              wm * vel[2] + (contrib[2] * pos[0] + contrib[5] * pos[1] +
                             contrib[8] * pos[2]) *
                                W); //< mvi_star z
          atomicAdd(
              &p2gbuffer[4][local_base_index[0] + i][local_base_index[1] + j]
                        [local_base_index[2] + k],
              wm * (vel[0] + Dp_inv * (C[0] * pos[0] + C[3] * pos[1] +
                             C[6] * pos[2]))); //< mvi_n x ASFLIP
          atomicAdd(
              &p2gbuffer[5][local_base_index[0] + i][local_base_index[1] + j]
                        [local_base_index[2] + k],
              wm * (vel[1] + Dp_inv * (C[1] * pos[0] + C[4] * pos[1] +
                             C[7] * pos[2]))); //< mvi_n y ASFLIP
          atomicAdd(
              &p2gbuffer[6][local_base_index[0] + i][local_base_index[1] + j]
                        [local_base_index[2] + k],
              wm * (vel[2] + Dp_inv * (C[2] * pos[0] + C[5] * pos[1] +
                             C[8] * pos[2]))); //< mvi_n z ASFLIP
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
    channelid += 3;
    float val =
        p2gbuffer[channelid][cx + (local_block_id & 4 ? g_blocksize : 0)]
                 [cy + (local_block_id & 2 ? g_blocksize : 0)]
                 [cz + (local_block_id & 1 ? g_blocksize : 0)];
    if (channelid == 4)
      atomicAdd(&next_grid.ch(_0, blockno).val_1d(_4, c), val);
    else if (channelid == 5)
      atomicAdd(&next_grid.ch(_0, blockno).val_1d(_5, c), val);
    else if (channelid == 6)
      atomicAdd(&next_grid.ch(_0, blockno).val_1d(_6, c), val);
  }
}

template <typename Partition, typename Grid, typename VerticeArray>
__global__ void g2p2g(float dt, float newDt, const ivec3 *__restrict__ blocks,
                      const ParticleBuffer<material_e::Sand> pbuffer,
                      ParticleBuffer<material_e::Sand> next_pbuffer,
                      const Partition prev_partition, Partition partition,
                      const Grid grid, Grid next_grid,
                      VerticeArray vertice_array) {
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

  ivec3 blockid;
  int src_blockno;
  if (blocks != nullptr) {
    blockid = blocks[blockIdx.x];
    src_blockno = partition.query(blockid);
  } else {
    if (partition._haloMarks[blockIdx.x])
      return;
    blockid = partition._activeKeys[blockIdx.x];
    src_blockno = blockIdx.x;
  }
  // auto blockid = partition._activeKeys[blockIdx.x];

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

  for (int pidib = threadIdx.x; pidib < partition._ppbs[src_blockno];
       pidib += blockDim.x) {
    int source_blockno, source_pidib;
    ivec3 base_index;
    {
      int advect =
          partition
              ._blockbuckets[src_blockno * g_particle_num_per_block + pidib];
      dir_components(advect / g_particle_num_per_block, base_index);
      base_index += blockid;
      source_blockno = prev_partition.query(base_index);
      source_pidib = advect & (g_particle_num_per_block - 1);
      source_blockno = prev_partition._binsts[source_blockno] +
                       source_pidib / g_bin_capacity;
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
    float Dp_inv; //< Inverse Intertia-Like Tensor (1/m^2)
    float scale = g_length * g_length; //< Area scale (m^2)
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
      dws.val(d) = C[d] * dt * Dp_inv + ((d & 0x3) ? 0.f : 1.f);

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
        auto particle_bin = next_pbuffer.ch(_0, partition._binsts[src_blockno] +
                                                    pidib / g_bin_capacity);
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
      partition.add_advection(local_base_index - 1, dirtag, pidib);
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

template <typename Partition, typename Grid, typename VerticeArray>
__global__ void g2p2g(float dt, float newDt, const ivec3 *__restrict__ blocks,
                      const ParticleBuffer<material_e::NACC> pbuffer,
                      ParticleBuffer<material_e::NACC> next_pbuffer,
                      const Partition prev_partition, Partition partition,
                      const Grid grid, Grid next_grid,
                      VerticeArray vertice_array) {
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

  ivec3 blockid;
  int src_blockno;
  if (blocks != nullptr) {
    blockid = blocks[blockIdx.x];
    src_blockno = partition.query(blockid);
  } else {
    if (partition._haloMarks[blockIdx.x])
      return;
    blockid = partition._activeKeys[blockIdx.x];
    src_blockno = blockIdx.x;
  }
  // auto blockid = partition._activeKeys[blockIdx.x];

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

  for (int pidib = threadIdx.x; pidib < partition._ppbs[src_blockno];
       pidib += blockDim.x) {
    int source_blockno, source_pidib;
    ivec3 base_index;
    {
      int advect =
          partition
              ._blockbuckets[src_blockno * g_particle_num_per_block + pidib];
      dir_components(advect / g_particle_num_per_block, base_index);
      base_index += blockid;
      source_blockno = prev_partition.query(base_index);
      source_pidib = advect & (g_particle_num_per_block - 1);
      source_blockno = prev_partition._binsts[source_blockno] +
                       source_pidib / g_bin_capacity;
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
    float Dp_inv; //< Inverse Intertia-Like Tensor (1/m^2)
    float scale = g_length * g_length; //< Area scale (m^2)
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
      dws.val(d) = C[d] * dt * Dp_inv + ((d & 0x3) ? 0.f : 1.f);

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
        auto particle_bin = next_pbuffer.ch(_0, partition._binsts[src_blockno] +
                                                    pidib / g_bin_capacity);
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
      partition.add_advection(local_base_index - 1, dirtag, pidib);
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


// Grid-to-Particle-to-Grid + Mesh Update - ASFLIP Transfer
// Stress/Strain not computed here, displacements sent to FE mesh for force calc
template <typename Partition, typename Grid, typename VerticeArray>
__global__ void g2p2g(float dt, float newDt, const ivec3 *__restrict__ blocks,
                      const ParticleBuffer<material_e::Meshed> pbuffer,
                      ParticleBuffer<material_e::Meshed> next_pbuffer,
                      const Partition prev_partition, Partition partition,
                      const Grid grid, Grid next_grid,
                      VerticeArray vertice_array) {
  static constexpr uint64_t numViPerBlock = g_blockvolume * 3;
  static constexpr uint64_t numViInArena = numViPerBlock << 3;
  static constexpr uint64_t numViInArena_ASFLIP = (g_blockvolume * 6) << 3;

  static constexpr uint64_t numMViPerBlock = g_blockvolume * 4;
  static constexpr uint64_t numMViInArena = numMViPerBlock << 3;

  static constexpr unsigned arenamask = (g_blocksize << 1) - 1;
  static constexpr unsigned arenabits = g_blockbits + 1;

  extern __shared__ char shmem[];
  using ViArena =
      float(*)[6][g_blocksize << 1][g_blocksize << 1][g_blocksize << 1];
  using ViArenaRef =
      float(&)[6][g_blocksize << 1][g_blocksize << 1][g_blocksize << 1];
  ViArenaRef __restrict__ g2pbuffer = *reinterpret_cast<ViArena>(shmem);
  using MViArena =
      float(*)[7][g_blocksize << 1][g_blocksize << 1][g_blocksize << 1];
  using MViArenaRef =
      float(&)[7][g_blocksize << 1][g_blocksize << 1][g_blocksize << 1];
  MViArenaRef __restrict__ p2gbuffer =
      *reinterpret_cast<MViArena>(shmem + numViInArena_ASFLIP * sizeof(float));

  ivec3 blockid;
  int src_blockno;
  if (blocks != nullptr) {
    blockid = blocks[blockIdx.x];
    src_blockno = partition.query(blockid);
  } else {
    if (partition._haloMarks[blockIdx.x])
      return;
    blockid = partition._activeKeys[blockIdx.x];
    src_blockno = blockIdx.x;
  }

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
    else if (channelid == 2) 
      val = grid_block.val_1d(_3, c);
    g2pbuffer[channelid][cx + (local_block_id & 4 ? g_blocksize : 0)]
             [cy + (local_block_id & 2 ? g_blocksize : 0)]
             [cz + (local_block_id & 1 ? g_blocksize : 0)] = val;
  }
  __syncthreads();
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
    channelid += 3;
    float val;
    if (channelid == 3) {
      val = grid_block.val_1d(_4, c);
    } else if (channelid == 4) {
      val = grid_block.val_1d(_5, c);
    } else if (channelid == 5) {
      val = grid_block.val_1d(_6, c);
    }
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
  for (int base = threadIdx.x; base < numMViInArena; base += blockDim.x) {
    int loc = base;
    char z = loc & arenamask;
    char y = (loc >>= arenabits) & arenamask;
    char x = (loc >>= arenabits) & arenamask;
    int channelid = loc >> arenabits;
    if (channelid != 0) {
      channelid += 3;
      p2gbuffer[channelid][x][y][z] = 0.f;
    }
  }
  __syncthreads();
  for (int pidib = threadIdx.x; pidib < partition._ppbs[src_blockno];
       pidib += blockDim.x) {
    int source_blockno, source_pidib;
    ivec3 base_index;
    {
      int advect =
          partition
              ._blockbuckets[src_blockno * g_particle_num_per_block + pidib];
      dir_components(advect / g_particle_num_per_block, base_index);
      base_index += blockid;
      source_blockno = prev_partition.query(base_index);
      source_pidib = advect & (g_particle_num_per_block - 1);
      source_blockno = prev_partition._binsts[source_blockno] +
                       source_pidib / g_bin_capacity;
    }
    vec3 pos;  //< Particle position at n
    int ID; // Vertice ID for mesh
    vec3 vp_n; //< Particle vel. at n
    float tension;
    //float J;   //< Particle volume ratio at n
    vec3 b;
    {
      auto source_particle_bin = pbuffer.ch(_0, source_blockno);
      pos[0] = source_particle_bin.val(_0, source_pidib % g_bin_capacity);  //< x
      pos[1] = source_particle_bin.val(_1, source_pidib % g_bin_capacity);  //< y
      pos[2] = source_particle_bin.val(_2, source_pidib % g_bin_capacity);  //< z
      ID = (int)source_particle_bin.val(_3, source_pidib % g_bin_capacity); //< ID
      vp_n[0] = source_particle_bin.val(_4, source_pidib % g_bin_capacity); //< vx
      vp_n[1] = source_particle_bin.val(_5, source_pidib % g_bin_capacity); //< vy
      vp_n[2] = source_particle_bin.val(_6, source_pidib % g_bin_capacity); //< vz
      tension = source_particle_bin.val(_7, source_pidib % g_bin_capacity); //< vz
      b[0] = source_particle_bin.val(_8, source_pidib % g_bin_capacity); //< bx
      b[1] = source_particle_bin.val(_9, source_pidib % g_bin_capacity); //< by
      b[2] = source_particle_bin.val(_10, source_pidib % g_bin_capacity); //< bz
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
    vec3 vel;   //< Stressed, collided grid velocity
    vec3 vel_n; //< Unstressed, uncollided grid velocity
    vec9 C;     //< APIC affine matrix, used a few times
    vel.set(0.f); 
    vel_n.set(0.f);
    C.set(0.f);

    // Dp^n = Dp^n+1 = (1/4) * dx^2 * I (Quad.)
    float Dp_inv; //< Inverse Intertia-Like Tensor (1/m^2)
    float scale = g_length * g_length; //< Area scale (m^2)
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
#pragma unroll 3
    for (char i = 0; i < 3; i++)
#pragma unroll 3
      for (char j = 0; j < 3; j++)
#pragma unroll 3
        for (char k = 0; k < 3; k++) {
          vec3 xixp = vec3{(float)i, (float)j, (float)k} * g_dx - local_pos;
          float W = dws(0, i) * dws(1, j) * dws(2, k);
          vec3 vi_n{g2pbuffer[3][local_base_index[0] + i][local_base_index[1] + j]
                            [local_base_index[2] + k],
                    g2pbuffer[4][local_base_index[0] + i][local_base_index[1] + j]
                            [local_base_index[2] + k],
                    g2pbuffer[5][local_base_index[0] + i][local_base_index[1] + j]
                            [local_base_index[2] + k]};
          vel_n += vi_n * W; 
        }
    //J = (1 + (C[0] + C[4] + C[8]) * dt * Dp_inv) * J;
    float beta;

    // if (J >= 1.f) beta = pbuffer.beta_max; //< Position correction factor (ASFLIP)
    // else if (J < 1.f) beta = pbuffer.beta_min;
    // else beta = 0.f;
    if (tension >= 1.f) beta = pbuffer.beta_max; //< Position correction factor (ASFLIP)
    else if (tension <= -1.f) beta = 0.f;
    else beta = pbuffer.beta_min;
    

    // Clean this up - JB
    float count;
    count = vertice_array.val(_10,ID); //< count

    int use_surface = 0;
    // Interior
    if (use_surface && count > 6.f) {
      pos += dt * (vel + beta * pbuffer.alpha * (vp_n - vel_n));
      vel += pbuffer.alpha * (vp_n - vel_n);
    }
    // Exterior
    // Modified Sim. for Vol. Obj. with Frictional Contact
    // Mixed with FLIP/PIC style
    // Collision Particles = Vertices (i.e. q = p)
    // No grid impulse set for cohesionless contact yet
    else if (use_surface && count <= 6.f) {
      float b_mag;
      b_mag = sqrt(b[0]*b[0] + b[1]*b[1] + b[2]*b[2]);
      float small, tiny;
      small = 1.f;
      tiny = 1e-5;
      if ( fabs(b_mag) < small ) {
        pos += dt * (vel + beta * pbuffer.alpha * (vp_n - vel_n));
        vel += pbuffer.alpha * (vp_n - vel_n);
      } else {
        vec3 n;
        n = b / b_mag;
        vec3 vr;
        //vr = vel_n - vp_n;
        vr = (vel - vp_n);
        float vrdotn;
        vrdotn = (vr[0]*n[0] + vr[1]*n[1] + vr[2]*n[2]);

        // Contact
        if (vrdotn >= 0.f) {
          // Seperate
          
          //vel = vp_n;
          // Consider using a ASFLIP style advection here
          // Might avoid grid impulses? Or atleast 
          // breaks positional trap
          vel += pbuffer.alpha * (vp_n - vel_n);        
          pos += vel * dt;
        } else {
          // Collide
          vec3 vt;  
          float I;
          I = 2 * vrdotn; // Tech. has mass but divides out later
          vt[0] = vr[0] - vrdotn * vr[0];
          vt[1] = vr[1] - vrdotn * vr[1];
          vt[2] = vr[2] - vrdotn * vr[2];

        
          //float vt_mag;
          //vt_mag = sqrt(vt[0]*vt[0] + vt[1]*vt[1] + vt[2]*vt[2]);
          float mu; // Coulumb friction parameter
          mu = 0.f;
          //if ( fabs(vt_mag) < tiny) vt_mag = 1.f;

          vel += pbuffer.alpha * I*n;// + fmin(vt_mag, -mu*I) * vt / vt_mag ;
          pos += dt * vel;
          
        }

      }
    } else {
      pos += dt * (vel + beta * pbuffer.alpha * (vp_n - vel_n));
      vel += pbuffer.alpha * (vp_n - vel_n);
    }
    
    {
      {
        vertice_array.val(_0, ID) = pos[0];
        vertice_array.val(_1, ID) = pos[1];
        vertice_array.val(_2, ID) = pos[2];
        vertice_array.val(_3, ID) = 0.f; //< bx
        vertice_array.val(_4, ID) = 0.f; //< by
        vertice_array.val(_5, ID) = 0.f; //< bz
        vertice_array.val(_6, ID) = 0.f; //< restMass
        vertice_array.val(_7, ID) = 0.f; //< fx
        vertice_array.val(_8, ID) = 0.f; //< fy
        vertice_array.val(_9, ID) = 0.f; //< fz
        vertice_array.val(_10,ID) = 0.f; //< count
      }
    }
  }
}


// Grid-to-Particle-to-Grid + Mesh Update - ASFLIP Transfer
// Stress/Strain not computed here, displacements sent to FE mesh for force calc
template <typename VerticeArray, typename ElementArray>
__global__ void v2fem2v(float dt, float newDt,
                      VerticeArray vertice_array,
                      ElementArray element_array,
                      ElementBuffer<fem_e::Tetrahedron> elementBins) {

  if (blockIdx.x >= g_max_fem_element_num) return;
  int IDs[4];
  vec3 p[4];
  auto element = elementBins.ch(_0, blockIdx.x);


  /* matrix indexes
          mat1   |  diag   |  mat2^T
          0 3 6  |  0      |  0 1 2
          1 4 7  |    1    |  3 4 5
          2 5 8  |      2  |  6 7 8
  */

  /// Precomputed
  // Bm is undeformed face normal matrix
  // relative to a point 0
  // Bm^T = Dm^-1
  vec9 BmT;
  BmT.set(0.f);
  IDs[0] = element.val(_0, 0);
  IDs[1] = element.val(_1, 0);
  IDs[2] = element.val(_2, 0);
  IDs[3] = element.val(_3, 0);
  BmT[0] = element.val(_4, 0);
  BmT[1] = element.val(_5, 0);
  BmT[2] = element.val(_6, 0);
  BmT[3] = element.val(_7, 0);
  BmT[4] = element.val(_8, 0);
  BmT[5] = element.val(_9, 0);
  BmT[6] = element.val(_10, 0);
  BmT[7] = element.val(_11, 0);
  BmT[8] = element.val(_12, 0);
  float restVolume = element.val(_13, 0);

  // Set position of vertices
  for (int v = 0; v < 4; v++) {
    int ID = IDs[v] - 1; //< Index at 0, my elements input files index from 1
    p[v][0] = vertice_array.val(_0, ID) * g_length;
    p[v][1] = vertice_array.val(_1, ID) * g_length;
    p[v][2] = vertice_array.val(_2, ID) * g_length;
  }
  __syncthreads();

  /// Run-Time
  // Ds is deformed edge vector matrix
  // relative to a point 0
  vec9 Ds;
  Ds.set(0.f);
  Ds[0] = p[1][0] - p[0][0];
  Ds[1] = p[1][1] - p[0][1];
  Ds[2] = p[1][2] - p[0][2];
  Ds[3] = p[2][0] - p[0][0];
  Ds[4] = p[2][1] - p[0][1];
  Ds[5] = p[2][2] - p[0][2];
  Ds[6] = p[3][0] - p[0][0];
  Ds[7] = p[3][1] - p[0][1];
  Ds[8] = p[3][2] - p[0][2];

  // F is deformation gradient tensor 
  vec9 F; // Deformation gradient at Gauss point
  F.set(0.f);
  // F = Ds Dm^-1 = Ds Bm^T; Bm^T = Dm^-1
  matrixMatrixMultiplication3d(Ds.data(), BmT.data(), F.data());

  // F^-1 = inv(F)
  vec9 Finv;
  Finv.set(0.f);
  matrixInverse(F.data(), Finv.data());

  // J = det | F |,  i.e. volume change ratio undef -> def
  float J;
  J = matrixDeterminant3d(F.data());

  float tension;
  float Jc = 1.0; // Characteristic J for material
  if (J >= Jc) tension = 1.f;
  else if (J < 0.99f) tension = -1.f;
  else tension = 0.f;

  // P is First Piola-Kirchoff stress at element Gauss point
  vec9 P;
  P.set(0.f);
  // P = (dPsi/dF){F}
  // Fixed-Corotated solid model here for energy potential
  compute_stress_FEM_fixedcorotated(restVolume, elementBins.mu, elementBins.lambda, F, P);

  // G is deformed internal force contribution matrix
  // at vertices relative to a vertex 0
  vec9 G;
  G.set(0.f);
  // G = P Bm
  matrixMatrixTransposeMultiplication3d(P.data(), BmT.data(), G.data());
  
  __syncthreads();

  // Bs is the deformed face normal matrix
  vec9 BsT; 
  BsT.set(0.f);
  // Bs = J F^-1^T Bm ;  Bs^T = J Bm^T F^-1
  // Note: (AB)^T = B^T A^T
  matrixMatrixMultiplication3d(BmT.data(), Finv.data(), BsT.data());

  // f is the internal force vector at deformed vertex
  vec3 f; // Internal force vector
  f.set(0.f);

  // b is the vertex normal deformed vector
  vec3 b; // Vertex normal deformed
  b.set(0.f);

  // Vertices = {0,1,2,3}
  // Thread 1,..,n handles vertex 1,..,n
  if (threadIdx.x == 1){
    f[0] = G[0];
    f[1] = G[1];
    f[2] = G[2];
    b[0] = J * BsT[0];
    b[1] = J * BsT[3];
    b[2] = J * BsT[6];
  } else if (threadIdx.x == 2) {
    f[0] = G[3];
    f[1] = G[4];
    f[2] = G[5];
    b[0] = J * BsT[1];
    b[1] = J * BsT[4];
    b[2] = J * BsT[7];
  }  else if (threadIdx.x == 3) {
    f[0] = G[6];
    f[1] = G[7];
    f[2] = G[8];
    b[0] = J * BsT[2];
    b[1] = J * BsT[5];
    b[2] = J * BsT[8];
  } else {
    f[0] = - (G[0] + G[3] + G[6]);
    f[1] = - (G[1] + G[4] + G[7]);
    f[2] = - (G[2] + G[5] + G[8]);
    b[0] = - J * (BsT[0] + BsT[1] + BsT[2]);
    b[1] = - J * (BsT[3] + BsT[4] + BsT[5]);
    b[2] = - J * (BsT[6] + BsT[7] + BsT[8]);
  }
  float b_mag;
  b_mag = sqrt(b[0]*b[0] + b[1]*b[1] + b[2]*b[2]);
  float scale = restVolume / g_length;
  f[0] = f[0] * scale;
  f[1] = f[1] * scale; 
  f[2] = f[2] * scale;
  b[0] = b[0] / b_mag;
  b[1] = b[1] / b_mag;
  b[2] = b[2] / b_mag;

  if (0) {
    if ((blockIdx.x) == 0){
        printf("Element (%d) Node (%d) force is: (%f %f %f)\n", blockIdx.x, IDs[threadIdx.x], f[0], f[1], f[2]);
        printf("Element (%d) Node (%d) position is: (%f %f %f)\n", blockIdx.x, IDs[threadIdx.x], p[threadIdx.x][0], p[threadIdx.x][1], p[threadIdx.x][2]);
        if ((threadIdx.x) == 0){
          printf("Element (%d) restVolume is: (%f)\n", blockIdx.x, restVolume);
        }
    }
  }
  __syncthreads();
  {
    int ID = IDs[threadIdx.x] - 1;
    atomicAdd(&vertice_array.val(_3, ID), b[0]); //< bx
    atomicAdd(&vertice_array.val(_4, ID), b[1]); //< by
    atomicAdd(&vertice_array.val(_5, ID), b[2]); //< bz
    atomicAdd(&vertice_array.val(_6, ID), fabs(restVolume) / 4.f); //< vol
    atomicAdd(&vertice_array.val(_7, ID), f[0]); //< fx
    atomicAdd(&vertice_array.val(_8, ID), f[1]); //< fy
    atomicAdd(&vertice_array.val(_9, ID), f[2]); //< fz
    atomicAdd(&vertice_array.val(_10, ID), 1.f); //< fz
  }
  __syncthreads();
}

template <typename VerticeArray, typename ElementArray>
__global__ void v2fem2v(float dt, float newDt,
                      VerticeArray vertice_array,
                      ElementArray element_array,
                      ElementBuffer<fem_e::Brick> elementBins) {
                        return;
}

template <typename Partition, typename Grid, typename VerticeArray>
__global__ void fem2p2g(float dt, float newDt, const ivec3 *__restrict__ blocks,
                      const ParticleBuffer<material_e::JFluid> pbuffer,
                      ParticleBuffer<material_e::JFluid> next_pbuffer,
                      const Partition prev_partition, Partition partition,
                      const Grid grid, Grid next_grid,
                      VerticeArray vertice_array) {
                        return;
                      }

template <typename Partition, typename Grid, typename VerticeArray>
__global__ void fem2p2g(float dt, float newDt, const ivec3 *__restrict__ blocks,
                      const ParticleBuffer<material_e::JFluid_ASFLIP> pbuffer,
                      ParticleBuffer<material_e::JFluid_ASFLIP> next_pbuffer,
                      const Partition prev_partition, Partition partition,
                      const Grid grid, Grid next_grid,
                      VerticeArray vertice_array) {
                        return;
                      }
template <typename Partition, typename Grid, typename VerticeArray>
__global__ void fem2p2g(float dt, float newDt, const ivec3 *__restrict__ blocks,
                      const ParticleBuffer<material_e::FixedCorotated> pbuffer,
                      ParticleBuffer<material_e::FixedCorotated> next_pbuffer,
                      const Partition prev_partition, Partition partition,
                      const Grid grid, Grid next_grid,
                      VerticeArray vertice_array) {
                        return;
                      }
template <typename Partition, typename Grid, typename VerticeArray>
__global__ void fem2p2g(float dt, float newDt, const ivec3 *__restrict__ blocks,
                      const ParticleBuffer<material_e::FixedCorotated_ASFLIP> pbuffer,
                      ParticleBuffer<material_e::FixedCorotated_ASFLIP> next_pbuffer,
                      const Partition prev_partition, Partition partition,
                      const Grid grid, Grid next_grid,
                      VerticeArray vertice_array) {
                        return;
                      }
template <typename Partition, typename Grid, typename VerticeArray>
__global__ void fem2p2g(float dt, float newDt, const ivec3 *__restrict__ blocks,
                      const ParticleBuffer<material_e::Sand> pbuffer,
                      ParticleBuffer<material_e::Sand> next_pbuffer,
                      const Partition prev_partition, Partition partition,
                      const Grid grid, Grid next_grid,
                      VerticeArray vertice_array) {
                        return;
                      }
template <typename Partition, typename Grid, typename VerticeArray>
__global__ void fem2p2g(float dt, float newDt, const ivec3 *__restrict__ blocks,
                      const ParticleBuffer<material_e::NACC> pbuffer,
                      ParticleBuffer<material_e::NACC> next_pbuffer,
                      const Partition prev_partition, Partition partition,
                      const Grid grid, Grid next_grid,
                      VerticeArray vertice_array) {
                        return;
                      }


// Grid-to-Particle-to-Grid + Mesh Update - ASFLIP Transfer
// Stress/Strain not computed here, displacements sent to FE mesh for force calc
template <typename Partition, typename Grid, typename VerticeArray>
__global__ void fem2p2g(float dt, float newDt, const ivec3 *__restrict__ blocks,
                      const ParticleBuffer<material_e::Meshed> pbuffer,
                      ParticleBuffer<material_e::Meshed> next_pbuffer,
                      const Partition prev_partition, Partition partition,
                      const Grid grid, Grid next_grid,
                      VerticeArray vertice_array) {
  static constexpr uint64_t numViPerBlock = g_blockvolume * 3;
  static constexpr uint64_t numViInArena = numViPerBlock << 3;
  static constexpr uint64_t numViInArena_ASFLIP = (g_blockvolume * 6) << 3;

  static constexpr uint64_t numMViPerBlock = g_blockvolume * 4;
  static constexpr uint64_t numMViInArena = numMViPerBlock << 3;

  static constexpr unsigned arenamask = (g_blocksize << 1) - 1;
  static constexpr unsigned arenabits = g_blockbits + 1;

  extern __shared__ char shmem[];
  using ViArena =
      float(*)[6][g_blocksize << 1][g_blocksize << 1][g_blocksize << 1];
  using ViArenaRef =
      float(&)[6][g_blocksize << 1][g_blocksize << 1][g_blocksize << 1];
  ViArenaRef __restrict__ g2pbuffer = *reinterpret_cast<ViArena>(shmem);
  using MViArena =
      float(*)[7][g_blocksize << 1][g_blocksize << 1][g_blocksize << 1];
  using MViArenaRef =
      float(&)[7][g_blocksize << 1][g_blocksize << 1][g_blocksize << 1];
  MViArenaRef __restrict__ p2gbuffer =
      *reinterpret_cast<MViArena>(shmem + numViInArena_ASFLIP * sizeof(float));

  ivec3 blockid;
  int src_blockno;
  if (blocks != nullptr) {
    blockid = blocks[blockIdx.x];
    src_blockno = partition.query(blockid);
  } else {
    if (partition._haloMarks[blockIdx.x])
      return;
    blockid = partition._activeKeys[blockIdx.x];
    src_blockno = blockIdx.x;
  }

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
    else if (channelid == 2) 
      val = grid_block.val_1d(_3, c);
    g2pbuffer[channelid][cx + (local_block_id & 4 ? g_blocksize : 0)]
             [cy + (local_block_id & 2 ? g_blocksize : 0)]
             [cz + (local_block_id & 1 ? g_blocksize : 0)] = val;
  }
  __syncthreads();
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
    channelid += 3;
    float val;
    if (channelid == 3) {
      val = grid_block.val_1d(_4, c);
    } else if (channelid == 4) {
      val = grid_block.val_1d(_5, c);
    } else if (channelid == 5) {
      val = grid_block.val_1d(_6, c);
    }
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
  for (int base = threadIdx.x; base < numMViInArena; base += blockDim.x) {
    int loc = base;
    char z = loc & arenamask;
    char y = (loc >>= arenabits) & arenamask;
    char x = (loc >>= arenabits) & arenamask;
    int channelid = loc >> arenabits;
    if (channelid != 0) {
      channelid += 3;
      p2gbuffer[channelid][x][y][z] = 0.f;
    }
  }
  __syncthreads();
  for (int pidib = threadIdx.x; pidib < partition._ppbs[src_blockno];
       pidib += blockDim.x) {
    int source_blockno, source_pidib;
    ivec3 base_index;
    {
      int advect =
          partition
              ._blockbuckets[src_blockno * g_particle_num_per_block + pidib];
      dir_components(advect / g_particle_num_per_block, base_index);
      base_index += blockid;
      source_blockno = prev_partition.query(base_index);
      source_pidib = advect & (g_particle_num_per_block - 1);
      source_blockno = prev_partition._binsts[source_blockno] +
                       source_pidib / g_bin_capacity;
    }
    vec3 pos;  //< Particle position at n
    int ID; // Vertice ID for mesh
    vec3 vp_n; //< Particle vel. at n
    float tension;
    vec3 b;
    //float J;   //< Particle volume ratio at n
    {
      auto source_particle_bin = pbuffer.ch(_0, source_blockno);
      pos[0] = source_particle_bin.val(_0, source_pidib % g_bin_capacity);  //< x
      pos[1] = source_particle_bin.val(_1, source_pidib % g_bin_capacity);  //< y
      pos[2] = source_particle_bin.val(_2, source_pidib % g_bin_capacity);  //< z
      ID = (int)source_particle_bin.val(_3, source_pidib % g_bin_capacity); //< ID
      vp_n[0] = source_particle_bin.val(_4, source_pidib % g_bin_capacity); //< vx
      vp_n[1] = source_particle_bin.val(_5, source_pidib % g_bin_capacity); //< vy
      vp_n[2] = source_particle_bin.val(_6, source_pidib % g_bin_capacity); //< vz
      tension = source_particle_bin.val(_7, source_pidib % g_bin_capacity); //< vz
      b[0] = source_particle_bin.val(_8, source_pidib % g_bin_capacity); //< bx
      b[1] = source_particle_bin.val(_9, source_pidib % g_bin_capacity); //< by
      b[2] = source_particle_bin.val(_10, source_pidib % g_bin_capacity); //< bz
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
    vec3 vel;   //< Stressed, collided grid velocity
    vec3 vel_n; //< Unstressed, uncollided grid velocity
    vec9 C;     //< APIC affine matrix, used a few times
    vel.set(0.f); 
    vel_n.set(0.f);
    C.set(0.f);

    // Dp^n = Dp^n+1 = (1/4) * dx^2 * I (Quad.)
    float Dp_inv; //< Inverse Intertia-Like Tensor (1/m^2)
    float scale = g_length * g_length; //< Area scale (m^2)
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
#pragma unroll 3
    for (char i = 0; i < 3; i++)
#pragma unroll 3
      for (char j = 0; j < 3; j++)
#pragma unroll 3
        for (char k = 0; k < 3; k++) {
          vec3 xixp = vec3{(float)i, (float)j, (float)k} * g_dx - local_pos;
          float W = dws(0, i) * dws(1, j) * dws(2, k);
          vec3 vi_n{g2pbuffer[3][local_base_index[0] + i][local_base_index[1] + j]
                            [local_base_index[2] + k],
                    g2pbuffer[4][local_base_index[0] + i][local_base_index[1] + j]
                            [local_base_index[2] + k],
                    g2pbuffer[5][local_base_index[0] + i][local_base_index[1] + j]
                            [local_base_index[2] + k]};
          vel_n += vi_n * W; 
        }
    //J = (1 + (C[0] + C[4] + C[8]) * dt * Dp_inv) * J;

    vec3 b_new; //< Vertex normal, area weighted
    b_new.set(0.f);
    float restVolume;
    vec3 f; //< Internal nodes forces
    f.set(0.f);
    float restMass;
    float count;
    count = 0.f;
    {
      b_new[0] = vertice_array.val(_3, ID);
      b_new[1] = vertice_array.val(_4, ID);
      b_new[2] = vertice_array.val(_5, ID);
      restVolume = vertice_array.val(_6, ID);
      f[0] = vertice_array.val(_7, ID);
      f[1] = vertice_array.val(_8, ID);
      f[2] = vertice_array.val(_9, ID);
      count = vertice_array.val(_10, ID);
      restMass = restVolume * pbuffer.rho;
      // Zero-out 
      vertice_array.val(_3, ID) = 0.f; // bx
      vertice_array.val(_4, ID) = 0.f; // by
      vertice_array.val(_5, ID) = 0.f; // bz
      vertice_array.val(_6, ID) = 0.f; // vol
      vertice_array.val(_7, ID) = 0.f; // fx
      vertice_array.val(_8, ID) = 0.f; // fy
      vertice_array.val(_9, ID) = 0.f; // fz
      vertice_array.val(_10, ID) = 0.f; // count

    }
    //restMass = pbuffer.mass;

    float beta;
    // if (J >= 1.f) beta = pbuffer.beta_max; //< Position correction factor (ASFLIP)
    // else if (J < 1.f) beta = pbuffer.beta_min;
    // else beta = 0.f;
    if (tension >= 1.f) beta = pbuffer.beta_max; //< Position correction factor (ASFLIP)
    else if (tension <= -1.f) beta = 0.f;
    else beta = pbuffer.beta_min;
    
    int use_surface = 0;
    // Interior
    if (use_surface && count > 6.f){
      pos += dt * (vel + beta * pbuffer.alpha * (vp_n - vel_n));
      vel += pbuffer.alpha * (vp_n - vel_n);
    }
    // Exterior
    // Modified Sim. for Vol. Obj. with Frictional Contact
    // Mixed with FLIP/PIC style
    // Collision Particles = Vertices (i.e. q = p)
    // No grid impulse set for cohesionless contact yet
    else if (use_surface && count <= 6.f) {
      float b_mag;
      b_mag = sqrt(b[0]*b[0] + b[1]*b[1] + b[2]*b[2]);
      float small, tiny;
      small = 1.f;
      tiny = 1e-5;
      if ( fabs(b_mag) < small ) {
        pos += dt * (vel + beta * pbuffer.alpha * (vp_n - vel_n));
        vel += pbuffer.alpha * (vp_n - vel_n);
      } else {
        vec3 n;
        n = b / b_mag;
        vec3 vr;
        //vr = vel_n - vp_n;
        vr = (vel - vp_n);
        float vrdotn;
        vrdotn = (vr[0]*n[0] + vr[1]*n[1] + vr[2]*n[2]);

        // Contact
        if (vrdotn >= 0.f) {
          // Seperate
          
          //vel = vp_n;
          // Consider using a ASFLIP style advection here
          // Might avoid grid impulses? Or atleast 
          // breaks positional trap
          vel += pbuffer.alpha * (vp_n - vel_n);        
          pos += vel * dt;
        } else {
          // Collide
          vec3 vt;  
          float I;
          I = 2 * vrdotn; // Tech. has mass but divides out later
          vt[0] = vr[0] - vrdotn * vr[0];
          vt[1] = vr[1] - vrdotn * vr[1];
          vt[2] = vr[2] - vrdotn * vr[2];

        
          //float vt_mag;
          //vt_mag = sqrt(vt[0]*vt[0] + vt[1]*vt[1] + vt[2]*vt[2]);
          float mu; // Coulumb friction parameter
          mu = 0.f;
          //if ( fabs(vt_mag) < tiny) vt_mag = 1.f;
          vel += pbuffer.alpha * I*n ;//+ fmin(vt_mag, -mu*I) * vt / vt_mag ;
          pos += dt * vel;
        }

      }
    } else {
      pos += dt * (vel + beta * pbuffer.alpha * (vp_n - vel_n));
      vel += pbuffer.alpha * (vp_n - vel_n);
    }

    {
      {
        auto particle_bin = next_pbuffer.ch(_0, partition._binsts[src_blockno] +
                                                    pidib / g_bin_capacity);
        particle_bin.val(_0, pidib % g_bin_capacity) = pos[0]; //< x
        particle_bin.val(_1, pidib % g_bin_capacity) = pos[1]; //< y
        particle_bin.val(_2, pidib % g_bin_capacity) = pos[2]; //< z
        particle_bin.val(_3, pidib % g_bin_capacity) = (float)ID; //< ID
        particle_bin.val(_4, pidib % g_bin_capacity) = vel[0]; //< vx
        particle_bin.val(_5, pidib % g_bin_capacity) = vel[1]; //< vy
        particle_bin.val(_6, pidib % g_bin_capacity) = vel[2]; //< vz
        // Recheck this
        particle_bin.val(_7, pidib % g_bin_capacity) = restMass; //< tension
        particle_bin.val(_8, pidib % g_bin_capacity) = b_new[0]; // bx
        particle_bin.val(_9, pidib % g_bin_capacity) = b_new[1]; // by
        particle_bin.val(_10, pidib % g_bin_capacity) = b_new[2]; // bz

      }
    }

    local_base_index = (pos * g_dx_inv + 0.5f).cast<int>() - 1;
    {
      int dirtag = dir_offset((base_index - 1) / g_blocksize -
                              (local_base_index - 1) / g_blocksize);
      partition.add_advection(local_base_index - 1, dirtag, pidib);
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
          auto wm = restMass * W;
          atomicAdd(
              &p2gbuffer[0][local_base_index[0] + i][local_base_index[1] + j]
                        [local_base_index[2] + k],
              wm);
          // ASFLIP velocity, force sent after FE mesh calc
          atomicAdd(
              &p2gbuffer[1][local_base_index[0] + i][local_base_index[1] + j]
                        [local_base_index[2] + k],
              wm * (vel[0] + Dp_inv * (C[0] * pos[0] + C[3] * pos[1] +
                             C[6] * pos[2])) - W * (f[0] * newDt));
          atomicAdd(
              &p2gbuffer[2][local_base_index[0] + i][local_base_index[1] + j]
                        [local_base_index[2] + k],
              wm * (vel[1] + Dp_inv * (C[1] * pos[0] + C[4] * pos[1] +
                             C[7] * pos[2])) - W * (f[1] * newDt));
          atomicAdd(
              &p2gbuffer[3][local_base_index[0] + i][local_base_index[1] + j]
                        [local_base_index[2] + k],
              wm * (vel[2] + Dp_inv * (C[2] * pos[0] + C[5] * pos[1] +
                             C[8] * pos[2])) - W * (f[2] * newDt));
          // ASFLIP unstressed velocity
          atomicAdd(
              &p2gbuffer[4][local_base_index[0] + i][local_base_index[1] + j]
                        [local_base_index[2] + k],
              wm * (vel[0] + Dp_inv * (C[0] * pos[0] + C[3] * pos[1] +
                             C[6] * pos[2])));
          atomicAdd(
              &p2gbuffer[5][local_base_index[0] + i][local_base_index[1] + j]
                        [local_base_index[2] + k],
              wm * (vel[1] + Dp_inv * (C[1] * pos[0] + C[4] * pos[1] +
                             C[7] * pos[2])));
          atomicAdd(
              &p2gbuffer[6][local_base_index[0] + i][local_base_index[1] + j]
                        [local_base_index[2] + k],
              wm * (vel[2] + Dp_inv * (C[2] * pos[0] + C[5] * pos[1] +
                             C[8] * pos[2])));
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
    if (channelid == 0) {
      atomicAdd(&next_grid.ch(_0, blockno).val_1d(_0, c), val);
    } else if (channelid == 1) {
      atomicAdd(&next_grid.ch(_0, blockno).val_1d(_1, c), val);
    } else if (channelid == 2) {
      atomicAdd(&next_grid.ch(_0, blockno).val_1d(_2, c), val);
    } else if (channelid == 3) {
      atomicAdd(&next_grid.ch(_0, blockno).val_1d(_3, c), val);
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
    int channelid = base & (numMViPerBlock - 1);
    char c = channelid % g_blockvolume;
    char cz = channelid & g_blockmask;
    char cy = (channelid >>= g_blockbits) & g_blockmask;
    char cx = (channelid >>= g_blockbits) & g_blockmask;
    channelid >>= g_blockbits;
    channelid += 3;
    float val =
        p2gbuffer[channelid][cx + (local_block_id & 4 ? g_blocksize : 0)]
                 [cy + (local_block_id & 2 ? g_blocksize : 0)]
                 [cz + (local_block_id & 1 ? g_blocksize : 0)];
    if (channelid == 4) {
      atomicAdd(&next_grid.ch(_0, blockno).val_1d(_4, c), val);
    } else if (channelid == 5) {
      atomicAdd(&next_grid.ch(_0, blockno).val_1d(_5, c), val);
    } else if (channelid == 6) {
      atomicAdd(&next_grid.ch(_0, blockno).val_1d(_6, c), val);
    }
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
  _marks[blockno] = _ppbs[blockno] > 0 ? 1 : 0;
}

template <typename Partition>
__global__ void
update_partition(uint32_t blockCount, const int *__restrict__ _sourceNos,
                 const Partition partition, Partition next_partition) {
  __shared__ std::size_t sourceNo[1];
  std::size_t blockno = blockIdx.x;
  if (blockno >= blockCount)
    return;
  if (threadIdx.x == 0) {
    sourceNo[0] = _sourceNos[blockno];
    auto sourceBlockid = partition._activeKeys[sourceNo[0]];
    next_partition._activeKeys[blockno] = sourceBlockid;
    next_partition.reinsert(blockno);
    next_partition._ppbs[blockno] = partition._ppbs[sourceNo[0]];
  }
  __syncthreads();

  auto pcnt = next_partition._ppbs[blockno];
  for (int pidib = threadIdx.x; pidib < pcnt; pidib += blockDim.x)
    next_partition._blockbuckets[blockno * g_particle_num_per_block + pidib] =
        partition._blockbuckets[sourceNo[0] * g_particle_num_per_block + pidib];
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
    // ASFLIP velocities unstressed
    targetblock.val_1d(_4, threadIdx.x) = sourceblock.val_1d(_4, threadIdx.x);
    targetblock.val_1d(_5, threadIdx.x) = sourceblock.val_1d(_5, threadIdx.x);
    targetblock.val_1d(_6, threadIdx.x) = sourceblock.val_1d(_6, threadIdx.x);
  }
}


template <typename Partition>
__global__ void check_table(uint32_t blockCount, Partition partition) {
  uint32_t blockno = blockIdx.x * blockDim.x + threadIdx.x;
  if (blockno >= blockCount)
    return;
  auto blockid = partition._activeKeys[blockno];
  if (partition.query(blockid) != blockno)
    printf("DAMN, partition table is wrong!\n");
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
    printf("\tDAMN, encountered zero block record\n");
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

template <typename Partition, typename ParticleBuffer, typename ParticleArray>
__global__ void retrieve_particle_buffer(Partition partition,
                                         Partition prev_partition,
                                         ParticleBuffer pbuffer,
                                         ParticleArray parray, 
                                         float *dispVal, 
                                         int *_parcnt) {
  int pcnt = partition._ppbs[blockIdx.x];
  ivec3 blockid = partition._activeKeys[blockIdx.x];
  auto advection_bucket =
      partition._blockbuckets + blockIdx.x * g_particle_num_per_block;
  // auto particle_offset = partition._binsts[blockIdx.x];
  for (int pidib = threadIdx.x; pidib < pcnt; pidib += blockDim.x) {
    auto advect = advection_bucket[pidib];
    ivec3 source_blockid;
    dir_components(advect / g_particle_num_per_block, source_blockid);
    source_blockid += blockid;
    auto source_blockno = prev_partition.query(source_blockid);
    auto source_pidib = advect % g_particle_num_per_block;
    auto source_bin = pbuffer.ch(_0, prev_partition._binsts[source_blockno] +
                                         source_pidib / g_bin_capacity);
    auto _source_pidib = source_pidib % g_bin_capacity;

    auto parid = atomicAdd(_parcnt, 1);
    /// pos
    parray.val(_0, parid) = source_bin.val(_0, _source_pidib);
    parray.val(_1, parid) = source_bin.val(_1, _source_pidib);
    parray.val(_2, parid) = source_bin.val(_2, _source_pidib);
  }
}


/// Functions to retrieve particle outputs (JB)
/// Copies from particle buffer to two particle arrays (device --> device)
/// Depends on material model, copy/paste/modify function for new material
template <typename Partition, typename ParticleArray>
__global__ void
retrieve_particle_buffer_attributes(Partition partition,
                                         Partition prev_partition,
                                         ParticleBuffer<material_e::JFluid> pbuffer,
                                         ParticleArray parray, 
                                         ParticleArray pattrib,
                                         float *dispVal, 
                                         int *_parcnt) {
  int pcnt = partition._ppbs[blockIdx.x];
  ivec3 blockid = partition._activeKeys[blockIdx.x];
  auto advection_bucket =
      partition._blockbuckets + blockIdx.x * g_particle_num_per_block;
  for (int pidib = threadIdx.x; pidib < pcnt; pidib += blockDim.x) {
    auto advect = advection_bucket[pidib];
    ivec3 source_blockid;
    dir_components(advect / g_particle_num_per_block, source_blockid);
    source_blockid += blockid;
    auto source_blockno = prev_partition.query(source_blockid);
    auto source_pidib = advect % g_particle_num_per_block;
    auto source_bin = pbuffer.ch(_0, prev_partition._binsts[source_blockno] +
                                         source_pidib / g_bin_capacity);
    auto _source_pidib = source_pidib % g_bin_capacity;

    /// Increase particle ID
    auto parid = atomicAdd(_parcnt, 1);
    
    /// Send positions (x,y,z) [0.0, 1.0] to parray (device --> device)
    parray.val(_0, parid) = source_bin.val(_0, _source_pidib) * g_length;
    parray.val(_1, parid) = source_bin.val(_1, _source_pidib) * g_length;
    parray.val(_2, parid) = source_bin.val(_2, _source_pidib) * g_length;

    if (1) {
      /// Send attributes (J, P, P - Patm) to pattribs (device --> device)
      float J = source_bin.val(_3, _source_pidib);
      float pressure = (pbuffer.bulk / pbuffer.gamma) * 
        (powf(J, -pbuffer.gamma) - 1.f);       //< Tait-Murnaghan Pressure (Pa)
      pattrib.val(_0, parid) = J;              //< J (V/Vo)
      pattrib.val(_1, parid) = pressure;       //< Pressure (Pa)
      pattrib.val(_2, parid) = (float)pcnt;    //< Particle count for block (#)
    }
  }
}

template <typename Partition, typename ParticleArray>
__global__ void
retrieve_particle_buffer_attributes(Partition partition,
                                         Partition prev_partition,
                                         ParticleBuffer<material_e::JFluid_ASFLIP> pbuffer,
                                         ParticleArray parray, 
                                         ParticleArray pattrib,
                                         float *dispVal, 
                                         int *_parcnt) {
  int pcnt = partition._ppbs[blockIdx.x];
  ivec3 blockid = partition._activeKeys[blockIdx.x];
  auto advection_bucket =
      partition._blockbuckets + blockIdx.x * g_particle_num_per_block;
  for (int pidib = threadIdx.x; pidib < pcnt; pidib += blockDim.x) {
    auto advect = advection_bucket[pidib];
    ivec3 source_blockid;
    dir_components(advect / g_particle_num_per_block, source_blockid);
    source_blockid += blockid;
    auto source_blockno = prev_partition.query(source_blockid);
    auto source_pidib = advect % g_particle_num_per_block;
    auto source_bin = pbuffer.ch(_0, prev_partition._binsts[source_blockno] +
                                         source_pidib / g_bin_capacity);
    auto _source_pidib = source_pidib % g_bin_capacity;

    /// Increase particle ID
    auto parid = atomicAdd(_parcnt, 1);
    
    /// Send positions (x,y,z) [m] to parray (device --> device)
    parray.val(_0, parid) = source_bin.val(_0, _source_pidib) * g_length;
    parray.val(_1, parid) = source_bin.val(_1, _source_pidib) * g_length;
    parray.val(_2, parid) = source_bin.val(_2, _source_pidib) * g_length;

    if (1) {
      /// Send attributes (J, P, Vel_y) to pattribs (device --> device)
      float J = source_bin.val(_3, _source_pidib);
      float pressure = (pbuffer.bulk / pbuffer.gamma) * 
        (powf(J, -pbuffer.gamma) - 1.f);       //< Tait-Murnaghan Pressure (Pa)
      pattrib.val(_0, parid) = J;              //< J (V/Vo)
      pattrib.val(_1, parid) = pressure;       //< Pressure (Pa)
      pattrib.val(_2, parid) = source_bin.val(_5, _source_pidib) * g_length; // Vel_y
    }
    if (0) {
      /// Send attributes (Vel_x, Vel_y, Vel_z) to pattribs (device --> device)
      pattrib.val(_0, parid) = source_bin.val(_4, _source_pidib) * g_length; // Vel_x
      pattrib.val(_1, parid) = source_bin.val(_5, _source_pidib) * g_length; // Vel_y
      pattrib.val(_2, parid) = source_bin.val(_6, _source_pidib) * g_length; // Vel_z
    }
  }
}

template <typename Partition, typename ParticleArray>
__global__ void
retrieve_particle_buffer_attributes(Partition partition,
                                         Partition prev_partition,
                                         ParticleBuffer<material_e::FixedCorotated> pbuffer,
                                         ParticleArray parray, 
                                         ParticleArray pattrib,
                                         float *dispVal, 
                                         int *_parcnt) {
  int pcnt = partition._ppbs[blockIdx.x];
  ivec3 blockid = partition._activeKeys[blockIdx.x];
  auto advection_bucket =
      partition._blockbuckets + blockIdx.x * g_particle_num_per_block;
  for (int pidib = threadIdx.x; pidib < pcnt; pidib += blockDim.x) {
    auto advect = advection_bucket[pidib];
    ivec3 source_blockid;
    dir_components(advect / g_particle_num_per_block, source_blockid);
    source_blockid += blockid;
    auto source_blockno = prev_partition.query(source_blockid);
    auto source_pidib = advect % g_particle_num_per_block;
    auto source_bin = pbuffer.ch(_0, prev_partition._binsts[source_blockno] +
                                         source_pidib / g_bin_capacity);
    auto _source_pidib = source_pidib % g_bin_capacity;

    /// Increase particle ID
    auto parid = atomicAdd(_parcnt, 1);
    
    /// Send positions (x,y,z) [0.0, 1.0] to parray (device --> device)
    parray.val(_0, parid) = source_bin.val(_0, _source_pidib) * g_length;
    parray.val(_1, parid) = source_bin.val(_1, _source_pidib) * g_length;
    parray.val(_2, parid) = source_bin.val(_2, _source_pidib) * g_length;

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
      float J = F[0]*F[4]*F[8] + F[3]*F[7]*F[2] + 
                F[6]*F[1]*F[5] - F[6]*F[4]*F[2] - 
                F[3]*F[1]*F[8]; //< J = V/Vo = ||F||
      I2 = U[0]*U[4] + U[4]*U[8] + 
           U[0]*U[8] - U[3]*U[3] - 
           U[6]*U[6] - U[7]*U[7]; //< I2 = 1/2((tr(C))^2 - tr(C^2))
      // I3 = U[0]*U[4]*U[8] - U[0]*U[7]*U[7] - 
      //      U[4]*U[6]*U[6] - U[8]*U[3]*U[3] + 
      //      U[3]*U[6]*U[7]*2.f;      //< I3 = ||C||
      I3= U[0]*U[4]*U[8] + U[3]*U[7]*U[2] + 
          U[6]*U[1]*U[5] - U[6]*U[4]*U[2] - 
          U[3]*U[1]*U[8]; //< J = V/Vo = ||F||
      // Set pattribs for particle to Principal Strain Invariants
      pattrib.val(_0, parid) = J;
      pattrib.val(_1, parid) = I2;
      pattrib.val(_2, parid) = I3;
    }
  }
}


template <typename Partition, typename ParticleArray>
__global__ void
retrieve_particle_buffer_attributes(Partition partition,
                                         Partition prev_partition,
                                         ParticleBuffer<material_e::FixedCorotated_ASFLIP> pbuffer,
                                         ParticleArray parray, 
                                         ParticleArray pattrib,
                                         float *dispVal, 
                                         int *_parcnt) {
  int pcnt = partition._ppbs[blockIdx.x];
  ivec3 blockid = partition._activeKeys[blockIdx.x];
  auto advection_bucket =
      partition._blockbuckets + blockIdx.x * g_particle_num_per_block;
  for (int pidib = threadIdx.x; pidib < pcnt; pidib += blockDim.x) {
    auto advect = advection_bucket[pidib];
    ivec3 source_blockid;
    dir_components(advect / g_particle_num_per_block, source_blockid);
    source_blockid += blockid;
    auto source_blockno = prev_partition.query(source_blockid);
    auto source_pidib = advect % g_particle_num_per_block;
    auto source_bin = pbuffer.ch(_0, prev_partition._binsts[source_blockno] +
                                         source_pidib / g_bin_capacity);
    auto _source_pidib = source_pidib % g_bin_capacity;

    /// Increase particle ID
    auto parid = atomicAdd(_parcnt, 1);
    
    /// Send positions (x,y,z) [0.0, 1.0] to parray (device --> device)
    parray.val(_0, parid) = source_bin.val(_0, _source_pidib) * g_length;
    parray.val(_1, parid) = source_bin.val(_1, _source_pidib) * g_length;
    parray.val(_2, parid) = source_bin.val(_2, _source_pidib) * g_length;

    if (1) {
      /// Send attributes to pattribs (device --> device) 
      pattrib.val(_0, parid) = source_bin.val(_12, _source_pidib) * g_length; //< vx
      pattrib.val(_1, parid) = source_bin.val(_13, _source_pidib) * g_length; //< vy
      pattrib.val(_2, parid) = source_bin.val(_14, _source_pidib) * g_length; //< vz
    }
  }
}

template <typename Partition, typename ParticleArray>
__global__ void
retrieve_particle_buffer_attributes(Partition partition,
                                         Partition prev_partition,
                                         ParticleBuffer<material_e::NACC> pbuffer,
                                         ParticleArray parray, 
                                         ParticleArray pattrib,
                                         float *dispVal, 
                                         int *_parcnt) {
  int pcnt = partition._ppbs[blockIdx.x];
  ivec3 blockid = partition._activeKeys[blockIdx.x];
  auto advection_bucket =
      partition._blockbuckets + blockIdx.x * g_particle_num_per_block;
  for (int pidib = threadIdx.x; pidib < pcnt; pidib += blockDim.x) {
    auto advect = advection_bucket[pidib];
    ivec3 source_blockid;
    dir_components(advect / g_particle_num_per_block, source_blockid);
    source_blockid += blockid;
    auto source_blockno = prev_partition.query(source_blockid);
    auto source_pidib = advect % g_particle_num_per_block;
    auto source_bin = pbuffer.ch(_0, prev_partition._binsts[source_blockno] +
                                         source_pidib / g_bin_capacity);
    auto _source_pidib = source_pidib % g_bin_capacity;

    /// Increase particle ID
    auto parid = atomicAdd(_parcnt, 1);
    
    /// Send positions (x,y,z) [0.0, 1.0] to parray (device --> device)
    parray.val(_0, parid) = source_bin.val(_0, _source_pidib) * g_length;
    parray.val(_1, parid) = source_bin.val(_1, _source_pidib) * g_length;
    parray.val(_2, parid) = source_bin.val(_2, _source_pidib) * g_length;

    if (1) {
      /// Send attributes to pattribs (device --> device)
      pattrib.val(_0, parid) = 0.f; 
      pattrib.val(_1, parid) = 0.f; 
      pattrib.val(_2, parid) = (float)pcnt;    //< Particle count for block (#)
    }
  }
}

template <typename Partition, typename ParticleArray>
__global__ void
retrieve_particle_buffer_attributes(Partition partition,
                                         Partition prev_partition,
                                         ParticleBuffer<material_e::Sand> pbuffer,
                                         ParticleArray parray, 
                                         ParticleArray pattrib,
                                         float *dispVal, 
                                         int *_parcnt) {
  int pcnt = partition._ppbs[blockIdx.x];
  ivec3 blockid = partition._activeKeys[blockIdx.x];
  auto advection_bucket =
      partition._blockbuckets + blockIdx.x * g_particle_num_per_block;
  for (int pidib = threadIdx.x; pidib < pcnt; pidib += blockDim.x) {
    auto advect = advection_bucket[pidib];
    ivec3 source_blockid;
    dir_components(advect / g_particle_num_per_block, source_blockid);
    source_blockid += blockid;
    auto source_blockno = prev_partition.query(source_blockid);
    auto source_pidib = advect % g_particle_num_per_block;
    auto source_bin = pbuffer.ch(_0, prev_partition._binsts[source_blockno] +
                                         source_pidib / g_bin_capacity);
    auto _source_pidib = source_pidib % g_bin_capacity;

    /// Increase particle ID
    auto parid = atomicAdd(_parcnt, 1);
    
    /// Send positions (x,y,z) [0.0, 1.0] to parray (device --> device)
    parray.val(_0, parid) = source_bin.val(_0, _source_pidib) * g_length;
    parray.val(_1, parid) = source_bin.val(_1, _source_pidib) * g_length;
    parray.val(_2, parid) = source_bin.val(_2, _source_pidib) * g_length;

    if (1) {
      /// Send attributes to pattribs (device --> device)
      pattrib.val(_0, parid) = 0.f;    
      pattrib.val(_1, parid) = 0.f; 
      pattrib.val(_2, parid) = (float)pcnt;    //< Particle count for block (#)
    }
  }
}

template <typename Partition, typename ParticleArray>
__global__ void
retrieve_particle_buffer_attributes(Partition partition,
                                         Partition prev_partition,
                                         ParticleBuffer<material_e::Meshed> pbuffer,
                                         ParticleArray parray, 
                                         ParticleArray pattrib,
                                         float *dispVal, 
                                         int *_parcnt) {
  int pcnt = partition._ppbs[blockIdx.x];
  ivec3 blockid = partition._activeKeys[blockIdx.x];
  auto advection_bucket =
      partition._blockbuckets + blockIdx.x * g_particle_num_per_block;
  for (int pidib = threadIdx.x; pidib < pcnt; pidib += blockDim.x) {
    auto advect = advection_bucket[pidib];
    ivec3 source_blockid;
    dir_components(advect / g_particle_num_per_block, source_blockid);
    source_blockid += blockid;
    auto source_blockno = prev_partition.query(source_blockid);
    auto source_pidib = advect % g_particle_num_per_block;
    auto source_bin = pbuffer.ch(_0, prev_partition._binsts[source_blockno] +
                                         source_pidib / g_bin_capacity);
    auto _source_pidib = source_pidib % g_bin_capacity;

    /// Increase particle ID
    auto parid = atomicAdd(_parcnt, 1);
    
    /// Send positions (x,y,z) [0.0, 1.0] to parray (device --> device)
    parray.val(_0, parid) = source_bin.val(_0, _source_pidib) * g_length;
    parray.val(_1, parid) = source_bin.val(_1, _source_pidib) * g_length;
    parray.val(_2, parid) = source_bin.val(_2, _source_pidib) * g_length;

    if (1) {
      /// Send attributes to pattribs (device --> device)
      // pattrib.val(_0, parid) = source_bin.val(_3, _source_pidib); //< ID
      // pattrib.val(_1, parid) = source_bin.val(_4, _source_pidib) * g_length; //< vel_x
      // pattrib.val(_2, parid) = source_bin.val(_7, _source_pidib); //< mass or J/Tension
      pattrib.val(_0, parid) = source_bin.val(_3, _source_pidib); //< ID
      pattrib.val(_1, parid) = source_bin.val(_7, _source_pidib); //< vol
      pattrib.val(_2, parid) = source_bin.val(_4, _source_pidib) * g_length; //< v_x
    }

    float offset = 8.f * g_dx;

    /// Set x value  host csv output
    auto part_ID = source_bin.val(_3, _source_pidib);
    if (part_ID == g_track_ID){
      float dv = (source_bin.val(_0, _source_pidib) - offset) * g_length ;
      atomicAdd(dispVal, dv);
    }

  }
}

/// Retrieve grid-cells between points a & b from grid-buffer to gridTarget (JB)
template <typename Partition, typename Grid, typename GridTarget>
__global__ void retrieve_selected_grid_cells(
    uint32_t blockCount, const Partition partition,
    Grid prev_grid, GridTarget garray,
    float dt, float *forceSum, vec3 point_a, vec3 point_b) {

  auto blockno = blockIdx.x;  //< Block number in partition
  auto blockid = partition._activeKeys[blockno];

  // Check if gridblock contains part of the point_a to point_b region
  // End all threads in block if not
  if ((4.f*blockid[0] + 3.f)*g_dx < point_a[0] || (4.f*blockid[0])*g_dx > point_b[0]) return;
  if ((4.f*blockid[1] + 3.f)*g_dx < point_a[1] || (4.f*blockid[1])*g_dx > point_b[1]) return;
  if ((4.f*blockid[2] + 3.f)*g_dx < point_a[2] || (4.f*blockid[2])*g_dx > point_b[2]) return;
  

  //auto blockid = prev_blockids[blockno]; //< 3D grid-block index
  if (blockno < blockCount) {

    auto sourceblock = prev_grid.ch(_0, blockno); //< Set grid-block by block index
    float tol = g_dx * 0.0f; // Tolerance layer around target domain

    // Add +1 to each? For point_b ~= point_a...
    ivec3 maxNodes_coord;
    maxNodes_coord[0] = (int)((point_b[0] + tol - point_a[0] + tol) * g_dx_inv + 1);
    maxNodes_coord[1] = (int)((point_b[1] + tol - point_a[1] + tol) * g_dx_inv + 1);
    maxNodes_coord[2] = (int)((point_b[2] + tol - point_a[2] + tol) * g_dx_inv + 1);
    int maxNodes = maxNodes_coord[0] * maxNodes_coord[1] * maxNodes_coord[2];
    if (maxNodes >= g_target_cells && threadIdx.x == 0) printf("Allocate more space for gridTarget!\n");

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
      if (xc < point_a[0] - tol || xc > point_b[0] + tol) continue;
      if (yc < point_a[1] - tol || yc > point_b[1] + tol) continue;
      if (zc < point_a[2] - tol || zc > point_b[2] + tol) continue;
      

      // Unique ID by spatial position of cell in target [0 to g_target_cells-1]
      int node_id;
      node_id = ((int)((xc - point_a[0] + tol) * g_dx_inv) * maxNodes_coord[1] * maxNodes_coord[2]) +
                ((int)((yc - point_a[1] + tol) * g_dx_inv) * maxNodes_coord[2]) +
                ((int)((zc - point_a[2] + tol) * g_dx_inv));
      // while (garray.val(_3, node_id) != 0.f) {
      //   node_id += 1;
      //   if (node_id > g_target_cells) {
      //     printf("node_id bigger than g_target_cells!");
      //     break;
      //   }
      // }
      //__syncthreads(); // Sync threads in block

      /// Set values in grid-array to specific cell from grid-buffer
      garray.val(_0, node_id) = xc * g_length;
      garray.val(_1, node_id) = yc * g_length;
      garray.val(_2, node_id) = zc * g_length;
      garray.val(_3, node_id) = sourceblock.val(_0, i, j, k);
      garray.val(_4, node_id) = sourceblock.val(_1, i, j, k) * g_length;
      garray.val(_5, node_id) = sourceblock.val(_2, i, j, k) * g_length;
      garray.val(_6, node_id) = sourceblock.val(_3, i, j, k) * g_length;

      /// Set values in grid-array to specific cell from grid-buffer
      float m1  = garray.val(_3, node_id);
      if (m1 <= 0.f) continue;
      float m2  = m1;
      float m = m1;
      m1 = 1.f / m1; //< Invert mass, avoids division operator
      m2 = 1.f / m2; //< Invert mass, avoids division operator
      

      float vx1 = garray.val(_4, node_id) * m1;
      float vy1 = garray.val(_5, node_id) * m1;
      float vz1 = garray.val(_6, node_id) * m1;
      float vx2 = 0.f;
      float vy2 = 0.f;
      float vz2 = 0.f;

      float fx = m * (vx1 - vx2) / dt;
      float fy = m * (vy1 - vy2) / dt;
      float fz = m * (vz1 - vz2) / dt;
      

      // Set load direction x/y/z
      // if ( dir_val == 0 || dir_val == 1 || dir_val == 2 ) force = fx;
      // else if ( dir_val == 3 || dir_val == 4 || dir_val == 5 ) force = fy;
      // else if ( dir_val == 6 || dir_val == 7 || dir_val == 8 ) force = fz;
      // else force = 0.f;
      // // Seperation -+/-/+ condition for load
      // if ((dir_val % 3) == 1) {
      //   if (force >= 0.f) continue;
      // } else if ((dir_val % 3) == 2) {
      //   if (force <= 0.f) continue;
      // }

      garray.val(_7, node_id) = fx;
      garray.val(_8, node_id) = fy;
      garray.val(_9, node_id) = fz; 

      float force;
      force = fx;
      if (force <= 0.f) continue;
      atomicAdd(forceSum, force);
    }
  }
}

/// Retrieve wave-gauge surface elevation between points a & b from grid-buffer to waveMax (JB)
template <typename Partition, typename Grid>
__global__ void retrieve_wave_gauge(
    uint32_t blockCount, const Partition partition,
    Grid prev_grid,
    float dt, float *waveMax, vec3 point_a, vec3 point_b) {

  auto blockno = blockIdx.x;  //< Block number in partition
  //if (blockno == -1) return;
  auto blockid = partition._activeKeys[blockno];

  // Check if gridblock contains part of the point_a to point_b region
  // End all threads in block if not
  if ((4.f*blockid[0] + 3.f)*g_dx < point_a[0] || (4.f*blockid[0])*g_dx > point_b[0]) return;
  if ((4.f*blockid[1] + 3.f)*g_dx < point_a[1] || (4.f*blockid[1])*g_dx > point_b[1]) return;
  if ((4.f*blockid[2] + 3.f)*g_dx < point_a[2] || (4.f*blockid[2])*g_dx > point_b[2]) return;
  

  //auto blockid = prev_blockids[blockno]; //< 3D grid-block index
  if (blockno < blockCount) {
    auto sourceblock = prev_grid.ch(_0, blockno); //< Set grid-block by block index

    // Tolerance layer thickness around wg space
    float tol = g_dx * 0.0f;

    // Loop through cells in grid-block, stride by 32 to avoid thread conflicts
    for (int cidib = threadIdx.x % 32; cidib < g_blockvolume; cidib += 32) {

      // Grid node coordinate [i,j,k] in grid-block
      int i = (cidib >> (g_blockbits << 1)) & g_blockmask;
      int j = (cidib >> g_blockbits) & g_blockmask;
      int k = cidib & g_blockmask;

      // Grid node position [x,y,z] in entire domain 
      float xc = (4*blockid[0]*g_dx) + (i*g_dx);
      float yc = (4*blockid[1]*g_dx) + (j*g_dx);
      float zc = (4*blockid[2]*g_dx) + (k*g_dx);
      float offset = (8.f * g_dx);

      // Exit thread if cell is not inside wave-gauge domain +/- tol
      if (xc < point_a[0] - tol || xc > point_b[0] + tol) continue;
      if (yc < point_a[1] - tol || yc > point_b[1] + tol) continue;
      if (zc < point_a[2] - tol || zc > point_b[2] + tol) continue;

      /// Set values of cell (mass, momentum) from grid-buffer
      float mass = sourceblock.val(_0, i, j, k); // Mass [kg]
      if (mass <= 0.f) continue;
      float elev = (yc - offset) * g_length; // Elevation [m]

      // Check for mass (material in cell, i.e wave surface)
      atomicMax(waveMax, elev);
    }
  }

}

} // namespace mn

#endif