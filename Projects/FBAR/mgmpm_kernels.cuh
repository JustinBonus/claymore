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
      int(std::lround(parray.val(_0, parid) * g_dx_inv) - 2) / g_blocksize,
      int(std::lround(parray.val(_1, parid) * g_dx_inv) - 2) / g_blocksize,
      int(std::lround(parray.val(_2, parid) * g_dx_inv) - 2) / g_blocksize};
  partition.insert(blockid);
}
template <typename ParticleArray, typename Partition>
__global__ void build_particle_cell_buckets(uint32_t particleCount,
                                            ParticleArray parray,
                                            Partition partition) {
  uint32_t parid = blockIdx.x * blockDim.x + threadIdx.x;
  if (parid >= particleCount)
    return;
  ivec3 coord{int(std::lround(parray.val(_0, parid) * g_dx_inv) - 2),
              int(std::lround(parray.val(_1, parid) * g_dx_inv) - 2),
              int(std::lround(parray.val(_2, parid) * g_dx_inv) - 2)};
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
    // Mass, Mass*(vel + dt * fint)
    gridblock.val_1d(_0, cidib) = 0.0;
    gridblock.val_1d(_1, cidib) = 0.0;
    gridblock.val_1d(_2, cidib) = 0.0;
    gridblock.val_1d(_3, cidib) = 0.0;
    // Mass*vel [ASFLIP, FLIP]
    gridblock.val_1d(_4, cidib) = 0.0;
    gridblock.val_1d(_5, cidib) = 0.0;
    gridblock.val_1d(_6, cidib) = 0.0;
    // Vol, JBar [Simple FBar]
    gridblock.val_1d(_7, cidib) = 0.0;
    gridblock.val_1d(_8, cidib) = 0.0;
  }
}
template <typename Grid> __global__ void clear_grid_FBar(Grid grid) {
  auto gridblock = grid.ch(_0, blockIdx.x);
  for (int cidib = threadIdx.x; cidib < g_blockvolume; cidib += blockDim.x) {
    // Vol, JBar [Simple FBar]
    gridblock.val_1d(_7, cidib) = 0.0;
    gridblock.val_1d(_8, cidib) = 0.0;
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
                          PREC mass, PREC volume, pvec3 vel0, PREC length) {
  uint32_t parid = blockIdx.x * blockDim.x + threadIdx.x;
  if (parid >= particleCount)
    return;

  pvec3 local_pos{(PREC)parray.val(_0, parid), (PREC)parray.val(_1, parid),
                 (PREC)parray.val(_2, parid)};
  pvec3 vel;
  pvec9 contrib, C;
  vel.set(0.), contrib.set(0.), C.set(0.);

  vel[0] = vel0[0];
  vel[1] = vel0[1];
  vel[2] = vel0[2];

  // Dp^n = Dp^n+1 = (1/4) * dx^2 * I (Quad.)
  PREC Dp_inv; //< Inverse Intertia-Like Tensor (1/m^2)
  PREC scale = length * length; //< Area scale (m^2)
  Dp_inv = g_D_inv / scale;     //< Scalar 4/(dx^2) for Quad. B-Spline
  contrib = (C * mass - contrib * dt) * Dp_inv;
  ivec3 global_base_index{int(std::lround(local_pos[0] * g_dx_inv) - 1),
                          int(std::lround(local_pos[1] * g_dx_inv) - 1),
                          int(std::lround(local_pos[2] * g_dx_inv) - 1)};
  local_pos = local_pos - global_base_index * g_dx;
  vec<vec<PREC, 3>, 3> dws;
  //pvec3x3 dws;
  for (int d = 0; d < 3; ++d)
    dws[d] = bspline_weight((PREC)local_pos[d]);
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j)
      for (int k = 0; k < 3; ++k) {
        ivec3 offset{i, j, k};
        pvec3 xixp = offset * g_dx - local_pos;
        PREC W = dws[0][i] * dws[1][j] * dws[2][k];
        ivec3 local_index = global_base_index + offset;
        PREC wm = mass * W;
        PREC wv = volume * W;
        PREC J = 1.0; // Volume ratio, Det def. gradient. 1 for t0 
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
        // Simple FBar: Vol, Vol JBar
        atomicAdd(
            &grid_block.val(_7, local_index[0], local_index[1], local_index[2]),
            wv);
        atomicAdd(
            &grid_block.val(_8, local_index[0], local_index[1], local_index[2]),
            wv * (1.0 - J));
      }
}

// %% ============================================================= %%
//     MPM-FEM Precompute Element Quantities
// %% ============================================================= %%

template <typename VerticeArray, typename ElementArray>
__global__ void fem_precompute(VerticeArray vertice_array,
                               ElementArray element_array,
                               ElementBuffer<fem_e::Tetrahedron> elementBins) {
    if (blockIdx.x >= g_max_fem_element_num) return;
    auto element = elementBins.ch(_0, blockIdx.x);
    int IDs[4];
    pvec3 p[4];
    pvec9 D, Dinv;
    PREC restVolume;
    IDs[0] = element_array.val(_0, blockIdx.x);
    IDs[1] = element_array.val(_1, blockIdx.x);
    IDs[2] = element_array.val(_2, blockIdx.x);
    IDs[3] = element_array.val(_3, blockIdx.x);

    for (int v = 0; v < 4; v++) {
      int ID = IDs[v] - 1;
      p[v][0] = vertice_array.val(_0, ID) * elementBins.length;
      p[v][1] = vertice_array.val(_1, ID) * elementBins.length;
      p[v][2] = vertice_array.val(_2, ID) * elementBins.length;
    }
    D.set(0.0);
    D[0] = p[1][0] - p[0][0];
    D[1] = p[1][1] - p[0][1];
    D[2] = p[1][2] - p[0][2];
    D[3] = p[2][0] - p[0][0];
    D[4] = p[2][1] - p[0][1];
    D[5] = p[2][2] - p[0][2];
    D[6] = p[3][0] - p[0][0];
    D[7] = p[3][1] - p[0][1];
    D[8] = p[3][2] - p[0][2];
    
    int sV0 = 0;
    restVolume = matrixDeterminant3d(D.data()) / 6.0;
    sV0 = (restVolume > 0.f) - (restVolume < 0.f);
    if (sV0 < 0.f) printf("Element %d inverted volume! \n", blockIdx.x);

    Dinv.set(0.0);
    matrixInverse(D.data(), Dinv.data());
    {
      element.val(_0, 0) = IDs[0];
      element.val(_1, 0) = IDs[1];
      element.val(_2, 0) = IDs[2];
      element.val(_3, 0) = IDs[3];
      element.val(_4,  0) = Dinv[0];
      element.val(_5,  0) = Dinv[1];
      element.val(_6,  0) = Dinv[2];
      element.val(_7,  0) = Dinv[3];
      element.val(_8,  0) = Dinv[4];
      element.val(_9,  0) = Dinv[5];
      element.val(_10, 0) = Dinv[6];
      element.val(_11, 0) = Dinv[7];
      element.val(_12, 0) = Dinv[8];
      element.val(_13, 0) = restVolume;
    }
}


template <typename VerticeArray, typename ElementArray>
__global__ void fem_precompute(VerticeArray vertice_array,
                               ElementArray element_array,
                               ElementBuffer<fem_e::Tetrahedron_FBar> elementBins) {
    if (blockIdx.x >= g_max_fem_element_num) return;
    auto element = elementBins.ch(_0, blockIdx.x);
    int IDs[4];
    pvec3 p[4];
    pvec9 D, Dinv;
    PREC restVolume;
    IDs[0] = element_array.val(_0, blockIdx.x);
    IDs[1] = element_array.val(_1, blockIdx.x);
    IDs[2] = element_array.val(_2, blockIdx.x);
    IDs[3] = element_array.val(_3, blockIdx.x);

    for (int v = 0; v < 4; v++) {
      int ID = IDs[v] - 1;
      p[v][0] = vertice_array.val(_0, ID) * elementBins.length;
      p[v][1] = vertice_array.val(_1, ID) * elementBins.length;
      p[v][2] = vertice_array.val(_2, ID) * elementBins.length;
    }
    D.set(0.0);
    D[0] = p[1][0] - p[0][0];
    D[1] = p[1][1] - p[0][1];
    D[2] = p[1][2] - p[0][2];
    D[3] = p[2][0] - p[0][0];
    D[4] = p[2][1] - p[0][1];
    D[5] = p[2][2] - p[0][2];
    D[6] = p[3][0] - p[0][0];
    D[7] = p[3][1] - p[0][1];
    D[8] = p[3][2] - p[0][2];
    
    int sV0 = 0;
    restVolume = matrixDeterminant3d(D.data()) / 6.0;
    sV0 = (restVolume > 0.f) - (restVolume < 0.f);
    if (sV0 < 0.f) printf("Element %d inverted volume! \n", blockIdx.x);

    PREC J = 1.0;
    PREC JBar = 1.0;
    Dinv.set(0.0);
    matrixInverse(D.data(), Dinv.data());
    {
      element.val(_0, 0) = IDs[0];
      element.val(_1, 0) = IDs[1];
      element.val(_2, 0) = IDs[2];
      element.val(_3, 0) = IDs[3];
      element.val(_4,  0) = Dinv[0];
      element.val(_5,  0) = Dinv[1];
      element.val(_6,  0) = Dinv[2];
      element.val(_7,  0) = Dinv[3];
      element.val(_8,  0) = Dinv[4];
      element.val(_9,  0) = Dinv[5];
      element.val(_10, 0) = Dinv[6];
      element.val(_11, 0) = Dinv[7];
      element.val(_12, 0) = Dinv[8];
      element.val(_13, 0) = restVolume;
      element.val(_14, 0) = 1.0 - J;
      element.val(_15, 0) = 1.0 - JBar;
    }
}

template <typename VerticeArray, typename ElementArray>
__global__ void fem_precompute(VerticeArray vertice_array,
                               ElementArray element_array,
                               ElementBuffer<fem_e::Brick> elementBins) {
                                 return;
}

// %% ============================================================= %%
//     MPM Move Particles From Device Array to Particle Buffers
// %% ============================================================= %%


template <typename ParticleArray, typename Partition>
__global__ void array_to_buffer(ParticleArray parray,
                                ParticleBuffer<material_e::JFluid> pbuffer,
                                Partition partition, vec<PREC, 3> vel) {
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
                                Partition partition, vec<PREC, 3> vel) {
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
    pbin.val(_3, pidib % g_bin_capacity) = 0.0; //< 1 - J , 1 - V/Vo, 
    /// vel
    pbin.val(_4, pidib % g_bin_capacity) = vel[0]; //< Vel_x m/s
    pbin.val(_5, pidib % g_bin_capacity) = vel[1]; //< Vel_y m/s
    pbin.val(_6, pidib % g_bin_capacity) = vel[2]; //< Vel_z m/s

  }
}

template <typename ParticleArray, typename Partition>
__global__ void array_to_buffer(ParticleArray parray,
                                ParticleBuffer<material_e::JBarFluid> pbuffer,
                                Partition partition, vec<PREC, 3> vel) {
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
    pbin.val(_3, pidib % g_bin_capacity) = 0.0; //< (1 - J) = (1 - V/Vo)
    /// vel (ASFLIP)
    pbin.val(_4, pidib % g_bin_capacity) = vel[0]; //< Vel_x m/s
    pbin.val(_5, pidib % g_bin_capacity) = vel[1]; //< Vel_y m/s
    pbin.val(_6, pidib % g_bin_capacity) = vel[2]; //< Vel_z m/s
    /// Vol, J (Simple FBar)
    pbin.val(_7, pidib % g_bin_capacity) = pbuffer.volume; //< Vol_0
    pbin.val(_8, pidib % g_bin_capacity) = 0.0; //< JBar
  }
}

template <typename ParticleArray, typename Partition>
__global__ void
array_to_buffer(ParticleArray parray,
                ParticleBuffer<material_e::FixedCorotated> pbuffer,
                Partition partition, vec<PREC, 3> vel) {
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
                Partition partition, vec<PREC, 3> vel) {
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
                                Partition partition, vec<PREC, 3> vel) {
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
                                Partition partition, vec<PREC, 3> vel) {
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
                                Partition partition, vec<PREC, 3> vel) {
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
    pbin.val(_7, pidib % g_bin_capacity) = 0.0; //< J
    // Normals
    pbin.val(_8, pidib % g_bin_capacity)  = 0.0; //< b_x   
    pbin.val(_9, pidib % g_bin_capacity)  = 0.0; //< b_y
    pbin.val(_10, pidib % g_bin_capacity) = 0.0; //< b_z
  }
}

// %% ============================================================= %%
//     MPM Grid Update Functions
// %% ============================================================= %%

template <typename Grid, typename Partition>
__global__ void update_grid_velocity_query_max(uint32_t blockCount, Grid grid,
                                               Partition partition, float dt,
                                               float *maxVel, float curTime,
                                               float grav, vec7 walls, vec7 boxes, PREC length) {
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
  int isInBuffer = ((blockid[0] < bc || blockid[0] >= g_grid_size_x - bc) << 2) |
                   ((blockid[1] < bc || blockid[1] >= g_grid_size_y - bc) << 1) |
                    (blockid[2] < bc || blockid[2] >= g_grid_size_z - bc);

  if (threadIdx.x < numWarps)
    sh_maxvels[threadIdx.x] = 0.0f;
  __syncthreads();
  
  PREC_G o = g_offset; // Domain offset [ ], for Quad. B-Splines (Off-by-2, Wang 2020)
  PREC_G l = length; // Length of domain [m]

  /// within-warp computations
  if (blockno < blockCount) {
    auto grid_block = grid.ch(_0, blockno);
    for (int cidib = threadIdx.x % 32; cidib < g_blockvolume; cidib += 32) {
      PREC_G mass = grid_block.val_1d(_0, cidib), velSqr = 0.f, vel[3], vel_n[3];
      if (mass > 0.0) {
        mass = (1.0 / mass);

        // Grid node coordinate [i,j,k] in grid-block
        int i = (cidib >> (g_blockbits << 1)) & g_blockmask;
        int j = (cidib >> g_blockbits) & g_blockmask;
        int k =  cidib & g_blockmask;
        // Grid node position [x,y,z] in entire domain
        PREC_G xc = (g_blocksize * blockid[0] + i) * g_dx; // Node x position [1]
        PREC_G yc = (g_blocksize * blockid[1] + j) * g_dx; // Node y position [1]
        PREC_G zc = (g_blocksize * blockid[2] + k) * g_dx; // Node z position [1]
        // PREC_G x  = (g_dx * (g_blocksize * blockid[0] + i) - o) * l; // Node x position [m]
        // PREC_G y  = (g_dx * (g_blocksize * blockid[1] + j) - o) * l; // Node y position [m]
        // PREC_G z  = (g_dx * (g_blocksize * blockid[2] + k) - o) * l; // Node z position [m]

        // Retrieve grid momentums (kg*m/s2)
        vel[0] = grid_block.val_1d(_1, cidib); //< mvx+dt(fint)
        vel[1] = grid_block.val_1d(_2, cidib); //< mvy+dt(fint)
        vel[2] = grid_block.val_1d(_3, cidib); //< mvz+dt(fint)
        vel_n[0] = grid_block.val_1d(_4, cidib); //< mvx
        vel_n[1] = grid_block.val_1d(_5, cidib); //< mvy
        vel_n[2] = grid_block.val_1d(_6, cidib); //< mvz

        // int isInBound = (((blockid[0] < bc && vel[0] < 0.f) || (blockid[0] >= g_grid_size_x - bc && vel[0] > 0.f)) << 2) |
        //                 (((blockid[1] < bc && vel[1] < 0.f) || (blockid[1] >= g_grid_size_y - bc && vel[1] > 0.f)) << 1) |
        //                  ((blockid[2] < bc && vel[2] < 0.f) || (blockid[2] >= g_grid_size_z - bc && vel[2] > 0.f));
        // int isInBound = (((blockid[0] < 1) || (blockid[0] >= g_grid_size_x - 1)) << 2) |
        //                 (((blockid[1] < 1) || (blockid[1] >= g_grid_size_y - 1)) << 1) |
        //                  ((blockid[2] < 1) || (blockid[2] >= g_grid_size_z - 1));
        int isInBound = 0;
        // Set boundaries of scene/flume
        gvec3 walls_dim, walls_pos;
        walls_dim[0] = walls[3] - walls[0];//6.0f;  // Length
        walls_dim[1] = walls[4] - walls[1];//6.0f;  // Depth
        walls_dim[2] = walls[5] - walls[2];//0.08f; // 0.08f; // Width
        walls_pos[0] = walls[0];
        walls_pos[1] = walls[1];
        walls_pos[2] = walls[2];

        PREC_G tol = g_dx / 100.0; // Tolerance
        tol = 0;
        // Slip
        if (walls[6] == 1) {
          int isOutFlume =  (((xc <= walls_pos[0]  + tol - 1.5*g_dx) || (xc >= walls_pos[0] + walls_dim[0] - tol + 1.5*g_dx)) << 2) |
                             (((yc <= walls_pos[1] + tol - 1.5*g_dx) || (yc >= walls_pos[1] + walls_dim[1] - tol + 1.5*g_dx)) << 1) |
                              ((zc <= walls_pos[2] + tol - 1.5*g_dx) || (zc >= walls_pos[2] + walls_dim[2] - tol + 1.5*g_dx));                          
          if (isOutFlume) isOutFlume = ((1 << 2) | (1 << 1) | 1);
          isInBound |= isOutFlume; // Update with regular boundary for efficiency
          int isSlipFlume =  (((xc <= walls_pos[0] + tol) || (xc >= walls_pos[0] + walls_dim[0] - tol)) << 2) |
                             (((yc <= walls_pos[1] + tol) || (yc >= walls_pos[1] + walls_dim[1] - tol)) << 1) |
                              ((zc <= walls_pos[2] + tol) || (zc >= walls_pos[2] + walls_dim[2] - tol));                          
          isInBound |= isSlipFlume; // Update with regular boundary for efficiency
        }
        // Seperable
        else if (walls[6] == 2) {
          int isOutFlume =  (((xc <= walls_pos[0]  + tol - 1.5*g_dx) || (xc >= walls_pos[0] + walls_dim[0] - tol + 1.5*g_dx)) << 2) |
                             (((yc <= walls_pos[1] + tol - 1.5*g_dx) || (yc >= walls_pos[1] + walls_dim[1] - tol + 1.5*g_dx)) << 1) |
                              ((zc <= walls_pos[2] + tol - 1.5*g_dx) || (zc >= walls_pos[2] + walls_dim[2] - tol + 1.5*g_dx));                          
          if (isOutFlume) isOutFlume = ((1 << 2) | (1 << 1) | 1);
          int isSepFlume = (((xc <= walls_pos[0] + tol && vel[0] < 0) || (xc >= walls_pos[0] + walls_dim[0] - tol && vel[0] > 0)) << 2) |
                           (((yc <= walls_pos[1] + tol && vel[1] < 0) || (yc >= walls_pos[1] + walls_dim[1] - tol && vel[1] > 0)) << 1) |
                            ((zc <= walls_pos[2] + tol && vel[2] < 0) || (zc >= walls_pos[2] + walls_dim[2] - tol && vel[2] > 0));                          
          isInBound |= isSepFlume; // Update with regular boundary for efficiency
        }

        // Rigid box boundary
        if (boxes[6]==2) {
          gvec3 struct_dim; //< Dimensions of structure in [m]
          struct_dim[0] = boxes[3] - boxes[0]; //(1.f);
          struct_dim[1] = boxes[4] - boxes[1]; //(1.f);
          struct_dim[2] = boxes[5] - boxes[2]; //(0.6f + tol);
          gvec3 struct_pos; //< Position of structures in [m]
          struct_pos[0] = boxes[0]; //(1.0f);
          struct_pos[1] = boxes[1]; //(-0.5f);
          struct_pos[2] = boxes[2]; //(1.52f - tol); // 1.325 rounded to nearest dx (0.02)
          PREC_G t = 1.0 * g_dx + tol; // Slip layer thickness for box

          // Check if grid-cell is within sticky interior of box
          // Subtract slip-layer thickness from structural box dimension for geometry
          int isOutStruct  = ((xc >= struct_pos[0] + t && xc < struct_pos[0] + struct_dim[0] - t) << 2) | 
                             ((yc >= struct_pos[1] + t && yc < struct_pos[1] + struct_dim[1] - t) << 1) |
                              (zc >= struct_pos[2] + t && zc < struct_pos[2] + struct_dim[2] - t);
          if (isOutStruct != 7) isOutStruct = 0; // Check if 111, reset otherwise
          isInBound |= isOutStruct; // Update with regular boundary for efficiency
          
          // Check exterior slip-layer of rigid box
          // One-cell depth, six-faces, order matters! (over-writes on edges, favors front) 
          int isOnStructFace[6];
          // Right (z+)
          isOnStructFace[0] = ((xc >= struct_pos[0] && xc <= struct_pos[0] + struct_dim[0]) << 2) | 
                              ((yc >= struct_pos[1] && yc <= struct_pos[1] + struct_dim[1]) << 1) |
                               (zc >= struct_pos[2] + struct_dim[2] - t && vel[2] < 0.f && zc < struct_pos[2] + struct_dim[2]);
          // Left (z-)
          isOnStructFace[1] = ((xc >= struct_pos[0] && xc <= struct_pos[0] + struct_dim[0]) << 2) | 
                              ((yc >= struct_pos[1] && yc <= struct_pos[1] + struct_dim[1]) << 1) |
                               (zc >= struct_pos[2] && vel[2] > 0.f && zc < struct_pos[2] + t);        
          // Top (y+)
          isOnStructFace[2] = ((xc >= struct_pos[0] && xc <= struct_pos[0] + struct_dim[0]) << 2) | 
                              ((yc >= struct_pos[1] + struct_dim[1] - t &&  vel[1] < 0.f && yc <= struct_pos[1] + struct_dim[1]) << 1) |
                               (zc >= struct_pos[2] && zc <= struct_pos[2] + struct_dim[2]);
          // Bottom (y-)
          isOnStructFace[3] = ((xc >= struct_pos[0] && xc <= struct_pos[0] + struct_dim[0]) << 2) | 
                              ((yc >= struct_pos[1] && vel[1] > 0.f && yc < struct_pos[1] + t) << 1) |
                               (zc >= struct_pos[2] && zc <= struct_pos[2] + struct_dim[2]);
          // Back (x+)
          isOnStructFace[4] = ((xc >= struct_pos[0] + struct_dim[0] - t && vel[0] < 0.f && xc < struct_pos[0] + struct_dim[0]) << 2) | 
                              ((yc >= struct_pos[1] && yc <= struct_pos[1] + struct_dim[1]) << 1) |
                               (zc >= struct_pos[2] && zc <= struct_pos[2] + struct_dim[2]);
          // Front (x-)
          isOnStructFace[5] = ((xc >= struct_pos[0] && vel[0] > 0.f && xc < struct_pos[0] + t) << 2) | 
                              ((yc >= struct_pos[1] && yc <= struct_pos[1] + struct_dim[1]) << 1) |
                               (zc >= struct_pos[2] && zc <= struct_pos[2] + struct_dim[2]);                             
          // Reduce results from box faces to single result
          int isOnStruct = 0; // Collision reduction variable
          for (int iter=0; iter<6; iter++) {
            if (isOnStructFace[iter] != 7) {
              // Check if 111 (7, all flags), set 000 (0, now no flags) otherwise
              isOnStructFace[iter] = 0;
            } else {
              // iter [0, 1, 2, 3, 4, 5] -> iter/2 [0, 1, 2] <->  [z, y, z], used as bit shift.
              isOnStructFace[iter] = (1 << iter / 2);
            }
            isOnStruct |= isOnStructFace[iter]; // OR reduces face results into one int
          }
          if (isOnStruct == 6 || isOnStruct == 5 || isOnStruct == 7) isOnStruct = 4; // Overlaps on front (XY, XZ, XYZ) -> (X)
          else if (isOnStruct == 3) isOnStruct = 0; // Overlaps on sides (YZ) -> (None)
          isInBound |= isOnStruct; // OR reduce into regular boundary for efficiency
        }


        // Wall boundary, releases after wait time
        if (0) {
          PREC_G gate = (4.f + o) / l; // Gate position [m]
          PREC_G wait = 0.25f; // Time til release [sec]
          if (curTime < wait && xc >= gate - tol) {
              vel[0] = 0.f; 
              vel_n[0] = 0.f;
          }
        }

#if 1
        ///< Slip contact        
        // Set grid node velocity
        vel[0]  = isInBound & 4 ? 0.f : vel[0] * mass; //< vx = mvx / m
        vel[1]  = isInBound & 2 ? 0.f : vel[1] * mass; //< vy = mvy / m
        vel[1] += isInBound & 2 ? 0.f : (grav / l) * dt;  //< fg = dt * g, Grav. force
        vel[2]  = isInBound & 1 ? 0.f : vel[2] * mass; //< vz = mvz / m
        vel_n[0] = isInBound & 4 ? 0.f : vel_n[0] * mass; //< vx = mvx / m
        vel_n[1] = isInBound & 2 ? 0.f : vel_n[1] * mass; //< vy = mvy / m
        vel_n[2] = isInBound & 1 ? 0.f : vel_n[2] * mass; //< vz = mvz / m
        // vel_n[0] = vel_n[0] * mass; //< vx = mvx / m
        // vel_n[1] = vel_n[1] * mass; //< vy = mvy / m
        // vel_n[2] = vel_n[2] * mass; //< vz = mvz / m
        // PREC_G vol = isInBound == 7 ? 1.0 : 0.0;
        // PREC_G JBar = isInBound == 7 ? 0.0 : 1.0;
#endif        

#if 0
        ///< Sticky contact
        if (isInBound) ///< sticky
          vel.set(0.f);
#endif

        grid_block.val_1d(_1, cidib) = vel[0];
        velSqr += vel[0] * vel[0];
        grid_block.val_1d(_2, cidib) = vel[1];
        velSqr += vel[1] * vel[1];
        grid_block.val_1d(_3, cidib) = vel[2];
        velSqr += vel[2] * vel[2];
        grid_block.val_1d(_4, cidib) = vel_n[0];
        grid_block.val_1d(_5, cidib) = vel_n[1];
        grid_block.val_1d(_6, cidib) = vel_n[2];
        grid_block.val_1d(_7, cidib) = 0;
        grid_block.val_1d(_8, cidib) = 0;
      }
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
                                               float *maxVel, float curTime,
                                               float grav) {
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
        int k = (cidib & g_blockmask);
        // Grid node position [x,y,z] in entire domain
        float xc = (4*blockid[0]*g_dx) + (i*g_dx); // + (g_dx/2.f);
        float yc = (4*blockid[1]*g_dx) + (j*g_dx); // + (g_dx/2.f);
        float zc = (4*blockid[2]*g_dx) + (k*g_dx); // + (g_dx/2.f);

        // Offset condition for Off-by-2 (see Xinlei & Fang et al.)
        // Note you should subtract 16 nodes from total
        // (or 4 grid blocks) to have total available length
        float o = g_offset;

        // Retrieve grid momentums (kg*m/s2)
        vel[0] = grid_block.val_1d(_1, cidib); //< mvx
        vel[1] = grid_block.val_1d(_2, cidib); //< mvy
        vel[2] = grid_block.val_1d(_3, cidib); //< mvz


        // Tank Dimensions
        // Acts on individual grid-cell velocities
        float flumex = 3.2f / g_length; // Actually 12m, added run-in/out
        float flumey = 6.4f / g_length; // 1.22m Depth
        float flumez = 0.4f / g_length; // 0.91m Width
        int isInFlume =  ((xc < o || xc >= flumex + o) << 2) |
                         ((yc < o || yc >= flumey + o) << 1) |
                          (zc < o || zc >= flumez + o);
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

template <typename Grid, typename Partition>
__global__ void update_grid_FBar(uint32_t blockCount, Grid grid,
                                               Partition partition, float dt,
                                                float curTime) {
  auto grid_block = grid.ch(_0, blockIdx.x);
  for (int cidib = threadIdx.x; cidib < g_blockvolume; cidib += blockDim.x) {
    // Vol, JBar [Simple FBar]
    PREC_G vol  = grid_block.val_1d(_7, cidib);
    if (vol > 0){
      PREC_G JBar = grid_block.val_1d(_8, cidib);
      grid_block.val_1d(_7, cidib) = 0.f;
      grid_block.val_1d(_8, cidib) = JBar / vol;
    }
  }
}


// %% ============================================================= %%
//     MPM Grid-to-Particle-to-Grid Functions
// %% ============================================================= %%

template <typename ParticleBuffer, typename Partition, typename Grid, typename VerticeArray>
__global__ void g2p2g(float dt, float newDt, const ivec3 *__restrict__ blocks,
                      const ParticleBuffer pbuffer,
                      ParticleBuffer next_pbuffer,
                      const Partition prev_partition, Partition partition,
                      const Grid grid, Grid next_grid,
                      VerticeArray vertice_array) { }

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
    pvec3 pos;
    PREC J;
    {
      auto source_particle_bin = pbuffer.ch(_0, source_blockno);
      pos[0] = source_particle_bin.val(_0, source_pidib % g_bin_capacity);
      pos[1] = source_particle_bin.val(_1, source_pidib % g_bin_capacity);
      pos[2] = source_particle_bin.val(_2, source_pidib % g_bin_capacity);
      J = source_particle_bin.val(_3, source_pidib % g_bin_capacity);
    }
    ivec3 local_base_index = (pos * g_dx_inv + 0.5f).cast<int>() - 1;
    pvec3 local_pos = pos - local_base_index * g_dx;
    base_index = local_base_index;

    pvec3x3 dws;
#pragma unroll 3
    for (int dd = 0; dd < 3; ++dd) {
      PREC d =
          (local_pos[dd] - ((int)(local_pos[dd] * g_dx_inv + 0.5) - 1) * g_dx) *
          g_dx_inv;
      dws(dd, 0) = 0.5f * (1.5 - d) * (1.5 - d);
      d -= 1.0f;
      dws(dd, 1) = 0.75 - d * d;
      d = 0.5f + d;
      dws(dd, 2) = 0.5 * d * d;
      local_base_index[dd] = ((local_base_index[dd] - 1) & g_blockmask) + 1;
    }
    pvec3 vel;
    vel.set(0.0);
    pvec9 C;
    C.set(0.0);

    // Dp^n = Dp^n+1 = (1/4) * dx^2 * I (Quad.)
    PREC Dp_inv; //< Inverse Intertia-Like Tensor (1/m^2)
    PREC scale = pbuffer.length * pbuffer.length; //< Area scale (m^2)
    Dp_inv = g_D_inv / scale; //< Scalar 4/(dx^2) for Quad. B-Spline

#pragma unroll 3
    for (char i = 0; i < 3; i++)
#pragma unroll 3
      for (char j = 0; j < 3; j++)
#pragma unroll 3
        for (char k = 0; k < 3; k++) {
          pvec3 xixp = pvec3{(PREC)i, (PREC)j, (PREC)k} * g_dx - local_pos;
          PREC W = dws(0, i) * dws(1, j) * dws(2, k);
          pvec3 vi{g2pbuffer[0][local_base_index[0] + i][local_base_index[1] + j]
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
    // if (J > 1.0f) J = 1.0f;
    // else if (J < 0.1f) J = 0.1;
    pvec9 contrib;
    {
      PREC voln = J * pbuffer.volume;
      PREC pressure = (pbuffer.bulk / pbuffer.gamma) * (pow(J, -pbuffer.gamma) - 1.f);
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

#pragma unroll 3
    for (char dd = 0; dd < 3; ++dd) {
      local_pos[dd] = pos[dd] - local_base_index[dd] * g_dx;
      PREC d =
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
          pos = pvec3{(PREC)i, (PREC)j, (PREC)k} * g_dx - local_pos;
          PREC W = dws(0, i) * dws(1, j) * dws(2, k);
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
  static constexpr uint64_t numViPerBlock = g_blockvolume * 6;
  static constexpr uint64_t numViInArena = numViPerBlock << 3;
  static constexpr uint64_t shmem_offset = (g_blockvolume * 6) << 3;

  static constexpr uint64_t numMViPerBlock = g_blockvolume * 7;
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
      *reinterpret_cast<MViArena>(shmem + shmem_offset * sizeof(float));

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
    else if (channelid == 3) 
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
    pvec3 pos;  //< Particle position at n
    pvec3 vp_n; //< Particle vel. at n
    PREC J;   //< Particle volume ratio at n
    {
      auto source_particle_bin = pbuffer.ch(_0, source_blockno);
      pos[0] = source_particle_bin.val(_0, source_pidib % g_bin_capacity);  //< x
      pos[1] = source_particle_bin.val(_1, source_pidib % g_bin_capacity);  //< y
      pos[2] = source_particle_bin.val(_2, source_pidib % g_bin_capacity);  //< z
      J = 1.0 - source_particle_bin.val(_3, source_pidib % g_bin_capacity);       //< Vo/V
      vp_n[0] = source_particle_bin.val(_4, source_pidib % g_bin_capacity); //< vx
      vp_n[1] = source_particle_bin.val(_5, source_pidib % g_bin_capacity); //< vy
      vp_n[2] = source_particle_bin.val(_6, source_pidib % g_bin_capacity); //< vz
    }
    ivec3 local_base_index = (pos * g_dx_inv + 0.5f).cast<int>() - 1;
    pvec3 local_pos = pos - local_base_index * g_dx;
    base_index = local_base_index;

    pvec3x3 dws;
#pragma unroll 3
    for (int dd = 0; dd < 3; ++dd) {
      PREC d =
          (local_pos[dd] - ((int)(local_pos[dd] * g_dx_inv + 0.5f) - 1) * g_dx) *
          g_dx_inv;
      dws(dd, 0) = 0.5f * (1.5 - d) * (1.5 - d);
      d -= 1.0f;
      dws(dd, 1) = 0.75 - d * d;
      d = 0.5f + d;
      dws(dd, 2) = 0.5 * d * d;
      local_base_index[dd] = ((local_base_index[dd] - 1) & g_blockmask) + 1;
    }
    pvec3 vel;   //< Stressed, collided grid velocity
    pvec3 vel_n; //< Unstressed, uncollided grid velocity
    pvec9 C;     //< APIC affine matrix, used a few times
    vel.set(0.0); 
    vel_n.set(0.0);
    C.set(0.0);

    // Dp^n = Dp^n+1 = (1/4) * dx^2 * I (Quad.)
    PREC Dp_inv; //< Inverse Intertia-Like Tensor (1/m^2)
    PREC scale = pbuffer.length * pbuffer.length; //< Area scale (m^2)
    Dp_inv = g_D_inv / scale; //< Scalar 4/(dx^2) for Quad. B-Spline

#pragma unroll 3
    for (char i = 0; i < 3; i++)
#pragma unroll 3
      for (char j = 0; j < 3; j++)
#pragma unroll 3
        for (char k = 0; k < 3; k++) {
          pvec3 xixp = pvec3{(PREC)i, (PREC)j, (PREC)k} * g_dx - local_pos;
          PREC W = dws(0, i) * dws(1, j) * dws(2, k);
          pvec3 vi{g2pbuffer[0][local_base_index[0] + i][local_base_index[1] + j]
                           [local_base_index[2] + k],
                  g2pbuffer[1][local_base_index[0] + i][local_base_index[1] + j]
                           [local_base_index[2] + k],
                  g2pbuffer[2][local_base_index[0] + i][local_base_index[1] + j]
                           [local_base_index[2] + k]};
          pvec3 vi_n{g2pbuffer[3][local_base_index[0] + i][local_base_index[1] + j]
                            [local_base_index[2] + k],
                    g2pbuffer[4][local_base_index[0] + i][local_base_index[1] + j]
                            [local_base_index[2] + k],
                    g2pbuffer[5][local_base_index[0] + i][local_base_index[1] + j]
                            [local_base_index[2] + k]};
          vel += vi * W;
          vel_n += vi_n * W; 
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

    J = (1 + (C[0] + C[4] + C[8]) * dt * Dp_inv) * J;

    PREC beta; //< Position correction factor (ASFLIP)
    if (J >= 1) beta = pbuffer.beta_max;  // beta max
    else beta = pbuffer.beta_min; // beta min

    pos += dt * (vel + beta * pbuffer.alpha * (vp_n - vel_n));
    vel += pbuffer.alpha * (vp_n - vel_n);

    pvec9 contrib;
    {
      PREC voln = J * pbuffer.volume;
      PREC pressure = (pbuffer.bulk / pbuffer.gamma) * (pow(J, -pbuffer.gamma) - 1.f);
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
        particle_bin.val(_3, pidib % g_bin_capacity) = 1.0 - J; //< 1 - V/Vo
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

#pragma unroll 3
    for (char dd = 0; dd < 3; ++dd) {
      local_pos[dd] = pos[dd] - local_base_index[dd] * g_dx;
      PREC d =
          (local_pos[dd] - ((int)(local_pos[dd] * g_dx_inv + 0.5f) - 1) * g_dx) *
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
          pos = pvec3{(PREC)i, (PREC)j, (PREC)k} * g_dx - local_pos;
          PREC W = dws(0, i) * dws(1, j) * dws(2, k);
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
    int channelid = base % numMViPerBlock;
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
    } else if (channelid == 4) {
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
                      const ParticleBuffer<material_e::JBarFluid> pbuffer,
                      ParticleBuffer<material_e::JBarFluid> next_pbuffer,
                      const Partition prev_partition, Partition partition,
                      const Grid grid, Grid next_grid,
                      VerticeArray vertice_array) { }

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
    pvec3 pos;
    {
      auto source_particle_bin = pbuffer.ch(_0, source_blockno);
      pos[0] = source_particle_bin.val(_0, source_pidib % g_bin_capacity);
      pos[1] = source_particle_bin.val(_1, source_pidib % g_bin_capacity);
      pos[2] = source_particle_bin.val(_2, source_pidib % g_bin_capacity);
    }
    ivec3 local_base_index = (pos * g_dx_inv + 0.5f).cast<int>() - 1;
    pvec3 local_pos = pos - local_base_index * g_dx;
    base_index = local_base_index;

    pvec3x3 dws;
#pragma unroll 3
    for (int dd = 0; dd < 3; ++dd) {
      PREC d =
          (local_pos[dd] - ((int)(local_pos[dd] * g_dx_inv + 0.5) - 1) * g_dx) *
          g_dx_inv;
      dws(dd, 0) = 0.5f * (1.5 - d) * (1.5 - d);
      d -= 1.0f;
      dws(dd, 1) = 0.75 - d * d;
      d = 0.5f + d;
      dws(dd, 2) = 0.5 * d * d;
      local_base_index[dd] = ((local_base_index[dd] - 1) & g_blockmask) + 1;
    }
    pvec3 vel;
    vel.set(0.0);
    pvec9 C;
    C.set(0.0);

    // Dp^n = Dp^n+1 = (1/4) * dx^2 * I (Quad.)
    PREC Dp_inv; //< Inverse Intertia-Like Tensor (1/m^2)
    PREC scale = pbuffer.length * pbuffer.length; //< Area scale (m^2)
    Dp_inv = g_D_inv / scale; //< Scalar 4/(dx^2) for Quad. B-Spline

#pragma unroll 3
    for (char i = 0; i < 3; i++)
#pragma unroll 3
      for (char j = 0; j < 3; j++)
#pragma unroll 3
        for (char k = 0; k < 3; k++) {
          pvec3 xixp = pvec3{(PREC)i, (PREC)j, (PREC)k} * g_dx - local_pos;
          PREC W = dws(0, i) * dws(1, j) * dws(2, k);
          pvec3 vi{g2pbuffer[0][local_base_index[0] + i][local_base_index[1] + j]
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

    pvec9 contrib;
    {
      pvec9 F;
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

#pragma unroll 3
    for (char dd = 0; dd < 3; ++dd) {
      local_pos[dd] = pos[dd] - local_base_index[dd] * g_dx;
      PREC d =
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
          pos = pvec3{(PREC)i, (PREC)j, (PREC)k} * g_dx - local_pos;
          PREC W = dws(0, i) * dws(1, j) * dws(2, k);
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
  static constexpr uint64_t numViPerBlock = g_blockvolume * 6;
  static constexpr uint64_t numViInArena = numViPerBlock << 3;
  static constexpr uint64_t shmem_offset = (g_blockvolume * 6) << 3;

  static constexpr uint64_t numMViPerBlock = g_blockvolume * 7;
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
      *reinterpret_cast<MViArena>(shmem + shmem_offset * sizeof(float));

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
    else if (channelid == 3)
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
    pvec3 pos, vp_n;
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
    pvec3 local_pos = pos - local_base_index * g_dx;
    base_index = local_base_index;

    pvec3x3 dws;
#pragma unroll 3
    for (int dd = 0; dd < 3; ++dd) {
      PREC d =
          (local_pos[dd] - ((int)(local_pos[dd] * g_dx_inv + 0.5) - 1) * g_dx) *
          g_dx_inv;
      dws(dd, 0) = 0.5f * (1.5 - d) * (1.5 - d);
      d -= 1.0f;
      dws(dd, 1) = 0.75 - d * d;
      d = 0.5f + d;
      dws(dd, 2) = 0.5 * d * d;
      local_base_index[dd] = ((local_base_index[dd] - 1) & g_blockmask) + 1;
    }
    pvec3 vel, vel_n;
    vel.set(0.0);
    vel_n.set(0.0);
    pvec9 C;
    C.set(0.0);

    // Dp^n = Dp^n+1 = (1/4) * dx^2 * I (Quad.)
    PREC Dp_inv; //< Inverse Intertia-Like Tensor (1/m^2)
    PREC scale = pbuffer.length * pbuffer.length; //< Area scale (m^2)
    Dp_inv = g_D_inv / scale; //< Scalar 4/(dx^2) for Quad. B-Spline

#pragma unroll 3
    for (char i = 0; i < 3; i++)
#pragma unroll 3
      for (char j = 0; j < 3; j++)
#pragma unroll 3
        for (char k = 0; k < 3; k++) {
          pvec3 xixp = pvec3{(PREC)i, (PREC)j, (PREC)k} * g_dx - local_pos;
          PREC W = dws(0, i) * dws(1, j) * dws(2, k);
          pvec3 vi{g2pbuffer[0][local_base_index[0] + i][local_base_index[1] + j]
                           [local_base_index[2] + k],
                  g2pbuffer[1][local_base_index[0] + i][local_base_index[1] + j]
                           [local_base_index[2] + k],
                  g2pbuffer[2][local_base_index[0] + i][local_base_index[1] + j]
                           [local_base_index[2] + k]};
          pvec3 vi_n{g2pbuffer[3][local_base_index[0] + i][local_base_index[1] + j]
                           [local_base_index[2] + k],
                  g2pbuffer[4][local_base_index[0] + i][local_base_index[1] + j]
                           [local_base_index[2] + k],
                  g2pbuffer[5][local_base_index[0] + i][local_base_index[1] + j]
                           [local_base_index[2] + k]};
          vel += vi * W;
          vel_n += vi_n * W;
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

#pragma unroll 9
    for (int d = 0; d < 9; ++d)
      dws.val(d) = C[d] * dt * Dp_inv + ((d & 0x3) ? 0.0 : 1.0);
    pvec9 contrib;
    {
      pvec9 F;
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
      PREC J  = matrixDeterminant3d(F.data());
      PREC beta;
      if (J >= 1.0) beta = pbuffer.beta_max; //< beta max
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

#pragma unroll 3
    for (char dd = 0; dd < 3; ++dd) {
      local_pos[dd] = pos[dd] - local_base_index[dd] * g_dx;
      PREC d =
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
          pos = pvec3{(PREC)i, (PREC)j, (PREC)k} * g_dx - local_pos; //< (xi-xp)
          PREC W = dws(0, i) * dws(1, j) * dws(2, k); //< Weight (2nd B-Spline)
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
    int channelid = base % numMViPerBlock;
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
    else if (channelid == 3)
      atomicAdd(&next_grid.ch(_0, blockno).val_1d(_3, c), val);
    else if (channelid == 4)
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
    pvec3 pos;
    {
      auto source_particle_bin = pbuffer.ch(_0, source_blockno);
      pos[0] = source_particle_bin.val(_0, source_pidib % g_bin_capacity);
      pos[1] = source_particle_bin.val(_1, source_pidib % g_bin_capacity);
      pos[2] = source_particle_bin.val(_2, source_pidib % g_bin_capacity);
    }
    ivec3 local_base_index = (pos * g_dx_inv + 0.5f).cast<int>() - 1;
    pvec3 local_pos = pos - local_base_index * g_dx;
    base_index = local_base_index;

    pvec3x3 dws;
#pragma unroll 3
    for (int dd = 0; dd < 3; ++dd) {
      PREC d =
          (local_pos[dd] - ((int)(local_pos[dd] * g_dx_inv + 0.5) - 1) * g_dx) *
          g_dx_inv;
      dws(dd, 0) = 0.5f * (1.5 - d) * (1.5 - d);
      d -= 1.0f;
      dws(dd, 1) = 0.75 - d * d;
      d = 0.5f + d;
      dws(dd, 2) = 0.5 * d * d;
      local_base_index[dd] = ((local_base_index[dd] - 1) & g_blockmask) + 1;
    }
    pvec3 vel;
    vel.set(0.0);
    pvec9 C;
    C.set(0.0);

    // Dp^n = Dp^n+1 = (1/4) * dx^2 * I (Quad.)
    PREC Dp_inv; //< Inverse Intertia-Like Tensor (1/m^2)
    PREC scale = pbuffer.length * pbuffer.length; //< Area scale (m^2)
    Dp_inv = g_D_inv / scale; //< Scalar 4/(dx^2) for Quad. B-Spline
    
#pragma unroll 3
    for (char i = 0; i < 3; i++)
#pragma unroll 3
      for (char j = 0; j < 3; j++)
#pragma unroll 3
        for (char k = 0; k < 3; k++) {
          pvec3 xixp = pvec3{(PREC)i, (PREC)j, (PREC)k} * g_dx - local_pos;
          PREC W = dws(0, i) * dws(1, j) * dws(2, k);
          pvec3 vi{g2pbuffer[0][local_base_index[0] + i][local_base_index[1] + j]
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
      dws.val(d) = C[d] * dt * Dp_inv + ((d & 0x3) ? 0.0 : 1.0);

    pvec9 contrib;
    {
      pvec9 F;
      PREC logJp;
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
      PREC d =
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
          pos = pvec3{(PREC)i, (PREC)j, (PREC)k} * g_dx - local_pos;
          PREC W = dws(0, i) * dws(1, j) * dws(2, k);
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
    pvec3 pos;
    {
      auto source_particle_bin = pbuffer.ch(_0, source_blockno);
      pos[0] = source_particle_bin.val(_0, source_pidib % g_bin_capacity);
      pos[1] = source_particle_bin.val(_1, source_pidib % g_bin_capacity);
      pos[2] = source_particle_bin.val(_2, source_pidib % g_bin_capacity);
    }
    ivec3 local_base_index = (pos * g_dx_inv + 0.5f).cast<int>() - 1;
    pvec3 local_pos = pos - local_base_index * g_dx;
    base_index = local_base_index;

    pvec3x3 dws;
#pragma unroll 3
    for (int dd = 0; dd < 3; ++dd) {
      PREC d =
          (local_pos[dd] - ((int)(local_pos[dd] * g_dx_inv + 0.5) - 1) * g_dx) *
          g_dx_inv;
      dws(dd, 0) = 0.5f * (1.5 - d) * (1.5 - d);
      d -= 1.0f;
      dws(dd, 1) = 0.75 - d * d;
      d = 0.5f + d;
      dws(dd, 2) = 0.5 * d * d;
      local_base_index[dd] = ((local_base_index[dd] - 1) & g_blockmask) + 1;
    }
    pvec3 vel;
    vel.set(0.0);
    pvec9 C;
    C.set(0.0);

    // Dp^n = Dp^n+1 = (1/4) * dx^2 * I (Quad.)
    PREC Dp_inv; //< Inverse Intertia-Like Tensor (1/m^2)
    PREC scale = pbuffer.length * pbuffer.length; //< Area scale (m^2)
    Dp_inv = g_D_inv / scale; //< Scalar 4/(dx^2) for Quad. B-Spline

#pragma unroll 3
    for (char i = 0; i < 3; i++)
#pragma unroll 3
      for (char j = 0; j < 3; j++)
#pragma unroll 3
        for (char k = 0; k < 3; k++) {
          pvec3 xixp = pvec3{(PREC)i, (PREC)j, (PREC)k} * g_dx - local_pos;
          PREC W = dws(0, i) * dws(1, j) * dws(2, k);
          pvec3 vi{g2pbuffer[0][local_base_index[0] + i][local_base_index[1] + j]
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

    pvec9 contrib;
    {
      pvec9 F;
      PREC logJp;
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
      PREC d =
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
          pos = pvec3{(PREC)i, (PREC)j, (PREC)k} * g_dx - local_pos;
          PREC W = dws(0, i) * dws(1, j) * dws(2, k);
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

template <typename ParticleBuffer, typename Partition, typename Grid, typename VerticeArray>
__global__ void g2p2v(float dt, float newDt, const ivec3 *__restrict__ blocks,
                      const ParticleBuffer pbuffer,
                      ParticleBuffer next_pbuffer,
                      const Partition prev_partition, Partition partition,
                      const Grid grid, Grid next_grid,
                      VerticeArray vertice_array) { } 

// Grid-to-Particle-to-Grid + Mesh Update - ASFLIP Transfer
// Stress/Strain not computed here, displacements sent to FE mesh for force calc
template <typename Partition, typename Grid, typename VerticeArray>
__global__ void g2p2v(float dt, float newDt, const ivec3 *__restrict__ blocks,
                      const ParticleBuffer<material_e::Meshed> pbuffer,
                      ParticleBuffer<material_e::Meshed> next_pbuffer,
                      const Partition prev_partition, Partition partition,
                      const Grid grid, Grid next_grid,
                      VerticeArray vertice_array) {
  static constexpr uint64_t numViPerBlock = g_blockvolume * 6;
  static constexpr uint64_t numViInArena = numViPerBlock << 3;
  static constexpr uint64_t shmem_offset = (g_blockvolume * 6) << 3;

  static constexpr uint64_t numMViPerBlock = g_blockvolume * 7;
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
      *reinterpret_cast<MViArena>(shmem + shmem_offset * sizeof(float));

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
    else if (channelid == 3) 
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
    pvec3 pos;  //< Particle position at n
    int ID; // Vertice ID for mesh
    pvec3 vp_n; //< Particle vel. at n
    //float tension;
    PREC beta = 0;
    //float J;   //< Particle volume ratio at n
    pvec3 b;
    {
      auto source_particle_bin = pbuffer.ch(_0, source_blockno);
      pos[0] = source_particle_bin.val(_0, source_pidib % g_bin_capacity);  //< x
      pos[1] = source_particle_bin.val(_1, source_pidib % g_bin_capacity);  //< y
      pos[2] = source_particle_bin.val(_2, source_pidib % g_bin_capacity);  //< z
      ID = (int)source_particle_bin.val(_3, source_pidib % g_bin_capacity); //< ID
      vp_n[0] = source_particle_bin.val(_4, source_pidib % g_bin_capacity); //< vx
      vp_n[1] = source_particle_bin.val(_5, source_pidib % g_bin_capacity); //< vy
      vp_n[2] = source_particle_bin.val(_6, source_pidib % g_bin_capacity); //< vz
      //beta = source_particle_bin.val(_7, source_pidib % g_bin_capacity); //< 
      b[0] = source_particle_bin.val(_8, source_pidib % g_bin_capacity); //< bx
      b[1] = source_particle_bin.val(_9, source_pidib % g_bin_capacity); //< by
      b[2] = source_particle_bin.val(_10, source_pidib % g_bin_capacity); //< bz
    }
    ivec3 local_base_index = (pos * g_dx_inv + 0.5f).cast<int>() - 1;
    pvec3 local_pos = pos - local_base_index * g_dx;
    base_index = local_base_index;

    pvec3x3 dws;
#pragma unroll 3
    for (int dd = 0; dd < 3; ++dd) {
      PREC d =
          (local_pos[dd] - ((int)(local_pos[dd] * g_dx_inv + 0.5) - 1) * g_dx) *
          g_dx_inv;
      dws(dd, 0) = 0.5f * (1.5 - d) * (1.5 - d);
      d -= 1.0f;
      dws(dd, 1) = 0.75 - d * d;
      d = 0.5f + d;
      dws(dd, 2) = 0.5 * d * d;
      local_base_index[dd] = ((local_base_index[dd] - 1) & g_blockmask) + 1;
    }
    pvec3 vel;   //< Stressed, collided grid velocity
    pvec3 vel_n; //< Unstressed, uncollided grid velocity
    pvec9 C;     //< APIC affine matrix, used a few times
    vel.set(0.0); 
    vel_n.set(0.0);
    C.set(0.0);

    // Dp^n = Dp^n+1 = (1/4) * dx^2 * I (Quad.)
    PREC Dp_inv; //< Inverse Intertia-Like Tensor (1/m^2)
    PREC scale = pbuffer.length * pbuffer.length; //< Area scale (m^2)
    Dp_inv = g_D_inv / scale; //< Scalar 4/(dx^2) for Quad. B-Spline

#pragma unroll 3
    for (char i = 0; i < 3; i++)
#pragma unroll 3
      for (char j = 0; j < 3; j++)
#pragma unroll 3
        for (char k = 0; k < 3; k++) {
          pvec3 xixp = pvec3{(PREC)i, (PREC)j, (PREC)k} * g_dx - local_pos;
          PREC W = dws(0, i) * dws(1, j) * dws(2, k);
          pvec3 vi{g2pbuffer[0][local_base_index[0] + i][local_base_index[1] + j]
                           [local_base_index[2] + k],
                  g2pbuffer[1][local_base_index[0] + i][local_base_index[1] + j]
                           [local_base_index[2] + k],
                  g2pbuffer[2][local_base_index[0] + i][local_base_index[1] + j]
                           [local_base_index[2] + k]};
          pvec3 vi_n{g2pbuffer[3][local_base_index[0] + i][local_base_index[1] + j]
                            [local_base_index[2] + k],
                    g2pbuffer[4][local_base_index[0] + i][local_base_index[1] + j]
                            [local_base_index[2] + k],
                    g2pbuffer[5][local_base_index[0] + i][local_base_index[1] + j]
                            [local_base_index[2] + k]};
          vel += vi * W;
          vel_n += vi_n * W; 
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

    //J = (1 + (C[0] + C[4] + C[8]) * dt * Dp_inv) * J;
    //float beta;
    // if (J >= 1.f) beta = pbuffer.beta_max; //< Position correction factor (ASFLIP)
    // else if (J < 1.f) beta = pbuffer.beta_min;
    // else beta = 0.f;
    // if (tension >= 2.f) beta = pbuffer.beta_max; //< Position correction factor (ASFLIP)
    // else if (tension <= 0.f) beta = 0.f;
    // else beta = pbuffer.beta_min;

    PREC count;
    count = vertice_array.val(_10, ID); //< count

    int surface_ASFLIP = 0;
    if (surface_ASFLIP && count > 10.f) { 
      // Interior of Mesh
      pos += dt * vel;
      vel += pbuffer.alpha * (vp_n - vel_n);
    } else if (surface_ASFLIP && count <= 10.f) { 
      // Surface of Mesh
      pos += dt * (vel + beta * pbuffer.alpha * (vp_n - vel_n));
      vel += pbuffer.alpha * (vp_n - vel_n);
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
        vertice_array.val(_6, ID) = 0.f; //< 
        vertice_array.val(_7, ID) = 0.f; //< fx
        vertice_array.val(_8, ID) = 0.f; //< fy
        vertice_array.val(_9, ID) = 0.f; //< fz
        vertice_array.val(_10,ID) = 0.f; //< count
      }
    }
  }
}

template <typename ParticleBuffer, typename Partition, typename Grid, typename VerticeArray>
__global__ void g2p2v_FBar(float dt, float newDt, const ivec3 *__restrict__ blocks,
                      const ParticleBuffer pbuffer,
                      ParticleBuffer next_pbuffer,
                      const Partition prev_partition, Partition partition,
                      const Grid grid, Grid next_grid,
                      VerticeArray vertice_array) { }

// Grid-to-Particle-to-Grid + Mesh Update - F-Bar ASFLIP Transfer
// Stress/Strain not computed here, displacements sent to FE mesh for force calc
template <typename Partition, typename Grid, typename VerticeArray>
__global__ void g2p2v_FBar(float dt, float newDt, const ivec3 *__restrict__ blocks,
                      const ParticleBuffer<material_e::Meshed> pbuffer,
                      ParticleBuffer<material_e::Meshed> next_pbuffer,
                      const Partition prev_partition, Partition partition,
                      const Grid grid, Grid next_grid,
                      VerticeArray vertice_array) {
  static constexpr uint64_t numViPerBlock = g_blockvolume * 6;
  static constexpr uint64_t numViInArena = numViPerBlock << 3;
  static constexpr uint64_t shmem_offset = (g_blockvolume * 6) << 3;

  static constexpr uint64_t numMViPerBlock = g_blockvolume * 7;
  static constexpr uint64_t numMViInArena = numMViPerBlock << 3;

  static constexpr unsigned arenamask = (g_blocksize << 1) - 1;
  static constexpr unsigned arenabits = g_blockbits + 1;

  extern __shared__ char shmem[];
  using ViArena =
      float(*)[6][g_blocksize << 1][g_blocksize << 1][g_blocksize << 1];
  using ViArenaRef =
      float(&)[6][g_blocksize << 1][g_blocksize << 1][g_blocksize << 1];
  ViArenaRef __restrict__ g2pbuffer = *reinterpret_cast<ViArena>(shmem);

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
    else if (channelid == 3) 
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
    pvec3 pos;  //< Particle position at n
    int ID; // Vertice ID for mesh
    pvec3 vp_n; //< Particle vel. at n
    //float tension;
    PREC beta = 0;
    //float J;   //< Particle volume ratio at n
    pvec3 b;
    {
      auto source_particle_bin = pbuffer.ch(_0, source_blockno);
      pos[0] = source_particle_bin.val(_0, source_pidib % g_bin_capacity);  //< x
      pos[1] = source_particle_bin.val(_1, source_pidib % g_bin_capacity);  //< y
      pos[2] = source_particle_bin.val(_2, source_pidib % g_bin_capacity);  //< z
      ID = (int)source_particle_bin.val(_3, source_pidib % g_bin_capacity); //< ID
      vp_n[0] = source_particle_bin.val(_4, source_pidib % g_bin_capacity); //< vx
      vp_n[1] = source_particle_bin.val(_5, source_pidib % g_bin_capacity); //< vy
      vp_n[2] = source_particle_bin.val(_6, source_pidib % g_bin_capacity); //< vz
      //beta = source_particle_bin.val(_7, source_pidib % g_bin_capacity); //< 
      b[0] = source_particle_bin.val(_8, source_pidib % g_bin_capacity); //< bx
      b[1] = source_particle_bin.val(_9, source_pidib % g_bin_capacity); //< by
      b[2] = source_particle_bin.val(_10, source_pidib % g_bin_capacity); //< bz
    }
    ivec3 local_base_index = (pos * g_dx_inv + 0.5f).cast<int>() - 1;
    pvec3 local_pos = pos - local_base_index * g_dx;
    base_index = local_base_index;

    pvec3x3 dws;
#pragma unroll 3
    for (int dd = 0; dd < 3; ++dd) {
      PREC d =
          (local_pos[dd] - ((int)(local_pos[dd] * g_dx_inv + 0.5) - 1) * g_dx) *
          g_dx_inv;
      dws(dd, 0) = 0.5f * (1.5 - d) * (1.5 - d);
      d -= 1.0f;
      dws(dd, 1) = 0.75 - d * d;
      d = 0.5f + d;
      dws(dd, 2) = 0.5 * d * d;
      local_base_index[dd] = ((local_base_index[dd] - 1) & g_blockmask) + 1;
    }
    pvec3 vel;   //< Stressed, collided grid velocity
    pvec3 vel_n; //< Unstressed, uncollided grid velocity
    pvec9 C;     //< APIC affine matrix, used a few times
    vel.set(0.0); 
    vel_n.set(0.0);
    C.set(0.0);

    // Dp^n = Dp^n+1 = (1/4) * dx^2 * I (Quad.)
    PREC Dp_inv; //< Inverse Intertia-Like Tensor (1/m^2)
    PREC scale = pbuffer.length * pbuffer.length; //< Area scale (m^2)
    Dp_inv = g_D_inv / scale; //< Scalar 4/(dx^2) for Quad. B-Spline

#pragma unroll 3
    for (char i = 0; i < 3; i++)
#pragma unroll 3
      for (char j = 0; j < 3; j++)
#pragma unroll 3
        for (char k = 0; k < 3; k++) {
          pvec3 xixp = pvec3{(PREC)i, (PREC)j, (PREC)k} * g_dx - local_pos;
          PREC W = dws(0, i) * dws(1, j) * dws(2, k);
          pvec3 vi{g2pbuffer[0][local_base_index[0] + i][local_base_index[1] + j]
                           [local_base_index[2] + k],
                  g2pbuffer[1][local_base_index[0] + i][local_base_index[1] + j]
                           [local_base_index[2] + k],
                  g2pbuffer[2][local_base_index[0] + i][local_base_index[1] + j]
                           [local_base_index[2] + k]};
          pvec3 vi_n{g2pbuffer[3][local_base_index[0] + i][local_base_index[1] + j]
                            [local_base_index[2] + k],
                    g2pbuffer[4][local_base_index[0] + i][local_base_index[1] + j]
                            [local_base_index[2] + k],
                    g2pbuffer[5][local_base_index[0] + i][local_base_index[1] + j]
                            [local_base_index[2] + k]};
          vel += vi * W;
          vel_n += vi_n * W; 
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

    //J = (1 + (C[0] + C[4] + C[8]) * dt * Dp_inv) * J;
    //float beta;
    // if (J >= 1.f) beta = pbuffer.beta_max; //< Position correction factor (ASFLIP)
    // else if (J < 1.f) beta = pbuffer.beta_min;
    // else beta = 0.f;
    // if (tension >= 2.f) beta = pbuffer.beta_max; //< Position correction factor (ASFLIP)
    // else if (tension <= 0.f) beta = 0.f;
    // else beta = pbuffer.beta_min;

    PREC count;
    count = vertice_array.val(_10, ID); //< count

    int surface_ASFLIP = 0;
    if (surface_ASFLIP && count > 10.f) { 
      // Interior of Mesh
      pos += dt * vel;
      vel += pbuffer.alpha * (vp_n - vel_n);
    } else if (surface_ASFLIP && count <= 10.f) { 
      // Surface of Mesh
      pos += dt * (vel + beta * pbuffer.alpha * (vp_n - vel_n));
      vel += pbuffer.alpha * (vp_n - vel_n);
    } else {
      pos += dt * (vel + beta * pbuffer.alpha * (vp_n - vel_n));
      vel += pbuffer.alpha * (vp_n - vel_n);
    }    
    
    {
      vertice_array.val(_0, ID) = pos[0];
      vertice_array.val(_1, ID) = pos[1];
      vertice_array.val(_2, ID) = pos[2];
      vertice_array.val(_3, ID) = 0.0; //< bx
      vertice_array.val(_4, ID) = 0.0; //< by
      vertice_array.val(_5, ID) = 0.0; //< bz
      vertice_array.val(_6, ID) = 0.0; //< 
      vertice_array.val(_7, ID) = 0.0; //< fx
      vertice_array.val(_8, ID) = 0.0; //< fy
      vertice_array.val(_9, ID) = 0.0; //< fz
      vertice_array.val(_10,ID) = 0.0; //< count
      vertice_array.val(_11, ID) = 0.0; //< Vol
      vertice_array.val(_12, ID) = 0.0; //< JBar Vol
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
  pvec3 p[4];
  auto element = elementBins.ch(_0, blockIdx.x);

  /* matrix indexes
          mat1   |  diag   |  mat2^T
          0 3 6  |  0      |  0 1 2
          1 4 7  |    1    |  3 4 5
          2 5 8  |      2  |  6 7 8
  */

  /// Precomputed
  // Dm is undeformed edge vector matrix relative to a vertex 0
  // Bm is undeformed face normal matrix relative to a vertex 0
  // Bm^T = Dm^-1
  // Restvolume is underformed volume [m^3]
  pvec9 DmI, Bm;
  DmI.set(0.0);
  Bm.set(0.0);
  IDs[0] = element.val(_0, 0); //< ID of node 0
  IDs[1] = element.val(_1, 0); //< ID of node 1
  IDs[2] = element.val(_2, 0); //< ID of node 2 
  IDs[3] = element.val(_3, 0); //< ID of node 3

  PREC V0 = element.val(_13, 0); //< Undeformed volume [m^3]
  float sV0 = V0<0.f ? -1.f : 1.f;

  DmI[0] = element.val(_4, 0); //< Bm^T, undef. area weighted face normals^T
  DmI[1] = element.val(_5, 0);
  DmI[2] = element.val(_6, 0);
  DmI[3] = element.val(_7, 0);
  DmI[4] = element.val(_8, 0);
  DmI[5] = element.val(_9, 0);
  DmI[6] = element.val(_10, 0);
  DmI[7] = element.val(_11, 0);
  DmI[8] = element.val(_12, 0);

  // Set position of vertices
  for (int v = 0; v < 4; v++) {
    int ID = IDs[v] - 1; //< Index at 0, my elements input files index from 1
    p[v][0] = vertice_array.val(_0, ID) * elementBins.length; //< x [m]
    p[v][1] = vertice_array.val(_1, ID) * elementBins.length; //< y [m]
    p[v][2] = vertice_array.val(_2, ID) * elementBins.length; //< z [m]
  }
  __syncthreads();

  /// Run-Time
  // Ds is deformed edge vector matrix relative to node 0
  pvec9 Ds;
  Ds.set(0.0);
  Ds[0] = p[1][0] - p[0][0];
  Ds[1] = p[1][1] - p[0][1];
  Ds[2] = p[1][2] - p[0][2];
  Ds[3] = p[2][0] - p[0][0];
  Ds[4] = p[2][1] - p[0][1];
  Ds[5] = p[2][2] - p[0][2];
  Ds[6] = p[3][0] - p[0][0];
  Ds[7] = p[3][1] - p[0][1];
  Ds[8] = p[3][2] - p[0][2];

  // F is 3x3 deformation gradient at Gauss Points (Centroid for Lin. Tet.)
  pvec9 F; //< Deformation gradient
  F.set(0.0);
  // F = Ds Dm^-1 = Ds Bm^T; Bm^T = Dm^-1
  matrixMatrixMultiplication3d(Ds.data(), DmI.data(), F.data());

  // J = det | F |,  i.e. volume change ratio undef -> def
  PREC J = matrixDeterminant3d(F.data());
  PREC Vn = V0 * J;
  Bm.set(0.0);
  Bm[0] = V0 * DmI[0];
  Bm[1] = V0 * DmI[3];
  Bm[2] = V0 * DmI[6];
  Bm[3] = V0 * DmI[1];
  Bm[4] = V0 * DmI[4];
  Bm[5] = V0 * DmI[7];
  Bm[6] = V0 * DmI[2];
  Bm[7] = V0 * DmI[5];
  Bm[8] = V0 * DmI[8];

  // P is First Piola-Kirchoff stress at element Gauss point
  // P = (dPsi/dF){F}, Derivative of Free Energy w.r.t. Def. Grad.
  // Maps undeformed area weighted face normals to deformed traction vectors
  pvec9 P; //< PK1, First Piola-Kirchoff stress at element Gauss point
  pvec9 G; //< Deformed internal forces at nodes 1,2,3 relative to node 0
  pvec9 Bs; //< BsT is the deformed. area-weighted node normal matrix
  P.set(0.0);
  G.set(0.0);
  Bs.set(0.0);
  // P = (dPsi/dF){F}, Fixed-Corotated model for energy potential
  compute_stress_fixedcorotated_PK1(elementBins.mu, elementBins.lambda, F, P);

  // G = P Bm
  matrixMatrixMultiplication3d(P.data(), Bm.data(), G.data());
  
  // F^-1 = inv(F)
  pvec9 Finv; //< Deformation gradient inverse
  Finv.set(0.0);
  matrixInverse(F.data(), Finv.data());

  // Bs = J F^-1^T Bm ;  Bs^T = J Bm^T F^-1; Note: (AB)^T = B^T A^T
  matrixTransposeMatrixMultiplication3d(Finv.data(), Bm.data(), Bs.data());
  // Bs = vol_n * Ds^-T 
  //matrixInverse(Ds.data(), Bs.data());

  Bm.set(0.0); //< Now use for Cauchy stress
  matrixMatrixTransposeMultiplication3d(P.data(), F.data(), Bm.data());
  PREC pressure = - (Bm[0] + Bm[4] + Bm[8]) / (3.0 * J);
  PREC von_mises = sqrt( (  (Bm[0]-Bm[4])*(Bm[0]-Bm[4]) + 
                            (Bm[4]-Bm[8])*(Bm[4]-Bm[8]) + 
                            (Bm[8]-Bm[0])*(Bm[8]-Bm[0]) + 
                            6*(Bm[3]*Bm[3] + Bm[6]*Bm[6] + Bm[7]*Bm[7]) ) / (2.0 * J));
  

  pvec3 f; // Internal force vector at deformed vertex
  pvec3 n;
  f.set(0.0);
  n.set(0.0);
  if (threadIdx.x == 1){ // Node 1; Face a
    f[0] = G[0];
    f[1] = G[1];
    f[2] = G[2];
    n[0] = J * Bs[0];
    n[1] = J * Bs[1];
    n[2] = J * Bs[2];
  } else if (threadIdx.x == 2) { // Node 2; Face b
    f[0] = G[3];
    f[1] = G[4];
    f[2] = G[5];
    n[0] = J * Bs[3];
    n[1] = J * Bs[4];
    n[2] = J * Bs[5];
  }  else if (threadIdx.x == 3) { // Node 3; Face c
    f[0] = G[6]; 
    f[1] = G[7];
    f[2] = G[8];
    n[0] = J * Bs[6];
    n[1] = J * Bs[7];
    n[2] = J * Bs[8];
  } else { // Node 4; Face d
    f[0] = - (G[0] + G[3] + G[6]);
    f[1] = - (G[1] + G[4] + G[7]);
    f[2] = - (G[2] + G[5] + G[8]);
    n[0] = - J * (Bs[0] + Bs[3] + Bs[6]) ;
    n[1] = - J * (Bs[1] + Bs[4] + Bs[7]) ;
    n[2] = - J * (Bs[2] + Bs[5] + Bs[8]) ;
  }
  __syncthreads();
  

  f[0] =  f[0] / elementBins.length;
  f[1] =  f[1] / elementBins.length; 
  f[2] =  f[2] / elementBins.length;

  __syncthreads();

  {
    int ID = IDs[threadIdx.x] - 1; // Index from 0
    // atomicAdd(&vertice_array.val(_0, ID), b[0]); //< x
    // atomicAdd(&vertice_array.val(_1, ID), b[1]); //< y
    // atomicAdd(&vertice_array.val(_2, ID), b[2]); //< z
    atomicAdd(&vertice_array.val(_3, ID), (PREC)(J-1.0)*abs(V0)*0.25); //< bx
    atomicAdd(&vertice_array.val(_4, ID), (PREC)pressure*abs(V0)*0.25); //< by
    atomicAdd(&vertice_array.val(_5, ID), (PREC)von_mises*abs(V0)*0.25); //< bz
    atomicAdd(&vertice_array.val(_6, ID), (PREC)abs(V0)*0.25); //< Node undef. volume [m3]
    atomicAdd(&vertice_array.val(_7, ID), (PREC)f[0]); //< fx
    atomicAdd(&vertice_array.val(_8, ID), (PREC)f[1]); //< fy
    atomicAdd(&vertice_array.val(_9, ID), (PREC)f[2]); //< fz
    atomicAdd(&vertice_array.val(_10, ID), (PREC)1.f); //< Counter
  }
}

template <typename VerticeArray, typename ElementArray>
__global__ void v2fem2v(float dt, float newDt,
                      VerticeArray vertice_array,
                      ElementArray element_array,
                      ElementBuffer<fem_e::Tetrahedron_FBar> elementBins) { }
template <typename VerticeArray, typename ElementArray, typename ElementBuffer>
__global__ void v2fem2v(float dt, float newDt,
                      VerticeArray vertice_array,
                      ElementArray element_array,
                      ElementBuffer elementBins) { }

template <typename VerticeArray, typename ElementArray, typename ElementBuffer>
__global__ void v2fem_FBar(float dt, float newDt,
                      VerticeArray vertice_array,
                      ElementArray element_array,
                      ElementBuffer elementBins) { }

// Grid-to-Particle-to-Grid + Mesh Update - ASFLIP Transfer
// Stress/Strain not computed here, displacements sent to FE mesh for force calc
template <typename VerticeArray, typename ElementArray>
__global__ void v2fem_FBar(float dt, float newDt,
                      VerticeArray vertice_array,
                      ElementArray element_array,
                      ElementBuffer<fem_e::Tetrahedron_FBar> elementBins) {

  if (blockIdx.x >= g_max_fem_element_num) return;
  int IDs[4];
  pvec3 p[4];
  auto element = elementBins.ch(_0, blockIdx.x);

  /* matrix indexes
          mat1   |  diag   |  mat2^T
          0 3 6  |  0      |  0 1 2
          1 4 7  |    1    |  3 4 5
          2 5 8  |      2  |  6 7 8
  */

  /// Precomputed
  // Dm is undeformed edge vector matrix relative to a vertex 0
  // Bm is undeformed face normal matrix relative to a vertex 0
  // Bm^T = Dm^-1
  // Restvolume is underformed volume [m^3]
  pvec9 DmI, Bm;
  DmI.set(0.0);
  Bm.set(0.0);
  IDs[0] = element.val(_0, 0); //< ID of node 0
  IDs[1] = element.val(_1, 0); //< ID of node 1
  IDs[2] = element.val(_2, 0); //< ID of node 2 
  IDs[3] = element.val(_3, 0); //< ID of node 3

  PREC V0, Jn, JBar;

  DmI[0] = element.val(_4, 0); //< Bm^T, undef. area weighted face normals^T
  DmI[1] = element.val(_5, 0);
  DmI[2] = element.val(_6, 0);
  DmI[3] = element.val(_7, 0);
  DmI[4] = element.val(_8, 0);
  DmI[5] = element.val(_9, 0);
  DmI[6] = element.val(_10, 0);
  DmI[7] = element.val(_11, 0);
  DmI[8] = element.val(_12, 0);
  V0 = element.val(_13, 0); //< Undeformed volume [m^3]
  Jn   = 1.0 - element.val(_14, 0);
  JBar = 1.0 - element.val(_15, 0);
  PREC sV0 = V0<0.f ? -1.f : 1.f;

  // Set position of vertices
  for (int v = 0; v < 4; v++) {
    int ID = IDs[v] - 1; //< Index at 0, my elements input files index from 1
    p[v][0] = vertice_array.val(_0, ID) * elementBins.length; //< x [m]
    p[v][1] = vertice_array.val(_1, ID) * elementBins.length; //< y [m]
    p[v][2] = vertice_array.val(_2, ID) * elementBins.length; //< z [m]
  }
  __syncthreads();

  /// Run-Time
  // Ds is deformed edge vector matrix relative to node 0
  pvec9 Ds;
  Ds.set(0.0);
  Ds[0] = p[1][0] - p[0][0];
  Ds[1] = p[1][1] - p[0][1];
  Ds[2] = p[1][2] - p[0][2];
  Ds[3] = p[2][0] - p[0][0];
  Ds[4] = p[2][1] - p[0][1];
  Ds[5] = p[2][2] - p[0][2];
  Ds[6] = p[3][0] - p[0][0];
  Ds[7] = p[3][1] - p[0][1];
  Ds[8] = p[3][2] - p[0][2];

  // F is 3x3 deformation gradient at Gauss Points (Centroid for Lin. Tet.)
  pvec9 F; //< Deformation gradient
  F.set(0.0);
  // F = Ds Dm^-1 = Ds Bm^T; Bm^T = Dm^-1
  matrixMatrixMultiplication3d(Ds.data(), DmI.data(), F.data());

  // J = det | F |,  i.e. volume change ratio undef -> def
  PREC J = matrixDeterminant3d(F.data());
  PREC Vn = V0 * J;
  PREC JInc = J / Jn; // J_n+1 = J_Inc * J_n
  
  __syncthreads();

  {
    int ID = IDs[threadIdx.x] - 1; // Index from 0
    atomicAdd(&vertice_array.val(_11, ID), abs(Vn)*0.25); //< Counter
    atomicAdd(&vertice_array.val(_12, ID), JInc*JBar*abs(Vn)*0.25); //< Counter
  }
  __syncthreads();
}

template <typename VerticeArray, typename ElementArray, typename ElementBuffer>
__global__ void fem2v_FBar(float dt, float newDt,
                      VerticeArray vertice_array,
                      ElementArray element_array,
                      ElementBuffer elementBins) { }

// Grid-to-Particle-to-Grid + Mesh Update - ASFLIP Transfer
// Stress/Strain not computed here, displacements sent to FE mesh for force calc
template <typename VerticeArray, typename ElementArray>
__global__ void fem2v_FBar(float dt, float newDt,
                      VerticeArray vertice_array,
                      ElementArray element_array,
                      ElementBuffer<fem_e::Tetrahedron_FBar> elementBins) {

  if (blockIdx.x >= g_max_fem_element_num) return;
  int IDs[4];
  pvec3 p[4];
  auto element = elementBins.ch(_0, blockIdx.x);

  /* matrix indexes
          mat1   |  diag   |  mat2^T
          0 3 6  |  0      |  0 1 2
          1 4 7  |    1    |  3 4 5
          2 5 8  |      2  |  6 7 8
  */

  /// Precomputed
  // Dm is undeformed edge vector matrix relative to a vertex 0
  // Bm is undeformed face normal matrix relative to a vertex 0
  // Bm^T = Dm^-1
  // Restvolume is underformed volume [m^3]
  pvec9 DmI, Bm;
  DmI.set(0.0);
  Bm.set(0.0);
  IDs[0] = element.val(_0, 0); //< ID of node 0
  IDs[1] = element.val(_1, 0); //< ID of node 1
  IDs[2] = element.val(_2, 0); //< ID of node 2 
  IDs[3] = element.val(_3, 0); //< ID of node 3

  PREC V0, Jn, JBar;

  DmI[0] = element.val(_4, 0); //< Bm^T, undef. area weighted face normals^T
  DmI[1] = element.val(_5, 0);
  DmI[2] = element.val(_6, 0);
  DmI[3] = element.val(_7, 0);
  DmI[4] = element.val(_8, 0);
  DmI[5] = element.val(_9, 0);
  DmI[6] = element.val(_10, 0);
  DmI[7] = element.val(_11, 0);
  DmI[8] = element.val(_12, 0);
  V0 = element.val(_13, 0); //< Undeformed volume [m^3]
  Jn   = 1.0 - element.val(_14, 0);
  //JBar = 1.0 - element.val(_15, 0);
  PREC sV0 = V0<0.f ? -1.f : 1.f;
  PREC JBar_i = 0.0;
  PREC Vn_i = 0.0;
  __syncthreads();

  // Set position of vertices
  for (int v = 0; v < 4; v++) {
    int ID = IDs[v] - 1; //< Index at 0, my elements input files index from 1
    p[v][0] = vertice_array.val(_0, ID) * elementBins.length; //< x [m]
    p[v][1] = vertice_array.val(_1, ID) * elementBins.length; //< y [m]
    p[v][2] = vertice_array.val(_2, ID) * elementBins.length; //< z [m]
    Vn_i += vertice_array.val(_11, ID);
    JBar_i += vertice_array.val(_12, ID);
  }
  __syncthreads();

  JBar = JBar_i / Vn_i;

  /// Run-Time
  // Ds is deformed edge vector matrix relative to node 0
  pvec9 Ds;
  Ds.set(0.0);
  Ds[0] = p[1][0] - p[0][0];
  Ds[1] = p[1][1] - p[0][1];
  Ds[2] = p[1][2] - p[0][2];
  Ds[3] = p[2][0] - p[0][0];
  Ds[4] = p[2][1] - p[0][1];
  Ds[5] = p[2][2] - p[0][2];
  Ds[6] = p[3][0] - p[0][0];
  Ds[7] = p[3][1] - p[0][1];
  Ds[8] = p[3][2] - p[0][2];
  __syncthreads();

  // F is 3x3 deformation gradient at Gauss Points (Centroid for Lin. Tet.)
  pvec9 F; //< Deformation gradient
  F.set(0.0);
  // F = Ds Dm^-1 = Ds Bm^T; Bm^T = Dm^-1
  matrixMatrixMultiplication3d(Ds.data(), DmI.data(), F.data());

  Bm.set(0.0);
  Bm[0] = V0 * DmI[0];
  Bm[1] = V0 * DmI[3];
  Bm[2] = V0 * DmI[6];
  Bm[3] = V0 * DmI[1];
  Bm[4] = V0 * DmI[4];
  Bm[5] = V0 * DmI[7];
  Bm[6] = V0 * DmI[2];
  Bm[7] = V0 * DmI[5];
  Bm[8] = V0 * DmI[8];

  // F^-1 = inv(F)
  pvec9 Finv; //< Deformation gradient inverse
  Finv.set(0.0);
  matrixInverse(F.data(), Finv.data());
  
  pvec9 Bs; //< BsT is the deformed. area-weighted node normal matrix
  Bs.set(0.0);

  // Bs = J F^-1^T Bm ;  Bs^T = J Bm^T F^-1; Note: (AB)^T = B^T A^T
  matrixTransposeMatrixMultiplication3d(Finv.data(), Bm.data(), Bs.data());
  // Bs = vol_n * Ds^-T 
  //matrixInverse(Ds.data(), Bs.data());


  // J = det | F |,  i.e. volume change ratio undef -> def
  PREC J = matrixDeterminant3d(F.data());
  PREC Vn = V0 * J;
  PREC J_Scale =  cbrt((JBar / J));
  F[0] = J_Scale * F[0];
  F[1] = J_Scale * F[1];
  F[2] = J_Scale * F[2];
  F[3] = J_Scale * F[3];
  F[4] = J_Scale * F[4];
  F[5] = J_Scale * F[5];
  F[6] = J_Scale * F[6];
  F[7] = J_Scale * F[7];
  F[8] = J_Scale * F[8];

  // P is First Piola-Kirchoff stress at element Gauss point
  // P = (dPsi/dF){F}, Derivative of Free Energy w.r.t. Def. Grad.
  // Maps undeformed area weighted face normals to deformed traction vectors
  pvec9 P; //< PK1, First Piola-Kirchoff stress at element Gauss point
  pvec9 G; //< Deformed internal forces at nodes 1,2,3 relative to node 0
  P.set(0.0);
  G.set(0.0);
  // P = (dPsi/dF){F}, Fixed-Corotated model for energy potential
  compute_stress_fixedcorotated_PK1(elementBins.mu, elementBins.lambda, F, P);

  // G = P Bm
  matrixMatrixMultiplication3d(P.data(), Bm.data(), G.data());

  Bm.set(0.0); //< Now use for Cauchy stress
  matrixMatrixTransposeMultiplication3d(P.data(), F.data(), Bm.data());
  PREC pressure = - (Bm[0] + Bm[4] + Bm[8]) / (3.0 * J);
  PREC von_mises = sqrt( (  (Bm[0]-Bm[4])*(Bm[0]-Bm[4]) + 
                            (Bm[4]-Bm[8])*(Bm[4]-Bm[8]) + 
                            (Bm[8]-Bm[0])*(Bm[8]-Bm[0]) + 
                            6*(Bm[3]*Bm[3] + Bm[6]*Bm[6] + Bm[7]*Bm[7]) ) / (2.0 * J));

  pvec3 f; // Internal force vector at deformed vertex
  pvec3 n;
  f.set(0.0);
  n.set(0.0);
  if (threadIdx.x == 1){ // Node 1; Face a
    f[0] = G[0];
    f[1] = G[1];
    f[2] = G[2];
    n[0] = J * Bs[0];
    n[1] = J * Bs[1];
    n[2] = J * Bs[2];
  } else if (threadIdx.x == 2) { // Node 2; Face b
    f[0] = G[3];
    f[1] = G[4];
    f[2] = G[5];
    n[0] = J * Bs[3];
    n[1] = J * Bs[4];
    n[2] = J * Bs[5];
  }  else if (threadIdx.x == 3) { // Node 3; Face c
    f[0] = G[6]; 
    f[1] = G[7];
    f[2] = G[8];
    n[0] = J * Bs[6];
    n[1] = J * Bs[7];
    n[2] = J * Bs[8];
  } else { // Node 4; Face d
    f[0] = - (G[0] + G[3] + G[6]);
    f[1] = - (G[1] + G[4] + G[7]);
    f[2] = - (G[2] + G[5] + G[8]);
    n[0] = - J * (Bs[0] + Bs[3] + Bs[6]) ;
    n[1] = - J * (Bs[1] + Bs[4] + Bs[7]) ;
    n[2] = - J * (Bs[2] + Bs[5] + Bs[8]) ;
  }
  __syncthreads();
   
  f[0] =  f[0] / elementBins.length;
  f[1] =  f[1] / elementBins.length; 
  f[2] =  f[2] / elementBins.length;

  __syncthreads();

  {
    element.val(_14, 0) = (PREC)(J    - 1.0);
    element.val(_15, 0) = (PREC)(JBar - 1.0);
  }
  __syncthreads();

  {
    int ID = IDs[threadIdx.x] - 1; // Index from 0
    // atomicAdd(&vertice_array.val(_3, ID), (PREC)n[0]); //< bx
    // atomicAdd(&vertice_array.val(_4, ID), (PREC)n[1]); //< by
    // atomicAdd(&vertice_array.val(_5, ID), (PREC)n[2]); //< bz
    atomicAdd(&vertice_array.val(_3, ID), (PREC)(1.0 - JBar)*abs(V0)*0.25); //< bz
    atomicAdd(&vertice_array.val(_4, ID), (PREC)pressure*abs(V0)*0.25); //< bx
    atomicAdd(&vertice_array.val(_5, ID), (PREC)von_mises*abs(V0)*0.25); //< by
    atomicAdd(&vertice_array.val(_6, ID), (PREC)abs(V0) * 0.25); //< Node undef. volume [m3]
    atomicAdd(&vertice_array.val(_7, ID), (PREC)f[0]); //< fx
    atomicAdd(&vertice_array.val(_8, ID), (PREC)f[1]); //< fy
    atomicAdd(&vertice_array.val(_9, ID), (PREC)f[2]); //< fz
    atomicAdd(&vertice_array.val(_10, ID), (PREC)1.f); //< Counter
    // atomicAdd(&vertice_array.val(_11, ID), Vn*0.25); //< Counter
    // atomicAdd(&vertice_array.val(_12, ID), (1.0 - JBar)*Vn*0.25); //< Counter
  }
}

template <typename VerticeArray, typename ElementArray>
__global__ void v2fem2v(float dt, float newDt,
                      VerticeArray vertice_array,
                      ElementArray element_array,
                      ElementBuffer<fem_e::Brick> elementBins) {
                        return;
}

template <typename ParticleBuffer, typename Partition, typename Grid>
__global__ void g2p_FBar(float dt, float newDt, const ivec3 *__restrict__ blocks,
                      const ParticleBuffer pbuffer,
                      ParticleBuffer next_pbuffer,
                      const Partition prev_partition, Partition partition,
                      Grid grid, Grid next_grid, PREC length) {}
template <typename Partition, typename Grid>
__global__ void g2p_FBar(float dt, float newDt, const ivec3 *__restrict__ blocks,
                      const ParticleBuffer<material_e::JBarFluid> pbuffer,
                      ParticleBuffer<material_e::JBarFluid> next_pbuffer,
                      const Partition prev_partition, Partition partition,
                      Grid grid, Grid next_grid, PREC length) {
  static constexpr uint64_t numViPerBlock = g_blockvolume * 3;
  static constexpr uint64_t numViInArena = numViPerBlock << 3;
  static constexpr uint64_t shmem_offset = (g_blockvolume * 3) << 3;
  static constexpr uint64_t numMViPerBlock = g_blockvolume * 2;
  static constexpr uint64_t numMViInArena = numMViPerBlock << 3;

  static constexpr unsigned arenamask = (g_blocksize << 1) - 1;
  static constexpr unsigned arenabits = g_blockbits + 1;

  extern __shared__ char shmem[];
  using ViArena =
      PREC_G(*)[3][g_blocksize << 1][g_blocksize << 1][g_blocksize << 1];
  using ViArenaRef =
      PREC_G(&)[3][g_blocksize << 1][g_blocksize << 1][g_blocksize << 1];
  ViArenaRef __restrict__ g2pbuffer = *reinterpret_cast<ViArena>(shmem);
  using MViArena =
      PREC_G(*)[2][g_blocksize << 1][g_blocksize << 1][g_blocksize << 1];
  using MViArenaRef =
      PREC_G(&)[2][g_blocksize << 1][g_blocksize << 1][g_blocksize << 1];
  MViArenaRef __restrict__ p2gbuffer =
      *reinterpret_cast<MViArena>(shmem + shmem_offset * sizeof(PREC_G));

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

    PREC_G val;
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
  for (int base = threadIdx.x; base < numMViInArena; base += blockDim.x) {
    int loc = base;
    char z = loc & arenamask;
    char y = (loc >>= arenabits) & arenamask;
    char x = (loc >>= arenabits) & arenamask;
    p2gbuffer[loc >> arenabits][x][y][z] = 0.0;
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
    pvec3 pos;  //< Particle position at n
    PREC J;   //< Particle volume ratio at n
    //PREC vol; //< Volumem at time n
    PREC JBar; //< Assumed particle volume ratio at n (Simple FBar)
    {
      auto source_particle_bin = pbuffer.ch(_0, source_blockno);
      pos[0] = source_particle_bin.val(_0, source_pidib % g_bin_capacity);  //< x
      pos[1] = source_particle_bin.val(_1, source_pidib % g_bin_capacity);  //< y
      pos[2] = source_particle_bin.val(_2, source_pidib % g_bin_capacity);  //< z
      J = 1.0 - source_particle_bin.val(_3, source_pidib % g_bin_capacity);       //< Vo/V
      // vp_n[0] = source_particle_bin.val(_4, source_pidib % g_bin_capacity); //< vx
      // vp_n[1] = source_particle_bin.val(_5, source_pidib % g_bin_capacity); //< vy
      // vp_n[2] = source_particle_bin.val(_6, source_pidib % g_bin_capacity); //< vz
      //vol  = source_particle_bin.val(_7, source_pidib % g_bin_capacity); //< Volume tn
      JBar = 1.0 - source_particle_bin.val(_8, source_pidib % g_bin_capacity); //< JBar tn
    }
    ivec3 local_base_index = (pos * g_dx_inv + 0.5f).cast<int>() - 1;
    pvec3 local_pos = pos - local_base_index * g_dx;
    base_index = local_base_index;

    pvec3x3 dws;
#pragma unroll 3
    for (int dd = 0; dd < 3; ++dd) {
      PREC d =
          (local_pos[dd] - ((int)(local_pos[dd] * g_dx_inv + 0.5f) - 1) * g_dx) *
          g_dx_inv;
      dws(dd, 0) = 0.5f * (1.5 - d) * (1.5 - d);
      d -= 1.f;
      dws(dd, 1) = 0.75 - d * d;
      d = 0.5f + d;
      dws(dd, 2) = 0.5 * d * d;
      local_base_index[dd] = ((local_base_index[dd] - 1) & g_blockmask) + 1;
    }
    pvec3 vel;   //< PIC, Stressed, collided grid velocity
    pvec9 C;     //< APIC affine matrix, used a few times
    vel.set(0.0); 
    C.set(0.0);

    // Dp^n = Dp^n+1 = (1/4) * dx^2 * I (Quad.)
    PREC Dp_inv; //< Inverse Intertia-Like Tensor (1/m^2)
    PREC scale = pbuffer.length * pbuffer.length; //< Area scale (m^2)
    Dp_inv = g_D_inv / scale; //< Scalar 4/(dx^2) for Quad. B-Spline

#pragma unroll 3
    for (char i = 0; i < 3; i++)
#pragma unroll 3
      for (char j = 0; j < 3; j++)
#pragma unroll 3
        for (char k = 0; k < 3; k++) {
          pvec3 xixp = pvec3{(PREC)i, (PREC)j, (PREC)k} * g_dx - local_pos;
          PREC W = dws(0, i) * dws(1, j) * dws(2, k);
          pvec3 vi{g2pbuffer[0][local_base_index[0] + i][local_base_index[1] + j]
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
    PREC JInc = (1 + (C[0] + C[4] + C[8]) * dt * Dp_inv); // J^n+1 / J^n
    PREC voln = J * pbuffer.volume; // vol^n+1, Send to grid for Simple FBar 
    J = JInc * J; // J^n+1
    //float voln = J * pbuffer.volume; // vol^n+1, Send to grid for Simple FBar 
    
#pragma unroll 3
    for (char i = 0; i < 3; i++)
#pragma unroll 3
      for (char j = 0; j < 3; j++)
#pragma unroll 3
        for (char k = 0; k < 3; k++) {
          pos = pvec3{(PREC)i, (PREC)j, (PREC)k} * g_dx - local_pos;
          PREC W = dws(0, i) * dws(1, j) * dws(2, k);
          //auto wm = pbuffer.mass * W; // Weighted mass
          PREC wv = voln * W; // Weighted volume
          atomicAdd(
              &p2gbuffer[0][local_base_index[0] + i][local_base_index[1] + j]
                        [local_base_index[2] + k],
              wv);
          atomicAdd(
              &p2gbuffer[1][local_base_index[0] + i][local_base_index[1] + j]
                        [local_base_index[2] + k],
              wv * (1.0 - JBar * JInc));
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
    int channelid = base % numMViPerBlock; // & (numMViPerBlock - 1);
    char c = channelid % g_blockvolume;
    char cz = channelid & g_blockmask;
    char cy = (channelid >>= g_blockbits) & g_blockmask;
    char cx = (channelid >>= g_blockbits) & g_blockmask;
    channelid >>= g_blockbits;
    PREC_G val =
        p2gbuffer[channelid][cx + (local_block_id & 4 ? g_blocksize : 0)]
                [cy + (local_block_id & 2 ? g_blocksize : 0)]
                [cz + (local_block_id & 1 ? g_blocksize : 0)];
    if (channelid == 0) {
      atomicAdd(&grid.ch(_0, blockno).val_1d(_7, c), val); //< Vol
    } else if (channelid == 1) {
      atomicAdd(&grid.ch(_0, blockno).val_1d(_8, c), val); //< JBar_Jinc Vol
    } 
  }
}

template <typename ParticleBuffer, typename Partition, typename Grid>
__global__ void p2g_FBar(float dt, float newDt, const ivec3 *__restrict__ blocks,
                      const ParticleBuffer pbuffer,
                      ParticleBuffer next_pbuffer,
                      const Partition prev_partition, Partition partition,
                      const Grid grid, Grid next_grid, PREC length) {}
template <typename Partition, typename Grid>
__global__ void p2g_FBar(float dt, float newDt, const ivec3 *__restrict__ blocks,
                      const ParticleBuffer<material_e::JBarFluid> pbuffer,
                      ParticleBuffer<material_e::JBarFluid> next_pbuffer,
                      const Partition prev_partition, Partition partition,
                      const Grid grid, Grid next_grid, PREC length) { 
  static constexpr uint64_t numViPerBlock = g_blockvolume * 8;
  static constexpr uint64_t numViInArena = numViPerBlock << 3;
  static constexpr uint64_t shmem_offset = (g_blockvolume * 8) << 3;

  static constexpr uint64_t numMViPerBlock = g_blockvolume * 7;
  static constexpr uint64_t numMViInArena = numMViPerBlock << 3;

  static constexpr unsigned arenamask = (g_blocksize << 1) - 1;
  static constexpr unsigned arenabits = g_blockbits + 1;

  extern __shared__ char shmem[];
  using ViArena =
      PREC_G(*)[8][g_blocksize << 1][g_blocksize << 1][g_blocksize << 1];
  using ViArenaRef =
      PREC_G(&)[8][g_blocksize << 1][g_blocksize << 1][g_blocksize << 1];
  ViArenaRef __restrict__ g2pbuffer = *reinterpret_cast<ViArena>(shmem);
  using MViArena =
      PREC_G(*)[7][g_blocksize << 1][g_blocksize << 1][g_blocksize << 1];
  using MViArenaRef =
      PREC_G(&)[7][g_blocksize << 1][g_blocksize << 1][g_blocksize << 1];
  MViArenaRef __restrict__ p2gbuffer =
      *reinterpret_cast<MViArena>(shmem + shmem_offset * sizeof(PREC_G));

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

    PREC_G val;
    if (channelid == 0) 
      val = grid_block.val_1d(_1, c);
    else if (channelid == 1)
      val = grid_block.val_1d(_2, c);
    else if (channelid == 2)
      val = grid_block.val_1d(_3, c);
    else if (channelid == 3) 
      val = grid_block.val_1d(_4, c);
    else if (channelid == 4) 
      val = grid_block.val_1d(_5, c);
    else if (channelid == 5) 
      val = grid_block.val_1d(_6, c);
    else if (channelid == 6) 
      val = grid_block.val_1d(_7, c);
    else if (channelid == 7) 
      val = grid_block.val_1d(_8, c);
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
  // Start Grid-to-Particle, threads are particle
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
    pvec3 pos;  //< Particle position at n
    pvec3 vp_n; //< Particle vel. at n
    PREC J;   //< Particle volume ratio at n
    PREC JBar; //< Assumed particle volume ratio at n (Simple FBar)
    PREC vol;
    {
      auto source_particle_bin = pbuffer.ch(_0, source_blockno);
      pos[0] = source_particle_bin.val(_0, source_pidib % g_bin_capacity);  //< x
      pos[1] = source_particle_bin.val(_1, source_pidib % g_bin_capacity);  //< y
      pos[2] = source_particle_bin.val(_2, source_pidib % g_bin_capacity);  //< z
      J =  1.0 - source_particle_bin.val(_3, source_pidib % g_bin_capacity);       //< Vo/V
      vp_n[0] = source_particle_bin.val(_4, source_pidib % g_bin_capacity); //< vx
      vp_n[1] = source_particle_bin.val(_5, source_pidib % g_bin_capacity); //< vy
      vp_n[2] = source_particle_bin.val(_6, source_pidib % g_bin_capacity); //< vz
      vol  = source_particle_bin.val(_7, source_pidib % g_bin_capacity); //< Volume tn
      JBar = 1.0 - source_particle_bin.val(_8, source_pidib % g_bin_capacity); //< JBar tn
    }
    ivec3 local_base_index = (pos * g_dx_inv + 0.5f).cast<int>() - 1;
    pvec3 local_pos = pos - local_base_index * g_dx;
    base_index = local_base_index;

    pvec3x3 dws;
#pragma unroll 3
    for (int dd = 0; dd < 3; ++dd) {
      PREC d =
          (local_pos[dd] - ((int)(local_pos[dd] * g_dx_inv + 0.5f) - 1) * g_dx) *
          g_dx_inv;
      dws(dd, 0) = 0.5f * (1.5 - d) * (1.5 - d);
      d -= 1.f;
      dws(dd, 1) = 0.75 - d * d;
      d = 0.5f + d;
      dws(dd, 2) = 0.5 * d * d;
      local_base_index[dd] = ((local_base_index[dd] - 1) & g_blockmask) + 1;
    }
    pvec3 vel;   //< Stressed, collided grid velocity
    pvec3 vel_n; //< Unstressed, uncollided grid velocity
    pvec9 C;     //< APIC affine matrix, used a few times
    PREC JBar_new; //< Simple FBar G2P JBar^n+1
    vel.set(0.0); 
    vel_n.set(0.0);
    C.set(0.0);
    JBar_new = 0.0;

    // Dp^n = Dp^n+1 = (1/4) * dx^2 * I (Quad.)
    PREC Dp_inv; //< Inverse Intertia-Like Tensor (1/m^2)
    PREC scale = length * length; //< Area scale (m^2)
    Dp_inv = g_D_inv / scale; //< Scalar 4/(dx^2) for Quad. B-Spline

#pragma unroll 3
    for (char i = 0; i < 3; i++)
#pragma unroll 3
      for (char j = 0; j < 3; j++)
#pragma unroll 3
        for (char k = 0; k < 3; k++) {
          pvec3 xixp = pvec3{(PREC)i, (PREC)j, (PREC)k} * g_dx - local_pos;
          PREC W = dws(0, i) * dws(1, j) * dws(2, k);
          pvec3 vi{g2pbuffer[0][local_base_index[0] + i][local_base_index[1] + j]
                           [local_base_index[2] + k],
                  g2pbuffer[1][local_base_index[0] + i][local_base_index[1] + j]
                           [local_base_index[2] + k],
                  g2pbuffer[2][local_base_index[0] + i][local_base_index[1] + j]
                           [local_base_index[2] + k]};
          pvec3 vi_n{g2pbuffer[3][local_base_index[0] + i][local_base_index[1] + j]
                            [local_base_index[2] + k],
                    g2pbuffer[4][local_base_index[0] + i][local_base_index[1] + j]
                            [local_base_index[2] + k],
                    g2pbuffer[5][local_base_index[0] + i][local_base_index[1] + j]
                            [local_base_index[2] + k]};
          PREC JBar_i = g2pbuffer[7][local_base_index[0] + i][local_base_index[1] + j]
                            [local_base_index[2] + k] / g2pbuffer[6][local_base_index[0] + i][local_base_index[1] + j]
                            [local_base_index[2] + k];
          vel   += vi * W;
          vel_n += vi_n * W; 
          C[0] += W * vi[0] * xixp[0] * scale;
          C[1] += W * vi[1] * xixp[0] * scale;
          C[2] += W * vi[2] * xixp[0] * scale;
          C[3] += W * vi[0] * xixp[1] * scale;
          C[4] += W * vi[1] * xixp[1] * scale;
          C[5] += W * vi[2] * xixp[1] * scale;
          C[6] += W * vi[0] * xixp[2] * scale;
          C[7] += W * vi[1] * xixp[2] * scale;
          C[8] += W * vi[2] * xixp[2] * scale;
          JBar_new += JBar_i * W;
        }
    JBar_new = 1.0 - JBar_new;
    J = (1 + (C[0] + C[4] + C[8]) * dt * Dp_inv) * J;
    //C = C.cast<float>();
    // FBar_n+1 = (JBar_n+1 / J_n+1)^(1/3) * F_n+1

    PREC beta; //< Position correction factor (ASFLIP)
    if (JBar_new >= 1.f) {
      //JStress = Jc;       // No vol. expansion, Tamp. 2017
      //J  = Jc;
      beta = pbuffer.beta_max;  // beta max
    } else beta = pbuffer.beta_min; // beta min
    
    // ASFLIP advection
    pos += dt * (vel + beta * pbuffer.alpha * (vp_n - vel_n));
    vel += pbuffer.alpha * (vp_n - vel_n);

    pvec9 contrib;
    contrib.set(0.0);
    {
      PREC voln = J * pbuffer.volume;
      PREC pressure = (pbuffer.bulk / pbuffer.gamma) * (pow(JBar_new, -pbuffer.gamma) - 1.f);
      {
        contrib[0] = ((C[0] + C[0]) * Dp_inv * pbuffer.visco - pressure) * voln ;
        contrib[1] = (C[1] + C[3]) * Dp_inv *  pbuffer.visco * voln ;
        contrib[2] = (C[2] + C[6]) * Dp_inv *  pbuffer.visco * voln ;
        contrib[3] = (C[3] + C[1]) * Dp_inv *  pbuffer.visco * voln ;
        contrib[4] = ((C[4] + C[4]) * Dp_inv *  pbuffer.visco - pressure) * voln ;
        contrib[5] = (C[5] + C[7]) * Dp_inv *  pbuffer.visco * voln ;
        contrib[6] = (C[6] + C[2]) * Dp_inv *  pbuffer.visco * voln ;
        contrib[7] = (C[7] + C[5]) * Dp_inv *  pbuffer.visco * voln ;
        contrib[8] = ((C[8] + C[8]) * Dp_inv *  pbuffer.visco - pressure) * voln ;
      }
      //contrib = contrib.cast<float>();
      // Merged affine matrix and stress contribution for MLS-MPM P2G
      contrib = (C * pbuffer.mass - contrib * newDt) * Dp_inv;
      {
        auto particle_bin = next_pbuffer.ch(_0, partition._binsts[src_blockno] +
                                                    pidib / g_bin_capacity);
        particle_bin.val(_0, pidib % g_bin_capacity) = pos[0]; //< x
        particle_bin.val(_1, pidib % g_bin_capacity) = pos[1]; //< y
        particle_bin.val(_2, pidib % g_bin_capacity) = pos[2]; //< z
        particle_bin.val(_3, pidib % g_bin_capacity) = 1.0 - J;      //< V/Vo
        particle_bin.val(_4, pidib % g_bin_capacity) = vel[0]; //< vx
        particle_bin.val(_5, pidib % g_bin_capacity) = vel[1]; //< vy
        particle_bin.val(_6, pidib % g_bin_capacity) = vel[2]; //< vz
        particle_bin.val(_7, pidib % g_bin_capacity) = voln;   //< FBar volume [m3]
        particle_bin.val(_8, pidib % g_bin_capacity) = 1.0 - JBar_new; //< JBar [ ]
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
      PREC d =
          (local_pos[dd] - ((int)(local_pos[dd] * g_dx_inv + 0.5f) - 1) * g_dx) *
          g_dx_inv;
      dws(dd, 0) = 0.5f * (1.5 - d) * (1.5 - d);
      d -= 1.f;
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
          pos = pvec3{(PREC)i, (PREC)j, (PREC)k} * g_dx - local_pos;
          PREC W = dws(0, i) * dws(1, j) * dws(2, k);
          PREC wm = pbuffer.mass * W;
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
    int channelid = base % numMViPerBlock; //& (numMViPerBlock - 1);
    char c = channelid % g_blockvolume;
    char cz = channelid & g_blockmask;
    char cy = (channelid >>= g_blockbits) & g_blockmask;
    char cx = (channelid >>= g_blockbits) & g_blockmask;
    channelid >>= g_blockbits;
    PREC_G val =
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
    } else if (channelid == 4) {
      atomicAdd(&next_grid.ch(_0, blockno).val_1d(_4, c), val);
    } else if (channelid == 5) {
      atomicAdd(&next_grid.ch(_0, blockno).val_1d(_5, c), val);
    } else if (channelid == 6) {
      atomicAdd(&next_grid.ch(_0, blockno).val_1d(_6, c), val);
    }
  }
}

template <typename ParticleBuffer, typename Partition, typename Grid, typename VerticeArray>
__global__ void v2p2g(float dt, float newDt, const ivec3 *__restrict__ blocks,
                      const ParticleBuffer pbuffer,
                      ParticleBuffer next_pbuffer,
                      const Partition prev_partition, Partition partition,
                      const Grid grid, Grid next_grid,
                      VerticeArray vertice_array) {
                        return;
                      }

// Grid-to-Particle-to-Grid + Mesh Update - ASFLIP Transfer
// Stress/Strain not computed here, displacements sent to FE mesh for force calc
template <typename Partition, typename Grid, typename VerticeArray>
__global__ void v2p2g(float dt, float newDt, const ivec3 *__restrict__ blocks,
                      const ParticleBuffer<material_e::Meshed> pbuffer,
                      ParticleBuffer<material_e::Meshed> next_pbuffer,
                      const Partition prev_partition, Partition partition,
                      const Grid grid, Grid next_grid,
                      VerticeArray vertice_array) {
  static constexpr uint64_t numViPerBlock = g_blockvolume * 6;
  static constexpr uint64_t numViInArena = numViPerBlock << 3;
  static constexpr uint64_t shmem_offset = (g_blockvolume * 6) << 3;

  static constexpr uint64_t numMViPerBlock = g_blockvolume * 7;
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
      *reinterpret_cast<MViArena>(shmem + shmem_offset * sizeof(float));

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
    else if (channelid == 3) 
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
    pvec3 pos;  //< Particle position at n
    int ID; // Vertice ID for mesh
    pvec3 vp_n; //< Particle vel. at n
    //float tension;
    PREC beta = 0;
    pvec3 b;
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
      //beta = source_particle_bin.val(_7, source_pidib % g_bin_capacity); //< 
      b[0] = source_particle_bin.val(_8, source_pidib % g_bin_capacity); //< bx
      b[1] = source_particle_bin.val(_9, source_pidib % g_bin_capacity); //< by
      b[2] = source_particle_bin.val(_10, source_pidib % g_bin_capacity); //< bz
    }
    ivec3 local_base_index = (pos * g_dx_inv + 0.5f).cast<int>() - 1;
    pvec3 local_pos = pos - local_base_index * g_dx;
    base_index = local_base_index;

    pvec3x3 dws;
#pragma unroll 3
    for (int dd = 0; dd < 3; ++dd) {
      PREC d =
          (local_pos[dd] - ((int)(local_pos[dd] * g_dx_inv + 0.5f) - 1) * g_dx) *
          g_dx_inv;
      dws(dd, 0) = 0.5f * (1.5 - d) * (1.5 - d);
      d -= 1.0f;
      dws(dd, 1) = 0.75 - d * d;
      d = 0.5f + d;
      dws(dd, 2) = 0.5 * d * d;
      local_base_index[dd] = ((local_base_index[dd] - 1) & g_blockmask) + 1;
    }
    pvec3 vel;   //< Stressed, collided grid velocity
    pvec3 vel_n; //< Unstressed, uncollided grid velocity
    pvec9 C;     //< APIC affine matrix, used a few times
    vel.set(0.0); 
    vel_n.set(0.0);
    C.set(0.0);

    // Dp^n = Dp^n+1 = (1/4) * dx^2 * I (Quad.)
    PREC Dp_inv; //< Inverse Intertia-Like Tensor (1/m^2)
    PREC scale = pbuffer.length * pbuffer.length; //< Area scale (m^2)
    Dp_inv = g_D_inv / scale; //< Scalar 4/(dx^2) for Quad. B-Spline

#pragma unroll 3
    for (char i = 0; i < 3; i++)
#pragma unroll 3
      for (char j = 0; j < 3; j++)
#pragma unroll 3
        for (char k = 0; k < 3; k++) {
          pvec3 xixp = pvec3{(PREC)i, (PREC)j, (PREC)k} * g_dx - local_pos;
          PREC W = dws(0, i) * dws(1, j) * dws(2, k);
          pvec3 vi{g2pbuffer[0][local_base_index[0] + i][local_base_index[1] + j]
                           [local_base_index[2] + k],
                  g2pbuffer[1][local_base_index[0] + i][local_base_index[1] + j]
                           [local_base_index[2] + k],
                  g2pbuffer[2][local_base_index[0] + i][local_base_index[1] + j]
                           [local_base_index[2] + k]};
          pvec3 vi_n{g2pbuffer[3][local_base_index[0] + i][local_base_index[1] + j]
                            [local_base_index[2] + k],
                    g2pbuffer[4][local_base_index[0] + i][local_base_index[1] + j]
                            [local_base_index[2] + k],
                    g2pbuffer[5][local_base_index[0] + i][local_base_index[1] + j]
                            [local_base_index[2] + k]};
          vel += vi * W;
          vel_n += vi_n * W; 
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

    //J = (1 + (C[0] + C[4] + C[8]) * dt * Dp_inv) * J;

    //vec3 b_new; //< Vertex normal, area weighted
    //b_new.set(0.f);
    pvec3 f; //< Internal force at FEM nodes
    f.set(0.0);
    PREC restVolume;
    PREC new_tension;
    PREC restMass;
    PREC count;
    PREC voln;
    PREC J, pressure, von_mises;
    count = 0.f;
    {
      // b[0] = vertice_array.val(_3, ID);
      // b[1] = vertice_array.val(_4, ID);
      // b[2] = vertice_array.val(_5, ID);
      J = vertice_array.val(_3, ID);
      pressure = vertice_array.val(_4, ID);
      von_mises = vertice_array.val(_5, ID);
      restVolume = vertice_array.val(_6, ID);
      f[0] = vertice_array.val(_7, ID);
      f[1] = vertice_array.val(_8, ID);
      f[2] = vertice_array.val(_9, ID);
      count = vertice_array.val(_10, ID);

      J = J / restVolume;
      pressure = pressure / restVolume;
      von_mises = von_mises / restVolume;
      restMass =  pbuffer.rho * fabs(restVolume);

      // float bMag;
      //float bMag = count;
      // float bMag = sqrtf(b[0]*b[0] + b[1]*b[1] + b[2]*b[2]);
      // if (bMag < 0.00001f) b.set(0.f);
      // else {
      //   b[0] = b[0]/bMag;
      //   b[1] = b[1]/bMag; 
      //   b[2] = b[2]/bMag;  
      // }
      // Zero-out FEM mesh nodal values
      vertice_array.val(_3, ID) = 0.f; // bx
      vertice_array.val(_4, ID) = 0.f; // by
      vertice_array.val(_5, ID) = 0.f; // bz 
      vertice_array.val(_6, ID) = 0.f; // vol
      vertice_array.val(_7, ID) = 0.f; // fx
      vertice_array.val(_8, ID) = 0.f; // fy
      vertice_array.val(_9, ID) = 0.f; // fz
      vertice_array.val(_10, ID) = 0.f; // count

    }
    
    // float new_beta;
    // if (new_tension >= 1.f) new_beta = pbuffer.beta_max; //< Position correction factor (ASFLIP)
    // else if (new_tension <= -1.f) new_beta = 0.f;
    // else new_beta = pbuffer.beta_min;

    int surface_ASFLIP = 0; //< 1 only applies ASFLIP to surface of mesh
    if (surface_ASFLIP && count > 10.f) {
      // Interior (ASFLIP)
      pos += dt * vel;
      vel += pbuffer.alpha * (vp_n - vel_n);
    } else if (surface_ASFLIP && count <= 10.f) {
      // Surface (FLIP/PIC)
      pos += dt * (vel + beta * pbuffer.alpha * (vp_n - vel_n));
      vel += pbuffer.alpha * (vp_n - vel_n);
    } else {
      // Interior and Surface (ASFLIP)
      pos += dt * (vel + beta * pbuffer.alpha * (vp_n - vel_n));
      vel += pbuffer.alpha * (vp_n - vel_n);
    } 

    {
      {
        auto particle_bin = next_pbuffer.ch(_0, partition._binsts[src_blockno] +
                                                    pidib / g_bin_capacity);
        particle_bin.val(_0, pidib % g_bin_capacity) = pos[0]; //< x [ ]
        particle_bin.val(_1, pidib % g_bin_capacity) = pos[1]; //< y [ ]
        particle_bin.val(_2, pidib % g_bin_capacity) = pos[2]; //< z [ ]
        particle_bin.val(_3, pidib % g_bin_capacity) = (PREC)ID; //< ID
        particle_bin.val(_4, pidib % g_bin_capacity) = vel[0]; //< vx [ /s]
        particle_bin.val(_5, pidib % g_bin_capacity) = vel[1]; //< vy [ /s]
        particle_bin.val(_6, pidib % g_bin_capacity) = vel[2]; //< vz [ /s]
        particle_bin.val(_7, pidib % g_bin_capacity) = restVolume; // volume [m3]
        particle_bin.val(_8, pidib % g_bin_capacity)  = J; //f[0]; // bx
        particle_bin.val(_9, pidib % g_bin_capacity)  = pressure; //f[1]; // by
        particle_bin.val(_10, pidib % g_bin_capacity) = von_mises; //f[2]; // bz

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
      PREC d =
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
          pos = pvec3{(PREC)i, (PREC)j, (PREC)k} * g_dx - local_pos;
          PREC W = dws(0, i) * dws(1, j) * dws(2, k);
          PREC wm = restMass * W;
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
    int channelid = base % numMViPerBlock;
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
    } else if (channelid == 4) {
      atomicAdd(&next_grid.ch(_0, blockno).val_1d(_4, c), val);
    } else if (channelid == 5) {
      atomicAdd(&next_grid.ch(_0, blockno).val_1d(_5, c), val);
    } else if (channelid == 6) {
      atomicAdd(&next_grid.ch(_0, blockno).val_1d(_6, c), val);
    }
  }
}


template <typename Partition, typename Grid, typename ParticleBuffer, typename VerticeArray>
__global__ void v2p2g_FBar(float dt, float newDt, const ivec3 *__restrict__ blocks,
                      const ParticleBuffer pbuffer,
                      ParticleBuffer next_pbuffer,
                      const Partition prev_partition, Partition partition,
                      const Grid grid, Grid next_grid,
                      VerticeArray vertice_array) {
                        return;
                      }

// Grid-to-Particle-to-Grid + Mesh Update - F-Bar ASFLIP Transfer
// Stress/Strain not computed here, displacements sent to FE mesh for force calc
template <typename Partition, typename Grid, typename VerticeArray>
__global__ void v2p2g_FBar(float dt, float newDt, const ivec3 *__restrict__ blocks,
                      const ParticleBuffer<material_e::Meshed> pbuffer,
                      ParticleBuffer<material_e::Meshed> next_pbuffer,
                      const Partition prev_partition, Partition partition,
                      const Grid grid, Grid next_grid,
                      VerticeArray vertice_array) {
  static constexpr uint64_t numViPerBlock = g_blockvolume * 6;
  static constexpr uint64_t numViInArena = numViPerBlock << 3;
  static constexpr uint64_t shmem_offset = (g_blockvolume * 6) << 3;

  static constexpr uint64_t numMViPerBlock = g_blockvolume * 7;
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
      *reinterpret_cast<MViArena>(shmem + shmem_offset * sizeof(float));

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
    else if (channelid == 3) 
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
    pvec3 pos;  //< Particle position at n
    int ID; // Vertice ID for mesh
    pvec3 vp_n; //< Particle vel. at n
    //float tension;
    PREC beta = 0;
    pvec3 b;
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
      //beta = source_particle_bin.val(_7, source_pidib % g_bin_capacity); //< 
      b[0] = source_particle_bin.val(_8, source_pidib % g_bin_capacity); //< bx
      b[1] = source_particle_bin.val(_9, source_pidib % g_bin_capacity); //< by
      b[2] = source_particle_bin.val(_10, source_pidib % g_bin_capacity); //< bz
    }
    ivec3 local_base_index = (pos * g_dx_inv + 0.5f).cast<int>() - 1;
    pvec3 local_pos = pos - local_base_index * g_dx;
    base_index = local_base_index;

    pvec3x3 dws;
#pragma unroll 3
    for (int dd = 0; dd < 3; ++dd) {
      PREC d =
          (local_pos[dd] - ((int)(local_pos[dd] * g_dx_inv + 0.5f) - 1) * g_dx) *
          g_dx_inv;
      dws(dd, 0) = 0.5 * (1.5 - d) * (1.5 - d);
      d -= 1.0;
      dws(dd, 1) = 0.75 - d * d;
      d = 0.5 + d;
      dws(dd, 2) = 0.5 * d * d;
      local_base_index[dd] = ((local_base_index[dd] - 1) & g_blockmask) + 1;
    }
    pvec3 vel;   //< Stressed, collided grid velocity
    pvec3 vel_n; //< Unstressed, uncollided grid velocity
    pvec9 C;     //< APIC affine matrix, used a few times
    vel.set(0.0); 
    vel_n.set(0.0);
    C.set(0.0);

    // Dp^n = Dp^n+1 = (1/4) * dx^2 * I (Quad.)
    PREC Dp_inv; //< Inverse Intertia-Like Tensor (1/m^2)
    PREC scale = pbuffer.length * pbuffer.length; //< Area scale (m^2)
    Dp_inv = g_D_inv / scale; //< Scalar 4/(dx^2) for Quad. B-Spline

#pragma unroll 3
    for (char i = 0; i < 3; i++)
#pragma unroll 3
      for (char j = 0; j < 3; j++)
#pragma unroll 3
        for (char k = 0; k < 3; k++) {
          pvec3 xixp = pvec3{(PREC)i, (PREC)j, (PREC)k} * g_dx - local_pos;
          PREC W = dws(0, i) * dws(1, j) * dws(2, k);
          pvec3 vi{g2pbuffer[0][local_base_index[0] + i][local_base_index[1] + j]
                           [local_base_index[2] + k],
                  g2pbuffer[1][local_base_index[0] + i][local_base_index[1] + j]
                           [local_base_index[2] + k],
                  g2pbuffer[2][local_base_index[0] + i][local_base_index[1] + j]
                           [local_base_index[2] + k]};
          pvec3 vi_n{g2pbuffer[3][local_base_index[0] + i][local_base_index[1] + j]
                            [local_base_index[2] + k],
                    g2pbuffer[4][local_base_index[0] + i][local_base_index[1] + j]
                            [local_base_index[2] + k],
                    g2pbuffer[5][local_base_index[0] + i][local_base_index[1] + j]
                            [local_base_index[2] + k]};
          vel += vi * W;
          vel_n += vi_n * W; 
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

    pvec3 f; //< Internal force at FEM nodes
    f.set(0.0);
    PREC restVolume;
    PREC new_tension;
    PREC restMass;
    PREC count;
    PREC voln, Vn, JBar;
    PREC J, pressure, von_mises;
    count = 0;
    {
      // b[0] = vertice_array.val(_3, ID);
      // b[1] = vertice_array.val(_4, ID);
      // b[2] = vertice_array.val(_5, ID);
      J = vertice_array.val(_3, ID);
      pressure = vertice_array.val(_4, ID);
      von_mises = vertice_array.val(_5, ID);
      restVolume = vertice_array.val(_6, ID);
      f[0] = vertice_array.val(_7, ID);
      f[1] = vertice_array.val(_8, ID);
      f[2] = vertice_array.val(_9, ID);
      count = vertice_array.val(_10, ID);
      Vn = vertice_array.val(_11, ID);
      JBar = vertice_array.val(_12, ID);
      
      J = (J / restVolume);
      JBar = (JBar / restVolume);
      pressure = (pressure / restVolume);
      von_mises = (von_mises / restVolume);

      restMass =  pbuffer.rho * abs(restVolume);


      // float bMag;
      //float bMag = count;
      // PREC bMag = sqrt(b[0]*b[0] + b[1]*b[1] + b[2]*b[2]);
      // if (bMag < 0.00001f) b.set(0.0);
      // else {
      //   b[0] = b[0]/bMag;
      //   b[1] = b[1]/bMag; 
      //   b[2] = b[2]/bMag;  
      // }
      // Zero-out FEM mesh nodal values
      vertice_array.val(_3, ID) = 0.0; // bx
      vertice_array.val(_4, ID) = 0.0; // by
      vertice_array.val(_5, ID) = 0.0; // bz 
      vertice_array.val(_6, ID) = 0.0; // vol
      vertice_array.val(_7, ID) = 0.0; // fx
      vertice_array.val(_8, ID) = 0.0; // fy
      vertice_array.val(_9, ID) = 0.0; // fz
      vertice_array.val(_10, ID) = 0.0; // count
      vertice_array.val(_11, ID) = 0.0; // Vn
      vertice_array.val(_12, ID) = 0.0; // (1 - JInc JBar) Vn
    }
    
    // PREC new_beta;
    // if (new_tension >= 1.f) new_beta = pbuffer.beta_max; //< Position correction factor (ASFLIP)
    // else if (new_tension <= -1.f) new_beta = 0.f;
    // else new_beta = pbuffer.beta_min;

    int surface_ASFLIP = 0; //< 1 only applies ASFLIP to surface of mesh
    if (surface_ASFLIP && count > 10.f) {
      // Interior (ASFLIP)
      pos += dt * vel;
      vel += pbuffer.alpha * (vp_n - vel_n);
    } else if (surface_ASFLIP && count <= 10.f) {
      // Surface (FLIP/PIC)
      pos += dt * (vel + beta * pbuffer.alpha * (vp_n - vel_n));
      vel += pbuffer.alpha * (vp_n - vel_n);
    } else {
      // Interior and Surface (ASFLIP)
      pos += dt * (vel + beta * pbuffer.alpha * (vp_n - vel_n));
      vel += pbuffer.alpha * (vp_n - vel_n);
    } 

    {
      {
        auto particle_bin = next_pbuffer.ch(_0, partition._binsts[src_blockno] +
                                                    pidib / g_bin_capacity);
        particle_bin.val(_0, pidib % g_bin_capacity) = pos[0]; //< x [ ]
        particle_bin.val(_1, pidib % g_bin_capacity) = pos[1]; //< y [ ]
        particle_bin.val(_2, pidib % g_bin_capacity) = pos[2]; //< z [ ]
        particle_bin.val(_3, pidib % g_bin_capacity) = (PREC)ID; //< ID
        particle_bin.val(_4, pidib % g_bin_capacity) = vel[0]; //< vx [ /s]
        particle_bin.val(_5, pidib % g_bin_capacity) = vel[1]; //< vy [ /s]
        particle_bin.val(_6, pidib % g_bin_capacity) = vel[2]; //< vz [ /s]
        particle_bin.val(_7, pidib % g_bin_capacity) = restVolume; // volume [m3]
        particle_bin.val(_8, pidib % g_bin_capacity)  = JBar; // bx
        particle_bin.val(_9, pidib % g_bin_capacity)  = pressure; // by
        particle_bin.val(_10, pidib % g_bin_capacity) = von_mises; // bz

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
      PREC d =
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
          pos = pvec3{(PREC)i, (PREC)j, (PREC)k} * g_dx - local_pos;
          PREC W = dws(0, i) * dws(1, j) * dws(2, k);
          PREC wm = restMass * W;
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
    int channelid = base % numMViPerBlock;
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
    } else if (channelid == 4) {
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
    // MLS-MPM: grid-node mass and fused moemntum + internal force
    targetblock.val_1d(_0, threadIdx.x) = sourceblock.val_1d(_0, threadIdx.x); //< mass
    targetblock.val_1d(_1, threadIdx.x) = sourceblock.val_1d(_1, threadIdx.x); //< mvel_x + fint_x
    targetblock.val_1d(_2, threadIdx.x) = sourceblock.val_1d(_2, threadIdx.x); //< mvel_y + fint_y
    targetblock.val_1d(_3, threadIdx.x) = sourceblock.val_1d(_3, threadIdx.x); //< mvel_z + fint_z
    // ASFLIP: Velocities unstressed
    targetblock.val_1d(_4, threadIdx.x) = sourceblock.val_1d(_4, threadIdx.x); //< mvel_x
    targetblock.val_1d(_5, threadIdx.x) = sourceblock.val_1d(_5, threadIdx.x); //< mvel_y
    targetblock.val_1d(_6, threadIdx.x) = sourceblock.val_1d(_6, threadIdx.x); //< mvel_z
    // Simple F-Bar: Volume and volume change ratio
    targetblock.val_1d(_7, threadIdx.x) = sourceblock.val_1d(_7, threadIdx.x); // Volume
    targetblock.val_1d(_8, threadIdx.x) = sourceblock.val_1d(_8, threadIdx.x); // JBar
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
  atomicAdd(sum, (float)grid.ch(_0, blockIdx.x).val_1d(_0, threadIdx.x));
}
// Added for Simple FBar Method - Justin
template <typename Grid> __global__ void sum_grid_volume(Grid grid, float *sum) {
  atomicAdd(sum, (float)grid.ch(_0, blockIdx.x).val_1d(_7, threadIdx.x));
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
                                         PREC *dispVal, 
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
                                         PREC *dispVal, 
                                         int *_parcnt, PREC length) {
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
    PREC o = g_offset;
    PREC l = length;
    /// Send positions (x,y,z) [m] to parray (device --> device)
    parray.val(_0, parid) = (source_bin.val(_0, _source_pidib) - o) * l;
    parray.val(_1, parid) = (source_bin.val(_1, _source_pidib) - o) * l;
    parray.val(_2, parid) = (source_bin.val(_2, _source_pidib) - o) * l;

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
                                         PREC *dispVal, 
                                         int *_parcnt, PREC length) {
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
    PREC o = g_offset;
    PREC l = length;
    /// Send positions (x,y,z) [m] to parray (device --> device)
    parray.val(_0, parid) = (source_bin.val(_0, _source_pidib) - o) * l;
    parray.val(_1, parid) = (source_bin.val(_1, _source_pidib) - o) * l;
    parray.val(_2, parid) = (source_bin.val(_2, _source_pidib) - o) * l;
    if (1) {
      /// Send attributes (J, P, Vel_y) to pattribs (device --> device)
      PREC J = 1.f - source_bin.val(_3, _source_pidib);
      PREC pressure = (pbuffer.bulk / pbuffer.gamma) * 
        (pow(J, -pbuffer.gamma) - 1.f);       //< Tait-Murnaghan Pressure (Pa)
      pattrib.val(_0, parid) = J;       //< V/Vo
      pattrib.val(_1, parid) = pressure; //< Pressure (Pa)
      pattrib.val(_2, parid) = source_bin.val(_5, _source_pidib) * l; // Vel_y
    }
  }
}


template <typename Partition, typename ParticleArray>
__global__ void
retrieve_particle_buffer_attributes(Partition partition,
                                         Partition prev_partition,
                                         ParticleBuffer<material_e::JBarFluid> pbuffer,
                                         ParticleArray parray, 
                                         ParticleArray pattrib,
                                         PREC *dispVal, 
                                         int *_parcnt, PREC length) {
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
    PREC o = g_offset;
    PREC l = length;
    auto output_attribs = pbuffer.output_attribs;
    /// Send positions (x,y,z) [m] to parray (device --> device)
    parray.val(_0, parid) = (source_bin.val(_0, _source_pidib) - o) * l;
    parray.val(_1, parid) = (source_bin.val(_1, _source_pidib) - o) * l;
    parray.val(_2, parid) = (source_bin.val(_2, _source_pidib) - o) * l;

    PREC J    = 1.0 - source_bin.val(_3, _source_pidib);
    PREC JBar = 1.0 - source_bin.val(_8, _source_pidib);
    PREC pressure = (pbuffer.bulk / pbuffer.gamma) * 
      (pow(JBar, -pbuffer.gamma) - 1.0); //< Tait-Murnaghan Pressure (Pa)

    for (int i = 0; i < 3; i++) {
      /// Send attributes to pattribs (device --> device)
      int idx = output_attribs[i]; //< Map index for output attribute (particle_buffer.cuh)
      PREC val;
      if      (idx == 0)
        val = (source_bin.val(_0, _source_pidib) - o) * l; // Vel_x [m/s]
      else if (idx == 1)
        val = (source_bin.val(_1, _source_pidib) - o) * l; // Vel_y [m/s]
      else if (idx == 2)
        val = (source_bin.val(_2, _source_pidib) - o) * l; // Vel_z [m/s]
      else if (idx == 3)
        val = source_bin.val(_4, _source_pidib) * l; // Vel_x [m/s]
      else if (idx == 4)
        val = source_bin.val(_5, _source_pidib) * l; // Vel_y [m/s]
      else if (idx == 5)
        val = source_bin.val(_6, _source_pidib) * l; // Vel_z [m/s]
      else if (idx == 6)
        val = J;  //< J [], Vo/V, det| F |
      else if (idx == 7)
        val = JBar;  //< J_Bar [], Simple F-Bar method
      else if (idx == 8)
        val = pressure; // Pressure [Pa]
      else
        val = -1; // Incorrect output attributes name
      if      (i == 0) pattrib.val(_0, parid) = val;  
      else if (i == 1) pattrib.val(_1, parid) = val; 
      else if (i == 2) pattrib.val(_2, parid) = val; 
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
                                         PREC *dispVal, 
                                         int *_parcnt, PREC length) {
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
    PREC o = g_offset;
    PREC l = length;
    /// Send positions (x,y,z) [m] to parray (device --> device)
    parray.val(_0, parid) = (source_bin.val(_0, _source_pidib) - o) * l;
    parray.val(_1, parid) = (source_bin.val(_1, _source_pidib) - o) * l;
    parray.val(_2, parid) = (source_bin.val(_2, _source_pidib) - o) * l;

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
                                         PREC *dispVal, 
                                         int *_parcnt, PREC length) {
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
    PREC o = g_offset;
    PREC l = length;
    /// Send positions (x,y,z) [m] to parray (device --> device)
    parray.val(_0, parid) = (source_bin.val(_0, _source_pidib) - o) * l;
    parray.val(_1, parid) = (source_bin.val(_1, _source_pidib) - o) * l;
    parray.val(_2, parid) = (source_bin.val(_2, _source_pidib) - o) * l;

    if (1) {
      /// Send attributes (Left-Strain Invariants) to pattribs (device --> device)
      pvec9 F; //< Deformation Gradient
      F[0] = source_bin.val(_3,  _source_pidib);
      F[1] = source_bin.val(_4,  _source_pidib);
      F[2] = source_bin.val(_5,  _source_pidib);
      F[3] = source_bin.val(_6,  _source_pidib);
      F[4] = source_bin.val(_7,  _source_pidib);
      F[5] = source_bin.val(_8,  _source_pidib);
      F[6] = source_bin.val(_9,  _source_pidib);
      F[7] = source_bin.val(_10, _source_pidib);
      F[8] = source_bin.val(_11, _source_pidib);
      PREC J  = matrixDeterminant3d(F.data());

      // PREC U[9], S[3], V[9]; //< Left, Singulars, and Right Values of Strain
      // math::svd(F[0], F[3], F[6], F[1], F[4], F[7], F[2], F[5], F[8], U[0], U[3],
      //           U[6], U[1], U[4], U[7], U[2], U[5], U[8], S[0], S[1], S[2], V[0],
      //           V[3], V[6], V[1], V[4], V[7], V[2], V[5], V[8]); // SVD Operation
      // PREC I1, I2, I3; // Principal Invariants
      // I1 = U[0] + U[4] + U[8];    //< I1 = tr(C)
      // I2 = U[0]*U[4] + U[4]*U[8] + 
      //      U[0]*U[8] - U[3]*U[3] - 
      //      U[6]*U[6] - U[7]*U[7]; //< I2 = 1/2((tr(C))^2 - tr(C^2))
      // I3 = matrixDeterminant3d(F.data());
      pvec9 C;
      compute_stress_fixedcorotated((PREC)(1/J), pbuffer.mu, pbuffer.lambda, F, C);
      PREC pressure = - (C[0] + C[4] + C[8]) / (3.0);

      // Set pattribs for particle to Principal Strain Invariants
      
      PREC von_mises = sqrt( (  (C[0]-C[4])*(C[0]-C[4]) + 
                                (C[4]-C[8])*(C[4]-C[8]) + 
                                (C[8]-C[0])*(C[8]-C[0]) + 
                                6*(C[3]*C[3] + C[6]*C[6] + C[7]*C[7]) ) / 2.0);
      
      pattrib.val(_0, parid) = J;
      pattrib.val(_1, parid) = pressure; //< I2
      pattrib.val(_2, parid) = von_mises; //< vy
      //pattrib.val(_2, parid) = source_bin.val(_13, _source_pidib) * l; //< vy
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
                                         PREC *dispVal, 
                                         int *_parcnt, PREC length) {
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
    PREC o = g_offset;
    PREC l = length;
    /// Send positions (x,y,z) [m] to parray (device --> device)
    parray.val(_0, parid) = (source_bin.val(_0, _source_pidib) - o) * l;
    parray.val(_1, parid) = (source_bin.val(_1, _source_pidib) - o) * l;
    parray.val(_2, parid) = (source_bin.val(_2, _source_pidib) - o) * l;

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
                                         PREC *dispVal, 
                                         int *_parcnt, PREC length) {
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
    PREC o = g_offset;
    PREC l = length;
    /// Send positions (x,y,z) [m] to parray (device --> device)
    parray.val(_0, parid) = (source_bin.val(_0, _source_pidib) - o) * l;
    parray.val(_1, parid) = (source_bin.val(_1, _source_pidib) - o) * l;
    parray.val(_2, parid) = (source_bin.val(_2, _source_pidib) - o) * l;

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
                                         PREC *trackVal, 
                                         int *_parcnt, PREC length) {
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
    PREC o = g_offset; // Off-by-two buffer, 8 dx, on sim. domain
    PREC l = length;
    /// Send positions (x,y,z) [0.0, 1.0] to parray (device --> device)
    parray.val(_0, parid) = (source_bin.val(_0, _source_pidib) - o) * l;
    parray.val(_1, parid) = (source_bin.val(_1, _source_pidib) - o) * l;
    parray.val(_2, parid) = (source_bin.val(_2, _source_pidib) - o) * l;

    if (1) {
      /// Send attributes to pattribs (device --> device)
      PREC JBar = 1.0 - source_bin.val(_8, _source_pidib);
      PREC pressure  = source_bin.val(_9, _source_pidib);
      PREC von_mises = source_bin.val(_10, _source_pidib);
      
      pattrib.val(_0, parid) = JBar;
      pattrib.val(_1, parid) = pressure; //< I2
      pattrib.val(_2, parid) = von_mises; //< vy
    }
    if (1) {
      /// Set desired value of tracked particle
      auto ID = source_bin.val(_3, _source_pidib);
      for (int i = 0; i < sizeof(g_track_IDs)/4; i++) {
        if (ID == g_track_ID) {
          PREC v = (source_bin.val(_1, _source_pidib) - o) * l;
          atomicAdd(trackVal, v);
        }
      }
    }
  }
}

/// Retrieve grid-cells between points a & b from grid-buffer to gridTarget (JB)
template <typename Partition, typename Grid, typename GridTarget>
__global__ void retrieve_selected_grid_cells(
    uint32_t blockCount, const Partition partition,
    Grid prev_grid, GridTarget garray,
    float dt, float *forceSum, vec3 point_a, vec3 point_b, PREC length) {

  auto blockno = blockIdx.x;  //< Block number in partition
  auto blockid = partition._activeKeys[blockno];

  // Check if gridblock contains part of the point_a to point_b region
  // End all threads in block if not
  if ((4*blockid[0] + 3)*g_dx < point_a[0] || (4*blockid[0])*g_dx > point_b[0]) return;
  if ((4*blockid[1] + 3)*g_dx < point_a[1] || (4*blockid[1])*g_dx > point_b[1]) return;
  if ((4*blockid[2] + 3)*g_dx < point_a[2] || (4*blockid[2])*g_dx > point_b[2]) return;
  

  //auto blockid = prev_blockids[blockno]; //< 3D grid-block index
  if (blockno < blockCount) {

    auto sourceblock = prev_grid.ch(_0, blockno); //< Set grid-block by block index
    float tol = g_dx * 0.0f; // Tolerance layer around target domain
    float o = g_offset; // Offset [1]
    float l = length; // Domain length [m]
    // Add +1 to each? For point_b ~= point_a...
    ivec3 maxCoord;
    maxCoord[0] = (int)((point_b[0] + tol - point_a[0] + tol) * g_dx_inv + 1);
    maxCoord[1] = (int)((point_b[1] + tol - point_a[1] + tol) * g_dx_inv + 1);
    maxCoord[2] = (int)((point_b[2] + tol - point_a[2] + tol) * g_dx_inv + 1);
    int maxNodes = maxCoord[0] * maxCoord[1] * maxCoord[2];
    if (maxNodes >= g_target_cells && threadIdx.x == 0) printf("Allocate more space for gridTarget!\n");

    // Loop through cells in grid-block, stride by 32 to avoid thread conflicts
    for (int cidib = threadIdx.x % 32; cidib < g_blockvolume; cidib += 32) {

      // Grid node coordinate [i,j,k] in grid-block
      int i = (cidib >> (g_blockbits << 1)) & g_blockmask;
      int j = (cidib >> g_blockbits) & g_blockmask;
      int k = cidib & g_blockmask;

      // Grid node position [x,y,z] in entire domain 
      float xc = (4*blockid[0] + i)*g_dx; //< Node x position [1m]
      float yc = (4*blockid[1] + j)*g_dx; //< Node y position [1m]
      float zc = (4*blockid[2] + k)*g_dx; //< Node z position [1m]

      // Exit thread if cell is not inside grid-target +/- tol
      if (xc < point_a[0] - tol || xc > point_b[0] + tol) continue;
      if (yc < point_a[1] - tol || yc > point_b[1] + tol) continue;
      if (zc < point_a[2] - tol || zc > point_b[2] + tol) continue;

      // Unique ID by spatial position of cell in target [0 to g_target_cells-1]
      int node_id;
      node_id = ((int)((xc - point_a[0] + tol) * g_dx_inv) * maxCoord[1] * maxCoord[2]) +
                ((int)((yc - point_a[1] + tol) * g_dx_inv) * maxCoord[2]) +
                ((int)((zc - point_a[2] + tol) * g_dx_inv));

      /// Set values in grid-array to specific cell from grid-buffer
      garray.val(_0, node_id) = (xc - o) * l;
      garray.val(_1, node_id) = (yc - o) * l;
      garray.val(_2, node_id) = (zc - o) * l;
      garray.val(_3, node_id) = sourceblock.val(_0, i, j, k);
      garray.val(_4, node_id) = sourceblock.val(_1, i, j, k) * l;
      garray.val(_5, node_id) = sourceblock.val(_2, i, j, k) * l;
      garray.val(_6, node_id) = sourceblock.val(_3, i, j, k) * l;

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
    float dt, float *waveMax, vec3 point_a, vec3 point_b, PREC length) {

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
    float o = g_offset;
    float l = length;
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

      // Exit thread if cell is not inside wave-gauge domain +/- tol
      if (xc < point_a[0] - tol || xc > point_b[0] + tol) continue;
      if (yc < point_a[1] - tol || yc > point_b[1] + tol) continue;
      if (zc < point_a[2] - tol || zc > point_b[2] + tol) continue;

      /// Set values of cell (mass, momentum) from grid-buffer
      float mass = sourceblock.val(_0, i, j, k); // Mass [kg]
      if (mass <= 0.f) continue;
      float elev = (yc - o) * l; // Elevation [m]

      // Check for mass (material in cell, i.e wave surface)
      atomicMax(waveMax, elev);
    }
  }
}

} // namespace mn

#endif