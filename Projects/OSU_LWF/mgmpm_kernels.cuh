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
  uint32_t parid = blockIdx.x * blockDim.x + threadIdx.x; // Global particle ID
  if (parid >= particleCount) return; // Return if global ID exceeds particle count
  const ivec3 coord	= get_block_id<PREC>({parray.val(_0, parid), parray.val(_1, parid), parray.val(_2, parid)}) - 2;
  const ivec3 blockid = coord / static_cast<int>(config::g_blocksize); 
  partition.insert(blockid); // Create block in partition
}

// Used to associate particles with grid-cells. 
// Cell bucket array fills each bucket with contained particles' IDs in respective cell
template <typename ParticleArray, typename Partition>
__global__ void build_particle_cell_buckets(uint32_t particleCount,
                                            ParticleArray parray,
                                            Partition partition) {
  const uint32_t parid = blockIdx.x * blockDim.x + threadIdx.x; // Global particle ID
  if (parid >= particleCount) return; // Return if global ID exceeds particle count

	//Get block id by particle pos
	const ivec3 coord	= get_block_id<PREC>({parray.val(_0, parid), parray.val(_1, parid), parray.val(_2, parid)}) - 2;
	const ivec3 blockid = coord / static_cast<int>(config::g_blocksize);
  auto blockno = partition.query(blockid); // Block number
  int cellno = (coord[0] & g_blockmask) * g_blocksize * g_blocksize +
               (coord[1] & g_blockmask) * g_blocksize +
               (coord[2] & g_blockmask); // Cell number
  auto pidic = atomicAdd(partition._ppcs + blockno * g_blockvolume + cellno, 1); // ID in cell
  if (pidic >= g_max_ppc) {
    atomicSub(partition._ppcs + blockno * g_blockvolume + cellno, 1); // Reduce count
#if PRINT_CELL_OVERFLOW
    printf("No space left in cell: block(%d), cell(%d)\n", blockno, cellno);
#endif
    return;
  }
  partition._cellbuckets[blockno * g_particle_num_per_block + cellno * g_max_ppc + 
                         pidic] = static_cast<int>(parid); // Store particle ID in cell bucket. Max IDs: g_max_ppc
}
template <typename ParticleArray, typename ParticleBuffer, typename Partition>
__global__ void
build_particle_cell_buckets(uint32_t particleCount, ParticleArray parray,
                            ParticleBuffer pbuffer, Partition partition) {
  const uint32_t parid = blockIdx.x * blockDim.x + threadIdx.x; // Global particle ID
  if (parid >= particleCount) return; // Return if global ID exceeds particle count

	//Get block id by particle pos
	const ivec3 coord	= get_block_id<PREC>({parray.val(_0, parid), parray.val(_1, parid), parray.val(_2, parid)}) - 2;
	const ivec3 blockid = coord / static_cast<int>(config::g_blocksize);
  auto blockno = partition.query(blockid); // Block number
  int cellno = (coord[0] & g_blockmask) * g_blocksize * g_blocksize +
               (coord[1] & g_blockmask) * g_blocksize +
               (coord[2] & g_blockmask); // Cell number in block
  auto pidic = atomicAdd(pbuffer._ppcs + blockno * g_blockvolume + cellno, 1); // ID in cell
	if (pidic >= g_max_ppc) {
	  // If no space is left, don't store the particle. Reduce count again
    atomicSub(pbuffer._ppcs + blockno * g_blockvolume + cellno, 1);
#if PRINT_CELL_OVERFLOW
    printf("No space left in cell: block(%d), cell(%d)\n", blockno, cellno);
#endif
    return;
  }
  pbuffer._cellbuckets[blockno * g_particle_num_per_block + cellno * g_max_ppc +
                        pidic] = static_cast<int>(parid); // Store par. ID in cell bucket. Max IDs: g_max_ppc
}

// GMPM
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
// Compute num. particle bins needed per block. Used to bin particles, i.e. memory allocation
__global__ void compute_bin_capacity(uint32_t blockCount, int const *_ppbs,
                                     int *_bincaps) {
  uint32_t blockno = blockIdx.x * blockDim.x + threadIdx.x;
  if (blockno >= blockCount) return;
  _bincaps[blockno] = (_ppbs[blockno] + g_bin_capacity - 1) / g_bin_capacity;
}
// Init. block buckets with particle IDs, no advection (movement) for first time-step
__global__ void init_adv_bucket(const int *_ppbs, int *_buckets) {
  auto pcnt = _ppbs[blockIdx.x]; // Particle count in this block
  auto bucket = _buckets + blockIdx.x * g_particle_num_per_block; // Block bucket start adress
  // Put particle ID in block bucket. No advecting particles so offset {0,0,0} blocks
  for (int pidib = threadIdx.x; pidib < pcnt; pidib += blockDim.x) 
    // TODO : Check if original bitwise OR is correct, seems like it should be addition for any g_particle_num_per_block not power of 2 (i.e. if MAX_PPC is not a power of 2)
    // if is_power_of_2(g_particle_num_per_block)
    if (g_ppc_pow2)
      bucket[pidib] = (dir_offset(ivec3{0, 0, 0}) * g_particle_num_per_block) | pidib; 
    else
      bucket[pidib] = (dir_offset(ivec3{0, 0, 0}) * g_particle_num_per_block) + pidib; 
}
template <typename Grid> __global__ void clear_grid(Grid grid) {
  auto gridblock = grid.ch(_0, blockIdx.x);
  for (int cidib = threadIdx.x; cidib < g_blockvolume; cidib += blockDim.x) {
    // Mass, Mass*(vel + dt * fint) [MLS-MPM]
    gridblock.val_1d(_0, cidib) = 0.f;
    gridblock.val_1d(_1, cidib) = 0.f;
    gridblock.val_1d(_2, cidib) = 0.f;
    gridblock.val_1d(_3, cidib) = 0.f;
    // Mass*vel [ASFLIP, FLIP]
    gridblock.val_1d(_4, cidib) = 0.f;
    gridblock.val_1d(_5, cidib) = 0.f;
    gridblock.val_1d(_6, cidib) = 0.f;
    // Vol, JBar [Simple FBAR]
    gridblock.val_1d(_7, cidib) = 0.f;
    gridblock.val_1d(_8, cidib) = 0.f; 
#if DEBUG_COUPLED_UP
    // mass_water, pressure_water [CoupledUP]
    gridblock.val_1d(_9, cidib) = 0.f;
    gridblock.val_1d(_10, cidib) = 0.f; 
#endif
  }
}
template <typename Grid> __global__ void clear_grid_FBar(Grid grid) {
  auto gridblock = grid.ch(_0, blockIdx.x);
  for (int cidib = threadIdx.x; cidib < g_blockvolume; cidib += blockDim.x) {
    // Vol, JBar [Simple FBar]
    gridblock.val_1d(_7, cidib) = 0.f;
    gridblock.val_1d(_8, cidib) = 0.f;
  }
}
// Update partition's hash-table with neighbor blocks of active particle blocks. 
// Neighbor grid-blocks needed for grid-to-particle MPM interp. of particle block.
// Neighbor grid-blocks encase a given particle block (shifted half-block, so 2x2x2 neighbors).
template <typename Partition>
__global__ void 
register_neighbor_blocks(uint32_t blockCount, Partition partition) {
  uint32_t blockno = blockIdx.x * blockDim.x + threadIdx.x;
  if (blockno >= blockCount) return;
  auto blockid = partition._activeKeys[blockno];
#pragma unroll 2
  for (char i = 0; i < 2; ++i)
#pragma unroll 2
    for (char j = 0; j < 2; ++j)
#pragma unroll 2
      for (char k = 0; k < 2; ++k)
        partition.insert(ivec3{blockid[0] + i, blockid[1] + j, blockid[2] + k});
}
// Update partition's hash-table with exterior blocks of active particle blocks. 
// Exterior grid-blocks needed for particle advection and sharing GPU halo grid-blocks.
// Exterior grid-blocks encase a given particle block + any possible advection (3x3x3 exteriors).
template <typename Partition>
__global__ void 
register_exterior_blocks(uint32_t blockCount, Partition partition) {
  uint32_t blockno = blockIdx.x * blockDim.x + threadIdx.x;
  if (blockno >= blockCount) return;
  auto blockid = partition._activeKeys[blockno];
#pragma unroll 3
  for (char i = -1; i < 2; ++i)
#pragma unroll 3
    for (char j = -1; j < 2; ++j)
#pragma unroll 3
      for (char k = -1; k < 2; ++k)
        partition.insert(ivec3{blockid[0] + i, blockid[1] + j, blockid[2] + k});
}


template <typename ParticleBuffer, typename ParticleArray, typename Grid, typename Partition>
__global__ void rasterize(uint32_t particleCount, const ParticleBuffer pbuffer, const ParticleArray parray, Grid grid, const Partition partition, double dt, pvec3 vel0, PREC grav) {
  uint32_t parid = blockIdx.x * blockDim.x + threadIdx.x;
  if (parid >= particleCount) return;
  PREC length = pbuffer.length;
  PREC mass = pbuffer.mass;
  PREC volume = pbuffer.volume;
  pvec3 local_pos{(PREC)parray.val(_0, parid), (PREC)parray.val(_1, parid),
                 (PREC)parray.val(_2, parid)};
  pvec3 vel;
  pvec9 contrib, C;
  vel.set(0.), contrib.set(0.), C.set(0.);
  PREC J = 1.0;  // Volume ratio, Det def. gradient. 1 for t0 

  bool hydrostatic_init = false;
  if (hydrostatic_init) {
    PREC pressure = (mass/volume) * (grav) * (local_pos[1] - g_offset)*length; //< P = pgh
    PREC voln = J * volume;
    contrib[0] = - pressure * voln;
    contrib[4] = - pressure * voln;
    contrib[8] = - pressure * voln;
  }
  // // Leap-frog init vs symplectic Euler
  // /if (1) dt = dt / 2.0;

  for (int d=0; d<3; d++) 
    vel[d] = vel0[d];

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
        // PREC wmw = 1000 * volume * W; // TODO : Change init. water_mass for CoupledUP
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
            wm * (vel[0] + Dp_inv * (C[0] * xixp[0] + C[3] * xixp[1] +
                           C[6] * xixp[2]) ) );
        atomicAdd(
            &grid_block.val(_5, local_index[0], local_index[1], local_index[2]),
            wm * (vel[1] + Dp_inv * (C[1] * xixp[0] + C[4] * xixp[1] +
                           C[7] * xixp[2]) ) );
        atomicAdd(
            &grid_block.val(_6, local_index[0], local_index[1], local_index[2]),
            wm * (vel[2] + Dp_inv * (C[2] * xixp[0] + C[5] * xixp[1] +
                           C[8] * xixp[2]) ) );
        // Simple FBar: Vol, Vol JBar
        atomicAdd(
            &grid_block.val(_7, local_index[0], local_index[1], local_index[2]),
            wv);
        atomicAdd(
            &grid_block.val(_8, local_index[0], local_index[1], local_index[2]),
            wv * (1.0 - J));
#if DEBUG_COUPLED_UP
        // Simple FBar: Vol, Vol JBar
        atomicAdd(
            &grid_block.val(_9, local_index[0], local_index[1], local_index[2]),
            wmw);
        atomicAdd(
            &grid_block.val(_10, local_index[0], local_index[1], local_index[2]),
            wmw * 0.0); // TODO : Change initial pressure_water p2g write to not be 0
#endif
      }
}


template <typename ParticleBuffer, typename ParticleArray, num_attribs_e N, typename Grid, typename Partition>
__global__ void rasterize(uint32_t particleCount, const ParticleBuffer pbuffer,
                          const ParticleArray parray, const ParticleAttrib<N> pattrib,
                          Grid grid, const Partition partition, double dt, pvec3 vel0, PREC grav) {
  uint32_t parid = blockIdx.x * blockDim.x + threadIdx.x;
  if (parid >= particleCount) return;
  PREC length = pbuffer.length;
  PREC mass = pbuffer.mass;
  PREC volume = pbuffer.volume;
  pvec3 local_pos{(PREC)parray.val(_0, parid), (PREC)parray.val(_1, parid),
                 (PREC)parray.val(_2, parid)};
  pvec3 vel;
  pvec9 contrib, C;
  vel.set(0.), contrib.set(0.), C.set(0.);
  PREC J = 1.0;  // Volume ratio, Det def. gradient. 1 for t0 

  bool hydrostatic_init = false;
  if (hydrostatic_init) {
    PREC pressure = (mass/volume) * (grav) * (local_pos[1] - g_offset)*length; //< P = pgh
    PREC voln = J * volume;
    contrib[0] = - pressure * voln;
    contrib[4] = - pressure * voln;
    contrib[8] = - pressure * voln;
  }
  // // Leap-frog init vs symplectic Euler
  // /if (1) dt = dt / 2.0;

  for (int d=0; d<3; d++) 
    vel[d] = vel0[d];

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
        // PREC wmw = 1000 * volume * W; // TODO : Change init. water_mass for CoupledUP
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
            wm * (vel[0] + Dp_inv * (C[0] * xixp[0] + C[3] * xixp[1] +
                           C[6] * xixp[2]) ) );
        atomicAdd(
            &grid_block.val(_5, local_index[0], local_index[1], local_index[2]),
            wm * (vel[1] + Dp_inv * (C[1] * xixp[0] + C[4] * xixp[1] +
                           C[7] * xixp[2]) ) );
        atomicAdd(
            &grid_block.val(_6, local_index[0], local_index[1], local_index[2]),
            wm * (vel[2] + Dp_inv * (C[2] * xixp[0] + C[5] * xixp[1] +
                           C[8] * xixp[2]) ) );
        // Simple FBar: Vol, Vol JBar
        atomicAdd(
            &grid_block.val(_7, local_index[0], local_index[1], local_index[2]),
            wv);
        atomicAdd(
            &grid_block.val(_8, local_index[0], local_index[1], local_index[2]),
            wv * (1.0 - J));
#if DEBUG_COUPLED_UP
        // Simple FBar: Vol, Vol JBar
        atomicAdd(
            &grid_block.val(_9, local_index[0], local_index[1], local_index[2]),
            wmw);
        atomicAdd(
            &grid_block.val(_10, local_index[0], local_index[1], local_index[2]),
            wmw * 0.0); // TODO : Change initial pressure_water p2g write to not be 0
#endif
      }
}

// Specialize particle-to-grid rasterize for materials with initial input attributes
template <typename ParticleArray, num_attribs_e N, typename Grid, typename Partition>
__global__ void rasterize(uint32_t particleCount, 
                          const ParticleBuffer<material_e::JBarFluid> pbuffer,
                          const ParticleArray parray, const ParticleAttrib<N> pattrib,
                          Grid grid, const Partition partition, double dt, pvec3 vel0, PREC grav) {
  uint32_t parid = blockIdx.x * blockDim.x + threadIdx.x;
  if (parid >= particleCount)
    return;
  PREC length = pbuffer.length;
  PREC mass = pbuffer.mass;
  PREC volume = pbuffer.volume;
  pvec3 local_pos{(PREC)parray.val(_0, parid), (PREC)parray.val(_1, parid),
                 (PREC)parray.val(_2, parid)};
  pvec3 vel;
  pvec9 contrib, C;
  vel.set(0.), contrib.set(0.), C.set(0.);

  // TODO : Clean this up to use a loop and enums
  PREC val = 0;
  unsigned i = 0;
  getParticleAttrib(pattrib, i, parid, val); 
  PREC sJ = val;
  // PREC J = 1.0 - sJ;  // Volume ratio, Det def. gradient. 1 for t0 
  PREC JInc = 0.0; // sJInc, assume 0 i.e. no change in J this time-step

  i++;
  getParticleAttrib(pattrib, i, parid, val); 
  vel[0] = val;

  i++;
  getParticleAttrib(pattrib, i, parid, val); 
  vel[1] = val;

  i++;
  getParticleAttrib(pattrib, i, parid, val); 
  vel[2] = val;

  i++;
  getParticleAttrib(pattrib, i, parid, val); 
  PREC sJBar = val;

  i++;
  getParticleAttrib(pattrib, i, parid, val); 
  PREC ID = val;

  // TODO : Init. with material's Cauchy stress, not just pressure for fluid
  // PREC pressure =  (pbuffer.bulk / pbuffer.gamma) * (  pow((1.0-sJBar), -pbuffer.gamma) - 1.0 );
  PREC pressure = (pbuffer.bulk / pbuffer.gamma) * expm1(-pbuffer.gamma*log1p(-sJBar));
  PREC voln = (1.0 - sJ) * volume;
  contrib[0] = - pressure * voln;
  contrib[4] = - pressure * voln;
  contrib[8] = - pressure * voln;

  bool hydrostatic_init = false;
  if (hydrostatic_init) {
    PREC pressure = (mass/volume) * (grav) * (local_pos[1] - g_offset)*length; //< P = pgh
    PREC voln = (1.0 - sJ) * volume;
    contrib[0] = - pressure * voln;
    contrib[4] = - pressure * voln;
    contrib[8] = - pressure * voln;
  }
  // * Leap-frog init vs symplectic Euler
  // /if (1) dt = dt / 2.0;

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
            wm * (vel[0] + Dp_inv * (C[0] * xixp[0] + C[3] * xixp[1] +
                           C[6] * xixp[2]) )
                             );
        atomicAdd(
            &grid_block.val(_5, local_index[0], local_index[1], local_index[2]),
            wm * (vel[1] + Dp_inv * (C[1] * xixp[0] + C[4] * xixp[1] +
                           C[7] * xixp[2]) )
                              );
        atomicAdd(
            &grid_block.val(_6, local_index[0], local_index[1], local_index[2]),
            wm * (vel[2] + Dp_inv * (C[2] * xixp[0] + C[5] * xixp[1] +
                           C[8] * xixp[2]) )
                              );
        // Simple FBar: Vol, Vol JBar
        atomicAdd(
            &grid_block.val(_7, local_index[0], local_index[1], local_index[2]),
            wv);
        // Recheck if JInc is sJInc, difference between 1 and 0
        atomicAdd(
            &grid_block.val(_8, local_index[0], local_index[1], local_index[2]),
            wv * (sJBar + sJBar * JInc - JInc));//(sJBar + sJBar * JInc - JInc));

#if DEBUG_COUPLED_UP
        atomicAdd(
            &grid_block.val(_9, local_index[0], local_index[1], local_index[2]),
            wv * 1000);
        atomicAdd(
            &grid_block.val(_10, local_index[0], local_index[1], local_index[2]),
            wv * 1000 * 0);
#endif
      }
}


// Specialize particle-to-grid rasterize for materials with initial input attributes
template <typename ParticleArray, num_attribs_e N, typename Grid, typename Partition>
__global__ void rasterize(uint32_t particleCount, 
                          const ParticleBuffer<material_e::JFluid_FBAR> pbuffer,
                          const ParticleArray parray, const ParticleAttrib<N> pattrib,
                          Grid grid, const Partition partition, double dt, pvec3 vel0, PREC grav) {
  uint32_t parid = blockIdx.x * blockDim.x + threadIdx.x;
  if (parid >= particleCount)
    return;
  PREC length = pbuffer.length;
  PREC mass = pbuffer.mass;
  PREC volume = pbuffer.volume;
  pvec3 local_pos{(PREC)parray.val(_0, parid), (PREC)parray.val(_1, parid),
                 (PREC)parray.val(_2, parid)};
  pvec3 vel;
  pvec9 contrib, C;
  vel.set(0.), contrib.set(0.), C.set(0.);

  // TODO : Clean this up to use a loop and enums
  PREC val = 0;
  unsigned i = 0;
  getParticleAttrib(pattrib, i, parid, val); 
  PREC sJ = val;
  // PREC J = 1.0 - sJ;  // Volume ratio, Det def. gradient. 1 for t0 
  PREC JInc = 0.0; // sJInc, assume 0 i.e. no change in J this time-step

  i++;
  getParticleAttrib(pattrib, i, parid, val); 
  vel[0] = val;

  i++;
  getParticleAttrib(pattrib, i, parid, val); 
  vel[1] = val;

  i++;
  getParticleAttrib(pattrib, i, parid, val); 
  vel[2] = val;

  i++;
  getParticleAttrib(pattrib, i, parid, val); 
  PREC sJBar = val;

  i++;
  getParticleAttrib(pattrib, i, parid, val); 
  PREC ID = val;

  // TODO : Init. with material's Cauchy stress, not just pressure for fluid
  // PREC pressure =  (pbuffer.bulk / pbuffer.gamma) * (  pow((1.0-sJBar), -pbuffer.gamma) - 1.0 );
  PREC pressure = (pbuffer.bulk / pbuffer.gamma) * expm1(-pbuffer.gamma*log1p(-sJBar));
  PREC voln = (1.0 - sJ) * volume;
  contrib[0] = - pressure * voln;
  contrib[4] = - pressure * voln;
  contrib[8] = - pressure * voln;

  // * Leap-frog init vs symplectic Euler
  // /if (1) dt = dt / 2.0;

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
            wm * (vel[0] + Dp_inv * (C[0] * xixp[0] + C[3] * xixp[1] +
                           C[6] * xixp[2]) )
                             );
        atomicAdd(
            &grid_block.val(_5, local_index[0], local_index[1], local_index[2]),
            wm * (vel[1] + Dp_inv * (C[1] * xixp[0] + C[4] * xixp[1] +
                           C[7] * xixp[2]) )
                              );
        atomicAdd(
            &grid_block.val(_6, local_index[0], local_index[1], local_index[2]),
            wm * (vel[2] + Dp_inv * (C[2] * xixp[0] + C[5] * xixp[1] +
                           C[8] * xixp[2]) )
                              );
        // Simple FBar: Vol, Vol JBar
        atomicAdd(
            &grid_block.val(_7, local_index[0], local_index[1], local_index[2]),
            wv);
        // Recheck if JInc is sJInc, difference between 1 and 0
        atomicAdd(
            &grid_block.val(_8, local_index[0], local_index[1], local_index[2]),
            wv * (sJBar + sJBar * JInc - JInc));//(sJBar + sJBar * JInc - JInc));

#if DEBUG_COUPLED_UP
        atomicAdd(
            &grid_block.val(_9, local_index[0], local_index[1], local_index[2]),
            wv * 1000);
        atomicAdd(
            &grid_block.val(_10, local_index[0], local_index[1], local_index[2]),
            wv * 1000 * 0);
#endif
      }
}

// %% ============================================================= %%
//     MPM-FEM Precompute Element Quantities
// %% ============================================================= %%

template <typename VerticeArray, typename ElementArray>
__global__ void fem_precompute(uint32_t blockCount, const VerticeArray vertice_array,
                               const ElementArray element_array,
                               ElementBuffer<fem_e::Tetrahedron> elementBins) {
    auto element_number = blockIdx.x * blockDim.x + threadIdx.x;
    if (element_number >= g_max_fem_element_num) {return; }
    else if (element_number >= blockCount) {return; }
    else {
    auto element = elementBins.ch(_0, element_number );
    int IDs[4];
    pvec3 p[4];
    pvec9 D, Dinv;
    PREC restVolume;
    IDs[0] = element_array.val(_0, element_number );
    IDs[1] = element_array.val(_1, element_number );
    IDs[2] = element_array.val(_2, element_number );
    IDs[3] = element_array.val(_3, element_number );

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
    if (sV0 < 0.f) printf("Element %d inverted volume! \n", element_number);

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
}



template <typename VerticeArray, typename ElementArray>
__global__ void fem_precompute(uint32_t blockCount, const VerticeArray vertice_array,
                               const ElementArray element_array,
                               ElementBuffer<fem_e::Tetrahedron_FBar> elementBins) {
    auto element_number = blockIdx.x * blockDim.x + threadIdx.x;
    if (element_number >= g_max_fem_element_num) {return;}
    else if (element_number < blockCount)
    {

    auto element = elementBins.ch(_0, element_number);
    int IDs[4];
    pvec3 p[4];
    pvec9 D, Dinv;
    PREC restVolume;
    IDs[0] = element_array.val(_0, element_number);
    IDs[1] = element_array.val(_1, element_number);
    IDs[2] = element_array.val(_2, element_number);
    IDs[3] = element_array.val(_3, element_number);

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
    if (sV0 < 0.f) printf("Element %d inverted volume! \n", element_number);

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
      element.val(_14, 0) = 0.0;
      element.val(_15, 0) = 0.0;
    }
  }
}

template <typename VerticeArray, typename ElementArray>
__global__ void fem_precompute(uint32_t blockCount, const VerticeArray vertice_array,
                               const ElementArray element_array,
                               ElementBuffer<fem_e::Brick> elementBins) {
                                 return;
}

// %% ============================================================= %%
//     MPM Move Particles From Device Array to Particle Buffers
// %% ============================================================= %%

template <typename ParticleArray, typename ParticleAttribs, typename ParticleBuffer, typename Partition>
__global__ void array_to_buffer(ParticleArray parray, ParticleAttribs pattribs,
                                ParticleBuffer pbuffer,
                                Partition partition, vec<PREC, 3> vel) { }


template <typename ParticleArray, typename Partition>
__global__ void array_to_buffer(ParticleArray parray,
                                ParticleBuffer<material_e::JFluid> pbuffer,
                                Partition partition, vec<PREC, 3> vel) {
  uint32_t blockno = blockIdx.x;
  int pcnt = (g_buckets_on_particle_buffer) 
              ? pbuffer._ppbs[blockno] 
              : partition._ppbs[blockno];
  auto bucket = (g_buckets_on_particle_buffer) 
                ? pbuffer._blockbuckets + blockno * g_particle_num_per_block 
                : partition._blockbuckets + blockno * g_particle_num_per_block;
  for (int pidib = threadIdx.x; pidib < pcnt; pidib += blockDim.x) {
    auto parid = bucket[pidib];
    auto pbin = (g_buckets_on_particle_buffer) 
                ? pbuffer.ch(_0, pbuffer._binsts[blockno] + pidib / g_bin_capacity)
                : pbuffer.ch(_0, partition._binsts[blockno] + pidib / g_bin_capacity);
    /// Position of particle (x,y,z) in 3D dimensionless grid-space [0, 1]
    pbin.val(_0, pidib % g_bin_capacity) = parray.val(_0, parid);
    pbin.val(_1, pidib % g_bin_capacity) = parray.val(_1, parid);
    pbin.val(_2, pidib % g_bin_capacity) = parray.val(_2, parid);
    /// J
    pbin.val(_3, pidib % g_bin_capacity) = 1.f;
  }
}

template <typename ParticleArray, typename ParticleAttribs, typename Partition>
__global__ void array_to_buffer(ParticleArray parray, ParticleAttribs pattribs,
                                ParticleBuffer<material_e::JFluid> pbuffer,
                                Partition partition) {
  uint32_t blockno = blockIdx.x;
  int pcnt = (g_buckets_on_particle_buffer) 
              ? pbuffer._ppbs[blockno] 
              : partition._ppbs[blockno];
  auto bucket = (g_buckets_on_particle_buffer) 
                ? pbuffer._blockbuckets + blockno * g_particle_num_per_block 
                : partition._blockbuckets + blockno * g_particle_num_per_block;
  for (int pidib = threadIdx.x; pidib < pcnt; pidib += blockDim.x) {
    auto parid = bucket[pidib];
    auto pbin = (g_buckets_on_particle_buffer) 
                ? pbuffer.ch(_0, pbuffer._binsts[blockno] + pidib / g_bin_capacity)
                : pbuffer.ch(_0, partition._binsts[blockno] + pidib / g_bin_capacity);
    /// Position of particle (x,y,z) in 3D dimensionless grid-space [0, 1]
    pbin.val(_0, pidib % g_bin_capacity) = parray.val(_0, parid);
    pbin.val(_1, pidib % g_bin_capacity) = parray.val(_1, parid);
    pbin.val(_2, pidib % g_bin_capacity) = parray.val(_2, parid);
    /// J
    pbin.val(_3, pidib % g_bin_capacity) = pattribs.val(_0, parid);
  }
}

template <typename ParticleArray, typename Partition>
__global__ void array_to_buffer(ParticleArray parray,
                                ParticleBuffer<material_e::JFluid_ASFLIP> pbuffer,
                                Partition partition, vec<PREC, 3> vel) {
  uint32_t blockno = blockIdx.x;
  int pcnt = (g_buckets_on_particle_buffer) 
              ? pbuffer._ppbs[blockno] 
              : partition._ppbs[blockno];
  auto bucket = (g_buckets_on_particle_buffer) 
                ? pbuffer._blockbuckets + blockno * g_particle_num_per_block 
                : partition._blockbuckets + blockno * g_particle_num_per_block;
  for (int pidib = threadIdx.x; pidib < pcnt; pidib += blockDim.x) {
    auto parid = bucket[pidib];
    auto pbin = (g_buckets_on_particle_buffer) 
                ? pbuffer.ch(_0, pbuffer._binsts[blockno] + pidib / g_bin_capacity)
                : pbuffer.ch(_0, partition._binsts[blockno] + pidib / g_bin_capacity);
    /// Position of particle (x,y,z) in 3D dimensionless grid-space [0, 1]
    pbin.val(_0, pidib % g_bin_capacity) = parray.val(_0, parid);
    pbin.val(_1, pidib % g_bin_capacity) = parray.val(_1, parid);
    pbin.val(_2, pidib % g_bin_capacity) = parray.val(_2, parid);
    /// J
    pbin.val(_3, pidib % g_bin_capacity) = 0.0; //< 1 - J , 1 - V/Vo, 
    /// vel
    pbin.val(_4, pidib % g_bin_capacity) = vel[0]; //< Vel_x 
    pbin.val(_5, pidib % g_bin_capacity) = vel[1]; //< Vel_y 
    pbin.val(_6, pidib % g_bin_capacity) = vel[2]; //< Vel_z 

  }
}


template <typename ParticleArray, typename Partition>
__global__ void array_to_buffer(ParticleArray parray, ParticleAttrib<num_attribs_e::Six> pattribs,
                                ParticleBuffer<material_e::JFluid_ASFLIP> pbuffer,
                                Partition partition, vec<PREC, 3> vel) {
  uint32_t blockno = blockIdx.x;
  int pcnt = (g_buckets_on_particle_buffer) 
              ? pbuffer._ppbs[blockno] 
              : partition._ppbs[blockno];
  auto bucket = (g_buckets_on_particle_buffer) 
                ? pbuffer._blockbuckets + blockno * g_particle_num_per_block 
                : partition._blockbuckets + blockno * g_particle_num_per_block;
  for (int pidib = threadIdx.x; pidib < pcnt; pidib += blockDim.x) {
    auto parid = bucket[pidib];
    auto pbin = (g_buckets_on_particle_buffer) 
                ? pbuffer.ch(_0, pbuffer._binsts[blockno] + pidib / g_bin_capacity)
                : pbuffer.ch(_0, partition._binsts[blockno] + pidib / g_bin_capacity);
    /// Position of particle (x,y,z) in 3D dimensionless grid-space [0, 1]
    pbin.val(_0, pidib % g_bin_capacity) = parray.val(_0, parid);
    pbin.val(_1, pidib % g_bin_capacity) = parray.val(_1, parid);
    pbin.val(_2, pidib % g_bin_capacity) = parray.val(_2, parid);
    /// J
    pbin.val(_3, pidib % g_bin_capacity) = pattribs.val(_0, parid); //< 1 - J , 1 - V/Vo, 
    /// vel
    pbin.val(_4, pidib % g_bin_capacity) = pattribs.val(_1, parid); //< Vel_x m/s
    pbin.val(_5, pidib % g_bin_capacity) = pattribs.val(_2, parid); //< Vel_y m/s
    pbin.val(_6, pidib % g_bin_capacity) = pattribs.val(_3, parid); //< Vel_z m/s 
  }
}

template <typename ParticleArray, typename Partition>
__global__ void array_to_buffer(ParticleArray parray,
                                ParticleBuffer<material_e::JFluid_FBAR> pbuffer,
                                Partition partition, vec<PREC, 3> vel) {
  uint32_t blockno = blockIdx.x;
  int pcnt = (g_buckets_on_particle_buffer) 
              ? pbuffer._ppbs[blockno] 
              : partition._ppbs[blockno];
  auto bucket = (g_buckets_on_particle_buffer) 
                ? pbuffer._blockbuckets + blockno * g_particle_num_per_block 
                : partition._blockbuckets + blockno * g_particle_num_per_block;
  for (int pidib = threadIdx.x; pidib < pcnt; pidib += blockDim.x) {
    auto parid = bucket[pidib];
    auto pbin = (g_buckets_on_particle_buffer) 
                ? pbuffer.ch(_0, pbuffer._binsts[blockno] + pidib / g_bin_capacity)
                : pbuffer.ch(_0, partition._binsts[blockno] + pidib / g_bin_capacity);
    /// Position of particle (x,y,z) in 3D dimensionless grid-space [0, 1]
    pbin.val(_0, pidib % g_bin_capacity) = parray.val(_0, parid); // x
    pbin.val(_1, pidib % g_bin_capacity) = parray.val(_1, parid); // y
    pbin.val(_2, pidib % g_bin_capacity) = parray.val(_2, parid); // z
    pbin.val(_3, pidib % g_bin_capacity) = 0.0; //< (1 - J) = (1 - V/Vo)
    pbin.val(_4, pidib % g_bin_capacity) = 0.0; //< (1 -JBar) : Simple FBAR 
    pbin.val(_5, pidib % g_bin_capacity) = (PREC)parid; //< ID
  }
}



template <typename ParticleArray, typename Partition>
__global__ void array_to_buffer(ParticleArray parray, ParticleAttrib<num_attribs_e::Six> pattribs,
                                ParticleBuffer<material_e::JFluid_FBAR> pbuffer,
                                Partition partition, vec<PREC, 3> vel) {
  uint32_t blockno = blockIdx.x;
  int pcnt = (g_buckets_on_particle_buffer) 
              ? pbuffer._ppbs[blockno] 
              : partition._ppbs[blockno];
  auto bucket = (g_buckets_on_particle_buffer) 
                ? pbuffer._blockbuckets + blockno * g_particle_num_per_block 
                : partition._blockbuckets + blockno * g_particle_num_per_block;
  for (int pidib = threadIdx.x; pidib < pcnt; pidib += blockDim.x) {
    auto parid = bucket[pidib];
    auto pbin = (g_buckets_on_particle_buffer) 
                ? pbuffer.ch(_0, pbuffer._binsts[blockno] + pidib / g_bin_capacity)
                : pbuffer.ch(_0, partition._binsts[blockno] + pidib / g_bin_capacity);
    /// Position of particle (x,y,z) in 3D dimensionless grid-space [0, 1]
    pbin.val(_0, pidib % g_bin_capacity) = parray.val(_0, parid); // x
    pbin.val(_1, pidib % g_bin_capacity) = parray.val(_1, parid); // y
    pbin.val(_2, pidib % g_bin_capacity) = parray.val(_2, parid); // z
    // Uses same attribs layout as JBarFluid (sJ, Velocity_X, Velocity_Y, Velocity_Z, sJBar, ID)
    pbin.val(_3, pidib % g_bin_capacity) = pattribs.val(_0, parid); //< (1 - J) = (1 - V/Vo)
    pbin.val(_4, pidib % g_bin_capacity) = pattribs.val(_4, parid); //< (1 -JBar) : Simple FBAR 
    pbin.val(_5, pidib % g_bin_capacity) = pattribs.val(_5, parid); //< ID
  }
}

template <typename ParticleArray, typename Partition>
__global__ void array_to_buffer(ParticleArray parray,
                                ParticleBuffer<material_e::JBarFluid> pbuffer,
                                Partition partition, vec<PREC, 3> vel) {
  uint32_t blockno = blockIdx.x;
  int pcnt = (g_buckets_on_particle_buffer) 
              ? pbuffer._ppbs[blockno] 
              : partition._ppbs[blockno];
  auto bucket = (g_buckets_on_particle_buffer) 
                ? pbuffer._blockbuckets + blockno * g_particle_num_per_block 
                : partition._blockbuckets + blockno * g_particle_num_per_block;
  for (int pidib = threadIdx.x; pidib < pcnt; pidib += blockDim.x) {
    auto parid = bucket[pidib];
    auto pbin = (g_buckets_on_particle_buffer) 
                ? pbuffer.ch(_0, pbuffer._binsts[blockno] + pidib / g_bin_capacity)
                : pbuffer.ch(_0, partition._binsts[blockno] + pidib / g_bin_capacity);
    /// Position of particle (x,y,z) in 3D dimensionless grid-space [0, 1]
    pbin.val(_0, pidib % g_bin_capacity) = parray.val(_0, parid);
    pbin.val(_1, pidib % g_bin_capacity) = parray.val(_1, parid);
    pbin.val(_2, pidib % g_bin_capacity) = parray.val(_2, parid);
    /// 1 - J
    pbin.val(_3, pidib % g_bin_capacity) = 0.0; //< (1 - J) = (1 - V/Vo)
    /// vel (ASFLIP)
    pbin.val(_4, pidib % g_bin_capacity) = vel[0]; //< Vel_x m/s
    pbin.val(_5, pidib % g_bin_capacity) = vel[1]; //< Vel_y m/s
    pbin.val(_6, pidib % g_bin_capacity) = vel[2]; //< Vel_z m/s
    /// JBar (Simple FBar)
    pbin.val(_7, pidib % g_bin_capacity) = 0.0; //< 1 - JBar
    pbin.val(_8, pidib % g_bin_capacity) = (PREC)parid; //< ID
  }
}


template <typename ParticleArray, typename Partition>
__global__ void array_to_buffer(ParticleArray parray, ParticleAttrib<num_attribs_e::Six> pattribs,
                                ParticleBuffer<material_e::JBarFluid> pbuffer,
                                Partition partition, vec<PREC, 3> vel) {
  uint32_t blockno = blockIdx.x;
  int pcnt = (g_buckets_on_particle_buffer) 
              ? pbuffer._ppbs[blockno] 
              : partition._ppbs[blockno];
  auto bucket = (g_buckets_on_particle_buffer) 
                ? pbuffer._blockbuckets + blockno * g_particle_num_per_block 
                : partition._blockbuckets + blockno * g_particle_num_per_block;
  for (int pidib = threadIdx.x; pidib < pcnt; pidib += blockDim.x) {
    auto parid = bucket[pidib];
    auto pbin = (g_buckets_on_particle_buffer) 
                ? pbuffer.ch(_0, pbuffer._binsts[blockno] + pidib / g_bin_capacity)
                : pbuffer.ch(_0, partition._binsts[blockno] + pidib / g_bin_capacity);
    /// Position of particle (x,y,z) in 3D dimensionless grid-space [0, 1]
    pbin.val(_0, pidib % g_bin_capacity) = parray.val(_0, parid);
    pbin.val(_1, pidib % g_bin_capacity) = parray.val(_1, parid);
    pbin.val(_2, pidib % g_bin_capacity) = parray.val(_2, parid);
    /// 1 - J
    pbin.val(_3, pidib % g_bin_capacity) = pattribs.val(_0, parid); //< (1 - J) = (1 - V/Vo)
    /// vel (ASFLIP)
    pbin.val(_4, pidib % g_bin_capacity) = pattribs.val(_1, parid); //< Vel_x m/s
    pbin.val(_5, pidib % g_bin_capacity) = pattribs.val(_2, parid); //< Vel_y m/s
    pbin.val(_6, pidib % g_bin_capacity) = pattribs.val(_3, parid); //< Vel_z m/s
    /// 1 - JBar (Simple FBar)
    pbin.val(_7, pidib % g_bin_capacity) = pattribs.val(_4, parid); //< 1 - JBar

    pbin.val(_8, pidib % g_bin_capacity) = pattribs.val(_5, parid); //< ID
  }
}


template <typename ParticleArray, typename Partition>
__global__ void
array_to_buffer(ParticleArray parray,
                ParticleBuffer<material_e::VonMises> pbuffer,
                Partition partition, vec<PREC, 3> vel) {
  uint32_t blockno = blockIdx.x;
  int pcnt = (g_buckets_on_particle_buffer) 
              ? pbuffer._ppbs[blockno] 
              : partition._ppbs[blockno];
  auto bucket = (g_buckets_on_particle_buffer) 
                ? pbuffer._blockbuckets + blockno * g_particle_num_per_block 
                : partition._blockbuckets + blockno * g_particle_num_per_block;
  for (int pidib = threadIdx.x; pidib < pcnt; pidib += blockDim.x) {
    auto parid = bucket[pidib];
    auto pbin = (g_buckets_on_particle_buffer) 
                ? pbuffer.ch(_0, pbuffer._binsts[blockno] + pidib / g_bin_capacity)
                : pbuffer.ch(_0, partition._binsts[blockno] + pidib / g_bin_capacity);
    /// Position of particle (x,y,z) in 3D dimensionless grid-space [0, 1]
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
    pbin.val(_12, pidib % g_bin_capacity) = (PREC)parid; //< Particle ID
  }
}

template <typename ParticleArray, typename Partition>
__global__ void
array_to_buffer(ParticleArray parray,
                ParticleBuffer<material_e::FixedCorotated> pbuffer,
                Partition partition, vec<PREC, 3> vel) {
  uint32_t blockno = blockIdx.x;
  int pcnt = (g_buckets_on_particle_buffer) 
              ? pbuffer._ppbs[blockno] 
              : partition._ppbs[blockno];
  auto bucket = (g_buckets_on_particle_buffer) 
                ? pbuffer._blockbuckets + blockno * g_particle_num_per_block 
                : partition._blockbuckets + blockno * g_particle_num_per_block;
  for (int pidib = threadIdx.x; pidib < pcnt; pidib += blockDim.x) {
    auto parid = bucket[pidib];
    auto pbin = (g_buckets_on_particle_buffer) 
                ? pbuffer.ch(_0, pbuffer._binsts[blockno] + pidib / g_bin_capacity)
                : pbuffer.ch(_0, partition._binsts[blockno] + pidib / g_bin_capacity);
    /// Position of particle (x,y,z) in 3D dimensionless grid-space [0, 1]
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
    pbin.val(_12, pidib % g_bin_capacity) = (PREC)parid; //< Particle ID
  }
}

template <typename ParticleArray, typename Partition>
__global__ void
array_to_buffer(ParticleArray parray,
                ParticleBuffer<material_e::FixedCorotated_ASFLIP> pbuffer,
                Partition partition, vec<PREC, 3> vel) {
  uint32_t blockno = blockIdx.x;
  int pcnt = (g_buckets_on_particle_buffer) 
              ? pbuffer._ppbs[blockno] 
              : partition._ppbs[blockno];
  auto bucket = (g_buckets_on_particle_buffer) 
                ? pbuffer._blockbuckets + blockno * g_particle_num_per_block 
                : partition._blockbuckets + blockno * g_particle_num_per_block;
  for (int pidib = threadIdx.x; pidib < pcnt; pidib += blockDim.x) {
    auto parid = bucket[pidib];
    auto pbin = (g_buckets_on_particle_buffer) 
                ? pbuffer.ch(_0, pbuffer._binsts[blockno] + pidib / g_bin_capacity)
                : pbuffer.ch(_0, partition._binsts[blockno] + pidib / g_bin_capacity);
    /// Position of particle (x,y,z) in 3D dimensionless grid-space [0, 1]
    pbin.val(_0, pidib % g_bin_capacity) = parray.val(_0, parid);
    pbin.val(_1, pidib % g_bin_capacity) = parray.val(_1, parid);
    pbin.val(_2, pidib % g_bin_capacity) = parray.val(_2, parid);
    /// F
    pbin.val(_3, pidib % g_bin_capacity) = 1.0;
    pbin.val(_4, pidib % g_bin_capacity) = 0.0;
    pbin.val(_5, pidib % g_bin_capacity) = 0.0;
    pbin.val(_6, pidib % g_bin_capacity) = 0.0;
    pbin.val(_7, pidib % g_bin_capacity) = 1.0;
    pbin.val(_8, pidib % g_bin_capacity) = 0.0;
    pbin.val(_9, pidib % g_bin_capacity) = 0.0;
    pbin.val(_10, pidib % g_bin_capacity) = 0.0;
    pbin.val(_11, pidib % g_bin_capacity) = 1.0;
    /// vel
    pbin.val(_12, pidib % g_bin_capacity) = vel[0]; //< Vel_x m/s
    pbin.val(_13, pidib % g_bin_capacity) = vel[1]; //< Vel_y m/s
    pbin.val(_14, pidib % g_bin_capacity) = vel[2]; //< Vel_z m/s
    /// Particle ID
    pbin.val(_15, pidib % g_bin_capacity) = (PREC)parid; //< Particle ID
  }
}

template <typename ParticleArray, typename Partition>
__global__ void
array_to_buffer(ParticleArray parray,
                ParticleBuffer<material_e::FixedCorotated_FBAR> pbuffer,
                Partition partition, vec<PREC, 3> vel) {
  uint32_t blockno = blockIdx.x;
  int pcnt = (g_buckets_on_particle_buffer) 
              ? pbuffer._ppbs[blockno] 
              : partition._ppbs[blockno];
  auto bucket = (g_buckets_on_particle_buffer) 
                ? pbuffer._blockbuckets + blockno * g_particle_num_per_block 
                : partition._blockbuckets + blockno * g_particle_num_per_block;
  for (int pidib = threadIdx.x; pidib < pcnt; pidib += blockDim.x) {
    auto parid = bucket[pidib];
    auto pbin = (g_buckets_on_particle_buffer) 
                ? pbuffer.ch(_0, pbuffer._binsts[blockno] + pidib / g_bin_capacity)
                : pbuffer.ch(_0, partition._binsts[blockno] + pidib / g_bin_capacity);
    /// Position of particle (x,y,z) in 3D dimensionless grid-space [0, 1]
    pbin.val(_0, pidib % g_bin_capacity) = parray.val(_0, parid);
    pbin.val(_1, pidib % g_bin_capacity) = parray.val(_1, parid);
    pbin.val(_2, pidib % g_bin_capacity) = parray.val(_2, parid);
    /// Deformation Gradient F
    pbin.val(_3, pidib % g_bin_capacity) = 1.0;
    pbin.val(_4, pidib % g_bin_capacity) = 0.0;
    pbin.val(_5, pidib % g_bin_capacity) = 0.0;
    pbin.val(_6, pidib % g_bin_capacity) = 0.0;
    pbin.val(_7, pidib % g_bin_capacity) = 1.0;
    pbin.val(_8, pidib % g_bin_capacity) = 0.0;
    pbin.val(_9, pidib % g_bin_capacity) = 0.0;
    pbin.val(_10, pidib % g_bin_capacity) = 0.0;
    pbin.val(_11, pidib % g_bin_capacity) = 1.0;
    /// F-Bar Anti-Locking
    pbin.val(_12, pidib % g_bin_capacity) = 0.0; //< 1 - JBar
    /// Particle ID
    pbin.val(_13, pidib % g_bin_capacity) = (PREC)parid; //< Particle ID
  }
}

template <typename ParticleArray, typename Partition>
__global__ void
array_to_buffer(ParticleArray parray,
                ParticleBuffer<material_e::FixedCorotated_ASFLIP_FBAR> pbuffer,
                Partition partition, vec<PREC, 3> vel) {
  uint32_t blockno = blockIdx.x;
  int pcnt = (g_buckets_on_particle_buffer) 
              ? pbuffer._ppbs[blockno] 
              : partition._ppbs[blockno];
  auto bucket = (g_buckets_on_particle_buffer) 
                ? pbuffer._blockbuckets + blockno * g_particle_num_per_block 
                : partition._blockbuckets + blockno * g_particle_num_per_block;
  for (int pidib = threadIdx.x; pidib < pcnt; pidib += blockDim.x) {
    auto parid = bucket[pidib];
    auto pbin = (g_buckets_on_particle_buffer) 
                ? pbuffer.ch(_0, pbuffer._binsts[blockno] + pidib / g_bin_capacity)
                : pbuffer.ch(_0, partition._binsts[blockno] + pidib / g_bin_capacity);
    /// Position of particle (x,y,z) in 3D dimensionless grid-space [0, 1]
    pbin.val(_0, pidib % g_bin_capacity) = parray.val(_0, parid);
    pbin.val(_1, pidib % g_bin_capacity) = parray.val(_1, parid);
    pbin.val(_2, pidib % g_bin_capacity) = parray.val(_2, parid);
    /// Deformation Gradient F
    pbin.val(_3, pidib % g_bin_capacity) = 1.0;
    pbin.val(_4, pidib % g_bin_capacity) = 0.0;
    pbin.val(_5, pidib % g_bin_capacity) = 0.0;
    pbin.val(_6, pidib % g_bin_capacity) = 0.0;
    pbin.val(_7, pidib % g_bin_capacity) = 1.0;
    pbin.val(_8, pidib % g_bin_capacity) = 0.0;
    pbin.val(_9, pidib % g_bin_capacity) = 0.0;
    pbin.val(_10, pidib % g_bin_capacity) = 0.0;
    pbin.val(_11, pidib % g_bin_capacity) = 1.0;
    /// Velocity for FLIP/ASFLIP
    pbin.val(_12, pidib % g_bin_capacity) = vel[0]; //< Vel_x
    pbin.val(_13, pidib % g_bin_capacity) = vel[1]; //< Vel_y
    pbin.val(_14, pidib % g_bin_capacity) = vel[2]; //< Vel_z
    /// F-Bar Anti-Locking
    pbin.val(_15, pidib % g_bin_capacity) = 0.0; //< 1 - JBar
    /// Particle ID
    pbin.val(_16, pidib % g_bin_capacity) = (PREC)parid; //< Particle ID
  }
}

template <typename ParticleArray, typename Partition>
__global__ void
array_to_buffer(ParticleArray parray,
                ParticleBuffer<material_e::NeoHookean> pbuffer,
                Partition partition, vec<PREC, 3> vel) {
  uint32_t blockno = blockIdx.x;
  int pcnt = (g_buckets_on_particle_buffer) 
              ? pbuffer._ppbs[blockno] 
              : partition._ppbs[blockno];
  auto bucket = (g_buckets_on_particle_buffer) 
                ? pbuffer._blockbuckets + blockno * g_particle_num_per_block 
                : partition._blockbuckets + blockno * g_particle_num_per_block;
  for (int pidib = threadIdx.x; pidib < pcnt; pidib += blockDim.x) {
    auto parid = bucket[pidib];
    auto pbin = (g_buckets_on_particle_buffer) 
                ? pbuffer.ch(_0, pbuffer._binsts[blockno] + pidib / g_bin_capacity)
                : pbuffer.ch(_0, partition._binsts[blockno] + pidib / g_bin_capacity);
    /// Position of particle (x,y,z) in 3D dimensionless grid-space [0, 1]
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
    pbin.val(_12, pidib % g_bin_capacity) = (PREC)parid; //< Particle ID
  }
}

template <typename ParticleArray, typename Partition>
__global__ void
array_to_buffer(ParticleArray parray,
                ParticleBuffer<material_e::NeoHookean_ASFLIP> pbuffer,
                Partition partition, vec<PREC, 3> vel) {
  uint32_t blockno = blockIdx.x;
  int pcnt = (g_buckets_on_particle_buffer) 
              ? pbuffer._ppbs[blockno] 
              : partition._ppbs[blockno];
  auto bucket = (g_buckets_on_particle_buffer) 
                ? pbuffer._blockbuckets + blockno * g_particle_num_per_block 
                : partition._blockbuckets + blockno * g_particle_num_per_block;
  for (int pidib = threadIdx.x; pidib < pcnt; pidib += blockDim.x) {
    auto parid = bucket[pidib];
    auto pbin = (g_buckets_on_particle_buffer) 
                ? pbuffer.ch(_0, pbuffer._binsts[blockno] + pidib / g_bin_capacity)
                : pbuffer.ch(_0, partition._binsts[blockno] + pidib / g_bin_capacity);
    /// Position of particle (x,y,z) in 3D dimensionless grid-space [0, 1]
    pbin.val(_0, pidib % g_bin_capacity) = parray.val(_0, parid);
    pbin.val(_1, pidib % g_bin_capacity) = parray.val(_1, parid);
    pbin.val(_2, pidib % g_bin_capacity) = parray.val(_2, parid);
    /// F
    pbin.val(_3, pidib % g_bin_capacity) = 1.0;
    pbin.val(_4, pidib % g_bin_capacity) = 0.0;
    pbin.val(_5, pidib % g_bin_capacity) = 0.0;
    pbin.val(_6, pidib % g_bin_capacity) = 0.0;
    pbin.val(_7, pidib % g_bin_capacity) = 1.0;
    pbin.val(_8, pidib % g_bin_capacity) = 0.0;
    pbin.val(_9, pidib % g_bin_capacity) = 0.0;
    pbin.val(_10, pidib % g_bin_capacity) = 0.0;
    pbin.val(_11, pidib % g_bin_capacity) = 1.0;
    /// vel
    pbin.val(_12, pidib % g_bin_capacity) = vel[0]; //< Vel_x m/s
    pbin.val(_13, pidib % g_bin_capacity) = vel[1]; //< Vel_y m/s
    pbin.val(_14, pidib % g_bin_capacity) = vel[2]; //< Vel_z m/s
    /// Particle ID
    pbin.val(_15, pidib % g_bin_capacity) = (PREC)parid; //< Particle ID
  }
}

template <typename ParticleArray, typename Partition>
__global__ void
array_to_buffer(ParticleArray parray,
                ParticleBuffer<material_e::NeoHookean_FBAR> pbuffer,
                Partition partition, vec<PREC, 3> vel) {
  uint32_t blockno = blockIdx.x;
  int pcnt = (g_buckets_on_particle_buffer) 
              ? pbuffer._ppbs[blockno] 
              : partition._ppbs[blockno];
  auto bucket = (g_buckets_on_particle_buffer) 
                ? pbuffer._blockbuckets + blockno * g_particle_num_per_block 
                : partition._blockbuckets + blockno * g_particle_num_per_block;
  for (int pidib = threadIdx.x; pidib < pcnt; pidib += blockDim.x) {
    auto parid = bucket[pidib];
    auto pbin = (g_buckets_on_particle_buffer) 
                ? pbuffer.ch(_0, pbuffer._binsts[blockno] + pidib / g_bin_capacity)
                : pbuffer.ch(_0, partition._binsts[blockno] + pidib / g_bin_capacity);
    /// Position of particle (x,y,z) in 3D dimensionless grid-space [0, 1]
    pbin.val(_0, pidib % g_bin_capacity) = parray.val(_0, parid);
    pbin.val(_1, pidib % g_bin_capacity) = parray.val(_1, parid);
    pbin.val(_2, pidib % g_bin_capacity) = parray.val(_2, parid);
    /// Deformation Gradient F
    pbin.val(_3, pidib % g_bin_capacity) = 1.0;
    pbin.val(_4, pidib % g_bin_capacity) = 0.0;
    pbin.val(_5, pidib % g_bin_capacity) = 0.0;
    pbin.val(_6, pidib % g_bin_capacity) = 0.0;
    pbin.val(_7, pidib % g_bin_capacity) = 1.0;
    pbin.val(_8, pidib % g_bin_capacity) = 0.0;
    pbin.val(_9, pidib % g_bin_capacity) = 0.0;
    pbin.val(_10, pidib % g_bin_capacity) = 0.0;
    pbin.val(_11, pidib % g_bin_capacity) = 1.0;
    /// F-Bar Anti-Locking
    pbin.val(_12, pidib % g_bin_capacity) = 0.0; //< 1 - JBar
    /// Particle ID
    pbin.val(_13, pidib % g_bin_capacity) = (PREC)parid; //< Particle ID
  }
}

template <typename ParticleArray, typename Partition>
__global__ void
array_to_buffer(ParticleArray parray,
                ParticleBuffer<material_e::NeoHookean_ASFLIP_FBAR> pbuffer,
                Partition partition, vec<PREC, 3> vel) {
  uint32_t blockno = blockIdx.x;
  int pcnt = (g_buckets_on_particle_buffer) 
              ? pbuffer._ppbs[blockno] 
              : partition._ppbs[blockno];
  auto bucket = (g_buckets_on_particle_buffer) 
                ? pbuffer._blockbuckets + blockno * g_particle_num_per_block 
                : partition._blockbuckets + blockno * g_particle_num_per_block;
  for (int pidib = threadIdx.x; pidib < pcnt; pidib += blockDim.x) {
    auto parid = bucket[pidib];
    auto pbin = (g_buckets_on_particle_buffer) 
                ? pbuffer.ch(_0, pbuffer._binsts[blockno] + pidib / g_bin_capacity)
                : pbuffer.ch(_0, partition._binsts[blockno] + pidib / g_bin_capacity);
    /// Position of particle (x,y,z) in 3D dimensionless grid-space [0, 1]
    pbin.val(_0, pidib % g_bin_capacity) = parray.val(_0, parid);
    pbin.val(_1, pidib % g_bin_capacity) = parray.val(_1, parid);
    pbin.val(_2, pidib % g_bin_capacity) = parray.val(_2, parid);
    /// Deformation Gradient F
    pbin.val(_3, pidib % g_bin_capacity) = 1.0;
    pbin.val(_4, pidib % g_bin_capacity) = 0.0;
    pbin.val(_5, pidib % g_bin_capacity) = 0.0;
    pbin.val(_6, pidib % g_bin_capacity) = 0.0;
    pbin.val(_7, pidib % g_bin_capacity) = 1.0;
    pbin.val(_8, pidib % g_bin_capacity) = 0.0;
    pbin.val(_9, pidib % g_bin_capacity) = 0.0;
    pbin.val(_10, pidib % g_bin_capacity) = 0.0;
    pbin.val(_11, pidib % g_bin_capacity) = 1.0;
    /// Velocity for FLIP/ASFLIP
    pbin.val(_12, pidib % g_bin_capacity) = vel[0]; //< Vel_x m/s
    pbin.val(_13, pidib % g_bin_capacity) = vel[1]; //< Vel_y m/s
    pbin.val(_14, pidib % g_bin_capacity) = vel[2]; //< Vel_z m/s
    /// F-Bar Anti-Locking
    pbin.val(_15, pidib % g_bin_capacity) = 0.0; //< 1 - JBar
    pbin.val(_16, pidib % g_bin_capacity) = (PREC)parid; //< Particle ID
  }
}


template <typename ParticleArray, typename Partition>
__global__ void array_to_buffer(ParticleArray parray,
                                ParticleBuffer<material_e::Sand> pbuffer,
                                Partition partition, vec<PREC, 3> vel) {
  uint32_t blockno = blockIdx.x;
  int pcnt = (g_buckets_on_particle_buffer) 
              ? pbuffer._ppbs[blockno] 
              : partition._ppbs[blockno];
  auto bucket = (g_buckets_on_particle_buffer) 
                ? pbuffer._blockbuckets + blockno * g_particle_num_per_block 
                : partition._blockbuckets + blockno * g_particle_num_per_block;
  for (int pidib = threadIdx.x; pidib < pcnt; pidib += blockDim.x) {
    auto parid = bucket[pidib];
    auto pbin = (g_buckets_on_particle_buffer) 
                ? pbuffer.ch(_0, pbuffer._binsts[blockno] + pidib / g_bin_capacity)
                : pbuffer.ch(_0, partition._binsts[blockno] + pidib / g_bin_capacity);
    /// Position of particle (x,y,z) in 3D dimensionless grid-space [0, 1]
    pbin.val(_0, pidib % g_bin_capacity) = parray.val(_0, parid);
    pbin.val(_1, pidib % g_bin_capacity) = parray.val(_1, parid);
    pbin.val(_2, pidib % g_bin_capacity) = parray.val(_2, parid);
    /// F
    pbin.val(_3, pidib % g_bin_capacity) = 1.0;
    pbin.val(_4, pidib % g_bin_capacity) = 0.0;
    pbin.val(_5, pidib % g_bin_capacity) = 0.0;
    pbin.val(_6, pidib % g_bin_capacity) = 0.0;
    pbin.val(_7, pidib % g_bin_capacity) = 1.0;
    pbin.val(_8, pidib % g_bin_capacity) = 0.0;
    pbin.val(_9, pidib % g_bin_capacity) = 0.0;
    pbin.val(_10, pidib % g_bin_capacity) = 0.0;
    pbin.val(_11, pidib % g_bin_capacity) = 1.0;
    pbin.val(_12, pidib % g_bin_capacity) = pbuffer.logJp0;
    pbin.val(_13, pidib % g_bin_capacity) = (PREC)parid; //< Particle ID
  }
}

template <typename ParticleArray, typename Partition>
__global__ void array_to_buffer(ParticleArray parray,
                                ParticleBuffer<material_e::Sand_ASFLIP> pbuffer,
                                Partition partition, vec<PREC, 3> vel) {
  uint32_t blockno = blockIdx.x;
  int pcnt = (g_buckets_on_particle_buffer) 
              ? pbuffer._ppbs[blockno] 
              : partition._ppbs[blockno];
  auto bucket = (g_buckets_on_particle_buffer) 
                ? pbuffer._blockbuckets + blockno * g_particle_num_per_block 
                : partition._blockbuckets + blockno * g_particle_num_per_block;
  for (int pidib = threadIdx.x; pidib < pcnt; pidib += blockDim.x) {
    auto parid = bucket[pidib];
    auto pbin = (g_buckets_on_particle_buffer) 
                ? pbuffer.ch(_0, pbuffer._binsts[blockno] + pidib / g_bin_capacity)
                : pbuffer.ch(_0, partition._binsts[blockno] + pidib / g_bin_capacity);
    /// Position of particle (x,y,z) in 3D dimensionless grid-space [0, 1]
    pbin.val(_0, pidib % g_bin_capacity) = parray.val(_0, parid);
    pbin.val(_1, pidib % g_bin_capacity) = parray.val(_1, parid);
    pbin.val(_2, pidib % g_bin_capacity) = parray.val(_2, parid);
    /// F
    pbin.val(_3, pidib % g_bin_capacity) = 1.0;
    pbin.val(_4, pidib % g_bin_capacity) = 0.0;
    pbin.val(_5, pidib % g_bin_capacity) = 0.0;
    pbin.val(_6, pidib % g_bin_capacity) = 0.0;
    pbin.val(_7, pidib % g_bin_capacity) = 1.0;
    pbin.val(_8, pidib % g_bin_capacity) = 0.0;
    pbin.val(_9, pidib % g_bin_capacity) = 0.0;
    pbin.val(_10, pidib % g_bin_capacity) = 0.0;
    pbin.val(_11, pidib % g_bin_capacity) = 1.0;
    pbin.val(_12, pidib % g_bin_capacity) = pbuffer.logJp0;
    pbin.val(_13, pidib % g_bin_capacity) = vel[0];
    pbin.val(_14, pidib % g_bin_capacity) = vel[1];
    pbin.val(_15, pidib % g_bin_capacity) = vel[2];
    pbin.val(_16, pidib % g_bin_capacity) = (PREC)parid; //< Particle ID
  }
}

template <typename ParticleArray, typename Partition>
__global__ void array_to_buffer(ParticleArray parray,
                                ParticleBuffer<material_e::Sand_FBAR> pbuffer,
                                Partition partition, vec<PREC, 3> vel) {
  uint32_t blockno = blockIdx.x;
  int pcnt = (g_buckets_on_particle_buffer) 
              ? pbuffer._ppbs[blockno] 
              : partition._ppbs[blockno];
  auto bucket = (g_buckets_on_particle_buffer) 
                ? pbuffer._blockbuckets + blockno * g_particle_num_per_block 
                : partition._blockbuckets + blockno * g_particle_num_per_block;
  for (int pidib = threadIdx.x; pidib < pcnt; pidib += blockDim.x) {
    auto parid = bucket[pidib];
    auto pbin = (g_buckets_on_particle_buffer) 
                ? pbuffer.ch(_0, pbuffer._binsts[blockno] + pidib / g_bin_capacity)
                : pbuffer.ch(_0, partition._binsts[blockno] + pidib / g_bin_capacity);
    /// Position of particle (x,y,z) in 3D dimensionless grid-space [0, 1]
    pbin.val(_0, pidib % g_bin_capacity) = parray.val(_0, parid);
    pbin.val(_1, pidib % g_bin_capacity) = parray.val(_1, parid);
    pbin.val(_2, pidib % g_bin_capacity) = parray.val(_2, parid);
    /// F
    pbin.val(_3, pidib % g_bin_capacity) = 1.0;
    pbin.val(_4, pidib % g_bin_capacity) = 0.0;
    pbin.val(_5, pidib % g_bin_capacity) = 0.0;
    pbin.val(_6, pidib % g_bin_capacity) = 0.0;
    pbin.val(_7, pidib % g_bin_capacity) = 1.0;
    pbin.val(_8, pidib % g_bin_capacity) = 0.0;
    pbin.val(_9, pidib % g_bin_capacity) = 0.0;
    pbin.val(_10, pidib % g_bin_capacity) = 0.0;
    pbin.val(_11, pidib % g_bin_capacity) = 1.0;
    pbin.val(_12, pidib % g_bin_capacity) = pbuffer.logJp0;
    pbin.val(_13, pidib % g_bin_capacity) = 0.0; //< sJBar
    pbin.val(_14, pidib % g_bin_capacity) = (PREC)parid; //< Particle ID
  }
}

template <typename ParticleArray, typename Partition>
__global__ void array_to_buffer(ParticleArray parray,
                                ParticleBuffer<material_e::Sand_ASFLIP_FBAR> pbuffer,
                                Partition partition, vec<PREC, 3> vel) {
  uint32_t blockno = blockIdx.x;
  int pcnt = (g_buckets_on_particle_buffer) 
              ? pbuffer._ppbs[blockno] 
              : partition._ppbs[blockno];
  auto bucket = (g_buckets_on_particle_buffer) 
                ? pbuffer._blockbuckets + blockno * g_particle_num_per_block 
                : partition._blockbuckets + blockno * g_particle_num_per_block;
  for (int pidib = threadIdx.x; pidib < pcnt; pidib += blockDim.x) {
    auto parid = bucket[pidib];
    auto pbin = (g_buckets_on_particle_buffer) 
                ? pbuffer.ch(_0, pbuffer._binsts[blockno] + pidib / g_bin_capacity)
                : pbuffer.ch(_0, partition._binsts[blockno] + pidib / g_bin_capacity);
    /// Position of particle (x,y,z) in 3D dimensionless grid-space [0, 1]
    pbin.val(_0, pidib % g_bin_capacity) = parray.val(_0, parid);
    pbin.val(_1, pidib % g_bin_capacity) = parray.val(_1, parid);
    pbin.val(_2, pidib % g_bin_capacity) = parray.val(_2, parid);
    /// F
    pbin.val(_3, pidib % g_bin_capacity) = 1.0;
    pbin.val(_4, pidib % g_bin_capacity) = 0.0;
    pbin.val(_5, pidib % g_bin_capacity) = 0.0;
    pbin.val(_6, pidib % g_bin_capacity) = 0.0;
    pbin.val(_7, pidib % g_bin_capacity) = 1.0;
    pbin.val(_8, pidib % g_bin_capacity) = 0.0;
    pbin.val(_9, pidib % g_bin_capacity) = 0.0;
    pbin.val(_10, pidib % g_bin_capacity) = 0.0;
    pbin.val(_11, pidib % g_bin_capacity) = 1.0;
    pbin.val(_12, pidib % g_bin_capacity) = pbuffer.logJp0;
    pbin.val(_13, pidib % g_bin_capacity) = vel[0];
    pbin.val(_14, pidib % g_bin_capacity) = vel[1];
    pbin.val(_15, pidib % g_bin_capacity) = vel[2];
    pbin.val(_16, pidib % g_bin_capacity) = 0.0; //< sJBar
    pbin.val(_17, pidib % g_bin_capacity) = (PREC)parid; //< Particle ID
  }
}

template <typename ParticleArray, typename Partition>
__global__ void array_to_buffer(ParticleArray parray,
                                ParticleBuffer<material_e::NACC> pbuffer,
                                Partition partition, vec<PREC, 3> vel) {
  uint32_t blockno = blockIdx.x;
  int pcnt = (g_buckets_on_particle_buffer) 
              ? pbuffer._ppbs[blockno] 
              : partition._ppbs[blockno];
  auto bucket = (g_buckets_on_particle_buffer) 
                ? pbuffer._blockbuckets + blockno * g_particle_num_per_block 
                : partition._blockbuckets + blockno * g_particle_num_per_block;
  for (int pidib = threadIdx.x; pidib < pcnt; pidib += blockDim.x) {
    auto parid = bucket[pidib];
    auto pbin = (g_buckets_on_particle_buffer) 
                ? pbuffer.ch(_0, pbuffer._binsts[blockno] + pidib / g_bin_capacity)
                : pbuffer.ch(_0, partition._binsts[blockno] + pidib / g_bin_capacity);
    /// Position of particle (x,y,z) in 3D dimensionless grid-space [0, 1]
    pbin.val(_0, pidib % g_bin_capacity) = parray.val(_0, parid);
    pbin.val(_1, pidib % g_bin_capacity) = parray.val(_1, parid);
    pbin.val(_2, pidib % g_bin_capacity) = parray.val(_2, parid);
    /// F
    pbin.val(_3, pidib % g_bin_capacity) = 1.0;
    pbin.val(_4, pidib % g_bin_capacity) = 0.0;
    pbin.val(_5, pidib % g_bin_capacity) = 0.0;
    pbin.val(_6, pidib % g_bin_capacity) = 0.0;
    pbin.val(_7, pidib % g_bin_capacity) = 1.0;
    pbin.val(_8, pidib % g_bin_capacity) = 0.0;
    pbin.val(_9, pidib % g_bin_capacity) = 0.0;
    pbin.val(_10, pidib % g_bin_capacity) = 0.0;
    pbin.val(_11, pidib % g_bin_capacity) = 1.0;
    pbin.val(_12, pidib % g_bin_capacity) = pbuffer.logJp0;
    pbin.val(_13, pidib % g_bin_capacity) = (PREC)parid; //< Particle ID
  }
}

template <typename ParticleArray, typename Partition>
__global__ void array_to_buffer(ParticleArray parray,
                                ParticleBuffer<material_e::NACC_ASFLIP> pbuffer,
                                Partition partition, vec<PREC, 3> vel) {
  uint32_t blockno = blockIdx.x;
  int pcnt = (g_buckets_on_particle_buffer) 
              ? pbuffer._ppbs[blockno] 
              : partition._ppbs[blockno];
  auto bucket = (g_buckets_on_particle_buffer) 
                ? pbuffer._blockbuckets + blockno * g_particle_num_per_block 
                : partition._blockbuckets + blockno * g_particle_num_per_block;
  for (int pidib = threadIdx.x; pidib < pcnt; pidib += blockDim.x) {
    auto parid = bucket[pidib];
    auto pbin = (g_buckets_on_particle_buffer) 
                ? pbuffer.ch(_0, pbuffer._binsts[blockno] + pidib / g_bin_capacity)
                : pbuffer.ch(_0, partition._binsts[blockno] + pidib / g_bin_capacity);
    /// Position of particle (x,y,z) in 3D dimensionless grid-space [0, 1]
    pbin.val(_0, pidib % g_bin_capacity) = parray.val(_0, parid);
    pbin.val(_1, pidib % g_bin_capacity) = parray.val(_1, parid);
    pbin.val(_2, pidib % g_bin_capacity) = parray.val(_2, parid);
    /// F
    pbin.val(_3, pidib % g_bin_capacity) = 1.0;
    pbin.val(_4, pidib % g_bin_capacity) = 0.0;
    pbin.val(_5, pidib % g_bin_capacity) = 0.0;
    pbin.val(_6, pidib % g_bin_capacity) = 0.0;
    pbin.val(_7, pidib % g_bin_capacity) = 1.0;
    pbin.val(_8, pidib % g_bin_capacity) = 0.0;
    pbin.val(_9, pidib % g_bin_capacity) = 0.0;
    pbin.val(_10, pidib % g_bin_capacity) = 0.0;
    pbin.val(_11, pidib % g_bin_capacity) = 1.0;
    pbin.val(_12, pidib % g_bin_capacity) = pbuffer.logJp0;
    pbin.val(_13, pidib % g_bin_capacity) = vel[0];
    pbin.val(_14, pidib % g_bin_capacity) = vel[1];
    pbin.val(_15, pidib % g_bin_capacity) = vel[2];
    pbin.val(_16, pidib % g_bin_capacity) = (PREC)parid; //< Particle ID
  }
}


template <typename ParticleArray, typename Partition>
__global__ void array_to_buffer(ParticleArray parray,
                                ParticleBuffer<material_e::NACC_FBAR> pbuffer,
                                Partition partition, vec<PREC, 3> vel) {
  uint32_t blockno = blockIdx.x;
  int pcnt = (g_buckets_on_particle_buffer) 
              ? pbuffer._ppbs[blockno] 
              : partition._ppbs[blockno];
  auto bucket = (g_buckets_on_particle_buffer) 
                ? pbuffer._blockbuckets + blockno * g_particle_num_per_block 
                : partition._blockbuckets + blockno * g_particle_num_per_block;
  for (int pidib = threadIdx.x; pidib < pcnt; pidib += blockDim.x) {
    auto parid = bucket[pidib];
    auto pbin = (g_buckets_on_particle_buffer) 
                ? pbuffer.ch(_0, pbuffer._binsts[blockno] + pidib / g_bin_capacity)
                : pbuffer.ch(_0, partition._binsts[blockno] + pidib / g_bin_capacity);
    /// Position of particle (x,y,z) in 3D dimensionless grid-space [0, 1]
    pbin.val(_0, pidib % g_bin_capacity) = parray.val(_0, parid);
    pbin.val(_1, pidib % g_bin_capacity) = parray.val(_1, parid);
    pbin.val(_2, pidib % g_bin_capacity) = parray.val(_2, parid);
    /// F
    pbin.val(_3, pidib % g_bin_capacity) = 1.0;
    pbin.val(_4, pidib % g_bin_capacity) = 0.0;
    pbin.val(_5, pidib % g_bin_capacity) = 0.0;
    pbin.val(_6, pidib % g_bin_capacity) = 0.0;
    pbin.val(_7, pidib % g_bin_capacity) = 1.0;
    pbin.val(_8, pidib % g_bin_capacity) = 0.0;
    pbin.val(_9, pidib % g_bin_capacity) = 0.0;
    pbin.val(_10, pidib % g_bin_capacity) = 0.0;
    pbin.val(_11, pidib % g_bin_capacity) = 1.0;
    pbin.val(_12, pidib % g_bin_capacity) = pbuffer.logJp0;
    pbin.val(_13, pidib % g_bin_capacity) = 0.0; //< sJBar
    pbin.val(_14, pidib % g_bin_capacity) = (PREC)parid; //< Particle ID
  }
}

template <typename ParticleArray, typename Partition>
__global__ void array_to_buffer(ParticleArray parray,
                                ParticleBuffer<material_e::NACC_ASFLIP_FBAR> pbuffer,
                                Partition partition, vec<PREC, 3> vel) {
  uint32_t blockno = blockIdx.x;
  int pcnt = (g_buckets_on_particle_buffer) 
              ? pbuffer._ppbs[blockno] 
              : partition._ppbs[blockno];
  auto bucket = (g_buckets_on_particle_buffer) 
                ? pbuffer._blockbuckets + blockno * g_particle_num_per_block 
                : partition._blockbuckets + blockno * g_particle_num_per_block;
  for (int pidib = threadIdx.x; pidib < pcnt; pidib += blockDim.x) {
    auto parid = bucket[pidib];
    auto pbin = (g_buckets_on_particle_buffer) 
                ? pbuffer.ch(_0, pbuffer._binsts[blockno] + pidib / g_bin_capacity)
                : pbuffer.ch(_0, partition._binsts[blockno] + pidib / g_bin_capacity);
    /// Position of particle (x,y,z) in 3D dimensionless grid-space [0, 1]
    pbin.val(_0, pidib % g_bin_capacity) = parray.val(_0, parid);
    pbin.val(_1, pidib % g_bin_capacity) = parray.val(_1, parid);
    pbin.val(_2, pidib % g_bin_capacity) = parray.val(_2, parid);
    /// F
    pbin.val(_3, pidib % g_bin_capacity) = 1.0;
    pbin.val(_4, pidib % g_bin_capacity) = 0.0;
    pbin.val(_5, pidib % g_bin_capacity) = 0.0;
    pbin.val(_6, pidib % g_bin_capacity) = 0.0;
    pbin.val(_7, pidib % g_bin_capacity) = 1.0;
    pbin.val(_8, pidib % g_bin_capacity) = 0.0;
    pbin.val(_9, pidib % g_bin_capacity) = 0.0;
    pbin.val(_10, pidib % g_bin_capacity) = 0.0;
    pbin.val(_11, pidib % g_bin_capacity) = 1.0;
    pbin.val(_12, pidib % g_bin_capacity) = pbuffer.logJp0;
    pbin.val(_13, pidib % g_bin_capacity) = vel[0];
    pbin.val(_14, pidib % g_bin_capacity) = vel[1];
    pbin.val(_15, pidib % g_bin_capacity) = vel[2];
    pbin.val(_16, pidib % g_bin_capacity) = 0.0; //< sJBar
    pbin.val(_17, pidib % g_bin_capacity) = (PREC)parid; //< Particle ID
  }
}

template <typename ParticleArray, typename Partition>
__global__ void array_to_buffer(ParticleArray parray,
                                ParticleBuffer<material_e::CoupledUP> pbuffer,
                                Partition partition, vec<PREC, 3> vel) {
  uint32_t blockno = blockIdx.x;
  int pcnt = (g_buckets_on_particle_buffer) 
              ? pbuffer._ppbs[blockno] 
              : partition._ppbs[blockno];
  auto bucket = (g_buckets_on_particle_buffer) 
                ? pbuffer._blockbuckets + blockno * g_particle_num_per_block 
                : partition._blockbuckets + blockno * g_particle_num_per_block;
  for (int pidib = threadIdx.x; pidib < pcnt; pidib += blockDim.x) {
    auto parid = bucket[pidib];
    auto pbin = (g_buckets_on_particle_buffer) 
                ? pbuffer.ch(_0, pbuffer._binsts[blockno] + pidib / g_bin_capacity)
                : pbuffer.ch(_0, partition._binsts[blockno] + pidib / g_bin_capacity);
    /// Position of particle (x,y,z) in 3D dimensionless grid-space [0, 1]
    pbin.val(_0, pidib % g_bin_capacity) = parray.val(_0, parid);
    pbin.val(_1, pidib % g_bin_capacity) = parray.val(_1, parid);
    pbin.val(_2, pidib % g_bin_capacity) = parray.val(_2, parid);
    /// F
    pbin.val(_3, pidib % g_bin_capacity) = 1.0;
    pbin.val(_4, pidib % g_bin_capacity) = 0.0;
    pbin.val(_5, pidib % g_bin_capacity) = 0.0;
    pbin.val(_6, pidib % g_bin_capacity) = 0.0;
    pbin.val(_7, pidib % g_bin_capacity) = 1.0;
    pbin.val(_8, pidib % g_bin_capacity) = 0.0;
    pbin.val(_9, pidib % g_bin_capacity) = 0.0;
    pbin.val(_10, pidib % g_bin_capacity) = 0.0;
    pbin.val(_11, pidib % g_bin_capacity) = 1.0;
    pbin.val(_12, pidib % g_bin_capacity) = pbuffer.logJp0;
    pbin.val(_13, pidib % g_bin_capacity) = vel[0];
    pbin.val(_14, pidib % g_bin_capacity) = vel[1];
    pbin.val(_15, pidib % g_bin_capacity) = vel[2];
    pbin.val(_16, pidib % g_bin_capacity) = 0.0; //< sJBar
    pbin.val(_17, pidib % g_bin_capacity) = 0.0; //< place-holder or mass_water CoupledUP
    pbin.val(_18, pidib % g_bin_capacity) = 0.0; //< pressure_water CoupledUP
    pbin.val(_19, pidib % g_bin_capacity) = (PREC)parid; //< Particle ID
  }
}

template <typename ParticleArray, typename Partition>
__global__ void array_to_buffer(ParticleArray parray,
                                ParticleBuffer<material_e::Meshed> pbuffer,
                                Partition partition, vec<PREC, 3> vel) {
  uint32_t blockno = blockIdx.x;
  int pcnt = (g_buckets_on_particle_buffer) 
              ? pbuffer._ppbs[blockno] 
              : partition._ppbs[blockno];
  auto bucket = (g_buckets_on_particle_buffer) 
                ? pbuffer._blockbuckets + blockno * g_particle_num_per_block 
                : partition._blockbuckets + blockno * g_particle_num_per_block;
  for (int pidib = threadIdx.x; pidib < pcnt; pidib += blockDim.x) {
    auto parid = bucket[pidib];
    auto pbin = (g_buckets_on_particle_buffer) 
                ? pbuffer.ch(_0, pbuffer._binsts[blockno] + pidib / g_bin_capacity)
                : pbuffer.ch(_0, partition._binsts[blockno] + pidib / g_bin_capacity);
    if (parid >= g_max_fem_vertice_num) { printf("FEM Particle %d invalid ID! Bigger than g_max_fem_vertice_num, increase and recompile.\n", parid); break; }
    /// Position of particle (x,y,z) in 3D dimensionless grid-space [0, 1]
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


template <typename T = PREC_G>
__device__ void apply_friction_to_grid_velocity(T* vel, T* normal, T friction_static, T friction_dynamic, T inv_mass, double dt, T length, double* grav, T decay_coef = 1.f) {
  // TODO: Currently just does static friction, update for dynamic friction
  // Assume velocity is GPU scale, assume gravity Real Scale (m/s^2)
  T force[3]; // Force on grid node. Assumes external force (e.g. gravity) not yet included.
  for (int d=0; d<3; ++d) force[d] = (vel[d] / dt + grav[d] / length) / inv_mass; 
  
  // Check for contact with friction surface (normal force magnitude is negative)
  T normal_force_magnitude = force[0]*normal[0] + force[1]*normal[1] + force[2]*normal[2]; 
  if (normal_force_magnitude >= 0.f) return;

  T tangent_force[3];
  for (int d=0; d<3; d++) tangent_force[d] = force[d] - normal_force_magnitude * normal[d]; 
  T tangent_force_magnitude = sqrt( tangent_force[0]*tangent_force[0] + 
                                    tangent_force[1]*tangent_force[1] + 
                                    tangent_force[2]*tangent_force[2]);

  // Check if exceeding static friction: set to dynamic friction if so, else use static friction
  T ratio_lost_to_friction = (tangent_force_magnitude > -normal_force_magnitude * friction_static) 
                           ? min((-normal_force_magnitude * friction_dynamic) / tangent_force_magnitude, (T) 1.f) : (T) 1.f;
  decay_coef = max(min(decay_coef, 1.f), 0.f); //< Clamp decay coef to [0, 1]
  for (int d=0; d<3; d++) 
    vel[d] -= decay_coef * tangent_force[d] * ratio_lost_to_friction * inv_mass * dt;
}

template <typename T = PREC_G>
__device__ void apply_friction_to_grid_momentum(T* momentum, T* normal, T friction_static, T friction_dynamic, T inv_mass, double dt, T length, double* grav, T decay_coef = 1.f) {
  // TODO: Currently just does static friction, update for dynamic friction
  // Assume velocity is GPU scale, assume gravity Real Scale (m/s^2)
  T force[3]; // Force on grid node. Assumes external force (e.g. gravity) not yet included.
  for (int d=0; d<3; ++d) force[d] = (momentum[d] / dt) + (grav[d] / length) / inv_mass; 
  
  // Check for contact with friction surface (normal force magnitude is negative)
  T normal_force_magnitude = force[0]*normal[0] + force[1]*normal[1] + force[2]*normal[2]; 
  if (normal_force_magnitude >= 0.f) return;

  T tangent_force[3];
  for (int d=0; d<3; d++) tangent_force[d] = force[d] - normal_force_magnitude * normal[d]; 
  T tangent_force_magnitude = sqrt( tangent_force[0]*tangent_force[0] + 
                                    tangent_force[1]*tangent_force[1] + 
                                    tangent_force[2]*tangent_force[2]);

  // Check if exceeding static friction: set to dynamic friction if so, else use static friction
  T ratio_lost_to_friction = (tangent_force_magnitude > -normal_force_magnitude * friction_static) 
                           ? min((-normal_force_magnitude * friction_dynamic) / tangent_force_magnitude, (T) 1.f) : (T) 1.f;
  decay_coef = max(min(decay_coef, 1.f), 0.f); //< Clamp decay coef to [0, 1]
  for (int d=0; d<3; d++) 
    momentum[d] -= decay_coef * tangent_force[d] * ratio_lost_to_friction * dt;
}

template <typename Grid, typename Partition, int boundary_cnt = g_max_grid_boundaries>
__global__ void update_grid_velocity_query_max(uint32_t blockCount, Grid grid,
                                               Partition partition, double dt,
                                               float *maxVel, double curTime, pvec3 grav, 
                                               vec<vec7, boundary_cnt> boundary_array,
                                               struct GridBoundaryConfigs * gridBoundary_array, 
                                               vec3 boundary_motion, PREC length, double fr_scale) {
  constexpr int bc = g_bc;
  constexpr int numWarps =
      g_num_grid_blocks_per_cuda_block * g_num_warps_per_grid_block; // Num. warps per block
  constexpr unsigned activeMask = 0xffffffff; // Active mask for all threads
  extern __shared__ float sh_maxvels[]; // Shared memory for max. velocity reduction
  if (threadIdx.x < numWarps) sh_maxvels[threadIdx.x] = 0.0f; // Zero shared memory
  __syncthreads();

  std::size_t blockno = blockIdx.x * g_num_grid_blocks_per_cuda_block +
                        threadIdx.x / 32 / g_num_warps_per_grid_block; 

  PREC_G o = g_offset; // Domain offset [ ], for Quad. B-Splines (Off-by-2, Wang 2020)
  PREC_G l = length; // Length of domain [m]

  /// Within-warp computations
  if (blockno < blockCount) 
  {
    auto blockid = partition._activeKeys[blockno];
    int isInBuffer = ((blockid[0] < bc || blockid[0] >= g_grid_size_x - bc) << 2) |
                     ((blockid[1] < bc || blockid[1] >= g_grid_size_y - bc) << 1) |
                     ((blockid[2] < bc || blockid[2] >= g_grid_size_z - bc));
    auto grid_block = grid.ch(_0, blockno);
    PREC_G velSqr = 0.f;

    for (int cidib = threadIdx.x % 32; cidib < g_blockvolume; cidib += 32) 
    {
      PREC_G mass = grid_block.val_1d(_0, cidib), vel[3], vel_FLIP[3], voln, sJBar;
#if DEBUG_COUPLED_UP
      PREC_G mass_water, pressure_water;
#endif
      if (mass > 0.f) {
        mass = (1.0 / mass); //< Now represents inverse mass, 1 / mass, for efficiency

        // Grid node coordinate [i,j,k] in grid-block
        int i = (cidib >> (g_blockbits << 1)) & g_blockmask;
        int j = (cidib >> g_blockbits) & g_blockmask;
        int k =  cidib & g_blockmask;
        // Grid node position [x,y,z] in entire domain
        PREC_G xc = (g_blocksize * blockid[0] + i) * g_dx; // Node x position [0,1]
        PREC_G yc = (g_blocksize * blockid[1] + j) * g_dx; // Node y position [0,1]
        PREC_G zc = (g_blocksize * blockid[2] + k) * g_dx; // Node z position [0,1]
        // Retrieve grid momentums (kg*m/s2)
        vel[0] = grid_block.val_1d(_1, cidib); //< mvx+dt(fint)
        vel[1] = grid_block.val_1d(_2, cidib); //< mvy+dt(fint)
        vel[2] = grid_block.val_1d(_3, cidib); //< mvz+dt(fint)
        vel_FLIP[0] = grid_block.val_1d(_4, cidib); //< mvx
        vel_FLIP[1] = grid_block.val_1d(_5, cidib); //< mvy
        vel_FLIP[2] = grid_block.val_1d(_6, cidib); //< mvz
        voln = grid_block.val_1d(_7, cidib); //< Sum_p ( voln )
        sJBar = grid_block.val_1d(_8, cidib); //< Sum_p ( sJBar * voln )
#if DEBUG_COUPLED_UP
        mass_water = grid_block.val_1d(_9, cidib); //< Sum_p ( mass_water )
        pressure_water = grid_block.val_1d(_10, cidib); //< Sum_p ( pressure_water )
#endif

        //< sJBar = (1 - JBar) = Sum_p ( sJBar * voln ) / Sum_p ( voln ), zero-out if no volume
        sJBar = (voln > 0.f) ? sJBar / voln : 0.f; // No division by zero or negative volumes
        constexpr PREC_G deformation_limit = 0.9f; // Standard limit for vol. deformation in common equations of state (e.g. Tait-Murnaghan Water)
        sJBar = (abs(sJBar) > deformation_limit) ? copysign(deformation_limit , sJBar) : sJBar; //< Clamp sJBar to [-deformation_limit, deformation_limit], done for stability of compressible fluids, though may be unnecassary for some materials
  
        // ! TODO : move into scopes to save registers
        int isInBound = 0;
        int isOutFlume = 0, isSlipFlume = 0, isSepFlume = 0;
        int isOutStruct = 0, isOnStruct = 0; 
        constexpr PREC_G tol = 0.f ; // g_dx * 0.0078125f; // Tolerance 1/128 grid-cell

        // Loop over all grid-boundaries for every active grid-node
        // NOTE: This loop only applies boundaries that are momentum based. Next loop handles velocity ones.
        // TODO: Move to separate kernel
        vec7 boundary; // ! Deprecated, better to use structures
// #pragma unroll g_max_grid_boundaries
        for (int g = 0; g < g_max_grid_boundaries; g++) {
          isOutFlume = isSlipFlume = isSepFlume = isOnStruct = 0; // Tracks if node is within grid-boundary domain per dimension
// #pragma unroll 7
          for (int d = 0; d < g_grid_boundary_attribs; d++)
            boundary[d] = boundary_array[g][d];
          if (boundary[6] == -1) continue; // TODO : Deprecate. Base on grid-boundary structure instead. 
          // Read boundary data
          auto gb = gridBoundary_array[g]; // Grid-boundary structure

          // Check if boundary is active via user-defined boundary time-interval
          if ( (curTime < gb._time[0]) || (curTime > gb._time[1]) ) continue;
          // Set boundary friction values
          PREC_G friction_static = gb._friction_static;
          PREC_G friction_dynamic = gb._friction_dynamic;

          // Loop over array of the boundary, default array of [1,1,1] i.e. single boundary
          // TODO : Deprecate boundary_dim and boundary_pos, just use grid-boundary structure instead. Maybe read in values to device / constant memory.
          gvec3 boundary_dim, boundary_pos;
          boundary_dim[0] = gb._domain_end[0] - gb._domain_start[0];
          boundary_pos[0] = gb._domain_start[0];
          for (int arr_i = 0; arr_i < gb._array[0]; ++arr_i) {
            boundary_dim[1] = gb._domain_end[1] - gb._domain_start[1];
            boundary_pos[1] = gb._domain_start[1];
            for (int arr_j = 0; arr_j < gb._array[1]; ++arr_j) {
              boundary_dim[2] = gb._domain_end[2] - gb._domain_start[2];
              boundary_pos[2] = gb._domain_start[2]; 
              for (int arr_k = 0; arr_k < gb._array[2]; ++arr_k) {
                // Evaluate type of boundary. 
                // TODO: Move to separate kernels

                // Exterior wall boundary (momentum based). Particles will not exit (if stable sim). Creates effective simulation domain, can embed objects ~2 cells deep into it usually for fixed/roller/etc BCs. Careful not to exit the actual simulation domain, will crash.
                if (gb._object == boundary_object_t::Walls) {
                  if (gb._contact == boundary_contact_t::Sticky) {
                    isOutFlume =  (((xc <= boundary_pos[0]  + tol ) || (xc >= boundary_pos[0] + boundary_dim[0] - tol )) << 2) |
                                      (((yc <= boundary_pos[1] + tol ) || (yc >= boundary_pos[1] + boundary_dim[1] - tol )) << 1) |
                                        ((zc <= boundary_pos[2] + tol ) || (zc >= boundary_pos[2] + boundary_dim[2] - tol ));                          
                    if (isOutFlume) isOutFlume = ((1 << 2) | (1 << 1) | 1);
                    isInBound |= isOutFlume; // Update with regular boundary for efficiency
                  } else if (gb._contact == boundary_contact_t::Slip) {
                    isOutFlume =  (((xc <= boundary_pos[0]  + tol - 1.5*g_dx) || (xc >= boundary_pos[0] + boundary_dim[0] - tol + 1.5*g_dx)) << 2) |
                                      (((yc <= boundary_pos[1] + tol - 1.5*g_dx) || (yc >= boundary_pos[1] + boundary_dim[1] - tol + 1.5*g_dx)) << 1) |
                                        ((zc <= boundary_pos[2] + tol - 1.5*g_dx) || (zc >= boundary_pos[2] + boundary_dim[2] - tol + 1.5*g_dx));                          
                    if (isOutFlume) isOutFlume = ((1 << 2) | (1 << 1) | 1);
                    isInBound |= isOutFlume; // Update with regular boundary for efficiency
                    
                    isSlipFlume =  (((xc <= boundary_pos[0] + tol) || (xc >= boundary_pos[0] + boundary_dim[0] - tol)) << 2) |
                                      (((yc <= boundary_pos[1] + tol) || (yc >= boundary_pos[1] + boundary_dim[1] - tol)) << 1) |
                                        ((zc <= boundary_pos[2] + tol) || (zc >= boundary_pos[2] + boundary_dim[2] - tol));                          
                    isInBound |= isSlipFlume; // Update with regular boundary for efficiency
                  } else if (gb._contact == boundary_contact_t::Separable) {
                    isOutFlume =  (((xc <= boundary_pos[0] + tol - 1.5*g_dx) || (xc >= boundary_pos[0] + boundary_dim[0] - tol + 1.5*g_dx)) << 2) |
                                      (((yc <= boundary_pos[1] + tol - 1.5*g_dx) || (yc >= boundary_pos[1] + boundary_dim[1] - tol + 1.5*g_dx)) << 1) |
                                      (((zc <= boundary_pos[2] + tol - 1.5*g_dx) || (zc >= boundary_pos[2] + boundary_dim[2] - tol + 1.5*g_dx)));                          
                    if (isOutFlume) isOutFlume = ((1 << 2) | (1 << 1) | 1);
                    isInBound |= isOutFlume; // Update with regular boundary for efficiency

                    isSepFlume = (((xc <= boundary_pos[0] + tol && vel[0] < 0) || (xc >= boundary_pos[0] + boundary_dim[0] - tol && vel[0] > 0)) << 2) |
                                    (((yc <= boundary_pos[1] + tol && vel[1] < 0) || (yc >= boundary_pos[1] + boundary_dim[1] - tol && vel[1] > 0)) << 1) |
                                      ((zc <= boundary_pos[2] + tol && vel[2] < 0) || (zc >= boundary_pos[2] + boundary_dim[2] - tol && vel[2] > 0));                          
                    isInBound |= isSepFlume; // Update with regular boundary for efficiency
                  }
                }
                // Box rigid boundary
                else if (gb._object == boundary_object_t::Box) {
                  constexpr int NUM_BOX_FACES = 6;
                  if (gb._contact == boundary_contact_t::Sticky) {

                    // Check if grid-cell is within sticky interior of box
                    isOutStruct  = ((xc >= boundary_pos[0] && xc <= boundary_pos[0] + boundary_dim[0]) << 2) | 
                                      ((yc >= boundary_pos[1] && yc <= boundary_pos[1] + boundary_dim[1]) << 1) |
                                        (zc >= boundary_pos[2] && zc <= boundary_pos[2] + boundary_dim[2]);
                    if (isOutStruct != ((1 << 2) | (1 << 1) | 1)) isOutStruct = 0; // Check if 111, reset otherwise
                    isInBound |= isOutStruct; // Update with regular boundary for efficiency
                  } else if (gb._contact == boundary_contact_t::Slip) {
                    constexpr PREC_G outer_tol = 0.99f * g_dx; // Slip layer thickness for box
                    constexpr PREC_G t = outer_tol + tol; // Slip layer thickness for box

                    // Check if grid-cell is within sticky interior of box
                    // Subtract slip-layer thickness from structural box dimension for geometry
                    isOutStruct  = ((xc > boundary_pos[0] + t && xc < boundary_pos[0] + boundary_dim[0] - t) << 2) | 
                                      ((yc > boundary_pos[1] + t && yc < boundary_pos[1] + boundary_dim[1] - t) << 1) |
                                        (zc > boundary_pos[2] + t && zc < boundary_pos[2] + boundary_dim[2] - t);
                    if (isOutStruct != ((1 << 2) | (1 << 1) | 1)) isOutStruct = 0; // Check if 111, reset otherwise
                    isInBound |= isOutStruct; // Update with regular boundary for efficiency
                    
                    // Check exterior slip-layer of rigid box
                    // One-cell depth, six-faces, order matters! (over-writes on edges, favors front) 
                    int isOnStructFace[NUM_BOX_FACES];
                    // Right (z+)
                    isOnStructFace[0] = ((xc >= boundary_pos[0] && xc <= boundary_pos[0] + boundary_dim[0]) << 2) | 
                                        ((yc >= boundary_pos[1] && yc <= boundary_pos[1] + boundary_dim[1]) << 1) |
                                        (zc >= boundary_pos[2] + boundary_dim[2] - outer_tol && zc <= boundary_pos[2] + boundary_dim[2]);
                    // Left (z-)
                    isOnStructFace[1] = ((xc >= boundary_pos[0] && xc <= boundary_pos[0] + boundary_dim[0]) << 2) | 
                                        ((yc >= boundary_pos[1] && yc <= boundary_pos[1] + boundary_dim[1]) << 1) |
                                        (zc >= boundary_pos[2] && zc <= boundary_pos[2] + outer_tol);        
                    // Top (y+)
                    isOnStructFace[2] = ((xc >= boundary_pos[0] && xc <= boundary_pos[0] + boundary_dim[0]) << 2) | 
                                        ((yc >= boundary_pos[1] + boundary_dim[1] - outer_tol && yc <= boundary_pos[1] + boundary_dim[1]) << 1) |
                                        (zc >= boundary_pos[2] && zc <= boundary_pos[2] + boundary_dim[2]);
                    // Bottom (y-)
                    isOnStructFace[3] = ((xc >= boundary_pos[0] && xc <= boundary_pos[0] + boundary_dim[0]) << 2) | 
                                        ((yc >= boundary_pos[1] && yc <= boundary_pos[1] + outer_tol) << 1) |
                                        (zc >= boundary_pos[2] && zc <= boundary_pos[2] + boundary_dim[2]);
                    // Back (x+)
                    isOnStructFace[4] = ((xc >= boundary_pos[0] + boundary_dim[0] - outer_tol && xc <= boundary_pos[0] + boundary_dim[0]) << 2) | 
                                        ((yc >= boundary_pos[1] && yc <= boundary_pos[1] + boundary_dim[1]) << 1) |
                                        (zc >= boundary_pos[2] && zc <= boundary_pos[2] + boundary_dim[2]);
                    // Front (x-)
                    isOnStructFace[5] = ((xc >= boundary_pos[0] && xc <= boundary_pos[0] + outer_tol) << 2) | 
                                        ((yc >= boundary_pos[1] && yc <= boundary_pos[1] + boundary_dim[1]) << 1) |
                                        (zc >= boundary_pos[2] && zc <= boundary_pos[2] + boundary_dim[2]);                             
                    // TODO: Make below box collider a function
                    // Combine box face collisions (isOnStructFace[]) into isOnStruct, efficient
                    // Check if 111 (i.e. within box = 7, all flags), otherwise set 000 (no collision)
                    // iter [0, 1, 2, 3, 4, 5] -> iter/2 [0, 1, 2] <->  [z, y, z], used as bit shift.
                    isOnStruct = 0; // Collision reduction variable

        #pragma unroll 6
                    for (int iter=0; iter<NUM_BOX_FACES; iter++) {
                      isOnStructFace[iter] = (isOnStructFace[iter]==((1 << 2) | (1 << 1) | 1)) ? (1 << iter / 2) : 0;
                      isOnStruct |= isOnStructFace[iter]; // OR (|=) combines face collision flags into one variable
                    }
                    // X-Favoring currently, will make an option in scene for user later (e.g. X, bevel)
                    if (isOnStruct == ((1 << 2) | (1 << 1) | 0) || isOnStruct == ((1 << 2) | 0 | 1) || isOnStruct == ((1 << 2) | 1 | 1) ) isOnStruct = (1 << 2); // Overlaps front (XY,XZ,XYZ)->(X)
                    else if (isOnStruct == (0 | (1 << 1) | 1) ) isOnStruct = 0; // Overlaps (YZ)->(0)
                    isInBound |= isOnStruct; // OR reduce box sticky collision into isInBound
                  }
                  else if (gb._contact == boundary_contact_t::Separable) {
                    // TODO: Base tolerance for outer layer on user-input
                    constexpr PREC_G outer_tol = 0.99f * g_dx; // Outer layer thickness for box
                    constexpr PREC_G t = outer_tol + tol; // Outer layer thickness for box

                    // Check if grid-cell is within sticky interior of box
                    // Subtract outer-layer thickness from structural box dimension for geometry
                    // Exclusive boundary, i.e. not including nodes at transitiion to outer-layer
                    isOutStruct  = ((xc > boundary_pos[0] + t && xc < boundary_pos[0] + boundary_dim[0] - t) << 2) | 
                                      ((yc > boundary_pos[1] + t && yc < boundary_pos[1] + boundary_dim[1] - t) << 1) |
                                        (zc > boundary_pos[2] + t && zc < boundary_pos[2] + boundary_dim[2] - t);
                    if (isOutStruct != ((1 << 2) | (1 << 1) | 1)) isOutStruct = 0; // Check if 111, reset otherwise
                    isInBound |= isOutStruct; // Update with regular boundary for efficiency
                    
                    // Check exterior slip-layer of rigid box
                    // One-cell depth, six-faces, order matters! (over-writes on edges, favors front) 
                    int isOnStructFace[NUM_BOX_FACES];
                    // Right (z+)
                    isOnStructFace[0] = ((xc >= boundary_pos[0] && xc <= boundary_pos[0] + boundary_dim[0]) << 2) | 
                                        ((yc >= boundary_pos[1] && yc <= boundary_pos[1] + boundary_dim[1]) << 1) |
                                        (zc >= boundary_pos[2] + boundary_dim[2] - t && vel[2] < 0.f && zc <= boundary_pos[2] + boundary_dim[2]);
                    // Left (z-)
                    isOnStructFace[1] = ((xc >= boundary_pos[0] && xc <= boundary_pos[0] + boundary_dim[0]) << 2) | 
                                        ((yc >= boundary_pos[1] && yc <= boundary_pos[1] + boundary_dim[1]) << 1) |
                                        (zc >= boundary_pos[2] && vel[2] > 0.f && zc <= boundary_pos[2] + t);        
                    // Top (y+)
                    isOnStructFace[2] = ((xc >= boundary_pos[0] && xc <= boundary_pos[0] + boundary_dim[0]) << 2) | 
                                        ((yc >= boundary_pos[1] + boundary_dim[1] - t && vel[1] < 0.f && yc <= boundary_pos[1] + boundary_dim[1]) << 1) |
                                        (zc >= boundary_pos[2] && zc <= boundary_pos[2] + boundary_dim[2]);
                    // Bottom (y-)
                    isOnStructFace[3] = ((xc >= boundary_pos[0] && xc <= boundary_pos[0] + boundary_dim[0]) << 2) | 
                                        ((yc >= boundary_pos[1] && vel[1] > 0.f && yc <= boundary_pos[1] + t) << 1) |
                                        (zc >= boundary_pos[2] && zc <= boundary_pos[2] + boundary_dim[2]);
                    // Back (x+)
                    isOnStructFace[4] = ((xc >= boundary_pos[0] + boundary_dim[0] - t && vel[0] < 0.f && xc <= boundary_pos[0] + boundary_dim[0]) << 2) | 
                                        ((yc >= boundary_pos[1] && yc <= boundary_pos[1] + boundary_dim[1]) << 1) |
                                        (zc >= boundary_pos[2] && zc <= boundary_pos[2] + boundary_dim[2]);
                    // Front (x-)
                    isOnStructFace[5] = ((xc >= boundary_pos[0] && vel[0] > 0.f && xc <= boundary_pos[0] + t) << 2) | 
                                        ((yc >= boundary_pos[1] && yc <= boundary_pos[1] + boundary_dim[1]) << 1) |
                                        (zc >= boundary_pos[2] && zc <= boundary_pos[2] + boundary_dim[2]);                             
                    // Reduce results from box faces to single result
                    isOnStruct = 0; // Collision reduction variable
        #pragma unroll 6
                    for (int iter=0; iter<NUM_BOX_FACES; iter++) {
                      isOnStructFace[iter] = (isOnStructFace[iter]==((1 << 2) | (1 << 1) | 1)) ? (1 << iter / 2) : 0;
                      isOnStruct |= isOnStructFace[iter]; // OR (|=) combines face collisions into one int
                    }
                    // X-favored, TODO: Make option in scene file
                    if (isOnStruct == ((1 << 2) | (1 << 1) | 0) || isOnStruct == ((1 << 2) | (0 << 1) | 1) || isOnStruct == ((1 << 2) | (1 << 1) | 1)) isOnStruct = ((1 << 2) | (0 << 1) | 0); // Overlaps front (XY,XZ, XYZ)->(X)
                    else if (isOnStruct == ((0 << 2) | (1 << 1) | 1) ) isOnStruct = 0; // Overlaps (YZ)->(0)
                    isInBound |= isOnStruct; // OR reduce into regular boundary for efficiency
                  }
                }
                // TODO: 1) Right now cylinder assumes perfect circle cross-section, i.e. not ellipse. 
                //          Affects normal vector and penetration condition calc.
                //       2) Assumes Y+ oriented long axis for now
                //       3) No "caps" on the cylinder faces, only the cylinder body
                else if (gb._object == boundary_object_t::Cylinder) { // Assume Y+ oriented axis for now
                  PREC_G r = boundary_dim[0] * 0.5f; // Cylinder radius assuming perfect circle cross-section and Y+ axis
                  PREC_G d = sqrt((xc - (boundary_pos[0] + boundary_dim[0] * 0.5f)) * (xc - (boundary_pos[0] + boundary_dim[0] * 0.5f)) 
                                + (zc - (boundary_pos[2] + boundary_dim[2] * 0.5f)) * (zc - (boundary_pos[2] + boundary_dim[2] * 0.5f))); // Distance from center
                  const bool isInRadius = (d <= r);
                  const bool isInHeight = (yc >= boundary_pos[1]) && (yc <= boundary_pos[1] + boundary_dim[1]);
                  const bool isInside = isInRadius && isInHeight;
                  if (gb._contact == boundary_contact_t::Sticky) {
                    isOutStruct = isInside * ((1 << 2) | (1 << 1) | 1); // Check if within cylinder radius
                    isInBound |= isOutStruct; // Update with regular boundary for efficiency
                  } else if (gb._contact == boundary_contact_t::Slip || gb._contact == boundary_contact_t::Separable) {
                    // Assumes perfect circle cross-section and Y+ oriented axis for now
                    if (isInside) {
                      // Distance vector from center axis
                      PREC_G dist[3];
                      dist[0] = xc - (boundary_pos[0] + boundary_dim[0] * 0.5f);
                      dist[1] = 0.f; // Assume Y+ oriented axis for now
                      dist[2] = zc - (boundary_pos[2] + boundary_dim[2] * 0.5f);

                      // Normalize distance vector
                      PREC_G norm = 0.f;
                      for (int i = 0; i < 3; i++) norm += dist[i] * dist[i];
                      const PREC_G small = 1e-6f; // Small value to prevent division by zero
                      norm = sqrt(norm);
                      if (norm > 0.f) for (int i = 0; i < 3; i++) dist[i] /= norm; 
                      
                      // Dot product of normalized distance vector and velocity vector
                      PREC_G dot = 0.f; 
                      for (int i = 0; i < 3; i++) dot += dist[i] * vel[i];

                      // Subtract the projection of velocity vector onto distance vector from velocity vector
                      if (gb._contact == boundary_contact_t::Slip) {
                        for (int i = 0; i < 3; i++) vel[i] = vel[i] - dot * dist[i];
                      } else {
                        // Separable boundary, check if grid node velocity is pointing into boundary
                        const bool isDecoupling = (dot >= 0.f);
                        if (isDecoupling == false) {
                          for (int i = 0; i < 3; i++) vel[i] = vel[i] - dot * dist[i];
                        }
                      }
                    }
                  }
                } 
                // TODO: Right now sphere assumes perfect sphere, i.e. not ellipsoid. 
                //       Affects normal vector and penetration condition calc.
                else if (gb._object == boundary_object_t::Sphere) {
                  PREC_G r = boundary_dim[0] * 0.5f; // Sphere radius
                  PREC_G d = sqrt((xc - (boundary_pos[0] + boundary_dim[0] * 0.5f)) * (xc - (boundary_pos[0] + boundary_dim[0] * 0.5f)) 
                                + (yc - (boundary_pos[1] + boundary_dim[1] * 0.5f)) * (yc - (boundary_pos[1] + boundary_dim[1] * 0.5f)) 
                                + (zc - (boundary_pos[2] + boundary_dim[2] * 0.5f)) * (zc - (boundary_pos[2] + boundary_dim[2] * 0.5f))); // Distance from center
                  const bool isInRadius = (d <= r);
                  if (gb._contact == boundary_contact_t::Sticky) {
                    // Check if grid-cell is within sticky interior of sphere
                    isOutStruct = isInRadius * ((1 << 2) | (1 << 1) | 1); // If within sphere radius set xyz boundary bit flags
                    isInBound |= isOutStruct; // Update with regular boundary for efficiency
                  } else if (gb._contact == boundary_contact_t::Slip || gb._contact == boundary_contact_t::Separable) {
                    // Check if grid-cell is within sticky interior of sphere
                    if (isInRadius) {
                      // distance vector from center axis
                      PREC_G dist[3];
                      dist[0] = xc - (boundary_pos[0] + boundary_dim[0] * 0.5f);
                      dist[1] = yc - (boundary_pos[1] + boundary_dim[1] * 0.5f);
                      dist[2] = zc - (boundary_pos[2] + boundary_dim[2] * 0.5f);

                      // normalize distance vector
                      PREC_G norm = 0.f;
                      for (int i = 0; i < 3; i++) norm += dist[i] * dist[i];
                      norm = sqrt(norm);
                      if (norm > 0.f) for (int i = 0; i < 3; i++) dist[i] /= norm;
                      
                      // dot product of normalized distance vector and velocity vector
                      PREC_G dot = 0.f;
                      for (int i = 0; i < 3; i++) dot += dist[i] * vel[i];

                      // subtract the projection of velocity vector onto distance vector from velocity vector
                      if (gb._contact == boundary_contact_t::Slip) {
                        for (int i = 0; i < 3; i++) vel[i] = vel[i] - dot * dist[i];
                      } else {
                        // Separable boundary, check if node is moving towards boundary
                        const bool isDecoupling = (dot >= 0.f);
                        if (isDecoupling == false) {
                          // subtract the projection of velocity vector onto distance vector from velocity vector
                          for (int i = 0; i < 3; i++) vel[i] = vel[i] - dot * dist[i];
                        }
                      }
                    }
                  }
                }

                else if (gb._object == boundary_object_t::USGS_GATE) {
                  if (gb._contact == boundary_contact_t::Separable) {
                    // TODO : Reimplement timed boundaries
                    PREC_G time_start = 1.0f * sqrt(fr_scale); // Time til motion start [sec]
                    PREC_G time_end   = 2.0f * sqrt(fr_scale); // Time at motion finish [sec]

                    PREC_G gate_x  = 4.7f * fr_scale; // Gate base X position [m]
                    PREC_G gate_z1 = 4.0f * fr_scale; // Gate center Z position [m]
                    PREC_G gate_z2 = 4.0f * fr_scale; // Gate center Z position [m]
                    PREC_G gate_width = 1.f * fr_scale; // Gate single door (x2) Z width [m]
                    

                    constexpr PREC_G gate_slope = -1.73205080757f; // -tangent(60.f * 3.14159265358979323846f / 180.f); // Gate slope [m]
                    PREC_G gate_y = gate_slope * ((xc-o)*l - gate_x); // Gate Y for an X [m]

                    if (curTime < time_start && (xc >= gate_x / l + o || yc >= gate_y / l + o) ) {
                      constexpr PREC_G gate_n[3] = {-0.866025403784f, -0.5f, 0.f}; // Gate normal vector, 60 deg slant backwards relative to 30 deg flume (XZ Plane)
                      PREC_G dot = gate_n[0] * vel[0] + gate_n[1] * vel[1] + gate_n[2] * vel[2];
                      if (dot < 0.f) 
                        for (int d=0; d<3; ++d) vel[d] = vel_FLIP[d] = vel[d] - dot * gate_n[d];

                    }
                    else if (curTime >= time_start && curTime < time_end && (xc >= gate_x / l + o || yc >= gate_y / l + o)) {
                      gate_z1 = (gate_z1 / l + o) - (gate_width / l) * ((time_end - time_start) - (time_end - curTime));
                      gate_z2 = (gate_z2 / l + o) + (gate_width / l) * ((time_end - time_start) - (time_end - curTime));
                      if (zc <= gate_z1 || zc >= gate_z2) {
                        constexpr PREC_G gate_n[3] = {-0.866025403784f, -0.5f, 0.f}; // Gate normal vector, 60 deg slant backwards relative to 30 deg flume (XZ Plane)
                        PREC_G dot = gate_n[0] * vel[0] + gate_n[1] * vel[1] + gate_n[2] * vel[2];
                        if (dot < 0.f) 
                          for (int d=0; d<3; ++d) vel[d] = vel_FLIP[d] = vel[d] - dot * gate_n[d];
                      }
                    }
                  }
                }
                else if (gb._object == boundary_object_t::WASIRF_PUMP) {
                  if (gb._contact == boundary_contact_t::Separable) {
                    for (int d=0; d<3; d++) {
                      if (xc >= boundary_pos[0] && xc <= boundary_pos[0] + boundary_dim[0] && 
                          yc >= boundary_pos[1] && yc <= boundary_pos[1] + boundary_dim[1] && 
                          zc >= boundary_pos[2] && zc <= boundary_pos[2] + boundary_dim[2]) {
                        if (abs(vel[d]) < gb._velocity[d] / mass) {
                          vel[d] = gb._velocity[d] / mass; // Set boundary velocity to node
                          // vel_FLIP[d] = gb._velocity[d]; // Set boundary velocity to node
                        }
                      }
                    }
                  }
                }

                boundary_pos[2] += gb._spacing[2];
              }
              boundary_pos[1] += gb._spacing[1];
            }
            boundary_pos[0] += gb._spacing[0];
          }


        }

#if 0   ///< Sticky contact only
        if (isInBound) vel.set(0.f);
#endif

#if 1   ///< Allow all contact types    
        // Set grid-node regular vel. (PIC, yes affine, yes internal forces, yes body forces)
        vel[0]  = isInBound & 4 ? 0.0 : vel[0] * mass; //< vx = mvx / m 
        vel[1]  = isInBound & 2 ? 0.0 : vel[1] * mass; //< vy = mvy / m
        // vel[1] += grav / l * dt;  //< fg = dt * g, Grav. force
        // vel[1] += isInBound & 2 ? 0.f : (grav / l) * dt;  //< fg = dt * g, Grav. force
        vel[2]  = isInBound & 1 ? 0.0 : vel[2] * mass; //< vz = mvz / m

        // Set grid-node internal force (reclaim from MLS-MPM fused vel. + int. force write)
        // PREC fint[3];
        // fint[0] = (vel[0] - vel_FLIP[0]) * mass;
        // fint[1] = (vel[1] - vel_FLIP[1]) * mass;
        // fint[2] = (vel[2] - vel_FLIP[2]) * mass;

        // Set FLIP velocity (yes affine, no internal forces, no body forces)
        if (0) { // Collision for FLIP velocity
          vel_FLIP[0] = isInBound & 4 ? 0.0 : vel_FLIP[0] * mass; //< vx = mvx / m
          vel_FLIP[1] = isInBound & 2 ? 0.0 : vel_FLIP[1] * mass; //< vy = mvy / m
          vel_FLIP[2] = isInBound & 1 ? 0.0 : vel_FLIP[2] * mass; //< vz = mvz / m
        } else { // No collisions for FLIP velocity
          vel_FLIP[0] = vel_FLIP[0] * mass; //< vx = mvx / m
          vel_FLIP[1] = vel_FLIP[1] * mass; //< vy = mvy / m
          vel_FLIP[2] = vel_FLIP[2] * mass; //< vz = mvz / m
        }
        // TODO : Implmenet vol JBar switch and pressure boundary
        //PREC_G vol = isInBound == 7 ? 1.0 : 0.0;
        //PREC_G JBar = isInBound == 7 ? 0.0 : 1.0;
#endif        
        // TODO : Reduce register usage for performance. Put boundaries in shared mem.?
        // Velocity boundaries
        for (int g = 0; g < g_max_grid_boundaries; g++)
        {
          vec7 boundary;
          for (int d = 0; d < g_grid_boundary_attribs; d++) boundary[d] = boundary_array[g][d];
          // TODO: Deprecate naive boundary[6] check, make it use gb object
          if (boundary[6] == -1) continue;

          // Read grid boundary data
          auto gb = gridBoundary_array[g];

          // Check if boundary is active
          if ((curTime < gb._time[0]) || (curTime > gb._time[1])) continue;
          // Set boundary friction values
          PREC_G friction_static = gb._friction_static;
          PREC_G friction_dynamic = gb._friction_dynamic;

          // Set boundaries of scene/flume
          gvec3 boundary_dim, boundary_pos;
          boundary_dim[0] = boundary[3] - boundary[0]; // Length
          boundary_dim[1] = boundary[4] - boundary[1]; // Depth
          boundary_dim[2] = boundary[5] - boundary[2]; // Width
          boundary_pos[0] = boundary[0];
          boundary_pos[1] = boundary[1];
          boundary_pos[2] = boundary[2];
          if (( gb._object == boundary_object_t::BATHYMETRY
              || gb._object == boundary_object_t::OSU_LWF_RAMP 
              || gb._object == boundary_object_t::OSU_TWB_RAMP 
              || gb._object == boundary_object_t::USGS_RAMP
              || gb._object == boundary_object_t::TOKYO_HARBOR 
              || gb._object == boundary_object_t::FLOOR
              || gb._object == boundary_object_t::PLANE)
              && gb._num_bathymetry_points > 0) {
            int num_bathymetry_points = gb._num_bathymetry_points;
            int num_bathymetry_panels = num_bathymetry_points - 1;
            if (num_bathymetry_panels < 1) continue;
            for (int i = 0; i < num_bathymetry_panels; ++i) {
              vec3 ns; //< Ramp boundary surface normal
              // Ramp X bounds
              if (xc < gb._bathymetry_points[i][0]) continue;
              if (xc >= gb._bathymetry_points[i+1][0]) continue;
              PREC_G slope = (gb._bathymetry_points[i+1][1] - gb._bathymetry_points[i][1]) / (gb._bathymetry_points[i+1][0] - gb._bathymetry_points[i][0]);
              PREC_G ys, xo, yo;
              xo = gb._bathymetry_points[i][0];
              yo = gb._bathymetry_points[i][1];
              ys = slope * (xc - xo) + yo;
              
              // Ramp Y bounds
              if (yc > ys + 1.f * g_dx) continue;

              // Ramp normal vector
              // TODO: Check for non ascending bathymetry point x values
              ns.set(0.f);
              if (gb._bathymetry_points[i+1][1] < gb._bathymetry_points[i][1]) {
                ns[0] = gb._bathymetry_points[i][1] - gb._bathymetry_points[i+1][1];
                ns[1] = gb._bathymetry_points[i+1][0] - gb._bathymetry_points[i][0];
              } else {
                ns[0] = gb._bathymetry_points[i][1] - gb._bathymetry_points[i+1][1];
                ns[1] = gb._bathymetry_points[i+1][0] - gb._bathymetry_points[i][0];
              }
              PREC_G ns_norm = sqrt(ns[0] * ns[0] + ns[1] * ns[1]);
              if (ns_norm > 0.f) {
                ns[0] /= ns_norm;
                ns[1] /= ns_norm;
              } else {
                ns[0] = 0.f;
                ns[1] = 1.f; // Default flat panel if poorly defined coordinates, points up (y+)
              }
              PREC_G vdotns = vel[0]*ns[0] + vel[1]*ns[1]; // Projected velocity on normal, negative values mean moving into the boundary, positive values mean moving away from the boundary surface

              // --------- Wen-Chia Yang's disseration, UW 2016, p. 50? ---------
              {
                // f_boundary = -f_internal - f_external - (1/dt)*p
                // Node accel. = (f_internal + f_external + ySf*f_boundary) / mass
                // TODO : Reimplement for all Wen-Chia Yang boundary conditions (e.g. linear, quadratic, sinusoidal, etc.)
                // In throw-away scope for register reuse

                // Boundary thickness and cell distance
                //h  = sqrt((g_dx*g_dx*ns[0]) + (g_dx*g_dx*ns[1]) + (g_dx*g_dx*ns[2]));
                //float r  = sqrt((xc - xs)*(xc - xs) + (yc - ys)*(yc - ys));

                // TODO : Bound. surf. elev. (ys) inaccurate (close for small angles). Can use trig. to calc, but may use too many registers.
                // Calc. decay coef. (ySF) for grid-node pos. Adjusts vel. irregular boundaries.
                // Decay coef: 0 = free, (0-1) = decayed slip/sep., 1 = slip/sep., 2 = rigid
                PREC_G ySF=0.f; // Boundary decay coef. 
                if (yc >= ys + 1.f * g_dx) ySF = 0.f; // Above decay layer
                else if (yc <= ys && yc > (ys - 2.5f*g_dx)) {
                  ySF = (gb._contact == boundary_contact_t::Separable) ? 1.0f 
                        : ((gb._contact == boundary_contact_t::Slip) ? 1.5f 
                        : 2.f);
                  } // Below boundary surface
                else if (yc <= (ys - 2.5f*g_dx)) ySF = 2.f; // Far below bound. surf.
                else {
                  if (1) { 
                    // linear decay
                    ySF = 1.f - (yc - ys) / g_dx ; 
                  } else if (0) { 
                    // quadratic decay
                    ySF = 1.f - (yc - ys) / g_dx; 
                    ySF *= ySF;
                  } else {  
                    // cubic decay
                    ySF = 1.f - (yc - ys) / g_dx; 
                    ySF *= ySF; 
                    ySF *= ySF;
                  }
                }

                // Adjust grid-node velocity via decay layer coef. and surface normal
                if (ySF > 1.5f) // Fix vel. rigidly if deep below boundary surf.
                  for (int d=0; d<3; ++d) vel[d] = 0.f;
                else if (ySF == 1.f) // Separable vel. at boundary surf.
                  if (vdotns < 0.f) for (int d=0; d<3; ++d) vel[d] -= ySF * (vdotns * ns[d]);// ySF*(vel - vdotns*ns)??
                else if (ySF == 1.5f) // Slip vel. below boundary surf.
                  for (int d=0; d<3; ++d) vel[d] -= ySF * (vdotns * ns[d]);// ySF*(vel - vdotns*ns)??
                else // Adjust vel. normal if in decay layer 
                  if (vdotns < 0.f) for (int d=0; d<3; ++d) vel[d] -= ySF * (vdotns * ns[d]);
              
              }
              // Frictional forces
              if (yc <= ys) {
                if ( (vel_FLIP[0] + vel_FLIP[1] + vel_FLIP[2] != 0.f) && (friction_static != 0.f)) {
                  PREC_G normal[3] = {ns[0], ns[1], ns[2]}; // Set normal vector
                  apply_friction_to_grid_velocity(vel, normal, friction_static, friction_dynamic, mass, dt, l, grav.data());
                }
              }
            }
          } else if (gb._object == boundary_object_t::OSU_LWF_RAMP && (gb._num_bathymetry_points == 0)) {
            PREC_G wave_maker_neutral = -2.f * fr_scale; // Wave-maker neutral X pos. [m] at OSU LWF        
            PREC_G ys=0.f, xo=0.f, yo=0.f;
            vec3 ns; //< Ramp boundary surface normal
            ns.set(0.f); ns[1] = 1.f; // Default flat panel, points up (y+)

            // Ramp segment bounds for OSU flume, Used diagram, Dakota Mascarenas Feb. 2021
            if (xc < ((fr_scale*14.275 + 0.0 - wave_maker_neutral)/l)+o) {
              // Flat, 0' elev., 0' - 46'10
              xo = (0.0 - wave_maker_neutral) / l + o;
              yo = (fr_scale*0.225f) / l + o;
              ys = 0.f * (xc-xo) + yo;
            } else if (xc >= ((fr_scale*14.275 + 0.0 - wave_maker_neutral)/l)+o && xc < ((fr_scale*3.65 + fr_scale*14.275 + 0.0 - wave_maker_neutral)/l)+o){
              // Flat (adjustable), 0' elev., 46'10 - 58'10
              xo = (fr_scale*14.275  + 0.0 - wave_maker_neutral) / l + o;
              yo = (fr_scale*0.225) / l + o; // 0.226 rounded down 
              //yo = (0.2)/l + o; // TODO: Reavaluate treatment of flat bathymetry panels, i.e. no decay?
              ys = yo;
            } else if (xc >= ((fr_scale*3.65 + fr_scale*14.275 + 0.0 - wave_maker_neutral)/l)+o && xc < ((fr_scale*10.975 + fr_scale*3.65 + fr_scale*14.275 + 0.0 - wave_maker_neutral)/l)+o) {
              // 1:12, 0' elev., 58'10 - 94'10
              ns[0] = -1.f/11.8648961f; ns[1] = 0.996441901109f; ns[2] = 0.f;
              xo = (fr_scale*3.65 + fr_scale*14.275 + 0.0 - wave_maker_neutral) / l + o;
              yo = (fr_scale*0.225) / l + o;
              ys = 1.f/11.864861f * (xc - xo) + yo;

            } else if (xc >= ((fr_scale*10.975 + fr_scale*3.65 + fr_scale*14.275 + 0.0 - wave_maker_neutral)/l)+o && xc < ((fr_scale*14.625 + fr_scale*10.975 + fr_scale*3.65 + fr_scale*14.275 + 0.0 - wave_maker_neutral)/l)+o) {
              // 1:24, 3' elev., 94'10 - 142'10
              ns[0] = -1.f/24.375f; ns[1] = 0.999158093986f; ns[2] = 0.f;
              xo = (fr_scale*10.975 + fr_scale*3.65 + fr_scale*14.275 + 0.0 - wave_maker_neutral) / l + o;
              yo = (fr_scale*1.15) / l + o; // 1.15 for grid adherence, physically it should be ~1.14ish at a 1:24 slope, modified for cell size of 0.025 m 
              ys = 1.f/24.375f * (xc - xo) + yo;

            } else if (xc >= ((fr_scale*14.625 + fr_scale*10.975 + fr_scale*3.65 + fr_scale*14.275 + 0.0 - wave_maker_neutral)/l)+o && xc < ((fr_scale*36.575 + fr_scale*14.625 + fr_scale*10.975 + fr_scale*3.65 + fr_scale*14.275 + 0.0 - wave_maker_neutral)/l)+o) {
              // Flat, 5' elev., 142'10 - 262'10
              xo = (fr_scale*14.625 + fr_scale*10.975 + fr_scale*3.65 + fr_scale*14.275 + 0.0 - wave_maker_neutral) / l + o;
              yo = (fr_scale*1.75) / l + o;
              ys = yo;

            } else if (xc >= ((fr_scale*36.575 + fr_scale*14.625 + fr_scale*10.975 + fr_scale*3.65 + fr_scale*14.275 + 0.0 - wave_maker_neutral)/l)+o && xc < ((fr_scale*7.35 + fr_scale*36.575 + fr_scale*14.625 + fr_scale*10.975 + fr_scale*3.65 + fr_scale*14.275 + 0.0 - wave_maker_neutral)/l)+o) {
              // 1:12, 5' elev., 262'10 - 286'10
              ns[0] = -1.f/12.25f; ns[1] = 0.996662485475f; ns[2] = 0.f;
              xo = (fr_scale*36.575 + fr_scale*14.625 + fr_scale*10.975 + fr_scale*3.65 + fr_scale*14.275 + 0.0 - wave_maker_neutral) / l + o;
              yo = (fr_scale*1.75) / l + o;
              ys = 1.f/12.25f * (xc - xo) + yo;
            } else {
              // Flat, 7' elev., 286'10 onward
              xo = (fr_scale*36.575 + fr_scale*14.625 + fr_scale*10.975 + fr_scale*3.65 + fr_scale*14.275 + 0.0 - wave_maker_neutral) / l + o; // Maybe change, recheck OSU LWF bathymetry far downstream
              yo = (fr_scale*2.35) / l + o;
              ys = yo;        
            }

            {
              PREC_G normal_inv_magnitude = rsqrt(ns[0]*ns[0] + ns[1]*ns[1] + ns[2]*ns[2]);
              for (int d=0; d<3; d++) ns[d] *= normal_inv_magnitude;
            }
            PREC_G vdotns = vel[0]*ns[0] + vel[1]*ns[1] + vel[2]*ns[2]; // Projected velocity on normal, negative values mean moving into the boundary, positive values mean moving away from the boundary surface

            // --------- Wen-Chia Yang's disseration, UW 2016, p. 50? ---------
            {
              // f_boundary = -f_internal - f_external - (1/dt)*p
              // Node accel. = (f_internal + f_external + ySf*f_boundary) / mass
              // TODO : Reimplement for all Wen-Chia Yang boundary conditions (e.g. linear, quadratic, sinusoidal, etc.)
              // In throw-away scope for register reuse

              // Boundary thickness and cell distance
              //h  = sqrt((g_dx*g_dx*ns[0]) + (g_dx*g_dx*ns[1]) + (g_dx*g_dx*ns[2]));
              //float r  = sqrt((xc - xs)*(xc - xs) + (yc - ys)*(yc - ys));

              // TODO : Bound. surf. elev. (ys) inaccurate (close for small angles). Can use trig. to calc, but may use too many registers.
              // Calc. decay coef. (ySF) for grid-node pos. Adjusts vel. irregular boundaries.
              // Decay coef: 0 = free, (0-1) = decayed slip/sep., 1 = slip/sep., 2 = rigid
              PREC_G ySF=0.f; // Boundary decay coef. 
              if (yc >= ys + 1.f * g_dx) ySF = 0.f; // Above decay layer
              else if (yc <= ys && yc > (ys - 2.5f*g_dx)) {
                ySF = (gb._contact == boundary_contact_t::Separable) ? 1.0f 
                      : ((gb._contact == boundary_contact_t::Slip) ? 1.5f 
                      : 2.f);
                } // Below boundary surface
              else if (yc <= (ys - 2.5f*g_dx)) ySF = 2.f; // Far below bound. surf.
              else {
                if (1) { 
                  // linear decay
                  ySF = 1.f - (yc - ys) / g_dx ; 
                } else if (0) { 
                  // quadratic decay
                  ySF = 1.f - (yc - ys) / g_dx; 
                  ySF *= ySF;
                } else {  
                  // cubic decay
                  ySF = 1.f - (yc - ys) / g_dx; 
                  ySF *= ySF; 
                  ySF *= ySF;
                }
              }

              // Adjust grid-node velocity via decay layer coef. and surface normal
              if (ySF > 1.5f) // Fix vel. rigidly if deep below boundary surf.
                for (int d=0; d<3; ++d) vel[d] = 0.f;
              else if (ySF == 1.f) // Separable vel. at boundary surf.
                if (vdotns < 0.f) for (int d=0; d<3; ++d) vel[d] -= ySF * (vdotns * ns[d]);// ySF*(vel - vdotns*ns)??
              else if (ySF == 1.5f) // Slip vel. below boundary surf.
                for (int d=0; d<3; ++d) vel[d] -= ySF * (vdotns * ns[d]);// ySF*(vel - vdotns*ns)??
              else // Adjust vel. normal if in decay layer 
                if (vdotns < 0.f) for (int d=0; d<3; ++d) vel[d] -= ySF * (vdotns * ns[d]);
            
            }
          } else if (gb._object == boundary_object_t::OSU_LWF_PADDLE) // OSU LWF wave-maker paddle
          {
            // OSU Wave-Maker - CSV Controlled
            //PREC_G wave_maker_neutral = (1.915f); // Streamwise offset from origin (m)
            // TODO: Run-time wave_maker_neutral X position
            PREC_G wave_maker_neutral = gb._domain_end[0] - o; // Streamwise offset from origin (m), already froude scaled
            if (xc <= ((boundary_motion[1]) / l + (wave_maker_neutral)) + o) {
              // TODO: Add reflection and/or decay layer?
              if (gb._contact == boundary_contact_t::Separable) {
                  if (vel[0] < boundary_motion[2] / l) vel[0] = boundary_motion[2] / l; 
              } else if (gb._contact == boundary_contact_t::Slip) {
                  vel[0] = boundary_motion[2] / l; 
              }
              // if (vel[1] < boundary_motion.vy / l) vel[1] = boundary_motion.vy / l;
              // if (vel[2] < boundary_motion.vz / l) vel[2] = boundary_motion.vz / l;
            }
          } else if (gb._object == boundary_object_t::OSU_TWB_PADDLE) // Moveable boundary - OSU LWF wave-maker
          {
            // OSU Wave-Maker - CSV Controlled
            // TODO: Run-time wave_maker_neutral X position
            PREC_G wave_maker_neutral = -0.f * fr_scale; // Streamwise offset from origin (m)
            if (xc <= (boundary_motion[1] - wave_maker_neutral) / l + o) {
              // TODO: Add reflection and/or decay layer?
#if 1 // Slip vel. (YES seperable)
              if (vel[0] < boundary_motion[2] / l) vel[0] = boundary_motion[2] / l; 
              // if (vel[1] < boundary_motion.vy / l) vel[1] = boundary_motion.vy / l;
              // if (vel[2] < boundary_motion.vz / l) vel[2] = boundary_motion.vz / l;
#else // Slip vel. (NO seperable) 
              vel[0] = (boundary_motion[2] / l); // Slip vel. (NO seperable)
#endif
            }
          } else if (gb._object == boundary_object_t::OSU_TWB_RAMP && (gb._num_bathymetry_points == 0)) {
            if (gb._contact == boundary_contact_t::Separable) {
              PREC_G wave_maker_neutral = 0.f * fr_scale; // Wave-maker neutral X pos. [m] at OSU TWB       
              PREC_G ys=0.f, xo=0.f, yo=0.f;
              vec3 ns; //< Ramp boundary surface normal
              ns.set(0.f); ns[1] = 1.f; // Default flat panel, points up (y+)

              // Ramp bathymetry for OSU TWB Flume, H. Park et al. 2021
              if (xc < ((fr_scale*11.3 + 0.0 - wave_maker_neutral)/l)+o) {
                // Flat, 0 elev., 0 - 11.3m (Wave Far-Field)
                xo = (0.0 - wave_maker_neutral) / l + o;
                yo = 0.f / l + o;
                ys = 0.f * (xc-xo) + yo;
              } else if (xc >= ((fr_scale*11.3 + 0.0 - wave_maker_neutral)/l)+o && xc < ((fr_scale*20.0 + fr_scale*11.3 + 0.0 - wave_maker_neutral)/l)+o){
                // 1:20, 0 elev., 11.3 - 31.3 m (Ramp)
                ns[0] = -1.f/20.f; ns[1] = 1.f; ns[2] = 0.f;
                xo = (fr_scale*11.3 + 0.0 - wave_maker_neutral) / l + o;
                yo = 0.f / l + o;
                ys = 1.f/20.f * (xc - xo) + yo;
              } else if (xc >= ((fr_scale*20.0 + fr_scale*11.3 + 0.0 - wave_maker_neutral)/l)+o && xc < ((fr_scale*10.0 + fr_scale*20.0 + fr_scale*11.3 + 0.0 - wave_maker_neutral)/l)+o) {
                // Flat, 1 m elev., 31.3 - 41.3 m (Debris Staging Platform)
                xo = (fr_scale*20.f + fr_scale*11.3 + 0.0 - wave_maker_neutral) / l + o;
                yo = (fr_scale*1.f) / l + o;
                ys = 0.f * (xc - xo) + yo;
              } else {
                // Flat, 0 m elev., 41.3+ m  (Water Overflow Basin)
                xo = (fr_scale*10.0 + fr_scale*20.0 + fr_scale*11.3 + 0.0 - wave_maker_neutral) / l + o;
                yo = 0.f / l + o;
                ys = 0.f * (xc - xo) + yo;
              }

              {
                PREC_G normal_inv_magnitude = rsqrt(ns[0]*ns[0] + ns[1]*ns[1] + ns[2]*ns[2]);
                for (int d=0; d<3; d++) ns[d] *= normal_inv_magnitude;
              }
              PREC_G vdotns = vel[0]*ns[0] + vel[1]*ns[1] + vel[2]*ns[2];

              // --------- Wen-Chia Yang's disseration, UW 2016, p. 50? ---------
              {
                // f_boundary = -f_internal - f_external - (1/dt)*p
                // Node accel. = (f_internal + f_external + ySf*f_boundary) / mass
                // TODO : Reimplement for all Wen-Chia Yang boundary conditions (e.g. linear, quadratic, sinusoidal, etc.)
                // In throw-away scope for register reuse

                // Boundary thickness and cell distance
                //h  = sqrt((g_dx*g_dx*ns[0]) + (g_dx*g_dx*ns[1]) + (g_dx*g_dx*ns[2]));
                //float r  = sqrt((xc - xs)*(xc - xs) + (yc - ys)*(yc - ys));

                // TODO : Bound. surf. elev. (ys) inaccurate (close for small angles). Can use trig. to calc, but may use too many registers.
                // Calc. decay coef. (ySF) for grid-node pos. Adjusts vel. irregular boundaries.
                PREC_G ySF=0.f; // Boundary decay coef. 
                // Decay coef: 0 = free, (0-1) = decayed slip/sep., 1 = slip/sep., 2 = rigid
                if (yc >= ys + 1.f * g_dx) ySF = 0.f; // Above decay layer
                else if (yc <= ys && yc > (ys - 1.5f*g_dx)) ySF = 1.f; // Below boundary surface
                else if (yc <= (ys - 1.5f*g_dx)) ySF = 2.f; // Far below bound. surf.
                else { ySF = (g_dx - (yc - ys)) / g_dx ; ySF *= ySF; } // Decay layer, quadratic

  #pragma unroll 3
                for (int d=0; d<3; ++d) {
                  // Adjust grid-node velocity via decay layer coef. and surface normal
                  if (ySF > 1.5f) // Fix vel. rigidly if deep below boundary surf.
                    vel[d] = 0.f;
                  else if (ySF == 1.5f) // Slip vel. below boundary surf. (NO seperable)
                    vel[d] -= ySF * (vdotns * ns[d]);// ySF*(vel - vdotns*ns)??
                  else if (ySF == 1.f) // Slip vel. below boundary surf. (NO seperable)
                    if (vdotns < 0.f) vel[d] -= ySF * (vdotns * ns[d]);// ySF*(vel - vdotns*ns)??
                  else // Adjust vel. normal if in decay layer (YES seperable)
                    if (vdotns < 0.f) vel[d] -= ySF * (vdotns * ns[d]);
                } 
              }

              // Friction
              PREC_G START_FRICTION_X = 31.3f * fr_scale; // Start friction at X position [m]
              if (xc > START_FRICTION_X / l + o)
              if (yc <= ys) {
                if ( vel_FLIP[0] + vel_FLIP[1] + vel_FLIP[2] != 0.f && friction_static != 0.f) {
                  PREC_G force[3];
                  force[0] = ((vel[0] ) / dt + grav[0] / l) / mass;
                  force[1] = ((vel[1] ) / dt + grav[1] / l) / mass;
                  force[2] = ((vel[2] ) / dt + grav[2] / l) / mass;
                  vdotns = force[0]*ns[0] + force[1]*ns[1] + force[2]*ns[2]; // Norm force
                  if (vdotns < 0.f) {
                    PREC_G tangent_force[3];
                    tangent_force[0] = force[0] - vdotns * ns[0];
                    tangent_force[1] = force[1] - vdotns * ns[1];
                    tangent_force[2] = force[2] - vdotns * ns[2];
                    PREC_G max_friction_force = - vdotns * friction_static;
                    PREC_G friction_force = sqrt(tangent_force[0]*tangent_force[0] + tangent_force[1]*tangent_force[1] + tangent_force[2]*tangent_force[2]);

                    // Check if exceeding static friction
                    if (friction_force > max_friction_force) {
                      PREC_G friction_force_scale = min(max_friction_force / friction_force, 1.f);
                      vel[0] -= tangent_force[0] * friction_force_scale * mass * dt;
                      vel[1] -= tangent_force[1] * friction_force_scale * mass * dt;
                      vel[2] -= tangent_force[2] * friction_force_scale * mass * dt;
                    }
                  }
                }
              }

            }
          } else if (gb._object == boundary_object_t::FLOOR && (gb._num_bathymetry_points == 0)) {
            if ((xc >= gb._domain_start[0]) && (xc <= gb._domain_end[0])) 
            if ((yc >= gb._domain_start[1]) && (yc <= gb._domain_end[1])) 
            if ((zc >= gb._domain_start[2]) && (zc <= gb._domain_end[2])) {
              if (gb._contact == boundary_contact_t::Sticky) {
                // No friction needed
                for (int d=0; d<3; d++) vel[d] = 0.f;
              } else if (gb._contact == boundary_contact_t::Slip) {
                // Apply friction transversely, zero out normal velocity
                PREC_G ns[3] = {0.f, 1.f, 0.f}; //< Floor boundary surface normal
                if ( (vel_FLIP[0] + vel_FLIP[1] + vel_FLIP[2] != 0.f) && (friction_static != 0.f)) {
                  apply_friction_to_grid_velocity(vel, ns, friction_static, friction_dynamic, mass, dt, l, grav.data());
                }
                vel[1] = 0.f;
              } else if (gb._contact == boundary_contact_t::Separable) {
                // Apply friction transversely, zero out normal velocity pointing into floor
                PREC_G ns[3] = {0.f, 1.f, 0.f}; //< Floor boundary surface normal
                // if ( (vel_FLIP[0] + vel_FLIP[1] + vel_FLIP[2] != 0.f) && (friction_static != 0.f)) {
                //   apply_friction_to_grid_velocity(vel, ns, friction_static, friction_dynamic, mass, dt, l, grav.data());
                // }
                if (friction_static != 0.f) {
                  apply_friction_to_grid_velocity(vel, ns, friction_static, friction_dynamic, mass, dt, l, grav.data());
                }
                if (vel[1] < 0.f) vel[1] = 0.f;
              }
            }
          } else if (gb._object == boundary_object_t::TOKYO_HARBOR && (gb._num_bathymetry_points == 0)) {
            if (gb._contact == boundary_contact_t::Separable) {
              PREC_G ns[3] = {0.f, 1.f, 0.f}; //< Ramp surface normal, flat panel, points up (y+)
              PREC_G START_HARBOR_X = 4.45f * fr_scale; // Start harbor at X position [m]
              PREC_G START_HARBOR_Y = 0.255f * fr_scale; // Start harbor at Y position [m]

              // Friction
              PREC_G DIST_DEBRIS_TO_HARBOR_X = 0.20f;
              PREC_G START_FRICTION_X = START_HARBOR_X + DIST_DEBRIS_TO_HARBOR_X * fr_scale; // Start friction at debris X position [m]
              if (xc >= START_FRICTION_X/l+o && yc <= START_HARBOR_Y/l+o) {
                // Only apply friction if an FLIP/ASFLIP particle is in grid-node's domain
                // Useful for localizing friction to debris, etc. (e.g. not apply friction to water)
                if ( (vel_FLIP[0] + vel_FLIP[1] + vel_FLIP[2] != 0.f) && (friction_static != 0.f) ) {
                  apply_friction_to_grid_velocity(vel, ns, friction_static, friction_dynamic, mass, dt, l, grav.data());
                }
              }

              // If past quay wall i.e. harbor face and moving forward
              if (((xc >= START_HARBOR_X/l+o && xc <= START_HARBOR_X/l+o + g_dx) && yc <= START_HARBOR_Y / l + o - g_dx) && vel[0] > 0.f){
                vel[0] = 0.f; // Remove forward velocity
              } 
              // If below harbor floor and moving down
              if ((xc > START_HARBOR_X/l+o && yc <= START_HARBOR_Y/l+o) && vel[1] < 0.f){
                vel[1] = 0.f; // Remove downward velocity
              } 

            }
          } else if (gb._object == boundary_object_t::Plane && (gb._num_bathymetry_points == 0)) {
            if (1) {
              PREC_G X_GATE = 0.f * fr_scale; // Gate X position [m]

              PREC_G ys = 0.f, xo = 0.f, yo = 0.f;
              vec3 ns;
              ns.set(0.f); ns[1] = 1.f;
              
              if ((xc >= (boundary_pos[0]) ) && (xc < ( (boundary_dim[0] + boundary_pos[0])  ) ) ) {
                ns[0] = -boundary_dim[1] / sqrt(boundary_dim[0]* boundary_dim[0] + boundary_dim[1]*boundary_dim[1]); 
                ns[1] = boundary_dim[0] / sqrt(boundary_dim[0]*boundary_dim[0] + boundary_dim[1]*boundary_dim[1]); 
                ns[2] = 0.f;
                xo = boundary_pos[0] ;
                yo = boundary_pos[1] ;
                ys = ((boundary_dim[1]) / (boundary_dim[0])) * (xc - xo) + yo;

              
              }
              else {
                // Flat, 0 m elev., 41.3+ m  (Water Overflow Basin)
                ns[0] = -1.f/1.732f; ns[1] = 1.f; ns[2] = 0.f;
                xo = (fr_scale*2.5 + fr_scale*80.0 + X_GATE) / l + o;
                yo = (fr_scale*1.25f/2.747f + fr_scale*1.25f/5.675f) / l + o;
                ys = (1.f/1.732) * (xc - xo) + yo;
              }

              {
                PREC_G normal_inv_magnitude = rsqrt(ns[0]*ns[0] + ns[1]*ns[1] + ns[2]*ns[2]);
                for (int d=0; d<3; d++) ns[d] *= normal_inv_magnitude;
              }

              PREC_G vdotns = vel[0]*ns[0] + vel[1]*ns[1] + vel[2]*ns[2];
               
// --------- Wen-Chia Yang's disseration, UW 2016, p. 50? ---------
              PREC_G ySF=0.f; // Boundary decay coef. 
              {
                // f_boundary = -f_internal - f_external - (1/dt)*p
                // Node accel. = (f_internal + f_external + ySf*f_boundary) / mass
                // TODO : Reimplement for all Wen-Chia Yang boundary conditions (e.g. linear, quadratic, sinusoidal, etc.)

                // Boundary thickness and cell distance
                //h  = sqrt((g_dx*g_dx*ns[0]) + (g_dx*g_dx*ns[1]) + (g_dx*g_dx*ns[2]));
                //float r  = sqrt((xc - xs)*(xc - xs) + (yc - ys)*(yc - ys));

                // TODO : Bound. surf. elev. (ys) inaccurate (close for small angles). Can use trig. to calc, but may use too many registers.
                // Calc. decay coef. (ySF) for grid-node pos. Adjusts vel. irregular boundaries.
                // Decay coef: 0 = free, (0-1) = decayed slip/sep., 1 = slip/sep., 2 = rigid
                if (yc >= ys + 1.f * g_dx) ySF = 0.f; // Above decay layer always "free" from BC
                else if (yc <= ys && yc > (ys - 1.5f*g_dx)) { // On and below boundary surface
                  if (gb._contact == boundary_contact_t::Separable) ySF = 1.f; // Separable
                  else if (gb._contact == boundary_contact_t::Slip) ySF = 1.5f; // Slip
                  else ySF = 2.f; // Rigid
                }
                else if (yc <= (ys - 1.5f*g_dx)) ySF = 2.f; // Far below bound. surf. always rigid
                else { ySF = (g_dx - (yc - ys)) / g_dx ; ySF *= ySF; } // Decay layer, quadratic

  #pragma unroll 3
                for (int d=0; d<3; ++d) {
                  // Adjust grid-node velocity via decay layer coef. and surface normal
                  if (ySF > 1.f) // Fix vel. rigidly if deep below boundary surf.
                    vel[d] = 0.f;
                  else if (ySF == 1.f) // Separable vel. below boundary surf. 
                    if (vdotns < 0.f) vel[d] -= ySF * (vdotns * ns[d]);// ySF*(vel - vdotns*ns)??
                  else if (ySF == 1.5f) // Slip vel. below boundary surf. 
                    vel[d] -= ySF * (vdotns * ns[d]);
                  else // Adjust vel. normal if in decay layer (YES seperable)
                    if (vdotns < 0.f) vel[d] -= ySF * (vdotns * ns[d]);
                } 
              }

              // Friction
              //constexpr PREC_G FRICTION_OF_SAND_ON_CONCRETE = 2.0f;
              //PREC_G X_BEGIN_BUMPY_TILES = 6.f * fr_scale + X_GATE; // Bumpy flume tiles go from 6 - 79 m;
              //PREC_G X_END_BUMPY_TILES = 79.f * fr_scale + X_GATE;
              //if (xc < X_BEGIN_BUMPY_TILES / l + o || xc > X_END_BUMPY_TILES / l + o)
              //  friction_static = FRICTION_OF_SAND_ON_CONCRETE; 
              if ( ySF > 0.f && ySF <= 1.f) {
                if ( vel_FLIP[0] + vel_FLIP[1] + vel_FLIP[2] != 0.f && friction_static != 0.f) {
                  PREC_G force[3];
                  force[0] = ((vel[0] ) / dt + grav[0] / l) / mass;
                  force[1] = ((vel[1] ) / dt + grav[1] / l) / mass;
                  force[2] = ((vel[2] ) / dt + grav[2] / l) / mass;
                  vdotns = force[0]*ns[0] + force[1]*ns[1] + force[2]*ns[2]; // Norm force
                  if (vdotns < 0.f) {
                    PREC_G tangent_force[3];
                    tangent_force[0] = force[0] - vdotns * ns[0];
                    tangent_force[1] = force[1] - vdotns * ns[1];
                    tangent_force[2] = force[2] - vdotns * ns[2];
                    PREC_G max_friction_force = - vdotns * friction_static;
                    PREC_G friction_force = sqrt(tangent_force[0]*tangent_force[0] + tangent_force[1]*tangent_force[1] + tangent_force[2]*tangent_force[2]);

                    // Check if exceeding static friction
                    if (friction_force > max_friction_force) {
                      PREC_G friction_force_scale = min(max_friction_force / friction_force, 1.f);
                      vel[0] -= ySF * tangent_force[0] * friction_force_scale * mass * dt;
                      vel[1] -= ySF * tangent_force[1] * friction_force_scale * mass * dt;
                      vel[2] -= ySF * tangent_force[2] * friction_force_scale * mass * dt;
                    } else {
                      vel[0] -= ySF * tangent_force[0] * mass * dt;
                      vel[1] -= ySF * tangent_force[1] * mass * dt;
                      vel[2] -= ySF * tangent_force[2] * mass * dt;
                    }
                  }
                }
              }
              

            }


          } else if (gb._object == boundary_object_t::USGS_RAMP && (gb._num_bathymetry_points == 0)) {
            if (gb._contact == boundary_contact_t::Separable) {
              PREC_G X_GATE = 0.f * fr_scale; // Gate X position [m]

              PREC_G ys=0.f, xo=0.f, yo=0.f;
              vec3 ns; //< Ramp boundary surface normal
              ns.set(0.f); ns[1] = 1.f; // Default flat panel, points up (y+)

              // Ramp bathymetry for OSU TWB Flume, H. Park et al. 2021
              if (xc < ((fr_scale*80.0 + X_GATE)/l)+o) {
                // Flat, 0 elev., 0 - 11.3m (Wave Far-Field)
                xo = (0.0) / l + o;
                yo = 0.f / l + o;
                ys = 0.f * (xc-xo) + yo;
              } else if (xc >= ((fr_scale*80.0 + X_GATE)/l)+o && xc < ((fr_scale*1.25 + fr_scale*80.0 + X_GATE)/l)+o){
                // 1:20, 0 elev., 11.3 - 31.3 m (Ramp)
                ns[0] = -1.f/5.675f; ns[1] = 1.f; ns[2] = 0.f;
                xo = (fr_scale*80.f + X_GATE) / l + o;
                yo = 0.f / l + o;
                ys = (1.f/5.675f) * (xc - xo) + yo;
              } else if (xc >= ((fr_scale*1.25 + fr_scale*80.0 + X_GATE)/l)+o && xc < ((fr_scale*2.5 + fr_scale*80.0 + X_GATE)/l)+o) {
                // Flat, 1 m elev., 31.3 - 41.3 m (Debris Staging Platform)
                ns[0] = -1.f/2.747f; ns[1] = 1.f; ns[2] = 0.f;
                xo = (fr_scale*1.25f + fr_scale*80.0 + X_GATE) / l + o;
                yo = (1.25f/5.675f) / l + o;
                ys = (1.f/2.747f) * (xc - xo) + yo;
              } else {
                // Flat, 0 m elev., 41.3+ m  (Water Overflow Basin)
                ns[0] = -1.f/1.732f; ns[1] = 1.f; ns[2] = 0.f;
                xo = (fr_scale*2.5 + fr_scale*80.0 + X_GATE) / l + o;
                yo = (fr_scale*1.25f/2.747f + fr_scale*1.25f/5.675f) / l + o;
                ys = (1.f/1.732) * (xc - xo) + yo;
              }

              {
                PREC_G normal_inv_magnitude = rsqrt(ns[0]*ns[0] + ns[1]*ns[1] + ns[2]*ns[2]);
                for (int d=0; d<3; d++) ns[d] *= normal_inv_magnitude;
              }
              PREC_G vdotns = vel[0]*ns[0] + vel[1]*ns[1] + vel[2]*ns[2];

              // --------- Wen-Chia Yang's disseration, UW 2016, p. 50? ---------
              PREC_G ySF=0.f; // Boundary decay coef. 
              {
                // f_boundary = -f_internal - f_external - (1/dt)*p
                // Node accel. = (f_internal + f_external + ySf*f_boundary) / mass
                // TODO : Reimplement for all Wen-Chia Yang boundary conditions (e.g. linear, quadratic, sinusoidal, etc.)

                // Boundary thickness and cell distance
                //h  = sqrt((g_dx*g_dx*ns[0]) + (g_dx*g_dx*ns[1]) + (g_dx*g_dx*ns[2]));
                //float r  = sqrt((xc - xs)*(xc - xs) + (yc - ys)*(yc - ys));

                // TODO : Bound. surf. elev. (ys) inaccurate (close for small angles). Can use trig. to calc, but may use too many registers.
                // Calc. decay coef. (ySF) for grid-node pos. Adjusts vel. irregular boundaries.
                // Decay coef: 0 = free, (0-1) = decayed slip/sep., 1 = slip/sep., 2 = rigid
                if (yc >= ys + 1.f * g_dx) ySF = 0.f; // Above decay layer always "free" from BC
                else if (yc <= ys && yc > (ys - 1.5f*g_dx)) { // On and below boundary surface
                  if (gb._contact == boundary_contact_t::Separable) ySF = 1.f; // Separable
                  else if (gb._contact == boundary_contact_t::Slip) ySF = 1.5f; // Slip
                  else ySF = 2.f; // Rigid
                }
                else if (yc <= (ys - 1.5f*g_dx)) ySF = 2.f; // Far below bound. surf. always rigid
                else { ySF = (g_dx - (yc - ys)) / g_dx ; ySF *= ySF; } // Decay layer, quadratic

  #pragma unroll 3
                for (int d=0; d<3; ++d) {
                  // Adjust grid-node velocity via decay layer coef. and surface normal
                  if (ySF > 1.f) // Fix vel. rigidly if deep below boundary surf.
                    vel[d] = 0.f;
                  else if (ySF == 1.f) // Separable vel. below boundary surf. 
                    if (vdotns < 0.f) vel[d] -= ySF * (vdotns * ns[d]);// ySF*(vel - vdotns*ns)??
                  else if (ySF == 1.5f) // Slip vel. below boundary surf. 
                    vel[d] -= ySF * (vdotns * ns[d]);
                  else // Adjust vel. normal if in decay layer (YES seperable)
                    if (vdotns < 0.f) vel[d] -= ySF * (vdotns * ns[d]);
                } 
              }

              // Friction
              constexpr PREC_G FRICTION_OF_SAND_ON_CONCRETE = 2.0f;
              PREC_G X_BEGIN_BUMPY_TILES = 6.f * fr_scale + X_GATE; // Bumpy flume tiles go from 6 - 79 m;
              PREC_G X_END_BUMPY_TILES = 79.f * fr_scale + X_GATE;
              if (xc < X_BEGIN_BUMPY_TILES / l + o || xc > X_END_BUMPY_TILES / l + o)
                friction_static = FRICTION_OF_SAND_ON_CONCRETE; 
              if ( ySF > 0.f && ySF <= 1.f) {
                if ( vel_FLIP[0] + vel_FLIP[1] + vel_FLIP[2] != 0.f && friction_static != 0.f) {
                  PREC_G force[3];
                  force[0] = ((vel[0] ) / dt + grav[0] / l) / mass;
                  force[1] = ((vel[1] ) / dt + grav[1] / l) / mass;
                  force[2] = ((vel[2] ) / dt + grav[2] / l) / mass;
                  vdotns = force[0]*ns[0] + force[1]*ns[1] + force[2]*ns[2]; // Norm force
                  if (vdotns < 0.f) {
                    PREC_G tangent_force[3];
                    tangent_force[0] = force[0] - vdotns * ns[0];
                    tangent_force[1] = force[1] - vdotns * ns[1];
                    tangent_force[2] = force[2] - vdotns * ns[2];
                    PREC_G max_friction_force = - vdotns * friction_static;
                    PREC_G friction_force = sqrt(tangent_force[0]*tangent_force[0] + tangent_force[1]*tangent_force[1] + tangent_force[2]*tangent_force[2]);

                    // Check if exceeding static friction
                    if (friction_force > max_friction_force) {
                      PREC_G friction_force_scale = min(max_friction_force / friction_force, 1.f);
                      vel[0] -= ySF * tangent_force[0] * friction_force_scale * mass * dt;
                      vel[1] -= ySF * tangent_force[1] * friction_force_scale * mass * dt;
                      vel[2] -= ySF * tangent_force[2] * friction_force_scale * mass * dt;
                    } else {
                      vel[0] -= ySF * tangent_force[0] * mass * dt;
                      vel[1] -= ySF * tangent_force[1] * mass * dt;
                      vel[2] -= ySF * tangent_force[2] * mass * dt;
                    }
                  }
                }
              }
            }
          } else if (gb._object == boundary_object_t::VELOCITY_BOUNDARY) {
            
            // Code for non-moving, untimed velocity boundary
            if ((xc >= gb._domain_start[0]) && (xc <= gb._domain_end[0]))
            if ((yc >= gb._domain_start[1]) && (yc <= gb._domain_end[1]))
            if ((zc >= gb._domain_start[2]) && (zc <= gb._domain_end[2])) {
              for (int d=0; d<3; d++) vel[d] = gb._velocity[d];
            }

            // Old code for a moving, time velocity boundary            
            // // * It is assumed that the velocity boundary has normal equivalent to unit vector in the direction of the velocity
            // if ((xc >= gb._domain_start[0] + gb._velocity[0]*(curTime)) && (xc <= gb._domain_end[0] + gb._velocity[0]*(curTime))) 
            // if ((yc >= gb._domain_start[1] + gb._velocity[1]*(curTime)) && (yc <= gb._domain_end[1] + gb._velocity[1]*(curTime))) 
            // if ((zc >= gb._domain_start[2] + gb._velocity[2]*(curTime)) && (zc <= gb._domain_end[2] + gb._velocity[2]*(curTime))) {

            //   // * Velocity boundary is active
            //   // * Velocity boundary is assumed to be a box
            //   // * V
            //   // TODO FIX THIS TO NOT JUST WORK IN X DiRECTION
            //   // if (gb._contact == boundary_contact_t::Sticky) {
            //   // for (int d=0; d<3; d++) vel[d] = gb._velocity[d];
            //   constexpr PREC_G TIME_LIMIT = 0.2f; // Time limit for velocity boundary, [sec]
            //   if (curTime < TIME_LIMIT) {
            //     if (vel[0] < gb._velocity[0]) vel[0] = gb._velocity[0];
            //   }
            

            // else if (gb._contact == boundary_contact_t::Slip) {
            //   PREC_G unit_vel[3];
            //   {
            //     PREC_G vel_mag = sqrt(gb._velocity[0]*gb._velocity[0] + gb._velocity[1]*gb._velocity[1] + gb._velocity[2]*gb._velocity[2]);
            //     for (int d=0; d<3; d++) unit_vel[d] = gb._velocity[d] / vel_mag;
            //   }
            //   {
            //     PREC_G rel_vel[3];
            //     for (int d=0; d<3; d++) rel_vel[d] = gb._velocity[d] - vel[d];
            //     PREC_G rel_vel_mag = sqrt(abs(rel_vel[0]*unit_vel[0] + rel_vel[1]*unit_vel[1] + rel_vel[2]*unit_vel[2]));
            //     for (int d=0; d<3; d++) {
            //       // ignore transverse velocity
            //       // adjust normal velocity with non-separable condition
            //       vel[d] += unit_vel[d] * rel_vel_mag;
  
            //     }
            //   }
            // } else if (gb._contact == boundary_contact_t::Separable){
            //   PREC_G unit_vel[3];
            //   PREC_G vel_mag = sqrt(gb._velocity[0]*gb._velocity[0] + gb._velocity[1]*gb._velocity[1] + gb._velocity[2]*gb._velocity[2]);
            //   {
            //     for (int d=0; d<3; d++) unit_vel[d] = gb._velocity[d] / vel_mag;
            //   }
            //   PREC_G vdotns = vel[0]*unit_vel[0] + vel[1]*unit_vel[1] + vel[2]*unit_vel[2];
            //   bool apply_boundary_velocity;
            //   {
            //     PREC_G grid_vel_mag = sqrt(abs(vdotns));
            //     bool is_inward = vdotns < 0.f;
            //     bool is_outward_and_faster = (!is_inward) && (grid_vel_mag < vel_mag);
            //     apply_boundary_velocity = is_inward || is_outward_and_faster;
            //   }
            //   if (apply_boundary_velocity) {
            //     PREC_G rel_vel[3];
            //     for (int d=0; d<3; d++) rel_vel[d] = gb._velocity[d] - vel[d];
            //     PREC_G rel_vel_mag = sqrt(abs(rel_vel[0]*unit_vel[0] + rel_vel[1]*unit_vel[1] + rel_vel[2]*unit_vel[2]));
            //     for (int d=0; d<3; d++) {
            //       // ignore transverse velocity
            //       // adjust normal velocity with separable condition
            //         vel[d] += unit_vel[d] * rel_vel_mag;
            //     }
            //   }

            // }
          }
        } //< End boundaries

        // * External body forces applied (i.e. gravity)
        // ! May need to be moved after all boundary conditions, or before initial update
#if 0
        vel[1] += grav / l * dt;  //< fg = dt * g, Grav. force
#else 
        for (int d=0; d<3; d++) vel[d] += grav[d] / l * dt; // 
#endif


        velSqr = 0.f;
        velSqr += vel[0] * vel[0] + vel[1] * vel[1] + vel[2] * vel[2];
        grid_block.val_1d(_1, cidib) = vel[0];
        grid_block.val_1d(_2, cidib) = vel[1];
        grid_block.val_1d(_3, cidib) = vel[2];
        grid_block.val_1d(_4, cidib) = vel_FLIP[0];
        grid_block.val_1d(_5, cidib) = vel_FLIP[1];
        grid_block.val_1d(_6, cidib) = vel_FLIP[2];
        grid_block.val_1d(_7, cidib) = voln;
        grid_block.val_1d(_8, cidib) = sJBar;
#if DEBUG_COUPLED_UP
        grid_block.val_1d(_9, cidib) = mass_water;
        grid_block.val_1d(_10, cidib) = pressure_water;
#endif

      }
      // Max velSqr in warp saved to first core's register in warp (threadIdx.x % 32 == 0)
      for (int iter = 1; iter % 32; iter <<= 1) 
      {
        float tmp = __shfl_down_sync(activeMask, velSqr, iter); //,32)
        if ((threadIdx.x % 32) + iter < 32)
          velSqr = tmp > velSqr ? (PREC_G) tmp : (PREC_G) velSqr;
      }
      // Put max velSqr of warp into shared memory array (1 element per warp)
      if ((threadIdx.x % 32) == 0)
        if(velSqr > sh_maxvels[threadIdx.x / 32])
          sh_maxvels[threadIdx.x / 32] = (PREC_G) velSqr;
    }
  }
  __syncthreads();
  /// Various assumptions about GPU's warps, block sizein kernel, etc
  // Max value in shared memory for block saved to thread 0
  for (int interval = numWarps >> 1; interval > 0; interval >>= 1) 
  {
    if (threadIdx.x < interval) 
      if (sh_maxvels[threadIdx.x + interval] > sh_maxvels[threadIdx.x])
          sh_maxvels[threadIdx.x] = sh_maxvels[threadIdx.x + interval];
    __syncthreads();
  }
  // Max velSqr from all blocks saved to global memeory maxVel
  if (threadIdx.x == 0)
    atomicMax(maxVel, sh_maxvels[0]);
}

template <typename Grid, typename Partition, typename Boundary>
__global__ void update_grid_velocity_query_max(uint32_t blockCount, Grid grid,
                                               Partition partition, double dt,
                                               Boundary boundary,
                                               float *maxVel, double curTime,
                                               PREC grav) {
  constexpr int bc = 2;
  constexpr int numWarps =
      g_num_grid_blocks_per_cuda_block * g_num_warps_per_grid_block;
  constexpr unsigned activeMask = 0xffffffff;
  //__shared__ float sh_maxvels[g_blockvolume * g_num_grid_blocks_per_cuda_block
  /// 32];
  extern __shared__ float sh_maxvels[];
  if (threadIdx.x < numWarps) sh_maxvels[threadIdx.x] = 0.f;
  __syncthreads();

  std::size_t blockno = blockIdx.x * g_num_grid_blocks_per_cuda_block +
                        threadIdx.x / 32 / g_num_warps_per_grid_block;
  if (blockno < blockCount) 
  {

    auto blockid = partition._activeKeys[blockno];
    int isInBound = ((blockid[0] < bc || blockid[0] >= g_grid_size_x - bc) << 2) |
                    ((blockid[1] < bc || blockid[1] >= g_grid_size_y - bc) << 1) |
                     (blockid[2] < bc || blockid[2] >= g_grid_size_z - bc);

    /// within-warp computations
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
                                               Partition partition, double dt,
                                                double curTime) 
{
  auto grid_block = grid.ch(_0, blockIdx.x);
  for (int cidib = threadIdx.x; cidib < g_blockvolume; cidib += blockDim.x) 
  {
    // Vol, JBar [Simple FBar]
    PREC_G vol  = grid_block.val_1d(_7, cidib);
    if (vol > 0)
    {
      PREC_G JBar = grid_block.val_1d(_8, cidib);
      grid_block.val_1d(_7, cidib) = 0.f;
      grid_block.val_1d(_8, cidib) = JBar / vol;
    }
  }
}



template <typename Grid, typename Partition>
__global__ void query_energy_grid(uint32_t blockCount, Grid grid, Partition partition, double dt, 
                                  PREC_G *sumKinetic, PREC_G *sumGravity, 
                                  double curTime, pvec3 grav, 
                                  vec<vec7, g_max_grid_boundaries> boundary_array, 
                                  vec3 boundary_motion, PREC length) {
  constexpr int numWarps =
      g_num_grid_blocks_per_cuda_block * g_num_warps_per_grid_block;
  constexpr unsigned activeMask = 0xffffffff;
                                  
  extern __shared__ PREC_G sh_energys[];
  if (threadIdx.x < numWarps) sh_energys[threadIdx.x] = 0.0f;
  __syncthreads();
  
  std::size_t blockno = blockIdx.x * g_num_grid_blocks_per_cuda_block +
                        threadIdx.x / 32 / g_num_warps_per_grid_block;

  // PREC_G o = g_offset; // Domain offset [ ], for Quad. B-Splines (Off-by-2, Wang 2020)
  PREC_G l = length; // Length of domain [m]

  // Kinetic energy determination, within-warp computations for efficiency
  if (blockno < blockCount) {
    //auto blockid = partition._activeKeys[blockno];
    auto grid_block = grid.ch(_0, blockno);
    for (int cidib = threadIdx.x % 32; cidib < g_blockvolume; cidib += 32) {
      PREC_G mass = grid_block.val_1d(_0, cidib);
      PREC_G kinetic_energy = 0.0;
      PREC_G velSqr = 0.0;
      PREC_G vel[3];
      if (mass > 0) 
      {
        // Retrieve grid momentums (kg*m/s2)
        vel[0] = grid_block.val_1d(_4, cidib) * l; //< vx + dt*(fint + g)/mass
        vel[1] = grid_block.val_1d(_5, cidib) * l; //< vy + dt*(fint + g)/mass
        vel[2] = grid_block.val_1d(_6, cidib) * l; //< vz + dt*(fint + g)/mass
        for (int d=0; d<3; ++d) velSqr += vel[d] * vel[d]; // Vel. magn.^2. Sqrt on host->save registers.
        kinetic_energy += (0.5 * mass) * velSqr; // E_kinetic = 1/2 * m * v^2
      }
      for (int iter = 1; iter % 32; iter <<= 1) {
        PREC_G tmp = __shfl_down_sync(activeMask, kinetic_energy, iter, 32);
        if ((threadIdx.x % 32) + iter < 32) kinetic_energy += tmp;
      }
      if ((threadIdx.x % 32) == 0) sh_energys[threadIdx.x / 32] += (PREC_G)kinetic_energy;
    }
  }
  __syncthreads();

  // Sum kinetic energy from warps for the CUDA block in shared memory, various assumptions
  for (int interval = numWarps >> 1; interval > 0; interval >>= 1)  {
    if (threadIdx.x < interval) 
        sh_energys[threadIdx.x] += sh_energys[threadIdx.x + interval];
    __syncthreads();
  }
  // Sum kinetic energy from CUDA block for CUDA GRID into global memory
  if (threadIdx.x == 0) {
    atomicAdd(sumKinetic, (PREC_G)sh_energys[0]);
  }
  __syncthreads();

  // Zero-out shared memory for gravity energy per warp in CUDA block
  if (threadIdx.x < numWarps) sh_energys[threadIdx.x] = static_cast<PREC_G>(0.0);
  __syncthreads();

  // Gravity energy determination, within-warp computations for efficiency
  if (blockno < blockCount) 
  {
    //auto blockid = partition._activeKeys[blockno];
    auto grid_block = grid.ch(_0, blockno);
    for (int cidib = threadIdx.x % 32; cidib < g_blockvolume; cidib += 32) 
    {
      PREC_G mass = grid_block.val_1d(_0, cidib);
      PREC_G gravity_energy = 0.0;
      if (mass > 0) 
      {
        constexpr PREC_G elevation_datum[3] = {0.f, 0.f, 0.f}; // Gravity potential datum, all zero for now [m]
        PREC_G i = (cidib >> (g_blockbits << 1)) & g_blockmask;
        PREC_G j = (cidib >> g_blockbits) & g_blockmask;
        PREC_G k = (cidib & g_blockmask);
        PREC_G position[3] = { (i - g_offset) * l , (j - g_offset) * l , (k - g_offset) * l }; // Grid-node position [m]
        for (int d=0; d<3; ++d) {
          gravity_energy += mass * grav[d] * (position[d] - elevation_datum[d]); // E_gravity = mgh
        }
      }
      for (int iter = 1; iter % 32; iter <<= 1) 
      {
        PREC_G tmp = __shfl_down_sync(activeMask, gravity_energy, iter, 32);
        if ((threadIdx.x % 32) + iter < 32) gravity_energy += tmp;
      }
      if ((threadIdx.x % 32) == 0) sh_energys[threadIdx.x / 32] += (PREC_G)gravity_energy;
    }
  }
  __syncthreads();
  /// various assumptions
  for (int interval = numWarps >> 1; interval > 0; interval >>= 1) 
  {
    if (threadIdx.x < interval) 
        sh_energys[threadIdx.x] += sh_energys[threadIdx.x + interval];
    __syncthreads();
  }
  if (threadIdx.x == 0)
  {
    atomicAdd(sumGravity, (PREC_G)sh_energys[0]);
  }

}



template <typename Partition, typename ParticleBuffer>
__global__ void query_energy_particles(Partition partition, Partition prev_partition,
                              ParticleBuffer pbuffer,
                              PREC_G *kinetic_energy, PREC_G *gravity_energy, PREC_G *strain_energy, PREC grav)  { 
                                
                                if(threadIdx.x + blockIdx.x*blockDim.x == 0) printf("ERROR: GPU function query_energy_particles() kernel not implemented for this material + algorithm yet.\n");
                               }
//   int pcnt = g_buckets_on_particle_buffer ? pbuffer._ppbs[blockIdx.x] : partition._ppbs[blockIdx.x];
//   ivec3 blockid = partition._activeKeys[blockIdx.x];
//   auto advection_bucket = g_buckets_on_particle_buffer
//       ? pbuffer._blockbuckets + blockIdx.x * g_particle_num_per_block
//       : partition._blockbuckets + blockIdx.x * g_particle_num_per_block;
//   // auto particle_offset = partition._binsts[blockIdx.x];
//   PREC_G thread_kinetic_energy = (PREC_G)0;
//   PREC_G thread_gravity_energy = (PREC_G)0;
//   PREC_G thread_strain_energy = (PREC_G)0;
//   PREC o = g_offset;
//   PREC l = pbuffer.length;
//   for (int pidib = threadIdx.x; pidib < pcnt; pidib += blockDim.x) 
//   {
//     auto advect = advection_bucket[pidib];
//     ivec3 source_blockid;
//     dir_components(advect / g_particle_num_per_block, source_blockid);
//     source_blockid += blockid;
//     auto source_blockno = prev_partition.query(source_blockid);
//     auto source_pidib = advect % g_particle_num_per_block;
//     auto source_bin = g_buckets_on_particle_buffer
//           ? pbuffer.ch(_0, pbuffer._binsts[source_blockno] + source_pidib / g_bin_capacity)
//           : pbuffer.ch(_0, prev_partition._binsts[source_blockno] + source_pidib / g_bin_capacity);
//     auto _source_pidib = source_pidib % g_bin_capacity;
//     auto _source_bin = g_buckets_on_particle_buffer ? pbuffer._binsts[source_blockno] + source_pidib / g_bin_capacity : prev_partition._binsts[source_blockno] + source_pidib / g_bin_capacity;
//     PREC elevation = (source_bin.val(_1, _source_pidib) - o) * l;
//     PREC particle_kinetic_energy = 0;
//     PREC particle_gravity_energy = pbuffer.mass * (grav / l) * elevation; //< E_gravity = mgh
//     PREC particle_strain_energy;
//     pbuffer.getEnergy_Strain(_source_bin, _source_pidib, pbuffer.volume, particle_strain_energy);
//     thread_strain_energy  += (PREC_G)particle_strain_energy;
//     thread_gravity_energy += (PREC_G)particle_gravity_energy;
//     thread_kinetic_energy += (PREC_G)particle_kinetic_energy;
//   }

//   __syncthreads();
//   atomicAdd(kinetic_energy, thread_kinetic_energy);
//   atomicAdd(gravity_energy, thread_gravity_energy);
//   atomicAdd(strain_energy, thread_strain_energy);
// }


template <typename Partition>
__global__ void query_energy_particles(Partition partition, Partition prev_partition,
                              ParticleBuffer<material_e::JFluid> pbuffer,
                              PREC_G *kinetic_energy, PREC_G *gravity_energy, PREC_G *strain_energy, PREC grav)  {
  int pcnt = g_buckets_on_particle_buffer ? pbuffer._ppbs[blockIdx.x] : partition._ppbs[blockIdx.x];
  ivec3 blockid = partition._activeKeys[blockIdx.x];
  auto advection_bucket = g_buckets_on_particle_buffer
      ? pbuffer._blockbuckets + blockIdx.x * g_particle_num_per_block
      : partition._blockbuckets + blockIdx.x * g_particle_num_per_block;
  // auto particle_offset = partition._binsts[blockIdx.x];
  PREC_G thread_kinetic_energy = (PREC_G)0;
  PREC_G thread_gravity_energy = (PREC_G)0;
  PREC_G thread_strain_energy = (PREC_G)0;
  PREC o = g_offset;
  PREC l = pbuffer.length;
  for (int pidib = threadIdx.x; pidib < pcnt; pidib += blockDim.x) 
  {
    auto advect = advection_bucket[pidib];
    ivec3 source_blockid;
    dir_components(advect / g_particle_num_per_block, source_blockid);
    source_blockid += blockid;
    auto source_blockno = prev_partition.query(source_blockid);
    auto source_pidib = advect % g_particle_num_per_block;
    auto source_bin = g_buckets_on_particle_buffer
          ? pbuffer.ch(_0, pbuffer._binsts[source_blockno] + source_pidib / g_bin_capacity)
          : pbuffer.ch(_0, prev_partition._binsts[source_blockno] + source_pidib / g_bin_capacity);
    auto _source_pidib = source_pidib % g_bin_capacity;
    
    PREC elevation = (source_bin.val(_1, _source_pidib) - o) * l;
    PREC J = 1.0 - source_bin.val(_3, _source_pidib);

    PREC one_minus_bwp = 1.0 - pbuffer.gamma;

    PREC particle_kinetic_energy = 0.0;// 0.5 * pbuffer.mass * (vel[0]*vel[0] + vel[1]*vel[1] + vel[2]*vel[2]);
    PREC particle_gravity_energy = pbuffer.mass * (grav / l) * elevation; //< E_gravity = mgh
    PREC particle_strain_energy = pbuffer.volume * pbuffer.bulk * 
                      ((1.0/(pbuffer.gamma*(pbuffer.gamma - 1.0))) * pow(J, one_minus_bwp) + (1.0/pbuffer.gamma)*J - (1.0/(pbuffer.gamma - 1.0)));

    thread_strain_energy  += (PREC_G)particle_strain_energy;
    thread_gravity_energy += (PREC_G)particle_gravity_energy;
    thread_kinetic_energy += (PREC_G)particle_kinetic_energy;
  }

  __syncthreads();
  atomicAdd(kinetic_energy, thread_kinetic_energy);
  atomicAdd(gravity_energy, thread_gravity_energy);
  atomicAdd(strain_energy, thread_strain_energy);
}

template <typename Partition>
__global__ void query_energy_particles(Partition partition, Partition prev_partition,
                              ParticleBuffer<material_e::JFluid_ASFLIP> pbuffer,
                              PREC_G *kinetic_energy, PREC_G *gravity_energy, PREC_G *strain_energy, PREC grav)  {
  int pcnt = g_buckets_on_particle_buffer ? pbuffer._ppbs[blockIdx.x] : partition._ppbs[blockIdx.x];
  ivec3 blockid = partition._activeKeys[blockIdx.x];
  auto advection_bucket = g_buckets_on_particle_buffer
      ? pbuffer._blockbuckets + blockIdx.x * g_particle_num_per_block
      : partition._blockbuckets + blockIdx.x * g_particle_num_per_block;
  // auto particle_offset = partition._binsts[blockIdx.x];
  PREC_G thread_kinetic_energy = (PREC_G)0;
  PREC_G thread_gravity_energy = (PREC_G)0;
  PREC_G thread_strain_energy = (PREC_G)0;
  PREC_G vel[3];
  PREC o = g_offset;
  PREC l = pbuffer.length;
  for (int pidib = threadIdx.x; pidib < pcnt; pidib += blockDim.x) 
  {
    auto advect = advection_bucket[pidib];
    ivec3 source_blockid;
    dir_components(advect / g_particle_num_per_block, source_blockid);
    source_blockid += blockid;
    auto source_blockno = prev_partition.query(source_blockid);
    auto source_pidib = advect % g_particle_num_per_block;
    auto source_bin = g_buckets_on_particle_buffer
          ? pbuffer.ch(_0, pbuffer._binsts[source_blockno] + source_pidib / g_bin_capacity)
          : pbuffer.ch(_0, prev_partition._binsts[source_blockno] + source_pidib / g_bin_capacity);
    auto _source_pidib = source_pidib % g_bin_capacity;
    
    PREC elevation = (source_bin.val(_1, _source_pidib) - o) * l;
    PREC J = 1.0 - source_bin.val(_3, _source_pidib);
    vel[0] = source_bin.val(_4, _source_pidib) * l;
    vel[1] = source_bin.val(_5, _source_pidib) * l;
    vel[2] = source_bin.val(_6, _source_pidib) * l;
    PREC one_minus_bwp = 1.0 - pbuffer.gamma;

    PREC particle_kinetic_energy = 0.5 * pbuffer.mass * (vel[0]*vel[0] + vel[1]*vel[1] + vel[2]*vel[2]);
    PREC particle_gravity_energy = pbuffer.mass * (grav / l) * elevation; //< E_gravity = mgh
    PREC particle_strain_energy = pbuffer.volume * pbuffer.bulk * 
                      ((1.0/(pbuffer.gamma*(pbuffer.gamma - 1.0))) * pow(J, one_minus_bwp) + (1.0/pbuffer.gamma)*J - (1.0/(pbuffer.gamma - 1.0)));

    thread_strain_energy  += (PREC_G)particle_strain_energy;
    thread_gravity_energy += (PREC_G)particle_gravity_energy;
    thread_kinetic_energy += (PREC_G)particle_kinetic_energy;
  }

  __syncthreads();
  atomicAdd(kinetic_energy, thread_kinetic_energy);
  atomicAdd(gravity_energy, thread_gravity_energy);
  atomicAdd(strain_energy, thread_strain_energy);
}

template <typename Partition>
__global__ void query_energy_particles(Partition partition, Partition prev_partition,
                              ParticleBuffer<material_e::JBarFluid> pbuffer,
                              PREC_G *kinetic_energy, PREC_G *gravity_energy, PREC_G *strain_energy, PREC grav)  {
  int pcnt = g_buckets_on_particle_buffer ? pbuffer._ppbs[blockIdx.x] : partition._ppbs[blockIdx.x];
  ivec3 blockid = partition._activeKeys[blockIdx.x];
  auto advection_bucket = g_buckets_on_particle_buffer
      ? pbuffer._blockbuckets + blockIdx.x * g_particle_num_per_block
      : partition._blockbuckets + blockIdx.x * g_particle_num_per_block;
  // auto particle_offset = partition._binsts[blockIdx.x];
  PREC_G thread_kinetic_energy = (PREC_G)0;
  PREC_G thread_gravity_energy = (PREC_G)0;
  PREC_G thread_strain_energy = (PREC_G)0;
  PREC_G vel[3];
  PREC o = g_offset;
  PREC l = pbuffer.length;
  for (int pidib = threadIdx.x; pidib < pcnt; pidib += blockDim.x) 
  {
    auto advect = advection_bucket[pidib];
    ivec3 source_blockid;
    dir_components(advect / g_particle_num_per_block, source_blockid);
    source_blockid += blockid;
    auto source_blockno = prev_partition.query(source_blockid);
    auto source_pidib = advect % g_particle_num_per_block;
    auto source_bin = g_buckets_on_particle_buffer
          ? pbuffer.ch(_0, pbuffer._binsts[source_blockno] + source_pidib / g_bin_capacity)
          : pbuffer.ch(_0, prev_partition._binsts[source_blockno] + source_pidib / g_bin_capacity);
    auto _source_pidib = source_pidib % g_bin_capacity;
    
    PREC elevation = (source_bin.val(_1, _source_pidib) - o) * l;
    PREC J = 1.0 - source_bin.val(_3, _source_pidib);
    vel[0] = source_bin.val(_4, _source_pidib) * l;
    vel[1] = source_bin.val(_5, _source_pidib) * l;
    vel[2] = source_bin.val(_6, _source_pidib) * l;
    PREC JBar = 1.0 - source_bin.val(_7, _source_pidib);
    // PREC voln = pbuffer.volume * J;

    PREC particle_kinetic_energy = 0.5 * pbuffer.mass * (vel[0]*vel[0] + vel[1]*vel[1] + vel[2]*vel[2]);
    PREC particle_gravity_energy = pbuffer.mass * (grav / l) * elevation; //< E_gravity = mgh
    // PREC particle_strain_energy = pbuffer.volume * pbuffer.bulk * 
    //                   ((1.0/(pbuffer.gamma*(pbuffer.gamma - 1.0))) * pow(JBar, one_minus_bwp) + (1.0/pbuffer.gamma)*JBar - (1.0/(pbuffer.gamma - 1.0)));
    PREC particle_strain_energy;
    compute_energy_jfluid(pbuffer.volume, pbuffer.bulk, pbuffer.gamma, J, particle_strain_energy);

    //compute_energy_jfluid(pbuffer.volume, pbuffer.bulk, pbuffer.gamma, JBar, particle_strain_energy);
    thread_strain_energy  += (PREC_G)particle_strain_energy;
    thread_gravity_energy += (PREC_G)particle_gravity_energy;
    thread_kinetic_energy += (PREC_G)particle_kinetic_energy;

  }

  __syncthreads();
  atomicAdd(kinetic_energy, thread_kinetic_energy);
  atomicAdd(gravity_energy, thread_gravity_energy);
  atomicAdd(strain_energy, thread_strain_energy);
}


template <typename Partition>
__global__ void query_energy_particles(Partition partition, Partition prev_partition,
                              ParticleBuffer<material_e::JFluid_FBAR> pbuffer,
                              PREC_G *kinetic_energy, PREC_G *gravity_energy, PREC_G *strain_energy, PREC grav)  {
  int pcnt = g_buckets_on_particle_buffer ? pbuffer._ppbs[blockIdx.x] : partition._ppbs[blockIdx.x];
  ivec3 blockid = partition._activeKeys[blockIdx.x];
  auto advection_bucket = g_buckets_on_particle_buffer
      ? pbuffer._blockbuckets + blockIdx.x * g_particle_num_per_block
      : partition._blockbuckets + blockIdx.x * g_particle_num_per_block;
  // auto particle_offset = partition._binsts[blockIdx.x];
  PREC_G thread_kinetic_energy = (PREC_G)0;
  PREC_G thread_gravity_energy = (PREC_G)0;
  PREC_G thread_strain_energy = (PREC_G)0;
  PREC o = g_offset;
  PREC l = pbuffer.length;
  for (int pidib = threadIdx.x; pidib < pcnt; pidib += blockDim.x) 
  {
    auto advect = advection_bucket[pidib];
    ivec3 source_blockid;
    dir_components(advect / g_particle_num_per_block, source_blockid);
    source_blockid += blockid;
    auto source_blockno = prev_partition.query(source_blockid);
    auto source_pidib = advect % g_particle_num_per_block;
    auto source_bin = g_buckets_on_particle_buffer
          ? pbuffer.ch(_0, pbuffer._binsts[source_blockno] + source_pidib / g_bin_capacity)
          : pbuffer.ch(_0, prev_partition._binsts[source_blockno] + source_pidib / g_bin_capacity);
    auto _source_pidib = source_pidib % g_bin_capacity;
    
    PREC elevation = (source_bin.val(_1, _source_pidib) - o) * l;
    PREC J = 1.0 - source_bin.val(_3, _source_pidib);
    PREC JBar = 1.0 - source_bin.val(_4, _source_pidib);
    PREC particle_kinetic_energy = 0;
    PREC particle_gravity_energy = pbuffer.mass * (grav / l) * elevation; //< E_gravity = mgh
    PREC particle_strain_energy;
    compute_energy_jfluid(pbuffer.volume, pbuffer.bulk, pbuffer.gamma, JBar, particle_strain_energy);
    thread_strain_energy  += (PREC_G)particle_strain_energy;
    thread_gravity_energy += (PREC_G)particle_gravity_energy;
    thread_kinetic_energy += (PREC_G)particle_kinetic_energy;
  }

  __syncthreads();
  atomicAdd(kinetic_energy, thread_kinetic_energy);
  atomicAdd(gravity_energy, thread_gravity_energy);
  atomicAdd(strain_energy, thread_strain_energy);
}


template <typename Partition>
__global__ void query_energy_particles(Partition partition, Partition prev_partition,
                              ParticleBuffer<material_e::VonMises> pbuffer,
                              PREC_G *kinetic_energy, PREC_G *gravity_energy, PREC_G *strain_energy, PREC grav)  {
  int pcnt = g_buckets_on_particle_buffer ? pbuffer._ppbs[blockIdx.x] : partition._ppbs[blockIdx.x];
  ivec3 blockid = partition._activeKeys[blockIdx.x];
  auto advection_bucket = g_buckets_on_particle_buffer
      ? pbuffer._blockbuckets + blockIdx.x * g_particle_num_per_block
      : partition._blockbuckets + blockIdx.x * g_particle_num_per_block;
  // auto particle_offset = partition._binsts[blockIdx.x];
  PREC_G thread_kinetic_energy = (PREC_G)0;
  PREC_G thread_gravity_energy = (PREC_G)0;
  PREC_G thread_strain_energy = (PREC_G)0;
  PREC o = g_offset;
  PREC l = pbuffer.length;
  for (int pidib = threadIdx.x; pidib < pcnt; pidib += blockDim.x) 
  {
    auto advect = advection_bucket[pidib];
    ivec3 source_blockid;
    dir_components(advect / g_particle_num_per_block, source_blockid);
    source_blockid += blockid;
    auto source_blockno = prev_partition.query(source_blockid);
    auto source_pidib = advect % g_particle_num_per_block;
    auto source_bin = g_buckets_on_particle_buffer
          ? pbuffer.ch(_0, pbuffer._binsts[source_blockno] + source_pidib / g_bin_capacity)
          : pbuffer.ch(_0, prev_partition._binsts[source_blockno] + source_pidib / g_bin_capacity);
    auto _source_pidib = source_pidib % g_bin_capacity;
    pvec9 F;
    F.set(0.0);
    PREC elevation = (source_bin.val(_1, _source_pidib) - o) * l;
    F[0] = source_bin.val(_3, _source_pidib) ;
    F[1] = source_bin.val(_4, _source_pidib) ;
    F[2] = source_bin.val(_5, _source_pidib) ;
    F[3] = source_bin.val(_6, _source_pidib) ;
    F[4] = source_bin.val(_7, _source_pidib) ;
    F[5] = source_bin.val(_8, _source_pidib) ;
    F[6] = source_bin.val(_9, _source_pidib) ;
    F[7] = source_bin.val(_10, _source_pidib) ;
    F[8] = source_bin.val(_11, _source_pidib) ;



    PREC particle_kinetic_energy = 0.0;// 0.5 * pbuffer.mass * (vel[0]*vel[0] + vel[1]*vel[1] + vel[2]*vel[2]);
    PREC particle_gravity_energy = pbuffer.mass * (grav / l) * elevation; //< E_gravity = mgh
    PREC particle_strain_energy = 0.0;
    compute_energy_fixedcorotated(pbuffer.volume, pbuffer.mu, pbuffer.lambda, F, particle_strain_energy);

    thread_strain_energy  += (PREC_G)particle_strain_energy;
    thread_gravity_energy += (PREC_G)particle_gravity_energy;
    thread_kinetic_energy += (PREC_G)particle_kinetic_energy;
  }

  __syncthreads();
  atomicAdd(kinetic_energy, thread_kinetic_energy);
  atomicAdd(gravity_energy, thread_gravity_energy);
  atomicAdd(strain_energy, thread_strain_energy);
}

template <typename Partition>
__global__ void query_energy_particles(Partition partition, Partition prev_partition,
                              ParticleBuffer<material_e::FixedCorotated> pbuffer,
                              PREC_G *kinetic_energy, PREC_G *gravity_energy, PREC_G *strain_energy, PREC grav)  {
  int pcnt = g_buckets_on_particle_buffer ? pbuffer._ppbs[blockIdx.x] : partition._ppbs[blockIdx.x];
  ivec3 blockid = partition._activeKeys[blockIdx.x];
  auto advection_bucket = g_buckets_on_particle_buffer
      ? pbuffer._blockbuckets + blockIdx.x * g_particle_num_per_block
      : partition._blockbuckets + blockIdx.x * g_particle_num_per_block;
  // auto particle_offset = partition._binsts[blockIdx.x];
  PREC_G thread_kinetic_energy = (PREC_G)0;
  PREC_G thread_gravity_energy = (PREC_G)0;
  PREC_G thread_strain_energy = (PREC_G)0;
  PREC o = g_offset;
  PREC l = pbuffer.length;
  for (int pidib = threadIdx.x; pidib < pcnt; pidib += blockDim.x) 
  {
    auto advect = advection_bucket[pidib];
    ivec3 source_blockid;
    dir_components(advect / g_particle_num_per_block, source_blockid);
    source_blockid += blockid;
    auto source_blockno = prev_partition.query(source_blockid);
    auto source_pidib = advect % g_particle_num_per_block;
    auto source_bin = g_buckets_on_particle_buffer
          ? pbuffer.ch(_0, pbuffer._binsts[source_blockno] + source_pidib / g_bin_capacity)
          : pbuffer.ch(_0, prev_partition._binsts[source_blockno] + source_pidib / g_bin_capacity);
    auto _source_pidib = source_pidib % g_bin_capacity;
    pvec9 F;
    F.set(0.0);
    PREC elevation = (source_bin.val(_1, _source_pidib) - o) * l;
    F[0] = source_bin.val(_3, _source_pidib) ;
    F[1] = source_bin.val(_4, _source_pidib) ;
    F[2] = source_bin.val(_5, _source_pidib) ;
    F[3] = source_bin.val(_6, _source_pidib) ;
    F[4] = source_bin.val(_7, _source_pidib) ;
    F[5] = source_bin.val(_8, _source_pidib) ;
    F[6] = source_bin.val(_9, _source_pidib) ;
    F[7] = source_bin.val(_10, _source_pidib) ;
    F[8] = source_bin.val(_11, _source_pidib) ;



    PREC particle_kinetic_energy = 0.0;// 0.5 * pbuffer.mass * (vel[0]*vel[0] + vel[1]*vel[1] + vel[2]*vel[2]);
    PREC particle_gravity_energy = pbuffer.mass * (grav / l) * elevation; //< E_gravity = mgh
    PREC particle_strain_energy = 0.0;
    compute_energy_fixedcorotated(pbuffer.volume, pbuffer.mu, pbuffer.lambda, F, particle_strain_energy);

    thread_strain_energy  += (PREC_G)particle_strain_energy;
    thread_gravity_energy += (PREC_G)particle_gravity_energy;
    thread_kinetic_energy += (PREC_G)particle_kinetic_energy;
  }

  __syncthreads();
  atomicAdd(kinetic_energy, thread_kinetic_energy);
  atomicAdd(gravity_energy, thread_gravity_energy);
  atomicAdd(strain_energy, thread_strain_energy);
}

template <typename Partition>
__global__ void query_energy_particles(Partition partition, Partition prev_partition,
                              ParticleBuffer<material_e::FixedCorotated_ASFLIP> pbuffer,
                              PREC_G *kinetic_energy, PREC_G *gravity_energy, PREC_G *strain_energy, PREC grav)  {
  int pcnt = g_buckets_on_particle_buffer ? pbuffer._ppbs[blockIdx.x] : partition._ppbs[blockIdx.x];
  ivec3 blockid = partition._activeKeys[blockIdx.x];
  auto advection_bucket = g_buckets_on_particle_buffer
      ? pbuffer._blockbuckets + blockIdx.x * g_particle_num_per_block
      : partition._blockbuckets + blockIdx.x * g_particle_num_per_block;
  // auto particle_offset = partition._binsts[blockIdx.x];
  PREC_G thread_kinetic_energy = (PREC_G)0;
  PREC_G thread_gravity_energy = (PREC_G)0;
  PREC_G thread_strain_energy = (PREC_G)0;
  PREC_G vel[3];
  PREC o = g_offset;
  PREC l = pbuffer.length;
  for (int pidib = threadIdx.x; pidib < pcnt; pidib += blockDim.x) 
  {
    auto advect = advection_bucket[pidib];
    ivec3 source_blockid;
    dir_components(advect / g_particle_num_per_block, source_blockid);
    source_blockid += blockid;
    auto source_blockno = prev_partition.query(source_blockid);
    auto source_pidib = advect % g_particle_num_per_block;
    auto source_bin = g_buckets_on_particle_buffer
          ? pbuffer.ch(_0, pbuffer._binsts[source_blockno] + source_pidib / g_bin_capacity)
          : pbuffer.ch(_0, prev_partition._binsts[source_blockno] + source_pidib / g_bin_capacity);
    auto _source_pidib = source_pidib % g_bin_capacity;
    pvec9 F;
    F.set(0.0);
    PREC elevation = (source_bin.val(_1, _source_pidib) - o) * l;
    F[0] = source_bin.val(_3, _source_pidib) ;
    F[1] = source_bin.val(_4, _source_pidib) ;
    F[2] = source_bin.val(_5, _source_pidib) ;
    F[3] = source_bin.val(_6, _source_pidib) ;
    F[4] = source_bin.val(_7, _source_pidib) ;
    F[5] = source_bin.val(_8, _source_pidib) ;
    F[6] = source_bin.val(_9, _source_pidib) ;
    F[7] = source_bin.val(_10, _source_pidib) ;
    F[8] = source_bin.val(_11, _source_pidib) ;
    vel[0] = source_bin.val(_12, _source_pidib) * l;
    vel[1] = source_bin.val(_13, _source_pidib) * l;
    vel[2] = source_bin.val(_14, _source_pidib) * l;


    PREC particle_kinetic_energy = 0.5 * pbuffer.mass * (vel[0]*vel[0] + vel[1]*vel[1] + vel[2]*vel[2]);
    PREC particle_gravity_energy = pbuffer.mass * (grav / l) * elevation; //< E_gravity = mgh
    PREC particle_strain_energy = 0.0;
    compute_energy_fixedcorotated(pbuffer.volume, pbuffer.mu, pbuffer.lambda, F, particle_strain_energy);

    thread_strain_energy  += (PREC_G)particle_strain_energy;
    thread_gravity_energy += (PREC_G)particle_gravity_energy;
    thread_kinetic_energy += (PREC_G)particle_kinetic_energy;
  }

  __syncthreads();
  atomicAdd(kinetic_energy, thread_kinetic_energy);
  atomicAdd(gravity_energy, thread_gravity_energy);
  atomicAdd(strain_energy, thread_strain_energy);
}


template <typename Partition>
__global__ void query_energy_particles(Partition partition, Partition prev_partition,
                              ParticleBuffer<material_e::FixedCorotated_FBAR> pbuffer,
                              PREC_G *kinetic_energy, PREC_G *gravity_energy, PREC_G *strain_energy, PREC grav)  {
  int pcnt = g_buckets_on_particle_buffer ? pbuffer._ppbs[blockIdx.x] : partition._ppbs[blockIdx.x];
  ivec3 blockid = partition._activeKeys[blockIdx.x];
  auto advection_bucket = g_buckets_on_particle_buffer
      ? pbuffer._blockbuckets + blockIdx.x * g_particle_num_per_block
      : partition._blockbuckets + blockIdx.x * g_particle_num_per_block;
  // auto particle_offset = partition._binsts[blockIdx.x];
  PREC_G thread_kinetic_energy = (PREC_G)0;
  PREC_G thread_gravity_energy = (PREC_G)0;
  PREC_G thread_strain_energy = (PREC_G)0;
  PREC o = g_offset;
  PREC l = pbuffer.length;
  for (int pidib = threadIdx.x; pidib < pcnt; pidib += blockDim.x) 
  {
    auto advect = advection_bucket[pidib];
    ivec3 source_blockid;
    dir_components(advect / g_particle_num_per_block, source_blockid);
    source_blockid += blockid;
    auto source_blockno = prev_partition.query(source_blockid);
    auto source_pidib = advect % g_particle_num_per_block;
    auto source_bin = g_buckets_on_particle_buffer
          ? pbuffer.ch(_0, pbuffer._binsts[source_blockno] + source_pidib / g_bin_capacity)
          : pbuffer.ch(_0, prev_partition._binsts[source_blockno] + source_pidib / g_bin_capacity);
    auto _source_pidib = source_pidib % g_bin_capacity;
    pvec9 F;
    F.set(0.0);
    PREC elevation = (source_bin.val(_1, _source_pidib) - o) * l;
    F[0] = source_bin.val(_3, _source_pidib) ;
    F[1] = source_bin.val(_4, _source_pidib) ;
    F[2] = source_bin.val(_5, _source_pidib) ;
    F[3] = source_bin.val(_6, _source_pidib) ;
    F[4] = source_bin.val(_7, _source_pidib) ;
    F[5] = source_bin.val(_8, _source_pidib) ;
    F[6] = source_bin.val(_9, _source_pidib) ;
    F[7] = source_bin.val(_10, _source_pidib) ;
    F[8] = source_bin.val(_11, _source_pidib) ;
    PREC JBar = 1.0 - source_bin.val(_12, _source_pidib);

    PREC particle_kinetic_energy = 0.0;
    PREC particle_gravity_energy = pbuffer.mass * (grav / l) * elevation; //< E_gravity = mgh
    PREC particle_strain_energy = 0.0;
    compute_energy_fixedcorotated(pbuffer.volume, pbuffer.mu, pbuffer.lambda, F, particle_strain_energy);

    thread_strain_energy  += (PREC_G)particle_strain_energy;
    thread_gravity_energy += (PREC_G)particle_gravity_energy;
    thread_kinetic_energy += (PREC_G)particle_kinetic_energy;
  }

  __syncthreads();
  atomicAdd(kinetic_energy, thread_kinetic_energy);
  atomicAdd(gravity_energy, thread_gravity_energy);
  atomicAdd(strain_energy, thread_strain_energy);
}

template <typename Partition>
__global__ void query_energy_particles(Partition partition, Partition prev_partition,
                              ParticleBuffer<material_e::FixedCorotated_ASFLIP_FBAR> pbuffer,
                              PREC_G *kinetic_energy, PREC_G *gravity_energy, PREC_G *strain_energy, PREC grav)  {
  int pcnt = g_buckets_on_particle_buffer ? pbuffer._ppbs[blockIdx.x] : partition._ppbs[blockIdx.x];
  ivec3 blockid = partition._activeKeys[blockIdx.x];
  auto advection_bucket = g_buckets_on_particle_buffer
      ? pbuffer._blockbuckets + blockIdx.x * g_particle_num_per_block
      : partition._blockbuckets + blockIdx.x * g_particle_num_per_block;
  // auto particle_offset = partition._binsts[blockIdx.x];
  PREC_G thread_kinetic_energy = (PREC_G)0;
  PREC_G thread_gravity_energy = (PREC_G)0;
  PREC_G thread_strain_energy = (PREC_G)0;
  PREC_G vel[3];
  PREC o = g_offset;
  PREC l = pbuffer.length;
  for (int pidib = threadIdx.x; pidib < pcnt; pidib += blockDim.x) 
  {
    auto advect = advection_bucket[pidib];
    ivec3 source_blockid;
    dir_components(advect / g_particle_num_per_block, source_blockid);
    source_blockid += blockid;
    auto source_blockno = prev_partition.query(source_blockid);
    auto source_pidib = advect % g_particle_num_per_block;
    auto source_bin = g_buckets_on_particle_buffer
          ? pbuffer.ch(_0, pbuffer._binsts[source_blockno] + source_pidib / g_bin_capacity)
          : pbuffer.ch(_0, prev_partition._binsts[source_blockno] + source_pidib / g_bin_capacity);
    auto _source_pidib = source_pidib % g_bin_capacity;
    pvec9 F;
    F.set(0.0);
    PREC elevation = (source_bin.val(_1, _source_pidib) - o) * l;
    F[0] = source_bin.val(_3, _source_pidib) ;
    F[1] = source_bin.val(_4, _source_pidib) ;
    F[2] = source_bin.val(_5, _source_pidib) ;
    F[3] = source_bin.val(_6, _source_pidib) ;
    F[4] = source_bin.val(_7, _source_pidib) ;
    F[5] = source_bin.val(_8, _source_pidib) ;
    F[6] = source_bin.val(_9, _source_pidib) ;
    F[7] = source_bin.val(_10, _source_pidib) ;
    F[8] = source_bin.val(_11, _source_pidib) ;
    vel[0] = source_bin.val(_12, _source_pidib) * l;
    vel[1] = source_bin.val(_13, _source_pidib) * l;
    vel[2] = source_bin.val(_14, _source_pidib) * l;
    PREC JBar = 1.0 - source_bin.val(_15, _source_pidib);

    PREC particle_kinetic_energy = 0.5 * pbuffer.mass * (vel[0]*vel[0] + vel[1]*vel[1] + vel[2]*vel[2]);
    PREC particle_gravity_energy = pbuffer.mass * (grav / l) * elevation; //< E_gravity = mgh
    PREC particle_strain_energy = 0.0;
    compute_energy_fixedcorotated(pbuffer.volume, pbuffer.mu, pbuffer.lambda, F, particle_strain_energy);

    thread_strain_energy  += (PREC_G)particle_strain_energy;
    thread_gravity_energy += (PREC_G)particle_gravity_energy;
    thread_kinetic_energy += (PREC_G)particle_kinetic_energy;
  }

  __syncthreads();
  atomicAdd(kinetic_energy, thread_kinetic_energy);
  atomicAdd(gravity_energy, thread_gravity_energy);
  atomicAdd(strain_energy, thread_strain_energy);
}

template <typename Partition>
__global__ void query_energy_particles(Partition partition, Partition prev_partition,
                              ParticleBuffer<material_e::NeoHookean> pbuffer,
                              PREC_G *kinetic_energy, PREC_G *gravity_energy, PREC_G *strain_energy, PREC grav)  {
  int pcnt = g_buckets_on_particle_buffer ? pbuffer._ppbs[blockIdx.x] : partition._ppbs[blockIdx.x];
  ivec3 blockid = partition._activeKeys[blockIdx.x];
  auto advection_bucket = g_buckets_on_particle_buffer
      ? pbuffer._blockbuckets + blockIdx.x * g_particle_num_per_block
      : partition._blockbuckets + blockIdx.x * g_particle_num_per_block;
  // auto particle_offset = partition._binsts[blockIdx.x];
  PREC_G thread_kinetic_energy = (PREC_G)0;
  PREC_G thread_gravity_energy = (PREC_G)0;
  PREC_G thread_strain_energy = (PREC_G)0;
  PREC o = g_offset;
  PREC l = pbuffer.length;
  for (int pidib = threadIdx.x; pidib < pcnt; pidib += blockDim.x) 
  {
    auto advect = advection_bucket[pidib];
    ivec3 source_blockid;
    dir_components(advect / g_particle_num_per_block, source_blockid);
    source_blockid += blockid;
    auto source_blockno = prev_partition.query(source_blockid);
    auto source_pidib = advect % g_particle_num_per_block;
    auto source_bin = g_buckets_on_particle_buffer
          ? pbuffer.ch(_0, pbuffer._binsts[source_blockno] + source_pidib / g_bin_capacity)
          : pbuffer.ch(_0, prev_partition._binsts[source_blockno] + source_pidib / g_bin_capacity);
    auto _source_pidib = source_pidib % g_bin_capacity;
    pvec9 F;
    F.set(0.0);
    PREC elevation = (source_bin.val(_1, _source_pidib) - o) * l;
    F[0] = source_bin.val(_3, _source_pidib) ;
    F[1] = source_bin.val(_4, _source_pidib) ;
    F[2] = source_bin.val(_5, _source_pidib) ;
    F[3] = source_bin.val(_6, _source_pidib) ;
    F[4] = source_bin.val(_7, _source_pidib) ;
    F[5] = source_bin.val(_8, _source_pidib) ;
    F[6] = source_bin.val(_9, _source_pidib) ;
    F[7] = source_bin.val(_10, _source_pidib) ;
    F[8] = source_bin.val(_11, _source_pidib) ;



    PREC particle_kinetic_energy = 0.0;// 0.5 * pbuffer.mass * (vel[0]*vel[0] + vel[1]*vel[1] + vel[2]*vel[2]);
    PREC particle_gravity_energy = pbuffer.mass * (grav / l) * elevation; //< E_gravity = mgh
    PREC particle_strain_energy = 0.0;
    compute_energy_neohookean(pbuffer.volume, pbuffer.mu, pbuffer.lambda, F, particle_strain_energy);

    thread_strain_energy  += (PREC_G)particle_strain_energy;
    thread_gravity_energy += (PREC_G)particle_gravity_energy;
    thread_kinetic_energy += (PREC_G)particle_kinetic_energy;
  }

  __syncthreads();
  atomicAdd(kinetic_energy, thread_kinetic_energy);
  atomicAdd(gravity_energy, thread_gravity_energy);
  atomicAdd(strain_energy, thread_strain_energy);
}

template <typename Partition>
__global__ void query_energy_particles(Partition partition, Partition prev_partition,
                              ParticleBuffer<material_e::NeoHookean_ASFLIP> pbuffer,
                              PREC_G *kinetic_energy, PREC_G *gravity_energy, PREC_G *strain_energy, PREC grav)  {
  int pcnt = g_buckets_on_particle_buffer ? pbuffer._ppbs[blockIdx.x] : partition._ppbs[blockIdx.x];
  ivec3 blockid = partition._activeKeys[blockIdx.x];
  auto advection_bucket = g_buckets_on_particle_buffer
      ? pbuffer._blockbuckets + blockIdx.x * g_particle_num_per_block
      : partition._blockbuckets + blockIdx.x * g_particle_num_per_block;
  // auto particle_offset = partition._binsts[blockIdx.x];
  PREC_G thread_kinetic_energy = (PREC_G)0;
  PREC_G thread_gravity_energy = (PREC_G)0;
  PREC_G thread_strain_energy = (PREC_G)0;
  PREC_G vel[3];
  PREC o = g_offset;
  PREC l = pbuffer.length;
  for (int pidib = threadIdx.x; pidib < pcnt; pidib += blockDim.x) 
  {
    auto advect = advection_bucket[pidib];
    ivec3 source_blockid;
    dir_components(advect / g_particle_num_per_block, source_blockid);
    source_blockid += blockid;
    auto source_blockno = prev_partition.query(source_blockid);
    auto source_pidib = advect % g_particle_num_per_block;
    auto source_bin = g_buckets_on_particle_buffer
          ? pbuffer.ch(_0, pbuffer._binsts[source_blockno] + source_pidib / g_bin_capacity)
          : pbuffer.ch(_0, prev_partition._binsts[source_blockno] + source_pidib / g_bin_capacity);
    auto _source_pidib = source_pidib % g_bin_capacity;
    pvec9 F;
    F.set(0.0);
    PREC elevation = (source_bin.val(_1, _source_pidib) - o) * l;
    F[0] = source_bin.val(_3, _source_pidib) ;
    F[1] = source_bin.val(_4, _source_pidib) ;
    F[2] = source_bin.val(_5, _source_pidib) ;
    F[3] = source_bin.val(_6, _source_pidib) ;
    F[4] = source_bin.val(_7, _source_pidib) ;
    F[5] = source_bin.val(_8, _source_pidib) ;
    F[6] = source_bin.val(_9, _source_pidib) ;
    F[7] = source_bin.val(_10, _source_pidib) ;
    F[8] = source_bin.val(_11, _source_pidib) ;
    vel[0] = source_bin.val(_12, _source_pidib) * l;
    vel[1] = source_bin.val(_13, _source_pidib) * l;
    vel[2] = source_bin.val(_14, _source_pidib) * l;


    PREC particle_kinetic_energy = 0.5 * pbuffer.mass * (vel[0]*vel[0] + vel[1]*vel[1] + vel[2]*vel[2]);
    PREC particle_gravity_energy = pbuffer.mass * (grav / l) * elevation; //< E_gravity = mgh
    PREC particle_strain_energy = 0.0;
    compute_energy_neohookean(pbuffer.volume, pbuffer.mu, pbuffer.lambda, F, particle_strain_energy);

    thread_strain_energy  += (PREC_G)particle_strain_energy;
    thread_gravity_energy += (PREC_G)particle_gravity_energy;
    thread_kinetic_energy += (PREC_G)particle_kinetic_energy;
  }

  __syncthreads();
  atomicAdd(kinetic_energy, thread_kinetic_energy);
  atomicAdd(gravity_energy, thread_gravity_energy);
  atomicAdd(strain_energy, thread_strain_energy);
}


template <typename Partition>
__global__ void query_energy_particles(Partition partition, Partition prev_partition,
                              ParticleBuffer<material_e::NeoHookean_FBAR> pbuffer,
                              PREC_G *kinetic_energy, PREC_G *gravity_energy, PREC_G *strain_energy, PREC grav)  {
  int pcnt = g_buckets_on_particle_buffer ? pbuffer._ppbs[blockIdx.x] : partition._ppbs[blockIdx.x];
  ivec3 blockid = partition._activeKeys[blockIdx.x];
  auto advection_bucket = g_buckets_on_particle_buffer
      ? pbuffer._blockbuckets + blockIdx.x * g_particle_num_per_block
      : partition._blockbuckets + blockIdx.x * g_particle_num_per_block;
  // auto particle_offset = partition._binsts[blockIdx.x];
  PREC_G thread_kinetic_energy = (PREC_G)0;
  PREC_G thread_gravity_energy = (PREC_G)0;
  PREC_G thread_strain_energy = (PREC_G)0;
  PREC o = g_offset;
  PREC l = pbuffer.length;
  for (int pidib = threadIdx.x; pidib < pcnt; pidib += blockDim.x) 
  {
    auto advect = advection_bucket[pidib];
    ivec3 source_blockid;
    dir_components(advect / g_particle_num_per_block, source_blockid);
    source_blockid += blockid;
    auto source_blockno = prev_partition.query(source_blockid);
    auto source_pidib = advect % g_particle_num_per_block;
    auto source_bin = g_buckets_on_particle_buffer
          ? pbuffer.ch(_0, pbuffer._binsts[source_blockno] + source_pidib / g_bin_capacity)
          : pbuffer.ch(_0, prev_partition._binsts[source_blockno] + source_pidib / g_bin_capacity);
    auto _source_pidib = source_pidib % g_bin_capacity;
    pvec9 F;
    F.set(0.0);
    PREC elevation = (source_bin.val(_1, _source_pidib) - o) * l;
    F[0] = source_bin.val(_3, _source_pidib) ;
    F[1] = source_bin.val(_4, _source_pidib) ;
    F[2] = source_bin.val(_5, _source_pidib) ;
    F[3] = source_bin.val(_6, _source_pidib) ;
    F[4] = source_bin.val(_7, _source_pidib) ;
    F[5] = source_bin.val(_8, _source_pidib) ;
    F[6] = source_bin.val(_9, _source_pidib) ;
    F[7] = source_bin.val(_10, _source_pidib) ;
    F[8] = source_bin.val(_11, _source_pidib) ;
    PREC JBar = 1.0 - source_bin.val(_12, _source_pidib);

    PREC particle_kinetic_energy = 0.0;
    PREC particle_gravity_energy = pbuffer.mass * (grav / l) * elevation; //< E_gravity = mgh
    PREC particle_strain_energy = 0.0;
    compute_energy_neohookean(pbuffer.volume, pbuffer.mu, pbuffer.lambda, F, particle_strain_energy);

    thread_strain_energy  += (PREC_G)particle_strain_energy;
    thread_gravity_energy += (PREC_G)particle_gravity_energy;
    thread_kinetic_energy += (PREC_G)particle_kinetic_energy;
  }

  __syncthreads();
  atomicAdd(kinetic_energy, thread_kinetic_energy);
  atomicAdd(gravity_energy, thread_gravity_energy);
  atomicAdd(strain_energy, thread_strain_energy);
}

template <typename Partition>
__global__ void query_energy_particles(Partition partition, Partition prev_partition,
                              ParticleBuffer<material_e::NeoHookean_ASFLIP_FBAR> pbuffer,
                              PREC_G *kinetic_energy, PREC_G *gravity_energy, PREC_G *strain_energy, PREC grav)  {
  int pcnt = g_buckets_on_particle_buffer ? pbuffer._ppbs[blockIdx.x] : partition._ppbs[blockIdx.x];
  ivec3 blockid = partition._activeKeys[blockIdx.x];
  auto advection_bucket = g_buckets_on_particle_buffer
      ? pbuffer._blockbuckets + blockIdx.x * g_particle_num_per_block
      : partition._blockbuckets + blockIdx.x * g_particle_num_per_block;
  // auto particle_offset = partition._binsts[blockIdx.x];
  PREC_G thread_kinetic_energy = (PREC_G)0;
  PREC_G thread_gravity_energy = (PREC_G)0;
  PREC_G thread_strain_energy = (PREC_G)0;
  pvec3 vel;
  PREC o = g_offset;
  PREC l = pbuffer.length;
  for (int pidib = threadIdx.x; pidib < pcnt; pidib += blockDim.x) 
  {
    auto advect = advection_bucket[pidib];
    ivec3 source_blockid;
    dir_components(advect / g_particle_num_per_block, source_blockid);
    source_blockid += blockid;
    auto source_blockno = prev_partition.query(source_blockid);
    auto source_pidib = advect % g_particle_num_per_block;
    auto source_bin = g_buckets_on_particle_buffer
          ? pbuffer.ch(_0, pbuffer._binsts[source_blockno] + source_pidib / g_bin_capacity)
          : pbuffer.ch(_0, prev_partition._binsts[source_blockno] + source_pidib / g_bin_capacity);
    auto _source_pidib = source_pidib % g_bin_capacity;
    pvec9 F;
    F.set(0.0);
    PREC elevation = (source_bin.val(_1, _source_pidib) - o) * l;
    F[0] = source_bin.val(_3, _source_pidib) ;
    F[1] = source_bin.val(_4, _source_pidib) ;
    F[2] = source_bin.val(_5, _source_pidib) ;
    F[3] = source_bin.val(_6, _source_pidib) ;
    F[4] = source_bin.val(_7, _source_pidib) ;
    F[5] = source_bin.val(_8, _source_pidib) ;
    F[6] = source_bin.val(_9, _source_pidib) ;
    F[7] = source_bin.val(_10, _source_pidib) ;
    F[8] = source_bin.val(_11, _source_pidib) ;
    vel[0] = source_bin.val(_12, _source_pidib) * l;
    vel[1] = source_bin.val(_13, _source_pidib) * l;
    vel[2] = source_bin.val(_14, _source_pidib) * l;
    PREC voln = source_bin.val(_15, _source_pidib);
    PREC JBar = 1.0 - source_bin.val(_16, _source_pidib);
    PREC J = matrixDeterminant3d(F.data());
    for (int d=0; d < 9; d++) F[d] = cbrt(JBar / J) * F[d];

    PREC particle_kinetic_energy = 0.0;    
    pbuffer.getEnergy_Kinetic(vel, particle_kinetic_energy);

    PREC particle_gravity_energy = pbuffer.mass * (grav / l) * elevation; //< E_gravity = mgh
    PREC particle_strain_energy = 0.0;
    // compute_energy_neohookean(pbuffer.volume, pbuffer.mu, pbuffer.lambda, F, particle_strain_energy);
    pbuffer.getEnergy_Strain(F, particle_strain_energy); // deviates from other neohookean variants

    thread_strain_energy  += (PREC_G)particle_strain_energy;
    thread_gravity_energy += (PREC_G)particle_gravity_energy;
    thread_kinetic_energy += (PREC_G)particle_kinetic_energy;
  }

  __syncthreads();
  atomicAdd(kinetic_energy, thread_kinetic_energy);
  atomicAdd(gravity_energy, thread_gravity_energy);
  atomicAdd(strain_energy, thread_strain_energy);
}


template <typename Partition>
__global__ void query_energy_particles(Partition partition, Partition prev_partition,
                              ParticleBuffer<material_e::Sand> pbuffer,
                              PREC_G *kinetic_energy, PREC_G *gravity_energy, PREC_G *strain_energy, PREC grav)  {
  int pcnt = g_buckets_on_particle_buffer ? pbuffer._ppbs[blockIdx.x] : partition._ppbs[blockIdx.x];
  ivec3 blockid = partition._activeKeys[blockIdx.x];
  auto advection_bucket = g_buckets_on_particle_buffer
      ? pbuffer._blockbuckets + blockIdx.x * g_particle_num_per_block
      : partition._blockbuckets + blockIdx.x * g_particle_num_per_block;
  // auto particle_offset = partition._binsts[blockIdx.x];
  PREC_G thread_kinetic_energy = (PREC_G)0;
  PREC_G thread_gravity_energy = (PREC_G)0;
  PREC_G thread_strain_energy = (PREC_G)0;
  PREC_G vel[3];
  PREC o = g_offset;
  PREC l = pbuffer.length;
  for (int pidib = threadIdx.x; pidib < pcnt; pidib += blockDim.x) 
  {
    auto advect = advection_bucket[pidib];
    ivec3 source_blockid;
    dir_components(advect / g_particle_num_per_block, source_blockid);
    source_blockid += blockid;
    auto source_blockno = prev_partition.query(source_blockid);
    auto source_pidib = advect % g_particle_num_per_block;
    auto source_bin = g_buckets_on_particle_buffer
          ? pbuffer.ch(_0, pbuffer._binsts[source_blockno] + source_pidib / g_bin_capacity)
          : pbuffer.ch(_0, prev_partition._binsts[source_blockno] + source_pidib / g_bin_capacity);
    auto _source_pidib = source_pidib % g_bin_capacity;
    pvec9 F;
    F.set(0.0);
    PREC logJp;
    PREC elevation = (source_bin.val(_1, _source_pidib) - o) * l;
    F[0] = source_bin.val(_3, _source_pidib) ;
    F[1] = source_bin.val(_4, _source_pidib) ;
    F[2] = source_bin.val(_5, _source_pidib) ;
    F[3] = source_bin.val(_6, _source_pidib) ;
    F[4] = source_bin.val(_7, _source_pidib) ;
    F[5] = source_bin.val(_8, _source_pidib) ;
    F[6] = source_bin.val(_9, _source_pidib) ;
    F[7] = source_bin.val(_10, _source_pidib) ;
    F[8] = source_bin.val(_11, _source_pidib) ;
    logJp =  source_bin.val(_12, _source_pidib);

    PREC particle_kinetic_energy = 0.0;
    PREC particle_gravity_energy = pbuffer.mass * (grav / l) * elevation; //< E_gravity = mgh
    PREC particle_strain_energy = 0.0;
    compute_energy_sand<PREC>(pbuffer.volume, pbuffer.mu, pbuffer.lambda, pbuffer.cohesion, pbuffer.beta, pbuffer.yieldSurface, pbuffer.volumeCorrection, logJp, F, particle_strain_energy);

    thread_strain_energy  += (PREC_G)particle_strain_energy;
    thread_gravity_energy += (PREC_G)particle_gravity_energy;
    thread_kinetic_energy += (PREC_G)particle_kinetic_energy;
  }

  __syncthreads();
  atomicAdd(kinetic_energy, thread_kinetic_energy);
  atomicAdd(gravity_energy, thread_gravity_energy);
  atomicAdd(strain_energy, thread_strain_energy);
}


template <typename Partition>
__global__ void query_energy_particles(Partition partition, Partition prev_partition,
                              ParticleBuffer<material_e::Sand_ASFLIP> pbuffer,
                              PREC_G *kinetic_energy, PREC_G *gravity_energy, PREC_G *strain_energy, PREC grav)  {
  int pcnt = g_buckets_on_particle_buffer ? pbuffer._ppbs[blockIdx.x] : partition._ppbs[blockIdx.x];
  ivec3 blockid = partition._activeKeys[blockIdx.x];
  auto advection_bucket = g_buckets_on_particle_buffer
      ? pbuffer._blockbuckets + blockIdx.x * g_particle_num_per_block
      : partition._blockbuckets + blockIdx.x * g_particle_num_per_block;
  // auto particle_offset = partition._binsts[blockIdx.x];
  PREC_G thread_kinetic_energy = (PREC_G)0;
  PREC_G thread_gravity_energy = (PREC_G)0;
  PREC_G thread_strain_energy = (PREC_G)0;
  PREC_G vel[3];
  PREC o = g_offset;
  PREC l = pbuffer.length;
  for (int pidib = threadIdx.x; pidib < pcnt; pidib += blockDim.x) 
  {
    auto advect = advection_bucket[pidib];
    ivec3 source_blockid;
    dir_components(advect / g_particle_num_per_block, source_blockid);
    source_blockid += blockid;
    auto source_blockno = prev_partition.query(source_blockid);
    auto source_pidib = advect % g_particle_num_per_block;
    auto source_bin = g_buckets_on_particle_buffer
          ? pbuffer.ch(_0, pbuffer._binsts[source_blockno] + source_pidib / g_bin_capacity)
          : pbuffer.ch(_0, prev_partition._binsts[source_blockno] + source_pidib / g_bin_capacity);
    auto _source_pidib = source_pidib % g_bin_capacity;
    pvec9 F;
    F.set(0.0);
    PREC logJp;
    PREC elevation = (source_bin.val(_1, _source_pidib) - o) * l;
    F[0] = source_bin.val(_3, _source_pidib) ;
    F[1] = source_bin.val(_4, _source_pidib) ;
    F[2] = source_bin.val(_5, _source_pidib) ;
    F[3] = source_bin.val(_6, _source_pidib) ;
    F[4] = source_bin.val(_7, _source_pidib) ;
    F[5] = source_bin.val(_8, _source_pidib) ;
    F[6] = source_bin.val(_9, _source_pidib) ;
    F[7] = source_bin.val(_10, _source_pidib) ;
    F[8] = source_bin.val(_11, _source_pidib) ;
    logJp =  source_bin.val(_12, _source_pidib);
    vel[0] = source_bin.val(_13, _source_pidib) * l;
    vel[1] = source_bin.val(_14, _source_pidib) * l;
    vel[2] = source_bin.val(_15, _source_pidib) * l;

    PREC particle_kinetic_energy = 0.5 * pbuffer.mass * (vel[0]*vel[0] + vel[1]*vel[1] + vel[2]*vel[2]);
    PREC particle_gravity_energy = pbuffer.mass * (grav / l) * elevation; //< E_gravity = mgh
    PREC particle_strain_energy = 0.0;
    compute_energy_sand<PREC>(pbuffer.volume, pbuffer.mu, pbuffer.lambda, pbuffer.cohesion, pbuffer.beta, pbuffer.yieldSurface, pbuffer.volumeCorrection, logJp, F, particle_strain_energy);

    thread_strain_energy  += (PREC_G)particle_strain_energy;
    thread_gravity_energy += (PREC_G)particle_gravity_energy;
    thread_kinetic_energy += (PREC_G)particle_kinetic_energy;
  }

  __syncthreads();
  atomicAdd(kinetic_energy, thread_kinetic_energy);
  atomicAdd(gravity_energy, thread_gravity_energy);
  atomicAdd(strain_energy, thread_strain_energy);
}


template <typename Partition>
__global__ void query_energy_particles(Partition partition, Partition prev_partition,
                              ParticleBuffer<material_e::Sand_FBAR> pbuffer,
                              PREC_G *kinetic_energy, PREC_G *gravity_energy, PREC_G *strain_energy, PREC grav)  {
  int pcnt = g_buckets_on_particle_buffer ? pbuffer._ppbs[blockIdx.x] : partition._ppbs[blockIdx.x];
  ivec3 blockid = partition._activeKeys[blockIdx.x];
  auto advection_bucket = g_buckets_on_particle_buffer
      ? pbuffer._blockbuckets + blockIdx.x * g_particle_num_per_block
      : partition._blockbuckets + blockIdx.x * g_particle_num_per_block;
  // auto particle_offset = partition._binsts[blockIdx.x];
  PREC_G thread_kinetic_energy = (PREC_G)0;
  PREC_G thread_gravity_energy = (PREC_G)0;
  PREC_G thread_strain_energy = (PREC_G)0;
  PREC o = g_offset;
  PREC l = pbuffer.length;
  for (int pidib = threadIdx.x; pidib < pcnt; pidib += blockDim.x) 
  {
    auto advect = advection_bucket[pidib];
    ivec3 source_blockid;
    dir_components(advect / g_particle_num_per_block, source_blockid);
    source_blockid += blockid;
    auto source_blockno = prev_partition.query(source_blockid);
    auto source_pidib = advect % g_particle_num_per_block;
    auto source_bin = g_buckets_on_particle_buffer
          ? pbuffer.ch(_0, pbuffer._binsts[source_blockno] + source_pidib / g_bin_capacity)
          : pbuffer.ch(_0, prev_partition._binsts[source_blockno] + source_pidib / g_bin_capacity);
    auto _source_pidib = source_pidib % g_bin_capacity;
    pvec9 F;
    F.set(0.0);
    PREC logJp;
    PREC elevation = (source_bin.val(_1, _source_pidib) - o) * l;
    F[0] = source_bin.val(_3, _source_pidib) ;
    F[1] = source_bin.val(_4, _source_pidib) ;
    F[2] = source_bin.val(_5, _source_pidib) ;
    F[3] = source_bin.val(_6, _source_pidib) ;
    F[4] = source_bin.val(_7, _source_pidib) ;
    F[5] = source_bin.val(_8, _source_pidib) ;
    F[6] = source_bin.val(_9, _source_pidib) ;
    F[7] = source_bin.val(_10, _source_pidib) ;
    F[8] = source_bin.val(_11, _source_pidib) ;
    logJp =  source_bin.val(_12, _source_pidib);

    PREC particle_kinetic_energy = 0.0;
    PREC particle_gravity_energy = pbuffer.mass * (grav / l) * elevation; //< E_gravity = mgh
    PREC particle_strain_energy = 0.0;
    compute_energy_sand<PREC>(pbuffer.volume, pbuffer.mu, pbuffer.lambda, pbuffer.cohesion, pbuffer.beta, pbuffer.yieldSurface, pbuffer.volumeCorrection, logJp, F, particle_strain_energy);

    thread_strain_energy  += (PREC_G)particle_strain_energy;
    thread_gravity_energy += (PREC_G)particle_gravity_energy;
    thread_kinetic_energy += (PREC_G)particle_kinetic_energy;
  }

  __syncthreads();
  atomicAdd(kinetic_energy, thread_kinetic_energy);
  atomicAdd(gravity_energy, thread_gravity_energy);
  atomicAdd(strain_energy, thread_strain_energy);
}

template <typename Partition>
__global__ void query_energy_particles(Partition partition, Partition prev_partition,
                              ParticleBuffer<material_e::Sand_ASFLIP_FBAR> pbuffer,
                              PREC_G *kinetic_energy, PREC_G *gravity_energy, PREC_G *strain_energy, PREC grav)  {
  int pcnt = g_buckets_on_particle_buffer ? pbuffer._ppbs[blockIdx.x] : partition._ppbs[blockIdx.x];
  ivec3 blockid = partition._activeKeys[blockIdx.x];
  auto advection_bucket = g_buckets_on_particle_buffer
      ? pbuffer._blockbuckets + blockIdx.x * g_particle_num_per_block
      : partition._blockbuckets + blockIdx.x * g_particle_num_per_block;
  // auto particle_offset = partition._binsts[blockIdx.x];
  PREC_G thread_kinetic_energy = (PREC_G)0;
  PREC_G thread_gravity_energy = (PREC_G)0;
  PREC_G thread_strain_energy = (PREC_G)0;
  PREC_G vel[3];
  PREC o = g_offset;
  PREC l = pbuffer.length;
  for (int pidib = threadIdx.x; pidib < pcnt; pidib += blockDim.x) 
  {
    auto advect = advection_bucket[pidib];
    ivec3 source_blockid;
    dir_components(advect / g_particle_num_per_block, source_blockid);
    source_blockid += blockid;
    auto source_blockno = prev_partition.query(source_blockid);
    auto source_pidib = advect % g_particle_num_per_block;
    auto source_bin = g_buckets_on_particle_buffer
          ? pbuffer.ch(_0, pbuffer._binsts[source_blockno] + source_pidib / g_bin_capacity)
          : pbuffer.ch(_0, prev_partition._binsts[source_blockno] + source_pidib / g_bin_capacity);
    auto _source_pidib = source_pidib % g_bin_capacity;
    pvec9 F;
    F.set(0.0);
    PREC logJp;
    PREC elevation = (source_bin.val(_1, _source_pidib) - o) * l;
    F[0] = source_bin.val(_3, _source_pidib) ;
    F[1] = source_bin.val(_4, _source_pidib) ;
    F[2] = source_bin.val(_5, _source_pidib) ;
    F[3] = source_bin.val(_6, _source_pidib) ;
    F[4] = source_bin.val(_7, _source_pidib) ;
    F[5] = source_bin.val(_8, _source_pidib) ;
    F[6] = source_bin.val(_9, _source_pidib) ;
    F[7] = source_bin.val(_10, _source_pidib) ;
    F[8] = source_bin.val(_11, _source_pidib) ;
    logJp =  source_bin.val(_12, _source_pidib);
    vel[0] = source_bin.val(_13, _source_pidib) * l;
    vel[1] = source_bin.val(_14, _source_pidib) * l;
    vel[2] = source_bin.val(_15, _source_pidib) * l;

    PREC particle_kinetic_energy = 0.5 * pbuffer.mass * (vel[0]*vel[0] + vel[1]*vel[1] + vel[2]*vel[2]);
    PREC particle_gravity_energy = pbuffer.mass * (grav / l) * elevation; //< E_gravity = mgh
    PREC particle_strain_energy = 0.0;
    compute_energy_sand<PREC>(pbuffer.volume, pbuffer.mu, pbuffer.lambda, pbuffer.cohesion, pbuffer.beta, pbuffer.yieldSurface, pbuffer.volumeCorrection, logJp, F, particle_strain_energy);

    thread_strain_energy  += (PREC_G)particle_strain_energy;
    thread_gravity_energy += (PREC_G)particle_gravity_energy;
    thread_kinetic_energy += (PREC_G)particle_kinetic_energy;
  }

  __syncthreads();
  atomicAdd(kinetic_energy, thread_kinetic_energy);
  atomicAdd(gravity_energy, thread_gravity_energy);
  atomicAdd(strain_energy, thread_strain_energy);
}


template <typename Partition>
__global__ void query_energy_particles(Partition partition, Partition prev_partition,
                              ParticleBuffer<material_e::CoupledUP> pbuffer,
                              PREC_G *kinetic_energy, PREC_G *gravity_energy, PREC_G *strain_energy, float grav)  {
  int pcnt = g_buckets_on_particle_buffer ? pbuffer._ppbs[blockIdx.x] : partition._ppbs[blockIdx.x];
  ivec3 blockid = partition._activeKeys[blockIdx.x];
  auto advection_bucket = g_buckets_on_particle_buffer
      ? pbuffer._blockbuckets + blockIdx.x * g_particle_num_per_block
      : partition._blockbuckets + blockIdx.x * g_particle_num_per_block;
  // auto particle_offset = partition._binsts[blockIdx.x];
  PREC_G thread_kinetic_energy = (PREC_G)0;
  PREC_G thread_gravity_energy = (PREC_G)0;
  PREC_G thread_strain_energy = (PREC_G)0;
  PREC_G vel[3];
  PREC o = g_offset;
  PREC l = pbuffer.length;
  for (int pidib = threadIdx.x; pidib < pcnt; pidib += blockDim.x) 
  {
    auto advect = advection_bucket[pidib];
    ivec3 source_blockid;
    dir_components(advect / g_particle_num_per_block, source_blockid);
    source_blockid += blockid;
    auto source_blockno = prev_partition.query(source_blockid);
    auto source_pidib = advect % g_particle_num_per_block;
    auto source_bin = g_buckets_on_particle_buffer
          ? pbuffer.ch(_0, pbuffer._binsts[source_blockno] + source_pidib / g_bin_capacity)
          : pbuffer.ch(_0, prev_partition._binsts[source_blockno] + source_pidib / g_bin_capacity);
    auto _source_pidib = source_pidib % g_bin_capacity;
    pvec9 F;
    F.set(0.0);
    PREC logJp;
    PREC elevation = (source_bin.val(_1, _source_pidib) - o) * l;
    F[0] = source_bin.val(_3, _source_pidib) ;
    F[1] = source_bin.val(_4, _source_pidib) ;
    F[2] = source_bin.val(_5, _source_pidib) ;
    F[3] = source_bin.val(_6, _source_pidib) ;
    F[4] = source_bin.val(_7, _source_pidib) ;
    F[5] = source_bin.val(_8, _source_pidib) ;
    F[6] = source_bin.val(_9, _source_pidib) ;
    F[7] = source_bin.val(_10, _source_pidib) ;
    F[8] = source_bin.val(_11, _source_pidib) ;
    logJp =  source_bin.val(_12, _source_pidib);
    vel[0] = source_bin.val(_13, _source_pidib) * l;
    vel[1] = source_bin.val(_14, _source_pidib) * l;
    vel[2] = source_bin.val(_15, _source_pidib) * l;
    // PREC sJBar =  source_bin.val(_16, _source_pidib); // (1 - JBar)
    // PREC empty_value =  source_bin.val(_17, _source_pidib); // Place-holder element, deprecate soon
    // PREC pw =  source_bin.val(_18, _source_pidib); // Water pressure, [Pa]
    // PREC ID =  source_bin.val(_19, _source_pidib); // Particle ID

    PREC particle_kinetic_energy = 0.5 * pbuffer.mass * (vel[0]*vel[0] + vel[1]*vel[1] + vel[2]*vel[2]);
    PREC particle_gravity_energy = pbuffer.mass * (grav / l) * elevation; //< E_gravity = mgh
    PREC particle_strain_energy = 0.0;
    compute_energy_CoupledUP<PREC>(pbuffer.volume, pbuffer.mu, pbuffer.lambda, pbuffer.cohesion, pbuffer.beta, pbuffer.yieldSurface, pbuffer.volumeCorrection, logJp, F, particle_strain_energy);

    thread_kinetic_energy += (PREC_G)particle_kinetic_energy;
    thread_gravity_energy += (PREC_G)particle_gravity_energy;
    thread_strain_energy  += (PREC_G)particle_strain_energy; 
  }

  __syncthreads();
  atomicAdd(kinetic_energy, thread_kinetic_energy);
  atomicAdd(gravity_energy, thread_gravity_energy);
  atomicAdd(strain_energy, thread_strain_energy);
}

// %% ============================================================= %%
//     MPM Grid-to-Particle-to-Grid Functions
// %% ============================================================= %%

template <typename ParticleBuffer, typename Partition, typename Grid>
__global__ void g2p2g(double dt, double newDt, const ivec3 *__restrict__ blocks,
                      const ParticleBuffer pbuffer,
                      ParticleBuffer next_pbuffer,
                      const Partition prev_partition, Partition partition,
                      const Grid grid, Grid next_grid) { }

template <typename Partition, typename Grid>
__global__ void g2p2g(double dt, double newDt, const ivec3 *__restrict__ blocks,
                      const ParticleBuffer<material_e::JFluid> pbuffer,
                      ParticleBuffer<material_e::JFluid> next_pbuffer,
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

  ivec3 blockid;
  int src_blockno;
  if (blocks != nullptr) { 
    blockid = blocks[blockIdx.x]; // Halo block ID
    src_blockno = partition.query(blockid); // Halo block number
  } else { 
    if (partition._haloMarks[blockIdx.x]) return; // Return early if halo block
    blockid = partition._activeKeys[blockIdx.x]; // Non-halo block ID
    src_blockno = blockIdx.x; // Non-halo block number
  }
  if (src_blockno < 0) return; // Return early if negative block number
//  else if (src_blockno > blockcnt) return; // Return early if excessive block number
  int ppb = g_buckets_on_particle_buffer
            ? next_pbuffer._ppbs[src_blockno] 
            : partition._ppbs[src_blockno]; // Particles in block
  if (ppb == 0) return; // Return early if no particles

  // Load G2P global grid data into shared memory. All particles in block will interpolate from it
  // Pull in G2P attributes per grid-node (e.g. velocity) in each grid-block in the grid-arena
  for (int base = threadIdx.x; base < numViInArena; base += blockDim.x) {
    char local_block_id = base / numViPerBlock;
    auto blockno = partition.query(
        ivec3{blockid[0] + ((local_block_id & 4) != 0 ? 1 : 0),
              blockid[1] + ((local_block_id & 2) != 0 ? 1 : 0),
              blockid[2] + ((local_block_id & 1) != 0 ? 1 : 0)});
    auto grid_block = grid.ch(_0, blockno);
    int channelid = base % numViPerBlock;
    char c = channelid & 0x3f; // channelid % g_blockvolume
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
  // Zero out shared memory for P2G grid-data (filled later)
  for (int base = threadIdx.x; base < numMViInArena; base += blockDim.x) {
    int loc = base;
    char z = loc & arenamask;
    char y = (loc >>= arenabits) & arenamask;
    char x = (loc >>= arenabits) & arenamask;
    p2gbuffer[loc >> arenabits][x][y][z] = 0.f;
  }
  __syncthreads();
  // 
  for (int pidib = threadIdx.x; pidib < ppb;
       pidib += blockDim.x) {
  // for (int pidib = threadIdx.x; pidib < partition._ppbs[src_blockno];
  //      pidib += blockDim.x) {
    int source_blockno, source_pidib;
    ivec3 base_index;
    {
      int advect = (g_buckets_on_particle_buffer)
                    ? next_pbuffer._blockbuckets[src_blockno * g_particle_num_per_block + pidib]
                    : partition._blockbuckets[src_blockno * g_particle_num_per_block + pidib];
      dir_components(advect / g_particle_num_per_block, base_index);
      base_index += blockid;
      source_blockno = prev_partition.query(base_index);
      source_pidib = advect % g_particle_num_per_block; // & (g_particle_num_per_block - 1);
      source_blockno = (g_buckets_on_particle_buffer)
                        ? pbuffer._binsts[source_blockno] + source_pidib / g_bin_capacity
                        : prev_partition._binsts[source_blockno] + source_pidib / g_bin_capacity;
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
    ivec3 local_base_index = (pos * g_dx_inv_d + 0.5).cast<int>() - 1;
    pvec3 local_pos = pos - local_base_index * g_dx_d;
    base_index = local_base_index;

    pvec3x3 dws;
#pragma unroll 3
    for (int dd = 0; dd < 3; ++dd) {
      PREC d = (local_pos[dd] - static_cast<PREC>(std::floor(local_pos[dd] * g_dx_inv_d + 0.5) - 1) * g_dx_d) * g_dx_inv_d;
      dws(dd, 0) = 0.5 * (1.5 - d) * (1.5 - d);
      d -= 1.0;
      dws(dd, 1) = 0.75 - d * d;
      d = 0.5 + d;
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
      // PREC pressure = (pbuffer.bulk / pbuffer.gamma) * expm1(-pbuffer.gamma*log1p(-sJ));
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
        auto particle_bin = g_buckets_on_particle_buffer 
            ? next_pbuffer.ch(_0, next_pbuffer._binsts[src_blockno] + pidib / g_bin_capacity) 
            : next_pbuffer.ch(_0, partition._binsts[src_blockno] + pidib / g_bin_capacity);
        particle_bin.val(_0, pidib % g_bin_capacity) = pos[0];
        particle_bin.val(_1, pidib % g_bin_capacity) = pos[1];
        particle_bin.val(_2, pidib % g_bin_capacity) = pos[2];
        particle_bin.val(_3, pidib % g_bin_capacity) = J;
      }
    }

    local_base_index = (pos * g_dx_inv + 0.5f).cast<int>() - 1;
    {
      int direction_tag = dir_offset((base_index - 1) / g_blocksize -
                              (local_base_index - 1) / g_blocksize);
      if (g_buckets_on_particle_buffer)
        next_pbuffer.add_advection(partition, local_base_index - 1, direction_tag, pidib);
      else 
        partition.add_advection(local_base_index - 1, direction_tag, pidib);
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
template <typename Partition, typename Grid>
__global__ void g2p2g(double dt, double newDt, const ivec3 *__restrict__ blocks,
                      const ParticleBuffer<material_e::JFluid_ASFLIP> pbuffer,
                      ParticleBuffer<material_e::JFluid_ASFLIP> next_pbuffer,
                      const Partition prev_partition, Partition partition,
                      const Grid grid, Grid next_grid) {
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
    blockid = blocks[blockIdx.x]; // Halo block ID
    src_blockno = partition.query(blockid); // Halo block number
  } else { 
    if (partition._haloMarks[blockIdx.x]) return; // Return early if halo block
    blockid = partition._activeKeys[blockIdx.x]; // Non-halo block ID
    src_blockno = blockIdx.x; // Non-halo block number
  }
  if (src_blockno < 0) return; // Return early if negative block number
//  else if (src_blockno > blockcnt) return; // Return early if excessive block number
  int ppb = g_buckets_on_particle_buffer
            ? next_pbuffer._ppbs[src_blockno] 
            : partition._ppbs[src_blockno]; // Particles in block
  if (ppb == 0) return; // Return early if no particles

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
  for (int pidib = threadIdx.x; pidib < ppb; pidib += blockDim.x) {
    int source_blockno, source_pidib;
    ivec3 base_index;
    {
      int advect = g_buckets_on_particle_buffer
          ? next_pbuffer._blockbuckets[src_blockno * g_particle_num_per_block + pidib]
          : partition._blockbuckets[src_blockno * g_particle_num_per_block + pidib];
      dir_components(advect / g_particle_num_per_block, base_index);
      base_index += blockid;
      source_blockno = prev_partition.query(base_index);
      source_pidib = advect % g_particle_num_per_block;// & (g_particle_num_per_block - 1);
      source_blockno = g_buckets_on_particle_buffer
          ? pbuffer._binsts[source_blockno] + source_pidib / g_bin_capacity
          : prev_partition._binsts[source_blockno] + source_pidib / g_bin_capacity;
    }
    pvec3 pos;  //< Particle position at n
    pvec3 vel_p; //< Particle vel. at n
    PREC J;   //< Particle volume ratio at n
    {
      auto source_particle_bin = pbuffer.ch(_0, source_blockno);
      pos[0] = source_particle_bin.val(_0, source_pidib % g_bin_capacity);  //< x
      pos[1] = source_particle_bin.val(_1, source_pidib % g_bin_capacity);  //< y
      pos[2] = source_particle_bin.val(_2, source_pidib % g_bin_capacity);  //< z
      J = 1.0 - source_particle_bin.val(_3, source_pidib % g_bin_capacity);       //< Vo/V
      vel_p[0] = source_particle_bin.val(_4, source_pidib % g_bin_capacity); //< vx
      vel_p[1] = source_particle_bin.val(_5, source_pidib % g_bin_capacity); //< vy
      vel_p[2] = source_particle_bin.val(_6, source_pidib % g_bin_capacity); //< vz
    }
    ivec3 local_base_index = (pos * g_dx_inv_d + 0.5).cast<int>() - 1;
    pvec3 local_pos = pos - local_base_index * g_dx_d;
    base_index = local_base_index;

    pvec3x3 dws;
#pragma unroll 3
    for (int dd = 0; dd < 3; ++dd) {
      PREC d = (local_pos[dd] - static_cast<PREC>(std::floor(local_pos[dd] * g_dx_inv_d + 0.5) - 1) * g_dx_d) * g_dx_inv_d;
      dws(dd, 0) = 0.5 * (1.5 - d) * (1.5 - d);
      d -= 1.0;
      dws(dd, 1) = 0.75 - d * d;
      d = 0.5 + d;
      dws(dd, 2) = 0.5 * d * d;
      local_base_index[dd] = ((local_base_index[dd] - 1) & g_blockmask) + 1;
    }
    pvec3 vel;   //< Stressed, collided grid velocity
    pvec3 vel_FLIP; //< Unstressed, uncollided grid velocity
    pvec9 C;     //< APIC affine matrix, used a few times
    vel.set(0.0); 
    vel_FLIP.set(0.0);
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
          vel_FLIP += vi_n * W; 
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

    pos += dt * (vel + beta * pbuffer.alpha * (vel_p - vel_FLIP));
    vel += pbuffer.alpha * (vel_p - vel_FLIP);

    pvec9 contrib;
    {
      PREC voln = J * pbuffer.volume;
      PREC pressure = (pbuffer.bulk / pbuffer.gamma) * (pow(J, -pbuffer.gamma) - 1.f);
      // PREC pressure = (pbuffer.bulk / pbuffer.gamma) * expm1(-pbuffer.gamma*log1p(-sJ));
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
        auto particle_bin = g_buckets_on_particle_buffer 
            ? next_pbuffer.ch(_0, next_pbuffer._binsts[src_blockno] + pidib / g_bin_capacity) 
            : next_pbuffer.ch(_0, partition._binsts[src_blockno] + pidib / g_bin_capacity);
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
      const int direction_tag = dir_offset(((base_index - 1) / static_cast<int>(g_blocksize) -
                              (local_base_index - 1) / static_cast<int>(g_blocksize)));
      if (g_buckets_on_particle_buffer)
        next_pbuffer.add_advection(partition, local_base_index - 1, direction_tag, pidib);
      else 
        partition.add_advection(local_base_index - 1, direction_tag, pidib);
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

		//Dim of p2gbuffer is (4, 8, 8, 8). So if values are too big, discard whole particle
		if(local_base_index[0] < 0 || local_base_index[1] < 0 || local_base_index[2] < 0 || local_base_index[0] + 2 >= 8 || local_base_index[1] + 2 >= 8 || local_base_index[2] + 2 >= 8) {
			//NOLINTNEXTLINE(cppcoreguidelines-pro-type-vararg) Cuda has no other way to print; Numbers are array indices to be printed
			printf("local_base_index out of range: %d %d %d\n", local_base_index[0], local_base_index[1], local_base_index[2]);
			return;
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
  for (int base = static_cast<int>(threadIdx.x); base < numMViInArena; base += static_cast<int>(blockDim.x)) {
    const char local_block_id = static_cast<char>(base / numMViPerBlock);
    const auto blockno = partition.query(
        ivec3{blockid[0] + ((local_block_id & 4) != 0 ? 1 : 0),
              blockid[1] + ((local_block_id & 2) != 0 ? 1 : 0),
              blockid[2] + ((local_block_id & 1) != 0 ? 1 : 0)});
    // auto grid_block = next_grid.template ch<0>(blockno);
    int channelid = static_cast<int>(base % numMViPerBlock);
    const char c = static_cast<char>(channelid % g_blockvolume);
    const char cz = static_cast<char>(channelid & g_blockmask);
    const char cy = static_cast<char>((channelid >>= g_blockbits) & g_blockmask);
    const char cx = static_cast<char>((channelid >>= g_blockbits) & g_blockmask);
    channelid >>= g_blockbits;
		float val = p2gbuffer[channelid][static_cast<size_t>(static_cast<size_t>(cx) + (local_block_id & 4 ? config::g_blocksize : 0))][static_cast<size_t>(static_cast<size_t>(cy) + (local_block_id & 2 ? config::g_blocksize : 0))][static_cast<size_t>(static_cast<size_t>(cz) + (local_block_id & 1 ? config::g_blocksize : 0))];

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

template <typename Partition, typename Grid>
__global__ void g2p2g(double dt, double newDt, const ivec3 *__restrict__ blocks,
                      const ParticleBuffer<material_e::JBarFluid> pbuffer,
                      ParticleBuffer<material_e::JBarFluid> next_pbuffer,
                      const Partition prev_partition, Partition partition,
                      const Grid grid, Grid next_grid) { }


template <typename Partition, typename Grid>
__global__ void g2p2g(double dt, double newDt, const ivec3 *__restrict__ blocks,
                      const ParticleBuffer<material_e::VonMises> pbuffer,
                      ParticleBuffer<material_e::VonMises> next_pbuffer,
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

  ivec3 blockid;
  int src_blockno;
  if (blocks != nullptr) { 
    blockid = blocks[blockIdx.x]; // Halo block ID
    src_blockno = partition.query(blockid); // Halo block number
  } else { 
    if (partition._haloMarks[blockIdx.x]) return; // Return early if halo block
    blockid = partition._activeKeys[blockIdx.x]; // Non-halo block ID
    src_blockno = blockIdx.x; // Non-halo block number
  }
  if (src_blockno < 0) return; // Return early if negative block number
//  else if (src_blockno > blockcnt) return; // Return early if excessive block number
  int ppb = g_buckets_on_particle_buffer
            ? next_pbuffer._ppbs[src_blockno] 
            : partition._ppbs[src_blockno]; // Particles in block
  if (ppb == 0) return; // Return early if no particles

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
      int advect = (g_buckets_on_particle_buffer)
                    ? next_pbuffer._blockbuckets[src_blockno * g_particle_num_per_block + pidib]
                    : partition._blockbuckets[src_blockno * g_particle_num_per_block + pidib];
      dir_components(advect / g_particle_num_per_block, base_index);
      base_index += blockid;
      source_blockno = prev_partition.query(base_index);
      source_pidib = advect % g_particle_num_per_block; // & (g_particle_num_per_block - 1);
      source_blockno = (g_buckets_on_particle_buffer)
                        ? pbuffer._binsts[source_blockno] + source_pidib / g_bin_capacity
                        : prev_partition._binsts[source_blockno] + source_pidib / g_bin_capacity;
    }
    pvec3 pos;
    {
      auto source_particle_bin = pbuffer.ch(_0, source_blockno);
      pos[0] = source_particle_bin.val(_0, source_pidib % g_bin_capacity);
      pos[1] = source_particle_bin.val(_1, source_pidib % g_bin_capacity);
      pos[2] = source_particle_bin.val(_2, source_pidib % g_bin_capacity);
    }
    ivec3 local_base_index = (pos * g_dx_inv_d + 0.5).cast<int>() - 1;
    pvec3 local_pos = pos - local_base_index * g_dx_d;
    base_index = local_base_index;

    pvec3x3 dws;
#pragma unroll 3
    for (int dd = 0; dd < 3; ++dd) {
      PREC d = (local_pos[dd] - static_cast<PREC>(std::floor(local_pos[dd] * g_dx_inv_d + 0.5) - 1) * g_dx_d) * g_dx_inv_d;
      dws(dd, 0) = 0.5 * (1.5 - d) * (1.5 - d);
      d -= 1.0;
      dws(dd, 1) = 0.75 - d * d;
      d = 0.5 + d;
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
      PREC ID = source_particle_bin.val(_12, source_pidib % g_bin_capacity);
      matrixMatrixMultiplication3d(dws.data(), contrib.data(), F.data());
      {
        auto particle_bin = g_buckets_on_particle_buffer 
            ? next_pbuffer.ch(_0, next_pbuffer._binsts[src_blockno] + pidib / g_bin_capacity) 
            : next_pbuffer.ch(_0, partition._binsts[src_blockno] + pidib / g_bin_capacity);
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
        particle_bin.val(_12, pidib % g_bin_capacity) = ID;

      }
      compute_stress_vonmises(pbuffer.volume, pbuffer.mu, pbuffer.lambda, 
                                    pbuffer.tensile_yield_strength, F, contrib);
      contrib = (C * pbuffer.mass - contrib * newDt) * Dp_inv;
    }

    local_base_index = (pos * g_dx_inv + 0.5f).cast<int>() - 1;
    {
      int direction_tag = dir_offset((base_index - 1) / g_blocksize -
                              (local_base_index - 1) / g_blocksize);
      if (g_buckets_on_particle_buffer)
        next_pbuffer.add_advection(partition, local_base_index - 1, direction_tag, pidib);
      else 
        partition.add_advection(local_base_index - 1, direction_tag, pidib);
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
    int channelid = base % numMViPerBlock; // & (numMViPerBlock - 1);
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
__global__ void g2p2g(double dt, double newDt, const ivec3 *__restrict__ blocks,
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

  ivec3 blockid;
  int src_blockno;
  if (blocks != nullptr) { 
    blockid = blocks[blockIdx.x]; // Halo block ID
    src_blockno = partition.query(blockid); // Halo block number
  } else { 
    if (partition._haloMarks[blockIdx.x]) return; // Return early if halo block
    blockid = partition._activeKeys[blockIdx.x]; // Non-halo block ID
    src_blockno = blockIdx.x; // Non-halo block number
  }
  if (src_blockno < 0) return; // Return early if negative block number
//  else if (src_blockno > blockcnt) return; // Return early if excessive block number
  int ppb = g_buckets_on_particle_buffer
            ? next_pbuffer._ppbs[src_blockno] 
            : partition._ppbs[src_blockno]; // Particles in block
  if (ppb == 0) return; // Return early if no particles

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
      int advect = (g_buckets_on_particle_buffer)
                    ? next_pbuffer._blockbuckets[src_blockno * g_particle_num_per_block + pidib]
                    : partition._blockbuckets[src_blockno * g_particle_num_per_block + pidib];
      dir_components(advect / g_particle_num_per_block, base_index);
      base_index += blockid;
      source_blockno = prev_partition.query(base_index);
      source_pidib = advect % g_particle_num_per_block; // & (g_particle_num_per_block - 1);
      source_blockno = (g_buckets_on_particle_buffer)
                        ? pbuffer._binsts[source_blockno] + source_pidib / g_bin_capacity
                        : prev_partition._binsts[source_blockno] + source_pidib / g_bin_capacity;
    }
    pvec3 pos;
    {
      auto source_particle_bin = pbuffer.ch(_0, source_blockno);
      pos[0] = source_particle_bin.val(_0, source_pidib % g_bin_capacity);
      pos[1] = source_particle_bin.val(_1, source_pidib % g_bin_capacity);
      pos[2] = source_particle_bin.val(_2, source_pidib % g_bin_capacity);
    }
    ivec3 local_base_index = (pos * g_dx_inv_d + 0.5).cast<int>() - 1;
    pvec3 local_pos = pos - local_base_index * g_dx_d;
    base_index = local_base_index;

    pvec3x3 dws;
#pragma unroll 3
    for (int dd = 0; dd < 3; ++dd) {
      PREC d = (local_pos[dd] - static_cast<PREC>(std::floor(local_pos[dd] * g_dx_inv_d + 0.5) - 1) * g_dx_d) * g_dx_inv_d;
      dws(dd, 0) = 0.5 * (1.5 - d) * (1.5 - d);
      d -= 1.0;
      dws(dd, 1) = 0.75 - d * d;
      d = 0.5 + d;
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
      PREC ID = source_particle_bin.val(_12, source_pidib % g_bin_capacity);
      matrixMatrixMultiplication3d(dws.data(), contrib.data(), F.data());
      {
        auto particle_bin = g_buckets_on_particle_buffer 
            ? next_pbuffer.ch(_0, next_pbuffer._binsts[src_blockno] + pidib / g_bin_capacity) 
            : next_pbuffer.ch(_0, partition._binsts[src_blockno] + pidib / g_bin_capacity);
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
        particle_bin.val(_12, pidib % g_bin_capacity) = ID;

      }
      compute_stress_fixedcorotated(pbuffer.volume, pbuffer.mu, pbuffer.lambda,
                                    F, contrib);
      contrib = (C * pbuffer.mass - contrib * newDt) * Dp_inv;
    }

    local_base_index = (pos * g_dx_inv + 0.5f).cast<int>() - 1;
    {
      int direction_tag = dir_offset((base_index - 1) / g_blocksize -
                              (local_base_index - 1) / g_blocksize);
      if (g_buckets_on_particle_buffer)
        next_pbuffer.add_advection(partition, local_base_index - 1, direction_tag, pidib);
      else 
        partition.add_advection(local_base_index - 1, direction_tag, pidib);
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
    int channelid = base % numMViPerBlock; // & (numMViPerBlock - 1);
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
template <typename Partition, typename Grid>
__global__ void 
__launch_bounds__(g_particle_batch, 3)
g2p2g(double dt, double newDt, const ivec3 *__restrict__ blocks,
                      const ParticleBuffer<material_e::FixedCorotated_ASFLIP> pbuffer,
                      ParticleBuffer<material_e::FixedCorotated_ASFLIP> next_pbuffer,
                      const Partition prev_partition, Partition partition,
                      const Grid grid, Grid next_grid) {
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
    blockid = blocks[blockIdx.x]; // Halo block ID
    src_blockno = partition.query(blockid); // Halo block number
  } else { 
    if (partition._haloMarks[blockIdx.x]) return; // Return early if halo block
    blockid = partition._activeKeys[blockIdx.x]; // Non-halo block ID
    src_blockno = blockIdx.x; // Non-halo block number
  }
  if (src_blockno < 0) return; // Return early if negative block number
//  else if (src_blockno > blockcnt) return; // Return early if excessive block number
  int ppb = g_buckets_on_particle_buffer
            ? next_pbuffer._ppbs[src_blockno] 
            : partition._ppbs[src_blockno]; // Particles in block
  if (ppb == 0) return; // Return early if no particles

  
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
  for (int pidib = threadIdx.x; pidib < ppb; pidib += blockDim.x) {
    int source_blockno, source_pidib;
    ivec3 base_index;
    {
      int advect = (g_buckets_on_particle_buffer)
                    ? next_pbuffer._blockbuckets[src_blockno * g_particle_num_per_block + pidib]
                    : partition._blockbuckets[src_blockno * g_particle_num_per_block + pidib];
      dir_components(advect / g_particle_num_per_block, base_index);
      base_index += blockid;
      source_blockno = prev_partition.query(base_index);
      source_pidib = advect % g_particle_num_per_block; // & (g_particle_num_per_block - 1);
      source_blockno = (g_buckets_on_particle_buffer)
                        ? pbuffer._binsts[source_blockno] + source_pidib / g_bin_capacity
                        : prev_partition._binsts[source_blockno] + source_pidib / g_bin_capacity;
    }
    pvec3 pos, vel_p;
    {
      auto source_particle_bin = pbuffer.ch(_0, source_blockno);
      pos[0] = source_particle_bin.val(_0, source_pidib % g_bin_capacity);
      pos[1] = source_particle_bin.val(_1, source_pidib % g_bin_capacity);
      pos[2] = source_particle_bin.val(_2, source_pidib % g_bin_capacity);
      vel_p[0] = source_particle_bin.val(_12, source_pidib % g_bin_capacity);
      vel_p[1] = source_particle_bin.val(_13, source_pidib % g_bin_capacity);
      vel_p[2] = source_particle_bin.val(_14, source_pidib % g_bin_capacity);
    }
    ivec3 local_base_index = (pos * g_dx_inv_d + 0.5).cast<int>() - 1;
    pvec3 local_pos = pos - local_base_index * g_dx_d;
    base_index = local_base_index;

    pvec3x3 dws;
#pragma unroll 3
    for (int dd = 0; dd < 3; ++dd) {
      PREC d = (local_pos[dd] - static_cast<PREC>(std::floor(local_pos[dd] * g_dx_inv_d + 0.5) - 1) * g_dx_d) * g_dx_inv_d;
      dws(dd, 0) = 0.5 * (1.5 - d) * (1.5 - d);
      d -= 1.0;
      dws(dd, 1) = 0.75 - d * d;
      d = 0.5 + d;
      dws(dd, 2) = 0.5 * d * d;
      local_base_index[dd] = ((local_base_index[dd] - 1) & g_blockmask) + 1;
    }
    pvec3 vel, vel_FLIP;
    vel.set(0.0);
    vel_FLIP.set(0.0);
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
          vel_FLIP += vi_n * W;
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
      PREC ID = source_particle_bin.val(_15, source_pidib % g_bin_capacity);

      matrixMatrixMultiplication3d(dws.data(), contrib.data(), F.data());
      PREC J  = matrixDeterminant3d(F.data());
      {
        // > or >= ?
        PREC beta = (J > 1.0) ? pbuffer.beta_max : pbuffer.beta_min; //< ASFLIP pos. corr.
        pos += dt * (vel + beta * pbuffer.alpha * (vel_p - vel_FLIP)); //< pos update
        vel += pbuffer.alpha * (vel_p - vel_FLIP); //< vel update
      }

      {
        auto particle_bin = g_buckets_on_particle_buffer 
            ? next_pbuffer.ch(_0, next_pbuffer._binsts[src_blockno] + pidib / g_bin_capacity) 
            : next_pbuffer.ch(_0, partition._binsts[src_blockno] + pidib / g_bin_capacity);
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
        particle_bin.val(_15, pidib % g_bin_capacity) = ID;
      }
      compute_stress_fixedcorotated(pbuffer.volume, pbuffer.mu, pbuffer.lambda,
                                    F, contrib);
      contrib = (C * pbuffer.mass - contrib * newDt) * Dp_inv;
    }

    local_base_index = (pos * g_dx_inv + 0.5f).cast<int>() - 1;
    {
      int direction_tag = dir_offset((base_index - 1) / g_blocksize -
                              (local_base_index - 1) / g_blocksize);
      if (g_buckets_on_particle_buffer)
        next_pbuffer.add_advection(partition, local_base_index - 1, direction_tag, pidib);
      else 
        partition.add_advection(local_base_index - 1, direction_tag, pidib);
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


template <typename Partition, typename Grid>
__global__ void g2p2g(double dt, double newDt, const ivec3 *__restrict__ blocks,
                      const ParticleBuffer<material_e::NeoHookean> pbuffer,
                      ParticleBuffer<material_e::NeoHookean> next_pbuffer,
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

  ivec3 blockid;
  int src_blockno;
  if (blocks != nullptr) { 
    blockid = blocks[blockIdx.x]; // Halo block ID
    src_blockno = partition.query(blockid); // Halo block number
  } else { 
    if (partition._haloMarks[blockIdx.x]) return; // Return early if halo block
    blockid = partition._activeKeys[blockIdx.x]; // Non-halo block ID
    src_blockno = blockIdx.x; // Non-halo block number
  }
  if (src_blockno < 0) return; // Return early if negative block number
//  else if (src_blockno > blockcnt) return; // Return early if excessive block number
  int ppb = g_buckets_on_particle_buffer
            ? next_pbuffer._ppbs[src_blockno] 
            : partition._ppbs[src_blockno]; // Particles in block
  if (ppb == 0) return; // Return early if no particles

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
      int advect = (g_buckets_on_particle_buffer)
                    ? next_pbuffer._blockbuckets[src_blockno * g_particle_num_per_block + pidib]
                    : partition._blockbuckets[src_blockno * g_particle_num_per_block + pidib];
      dir_components(advect / g_particle_num_per_block, base_index);
      base_index += blockid;
      source_blockno = prev_partition.query(base_index);
      source_pidib = advect % g_particle_num_per_block; // & (g_particle_num_per_block - 1);
      source_blockno = (g_buckets_on_particle_buffer)
                        ? pbuffer._binsts[source_blockno] + source_pidib / g_bin_capacity
                        : prev_partition._binsts[source_blockno] + source_pidib / g_bin_capacity;
    }
    pvec3 pos;
    {
      auto source_particle_bin = pbuffer.ch(_0, source_blockno);
      pos[0] = source_particle_bin.val(_0, source_pidib % g_bin_capacity);
      pos[1] = source_particle_bin.val(_1, source_pidib % g_bin_capacity);
      pos[2] = source_particle_bin.val(_2, source_pidib % g_bin_capacity);
    }
    ivec3 local_base_index = (pos * g_dx_inv_d + 0.5).cast<int>() - 1;
    pvec3 local_pos = pos - local_base_index * g_dx_d;
    base_index = local_base_index;

    pvec3x3 dws;
#pragma unroll 3
    for (int dd = 0; dd < 3; ++dd) {
      PREC d = (local_pos[dd] - static_cast<PREC>(std::floor(local_pos[dd] * g_dx_inv_d + 0.5) - 1) * g_dx_d) * g_dx_inv_d;
      dws(dd, 0) = 0.5 * (1.5 - d) * (1.5 - d);
      d -= 1.0;
      dws(dd, 1) = 0.75 - d * d;
      d = 0.5 + d;
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
      PREC ID = source_particle_bin.val(_12, source_pidib % g_bin_capacity);
      matrixMatrixMultiplication3d(dws.data(), contrib.data(), F.data());
      {
        auto particle_bin = g_buckets_on_particle_buffer 
            ? next_pbuffer.ch(_0, next_pbuffer._binsts[src_blockno] + pidib / g_bin_capacity) 
            : next_pbuffer.ch(_0, partition._binsts[src_blockno] + pidib / g_bin_capacity);
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
        particle_bin.val(_12, pidib % g_bin_capacity) = ID;

      }
      compute_stress_neohookean(pbuffer.volume, pbuffer.mu, pbuffer.lambda,
                                    F, contrib);
      contrib = (C * pbuffer.mass - contrib * newDt) * Dp_inv;
    }

    local_base_index = (pos * g_dx_inv + 0.5f).cast<int>() - 1;
    {
      int direction_tag = dir_offset((base_index - 1) / g_blocksize -
                              (local_base_index - 1) / g_blocksize);
      if (g_buckets_on_particle_buffer)
        next_pbuffer.add_advection(partition, local_base_index - 1, direction_tag, pidib);
      else 
        partition.add_advection(local_base_index - 1, direction_tag, pidib);
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
    int channelid = base % numMViPerBlock; // & (numMViPerBlock - 1);
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
template <typename Partition, typename Grid>
__global__ void 
__launch_bounds__(g_particle_batch, 3)
g2p2g(double dt, double newDt, const ivec3 *__restrict__ blocks,
                      const ParticleBuffer<material_e::NeoHookean_ASFLIP> pbuffer,
                      ParticleBuffer<material_e::NeoHookean_ASFLIP> next_pbuffer,
                      const Partition prev_partition, Partition partition,
                      const Grid grid, Grid next_grid) {
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
    blockid = blocks[blockIdx.x]; // Halo block ID
    src_blockno = partition.query(blockid); // Halo block number
  } else { 
    if (partition._haloMarks[blockIdx.x]) return; // Return early if halo block
    blockid = partition._activeKeys[blockIdx.x]; // Non-halo block ID
    src_blockno = blockIdx.x; // Non-halo block number
  }
  if (src_blockno < 0) return; // Return early if negative block number
//  else if (src_blockno > blockcnt) return; // Return early if excessive block number
  int ppb = g_buckets_on_particle_buffer
            ? next_pbuffer._ppbs[src_blockno] 
            : partition._ppbs[src_blockno]; // Particles in block
  if (ppb == 0) return; // Return early if no particles

  
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
  for (int pidib = threadIdx.x; pidib < ppb; pidib += blockDim.x) {
    int source_blockno, source_pidib;
    ivec3 base_index;
    {
      int advect = (g_buckets_on_particle_buffer)
                    ? next_pbuffer._blockbuckets[src_blockno * g_particle_num_per_block + pidib]
                    : partition._blockbuckets[src_blockno * g_particle_num_per_block + pidib];
      dir_components(advect / g_particle_num_per_block, base_index);
      base_index += blockid;
      source_blockno = prev_partition.query(base_index);
      source_pidib = advect % g_particle_num_per_block; // & (g_particle_num_per_block - 1);
      source_blockno = (g_buckets_on_particle_buffer)
                        ? pbuffer._binsts[source_blockno] + source_pidib / g_bin_capacity
                        : prev_partition._binsts[source_blockno] + source_pidib / g_bin_capacity;
    }
    pvec3 pos, vel_p;
    {
      auto source_particle_bin = pbuffer.ch(_0, source_blockno);
      pos[0] = source_particle_bin.val(_0, source_pidib % g_bin_capacity);
      pos[1] = source_particle_bin.val(_1, source_pidib % g_bin_capacity);
      pos[2] = source_particle_bin.val(_2, source_pidib % g_bin_capacity);
      vel_p[0] = source_particle_bin.val(_12, source_pidib % g_bin_capacity);
      vel_p[1] = source_particle_bin.val(_13, source_pidib % g_bin_capacity);
      vel_p[2] = source_particle_bin.val(_14, source_pidib % g_bin_capacity);
    }
    ivec3 local_base_index = (pos * g_dx_inv_d + 0.5).cast<int>() - 1;
    pvec3 local_pos = pos - local_base_index * g_dx_d;
    base_index = local_base_index;

    pvec3x3 dws;
#pragma unroll 3
    for (int dd = 0; dd < 3; ++dd) {
      PREC d = (local_pos[dd] - static_cast<PREC>(std::floor(local_pos[dd] * g_dx_inv_d + 0.5) - 1) * g_dx_d) * g_dx_inv_d;
      dws(dd, 0) = 0.5 * (1.5 - d) * (1.5 - d);
      d -= 1.0;
      dws(dd, 1) = 0.75 - d * d;
      d = 0.5 + d;
      dws(dd, 2) = 0.5 * d * d;
      local_base_index[dd] = ((local_base_index[dd] - 1) & g_blockmask) + 1;
    }
    pvec3 vel, vel_FLIP;
    vel.set(0.0);
    vel_FLIP.set(0.0);
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
          vel_FLIP += vi_n * W;
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
      PREC ID = source_particle_bin.val(_15, source_pidib % g_bin_capacity);

      matrixMatrixMultiplication3d(dws.data(), contrib.data(), F.data());
      PREC J  = matrixDeterminant3d(F.data());
      {
        // > or >= ?
        PREC beta = (J > 1.0) ? pbuffer.beta_max : pbuffer.beta_min; //< ASFLIP pos. corr.
        pos += dt * (vel + beta * pbuffer.alpha * (vel_p - vel_FLIP)); //< pos update
        vel += pbuffer.alpha * (vel_p - vel_FLIP); //< vel update
      }

      {
        auto particle_bin = g_buckets_on_particle_buffer 
            ? next_pbuffer.ch(_0, next_pbuffer._binsts[src_blockno] + pidib / g_bin_capacity) 
            : next_pbuffer.ch(_0, partition._binsts[src_blockno] + pidib / g_bin_capacity);
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
        particle_bin.val(_15, pidib % g_bin_capacity) = ID;
      }
      compute_stress_neohookean(pbuffer.volume, pbuffer.mu, pbuffer.lambda,
                                    F, contrib);
      contrib = (C * pbuffer.mass - contrib * newDt) * Dp_inv;
    }

    local_base_index = (pos * g_dx_inv + 0.5f).cast<int>() - 1;
    {
      int direction_tag = dir_offset((base_index - 1) / g_blocksize -
                              (local_base_index - 1) / g_blocksize);
      if (g_buckets_on_particle_buffer)
        next_pbuffer.add_advection(partition, local_base_index - 1, direction_tag, pidib);
      else 
        partition.add_advection(local_base_index - 1, direction_tag, pidib);
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


template <typename Partition, typename Grid>
__global__ void g2p2g(double dt, double newDt, const ivec3 *__restrict__ blocks,
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

  ivec3 blockid;
  int src_blockno;
  if (blocks != nullptr) { 
    blockid = blocks[blockIdx.x]; // Halo block ID
    src_blockno = partition.query(blockid); // Halo block number
  } else { 
    if (partition._haloMarks[blockIdx.x]) return; // Return early if halo block
    blockid = partition._activeKeys[blockIdx.x]; // Non-halo block ID
    src_blockno = blockIdx.x; // Non-halo block number
  }
  if (src_blockno < 0) return; // Return early if negative block number
//  else if (src_blockno > blockcnt) return; // Return early if excessive block number
  int ppb = g_buckets_on_particle_buffer
            ? next_pbuffer._ppbs[src_blockno] 
            : partition._ppbs[src_blockno]; // Particles in block
  if (ppb == 0) return; // Return early if no particles


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
      int advect = (g_buckets_on_particle_buffer)
                    ? next_pbuffer._blockbuckets[src_blockno * g_particle_num_per_block + pidib]
                    : partition._blockbuckets[src_blockno * g_particle_num_per_block + pidib];
      dir_components(advect / g_particle_num_per_block, base_index);
      base_index += blockid;
      source_blockno = prev_partition.query(base_index);
      source_pidib = advect % g_particle_num_per_block; // & (g_particle_num_per_block - 1);
      source_blockno = (g_buckets_on_particle_buffer)
                        ? pbuffer._binsts[source_blockno] + source_pidib / g_bin_capacity
                        : prev_partition._binsts[source_blockno] + source_pidib / g_bin_capacity;
    }
    pvec3 pos;
    {
      auto source_particle_bin = pbuffer.ch(_0, source_blockno);
      pos[0] = source_particle_bin.val(_0, source_pidib % g_bin_capacity);
      pos[1] = source_particle_bin.val(_1, source_pidib % g_bin_capacity);
      pos[2] = source_particle_bin.val(_2, source_pidib % g_bin_capacity);
    }
    ivec3 local_base_index = (pos * g_dx_inv_d + 0.5).cast<int>() - 1;
    pvec3 local_pos = pos - local_base_index * g_dx_d;
    base_index = local_base_index;

    pvec3x3 dws;
#pragma unroll 3
    for (int dd = 0; dd < 3; ++dd) {
      PREC d = (local_pos[dd] - static_cast<PREC>(std::floor(local_pos[dd] * g_dx_inv_d + 0.5) - 1) * g_dx_d) * g_dx_inv_d;
      dws(dd, 0) = 0.5 * (1.5 - d) * (1.5 - d);
      d -= 1.0;
      dws(dd, 1) = 0.75 - d * d;
      d = 0.5 + d;
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
      PREC ID = source_particle_bin.val(_13, source_pidib % g_bin_capacity);

      matrixMatrixMultiplication3d(dws.data(), contrib.data(), F.data());
      PREC J = matrixDeterminant3d(F.data());
      pos += dt * vel;

      compute_stress_sand(pbuffer.volume, pbuffer.mu, pbuffer.lambda,
                          pbuffer.cohesion, pbuffer.beta, pbuffer.yieldSurface,
                          pbuffer.volumeCorrection, logJp, F, contrib);
      {
        auto particle_bin = g_buckets_on_particle_buffer 
            ? next_pbuffer.ch(_0, next_pbuffer._binsts[src_blockno] + pidib / g_bin_capacity) 
            : next_pbuffer.ch(_0, partition._binsts[src_blockno] + pidib / g_bin_capacity);
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
        particle_bin.val(_13, pidib % g_bin_capacity) = ID;

      }

      contrib = (C * pbuffer.mass - contrib * newDt) * Dp_inv;
    }

    local_base_index = (pos * g_dx_inv_d + 0.5).cast<int>() - 1;
    {
      int direction_tag = dir_offset((base_index - 1) / g_blocksize -
                              (local_base_index - 1) / g_blocksize);
      if (g_buckets_on_particle_buffer)
        next_pbuffer.add_advection(partition, local_base_index - 1, direction_tag, pidib);
      else 
        partition.add_advection(local_base_index - 1, direction_tag, pidib);
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
  }
}


template <typename Partition, typename Grid>
__global__ void g2p2g(double dt, double newDt, const ivec3 *__restrict__ blocks,
                      const ParticleBuffer<material_e::Sand_ASFLIP> pbuffer,
                      ParticleBuffer<material_e::Sand_ASFLIP> next_pbuffer,
                      const Partition prev_partition, Partition partition,
                      const Grid grid, Grid next_grid) {
  static constexpr uint64_t numViPerBlock = g_blockvolume * 6;
  static constexpr uint64_t numViInArena = numViPerBlock << 3;

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
      *reinterpret_cast<MViArena>(shmem + numViInArena * sizeof(float));

  ivec3 blockid;
  int src_blockno;
  if (blocks != nullptr) { 
    blockid = blocks[blockIdx.x]; // Halo block ID
    src_blockno = partition.query(blockid); // Halo block number
  } else { 
    if (partition._haloMarks[blockIdx.x]) return; // Return early if halo block
    blockid = partition._activeKeys[blockIdx.x]; // Non-halo block ID
    src_blockno = blockIdx.x; // Non-halo block number
  }
  if (src_blockno < 0) return; // Return early if negative block number
//  else if (src_blockno > blockcnt) return; // Return early if excessive block number
  int ppb = g_buckets_on_particle_buffer
            ? next_pbuffer._ppbs[src_blockno] 
            : partition._ppbs[src_blockno]; // Particles in block
  if (ppb == 0) return; // Return early if no particles


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

  for (int pidib = threadIdx.x; pidib < ppb; pidib += blockDim.x) {
    int source_blockno, source_pidib;
    ivec3 base_index;
    {
      int advect = (g_buckets_on_particle_buffer)
                    ? next_pbuffer._blockbuckets[src_blockno * g_particle_num_per_block + pidib]
                    : partition._blockbuckets[src_blockno * g_particle_num_per_block + pidib];
      dir_components(advect / g_particle_num_per_block, base_index);
      base_index += blockid;
      source_blockno = prev_partition.query(base_index);
      source_pidib = advect % g_particle_num_per_block; // & (g_particle_num_per_block - 1);
      source_blockno = (g_buckets_on_particle_buffer)
                        ? pbuffer._binsts[source_blockno] + source_pidib / g_bin_capacity
                        : prev_partition._binsts[source_blockno] + source_pidib / g_bin_capacity;
    }
    pvec3 pos, vel_p;
    {
      auto source_particle_bin = pbuffer.ch(_0, source_blockno);
      pos[0] = source_particle_bin.val(_0, source_pidib % g_bin_capacity);
      pos[1] = source_particle_bin.val(_1, source_pidib % g_bin_capacity);
      pos[2] = source_particle_bin.val(_2, source_pidib % g_bin_capacity);
      vel_p[0] = source_particle_bin.val(_13, source_pidib % g_bin_capacity);
      vel_p[1] = source_particle_bin.val(_14, source_pidib % g_bin_capacity);
      vel_p[2] = source_particle_bin.val(_15, source_pidib % g_bin_capacity);
    }
    ivec3 local_base_index = (pos * g_dx_inv_d + 0.5).cast<int>() - 1;
    pvec3 local_pos = pos - local_base_index * g_dx_d;
    base_index = local_base_index;

    pvec3x3 dws;
#pragma unroll 3
    for (int dd = 0; dd < 3; ++dd) {
      PREC d = (local_pos[dd] - static_cast<PREC>(std::floor(local_pos[dd] * g_dx_inv_d + 0.5) - 1) * g_dx_d) * g_dx_inv_d;
      dws(dd, 0) = 0.5 * (1.5 - d) * (1.5 - d);
      d -= 1.0;
      dws(dd, 1) = 0.75 - d * d;
      d = 0.5 + d;
      dws(dd, 2) = 0.5 * d * d;
      local_base_index[dd] = ((local_base_index[dd] - 1) & g_blockmask) + 1;
    }
    pvec3 vel, vel_FLIP;
    vel.set(0.0);
    vel_FLIP.set(0.0);
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
          vel_FLIP += vi_n * W;
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
      PREC ID = source_particle_bin.val(_16, source_pidib % g_bin_capacity);

      matrixMatrixMultiplication3d(dws.data(), contrib.data(), F.data());
      PREC J = matrixDeterminant3d(F.data());
      PREC beta;
      if (J >= 1.0) beta = pbuffer.beta_max; //< beta max
      else beta = pbuffer.beta_min;          //< beta min
      pos += dt * (vel + beta * pbuffer.alpha * (vel_p - vel_FLIP)); //< pos update
      vel += pbuffer.alpha * (vel_p - vel_FLIP); //< vel update

      compute_stress_sand(pbuffer.volume, pbuffer.mu, pbuffer.lambda,
                          pbuffer.cohesion, pbuffer.beta, pbuffer.yieldSurface,
                          pbuffer.volumeCorrection, logJp, F, contrib);
      {
        auto particle_bin = g_buckets_on_particle_buffer 
            ? next_pbuffer.ch(_0, next_pbuffer._binsts[src_blockno] + pidib / g_bin_capacity) 
            : next_pbuffer.ch(_0, partition._binsts[src_blockno] + pidib / g_bin_capacity);
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
        particle_bin.val(_13, pidib % g_bin_capacity) = vel[0];
        particle_bin.val(_14, pidib % g_bin_capacity) = vel[1];
        particle_bin.val(_15, pidib % g_bin_capacity) = vel[2];
        particle_bin.val(_16, pidib % g_bin_capacity) = ID;

      }

      contrib = (C * pbuffer.mass - contrib * newDt) * Dp_inv;
    }

    local_base_index = (pos * g_dx_inv_d + 0.5).cast<int>() - 1;
    {
      int direction_tag = dir_offset((base_index - 1) / g_blocksize -
                              (local_base_index - 1) / g_blocksize);
      if (g_buckets_on_particle_buffer)
        next_pbuffer.add_advection(partition, local_base_index - 1, direction_tag, pidib);
      else 
        partition.add_advection(local_base_index - 1, direction_tag, pidib);
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

template <typename Partition, typename Grid>
__global__ void g2p2g(double dt, double newDt, const ivec3 *__restrict__ blocks,
                      const ParticleBuffer<material_e::Sand_ASFLIP_FBAR> pbuffer,
                      ParticleBuffer<material_e::Sand_ASFLIP_FBAR> next_pbuffer,
                      const Partition prev_partition, Partition partition,
                      const Grid grid, Grid next_grid) {
  static constexpr uint64_t numViPerBlock = g_blockvolume * 6;
  static constexpr uint64_t numViInArena = numViPerBlock << 3;

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
      *reinterpret_cast<MViArena>(shmem + numViInArena * sizeof(float));

  ivec3 blockid;
  int src_blockno;
  if (blocks != nullptr) { 
    blockid = blocks[blockIdx.x]; // Halo block ID
    src_blockno = partition.query(blockid); // Halo block number
  } else { 
    if (partition._haloMarks[blockIdx.x]) return; // Return early if halo block
    blockid = partition._activeKeys[blockIdx.x]; // Non-halo block ID
    src_blockno = blockIdx.x; // Non-halo block number
  }
  if (src_blockno < 0) return; // Return early if negative block number
//  else if (src_blockno > blockcnt) return; // Return early if excessive block number
  int ppb = g_buckets_on_particle_buffer
            ? next_pbuffer._ppbs[src_blockno] 
            : partition._ppbs[src_blockno]; // Particles in block
  if (ppb == 0) return; // Return early if no particles


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

  for (int pidib = threadIdx.x; pidib < ppb; pidib += blockDim.x) {
    int source_blockno, source_pidib;
    ivec3 base_index;
    {
      int advect = (g_buckets_on_particle_buffer)
                    ? next_pbuffer._blockbuckets[src_blockno * g_particle_num_per_block + pidib]
                    : partition._blockbuckets[src_blockno * g_particle_num_per_block + pidib];
      dir_components(advect / g_particle_num_per_block, base_index);
      base_index += blockid;
      source_blockno = prev_partition.query(base_index);
      source_pidib = advect % g_particle_num_per_block; // & (g_particle_num_per_block - 1);
      source_blockno = (g_buckets_on_particle_buffer)
                        ? pbuffer._binsts[source_blockno] + source_pidib / g_bin_capacity
                        : prev_partition._binsts[source_blockno] + source_pidib / g_bin_capacity;
    }
    pvec3 pos, vel_p;
    {
      auto source_particle_bin = pbuffer.ch(_0, source_blockno);
      pos[0] = source_particle_bin.val(_0, source_pidib % g_bin_capacity);
      pos[1] = source_particle_bin.val(_1, source_pidib % g_bin_capacity);
      pos[2] = source_particle_bin.val(_2, source_pidib % g_bin_capacity);
      vel_p[0] = source_particle_bin.val(_13, source_pidib % g_bin_capacity);
      vel_p[1] = source_particle_bin.val(_14, source_pidib % g_bin_capacity);
      vel_p[2] = source_particle_bin.val(_15, source_pidib % g_bin_capacity);
    }
    ivec3 local_base_index = (pos * g_dx_inv_d + 0.5).cast<int>() - 1;
    pvec3 local_pos = pos - local_base_index * g_dx_d;
    base_index = local_base_index;

    pvec3x3 dws;
#pragma unroll 3
    for (int dd = 0; dd < 3; ++dd) {
      PREC d = (local_pos[dd] - static_cast<PREC>(std::floor(local_pos[dd] * g_dx_inv_d + 0.5) - 1) * g_dx_d) * g_dx_inv_d;
      dws(dd, 0) = 0.5 * (1.5 - d) * (1.5 - d);
      d -= 1.0;
      dws(dd, 1) = 0.75 - d * d;
      d = 0.5 + d;
      dws(dd, 2) = 0.5 * d * d;
      local_base_index[dd] = ((local_base_index[dd] - 1) & g_blockmask) + 1;
    }
    pvec3 vel, vel_FLIP;
    vel.set(0.0);
    vel_FLIP.set(0.0);
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
          vel_FLIP += vi_n * W;
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
      PREC ID = source_particle_bin.val(_17, source_pidib % g_bin_capacity);

      matrixMatrixMultiplication3d(dws.data(), contrib.data(), F.data());
      PREC J = matrixDeterminant3d(F.data());
      PREC beta;
      if (J >= 1.0) beta = pbuffer.beta_max; //< beta max
      else beta = pbuffer.beta_min;          //< beta min
      pos += dt * (vel + beta * pbuffer.alpha * (vel_p - vel_FLIP)); //< pos update
      vel += pbuffer.alpha * (vel_p - vel_FLIP); //< vel update

      compute_stress_sand(pbuffer.volume, pbuffer.mu, pbuffer.lambda,
                          pbuffer.cohesion, pbuffer.beta, pbuffer.yieldSurface,
                          pbuffer.volumeCorrection, logJp, F, contrib);
      {
        auto particle_bin = g_buckets_on_particle_buffer 
            ? next_pbuffer.ch(_0, next_pbuffer._binsts[src_blockno] + pidib / g_bin_capacity) 
            : next_pbuffer.ch(_0, partition._binsts[src_blockno] + pidib / g_bin_capacity);
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
        particle_bin.val(_13, pidib % g_bin_capacity) = vel[0];
        particle_bin.val(_14, pidib % g_bin_capacity) = vel[1];
        particle_bin.val(_15, pidib % g_bin_capacity) = vel[2];
        // sJBar
        particle_bin.val(_17, pidib % g_bin_capacity) = ID;

      }

      contrib = (C * pbuffer.mass - contrib * newDt) * Dp_inv;
    }

    local_base_index = (pos * g_dx_inv_d + 0.5).cast<int>() - 1;
    {
      int direction_tag = dir_offset((base_index - 1) / g_blocksize -
                              (local_base_index - 1) / g_blocksize);
      if (g_buckets_on_particle_buffer)
        next_pbuffer.add_advection(partition, local_base_index - 1, direction_tag, pidib);
      else 
        partition.add_advection(local_base_index - 1, direction_tag, pidib);
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


template <typename Partition, typename Grid>
__global__ void g2p2g(double dt, double newDt, const ivec3 *__restrict__ blocks,
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

  ivec3 blockid;
  int src_blockno;
  if (blocks != nullptr) { 
    blockid = blocks[blockIdx.x]; // Halo block ID
    src_blockno = partition.query(blockid); // Halo block number
  } else { 
    if (partition._haloMarks[blockIdx.x]) return; // Return early if halo block
    blockid = partition._activeKeys[blockIdx.x]; // Non-halo block ID
    src_blockno = blockIdx.x; // Non-halo block number
  }
  if (src_blockno < 0) return; // Return early if negative block number
//  else if (src_blockno > blockcnt) return; // Return early if excessive block number
  int ppb = g_buckets_on_particle_buffer
            ? next_pbuffer._ppbs[src_blockno] 
            : partition._ppbs[src_blockno]; // Particles in block
  if (ppb == 0) return; // Return early if no particles


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
    p2gbuffer[loc >> arenabits][x][y][z] = static_cast<PREC_G>(0);
  }
  __syncthreads();

  for (int pidib = threadIdx.x; pidib < ppb; pidib += blockDim.x) {
    int source_blockno, source_pidib;
    ivec3 base_index;
    {
      int advect = (g_buckets_on_particle_buffer)
                    ? next_pbuffer._blockbuckets[src_blockno * g_particle_num_per_block + pidib]
                    : partition._blockbuckets[src_blockno * g_particle_num_per_block + pidib];
      dir_components(advect / g_particle_num_per_block, base_index);
      base_index += blockid;
      source_blockno = prev_partition.query(base_index);
      source_pidib = advect % g_particle_num_per_block; // & (g_particle_num_per_block - 1);
      source_blockno = (g_buckets_on_particle_buffer)
                        ? pbuffer._binsts[source_blockno] + source_pidib / g_bin_capacity
                        : prev_partition._binsts[source_blockno] + source_pidib / g_bin_capacity;
    }
    pvec3 pos;
    {
      auto source_particle_bin = pbuffer.ch(_0, source_blockno);
      pos[0] = source_particle_bin.val(_0, source_pidib % g_bin_capacity);
      pos[1] = source_particle_bin.val(_1, source_pidib % g_bin_capacity);
      pos[2] = source_particle_bin.val(_2, source_pidib % g_bin_capacity);
    }
    ivec3 local_base_index = (pos * g_dx_inv_d + 0.5).cast<int>() - 1;
    pvec3 local_pos = pos - local_base_index * g_dx_d;
    base_index = local_base_index;

    pvec3x3 dws;
#pragma unroll 3
    for (int dd = 0; dd < 3; ++dd) {
      PREC d = (local_pos[dd] - static_cast<PREC>(std::floor(local_pos[dd] * g_dx_inv_d + 0.5) - 1) * g_dx_d) * g_dx_inv_d;
      dws(dd, 0) = 0.5 * (1.5 - d) * (1.5 - d);
      d -= 1.0;
      dws(dd, 1) = 0.75 - d * d;
      d = 0.5 + d;
      dws(dd, 2) = 0.5 * d * d;
      local_base_index[dd] = ((local_base_index[dd] - 1) & g_blockmask) + 1;
    }
    pvec3 vel;
    pvec9 C;
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
      // Incremental deformation gradient FInc, (F^n+1 = FInc : F^n)
#pragma unroll 9
    for (int d = 0; d < 9; ++d)
      dws.val(d) = C[d] * dt * Dp_inv + ((d & 0x3) ? static_cast<PREC>(0) : static_cast<PREC>(1));

    pvec9 contrib;
    {
      pvec9 F;
      auto source_particle_bin = pbuffer.ch(_0, source_blockno);
      // vel x,y,z not read in here
      contrib[0] = source_particle_bin.val(_3, source_pidib % g_bin_capacity);
      contrib[1] = source_particle_bin.val(_4, source_pidib % g_bin_capacity);
      contrib[2] = source_particle_bin.val(_5, source_pidib % g_bin_capacity);
      contrib[3] = source_particle_bin.val(_6, source_pidib % g_bin_capacity);
      contrib[4] = source_particle_bin.val(_7, source_pidib % g_bin_capacity);
      contrib[5] = source_particle_bin.val(_8, source_pidib % g_bin_capacity);
      contrib[6] = source_particle_bin.val(_9, source_pidib % g_bin_capacity);
      contrib[7] = source_particle_bin.val(_10, source_pidib % g_bin_capacity);
      contrib[8] = source_particle_bin.val(_11, source_pidib % g_bin_capacity);
      PREC logJp = source_particle_bin.val(_12, source_pidib % g_bin_capacity);
      PREC ID = source_particle_bin.val(_13, source_pidib % g_bin_capacity);

      // Deformation gradient update to F_p^n+1, (F_p^n+1 = FInc : F_p^n)
      matrixMatrixMultiplication3d(dws.data(), contrib.data(), F.data());
      PREC J = matrixDeterminant3d(F.data()); // J_p^n+1 = det(F_p^n+1)
      
      // Particle advection
      pos += dt * vel; // < pos update
      
      // Update particle stress, adjust for plasticity in deformation gradient
      compute_stress_nacc(pbuffer.volume, pbuffer.mu, pbuffer.lambda,
                            pbuffer.bm, pbuffer.xi, pbuffer.beta, pbuffer.Msqr,
                            pbuffer.hardeningOn, logJp, F, contrib);
      {
        auto particle_bin = g_buckets_on_particle_buffer 
            ? next_pbuffer.ch(_0, next_pbuffer._binsts[src_blockno] + pidib / g_bin_capacity) 
            : next_pbuffer.ch(_0, partition._binsts[src_blockno] + pidib / g_bin_capacity);
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
        particle_bin.val(_13, pidib % g_bin_capacity) = ID;

      }
      contrib = (C * pbuffer.mass - contrib * newDt) * Dp_inv;
    }

    local_base_index = (pos * g_dx_inv + 0.5f).cast<int>() - 1;
    {
      int direction_tag = dir_offset((base_index - 1) / g_blocksize -
                              (local_base_index - 1) / g_blocksize);
      if (g_buckets_on_particle_buffer)
        next_pbuffer.add_advection(partition, local_base_index - 1, direction_tag, pidib);
      else 
        partition.add_advection(local_base_index - 1, direction_tag, pidib);
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
    int channelid = base % numMViPerBlock; // & (numMViPerBlock - 1);
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
  }
}


template <typename Partition, typename Grid>
__global__ void g2p2g(double dt, double newDt, const ivec3 *__restrict__ blocks,
                      const ParticleBuffer<material_e::NACC_ASFLIP> pbuffer,
                      ParticleBuffer<material_e::NACC_ASFLIP> next_pbuffer,
                      const Partition prev_partition, Partition partition,
                      const Grid grid, Grid next_grid) {
  static constexpr uint64_t numViPerBlock = g_blockvolume * 6;
  static constexpr uint64_t numViInArena = numViPerBlock << 3;

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
      *reinterpret_cast<MViArena>(shmem + numViInArena * sizeof(float));

  ivec3 blockid;
  int src_blockno;
  if (blocks != nullptr) { 
    blockid = blocks[blockIdx.x]; // Halo block ID
    src_blockno = partition.query(blockid); // Halo block number
  } else { 
    if (partition._haloMarks[blockIdx.x]) return; // Return early if halo block
    blockid = partition._activeKeys[blockIdx.x]; // Non-halo block ID
    src_blockno = blockIdx.x; // Non-halo block number
  }
  if (src_blockno < 0) return; // Return early if negative block number
//  else if (src_blockno > blockcnt) return; // Return early if excessive block number
  int ppb = g_buckets_on_particle_buffer
            ? next_pbuffer._ppbs[src_blockno] 
            : partition._ppbs[src_blockno]; // Particles in block
  if (ppb == 0) return; // Return early if no particles


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
    p2gbuffer[loc >> arenabits][x][y][z] = static_cast<PREC_G>(0);
  }
  __syncthreads();

  for (int pidib = threadIdx.x; pidib < ppb; pidib += blockDim.x) {
    int source_blockno, source_pidib;
    ivec3 base_index;
    {
      int advect = (g_buckets_on_particle_buffer)
                    ? next_pbuffer._blockbuckets[src_blockno * g_particle_num_per_block + pidib]
                    : partition._blockbuckets[src_blockno * g_particle_num_per_block + pidib];
      dir_components(advect / g_particle_num_per_block, base_index);
      base_index += blockid;
      source_blockno = prev_partition.query(base_index);
      source_pidib = advect % g_particle_num_per_block; // & (g_particle_num_per_block - 1);
      source_blockno = (g_buckets_on_particle_buffer)
                        ? pbuffer._binsts[source_blockno] + source_pidib / g_bin_capacity
                        : prev_partition._binsts[source_blockno] + source_pidib / g_bin_capacity;
    }
    pvec3 pos;
    pvec3 vel_p;
    {
      auto source_particle_bin = pbuffer.ch(_0, source_blockno);
      pos[0] = source_particle_bin.val(_0, source_pidib % g_bin_capacity);
      pos[1] = source_particle_bin.val(_1, source_pidib % g_bin_capacity);
      pos[2] = source_particle_bin.val(_2, source_pidib % g_bin_capacity);
      vel_p[0] = source_particle_bin.val(_13, source_pidib % g_bin_capacity);
      vel_p[1] = source_particle_bin.val(_14, source_pidib % g_bin_capacity);
      vel_p[2] = source_particle_bin.val(_15, source_pidib % g_bin_capacity);
    }
    ivec3 local_base_index = (pos * g_dx_inv_d + 0.5).cast<int>() - 1;
    pvec3 local_pos = pos - local_base_index * g_dx_d;
    base_index = local_base_index;

    pvec3x3 dws;
#pragma unroll 3
    for (int dd = 0; dd < 3; ++dd) {
      PREC d = (local_pos[dd] - static_cast<PREC>(std::floor(local_pos[dd] * g_dx_inv_d + 0.5) - 1) * g_dx_d) * g_dx_inv_d;
      dws(dd, 0) = 0.5 * (1.5 - d) * (1.5 - d);
      d -= 1.0;
      dws(dd, 1) = 0.75 - d * d;
      d = 0.5 + d;
      dws(dd, 2) = 0.5 * d * d;
      local_base_index[dd] = ((local_base_index[dd] - 1) & g_blockmask) + 1;
    }
    pvec3 vel, vel_FLIP;
    pvec9 C;
    vel.set(0.0);
    vel_FLIP.set(0.0);
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
          pvec3 vi_FLIP{g2pbuffer[3][local_base_index[0] + i][local_base_index[1] + j]
                           [local_base_index[2] + k],
                  g2pbuffer[4][local_base_index[0] + i][local_base_index[1] + j]
                           [local_base_index[2] + k],
                  g2pbuffer[5][local_base_index[0] + i][local_base_index[1] + j]
                           [local_base_index[2] + k]};
          vel += vi * W;
          vel_FLIP += vi_FLIP * W;
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
      // Incremental deformation gradient FInc, (F^n+1 = FInc : F^n)
#pragma unroll 9
    for (int d = 0; d < 9; ++d)
      dws.val(d) = C[d] * dt * Dp_inv + ((d & 0x3) ? static_cast<PREC>(0) : static_cast<PREC>(1));

    pvec9 contrib;
    {
      pvec9 F;
      auto source_particle_bin = pbuffer.ch(_0, source_blockno);
      // vel x,y,z not read in here
      contrib[0] = source_particle_bin.val(_3, source_pidib % g_bin_capacity);
      contrib[1] = source_particle_bin.val(_4, source_pidib % g_bin_capacity);
      contrib[2] = source_particle_bin.val(_5, source_pidib % g_bin_capacity);
      contrib[3] = source_particle_bin.val(_6, source_pidib % g_bin_capacity);
      contrib[4] = source_particle_bin.val(_7, source_pidib % g_bin_capacity);
      contrib[5] = source_particle_bin.val(_8, source_pidib % g_bin_capacity);
      contrib[6] = source_particle_bin.val(_9, source_pidib % g_bin_capacity);
      contrib[7] = source_particle_bin.val(_10, source_pidib % g_bin_capacity);
      contrib[8] = source_particle_bin.val(_11, source_pidib % g_bin_capacity);
      PREC logJp = source_particle_bin.val(_12, source_pidib % g_bin_capacity);
      PREC ID = source_particle_bin.val(_16, source_pidib % g_bin_capacity);

      // Deformation gradient update to F_p^n+1, (F_p^n+1 = FInc : F_p^n)
      matrixMatrixMultiplication3d(dws.data(), contrib.data(), F.data());
      PREC J = matrixDeterminant3d(F.data()); // J_p^n+1 = det(F_p^n+1)
      
      // Particle advection
      if (pbuffer.alpha > 0) {
        PREC beta = (J >= 1.0) ?  pbuffer.beta_max : pbuffer.beta_min;
        pos += dt * (vel + beta * pbuffer.alpha * (vel_p - vel_FLIP)); //< pos update
        vel += pbuffer.alpha * (vel_p - vel_FLIP); //< vel update
      } else {
        pos += dt * vel; //< pos update
      }
      
      // Update particle stress, adjust for plasticity in deformation gradient
      compute_stress_nacc(pbuffer.volume, pbuffer.mu, pbuffer.lambda,
                            pbuffer.bm, pbuffer.xi, pbuffer.beta, pbuffer.Msqr,
                            pbuffer.hardeningOn, logJp, F, contrib);
      {
        auto particle_bin = g_buckets_on_particle_buffer 
            ? next_pbuffer.ch(_0, next_pbuffer._binsts[src_blockno] + pidib / g_bin_capacity) 
            : next_pbuffer.ch(_0, partition._binsts[src_blockno] + pidib / g_bin_capacity);
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
        particle_bin.val(_13, pidib % g_bin_capacity) = vel[0];
        particle_bin.val(_14, pidib % g_bin_capacity) = vel[1];
        particle_bin.val(_15, pidib % g_bin_capacity) = vel[2];
        particle_bin.val(_16, pidib % g_bin_capacity) = ID;

      }
      contrib = (C * pbuffer.mass - contrib * newDt) * Dp_inv;
    }

    local_base_index = (pos * g_dx_inv + 0.5f).cast<int>() - 1;
    {
      int direction_tag = dir_offset((base_index - 1) / g_blocksize -
                              (local_base_index - 1) / g_blocksize);
      if (g_buckets_on_particle_buffer)
        next_pbuffer.add_advection(partition, local_base_index - 1, direction_tag, pidib);
      else 
        partition.add_advection(local_base_index - 1, direction_tag, pidib);
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
    int channelid = base % numMViPerBlock; // & (numMViPerBlock - 1);
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

template <typename Partition, typename Grid>
__global__ void g2p2g(double dt, double newDt, const ivec3 *__restrict__ blocks,
                      const ParticleBuffer<material_e::NACC_ASFLIP_FBAR> pbuffer,
                      ParticleBuffer<material_e::NACC_ASFLIP_FBAR> next_pbuffer,
                      const Partition prev_partition, Partition partition,
                      const Grid grid, Grid next_grid) {
  static constexpr uint64_t numViPerBlock = g_blockvolume * 6;
  static constexpr uint64_t numViInArena = numViPerBlock << 3;

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
      *reinterpret_cast<MViArena>(shmem + numViInArena * sizeof(float));

  ivec3 blockid;
  int src_blockno;
  if (blocks != nullptr) { 
    blockid = blocks[blockIdx.x]; // Halo block ID
    src_blockno = partition.query(blockid); // Halo block number
  } else { 
    if (partition._haloMarks[blockIdx.x]) return; // Return early if halo block
    blockid = partition._activeKeys[blockIdx.x]; // Non-halo block ID
    src_blockno = blockIdx.x; // Non-halo block number
  }
  if (src_blockno < 0) return; // Return early if negative block number
//  else if (src_blockno > blockcnt) return; // Return early if excessive block number
  int ppb = g_buckets_on_particle_buffer
            ? next_pbuffer._ppbs[src_blockno] 
            : partition._ppbs[src_blockno]; // Particles in block
  if (ppb == 0) return; // Return early if no particles


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
    p2gbuffer[loc >> arenabits][x][y][z] = static_cast<PREC_G>(0);
  }
  __syncthreads();

  for (int pidib = threadIdx.x; pidib < ppb; pidib += blockDim.x) {
    int source_blockno, source_pidib;
    ivec3 base_index;
    {
      int advect = (g_buckets_on_particle_buffer)
                    ? next_pbuffer._blockbuckets[src_blockno * g_particle_num_per_block + pidib]
                    : partition._blockbuckets[src_blockno * g_particle_num_per_block + pidib];
      dir_components(advect / g_particle_num_per_block, base_index);
      base_index += blockid;
      source_blockno = prev_partition.query(base_index);
      source_pidib = advect % g_particle_num_per_block; // & (g_particle_num_per_block - 1);
      source_blockno = (g_buckets_on_particle_buffer)
                        ? pbuffer._binsts[source_blockno] + source_pidib / g_bin_capacity
                        : prev_partition._binsts[source_blockno] + source_pidib / g_bin_capacity;
    }
    pvec3 pos;
    pvec3 vel_p;
    {
      auto source_particle_bin = pbuffer.ch(_0, source_blockno);
      pos[0] = source_particle_bin.val(_0, source_pidib % g_bin_capacity);
      pos[1] = source_particle_bin.val(_1, source_pidib % g_bin_capacity);
      pos[2] = source_particle_bin.val(_2, source_pidib % g_bin_capacity);
      vel_p[0] = source_particle_bin.val(_13, source_pidib % g_bin_capacity);
      vel_p[1] = source_particle_bin.val(_14, source_pidib % g_bin_capacity);
      vel_p[2] = source_particle_bin.val(_15, source_pidib % g_bin_capacity);
    }
    ivec3 local_base_index = (pos * g_dx_inv_d + 0.5).cast<int>() - 1;
    pvec3 local_pos = pos - local_base_index * g_dx_d;
    base_index = local_base_index;

    pvec3x3 dws;
#pragma unroll 3
    for (int dd = 0; dd < 3; ++dd) {
      PREC d = (local_pos[dd] - static_cast<PREC>(std::floor(local_pos[dd] * g_dx_inv_d + 0.5) - 1) * g_dx_d) * g_dx_inv_d;
      dws(dd, 0) = 0.5 * (1.5 - d) * (1.5 - d);
      d -= 1.0;
      dws(dd, 1) = 0.75 - d * d;
      d = 0.5 + d;
      dws(dd, 2) = 0.5 * d * d;
      local_base_index[dd] = ((local_base_index[dd] - 1) & g_blockmask) + 1;
    }
    pvec3 vel, vel_FLIP;
    pvec9 C;
    vel.set(0.0);
    vel_FLIP.set(0.0);
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
          pvec3 vi_FLIP{g2pbuffer[3][local_base_index[0] + i][local_base_index[1] + j]
                           [local_base_index[2] + k],
                  g2pbuffer[4][local_base_index[0] + i][local_base_index[1] + j]
                           [local_base_index[2] + k],
                  g2pbuffer[5][local_base_index[0] + i][local_base_index[1] + j]
                           [local_base_index[2] + k]};
          vel += vi * W;
          vel_FLIP += vi_FLIP * W;
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
      // Incremental deformation gradient FInc, (F^n+1 = FInc : F^n)
#pragma unroll 9
    for (int d = 0; d < 9; ++d)
      dws.val(d) = C[d] * dt * Dp_inv + ((d & 0x3) ? static_cast<PREC>(0) : static_cast<PREC>(1));

    pvec9 contrib;
    {
      pvec9 F;
      auto source_particle_bin = pbuffer.ch(_0, source_blockno);
      // vel x,y,z not read in here
      contrib[0] = source_particle_bin.val(_3, source_pidib % g_bin_capacity);
      contrib[1] = source_particle_bin.val(_4, source_pidib % g_bin_capacity);
      contrib[2] = source_particle_bin.val(_5, source_pidib % g_bin_capacity);
      contrib[3] = source_particle_bin.val(_6, source_pidib % g_bin_capacity);
      contrib[4] = source_particle_bin.val(_7, source_pidib % g_bin_capacity);
      contrib[5] = source_particle_bin.val(_8, source_pidib % g_bin_capacity);
      contrib[6] = source_particle_bin.val(_9, source_pidib % g_bin_capacity);
      contrib[7] = source_particle_bin.val(_10, source_pidib % g_bin_capacity);
      contrib[8] = source_particle_bin.val(_11, source_pidib % g_bin_capacity);
      PREC logJp = source_particle_bin.val(_12, source_pidib % g_bin_capacity);
      // vel_flip x,y,z not read in here
      PREC ID = source_particle_bin.val(_17, source_pidib % g_bin_capacity);

      // Deformation gradient update to F_p^n+1, (F_p^n+1 = FInc : F_p^n)
      matrixMatrixMultiplication3d(dws.data(), contrib.data(), F.data());
      PREC J = matrixDeterminant3d(F.data()); // J_p^n+1 = det(F_p^n+1)
      
      // Particle advection
      if (pbuffer.alpha > 0) {
        PREC beta = (J >= 1.0) ?  pbuffer.beta_max : pbuffer.beta_min;
        pos += dt * (vel + beta * pbuffer.alpha * (vel_p - vel_FLIP)); //< pos update
        vel += pbuffer.alpha * (vel_p - vel_FLIP); //< vel update
      } else {
        pos += dt * vel; //< pos update
      }
      
      // Update particle stress, adjust for plasticity in deformation gradient
      compute_stress_nacc(pbuffer.volume, pbuffer.mu, pbuffer.lambda,
                            pbuffer.bm, pbuffer.xi, pbuffer.beta, pbuffer.Msqr,
                            pbuffer.hardeningOn, logJp, F, contrib);
      {
        auto particle_bin = g_buckets_on_particle_buffer 
            ? next_pbuffer.ch(_0, next_pbuffer._binsts[src_blockno] + pidib / g_bin_capacity) 
            : next_pbuffer.ch(_0, partition._binsts[src_blockno] + pidib / g_bin_capacity);
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
        particle_bin.val(_13, pidib % g_bin_capacity) = vel[0];
        particle_bin.val(_14, pidib % g_bin_capacity) = vel[1];
        particle_bin.val(_15, pidib % g_bin_capacity) = vel[2];
        // sJBar
        particle_bin.val(_17, pidib % g_bin_capacity) = ID;

      }
      contrib = (C * pbuffer.mass - contrib * newDt) * Dp_inv;
    }

    local_base_index = (pos * g_dx_inv + 0.5f).cast<int>() - 1;
    {
      int direction_tag = dir_offset((base_index - 1) / g_blocksize -
                              (local_base_index - 1) / g_blocksize);
      if (g_buckets_on_particle_buffer)
        next_pbuffer.add_advection(partition, local_base_index - 1, direction_tag, pidib);
      else 
        partition.add_advection(local_base_index - 1, direction_tag, pidib);
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
    int channelid = base % numMViPerBlock; // & (numMViPerBlock - 1);
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

template <typename ParticleBuffer, typename Partition, typename Grid, typename VerticeArray>
__global__ void g2p2v(double dt, double newDt, const ivec3 *__restrict__ blocks,
                      const ParticleBuffer pbuffer,
                      ParticleBuffer next_pbuffer,
                      const Partition prev_partition, Partition partition,
                      const Grid grid, Grid next_grid,
                      VerticeArray vertice_array) { } 

// Grid-to-Particle-to-Grid + Mesh Update - ASFLIP Transfer
// Stress/Strain not computed here, displacements sent to FE mesh for force calc
template <typename Partition, typename Grid, typename VerticeArray>
__global__ void g2p2v(double dt, double newDt, const ivec3 *__restrict__ blocks,
                      const ParticleBuffer<material_e::Meshed> pbuffer,
                      ParticleBuffer<material_e::Meshed> next_pbuffer,
                      const Partition prev_partition, Partition partition,
                      const Grid grid, Grid next_grid,
                      VerticeArray vertice_array) {
  static constexpr uint64_t numViPerBlock = g_blockvolume * 6;
  static constexpr uint64_t numViInArena = numViPerBlock << 3;

  extern __shared__ char shmem[];
  using ViArena =
      float(*)[6][g_blocksize << 1][g_blocksize << 1][g_blocksize << 1];
  using ViArenaRef =
      float(&)[6][g_blocksize << 1][g_blocksize << 1][g_blocksize << 1];
  ViArenaRef __restrict__ g2pbuffer = *reinterpret_cast<ViArena>(shmem);

  ivec3 blockid;
  int src_blockno;
  if (blocks != nullptr) { 
    blockid = blocks[blockIdx.x]; // Halo block ID
    src_blockno = partition.query(blockid); // Halo block number
  } else { 
    if (partition._haloMarks[blockIdx.x]) return; // Return early if halo block
    blockid = partition._activeKeys[blockIdx.x]; // Non-halo block ID
    src_blockno = blockIdx.x; // Non-halo block number
  }
  if (src_blockno < 0) return; // Return early if negative block number
//  else if (src_blockno > blockcnt) return; // Return early if excessive block number
  int ppb = g_buckets_on_particle_buffer
            ? next_pbuffer._ppbs[src_blockno] 
            : partition._ppbs[src_blockno]; // Particles in block
  if (ppb == 0) return; // Return early if no particles


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
  for (int pidib = threadIdx.x; pidib < ppb; pidib += blockDim.x) {
    int source_blockno, source_pidib;
    ivec3 base_index;
    {
      int advect = (g_buckets_on_particle_buffer)
                    ? next_pbuffer._blockbuckets[src_blockno * g_particle_num_per_block + pidib]
                    : partition._blockbuckets[src_blockno * g_particle_num_per_block + pidib];
      dir_components(advect / g_particle_num_per_block, base_index);
      base_index += blockid;
      source_blockno = prev_partition.query(base_index);
      source_pidib = advect % g_particle_num_per_block; // & (g_particle_num_per_block - 1);
      source_blockno = (g_buckets_on_particle_buffer)
                        ? pbuffer._binsts[source_blockno] + source_pidib / g_bin_capacity
                        : prev_partition._binsts[source_blockno] + source_pidib / g_bin_capacity;
    }
    pvec3 pos;  //< Particle position at n
    int ID; // Vertice ID for mesh
    pvec3 vel_p; //< Particle vel. at n
    PREC beta = 0;
    pvec3 b;
    {
      auto source_particle_bin = pbuffer.ch(_0, source_blockno);
      pos[0] = source_particle_bin.val(_0, source_pidib % g_bin_capacity);  //< x
      pos[1] = source_particle_bin.val(_1, source_pidib % g_bin_capacity);  //< y
      pos[2] = source_particle_bin.val(_2, source_pidib % g_bin_capacity);  //< z
      ID = (int)source_particle_bin.val(_3, source_pidib % g_bin_capacity); //< ID
      vel_p[0] = source_particle_bin.val(_4, source_pidib % g_bin_capacity); //< vx
      vel_p[1] = source_particle_bin.val(_5, source_pidib % g_bin_capacity); //< vy
      vel_p[2] = source_particle_bin.val(_6, source_pidib % g_bin_capacity); //< vz
      //beta = source_particle_bin.val(_7, source_pidib % g_bin_capacity); //< 
      b[0] = source_particle_bin.val(_8, source_pidib % g_bin_capacity); //< bx
      b[1] = source_particle_bin.val(_9, source_pidib % g_bin_capacity); //< by
      b[2] = source_particle_bin.val(_10, source_pidib % g_bin_capacity); //< bz
    }
    ivec3 local_base_index = (pos * g_dx_inv_d + 0.5).cast<int>() - 1;
    pvec3 local_pos = pos - local_base_index * g_dx_d;
    base_index = local_base_index;

    pvec3x3 dws;
#pragma unroll 3
    for (int dd = 0; dd < 3; ++dd) {
      PREC d = (local_pos[dd] - static_cast<PREC>(std::floor(local_pos[dd] * g_dx_inv_d + 0.5) - 1) * g_dx_d) * g_dx_inv_d;
      dws(dd, 0) = 0.5 * (1.5 - d) * (1.5 - d);
      d -= 1.0;
      dws(dd, 1) = 0.75 - d * d;
      d = 0.5 + d;
      dws(dd, 2) = 0.5 * d * d;
      local_base_index[dd] = ((local_base_index[dd] - 1) & g_blockmask) + 1;
    }
    pvec3 vel;   //< Stressed, collided grid velocity
    pvec3 vel_FLIP; //< Unstressed, uncollided grid velocity
    pvec9 C;     //< APIC affine matrix, used a few times
    vel.set(0.0); 
    vel_FLIP.set(0.0);
    C.set(0.0);

    PREC scale = pbuffer.area;
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
          vel_FLIP += vi_n * W; 
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

    //PREC count = 0;
    int count = (int) vertice_array.val(_10, ID); //< count

    int surface_ASFLIP = 0;
    if (surface_ASFLIP && count > 10.f) { 
      // Interior of Mesh
      pos += dt * vel;
      vel += pbuffer.alpha * (vel_p - vel_FLIP);
    } else if (surface_ASFLIP && count <= 10.f) { 
      // Surface of Mesh
      pos += dt * (vel + beta * pbuffer.alpha * (vel_p - vel_FLIP));
      vel += pbuffer.alpha * (vel_p - vel_FLIP);
    } else {
      pos += dt * (vel + beta * pbuffer.alpha * (vel_p - vel_FLIP));
      vel += pbuffer.alpha * (vel_p - vel_FLIP);
    }

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

template <typename ParticleBuffer, typename Partition, typename Grid, typename VerticeArray>
__global__ void g2p2v_FBar(double dt, double newDt, const ivec3 *__restrict__ blocks,
                      const ParticleBuffer pbuffer,
                      const ParticleBuffer next_pbuffer,
                      const Partition prev_partition, const Partition partition,
                      const Grid grid, Grid next_grid,
                      VerticeArray vertice_array) { }

// Grid-to-Particle-to-Grid + Mesh Update - F-Bar ASFLIP Transfer
// Stress/Strain not computed here, displacements sent to FE mesh for force calc
template <typename Partition, typename Grid, typename VerticeArray>
__global__ void g2p2v_FBar(double dt, double newDt, const ivec3 *__restrict__ blocks,
                      const ParticleBuffer<material_e::Meshed> pbuffer,
                      const ParticleBuffer<material_e::Meshed> next_pbuffer,
                      const Partition prev_partition, const Partition partition,
                      const Grid grid, Grid next_grid,
                      VerticeArray vertice_array) {
  static constexpr uint64_t numViPerBlock = g_blockvolume * 6;
  static constexpr uint64_t numViInArena = numViPerBlock << 3;

  extern __shared__ char shmem[];
  using ViArena =
      float(*)[6][g_blocksize << 1][g_blocksize << 1][g_blocksize << 1];
  using ViArenaRef =
      float(&)[6][g_blocksize << 1][g_blocksize << 1][g_blocksize << 1];
  ViArenaRef __restrict__ g2pbuffer = *reinterpret_cast<ViArena>(shmem);

  ivec3 blockid;
  int src_blockno;
  if (blocks != nullptr) { 
    blockid = blocks[blockIdx.x]; // Halo block ID
    src_blockno = partition.query(blockid); // Halo block number
  } else { 
    if (partition._haloMarks[blockIdx.x]) return; // Return early if halo block
    blockid = partition._activeKeys[blockIdx.x]; // Non-halo block ID
    src_blockno = blockIdx.x; // Non-halo block number
  }
  if (src_blockno < 0) return; // Return early if negative block number
//  else if (src_blockno > blockcnt) return; // Return early if excessive block number
  int ppb = g_buckets_on_particle_buffer
            ? next_pbuffer._ppbs[src_blockno] 
            : partition._ppbs[src_blockno]; // Particles in block
  if (ppb == 0) return; // Return early if no particles


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
  for (int pidib = threadIdx.x; pidib < ppb; pidib += blockDim.x) {
    int source_blockno, source_pidib;
    ivec3 base_index;
    {
      int advect = (g_buckets_on_particle_buffer)
                    ? next_pbuffer._blockbuckets[src_blockno * g_particle_num_per_block + pidib]
                    : partition._blockbuckets[src_blockno * g_particle_num_per_block + pidib];
      dir_components(advect / g_particle_num_per_block, base_index);
      base_index += blockid;
      source_blockno = prev_partition.query(base_index);
      source_pidib = advect % g_particle_num_per_block; // & (g_particle_num_per_block - 1);
      source_blockno = (g_buckets_on_particle_buffer)
                        ? pbuffer._binsts[source_blockno] + source_pidib / g_bin_capacity
                        : prev_partition._binsts[source_blockno] + source_pidib / g_bin_capacity;
    }
    pvec3 pos;  //< Particle position at n
    int ID; // Vertice ID for mesh
    pvec3 vel_p; //< Particle vel. at n
    PREC beta = 0;
    pvec3 b;
    {
      auto source_particle_bin = pbuffer.ch(_0, source_blockno);
      pos[0] = source_particle_bin.val(_0, source_pidib % g_bin_capacity);  //< x
      pos[1] = source_particle_bin.val(_1, source_pidib % g_bin_capacity);  //< y
      pos[2] = source_particle_bin.val(_2, source_pidib % g_bin_capacity);  //< z
      ID = (int)source_particle_bin.val(_3, source_pidib % g_bin_capacity); //< ID
      vel_p[0] = source_particle_bin.val(_4, source_pidib % g_bin_capacity); //< vx
      vel_p[1] = source_particle_bin.val(_5, source_pidib % g_bin_capacity); //< vy
      vel_p[2] = source_particle_bin.val(_6, source_pidib % g_bin_capacity); //< vz
      //beta = source_particle_bin.val(_7, source_pidib % g_bin_capacity); //< 
      b[0] = source_particle_bin.val(_8, source_pidib % g_bin_capacity); //< bx
      b[1] = source_particle_bin.val(_9, source_pidib % g_bin_capacity); //< by
      b[2] = source_particle_bin.val(_10, source_pidib % g_bin_capacity); //< bz
    }
    ivec3 local_base_index = (pos * g_dx_inv_d + 0.5).cast<int>() - 1;
    pvec3 local_pos = pos - local_base_index * g_dx_d;
    base_index = local_base_index;

    pvec3x3 dws;
#pragma unroll 3
    for (int dd = 0; dd < 3; ++dd) {
      PREC d = (local_pos[dd] - static_cast<PREC>(std::floor(local_pos[dd] * g_dx_inv_d + 0.5) - 1) * g_dx_d) * g_dx_inv_d;
      dws(dd, 0) = 0.5 * (1.5 - d) * (1.5 - d);
      d -= 1.0;
      dws(dd, 1) = 0.75 - d * d;
      d = 0.5 + d;
      dws(dd, 2) = 0.5 * d * d;
      local_base_index[dd] = ((local_base_index[dd] - 1) & g_blockmask) + 1;
    }
    pvec3 vel;   //< Stressed, collided grid velocity
    pvec3 vel_FLIP; //< Unstressed, uncollided grid velocity
    pvec9 C;     //< APIC affine matrix, used a few times
    vel.set(0.0); 
    vel_FLIP.set(0.0);
    C.set(0.0);
    PREC scale = pbuffer.area;
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
          vel_FLIP += vi_n * W; 
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

    int count = vertice_array.val(_10, ID); //< count

    int surface_ASFLIP = 0;
    beta = 0;
    if (surface_ASFLIP && count > 10.f) { 
      // Interior of Mesh
      pos += dt * vel;
      vel += pbuffer.alpha * (vel_p - vel_FLIP);
    } else if (surface_ASFLIP && count <= 10.f) { 
      // Surface of Mesh
      pos += dt * (vel + beta * pbuffer.alpha * (vel_p - vel_FLIP));
      vel += pbuffer.alpha * (vel_p - vel_FLIP);
    } else {
      pos += dt * (vel + beta * pbuffer.alpha * (vel_p - vel_FLIP));
      vel += pbuffer.alpha * (vel_p - vel_FLIP);
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

template <typename VerticeArray, typename ElementArray, typename ElementBuffer>
__global__ void v2fem2v(uint32_t blockCount, double dt, double newDt,
                       VerticeArray vertice_array,
                       const ElementArray element_array,
                       ElementBuffer elementBins) { return; }

// Grid-to-Particle-to-Grid + Mesh Update - ASFLIP Transfer
// Stress/Strain not computed here, displacements sent to FE mesh for force calc
template <typename VerticeArray, typename ElementArray>
__global__ void v2fem2v(uint32_t blockCount, double dt, double newDt,
                      VerticeArray vertice_array,
                      const ElementArray element_array,
                      ElementBuffer<fem_e::Tetrahedron> elementBins) {
    auto element_number = blockIdx.x * blockDim.x + threadIdx.x;
    if (element_number < blockCount && element_number < g_max_fem_element_num)
    {
    auto element = elementBins.ch(_0, element_number);
    int IDs[4];
    pvec3 p[4];

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
    DmI[0] = element.val(_4, 0); //< Bm^T, undef. area weighted face normals^T
    DmI[1] = element.val(_5, 0);
    DmI[2] = element.val(_6, 0);
    DmI[3] = element.val(_7, 0);
    DmI[4] = element.val(_8, 0);
    DmI[5] = element.val(_9, 0);
    DmI[6] = element.val(_10, 0);
    DmI[7] = element.val(_11, 0);
    DmI[8] = element.val(_12, 0);
    PREC V0 = element.val(_13, 0); //< Undeformed volume [m^3]
    //PREC sV0 = V0<0.f ? -1.f : 1.f;

    // Set position of vertices
    #pragma unroll 4
    for (int v = 0; v < 4; v++) {
      int ID = IDs[v] - 1; //< Index at 0, my elements input files index from 1
      p[v][0] = vertice_array.val(_0, ID) * elementBins.length; //< x [m]
      p[v][1] = vertice_array.val(_1, ID) * elementBins.length; //< y [m]
      p[v][2] = vertice_array.val(_2, ID) * elementBins.length; //< z [m]
    }
    //__syncthreads();

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
    //PREC Vn = V0 * J;
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
    compute_stress_PK1_fixedcorotated((PREC)1.0, elementBins.mu, elementBins.lambda, F, P);

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
    PREC pressure  = compute_MeanStress_from_StressCauchy(Bm.data()) / J;
    PREC von_mises = compute_VonMisesStress_from_StressCauchy(Bm.data()) / sqrt(J);
    


#pragma unroll 4
    for (int v=0; v<4; v++)
    {
      pvec3 force; // Internal force vector at deformed vertex
      pvec3 n;
      if (v == 1) { // Node 1; Face a
        force[0] = G[0];
        force[1] = G[1];
        force[2] = G[2];
        n[0] = J * Bs[0];
        n[1] = J * Bs[1];
        n[2] = J * Bs[2];
      } else if (v == 2) { // Node 2; Face b
        force[0] = G[3];
        force[1] = G[4];
        force[2] = G[5];
        n[0] = J * Bs[3];
        n[1] = J * Bs[4];
        n[2] = J * Bs[5];
      }  else if (v == 3) { // Node 3; Face c
        force[0] = G[6]; 
        force[1] = G[7];
        force[2] = G[8];
        n[0] = J * Bs[6];
        n[1] = J * Bs[7];
        n[2] = J * Bs[8];
      } else { // Node 4; Face d
        force[0] = - (G[0] + G[3] + G[6]);
        force[1] = - (G[1] + G[4] + G[7]);
        force[2] = - (G[2] + G[5] + G[8]);
        n[0] = - J * (Bs[0] + Bs[3] + Bs[6]) ;
        n[1] = - J * (Bs[1] + Bs[4] + Bs[7]) ;
        n[2] = - J * (Bs[2] + Bs[5] + Bs[8]) ;
      }
      //__syncthreads();
      
#pragma unroll 3
      for (int d = 0; d < 3; d++) force[d] =  force[d] / elementBins.length;

      //__syncthreads();

      
      int ID = IDs[v] - 1; // Index from 0
      atomicAdd(&vertice_array.val(_3, ID), (PREC)(1.0-J)*abs(V0)*0.25); //< bx
      atomicAdd(&vertice_array.val(_4, ID), (PREC)pressure*abs(V0)*0.25); //< by
      atomicAdd(&vertice_array.val(_5, ID), (PREC)von_mises*abs(V0)*0.25); //< bz
      atomicAdd(&vertice_array.val(_6, ID), (PREC)abs(V0)*0.25); //< Node undef. volume [m3]
      atomicAdd(&vertice_array.val(_7, ID), (PREC)force[0]); //< fx
      atomicAdd(&vertice_array.val(_8, ID), (PREC)force[1]); //< fy
      atomicAdd(&vertice_array.val(_9, ID), (PREC)force[2]); //< fz
      atomicAdd(&vertice_array.val(_10, ID), (PREC)1.f); //< Counter
      
    }
  }
}
template <typename VerticeArray, typename ElementArray>
__global__ void v2fem2v(uint32_t blockCount, double dt, double newDt,
                      VerticeArray vertice_array,
                      ElementArray element_array,
                      ElementBuffer<fem_e::Tetrahedron_FBar> elementBins) { return; }
template <typename VerticeArray, typename ElementArray>
__global__ void v2fem2v(uint32_t blockCount, double dt, double newDt,
                      VerticeArray vertice_array,
                      ElementArray element_array,
                      ElementBuffer<fem_e::Brick> elementBins) { return; }


// Grid-to-Particle-to-Grid + Mesh Update - ASFLIP Transfer
// Stress/Strain not computed here, displacements sent to FE mesh for force calc
template <typename VerticeArray, typename ElementArray, typename ElementBuffer>
__global__ void v2fem_FBar(uint32_t blockCount, double dt, double newDt,
                           VerticeArray vertice_array,
                           const ElementArray element_array,
                           const ElementBuffer elementBins) { return; }

template <typename VerticeArray, typename ElementArray>
__global__ void v2fem_FBar(uint32_t blockCount, double dt, double newDt,
                      VerticeArray vertice_array,
                      const ElementArray element_array,
                      const ElementBuffer<fem_e::Tetrahedron_FBar> elementBins) 
{
  auto element_number = blockIdx.x * blockDim.x + threadIdx.x;
  if (element_number < blockCount && element_number < g_max_fem_element_num)
  {
    auto element = elementBins.ch(_0, element_number);

    /// Precomputed
    // Dm is undeformed edge vector matrix relative to a vertex 0
    // Bm is undeformed face normal matrix relative to a vertex 0
    // Bm^T = Dm^-1
    // Restvolume is underformed volume [m^3]
    int IDs[4];
    pvec9 DmI, Bm;
    DmI.set(0.0);
    Bm.set(0.0);
    PREC V0, Jn, JBar;

    IDs[0] = element.val(_0, 0); //< ID of node 0
    IDs[1] = element.val(_1, 0); //< ID of node 1
    IDs[2] = element.val(_2, 0); //< ID of node 2 
    IDs[3] = element.val(_3, 0); //< ID of node 3
    DmI[0] = element.val(_4, 0); //< Bm^T, undef. area weighted face normals^T
    DmI[1] = element.val(_5, 0);
    DmI[2] = element.val(_6, 0);
    DmI[3] = element.val(_7, 0);
    DmI[4] = element.val(_8, 0);
    DmI[5] = element.val(_9, 0);
    DmI[6] = element.val(_10, 0);
    DmI[7] = element.val(_11, 0);
    DmI[8] = element.val(_12, 0);
    V0     = element.val(_13, 0); //< Undeformed volume [m^3]
    Jn     = 1.0 - element.val(_14, 0);
    JBar   = 1.0 - element.val(_15, 0);
    //PREC sV0 = V0<0.f ? -1.f : 1.f;

    // Set position of vertices
    pvec3 p[4];
#pragma unroll 4
    for (int v = 0; v < 4; v++) {
      int ID = IDs[v] - 1; //< Index at 0, my elements input files index from 1
      p[v][0] = vertice_array.val(_0, ID) * elementBins.length; //< x [m]
      p[v][1] = vertice_array.val(_1, ID) * elementBins.length; //< y [m]
      p[v][2] = vertice_array.val(_2, ID) * elementBins.length; //< z [m]
    }

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
    PREC w = 0.25; // Gaussian weight for element nodes
#pragma unroll 4
    for (int v = 0; v < 4; v++)
    {
      int ID = IDs[v] - 1; // Index from 0
      atomicAdd(&vertice_array.val(_11, ID), Vn*w); //< Counter
      atomicAdd(&vertice_array.val(_12, ID), JInc*JBar*Vn*w); //< Counter
    }
  }
}


// Grid-to-Particle-to-Grid + Mesh Update - ASFLIP Transfer
// Stress/Strain not computed here, displacements sent to FE mesh for force calc
template <typename VerticeArray, typename ElementArray>
__global__ void fem2v_FBar(uint32_t blockCount, double dt, double newDt,
                           VerticeArray vertice_array,
                           const ElementArray element_array,
                           ElementBuffer<fem_e::Tetrahedron_FBar> elementBins) 
{  
  auto element_number = blockIdx.x * blockDim.x + threadIdx.x;
  if (element_number < blockCount && element_number < g_max_fem_element_num)
  {
    auto element = elementBins.ch(_0, element_number);

    /// Precomputed
    // Dm is undeformed edge vector matrix relative to a vertex 0
    // Bm is undeformed face normal matrix relative to a vertex 0
    // Bm^T = Dm^-1
    // Restvolume is underformed volume [m^3]
    int IDs[4];
    pvec3 p[4];
    pvec9 DmI, Bm;
    DmI.set(0.0);
    Bm.set(0.0);
    PREC V0;

    IDs[0] = element.val(_0, 0); //< ID of node 0
    IDs[1] = element.val(_1, 0); //< ID of node 1
    IDs[2] = element.val(_2, 0); //< ID of node 2 
    IDs[3] = element.val(_3, 0); //< ID of node 3
    DmI[0] = element.val(_4, 0); //< Bm^T, undef. area weighted face normals^T
    DmI[1] = element.val(_5, 0);
    DmI[2] = element.val(_6, 0);
    DmI[3] = element.val(_7, 0);
    DmI[4] = element.val(_8, 0);
    DmI[5] = element.val(_9, 0);
    DmI[6] = element.val(_10, 0);
    DmI[7] = element.val(_11, 0);
    DmI[8] = element.val(_12, 0);
    V0     = element.val(_13, 0); //< Undeformed volume [m^3]
    //Jn   = 1.0 - element.val(_14, 0);
    //JBar_n = 1.0 - element.val(_15, 0);
    //PREC sV0 = V0<0.f ? -1.f : 1.f;
    PREC JBar = 0.0;
    PREC JBar_i = 0.0;
    PREC Vn_i = 0.0;
    PREC w = 0.25; // Gaussian weight

    // Set position of vertices
#pragma unroll 4
    for (int v = 0; v < 4; v++) {
      int ID = IDs[v] - 1; //< Index at 0, my elements input files index from 1
      p[v][0] = vertice_array.val(_0, ID) * elementBins.length; //< x [m]
      p[v][1] = vertice_array.val(_1, ID) * elementBins.length; //< y [m]
      p[v][2] = vertice_array.val(_2, ID) * elementBins.length; //< z [m]
      Vn_i   = vertice_array.val(_11, ID);
      JBar_i = vertice_array.val(_12, ID);
      JBar += w * (JBar_i / Vn_i);
    }

    //PREC signJBar = JBar<0.f ? -1.f : 1.f;

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
    //__syncthreads();

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
    
    // Bs = J F^-1^T Bm ;  Bs^T = J Bm^T F^-1; Note: (AB)^T = B^T A^T
    pvec9 Bs; //< BsT is the deformed. area-weighted node normal matrix
    Bs.set(0.0);
    matrixTransposeMatrixMultiplication3d(Finv.data(), Bm.data(), Bs.data());
    // Bs = vol_n * Ds^-T 
    //matrixInverse(Ds.data(), Bs.data());


    // J = det | F |,  i.e. volume change ratio undef -> def
    PREC J = matrixDeterminant3d(F.data());
    //REC Vn = V0 * J;

    PREC J_Scale =  cbrt( (JBar / J) );
#pragma unroll 9
    for (int d=0; d<9; d++) F[d] = J_Scale * F[d];

    {
      elementBins.ch(_0, element_number).val(_14, 0) = (PREC)(1.0 - J);
      elementBins.ch(_0, element_number).val(_15, 0) = (PREC)(1.0 - JBar);
    }

    // P is First Piola-Kirchoff stress at element Gauss point
    // P = (dPsi/dF){F}, Derivative of Free Energy w.r.t. Def. Grad.
    // Maps undeformed area weighted face normals to deformed traction vectors
    pvec9 P; //< PK1, First Piola-Kirchoff stress at element Gauss point
    pvec9 G; //< Deformed internal forces at nodes 1,2,3 relative to node 0
    P.set(0.0);
    G.set(0.0);
    // P = (dPsi/dF){F}, Fixed-Corotated model for energy potential
    compute_stress_PK1_fixedcorotated((PREC)1.0, elementBins.mu, elementBins.lambda, F, P);

    // G = P Bm
    matrixMatrixMultiplication3d(P.data(), Bm.data(), G.data());

    pvec9 C;
    C.set(0.0); //< Now use for Cauchy stress
    // C = (1/J) P F^T
    matrixMatrixTransposeMultiplication3d(P.data(), F.data(), C.data());
// #pragma unroll 9
//     for (int d=0; d<9; d++) Bm[d] = Bm[d] / J;

    PREC pressure  = compute_MeanStress_from_StressCauchy(C.data()) / JBar;
    PREC von_mises = compute_VonMisesStress_from_StressCauchy(C.data()) / sqrt(JBar); 

#pragma unroll 9 
    for (int d=0; d<9; d++)
    { 
      G[d] = G[d]/elementBins.length;
      Bs[d] = J*Bs[d];
    }
    
    //__syncthreads();
#pragma unroll 4
    for (int v=0; v<4; v++) {
      pvec3 force; // Internal force vector at deformed vertex
      pvec3 n;
      if (v == 1){ // Node 1; Face a
        force[0] = G[0];
        force[1] = G[1];
        force[2] = G[2];
        n[0] = Bs[0];
        n[1] = Bs[1];
        n[2] = Bs[2];
      } else if (v == 2) { // Node 2; Face b
        force[0] = G[3];
        force[1] = G[4];
        force[2] = G[5];
        n[0] = Bs[3];
        n[1] = Bs[4];
        n[2] = Bs[5];
      }  else if (v == 3) { // Node 3; Face c
        force[0] = G[6]; 
        force[1] = G[7];
        force[2] = G[8];
        n[0] = Bs[6];
        n[1] = Bs[7];
        n[2] = Bs[8];
      } else { // Node 4; Face d
        force[0] = - (G[0] + G[3] + G[6]);
        force[1] = - (G[1] + G[4] + G[7]);
        force[2] = - (G[2] + G[5] + G[8]);
        n[0] = - (Bs[0] + Bs[3] + Bs[6]) ;
        n[1] = - (Bs[1] + Bs[4] + Bs[7]) ;
        n[2] = - (Bs[2] + Bs[5] + Bs[8]) ;
      }
      
      int ID = IDs[v] - 1; // Index from 0
      // atomicAdd(&vertice_array.val(_3, ID), (PREC)n[0]); //< bx
      // atomicAdd(&vertice_array.val(_4, ID), (PREC)n[1]); //< by
      // atomicAdd(&vertice_array.val(_5, ID), (PREC)n[2]); //< bz
      atomicAdd(&vertice_array.val(_3, ID), (PREC)(J * V0 * w)); //< bz
      atomicAdd(&vertice_array.val(_4, ID), (PREC)(pressure  * V0 * w)); //< bx
      atomicAdd(&vertice_array.val(_5, ID), (PREC)(von_mises * V0 * w)); //< by
      atomicAdd(&vertice_array.val(_6, ID), (PREC)(V0 * w)); //< Node undef. volume [m3]
      atomicAdd(&vertice_array.val(_7, ID), (PREC)force[0]); //< fx
      atomicAdd(&vertice_array.val(_8, ID), (PREC)force[1]); //< fy
      atomicAdd(&vertice_array.val(_9, ID), (PREC)force[2]); //< fz
      atomicAdd(&vertice_array.val(_10, ID), (PREC)1); //< Counter
      // atomicAdd(&vertice_array.val(_11, ID), Vn*0.25); //< Counter
      // atomicAdd(&vertice_array.val(_12, ID), (1.0 - JBar)*Vn*0.25); //< Counter
    
    }
  }
}

template <typename VerticeArray, typename ElementArray, typename ElementBuffer>
__global__ void fem2v_FBar(uint32_t blockCount, double dt, double newDt,
                      VerticeArray vertice_array,
                      const ElementArray element_array,
                      ElementBuffer elementBins) { }

template <typename ParticleBuffer, typename Partition, typename Grid>
__global__ void g2p2g_FBar(double dt, double newDt, const ivec3 *__restrict__ blocks,
                      const ParticleBuffer pbuffer,
                      ParticleBuffer next_pbuffer,
                      const Partition prev_partition, Partition partition,
                      Grid grid, Grid next_grid) {
                        if (threadIdx.x == 0 && blockIdx.x == 0) printf("ERROR: The g2p2g_FBar kernel has not yet been implemented for your selection of material and algorithm! Please try different settings...\n");
                      }

// Uses (3 + 3 + 1 + 1 + 3 + 3 + 2) * (512) * (4 bytes) shmem
// 32 kB shmem per block single precision, 64 kB per block double precision
// __launch_bounds__(g_particle_batch, (G_MAX_SHMEM_PER_SM / 32))
template <typename Partition, typename Grid>
__global__ void
__launch_bounds__(g_particle_batch, 3)
g2p2g_FBar(double dt, double newDt, const ivec3 *__restrict__ blocks,
                      const ParticleBuffer<material_e::JBarFluid> pbuffer,
                      ParticleBuffer<material_e::JBarFluid> next_pbuffer,
                      const Partition prev_partition, Partition partition,
                      const Grid grid, Grid next_grid) { 
  constexpr std::size_t num_g2p_variables = (3 + 3 + 1); // vel, vel_FLIP, 1 - JBar
  constexpr std::size_t num_p2g_variables = (1 + 3 + 3 + 2); // mass, momentum + fint, vel_FLIP,  vol, sJBar*vol
  
  static constexpr uint64_t numViPerBlock = g_blockvolume * num_g2p_variables;
  static constexpr uint64_t numViInArena = numViPerBlock << 3;
  static constexpr uint64_t shmem_offset = (g_blockvolume * num_g2p_variables) << 3;

  static constexpr uint64_t numMViPerBlock = g_blockvolume * num_p2g_variables;
  static constexpr uint64_t numMViInArena = numMViPerBlock << 3;

  static constexpr unsigned arenamask = (g_blocksize << 1) - 1;
  static constexpr unsigned arenabits = g_blockbits + 1;

  extern __shared__ char shmem[];
  using ViArena =
      PREC_G(*)[num_g2p_variables][g_blocksize << 1][g_blocksize << 1][g_blocksize << 1];
  using ViArenaRef =
      PREC_G(&)[num_g2p_variables][g_blocksize << 1][g_blocksize << 1][g_blocksize << 1];
  ViArenaRef __restrict__ g2pbuffer = *reinterpret_cast<ViArena>(shmem);
  using MViArena =
      PREC_G(*)[num_p2g_variables][g_blocksize << 1][g_blocksize << 1][g_blocksize << 1];
  using MViArenaRef =
      PREC_G(&)[num_p2g_variables][g_blocksize << 1][g_blocksize << 1][g_blocksize << 1];
  MViArenaRef __restrict__ p2gbuffer =
      *reinterpret_cast<MViArena>(shmem + shmem_offset * sizeof(PREC_G));

  ivec3 blockid;
  int src_blockno;
  if (blocks != nullptr) { 
    blockid = blocks[blockIdx.x]; // Halo block ID
    src_blockno = partition.query(blockid); // Halo block number
  } else { 
    if (partition._haloMarks[blockIdx.x]) return; // Return early if halo block
    src_blockno = blockIdx.x; // Non-halo block number
    blockid = partition._activeKeys[blockIdx.x]; // Non-halo block ID
  }
  int ppb = g_buckets_on_particle_buffer
          ? next_pbuffer._ppbs[src_blockno] 
          : partition._ppbs[src_blockno]; // Particles in block
  if (ppb == 0) return; // Return early if no particles in block

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
      val = grid_block.val_1d(_8, c); // Note, pulling 8th channel from grid into 6th channel of g2pbuffer as we skip channel 7 of grid
    g2pbuffer[channelid][cx + (local_block_id & 4 ? g_blocksize : 0)][cy + (local_block_id & 2 ? g_blocksize : 0)][cz + (local_block_id & 1 ? g_blocksize : 0)] = val;
  }
  __syncthreads();
  for (int base = threadIdx.x; base < numMViInArena; base += blockDim.x) {
    int loc = base;
    char z = loc & arenamask;
    char y = (loc >>= arenabits) & arenamask;
    char x = (loc >>= arenabits) & arenamask;
    p2gbuffer[loc >> arenabits][x][y][z] = static_cast<PREC_G>(0);
  }
  __syncthreads();
  // Start Grid-to-Particle, threads are particles, threads can iterate through multiple particles
  for (int pidib = threadIdx.x; pidib < ppb; pidib += blockDim.x) {
    int source_blockno, source_pidib;
    ivec3 base_index;
    {
      int advect = (g_buckets_on_particle_buffer)
                    ? next_pbuffer._blockbuckets[src_blockno * g_particle_num_per_block + pidib]
                    : partition._blockbuckets[src_blockno * g_particle_num_per_block + pidib];
      dir_components(advect / g_particle_num_per_block, base_index);
      base_index += blockid;
      source_blockno = prev_partition.query(base_index);
      source_pidib = advect % g_particle_num_per_block; // & (g_particle_num_per_block - 1);
      source_blockno = (g_buckets_on_particle_buffer)
                        ? pbuffer._binsts[source_blockno] + source_pidib / g_bin_capacity
                        : prev_partition._binsts[source_blockno] + source_pidib / g_bin_capacity;
    }
    pvec3 pos;  //< Particle position at n
    pvec3 vel_p; //< Particle vel. at n
    PREC sJ, sJBar_old; //< 1 - Particle volume ratio at n (previous step)
    PREC ID;
    {
      auto source_particle_bin = pbuffer.ch(_0, source_blockno);
      pos[0] = source_particle_bin.val(_0, source_pidib % g_bin_capacity);  //< x
      pos[1] = source_particle_bin.val(_1, source_pidib % g_bin_capacity);  //< y
      pos[2] = source_particle_bin.val(_2, source_pidib % g_bin_capacity);  //< z
      sJ = source_particle_bin.val(_3, source_pidib % g_bin_capacity);       //< Vo/V
      vel_p[0] = source_particle_bin.val(_4, source_pidib % g_bin_capacity); //< vx
      vel_p[1] = source_particle_bin.val(_5, source_pidib % g_bin_capacity); //< vy
      vel_p[2] = source_particle_bin.val(_6, source_pidib % g_bin_capacity); //< vz
      // sJBar_old = source_particle_bin.val(_7, source_pidib % g_bin_capacity); //< (1 - Vo/V)Bar
      ID  = source_particle_bin.val(_8, source_pidib % g_bin_capacity); //< Particle ID
    }
    ivec3 local_base_index = (pos * g_dx_inv_d + 0.5).cast<int>() - 1;
    pvec3 local_pos = pos - local_base_index * g_dx_d;
    base_index = local_base_index;

    pvec3x3 dws;
#pragma unroll 3
    for (int dd = 0; dd < 3; ++dd) {
      PREC d = (local_pos[dd] - static_cast<PREC>(std::floor(local_pos[dd] * g_dx_inv_d + 0.5) - 1) * g_dx_d) * g_dx_inv_d;
      dws(dd, 0) = 0.5 * (1.5 - d) * (1.5 - d);
      d -= 1.0;
      dws(dd, 1) = 0.75 - d * d;
      d = 0.5 + d;
      dws(dd, 2) = 0.5 * d * d;
      local_base_index[dd] = ((local_base_index[dd] - 1) & g_blockmask) + 1;
    }
    pvec3 vel;   //< Stressed, collided grid velocity
    pvec3 vel_FLIP; //< Unstressed, uncollided grid velocity
    pvec9 C;     //< APIC affine matrix, used a few times
    PREC sJBar_new; //< New value of (1 - JBar^n+1) to calculate this step
    vel.set(0.0); vel_FLIP.set(0.0); C.set(0.0); // Zero out all values
    sJBar_new = 0.0; // Zero out. We will sum into this variable

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

          PREC sJBar_i = g2pbuffer[6][local_base_index[0] + i][local_base_index[1] + j]
                            [local_base_index[2] + k]; // Assumes we already performed grid operation: sJBar = sJBar / voln 
          vel   += vi * W;
          vel_FLIP += vi_n * W; 
          C[0] += W * vi[0] * xixp[0] * scale;
          C[1] += W * vi[1] * xixp[0] * scale;
          C[2] += W * vi[2] * xixp[0] * scale;
          C[3] += W * vi[0] * xixp[1] * scale;
          C[4] += W * vi[1] * xixp[1] * scale;
          C[5] += W * vi[2] * xixp[1] * scale;
          C[6] += W * vi[0] * xixp[2] * scale;
          C[7] += W * vi[1] * xixp[2] * scale;
          C[8] += W * vi[2] * xixp[2] * scale;
          sJBar_new += sJBar_i * W;
        }

    PREC sJInc = (C[0] + C[4] + C[8]) * dt * Dp_inv; // sJInc = (tr(A)*dt*D_p^[inv] + 1 ) - 1)
    PREC sJBar_stress; //< Holds linearly mixed (1-JBar) for stress update, ((1 - FBAR_ratio)*(1-J) + (FBAR_ratio)*(1 - JBar))
    // PREC sJBar_strain;
    // PREC FBAR_ratio_strain , FBAR_ratio_stress;
    {
      PREC FBAR_ratio = pbuffer.FBAR_ratio; // Linear mixing ratio for FBAR anti-locking (0 = None, 1 = Full, Recommended 0.1 to 0.75 for very compressible materials, 0.99+ for nearly imcompressible materials)
      // FBAR_ratio_strain = pbuffer.FBAR_ratio_strain; // Linear ratio to mix sJ (normal) and sJBar (grid-averaged) values, produces value used for the strain update and saved as sJ on particle
      // FBAR_ratio_stress = pbuffer.FBAR_ratio_stress; // Linear ratio to mix sJ (normal) and sJBar (grid-averaged) values, produces value used for the stress update and saved as sJBar on particle
      sJ = sJ + (sJ * sJInc) - sJInc; // Update (1 - J)^n to (1 - J)^n+1

      // NOT SURE IF WE SHOULD EVOLVE sJBar_new HERE
      sJBar_new = sJBar_new + (sJInc * sJBar_new) - sJInc; // Update (1 - JBar)^n to (1 - JBar)^n+1
      
      // Mix (1-J^n+1) with (1-JBar^n+1) to get stable (1-JBar^n+1) for stress update
      // FBar_n+1 = (JBar_n+1 / J_n+1)^(1/3) * F_n+1
      sJBar_stress = (1.0 - FBAR_ratio) * sJ + (FBAR_ratio) * sJBar_new;

      if (pbuffer.alpha > 0) {
        // Position correction factor (ASFLIP). Larger if in tension
        PREC beta =  (sJBar_stress < 0) ? pbuffer.beta_max : pbuffer.beta_min; 
        // ASFLIP advection update
        pos += dt * (vel + beta * pbuffer.alpha * (vel_p - vel_FLIP));
        vel += pbuffer.alpha * (vel_p - vel_FLIP);
      } else {
        // Regular MLS-MPM / PIC advection update (No FLIP/ASFLIP)
        pos += dt * vel;
      }

    }

    PREC voln = (1.0 - sJ) * pbuffer.volume; //< Volume of the particle at this step
    pvec9 contrib;
    contrib.set(0.0);
    {
      // PREC pressure = (pbuffer.bulk / pbuffer.gamma) * (pow((1.0 - sJBar_new), -pbuffer.gamma) - 1.0);
      PREC pressure = (pbuffer.bulk / pbuffer.gamma) * expm1(-pbuffer.gamma*log1p(-sJBar_stress));
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
      // Merged affine matrix and stress contribution for MLS-MPM P2G
      // Q = APIC_affine_velocity_matrix * mass - cauchy_stress * volume * dt * (4 / dx^2)
      contrib = (C * pbuffer.mass - contrib * newDt) * Dp_inv; 
      {
        auto particle_bin = g_buckets_on_particle_buffer 
            ? next_pbuffer.ch(_0, next_pbuffer._binsts[src_blockno] + pidib / g_bin_capacity) 
            : next_pbuffer.ch(_0, partition._binsts[src_blockno] + pidib / g_bin_capacity);
        particle_bin.val(_0, pidib % g_bin_capacity) = pos[0]; //< x
        particle_bin.val(_1, pidib % g_bin_capacity) = pos[1]; //< y
        particle_bin.val(_2, pidib % g_bin_capacity) = pos[2]; //< z
        particle_bin.val(_3, pidib % g_bin_capacity) = sJ; //< (1 - J) 
        particle_bin.val(_4, pidib % g_bin_capacity) = vel[0]; //< vx
        particle_bin.val(_5, pidib % g_bin_capacity) = vel[1]; //< vy
        particle_bin.val(_6, pidib % g_bin_capacity) = vel[2]; //< vz
        particle_bin.val(_7, pidib % g_bin_capacity) = sJBar_stress; //< (1 - JBar)
        particle_bin.val(_8, pidib % g_bin_capacity) = ID;   //< ID
      }
    }

    // Handle particle advection (i.e. movement) between bins / blocks on our data-structure
    local_base_index = (pos * g_dx_inv_d + 0.5).cast<int>() - 1;
    {
      int direction_tag = dir_offset((base_index - 1) / g_blocksize -
                              (local_base_index - 1) / g_blocksize);
      if (g_buckets_on_particle_buffer)
        next_pbuffer.add_advection(partition, local_base_index - 1, direction_tag, pidib);
      else 
        partition.add_advection(local_base_index - 1, direction_tag, pidib);
    }
    // Recalculate shape-functions for new particle positions
#pragma unroll 3
    for (char dd = 0; dd < 3; ++dd) {
      local_pos[dd] = pos[dd] - local_base_index[dd] * g_dx_d;
      PREC d =
          (local_pos[dd] - static_cast<PREC>(std::floor(local_pos[dd] * g_dx_inv_d + 0.5) - 1) * g_dx_d) * g_dx_inv_d;
      dws(dd, 0) = 0.5 * (1.5 - d) * (1.5 - d);
      d -= 1.0;
      dws(dd, 1) = 0.75 - d * d;
      d = 0.5 + d;
      dws(dd, 2) = 0.5 * d * d;

      local_base_index[dd] = (((base_index[dd] - 1) & g_blockmask) + 1) +
                             local_base_index[dd] - base_index[dd];
    }
    // Add particle contributions to nearby grid-nodes
#pragma unroll 3
    for (char i = 0; i < 3; i++)
#pragma unroll 3
      for (char j = 0; j < 3; j++)
#pragma unroll 3
        for (char k = 0; k < 3; k++) {
          pos = pvec3{(PREC)i, (PREC)j, (PREC)k} * g_dx - local_pos;
          PREC W = dws(0, i) * dws(1, j) * dws(2, k);
          PREC wm = pbuffer.mass * W;
          PREC wv = voln * W; // Weighted volume
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
          // FBAR volume V and (1 - (V / Vo)Bar) * V
          atomicAdd(
              &p2gbuffer[7][local_base_index[0] + i][local_base_index[1] + j]
                        [local_base_index[2] + k],
              wv);
          atomicAdd(
              &p2gbuffer[8][local_base_index[0] + i][local_base_index[1] + j]
                        [local_base_index[2] + k],
              wv * sJBar_stress);
              // wv * (sJBar_new + sJBar_new * sJInc - sJInc));
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
    } else if (channelid == 7) {
      atomicAdd(&next_grid.ch(_0, blockno).val_1d(_7, c), val);
    } else if (channelid == 8) {
      atomicAdd(&next_grid.ch(_0, blockno).val_1d(_8, c), val);
    }
  }
}


// Uses (3 + 1 + 1 + 3 + 2) * (512) * (4 bytes) shmem
// 20 kB shmem per block single precision, 40 kB per block double precision
// __launch_bounds__(g_particle_batch, G_MAX_SHMEM_PER_SM / 20)
template <typename Partition, typename Grid>
__global__ void 
__launch_bounds__(g_particle_batch, 5)
g2p2g_FBar(double dt, double newDt, const ivec3 *__restrict__ blocks,
                      const ParticleBuffer<material_e::JFluid_FBAR> pbuffer,
                      ParticleBuffer<material_e::JFluid_FBAR> next_pbuffer,
                      const Partition prev_partition, Partition partition,
                      const Grid grid, Grid next_grid) { 
  constexpr std::size_t num_g2p_variables = (3 + 1); // vel, 1 - JBar
  constexpr std::size_t num_p2g_variables = (1 + 3 + 2); // mass, momentum + fint, vol, sJBar*vol
  
  static constexpr uint64_t numViPerBlock = g_blockvolume * num_g2p_variables;
  static constexpr uint64_t numViInArena = numViPerBlock << 3;
  static constexpr uint64_t shmem_offset = (g_blockvolume * num_g2p_variables) << 3;

  static constexpr uint64_t numMViPerBlock = g_blockvolume * num_p2g_variables;
  static constexpr uint64_t numMViInArena = numMViPerBlock << 3;

  static constexpr unsigned arenamask = (g_blocksize << 1) - 1;
  static constexpr unsigned arenabits = g_blockbits + 1;

  extern __shared__ char shmem[];
  using ViArena =
      PREC_G(*)[num_g2p_variables][g_blocksize << 1][g_blocksize << 1][g_blocksize << 1];
  using ViArenaRef =
      PREC_G(&)[num_g2p_variables][g_blocksize << 1][g_blocksize << 1][g_blocksize << 1];
  ViArenaRef __restrict__ g2pbuffer = *reinterpret_cast<ViArena>(shmem);
  using MViArena =
      PREC_G(*)[num_p2g_variables][g_blocksize << 1][g_blocksize << 1][g_blocksize << 1];
  using MViArenaRef =
      PREC_G(&)[num_p2g_variables][g_blocksize << 1][g_blocksize << 1][g_blocksize << 1];
  MViArenaRef __restrict__ p2gbuffer =
      *reinterpret_cast<MViArena>(shmem + shmem_offset * sizeof(PREC_G));

  ivec3 blockid;
  int src_blockno;
  if (blocks != nullptr) { 
    blockid = blocks[blockIdx.x]; // Halo block ID
    src_blockno = partition.query(blockid); // Halo block number
  } else { 
    if (partition._haloMarks[blockIdx.x]) return; // Return early if halo block
    src_blockno = blockIdx.x; // Non-halo block number
    blockid = partition._activeKeys[blockIdx.x]; // Non-halo block ID
  }
  int ppb = g_buckets_on_particle_buffer
          ? next_pbuffer._ppbs[src_blockno] 
          : partition._ppbs[src_blockno]; // Particles in block
  if (ppb == 0) return; // Return early if no particles in block

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
      val = grid_block.val_1d(_8, c); // Note, pulling 8th channel from grid into 6th channel of g2pbuffer as we skip channel 7 of grid
    g2pbuffer[channelid][cx + (local_block_id & 4 ? g_blocksize : 0)][cy + (local_block_id & 2 ? g_blocksize : 0)][cz + (local_block_id & 1 ? g_blocksize : 0)] = val;
  }
  __syncthreads();
  for (int base = threadIdx.x; base < numMViInArena; base += blockDim.x) {
    int loc = base;
    char z = loc & arenamask;
    char y = (loc >>= arenabits) & arenamask;
    char x = (loc >>= arenabits) & arenamask;
    p2gbuffer[loc >> arenabits][x][y][z] = static_cast<PREC_G>(0);
  }
  __syncthreads();
  // Start Grid-to-Particle, threads are particles, threads can iterate through multiple particles
  for (int pidib = threadIdx.x; pidib < ppb; pidib += blockDim.x) {
    int source_blockno, source_pidib;
    ivec3 base_index;
    {
      int advect = (g_buckets_on_particle_buffer)
                    ? next_pbuffer._blockbuckets[src_blockno * g_particle_num_per_block + pidib]
                    : partition._blockbuckets[src_blockno * g_particle_num_per_block + pidib];
      dir_components(advect / g_particle_num_per_block, base_index);
      base_index += blockid;
      source_blockno = prev_partition.query(base_index);
      source_pidib = advect % g_particle_num_per_block; // & (g_particle_num_per_block - 1);
      source_blockno = (g_buckets_on_particle_buffer)
                        ? pbuffer._binsts[source_blockno] + source_pidib / g_bin_capacity
                        : prev_partition._binsts[source_blockno] + source_pidib / g_bin_capacity;
    }
    pvec3 pos;  //< Particle position at n (previous step)
    PREC sJ, sJBar_old; //< 1 - Particle volume ratio at n (previous step)
    PREC ID; //< Particle ID
    {
      auto source_particle_bin = pbuffer.ch(_0, source_blockno);
      pos[0] = source_particle_bin.val(_0, source_pidib % g_bin_capacity);  //< x
      pos[1] = source_particle_bin.val(_1, source_pidib % g_bin_capacity);  //< y
      pos[2] = source_particle_bin.val(_2, source_pidib % g_bin_capacity);  //< z
      sJ = source_particle_bin.val(_3, source_pidib % g_bin_capacity);       //< Vo/V
      // sJBar_old = source_particle_bin.val(_4, source_pidib % g_bin_capacity); //< (1 - Vo/V)Bar
      ID  = source_particle_bin.val(_5, source_pidib % g_bin_capacity); //< Particle ID
    }
    ivec3 local_base_index = (pos * g_dx_inv_d + 0.5).cast<int>() - 1;
    pvec3 local_pos = pos - local_base_index * g_dx_d;
    base_index = local_base_index;

    pvec3x3 dws;
#pragma unroll 3
    for (int dd = 0; dd < 3; ++dd) {
      PREC d = (local_pos[dd] - static_cast<PREC>(std::floor(local_pos[dd] * g_dx_inv_d + 0.5) - 1) * g_dx_d) * g_dx_inv_d;
      dws(dd, 0) = 0.5 * (1.5 - d) * (1.5 - d);
      d -= 1.0;
      dws(dd, 1) = 0.75 - d * d;
      d = 0.5 + d;
      dws(dd, 2) = 0.5 * d * d;
      local_base_index[dd] = ((local_base_index[dd] - 1) & g_blockmask) + 1;
    }
    pvec3 vel;   //< Stressed, collided grid velocity
    pvec3 vel_FLIP; //< Unstressed, uncollided grid velocity
    pvec9 C;     //< APIC affine matrix, used a few times
    PREC sJBar_new; //< New value of (1 - JBar^n+1) to calculate this step
    vel.set(0.0); vel_FLIP.set(0.0); C.set(0.0); // Zero out all values
    sJBar_new = 0.0; // Zero out. We will sum into this variable

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
          pvec3 xixp = pvec3{(PREC)i, (PREC)j, (PREC)k} * g_dx_d - local_pos;
          PREC W = dws(0, i) * dws(1, j) * dws(2, k);
          pvec3 vi{g2pbuffer[0][local_base_index[0] + i][local_base_index[1] + j]
                           [local_base_index[2] + k],
                  g2pbuffer[1][local_base_index[0] + i][local_base_index[1] + j]
                           [local_base_index[2] + k],
                  g2pbuffer[2][local_base_index[0] + i][local_base_index[1] + j]
                           [local_base_index[2] + k]};
          PREC sJBar_i = g2pbuffer[3][local_base_index[0] + i][local_base_index[1] + j]
                            [local_base_index[2] + k]; // Assumes we ran following calc in grid update kernel: sJBar_i = Sum_p (wip * sJBar_p * voln_p) / Sum_p (wip * voln_p) 
          vel   += vi * W;
          C[0] += W * vi[0] * xixp[0] * scale;
          C[1] += W * vi[1] * xixp[0] * scale;
          C[2] += W * vi[2] * xixp[0] * scale;
          C[3] += W * vi[0] * xixp[1] * scale;
          C[4] += W * vi[1] * xixp[1] * scale;
          C[5] += W * vi[2] * xixp[1] * scale;
          C[6] += W * vi[0] * xixp[2] * scale;
          C[7] += W * vi[1] * xixp[2] * scale;
          C[8] += W * vi[2] * xixp[2] * scale;
          sJBar_new += sJBar_i * W;
        }

    PREC sJInc = (C[0] + C[4] + C[8]) * dt * Dp_inv; // sJInc = (tr(A)*dt*D_p^[inv] + 1 ) - 1)
    PREC sJBar_stress; //< Holds linearly mixed (1-JBar) for stress update, ((1 - FBAR_ratio)*(1-J) + (FBAR_ratio)*(1 - JBar))
    // PREC sJBar_strain;
    // PREC FBAR_ratio_strain , FBAR_ratio_stress;
    {
      PREC FBAR_ratio = pbuffer.FBAR_ratio; // Linear mixing ratio for FBAR anti-locking (0 = None, 1 = Full, Recommended 0.1 to 0.75 for very compressible materials, 0.99+ for nearly imcompressible materials)
      // FBAR_ratio_strain = pbuffer.FBAR_ratio_strain; // Linear ratio to mix sJ (normal) and sJBar (grid-averaged) values, produces value used for the strain update and saved as sJ on particle
      // FBAR_ratio_stress = pbuffer.FBAR_ratio_stress; // Linear ratio to mix sJ (normal) and sJBar (grid-averaged) values, produces value used for the stress update and saved as sJBar on particle
      sJ = sJ + (sJ * sJInc) - sJInc; // Update (1 - J)^n to (1 - J)^n+1

      // NOT SURE IF WE SHOULD EVOLVE sJBar_new HERE
      sJBar_new = sJBar_new + (sJInc * sJBar_new) - sJInc; // Update (1 - JBar)^n to (1 - JBar)^n+1
      
      // Mix (1-J^n+1) with (1-JBar^n+1) to get stable (1-JBar^n+1) for stress update
      // FBar_n+1 = (JBar_n+1 / J_n+1)^(1/3) * F_n+1
      sJBar_stress = (1.0 - FBAR_ratio) * sJ + (FBAR_ratio) * sJBar_new;
     
      // Regular MLS-MPM / PIC advection update (No FLIP/ASFLIP)
      pos += dt * vel;
    }

    PREC voln = (1.0 - sJ) * pbuffer.volume; //< Volume of the particle at this step
    pvec9 contrib;
    contrib.set(0.0);
    {
      // PREC pressure = (pbuffer.bulk / pbuffer.gamma) * (pow((1.0 - sJBar_new), -pbuffer.gamma) - 1.0);
      // Update particle stress to t = n+1
      PREC pressure = (pbuffer.bulk / pbuffer.gamma) * expm1(-pbuffer.gamma*log1p(-sJBar_stress));
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
      // Merged affine matrix and stress contribution for MLS-MPM P2G
      // Q = APIC_affine_velocity_matrix * mass - cauchy_stress * volume * dt * (4 / dx^2)
      contrib = (C * pbuffer.mass - contrib * newDt) * Dp_inv; 
      // Save particle values at t = n+1 to particleBuffer (global memory)
      {
        auto particle_bin = g_buckets_on_particle_buffer 
            ? next_pbuffer.ch(_0, next_pbuffer._binsts[src_blockno] + pidib / g_bin_capacity) 
            : next_pbuffer.ch(_0, partition._binsts[src_blockno] + pidib / g_bin_capacity);
        particle_bin.val(_0, pidib % g_bin_capacity) = pos[0]; //< x
        particle_bin.val(_1, pidib % g_bin_capacity) = pos[1]; //< y
        particle_bin.val(_2, pidib % g_bin_capacity) = pos[2]; //< z
        particle_bin.val(_3, pidib % g_bin_capacity) = sJ; //< (1 - J) 
        particle_bin.val(_4, pidib % g_bin_capacity) = sJBar_stress; //< (1 - JBar)
        particle_bin.val(_5, pidib % g_bin_capacity) = ID;   //< ID
      }
    }

    // Compute particle advection (i.e. movement of IDs) between bins / blocks
    local_base_index = (pos * g_dx_inv_d + 0.5).cast<int>() - 1;
    {
      int direction_tag = dir_offset((base_index - 1) / g_blocksize -
                              (local_base_index - 1) / g_blocksize);
      if (g_buckets_on_particle_buffer)
        next_pbuffer.add_advection(partition, local_base_index - 1, direction_tag, pidib);
      else 
        partition.add_advection(local_base_index - 1, direction_tag, pidib);
    }
    // Recalculate shape-functions for new particle positions
#pragma unroll 3
    for (char dd = 0; dd < 3; ++dd) {
      local_pos[dd] = pos[dd] - local_base_index[dd] * g_dx_d;
      PREC d =
          (local_pos[dd] - static_cast<PREC>(std::floor(local_pos[dd] * g_dx_inv_d + 0.5) - 1) * g_dx_d) * g_dx_inv_d;
      dws(dd, 0) = 0.5 * (1.5 - d) * (1.5 - d);
      d -= 1.0;
      dws(dd, 1) = 0.75 - d * d;
      d = 0.5 + d;
      dws(dd, 2) = 0.5 * d * d;
      local_base_index[dd] = (((base_index[dd] - 1) & g_blockmask) + 1) +
                             local_base_index[dd] - base_index[dd];
    }
    // Add particle contributions to nearby grid-nodes on p2gbuffer (shared memory)
#pragma unroll 3
    for (char i = 0; i < 3; i++)
#pragma unroll 3
      for (char j = 0; j < 3; j++)
#pragma unroll 3
        for (char k = 0; k < 3; k++) {
          pos = pvec3{(PREC)i, (PREC)j, (PREC)k} * g_dx - local_pos;
          PREC W = dws(0, i) * dws(1, j) * dws(2, k);
          PREC wm = pbuffer.mass * W;
          PREC wv = voln * W; // Weighted volume
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
          // FBAR volume V and (1 - (V / Vo)Bar) * V
          atomicAdd(
              &p2gbuffer[4][local_base_index[0] + i][local_base_index[1] + j]
                        [local_base_index[2] + k],
              wv);
          atomicAdd(
              &p2gbuffer[5][local_base_index[0] + i][local_base_index[1] + j]
                        [local_base_index[2] + k],
              wv * sJBar_stress);
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
      atomicAdd(&next_grid.ch(_0, blockno).val_1d(_7, c), val);
    } else if (channelid == 5) {
      atomicAdd(&next_grid.ch(_0, blockno).val_1d(_8, c), val);
    }
  }
}


template <typename ParticleBuffer, typename Partition, typename Grid>
__global__ void g2p_FBar(double dt, double newDt, const ivec3 *__restrict__ blocks,
                      const ParticleBuffer pbuffer,
                      ParticleBuffer next_pbuffer,
                      const Partition prev_partition, Partition partition,
                      Grid grid, Grid next_grid) {}

// shmem per block = (3 + 2) * (512) * (4 bytes) = 10 kB single precision, 20 kB double precision
template <typename Partition, typename Grid>
__global__ void 
__launch_bounds__(g_particle_batch, static_cast<unsigned>(G_MAX_SHMEM_PER_SM / 10))
g2p_FBar(double dt, double newDt, const ivec3 *__restrict__ blocks,
                      const ParticleBuffer<material_e::JBarFluid> pbuffer,
                      ParticleBuffer<material_e::JBarFluid> next_pbuffer,
                      const Partition prev_partition, Partition partition,
                      Grid grid, Grid next_grid) {
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
    blockid = blocks[blockIdx.x]; // Halo block ID
    src_blockno = partition.query(blockid); // Halo block number
  } else { 
    if (partition._haloMarks[blockIdx.x]) return; // Return early if halo block
    blockid = partition._activeKeys[blockIdx.x]; // Non-halo block ID
    src_blockno = blockIdx.x; // Non-halo block number
  }
  if (src_blockno < 0) return; // Return early if negative block number
//  else if (src_blockno > blockcnt) return; // Return early if excessive block number
  int ppb = g_buckets_on_particle_buffer
            ? next_pbuffer._ppbs[src_blockno] 
            : partition._ppbs[src_blockno]; // Particles in block
  if (ppb == 0) return; // Return early if no particles


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
    p2gbuffer[loc >> arenabits][x][y][z] = static_cast<PREC_G>(0);
  }
  __syncthreads();
  for (int pidib = threadIdx.x; pidib < ppb; pidib += blockDim.x) {
    int source_blockno, source_pidib;
    ivec3 base_index;
    {
      int advect = (g_buckets_on_particle_buffer)
                    ? next_pbuffer._blockbuckets[src_blockno * g_particle_num_per_block + pidib]
                    : partition._blockbuckets[src_blockno * g_particle_num_per_block + pidib];
      dir_components(advect / g_particle_num_per_block, base_index);
      base_index += blockid;
      source_blockno = prev_partition.query(base_index);
      source_pidib = advect % g_particle_num_per_block; // & (g_particle_num_per_block - 1);
      source_blockno = (g_buckets_on_particle_buffer)
                        ? pbuffer._binsts[source_blockno] + source_pidib / g_bin_capacity
                        : prev_partition._binsts[source_blockno] + source_pidib / g_bin_capacity;
    }
    pvec3 pos;  //< Particle position at n
    PREC sJ;   //< Particle volume ratio at n
    PREC sJBar;
    {
      auto source_particle_bin = pbuffer.ch(_0, source_blockno);
      pos[0] = source_particle_bin.val(_0, source_pidib % g_bin_capacity);  //< x
      pos[1] = source_particle_bin.val(_1, source_pidib % g_bin_capacity);  //< y
      pos[2] = source_particle_bin.val(_2, source_pidib % g_bin_capacity);  //< z
      sJ = source_particle_bin.val(_3, source_pidib % g_bin_capacity);       //< Vo/V
      sJBar = source_particle_bin.val(_7, source_pidib % g_bin_capacity); //< JBar tn
    }
    ivec3 local_base_index = (pos * g_dx_inv_d + 0.5).cast<int>() - 1;
    pvec3 local_pos = pos - local_base_index * g_dx_d;
    base_index = local_base_index;

    pvec3x3 dws;
#pragma unroll 3
    for (int dd = 0; dd < 3; ++dd) {
      PREC d = (local_pos[dd] - static_cast<PREC>(std::floor(local_pos[dd] * g_dx_inv_d + 0.5) - 1) * g_dx_d) * g_dx_inv_d;
      dws(dd, 0) = 0.5 * (1.5 - d) * (1.5 - d);
      d -= 1.0;
      dws(dd, 1) = 0.75 - d * d;
      d = 0.5 + d;
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
    PREC sJInc = ((C[0] + C[4] + C[8]) * dt * Dp_inv); // J^n+1 / J^n
    PREC voln = (1.0 - sJ) * pbuffer.volume; // vol^n+1, Send to grid for Simple FBar 
    //PREC J = JInc * (1.0 - sJ); // J^n+1
    //float voln = J * pbuffer.volume; // vol^n+1, Send to grid for Simple FBar 
    
#pragma unroll 3
    for (char i = 0; i < 3; i++)
#pragma unroll 3
      for (char j = 0; j < 3; j++)
#pragma unroll 3
        for (char k = 0; k < 3; k++) {
          pos = pvec3{(PREC)i, (PREC)j, (PREC)k} * g_dx - local_pos;
          PREC W = dws(0, i) * dws(1, j) * dws(2, k);
          PREC wv = voln * W; // Weighted volume
          atomicAdd(
              &p2gbuffer[0][local_base_index[0] + i][local_base_index[1] + j]
                        [local_base_index[2] + k],
              wv);
          atomicAdd(
              &p2gbuffer[1][local_base_index[0] + i][local_base_index[1] + j]
                        [local_base_index[2] + k],
              wv * (sJBar + sJBar * sJInc - sJInc));
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


template <typename Partition, typename Grid>
__global__ void g2p_FBar(double dt, double newDt, const ivec3 *__restrict__ blocks,
                      const ParticleBuffer<material_e::JFluid_FBAR> pbuffer,
                      ParticleBuffer<material_e::JFluid_FBAR> next_pbuffer,
                      const Partition prev_partition, Partition partition,
                      Grid grid, Grid next_grid) {
  static constexpr uint64_t numViPerBlock = g_blockvolume * 3;
  static constexpr uint64_t numViInArena = numViPerBlock << 3;
  static constexpr uint64_t shmem_offset = (g_blockvolume * 3) << 3;

  static constexpr uint64_t numMViPerBlock = g_blockvolume * 2;
  static constexpr uint64_t numMViInArena = numMViPerBlock << 3;

  static constexpr unsigned arenamask = (g_blocksize << 1) - 1;
  static constexpr unsigned arenabits = g_blockbits + 1;

  extern __shared__ char shmem[];
  using ViArena =
      PREC_G(*)[3][(g_blocksize << 1) ][(g_blocksize << 1) ][(g_blocksize << 1) ];
  using ViArenaRef =
      PREC_G(&)[3][(g_blocksize << 1) ][(g_blocksize << 1) ][(g_blocksize << 1) ];
  ViArenaRef __restrict__ g2pbuffer = *reinterpret_cast<ViArena>(shmem);
  using MViArena =
      PREC_G(*)[2][(g_blocksize << 1) ][(g_blocksize << 1) ][(g_blocksize << 1) ];
  using MViArenaRef =
      PREC_G(&)[2][(g_blocksize << 1) ][(g_blocksize << 1) ][(g_blocksize << 1) ];
  MViArenaRef __restrict__ p2gbuffer =
      *reinterpret_cast<MViArena>(shmem + shmem_offset * sizeof(PREC_G));

  ivec3 blockid;
  int src_blockno;
  if (blocks != nullptr) { 
    blockid = blocks[blockIdx.x]; // Halo block ID
    src_blockno = partition.query(blockid); // Halo block number
  } else { 
    if (partition._haloMarks[blockIdx.x]) return; // Return early if halo block
    blockid = partition._activeKeys[blockIdx.x]; // Non-halo block ID
    src_blockno = blockIdx.x; // Non-halo block number
  }
  if (src_blockno < 0) return; // Return early if negative block number
//  else if (src_blockno > blockcnt) return; // Return early if excessive block number
  int ppb = g_buckets_on_particle_buffer
            ? next_pbuffer._ppbs[src_blockno] 
            : partition._ppbs[src_blockno]; // Particles in block
  if (ppb == 0) return; // Return early if no particles


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
    p2gbuffer[loc >> arenabits][x][y][z] = static_cast<PREC_G>(0);
  }
  __syncthreads();
  for (int pidib = threadIdx.x; pidib < ppb; pidib += blockDim.x) {
    int source_blockno, source_pidib;
    ivec3 base_index;
    {
      int advect = (g_buckets_on_particle_buffer)
                    ? next_pbuffer._blockbuckets[src_blockno * g_particle_num_per_block + pidib]
                    : partition._blockbuckets[src_blockno * g_particle_num_per_block + pidib];
      dir_components(advect / g_particle_num_per_block, base_index);
      base_index += blockid;
      source_blockno = prev_partition.query(base_index);
      source_pidib = advect % g_particle_num_per_block; // & (g_particle_num_per_block - 1);
      source_blockno = (g_buckets_on_particle_buffer)
                        ? pbuffer._binsts[source_blockno] + source_pidib / g_bin_capacity
                        : prev_partition._binsts[source_blockno] + source_pidib / g_bin_capacity;
    }
    pvec3 pos;  //< Particle position at n
    PREC sJ;   //< Particle volume ratio at n
    PREC sJBar;
    {
      auto source_particle_bin = pbuffer.ch(_0, source_blockno);
      pos[0] = source_particle_bin.val(_0, source_pidib % g_bin_capacity);  //< x
      pos[1] = source_particle_bin.val(_1, source_pidib % g_bin_capacity);  //< y
      pos[2] = source_particle_bin.val(_2, source_pidib % g_bin_capacity);  //< z
      sJ = source_particle_bin.val(_3, source_pidib % g_bin_capacity);       //< Vo/V
      sJBar = source_particle_bin.val(_4, source_pidib % g_bin_capacity); //< JBar tn
    }
    ivec3 local_base_index = (pos * g_dx_inv_d + 0.5).cast<int>() - 1;
    pvec3 local_pos = pos - local_base_index * g_dx_d;
    base_index = local_base_index;

    pvec3x3 dws;
#pragma unroll 3
    for (int dd = 0; dd < 3; ++dd) {
      PREC d = (local_pos[dd] - static_cast<PREC>(std::floor(local_pos[dd] * g_dx_inv_d + 0.5) - 1) * g_dx_d) * g_dx_inv_d;
      dws(dd, 0) = 0.5 * (1.5 - d) * (1.5 - d);
      d -= 1.0;
      dws(dd, 1) = 0.75 - d * d;
      d = 0.5 + d;
      dws(dd, 2) = 0.5 * d * d;
      local_base_index[dd] = ((local_base_index[dd] - 1) & g_blockmask) + 1;
    }

    pvec3 vel;   //< PIC, Stressed, collided grid velocity
    pvec9 C;     //< APIC affine matrix, used a few times
    vel.set(0.0); 
    C.set(0.0);

    PREC Dp_inv; //< Inverse Intertia-Like Tensor (1/m^2)
    PREC scale = pbuffer.length * pbuffer.length; //< Area scale (m^2)
    Dp_inv = g_D_inv / scale; //< Scalar 4/(dx^2) for Quad. B-Spline

#pragma unroll 3
    for (char i = 0; i < 3; i++)
#pragma unroll 3
      for (char j = 0; j < 3; j++)
#pragma unroll 3
        for (char k = 0; k < 3; k++) {
            pvec3 xixp = pvec3{(PREC)i, (PREC)j, (PREC)k} * g_dx_d - local_pos;
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
    PREC sJInc = ((C[0] + C[4] + C[8]) * dt * Dp_inv); // J^n+1 / J^n
    PREC voln = (1.0 - sJ) * pbuffer.volume; // vol^n+1, Send to grid for Simple FBar 


// SHIFT KERNEL
    // local_base_index = (pos * g_dx_inv + 0.5f).cast<int>() - 1;
//     local_pos = (pos) - local_base_index * g_dx;

// #pragma unroll 3
//     for (int dd = 0; dd < 3; ++dd) {
//       PREC d =
//           (local_pos[dd] - ((int)(local_pos[dd] * g_dx_inv + 0.5f) - 1) * g_dx) *
//           g_dx_inv;
//       dws(dd, 0) = 0.5 * (1.5 - d) * (1.5 - d);
//       d -= 1.0;
//       dws(dd, 1) = 0.75 - d * d;
//       d = 0.5 + d;
//       dws(dd, 2) = 0.5 * d * d;
//       local_base_index[dd] = ((local_base_index[dd] - 1) & g_blockmask) + 1;
//     }


//     local_base_index = (pos * g_dx_inv + 0.5f).cast<int>();
//     local_pos = (pos +0.5*g_dx) - local_base_index * g_dx;

//     //pvec2x2 dws;
// #pragma unroll 3
//     for (int dd = 0; dd < 3; ++dd) {
//       local_base_index[dd] = ((local_base_index[dd] - 1) & g_blockmask) + 1;
//     }

// #pragma unroll 2
//     for (char i = -1; i < 2; i+=2)
// #pragma unroll 2
//       for (char j = -1; j < 2; j+=2)
// #pragma unroll 2
//         for (char k = -1; k < 2; k+=2) {
//           //pos = pvec3{(PREC)i, (PREC)j, (PREC)k} * g_dx - local_pos;
//           //PREC W = dws(0, i) * dws(1, j) * dws(2, k);
//           PREC W = 0.125 * (1 + i * 2 * (local_pos[0] * g_dx_inv - 0.5)) * (1 + j * 2 * (local_pos[1] * g_dx_inv - 0.5)) * (1 + k * 2 * (local_pos[2] * g_dx_inv - 0.5));
//           PREC wv = voln * W; // Weighted volume
//           atomicAdd(
//               &p2gbuffer[0][local_base_index[0] + i][local_base_index[1] + j]
//                         [local_base_index[2] + k],
//               wv);
//           atomicAdd(
//               &p2gbuffer[1][local_base_index[0] + i][local_base_index[1] + j]
//                         [local_base_index[2] + k],
//               wv * (sJBar + sJBar * JInc - JInc));
//         }


#pragma unroll 3
    for (char i = 0; i < 3; i++)
#pragma unroll 3
      for (char j = 0; j < 3; j++)
#pragma unroll 3
        for (char k = 0; k < 3; k++) {
          // if (local_base_index[0] + i > 0 && local_base_index[0] + i < arenamask) {
          //   if (local_base_index[1] + j > 0 && local_base_index[1] + j < arenamask) {
          //     if (local_base_index[2] + k > 0 && local_base_index[2] + k < arenamask) {
          PREC W = dws(0, i) * dws(1, j) * dws(2, k);
          PREC wv = voln * W; // Weighted volume
          atomicAdd(
              &p2gbuffer[0][local_base_index[0] + i][local_base_index[1] + j]
                        [local_base_index[2] + k],
              wv);
          atomicAdd(
              &p2gbuffer[1][local_base_index[0] + i][local_base_index[1] + j]
                        [local_base_index[2] + k],
              wv * (sJBar + sJBar * sJInc - sJInc));
          //     }
          //   }
          // }
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
    // if (cz > 0 && cz < arenamask) {      
    //   if (cy > 0 && cy < arenamask) {
    //     if (cx > 0 && cx < arenamask) {
    PREC_G val = p2gbuffer[channelid][cx + (local_block_id & 4 ? g_blocksize : 0)][cy + (local_block_id & 2 ? g_blocksize : 0)][cz + (local_block_id & 1 ? g_blocksize : 0)];
    if (channelid == 0) {
      atomicAdd(&grid.ch(_0, blockno).val_1d(_7, c), val); //< Vol
    } else if (channelid == 1) {
      atomicAdd(&grid.ch(_0, blockno).val_1d(_8, c), val); //< JBar_JInc Vol
    }
    //     }
    //   }
    // } 
  }
}


template <typename Partition, typename Grid>
__global__ void g2p_FBar(double dt, double newDt, const ivec3 *__restrict__ blocks,
                      const ParticleBuffer<material_e::FixedCorotated_FBAR> pbuffer,
                      ParticleBuffer<material_e::FixedCorotated_FBAR> next_pbuffer,
                      const Partition prev_partition, Partition partition,
                      Grid grid, Grid next_grid) {
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
    blockid = blocks[blockIdx.x]; // Halo block ID
    src_blockno = partition.query(blockid); // Halo block number
  } else { 
    if (partition._haloMarks[blockIdx.x]) return; // Return early if halo block
    blockid = partition._activeKeys[blockIdx.x]; // Non-halo block ID
    src_blockno = blockIdx.x; // Non-halo block number
  }
  if (src_blockno < 0) return; // Return early if negative block number
//  else if (src_blockno > blockcnt) return; // Return early if excessive block number
  int ppb = g_buckets_on_particle_buffer
            ? next_pbuffer._ppbs[src_blockno] 
            : partition._ppbs[src_blockno]; // Particles in block
  if (ppb == 0) return; // Return early if no particles


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
    p2gbuffer[loc >> arenabits][x][y][z] = (PREC_G)0.0;
  }
  __syncthreads();
  for (int pidib = threadIdx.x; pidib < ppb; pidib += blockDim.x) {
    int source_blockno, source_pidib;
    ivec3 base_index;
    {
      int advect = (g_buckets_on_particle_buffer)
                    ? next_pbuffer._blockbuckets[src_blockno * g_particle_num_per_block + pidib]
                    : partition._blockbuckets[src_blockno * g_particle_num_per_block + pidib];
      dir_components(advect / g_particle_num_per_block, base_index);
      base_index += blockid;
      source_blockno = prev_partition.query(base_index);
      source_pidib = advect % g_particle_num_per_block; // & (g_particle_num_per_block - 1);
      source_blockno = (g_buckets_on_particle_buffer)
                        ? pbuffer._binsts[source_blockno] + source_pidib / g_bin_capacity
                        : prev_partition._binsts[source_blockno] + source_pidib / g_bin_capacity;
    }
    pvec3 pos;  //< Particle position at n
    pvec9 F;
    PREC sJBar;
    {
      auto source_particle_bin = pbuffer.ch(_0, source_blockno);
      pos[0] = source_particle_bin.val(_0, source_pidib % g_bin_capacity);  //< x
      pos[1] = source_particle_bin.val(_1, source_pidib % g_bin_capacity);  //< y
      pos[2] = source_particle_bin.val(_2, source_pidib % g_bin_capacity);  //< z
      F[0] = source_particle_bin.val(_3, source_pidib % g_bin_capacity);
      F[1] = source_particle_bin.val(_4, source_pidib % g_bin_capacity);
      F[2] = source_particle_bin.val(_5, source_pidib % g_bin_capacity);
      F[3] = source_particle_bin.val(_6, source_pidib % g_bin_capacity);
      F[4] = source_particle_bin.val(_7, source_pidib % g_bin_capacity);
      F[5] = source_particle_bin.val(_8, source_pidib % g_bin_capacity);
      F[6] = source_particle_bin.val(_9, source_pidib % g_bin_capacity);
      F[7] = source_particle_bin.val(_10, source_pidib % g_bin_capacity);
      F[8] = source_particle_bin.val(_11, source_pidib % g_bin_capacity);
      // Skip velocity
      sJBar = source_particle_bin.val(_12, source_pidib % g_bin_capacity); //< JBar tn
      // Skip ID
    }
    ivec3 local_base_index = (pos * g_dx_inv_d + 0.5).cast<int>() - 1;
    pvec3 local_pos = pos - local_base_index * g_dx_d;
    base_index = local_base_index;

    pvec3x3 dws;
#pragma unroll 3
    for (int dd = 0; dd < 3; ++dd) {
      PREC d = (local_pos[dd] - static_cast<PREC>(std::floor(local_pos[dd] * g_dx_inv_d + 0.5) - 1) * g_dx_d) * g_dx_inv_d;
      dws(dd, 0) = 0.5 * (1.5 - d) * (1.5 - d);
      d -= 1.0;
      dws(dd, 1) = 0.75 - d * d;
      d = 0.5 + d;
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
          pvec3 xixp = pvec3{(PREC)i, (PREC)j, (PREC)k} * g_dx_d - local_pos;
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
    pvec9 FInc;
#pragma unroll 9
    for (int d = 0; d < 9; ++d)
      FInc[d] = C[d] * dt * Dp_inv + ((d & 0x3) ? 0.0 : 1.0);
    PREC JInc = matrixDeterminant3d(FInc.data()); // J^n+1 / J^n
    PREC J  = matrixDeterminant3d(F.data());
    PREC voln = J * pbuffer.volume; // vol^n+1, Send to grid for Simple FBar 
#pragma unroll 3
    for (char i = 0; i < 3; i++)
#pragma unroll 3
      for (char j = 0; j < 3; j++)
#pragma unroll 3
        for (char k = 0; k < 3; k++) {
          pos = pvec3{(PREC)i, (PREC)j, (PREC)k} * g_dx - local_pos;
          PREC W = dws(0, i) * dws(1, j) * dws(2, k);
          PREC wv = voln * W; // Weighted volume
          atomicAdd(
              &p2gbuffer[0][local_base_index[0] + i][local_base_index[1] + j]
                        [local_base_index[2] + k],
              wv);
          atomicAdd(
              &p2gbuffer[1][local_base_index[0] + i][local_base_index[1] + j]
                        [local_base_index[2] + k],
              wv * ((sJBar * JInc) - JInc + 1.0));
              //wv * (sJBar + sJBar * sJInc - sJInc)); // using sJInc = (1-JInc)
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

template <typename Partition, typename Grid>
__global__ void g2p_FBar(double dt, double newDt, const ivec3 *__restrict__ blocks,
                      const ParticleBuffer<material_e::FixedCorotated_ASFLIP_FBAR> pbuffer,
                      ParticleBuffer<material_e::FixedCorotated_ASFLIP_FBAR> next_pbuffer,
                      const Partition prev_partition, Partition partition,
                      Grid grid, Grid next_grid) {
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
    blockid = blocks[blockIdx.x]; // Halo block ID
    src_blockno = partition.query(blockid); // Halo block number
  } else { 
    if (partition._haloMarks[blockIdx.x]) return; // Return early if halo block
    blockid = partition._activeKeys[blockIdx.x]; // Non-halo block ID
    src_blockno = blockIdx.x; // Non-halo block number
  }
  if (src_blockno < 0) return; // Return early if negative block number
//  else if (src_blockno > blockcnt) return; // Return early if excessive block number
  int ppb = g_buckets_on_particle_buffer
            ? next_pbuffer._ppbs[src_blockno] 
            : partition._ppbs[src_blockno]; // Particles in block
  if (ppb == 0) return; // Return early if no particles


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
    p2gbuffer[loc >> arenabits][x][y][z] = (PREC_G)0.0;
  }
  __syncthreads();
  for (int pidib = threadIdx.x; pidib < ppb; pidib += blockDim.x) {
    int source_blockno, source_pidib;
    ivec3 base_index;
    {
      int advect = (g_buckets_on_particle_buffer)
                    ? next_pbuffer._blockbuckets[src_blockno * g_particle_num_per_block + pidib]
                    : partition._blockbuckets[src_blockno * g_particle_num_per_block + pidib];
      dir_components(advect / g_particle_num_per_block, base_index);
      base_index += blockid;
      source_blockno = prev_partition.query(base_index);
      source_pidib = advect % g_particle_num_per_block; // & (g_particle_num_per_block - 1);
      source_blockno = (g_buckets_on_particle_buffer)
                        ? pbuffer._binsts[source_blockno] + source_pidib / g_bin_capacity
                        : prev_partition._binsts[source_blockno] + source_pidib / g_bin_capacity;
    }
    pvec3 pos;  //< Particle position at n
    pvec9 F;
    PREC sJBar;
    {
      auto source_particle_bin = pbuffer.ch(_0, source_blockno);
      pos[0] = source_particle_bin.val(_0, source_pidib % g_bin_capacity);  //< x
      pos[1] = source_particle_bin.val(_1, source_pidib % g_bin_capacity);  //< y
      pos[2] = source_particle_bin.val(_2, source_pidib % g_bin_capacity);  //< z
      F[0] = source_particle_bin.val(_3, source_pidib % g_bin_capacity);
      F[1] = source_particle_bin.val(_4, source_pidib % g_bin_capacity);
      F[2] = source_particle_bin.val(_5, source_pidib % g_bin_capacity);
      F[3] = source_particle_bin.val(_6, source_pidib % g_bin_capacity);
      F[4] = source_particle_bin.val(_7, source_pidib % g_bin_capacity);
      F[5] = source_particle_bin.val(_8, source_pidib % g_bin_capacity);
      F[6] = source_particle_bin.val(_9, source_pidib % g_bin_capacity);
      F[7] = source_particle_bin.val(_10, source_pidib % g_bin_capacity);
      F[8] = source_particle_bin.val(_11, source_pidib % g_bin_capacity);
      // Skip velocity
      sJBar = source_particle_bin.val(_15, source_pidib % g_bin_capacity); //< JBar tn
      // Skip ID
    }
    ivec3 local_base_index = (pos * g_dx_inv_d + 0.5).cast<int>() - 1;
    pvec3 local_pos = pos - local_base_index * g_dx_d;
    base_index = local_base_index;

    pvec3x3 dws;
#pragma unroll 3
    for (int dd = 0; dd < 3; ++dd) {
      PREC d = (local_pos[dd] - static_cast<PREC>(std::floor(local_pos[dd] * g_dx_inv_d + 0.5) - 1) * g_dx_d) * g_dx_inv_d;
      dws(dd, 0) = 0.5 * (1.5 - d) * (1.5 - d);
      d -= 1.0;
      dws(dd, 1) = 0.75 - d * d;
      d = 0.5 + d;
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
          pvec3 xixp = pvec3{(PREC)i, (PREC)j, (PREC)k} * g_dx_d - local_pos;
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
    pvec9 FInc;
#pragma unroll 9
    for (int d = 0; d < 9; ++d)
      FInc[d] = C[d] * dt * Dp_inv + ((d & 0x3) ? 0.0 : 1.0);
    PREC JInc = matrixDeterminant3d(FInc.data()); // J^n+1 / J^n
    PREC J  = matrixDeterminant3d(F.data());
    PREC voln = J * pbuffer.volume; // vol^n+1, Send to grid for Simple FBar 
#pragma unroll 3
    for (char i = 0; i < 3; i++)
#pragma unroll 3
      for (char j = 0; j < 3; j++)
#pragma unroll 3
        for (char k = 0; k < 3; k++) {
          pos = pvec3{(PREC)i, (PREC)j, (PREC)k} * g_dx - local_pos;
          PREC W = dws(0, i) * dws(1, j) * dws(2, k);
          PREC wv = voln * W; // Weighted volume
          atomicAdd(
              &p2gbuffer[0][local_base_index[0] + i][local_base_index[1] + j]
                        [local_base_index[2] + k],
              wv);
          atomicAdd(
              &p2gbuffer[1][local_base_index[0] + i][local_base_index[1] + j]
                        [local_base_index[2] + k],
              wv * ((sJBar * JInc) - JInc + 1.0));
              //wv * (sJBar + sJBar * sJInc - sJInc)); // using sJInc = (1-JInc)
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



template <typename Partition, typename Grid>
__global__ void g2p_FBar(double dt, double newDt, const ivec3 *__restrict__ blocks,
                      const ParticleBuffer<material_e::NeoHookean_FBAR> pbuffer,
                      ParticleBuffer<material_e::NeoHookean_FBAR> next_pbuffer,
                      const Partition prev_partition, Partition partition,
                      Grid grid, Grid next_grid) {
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
    blockid = blocks[blockIdx.x]; // Halo block ID
    src_blockno = partition.query(blockid); // Halo block number
  } else { 
    if (partition._haloMarks[blockIdx.x]) return; // Return early if halo block
    blockid = partition._activeKeys[blockIdx.x]; // Non-halo block ID
    src_blockno = blockIdx.x; // Non-halo block number
  }
  if (src_blockno < 0) return; // Return early if negative block number
//  else if (src_blockno > blockcnt) return; // Return early if excessive block number
  int ppb = g_buckets_on_particle_buffer
            ? next_pbuffer._ppbs[src_blockno] 
            : partition._ppbs[src_blockno]; // Particles in block
  if (ppb == 0) return; // Return early if no particles


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
    p2gbuffer[loc >> arenabits][x][y][z] = (PREC_G)0.0;
  }
  __syncthreads();
  for (int pidib = threadIdx.x; pidib < ppb; pidib += blockDim.x) {
    int source_blockno, source_pidib;
    ivec3 base_index;
    {
      int advect = (g_buckets_on_particle_buffer)
                    ? next_pbuffer._blockbuckets[src_blockno * g_particle_num_per_block + pidib]
                    : partition._blockbuckets[src_blockno * g_particle_num_per_block + pidib];
      dir_components(advect / g_particle_num_per_block, base_index);
      base_index += blockid;
      source_blockno = prev_partition.query(base_index);
      source_pidib = advect % g_particle_num_per_block; // & (g_particle_num_per_block - 1);
      source_blockno = (g_buckets_on_particle_buffer)
                        ? pbuffer._binsts[source_blockno] + source_pidib / g_bin_capacity
                        : prev_partition._binsts[source_blockno] + source_pidib / g_bin_capacity;
    }
    pvec3 pos;  //< Particle position at n
    pvec9 F;
    PREC sJBar;
    {
      auto source_particle_bin = pbuffer.ch(_0, source_blockno);
      pos[0] = source_particle_bin.val(_0, source_pidib % g_bin_capacity);  //< x
      pos[1] = source_particle_bin.val(_1, source_pidib % g_bin_capacity);  //< y
      pos[2] = source_particle_bin.val(_2, source_pidib % g_bin_capacity);  //< z
      F[0] = source_particle_bin.val(_3, source_pidib % g_bin_capacity);
      F[1] = source_particle_bin.val(_4, source_pidib % g_bin_capacity);
      F[2] = source_particle_bin.val(_5, source_pidib % g_bin_capacity);
      F[3] = source_particle_bin.val(_6, source_pidib % g_bin_capacity);
      F[4] = source_particle_bin.val(_7, source_pidib % g_bin_capacity);
      F[5] = source_particle_bin.val(_8, source_pidib % g_bin_capacity);
      F[6] = source_particle_bin.val(_9, source_pidib % g_bin_capacity);
      F[7] = source_particle_bin.val(_10, source_pidib % g_bin_capacity);
      F[8] = source_particle_bin.val(_11, source_pidib % g_bin_capacity);
      // Skip velocity
      sJBar = source_particle_bin.val(_12, source_pidib % g_bin_capacity); //< JBar tn
      // Skip ID
    }
    ivec3 local_base_index = (pos * g_dx_inv_d + 0.5).cast<int>() - 1;
    pvec3 local_pos = pos - local_base_index * g_dx_d;
    base_index = local_base_index;

    pvec3x3 dws;
#pragma unroll 3
    for (int dd = 0; dd < 3; ++dd) {
      PREC d = (local_pos[dd] - static_cast<PREC>(std::floor(local_pos[dd] * g_dx_inv_d + 0.5) - 1) * g_dx_d) * g_dx_inv_d;
      dws(dd, 0) = 0.5 * (1.5 - d) * (1.5 - d);
      d -= 1.0;
      dws(dd, 1) = 0.75 - d * d;
      d = 0.5 + d;
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
          pvec3 xixp = pvec3{(PREC)i, (PREC)j, (PREC)k} * g_dx_d - local_pos;
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
    pvec9 FInc;
#pragma unroll 9
    for (int d = 0; d < 9; ++d)
      FInc[d] = C[d] * dt * Dp_inv + ((d & 0x3) ? 0.0 : 1.0);
    PREC JInc = matrixDeterminant3d(FInc.data()); // J^n+1 / J^n
    PREC J  = matrixDeterminant3d(F.data());
    PREC voln = J * pbuffer.volume; // vol^n+1, Send to grid for Simple FBar 
#pragma unroll 3
    for (char i = 0; i < 3; i++)
#pragma unroll 3
      for (char j = 0; j < 3; j++)
#pragma unroll 3
        for (char k = 0; k < 3; k++) {
          pos = pvec3{(PREC)i, (PREC)j, (PREC)k} * g_dx - local_pos;
          PREC W = dws(0, i) * dws(1, j) * dws(2, k);
          PREC wv = voln * W; // Weighted volume
          atomicAdd(
              &p2gbuffer[0][local_base_index[0] + i][local_base_index[1] + j]
                        [local_base_index[2] + k],
              wv);
          atomicAdd(
              &p2gbuffer[1][local_base_index[0] + i][local_base_index[1] + j]
                        [local_base_index[2] + k],
              wv * ((sJBar * JInc) - JInc + 1.0));
              //wv * (sJBar + sJBar * sJInc - sJInc)); // using sJInc = (1-JInc)
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


template <typename Partition, typename Grid>
__global__ void g2p_FBar(double dt, double newDt, const ivec3 *__restrict__ blocks,
                      const ParticleBuffer<material_e::NeoHookean_ASFLIP_FBAR> pbuffer,
                      ParticleBuffer<material_e::NeoHookean_ASFLIP_FBAR> next_pbuffer,
                      const Partition prev_partition, Partition partition,
                      Grid grid, Grid next_grid) {
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
    blockid = blocks[blockIdx.x]; // Halo block ID
    src_blockno = partition.query(blockid); // Halo block number
  } else { 
    if (partition._haloMarks[blockIdx.x]) return; // Return early if halo block
    blockid = partition._activeKeys[blockIdx.x]; // Non-halo block ID
    src_blockno = blockIdx.x; // Non-halo block number
  }
  if (src_blockno < 0) return; // Return early if negative block number
//  else if (src_blockno > blockcnt) return; // Return early if excessive block number
  int ppb = g_buckets_on_particle_buffer
            ? next_pbuffer._ppbs[src_blockno] 
            : partition._ppbs[src_blockno]; // Particles in block
  if (ppb == 0) return; // Return early if no particles


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
    p2gbuffer[loc >> arenabits][x][y][z] = (PREC_G)0.0;
  }
  __syncthreads();
  for (int pidib = threadIdx.x; pidib < ppb; pidib += blockDim.x) {
    int source_blockno, source_pidib;
    ivec3 base_index;
    {
      int advect = (g_buckets_on_particle_buffer)
                    ? next_pbuffer._blockbuckets[src_blockno * g_particle_num_per_block + pidib]
                    : partition._blockbuckets[src_blockno * g_particle_num_per_block + pidib];
      dir_components(advect / g_particle_num_per_block, base_index);
      base_index += blockid;
      source_blockno = prev_partition.query(base_index);
      source_pidib = advect % g_particle_num_per_block; // & (g_particle_num_per_block - 1);
      source_blockno = (g_buckets_on_particle_buffer)
                        ? pbuffer._binsts[source_blockno] + source_pidib / g_bin_capacity
                        : prev_partition._binsts[source_blockno] + source_pidib / g_bin_capacity;
    }
    pvec3 pos;  //< Particle position at n
    //PREC sJ;   //< Particle volume ratio at n
    pvec9 F;
    PREC sJBar;
    //PREC vol;
    //PREC ID;
    {
      auto source_particle_bin = pbuffer.ch(_0, source_blockno);
      pos[0] = source_particle_bin.val(_0, source_pidib % g_bin_capacity);  //< x
      pos[1] = source_particle_bin.val(_1, source_pidib % g_bin_capacity);  //< y
      pos[2] = source_particle_bin.val(_2, source_pidib % g_bin_capacity);  //< z
      F[0] = source_particle_bin.val(_3, source_pidib % g_bin_capacity);
      F[1] = source_particle_bin.val(_4, source_pidib % g_bin_capacity);
      F[2] = source_particle_bin.val(_5, source_pidib % g_bin_capacity);
      F[3] = source_particle_bin.val(_6, source_pidib % g_bin_capacity);
      F[4] = source_particle_bin.val(_7, source_pidib % g_bin_capacity);
      F[5] = source_particle_bin.val(_8, source_pidib % g_bin_capacity);
      F[6] = source_particle_bin.val(_9, source_pidib % g_bin_capacity);
      F[7] = source_particle_bin.val(_10, source_pidib % g_bin_capacity);
      F[8] = source_particle_bin.val(_11, source_pidib % g_bin_capacity);
      // vel_p[0] = source_particle_bin.val(_12, source_pidib % g_bin_capacity); //< vx
      // vel_p[1] = source_particle_bin.val(_13, source_pidib % g_bin_capacity); //< vy
      // vel_p[2] = source_particle_bin.val(_14, source_pidib % g_bin_capacity); //< vz
      //vol  = source_particle_bin.val(_15, source_pidib % g_bin_capacity); //< Volume tn
      sJBar = source_particle_bin.val(_16, source_pidib % g_bin_capacity); //< JBar tn 
      // ! TODO: Fix to not have vol, have ID after sJBAR <<<<<<<<<<<<<<<<<<<<
    }
    ivec3 local_base_index = (pos * g_dx_inv_d + 0.5).cast<int>() - 1;
    pvec3 local_pos = pos - local_base_index * g_dx_d;
    base_index = local_base_index;

    pvec3x3 dws;
#pragma unroll 3
    for (int dd = 0; dd < 3; ++dd) {
      PREC d = (local_pos[dd] - static_cast<PREC>(std::floor(local_pos[dd] * g_dx_inv_d + 0.5) - 1) * g_dx_d) * g_dx_inv_d;
      dws(dd, 0) = 0.5 * (1.5 - d) * (1.5 - d);
      d -= 1.0;
      dws(dd, 1) = 0.75 - d * d;
      d = 0.5 + d;
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
    pvec9 FInc;
#pragma unroll 9
    for (int d = 0; d < 9; ++d)
      FInc[d] = C[d] * dt * Dp_inv + ((d & 0x3) ? 0.0 : 1.0);
    PREC JInc = matrixDeterminant3d(FInc.data()); // J^n+1 / J^n
    //JInc = JInc - 1.0;

    //pvec9 F_new;
    //matrixMatrixMultiplication3d(FInc.data(), F.data(), F_new.data());
    //PREC J_new  = matrixDeterminant3d(F_new.data());
    
    PREC J  = matrixDeterminant3d(F.data());
    //PREC sJ = (1.0 - J);
    //PREC JInc = ((C[0] + C[4] + C[8]) * dt * Dp_inv); // J^n+1 / J^n
    PREC voln = J * pbuffer.volume; // vol^n+1, Send to grid for Simple FBar 
    //PREC J = JInc * (1.0 - sJ); // J^n+1
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
              wv * ((sJBar * JInc) - JInc + 1.0));
              //wv * (sJBar + sJBar * JInc - JInc));
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



template <typename Partition, typename Grid>
__global__ void g2p_FBar(double dt, double newDt, const ivec3 *__restrict__ blocks,
                      const ParticleBuffer<material_e::Sand_FBAR> pbuffer,
                      ParticleBuffer<material_e::Sand_FBAR> next_pbuffer,
                      const Partition prev_partition, Partition partition,
                      Grid grid, Grid next_grid) {
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
    blockid = blocks[blockIdx.x]; // Halo block ID
    src_blockno = partition.query(blockid); // Halo block number
  } else { 
    if (partition._haloMarks[blockIdx.x]) return; // Return early if halo block
    blockid = partition._activeKeys[blockIdx.x]; // Non-halo block ID
    src_blockno = blockIdx.x; // Non-halo block number
  }
  if (src_blockno < 0) return; // Return early if negative block number
//  else if (src_blockno > blockcnt) return; // Return early if excessive block number
  int ppb = g_buckets_on_particle_buffer
            ? next_pbuffer._ppbs[src_blockno] 
            : partition._ppbs[src_blockno]; // Particles in block
  if (ppb == 0) return; // Return early if no particles


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
    p2gbuffer[loc >> arenabits][x][y][z] = (PREC_G)0.0;
  }
  __syncthreads();
  for (int pidib = threadIdx.x; pidib < ppb; pidib += blockDim.x) {
    int source_blockno, source_pidib;
    ivec3 base_index;
    {
      int advect = (g_buckets_on_particle_buffer)
                    ? next_pbuffer._blockbuckets[src_blockno * g_particle_num_per_block + pidib]
                    : partition._blockbuckets[src_blockno * g_particle_num_per_block + pidib];
      dir_components(advect / g_particle_num_per_block, base_index);
      base_index += blockid;
      source_blockno = prev_partition.query(base_index);
      source_pidib = advect % g_particle_num_per_block; // & (g_particle_num_per_block - 1);
      source_blockno = (g_buckets_on_particle_buffer)
                        ? pbuffer._binsts[source_blockno] + source_pidib / g_bin_capacity
                        : prev_partition._binsts[source_blockno] + source_pidib / g_bin_capacity;
    }
    pvec3 pos;  //< Particle position at n
    pvec9 F;
    PREC sJBar;
    {
      auto source_particle_bin = pbuffer.ch(_0, source_blockno);
      pos[0] = source_particle_bin.val(_0, source_pidib % g_bin_capacity);  //< x
      pos[1] = source_particle_bin.val(_1, source_pidib % g_bin_capacity);  //< y
      pos[2] = source_particle_bin.val(_2, source_pidib % g_bin_capacity);  //< z
      F[0] = source_particle_bin.val(_3, source_pidib % g_bin_capacity);
      F[1] = source_particle_bin.val(_4, source_pidib % g_bin_capacity);
      F[2] = source_particle_bin.val(_5, source_pidib % g_bin_capacity);
      F[3] = source_particle_bin.val(_6, source_pidib % g_bin_capacity);
      F[4] = source_particle_bin.val(_7, source_pidib % g_bin_capacity);
      F[5] = source_particle_bin.val(_8, source_pidib % g_bin_capacity);
      F[6] = source_particle_bin.val(_9, source_pidib % g_bin_capacity);
      F[7] = source_particle_bin.val(_10, source_pidib % g_bin_capacity);
      F[8] = source_particle_bin.val(_11, source_pidib % g_bin_capacity);
      sJBar = source_particle_bin.val(_13, source_pidib % g_bin_capacity); //< JBar tn
    }
    ivec3 local_base_index = (pos * g_dx_inv_d + 0.5).cast<int>() - 1;
    pvec3 local_pos = pos - local_base_index * g_dx_d;
    base_index = local_base_index;

    pvec3x3 dws;
#pragma unroll 3
    for (int dd = 0; dd < 3; ++dd) {
      PREC d = (local_pos[dd] - static_cast<PREC>(std::floor(local_pos[dd] * g_dx_inv_d + 0.5) - 1) * g_dx_d) * g_dx_inv_d;
      dws(dd, 0) = 0.5 * (1.5 - d) * (1.5 - d);
      d -= 1.0;
      dws(dd, 1) = 0.75 - d * d;
      d = 0.5 + d;
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
    pvec9 FInc;
#pragma unroll 9
    for (int d = 0; d < 9; ++d)
      FInc[d] = C[d] * dt * Dp_inv + ((d & 0x3) ? 0.0 : 1.0);
    PREC JInc = matrixDeterminant3d(FInc.data()); // J^n+1 / J^n
    PREC J  = matrixDeterminant3d(F.data());
    PREC voln = J * pbuffer.volume; // vol^n+1, Send to grid for Simple FBar 

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
              wv * ((sJBar * JInc) - JInc + 1.0));
              //wv * (sJBar + sJBar * JInc - JInc));
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

template <typename Partition, typename Grid>
__global__ void g2p_FBar(double dt, double newDt, const ivec3 *__restrict__ blocks,
                      const ParticleBuffer<material_e::Sand_ASFLIP_FBAR> pbuffer,
                      ParticleBuffer<material_e::Sand_ASFLIP_FBAR> next_pbuffer,
                      const Partition prev_partition, Partition partition,
                      Grid grid, Grid next_grid) {
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
    blockid = blocks[blockIdx.x]; // Halo block ID
    src_blockno = partition.query(blockid); // Halo block number
  } else { 
    if (partition._haloMarks[blockIdx.x]) return; // Return early if halo block
    blockid = partition._activeKeys[blockIdx.x]; // Non-halo block ID
    src_blockno = blockIdx.x; // Non-halo block number
  }
  if (src_blockno < 0) return; // Return early if negative block number
//  else if (src_blockno > blockcnt) return; // Return early if excessive block number
  int ppb = g_buckets_on_particle_buffer
            ? next_pbuffer._ppbs[src_blockno] 
            : partition._ppbs[src_blockno]; // Particles in block
  if (ppb == 0) return; // Return early if no particles


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
    p2gbuffer[loc >> arenabits][x][y][z] = (PREC_G)0.0;
  }
  __syncthreads();
  for (int pidib = threadIdx.x; pidib < ppb; pidib += blockDim.x) {
    int source_blockno, source_pidib;
    ivec3 base_index;
    {
      int advect = (g_buckets_on_particle_buffer)
                    ? next_pbuffer._blockbuckets[src_blockno * g_particle_num_per_block + pidib]
                    : partition._blockbuckets[src_blockno * g_particle_num_per_block + pidib];
      dir_components(advect / g_particle_num_per_block, base_index);
      base_index += blockid;
      source_blockno = prev_partition.query(base_index);
      source_pidib = advect % g_particle_num_per_block; // & (g_particle_num_per_block - 1);
      source_blockno = (g_buckets_on_particle_buffer)
                        ? pbuffer._binsts[source_blockno] + source_pidib / g_bin_capacity
                        : prev_partition._binsts[source_blockno] + source_pidib / g_bin_capacity;
    }
    pvec3 pos;  //< Particle position at n
    pvec9 F;
    PREC sJBar;
    {
      auto source_particle_bin = pbuffer.ch(_0, source_blockno);
      pos[0] = source_particle_bin.val(_0, source_pidib % g_bin_capacity);  //< x
      pos[1] = source_particle_bin.val(_1, source_pidib % g_bin_capacity);  //< y
      pos[2] = source_particle_bin.val(_2, source_pidib % g_bin_capacity);  //< z
      F[0] = source_particle_bin.val(_3, source_pidib % g_bin_capacity);
      F[1] = source_particle_bin.val(_4, source_pidib % g_bin_capacity);
      F[2] = source_particle_bin.val(_5, source_pidib % g_bin_capacity);
      F[3] = source_particle_bin.val(_6, source_pidib % g_bin_capacity);
      F[4] = source_particle_bin.val(_7, source_pidib % g_bin_capacity);
      F[5] = source_particle_bin.val(_8, source_pidib % g_bin_capacity);
      F[6] = source_particle_bin.val(_9, source_pidib % g_bin_capacity);
      F[7] = source_particle_bin.val(_10, source_pidib % g_bin_capacity);
      F[8] = source_particle_bin.val(_11, source_pidib % g_bin_capacity);
      sJBar = source_particle_bin.val(_16, source_pidib % g_bin_capacity); //< JBar tn
    }
    ivec3 local_base_index = (pos * g_dx_inv_d + 0.5).cast<int>() - 1;
    pvec3 local_pos = pos - local_base_index * g_dx_d;
    base_index = local_base_index;

    pvec3x3 dws;
#pragma unroll 3
    for (int dd = 0; dd < 3; ++dd) {
      PREC d = (local_pos[dd] - static_cast<PREC>(std::floor(local_pos[dd] * g_dx_inv_d + 0.5) - 1) * g_dx_d) * g_dx_inv_d;
      dws(dd, 0) = 0.5 * (1.5 - d) * (1.5 - d);
      d -= 1.0;
      dws(dd, 1) = 0.75 - d * d;
      d = 0.5 + d;
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
    pvec9 FInc;
#pragma unroll 9
    for (int d = 0; d < 9; ++d)
      FInc[d] = C[d] * dt * Dp_inv + ((d & 0x3) ? 0.0 : 1.0);
    PREC JInc = matrixDeterminant3d(FInc.data()); // J^n+1 / J^n
    PREC J  = matrixDeterminant3d(F.data());
    PREC voln = J * pbuffer.volume; // vol^n+1, Send to grid for Simple FBar 

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
              wv * ((sJBar * JInc) - JInc + 1.0));
              //wv * (sJBar + sJBar * JInc - JInc));
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


template <typename Partition, typename Grid>
__global__ void g2p_FBar(double dt, double newDt, const ivec3 *__restrict__ blocks,
                      const ParticleBuffer<material_e::NACC_FBAR> pbuffer,
                      ParticleBuffer<material_e::NACC_FBAR> next_pbuffer,
                      const Partition prev_partition, Partition partition,
                      Grid grid, Grid next_grid) {
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
    blockid = blocks[blockIdx.x]; // Halo block ID
    src_blockno = partition.query(blockid); // Halo block number
  } else { 
    if (partition._haloMarks[blockIdx.x]) return; // Return early if halo block
    blockid = partition._activeKeys[blockIdx.x]; // Non-halo block ID
    src_blockno = blockIdx.x; // Non-halo block number
  }
  if (src_blockno < 0) return; // Return early if negative block number
//  else if (src_blockno > blockcnt) return; // Return early if excessive block number
  int ppb = g_buckets_on_particle_buffer
            ? next_pbuffer._ppbs[src_blockno] 
            : partition._ppbs[src_blockno]; // Particles in block
  if (ppb == 0) return; // Return early if no particles


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
    p2gbuffer[loc >> arenabits][x][y][z] = (PREC_G)0.0;
  }
  __syncthreads();
  for (int pidib = threadIdx.x; pidib < ppb; pidib += blockDim.x) {
    int source_blockno, source_pidib;
    ivec3 base_index;
    {
      int advect = (g_buckets_on_particle_buffer)
                    ? next_pbuffer._blockbuckets[src_blockno * g_particle_num_per_block + pidib]
                    : partition._blockbuckets[src_blockno * g_particle_num_per_block + pidib];
      dir_components(advect / g_particle_num_per_block, base_index);
      base_index += blockid;
      source_blockno = prev_partition.query(base_index);
      source_pidib = advect % g_particle_num_per_block; // & (g_particle_num_per_block - 1);
      source_blockno = (g_buckets_on_particle_buffer)
                        ? pbuffer._binsts[source_blockno] + source_pidib / g_bin_capacity
                        : prev_partition._binsts[source_blockno] + source_pidib / g_bin_capacity;
    }
    pvec3 pos;  //< Particle position at n
    pvec9 F;
    PREC sJBar;
    {
      auto source_particle_bin = pbuffer.ch(_0, source_blockno);
      pos[0] = source_particle_bin.val(_0, source_pidib % g_bin_capacity);  //< x
      pos[1] = source_particle_bin.val(_1, source_pidib % g_bin_capacity);  //< y
      pos[2] = source_particle_bin.val(_2, source_pidib % g_bin_capacity);  //< z
      F[0] = source_particle_bin.val(_3, source_pidib % g_bin_capacity);
      F[1] = source_particle_bin.val(_4, source_pidib % g_bin_capacity);
      F[2] = source_particle_bin.val(_5, source_pidib % g_bin_capacity);
      F[3] = source_particle_bin.val(_6, source_pidib % g_bin_capacity);
      F[4] = source_particle_bin.val(_7, source_pidib % g_bin_capacity);
      F[5] = source_particle_bin.val(_8, source_pidib % g_bin_capacity);
      F[6] = source_particle_bin.val(_9, source_pidib % g_bin_capacity);
      F[7] = source_particle_bin.val(_10, source_pidib % g_bin_capacity);
      F[8] = source_particle_bin.val(_11, source_pidib % g_bin_capacity);
      sJBar = source_particle_bin.val(_13, source_pidib % g_bin_capacity); //< JBar tn
    }
    ivec3 local_base_index = (pos * g_dx_inv_d + 0.5).cast<int>() - 1;
    pvec3 local_pos = pos - local_base_index * g_dx_d;
    base_index = local_base_index;

    pvec3x3 dws;
#pragma unroll 3
    for (int dd = 0; dd < 3; ++dd) {
      PREC d = (local_pos[dd] - static_cast<PREC>(std::floor(local_pos[dd] * g_dx_inv_d + 0.5) - 1) * g_dx_d) * g_dx_inv_d;
      dws(dd, 0) = 0.5 * (1.5 - d) * (1.5 - d);
      d -= 1.0;
      dws(dd, 1) = 0.75 - d * d;
      d = 0.5 + d;
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
    pvec9 FInc;
#pragma unroll 9
    for (int d = 0; d < 9; ++d)
      FInc[d] = C[d] * dt * Dp_inv + ((d & 0x3) ? 0.0 : 1.0);
    PREC JInc = matrixDeterminant3d(FInc.data()); // J^n+1 / J^n
    PREC J  = matrixDeterminant3d(F.data());
    PREC voln = J * pbuffer.volume; // vol^n+1, Send to grid for Simple FBar 

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
              wv * ((sJBar * JInc) - JInc + 1.0));
              //wv * (sJBar + sJBar * JInc - JInc));
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

template <typename Partition, typename Grid>
__global__ void g2p_FBar(double dt, double newDt, const ivec3 *__restrict__ blocks,
                      const ParticleBuffer<material_e::NACC_ASFLIP_FBAR> pbuffer,
                      ParticleBuffer<material_e::NACC_ASFLIP_FBAR> next_pbuffer,
                      const Partition prev_partition, Partition partition,
                      Grid grid, Grid next_grid) {
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
    blockid = blocks[blockIdx.x]; // Halo block ID
    src_blockno = partition.query(blockid); // Halo block number
  } else { 
    if (partition._haloMarks[blockIdx.x]) return; // Return early if halo block
    blockid = partition._activeKeys[blockIdx.x]; // Non-halo block ID
    src_blockno = blockIdx.x; // Non-halo block number
  }
  if (src_blockno < 0) return; // Return early if negative block number
//  else if (src_blockno > blockcnt) return; // Return early if excessive block number
  int ppb = g_buckets_on_particle_buffer
            ? next_pbuffer._ppbs[src_blockno] 
            : partition._ppbs[src_blockno]; // Particles in block
  if (ppb == 0) return; // Return early if no particles


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
    p2gbuffer[loc >> arenabits][x][y][z] = (PREC_G)0.0;
  }
  __syncthreads();
  for (int pidib = threadIdx.x; pidib < ppb; pidib += blockDim.x) {
    int source_blockno, source_pidib;
    ivec3 base_index;
    {
      int advect = (g_buckets_on_particle_buffer)
                    ? next_pbuffer._blockbuckets[src_blockno * g_particle_num_per_block + pidib]
                    : partition._blockbuckets[src_blockno * g_particle_num_per_block + pidib];
      dir_components(advect / g_particle_num_per_block, base_index);
      base_index += blockid;
      source_blockno = prev_partition.query(base_index);
      source_pidib = advect % g_particle_num_per_block; // & (g_particle_num_per_block - 1);
      source_blockno = (g_buckets_on_particle_buffer)
                        ? pbuffer._binsts[source_blockno] + source_pidib / g_bin_capacity
                        : prev_partition._binsts[source_blockno] + source_pidib / g_bin_capacity;
    }
    pvec3 pos;  //< Particle position at n
    pvec9 F;
    PREC sJBar;
    {
      auto source_particle_bin = pbuffer.ch(_0, source_blockno);
      pos[0] = source_particle_bin.val(_0, source_pidib % g_bin_capacity);  //< x
      pos[1] = source_particle_bin.val(_1, source_pidib % g_bin_capacity);  //< y
      pos[2] = source_particle_bin.val(_2, source_pidib % g_bin_capacity);  //< z
      F[0] = source_particle_bin.val(_3, source_pidib % g_bin_capacity);
      F[1] = source_particle_bin.val(_4, source_pidib % g_bin_capacity);
      F[2] = source_particle_bin.val(_5, source_pidib % g_bin_capacity);
      F[3] = source_particle_bin.val(_6, source_pidib % g_bin_capacity);
      F[4] = source_particle_bin.val(_7, source_pidib % g_bin_capacity);
      F[5] = source_particle_bin.val(_8, source_pidib % g_bin_capacity);
      F[6] = source_particle_bin.val(_9, source_pidib % g_bin_capacity);
      F[7] = source_particle_bin.val(_10, source_pidib % g_bin_capacity);
      F[8] = source_particle_bin.val(_11, source_pidib % g_bin_capacity);
      sJBar = source_particle_bin.val(_16, source_pidib % g_bin_capacity); //< JBar tn
    }
    ivec3 local_base_index = (pos * g_dx_inv_d + 0.5).cast<int>() - 1;
    pvec3 local_pos = pos - local_base_index * g_dx_d;
    base_index = local_base_index;

    pvec3x3 dws;
#pragma unroll 3
    for (int dd = 0; dd < 3; ++dd) {
      PREC d = (local_pos[dd] - static_cast<PREC>(std::floor(local_pos[dd] * g_dx_inv_d + 0.5) - 1) * g_dx_d) * g_dx_inv_d;
      dws(dd, 0) = 0.5 * (1.5 - d) * (1.5 - d);
      d -= 1.0;
      dws(dd, 1) = 0.75 - d * d;
      d = 0.5 + d;
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
    pvec9 FInc;
#pragma unroll 9
    for (int d = 0; d < 9; ++d)
      FInc[d] = C[d] * dt * Dp_inv + ((d & 0x3) ? 0.0 : 1.0);
    PREC JInc = matrixDeterminant3d(FInc.data()); // J^n+1 / J^n
    PREC J  = matrixDeterminant3d(F.data());
    PREC voln = J * pbuffer.volume; // vol^n+1, Send to grid for Simple FBar 

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
              wv * ((sJBar * JInc) - JInc + 1.0));
              //wv * (sJBar + sJBar * JInc - JInc));
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


template <typename Partition, typename Grid>
__global__ void g2p_FBar(double dt, double newDt, const ivec3 *__restrict__ blocks,
                      const ParticleBuffer<material_e::CoupledUP> pbuffer,
                      ParticleBuffer<material_e::CoupledUP> next_pbuffer,
                      const Partition prev_partition, Partition partition,
                      Grid grid, Grid next_grid) {
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
    blockid = blocks[blockIdx.x]; // Halo block ID
    src_blockno = partition.query(blockid); // Halo block number
  } else { 
    if (partition._haloMarks[blockIdx.x]) return; // Return early if halo block
    blockid = partition._activeKeys[blockIdx.x]; // Non-halo block ID
    src_blockno = blockIdx.x; // Non-halo block number
  }
  if (src_blockno < 0) return; // Return early if negative block number
//  else if (src_blockno > blockcnt) return; // Return early if excessive block number
  int ppb = g_buckets_on_particle_buffer
            ? next_pbuffer._ppbs[src_blockno] 
            : partition._ppbs[src_blockno]; // Particles in block
  if (ppb == 0) return; // Return early if no particles


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
    p2gbuffer[loc >> arenabits][x][y][z] = static_cast<PREC_G>(0.0);
  }
  __syncthreads();
  for (int pidib = threadIdx.x; pidib < ppb; pidib += blockDim.x) {
    int source_blockno, source_pidib;
    ivec3 base_index;
    {
      int advect = (g_buckets_on_particle_buffer)
                    ? next_pbuffer._blockbuckets[src_blockno * g_particle_num_per_block + pidib]
                    : partition._blockbuckets[src_blockno * g_particle_num_per_block + pidib];
      dir_components(advect / g_particle_num_per_block, base_index);
      base_index += blockid;
      source_blockno = prev_partition.query(base_index);
      source_pidib = advect % g_particle_num_per_block; // & (g_particle_num_per_block - 1);
      source_blockno = (g_buckets_on_particle_buffer)
                        ? pbuffer._binsts[source_blockno] + source_pidib / g_bin_capacity
                        : prev_partition._binsts[source_blockno] + source_pidib / g_bin_capacity;
    }
    pvec3 pos;  //< Particle position at n
    pvec9 F;
    PREC sJBar;
    {
      auto source_particle_bin = pbuffer.ch(_0, source_blockno);
      pos[0] = source_particle_bin.val(_0, source_pidib % g_bin_capacity);  //< x
      pos[1] = source_particle_bin.val(_1, source_pidib % g_bin_capacity);  //< y
      pos[2] = source_particle_bin.val(_2, source_pidib % g_bin_capacity);  //< z
      F[0] = source_particle_bin.val(_3, source_pidib % g_bin_capacity);
      F[1] = source_particle_bin.val(_4, source_pidib % g_bin_capacity);
      F[2] = source_particle_bin.val(_5, source_pidib % g_bin_capacity);
      F[3] = source_particle_bin.val(_6, source_pidib % g_bin_capacity);
      F[4] = source_particle_bin.val(_7, source_pidib % g_bin_capacity);
      F[5] = source_particle_bin.val(_8, source_pidib % g_bin_capacity);
      F[6] = source_particle_bin.val(_9, source_pidib % g_bin_capacity);
      F[7] = source_particle_bin.val(_10, source_pidib % g_bin_capacity);
      F[8] = source_particle_bin.val(_11, source_pidib % g_bin_capacity);
      sJBar = source_particle_bin.val(_16, source_pidib % g_bin_capacity); //< JBar tn
    }
    ivec3 local_base_index = (pos * g_dx_inv_d + 0.5).cast<int>() - 1;
    pvec3 local_pos = pos - local_base_index * g_dx_d;
    base_index = local_base_index;

    pvec3x3 dws;
#pragma unroll 3
    for (int dd = 0; dd < 3; ++dd) {
      PREC d = (local_pos[dd] - static_cast<PREC>(std::floor(local_pos[dd] * g_dx_inv_d + 0.5) - 1) * g_dx_d) * g_dx_inv_d;
      dws(dd, 0) = 0.5 * (1.5 - d) * (1.5 - d);
      d -= 1.0;
      dws(dd, 1) = 0.75 - d * d;
      d = 0.5 + d;
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
    pvec9 FInc;
#pragma unroll 9
    for (int d = 0; d < 9; ++d)
      FInc[d] = C[d] * dt * Dp_inv + ((d & 0x3) ? 0.0 : 1.0);
    PREC JInc = matrixDeterminant3d(FInc.data()); // J^n+1 / J^n
    PREC J  = matrixDeterminant3d(F.data());
    PREC voln = J * pbuffer.volume; // vol^n+1, Send to grid for Simple FBar 

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
              wv * ((sJBar * JInc) - JInc + 1.0));
              //wv * (sJBar + sJBar * JInc - JInc));
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
__global__ void p2g_FBar(double dt, double newDt, const ivec3 *__restrict__ blocks,
                      const ParticleBuffer pbuffer,
                      ParticleBuffer next_pbuffer,
                      const Partition prev_partition, Partition partition,
                      const Grid grid, Grid next_grid) {}


template <typename Partition, typename Grid>
__global__ void p2g_FBar(double dt, double newDt, const ivec3 *__restrict__ blocks,
                      const ParticleBuffer<material_e::JFluid_FBAR> pbuffer,
                      ParticleBuffer<material_e::JFluid_FBAR> next_pbuffer,
                      const Partition prev_partition, Partition partition,
                      const Grid grid, Grid next_grid) { 
  static constexpr uint64_t numViPerBlock = g_blockvolume * 5;
  static constexpr uint64_t numViInArena = numViPerBlock << 3;
  static constexpr uint64_t shmem_offset = (g_blockvolume * 5) << 3;

  static constexpr uint64_t numMViPerBlock = g_blockvolume * 4;
  static constexpr uint64_t numMViInArena = numMViPerBlock << 3;

  static constexpr unsigned arenamask = (g_blocksize << 1) - 1;
  static constexpr unsigned arenabits = g_blockbits + 1;

  extern __shared__ char shmem[];
  using ViArena =
      PREC_G(*)[5][g_blocksize << 1][g_blocksize << 1][g_blocksize << 1];
  using ViArenaRef =
      PREC_G(&)[5][g_blocksize << 1][g_blocksize << 1][g_blocksize << 1];
  ViArenaRef __restrict__ g2pbuffer = *reinterpret_cast<ViArena>(shmem);
  using MViArena =
      PREC_G(*)[4][g_blocksize << 1][g_blocksize << 1][g_blocksize << 1];
  using MViArenaRef =
      PREC_G(&)[4][g_blocksize << 1][g_blocksize << 1][g_blocksize << 1];
  MViArenaRef __restrict__ p2gbuffer =
      *reinterpret_cast<MViArena>(shmem + shmem_offset * sizeof(PREC_G));

  ivec3 blockid;
  int src_blockno;
  if (blocks != nullptr) { 
    blockid = blocks[blockIdx.x]; // Halo block ID
    src_blockno = partition.query(blockid); // Halo block number
  } else { 
    if (partition._haloMarks[blockIdx.x]) return; // Return early if halo block
    blockid = partition._activeKeys[blockIdx.x]; // Non-halo block ID
    src_blockno = blockIdx.x; // Non-halo block number
  }
  if (src_blockno < 0) return; // Return early if negative block number
//  else if (src_blockno > blockcnt) return; // Return early if excessive block number
  int ppb = g_buckets_on_particle_buffer
            ? next_pbuffer._ppbs[src_blockno] 
            : partition._ppbs[src_blockno]; // Particles in block
  if (ppb == 0) return; // Return early if no particles


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
      val = grid_block.val_1d(_7, c);
    else if (channelid == 4) 
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
    p2gbuffer[loc >> arenabits][x][y][z] = static_cast<PREC_G>(0);
  }
  __syncthreads();
  // Start Grid-to-Particle, threads are particle
  for (int pidib = threadIdx.x; pidib < ppb; pidib += blockDim.x) {
    int source_blockno, source_pidib;
    ivec3 base_index;
    {
      int advect = (g_buckets_on_particle_buffer)
                    ? next_pbuffer._blockbuckets[src_blockno * g_particle_num_per_block + pidib]
                    : partition._blockbuckets[src_blockno * g_particle_num_per_block + pidib];
      dir_components(advect / g_particle_num_per_block, base_index);
      base_index += blockid;
      source_blockno = prev_partition.query(base_index);
      source_pidib = advect % g_particle_num_per_block; // & (g_particle_num_per_block - 1);
      source_blockno = (g_buckets_on_particle_buffer)
                        ? pbuffer._binsts[source_blockno] + source_pidib / g_bin_capacity
                        : prev_partition._binsts[source_blockno] + source_pidib / g_bin_capacity;
    }
    pvec3 pos;  //< Particle position at n
    PREC sJ; // sJBar, ID; //< (1-J), (1-JBar), Particle ID
    PREC ID;
    {
      auto source_particle_bin = pbuffer.ch(_0, source_blockno);
      pos[0] = source_particle_bin.val(_0, source_pidib % g_bin_capacity);  //< x
      pos[1] = source_particle_bin.val(_1, source_pidib % g_bin_capacity);  //< y
      pos[2] = source_particle_bin.val(_2, source_pidib % g_bin_capacity);  //< z
      sJ =  source_particle_bin.val(_3, source_pidib % g_bin_capacity);       //< Vo/V
      //sJBar = source_particle_bin.val(_4, source_pidib % g_bin_capacity); //< JBar tn
      ID  = source_particle_bin.val(_5, source_pidib % g_bin_capacity); //< Volume tn
    }

    ivec3 local_base_index = (pos * g_dx_inv_d + 0.5).cast<int>() - 1;
    pvec3 local_pos = pos - local_base_index * g_dx_d;
    base_index = local_base_index;

    pvec3x3 dws;
#pragma unroll 3
    for (int dd = 0; dd < 3; ++dd) {
      PREC d = (local_pos[dd] - static_cast<PREC>(std::floor(local_pos[dd] * g_dx_inv_d + 0.5) - 1) * g_dx_d) * g_dx_inv_d;
      dws(dd, 0) = 0.5 * (1.5 - d) * (1.5 - d);
      d -= 1.0;
      dws(dd, 1) = 0.75 - d * d;
      d = 0.5 + d;
      dws(dd, 2) = 0.5 * d * d;
      local_base_index[dd] = ((local_base_index[dd] - 1) & g_blockmask) + 1;
    }
    pvec3 vel;   //< Stressed, collided grid velocity
    pvec9 C;     //< APIC affine matrix, used a few times
    PREC sJBar_new; //< Simple FBar G2P JBar^n+1
    vel.set(0.0); 
    C.set(0.0);
    sJBar_new = 0.0;

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
          PREC sJBar_i = g2pbuffer[4][local_base_index[0] + i][local_base_index[1] + j]
                            [local_base_index[2] + k] / g2pbuffer[3][local_base_index[0] + i][local_base_index[1] + j]
                            [local_base_index[2] + k];
          // PREC sJBar_i = g2pbuffer[3][local_base_index[0] + i][local_base_index[1] + j]
          //                   [local_base_index[2] + k];
          vel   += vi * W;
          C[0] += W * vi[0] * xixp[0] * scale;
          C[1] += W * vi[1] * xixp[0] * scale;
          C[2] += W * vi[2] * xixp[0] * scale;
          C[3] += W * vi[0] * xixp[1] * scale;
          C[4] += W * vi[1] * xixp[1] * scale;
          C[5] += W * vi[2] * xixp[1] * scale;
          C[6] += W * vi[0] * xixp[2] * scale;
          C[7] += W * vi[1] * xixp[2] * scale;
          C[8] += W * vi[2] * xixp[2] * scale;
          sJBar_new += sJBar_i * W;
        }

    // pos += dt * vel; //< Advect paricle positions

/// B-SPLINE SHIFTED KERNEL
    // local_base_index = (pos * g_dx_inv + 0.5f).cast<int>() - 1;
    // local_pos = (pos ) - local_base_index * g_dx;
//     //base_index = local_base_index;
   
// #pragma unroll 3
//     for (int dd = 0; dd < 3; ++dd) {
//       PREC d =
//           (local_pos[dd] - ((int)(local_pos[dd] * g_dx_inv + 0.5) - 1) * g_dx) *
//           g_dx_inv;
//       dws(dd, 0) = 0.5 * (1.5 - d) * (1.5 - d);
//       d -= 1.0;
//       dws(dd, 1) = 0.75 - d * d;
//       d = 0.5 + d;
//       dws(dd, 2) = 0.5 * d * d;
//       local_base_index[dd] = ((local_base_index[dd] - 1) & g_blockmask) + 1;
//     }

// #pragma unroll 3
//     for (char i = 0; i < 3; i++)
// #pragma unroll 3
//       for (char j = 0; j < 3; j++)
// #pragma unroll 3
//         for (char k = 0; k < 3; k++) {
//           pvec3 xixp = pvec3{(PREC)i, (PREC)j, (PREC)k} * g_dx - local_pos;
//           PREC W = dws(0, i) * dws(1, j) * dws(2, k);
//           PREC sJBar_i = g2pbuffer[4][local_base_index[0] + i][local_base_index[1] + j]
//                             [local_base_index[2] + k] / g2pbuffer[3][local_base_index[0] + i][local_base_index[1] + j]
//                             [local_base_index[2] + k];
//           sJBar_new += sJBar_i * W;
        // }
/// END


/// LINEAR KERNEL
//     local_base_index = (pos * g_dx_inv).cast<int>();
//     local_pos = (pos) - local_base_index * g_dx;

//     //pvec2x2 dws;
// #pragma unroll 3
//     for (int dd = 0; dd < 3; ++dd) {
//       local_base_index[dd] = ((local_base_index[dd] - 1) & g_blockmask) + 1;
//     }

// #pragma unroll 2
//     for (char i = -1; i < 2; i+=2)
// #pragma unroll 2
//       for (char j = -1; j < 2; j+=2)
// #pragma unroll 2
//         for (char k = -1; k < 2; k+=2) {
//           PREC W = 0.125 * (1 + i * 2 * (local_pos[0] * g_dx_inv - 0.5)) * (1 + j * 2 * (local_pos[1] * g_dx_inv - 0.5)) * (1 + k * 2 * (local_pos[2] * g_dx_inv - 0.5));
//           PREC sJBar_i = g2pbuffer[4][local_base_index[0] + i][local_base_index[1] + j]
//                             [local_base_index[2] + k] / g2pbuffer[3][local_base_index[0] + i][local_base_index[1] + j]
//                             [local_base_index[2] + k];
//           sJBar_new += sJBar_i * W;
//         }
/// END

    pos += dt * vel; //< Advect paricle positions

    // FBar_n+1 = (JBar_n+1 / J_n+1)^(1/3) * F_n+1
    PREC FBAR_ratio = pbuffer.FBAR_ratio;
    PREC JInc = (1.0 + (C[0] + C[4] + C[8]) * dt * Dp_inv);
    sJ = (JInc * sJ) - JInc + 1.0;
    sJBar_new = (1.0 - FBAR_ratio) * sJ + (FBAR_ratio) * sJBar_new;

    pvec9 contrib;
    contrib.set(0.0);
    {
      PREC voln = (1.0 - sJ) * pbuffer.volume;
      PREC pressure = (pbuffer.bulk / pbuffer.gamma) * expm1(-pbuffer.gamma*log1p(-sJBar_new));
      // PREC pressure = (pbuffer.bulk / pbuffer.gamma) * (pow((1.0 - sJBar_new), -pbuffer.gamma) - 1.0);
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
      // Merged affine matrix and stress contribution for MLS-MPM P2G
      contrib = (C * pbuffer.mass - contrib * newDt) * Dp_inv;
      {
        auto particle_bin = g_buckets_on_particle_buffer 
            ? next_pbuffer.ch(_0, next_pbuffer._binsts[src_blockno] + pidib / g_bin_capacity) 
            : next_pbuffer.ch(_0, partition._binsts[src_blockno] + pidib / g_bin_capacity);
        particle_bin.val(_0, pidib % g_bin_capacity) = pos[0]; //< x
        particle_bin.val(_1, pidib % g_bin_capacity) = pos[1]; //< y
        particle_bin.val(_2, pidib % g_bin_capacity) = pos[2]; //< z
        particle_bin.val(_3, pidib % g_bin_capacity) = sJ;// + (sJBar_new - sJBar) / 2; //< V/Vo
        particle_bin.val(_4, pidib % g_bin_capacity) = sJBar_new; //< JBar [ ]
        particle_bin.val(_5, pidib % g_bin_capacity) = ID;   //< ID
      }
    }

    local_base_index = (pos * g_dx_inv_d + 0.5).cast<int>() - 1;
    {
      int direction_tag = dir_offset((base_index - 1) / g_blocksize -
                              (local_base_index - 1) / g_blocksize);
      if (g_buckets_on_particle_buffer)
        next_pbuffer.add_advection(partition, local_base_index - 1, direction_tag, pidib);
      else 
        partition.add_advection(local_base_index - 1, direction_tag, pidib);
    }

#pragma unroll 3
    for (char dd = 0; dd < 3; ++dd) {
      local_pos[dd] = pos[dd] - static_cast<PREC>(local_base_index[dd]) * g_dx_d;
      PREC d = (local_pos[dd] - static_cast<PREC>(std::floor(local_pos[dd] * g_dx_inv_d + 0.5) - 1) * g_dx_d) * g_dx_inv_d;
      dws(dd, 0) = 0.5 * (1.5 - d) * (1.5 - d);
      d -= 1.0;
      dws(dd, 1) = 0.75 - d * d;
      d = 0.5 + d;
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
    } 
  }
}

template <typename Partition, typename Grid>
__global__ void p2g_FBar(double dt, double newDt, const ivec3 *__restrict__ blocks,
                      const ParticleBuffer<material_e::JBarFluid> pbuffer,
                      ParticleBuffer<material_e::JBarFluid> next_pbuffer,
                      const Partition prev_partition, Partition partition,
                      const Grid grid, Grid next_grid) { 
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
    blockid = blocks[blockIdx.x]; // Halo block ID
    src_blockno = partition.query(blockid); // Halo block number
  } else { 
    if (partition._haloMarks[blockIdx.x]) return; // Return early if halo block
    src_blockno = blockIdx.x; // Non-halo block number
    blockid = partition._activeKeys[blockIdx.x]; // Non-halo block ID
  }

  int ppb = g_buckets_on_particle_buffer
          ? next_pbuffer._ppbs[src_blockno] 
          : partition._ppbs[src_blockno]; // Particles in block
  if (ppb == 0) return; // Return early if no particles

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
    g2pbuffer[channelid][cx + (local_block_id & 4 ? g_blocksize : 0)][cy + (local_block_id & 2 ? g_blocksize : 0)][cz + (local_block_id & 1 ? g_blocksize : 0)] = val;
  }
  __syncthreads();
  for (int base = threadIdx.x; base < numMViInArena; base += blockDim.x) {
    int loc = base;
    char z = loc & arenamask;
    char y = (loc >>= arenabits) & arenamask;
    char x = (loc >>= arenabits) & arenamask;
    p2gbuffer[loc >> arenabits][x][y][z] = static_cast<PREC_G>(0);
  }
  __syncthreads();
  // Start Grid-to-Particle, threads are particle
  for (int pidib = threadIdx.x; pidib < ppb; pidib += blockDim.x) {
    int source_blockno, source_pidib;
    ivec3 base_index;
    {
      int advect = (g_buckets_on_particle_buffer)
                    ? next_pbuffer._blockbuckets[src_blockno * g_particle_num_per_block + pidib]
                    : partition._blockbuckets[src_blockno * g_particle_num_per_block + pidib];
      dir_components(advect / g_particle_num_per_block, base_index);
      base_index += blockid;
      source_blockno = prev_partition.query(base_index);
      source_pidib = advect % g_particle_num_per_block; // & (g_particle_num_per_block - 1);
      source_blockno = (g_buckets_on_particle_buffer)
                        ? pbuffer._binsts[source_blockno] + source_pidib / g_bin_capacity
                        : prev_partition._binsts[source_blockno] + source_pidib / g_bin_capacity;
    }
    pvec3 pos;  //< Particle position at n
    pvec3 vel_p; //< Particle vel. at n
    PREC sJ; //< Particle volume ratio at n
    PREC ID;
    {
      auto source_particle_bin = pbuffer.ch(_0, source_blockno);
      pos[0] = source_particle_bin.val(_0, source_pidib % g_bin_capacity);  //< x
      pos[1] = source_particle_bin.val(_1, source_pidib % g_bin_capacity);  //< y
      pos[2] = source_particle_bin.val(_2, source_pidib % g_bin_capacity);  //< z
      sJ = source_particle_bin.val(_3, source_pidib % g_bin_capacity);       //< (1 - J)
      vel_p[0] = source_particle_bin.val(_4, source_pidib % g_bin_capacity); //< vx
      vel_p[1] = source_particle_bin.val(_5, source_pidib % g_bin_capacity); //< vy
      vel_p[2] = source_particle_bin.val(_6, source_pidib % g_bin_capacity); //< vz
      //sJBar = source_particle_bin.val(_7, source_pidib % g_bin_capacity);       //< (1 - JBar)
      ID  = source_particle_bin.val(_8, source_pidib % g_bin_capacity); //< Volume tn
    }
    ivec3 local_base_index = (pos * g_dx_inv_d + 0.5).cast<int>() - 1;
    pvec3 local_pos = pos - local_base_index * g_dx_d;
    base_index = local_base_index;

    pvec3x3 dws;
#pragma unroll 3
    for (int dd = 0; dd < 3; ++dd) {
      PREC d = (local_pos[dd] - static_cast<PREC>(std::floor(local_pos[dd] * g_dx_inv_d + 0.5) - 1) * g_dx_d) * g_dx_inv_d;
      dws(dd, 0) = 0.5 * (1.5 - d) * (1.5 - d);
      d -= 1.0;
      dws(dd, 1) = 0.75 - d * d;
      d = 0.5 + d;
      dws(dd, 2) = 0.5 * d * d;
      local_base_index[dd] = ((local_base_index[dd] - 1) & g_blockmask) + 1;
    }
    pvec3 vel;   //< Stressed, collided grid velocity
    pvec3 vel_FLIP; //< Unstressed, uncollided grid velocity
    pvec9 C;     //< APIC affine matrix, used a few times
    PREC sJBar_new; //< Simple FBar G2P JBar^n+1
    vel.set(0.0); 
    vel_FLIP.set(0.0);
    C.set(0.0);
    sJBar_new = 0.0;

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
          // Assuming we didn't do sJBar_i = (sJBar_i * voln_i) / (voln_i) in grid update kernel
          PREC sJBar_i = g2pbuffer[7][local_base_index[0] + i][local_base_index[1] + j]
                            [local_base_index[2] + k] / g2pbuffer[6][local_base_index[0] + i][local_base_index[1] + j]
                            [local_base_index[2] + k];
          vel   += vi * W;
          vel_FLIP += vi_n * W; 
          C[0] += W * vi[0] * xixp[0] * scale;
          C[1] += W * vi[1] * xixp[0] * scale;
          C[2] += W * vi[2] * xixp[0] * scale;
          C[3] += W * vi[0] * xixp[1] * scale;
          C[4] += W * vi[1] * xixp[1] * scale;
          C[5] += W * vi[2] * xixp[1] * scale;
          C[6] += W * vi[0] * xixp[2] * scale;
          C[7] += W * vi[1] * xixp[2] * scale;
          C[8] += W * vi[2] * xixp[2] * scale;
          sJBar_new += sJBar_i * W;
        }
    // PREC JInc = (1.0 + (C[0] + C[4] + C[8]) * dt * Dp_inv);
    // sJ = (JInc * sJ) - JInc + 1.0;
    {
      PREC sJInc = (C[0] + C[4] + C[8]) * dt * Dp_inv;
      sJ = sJ + (sJ * sJInc) - sJInc;
      // FBar_n+1 = (JBar_n+1 / J_n+1)^(1/3) * F_n+1
      PREC FBAR_ratio = pbuffer.FBAR_ratio;
      sJBar_new = (1.0 - FBAR_ratio) * sJ + (FBAR_ratio) * sJBar_new;

      if (pbuffer.alpha > 0) {
        // Position correction factor (ASFLIP)
        PREC beta = (sJBar_new < 0.0) ? pbuffer.beta_max : pbuffer.beta_min; 
        
        // ASFLIP particle advection update
        pos += dt * (vel + beta * pbuffer.alpha * (vel_p - vel_FLIP));
        vel += pbuffer.alpha * (vel_p - vel_FLIP);
      } else {
        // Regular MLS-MPM / PIC particle advection update
        pos += dt * vel;
      }
    }

    pvec9 contrib;
    contrib.set(0.0);
    {
      PREC voln = (1.0 - sJ) * pbuffer.volume;
      // PREC pressure = (pbuffer.bulk / pbuffer.gamma) * (pow((1.0 - sJBar_new), -pbuffer.gamma) - 1.0);
      PREC pressure = (pbuffer.bulk / pbuffer.gamma) * expm1(-pbuffer.gamma*log1p(-sJBar_new));
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
      // Merged affine matrix and stress contribution for MLS-MPM P2G
      contrib = (C * pbuffer.mass - contrib * newDt) * Dp_inv;
      {
        auto particle_bin = g_buckets_on_particle_buffer 
            ? next_pbuffer.ch(_0, next_pbuffer._binsts[src_blockno] + pidib / g_bin_capacity) 
            : next_pbuffer.ch(_0, partition._binsts[src_blockno] + pidib / g_bin_capacity);
        particle_bin.val(_0, pidib % g_bin_capacity) = pos[0]; //< pos_x
        particle_bin.val(_1, pidib % g_bin_capacity) = pos[1]; //< pos_y
        particle_bin.val(_2, pidib % g_bin_capacity) = pos[2]; //< pos_z
        particle_bin.val(_3, pidib % g_bin_capacity) = sJ; //< (1 - J)
        particle_bin.val(_4, pidib % g_bin_capacity) = vel[0]; //< vx
        particle_bin.val(_5, pidib % g_bin_capacity) = vel[1]; //< vy
        particle_bin.val(_6, pidib % g_bin_capacity) = vel[2]; //< vz
        particle_bin.val(_7, pidib % g_bin_capacity) = sJBar_new; //< (1 - JBar)
        particle_bin.val(_8, pidib % g_bin_capacity) = ID;   //< ID
      }
    }

    local_base_index = (pos * g_dx_inv_d + static_cast<PREC>(0.5)).cast<int>() - 1;
    {
      int direction_tag = dir_offset((base_index - 1) / g_blocksize -
                              (local_base_index - 1) / g_blocksize);
      if (g_buckets_on_particle_buffer)
        next_pbuffer.add_advection(partition, local_base_index - 1, direction_tag, pidib);
      else 
        partition.add_advection(local_base_index - 1, direction_tag, pidib);
    }

#pragma unroll 3
    for (char dd = 0; dd < 3; ++dd) {
      local_pos[dd] = pos[dd] - local_base_index[dd] * g_dx_d;
      PREC d = (local_pos[dd] - static_cast<PREC>(std::floor(local_pos[dd] * g_dx_inv_d + 0.5) - 1) * g_dx_d) * g_dx_inv_d;
      dws(dd, 0) = 0.5 * (1.5 - d) * (1.5 - d);
      d -= 1.0;
      dws(dd, 1) = 0.75 - d * d;
      d = 0.5 + d;
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
          pos = pvec3{(PREC)i, (PREC)j, (PREC)k} * g_dx_d - local_pos;
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
    int channelid = base % numMViPerBlock; 
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


// template <typename Partition, typename Grid>
// __global__ void p2g_FBar(double dt, double newDt, const ivec3 *__restrict__ blocks,
//                       const ParticleBuffer<material_e::JBarFluid> pbuffer,
//                       ParticleBuffer<material_e::JBarFluid> next_pbuffer,
//                       const Partition prev_partition, Partition partition,
//                       const Grid grid, Grid next_grid) { 
//   static constexpr uint64_t numViPerBlock = g_blockvolume * 8;
//   static constexpr uint64_t numViInArena = numViPerBlock << 3;
//   static constexpr uint64_t shmem_offset = (g_blockvolume * 8) << 3;

//   static constexpr uint64_t numMViPerBlock = g_blockvolume * 7;
//   static constexpr uint64_t numMViInArena = numMViPerBlock << 3;

//   static constexpr unsigned arenamask = (g_blocksize << 1) - 1;
//   static constexpr unsigned arenabits = g_blockbits + 1;

//   extern __shared__ char shmem[];
//   using ViArena =
//       PREC_G(*)[8][g_blocksize << 1][g_blocksize << 1][g_blocksize << 1];
//   using ViArenaRef =
//       PREC_G(&)[8][g_blocksize << 1][g_blocksize << 1][g_blocksize << 1];
//   ViArenaRef __restrict__ g2pbuffer = *reinterpret_cast<ViArena>(shmem);
//   using MViArena =
//       PREC_G(*)[7][g_blocksize << 1][g_blocksize << 1][g_blocksize << 1];
//   using MViArenaRef =
//       PREC_G(&)[7][g_blocksize << 1][g_blocksize << 1][g_blocksize << 1];
//   MViArenaRef __restrict__ p2gbuffer =
//       *reinterpret_cast<MViArena>(shmem + shmem_offset * sizeof(PREC_G));

//   ivec3 blockid;
//   int src_blockno;
//   if (blocks != nullptr) {
//     blockid = blocks[blockIdx.x];
//     src_blockno = partition.query(blockid);
//   } else {
//     if (partition._haloMarks[blockIdx.x])
//       return;
//     blockid = partition._activeKeys[blockIdx.x];
//     src_blockno = blockIdx.x;
//   }

//   for (int base = threadIdx.x; base < numViInArena; base += blockDim.x) {
//     char local_block_id = base / numViPerBlock;
//     auto blockno = partition.query(
//         ivec3{blockid[0] + ((local_block_id & 4) != 0 ? 1 : 0),
//               blockid[1] + ((local_block_id & 2) != 0 ? 1 : 0),
//               blockid[2] + ((local_block_id & 1) != 0 ? 1 : 0)});
//     auto grid_block = grid.ch(_0, blockno);
//     int channelid = base % numViPerBlock;
//     char c = channelid & 0x3f;
//     char cz = channelid & g_blockmask;
//     char cy = (channelid >>= g_blockbits) & g_blockmask;
//     char cx = (channelid >>= g_blockbits) & g_blockmask;
//     channelid >>= g_blockbits;

//     PREC_G val;
//     if (channelid == 0) 
//       val = grid_block.val_1d(_1, c);
//     else if (channelid == 1)
//       val = grid_block.val_1d(_2, c);
//     else if (channelid == 2)
//       val = grid_block.val_1d(_3, c);
//     else if (channelid == 3) 
//       val = grid_block.val_1d(_4, c);
//     else if (channelid == 4) 
//       val = grid_block.val_1d(_5, c);
//     else if (channelid == 5) 
//       val = grid_block.val_1d(_6, c);
//     else if (channelid == 6) 
//       val = grid_block.val_1d(_7, c);
//     else if (channelid == 7) 
//       val = grid_block.val_1d(_8, c);
//     g2pbuffer[channelid][cx + (local_block_id & 4 ? g_blocksize : 0)]
//              [cy + (local_block_id & 2 ? g_blocksize : 0)]
//              [cz + (local_block_id & 1 ? g_blocksize : 0)] = val;
//   }
//   __syncthreads();
//   for (int base = threadIdx.x; base < numMViInArena; base += blockDim.x) {
//     int loc = base;
//     char z = loc & arenamask;
//     char y = (loc >>= arenabits) & arenamask;
//     char x = (loc >>= arenabits) & arenamask;
//     p2gbuffer[loc >> arenabits][x][y][z] = (PREC_G)0.0;
//   }
//   __syncthreads();
//   // Start Grid-to-Particle, threads are particle
//   for (int pidib = threadIdx.x; pidib < ppb; pidib += blockDim.x) {
//     int source_blockno, source_pidib;
//     ivec3 base_index;
//     {
//       int advect =
//           partition
//               ._blockbuckets[src_blockno * g_particle_num_per_block + pidib];
//       dir_components(advect / g_particle_num_per_block, base_index);
//       base_index += blockid;
//       source_blockno = prev_partition.query(base_index);
//       source_pidib = advect % g_particle_num_per_block; // & (g_particle_num_per_block - 1);
//       source_blockno = prev_partition._binsts[source_blockno] +
//                        source_pidib / g_bin_capacity;
//     }
//     pvec3 pos;  //< Particle position at n
//     pvec3 vel_p; //< Particle vel. at n
//     PREC sJ; //< Particle volume ratio at n
//     PREC ID;
//     {
//       auto source_particle_bin = pbuffer.ch(_0, source_blockno);
//       pos[0] = source_particle_bin.val(_0, source_pidib % g_bin_capacity);  //< x
//       pos[1] = source_particle_bin.val(_1, source_pidib % g_bin_capacity);  //< y
//       pos[2] = source_particle_bin.val(_2, source_pidib % g_bin_capacity);  //< z
//       sJ =  source_particle_bin.val(_3, source_pidib % g_bin_capacity);       //< Vo/V
//       vel_p[0] = source_particle_bin.val(_4, source_pidib % g_bin_capacity); //< vx
//       vel_p[1] = source_particle_bin.val(_5, source_pidib % g_bin_capacity); //< vy
//       vel_p[2] = source_particle_bin.val(_6, source_pidib % g_bin_capacity); //< vz
//       ID  = source_particle_bin.val(_7, source_pidib % g_bin_capacity); //< Volume tn
//     }
//     ivec3 local_base_index = (pos * g_dx_inv + 0.5f).cast<int>() - 1;
//     pvec3 local_pos = pos - local_base_index * g_dx;
//     base_index = local_base_index;

//     pvec3x3 dws;
// #pragma unroll 3
//     for (int dd = 0; dd < 3; ++dd) {
//       PREC d =
//           (local_pos[dd] - ((int)(local_pos[dd] * g_dx_inv + 0.5f) - 1) * g_dx) *
//           g_dx_inv;
//       dws(dd, 0) = 0.5 * (1.5 - d) * (1.5 - d);
//       d -= 1.0;
//       dws(dd, 1) = 0.75 - d * d;
//       d = 0.5 + d;
//       dws(dd, 2) = 0.5 * d * d;
//       local_base_index[dd] = ((local_base_index[dd] - 1) & g_blockmask) + 1;
//     }
//     pvec3 vel;   //< Stressed, collided grid velocity
//     pvec3 vel_FLIP; //< Unstressed, uncollided grid velocity
//     pvec9 C;     //< APIC affine matrix, used a few times
//     PREC sJBar_new; //< Simple FBar G2P JBar^n+1
//     vel.set(0.0); 
//     vel_FLIP.set(0.0);
//     C.set(0.0);
//     sJBar_new = 0.0;

//     // Dp^n = Dp^n+1 = (1/4) * dx^2 * I (Quad.)
//     PREC Dp_inv; //< Inverse Intertia-Like Tensor (1/m^2)
//     PREC scale = pbuffer.length * pbuffer.length; //< Area scale (m^2)
//     Dp_inv = g_D_inv / scale; //< Scalar 4/(dx^2) for Quad. B-Spline

// #pragma unroll 3
//     for (char i = 0; i < 3; i++)
// #pragma unroll 3
//       for (char j = 0; j < 3; j++)
// #pragma unroll 3
//         for (char k = 0; k < 3; k++) {
//           pvec3 xixp = pvec3{(PREC)i, (PREC)j, (PREC)k} * g_dx - local_pos;
//           PREC W = dws(0, i) * dws(1, j) * dws(2, k);
//           pvec3 vi{g2pbuffer[0][local_base_index[0] + i][local_base_index[1] + j]
//                            [local_base_index[2] + k],
//                   g2pbuffer[1][local_base_index[0] + i][local_base_index[1] + j]
//                            [local_base_index[2] + k],
//                   g2pbuffer[2][local_base_index[0] + i][local_base_index[1] + j]
//                            [local_base_index[2] + k]};
//           pvec3 vi_n{g2pbuffer[3][local_base_index[0] + i][local_base_index[1] + j]
//                             [local_base_index[2] + k],
//                     g2pbuffer[4][local_base_index[0] + i][local_base_index[1] + j]
//                             [local_base_index[2] + k],
//                     g2pbuffer[5][local_base_index[0] + i][local_base_index[1] + j]
//                             [local_base_index[2] + k]};

//           PREC sJBar_i = g2pbuffer[7][local_base_index[0] + i][local_base_index[1] + j]
//                             [local_base_index[2] + k] / g2pbuffer[6][local_base_index[0] + i][local_base_index[1] + j]
//                             [local_base_index[2] + k];
//           vel   += vi * W;
//           vel_FLIP += vi_n * W; 
//           C[0] += W * vi[0] * xixp[0] * scale;
//           C[1] += W * vi[1] * xixp[0] * scale;
//           C[2] += W * vi[2] * xixp[0] * scale;
//           C[3] += W * vi[0] * xixp[1] * scale;
//           C[4] += W * vi[1] * xixp[1] * scale;
//           C[5] += W * vi[2] * xixp[1] * scale;
//           C[6] += W * vi[0] * xixp[2] * scale;
//           C[7] += W * vi[1] * xixp[2] * scale;
//           C[8] += W * vi[2] * xixp[2] * scale;
//           sJBar_new += sJBar_i * W;
//         }
//     PREC JInc = (1.0 + (C[0] + C[4] + C[8]) * dt * Dp_inv);
//     sJ = (JInc * sJ) - JInc + 1.0;

//     // FBar_n+1 = (JBar_n+1 / J_n+1)^(1/3) * F_n+1
//     PREC FBAR_ratio = pbuffer.FBAR_ratio;
//     sJBar_new = (1.0 - FBAR_ratio) * sJ + (FBAR_ratio) * sJBar_new;
//     // sJ = sJ + (JInc * sJ) - JInc;

//     PREC beta; //< Position correction factor (ASFLIP)
//     if (sJBar_new < 0.0) beta = pbuffer.beta_max; 
//     else beta = pbuffer.beta_min; 
    
//     pos += dt * (vel + beta * pbuffer.alpha * (vel_p - vel_FLIP));
//     vel += pbuffer.alpha * (vel_p - vel_FLIP);

//     pvec9 contrib;
//     contrib.set(0.0);
//     {
//       PREC voln = (1.0 - sJ) * pbuffer.volume;
//       PREC pressure = (pbuffer.bulk / pbuffer.gamma) * (pow((1.0 - sJBar_new), -pbuffer.gamma) - 1.0);
//       {
//         contrib[0] = ((C[0] + C[0]) * Dp_inv * pbuffer.visco - pressure) * voln ;
//         contrib[1] = (C[1] + C[3]) * Dp_inv *  pbuffer.visco * voln ;
//         contrib[2] = (C[2] + C[6]) * Dp_inv *  pbuffer.visco * voln ;
//         contrib[3] = (C[3] + C[1]) * Dp_inv *  pbuffer.visco * voln ;
//         contrib[4] = ((C[4] + C[4]) * Dp_inv *  pbuffer.visco - pressure) * voln ;
//         contrib[5] = (C[5] + C[7]) * Dp_inv *  pbuffer.visco * voln ;
//         contrib[6] = (C[6] + C[2]) * Dp_inv *  pbuffer.visco * voln ;
//         contrib[7] = (C[7] + C[5]) * Dp_inv *  pbuffer.visco * voln ;
//         contrib[8] = ((C[8] + C[8]) * Dp_inv *  pbuffer.visco - pressure) * voln ;
//       }
//       // Merged affine matrix and stress contribution for MLS-MPM P2G
//       contrib = (C * pbuffer.mass - contrib * newDt) * Dp_inv;
//       {
//         auto particle_bin = next_pbuffer.ch(_0, partition._binsts[src_blockno] +
//                                                     pidib / g_bin_capacity);
//         particle_bin.val(_0, pidib % g_bin_capacity) = pos[0]; //< x
//         particle_bin.val(_1, pidib % g_bin_capacity) = pos[1]; //< y
//         particle_bin.val(_2, pidib % g_bin_capacity) = pos[2]; //< z
//         particle_bin.val(_3, pidib % g_bin_capacity) = sJ; //sJBar_new;      //< V/Vo
//         particle_bin.val(_4, pidib % g_bin_capacity) = vel[0]; //< vx
//         particle_bin.val(_5, pidib % g_bin_capacity) = vel[1]; //< vy
//         particle_bin.val(_6, pidib % g_bin_capacity) = vel[2]; //< vz
//         particle_bin.val(_7, pidib % g_bin_capacity) = ID;   //< ID
//         particle_bin.val(_8, pidib % g_bin_capacity) = sJBar_new; //< JBar [ ]
//       }
//     }

//     local_base_index = (pos * g_dx_inv + 0.5f).cast<int>() - 1;
//     {
//       int direction_tag = dir_offset((base_index - 1) / g_blocksize -
//                               (local_base_index - 1) / g_blocksize);
//       partition.add_advection(local_base_index - 1, direction_tag, pidib);
//     }

// #pragma unroll 3
//     for (char dd = 0; dd < 3; ++dd) {
//       local_pos[dd] = pos[dd] - local_base_index[dd] * g_dx;
//       PREC d =
//           (local_pos[dd] - ((int)(local_pos[dd] * g_dx_inv + 0.5) - 1) * g_dx) *
//           g_dx_inv;
//       dws(dd, 0) = 0.5 * (1.5 - d) * (1.5 - d);
//       d -= 1.0;
//       dws(dd, 1) = 0.75 - d * d;
//       d = 0.5 + d;
//       dws(dd, 2) = 0.5 * d * d;

//       local_base_index[dd] = (((base_index[dd] - 1) & g_blockmask) + 1) +
//                              local_base_index[dd] - base_index[dd];
//     }
// #pragma unroll 3
//     for (char i = 0; i < 3; i++)
// #pragma unroll 3
//       for (char j = 0; j < 3; j++)
// #pragma unroll 3
//         for (char k = 0; k < 3; k++) {
//           pos = pvec3{(PREC)i, (PREC)j, (PREC)k} * g_dx - local_pos;
//           PREC W = dws(0, i) * dws(1, j) * dws(2, k);
//           PREC wm = pbuffer.mass * W;
//           atomicAdd(
//               &p2gbuffer[0][local_base_index[0] + i][local_base_index[1] + j]
//                         [local_base_index[2] + k],
//               wm);
//           atomicAdd(
//               &p2gbuffer[1][local_base_index[0] + i][local_base_index[1] + j]
//                         [local_base_index[2] + k],
//               wm * vel[0] + (contrib[0] * pos[0] + contrib[3] * pos[1] +
//                              contrib[6] * pos[2]) *
//                                 W);
//           atomicAdd(
//               &p2gbuffer[2][local_base_index[0] + i][local_base_index[1] + j]
//                         [local_base_index[2] + k],
//               wm * vel[1] + (contrib[1] * pos[0] + contrib[4] * pos[1] +
//                              contrib[7] * pos[2]) *
//                                 W);
//           atomicAdd(
//               &p2gbuffer[3][local_base_index[0] + i][local_base_index[1] + j]
//                         [local_base_index[2] + k],
//               wm * vel[2] + (contrib[2] * pos[0] + contrib[5] * pos[1] +
//                              contrib[8] * pos[2]) *
//                                 W);
//           // ASFLIP unstressed velocity
//           atomicAdd(
//               &p2gbuffer[4][local_base_index[0] + i][local_base_index[1] + j]
//                         [local_base_index[2] + k],
//               wm * (vel[0] + Dp_inv * (C[0] * pos[0] + C[3] * pos[1] +
//                              C[6] * pos[2])));
//           atomicAdd(
//               &p2gbuffer[5][local_base_index[0] + i][local_base_index[1] + j]
//                         [local_base_index[2] + k],
//               wm * (vel[1] + Dp_inv * (C[1] * pos[0] + C[4] * pos[1] +
//                              C[7] * pos[2])));
//           atomicAdd(
//               &p2gbuffer[6][local_base_index[0] + i][local_base_index[1] + j]
//                         [local_base_index[2] + k],
//               wm * (vel[2] + Dp_inv * (C[2] * pos[0] + C[5] * pos[1] +
//                              C[8] * pos[2])));
//         }
//   }
//   __syncthreads();
//   /// arena no, channel no, cell no
//   for (int base = threadIdx.x; base < numMViInArena; base += blockDim.x) {
//     char local_block_id = base / numMViPerBlock;
//     auto blockno = partition.query(
//         ivec3{blockid[0] + ((local_block_id & 4) != 0 ? 1 : 0),
//               blockid[1] + ((local_block_id & 2) != 0 ? 1 : 0),
//               blockid[2] + ((local_block_id & 1) != 0 ? 1 : 0)});
//     // auto grid_block = next_grid.template ch<0>(blockno);
//     int channelid = base % numMViPerBlock; //& (numMViPerBlock - 1);
//     char c = channelid % g_blockvolume;
//     char cz = channelid & g_blockmask;
//     char cy = (channelid >>= g_blockbits) & g_blockmask;
//     char cx = (channelid >>= g_blockbits) & g_blockmask;
//     channelid >>= g_blockbits;
//     PREC_G val =
//         p2gbuffer[channelid][cx + (local_block_id & 4 ? g_blocksize : 0)]
//                  [cy + (local_block_id & 2 ? g_blocksize : 0)]
//                  [cz + (local_block_id & 1 ? g_blocksize : 0)];
//     if (channelid == 0) {
//       atomicAdd(&next_grid.ch(_0, blockno).val_1d(_0, c), val);
//     } else if (channelid == 1) {
//       atomicAdd(&next_grid.ch(_0, blockno).val_1d(_1, c), val);
//     } else if (channelid == 2) {
//       atomicAdd(&next_grid.ch(_0, blockno).val_1d(_2, c), val);
//     } else if (channelid == 3) {
//       atomicAdd(&next_grid.ch(_0, blockno).val_1d(_3, c), val);
//     } else if (channelid == 4) {
//       atomicAdd(&next_grid.ch(_0, blockno).val_1d(_4, c), val);
//     } else if (channelid == 5) {
//       atomicAdd(&next_grid.ch(_0, blockno).val_1d(_5, c), val);
//     } else if (channelid == 6) {
//       atomicAdd(&next_grid.ch(_0, blockno).val_1d(_6, c), val);
//     }
//   }
// }


template <typename Partition, typename Grid>
__global__ void p2g_FBar(double dt, double newDt, const ivec3 *__restrict__ blocks,
                      const ParticleBuffer<material_e::FixedCorotated_FBAR> pbuffer,
                      ParticleBuffer<material_e::FixedCorotated_FBAR> next_pbuffer,
                      const Partition prev_partition, Partition partition,
                      const Grid grid, Grid next_grid) { 
  static constexpr uint64_t numViPerBlock = g_blockvolume * 5;
  static constexpr uint64_t numViInArena = numViPerBlock << 3;
  static constexpr uint64_t shmem_offset = (g_blockvolume * 8) << 3;

  static constexpr uint64_t numMViPerBlock = g_blockvolume * 4;
  static constexpr uint64_t numMViInArena = numMViPerBlock << 3;

  static constexpr unsigned arenamask = (g_blocksize << 1) - 1;
  static constexpr unsigned arenabits = g_blockbits + 1;

  extern __shared__ char shmem[];
  using ViArena =
      PREC_G(*)[5][g_blocksize << 1][g_blocksize << 1][g_blocksize << 1];
  using ViArenaRef =
      PREC_G(&)[5][g_blocksize << 1][g_blocksize << 1][g_blocksize << 1];
  ViArenaRef __restrict__ g2pbuffer = *reinterpret_cast<ViArena>(shmem);
  using MViArena =
      PREC_G(*)[4][g_blocksize << 1][g_blocksize << 1][g_blocksize << 1];
  using MViArenaRef =
      PREC_G(&)[4][g_blocksize << 1][g_blocksize << 1][g_blocksize << 1];
  MViArenaRef __restrict__ p2gbuffer =
      *reinterpret_cast<MViArena>(shmem + shmem_offset * sizeof(PREC_G));

  ivec3 blockid;
  int src_blockno;
  if (blocks != nullptr) { 
    blockid = blocks[blockIdx.x]; // Halo block ID
    src_blockno = partition.query(blockid); // Halo block number
  } else { 
    if (partition._haloMarks[blockIdx.x]) return; // Return early if halo block
    blockid = partition._activeKeys[blockIdx.x]; // Non-halo block ID
    src_blockno = blockIdx.x; // Non-halo block number
  }
  if (src_blockno < 0) return; // Return early if negative block number
//  else if (src_blockno > blockcnt) return; // Return early if excessive block number
  int ppb = g_buckets_on_particle_buffer
            ? next_pbuffer._ppbs[src_blockno] 
            : partition._ppbs[src_blockno]; // Particles in block
  if (ppb == 0) return; // Return early if no particles


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
    p2gbuffer[loc >> arenabits][x][y][z] = static_cast<PREC_G>(0);
  }
  __syncthreads();
  // Start Grid-to-Particle, each thread is a unique particle at a given moment
  for (int pidib = threadIdx.x; pidib < ppb; pidib += blockDim.x) {
    int source_blockno, source_pidib;
    ivec3 base_index;
    {
      int advect = (g_buckets_on_particle_buffer)
                    ? next_pbuffer._blockbuckets[src_blockno * g_particle_num_per_block + pidib]
                    : partition._blockbuckets[src_blockno * g_particle_num_per_block + pidib];
      dir_components(advect / g_particle_num_per_block, base_index);
      base_index += blockid;
      source_blockno = prev_partition.query(base_index);
      source_pidib = advect % g_particle_num_per_block; // & (g_particle_num_per_block - 1);
      source_blockno = (g_buckets_on_particle_buffer)
                        ? pbuffer._binsts[source_blockno] + source_pidib / g_bin_capacity
                        : prev_partition._binsts[source_blockno] + source_pidib / g_bin_capacity;
    }
    pvec3 pos;  //< Particle position at n
    pvec3 vel_p; //< Particle vel. at n
    //PREC sJ, sJBar; //< Particle volume ratio at n
    PREC ID;
    {
      auto source_particle_bin = pbuffer.ch(_0, source_blockno);
      pos[0] = source_particle_bin.val(_0, source_pidib % g_bin_capacity);  //< x
      pos[1] = source_particle_bin.val(_1, source_pidib % g_bin_capacity);  //< y
      pos[2] = source_particle_bin.val(_2, source_pidib % g_bin_capacity);  //< z
      //sJBar = source_particle_bin.val(_12, source_pidib % g_bin_capacity); //< JBar tn
      ID =  source_particle_bin.val(_13, source_pidib % g_bin_capacity);
    }
    ivec3 local_base_index = (pos * g_dx_inv_d + 0.5).cast<int>() - 1;
    pvec3 local_pos = pos - local_base_index * g_dx_d;
    base_index = local_base_index;

    pvec3x3 dws;
#pragma unroll 3
    for (int dd = 0; dd < 3; ++dd) {
      PREC d = (local_pos[dd] - static_cast<PREC>(std::floor(local_pos[dd] * g_dx_inv_d + 0.5) - 1) * g_dx_d) * g_dx_inv_d;
      dws(dd, 0) = 0.5 * (1.5 - d) * (1.5 - d);
      d -= 1.0;
      dws(dd, 1) = 0.75 - d * d;
      d = 0.5 + d;
      dws(dd, 2) = 0.5 * d * d;
      local_base_index[dd] = ((local_base_index[dd] - 1) & g_blockmask) + 1;
    }
    pvec3 vel;   //< Stressed, collided grid velocity
    pvec9 C;     //< APIC affine matrix, used a few times
    PREC sJBar_new; //< Simple FBar G2P JBar^n+1
    vel.set(0.0); 
    C.set(0.0);
    sJBar_new = 0.0;

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
          pvec3 xixp = pvec3{(PREC)i, (PREC)j, (PREC)k} * g_dx_d - local_pos;
          PREC W = dws(0, i) * dws(1, j) * dws(2, k);
          pvec3 vi{g2pbuffer[0][local_base_index[0] + i][local_base_index[1] + j]
                           [local_base_index[2] + k],
                  g2pbuffer[1][local_base_index[0] + i][local_base_index[1] + j]
                           [local_base_index[2] + k],
                  g2pbuffer[2][local_base_index[0] + i][local_base_index[1] + j]
                           [local_base_index[2] + k]};

          PREC sJBar_i = g2pbuffer[7][local_base_index[0] + i][local_base_index[1] + j]
                            [local_base_index[2] + k] / g2pbuffer[6][local_base_index[0] + i][local_base_index[1] + j]
                            [local_base_index[2] + k];
          vel   += vi * W;
          C[0] += W * vi[0] * xixp[0] * scale;
          C[1] += W * vi[1] * xixp[0] * scale;
          C[2] += W * vi[2] * xixp[0] * scale;
          C[3] += W * vi[0] * xixp[1] * scale;
          C[4] += W * vi[1] * xixp[1] * scale;
          C[5] += W * vi[2] * xixp[1] * scale;
          C[6] += W * vi[0] * xixp[2] * scale;
          C[7] += W * vi[1] * xixp[2] * scale;
          C[8] += W * vi[2] * xixp[2] * scale;
          sJBar_new += sJBar_i * W;
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
      

      PREC JInc = matrixDeterminant3d(dws.data());
      PREC J  = matrixDeterminant3d(F.data());
      // PREC voln = J * pbuffer.volume;

      // Particle advection
      pos += dt * vel; //< pos update
      
      //FBar_n+1 = (JBar_n+1 / J_n+1)^(1/3) * F_n+1
      PREC FBAR_ratio = pbuffer.FBAR_ratio;
      sJBar_new = (1.0 - FBAR_ratio) * (1.0 - J) + (FBAR_ratio) * sJBar_new;
      PREC J_Scale = cbrt((1.0 - sJBar_new) / J);

      // Write particle values to particleBuffer (global memory)
      {
        auto particle_bin = g_buckets_on_particle_buffer 
            ? next_pbuffer.ch(_0, next_pbuffer._binsts[src_blockno] + pidib / g_bin_capacity) 
            : next_pbuffer.ch(_0, partition._binsts[src_blockno] + pidib / g_bin_capacity);
        particle_bin.val(_0, pidib % g_bin_capacity) = pos[0];
        particle_bin.val(_1, pidib % g_bin_capacity) = pos[1];
        particle_bin.val(_2, pidib % g_bin_capacity) = pos[2];
        particle_bin.val(_3, pidib % g_bin_capacity) = F[0] ; // * JVol
        particle_bin.val(_4, pidib % g_bin_capacity) = F[1] ; // * JVol
        particle_bin.val(_5, pidib % g_bin_capacity) = F[2] ; // * JVol
        particle_bin.val(_6, pidib % g_bin_capacity) = F[3] ; // * JVol
        particle_bin.val(_7, pidib % g_bin_capacity) = F[4] ; // * JVol
        particle_bin.val(_8, pidib % g_bin_capacity) = F[5] ; // * JVol
        particle_bin.val(_9, pidib % g_bin_capacity) = F[6] ; // * JVol
        particle_bin.val(_10, pidib % g_bin_capacity) = F[7]; // * JVol
        particle_bin.val(_11, pidib % g_bin_capacity) = F[8]; // * JVol
        particle_bin.val(_12, pidib % g_bin_capacity) = sJBar_new;
        particle_bin.val(_13, pidib % g_bin_capacity) = ID;
      }
      // Update particle stress
      {
#pragma unroll 9
        for (int d = 0; d < 9; d++) F[d] = F[d]  * J_Scale;
        compute_stress_fixedcorotated(pbuffer.volume, pbuffer.mu, pbuffer.lambda,
                                    F, contrib);
      }
      // MLS-MPM fused affine momentum + internal force matrix
      contrib = (C * pbuffer.mass - contrib * newDt) * Dp_inv;
    }

    local_base_index = (pos * g_dx_inv_d + 0.5).cast<int>() - 1;
    {
      int direction_tag = dir_offset((base_index - 1) / g_blocksize -
                              (local_base_index - 1) / g_blocksize);
      if (g_buckets_on_particle_buffer)
        next_pbuffer.add_advection(partition, local_base_index - 1, direction_tag, pidib);
      else 
        partition.add_advection(local_base_index - 1, direction_tag, pidib);
    }

#pragma unroll 3
    for (char dd = 0; dd < 3; ++dd) {
      local_pos[dd] = pos[dd] - local_base_index[dd] * g_dx_d;
      PREC d =
          (local_pos[dd] - ((int)(local_pos[dd] * g_dx_inv_d + 0.5) - 1) * g_dx_d) *
          g_dx_inv_d;
      dws(dd, 0) = 0.5 * (1.5 - d) * (1.5 - d);
      d -= 1.0;
      dws(dd, 1) = 0.75 - d * d;
      d = 0.5 + d;
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
          pos = pvec3{(PREC)i, (PREC)j, (PREC)k} * g_dx_d - local_pos;
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
    }
  }
}

template <typename Partition, typename Grid>
__global__ void p2g_FBar(double dt, double newDt, const ivec3 *__restrict__ blocks,
                      const ParticleBuffer<material_e::FixedCorotated_ASFLIP_FBAR> pbuffer,
                      ParticleBuffer<material_e::FixedCorotated_ASFLIP_FBAR> next_pbuffer,
                      const Partition prev_partition, Partition partition,
                      const Grid grid, Grid next_grid) { 
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
    blockid = blocks[blockIdx.x]; // Halo block ID
    src_blockno = partition.query(blockid); // Halo block number
  } else { 
    if (partition._haloMarks[blockIdx.x]) return; // Return early if halo block
    blockid = partition._activeKeys[blockIdx.x]; // Non-halo block ID
    src_blockno = blockIdx.x; // Non-halo block number
  }
  if (src_blockno < 0) return; // Return early if negative block number
//  else if (src_blockno > blockcnt) return; // Return early if excessive block number
  int ppb = g_buckets_on_particle_buffer
            ? next_pbuffer._ppbs[src_blockno] 
            : partition._ppbs[src_blockno]; // Particles in block
  if (ppb == 0) return; // Return early if no particles


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
    p2gbuffer[loc >> arenabits][x][y][z] = static_cast<PREC_G>(0);
  }
  __syncthreads();
  // Start Grid-to-Particle, each thread is a unique particle at a given moment
  for (int pidib = threadIdx.x; pidib < ppb; pidib += blockDim.x) {
    int source_blockno, source_pidib;
    ivec3 base_index;
    {
      int advect = (g_buckets_on_particle_buffer)
                    ? next_pbuffer._blockbuckets[src_blockno * g_particle_num_per_block + pidib]
                    : partition._blockbuckets[src_blockno * g_particle_num_per_block + pidib];
      dir_components(advect / g_particle_num_per_block, base_index);
      base_index += blockid;
      source_blockno = prev_partition.query(base_index);
      source_pidib = advect % g_particle_num_per_block; // & (g_particle_num_per_block - 1);
      source_blockno = (g_buckets_on_particle_buffer)
                        ? pbuffer._binsts[source_blockno] + source_pidib / g_bin_capacity
                        : prev_partition._binsts[source_blockno] + source_pidib / g_bin_capacity;
    }
    pvec3 pos;  //< Particle position at n
    pvec3 vel_p; //< Particle vel. at n
    //PREC sJ, sJBar; //< Particle volume ratio at n
    PREC ID;
    {
      auto source_particle_bin = pbuffer.ch(_0, source_blockno);
      pos[0] = source_particle_bin.val(_0, source_pidib % g_bin_capacity);  //< x
      pos[1] = source_particle_bin.val(_1, source_pidib % g_bin_capacity);  //< y
      pos[2] = source_particle_bin.val(_2, source_pidib % g_bin_capacity);  //< z
      vel_p[0] = source_particle_bin.val(_12, source_pidib % g_bin_capacity); //< vx
      vel_p[1] = source_particle_bin.val(_13, source_pidib % g_bin_capacity); //< vy
      vel_p[2] = source_particle_bin.val(_14, source_pidib % g_bin_capacity); //< vz
      //sJBar = source_particle_bin.val(_15, source_pidib % g_bin_capacity); //< JBar tn
      ID =  source_particle_bin.val(_16, source_pidib % g_bin_capacity);
    }
    ivec3 local_base_index = (pos * g_dx_inv_d + 0.5).cast<int>() - 1;
    pvec3 local_pos = pos - local_base_index * g_dx_d;
    base_index = local_base_index;

    pvec3x3 dws;
#pragma unroll 3
    for (int dd = 0; dd < 3; ++dd) {
      PREC d = (local_pos[dd] - static_cast<PREC>(std::floor(local_pos[dd] * g_dx_inv_d + 0.5) - 1) * g_dx_d) * g_dx_inv_d;
      dws(dd, 0) = 0.5 * (1.5 - d) * (1.5 - d);
      d -= 1.0;
      dws(dd, 1) = 0.75 - d * d;
      d = 0.5 + d;
      dws(dd, 2) = 0.5 * d * d;
      local_base_index[dd] = ((local_base_index[dd] - 1) & g_blockmask) + 1;
    }
    pvec3 vel;   //< Stressed, collided grid velocity
    pvec3 vel_FLIP; //< Unstressed, uncollided grid velocity
    pvec9 C;     //< APIC affine matrix, used a few times
    PREC sJBar_new; //< Simple FBar G2P JBar^n+1
    vel.set(0.0); 
    vel_FLIP.set(0.0);
    C.set(0.0);
    sJBar_new = 0.0;

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
          pvec3 xixp = pvec3{(PREC)i, (PREC)j, (PREC)k} * g_dx_d - local_pos;
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

          PREC sJBar_i = g2pbuffer[7][local_base_index[0] + i][local_base_index[1] + j]
                            [local_base_index[2] + k] / g2pbuffer[6][local_base_index[0] + i][local_base_index[1] + j]
                            [local_base_index[2] + k];
          vel   += vi * W;
          vel_FLIP += vi_n * W; 
          C[0] += W * vi[0] * xixp[0] * scale;
          C[1] += W * vi[1] * xixp[0] * scale;
          C[2] += W * vi[2] * xixp[0] * scale;
          C[3] += W * vi[0] * xixp[1] * scale;
          C[4] += W * vi[1] * xixp[1] * scale;
          C[5] += W * vi[2] * xixp[1] * scale;
          C[6] += W * vi[0] * xixp[2] * scale;
          C[7] += W * vi[1] * xixp[2] * scale;
          C[8] += W * vi[2] * xixp[2] * scale;
          sJBar_new += sJBar_i * W;
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
      

      PREC JInc = matrixDeterminant3d(dws.data());
      PREC J  = matrixDeterminant3d(F.data());
      // PREC voln = J * pbuffer.volume;

      // Particle advection
      if (pbuffer.alpha > 0.0) {
        // ASFLIP position correction factor
        PREC beta = (sJBar_new < 0.0) ? pbuffer.beta_max : pbuffer.beta_min;
        pos += dt * (vel + beta * pbuffer.alpha * (vel_p - vel_FLIP)); //< pos update
        vel += pbuffer.alpha * (vel_p - vel_FLIP); //< vel update
      } else {
        pos += dt * vel; //< pos update
      }
      //FBar_n+1 = (JBar_n+1 / J_n+1)^(1/3) * F_n+1
      PREC FBAR_ratio = pbuffer.FBAR_ratio;
      sJBar_new = (1.0 - FBAR_ratio) * (1.0 - J) + (FBAR_ratio) * sJBar_new;
      PREC J_Scale = cbrt((1.0 - sJBar_new) / J);

      // Write particle values to particleBuffer (global memory)
      {
        auto particle_bin = g_buckets_on_particle_buffer 
            ? next_pbuffer.ch(_0, next_pbuffer._binsts[src_blockno] + pidib / g_bin_capacity) 
            : next_pbuffer.ch(_0, partition._binsts[src_blockno] + pidib / g_bin_capacity);
        particle_bin.val(_0, pidib % g_bin_capacity) = pos[0];
        particle_bin.val(_1, pidib % g_bin_capacity) = pos[1];
        particle_bin.val(_2, pidib % g_bin_capacity) = pos[2];
        particle_bin.val(_3, pidib % g_bin_capacity) = F[0] ; // * JVol
        particle_bin.val(_4, pidib % g_bin_capacity) = F[1] ; // * JVol
        particle_bin.val(_5, pidib % g_bin_capacity) = F[2] ; // * JVol
        particle_bin.val(_6, pidib % g_bin_capacity) = F[3] ; // * JVol
        particle_bin.val(_7, pidib % g_bin_capacity) = F[4] ; // * JVol
        particle_bin.val(_8, pidib % g_bin_capacity) = F[5] ; // * JVol
        particle_bin.val(_9, pidib % g_bin_capacity) = F[6] ; // * JVol
        particle_bin.val(_10, pidib % g_bin_capacity) = F[7]; // * JVol
        particle_bin.val(_11, pidib % g_bin_capacity) = F[8]; // * JVol
        particle_bin.val(_12, pidib % g_bin_capacity) = vel[0];
        particle_bin.val(_13, pidib % g_bin_capacity) = vel[1];
        particle_bin.val(_14, pidib % g_bin_capacity) = vel[2];
        particle_bin.val(_15, pidib % g_bin_capacity) = sJBar_new;
        particle_bin.val(_16, pidib % g_bin_capacity) = ID;
      }
      // Update particle stress
      {
#pragma unroll 9
        for (int d = 0; d < 9; d++) F[d] = F[d]  * J_Scale;
        compute_stress_fixedcorotated(pbuffer.volume, pbuffer.mu, pbuffer.lambda,
                                    F, contrib);
      }
      // MLS-MPM fused affine momentum + internal force matrix
      contrib = (C * pbuffer.mass - contrib * newDt) * Dp_inv;
    }

    local_base_index = (pos * g_dx_inv_d + 0.5).cast<int>() - 1;
    {
      int direction_tag = dir_offset((base_index - 1) / g_blocksize -
                              (local_base_index - 1) / g_blocksize);
      if (g_buckets_on_particle_buffer)
        next_pbuffer.add_advection(partition, local_base_index - 1, direction_tag, pidib);
      else 
        partition.add_advection(local_base_index - 1, direction_tag, pidib);
    }

#pragma unroll 3
    for (char dd = 0; dd < 3; ++dd) {
      local_pos[dd] = pos[dd] - local_base_index[dd] * g_dx_d;
      PREC d =
          (local_pos[dd] - ((int)(local_pos[dd] * g_dx_inv_d + 0.5) - 1) * g_dx_d) *
          g_dx_inv_d;
      dws(dd, 0) = 0.5 * (1.5 - d) * (1.5 - d);
      d -= 1.0;
      dws(dd, 1) = 0.75 - d * d;
      d = 0.5 + d;
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
          pos = pvec3{(PREC)i, (PREC)j, (PREC)k} * g_dx_d - local_pos;
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
    int channelid = base % numMViPerBlock;
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


template <typename Partition, typename Grid>
__global__ void p2g_FBar(double dt, double newDt, const ivec3 *__restrict__ blocks,
                      const ParticleBuffer<material_e::NeoHookean_FBAR> pbuffer,
                      ParticleBuffer<material_e::NeoHookean_FBAR> next_pbuffer,
                      const Partition prev_partition, Partition partition,
                      const Grid grid, Grid next_grid) { 
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
    blockid = blocks[blockIdx.x]; // Halo block ID
    src_blockno = partition.query(blockid); // Halo block number
  } else { 
    if (partition._haloMarks[blockIdx.x]) return; // Return early if halo block
    blockid = partition._activeKeys[blockIdx.x]; // Non-halo block ID
    src_blockno = blockIdx.x; // Non-halo block number
  }
  if (src_blockno < 0) return; // Return early if negative block number
//  else if (src_blockno > blockcnt) return; // Return early if excessive block number
  int ppb = g_buckets_on_particle_buffer
            ? next_pbuffer._ppbs[src_blockno] 
            : partition._ppbs[src_blockno]; // Particles in block
  if (ppb == 0) return; // Return early if no particles


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
    p2gbuffer[loc >> arenabits][x][y][z] = static_cast<PREC_G>(0);
  }
  __syncthreads();
  // Start Grid-to-Particle, each thread is a unique particle at a given moment
  for (int pidib = threadIdx.x; pidib < ppb; pidib += blockDim.x) {
    int source_blockno, source_pidib;
    ivec3 base_index;
    {
      int advect = (g_buckets_on_particle_buffer)
                    ? next_pbuffer._blockbuckets[src_blockno * g_particle_num_per_block + pidib]
                    : partition._blockbuckets[src_blockno * g_particle_num_per_block + pidib];
      dir_components(advect / g_particle_num_per_block, base_index);
      base_index += blockid;
      source_blockno = prev_partition.query(base_index);
      source_pidib = advect % g_particle_num_per_block; // & (g_particle_num_per_block - 1);
      source_blockno = (g_buckets_on_particle_buffer)
                        ? pbuffer._binsts[source_blockno] + source_pidib / g_bin_capacity
                        : prev_partition._binsts[source_blockno] + source_pidib / g_bin_capacity;
    }
    pvec3 pos;  //< Particle position at n
    pvec3 vel_p; //< Particle vel. at n
    //PREC sJ, sJBar; //< Particle volume ratio at n
    PREC ID;
    {
      auto source_particle_bin = pbuffer.ch(_0, source_blockno);
      pos[0] = source_particle_bin.val(_0, source_pidib % g_bin_capacity);  //< x
      pos[1] = source_particle_bin.val(_1, source_pidib % g_bin_capacity);  //< y
      pos[2] = source_particle_bin.val(_2, source_pidib % g_bin_capacity);  //< z
      //sJBar = source_particle_bin.val(_12, source_pidib % g_bin_capacity); //< JBar tn
      ID =  source_particle_bin.val(_13, source_pidib % g_bin_capacity);
    }
    ivec3 local_base_index = (pos * g_dx_inv_d + 0.5).cast<int>() - 1;
    pvec3 local_pos = pos - local_base_index * g_dx_d;
    base_index = local_base_index;

    pvec3x3 dws;
#pragma unroll 3
    for (int dd = 0; dd < 3; ++dd) {
      PREC d = (local_pos[dd] - static_cast<PREC>(std::floor(local_pos[dd] * g_dx_inv_d + 0.5) - 1) * g_dx_d) * g_dx_inv_d;
      dws(dd, 0) = 0.5 * (1.5 - d) * (1.5 - d);
      d -= 1.0;
      dws(dd, 1) = 0.75 - d * d;
      d = 0.5 + d;
      dws(dd, 2) = 0.5 * d * d;
      local_base_index[dd] = ((local_base_index[dd] - 1) & g_blockmask) + 1;
    }
    pvec3 vel;   //< Stressed, collided grid velocity
    pvec3 vel_FLIP; //< Unstressed, uncollided grid velocity
    pvec9 C;     //< APIC affine matrix, used a few times
    PREC sJBar_new; //< Simple FBar G2P JBar^n+1
    vel.set(0.0); 
    vel_FLIP.set(0.0);
    C.set(0.0);
    sJBar_new = 0.0;

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
          pvec3 xixp = pvec3{(PREC)i, (PREC)j, (PREC)k} * g_dx_d - local_pos;
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

          PREC sJBar_i = g2pbuffer[7][local_base_index[0] + i][local_base_index[1] + j]
                            [local_base_index[2] + k] / g2pbuffer[6][local_base_index[0] + i][local_base_index[1] + j]
                            [local_base_index[2] + k];
          vel   += vi * W;
          C[0] += W * vi[0] * xixp[0] * scale;
          C[1] += W * vi[1] * xixp[0] * scale;
          C[2] += W * vi[2] * xixp[0] * scale;
          C[3] += W * vi[0] * xixp[1] * scale;
          C[4] += W * vi[1] * xixp[1] * scale;
          C[5] += W * vi[2] * xixp[1] * scale;
          C[6] += W * vi[0] * xixp[2] * scale;
          C[7] += W * vi[1] * xixp[2] * scale;
          C[8] += W * vi[2] * xixp[2] * scale;
          sJBar_new += sJBar_i * W;
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
      

      PREC JInc = matrixDeterminant3d(dws.data());
      PREC J  = matrixDeterminant3d(F.data());
      // PREC voln = J * pbuffer.volume;

      // Particle advection
      pos += dt * vel; //< pos update
      
      //FBar_n+1 = (JBar_n+1 / J_n+1)^(1/3) * F_n+1
      PREC FBAR_ratio = pbuffer.FBAR_ratio;
      sJBar_new = (1.0 - FBAR_ratio) * (1.0 - J) + (FBAR_ratio) * sJBar_new;
      PREC J_Scale = cbrt((1.0 - sJBar_new) / J);

      // Write particle values to particleBuffer (global memory)
      {
        auto particle_bin = g_buckets_on_particle_buffer 
            ? next_pbuffer.ch(_0, next_pbuffer._binsts[src_blockno] + pidib / g_bin_capacity) 
            : next_pbuffer.ch(_0, partition._binsts[src_blockno] + pidib / g_bin_capacity);
        particle_bin.val(_0, pidib % g_bin_capacity) = pos[0];
        particle_bin.val(_1, pidib % g_bin_capacity) = pos[1];
        particle_bin.val(_2, pidib % g_bin_capacity) = pos[2];
        particle_bin.val(_3, pidib % g_bin_capacity) = F[0] ; // * JVol
        particle_bin.val(_4, pidib % g_bin_capacity) = F[1] ; // * JVol
        particle_bin.val(_5, pidib % g_bin_capacity) = F[2] ; // * JVol
        particle_bin.val(_6, pidib % g_bin_capacity) = F[3] ; // * JVol
        particle_bin.val(_7, pidib % g_bin_capacity) = F[4] ; // * JVol
        particle_bin.val(_8, pidib % g_bin_capacity) = F[5] ; // * JVol
        particle_bin.val(_9, pidib % g_bin_capacity) = F[6] ; // * JVol
        particle_bin.val(_10, pidib % g_bin_capacity) = F[7]; // * JVol
        particle_bin.val(_11, pidib % g_bin_capacity) = F[8]; // * JVol
        particle_bin.val(_12, pidib % g_bin_capacity) = sJBar_new;
        particle_bin.val(_13, pidib % g_bin_capacity) = ID;
      }
      // Update particle stress
      {
#pragma unroll 9
        for (int d = 0; d < 9; d++) F[d] = F[d]  * J_Scale;
        compute_stress_neohookean(pbuffer.volume, pbuffer.mu, pbuffer.lambda,
                                    F, contrib);
      }
      // MLS-MPM fused affine momentum + internal force matrix
      contrib = (C * pbuffer.mass - contrib * newDt) * Dp_inv;
    }

    local_base_index = (pos * g_dx_inv_d + 0.5).cast<int>() - 1;
    {
      int direction_tag = dir_offset((base_index - 1) / g_blocksize -
                              (local_base_index - 1) / g_blocksize);
      if (g_buckets_on_particle_buffer)
        next_pbuffer.add_advection(partition, local_base_index - 1, direction_tag, pidib);
      else 
        partition.add_advection(local_base_index - 1, direction_tag, pidib);
    }

#pragma unroll 3
    for (char dd = 0; dd < 3; ++dd) {
      local_pos[dd] = pos[dd] - local_base_index[dd] * g_dx_d;
      PREC d =
          (local_pos[dd] - ((int)(local_pos[dd] * g_dx_inv_d + 0.5) - 1) * g_dx_d) *
          g_dx_inv_d;
      dws(dd, 0) = 0.5 * (1.5 - d) * (1.5 - d);
      d -= 1.0;
      dws(dd, 1) = 0.75 - d * d;
      d = 0.5 + d;
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
          pos = pvec3{(PREC)i, (PREC)j, (PREC)k} * g_dx_d - local_pos;
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
    int channelid = base % numMViPerBlock;
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

template <typename Partition, typename Grid>
__global__ void p2g_FBar(double dt, double newDt, const ivec3 *__restrict__ blocks,
                      const ParticleBuffer<material_e::NeoHookean_ASFLIP_FBAR> pbuffer,
                      ParticleBuffer<material_e::NeoHookean_ASFLIP_FBAR> next_pbuffer,
                      const Partition prev_partition, Partition partition,
                      const Grid grid, Grid next_grid) { 
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
    blockid = blocks[blockIdx.x]; // Halo block ID
    src_blockno = partition.query(blockid); // Halo block number
  } else { 
    if (partition._haloMarks[blockIdx.x]) return; // Return early if halo block
    blockid = partition._activeKeys[blockIdx.x]; // Non-halo block ID
    src_blockno = blockIdx.x; // Non-halo block number
  }
  if (src_blockno < 0) return; // Return early if negative block number
//  else if (src_blockno > blockcnt) return; // Return early if excessive block number
  int ppb = g_buckets_on_particle_buffer
            ? next_pbuffer._ppbs[src_blockno] 
            : partition._ppbs[src_blockno]; // Particles in block
  if (ppb == 0) return; // Return early if no particles


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
    p2gbuffer[loc >> arenabits][x][y][z] = static_cast<PREC_G>(0);
  }
  __syncthreads();
  // Start Grid-to-Particle, threads are particle
  for (int pidib = threadIdx.x; pidib < ppb; pidib += blockDim.x) {
    int source_blockno, source_pidib;
    ivec3 base_index;
    {
      int advect = (g_buckets_on_particle_buffer)
                    ? next_pbuffer._blockbuckets[src_blockno * g_particle_num_per_block + pidib]
                    : partition._blockbuckets[src_blockno * g_particle_num_per_block + pidib];
      dir_components(advect / g_particle_num_per_block, base_index);
      base_index += blockid;
      source_blockno = prev_partition.query(base_index);
      source_pidib = advect % g_particle_num_per_block; // & (g_particle_num_per_block - 1);
      source_blockno = (g_buckets_on_particle_buffer)
                        ? pbuffer._binsts[source_blockno] + source_pidib / g_bin_capacity
                        : prev_partition._binsts[source_blockno] + source_pidib / g_bin_capacity;
    }
    pvec3 pos;  //< Particle position at n
    pvec3 vel_p; //< Particle vel. at n
    //PREC sJBar; //< Particle volume ratio at n
    PREC ID;
    {
      auto source_particle_bin = pbuffer.ch(_0, source_blockno);
      pos[0] = source_particle_bin.val(_0, source_pidib % g_bin_capacity);  //< x
      pos[1] = source_particle_bin.val(_1, source_pidib % g_bin_capacity);  //< y
      pos[2] = source_particle_bin.val(_2, source_pidib % g_bin_capacity);  //< z
      vel_p[0] = source_particle_bin.val(_12, source_pidib % g_bin_capacity); //< vx
      vel_p[1] = source_particle_bin.val(_13, source_pidib % g_bin_capacity); //< vy
      vel_p[2] = source_particle_bin.val(_14, source_pidib % g_bin_capacity); //< vz
      //sJBar = source_particle_bin.val(_15, source_pidib % g_bin_capacity); //< 1 - JBar tn
      ID =  source_particle_bin.val(_16, source_pidib % g_bin_capacity); //< ID

    }
    ivec3 local_base_index = (pos * g_dx_inv_d + 0.5).cast<int>() - 1;
    pvec3 local_pos = pos - local_base_index * g_dx_d;
    base_index = local_base_index;

    pvec3x3 dws;
#pragma unroll 3
    for (int dd = 0; dd < 3; ++dd) {
      PREC d = (local_pos[dd] - static_cast<PREC>(std::floor(local_pos[dd] * g_dx_inv_d + 0.5) - 1) * g_dx_d) * g_dx_inv_d;
      dws(dd, 0) = 0.5 * (1.5 - d) * (1.5 - d);
      d -= 1.0;
      dws(dd, 1) = 0.75 - d * d;
      d = 0.5 + d;
      dws(dd, 2) = 0.5 * d * d;
      local_base_index[dd] = ((local_base_index[dd] - 1) & g_blockmask) + 1;
    }
    pvec3 vel;   //< Stressed, collided grid velocity
    pvec3 vel_FLIP; //< Unstressed, uncollided grid velocity
    pvec9 C;     //< APIC affine matrix, used a few times
    PREC sJBar_new; //< Simple FBar G2P JBar^n+1
    vel.set(0.0); 
    vel_FLIP.set(0.0);
    C.set(0.0);
    sJBar_new = 0.0;

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

          PREC sJBar_i = g2pbuffer[7][local_base_index[0] + i][local_base_index[1] + j]
                            [local_base_index[2] + k] / g2pbuffer[6][local_base_index[0] + i][local_base_index[1] + j]
                            [local_base_index[2] + k];
          vel   += vi * W;
          vel_FLIP += vi_n * W; 
          C[0] += W * vi[0] * xixp[0] * scale;
          C[1] += W * vi[1] * xixp[0] * scale;
          C[2] += W * vi[2] * xixp[0] * scale;
          C[3] += W * vi[0] * xixp[1] * scale;
          C[4] += W * vi[1] * xixp[1] * scale;
          C[5] += W * vi[2] * xixp[1] * scale;
          C[6] += W * vi[0] * xixp[2] * scale;
          C[7] += W * vi[1] * xixp[2] * scale;
          C[8] += W * vi[2] * xixp[2] * scale;
          sJBar_new += sJBar_i * W;
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
      

      PREC JInc = matrixDeterminant3d(dws.data());
      PREC J  = matrixDeterminant3d(F.data());
      // PREC voln = J * pbuffer.volume;

      PREC J_Scale;
      {
        PREC FBAR_ratio = pbuffer.FBAR_ratio;
        // Mix grid averaged (1-JBar) with regular particle (1-J) for stress update (1-JBar)     
        sJBar_new = (1.0 - FBAR_ratio) * (1.0 - J) + (FBAR_ratio) * (sJBar_new);

        //FBar_n+1 = (JBar_n+1 / J_n+1)^(1/3) * F_n+1
        J_Scale = cbrt((1.0 - sJBar_new) / J);

        // Particle advection
        if (pbuffer.alpha > 0) {
          // Position correction factor (ASFLIP)
          PREC beta = (sJBar_new < 0) ? pbuffer.beta_max : pbuffer.beta_min; // beta min
          pos += dt * (vel + beta * pbuffer.alpha * (vel_p - vel_FLIP)); //< pos update
          vel += pbuffer.alpha * (vel_p - vel_FLIP); //< vel update
        } else {
          pos += dt * vel; //< pos update
        }
      }
      // Save update particle values to particleBuffer global memory
      {
        auto particle_bin = g_buckets_on_particle_buffer 
            ? next_pbuffer.ch(_0, next_pbuffer._binsts[src_blockno] + pidib / g_bin_capacity) 
            : next_pbuffer.ch(_0, partition._binsts[src_blockno] + pidib / g_bin_capacity);
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
        particle_bin.val(_15, pidib % g_bin_capacity) = sJBar_new;
        particle_bin.val(_16, pidib % g_bin_capacity) = ID;
      }
      // Stress update
      {
#pragma unroll 9
        for (int d = 0; d < 9; d++) F[d] *= J_Scale;
        compute_stress_neohookean(pbuffer.volume, pbuffer.mu, pbuffer.lambda, F, contrib);
        contrib = (C * pbuffer.mass - contrib * newDt) * Dp_inv;
      }
    }

    local_base_index = (pos * g_dx_inv_d + 0.5).cast<int>() - 1;
    {
      int direction_tag = dir_offset((base_index - 1) / g_blocksize -
                              (local_base_index - 1) / g_blocksize);
      if (g_buckets_on_particle_buffer)
        next_pbuffer.add_advection(partition, local_base_index - 1, direction_tag, pidib);
      else 
        partition.add_advection(local_base_index - 1, direction_tag, pidib);
    }
    // Calculate shape function weights at new particle position
#pragma unroll 3
    for (char dd = 0; dd < 3; ++dd) {
      local_pos[dd] = pos[dd] - local_base_index[dd] * g_dx_d;
      PREC d = (local_pos[dd] - static_cast<PREC>(std::floor(local_pos[dd] * g_dx_inv_d + 0.5) - 1) * g_dx_d) * g_dx_inv_d;
      dws(dd, 0) = 0.5 * (1.5 - d) * (1.5 - d);
      d -= 1.0;
      dws(dd, 1) = 0.75 - d * d;
      d = 0.5 + d;
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
    int channelid = base % numMViPerBlock;
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


template <typename Partition, typename Grid>
__global__ void p2g_FBar(double dt, double newDt, const ivec3 *__restrict__ blocks,
                      const ParticleBuffer<material_e::Sand_FBAR> pbuffer,
                      ParticleBuffer<material_e::Sand_FBAR> next_pbuffer,
                      const Partition prev_partition, Partition partition,
                      const Grid grid, Grid next_grid) { 
  static constexpr uint64_t numViPerBlock = g_blockvolume * 5;
  static constexpr uint64_t numViInArena = numViPerBlock << 3;
  static constexpr uint64_t shmem_offset = (g_blockvolume * 8) << 3;

  static constexpr uint64_t numMViPerBlock = g_blockvolume * 4;
  static constexpr uint64_t numMViInArena = numMViPerBlock << 3;

  static constexpr unsigned arenamask = (g_blocksize << 1) - 1;
  static constexpr unsigned arenabits = g_blockbits + 1;

  extern __shared__ char shmem[];
  using ViArena =
      PREC_G(*)[5][g_blocksize << 1][g_blocksize << 1][g_blocksize << 1];
  using ViArenaRef =
      PREC_G(&)[5][g_blocksize << 1][g_blocksize << 1][g_blocksize << 1];
  ViArenaRef __restrict__ g2pbuffer = *reinterpret_cast<ViArena>(shmem);
  using MViArena =
      PREC_G(*)[4][g_blocksize << 1][g_blocksize << 1][g_blocksize << 1];
  using MViArenaRef =
      PREC_G(&)[4][g_blocksize << 1][g_blocksize << 1][g_blocksize << 1];
  MViArenaRef __restrict__ p2gbuffer =
      *reinterpret_cast<MViArena>(shmem + shmem_offset * sizeof(PREC_G));

  ivec3 blockid;
  int src_blockno;
  if (blocks != nullptr) { 
    blockid = blocks[blockIdx.x]; // Halo block ID
    src_blockno = partition.query(blockid); // Halo block number
  } else { 
    if (partition._haloMarks[blockIdx.x]) return; // Return early if halo block
    blockid = partition._activeKeys[blockIdx.x]; // Non-halo block ID
    src_blockno = blockIdx.x; // Non-halo block number
  }
  if (src_blockno < 0) return; // Return early if negative block number
//  else if (src_blockno > blockcnt) return; // Return early if excessive block number
  int ppb = g_buckets_on_particle_buffer
            ? next_pbuffer._ppbs[src_blockno] 
            : partition._ppbs[src_blockno]; // Particles in block
  if (ppb == 0) return; // Return early if no particles


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
    p2gbuffer[loc >> arenabits][x][y][z] = static_cast<PREC_G>(0);
  }
  __syncthreads();
  // Start Grid-to-Particle, threads are particle
  for (int pidib = threadIdx.x; pidib < ppb; pidib += blockDim.x) {
    int source_blockno, source_pidib;
    ivec3 base_index;
    {
      int advect = (g_buckets_on_particle_buffer)
                    ? next_pbuffer._blockbuckets[src_blockno * g_particle_num_per_block + pidib]
                    : partition._blockbuckets[src_blockno * g_particle_num_per_block + pidib];
      dir_components(advect / g_particle_num_per_block, base_index);
      base_index += blockid;
      source_blockno = prev_partition.query(base_index);
      source_pidib = advect % g_particle_num_per_block; // & (g_particle_num_per_block - 1);
      source_blockno = (g_buckets_on_particle_buffer)
                        ? pbuffer._binsts[source_blockno] + source_pidib / g_bin_capacity
                        : prev_partition._binsts[source_blockno] + source_pidib / g_bin_capacity;
    }
    pvec3 pos;  //< Particle position at n
    {
      auto source_particle_bin = pbuffer.ch(_0, source_blockno);
      pos[0] = source_particle_bin.val(_0, source_pidib % g_bin_capacity);  //< x
      pos[1] = source_particle_bin.val(_1, source_pidib % g_bin_capacity);  //< y
      pos[2] = source_particle_bin.val(_2, source_pidib % g_bin_capacity);  //< z
    }
    ivec3 local_base_index = (pos * g_dx_inv_d + 0.5).cast<int>() - 1;
    pvec3 local_pos = pos - local_base_index * g_dx_d;
    base_index = local_base_index;

    pvec3x3 dws;
#pragma unroll 3
    for (int dd = 0; dd < 3; ++dd) {
      PREC d = (local_pos[dd] - static_cast<PREC>(std::floor(local_pos[dd] * g_dx_inv_d + 0.5) - 1) * g_dx_d) * g_dx_inv_d;
      dws(dd, 0) = 0.5 * (1.5 - d) * (1.5 - d);
      d -= 1.0;
      dws(dd, 1) = 0.75 - d * d;
      d = 0.5 + d;
      dws(dd, 2) = 0.5 * d * d;
      local_base_index[dd] = ((local_base_index[dd] - 1) & g_blockmask) + 1;
    }
    pvec3 vel;   //< Stressed, collided grid velocity
    pvec9 C;     //< APIC affine matrix, used a few times
    vel.set(0.0); 
    C.set(0.0);
    PREC sJBar_new; //< Simple FBar G2P JBar^n+1
    sJBar_new = 0.0;

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

          PREC sJBar_i = g2pbuffer[7][local_base_index[0] + i][local_base_index[1] + j]
                            [local_base_index[2] + k] / g2pbuffer[6][local_base_index[0] + i][local_base_index[1] + j]
                            [local_base_index[2] + k];
          vel   += vi * W;
          C[0] += W * vi[0] * xixp[0] * scale;
          C[1] += W * vi[1] * xixp[0] * scale;
          C[2] += W * vi[2] * xixp[0] * scale;
          C[3] += W * vi[0] * xixp[1] * scale;
          C[4] += W * vi[1] * xixp[1] * scale;
          C[5] += W * vi[2] * xixp[1] * scale;
          C[6] += W * vi[0] * xixp[2] * scale;
          C[7] += W * vi[1] * xixp[2] * scale;
          C[8] += W * vi[2] * xixp[2] * scale;
          sJBar_new += sJBar_i * W;
        }

#pragma unroll 9
    for (int d = 0; d < 9; ++d)
      dws.val(d) = C[d] * dt * Dp_inv + ((d & 0x3) ? 0.0 : 1.0);
    pvec9 contrib;
    {
      pvec9 F;
      PREC logJp;
      // PREC sJBar;
      PREC ID;
      {
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
        logJp  = source_particle_bin.val(_12, source_pidib % g_bin_capacity); //< Volume tn
        // sJBar = source_particle_bin.val(_13, source_pidib % g_bin_capacity); //< JBar tn
        ID =  source_particle_bin.val(_14, source_pidib % g_bin_capacity);
      }
      matrixMatrixMultiplication3d(dws.data(), contrib.data(), F.data());
      
      PREC JInc = matrixDeterminant3d(dws.data());
      PREC J  = matrixDeterminant3d(F.data());
      PREC J_Scale;
      //PREC voln = J * pbuffer.volume;
      if (pbuffer.FBAR_ratio > 0.0) {
        PREC FBAR_ratio = pbuffer.FBAR_ratio;
        sJBar_new = (1.0 - FBAR_ratio) * (1.0 - J) + (FBAR_ratio) * (sJBar_new);
        // FBar_n+1 = (JBar_n+1 / J_n+1)^(1/3) * F_n+1
        J_Scale = cbrt((1.0 - sJBar_new) / J);
      }

      pos += dt * vel; //< pos update

      // May need to redefine FBAR for Sand. DefGrad. F is altered
      // In compute_stress_sand before saving to particle buffer + FBAR scaling
      {
  #pragma unroll 9
        for (int d = 0; d < 9; d++) F[d] = F[d] * J_Scale;
        compute_stress_sand(pbuffer.volume, pbuffer.mu, pbuffer.lambda, pbuffer.cohesion,
        pbuffer.beta, pbuffer.yieldSurface, pbuffer.volumeCorrection, logJp, F, contrib);
      }
      // Save updated particle values to particleBuffer global memory
      {
        auto particle_bin = g_buckets_on_particle_buffer 
            ? next_pbuffer.ch(_0, next_pbuffer._binsts[src_blockno] + pidib / g_bin_capacity) 
            : next_pbuffer.ch(_0, partition._binsts[src_blockno] + pidib / g_bin_capacity);
        particle_bin.val(_0, pidib % g_bin_capacity) = pos[0];
        particle_bin.val(_1, pidib % g_bin_capacity) = pos[1];
        particle_bin.val(_2, pidib % g_bin_capacity) = pos[2];
        particle_bin.val(_3, pidib % g_bin_capacity) = F[0] ;
        particle_bin.val(_4, pidib % g_bin_capacity) = F[1] ;
        particle_bin.val(_5, pidib % g_bin_capacity) = F[2] ;
        particle_bin.val(_6, pidib % g_bin_capacity) = F[3] ;
        particle_bin.val(_7, pidib % g_bin_capacity) = F[4] ;
        particle_bin.val(_8, pidib % g_bin_capacity) = F[5] ;
        particle_bin.val(_9, pidib % g_bin_capacity) = F[6] ;
        particle_bin.val(_10, pidib % g_bin_capacity) = F[7];
        particle_bin.val(_11, pidib % g_bin_capacity) = F[8];
        particle_bin.val(_12, pidib % g_bin_capacity) = logJp;
        particle_bin.val(_13, pidib % g_bin_capacity) = sJBar_new;
        particle_bin.val(_14, pidib % g_bin_capacity) = ID;
      }

      // Fused affine-momentum and stress contribution matrix (MLS-MPM)
      {
        contrib = (C * pbuffer.mass - contrib * newDt) * Dp_inv;
      }
    }

    local_base_index = (pos * g_dx_inv_d + 0.5).cast<int>() - 1;
    {
      int direction_tag = dir_offset((base_index - 1) / g_blocksize -
                              (local_base_index - 1) / g_blocksize);
      if (g_buckets_on_particle_buffer)
        next_pbuffer.add_advection(partition, local_base_index - 1, direction_tag, pidib);
      else 
        partition.add_advection(local_base_index - 1, direction_tag, pidib);
    }

#pragma unroll 3
    for (char dd = 0; dd < 3; ++dd) {
      local_pos[dd] = pos[dd] - local_base_index[dd] * g_dx_d;
      PREC d =  (local_pos[dd] - static_cast<PREC>(std::floor(local_pos[dd] * g_dx_inv + 0.5) - 1) * g_dx_d) * g_dx_inv_d;
      dws(dd, 0) = 0.5 * (1.5 - d) * (1.5 - d);
      d -= 1.0;
      dws(dd, 1) = 0.75 - d * d;
      d = 0.5 + d;
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
    }
  }
}

template <typename Partition, typename Grid>
__global__ void p2g_FBar(double dt, double newDt, const ivec3 *__restrict__ blocks,
                      const ParticleBuffer<material_e::Sand_ASFLIP_FBAR> pbuffer,
                      ParticleBuffer<material_e::Sand_ASFLIP_FBAR> next_pbuffer,
                      const Partition prev_partition, Partition partition,
                      const Grid grid, Grid next_grid) { 
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
    blockid = blocks[blockIdx.x]; // Halo block ID
    src_blockno = partition.query(blockid); // Halo block number
  } else { 
    if (partition._haloMarks[blockIdx.x]) return; // Return early if halo block
    blockid = partition._activeKeys[blockIdx.x]; // Non-halo block ID
    src_blockno = blockIdx.x; // Non-halo block number
  }
  if (src_blockno < 0) return; // Return early if negative block number
//  else if (src_blockno > blockcnt) return; // Return early if excessive block number
  int ppb = g_buckets_on_particle_buffer
            ? next_pbuffer._ppbs[src_blockno] 
            : partition._ppbs[src_blockno]; // Particles in block
  if (ppb == 0) return; // Return early if no particles


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
    p2gbuffer[loc >> arenabits][x][y][z] = static_cast<PREC_G>(0);
  }
  __syncthreads();
  // Start Grid-to-Particle, threads are particle
  for (int pidib = threadIdx.x; pidib < ppb; pidib += blockDim.x) {
    int source_blockno, source_pidib;
    ivec3 base_index;
    {
      int advect = (g_buckets_on_particle_buffer)
                    ? next_pbuffer._blockbuckets[src_blockno * g_particle_num_per_block + pidib]
                    : partition._blockbuckets[src_blockno * g_particle_num_per_block + pidib];
      dir_components(advect / g_particle_num_per_block, base_index);
      base_index += blockid;
      source_blockno = prev_partition.query(base_index);
      source_pidib = advect % g_particle_num_per_block; // & (g_particle_num_per_block - 1);
      source_blockno = (g_buckets_on_particle_buffer)
                        ? pbuffer._binsts[source_blockno] + source_pidib / g_bin_capacity
                        : prev_partition._binsts[source_blockno] + source_pidib / g_bin_capacity;
    }
    pvec3 pos;  //< Particle position at n
    {
      auto source_particle_bin = pbuffer.ch(_0, source_blockno);
      pos[0] = source_particle_bin.val(_0, source_pidib % g_bin_capacity);  //< x
      pos[1] = source_particle_bin.val(_1, source_pidib % g_bin_capacity);  //< y
      pos[2] = source_particle_bin.val(_2, source_pidib % g_bin_capacity);  //< z
    }
    ivec3 local_base_index = (pos * g_dx_inv_d + 0.5).cast<int>() - 1;
    pvec3 local_pos = pos - local_base_index * g_dx_d;
    base_index = local_base_index;

    pvec3x3 dws;
#pragma unroll 3
    for (int dd = 0; dd < 3; ++dd) {
      PREC d = (local_pos[dd] - static_cast<PREC>(std::floor(local_pos[dd] * g_dx_inv_d + 0.5) - 1) * g_dx_d) * g_dx_inv_d;
      dws(dd, 0) = 0.5 * (1.5 - d) * (1.5 - d);
      d -= 1.0;
      dws(dd, 1) = 0.75 - d * d;
      d = 0.5 + d;
      dws(dd, 2) = 0.5 * d * d;
      local_base_index[dd] = ((local_base_index[dd] - 1) & g_blockmask) + 1;
    }
    pvec3 vel;   //< Stressed, collided grid velocity
    pvec3 vel_FLIP; //< Unstressed, uncollided grid velocity
    pvec9 C;     //< APIC affine matrix, used a few times
    vel.set(0.0); 
    vel_FLIP.set(0.0);
    C.set(0.0);
    PREC sJBar_new; //< Simple FBar G2P JBar^n+1
    sJBar_new = 0.0;

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

          PREC sJBar_i = g2pbuffer[7][local_base_index[0] + i][local_base_index[1] + j]
                            [local_base_index[2] + k] / g2pbuffer[6][local_base_index[0] + i][local_base_index[1] + j]
                            [local_base_index[2] + k];
          vel   += vi * W;
          vel_FLIP += vi_n * W; 
          C[0] += W * vi[0] * xixp[0] * scale;
          C[1] += W * vi[1] * xixp[0] * scale;
          C[2] += W * vi[2] * xixp[0] * scale;
          C[3] += W * vi[0] * xixp[1] * scale;
          C[4] += W * vi[1] * xixp[1] * scale;
          C[5] += W * vi[2] * xixp[1] * scale;
          C[6] += W * vi[0] * xixp[2] * scale;
          C[7] += W * vi[1] * xixp[2] * scale;
          C[8] += W * vi[2] * xixp[2] * scale;
          sJBar_new += sJBar_i * W;
        }

#pragma unroll 9
    for (int d = 0; d < 9; ++d)
      dws.val(d) = C[d] * dt * Dp_inv + ((d & 0x3) ? 0.0 : 1.0);
    pvec9 contrib;
    {
      pvec9 F;
      PREC logJp;
      pvec3 vel_p; //< Particle vel. at n
      // PREC sJBar;
      PREC ID;
      {
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
        logJp  = source_particle_bin.val(_12, source_pidib % g_bin_capacity); //< Volume tn
        vel_p[0] = source_particle_bin.val(_13, source_pidib % g_bin_capacity); //< vx
        vel_p[1] = source_particle_bin.val(_14, source_pidib % g_bin_capacity); //< vy
        vel_p[2] = source_particle_bin.val(_15, source_pidib % g_bin_capacity); //< vz
        // sJBar = source_particle_bin.val(_16, source_pidib % g_bin_capacity); //< JBar tn
        ID =  source_particle_bin.val(_17, source_pidib % g_bin_capacity);
      }
      matrixMatrixMultiplication3d(dws.data(), contrib.data(), F.data());
      
      PREC JInc = matrixDeterminant3d(dws.data());
      PREC J  = matrixDeterminant3d(F.data());
      PREC J_Scale;
      //PREC voln = J * pbuffer.volume;
      if (pbuffer.FBAR_ratio > 0.0) {
        PREC FBAR_ratio = pbuffer.FBAR_ratio;
        sJBar_new = (1.0 - FBAR_ratio) * (1.0 - J) + (FBAR_ratio) * (sJBar_new);
        // FBar_n+1 = (JBar_n+1 / J_n+1)^(1/3) * F_n+1
        J_Scale = cbrt((1.0 - sJBar_new) / J);
      }

      if (pbuffer.alpha > 0.0) {
        // ASFLIP Position correction factor
        PREC beta = (sJBar_new < 0.0) ? pbuffer.beta_max : pbuffer.beta_min; 

        // * Advect particle position and velocity
        pos += dt * (vel + beta * pbuffer.alpha * (vel_p - vel_FLIP)); //< pos update
        vel += pbuffer.alpha * (vel_p - vel_FLIP); //< vel update
      } else {
        pos += dt * vel;
      }

      // May need to redefine FBAR for Sand. DefGrad. F is altered
      // In compute_stress_sand before saving to particle buffer + FBAR scaling
      {
  #pragma unroll 9
        for (int d = 0; d < 9; d++) F[d] = F[d] * J_Scale;
        compute_stress_sand(pbuffer.volume, pbuffer.mu, pbuffer.lambda, pbuffer.cohesion,
        pbuffer.beta, pbuffer.yieldSurface, pbuffer.volumeCorrection, logJp, F, contrib);
      }
      // Save updated particle values to particleBuffer global memory
      {
        auto particle_bin = g_buckets_on_particle_buffer 
            ? next_pbuffer.ch(_0, next_pbuffer._binsts[src_blockno] + pidib / g_bin_capacity) 
            : next_pbuffer.ch(_0, partition._binsts[src_blockno] + pidib / g_bin_capacity);
        particle_bin.val(_0, pidib % g_bin_capacity) = pos[0];
        particle_bin.val(_1, pidib % g_bin_capacity) = pos[1];
        particle_bin.val(_2, pidib % g_bin_capacity) = pos[2];
        particle_bin.val(_3, pidib % g_bin_capacity) = F[0] ;
        particle_bin.val(_4, pidib % g_bin_capacity) = F[1] ;
        particle_bin.val(_5, pidib % g_bin_capacity) = F[2] ;
        particle_bin.val(_6, pidib % g_bin_capacity) = F[3] ;
        particle_bin.val(_7, pidib % g_bin_capacity) = F[4] ;
        particle_bin.val(_8, pidib % g_bin_capacity) = F[5] ;
        particle_bin.val(_9, pidib % g_bin_capacity) = F[6] ;
        particle_bin.val(_10, pidib % g_bin_capacity) = F[7];
        particle_bin.val(_11, pidib % g_bin_capacity) = F[8];
        particle_bin.val(_12, pidib % g_bin_capacity) = logJp;
        particle_bin.val(_13, pidib % g_bin_capacity) = vel[0];
        particle_bin.val(_14, pidib % g_bin_capacity) = vel[1];
        particle_bin.val(_15, pidib % g_bin_capacity) = vel[2];
        particle_bin.val(_16, pidib % g_bin_capacity) = sJBar_new;
        particle_bin.val(_17, pidib % g_bin_capacity) = ID;
      }

      // Fused affine-momentum and stress contribution matrix (MLS-MPM)
      {
        contrib = (C * pbuffer.mass - contrib * newDt) * Dp_inv;
      }
    }

    local_base_index = (pos * g_dx_inv_d + 0.5).cast<int>() - 1;
    {
      int direction_tag = dir_offset((base_index - 1) / g_blocksize -
                              (local_base_index - 1) / g_blocksize);
      if (g_buckets_on_particle_buffer)
        next_pbuffer.add_advection(partition, local_base_index - 1, direction_tag, pidib);
      else 
        partition.add_advection(local_base_index - 1, direction_tag, pidib);
    }

#pragma unroll 3
    for (char dd = 0; dd < 3; ++dd) {
      local_pos[dd] = pos[dd] - local_base_index[dd] * g_dx_d;
      PREC d =  (local_pos[dd] - static_cast<PREC>(std::floor(local_pos[dd] * g_dx_inv + 0.5) - 1) * g_dx_d) * g_dx_inv_d;
      dws(dd, 0) = 0.5 * (1.5 - d) * (1.5 - d);
      d -= 1.0;
      dws(dd, 1) = 0.75 - d * d;
      d = 0.5 + d;
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
    int channelid = base % numMViPerBlock;
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


template <typename Partition, typename Grid>
__global__ void p2g_FBar(double dt, double newDt, const ivec3 *__restrict__ blocks,
                      const ParticleBuffer<material_e::NACC_FBAR> pbuffer,
                      ParticleBuffer<material_e::NACC_FBAR> next_pbuffer,
                      const Partition prev_partition, Partition partition,
                      const Grid grid, Grid next_grid) { 
  static constexpr uint64_t numViPerBlock = g_blockvolume * 5;
  static constexpr uint64_t numViInArena = numViPerBlock << 3;
  static constexpr uint64_t shmem_offset = (g_blockvolume * 8) << 3;

  static constexpr uint64_t numMViPerBlock = g_blockvolume * 4;
  static constexpr uint64_t numMViInArena = numMViPerBlock << 3;

  static constexpr unsigned arenamask = (g_blocksize << 1) - 1;
  static constexpr unsigned arenabits = g_blockbits + 1;

  extern __shared__ char shmem[];
  using ViArena =
      PREC_G(*)[5][g_blocksize << 1][g_blocksize << 1][g_blocksize << 1];
  using ViArenaRef =
      PREC_G(&)[5][g_blocksize << 1][g_blocksize << 1][g_blocksize << 1];
  ViArenaRef __restrict__ g2pbuffer = *reinterpret_cast<ViArena>(shmem);
  using MViArena =
      PREC_G(*)[4][g_blocksize << 1][g_blocksize << 1][g_blocksize << 1];
  using MViArenaRef =
      PREC_G(&)[4][g_blocksize << 1][g_blocksize << 1][g_blocksize << 1];
  MViArenaRef __restrict__ p2gbuffer =
      *reinterpret_cast<MViArena>(shmem + shmem_offset * sizeof(PREC_G));

  ivec3 blockid;
  int src_blockno;
  if (blocks != nullptr) { 
    blockid = blocks[blockIdx.x]; // Halo block ID
    src_blockno = partition.query(blockid); // Halo block number
  } else { 
    if (partition._haloMarks[blockIdx.x]) return; // Return early if halo block
    blockid = partition._activeKeys[blockIdx.x]; // Non-halo block ID
    src_blockno = blockIdx.x; // Non-halo block number
  }
  if (src_blockno < 0) return; // Return early if negative block number
//  else if (src_blockno > blockcnt) return; // Return early if excessive block number
  int ppb = g_buckets_on_particle_buffer
            ? next_pbuffer._ppbs[src_blockno] 
            : partition._ppbs[src_blockno]; // Particles in block
  if (ppb == 0) return; // Return early if no particles


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
    p2gbuffer[loc >> arenabits][x][y][z] = (PREC_G)0.0;
  }
  __syncthreads();
  // Start Grid-to-Particle, threads are particle
  for (int pidib = threadIdx.x; pidib < ppb; pidib += blockDim.x) {
    int source_blockno, source_pidib;
    ivec3 base_index;
    {
      int advect = (g_buckets_on_particle_buffer)
                    ? next_pbuffer._blockbuckets[src_blockno * g_particle_num_per_block + pidib]
                    : partition._blockbuckets[src_blockno * g_particle_num_per_block + pidib];
      dir_components(advect / g_particle_num_per_block, base_index);
      base_index += blockid;
      source_blockno = prev_partition.query(base_index);
      source_pidib = advect % g_particle_num_per_block; // & (g_particle_num_per_block - 1);
      source_blockno = (g_buckets_on_particle_buffer)
                        ? pbuffer._binsts[source_blockno] + source_pidib / g_bin_capacity
                        : prev_partition._binsts[source_blockno] + source_pidib / g_bin_capacity;
    }
    pvec3 pos;  //< Particle position at n
    {
      auto source_particle_bin = pbuffer.ch(_0, source_blockno);
      pos[0] = source_particle_bin.val(_0, source_pidib % g_bin_capacity);  //< x
      pos[1] = source_particle_bin.val(_1, source_pidib % g_bin_capacity);  //< y
      pos[2] = source_particle_bin.val(_2, source_pidib % g_bin_capacity);  //< z
    }
    ivec3 local_base_index = (pos * g_dx_inv_d + 0.5).cast<int>() - 1;
    pvec3 local_pos = pos - local_base_index * g_dx_d;
    base_index = local_base_index;

    pvec3x3 dws;
#pragma unroll 3
    for (int dd = 0; dd < 3; ++dd) {
      PREC d = (local_pos[dd] - static_cast<PREC>(std::floor(local_pos[dd] * g_dx_inv_d + 0.5) - 1) * g_dx_d) * g_dx_inv_d;
      dws(dd, 0) = 0.5 * (1.5 - d) * (1.5 - d);
      d -= 1.0;
      dws(dd, 1) = 0.75 - d * d;
      d = 0.5 + d;
      dws(dd, 2) = 0.5 * d * d;
      local_base_index[dd] = ((local_base_index[dd] - 1) & g_blockmask) + 1;
    }
    pvec3 vel; //< Unstressed, uncollided grid velocity
    pvec9 C;     //< APIC affine matrix, used a few times
    vel.set(0.0); 
    C.set(0.0);
    PREC sJBar_new = 0; //< Simple FBar G2P JBar^n+1

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

          PREC sJBar_i = g2pbuffer[7][local_base_index[0] + i][local_base_index[1] + j]
                            [local_base_index[2] + k] / g2pbuffer[6][local_base_index[0] + i][local_base_index[1] + j]
                            [local_base_index[2] + k];
          vel   += vi * W;
          C[0] += W * vi[0] * xixp[0] * scale;
          C[1] += W * vi[1] * xixp[0] * scale;
          C[2] += W * vi[2] * xixp[0] * scale;
          C[3] += W * vi[0] * xixp[1] * scale;
          C[4] += W * vi[1] * xixp[1] * scale;
          C[5] += W * vi[2] * xixp[1] * scale;
          C[6] += W * vi[0] * xixp[2] * scale;
          C[7] += W * vi[1] * xixp[2] * scale;
          C[8] += W * vi[2] * xixp[2] * scale;
          sJBar_new += sJBar_i * W;
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
      PREC logJp  = source_particle_bin.val(_12, source_pidib % g_bin_capacity); //< Volume tn
      PREC sJBar = source_particle_bin.val(_13, source_pidib % g_bin_capacity); //< JBar tn
      PREC ID =  source_particle_bin.val(_14, source_pidib % g_bin_capacity);

      matrixMatrixMultiplication3d(dws.data(), contrib.data(), F.data());
      
      PREC JInc = matrixDeterminant3d(dws.data());
      PREC J  = matrixDeterminant3d(F.data());
      //PREC voln = J * pbuffer.volume;

      PREC beta; //< Position correction factor (ASFLIP)
      if ((1.0 - sJBar_new) >= 1.0) beta = pbuffer.beta_max;  // beta max
      else beta = pbuffer.beta_min; // beta min

      // Advect particle position and velocity
      pos += dt * vel; //< pos update

      //FBar_n+1 = (JBar_n+1 / J_n+1)^(1/3) * F_n+1
      PREC J_Scale = cbrt((1.0 - sJBar_new) / J);

      PREC FBAR_ratio = pbuffer.FBAR_ratio;
#pragma unroll 9
      for (int d = 0; d < 9; d++) F[d] = F[d] * ((1.0 - FBAR_ratio) * 1.0 + (FBAR_ratio) * J_Scale);
      compute_stress_nacc(pbuffer.volume, pbuffer.mu, pbuffer.lambda, pbuffer.bm, pbuffer.xi, pbuffer.beta, pbuffer.Msqr, pbuffer.hardeningOn, logJp,
                                    F, contrib);
      {
        auto particle_bin = g_buckets_on_particle_buffer 
            ? next_pbuffer.ch(_0, next_pbuffer._binsts[src_blockno] + pidib / g_bin_capacity) 
            : next_pbuffer.ch(_0, partition._binsts[src_blockno] + pidib / g_bin_capacity);
        particle_bin.val(_0, pidib % g_bin_capacity) = pos[0];
        particle_bin.val(_1, pidib % g_bin_capacity) = pos[1];
        particle_bin.val(_2, pidib % g_bin_capacity) = pos[2];
        particle_bin.val(_3, pidib % g_bin_capacity) = F[0] ;
        particle_bin.val(_4, pidib % g_bin_capacity) = F[1] ;
        particle_bin.val(_5, pidib % g_bin_capacity) = F[2] ;
        particle_bin.val(_6, pidib % g_bin_capacity) = F[3] ;
        particle_bin.val(_7, pidib % g_bin_capacity) = F[4] ;
        particle_bin.val(_8, pidib % g_bin_capacity) = F[5] ;
        particle_bin.val(_9, pidib % g_bin_capacity) = F[6] ;
        particle_bin.val(_10, pidib % g_bin_capacity) = F[7];
        particle_bin.val(_11, pidib % g_bin_capacity) = F[8];
        particle_bin.val(_12, pidib % g_bin_capacity) = logJp;
        particle_bin.val(_13, pidib % g_bin_capacity) = sJBar_new;
        particle_bin.val(_14, pidib % g_bin_capacity) = ID;
      }

      {

      contrib = (C * pbuffer.mass - contrib * newDt) * Dp_inv;
      }
    }

    local_base_index = (pos * g_dx_inv + 0.5f).cast<int>() - 1;
    {
      int direction_tag = dir_offset((base_index - 1) / g_blocksize -
                              (local_base_index - 1) / g_blocksize);
      if (g_buckets_on_particle_buffer)
        next_pbuffer.add_advection(partition, local_base_index - 1, direction_tag, pidib);
      else 
        partition.add_advection(local_base_index - 1, direction_tag, pidib);
    }

#pragma unroll 3
    for (char dd = 0; dd < 3; ++dd) {
      local_pos[dd] = pos[dd] - local_base_index[dd] * g_dx;
      PREC d =
          (local_pos[dd] - ((int)(local_pos[dd] * g_dx_inv + 0.5) - 1) * g_dx) *
          g_dx_inv;
      dws(dd, 0) = 0.5 * (1.5 - d) * (1.5 - d);
      d -= 1.0;
      dws(dd, 1) = 0.75 - d * d;
      d = 0.5 + d;
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
    }
  }
}


template <typename Partition, typename Grid>
__global__ void p2g_FBar(double dt, double newDt, const ivec3 *__restrict__ blocks,
                      const ParticleBuffer<material_e::NACC_ASFLIP_FBAR> pbuffer,
                      ParticleBuffer<material_e::NACC_ASFLIP_FBAR> next_pbuffer,
                      const Partition prev_partition, Partition partition,
                      const Grid grid, Grid next_grid) { 
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
    blockid = blocks[blockIdx.x]; // Halo block ID
    src_blockno = partition.query(blockid); // Halo block number
  } else { 
    if (partition._haloMarks[blockIdx.x]) return; // Return early if halo block
    blockid = partition._activeKeys[blockIdx.x]; // Non-halo block ID
    src_blockno = blockIdx.x; // Non-halo block number
  }
  if (src_blockno < 0) return; // Return early if negative block number
//  else if (src_blockno > blockcnt) return; // Return early if excessive block number
  int ppb = g_buckets_on_particle_buffer
            ? next_pbuffer._ppbs[src_blockno] 
            : partition._ppbs[src_blockno]; // Particles in block
  if (ppb == 0) return; // Return early if no particles


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
    p2gbuffer[loc >> arenabits][x][y][z] = (PREC_G)0.0;
  }
  __syncthreads();
  // Start Grid-to-Particle, threads are particle
  for (int pidib = threadIdx.x; pidib < ppb; pidib += blockDim.x) {
    int source_blockno, source_pidib;
    ivec3 base_index;
    {
      int advect = (g_buckets_on_particle_buffer)
                    ? next_pbuffer._blockbuckets[src_blockno * g_particle_num_per_block + pidib]
                    : partition._blockbuckets[src_blockno * g_particle_num_per_block + pidib];
      dir_components(advect / g_particle_num_per_block, base_index);
      base_index += blockid;
      source_blockno = prev_partition.query(base_index);
      source_pidib = advect % g_particle_num_per_block; // & (g_particle_num_per_block - 1);
      source_blockno = (g_buckets_on_particle_buffer)
                        ? pbuffer._binsts[source_blockno] + source_pidib / g_bin_capacity
                        : prev_partition._binsts[source_blockno] + source_pidib / g_bin_capacity;
    }
    pvec3 pos;  //< Particle position at n
    {
      auto source_particle_bin = pbuffer.ch(_0, source_blockno);
      pos[0] = source_particle_bin.val(_0, source_pidib % g_bin_capacity);  //< x
      pos[1] = source_particle_bin.val(_1, source_pidib % g_bin_capacity);  //< y
      pos[2] = source_particle_bin.val(_2, source_pidib % g_bin_capacity);  //< z
    }
    ivec3 local_base_index = (pos * g_dx_inv_d + 0.5).cast<int>() - 1;
    pvec3 local_pos = pos - local_base_index * g_dx_d;
    base_index = local_base_index;

    pvec3x3 dws;
#pragma unroll 3
    for (int dd = 0; dd < 3; ++dd) {
      PREC d = (local_pos[dd] - static_cast<PREC>(std::floor(local_pos[dd] * g_dx_inv_d + 0.5) - 1) * g_dx_d) * g_dx_inv_d;
      dws(dd, 0) = 0.5 * (1.5 - d) * (1.5 - d);
      d -= 1.0;
      dws(dd, 1) = 0.75 - d * d;
      d = 0.5 + d;
      dws(dd, 2) = 0.5 * d * d;
      local_base_index[dd] = ((local_base_index[dd] - 1) & g_blockmask) + 1;
    }
    pvec3 vel, vel_FLIP; //< Unstressed, uncollided grid velocity
    pvec9 C;     //< APIC affine matrix, used a few times
    vel.set(0.0); 
    vel_FLIP.set(0.0);
    C.set(0.0);
    PREC sJBar_new = 0; //< Simple FBar G2P JBar^n+1

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

          PREC sJBar_i = g2pbuffer[7][local_base_index[0] + i][local_base_index[1] + j]
                            [local_base_index[2] + k] / g2pbuffer[6][local_base_index[0] + i][local_base_index[1] + j]
                            [local_base_index[2] + k];
          vel   += vi * W;
          vel_FLIP += vi_n * W; 
          C[0] += W * vi[0] * xixp[0] * scale;
          C[1] += W * vi[1] * xixp[0] * scale;
          C[2] += W * vi[2] * xixp[0] * scale;
          C[3] += W * vi[0] * xixp[1] * scale;
          C[4] += W * vi[1] * xixp[1] * scale;
          C[5] += W * vi[2] * xixp[1] * scale;
          C[6] += W * vi[0] * xixp[2] * scale;
          C[7] += W * vi[1] * xixp[2] * scale;
          C[8] += W * vi[2] * xixp[2] * scale;
          sJBar_new += sJBar_i * W;
        }

#pragma unroll 9
    for (int d = 0; d < 9; ++d)
      dws.val(d) = C[d] * dt * Dp_inv + ((d & 0x3) ? 0.0 : 1.0);
    pvec9 contrib;
    {
      pvec9 F;
      pvec3 vel_p; //< Particle vel. at n
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
      PREC logJp  = source_particle_bin.val(_12, source_pidib % g_bin_capacity); //< Volume tn
      vel_p[0] = source_particle_bin.val(_13, source_pidib % g_bin_capacity); //< vx
      vel_p[1] = source_particle_bin.val(_14, source_pidib % g_bin_capacity); //< vy
      vel_p[2] = source_particle_bin.val(_15, source_pidib % g_bin_capacity); //< vz
      PREC sJBar = source_particle_bin.val(_16, source_pidib % g_bin_capacity); //< JBar tn
      PREC ID =  source_particle_bin.val(_17, source_pidib % g_bin_capacity);

      matrixMatrixMultiplication3d(dws.data(), contrib.data(), F.data());
      
      PREC JInc = matrixDeterminant3d(dws.data());
      PREC J  = matrixDeterminant3d(F.data());
      //PREC voln = J * pbuffer.volume;

      PREC beta; //< Position correction factor (ASFLIP)
      if ((1.0 - sJBar_new) >= 1.0) beta = pbuffer.beta_max;  // beta max
      else beta = pbuffer.beta_min; // beta min

      // Advect particle position and velocity
      pos += dt * (vel + beta * pbuffer.alpha * (vel_p - vel_FLIP)); //< pos update
      vel += pbuffer.alpha * (vel_p - vel_FLIP); //< vel update
      
      //FBar_n+1 = (JBar_n+1 / J_n+1)^(1/3) * F_n+1
      PREC J_Scale = cbrt((1.0 - sJBar_new) / J);

      PREC FBAR_ratio = pbuffer.FBAR_ratio;
#pragma unroll 9
      for (int d = 0; d < 9; d++) F[d] = F[d] * ((1.0 - FBAR_ratio) * 1.0 + (FBAR_ratio) * J_Scale);
      compute_stress_nacc(pbuffer.volume, pbuffer.mu, pbuffer.lambda, pbuffer.bm, pbuffer.xi, pbuffer.beta, pbuffer.Msqr, pbuffer.hardeningOn, logJp,
                                    F, contrib);
      {
        auto particle_bin = g_buckets_on_particle_buffer 
            ? next_pbuffer.ch(_0, next_pbuffer._binsts[src_blockno] + pidib / g_bin_capacity) 
            : next_pbuffer.ch(_0, partition._binsts[src_blockno] + pidib / g_bin_capacity);
        particle_bin.val(_0, pidib % g_bin_capacity) = pos[0];
        particle_bin.val(_1, pidib % g_bin_capacity) = pos[1];
        particle_bin.val(_2, pidib % g_bin_capacity) = pos[2];
        particle_bin.val(_3, pidib % g_bin_capacity) = F[0] ;
        particle_bin.val(_4, pidib % g_bin_capacity) = F[1] ;
        particle_bin.val(_5, pidib % g_bin_capacity) = F[2] ;
        particle_bin.val(_6, pidib % g_bin_capacity) = F[3] ;
        particle_bin.val(_7, pidib % g_bin_capacity) = F[4] ;
        particle_bin.val(_8, pidib % g_bin_capacity) = F[5] ;
        particle_bin.val(_9, pidib % g_bin_capacity) = F[6] ;
        particle_bin.val(_10, pidib % g_bin_capacity) = F[7];
        particle_bin.val(_11, pidib % g_bin_capacity) = F[8];
        particle_bin.val(_12, pidib % g_bin_capacity) = logJp;
        particle_bin.val(_13, pidib % g_bin_capacity) = vel[0];
        particle_bin.val(_14, pidib % g_bin_capacity) = vel[1];
        particle_bin.val(_15, pidib % g_bin_capacity) = vel[2];
        particle_bin.val(_16, pidib % g_bin_capacity) = sJBar_new;
        particle_bin.val(_17, pidib % g_bin_capacity) = ID;
      }

      {

      contrib = (C * pbuffer.mass - contrib * newDt) * Dp_inv;
      }
    }

    local_base_index = (pos * g_dx_inv + 0.5f).cast<int>() - 1;
    {
      int direction_tag = dir_offset((base_index - 1) / g_blocksize -
                              (local_base_index - 1) / g_blocksize);
      if (g_buckets_on_particle_buffer)
        next_pbuffer.add_advection(partition, local_base_index - 1, direction_tag, pidib);
      else 
        partition.add_advection(local_base_index - 1, direction_tag, pidib);
    }

#pragma unroll 3
    for (char dd = 0; dd < 3; ++dd) {
      local_pos[dd] = pos[dd] - local_base_index[dd] * g_dx;
      PREC d =
          (local_pos[dd] - ((int)(local_pos[dd] * g_dx_inv + 0.5) - 1) * g_dx) *
          g_dx_inv;
      dws(dd, 0) = 0.5 * (1.5 - d) * (1.5 - d);
      d -= 1.0;
      dws(dd, 1) = 0.75 - d * d;
      d = 0.5 + d;
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


template <typename Partition, typename Grid>
__global__ void p2g_FBar(double dt, double newDt, const ivec3 *__restrict__ blocks,
                      const ParticleBuffer<material_e::CoupledUP> pbuffer,
                      ParticleBuffer<material_e::CoupledUP> next_pbuffer,
                      const Partition prev_partition, Partition partition,
                      const Grid grid, Grid next_grid) {
  static constexpr uint64_t numViPerBlock = g_blockvolume * 10;
  static constexpr uint64_t numViInArena = numViPerBlock << 3;
  static constexpr uint64_t shmem_offset = (g_blockvolume * 10) << 3;

  static constexpr uint64_t numMViPerBlock = g_blockvolume * 9;
  static constexpr uint64_t numMViInArena = numMViPerBlock << 3;

  static constexpr unsigned arenamask = (g_blocksize << 1) - 1;
  static constexpr unsigned arenabits = g_blockbits + 1;

#if !DEBUG_COUPLED_UP
  printf("ERROR: Turn on DEBUG_COUPLED_UP to use CoupledUP material.\n");
  return;
#endif
  extern __shared__ char shmem[];
  using ViArena =
      PREC_G(*)[10][g_blocksize << 1][g_blocksize << 1][g_blocksize << 1];
  using ViArenaRef =
      PREC_G(&)[10][g_blocksize << 1][g_blocksize << 1][g_blocksize << 1];
  ViArenaRef __restrict__ g2pbuffer = *reinterpret_cast<ViArena>(shmem);
  using MViArena =
      PREC_G(*)[9][g_blocksize << 1][g_blocksize << 1][g_blocksize << 1];
  using MViArenaRef =
      PREC_G(&)[9][g_blocksize << 1][g_blocksize << 1][g_blocksize << 1];
  MViArenaRef __restrict__ p2gbuffer =
      *reinterpret_cast<MViArena>(shmem + shmem_offset * sizeof(PREC_G));

  ivec3 blockid;
  int src_blockno;
  if (blocks != nullptr) { 
    blockid = blocks[blockIdx.x]; // Halo block ID
    src_blockno = partition.query(blockid); // Halo block number
  } else { 
    if (partition._haloMarks[blockIdx.x]) return; // Return early if halo block
    blockid = partition._activeKeys[blockIdx.x]; // Non-halo block ID
    src_blockno = blockIdx.x; // Non-halo block number
  }
  if (src_blockno < 0) return; // Return early if negative block number
//  else if (src_blockno > blockcnt) return; // Return early if excessive block number
  int ppb = g_buckets_on_particle_buffer
            ? next_pbuffer._ppbs[src_blockno] 
            : partition._ppbs[src_blockno]; // Particles in block
  if (ppb == 0) return; // Return early if no particles

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
      val = grid_block.val_1d(_1, c); // vel(1) + dt*fint(1)
    else if (channelid == 1)
      val = grid_block.val_1d(_2, c);
    else if (channelid == 2)
      val = grid_block.val_1d(_3, c);
    else if (channelid == 3) 
      val = grid_block.val_1d(_4, c); // vel(1)
    else if (channelid == 4) 
      val = grid_block.val_1d(_5, c);
    else if (channelid == 5) 
      val = grid_block.val_1d(_6, c);
    else if (channelid == 6) 
      val = grid_block.val_1d(_7, c); // Vol
    else if (channelid == 7) 
      val = grid_block.val_1d(_8, c); // J
#if DEBUG_COUPLED_UP
    else if (channelid == 8) 
      val = grid_block.val_1d(_9, c); // mass_w
    else if (channelid == 9) 
      val = grid_block.val_1d(_10, c); // pressure_w
#endif
    g2pbuffer[channelid]
             [cx + (local_block_id & 4 ? g_blocksize : 0)]
             [cy + (local_block_id & 2 ? g_blocksize : 0)]
             [cz + (local_block_id & 1 ? g_blocksize : 0)] = val;
  }
  __syncthreads();
  for (int base = threadIdx.x; base < numMViInArena; base += blockDim.x) {
    int loc = base;
    char z = loc & arenamask;
    char y = (loc >>= arenabits) & arenamask;
    char x = (loc >>= arenabits) & arenamask;
    p2gbuffer[loc >> arenabits][x][y][z] = (PREC_G)0.0;
  }
  __syncthreads();
  // Start Grid-to-Particle, threads are particle
  for (int pidib = threadIdx.x; pidib < ppb; pidib += blockDim.x) {
    int source_blockno, source_pidib;
    ivec3 base_index;
    {
      int advect = (g_buckets_on_particle_buffer)
                    ? next_pbuffer._blockbuckets[src_blockno * g_particle_num_per_block + pidib]
                    : partition._blockbuckets[src_blockno * g_particle_num_per_block + pidib];
      dir_components(advect / g_particle_num_per_block, base_index);
      base_index += blockid;
      source_blockno = prev_partition.query(base_index);
      source_pidib = advect % g_particle_num_per_block; // & (g_particle_num_per_block - 1);
      source_blockno = (g_buckets_on_particle_buffer)
                        ? pbuffer._binsts[source_blockno] + source_pidib / g_bin_capacity
                        : prev_partition._binsts[source_blockno] + source_pidib / g_bin_capacity;
    }
    pvec3 pos;  //< Particle position at n
    {
      auto source_particle_bin = pbuffer.ch(_0, source_blockno);
      pos[0] = source_particle_bin.val(_0, source_pidib % g_bin_capacity);  //< x
      pos[1] = source_particle_bin.val(_1, source_pidib % g_bin_capacity);  //< y
      pos[2] = source_particle_bin.val(_2, source_pidib % g_bin_capacity);  //< z
    }
    ivec3 local_base_index = (pos * g_dx_inv_d + 0.5).cast<int>() - 1;
    pvec3 local_pos = pos - local_base_index * g_dx_d;
    base_index = local_base_index;

    pvec3x3 dws;
#pragma unroll 3
    for (int dd = 0; dd < 3; ++dd) {
      PREC d = (local_pos[dd] - static_cast<PREC>(std::floor(local_pos[dd] * g_dx_inv_d + 0.5) - 1) * g_dx_d) * g_dx_inv_d;
      dws(dd, 0) = 0.5 * (1.5 - d) * (1.5 - d);
      d -= 1.0;
      dws(dd, 1) = 0.75 - d * d;
      d = 0.5 + d;
      dws(dd, 2) = 0.5 * d * d;
      local_base_index[dd] = ((local_base_index[dd] - 1) & g_blockmask) + 1;
    }
    pvec3 vel;   //< Stressed, collided grid velocity
    pvec3 vel_FLIP; //< Unstressed, uncollided grid velocity
    pvec9 C;     //< APIC affine matrix, used a few times
    vel.set(0.0); 
    vel_FLIP.set(0.0);
    C.set(0.0);
    PREC sJBar_new; //< Simple FBar G2P JBar^n+1
    sJBar_new = 0.0;

    PREC pw_new; //< CoupledUP G2P water pressure^n+1
    pw_new = 0.0;

    // Dp^n = Dp^n+1 = (1/4) * dx^2 * I (Quad.)
    PREC Dp_inv; //< Inverse Intertia-Like Tensor (1/m^2)
    PREC scale;
    scale = pbuffer.length * pbuffer.length; //< Area scale (m^2)
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

          PREC sJBar_i = g2pbuffer[7][local_base_index[0] + i][local_base_index[1] + j]
                            [local_base_index[2] + k] / g2pbuffer[6][local_base_index[0] + i][local_base_index[1] + j]
                            [local_base_index[2] + k];

          PREC pw_i = g2pbuffer[9][local_base_index[0] + i][local_base_index[1] + j]
                            [local_base_index[2] + k] / g2pbuffer[8][local_base_index[0] + i][local_base_index[1] + j][local_base_index[2] + k];
          vel   += vi * W;
          vel_FLIP += vi_n * W; 
          C[0] += W * vi[0] * xixp[0] * scale;
          C[1] += W * vi[1] * xixp[0] * scale;
          C[2] += W * vi[2] * xixp[0] * scale;
          C[3] += W * vi[0] * xixp[1] * scale;
          C[4] += W * vi[1] * xixp[1] * scale;
          C[5] += W * vi[2] * xixp[1] * scale;
          C[6] += W * vi[0] * xixp[2] * scale;
          C[7] += W * vi[1] * xixp[2] * scale;
          C[8] += W * vi[2] * xixp[2] * scale;
          sJBar_new += sJBar_i * W;
          pw_new += pw_i * W; //< CoupledUP G2P water pressure increment
        }

#pragma unroll 9
    for (int d = 0; d < 9; ++d)
      dws.val(d) = C[d] * dt * Dp_inv + ((d & 0x3) ? 0.0 : 1.0);
    pvec9 contrib;
    {
      pvec9 F;
      PREC logJp;
      pvec3 vel_p; //< Particle vel. at n
      // PREC sJBar;
      PREC ID;
      // PREC masw;
      // PREC pw; // CoupledUP water pressure old

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
      logJp  = source_particle_bin.val(_12, source_pidib % g_bin_capacity); //< Volume tn
      vel_p[0] = source_particle_bin.val(_13, source_pidib % g_bin_capacity); //< vx
      vel_p[1] = source_particle_bin.val(_14, source_pidib % g_bin_capacity); //< vy
      vel_p[2] = source_particle_bin.val(_15, source_pidib % g_bin_capacity); //< vz
      // sJBar = source_particle_bin.val(_16, source_pidib % g_bin_capacity); //< JBar tn
      //masw =  source_particle_bin.val(_17, source_pidib % g_bin_capacity); // water mass old
      // pw =  source_particle_bin.val(_18, source_pidib % g_bin_capacity); // water pore pressure old
      ID =  source_particle_bin.val(_19, source_pidib % g_bin_capacity);

      matrixMatrixMultiplication3d(dws.data(), contrib.data(), F.data());
      
      PREC JInc = matrixDeterminant3d(dws.data());
      PREC J  = matrixDeterminant3d(F.data());
      // PREC voln = J * pbuffer.volume;

      PREC beta; //< Position correction factor (ASFLIP)
      if ((1.0 - sJBar_new) >= 1.0) beta = pbuffer.beta_max;  // beta max
      else beta = pbuffer.beta_min; // beta min

      // * Advect particle position and velocity
      pos += dt * (vel + beta * pbuffer.alpha * (vel_p - vel_FLIP)); //< pos update
      vel += pbuffer.alpha * (vel_p - vel_FLIP); //< vel update
      
      // FBar_n+1 = (JBar_n+1 / J_n+1)^(1/3) * F_n+1
      PREC J_Scale = cbrt((1.0 - sJBar_new) / J);
      // May need to redefine FBAR for Sand. DefGrad. F is altered
      // In compute_stress_sand before saving to particle buffer + FBAR scaling
      PREC FBAR_ratio = pbuffer.FBAR_ratio;
#pragma unroll 9

      for (int d = 0; d < 9; d++) F[d] = F[d] * ((1.0 - FBAR_ratio) * 1.0 + (FBAR_ratio) * J_Scale);
      
      compute_stress_CoupledUP(pbuffer.volume, pbuffer.mu, pbuffer.lambda, pbuffer.cohesion,
      pbuffer.beta, pbuffer.yieldSurface, pbuffer.volumeCorrection, logJp, pw_new, F, contrib);

      contrib[0] -= pw_new * pbuffer.volume;
      contrib[4] -= pw_new * pbuffer.volume;
      contrib[8] -= pw_new * pbuffer.volume;


      {
        auto particle_bin = g_buckets_on_particle_buffer 
            ? next_pbuffer.ch(_0, next_pbuffer._binsts[src_blockno] + pidib / g_bin_capacity) 
            : next_pbuffer.ch(_0, partition._binsts[src_blockno] + pidib / g_bin_capacity);
        particle_bin.val(_0, pidib % g_bin_capacity) = pos[0];
        particle_bin.val(_1, pidib % g_bin_capacity) = pos[1];
        particle_bin.val(_2, pidib % g_bin_capacity) = pos[2];
        particle_bin.val(_3, pidib % g_bin_capacity) = F[0] ;
        particle_bin.val(_4, pidib % g_bin_capacity) = F[1] ;
        particle_bin.val(_5, pidib % g_bin_capacity) = F[2] ;
        particle_bin.val(_6, pidib % g_bin_capacity) = F[3] ;
        particle_bin.val(_7, pidib % g_bin_capacity) = F[4] ;
        particle_bin.val(_8, pidib % g_bin_capacity) = F[5] ;
        particle_bin.val(_9, pidib % g_bin_capacity) = F[6] ;
        particle_bin.val(_10, pidib % g_bin_capacity) = F[7];
        particle_bin.val(_11, pidib % g_bin_capacity) = F[8];
        particle_bin.val(_12, pidib % g_bin_capacity) = logJp;
        particle_bin.val(_13, pidib % g_bin_capacity) = vel[0];
        particle_bin.val(_14, pidib % g_bin_capacity) = vel[1];
        particle_bin.val(_15, pidib % g_bin_capacity) = vel[2];
        particle_bin.val(_16, pidib % g_bin_capacity) = sJBar_new;
        //particle_bin.val(_17, pidib % g_bin_capacity) = masw_new; // water mass new
        particle_bin.val(_18, pidib % g_bin_capacity) = pw_new; // water pore pressure new
        particle_bin.val(_19, pidib % g_bin_capacity) = ID;
      }

      {
        contrib = (C * pbuffer.mass - contrib * newDt) * Dp_inv;
      }
    }

    local_base_index = (pos * g_dx_inv + 0.5f).cast<int>() - 1;
    {
      int direction_tag = dir_offset((base_index - 1) / g_blocksize -
                              (local_base_index - 1) / g_blocksize);
      if (g_buckets_on_particle_buffer)
        next_pbuffer.add_advection(partition, local_base_index - 1, direction_tag, pidib);
      else 
        partition.add_advection(local_base_index - 1, direction_tag, pidib);
    }

#pragma unroll 3
    for (char dd = 0; dd < 3; ++dd) {
      local_pos[dd] = pos[dd] - local_base_index[dd] * g_dx;
      PREC d =
          (local_pos[dd] - ((int)(local_pos[dd] * g_dx_inv + 0.5) - 1) * g_dx) *
          g_dx_inv;
      dws(dd, 0) = 0.5 * (1.5 - d) * (1.5 - d);
      d -= 1.0;
      dws(dd, 1) = 0.75 - d * d;
      d = 0.5 + d;
      dws(dd, 2) = 0.5 * d * d;

      local_base_index[dd] = (((base_index[dd] - 1) & g_blockmask) + 1) +
                             local_base_index[dd] - base_index[dd];
    }
 
    scale = pbuffer.length * pbuffer.length; //< Area scale (m^2) 
    PREC M_inv = g_D_inv / scale; //< Scalar 4/(dx^2) for Quad. B-Spline

#pragma unroll 3
    for (char i = 0; i < 3; i++)
#pragma unroll 3
      for (char j = 0; j < 3; j++)
#pragma unroll 3
        for (char k = 0; k < 3; k++) {
          pos = pvec3{(PREC)i, (PREC)j, (PREC)k} * g_dx - local_pos;
          PREC W = dws(0, i) * dws(1, j) * dws(2, k);
          PREC wm = pbuffer.mass * W;
          //PREC wmw = pbuffer.masw * W; // water mass
          PREC wmw = (1) * pbuffer.masw * W; // Water mass should be changing relative to deformation? JB
          PREC newDtKm = newDt * pbuffer.Kperm * pbuffer.masw;
          atomicAdd(
              &p2gbuffer[0][local_base_index[0] + i][local_base_index[1] + j]
                        [local_base_index[2] + k],
              wm); 

          // Velocity*mass         
          atomicAdd(
              &p2gbuffer[1][local_base_index[0] + i][local_base_index[1] + j]
                        [local_base_index[2] + k],
              wm * vel[0] + (contrib[0] * pos[0] + contrib[3] * pos[1] +
                             contrib[6] * pos[2]) * W);   
          atomicAdd(
              &p2gbuffer[2][local_base_index[0] + i][local_base_index[1] + j]
                        [local_base_index[2] + k],
              wm * vel[1] + (contrib[1] * pos[0] + contrib[4] * pos[1] +
                             contrib[7] * pos[2]) * W);
          atomicAdd(
              &p2gbuffer[3][local_base_index[0] + i][local_base_index[1] + j]
                        [local_base_index[2] + k],
              wm * vel[2] + (contrib[2] * pos[0] + contrib[5] * pos[1] +
                             contrib[8] * pos[2]) * W);

          // ASFLIP unstressed velocity
          atomicAdd(
              &p2gbuffer[4][local_base_index[0] + i][local_base_index[1] + j]
                        [local_base_index[2] + k],
              wm * (vel[0] + Dp_inv * (C[0] * pos[0] + C[3] * pos[1] + C[6] * pos[2])));
          atomicAdd(
              &p2gbuffer[5][local_base_index[0] + i][local_base_index[1] + j]
                        [local_base_index[2] + k],
              wm * (vel[1] + Dp_inv * (C[1] * pos[0] + C[4] * pos[1] + C[7] * pos[2])));
          atomicAdd(
              &p2gbuffer[6][local_base_index[0] + i][local_base_index[1] + j]
                        [local_base_index[2] + k],
              wm * (vel[2] + Dp_inv * (C[2] * pos[0] + C[5] * pos[1] + C[8] * pos[2])));

          // CoupledUP
          atomicAdd(
              &p2gbuffer[7][local_base_index[0] + i][local_base_index[1] + j]
                        [local_base_index[2] + k],
              wmw);
          
          atomicAdd(
              &p2gbuffer[8][local_base_index[0] + i][local_base_index[1] + j]
                        [local_base_index[2] + k],
              wmw * pbuffer.Q_inv * pw_new 
              - (newDtKm * W * (M_inv)*(M_inv) * (pos[0] * pos[0] + pos[1] * pos[1] + pos[2] * pos[2]))*pw_new 
              - newDt * pbuffer.alpha1 * M_inv * W * wmw * (pos[0] * vel[0] + pos[1] * vel[1] + pos[2] * vel[2]) );

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
    // MLS-MPM 
    if (channelid == 0) {
      atomicAdd(&next_grid.ch(_0, blockno).val_1d(_0, c), val);
    } else if (channelid == 1) {
      atomicAdd(&next_grid.ch(_0, blockno).val_1d(_1, c), val);
    } else if (channelid == 2) {
      atomicAdd(&next_grid.ch(_0, blockno).val_1d(_2, c), val);
    } else if (channelid == 3) {
      atomicAdd(&next_grid.ch(_0, blockno).val_1d(_3, c), val);
    } 
    
    // ASFLIP
    else if (channelid == 4) {
      atomicAdd(&next_grid.ch(_0, blockno).val_1d(_4, c), val);
    } else if (channelid == 5) {
      atomicAdd(&next_grid.ch(_0, blockno).val_1d(_5, c), val);
    } else if (channelid == 6) {
      atomicAdd(&next_grid.ch(_0, blockno).val_1d(_6, c), val);
    } 
    // CoupledUP
#if DEBUG_COUPLED_UP
    else if (channelid == 7) {
      atomicAdd(&next_grid.ch(_0, blockno).val_1d(_9, c), val);
    } else if (channelid == 8) {
      atomicAdd(&next_grid.ch(_0, blockno).val_1d(_10, c), val);
    }
#endif
  }
}

template <typename ParticleBuffer, typename Partition, typename Grid, typename VerticeArray>
__global__ void v2p2g(double dt, double newDt, const ivec3 *__restrict__ blocks,
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
__global__ void v2p2g(double dt, double newDt, const ivec3 *__restrict__ blocks,
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
    blockid = blocks[blockIdx.x]; // Halo block ID
    src_blockno = partition.query(blockid); // Halo block number
  } else { 
    if (partition._haloMarks[blockIdx.x]) return; // Return early if halo block
    blockid = partition._activeKeys[blockIdx.x]; // Non-halo block ID
    src_blockno = blockIdx.x; // Non-halo block number
  }
  if (src_blockno < 0) return; // Return early if negative block number
//  else if (src_blockno > blockcnt) return; // Return early if excessive block number
  int ppb = g_buckets_on_particle_buffer
            ? next_pbuffer._ppbs[src_blockno] 
            : partition._ppbs[src_blockno]; // Particles in block
  if (ppb == 0) return; // Return early if no particles


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
  for (int pidib = threadIdx.x; pidib < ppb; pidib += blockDim.x) {
    int source_blockno, source_pidib;
    ivec3 base_index;
    {
      int advect = (g_buckets_on_particle_buffer)
                    ? next_pbuffer._blockbuckets[src_blockno * g_particle_num_per_block + pidib]
                    : partition._blockbuckets[src_blockno * g_particle_num_per_block + pidib];
      dir_components(advect / g_particle_num_per_block, base_index);
      base_index += blockid;
      source_blockno = prev_partition.query(base_index);
      source_pidib = advect % g_particle_num_per_block; // & (g_particle_num_per_block - 1);
      source_blockno = (g_buckets_on_particle_buffer)
                        ? pbuffer._binsts[source_blockno] + source_pidib / g_bin_capacity
                        : prev_partition._binsts[source_blockno] + source_pidib / g_bin_capacity;
    }
    pvec3 pos;  //< Particle position at n
    int ID; // Vertice ID for mesh
    pvec3 vel_p; //< Particle vel. at n
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
      vel_p[0] = source_particle_bin.val(_4, source_pidib % g_bin_capacity); //< vx
      vel_p[1] = source_particle_bin.val(_5, source_pidib % g_bin_capacity); //< vy
      vel_p[2] = source_particle_bin.val(_6, source_pidib % g_bin_capacity); //< vz
      //beta = source_particle_bin.val(_7, source_pidib % g_bin_capacity); //< 
      b[0] = source_particle_bin.val(_8, source_pidib % g_bin_capacity); //< bx
      b[1] = source_particle_bin.val(_9, source_pidib % g_bin_capacity); //< by
      b[2] = source_particle_bin.val(_10, source_pidib % g_bin_capacity); //< bz
    }
    ivec3 local_base_index = (pos * g_dx_inv_d + 0.5).cast<int>() - 1;
    pvec3 local_pos = pos - local_base_index * g_dx_d;
    base_index = local_base_index;

    pvec3x3 dws;
#pragma unroll 3
    for (int dd = 0; dd < 3; ++dd) {
      PREC d = (local_pos[dd] - static_cast<PREC>(std::floor(local_pos[dd] * g_dx_inv_d + 0.5) - 1) * g_dx_d) * g_dx_inv_d;
      dws(dd, 0) = 0.5 * (1.5 - d) * (1.5 - d);
      d -= 1.0;
      dws(dd, 1) = 0.75 - d * d;
      d = 0.5 + d;
      dws(dd, 2) = 0.5 * d * d;
      local_base_index[dd] = ((local_base_index[dd] - 1) & g_blockmask) + 1;
    }
    pvec3 vel;   //< Stressed, collided grid velocity
    pvec3 vel_FLIP; //< Unstressed, uncollided grid velocity
    pvec9 C;     //< APIC affine matrix, used a few times
    vel.set(0.0); 
    vel_FLIP.set(0.0);
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
          vel_FLIP += vi_n * W; 
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
    pvec3 force; //< Internal force at FEM nodes
    force.set(0.0);
    PREC restVolume;
    PREC restMass;
    int count;
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
      force[0] = vertice_array.val(_7, ID);
      force[1] = vertice_array.val(_8, ID);
      force[2] = vertice_array.val(_9, ID);
      count = (int) vertice_array.val(_10, ID);

      J = J / restVolume;
      pressure = pressure / restVolume;
      von_mises = von_mises / restVolume;
      restMass =  pbuffer.rho * abs(restVolume);

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
    if (surface_ASFLIP && count > 10) {
      // Interior (ASFLIP)
      pos += dt * vel;
      vel += pbuffer.alpha * (vel_p - vel_FLIP);
    } else if (surface_ASFLIP && count <= 10) {
      // Surface (FLIP/PIC)
      pos += dt * (vel + beta * pbuffer.alpha * (vel_p - vel_FLIP));
      vel += pbuffer.alpha * (vel_p - vel_FLIP);
    } else {
      // Interior and Surface (ASFLIP)
      pos += dt * (vel + beta * pbuffer.alpha * (vel_p - vel_FLIP));
      vel += pbuffer.alpha * (vel_p - vel_FLIP);
    } 

    {
      {
        auto particle_bin = g_buckets_on_particle_buffer 
            ? next_pbuffer.ch(_0, next_pbuffer._binsts[src_blockno] + pidib / g_bin_capacity) 
            : next_pbuffer.ch(_0, partition._binsts[src_blockno] + pidib / g_bin_capacity);
        particle_bin.val(_0, pidib % g_bin_capacity) = pos[0]; //< x [ ]
        particle_bin.val(_1, pidib % g_bin_capacity) = pos[1]; //< y [ ]
        particle_bin.val(_2, pidib % g_bin_capacity) = pos[2]; //< z [ ]
        particle_bin.val(_3, pidib % g_bin_capacity) = (PREC)ID; //< ID
        particle_bin.val(_4, pidib % g_bin_capacity) = vel[0]; //< vx [ /s]
        particle_bin.val(_5, pidib % g_bin_capacity) = vel[1]; //< vy [ /s]
        particle_bin.val(_6, pidib % g_bin_capacity) = vel[2]; //< vz [ /s]
        particle_bin.val(_7, pidib % g_bin_capacity) = restVolume; // volume [m3]
        particle_bin.val(_8, pidib % g_bin_capacity)  = J; //force[0]; // bx
        particle_bin.val(_9, pidib % g_bin_capacity)  = pressure; //force[1]; // by
        particle_bin.val(_10, pidib % g_bin_capacity) = von_mises; //force[2]; // bz
      }
    }

    local_base_index = (pos * g_dx_inv + 0.5f).cast<int>() - 1;
    {
      int direction_tag = dir_offset((base_index - 1) / g_blocksize -
                              (local_base_index - 1) / g_blocksize);
      if (g_buckets_on_particle_buffer)
        next_pbuffer.add_advection(partition, local_base_index - 1, direction_tag, pidib);
      else 
        partition.add_advection(local_base_index - 1, direction_tag, pidib);
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
                             C[6] * pos[2])) - W * (force[0] * newDt));
          atomicAdd(
              &p2gbuffer[2][local_base_index[0] + i][local_base_index[1] + j]
                        [local_base_index[2] + k],
              wm * (vel[1] + Dp_inv * (C[1] * pos[0] + C[4] * pos[1] +
                             C[7] * pos[2])) - W * (force[1] * newDt));
          atomicAdd(
              &p2gbuffer[3][local_base_index[0] + i][local_base_index[1] + j]
                        [local_base_index[2] + k],
              wm * (vel[2] + Dp_inv * (C[2] * pos[0] + C[5] * pos[1] +
                             C[8] * pos[2])) - W * (force[2] * newDt));
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
__global__ void v2p2g_FBar(double dt, double newDt, const ivec3 *__restrict__ blocks,
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
__global__ void v2p2g_FBar(double dt, double newDt, const ivec3 *__restrict__ blocks,
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
    blockid = blocks[blockIdx.x]; // Halo block ID
    src_blockno = partition.query(blockid); // Halo block number
  } else { 
    if (partition._haloMarks[blockIdx.x]) return; // Return early if halo block
    blockid = partition._activeKeys[blockIdx.x]; // Non-halo block ID
    src_blockno = blockIdx.x; // Non-halo block number
  }
  if (src_blockno < 0) return; // Return early if negative block number
//  else if (src_blockno > blockcnt) return; // Return early if excessive block number
  int ppb = g_buckets_on_particle_buffer
            ? next_pbuffer._ppbs[src_blockno] 
            : partition._ppbs[src_blockno]; // Particles in block
  if (ppb == 0) return; // Return early if no particles


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
  for (int pidib = threadIdx.x; pidib < ppb; pidib += blockDim.x) {
    int source_blockno, source_pidib;
    ivec3 base_index;
    {
      int advect = (g_buckets_on_particle_buffer)
                    ? next_pbuffer._blockbuckets[src_blockno * g_particle_num_per_block + pidib]
                    : partition._blockbuckets[src_blockno * g_particle_num_per_block + pidib];
      dir_components(advect / g_particle_num_per_block, base_index);
      base_index += blockid;
      source_blockno = prev_partition.query(base_index);
      source_pidib = advect % g_particle_num_per_block; // & (g_particle_num_per_block - 1);
      source_blockno = (g_buckets_on_particle_buffer)
                        ? pbuffer._binsts[source_blockno] + source_pidib / g_bin_capacity
                        : prev_partition._binsts[source_blockno] + source_pidib / g_bin_capacity;
    }
    int ID; // Vertice ID for mesh
    pvec3 pos;  //< Particle position at n
    pvec3 vel_p; //< Particle vel. at n
    pvec3 b; //< Normal at vertice
    PREC beta = 0; //< ASFLIP beta factor
    //float J;   //< Particle volume ratio at n
    {
      auto source_particle_bin = pbuffer.ch(_0, source_blockno);
      pos[0] = source_particle_bin.val(_0, source_pidib % g_bin_capacity);  //< x
      pos[1] = source_particle_bin.val(_1, source_pidib % g_bin_capacity);  //< y
      pos[2] = source_particle_bin.val(_2, source_pidib % g_bin_capacity);  //< z
      ID = (int)source_particle_bin.val(_3, source_pidib % g_bin_capacity); //< ID
      vel_p[0] = source_particle_bin.val(_4, source_pidib % g_bin_capacity); //< vx
      vel_p[1] = source_particle_bin.val(_5, source_pidib % g_bin_capacity); //< vy
      vel_p[2] = source_particle_bin.val(_6, source_pidib % g_bin_capacity); //< vz
      //beta = source_particle_bin.val(_7, source_pidib % g_bin_capacity); //< 
      b[0] = source_particle_bin.val(_8, source_pidib % g_bin_capacity); //< bx
      b[1] = source_particle_bin.val(_9, source_pidib % g_bin_capacity); //< by
      b[2] = source_particle_bin.val(_10, source_pidib % g_bin_capacity); //< bz
    }
    ivec3 local_base_index = (pos * g_dx_inv_d + 0.5).cast<int>() - 1;
    pvec3 local_pos = pos - local_base_index * g_dx_d;
    base_index = local_base_index;

    pvec3x3 dws;
#pragma unroll 3
    for (int dd = 0; dd < 3; ++dd) {
      PREC d = (local_pos[dd] - static_cast<PREC>(std::floor(local_pos[dd] * g_dx_inv_d + 0.5) - 1) * g_dx_d) * g_dx_inv_d;
      dws(dd, 0) = 0.5 * (1.5 - d) * (1.5 - d);
      d -= 1.0;
      dws(dd, 1) = 0.75 - d * d;
      d = 0.5 + d;
      dws(dd, 2) = 0.5 * d * d;
      local_base_index[dd] = ((local_base_index[dd] - 1) & g_blockmask) + 1;
    }
    pvec3 vel;   //< Stressed, collided grid velocity
    pvec3 vel_FLIP; //< Unstressed, uncollided grid velocity
    pvec9 C;     //< APIC affine matrix, used a few times
    vel.set(0.0); 
    vel_FLIP.set(0.0);
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
          vel_FLIP += vi_n * W; 
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

    pvec3 force; //< Internal force at FEM nodes
    force.set(0.0);
    PREC restVolume;
    PREC restMass;
    int count;
    PREC Vn, JBar;
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
      force[0] = vertice_array.val(_7, ID);
      force[1] = vertice_array.val(_8, ID);
      force[2] = vertice_array.val(_9, ID);
      count = (int) vertice_array.val(_10, ID);
      Vn = vertice_array.val(_11, ID);
      JBar = vertice_array.val(_12, ID);

      restMass =  pbuffer.rho * abs(restVolume);
      
      J = J / restVolume;
      JBar = JBar / Vn;
      pressure = pressure / restVolume;
      von_mises = von_mises / restVolume;

      // PREC bMag = sqrt(b[0]*b[0] + b[1]*b[1] + b[2]*b[2]);
      // if (bMag < 0.00001) b.set(0.0);
      // else for (int d = 0; d < 3; d++) b[d] = b[d]/bMag;

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
    beta = 0;
    if (surface_ASFLIP && count > 10) {
      // Interior (ASFLIP)
      pos += dt * vel;
      vel += pbuffer.alpha * (vel_p - vel_FLIP);
    } else if (surface_ASFLIP && count <= 10) {
      // Surface (FLIP/PIC)
      pos += dt * (vel + beta * pbuffer.alpha * (vel_p - vel_FLIP));
      vel += pbuffer.alpha * (vel_p - vel_FLIP);
    } else {
      // Interior and Surface (ASFLIP)
      pos += dt * (vel + beta * pbuffer.alpha * (vel_p - vel_FLIP));
      vel += pbuffer.alpha * (vel_p - vel_FLIP);
    } 

    {
      {
        auto particle_bin = g_buckets_on_particle_buffer 
            ? next_pbuffer.ch(_0, next_pbuffer._binsts[src_blockno] + pidib / g_bin_capacity) 
            : next_pbuffer.ch(_0, partition._binsts[src_blockno] + pidib / g_bin_capacity);
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
      int direction_tag = dir_offset((base_index - 1) / g_blocksize -
                              (local_base_index - 1) / g_blocksize);
      if (g_buckets_on_particle_buffer)
        next_pbuffer.add_advection(partition, local_base_index - 1, direction_tag, pidib);
      else 
        partition.add_advection(local_base_index - 1, direction_tag, pidib);
    }

#pragma unroll 3
    for (char dd = 0; dd < 3; ++dd) {
      local_pos[dd] = pos[dd] - local_base_index[dd] * g_dx;
      PREC d =
          (local_pos[dd] - ((int)(local_pos[dd] * g_dx_inv + 0.5) - 1) * g_dx) *
          g_dx_inv;
      dws(dd, 0) = 0.5 * (1.5 - d) * (1.5 - d);
      d -= 1.0;
      dws(dd, 1) = 0.75 - d * d;
      d = 0.5 + d;
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
                             C[6] * pos[2])) - W * (force[0] * newDt));
          atomicAdd(
              &p2gbuffer[2][local_base_index[0] + i][local_base_index[1] + j]
                        [local_base_index[2] + k],
              wm * (vel[1] + Dp_inv * (C[1] * pos[0] + C[4] * pos[1] +
                             C[7] * pos[2])) - W * (force[1] * newDt));
          atomicAdd(
              &p2gbuffer[3][local_base_index[0] + i][local_base_index[1] + j]
                        [local_base_index[2] + k],
              wm * (vel[2] + Dp_inv * (C[2] * pos[0] + C[5] * pos[1] +
                             C[8] * pos[2])) - W * (force[2] * newDt));
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

// MGSP
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


// // GMPM
// template <typename Grid>
// __global__ void mark_active_grid_blocks(uint32_t blockCount, const Grid grid,
//                                         int *_marks) {
//   auto idx = blockIdx.x * blockDim.x + threadIdx.x;
//   int blockno = idx / g_blockvolume, cellno = idx % g_blockvolume;
//   if (blockno >= blockCount)
//     return;
//   if (grid.ch(_0, blockno).val_1d(_0, cellno) != 0.f)
//     _marks[blockno] = 1;
// }

__global__ void mark_active_particle_blocks(uint32_t blockCount,
                                            const int *__restrict__ _ppbs,
                                            int *_marks) {
  std::size_t blockno = blockIdx.x * blockDim.x + threadIdx.x;
  if (blockno >= blockCount)
    return;
  if (g_buckets_on_particle_buffer){ // GMPM ParticlBins Buckets
    if (_ppbs[blockno] > 0)
      _marks[blockno] = 1;
  } else { // MGSP Partition Buckets
    _marks[blockno] = _ppbs[blockno] > 0 ? 1 : 0; // Wouldn't this not work for multiple models calling kernel? Maybe use OR?
  }
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

template <typename Partition>
__global__ void
update_partition(uint32_t blockCount, const int *__restrict__ _sourceNos,
                 const Partition partition, Partition next_partition) {
  __shared__ std::size_t sourceNo[1];
  std::size_t blockno = (g_buckets_on_particle_buffer) 
                        ? blockIdx.x * blockDim.x + threadIdx.x
                        : blockIdx.x; //recheck..
  if (blockno >= blockCount)
    return;

  if (g_buckets_on_particle_buffer) {
    //uint32_t sourceNo = _sourceNos[blockno];
    auto sourceBlockid = partition._activeKeys[(uint32_t)_sourceNos[blockno]];
    next_partition._activeKeys[blockno] = sourceBlockid;
    next_partition.reinsert(blockno);
  }
  else{
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
    // ! This shouldnt cause warp divergence, but double-check
  }
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
#if DEBUG_COUPLED_UP
    // CoupledUP: Mass and pressure of fluid
    targetblock.val_1d(_9, threadIdx.x) = sourceblock.val_1d(_9, threadIdx.x); // Volume
    targetblock.val_1d(_10, threadIdx.x) = sourceblock.val_1d(_10, threadIdx.x); // JBar
#endif
  }
}


template <typename Partition>
__global__ void check_table(uint32_t blockCount, Partition partition) {
  uint32_t blockno = blockIdx.x * blockDim.x + threadIdx.x;
  if (blockno >= blockCount) return;
  auto blockid = partition._activeKeys[blockno];
  if (partition.query(blockid) != blockno)
    printf("ERROR: Partition hash-table is wrong! Block 3D ID and block 1D number not related properly.\n");
}
template <typename Grid> __global__ void sum_grid_mass(Grid grid, PREC_G *sum) {
  atomicAdd(sum, (float)grid.ch(_0, blockIdx.x).val_1d(_0, threadIdx.x));
}
// Added for Simple FBar Method
template <typename Grid> __global__ void sum_grid_volume(Grid grid, PREC_G *sum) {
  atomicAdd(sum, (PREC_G)grid.ch(_0, blockIdx.x).val_1d(_7, threadIdx.x));
}
__global__ void sum_particle_count(uint32_t count, int *__restrict__ _cnts, int *sum) {
  auto idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= count) return;
  atomicAdd(sum, _cnts[idx]);
}

template <typename Partition>
__global__ void check_partition(uint32_t blockCount, Partition partition) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= blockCount)
    return;
  ivec3 blockid = partition._activeKeys[idx];
  if (blockid[0] == 0 || blockid[1] == 0 || blockid[2] == 0)
    printf("\tERROR: Encountered zero block record\n");
  if (partition.query(blockid) != idx) {
    int id = partition.query(blockid);
    ivec3 bid = partition._activeKeys[id];
    printf("\t\tERROR: Check partition index[%d], produced block ID (%d, %d, %d), produced feedback index[%d], produced feedback block ID (%d, %d, "
           "%d). Feedback and originals should be identical if partition query is working.\n",
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
        "Partition block[%d] (%d, %d, %d) in domain[%d], mins-maxs: (%d, %d, %d)-(%d, %d, %d)\n",
        idx, blockid[0], blockid[1], blockid[2], did, domain._min[0],
        domain._min[1], domain._min[2], domain._max[0], domain._max[1],
        domain._max[2]);
  }
}


template<num_attribs_e N, typename I, typename T>
__device__ void setParticleAttrib(ParticleAttrib<N> pattrib, I i, T parid, PREC val)
{ }
template<typename I, typename T>
__device__ void setParticleAttrib(ParticleAttrib<num_attribs_e::Zero> pattrib, I i, T parid, PREC val) { }
template<typename I, typename T>
__device__ void setParticleAttrib(ParticleAttrib<num_attribs_e::One> pattrib, I i, T parid, PREC val)
{
  if      (i == 0) pattrib.val(_0, parid) = val; 
}
template<typename I, typename T>
__device__ void setParticleAttrib(ParticleAttrib<num_attribs_e::Two> pattrib, I i, T parid, PREC val)
{
  if      (i == 0) pattrib.val(_0, parid) = val; 
  else if (i == 1) pattrib.val(_1, parid) = val; 
}

template<typename I, typename T>
__device__ void setParticleAttrib(ParticleAttrib<num_attribs_e::Three> pattrib, I i, T parid, PREC val)
{
  if      (i == 0) pattrib.val(_0, parid) = val; 
  else if (i == 1) pattrib.val(_1, parid) = val; 
  else if (i == 2) pattrib.val(_2, parid) = val; 
}

template<typename I, typename T>
__device__ void setParticleAttrib(ParticleAttrib<num_attribs_e::Four> pattrib, I i, T parid, PREC val)
{
  if      (i == 0) pattrib.val(_0, parid) = val; 
  else if (i == 1) pattrib.val(_1, parid) = val; 
  else if (i == 2) pattrib.val(_2, parid) = val; 
  else if (i == 3) pattrib.val(_3, parid) = val; 
}

template<typename I, typename T>
__device__ void setParticleAttrib(ParticleAttrib<num_attribs_e::Five> pattrib, I i, T parid, PREC val)
{
  if      (i == 0) pattrib.val(_0, parid) = val; 
  else if (i == 1) pattrib.val(_1, parid) = val; 
  else if (i == 2) pattrib.val(_2, parid) = val; 
  else if (i == 3) pattrib.val(_3, parid) = val; 
  else if (i == 4) pattrib.val(_4, parid) = val; 
}

template<typename I, typename T>
__device__ void setParticleAttrib(ParticleAttrib<num_attribs_e::Six> pattrib, I i, T parid, PREC val)
{
  if      (i == 0) pattrib.val(_0, parid) = val; 
  else if (i == 1) pattrib.val(_1, parid) = val; 
  else if (i == 2) pattrib.val(_2, parid) = val; 
  else if (i == 3) pattrib.val(_3, parid) = val; 
  else if (i == 4) pattrib.val(_4, parid) = val; 
  else if (i == 5) pattrib.val(_5, parid) = val; 
}

template<typename I, typename T>
__device__ void setParticleAttrib(ParticleAttrib<num_attribs_e::Seven> pattrib, I i, T parid, PREC val)
{
  if      (i == 0) pattrib.val(_0, parid) = val; 
  else if (i == 1) pattrib.val(_1, parid) = val; 
  else if (i == 2) pattrib.val(_2, parid) = val; 
  else if (i == 3) pattrib.val(_3, parid) = val; 
  else if (i == 4) pattrib.val(_4, parid) = val; 
  else if (i == 5) pattrib.val(_5, parid) = val; 
  else if (i == 6) pattrib.val(_6, parid) = val; 
}

template<typename I, typename T>
__device__ void setParticleAttrib(ParticleAttrib<num_attribs_e::Eight> pattrib, I i, T parid, PREC val)
{
  if      (i == 0) pattrib.val(_0, parid) = val; 
  else if (i == 1) pattrib.val(_1, parid) = val; 
  else if (i == 2) pattrib.val(_2, parid) = val; 
  else if (i == 3) pattrib.val(_3, parid) = val; 
  else if (i == 4) pattrib.val(_4, parid) = val; 
  else if (i == 5) pattrib.val(_5, parid) = val; 
  else if (i == 6) pattrib.val(_6, parid) = val; 
  else if (i == 7) pattrib.val(_7, parid) = val; 
}


template<typename I, typename T>
__device__ void setParticleAttrib(ParticleAttrib<num_attribs_e::Nine> pattrib, I i, T parid, PREC val)
{
  if      (i == 0) pattrib.val(_0, parid) = val; 
  else if (i == 1) pattrib.val(_1, parid) = val; 
  else if (i == 2) pattrib.val(_2, parid) = val; 
  else if (i == 3) pattrib.val(_3, parid) = val; 
  else if (i == 4) pattrib.val(_4, parid) = val; 
  else if (i == 5) pattrib.val(_5, parid) = val; 
  else if (i == 6) pattrib.val(_6, parid) = val; 
  else if (i == 7) pattrib.val(_7, parid) = val; 
  else if (i == 8) pattrib.val(_8, parid) = val; 
}


// template<typename I, typename T>
// __device__ void setParticleAttrib(ParticleAttrib<num_attribs_e::Ten> pattrib, I i, T parid, PREC val)
// {
//   if      (i == 0) pattrib.val(_0, parid) = val; 
//   else if (i == 1) pattrib.val(_1, parid) = val; 
//   else if (i == 2) pattrib.val(_2, parid) = val; 
//   else if (i == 3) pattrib.val(_3, parid) = val; 
//   else if (i == 4) pattrib.val(_4, parid) = val; 
//   else if (i == 5) pattrib.val(_5, parid) = val; 
//   else if (i == 6) pattrib.val(_6, parid) = val; 
//   else if (i == 7) pattrib.val(_7, parid) = val; 
//   else if (i == 8) pattrib.val(_8, parid) = val; 
//   else if (i == 9) pattrib.val(_9, parid) = val; 
// }

// template<typename I, typename T>
// __device__ void setParticleAttrib(ParticleAttrib<num_attribs_e::Eleven> pattrib, I i, T parid, PREC val)
// {
//   if      (i == 0) pattrib.val(_0, parid) = val; 
//   else if (i == 1) pattrib.val(_1, parid) = val; 
//   else if (i == 2) pattrib.val(_2, parid) = val; 
//   else if (i == 3) pattrib.val(_3, parid) = val; 
//   else if (i == 4) pattrib.val(_4, parid) = val; 
//   else if (i == 5) pattrib.val(_5, parid) = val; 
//   else if (i == 6) pattrib.val(_6, parid) = val; 
//   else if (i == 7) pattrib.val(_7, parid) = val; 
//   else if (i == 8) pattrib.val(_8, parid) = val; 
//   else if (i == 9) pattrib.val(_9, parid) = val; 
//   else if (i == 11) pattrib.val(_10, parid) = val; 
// }

// template<typename I, typename T>
// __device__ void setParticleAttrib(ParticleAttrib<num_attribs_e::Twelve> pattrib, I i, T parid, PREC val)
// {
//   if      (i == 0) pattrib.val(_0, parid) = val; 
//   else if (i == 1) pattrib.val(_1, parid) = val; 
//   else if (i == 2) pattrib.val(_2, parid) = val; 
//   else if (i == 3) pattrib.val(_3, parid) = val; 
//   else if (i == 4) pattrib.val(_4, parid) = val; 
//   else if (i == 5) pattrib.val(_5, parid) = val; 
//   else if (i == 6) pattrib.val(_6, parid) = val; 
//   else if (i == 7) pattrib.val(_7, parid) = val; 
//   else if (i == 8) pattrib.val(_8, parid) = val; 
//   else if (i == 9) pattrib.val(_9, parid) = val; 
//   else if (i == 10) pattrib.val(_10, parid) = val; 
//   else if (i == 11) pattrib.val(_11, parid) = val; 
// }
// template<typename I, typename T>
// __device__ void setParticleAttrib(ParticleAttrib<num_attribs_e::Thirteen> pattrib, I i, T parid, PREC val)
// {
//   if      (i == 0) pattrib.val(_0, parid) = val; 
//   else if (i == 1) pattrib.val(_1, parid) = val; 
//   else if (i == 2) pattrib.val(_2, parid) = val; 
//   else if (i == 3) pattrib.val(_3, parid) = val; 
//   else if (i == 4) pattrib.val(_4, parid) = val; 
//   else if (i == 5) pattrib.val(_5, parid) = val; 
//   else if (i == 6) pattrib.val(_6, parid) = val; 
//   else if (i == 7) pattrib.val(_7, parid) = val; 
//   else if (i == 8) pattrib.val(_8, parid) = val; 
//   else if (i == 9) pattrib.val(_9, parid) = val; 
//   else if (i == 10) pattrib.val(_10, parid) = val; 
//   else if (i == 11) pattrib.val(_11, parid) = val; 
//   else if (i == 12) pattrib.val(_12, parid) = val; 
// }
// template<typename I, typename T>
// __device__ void setParticleAttrib(ParticleAttrib<15> pattrib, I i, T parid, PREC val)
// {
//   if      (i == 0) pattrib.val(_0, parid) = val; 
//   else if (i == 1) pattrib.val(_1, parid) = val; 
//   else if (i == 2) pattrib.val(_2, parid) = val; 
//   else if (i == 3) pattrib.val(_3, parid) = val; 
//   else if (i == 4) pattrib.val(_4, parid) = val; 
//   else if (i == 5) pattrib.val(_5, parid) = val; 
//   else if (i == 6) pattrib.val(_6, parid) = val; 
//   else if (i == 7) pattrib.val(_7, parid) = val; 
//   else if (i == 8) pattrib.val(_8, parid) = val; 
//   else if (i == 9) pattrib.val(_9, parid) = val; 
//   else if (i == 10) pattrib.val(_10, parid) = val; 
//   else if (i == 11) pattrib.val(_11, parid) = val; 
//   else if (i == 12) pattrib.val(_12, parid) = val; 
//   else if (i == 13) pattrib.val(_13, parid) = val; 
//   else if (i == 14) pattrib.val(_14, parid) = val; 
// }
// template<typename I, typename T>
// __device__ void setParticleAttrib(ParticleAttrib<24> pattrib, I i, T parid, PREC val)
// {
//   if      (i == 0) pattrib.val(_0, parid) = val; 
//   else if (i == 1) pattrib.val(_1, parid) = val; 
//   else if (i == 2) pattrib.val(_2, parid) = val; 
//   else if (i == 3) pattrib.val(_3, parid) = val; 
//   else if (i == 4) pattrib.val(_4, parid) = val; 
//   else if (i == 5) pattrib.val(_5, parid) = val; 
//   else if (i == 6) pattrib.val(_6, parid) = val; 
//   else if (i == 7) pattrib.val(_7, parid) = val; 
//   else if (i == 8) pattrib.val(_8, parid) = val; 
//   else if (i == 9) pattrib.val(_9, parid) = val; 
//   else if (i == 10) pattrib.val(_10, parid) = val; 
//   else if (i == 11) pattrib.val(_11, parid) = val; 
//   else if (i == 12) pattrib.val(_12, parid) = val; 
//   else if (i == 13) pattrib.val(_13, parid) = val; 
//   else if (i == 14) pattrib.val(_14, parid) = val; 
//   else if (i == 15) pattrib.val(_15, parid) = val; 
//   else if (i == 16) pattrib.val(_16, parid) = val; 
//   else if (i == 17) pattrib.val(_17, parid) = val; 
//   else if (i == 18) pattrib.val(_18, parid) = val; 
//   else if (i == 19) pattrib.val(_19, parid) = val; 
//   else if (i == 20) pattrib.val(_20, parid) = val; 
//   else if (i == 21) pattrib.val(_21, parid) = val; 
//   else if (i == 22) pattrib.val(_22, parid) = val; 
//   else if (i == 23) pattrib.val(_23, parid) = val; 
// }


template<num_attribs_e N, typename I, typename T>
__device__ void getParticleAttrib(ParticleAttrib<N> pattrib, I i, T parid, PREC& val)
{ }
template<typename I, typename T>
__device__ void getParticleAttrib(ParticleAttrib<num_attribs_e::Zero> pattrib, I i, T parid, PREC&val) { }
template<typename I, typename T>
__device__ void getParticleAttrib(ParticleAttrib<num_attribs_e::One> pattrib, I i, T parid, PREC&val)
{
  if      (i == 0) val = pattrib.val(_0, parid) ; 
}
template<typename I, typename T>
__device__ void getParticleAttrib(ParticleAttrib<num_attribs_e::Two> pattrib, I i, T parid, PREC&val)
{
  if      (i == 0) val = pattrib.val(_0, parid) ; 
  else if (i == 1) val = pattrib.val(_1, parid) ; 
}
template<typename I, typename T>
__device__ void getParticleAttrib(ParticleAttrib<num_attribs_e::Three> pattrib, I i, T parid, PREC&val)
{
  if      (i == 0) val = pattrib.val(_0, parid) ; 
  else if (i == 1) val = pattrib.val(_1, parid) ; 
  else if (i == 2) val = pattrib.val(_2, parid) ; 
}

template<typename I, typename T>
__device__ void getParticleAttrib(ParticleAttrib<num_attribs_e::Four> pattrib, I i, T parid, PREC&val)
{
  if      (i == 0) val = pattrib.val(_0, parid) ; 
  else if (i == 1) val = pattrib.val(_1, parid) ; 
  else if (i == 2) val = pattrib.val(_2, parid) ; 
  else if (i == 3) val = pattrib.val(_3, parid) ; 
}

template<typename I, typename T>
__device__ void getParticleAttrib(ParticleAttrib<num_attribs_e::Five> pattrib, I i, T parid, PREC&val)
{
  if      (i == 0) val = pattrib.val(_0, parid) ; 
  else if (i == 1) val = pattrib.val(_1, parid) ; 
  else if (i == 2) val = pattrib.val(_2, parid) ; 
  else if (i == 3) val = pattrib.val(_3, parid) ; 
  else if (i == 4) val = pattrib.val(_4, parid) ; 
}

template<typename I, typename T>
__device__ void getParticleAttrib(ParticleAttrib<num_attribs_e::Six> pattrib, I i, T parid, PREC& val)
{
  if      (i == 0) val = pattrib.val(_0, parid) ; 
  else if (i == 1) val = pattrib.val(_1, parid) ; 
  else if (i == 2) val = pattrib.val(_2, parid) ; 
  else if (i == 3) val = pattrib.val(_3, parid) ; 
  else if (i == 4) val = pattrib.val(_4, parid) ; 
  else if (i == 5) val = pattrib.val(_5, parid) ; 
}

template<typename I, typename T>
__device__ void getParticleAttrib(ParticleAttrib<num_attribs_e::Seven> pattrib, I i, T parid, PREC& val)
{
  if      (i == 0) val = pattrib.val(_0, parid) ; 
  else if (i == 1) val = pattrib.val(_1, parid) ; 
  else if (i == 2) val = pattrib.val(_2, parid) ; 
  else if (i == 3) val = pattrib.val(_3, parid) ; 
  else if (i == 4) val = pattrib.val(_4, parid) ; 
  else if (i == 5) val = pattrib.val(_5, parid) ; 
  else if (i == 6) val = pattrib.val(_6, parid) ; 
}

template<typename I, typename T>
__device__ void getParticleAttrib(ParticleAttrib<num_attribs_e::Eight> pattrib, I i, T parid, PREC& val)
{
  if      (i == 0) val = pattrib.val(_0, parid) ; 
  else if (i == 1) val = pattrib.val(_1, parid) ; 
  else if (i == 2) val = pattrib.val(_2, parid) ; 
  else if (i == 3) val = pattrib.val(_3, parid) ; 
  else if (i == 4) val = pattrib.val(_4, parid) ; 
  else if (i == 5) val = pattrib.val(_5, parid) ; 
  else if (i == 6) val = pattrib.val(_6, parid) ; 
  else if (i == 7) val = pattrib.val(_7, parid) ; 
}

template<typename I, typename T>
__device__ void getParticleAttrib(ParticleAttrib<num_attribs_e::Nine> pattrib, I i, T parid, PREC& val)
{
  if      (i == 0) val = pattrib.val(_0, parid) ; 
  else if (i == 1) val = pattrib.val(_1, parid) ; 
  else if (i == 2) val = pattrib.val(_2, parid) ; 
  else if (i == 3) val = pattrib.val(_3, parid) ; 
  else if (i == 4) val = pattrib.val(_4, parid) ; 
  else if (i == 5) val = pattrib.val(_5, parid) ; 
  else if (i == 6) val = pattrib.val(_6, parid) ; 
  else if (i == 7) val = pattrib.val(_7, parid) ; 
  else if (i == 8) val = pattrib.val(_8, parid) ; 
}

// template<typename I, typename T>
// __device__ void getParticleAttrib(ParticleAttrib<num_attribs_e::Ten> pattrib, I i, T parid, PREC& val)
// {
//   if      (i == 0) val = pattrib.val(_0, parid) ; 
//   else if (i == 1) val = pattrib.val(_1, parid) ; 
//   else if (i == 2) val = pattrib.val(_2, parid) ; 
//   else if (i == 3) val = pattrib.val(_3, parid) ; 
//   else if (i == 4) val = pattrib.val(_4, parid) ; 
//   else if (i == 5) val = pattrib.val(_5, parid) ; 
//   else if (i == 6) val = pattrib.val(_6, parid) ; 
//   else if (i == 7) val = pattrib.val(_7, parid) ; 
//   else if (i == 8) val = pattrib.val(_8, parid) ; 
//   else if (i == 9) val = pattrib.val(_9, parid) ; 
// }

// template<typename I, typename T>
// __device__ void getParticleAttrib(ParticleAttrib<num_attribs_e::Eleven> pattrib, I i, T parid, PREC& val)
// {
//   if      (i == 0) val = pattrib.val(_0, parid) ; 
//   else if (i == 1) val = pattrib.val(_1, parid) ; 
//   else if (i == 2) val = pattrib.val(_2, parid) ; 
//   else if (i == 3) val = pattrib.val(_3, parid) ; 
//   else if (i == 4) val = pattrib.val(_4, parid) ; 
//   else if (i == 5) val = pattrib.val(_5, parid) ; 
//   else if (i == 6) val = pattrib.val(_6, parid) ; 
//   else if (i == 7) val = pattrib.val(_7, parid) ; 
//   else if (i == 8) val = pattrib.val(_8, parid) ; 
//   else if (i == 9) val = pattrib.val(_9, parid) ; 
//   else if (i == 10) val = pattrib.val(_10, parid) ; 
// }

// template<typename I, typename T>
// __device__ void getParticleAttrib(ParticleAttrib<num_attribs_e::Twelve> pattrib, I i, T parid, PREC& val)
// {
//   if      (i == 0) val = pattrib.val(_0, parid) ; 
//   else if (i == 1) val = pattrib.val(_1, parid) ; 
//   else if (i == 2) val = pattrib.val(_2, parid) ; 
//   else if (i == 3) val = pattrib.val(_3, parid) ; 
//   else if (i == 4) val = pattrib.val(_4, parid) ; 
//   else if (i == 5) val = pattrib.val(_5, parid) ; 
//   else if (i == 6) val = pattrib.val(_6, parid) ; 
//   else if (i == 7) val = pattrib.val(_7, parid) ; 
//   else if (i == 8) val = pattrib.val(_8, parid) ; 
//   else if (i == 9) val = pattrib.val(_9, parid) ; 
//   else if (i == 10) val = pattrib.val(_10, parid) ; 
//   else if (i == 11) val = pattrib.val(_11, parid) ; 
// }

// template<typename I, typename T>
// __device__ void getParticleAttrib(ParticleAttrib<num_attribs_e::Thirteen> pattrib, I i, T parid, PREC& val)
// {
//   if      (i == 0) val = pattrib.val(_0, parid) ; 
//   else if (i == 1) val = pattrib.val(_1, parid) ; 
//   else if (i == 2) val = pattrib.val(_2, parid) ; 
//   else if (i == 3) val = pattrib.val(_3, parid) ; 
//   else if (i == 4) val = pattrib.val(_4, parid) ; 
//   else if (i == 5) val = pattrib.val(_5, parid) ;
//   else if (i == 6) val = pattrib.val(_6, parid) ; 
//   else if (i == 7) val = pattrib.val(_7, parid) ; 
//   else if (i == 8) val = pattrib.val(_8, parid) ; 
//   else if (i == 9) val = pattrib.val(_9, parid) ; 
//   else if (i == 10) val = pattrib.val(_10, parid) ; 
//   else if (i == 11) val = pattrib.val(_11, parid) ; 
//   else if (i == 12) val = pattrib.val(_12, parid) ; 
// }

template <typename ParticleBuffer, typename T, typename I>
__device__ void caseSwitch_ParticleAttrib(ParticleBuffer pbuffer, T _source_bin, T _source_pidib, I idx, PREC& val) { }

template <material_e mt, typename T, typename I>
__device__ void caseSwitch_ParticleAttrib(ParticleBuffer<mt>& pbuffer, T _source_bin, T _source_pidib, I idx, PREC& val) {
    using attribs_e_ = typename ParticleBuffer<mt>::attribs_e;
    using output_e_ =  particle_output_attribs_e;
    PREC o = g_offset;
    PREC l = pbuffer.length;
    pvec9 F; 
    F.set(0.); 
    pvec3 Principals, Invariants; 
    Principals.set(0.); Invariants.set(0.);
    pbuffer.getDefGrad(_source_bin, _source_pidib, F.data());
    PREC J = matrixDeterminant3d(F.data());
    PREC sJBar = pbuffer.getAttribute<attribs_e_::JBar>(_source_bin, _source_pidib);
    switch (idx) {
      case output_e_::ID:
        val = pbuffer.getAttribute<attribs_e_::ID>(_source_bin, _source_pidib); return; // ID 
      case output_e_::Mass:
        val = pbuffer.mass; return; // Mass [kg]
      case output_e_::Volume:
        val = pbuffer.volume * pbuffer.getAttribute<attribs_e_::J>(_source_bin, _source_pidib); return; // Volume [m^3]
      case output_e_::Position_X:
        val = (pbuffer.getAttribute<attribs_e_::Position_X>(_source_bin, _source_pidib) - o) * l; return; // Position_X [m]
      case output_e_::Position_Y:
        val = (pbuffer.getAttribute<attribs_e_::Position_Y>(_source_bin, _source_pidib) - o) * l; return; // Position_Y [m]
      case output_e_::Position_Z:
        val = (pbuffer.getAttribute<attribs_e_::Position_Z>(_source_bin, _source_pidib) - o) * l; return; // Position_Z [m]
      case output_e_::Velocity_X:
        val = pbuffer.getAttribute<attribs_e_::Velocity_X>(_source_bin, _source_pidib) * l; return; // Velocity_X [m/s]
      case output_e_::Velocity_Y:
        val = pbuffer.getAttribute<attribs_e_::Velocity_Y>(_source_bin, _source_pidib) * l; return; // Velocity_Y [m/s]
      case output_e_::Velocity_Z:
        val = pbuffer.getAttribute<attribs_e_::Velocity_Z>(_source_bin, _source_pidib) * l; return; // Velocity_Z [m/s]
      case output_e_::Velocity_Magnitude:
        val = pbuffer.getAttribute<attribs_e_::Velocity_X>(_source_bin, _source_pidib) * l * pbuffer.getAttribute<attribs_e_::Velocity_X>(_source_bin, _source_pidib) * l; 
        val += pbuffer.getAttribute<attribs_e_::Velocity_Y>(_source_bin, _source_pidib) * l * pbuffer.getAttribute<attribs_e_::Velocity_Y>(_source_bin, _source_pidib) * l;
        val += pbuffer.getAttribute<attribs_e_::Velocity_Z>(_source_bin, _source_pidib) * l * pbuffer.getAttribute<attribs_e_::Velocity_Z>(_source_bin, _source_pidib) * l; 
        val = sqrt(val);
        return; // Velocity_Magnitude [m/s]
      case output_e_::DefGrad_XX:
        val = F[0]; return; // DefGrad_XX
      case output_e_::DefGrad_XY:
        val = F[1]; return; // DefGrad_XY
      case output_e_::DefGrad_XZ:
        val = F[2]; return; // DefGrad_XZ
      case output_e_::DefGrad_YX:
        val = F[3]; return; // DefGrad_YX
      case output_e_::DefGrad_YY:
        val = F[4]; return; // DefGrad_YY
      case output_e_::DefGrad_YZ:
        val = F[5]; return; // DefGrad_YZ
      case output_e_::DefGrad_ZX:
        val = F[6]; return; // DefGrad_ZX
      case output_e_::DefGrad_ZY:
        val = F[7]; return; // DefGrad_ZY
      case output_e_::DefGrad_ZZ:
        val = F[8]; return; // DefGrad_ZZ
      case output_e_::DefGrad_Determinant:
        //! NOTE: Currently outputs (1 - J), not J
        val = 1.0 - matrixDeterminant3d(F.data()); return; // 1 - J, 1 - V/Vo, 1 - det| F |
      case output_e_::DefGrad_Determinant_FBAR:
        //! NOTE: Currently outputs (1 - JBar), not JBar
        val = pbuffer.getAttribute<attribs_e_::JBar>(_source_bin, _source_pidib); return; // JBar
      // case output_e_::sJ:
      //   val = pbuffer.getAttribute<attribs_e_::J>(_source_bin, _source_pidib); return; // (1-J)
      // case output_e_::sJBar:
      //   val = pbuffer.getAttribute<attribs_e_::JBar>(_source_bin, _source_pidib); return; // 1-JBar
      case output_e_::logJp:
        val = pbuffer.getAttribute<attribs_e_::logJp>(_source_bin, _source_pidib); return; 
      case output_e_::DefGrad_Invariant1:
        val = compute_Invariant_1_from_3x3_Tensor(F.data()); return; // Def. Grad. Invariant 1
      case output_e_::DefGrad_Invariant2:
        val = compute_Invariant_2_from_3x3_Tensor(F.data()); return; // Def. Grad. Invariant 2
      case output_e_::DefGrad_Invariant3:
        val = compute_Invariant_3_from_3x3_Tensor(F.data()); return; // Def. Grad. Invariant 3
      case output_e_::PorePressure:
        val = pbuffer.getAttribute<attribs_e_::PorePressure>(_source_bin, _source_pidib); return; // Pore Pressue [Pa] (CoupledUP)
      case output_e_::EMPTY:
        val = 0.0; return; // val = zero if EMPTY requested, used to buffer output columns
      case output_e_::INVALID_CT:
        val = static_cast<PREC>(NAN); return; // Invalid compile-time request, e.g. Specifically disallowed output
    }

    pvec9 C; 
    C.set(0.); 
    pvec9 FBar;

    if (pbuffer.use_FBAR) {
      PREC J_Scale = cbrt((1.0-sJBar) / J);
      for (int d=0; d<9; d++) FBar[d] = F[d] * ((1. - pbuffer.FBAR_ratio) + pbuffer.FBAR_ratio * J_Scale);
      pbuffer.getStress_Cauchy(1., FBar, C);
    }
    else 
      pbuffer.getStress_Cauchy((1/J), F, C);
    compute_Invariants_from_3x3_Tensor(C.data(), Invariants.data());
    switch (idx) {
      case output_e_::StressCauchy_XX:
        val = C[0]; return; // StressCauchy_XX
      case output_e_::StressCauchy_XY:
        val = C[1]; return; // StressCauchy_XY
      case output_e_::StressCauchy_XZ:
        val = C[2]; return; // StressCauchy_XZ
      case output_e_::StressCauchy_YX:
        val = C[3]; return; // StressCauchy_YX
      case output_e_::StressCauchy_YY:
        val = C[4]; return; // StressCauchy_YY
      case output_e_::StressCauchy_YZ:
        val = C[5]; return; // StressCauchy_YZ
      case output_e_::StressCauchy_ZX:
        val = C[6]; return; // StressCauchy_ZX
      case output_e_::StressCauchy_ZY:
        val = C[7]; return; // StressCauchy_ZY
      case output_e_::StressCauchy_ZZ:
        val = C[8]; return; ;// StressCauchy_ZZ
      case output_e_::Pressure:
        val = compute_MeanStress_from_StressCauchy(C.data()); return; // Pressure [Pa], Mean Stress 
      case output_e_::VonMisesStress:
        val = compute_VonMisesStress_from_StressCauchy(C.data()); return; // Von Mises Stress [Pa]
      case output_e_::StressCauchy_Invariant1:
        val = compute_Invariant_1_from_3x3_Tensor(C.data()); return; // Cauchy Stress Invariant 1
      case output_e_::StressCauchy_Invariant2:
        val = compute_Invariant_2_from_3x3_Tensor(C.data()); return; // Cauchy Stress Invariant 2
      case output_e_::StressCauchy_Invariant3:
        val = compute_Invariant_3_from_3x3_Tensor(C.data()); return;// Cauchy Stress Invariant 3
      case output_e_::StressCauchy_1:
        compute_Principals_from_Invariants_3x3_Sym_Tensor(Invariants.data(), Principals.data()); 
        val = Principals[0]; return; // Cauchy Stress Principal 1
      case output_e_::StressCauchy_2:
        compute_Principals_from_Invariants_3x3_Sym_Tensor(Invariants.data(), Principals.data()); 
        val = Principals[1]; return; // Cauchy Stress Principal 2
      case output_e_::StressCauchy_3:
        compute_Principals_from_Invariants_3x3_Sym_Tensor(Invariants.data(), Principals.data()); 
        val = Principals[2]; return; // Cauchy Stress Principal 3
    }

    pvec9 e; 
    e.set(0.); 
    compute_StrainSmall_from_DefGrad(F.data(), e.data());
    compute_Invariants_from_3x3_Tensor(e.data(), Invariants.data());
    switch (idx) {
      case output_e_::StrainSmall_XX:
        val = C[0]; return; // StressCauchy_XX
      case output_e_::StrainSmall_XY:
        val = C[1]; return; // StressCauchy_XY
      case output_e_::StrainSmall_XZ:
        val = C[2]; return; // StressCauchy_XZ
      case output_e_::StrainSmall_YX:
        val = C[3]; return; // StressCauchy_YX
      case output_e_::StrainSmall_YY:
        val = C[4]; return; // StressCauchy_YY
      case output_e_::StrainSmall_YZ:
        val = C[5]; return; // StressCauchy_YZ
      case output_e_::StrainSmall_ZX:
        val = C[6]; return; // StressCauchy_ZX
      case output_e_::StrainSmall_ZY:
        val = C[7]; return; // StressCauchy_ZY
      case output_e_::StrainSmall_ZZ:
        val = C[8]; return; // StressCauchy_ZZ
      case output_e_::VonMisesStrain:
        val = compute_VonMisesStrain_from_StrainSmall(e.data()); return; // Von Mises Stress [Pa]
      case output_e_::StrainSmall_Invariant1:
        val = compute_Invariant_1_from_3x3_Tensor(e.data()); return; // Cauchy Stress Invariant 1
      case output_e_::StrainSmall_Invariant2:
        val = compute_Invariant_2_from_3x3_Tensor(e.data()); return; // Cauchy Stress Invariant 2
      case output_e_::StrainSmall_Invariant3:
        val = compute_Invariant_3_from_3x3_Tensor(e.data()); return; // Cauchy Stress Invariant 3
      case output_e_::StrainSmall_1:
        compute_Principals_from_Invariants_3x3_Sym_Tensor(Invariants.data(), Principals.data()); 
        val = Principals[0]; return; // Cauchy Stress Principal 1
      case output_e_::StrainSmall_2:
        compute_Principals_from_Invariants_3x3_Sym_Tensor(Invariants.data(), Principals.data()); 
        val = Principals[1]; return; // Cauchy Stress Principal 2
      case output_e_::StrainSmall_3:
        compute_Principals_from_Invariants_3x3_Sym_Tensor(Invariants.data(), Principals.data()); 
        val = Principals[2]; return; // Cauchy Stress Principal 3
      default:
        val = -1; return; // Invalid run-time request, e.g. Incorrect output attributes name
    }
  }


template <typename Partition, typename ParticleBuffer, typename ParticleArray>
__global__ void retrieve_particle_buffer(Partition partition, Partition prev_partition,
                                         ParticleBuffer pbuffer, ParticleBuffer next_pbuffer,
                                         ParticleArray parray, 
                                         PREC *trackVal, 
                                         int *_parcnt) {
  int pcnt = g_buckets_on_particle_buffer ? next_pbuffer._ppbs[blockIdx.x] : partition._ppbs[blockIdx.x];
  ivec3 blockid = partition._activeKeys[blockIdx.x];
  auto advection_bucket = g_buckets_on_particle_buffer
      ? next_pbuffer._blockbuckets + blockIdx.x * g_particle_num_per_block
      : partition._blockbuckets + blockIdx.x * g_particle_num_per_block;
  for (int pidib = threadIdx.x; pidib < pcnt; pidib += blockDim.x) {
    auto advect = advection_bucket[pidib];
    ivec3 source_blockid;
    dir_components(advect / g_particle_num_per_block, source_blockid);
    source_blockid += blockid;
    auto source_blockno = prev_partition.query(source_blockid);
    auto source_pidib = advect % g_particle_num_per_block;
    auto source_bin = g_buckets_on_particle_buffer
          ? pbuffer.ch(_0, pbuffer._binsts[source_blockno] + source_pidib / g_bin_capacity)
          : pbuffer.ch(_0, prev_partition._binsts[source_blockno] + source_pidib / g_bin_capacity);
    auto _source_pidib = source_pidib % g_bin_capacity;

    auto parid = atomicAdd(_parcnt, 1);
    /// pos
    parray.val(_0, parid) = source_bin.val(_0, _source_pidib);
    parray.val(_1, parid) = source_bin.val(_1, _source_pidib);
    parray.val(_2, parid) = source_bin.val(_2, _source_pidib);
  }
}

template <typename Partition, typename ParticleBuffer, typename ParticleArray, num_attribs_e N, typename ParticleTarget>
__global__ void
retrieve_particle_buffer_attributes_general(Partition partition,
                                        Partition prev_partition,
                                        ParticleBuffer pbuffer, ParticleBuffer next_pbuffer,
                                        ParticleArray parray, 
                                        ParticleAttrib<N> pattrib,
                                        PREC *trackVal, 
                                        int *_parcnt, 
                                        ParticleTarget particleTarget,
                                        PREC *valAgg, 
                                        const vec7 target, 
                                        int *_targetcnt, bool output_pt=false) {

}

template <typename Partition, material_e mt, typename ParticleArray, num_attribs_e N, typename ParticleTarget>
__global__ void
retrieve_particle_buffer_attributes_general(Partition partition,
                                        Partition prev_partition,
                                        ParticleBuffer<mt> pbuffer, ParticleBuffer<mt> next_pbuffer,
                                        ParticleArray parray, 
                                        ParticleAttrib<N> pattrib,
                                        PREC *trackVal, 
                                        int *_parcnt, 
                                        ParticleTarget particleTarget,
                                        PREC *valAgg, 
                                        const vec7 target, 
                                        int *_targetcnt, bool output_pt=false, bool particles_output_exterior_only=false) {
  int pcnt = g_buckets_on_particle_buffer ? next_pbuffer._ppbs[blockIdx.x] : partition._ppbs[blockIdx.x];
  ivec3 blockid = partition._activeKeys[blockIdx.x];

  auto advection_bucket = g_buckets_on_particle_buffer
      ? next_pbuffer._blockbuckets + blockIdx.x * g_particle_num_per_block
      : partition._blockbuckets + blockIdx.x * g_particle_num_per_block;
  //uint32_t blockno = blockIdx.x;

  // Check if any of the 27 blocks around the current block has no particles
  // If no particles, this is a particle block near exterior of model
  // Saves memory if particles_output_exterior_only is true
  if ( particles_output_exterior_only ) {
    bool exterior_particles = false;
#pragma unroll 3
    for (char i = -1; i < 2; ++i)
#pragma unroll 3
      for (char j = -1; j < 2; ++j)
#pragma unroll 3
        for (char k = -1; k < 2; ++k) {
          auto ext_blockno = partition.query(blockid + ivec3(i, j, k));
          // Check if valid block
          auto ext_pcnt = g_buckets_on_particle_buffer 
                    ? next_pbuffer._ppbs[ext_blockno] 
                    : partition._ppbs[ext_blockno];
          constexpr int too_few_particles = g_exterior_particles_cutoff;
          if (ext_pcnt <= too_few_particles) {
            exterior_particles = true;
            break;
          }
        }
    if (!exterior_particles) return;
  }


  for (int pidib = threadIdx.x; pidib < pcnt; pidib += blockDim.x) {
    auto advect = advection_bucket[pidib];
    ivec3 source_blockid;
    dir_components(advect / g_particle_num_per_block, source_blockid);
    source_blockid += blockid;
    auto source_blockno = prev_partition.query(source_blockid);
    auto source_pidib = advect % g_particle_num_per_block;
    auto source_bin = g_buckets_on_particle_buffer
          ? pbuffer.ch(_0, pbuffer._binsts[source_blockno] + source_pidib / g_bin_capacity)
          : pbuffer.ch(_0, prev_partition._binsts[source_blockno] + source_pidib / g_bin_capacity);
    auto _source_pidib = source_pidib % g_bin_capacity;
    auto _source_bin = g_buckets_on_particle_buffer
          ? pbuffer._binsts[source_blockno] + source_pidib / g_bin_capacity
          : prev_partition._binsts[source_blockno] + source_pidib / g_bin_capacity;
    using attribs_e_ = typename ParticleBuffer<mt>::attribs_e;
    using output_e_ = particle_output_attribs_e;

    //auto global_particle_ID = bucket[pidib];
    auto parid = atomicAdd(_parcnt, 1); //< Particle count, increments atomically
    constexpr PREC o = g_offset;
    PREC l = pbuffer.length;

    /// Send positions (x,y,z) [m] to parray (device --> device)
    float3 position; 
    position.x = (pbuffer.getAttribute<attribs_e_::Position_X>(_source_bin, _source_pidib) - o) * l;
    position.y = (pbuffer.getAttribute<attribs_e_::Position_Y>(_source_bin, _source_pidib) - o) * l;
    position.z = (pbuffer.getAttribute<attribs_e_::Position_Z>(_source_bin, _source_pidib) - o) * l;
    parray.val(_0, parid) = position.x;
    parray.val(_1, parid) = position.y;
    parray.val(_2, parid) = position.z;
    auto global_particle_ID = pbuffer.getAttribute<attribs_e_::ID>(_source_bin, _source_pidib);

    if (!output_pt) {
      vec<int,static_cast<int>(N)> output_attribs; // =  pbuffer.output_attribs;
      for (int j = 0; j < static_cast<int>(N);j++) output_attribs[j]= pbuffer.output_attribs_dyn[j];
      for (unsigned i=0; i < static_cast<unsigned>(N); i++ ) {
        const unsigned I = i;
        if (I < sizeof(pbuffer.output_attribs_dyn) / sizeof(int)) {
          output_e_ idx = static_cast<output_e_>(output_attribs[i]); //< Map index for output 
          PREC val;
          caseSwitch_ParticleAttrib<mt>(pbuffer, _source_bin, _source_pidib, idx, val);
          setParticleAttrib(pattrib, i, parid, val);
        }
      }

      unsigned NUM_RUNTIME_TRACKERS = pbuffer.num_runtime_trackers;
      unsigned NUM_RUNTIME_TRACKER_ATTRIBS = pbuffer.num_runtime_tracker_attribs; 
      /// Get value of tracked particles for desired attribute
      vec<int, g_max_particle_tracker_attribs> track_attribs = pbuffer.track_attribs;
      for (int t=0; t<g_max_particle_trackers; t++) {
        if (t >= NUM_RUNTIME_TRACKERS) continue;
        if (global_particle_ID != pbuffer.track_IDs[t]) continue;
        for (int i=0; i < g_max_particle_tracker_attribs; i++) {
          if (i >= NUM_RUNTIME_TRACKER_ATTRIBS) continue;
          output_e_ idx = static_cast<output_e_>(track_attribs[i]); 
          PREC val;
          caseSwitch_ParticleAttrib(pbuffer, _source_bin, _source_pidib, idx, val);
          atomicAdd(&trackVal[t*g_max_particle_tracker_attribs + i], val);
        }
      }
    } else {
      int target_type = (int)target[0];
      pvec3 point_a {target[1], target[2], target[3]};
      pvec3 point_b {target[4], target[5], target[6]};

      PREC tol = 0.0;
      // Continue thread if cell is not inside target +/- tol
      if ((position.x >= (point_a[0]-tol-o)*l && position.x <= (point_b[0]+tol-o)*l) && 
          (position.y >= (point_a[1]-tol-o)*l && position.y <= (point_b[1]+tol-o)*l) &&
          (position.z >= (point_a[2]-tol-o)*l && position.z <= (point_b[2]+tol-o)*l))
      {
        vec<int,1> target_attribs = pbuffer.target_attribs;
        auto target_id = atomicAdd(_targetcnt, 1);
        if (target_id >= g_max_particle_target_nodes) printf("Allocate more space for particleTarget! node_id of %d compared to preallocated %d nodes!\n", target_id, g_max_particle_target_nodes);

        for (int i=0; i < 1; i++ ) {
          //int idx = target_attribs[i]; //< Map index for output attribute (particle_buffer.cuh)
          PREC val;
          output_e_ idx = static_cast<output_e_>(target_attribs[i]); 
          caseSwitch_ParticleAttrib(pbuffer, _source_bin, _source_pidib, idx, val);
          particleTarget.val(_0, target_id) = position.x; 
          particleTarget.val(_1, target_id) = position.y; 
          particleTarget.val(_2, target_id) = position.z; 
          particleTarget.val(_3, target_id) = val; 

          //atomicAdd(valAgg, val);
          //atomicAdd(valAgg, val);
          atomicMax(valAgg, val);
          //atomicMin(valAgg, val);
        }
      }
    }
  }
}


template <typename Partition, material_e mt, typename ParticleTarget>
__global__ void
retrieve_particle_target_attributes_general(Partition partition,
                                        Partition prev_partition,
                                        ParticleBuffer<mt> pbuffer, ParticleBuffer<mt> next_pbuffer,
                                        PREC *trackVal, 
                                        int *_parcnt, 
                                        ParticleTarget particleTarget,
                                        PREC *valAgg, 
                                        const vec7 target, 
                                        int *_targetcnt) {
  // First thread init. count of particles and value of aggregate for atomic operations
  if (threadIdx.x + blockIdx.x * blockDim.x == 0) {
    *_targetcnt = 0;
    if (target[0] == 0) *valAgg = 0.0; // Bootstrap atomicMax
    else if (target[0] == 1) *valAgg = 1e10; // Bootstrap atomicMin 
    else if (target[0] == 2) *valAgg = 0.0; // Bootstrap atomicAdd
  }
  __syncthreads(); // All threads/blocks wait for first thread to finish

  int pcnt = g_buckets_on_particle_buffer 
             ? next_pbuffer._ppbs[blockIdx.x] : partition._ppbs[blockIdx.x];
  if (pcnt == 0) return; // Return early if no particles in block
  ivec3 blockid = partition._activeKeys[blockIdx.x];

  pvec3 point_a {target[1], target[2], target[3]}; // Target point A (low corner)
  pvec3 point_b {target[4], target[5], target[6]}; // Target point B (high corner)
  PREC operation = target[0]; // Atomic operation to perform on target, 0 = max, 1 = min, 2 = sum
  PREC tol = (2 * g_blocksize) * g_dx; // 8 cell tolerance, accounts for advection 
  if (((blockid[0]*g_blocksize + 2 + 3)*g_dx < point_a[0]-tol || 
       (blockid[0]*g_blocksize + 2)*g_dx >   point_b[0]+tol) || 
      ((blockid[1]*g_blocksize + 2 + 3)*g_dx < point_a[1]-tol || 
       (blockid[1]*g_blocksize + 2)*g_dx > point_b[1]+tol) ||
      ((blockid[2]*g_blocksize + 2 + 3)*g_dx < point_a[2]-tol || 
       (blockid[2]*g_blocksize + 2)*g_dx > point_b[2]+tol)) {
      return; // Return early if block not in target's region
  } 
  // Used to find advection direction of particle during time-step
  auto advection_bucket = g_buckets_on_particle_buffer
      ? next_pbuffer._blockbuckets + blockIdx.x * g_particle_num_per_block
      : partition._blockbuckets + blockIdx.x * g_particle_num_per_block;

  // Loop over particles in block
  for (int pidib = threadIdx.x; pidib < pcnt; pidib += blockDim.x) {
    auto advect = advection_bucket[pidib];
    ivec3 source_blockid;
    dir_components(advect / g_particle_num_per_block, source_blockid);
    source_blockid += blockid;
    auto source_blockno = prev_partition.query(source_blockid);
    auto source_pidib = advect % g_particle_num_per_block;
    auto _source_pidib = source_pidib % g_bin_capacity;
    auto _source_bin = g_buckets_on_particle_buffer
          ? pbuffer._binsts[source_blockno] + source_pidib / g_bin_capacity
          : prev_partition._binsts[source_blockno] + source_pidib / g_bin_capacity;
    auto source_bin = pbuffer.ch(_0, _source_bin);

    using attribs_e_ = typename ParticleBuffer<mt>::attribs_e;
    using output_e_ = particle_output_attribs_e;
    PREC l = pbuffer.length;
    PREC o = g_offset;
    //auto global_particle_ID = bucket[pidib];

    /// Send positions (x,y,z) [m] to parray (device --> device)
    float3 position; 
    position.x = (pbuffer.getAttribute<attribs_e_::Position_X>(_source_bin, _source_pidib) - o) * l;
    position.y = (pbuffer.getAttribute<attribs_e_::Position_Y>(_source_bin, _source_pidib) - o) * l;
    position.z = (pbuffer.getAttribute<attribs_e_::Position_Z>(_source_bin, _source_pidib) - o) * l;
    //auto global_particle_ID = pbuffer.getAttribute<attribs_e_::ID>(_source_bin, _source_pidib);

    int target_type = (int)target[0];

    // Skip particle if not inside target +/- tol
    tol = 0.0; // Tolerance of particle dist. from target
    if ((position.x >= (point_a[0]-tol-o)*l && position.x <= (point_b[0]+tol-o)*l) && 
        (position.y >= (point_a[1]-tol-o)*l && position.y <= (point_b[1]+tol-o)*l) &&
        (position.z >= (point_a[2]-tol-o)*l && position.z <= (point_b[2]+tol-o)*l))
    {
      auto parid = atomicAdd(_parcnt, 1); //< Particle count, increments atomically
      int parcnt_target;
      if (parid >= g_max_particle_target_nodes) {
          printf("ERROR: Particle Number[%d] in particleTarget >= g_max_particle_target_nodes[%d]! Increase value! Skipping particle...\n", parid, g_max_particle_target_nodes);
      } else {
        parcnt_target = atomicAdd(_targetcnt, 1); //< Particle count in target so far
        // Write to particleTarget if appropiate time-step (device)
        particleTarget.val(_0, parcnt_target) = position.x; 
        particleTarget.val(_1, parcnt_target) = position.y; 
        particleTarget.val(_2, parcnt_target) = position.z; 
      }

      vec<int,1> target_attribs = pbuffer.target_attribs;
      for (int i=0; i < 1; i++ ) {
        PREC val = 0.; //< Value of user requested output attribute on particle
        output_e_ idx = static_cast<output_e_>(target_attribs[i]); //< Output attrib. index
        caseSwitch_ParticleAttrib(pbuffer, _source_bin, _source_pidib, idx, val);
        
        // Write to particleTarget if appropiate time-step (device)
        if (parid < g_max_particle_target_nodes) 
          particleTarget.val(_3, parcnt_target) = val; 
        
        // TODO : Use warp reduction to reduce atomic operations
        // TODO : Must ensure all threads in warp are active
        // TODO : Reduce warp reduction to shared memory
        // TODO : Reduce shared memory to global memory
        if (operation == 0) // Max
          atomicMax(valAgg, val);
        else if (operation == 1) // Min
          atomicMin(valAgg, val);
        else if (operation == 2) // Sum
          atomicAdd(valAgg, val);
        // atomicMax(&valAgg[i], val);
      }
    }
  }
  // __syncthreads();
  // {
  //   atomicMax(trackVal, valAgg[0]);
  // }
}


/// @brief Functions to retrieve particle attributes.
/// Copies from particle buffer to particle arrays (device --> device)
/// Depends on material model, copy/paste/modify function for new materials
template <typename Partition, typename ParticleBuffer, typename ParticleArray, num_attribs_e N, typename ParticleTarget>
__global__ void
retrieve_particle_buffer_attributes(Partition partition,
                                        Partition prev_partition,
                                        ParticleBuffer pbuffer,
                                        ParticleArray parray, 
                                        ParticleAttrib<N> pattrib,
                                        PREC *trackVal, 
                                        int *_parcnt, 
                                        ParticleTarget particleTarget,
                                        PREC *valAgg, 
                                        const vec7 target, 
                                        int *_targetcnt, bool output_pt=false) { }

// TODO: Refactor all the Meshed outputs
template <typename Partition, typename ParticleArray, num_attribs_e N, typename ParticleTarget>
__global__ void
retrieve_particle_buffer_attributes(Partition partition,
                                         Partition prev_partition,
                                         ParticleBuffer<material_e::Meshed> pbuffer,
                                         ParticleArray parray, 
                                         ParticleAttrib<N> pattrib,
                                         PREC *trackVal, 
                                         int *_parcnt, 
                                         ParticleTarget particleTarget,
                                         PREC *valAgg, 
                                         const vec7 target, 
                                         int *_targetcnt, bool output_pt=false) {
  if (output_pt) { return; }
  int pcnt = g_buckets_on_particle_buffer ? pbuffer._ppbs[blockIdx.x] : partition._ppbs[blockIdx.x];
  ivec3 blockid = partition._activeKeys[blockIdx.x];
  auto advection_bucket = g_buckets_on_particle_buffer
      ? pbuffer._blockbuckets + blockIdx.x * g_particle_num_per_block
      : partition._blockbuckets + blockIdx.x * g_particle_num_per_block;
  for (int pidib = threadIdx.x; pidib < pcnt; pidib += blockDim.x) {
    auto advect = advection_bucket[pidib];
    ivec3 source_blockid;
    dir_components(advect / g_particle_num_per_block, source_blockid);
    source_blockid += blockid;
    auto source_blockno = prev_partition.query(source_blockid);
    auto source_pidib = advect % g_particle_num_per_block;
    auto source_bin = g_buckets_on_particle_buffer
          ? pbuffer.ch(_0, pbuffer._binsts[source_blockno] + source_pidib / g_bin_capacity)
          : pbuffer.ch(_0, prev_partition._binsts[source_blockno] + source_pidib / g_bin_capacity);
    auto _source_pidib = source_pidib % g_bin_capacity;

    /// Increase particle ID
    auto parid = atomicAdd(_parcnt, 1);
    PREC o = g_offset; // Off-by-two buffer, 8 dx, on sim. domain
    PREC l = pbuffer.length;
    /// Send positions (x,y,z) [0.0, 1.0] to parray (device --> device)
    parray.val(_0, parid) = (source_bin.val(_0, _source_pidib) - o) * l;
    parray.val(_1, parid) = (source_bin.val(_1, _source_pidib) - o) * l;
    parray.val(_2, parid) = (source_bin.val(_2, _source_pidib) - o) * l;

    for (unsigned i=0; i < 3; i++) {
      /// Send attributes to pattribs (device --> device)
      PREC JBar = 1.0 - source_bin.val(_8, _source_pidib);
      PREC pressure  = source_bin.val(_9, _source_pidib);
      PREC von_mises = source_bin.val(_10, _source_pidib);
      PREC val = pressure; // !Need to refactor 
      setParticleAttrib(pattrib, i, parid, val);
    }
    
    if (1) {
      /// Set desired value of tracked particle
      auto ID = source_bin.val(_3, _source_pidib);
      for (int i = 0; i < g_max_particle_trackers; i++) {
        if (ID == g_track_ID) { // ! Deprecated
          PREC v = (source_bin.val(_1, _source_pidib) - o) * l;
          atomicAdd(trackVal, v);
        }
      }
    }
  }
}


template <typename VerticeArray, typename ElementBuffer, typename ElementArray, typename ElementAttrib>
__global__ void
retrieve_element_buffer_attributes( uint32_t blockCount,
                                         const VerticeArray vertice_array,
                                         const ElementBuffer elementBins,
                                         const ElementArray element_array,
                                         ElementAttrib element_attribs,
                                         PREC *trackVal, 
                                         int *_elcnt) { }

template <typename VerticeArray, typename ElementArray, typename ElementAttrib>
__global__ void
retrieve_element_buffer_attributes( uint32_t blockCount,
                                         const VerticeArray vertice_array,
                                         const ElementBuffer<fem_e::Brick> elementBins,
                                         const ElementArray element_array,
                                         ElementAttrib element_attribs,
                                         PREC *trackVal, 
                                         int *_elcnt) { }

template <typename VerticeArray, typename ElementArray, typename ElementAttrib>
__global__ void
retrieve_element_buffer_attributes( uint32_t blockCount,
                                         const VerticeArray vertice_array,
                                         const ElementBuffer<fem_e::Tetrahedron> elementBins,
                                         const ElementArray element_array,
                                         ElementAttrib element_attribs,
                                         PREC *trackVal, 
                                         int *_elcnt) {
  auto element_number =  blockIdx.x * blockDim.x + threadIdx.x;
  if (element_number < blockCount && element_number < g_max_fem_element_num) 
  {
  auto element = elementBins.ch(_0, element_number);

  //auto element_ID = element_number; //= atomicAdd(_parcnt, 1);
  PREC o = g_offset; // Off-by-two buffer, 8 dx, on sim. domain
  PREC l = elementBins.length;

  int IDs[4];
  pvec3 p[4];

  pvec9 DmI, Bm;
  DmI.set(0.0);
  Bm.set(0.0);
  IDs[0] = element.val(_0, 0); //< ID of node 0
  IDs[1] = element.val(_1, 0); //< ID of node 1
  IDs[2] = element.val(_2, 0); //< ID of node 2 
  IDs[3] = element.val(_3, 0); //< ID of node 3retrieve_element
  DmI[0] = element.val(_4, 0); //< Bm^T, undef. area weighted face normals^T
  DmI[1] = element.val(_5, 0);
  DmI[2] = element.val(_6, 0);
  DmI[3] = element.val(_7, 0);
  DmI[4] = element.val(_8, 0);
  DmI[5] = element.val(_9, 0);
  DmI[6] = element.val(_10, 0);
  DmI[7] = element.val(_11, 0);
  DmI[8] = element.val(_12, 0);
  PREC V0 = element.val(_13, 0); //< Undeformed volume [m^3]
  //int sV0 = V0<0.f ? -1.f : 1.f;

  // Set position of vertices
#pragma unroll 4
  for (int v = 0; v < 4; v++) {
    int ID = IDs[v] - 1; //< Index at 0, my elements input files index from 1
    p[v][0] = vertice_array.val(_0, ID) * l; //< x [m]
    p[v][1] = vertice_array.val(_1, ID) * l; //< y [m]
    p[v][2] = vertice_array.val(_2, ID) * l; //< z [m]
  }

  pvec3 centroid;
  centroid.set(0.0);

#pragma unroll 3
  for (int d = 0; d < 3; d++)
    centroid[d] = (p[0][d] + p[1][d] + p[2][d] + p[3][d]) / 4.0;


  // __syncthreads();

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
  //PREC Vn = V0 * J;
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

  pvec9 P, G, Bs;
  P.set(0.0);
  G.set(0.0);
  Bs.set(0.0);
  compute_stress_PK1_fixedcorotated((PREC)1.0, elementBins.mu, elementBins.lambda, F, P);

  // G = P Bm
  matrixMatrixMultiplication3d(P.data(), Bm.data(), G.data());
  
  // F^-1 = inv(F)
  pvec9 Finv; //< Deformation gradient inverse
  Finv.set(0.0);
  matrixInverse(F.data(), Finv.data());

  // Bs = J F^-1^T Bm ;  Bs^T = J Bm^T F^-1; Note: (AB)^T = B^T A^T
  matrixTransposeMatrixMultiplication3d(Finv.data(), Bm.data(), Bs.data());

  Bm.set(0.0); //< Now use for Cauchy stress
  matrixMatrixTransposeMultiplication3d(P.data(), F.data(), Bm.data());
  PREC pressure  = compute_MeanStress_from_StressCauchy(Bm.data()) / J;
  PREC von_mises = compute_VonMisesStress_from_StressCauchy(Bm.data()) / sqrt(J);

  atomicAdd(_elcnt, 1);
    /// Increase particle ID
  /// Send positions (x,y,z) [0.0, 1.0] to parray (device --> device)
  element_attribs.val(_0, element_number) = (PREC)(centroid[0] - (o * l));
  element_attribs.val(_1, element_number) = (PREC)(centroid[1] - (o * l));
  element_attribs.val(_2, element_number) = (PREC)(centroid[2] - (o * l));


  /// Send attributes to element_attribs (device --> device)
  element_attribs.val(_3, element_number) = (PREC)(1.0-J);
  element_attribs.val(_4, element_number) = (PREC)pressure; //< I2
  element_attribs.val(_5, element_number) = (PREC)von_mises; //< vy

    // if (1) {
    //   /// Set desired value of tracked particle
    //   auto ID = source_bin.val(_3, _source_pidib);
    //   for (int i = 0; i < sizeof(g_track_IDs)/4; i++) {
    //     if (ID == g_track_ID) {
    //       PREC v = (source_bin.val(_1, _source_pidib) - o) * l;
    //       atomicAdd(trackVal, v);
    //     }
    //   }
    // }
  }
}

template <typename VerticeArray, typename ElementArray,  typename ElementAttrib>
__global__ void
retrieve_element_buffer_attributes(   uint32_t blockCount,
                                         const VerticeArray vertice_array,
                                         const ElementBuffer<fem_e::Tetrahedron_FBar> elementBins,
                                         const ElementArray element_array,
                                         ElementAttrib element_attribs,
                                         PREC *trackVal, 
                                         int *_elcnt) {
  auto element_number =  blockIdx.x * blockDim.x + threadIdx.x;
  if (element_number < blockCount && element_number < g_max_fem_element_num) 
  {
  auto element = elementBins.ch(_0, element_number);

  //auto element_ID = element_number; //= atomicAdd(_parcnt, 1);
  PREC o = g_offset; // Off-by-two buffer, 8 dx, on sim. domain
  PREC l = elementBins.length;

  int IDs[4];
  pvec3 p[4];

  pvec9 DmI, Bm;
  DmI.set(0.0);
  Bm.set(0.0);

  // IDs[0] = (int)element_array.val(_0, element_number);
  // IDs[1] = (int)element_array.val(_1, element_number);
  // IDs[2] = (int)element_array.val(_2, element_number);
  // IDs[3] = (int)element_array.val(_3, element_number);
  IDs[0] = (int)element.val(_0, 0); //< ID of node 0
  IDs[1] = (int)element.val(_1, 0); //< ID of node 1
  IDs[2] = (int)element.val(_2, 0); //< ID of node 2 
  IDs[3] = (int)element.val(_3, 0); //< ID of node 3
  DmI[0] = element.val(_4, 0); //< Bm^T, undef. area weighted face normals^T
  DmI[1] = element.val(_5, 0);
  DmI[2] = element.val(_6, 0);
  DmI[3] = element.val(_7, 0);
  DmI[4] = element.val(_8, 0);
  DmI[5] = element.val(_9, 0);
  DmI[6] = element.val(_10, 0);
  DmI[7] = element.val(_11, 0);
  DmI[8] = element.val(_12, 0);
  PREC V0 = element.val(_13, 0); //< Undeformed volume [m^3]
  PREC sJ = element.val(_14, 0);
  PREC sJBar = element.val(_15, 0);
  //int sV0 = V0<0.f ? -1.f : 1.f;

  // Set position of vertices
#pragma unroll 4
  for (int v = 0; v < 4; v++) {
    int ID = IDs[v] - 1; //< Index at 0, my elements input files index from 1
    p[v][0] = vertice_array.val(_0, ID) * l; //< x [m]
    p[v][1] = vertice_array.val(_1, ID) * l; //< y [m]
    p[v][2] = vertice_array.val(_2, ID) * l; //< z [m]
  }

  pvec3 centroid;
  centroid.set(0.0);

#pragma unroll 3
  for (int d = 0; d < 3; d++)
    centroid[d] = (p[0][d] + p[1][d] + p[2][d] + p[3][d]) / 4.0;


  //__syncthreads();

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
  //PREC Vn = V0 * J;
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

  PREC J_Scale =  cbrt( ((1.0 - sJBar) / J) );
#pragma unroll 9
  for (int d=0; d<9; d++) F[d] = J_Scale * F[d];
  PREC JBar = (1.0 - sJBar);

  pvec9 P;
  pvec9 G;
  pvec9 Bs;
  P.set(0.0);
  G.set(0.0);
  Bs.set(0.0);
  compute_stress_PK1_fixedcorotated((PREC)1.0, elementBins.mu, elementBins.lambda, F, P);

  // G = P Bm
  matrixMatrixMultiplication3d(P.data(), Bm.data(), G.data());
  
  // F^-1 = inv(F)
  pvec9 Finv; //< Deformation gradient inverse
  Finv.set(0.0);
  matrixInverse(F.data(), Finv.data());
  // Bs = J F^-1^T Bm ;  Bs^T = J Bm^T F^-1; Note: (AB)^T = B^T A^T
  matrixTransposeMatrixMultiplication3d(Finv.data(), Bm.data(), Bs.data());

  Bm.set(0.0); //< Now use for Cauchy stress
  matrixMatrixTransposeMultiplication3d(P.data(), F.data(), Bm.data());
  PREC pressure  = compute_MeanStress_from_StressCauchy(Bm.data()) / J;
  PREC von_mises = compute_VonMisesStress_from_StressCauchy(Bm.data()) / sqrt(J);
  atomicAdd(_elcnt, 1);
    /// Increase particle ID
  /// Send positions (x,y,z) [0.0, 1.0] to parray (device --> device)
  element_attribs.val(_0, element_number) = (PREC)(centroid[0] - (o * l));
  element_attribs.val(_1, element_number) = (PREC)(centroid[1] - (o * l));
  element_attribs.val(_2, element_number) = (PREC)(centroid[2] - (o * l));


  /// Send attributes to element_attribs (device --> device)
  element_attribs.val(_3, element_number) = (PREC)(1.0 - JBar);
  element_attribs.val(_4, element_number) = (PREC)pressure; //< I2
  element_attribs.val(_5, element_number) = (PREC)von_mises; //< vy
  }
}



/// Retrieve grid-cells between points a & b from grid-buffer to gridTarget (JB)
template <typename Partition, typename Grid, typename GridTarget>
__global__ void retrieve_selected_grid_cells(
    const uint32_t blockCount, const Partition partition,
    const Grid prev_grid, GridTarget garray,
    const double dt, PREC_G *forceSum, const vec7 target, int *_targetcnt, const PREC length) {

  auto blockno = blockIdx.x;  //< Grid block number in partition
  if (blockno < blockCount) 
  {
    auto blockid = partition._activeKeys[blockno];

    int target_type = (int)target[0];
    vec3 point_a {target[1], target[2], target[3]};
    vec3 point_b {target[4], target[5], target[6]};


    // Check if gridblock contains part of the point_a to point_b region
    // End all threads in block if not
    if ((4*blockid[0] + 3)*g_dx < point_a[0] || (4*blockid[0])*g_dx > point_b[0]) return;
    else if ((4*blockid[1] + 3)*g_dx < point_a[1] || (4*blockid[1])*g_dx > point_b[1]) return;
    else if ((4*blockid[2] + 3)*g_dx < point_a[2] || (4*blockid[2])*g_dx > point_b[2]) return;
    else
    {
    //auto blockid = prev_blockids[blockno]; //< 3D grid-block index

      auto sourceblock = prev_grid.ch(_0, blockno); //< Set grid-block by block index
      PREC_G tol = g_dx * 0; // Tolerance layer around target domain
      PREC_G o = g_offset; // Offset [1]
      PREC_G l = length; // Domain length [m]
      // Add +1 to each? For point_b ~= point_a...
      if (g_log_level > 2)
      {
        ivec3 maxCoord;
        maxCoord[0] = (int)(((point_b[0] + tol) - (point_a[0] + tol)) * g_dx_inv + 1);
        maxCoord[1] = (int)(((point_b[1] + tol) - (point_a[1] + tol)) * g_dx_inv + 1);
        maxCoord[2] = (int)(((point_b[2] + tol) - (point_a[2] + tol)) * g_dx_inv + 1);
        int maxNodes = maxCoord[0] * maxCoord[1] * maxCoord[2];
        if (maxNodes >= g_grid_target_cells && threadIdx.x == 0) printf("ERROR: Allocate more space for gridTarget! Max grid-target nodes of %d compared to preallocated %d nodes!\n", maxNodes, g_grid_target_cells);
      }
      // Loop through cells in grid-block, stride by 32 to avoid thread conflicts
      for (int cidib = threadIdx.x % 32; cidib < g_blockvolume; cidib += 32) {

        // Grid node coordinate [i,j,k] in grid-block
        int i = (cidib >> (g_blockbits << 1)) & g_blockmask;
        int j = (cidib >> g_blockbits) & g_blockmask;
        int k = cidib & g_blockmask;

        // Grid node position [x,y,z] in entire domain 
        PREC_G xc = (4*blockid[0] + i)*g_dx; //< Node x position [1m]
        PREC_G yc = (4*blockid[1] + j)*g_dx; //< Node y position [1m]
        PREC_G zc = (4*blockid[2] + k)*g_dx; //< Node z position [1m]

        // Continue thread if cell is not inside grid-target +/- tol
        // if (xc < point_a[0] - tol || xc > point_b[0] + tol) {};
        // else if (yc < point_a[1] - tol || yc > point_b[1] + tol) {};
        // else if (zc < point_a[2] - tol || zc > point_b[2] + tol) {};
        if ((xc >= (point_a[0]-tol) && xc <= (point_b[0]+tol)) && 
            (yc >= (point_a[1]-tol) && yc <= (point_b[1]+tol)) &&
            (zc >= (point_a[2]-tol) && zc <= (point_b[2]+tol)))
        {
        // Unique ID by spatial position of cell in target [0 to g_grid_target_cells-1]
          // int node_id;
          // node_id = ((int)((xc - point_a[0] + tol) * g_dx_inv) * maxCoord[1] * maxCoord[2]) +
          //           ((int)((yc - point_a[1] + tol) * g_dx_inv) * maxCoord[2]) +
          //           ((int)((zc - point_a[2] + tol) * g_dx_inv));
          
          /// Set values in grid-array to specific cell from grid-buffer
          PREC_G x = (xc - o) * l;
          PREC_G y = (yc - o) * l;
          PREC_G z = (zc - o) * l;
          PREC_G mass = sourceblock.val(_0, i, j, k);
          PREC_G mvx1 = sourceblock.val(_1, i, j, k) * l;
          PREC_G mvy1 = sourceblock.val(_2, i, j, k) * l;
          PREC_G mvz1 = sourceblock.val(_3, i, j, k) * l;
          PREC_G volume = sourceblock.val(_7, i, j, k) ;
          PREC_G JBar = sourceblock.val(_8, i, j, k) ;
#if DEBUG_COUPLED_UP
          PREC_G mass_water = sourceblock.val(_9, i, j, k) ;
          PREC_G pore_pressure = sourceblock.val(_10, i, j, k) ;
#endif
          /// Set values in grid-array to specific cell from grid-buffer
          //PREC_G m1  = mass;
          // PREC_G small = 1e-7;
          if (mass > 0.f) 
          {
            PREC_G m1 = 1.0 / mass; //< Invert mass, avoids division operator
            PREC_G vx1 = mvx1 * m1;
            PREC_G vy1 = mvy1 * m1;
            PREC_G vz1 = mvz1 * m1;
            // PREC_G vx2 = 0.f;
            // PREC_G vy2 = 0.f;
            // PREC_G vz2 = 0.f;
            gvec3 force;
            force[0] = mass * (vx1) / dt;
            force[1] = mass * (vy1) / dt;
            force[2] = mass * (vz1) / dt;


            JBar = (volume > 0.f) ? JBar / volume : static_cast<PREC_G>(0); 

            auto node_id = atomicAdd(_targetcnt, 1);

            if (node_id >= g_grid_target_cells) printf("Allocate more space for gridTarget! node_id of %d compared to preallocated %d nodes!\n", node_id, g_grid_target_cells);

            if (1)
            {
              PREC_G val = 0;
              if      ( target_type / 3 == 0 ) val =  force[0];
              else if ( target_type / 3 == 1 ) val =  force[1];
              else if ( target_type / 3 == 2 ) val =  force[2];
              else if      ( target_type / 3 == 3 ) val =  vx1;
              else if ( target_type / 3 == 4 ) val =  vy1;
              else if ( target_type / 3 == 5 ) val =  vz1;
              else if      ( target_type / 3 == 6 ) val =  mvx1;
              else if ( target_type / 3 == 7 ) val =  mvy1;
              else if ( target_type / 3 == 8 ) val =  mvz1;
              else if      ( target_type / 3 == 9 ) val =  mass;
              else if ( target_type / 3 == 10 ) val =  volume;
              else if ( target_type / 3 == 11 ) val =  JBar;
              if ((target_type % 3) == 1)
                val = (val > 0) ? 0 : val;
              else if ((target_type % 3) == 2) 
                val = (val < 0) ? 0 : val;
              else
                val = val;
              garray.val(_0, node_id) = x;
              garray.val(_1, node_id) = y;
              garray.val(_2, node_id) = z;
              garray.val(_3, node_id) = mass;
              garray.val(_4, node_id) = vx1;
              garray.val(_5, node_id) = vy1;
              garray.val(_6, node_id) = vz1;
              garray.val(_7, node_id) = (target_type / 3 == 0) ? val : 0;
              garray.val(_8, node_id) = (target_type / 3 == 1) ? val : 0;
              garray.val(_9, node_id) = (target_type / 3 == 2) ? val : 0; 
            }

            PREC_G val = 0;
            // Set load direction x/y/z
            if      ( target_type / 3 == 0 ) val =  force[0];
            else if ( target_type / 3 == 1 ) val =  force[1];
            else if ( target_type / 3 == 2 ) val =  force[2];
            else if      ( target_type / 3 == 3 ) val =  vx1;
            else if ( target_type / 3 == 4 ) val =  vy1;
            else if ( target_type / 3 == 5 ) val =  vz1;
            else if      ( target_type / 3 == 6 ) val =  mvx1;
            else if ( target_type / 3 == 7 ) val =  mvy1;
            else if ( target_type / 3 == 8 ) val =  mvz1;
            else if      ( target_type / 3 == 9 ) val =  mass;
            else if ( target_type / 3 == 10 ) val =  volume;
            else if ( target_type / 3 == 11 ) val =  JBar;
            //else val = 0.f;
            // Seperation -+/-/+ condition for load
            if ((target_type % 3) == 1)
              val = (val >= 0) ? 0 : -val;
            else if ((target_type % 3) == 2) 
              val = (val <= 0) ? 0 : val;
            else 
              val = val;
            
            atomicAdd(forceSum, val);
          }
        }
      }
    }
  }
}

                      
} // namespace mn

#endif
