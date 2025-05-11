#ifndef __SETTINGS_H_
#define __SETTINGS_H_
#include "partition_domain.h"
#include <MnBase/Math/Vec.h>
#include <MnBase/Object/Structural.h>
#include <string>
#include <iostream>
#include <array>
#include <vector>
// #include <chrono> // maybe place in another file
//NOLINTNEXTLINE(cppcoreguidelines-macro-usage) Macro usage necessary here for preprocessor if
#define PRINT_CELL_OVERFLOW 0//TODO: Move to another place

namespace mn {

/// * Enable/disable cluster communication for multi-node simulations
#define CLUSTER_COMM_STYLE 0 //< 0 = None, 1 = MPI, 2 = CUDA-aware MPI, 3 = NVSHMEM

/// * Set computational precisions (double = good but expensive, float = low but fast)
#define PREC double //< Particle floating-point precision
#define PREC_P double //< Particle floating-point precision
#define PREC_G float //< Grid floating-point precision

using f_ = structural_entity<PREC>; //< Set precision of particle attributes
using fp_ = structural_entity<PREC_P>; //< Set precision of particle attributes
using fg_ = structural_entity<PREC_G>; //< Set precision of particle attributes

/// Vector type shorthand
using ivec3 = vec<int, 3>;
using pvec3 = vec<PREC, 3>;
using pvec6 = vec<PREC, 6>;
using pvec7 = vec<PREC, 7>;
using pvec9 = vec<PREC, 9>;
using pvec3x3 = vec<PREC, 3, 3>;
using pvec3x4 = vec<PREC, 3, 4>;
using pvec3x3x3 = vec<PREC, 3, 3, 3>;
using gvec3 = vec<PREC_G, 3>;
using gvec6 = vec<PREC_G, 6>;
using gvec7 = vec<PREC_G, 7>;
using gvec9 = vec<PREC_G, 9>;
using gvec3x3 = vec<PREC_G, 3, 3>;
using gvec3x4 = vec<PREC_G, 3, 4>;
using gvec3x3x3 = vec<PREC_G, 3, 3, 3>;
using vec3 = vec<float, 3>;
using vec6 = vec<float, 6>;
using vec7 = vec<float, 7>;
using vec9 = vec<float, 9>;
using vec3x3 = vec<float, 3, 3>;
using vec3x4 = vec<float, 3, 4>;
using vec3x3x3 = vec<float, 3, 3, 3>;


// using Duration = std::chrono::duration<float>;

/// * Available material models
enum class material_e { JFluid, JFluid_ASFLIP, JFluid_FBAR, JBarFluid,
                        FixedCorotated, FixedCorotated_ASFLIP, FixedCorotated_ASFLIP_FBAR,
                        NeoHookean_ASFLIP_FBAR,
                        Sand, // Drucker Prager Plasticity, StvkHencky Elasticity
                        NACC, 
                        CoupledUP,
                        Meshed,
                        VonMises,
                        Total };

/// * Available FEM element types
enum class fem_e {  Tetrahedron, Tetrahedron_FBar,
                    Brick, //< Not implemented yet
                    Total };


/// * For convenience, binds I/O sizes to compile-time variable names
/// * Guarantees someone avoids unimplemented sizes (e.g. -1, 10004)
/// TODO : Make this run-time to avoid lengthy compilitation of templates
enum class num_attribs_e : int { Zero = 0, One = 1, Two = 2, Three = 3,
                                 Four = 4, Five = 5, Six = 6, Seven = 7,
                                 Eight = 8, Nine = 9 //, Ten = 10, 
                                 // Eleven = 11, Twelve = 12 , Thirteen = 13
                                //  Fourteen = 14, Fifteen = 15, Sixteen = 16, 
                                //  Eighteen = 18, Twentyfour = 24, Thirtytwo = 32 
                                 };
// Coupled-UP Soil Configs (Work-in-progress, model by Javier). Set to false if not a developer.
#define DEBUG_COUPLED_UP false //< Debugging flag for CoupleUP model. False = Don't reserve grid memory for debugging. True = Reserve grid memory for debugging.
constexpr bool g_debug_CoupledUP = DEBUG_COUPLED_UP; //< Debugging flag for CoupleUP

#define G_MAX_SHMEM_PER_BLOCK 64 //< Estimated max shared memory per block in kB. Used to estimate max. number of blocks per GPU. Used for debugging.
#define G_MAX_SHMEM_PER_SM 96 //< Estimated max shared memory per kernel on a streaming-multiprocessor in kB. Used to estimate max. number of blocks per GPU. Used for debugging.
constexpr bool g_g2p2g_favor_shmem_over_cache = true; //< ADVANCED. Tell CUDA to assign more of the "carveout" on new GPUs as shared-memory. Reduces the L1 Cache (i.e. greater risk of storing register temp values in global mem, can be very slow though large L2 cache on newer GPUs offsets this). Larger shmem pool allows more CUDA blocks to run concurrently, in some cases can ~2x simulation speed. Especially helpful if using ASFLIP+FBAR for G2P2G kernels (requires a lot of shmem), likewise if eventually using double precision vals in shmem (2x the shmem requirement).  

/// https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html, F.3.16.5
namespace config /// * Simulation config setup and name-space
{
// * GPU set-up. You will need to change these for your system/simulation. 
// * e.g. 4 GPU simulation on 4+ GPU system, each GPU holds up-to 3 unique particle models: 
// * g_device_cnt = 4, g_models_per_gpu = 2 ---> g_model_cnt = 8
// ! You will get errors if exceeding num. of:
// ! (i) Physical GPUs, check 'nvidia-smi' in terminal, (ii) Max. compiled particle models per GPU
constexpr int g_device_cnt = 1; //< IMPORTANT. Num. GPUs to compile for. Default 1.
constexpr int g_models_per_gpu = 2; //< IMPORTANT. Max num. particle models per GPU. Default 1.
constexpr uint32_t g_model_cnt = g_device_cnt * g_models_per_gpu; //< Max num. particle models on node

// * Output set-up
enum class log_e : int { None, Errors, Warn, Info, Tips };
constexpr int g_log_level = (int)log_e::Warn; //< 0 = Print Nothing, 1 = + Errors, 2 = + Warnings, 3 = + Info, 4 = + Tips.
constexpr int g_info_rate = 10; //< How often to print to stdout extra info messages of grid/particle memory use, etc. 1 = every frame, 2 = every other frame, etc.
constexpr bool g_particles_output_exterior_only = false; // Default if not set in input script. Output only particles in exteriors blocks per frame, reduces memory usage on disk. Turn off for FULL particle output. 
constexpr int g_exterior_particles_cutoff = 128; // Number of particles minimum in an exterior block to qualify for output. Avoid false positives when particle block is practically empty visually (e.g. 2 particles may as well be 0 cause you can see through it when visualizing)

// * Grid set-up
#define DOMAIN_BITS 11 //< Domain resolution. 8 -> (2^8)^3 grid-nodes. Increase = finer grids.
#define BLOCK_BITS 2 //< Block resolution. 2 -> (2^2)^3 grid-nodes. Set for Quadratic B-Spline.
#define ARENA_BITS 1 //< Arena resolution. 1 -> (2^1)^3 grid-blocks. Set for Quadratic B-Spline Shared Mem with Off-by-2.
#define DXINV (1.f * (1 << DOMAIN_BITS)) // Max grid-nodes in a direction, inverse of grid-spacing.
constexpr int g_domain_bits = DOMAIN_BITS; //< Grid block resolution.
constexpr int g_domain_size = (1 << DOMAIN_BITS); //< Max grid-nodes in domain direction.
constexpr float g_bc = 2; //< Offset for Off-by-2 (Claymore, Wang 2020)
constexpr float g_dx = 1.0 / DXINV; //< Grid-cell spacing
constexpr float g_dx_inv = DXINV; //< Max grid-nodes in a direction, inverse of grid-spacing.
constexpr float g_D_inv = 4.0 * DXINV * DXINV; //< D_p^-1 (Jiang 2015) for quad. B-spline
constexpr double g_dx_inv_d = (1.0 * (1 << DOMAIN_BITS)); 
constexpr double g_dx_d = 1.0 / g_dx_inv_d;
constexpr int g_blockbits = BLOCK_BITS; //< Sets num. grid-nodes in grid-block = 2^BLOCK_BITS^3
constexpr int g_blocksize = (1 << BLOCK_BITS); //< Grid-block size in direction. Default 4
constexpr int g_blockmask = ((1 << BLOCK_BITS) - 1); //< Mask for block. Default 3
constexpr int g_blockvolume = (1 << (BLOCK_BITS * 3)); //< No. grid-nodes in a block. Default 64.
constexpr int g_grid_bits = (DOMAIN_BITS - BLOCK_BITS); //< Resolution of grid-blocks in domain
constexpr int g_grid_size = (1 << (DOMAIN_BITS - BLOCK_BITS)); //< Max blocks in domain direction
constexpr int g_arena_bits = 1; //< Resolution of grid-block arena in shared memory for g2p2g.
constexpr int g_arenavolume = (g_blockvolume << (g_arena_bits*3)); //< No. grid-nodes in arena. Default 8.
constexpr float g_offset = g_dx * 8; //< Offset in grid-cells of sim origin from domain origin. Unstable below 2 (Off-by-2 G2P2G, Wang 2020). 4-6 saves memory. 8 creates extra space for boundary conditions at sim. edges.

// Convenient compiled ceiling function. May not work in C++ 11? Should be good for 14 and 17
constexpr int32_t constCeil(float num) {
    return (static_cast<float>(static_cast<int32_t>(num)) == num)
        ? static_cast<int32_t>(num)
        : static_cast<int32_t>(num) + ((num > 0) ? 1 : 0);
}
// Domain size
#define DOMAIN_LENGTH 1.0 //< Default max domain length 
#define DOMAIN_VOLUME DOMAIN_LENGTH * DOMAIN_LENGTH * DOMAIN_LENGTH //< Domain Vol. 
constexpr double g_length   = 1.0; //< Default domain full length (m)
constexpr double g_volume   = g_length * g_length * g_length; //< Default domain max volume [m^3]
constexpr double g_length_x = g_length / 1.0; //< Default domain x length (m)
constexpr double g_length_y = g_length / 1.0; //< Default domain y length (m)
constexpr double g_length_z = g_length / 8.0; //< Default domain z length (m)
constexpr double g_domain_volume = g_length * g_length * g_length;
constexpr double g_grid_ratio_x = g_length_x / g_length; //< Domain x ratio
constexpr double g_grid_ratio_y = g_length_y / g_length; //< Domain y ratio
constexpr double g_grid_ratio_z = g_length_z / g_length; //< Domain z ratio
constexpr int g_grid_size_x = (int)constCeil(static_cast<float>(g_grid_size * g_grid_ratio_x)); //< Domain x grid-blocks
constexpr int g_grid_size_y = (int)constCeil(static_cast<float>(g_grid_size * g_grid_ratio_y)); //< Domain y grid-blocks
constexpr int g_grid_size_z = (int)constCeil(static_cast<float>(g_grid_size * g_grid_ratio_z)); //< Domain z grid-blocks

/// ------------------
/// * Preallocate GPU memory for particles, grids, finite elements, grid-targets, etc
/// ------------------
// * Halo Blocks for Multi-GPU Communication. Set zero if only on Single-GPU
constexpr std::size_t g_max_halo_block = 1024 * 4; //< Max active halo blocks between any two GPUs. Preallocated, can resize. 

// * Grids
#define GBPCB 8 //< Grid blocks per cuda block. 8 or 16 is usually good. Can cause kernels to fail if a bad number for the specific GPU. Can slightly effect performance.
constexpr int g_num_grid_blocks_per_cuda_block = GBPCB;
constexpr int g_num_warps_per_grid_block = 1;
constexpr int g_num_warps_per_cuda_block = g_num_warps_per_grid_block * g_num_grid_blocks_per_cuda_block;
constexpr int g_max_active_block = 10000; //< Max active blocks in gridBlocks. Preallocated, can resize. Lower = less memory used.

// * Particles
#define MAX_PPC 128 //< VERY important. Max particles-per-cell. Must be a power of two, e.g. 16, 32, 64. Substantially effects memory/performance. Exceeding MAX_PPC deletes particles. Generally, use MAX_PPC = 8*(Actual PPC) to account for compression, if nearly incompressible materials this isn't as neccesary. 64 is usually reliable as default.
constexpr int g_max_ppc = MAX_PPC; //< Max particles per cell
constexpr bool is_powerof2(int val) {
    return val && ((val & (val - 1)) == 0);
}
constexpr bool g_ppc_pow2 = is_powerof2(g_max_ppc);
constexpr int g_max_particle_num = 4000000; //< Max no. particles. Very important, affects memory usage and performance. Preallocated, can resize.
constexpr int g_num_warps_per_particle_bin = 1; //< No. warps per particle bin. Usually 1 is good. Can cause kernels to fail if a bad number for the specific GPU. Can slightly effect performance.
constexpr int g_bin_capacity = 32 * g_num_warps_per_particle_bin; //< Particles / threads per particle bin. Multiple of 32 highly recommended (32 or 64 usually)
constexpr int g_particle_batch_capacity = 4 * g_bin_capacity; // Sets thread block size in g2p2g, etc. Usually 64, 128, 256, or 512 is good. If kernel uses a lot of shared memory (e.g. 32kB+ when using FBAR and ASFLIP) then raise num. for occupancy benefits. If said kernel uses a lot of registers (e.g. 64+), then lower for occupancy. See CUDA occupancy calculator online, varies by GPU/system. - JB
constexpr int g_particle_batch = g_particle_batch_capacity;
constexpr int g_particle_num_per_block = 
    (MAX_PPC * (1 << (BLOCK_BITS * 3))); //< Max no. particles in a block
constexpr std::size_t g_max_particle_bin = 
    g_max_particle_num / g_bin_capacity; //< Max no. particle bins. Preallocated
constexpr std::size_t calc_particle_bin_count(std::size_t numActiveBlocks) noexcept {
    return numActiveBlocks * (g_max_ppc * g_blockvolume / g_bin_capacity); } //< Return max particle bins that fit in the active blocks 
constexpr int g_particle_attribs = 3; //< No. attribute values to output per particle 
constexpr int g_max_particle_attribs = 9; //< No. attribute values to output per particle 
constexpr bool g_buckets_on_particle_buffer = true; //< ADVANCED. Default true. Controls if particle cell/block buckets, etc. are on partition (false) or particle-buffer (true). Used for compatability with original Multi-GPU and Single-GPU data-structure setup. Having them on particle buffer required if multiiple models per GPU. - JB

// * Particle-Trackers
constexpr int g_track_ID = 0; //< ID of particle to track, [0, g_max_fem_vertice_num)
constexpr int g_max_particle_trackers = 64; //< Max no. particle trackers. Preallocated, can resize.
std::array<int, g_max_particle_trackers> g_track_IDs = {g_track_ID}; //< IDs of particles to track for high-freq outputs
constexpr int g_max_particle_tracker_attribs = 6;

// * Particle-Targets
constexpr int g_max_particle_target_nodes = 1024 * 8; //< Max particles per particleTarget
constexpr int g_particle_target_cells = g_max_particle_target_nodes;
constexpr int g_particle_target_attribs = 10; //< No. values per gridTarget node

// * Grid-Targets
constexpr int g_max_grid_target_nodes = 1024 * 8; //< Max grid-nodes per gridTarget
constexpr int g_grid_target_cells = g_max_grid_target_nodes; //< Max grid-nodes per gridTarget
constexpr int g_grid_target_attribs = 10; //< No. values per gridTarget node

// * Grid Boundaries
constexpr int g_max_grid_boundaries = 6; //< Max grid-boundaries in scene
constexpr int g_grid_boundary_attribs = 7; //< No. of values per grid-boundary node

// * Finite Elements
constexpr int g_max_fem_vertice_num = 32;  // Max no. vertice on FEM mesh
constexpr int g_max_fem_element_num = 32; // Max no. element in FEM mesh
constexpr int g_fem_element_bin_capacity = 1; //< Finite elements per bin in elementBins
constexpr int g_max_fem_element_bin = 
    g_max_fem_element_num / g_fem_element_bin_capacity; // Max no. of finite element bins


// Resize ratios
// constexpr std::size_t g_block_check_ratio = 90;
// constexpr std::size_t g_block_resize_ratio = 110;
// constexpr std::size_t g_bin_check_ratio = 90;
// constexpr std::size_t g_bin_resize_ratio = 110;

// * Run-time/animation default settings
constexpr int g_total_frame_cnt = 30; //< Default simulation frames to output
constexpr int g_fps = 60; //< Default frames-per-second

// * Default material parameters, override at run-time
#define DENSITY 1000       //< Default density [kg/m^3]
#define YOUNGS_MODULUS 1e6 //< Default Young's Modulus [Pa]
#define POISSON_RATIO 0.3 // Default Poisson Ratio
#define CFL_CONDITION 0.5
#define MODEL_PPC 8.0 //< Default particles-per-cell
constexpr float g_cfl = CFL_CONDITION; //< Default CFL Condition Coefficient
constexpr float g_model_ppc = MODEL_PPC; //< Default particles-per-cell

// * Ambient parameters
#define PI_AS_A_DOUBLE 3.14159265358979323846
#define PI_AS_A_FLOAT  3.14159265358979323846f
#define GRAVITY -9.81 //< Default gravity (m/s^2)
constexpr float g_gravity = GRAVITY; //< Default gravity (m/s^2)

// * Utility functions
constexpr material_e get_material_type(int did) noexcept {
  material_e type{material_e::JFluid}; return type; }

constexpr fem_e get_element_type(int did) noexcept {
  fem_e type{fem_e::Tetrahedron}; return type;}

// * Structures for grouping variables
struct SimulatorConfigs {
  int dim;
  float dx, dxInv;
  int resolution;
  float gravity;
  std::vector<float> offset;
} simConfigs;

struct MaterialConfigs {
  PREC ppc, rho, CFL;
  PREC bulk, visco, gamma;
  PREC E, nu;
  PREC logJp0, frictionAngle, cohesion, beta;
  bool volumeCorrection;
  PREC xi;
  bool hardeningOn;
  PREC rhow, alpha1, poro, Kf, Ks, Kperm;
  PREC tensile_yield_strength;
  MaterialConfigs() : ppc(8.0), rho(1e3), bulk(2.2e7), visco(1e-3), gamma(7.1), E{1e7}, nu{0.3}, logJp0(0.), frictionAngle(30.), cohesion(0.), beta(0.5), volumeCorrection(false), xi(1.0), hardeningOn(true), rhow(1e3), alpha1(1.0), poro(0.2), Kf(1.0e7), Ks(2.2e7), Kperm(1.0e-5), tensile_yield_strength(2.0e6) {}
  ~MaterialConfigs() {}
};

struct AlgoConfigs {
  bool use_ASFLIP;
  bool use_FEM;
  bool use_FBAR;
  PREC ASFLIP_alpha, ASFLIP_beta_min, ASFLIP_beta_max;
  PREC FBAR_ratio;
  bool FBAR_fused_kernel;
  AlgoConfigs() : use_ASFLIP(true), use_FEM{false}, use_FBAR(false), ASFLIP_alpha(0.), ASFLIP_beta_min(0.), ASFLIP_beta_max(0.), FBAR_ratio(0.), FBAR_fused_kernel(true) {}
  ~AlgoConfigs() {}
};

constexpr int g_max_bathymetry_points = 32; //< Max no. of bathymetry coordinate points, i.e. 31 panels total

// * Boundary condition enumerators for easier reading
enum class boundary_contact_t { Sticky, Slip, Separate, Separable = Separate, Seperable = Separable};
enum class boundary_object_t { Walls, Box, Sphere, Cylinder, Plane, 
                                WALLS = Walls, BOX = Box, SPHERE = Sphere, CYLINDER = Cylinder, PLANE = Plane,
                                OSU_LWF_RAMP, OSU_LWF_PADDLE, 
                                USGS_RAMP, USGS_GATE, 
                                OSU_TWB_RAMP, OSU_TWB_PADDLE,
                                WASIRF_PUMP,
                                TOKYO_HARBOR,
                                VELOCITY_BOUNDARY,
                                ACCELERATION_BOUNDARY,
                                MOMENTUM_BOUNDARY,
                                FORCE_BOUNDARY,
                                FLOOR,
                                BATHYMETRY,
                                TOTAL };
// Structure for holding grid-boundary configs. Sent to GPU as a SMALL kernel parameter (GPU constant mem. is like 4kb)
struct GridBoundaryConfigs {
  int _ID; //< Specific grid-target ID, [0, number_of_targets)
  vec<PREC_G, 3> _domain_start; //< Start of boundary domain
  vec<PREC_G, 3> _domain_end; //< End of boundary domain
  boundary_object_t _object; //< Type of boundary object
  boundary_contact_t _contact; //< Type of contact
  PREC_G _friction_static, _friction_dynamic;
  vec<PREC_G, 3> _time; //< Start and end time of boundary. Last element deprecated. TODO: Only use 2 elements. Fix things in read_scene_input.h
  // vec3 _normal;
  // vec3 _trans, _transVel;
  vec<PREC_G, 3> _velocity; //< Velocity of boundary object
  // vec3x3 _rotMat;
  // vec3 _omega; 
  vec<int, 3> _array; //< Array of boundary objects
  vec<PREC_G, 3> _spacing; // Spacing between boundary objects
  int _num_bathymetry_points; //< Number of bathymetry points
  vec<PREC_G, 2> _bathymetry_points[static_cast<int>(g_max_bathymetry_points)]; //< Bathymetry points
};
// Deprecated structure for holding grid-target configs
// TODO: Undeprecate
struct GridTargetConfigs {
  static int number_of_targets; //< Total no. of grid-targets. TODO: May not work on multiple GPUs

  int target_ID; //< Specific grid-target ID, [0, number_of_targets)
  int idx_attribute; //< 
  int idx_operation; //<
  int idx_direction; //<
  float3 domain_start; //< 
  float3 domain_end;
  float output_frequency;

  GridTargetConfigs() : target_ID(number_of_targets), idx_attribute(-1), idx_operation(-1), idx_direction(-1), domain_start(make_float3(0.f,0.f,0.f)), domain_end(make_float3(0.f,0.f,0.f)), output_frequency(1.f) 
  { 
    number_of_targets++;
  }
  GridTargetConfigs(int attr, int oper, int dir,
                    float3 start, float3 end, 
                    float freq) : target_ID{number_of_targets}, idx_attribute(attr), idx_operation(oper), idx_direction(dir), domain_start(start), domain_end(end), output_frequency(freq) 
  { 
    number_of_targets++;
    if (output_frequency == 0) output_frequency = 1; //< Avoid potential divide by zero
    std::cout << "Target ID " << target_ID << " of " << number_of_targets << " total targets." << '\n';
  }
  ~GridTargetConfigs() 
  {
    number_of_targets--;
  }
};
int GridTargetConfigs::number_of_targets = 0; 

// Structure for holding particle-target/sensor configs. Sent to GPU as a SMALL kernel parameter (GPU constant mem. is like 4kb)
struct ParticleTargetConfigs  {
  static int number_of_targets; //< Total no. of targets/sensors

  int target_ID; //< Specific particle-target ID, [0, number_of_targets)
  int idx_attribute; //< 
  int idx_operation; //<
  int idx_direction; //<
  float3 domain_start; //< 
  float3 domain_end;
  float output_frequency;

  ParticleTargetConfigs() : target_ID{number_of_targets}, idx_attribute(-1), idx_operation(-1), idx_direction(-1), domain_start(make_float3(0.f,0.f,0.f)), domain_end(make_float3(0.f,0.f,0.f)), output_frequency(1.f) { 
    number_of_targets++;
  }
  ParticleTargetConfigs(int attr, int oper, int dir,
                    float3 start, float3 end, 
                    float freq) : target_ID(number_of_targets), idx_attribute(attr), idx_operation(oper), idx_direction(dir), domain_start(start), domain_end(end), output_frequency(freq) { 
    number_of_targets++;
    if (output_frequency == 0) output_frequency = 1; //< Avoid potential divide by zero
    std::cout << "Target ID " << target_ID << " of " << number_of_targets << " total targets." << '\n';
  }
  ~ParticleTargetConfigs() {
    number_of_targets--;
  }
};
int ParticleTargetConfigs::number_of_targets = 0; 


} // namespace config

// Set up max cartesian grid block domain dimensions for simulation. 
// Used to create data-structure for GPU hash table for grid/particle-block IDs. 
// May not want this defined here. TODO: Double-check.
using GridDomain = compact_domain<int, config::g_grid_size_x, config::g_grid_size_y,
                                  config::g_grid_size_z>;

} // namespace mn

#endif
