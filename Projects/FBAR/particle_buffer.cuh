#ifndef __PARTICLE_BUFFER_CUH_
#define __PARTICLE_BUFFER_CUH_
#include "settings.h"
#include "constitutive_models.cuh"
#include "utility_funcs.hpp"
#include <MnBase/Math/Vec.h>
#include <MnBase/Meta/Polymorphism.h>
#include <MnSystem/Cuda/HostUtils.hpp>
#include <MnBase/Meta/TypeMeta.h>


namespace mn {

using ParticleBinDomain = aligned_domain<char, config::g_bin_capacity>;
using ParticleBufferDomain = compact_domain<int, config::g_max_particle_bin>;
using ParticleArrayDomain = compact_domain<int, config::g_max_particle_num>;
using ParticleTargetDomain = compact_domain<int, config::g_max_particle_target_nodes>;

// * All  particle attributes available for ouput.
// * Not all materials will support every output.
enum class particle_output_attribs_e : int {
        EMPTY=-3, // Empty attribute request 
        INVALID_CT=-2, // Invalid compile-time request, e.g. deprecated variable (below END)
        INVALID_RT=-1, // Invalid run-time request e.g. "vel X" instead of "Velocity_X"
        START=0,
        ID = 0, Mass, Volume,
        Position_X, Position_Y, Position_Z,
        Velocity_X, Velocity_Y, Velocity_Z,
        DefGrad_XX, DefGrad_XY, DefGrad_XZ,
        DefGrad_YX, DefGrad_YY, DefGrad_YZ,
        DefGrad_ZX, DefGrad_ZY, DefGrad_ZZ,
        J, DefGrad_Determinant=J, JBar, DefGrad_Determinant_FBAR=JBar, 
        StressCauchy_XX, StressCauchy_XY, StressCauchy_XZ,
        StressCauchy_YX, StressCauchy_YY, StressCauchy_YZ,
        StressCauchy_ZX, StressCauchy_ZY, StressCauchy_ZZ,
        Pressure, VonMisesStress,
        DefGrad_Invariant1, DefGrad_Invariant2, DefGrad_Invariant3,
        DefGrad_1, DefGrad_2, DefGrad_3,
        StressCauchy_Invariant1, StressCauchy_Invariant2, StressCauchy_Invariant3,
        StressCauchy_1, StressCauchy_2, StressCauchy_3,
        StressPK1_XX, StressPK1_XY, StressPK1_XZ,
        StressPK1_YX, StressPK1_YY, StressPK1_YZ,
        StressPK1_ZX, StressPK1_ZY, StressPK1_ZZ,
        StressPK1_Invariant1, StressPK1_Invariant2, StressPK1_Invariant3,
        StressPK1_1, StressPK1_2, StressPK1_3,
        StressPK2_XX, StressPK2_XY, StressPK2_XZ,
        StressPK2_YX, StressPK2_YY, StressPK2_YZ,
        StressPK2_ZX, StressPK2_ZY, StressPK2_ZZ,
        StressPK2_Invariant1, StressPK2_Invariant2, StressPK2_Invariant3,
        StressPK2_1, StressPK2_2, StressPK2_3,
        StrainSmall_XX, StrainSmall_XY, StrainSmall_XZ,
        StrainSmall_YX, StrainSmall_YY, StrainSmall_YZ,
        StrainSmall_ZX, StrainSmall_ZY, StrainSmall_ZZ,
        StrainSmall_Invariant1, StrainSmall_Invariant2, StrainSmall_Invariant3,
        Dilation = StrainSmall_Invariant1, StrainSmall_Determinant = StrainSmall_Invariant3,
        StrainSmall_1,  StrainSmall_2, StrainSmall_3,
        VonMisesStrain,
        logJp=100,
        END,
        ExampleDeprecatedVariable //< Will give INVALID_CT output of -2
};

using particle_bin4_ =
    structural<structural_type::dense,
               decorator<structural_allocation_policy::full_allocation,
                         structural_padding_policy::sum_pow2_align>,
               ParticleBinDomain, attrib_layout::soa, f32_, f32_, f32_,
               f32_>; ///< pos, J
using particle_bin4_f_ =
    structural<structural_type::dense,
               decorator<structural_allocation_policy::full_allocation,
                         structural_padding_policy::sum_pow2_align>,
               ParticleBinDomain, attrib_layout::soa, f_, f_, f_,
               f_>; ///< pos, J
using particle_bin6_f_ =
    structural<structural_type::dense,
               decorator<structural_allocation_policy::full_allocation,
                         structural_padding_policy::sum_pow2_align>,
               ParticleBinDomain, attrib_layout::soa, f_, f_, f_,
               f_, f_, f_>; ///< pos, J, JBar, ID
using particle_bin7_f_ =
    structural<structural_type::dense,
               decorator<structural_allocation_policy::full_allocation,
                         structural_padding_policy::sum_pow2_align>,
               ParticleBinDomain, attrib_layout::soa, f_, f_, f_,
               f_, f_, f_, f_>; ///< pos, J / ID, vel
using particle_bin9_f_ =
    structural<structural_type::dense,
               decorator<structural_allocation_policy::full_allocation,
                         structural_padding_policy::sum_pow2_align>,
               ParticleBinDomain, attrib_layout::soa, f_, f_, f_, f_, 
               f_, f_, f_, 
               f_, f_>; ///< pos, J, vel, vol JBar
using particle_bin11_f_ =
    structural<structural_type::dense,
               decorator<structural_allocation_policy::full_allocation,
                         structural_padding_policy::sum_pow2_align>,
               ParticleBinDomain, attrib_layout::soa, f_, f_, f_,
               f_, f_, f_, f_, f_,
               f_, f_, f_>; ///< pos, ID, forces, restVolume, normals
using particle_bin12_f_ =
    structural<structural_type::dense,
               decorator<structural_allocation_policy::full_allocation,
                         structural_padding_policy::sum_pow2_align>,
               ParticleBinDomain, attrib_layout::soa, f_, f_, f_, 
               f_, f_, f_, f_, f_, f_, f_, f_, f_>; ///< pos, F
using particle_bin13_f_ =
    structural<structural_type::dense,
               decorator<structural_allocation_policy::full_allocation,
                         structural_padding_policy::sum_pow2_align>,
               ParticleBinDomain, attrib_layout::soa, f_, f_, f_, f_,
               f_, f_, f_, f_, f_, f_, f_, f_,
               f_>; ///< pos, F, logJp
using particle_bin15_f_ =
    structural<structural_type::dense,
               decorator<structural_allocation_policy::full_allocation,
                         structural_padding_policy::sum_pow2_align>,
               ParticleBinDomain, attrib_layout::soa, f_, f_, f_, f_,
               f_, f_, f_, f_, f_, f_, f_, f_, 
               f_, f_, f_>; ///< pos, F, vel
using particle_bin16_f_ =
    structural<structural_type::dense,
               decorator<structural_allocation_policy::full_allocation,
                         structural_padding_policy::sum_pow2_align>,
               ParticleBinDomain, attrib_layout::soa, f_, f_, f_, f_,
               f_, f_, f_, f_, f_, f_, f_, f_, 
               f_, f_, f_, f_>; ///< pos, F, logJp, vel
using particle_bin17_f_ =
    structural<structural_type::dense,
               decorator<structural_allocation_policy::full_allocation,
                         structural_padding_policy::sum_pow2_align>,
               ParticleBinDomain, attrib_layout::soa, f_, f_, f_, 
               f_, f_, f_, f_, f_, f_, f_, f_, f_, 
               f_, f_, f_,
               f_, f_>; ///< pos, F, vel, vol_Bar, J_Bar
using particle_bin18_f_ =
    structural<structural_type::dense,
               decorator<structural_allocation_policy::full_allocation,
                         structural_padding_policy::sum_pow2_align>,
               ParticleBinDomain, attrib_layout::soa, f_, f_, f_, 
               f_, f_, f_, f_, f_, f_, f_, f_, f_, 
               f_, f_, f_,
               f_, f_,
               f_>; ///< pos, F, vel, vol_Bar, J_Bar, ID

template <material_e mt> struct particle_bin_;
template <> struct particle_bin_<material_e::JFluid> : particle_bin4_f_ {};
template <> struct particle_bin_<material_e::JFluid_ASFLIP> : particle_bin7_f_ {};
template <> struct particle_bin_<material_e::JFluid_FBAR> : particle_bin6_f_ {};
template <> struct particle_bin_<material_e::JBarFluid> : particle_bin9_f_ {};
template <> struct particle_bin_<material_e::FixedCorotated> : particle_bin13_f_ {};
template <> struct particle_bin_<material_e::FixedCorotated_ASFLIP> : particle_bin16_f_ {};
template <> struct particle_bin_<material_e::FixedCorotated_ASFLIP_FBAR> : particle_bin18_f_ {};
template <> struct particle_bin_<material_e::NeoHookean_ASFLIP_FBAR> : particle_bin18_f_ {};
template <> struct particle_bin_<material_e::Sand> : particle_bin18_f_ {};
template <> struct particle_bin_<material_e::NACC> : particle_bin18_f_ {};
template <> struct particle_bin_<material_e::Meshed> : particle_bin11_f_ {};


template <typename ParticleBin>
using particle_buffer_ =
    structural<structural_type::dynamic,
               decorator<structural_allocation_policy::full_allocation,
                         structural_padding_policy::compact>,
               ParticleBufferDomain, attrib_layout::aos, ParticleBin>;

template <material_e mt>
struct ParticleBufferImpl : Instance<particle_buffer_<particle_bin_<mt>>> {
  static constexpr material_e materialType = mt;
  using base_t = Instance<particle_buffer_<particle_bin_<mt>>>;

  // Constructor
  template <typename Allocator>
  ParticleBufferImpl(Allocator allocator, std::size_t count)
      : base_t{spawn<particle_buffer_<particle_bin_<mt>>, orphan_signature>(
            allocator, count)}, _numActiveBlocks{0}, _ppcs{nullptr}, _ppbs{nullptr},
            _cellbuckets{nullptr}, _blockbuckets{nullptr}, _binsts{nullptr} {
              std::cout << "Constructing ParticleBufferImpl with buckets." << "\n";
            }
  template <typename Allocator>
  ParticleBufferImpl(Allocator allocator)
      : base_t{spawn<particle_buffer_<particle_bin_<mt>>, orphan_signature>(
            allocator)} {
              std::cout << "Constructing ParticleBufferImpl without buckets." << "\n";
            }

  // Check if particle buffer can hold n particle blocks, resize if not
  template <typename Allocator>
  void checkCapacity(Allocator allocator, std::size_t capacity) {
    if (capacity > this->_capacity)
      this->resize(allocator, capacity);
  }

  /// @brief Maps run-time string labels for any MPM particle attributes to int indices for GPU kernel output use
  /// @param n is a std::string containing a particle attribute label  
  /// @return Integer index assigned to particle attribute n, used in GPU kernels
  particle_output_attribs_e mapAttributeStringToIndex(const std::string& n) {
        using out_ = particle_output_attribs_e;
        if      (n == "ID") return out_:: ID; 
        else if (n == "Mass") return out_:: Mass;
        else if (n == "Volume") return out_:: Volume;
        else if (n == "Position_X") return out_:: Position_X; 
        else if (n == "Position_Y") return out_:: Position_Y;
        else if (n == "Position_Z") return out_:: Position_Z;
        else if (n == "Velocity_X") return out_:: Velocity_X;
        else if (n == "Velocity_Y") return out_:: Velocity_Y;
        else if (n == "Velocity_Z") return out_:: Velocity_Z;
        else if (n == "DefGrad_XX") return out_:: DefGrad_XX;
        else if (n == "DefGrad_XY") return out_:: DefGrad_XY;
        else if (n == "DefGrad_XZ") return out_:: DefGrad_XZ;
        else if (n == "DefGrad_YX") return out_:: DefGrad_YX;
        else if (n == "DefGrad_YY") return out_:: DefGrad_YY;
        else if (n == "DefGrad_YZ") return out_:: DefGrad_YZ;
        else if (n == "DefGrad_ZX") return out_:: DefGrad_ZX;
        else if (n == "DefGrad_ZY") return out_:: DefGrad_ZY;
        else if (n == "DefGrad_ZZ") return out_:: DefGrad_ZZ;
        else if (n == "J")          return out_:: J;
        else if (n == "JBar")       return out_:: JBar;
        else if (n == "StressCauchy_XX") return out_:: StressCauchy_XX;
        else if (n == "StressCauchy_XY") return out_:: StressCauchy_XY;
        else if (n == "StressCauchy_XZ") return out_:: StressCauchy_XZ;
        else if (n == "StressCauchy_YX") return out_:: StressCauchy_YX;
        else if (n == "StressCauchy_YY") return out_:: StressCauchy_YY;
        else if (n == "StressCauchy_YZ") return out_:: StressCauchy_YZ;
        else if (n == "StressCauchy_ZX") return out_:: StressCauchy_ZX;
        else if (n == "StressCauchy_ZY") return out_:: StressCauchy_ZY;
        else if (n == "StressCauchy_ZZ") return out_:: StressCauchy_ZZ;
        else if (n == "Pressure")        return out_:: Pressure;
        else if (n == "VonMisesStress")  return out_:: VonMisesStress;
        else if (n == "DefGrad_Invariant1") return out_:: DefGrad_Invariant1;
        else if (n == "DefGrad_Invariant2") return out_:: DefGrad_Invariant2;
        else if (n == "DefGrad_Invariant3") return out_:: DefGrad_Invariant3;
        else if (n == "DefGrad_1") return out_:: DefGrad_1;
        else if (n == "DefGrad_2") return out_:: DefGrad_2;
        else if (n == "DefGrad_3") return out_:: DefGrad_3;
        else if (n == "StressCauchy_Invariant1") return out_:: StressCauchy_Invariant1;
        else if (n == "StressCauchy_Invariant2") return out_:: StressCauchy_Invariant2;
        else if (n == "StressCauchy_Invariant3") return out_:: StressCauchy_Invariant3;
        else if (n == "StressCauchy_1") return out_:: StressCauchy_1;
        else if (n == "StressCauchy_2") return out_:: StressCauchy_2;
        else if (n == "StressCauchy_3") return out_:: StressCauchy_3;
        else if (n == "StressPK1_XX") return out_:: StressPK1_XX;
        else if (n == "StressPK1_XY") return out_:: StressPK1_XY;
        else if (n == "StressPK1_XZ") return out_:: StressPK1_XZ;
        else if (n == "StressPK1_YX") return out_:: StressPK1_YX;
        else if (n == "StressPK1_YY") return out_:: StressPK1_YY;
        else if (n == "StressPK1_YZ") return out_:: StressPK1_YZ;
        else if (n == "StressPK1_ZX") return out_:: StressPK1_ZX;
        else if (n == "StressPK1_ZY") return out_:: StressPK1_ZY;
        else if (n == "StressPK1_ZZ") return out_:: StressPK1_ZZ;
        else if (n == "StressPK1_Invariant1") return out_:: StressPK1_Invariant1;
        else if (n == "StressPK1_Invariant2") return out_:: StressPK1_Invariant2;
        else if (n == "StressPK1_Invariant3") return out_:: StressPK1_Invariant3;
        else if (n == "StressPK1_1") return out_:: StressPK1_1;
        else if (n == "StressPK1_2") return out_:: StressPK1_2;
        else if (n == "StressPK1_3") return out_:: StressPK1_3;
        else if (n == "StrainSmall_XX") return out_:: StrainSmall_XX;
        else if (n == "StrainSmall_XY") return out_:: StrainSmall_XY;
        else if (n == "StrainSmall_XZ") return out_:: StrainSmall_XZ;
        else if (n == "StrainSmall_YX") return out_:: StrainSmall_YX;
        else if (n == "StrainSmall_YY") return out_:: StrainSmall_YY;
        else if (n == "StrainSmall_YZ") return out_:: StrainSmall_YZ;
        else if (n == "StrainSmall_ZX") return out_:: StrainSmall_ZX;
        else if (n == "StrainSmall_ZY") return out_:: StrainSmall_ZY;
        else if (n == "StrainSmall_ZZ") return out_:: StrainSmall_ZZ;
        else if (n == "StrainSmall_Invariant1") return out_:: StrainSmall_Invariant1;
        else if (n == "StrainSmall_Invariant2") return out_:: StrainSmall_Invariant2;
        else if (n == "StrainSmall_Invariant3") return out_:: StrainSmall_Invariant3;
        else if (n == "StrainSmall_1") return out_:: StrainSmall_1;
        else if (n == "StrainSmall_2") return out_:: StrainSmall_2;
        else if (n == "StrainSmall_3") return out_:: StrainSmall_3;
        else if (n == "StrainSmall_Determinant")  return out_:: StrainSmall_Determinant;
        else if (n == "VonMisesStrain")  return out_:: VonMisesStrain;
        else if (n == "Dilation")  return out_:: Dilation;
        else if (n == "logJp") return out_:: logJp;
        else return out_:: INVALID_RT;
  }

  int track_ID = 0;
  vec<int, 1> track_attribs;
  std::vector<std::string> track_labels;   
  void updateTrack(std::vector<std::string> names, int trackID=0) {
    track_ID = trackID;
    int i = 0;
    for (auto n : names) {
      track_labels.emplace_back(n);
      track_attribs[i] = static_cast<int>(mapAttributeStringToIndex(n));
      i = i+1;
    }
  }

  vec<int, 3> output_attribs;
  vec<int, mn::config::g_max_particle_attribs> output_attribs_dyn;
  std::vector<std::string> output_labels;   
  void updateOutputs(std::vector<std::string> names) {
    int i = 0;
    for (auto n : names) {
      if (i>=mn::config::g_max_particle_attribs) continue;
      output_labels.emplace_back(n);
      output_attribs_dyn[i] = static_cast<int>(mapAttributeStringToIndex(n));
      if (i < 3) output_attribs[i] = static_cast<int>(mapAttributeStringToIndex(n));
      i++;
    }
  }

  vec<int, 1> target_attribs;
  std::vector<std::string> target_labels;   
  void updateTargets(std::vector<std::string> names) {
    int i = 0;
    for (auto n : names) {
      target_labels.emplace_back(n);
      target_attribs[i] = static_cast<int>(mapAttributeStringToIndex(n));
      i++;
    }
  }


  template <typename Allocator>
  void deallocateBuckets(Allocator allocator) {
    if (config::g_buckets_on_particle_buffer && _binsts) {
      allocator.deallocate(_ppcs, sizeof(int) * _numActiveBlocks *
                                            config::g_blockvolume);
      allocator.deallocate(_ppbs, sizeof(int) * _numActiveBlocks);
      allocator.deallocate(_cellbuckets, sizeof(int) * _numActiveBlocks *
                                            config::g_blockvolume *
                                            config::g_max_ppc);
      allocator.deallocate(_blockbuckets, sizeof(int) * _numActiveBlocks *
                                            config::g_particle_num_per_block);
      allocator.deallocate(_binsts, sizeof(int) * _numActiveBlocks);
    }
  }

  template <typename Allocator>
  void reserveBuckets(Allocator allocator, std::size_t numBlockCnt) {
    if (_binsts) {
      allocator.deallocate(_ppcs, sizeof(int) * _numActiveBlocks *
                                             config::g_blockvolume);
      allocator.deallocate(_ppbs, sizeof(int) * _numActiveBlocks);
      allocator.deallocate(_cellbuckets, sizeof(int) * _numActiveBlocks *
                                             config::g_blockvolume *
                                             config::g_max_ppc);
      allocator.deallocate(_blockbuckets, sizeof(int) * _numActiveBlocks *
                                             config::g_particle_num_per_block);
      allocator.deallocate(_binsts, sizeof(int) * _numActiveBlocks);
    }
    _numActiveBlocks = numBlockCnt;
    _ppcs = (int *)allocator.allocate(sizeof(int) * _numActiveBlocks *
                                             config::g_blockvolume);
    _ppbs = (int *)allocator.allocate(sizeof(int) * _numActiveBlocks);
    _cellbuckets = (int *)allocator.allocate(sizeof(int) * _numActiveBlocks *
                                             config::g_blockvolume * 
                                             config::g_max_ppc);
    _blockbuckets = (int *)allocator.allocate(sizeof(int) * _numActiveBlocks *
                                             config::g_particle_num_per_block);
    _binsts = (int *)allocator.allocate(sizeof(int) * _numActiveBlocks);
    resetPpcs();
  }
  // Reset particle per cell to zero on GPU
  void resetPpcs() {
    checkCudaErrors(cudaMemset( _ppcs, 0, sizeof(int) * _numActiveBlocks 
                                                      * config::g_blockvolume));
    printf("Reset particleBins._ppcs to zero.\n");
  }
  // Stream copy bin starts and particle per block to other buffer for next step
  void copy_to(ParticleBufferImpl &other, std::size_t blockCnt,
               cudaStream_t stream) const {
    checkCudaErrors(cudaMemcpyAsync(other._binsts, _binsts,
                                    sizeof(int) * (blockCnt + 1),
                                    cudaMemcpyDefault, stream));
    checkCudaErrors(cudaMemcpyAsync(other._ppbs, _ppbs, sizeof(int) * blockCnt,
                                    cudaMemcpyDefault, stream));
    printf("Copied particleBins._binsts to other._binsts.\n");
    printf("Copied particleBins._ppbs to other._ppbs.\n");
  }
  // 
  template<typename Partition> // May want to put this in mpmpm_kernels.cuh
  __forceinline__ __device__ void add_advection(Partition &table,
                                                ivec3 cellid,
                                                int dirtag,
                                                int pidib) noexcept {
    using namespace config;
    ivec3 blockid = cellid / g_blocksize;
    int blockno = table.query(blockid); // Get block number from 3D block ID
#if 1
    if (blockno == -1) {
      ivec3 offset{};
      dir_components(dirtag, offset);
      printf("loc(%d, %d, %d) dir(%d, %d, %d) pidib(%d)\n", cellid[0],
             cellid[1], cellid[2], offset[0], offset[1], offset[2], pidib);
      return;
    }
#endif
    int cellno = ((cellid[0] & g_blockmask) << (g_blockbits << 1)) |
                 ((cellid[1] & g_blockmask) << g_blockbits) |
                 (cellid[2] & g_blockmask);
    // +1 particle to particles per cell. pidic = old value particle ID in cell    
    int pidic = atomicAdd(_ppcs + blockno * g_blockvolume + cellno, 1); 
    _cellbuckets[blockno * g_particle_num_per_block + cellno * g_max_ppc +
                 pidic] = (dirtag * g_particle_num_per_block) | pidib; 
  }

  std::size_t _numActiveBlocks;
  int *_ppcs, *_ppbs;
  int *_cellbuckets, *_blockbuckets;
  int *_binsts;
};


template <material_e mt> struct ParticleBuffer;
template <>
struct ParticleBuffer<material_e::JFluid>
    : ParticleBufferImpl<material_e::JFluid> {
  using base_t = ParticleBufferImpl<material_e::JFluid>;
  PREC length = DOMAIN_LENGTH; // Domain total length [m] (scales volume, etc.)
  PREC rho = DENSITY; // Density [kg/m3]
  PREC volume = DOMAIN_VOLUME * ( 1.f / (1 << DOMAIN_BITS) / (1 << DOMAIN_BITS) /
                  (1 << DOMAIN_BITS) / MODEL_PPC); // Volume of Particle [m3]
  PREC mass = (volume * DENSITY); // Mass of particle [kg]
  PREC bulk = 5e6; //< Bulk Modulus [Pa]
  PREC gamma = 7.1f; //< Derivative Bulk w.r.t. Pressure
  PREC visco = 0.001f; //< Dynamic Viscosity, [Pa * s]
  bool use_ASFLIP = false; //< Use ASFLIP/PIC mixing? Default off.
  PREC alpha = 0.0;  //< FLIP/PIC Mixing Factor [0.1] -> [PIC, FLIP]
  PREC beta_min = 0.0; //< ASFLIP Minimum Position Correction Factor  
  PREC beta_max = 0.0; //< ASFLIP Maximum Position Correction Factor 
  PREC FBAR_ratio = 0.0; //< F-Bar Anti-locking mixing ratio (0 = None, 1 = Full)
  bool use_FEM = false; //< Use Finite Elements? Default off. Must set mesh
  bool use_FBAR = false; //< Use Simple F-Bar anti-locking? Default off.
  void updateParameters(PREC l, config::MaterialConfigs mat, 
                        config::AlgoConfigs algo) {
    length = l;
    rho = mat.rho;
    volume = length*length*length * ( 1.0 / (1 << DOMAIN_BITS) / (1 << DOMAIN_BITS) /
                    (1 << DOMAIN_BITS) / mat.ppc);
    mass = volume * mat.rho;
    bulk = mat.bulk;
    gamma = mat.gamma;
    visco = mat.visco;
    alpha = algo.ASFLIP_alpha;
    beta_min = algo.ASFLIP_beta_min;
    beta_max = algo.ASFLIP_beta_max;
    FBAR_ratio = algo.FBAR_ratio;
    use_ASFLIP = algo.use_ASFLIP;
    use_FEM = algo.use_FEM;
    use_FBAR = algo.use_FBAR;
  }

  // * Attributes saved on particles of this material. Given variable names for easy mapping
  // * REQUIRED : Variable order matches atttribute order in ParticleBuffer.val_1d(VARIABLE, ...)
  // * e.g. if ParticleBuffer<MATERIAL>.val_1d(4_, ...) is Velocity_X, then set Velocity_X = 4
  // * REQUIRED : Define material's unused base variables after END to avoid errors.
  // TODO : Write unit-test to guarantee all attribs_e have base set of variables.
  enum attribs_e : int {
          EMPTY=-3, // Empty attribute request 
          INVALID_CT=-2, // Invalid compile-time request, e.g. asking for variable after END
          INVALID_RT=-1, // Invalid run-time request e.g. "Speed_X" instead of "Velocity_X"
          START=0, // Values less than or equal to START not held on particle
          Position_X=0, Position_Y=1, Position_Z=2,
          J, DefGrad_Determinant=J, 
          END, // Values greater than or equal to END not held on particle
          // REQUIRED: Put N/A variables for specific material below END
          DefGrad_XX, DefGrad_XY, DefGrad_XZ,
          DefGrad_YX, DefGrad_YY, DefGrad_YZ,
          DefGrad_ZX, DefGrad_ZY, DefGrad_ZZ,
          Velocity_X, Velocity_Y, Velocity_Z,
          Volume_FBAR, 
          JBar, DefGrad_Determinant_FBAR=JBar, 
          ID,
          logJp
  };


  // TODO : Change if/else statement to case/switch. may require compile-time min-max guarantee
  template <attribs_e ATTRIBUTE, typename T>
   __device__ PREC
  getAttribute(const T bin, const T particle_id_in_bin){
    if (ATTRIBUTE < attribs_e::START) return (PREC)ATTRIBUTE;
    else if (ATTRIBUTE >= attribs_e::END) return (PREC)attribs_e::INVALID_CT;
    else return this->ch(std::integral_constant<unsigned, 0>{}, bin).val_1d(std::integral_constant<unsigned, std::min(abs(ATTRIBUTE), attribs_e::END-1)>{}, particle_id_in_bin);
  }

  template <typename T>
   __device__ constexpr void
  getDefGrad(T bin, T particle_id_in_bin, PREC& J) {
    J = 1.0 - this->ch(std::integral_constant<unsigned, 0>{}, bin).val_1d(std::integral_constant<unsigned, attribs_e::J>{}, particle_id_in_bin);
  }
  template <typename T>
   __device__ constexpr void
  getDefGrad(T bin, T particle_id_in_bin, PREC * DefGrad) { 
    PREC DefGrad_Det_cbrt = cbrt(1.0 - this->ch(std::integral_constant<unsigned, 0>{}, bin).val_1d(std::integral_constant<unsigned, attribs_e::J>{}, particle_id_in_bin));
    DefGrad[0] = DefGrad[4] = DefGrad[8] = DefGrad_Det_cbrt;
    DefGrad[1] = DefGrad[2] = DefGrad[3] = DefGrad[5] = DefGrad[6] = DefGrad[7] = 0.; 
  }

  template <typename T = PREC>
   __device__ void
  getPressure(T J, T& pressure){
    compute_pressure_jfluid(volume, bulk, gamma, J, pressure);
  }

  template <typename T = PREC>
   __device__ constexpr void
  getStress_Cauchy(const vec<T,9>& F, vec<T,9>& PF){
    //compute_stress_PK1_jfluid(volume, bulk, gamma, F, P);
    PREC Jp = F[0]*F[4]*F[8];
    PREC pressure;
    compute_pressure_jfluid(volume, bulk, gamma, Jp, pressure);
    PF[0] = PF[4] = PF[8] = pressure;
  }
  template <typename T = PREC>
   __device__ constexpr void
  getStress_Cauchy(T vol, const vec<T,9>& F, vec<T,9>& PF){
    //compute_stress_PK1_jfluid(vol, bulk, gamma, F, P);
    PREC Jp = F[0]*F[4]*F[8];
    PREC pressure;
    compute_pressure_jfluid(vol, bulk, gamma, Jp, pressure);
    PF[0] = PF[4] = PF[8] = pressure;
  }

  template <typename T = PREC>
   __device__ void
  getStrainEnergy(T J, T& strain_energy){
    compute_energy_jfluid(volume, bulk, gamma, J, strain_energy);
  }

  template <typename Allocator>
  ParticleBuffer(Allocator allocator) : base_t{allocator} {}

  template <typename Allocator>
  ParticleBuffer(Allocator allocator, std::size_t count)
      : base_t{allocator, count} {}
};

template <>
struct ParticleBuffer<material_e::JFluid_ASFLIP>
    : ParticleBufferImpl<material_e::JFluid_ASFLIP> {
  using base_t = ParticleBufferImpl<material_e::JFluid_ASFLIP>;
  PREC length = DOMAIN_LENGTH; // Domain total length [m] (scales volume, etc.)
  PREC rho = DENSITY; // Density [kg/m3]
  PREC volume = DOMAIN_VOLUME * ( 1.f / (1 << DOMAIN_BITS) / (1 << DOMAIN_BITS) /
                  (1 << DOMAIN_BITS) / MODEL_PPC); // Volume of Particle [m3]
  PREC mass = (volume * DENSITY); // Mass of particle [kg]
  PREC bulk = 2.2e9; //< Bulk Modulus [Pa]
  PREC gamma = 7.1; //< Derivative Bulk w.r.t. Pressure
  PREC visco = 0.001; //< Dynamic Viscosity, [Pa * s]
  bool use_ASFLIP = false; //< Use ASFLIP/PIC mixing? Default off.
  PREC alpha = 0.0;  //< FLIP/PIC Mixing Factor [0.1] -> [PIC, FLIP]
  PREC beta_min = 0.0; //< ASFLIP Minimum Position Correction Factor  
  PREC beta_max = 0.0; //< ASFLIP Maximum Position Correction Factor 
  PREC FBAR_ratio = 0.0; //< F-Bar Anti-locking mixing ratio (0 = None, 1 = Full)
  bool use_FEM = false; //< Use Finite Elements? Default off. Must set mesh
  bool use_FBAR = false; //< Use Simple F-Bar anti-locking? Default off.
  void updateParameters(PREC l, config::MaterialConfigs mat, 
                        config::AlgoConfigs algo) {
    length = l;
    rho = mat.rho;
    volume = length*length*length * ( 1.f / (1 << DOMAIN_BITS) / (1 << DOMAIN_BITS) /
                    (1 << DOMAIN_BITS) / mat.ppc);
    mass = volume * mat.rho;
    bulk = mat.bulk;
    gamma = mat.gamma;
    visco = mat.visco;
    alpha = algo.ASFLIP_alpha;
    beta_min = algo.ASFLIP_beta_min;
    beta_max = algo.ASFLIP_beta_max;
    FBAR_ratio = algo.FBAR_ratio;
    use_ASFLIP = algo.use_ASFLIP;
    use_FEM = algo.use_FEM;
    use_FBAR = algo.use_FBAR;
  }

  // * Attributes saved on particles of this material. Given variable names for easy mapping
  // * REQUIRED : Variable order matches atttribute order in ParticleBuffer.val_1d(VARIABLE, ...)
  // * e.g. if ParticleBuffer<MATERIAL>.val_1d(4_, ...) is Velocity_X, then set Velocity_X = 4
  // * REQUIRED : Define material's unused base variables after END to avoid errors.
  // TODO : Write unit-test to guarantee all attribs_e have base set of variables.
  enum attribs_e : int {
          EMPTY=-3, // Empty attribute request 
          INVALID_CT=-2, // Invalid compile-time request, e.g. asking for variable after END
          INVALID_RT=-1, // Invalid run-time request e.g. "Speed_X" instead of "Velocity_X"
          START=0, // Values less than or equal to START not held on particle
          Position_X=0, Position_Y=1, Position_Z=2,
          J, DefGrad_Determinant=J, 
          Velocity_X, Velocity_Y, Velocity_Z,
          END, // Values greater than or equal to END not held on particle
          // REQUIRED: Put N/A variables for specific material below END
          DefGrad_XX, DefGrad_XY, DefGrad_XZ,
          DefGrad_YX, DefGrad_YY, DefGrad_YZ,
          DefGrad_ZX, DefGrad_ZY, DefGrad_ZZ,
          Volume_FBAR, 
          JBar, DefGrad_Determinant_FBAR=JBar, 
          ID,
          logJp
  };


  // TODO : Change if/else statement to case/switch. may require compile-time min-max guarantee
  template <attribs_e ATTRIBUTE, typename T>
   __device__ PREC
  getAttribute(const T bin, const T particle_id_in_bin){
    if (ATTRIBUTE < attribs_e::START) return (PREC)ATTRIBUTE;
    else if (ATTRIBUTE >= attribs_e::END) return (PREC)attribs_e::INVALID_CT;
    else return this->ch(std::integral_constant<unsigned, 0>{}, bin).val_1d(std::integral_constant<unsigned, std::min(abs(ATTRIBUTE), attribs_e::END-1)>{}, particle_id_in_bin);
  }

  template <typename T>
   __device__ constexpr void
  getDefGrad(T bin, T particle_id_in_bin, PREC& J) {
    J = 1.0 - this->ch(std::integral_constant<unsigned, 0>{}, bin).val_1d(std::integral_constant<unsigned, attribs_e::J>{}, particle_id_in_bin);
  }
  template <typename T>
   __device__ constexpr void
  getDefGrad(T bin, T particle_id_in_bin, PREC * DefGrad) {  
    PREC DefGrad_Det_cbrt = cbrt(1.0 - this->ch(std::integral_constant<unsigned, 0>{}, bin).val_1d(std::integral_constant<unsigned, attribs_e::J>{}, particle_id_in_bin));
    DefGrad[0] = DefGrad[4] = DefGrad[8] = DefGrad_Det_cbrt;
    DefGrad[1] = DefGrad[2] = DefGrad[3] = DefGrad[5] = DefGrad[6] = DefGrad[7] = 0.; 
  }

  template <typename T = PREC>
   __device__ void
  getPressure(T J, T& pressure){
    compute_pressure_jfluid(volume, bulk, gamma, J, pressure);
  }

  template <typename T = PREC>
   __device__ constexpr void
  getStress_Cauchy(const vec<T,9>& F, vec<T,9>& PF){
    //compute_stress_PK1_jfluid(volume, bulk, gamma, F, P);
    PREC Jp = F[0]*F[4]*F[8];
    PREC pressure;
    compute_pressure_jfluid(volume, bulk, gamma, Jp, pressure);
    PF[0] = PF[4] = PF[8] = pressure;
  }
  template <typename T = PREC>
   __device__ constexpr void
  getStress_Cauchy(T vol, const vec<T,9>& F, vec<T,9>& PF){
    //compute_stress_PK1_jfluid(vol, bulk, gamma, F, P);
    PREC Jp = F[0]*F[4]*F[8];
    PREC pressure;
    compute_pressure_jfluid(vol, bulk, gamma, Jp, pressure);
    PF[0] = PF[4] = PF[8] = pressure;
  }

  template <typename T = PREC>
   __device__ void
  getStrainEnergy(T J, T& strain_energy){
    compute_energy_jfluid(volume, bulk, gamma, J, strain_energy);
  }

  template <typename Allocator>
  ParticleBuffer(Allocator allocator) : base_t{allocator} {}

  template <typename Allocator>
  ParticleBuffer(Allocator allocator, std::size_t count)
      : base_t{allocator, count} {}
};


template <>
struct ParticleBuffer<material_e::JFluid_FBAR>
    : ParticleBufferImpl<material_e::JFluid_FBAR> {
  using base_t = ParticleBufferImpl<material_e::JFluid_FBAR>;
  PREC length = DOMAIN_LENGTH; // Domain total length [m] (scales volume, etc.)
  PREC rho = DENSITY; // Density [kg/m3]
  PREC volume = DOMAIN_VOLUME * ( 1.0 / (1 << DOMAIN_BITS) / (1 << DOMAIN_BITS) /
                  (1 << DOMAIN_BITS) / MODEL_PPC); // Volume of Particle [m3]
  PREC mass = (volume * DENSITY); // Mass of particle [kg]
  PREC bulk = 2.2e9; //< Bulk Modulus [Pa]
  PREC gamma = 7.1; //< Derivative Bulk w.r.t. Pressure
  PREC visco = 0.001; //< Dynamic Viscosity, [Pa * s]
  
  bool use_ASFLIP = false; //< Use ASFLIP/PIC mixing? Default off.
  PREC alpha = 0.0;  //< FLIP/PIC Mixing Factor [0.1] -> [PIC, FLIP]
  PREC beta_min = 0.0; //< ASFLIP Minimum Position Correction Factor  
  PREC beta_max = 0.0; //< ASFLIP Maximum Position Correction Factor 
  PREC FBAR_ratio = 0.0; //< F-Bar Anti-locking mixing ratio (0 = None, 1 = Full)
  bool use_FEM = false; //< Use Finite Elements? Default off. Must set mesh
  bool use_FBAR = false; //< Use Simple F-Bar anti-locking? Default off.

  void updateParameters(PREC l, config::MaterialConfigs mat, 
                        config::AlgoConfigs algo) {
    length = l;
    rho = mat.rho;
    volume = length*length*length * (1.0 / (1 << DOMAIN_BITS) / (1 << DOMAIN_BITS) /
                    (1 << DOMAIN_BITS)) / mat.ppc;
    mass = volume * mat.rho;
    bulk = mat.bulk;
    gamma = mat.gamma;
    visco = mat.visco;
    alpha = algo.ASFLIP_alpha;
    beta_min = algo.ASFLIP_beta_min;
    beta_max = algo.ASFLIP_beta_max;
    FBAR_ratio = algo.FBAR_ratio;
    use_ASFLIP = algo.use_ASFLIP;
    use_FEM = algo.use_FEM;
    use_FBAR = algo.use_FBAR;
  }  

  // * Attributes saved on particles of this material. Given variable names for easy mapping
  // * REQUIRED : Variable order matches atttribute order in ParticleBuffer.val_1d(VARIABLE, ...)
  // * e.g. if ParticleBuffer<MATERIAL>.val_1d(4_, ...) is Velocity_X, then set Velocity_X = 4
  // * REQUIRED : Define material's unused base variables after END to avoid errors.
  // TODO : Write unit-test to guarantee all attribs_e have base set of variables.
  enum attribs_e : int {
          EMPTY=-3, // Empty attribute request 
          INVALID_CT=-2, // Invalid compile-time request, e.g. asking for variable after END
          INVALID_RT=-1, // Invalid run-time request e.g. "Speed_X" instead of "Velocity_X"
          START=0, // Values less than or equal to START not held on particle
          Position_X=0, Position_Y=1, Position_Z=2,
          J, DefGrad_Determinant=J, 
          JBar, DefGrad_Determinant_FBAR=JBar, 
          ID,
          END, // Values greater than or equal to END not held on particle
          // REQUIRED: Put N/A variables for specific material below END
          DefGrad_XX, DefGrad_XY, DefGrad_XZ,
          DefGrad_YX, DefGrad_YY, DefGrad_YZ,
          DefGrad_ZX, DefGrad_ZY, DefGrad_ZZ,
          Velocity_X, Velocity_Y, Velocity_Z,
          Volume_FBAR, 
          logJp
  };


  // TODO : Change if/else statement to case/switch. may require compile-time min-max guarantee
  template <attribs_e ATTRIBUTE, typename T>
   __device__ PREC
  getAttribute(const T bin, const T particle_id_in_bin){
    if (ATTRIBUTE < attribs_e::START) return (PREC)ATTRIBUTE;
    else if (ATTRIBUTE >= attribs_e::END) return (PREC)attribs_e::INVALID_CT;
    else return this->ch(std::integral_constant<unsigned, 0>{}, bin).val_1d(std::integral_constant<unsigned, std::min(abs(ATTRIBUTE), attribs_e::END-1)>{}, particle_id_in_bin);
  }

  template <typename T>
   __device__ constexpr void
  getDefGrad(const T bin, const T particle_id_in_bin, PREC& J) {
    J = 1.0 - this->ch(std::integral_constant<unsigned, 0>{}, bin).val_1d(std::integral_constant<unsigned, attribs_e::J>{}, particle_id_in_bin);
  }
  template <typename T>
   __device__ constexpr void
  getDefGrad(T bin, T particle_id_in_bin, PREC * DefGrad) { 
    PREC DefGrad_Det_cbrt = cbrt(1.0 - this->ch(std::integral_constant<unsigned, 0>{}, bin).val_1d(std::integral_constant<unsigned, attribs_e::J>{}, particle_id_in_bin));
    DefGrad[0] = DefGrad[4] = DefGrad[8] = DefGrad_Det_cbrt;
    DefGrad[1] = DefGrad[2] = DefGrad[3] = DefGrad[5] = DefGrad[6] = DefGrad[7] = 0.; 
  }

  template <typename T = PREC>
   __device__ constexpr void
  getStress_Cauchy(const vec<T,9>& F, vec<T,9>& PF){
    //compute_stress_PK1_jfluid(volume, bulk, gamma, F, P);
    PREC Jp = F[0]*F[4]*F[8];
    PREC pressure;
    compute_pressure_jfluid(volume, bulk, gamma, Jp, pressure);
    PF[0] = PF[4] = PF[8] = pressure;
  }
  template <typename T = PREC>
   __device__ constexpr void
  getStress_Cauchy(T vol, const vec<T,9>& F, vec<T,9>& PF){
    //compute_stress_PK1_jfluid(vol, bulk, gamma, F, P);
    PREC Jp = 1. - F[0]*F[4]*F[8];
    PREC pressure;
    compute_pressure_jfluid(vol, bulk, gamma, Jp, pressure);
    PF[0] = PF[4] = PF[8] = pressure;
  }

  template <typename T = PREC>
   __device__ void
  getPressure(T sJ, T& pressure){
    compute_pressure_jfluid(volume, bulk, gamma, sJ, pressure);
  }

  template <typename T = PREC>
   __device__ void
  getStrainEnergy(T J, T& strain_energy){
    compute_energy_jfluid(volume, bulk, gamma, J, strain_energy);
  }

  template <typename Allocator>
  ParticleBuffer(Allocator allocator) : base_t{allocator} {}

  template <typename Allocator>
  ParticleBuffer(Allocator allocator, std::size_t count)
      : base_t{allocator, count} {}
};


template <>
struct ParticleBuffer<material_e::JBarFluid>
    : ParticleBufferImpl<material_e::JBarFluid> {
  using base_t = ParticleBufferImpl<material_e::JBarFluid>;
  PREC length = DOMAIN_LENGTH; // Domain total length [m] (scales volume, etc.)
  PREC rho = DENSITY; // Density [kg/m3]
  PREC volume = DOMAIN_VOLUME * ( 1.0 / (1 << DOMAIN_BITS) / (1 << DOMAIN_BITS) /
                  (1 << DOMAIN_BITS) / MODEL_PPC); // Volume of Particle [m3]
  PREC mass = (volume * DENSITY); // Mass of particle [kg]
  PREC bulk = 2.2e9; //< Bulk Modulus [Pa]
  PREC gamma = 7.1; //< Derivative Bulk w.r.t. Pressure
  PREC visco = 0.001; //< Dynamic Viscosity, [Pa * s]
  
  bool use_ASFLIP = false; //< Use ASFLIP/PIC mixing? Default off.
  PREC alpha = 0.0;  //< FLIP/PIC Mixing Factor [0.1] -> [PIC, FLIP]
  PREC beta_min = 0.0; //< ASFLIP Minimum Position Correction Factor  
  PREC beta_max = 0.0; //< ASFLIP Maximum Position Correction Factor 
  PREC FBAR_ratio = 0.0; //< F-Bar Anti-locking mixing ratio (0 = None, 1 = Full)
  bool use_FEM = false; //< Use Finite Elements? Default off. Must set mesh
  bool use_FBAR = false; //< Use Simple F-Bar anti-locking? Default off.

  void updateParameters(PREC l, config::MaterialConfigs mat, 
                        config::AlgoConfigs algo) {
    length = l;
    volume = length*length*length * (1.0 / (1 << DOMAIN_BITS) / (1 << DOMAIN_BITS) /
                    (1 << DOMAIN_BITS)) / mat.ppc;
    rho = mat.rho;
    mass = volume * mat.rho;
    bulk = mat.bulk;
    gamma = mat.gamma;
    visco = mat.visco;
    alpha = algo.ASFLIP_alpha;
    beta_min = algo.ASFLIP_beta_min;
    beta_max = algo.ASFLIP_beta_max;
    FBAR_ratio = algo.FBAR_ratio;
    use_ASFLIP = algo.use_ASFLIP;
    use_FEM = algo.use_FEM;
    use_FBAR = algo.use_FBAR;
  }  

  // * Attributes saved on particles of this material. Given variable names for easy mapping
  // * REQUIRED : Variable order matches atttribute order in ParticleBuffer.val_1d(VARIABLE, ...)
  // * e.g. if ParticleBuffer<MATERIAL>.val_1d(4_, ...) is Velocity_X, then set Velocity_X = 4
  // * REQUIRED : Define material's unused base variables after END to avoid errors.
  // TODO : Write unit-test to guarantee all attribs_e have base set of variables.
  enum attribs_e : int {
          EMPTY=-3, // Empty attribute request 
          INVALID_CT=-2, // Invalid compile-time request, e.g. asking for variable after END
          INVALID_RT=-1, // Invalid run-time request e.g. "Speed_X" instead of "Velocity_X"
          START=0, // Values less than or equal to START not held on particle
          Position_X=0, Position_Y=1, Position_Z=2,
          J, DefGrad_Determinant=J, 
          Velocity_X, Velocity_Y, Velocity_Z,
          ID,
          JBar, DefGrad_Determinant_FBAR=JBar, 
          END, // Values greater than or equal to END not held on particle
          // REQUIRED: Put N/A variables for specific material below END
          DefGrad_XX, DefGrad_XY, DefGrad_XZ,
          DefGrad_YX, DefGrad_YY, DefGrad_YZ,
          DefGrad_ZX, DefGrad_ZY, DefGrad_ZZ,
          Volume_FBAR, 
          logJp
  };

  // TODO : Change if/else statement to case/switch. may require compile-time min-max guarantee
  template <attribs_e ATTRIBUTE, typename T>
   __device__ PREC
  getAttribute(const T bin, const T particle_id_in_bin){
    if (ATTRIBUTE < attribs_e::START) return (PREC)ATTRIBUTE;
    else if (ATTRIBUTE >= attribs_e::END) return (PREC)attribs_e::INVALID_CT;
    else return this->ch(std::integral_constant<unsigned, 0>{}, bin).val_1d(std::integral_constant<unsigned, std::min(abs(ATTRIBUTE), attribs_e::END-1)>{}, particle_id_in_bin);
  }
  
  template <typename T = PREC>
   __device__ void
  getVelocity(const T bin, const T particle_id_in_bin, PREC * velocity) {
    velocity = {
      this->ch(std::integral_constant<unsigned, 0>{}, bin).val_1d(std::integral_constant<unsigned, attribs_e::Velocity_X>{}, particle_id_in_bin),
      this->ch(std::integral_constant<unsigned, 0>{}, bin).val_1d(std::integral_constant<unsigned, attribs_e::Velocity_Y>{}, particle_id_in_bin),
      this->ch(std::integral_constant<unsigned, 0>{}, bin).val_1d(std::integral_constant<unsigned, attribs_e::Velocity_Z>{}, particle_id_in_bin)
    };
  }

  template <typename T>
   __device__ constexpr void
  getDefGrad(T bin, T particle_id_in_bin, PREC & J) {
    J = 1.0 - this->ch(std::integral_constant<unsigned, 0>{}, bin).val_1d(std::integral_constant<unsigned, attribs_e::J>{}, particle_id_in_bin);
  }

  template <typename T>
   __device__ constexpr void
  getDefGrad(T bin, T particle_id_in_bin, PREC * DefGrad) {
    PREC DefGrad_Det_cbrt = cbrt(1.0 - this->ch(std::integral_constant<unsigned, 0>{}, bin).val_1d(std::integral_constant<unsigned, attribs_e::J>{}, particle_id_in_bin));
    DefGrad[0] = DefGrad[4] = DefGrad[8] = DefGrad_Det_cbrt;
    DefGrad[1] = DefGrad[2] = DefGrad[3] = DefGrad[5] = DefGrad[6] = DefGrad[7] = 0.; 
  }
  template <typename T = PREC>
   __device__ void
  getPressure(const T bin, const T particle_id_in_bin, PREC& pressure){
    PREC Jp = 1.0 - this->ch(std::integral_constant<unsigned, 0>{}, bin).val_1d(std::integral_constant<unsigned, attribs_e::J>{}, particle_id_in_bin);
    compute_pressure_jfluid(volume, bulk, gamma, Jp, pressure);
  }

  // TODO : Make getStress accurate to JFluid for APIC
  template <typename T = PREC>
   __device__ void
  getStress_PK1(const vec<T,9>& F, vec<T,9>& P){
    //compute_stress_PK1_jfluid(volume, bulk, gamma, F, P);
    PREC Jp = F[0]*F[4]*F[8];
    PREC pressure;
    compute_pressure_jfluid(volume, bulk, gamma, Jp, pressure);
    P[0] = P[4] = P[8] = pressure;
  }

  template <typename T = PREC>
   __device__ void
  getStress_PK1(T vol, const vec<T,9>& F, vec<T,9>& P){
    //compute_stress_PK1_jfluid(vol, bulk, gamma, F, P);
    PREC Jp = F[0]*F[4]*F[8];
    PREC pressure;
    compute_pressure_jfluid(vol, bulk, gamma, Jp, pressure);
    P[0] = P[4] = P[8] = pressure;
  }

  template <typename T = PREC>
   __device__ constexpr void
  getStress_Cauchy(const vec<T,9>& F, vec<T,9>& PF){
    //compute_stress_PK1_jfluid(volume, bulk, gamma, F, P);
    PREC Jp = F[0]*F[4]*F[8];
    PREC pressure;
    compute_pressure_jfluid(volume, bulk, gamma, Jp, pressure);
    PF[0] = PF[4] = PF[8] = pressure;
  }
  template <typename T = PREC>
   __device__ constexpr void
  getStress_Cauchy(T vol, const vec<T,9>& F, vec<T,9>& PF){
    //compute_stress_PK1_jfluid(vol, bulk, gamma, F, P);
    PREC Jp = F[0]*F[4]*F[8];
    PREC pressure;
    compute_pressure_jfluid(vol, bulk, gamma, Jp, pressure);
    PF[0] = PF[4] = PF[8] = pressure;
  }
  
  template <typename T = PREC>
   __device__ void
  getStrainEnergy(T J, T& strain_energy){
    compute_energy_jfluid(volume, bulk, gamma, J, strain_energy);
  }

  template <typename T = PREC>
   __device__ void
  getEnergy_Kinetic(const vec<T,3>& velocity, T& kinetic_energy){  
    kinetic_energy = 0.5 * mass * __fma_rn(velocity[0], velocity[0], __fma_rn(velocity[1], velocity[1], (velocity[2], velocity[2])));
    }

  template <typename Allocator>
  ParticleBuffer(Allocator allocator) : base_t{allocator} {}

  template <typename Allocator>
  ParticleBuffer(Allocator allocator, std::size_t count)
      : base_t{allocator, count} {}
};

template <>
struct ParticleBuffer<material_e::FixedCorotated>
    : ParticleBufferImpl<material_e::FixedCorotated> {
  using base_t = ParticleBufferImpl<material_e::FixedCorotated>;
  PREC length = DOMAIN_LENGTH; // Domain total length [m] (scales volume, etc.)
  PREC rho = DENSITY;
  PREC volume = DOMAIN_VOLUME * (1.f / (1 << DOMAIN_BITS) / (1 << DOMAIN_BITS) /
                  (1 << DOMAIN_BITS) / MODEL_PPC);
  PREC mass = (volume * DENSITY);
  PREC E = YOUNGS_MODULUS;
  PREC nu = POISSON_RATIO;
  PREC lambda = YOUNGS_MODULUS * POISSON_RATIO /
                 ((1 + POISSON_RATIO) * (1 - 2 * POISSON_RATIO));
  PREC mu = YOUNGS_MODULUS / (2 * (1 + POISSON_RATIO));
  bool use_ASFLIP = false; //< Use ASFLIP/PIC mixing? Default off.
  PREC alpha = 0.0;  //< FLIP/PIC Mixing Factor [0.1] -> [PIC, FLIP]
  PREC beta_min = 0.0; //< ASFLIP Minimum Position Correction Factor  
  PREC beta_max = 0.0; //< ASFLIP Maximum Position Correction Factor 
  PREC FBAR_ratio = 0.0; //< F-Bar Anti-locking mixing ratio (0 = None, 1 = Full)
  bool use_FEM = false; //< Use Finite Elements? Default off. Must set mesh
  bool use_FBAR = false; //< Use Simple F-Bar anti-locking? Default off.
  void updateParameters(PREC l, config::MaterialConfigs mat, 
                        config::AlgoConfigs algo) {
    length = l;
    rho = mat.rho;
    volume = length*length*length * ( 1.f / (1 << DOMAIN_BITS) / (1 << DOMAIN_BITS) /
                    (1 << DOMAIN_BITS) / mat.ppc);
    mass = volume * mat.rho;
    lambda = mat.E * mat.nu / ((1 + mat.nu) * (1 - 2 * mat.nu));
    mu = mat.E / (2 * (1 + mat.nu));
    alpha = algo.ASFLIP_alpha;
    beta_min = algo.ASFLIP_beta_min;
    beta_max = algo.ASFLIP_beta_max;
    FBAR_ratio = algo.FBAR_ratio;
    use_ASFLIP = algo.use_ASFLIP;
    use_FEM = algo.use_FEM;
    use_FBAR = algo.use_FBAR;
  }

  // * Attributes saved on particles of this material. Given variable names for easy mapping
  // * REQUIRED : Variable order matches atttribute order in ParticleBuffer.val_1d(VARIABLE, ...)
  // * e.g. if ParticleBuffer<MATERIAL>.val_1d(4_, ...) is Velocity_X, then set Velocity_X = 4
  // * REQUIRED : Define material's unused base variables after END to avoid errors.
  // TODO : Write unit-test to guarantee all attribs_e have base set of variables.
  enum attribs_e : int {
          EMPTY=-3, // Empty attribute request 
          INVALID_CT=-2, // Invalid compile-time request, e.g. asking for variable after END
          INVALID_RT=-1, // Invalid run-time request e.g. "Speed_X" instead of "Velocity_X"
          START=0, // Values less than or equal to START not held on particle
          Position_X=0, Position_Y=1, Position_Z=2,
          DefGrad_XX, DefGrad_XY, DefGrad_XZ,
          DefGrad_YX, DefGrad_YY, DefGrad_YZ,
          DefGrad_ZX, DefGrad_ZY, DefGrad_ZZ,
          ID,
          END, // Values greater than or equal to END not held on particle
          // REQUIRED: Put N/A variables for specific material below END
          J, DefGrad_Determinant=J, 
          Velocity_X, Velocity_Y, Velocity_Z,
          Volume_FBAR, JBar, DefGrad_Determinant_FBAR=JBar, 
          logJp
  };


  // TODO : Change if/else statement to case/switch. may require compile-time min-max guarantee
  template <attribs_e ATTRIBUTE, typename T>
   __device__ PREC
  getAttribute(const T bin, const T particle_id_in_bin){
    if (ATTRIBUTE < attribs_e::START) return (PREC)ATTRIBUTE;
    else if (ATTRIBUTE >= attribs_e::END) return (PREC)attribs_e::INVALID_CT;
    else return this->ch(std::integral_constant<unsigned, 0>{}, bin).val_1d(std::integral_constant<unsigned, std::min(abs(ATTRIBUTE), attribs_e::END-1)>{}, particle_id_in_bin);
  }
  
  template <typename T = PREC>
   __device__ void
  getVelocity(const T bin, const T particle_id_in_bin, PREC * velocity) {
    velocity[0] = velocity[1] = velocity[2] = 0.;
  }

  template <typename T>
   __device__ constexpr void
  getDefGrad(T bin, T particle_id_in_bin, PREC * DefGrad) {
    DefGrad[0] = this->ch(std::integral_constant<unsigned, 0>{}, bin).val_1d(std::integral_constant<unsigned, attribs_e::DefGrad_XX>{}, particle_id_in_bin);
    DefGrad[1] = this->ch(std::integral_constant<unsigned, 0>{}, bin).val_1d(std::integral_constant<unsigned, attribs_e::DefGrad_XY>{}, particle_id_in_bin);
    DefGrad[2] = this->ch(std::integral_constant<unsigned, 0>{}, bin).val_1d(std::integral_constant<unsigned, attribs_e::DefGrad_XZ>{}, particle_id_in_bin);
    DefGrad[3] = this->ch(std::integral_constant<unsigned, 0>{}, bin).val_1d(std::integral_constant<unsigned, attribs_e::DefGrad_YX>{}, particle_id_in_bin);
    DefGrad[4] = this->ch(std::integral_constant<unsigned, 0>{}, bin).val_1d(std::integral_constant<unsigned, attribs_e::DefGrad_YY>{}, particle_id_in_bin);
    DefGrad[5] = this->ch(std::integral_constant<unsigned, 0>{}, bin).val_1d(std::integral_constant<unsigned, attribs_e::DefGrad_YZ>{}, particle_id_in_bin);
    DefGrad[6] = this->ch(std::integral_constant<unsigned, 0>{}, bin).val_1d(std::integral_constant<unsigned, attribs_e::DefGrad_ZX>{}, particle_id_in_bin);
    DefGrad[7] = this->ch(std::integral_constant<unsigned, 0>{}, bin).val_1d(std::integral_constant<unsigned, attribs_e::DefGrad_ZY>{}, particle_id_in_bin);
    DefGrad[8] = this->ch(std::integral_constant<unsigned, 0>{}, bin).val_1d(std::integral_constant<unsigned, attribs_e::DefGrad_ZZ>{}, particle_id_in_bin);
  }

  template <typename T = PREC>
   __device__ constexpr void
  getStress_Cauchy(const vec<T,9>& F, vec<T,9>& PF){
    compute_stress_fixedcorotated(volume, mu, lambda, F, PF);
  }
  template <typename T = PREC>
   __device__ constexpr void
  getStress_Cauchy(T vol, const vec<T,9>& F, vec<T,9>& PF){
    compute_stress_fixedcorotated(vol, mu, lambda, F, PF);
  }
  
  template <typename T = PREC>
   __device__ void
  getStress_PK1(const vec<T,9>& F, vec<T,9>& P){
    compute_stress_PK1_fixedcorotated(volume, mu, lambda, F, P);
  }
  template <typename T = PREC>
   __device__ void
  getStress_PK1(T vol, const vec<T,9>& F, vec<T,9>& P){
    compute_stress_PK1_fixedcorotated(vol, mu, lambda, F, P);
  }

  template <typename T = PREC>
   __device__ void
  getEnergy_Strain(const vec<T,9>& F, T& strain_energy, T vol){
    compute_energy_fixedcorotated(vol, mu, lambda, F, strain_energy);
  }
  template <typename T = PREC>
   __device__ void
  getEnergy_Strain(const vec<T,9>& F, T& strain_energy){
    compute_energy_fixedcorotated(volume, mu, lambda, F, strain_energy);
  }
  template <typename T = PREC>
   __device__ void
  getEnergy_Kinetic(const vec<T,3>& velocity, T& kinetic_energy){  
    kinetic_energy = 0.5 * mass * __fma_rn(velocity[0], velocity[0], __fma_rn(velocity[1], velocity[1], (velocity[2], velocity[2])));
    }

  template <typename Allocator>
  ParticleBuffer(Allocator allocator) : base_t{allocator} {}

  template <typename Allocator>
  ParticleBuffer(Allocator allocator, std::size_t count)
      : base_t{allocator, count} {}
};

template <>
struct ParticleBuffer<material_e::FixedCorotated_ASFLIP>
    : ParticleBufferImpl<material_e::FixedCorotated_ASFLIP> {
  using base_t = ParticleBufferImpl<material_e::FixedCorotated_ASFLIP>;
  PREC length = DOMAIN_LENGTH; // Domain total length [m] (scales volume, etc.)
  PREC rho = DENSITY;
  PREC volume = DOMAIN_VOLUME * (1.f / (1 << DOMAIN_BITS) / (1 << DOMAIN_BITS) /
                  (1 << DOMAIN_BITS) / MODEL_PPC);
  PREC mass = (volume * DENSITY);
  PREC E = YOUNGS_MODULUS;
  PREC nu = POISSON_RATIO;
  PREC lambda = YOUNGS_MODULUS * POISSON_RATIO /
                 ((1 + POISSON_RATIO) * (1 - 2 * POISSON_RATIO));
  PREC mu = YOUNGS_MODULUS / (2 * (1 + POISSON_RATIO));
  bool use_ASFLIP = false; //< Use ASFLIP/PIC mixing? Default off.
  PREC alpha = 0.0;  //< FLIP/PIC Mixing Factor [0.1] -> [PIC, FLIP]
  PREC beta_min = 0.0; //< ASFLIP Minimum Position Correction Factor  
  PREC beta_max = 0.0; //< ASFLIP Maximum Position Correction Factor 
  PREC FBAR_ratio = 0.0; //< F-Bar Anti-locking mixing ratio (0 = None, 1 = Full)
  bool use_FEM = false; //< Use Finite Elements? Default off. Must set mesh
  bool use_FBAR = false; //< Use Simple F-Bar anti-locking? Default off.
  void updateParameters(PREC l, config::MaterialConfigs mat, 
                        config::AlgoConfigs algo) {
    length = l;
    rho = mat.rho;
    volume = length*length*length * ( 1.f / (1 << DOMAIN_BITS) / (1 << DOMAIN_BITS) /
                    (1 << DOMAIN_BITS) / mat.ppc);
    mass = volume * mat.rho;
    lambda = mat.E * mat.nu / ((1 + mat.nu) * (1 - 2 * mat.nu));
    mu = mat.E / (2 * (1 + mat.nu));
    alpha = algo.ASFLIP_alpha;
    beta_min = algo.ASFLIP_beta_min;
    beta_max = algo.ASFLIP_beta_max;
    FBAR_ratio = algo.FBAR_ratio;
    use_ASFLIP = algo.use_ASFLIP;
    use_FEM = algo.use_FEM;
    use_FBAR = algo.use_FBAR;
  }

  // * Attributes saved on particles of this material. Given variable names for easy mapping
  // * REQUIRED : Variable order matches atttribute order in ParticleBuffer.val_1d(VARIABLE, ...)
  // * e.g. if ParticleBuffer<MATERIAL>.val_1d(4_, ...) is Velocity_X, then set Velocity_X = 4
  // * REQUIRED : Define material's unused base variables after END to avoid errors.
  // TODO : Write unit-test to guarantee all attribs_e have base set of variables.
  enum attribs_e : int {
          EMPTY=-3, // Empty attribute request 
          INVALID_CT=-2, // Invalid compile-time request, e.g. asking for variable after END
          INVALID_RT=-1, // Invalid run-time request e.g. "Speed_X" instead of "Velocity_X"
          START=0, // Values less than or equal to START not held on particle
          Position_X=0, Position_Y=1, Position_Z=2,
          DefGrad_XX, DefGrad_XY, DefGrad_XZ,
          DefGrad_YX, DefGrad_YY, DefGrad_YZ,
          DefGrad_ZX, DefGrad_ZY, DefGrad_ZZ,
          Velocity_X, Velocity_Y, Velocity_Z,
          ID,
          END, // Values greater than or equal to END not held on particle
          // REQUIRED: Put N/A variables for specific material below END
          J, DefGrad_Determinant=J, Volume_FBAR, JBar, DefGrad_Determinant_FBAR=JBar, logJp
  };


  // TODO : Change if/else statement to case/switch. may require compile-time min-max guarantee
  template <attribs_e ATTRIBUTE, typename T>
   __device__ PREC
  getAttribute(const T bin, const T particle_id_in_bin){
    if (ATTRIBUTE < attribs_e::START) return (PREC)ATTRIBUTE;
    else if (ATTRIBUTE >= attribs_e::END) return (PREC)attribs_e::INVALID_CT;
    else return this->ch(std::integral_constant<unsigned, 0>{}, bin).val_1d(std::integral_constant<unsigned, std::min(abs(ATTRIBUTE), attribs_e::END-1)>{}, particle_id_in_bin);
  }
  
  template <typename T = PREC>
   __device__ void
  getVelocity(const T bin, const T particle_id_in_bin, PREC * velocity) {
    velocity = {
      this->ch(std::integral_constant<unsigned, 0>{}, bin).val_1d(std::integral_constant<unsigned, attribs_e::Velocity_X>{}, particle_id_in_bin),
      this->ch(std::integral_constant<unsigned, 0>{}, bin).val_1d(std::integral_constant<unsigned, attribs_e::Velocity_Y>{}, particle_id_in_bin),
      this->ch(std::integral_constant<unsigned, 0>{}, bin).val_1d(std::integral_constant<unsigned, attribs_e::Velocity_Z>{}, particle_id_in_bin)
    };
  }

  template <typename T>
   __device__ constexpr void
  getDefGrad(T bin, T particle_id_in_bin, PREC * DefGrad) {
    DefGrad[0] = this->ch(std::integral_constant<unsigned, 0>{}, bin).val_1d(std::integral_constant<unsigned, attribs_e::DefGrad_XX>{}, particle_id_in_bin);
    DefGrad[1] = this->ch(std::integral_constant<unsigned, 0>{}, bin).val_1d(std::integral_constant<unsigned, attribs_e::DefGrad_XY>{}, particle_id_in_bin);
    DefGrad[2] = this->ch(std::integral_constant<unsigned, 0>{}, bin).val_1d(std::integral_constant<unsigned, attribs_e::DefGrad_XZ>{}, particle_id_in_bin);
    DefGrad[3] = this->ch(std::integral_constant<unsigned, 0>{}, bin).val_1d(std::integral_constant<unsigned, attribs_e::DefGrad_YX>{}, particle_id_in_bin);
    DefGrad[4] = this->ch(std::integral_constant<unsigned, 0>{}, bin).val_1d(std::integral_constant<unsigned, attribs_e::DefGrad_YY>{}, particle_id_in_bin);
    DefGrad[5] = this->ch(std::integral_constant<unsigned, 0>{}, bin).val_1d(std::integral_constant<unsigned, attribs_e::DefGrad_YZ>{}, particle_id_in_bin);
    DefGrad[6] = this->ch(std::integral_constant<unsigned, 0>{}, bin).val_1d(std::integral_constant<unsigned, attribs_e::DefGrad_ZX>{}, particle_id_in_bin);
    DefGrad[7] = this->ch(std::integral_constant<unsigned, 0>{}, bin).val_1d(std::integral_constant<unsigned, attribs_e::DefGrad_ZY>{}, particle_id_in_bin);
    DefGrad[8] = this->ch(std::integral_constant<unsigned, 0>{}, bin).val_1d(std::integral_constant<unsigned, attribs_e::DefGrad_ZZ>{}, particle_id_in_bin);
  }

  template <typename T = PREC>
   __device__ constexpr void
  getStress_Cauchy(const vec<T,9>& F, vec<T,9>& PF){
    compute_stress_fixedcorotated(volume, mu, lambda, F, PF);
  }
  template <typename T = PREC>
   __device__ constexpr void
  getStress_Cauchy(T vol, const vec<T,9>& F, vec<T,9>& PF){
    compute_stress_fixedcorotated(vol, mu, lambda, F, PF);
  }
  
  template <typename T = PREC>
   __device__ void
  getStress_PK1(const vec<T,9>& F, vec<T,9>& P){
    compute_stress_PK1_fixedcorotated(volume, mu, lambda, F, P);
  }
  template <typename T = PREC>
   __device__ void
  getStress_PK1(T vol, const vec<T,9>& F, vec<T,9>& P){
    compute_stress_PK1_fixedcorotated(vol, mu, lambda, F, P);
  }

  template <typename T = PREC>
   __device__ void
  getEnergy_Strain(const vec<T,9>& F, T& strain_energy, T vol){
    compute_energy_fixedcorotated(vol, mu, lambda, F, strain_energy);
  }
  template <typename T = PREC>
   __device__ void
  getEnergy_Strain(const vec<T,9>& F, T& strain_energy){
    compute_energy_fixedcorotated(volume, mu, lambda, F, strain_energy);
  }
  template <typename T = PREC>
   __device__ void
  getEnergy_Kinetic(const vec<T,3>& velocity, T& kinetic_energy){  
    kinetic_energy = 0.5 * mass * __fma_rn(velocity[0], velocity[0], __fma_rn(velocity[1], velocity[1], (velocity[2], velocity[2])));
    }

  template <typename Allocator>
  ParticleBuffer(Allocator allocator) : base_t{allocator} {}

  template <typename Allocator>
  ParticleBuffer(Allocator allocator, std::size_t count)
      : base_t{allocator, count} {}
};


template <>
struct ParticleBuffer<material_e::FixedCorotated_ASFLIP_FBAR>
    : ParticleBufferImpl<material_e::FixedCorotated_ASFLIP_FBAR> {
  using base_t = ParticleBufferImpl<material_e::FixedCorotated_ASFLIP_FBAR>;
  PREC length = DOMAIN_LENGTH; // Domain total length [m] (scales volume, etc.)
  PREC rho = DENSITY;
  PREC volume = DOMAIN_VOLUME * (1.f / (1 << DOMAIN_BITS) / (1 << DOMAIN_BITS) /
                  (1 << DOMAIN_BITS) / MODEL_PPC);
  PREC mass = (volume * DENSITY);
  PREC E = YOUNGS_MODULUS;
  PREC nu = POISSON_RATIO;
  PREC lambda = YOUNGS_MODULUS * POISSON_RATIO /
                 ((1 + POISSON_RATIO) * (1 - 2 * POISSON_RATIO));
  PREC mu = YOUNGS_MODULUS / (2 * (1 + POISSON_RATIO));
  bool use_ASFLIP = false; //< Use ASFLIP/PIC mixing? Default off.
  PREC alpha = 0.0;  //< FLIP/PIC Mixing Factor [0.1] -> [PIC, FLIP]
  PREC beta_min = 0.0; //< ASFLIP Minimum Position Correction Factor  
  PREC beta_max = 0.0; //< ASFLIP Maximum Position Correction Factor 
  PREC FBAR_ratio = 0.0; //< F-Bar Anti-locking mixing ratio (0 = None, 1 = Full)
  bool use_FEM = false; //< Use Finite Elements? Default off. Must set mesh
  bool use_FBAR = false; //< Use Simple F-Bar anti-locking? Default off.
  void updateParameters(PREC l, config::MaterialConfigs mat, 
                        config::AlgoConfigs algo) {
    length = l;
    rho = mat.rho;
    volume = length*length*length * ( 1.0 / (1 << DOMAIN_BITS) / (1 << DOMAIN_BITS) /
                    (1 << DOMAIN_BITS) / mat.ppc);
    mass = volume * mat.rho;
    lambda = mat.E * mat.nu / ((1 + mat.nu) * (1 - 2 * mat.nu));
    mu = mat.E / (2 * (1 + mat.nu));
    alpha = algo.ASFLIP_alpha;
    beta_min = algo.ASFLIP_beta_min;
    beta_max = algo.ASFLIP_beta_max;
    FBAR_ratio = algo.FBAR_ratio;
    use_ASFLIP = algo.use_ASFLIP;
    use_FEM = algo.use_FEM;
    use_FBAR = algo.use_FBAR;
  }

  // * Attributes saved on particles of this material. Given variable names for easy mapping
  // * REQUIRED : Variable order matches atttribute order in ParticleBuffer.val_1d(VARIABLE, ...)
  // * e.g. if ParticleBuffer<MATERIAL>.val_1d(4_, ...) is Velocity_X, then set Velocity_X = 4
  // * REQUIRED : Define material's unused base variables after END to avoid errors.
  // TODO : Write unit-test to guarantee all attribs_e have base set of variables.
  enum attribs_e : int {
          EMPTY=-3, // Empty attribute request 
          INVALID_CT=-2, // Invalid compile-time request, e.g. asking for variable after END
          INVALID_RT=-1, // Invalid run-time request e.g. "Speed_X" instead of "Velocity_X"
          START=0, // Values less than or equal to START not held on particle
          Position_X=0, Position_Y=1, Position_Z=2,
          DefGrad_XX, DefGrad_XY, DefGrad_XZ,
          DefGrad_YX, DefGrad_YY, DefGrad_YZ,
          DefGrad_ZX, DefGrad_ZY, DefGrad_ZZ,
          Velocity_X, Velocity_Y, Velocity_Z,
          Volume_FBAR, JBar, DefGrad_Determinant_FBAR=JBar, ID,
          END, // Values greater than or equal to END not held on particle
          // REQUIRED: Put N/A variables for specific material below END
          J, DefGrad_Determinant=J, logJp
  };


  // TODO : Change if/else statement to case/switch. may require compile-time min-max guarantee
  template <attribs_e ATTRIBUTE, typename T>
   __device__ PREC
  getAttribute(const T bin, const T particle_id_in_bin){
    if (ATTRIBUTE < attribs_e::START) return (PREC)ATTRIBUTE;
    else if (ATTRIBUTE >= attribs_e::END) return (PREC)attribs_e::INVALID_CT;
    else return this->ch(std::integral_constant<unsigned, 0>{}, bin).val_1d(std::integral_constant<unsigned, std::min(abs(ATTRIBUTE), attribs_e::END-1)>{}, particle_id_in_bin);
  }

  template <typename T = PREC>
   __device__ void
  getVelocity(const T bin, const T particle_id_in_bin, PREC * velocity) {
    velocity = {
      this->ch(std::integral_constant<unsigned, 0>{}, bin).val_1d(std::integral_constant<unsigned, attribs_e::Velocity_X>{}, particle_id_in_bin),
      this->ch(std::integral_constant<unsigned, 0>{}, bin).val_1d(std::integral_constant<unsigned, attribs_e::Velocity_Y>{}, particle_id_in_bin),
      this->ch(std::integral_constant<unsigned, 0>{}, bin).val_1d(std::integral_constant<unsigned, attribs_e::Velocity_Z>{}, particle_id_in_bin)
    };
  }

  template <typename T>
   __device__ constexpr void
  getDefGrad(T bin, T particle_id_in_bin, PREC * DefGrad) {
    DefGrad[0] = this->ch(std::integral_constant<unsigned, 0>{}, bin).val_1d(std::integral_constant<unsigned, attribs_e::DefGrad_XX>{}, particle_id_in_bin);
    DefGrad[1] = this->ch(std::integral_constant<unsigned, 0>{}, bin).val_1d(std::integral_constant<unsigned, attribs_e::DefGrad_XY>{}, particle_id_in_bin);
    DefGrad[2] = this->ch(std::integral_constant<unsigned, 0>{}, bin).val_1d(std::integral_constant<unsigned, attribs_e::DefGrad_XZ>{}, particle_id_in_bin);
    DefGrad[3] = this->ch(std::integral_constant<unsigned, 0>{}, bin).val_1d(std::integral_constant<unsigned, attribs_e::DefGrad_YX>{}, particle_id_in_bin);
    DefGrad[4] = this->ch(std::integral_constant<unsigned, 0>{}, bin).val_1d(std::integral_constant<unsigned, attribs_e::DefGrad_YY>{}, particle_id_in_bin);
    DefGrad[5] = this->ch(std::integral_constant<unsigned, 0>{}, bin).val_1d(std::integral_constant<unsigned, attribs_e::DefGrad_YZ>{}, particle_id_in_bin);
    DefGrad[6] = this->ch(std::integral_constant<unsigned, 0>{}, bin).val_1d(std::integral_constant<unsigned, attribs_e::DefGrad_ZX>{}, particle_id_in_bin);
    DefGrad[7] = this->ch(std::integral_constant<unsigned, 0>{}, bin).val_1d(std::integral_constant<unsigned, attribs_e::DefGrad_ZY>{}, particle_id_in_bin);
    DefGrad[8] = this->ch(std::integral_constant<unsigned, 0>{}, bin).val_1d(std::integral_constant<unsigned, attribs_e::DefGrad_ZZ>{}, particle_id_in_bin);
  }

  template <typename T = PREC>
   __device__ constexpr void
  getStress_Cauchy(const vec<T,9>& F, vec<T,9>& PF){
    compute_stress_fixedcorotated(volume, mu, lambda, F, PF);
  }
  template <typename T = PREC>
   __device__ constexpr void
  getStress_Cauchy(T vol, const vec<T,9>& F, vec<T,9>& PF){
    compute_stress_fixedcorotated(vol, mu, lambda, F, PF);
  }
  
  template <typename T = PREC>
   __device__ void
  getStress_PK1(const vec<T,9>& F, vec<T,9>& P){
    compute_stress_PK1_fixedcorotated(volume, mu, lambda, F, P);
  }
  template <typename T = PREC>
   __device__ void
  getStress_PK1(T vol, const vec<T,9>& F, vec<T,9>& P){
    compute_stress_PK1_fixedcorotated(vol, mu, lambda, F, P);
  }

  template <typename T = PREC>
   __device__ void
  getEnergy_Strain(const vec<T,9>& F, T& strain_energy, T vol){
    compute_energy_fixedcorotated(vol, mu, lambda, F, strain_energy);
  }
  template <typename T = PREC>
   __device__ void
  getEnergy_Strain(const vec<T,9>& F, T& strain_energy){
    compute_energy_fixedcorotated(volume, mu, lambda, F, strain_energy);
  }
  template <typename T = PREC>
   __device__ void
  getEnergy_Kinetic(const vec<T,3>& velocity, T& kinetic_energy){  
    kinetic_energy = 0.5 * mass * __fma_rn(velocity[0], velocity[0], __fma_rn(velocity[1], velocity[1], (velocity[2], velocity[2])));
    }

  template <typename Allocator>
  ParticleBuffer(Allocator allocator) : base_t{allocator} {}

  template <typename Allocator>
  ParticleBuffer(Allocator allocator, std::size_t count)
      : base_t{allocator, count} {}
};


template <>
struct ParticleBuffer<material_e::NeoHookean_ASFLIP_FBAR>
    : ParticleBufferImpl<material_e::NeoHookean_ASFLIP_FBAR> {
  using base_t = ParticleBufferImpl<material_e::NeoHookean_ASFLIP_FBAR>;
  PREC length = DOMAIN_LENGTH; // Domain total length [m] (scales volume, etc.)
  PREC rho = DENSITY;
  PREC volume = DOMAIN_VOLUME * (1.0 / (1 << DOMAIN_BITS) / (1 << DOMAIN_BITS) /
                  (1 << DOMAIN_BITS) / MODEL_PPC);
  PREC mass = (volume * DENSITY);
  PREC E = YOUNGS_MODULUS;
  PREC nu = POISSON_RATIO;
  PREC lambda = YOUNGS_MODULUS * POISSON_RATIO /
                 ((1 + POISSON_RATIO) * (1 - 2 * POISSON_RATIO));
  PREC mu = YOUNGS_MODULUS / (2 * (1 + POISSON_RATIO));
  bool use_ASFLIP = false; //< Use ASFLIP/PIC mixing? Default off.
  PREC alpha = 0.0;  //< FLIP/PIC Mixing Factor [0.1] -> [PIC, FLIP]
  PREC beta_min = 0.0; //< ASFLIP Minimum Position Correction Factor  
  PREC beta_max = 0.0; //< ASFLIP Maximum Position Correction Factor 
  PREC FBAR_ratio = 0.0; //< F-Bar Anti-locking mixing ratio (0 = None, 1 = Full)
  bool use_FEM = false; //< Use Finite Elements? Default off. Must set mesh
  bool use_FBAR = false; //< Use Simple F-Bar anti-locking? Default off.
  void updateParameters(PREC l, config::MaterialConfigs mat, 
                        config::AlgoConfigs algo) {
    length = l;
    volume = length*length*length * ( 1.f / (1 << DOMAIN_BITS) / (1 << DOMAIN_BITS) /
                    (1 << DOMAIN_BITS) / mat.ppc);
    rho = mat.rho;
    mass = volume * mat.rho;
    lambda = mat.E * mat.nu / ((1 + mat.nu) * (1 - 2 * mat.nu));
    mu = mat.E / (2 * (1 + mat.nu));
    alpha = algo.ASFLIP_alpha;
    beta_min = algo.ASFLIP_beta_min;
    beta_max = algo.ASFLIP_beta_max;
    FBAR_ratio = algo.FBAR_ratio;
    use_ASFLIP = algo.use_ASFLIP;
    use_FEM = algo.use_FEM;
    use_FBAR = algo.use_FBAR;
  }

  // * Attributes saved on particles of this material. Given variable names for easy mapping
  // * REQUIRED : Variable order matches atttribute order in ParticleBuffer.val_1d(VARIABLE, ...)
  // * e.g. if ParticleBuffer<MATERIAL>.val_1d(4_, ...) is Velocity_X, then set Velocity_X = 4
  // * REQUIRED : Define material's unused base variables after END to avoid errors.
  // TODO : Write unit-test to guarantee all attribs_e have base set of variables.
  enum attribs_e : int {
          EMPTY=-3, // Empty attribute request 
          INVALID_CT=-2, // Invalid compile-time request, e.g. asking for variable after END
          INVALID_RT=-1, // Invalid run-time request e.g. "Speed_X" instead of "Velocity_X"
          START=0, // Values less than or equal to START not held on particle
          Position_X=0, Position_Y=1, Position_Z=2,
          DefGrad_XX, DefGrad_XY, DefGrad_XZ,
          DefGrad_YX, DefGrad_YY, DefGrad_YZ,
          DefGrad_ZX, DefGrad_ZY, DefGrad_ZZ,
          Velocity_X, Velocity_Y, Velocity_Z,
          Volume_FBAR, JBar, DefGrad_Determinant_FBAR=JBar, ID,
          END, // Values greater than or equal to END not held on particle
          // REQUIRED: Put N/A variables for specific material below END
          J, DefGrad_Determinant=J, logJp
  };


  // TODO : Change if/else statement to case/switch. may require compile-time min-max guarantee
  template <attribs_e ATTRIBUTE, typename T>
   __device__ PREC
  getAttribute(const T bin, const T particle_id_in_bin){
    if (ATTRIBUTE < attribs_e::START) return (PREC)ATTRIBUTE;
    else if (ATTRIBUTE >= attribs_e::END) return (PREC)attribs_e::INVALID_CT;
    else return this->ch(std::integral_constant<unsigned, 0>{}, bin).val_1d(std::integral_constant<unsigned, std::min(abs(ATTRIBUTE), attribs_e::END-1)>{}, particle_id_in_bin);
  }

  template <typename T = int>
   __device__ void
  getVelocity(const T bin, const T particle_id_in_bin, PREC * velocity) {
    velocity = {
      this->ch(std::integral_constant<unsigned, 0>{}, bin).val_1d(std::integral_constant<unsigned, attribs_e::Velocity_X>{}, particle_id_in_bin),
      this->ch(std::integral_constant<unsigned, 0>{}, bin).val_1d(std::integral_constant<unsigned, attribs_e::Velocity_Y>{}, particle_id_in_bin),
      this->ch(std::integral_constant<unsigned, 0>{}, bin).val_1d(std::integral_constant<unsigned, attribs_e::Velocity_Z>{}, particle_id_in_bin)
    };
  }

  template <typename T>
   __device__ constexpr void
  getDefGrad(T bin, T particle_id_in_bin, PREC * DefGrad) {
    DefGrad[0] = this->ch(std::integral_constant<unsigned, 0>{}, bin).val_1d(std::integral_constant<unsigned, attribs_e::DefGrad_XX>{}, particle_id_in_bin);
    DefGrad[1] = this->ch(std::integral_constant<unsigned, 0>{}, bin).val_1d(std::integral_constant<unsigned, attribs_e::DefGrad_XY>{}, particle_id_in_bin);
    DefGrad[2] = this->ch(std::integral_constant<unsigned, 0>{}, bin).val_1d(std::integral_constant<unsigned, attribs_e::DefGrad_XZ>{}, particle_id_in_bin);
    DefGrad[3] = this->ch(std::integral_constant<unsigned, 0>{}, bin).val_1d(std::integral_constant<unsigned, attribs_e::DefGrad_YX>{}, particle_id_in_bin);
    DefGrad[4] = this->ch(std::integral_constant<unsigned, 0>{}, bin).val_1d(std::integral_constant<unsigned, attribs_e::DefGrad_YY>{}, particle_id_in_bin);
    DefGrad[5] = this->ch(std::integral_constant<unsigned, 0>{}, bin).val_1d(std::integral_constant<unsigned, attribs_e::DefGrad_YZ>{}, particle_id_in_bin);
    DefGrad[6] = this->ch(std::integral_constant<unsigned, 0>{}, bin).val_1d(std::integral_constant<unsigned, attribs_e::DefGrad_ZX>{}, particle_id_in_bin);
    DefGrad[7] = this->ch(std::integral_constant<unsigned, 0>{}, bin).val_1d(std::integral_constant<unsigned, attribs_e::DefGrad_ZY>{}, particle_id_in_bin);
    DefGrad[8] = this->ch(std::integral_constant<unsigned, 0>{}, bin).val_1d(std::integral_constant<unsigned, attribs_e::DefGrad_ZZ>{}, particle_id_in_bin);
  }

  template <typename T = PREC>
   __device__ constexpr void
  getStress_Cauchy(const vec<T,9>& F, vec<T,9>& PF){
    compute_stress_neohookean(volume, mu, lambda, F, PF);
  }
  template <typename T = PREC>
   __device__  constexpr void
  getStress_Cauchy(T vol, const vec<T,9>& F, vec<T,9>& PF){
    compute_stress_neohookean(vol, mu, lambda, F, PF);
  }
  
  template <typename T = PREC>
   __device__ void
  getStress_PK1(const vec<T,9>& F, vec<T,9>& P){
    compute_stress_PK1_neohookean(volume, mu, lambda, F, P);
  }
  template <typename T = PREC>
   __device__ void
  getStress_PK1(T vol, const vec<T,9>& F, vec<T,9>& P){
    compute_stress_PK1_neohookean(vol, mu, lambda, F, P);
  }

  template <typename T = PREC>
   __device__ void
  getEnergy_Strain(const vec<T,9>& F, T& strain_energy, T vol){
    compute_energy_neohookean(vol, mu, lambda, F, strain_energy);
  }
  template <typename T = PREC>
   __device__ void
  getEnergy_Strain(const vec<T,9>& F, T& strain_energy){
    compute_energy_neohookean(volume, mu, lambda, F, strain_energy);
  }
  template <typename T = PREC>
   __device__ void
  getEnergy_Kinetic(const vec<T,3>& velocity, T& kinetic_energy){  
    kinetic_energy = 0.5 * mass * __fma_rn(velocity[0], velocity[0], __fma_rn(velocity[1], velocity[1], (velocity[2], velocity[2])));
    }

  template <typename Allocator>
  ParticleBuffer(Allocator allocator) : base_t{allocator} {}

  template <typename Allocator>
  ParticleBuffer(Allocator allocator, std::size_t count)
      : base_t{allocator, count} {}
};

template <>
struct ParticleBuffer<material_e::Sand> : ParticleBufferImpl<material_e::Sand> {
  using base_t = ParticleBufferImpl<material_e::Sand>;
  PREC length = DOMAIN_LENGTH; // Domain total length [m] (scales volume, etc.)
  PREC rho = DENSITY;
  PREC volume = DOMAIN_VOLUME * 
      (1.f / (1 << DOMAIN_BITS) / (1 << DOMAIN_BITS) / (1 << DOMAIN_BITS) /
       MODEL_PPC);
  PREC mass = (volume * DENSITY);
  PREC E = YOUNGS_MODULUS;
  PREC nu = POISSON_RATIO;
  PREC lambda =
      YOUNGS_MODULUS * POISSON_RATIO /
      ((1 + POISSON_RATIO) * (1 - 2 * POISSON_RATIO));
  PREC mu = YOUNGS_MODULUS / (2 * (1 + POISSON_RATIO));

  PREC logJp0 = 0.f;
  PREC frictionAngle = 30.f;
  PREC cohesion = 0.f;
  PREC beta = 1.f;
  // std::sqrt(2.f/3.f) * 2.f * std::sin(30.f/180.f*3.141592741f)
  // 						/ (3.f -
  // std::sin(30.f/180.f*3.141592741f))
  PREC yieldSurface =
      0.816496580927726f * 2.f * 0.5f / (3.f - 0.5f);
  bool volumeCorrection = true;
  bool use_ASFLIP = false; //< Use ASFLIP/PIC mixing? Default off.
  PREC alpha = 0.0;  //< FLIP/PIC Mixing Factor [0.1] -> [PIC, FLIP]
  PREC beta_min = 0.0; //< ASFLIP Minimum Position Correction Factor  
  PREC beta_max = 0.0; //< ASFLIP Maximum Position Correction Factor 
  PREC FBAR_ratio = 0.0; //< F-Bar Anti-locking mixing ratio (0 = None, 1 = Full)
  bool use_FEM = false; //< Use Finite Elements? Default off. Must set mesh
  bool use_FBAR = false; //< Use Simple F-Bar anti-locking? Default off.
  void updateParameters(PREC l, config::MaterialConfigs mat, 
                        config::AlgoConfigs algo) {
    length = l;
    rho = mat.rho;
    volume = length*length*length * ( 1.f / (1 << DOMAIN_BITS) / (1 << DOMAIN_BITS) /
                    (1 << DOMAIN_BITS) / mat.ppc);
    mass = volume * mat.rho;
    lambda = mat.E * mat.nu / ((1 + mat.nu) * (1 - 2 * mat.nu));
    mu = mat.E / (2 * (1 + mat.nu));
    logJp0 = mat.logJp0;
    frictionAngle = mat.frictionAngle;
    yieldSurface = 0.816496580927726 * 2.0 * std::sin(mat.frictionAngle / 180.0 * 3.141592741) / (3.0 - std::sin(mat.frictionAngle / 180.0 * 3.141592741));
    cohesion = mat.cohesion;
    beta = mat.beta;
    volumeCorrection = mat.volumeCorrection;
    alpha = algo.ASFLIP_alpha;
    beta_min = algo.ASFLIP_beta_min;
    beta_max = algo.ASFLIP_beta_max;
    FBAR_ratio = algo.FBAR_ratio;
    use_ASFLIP = algo.use_ASFLIP;
    use_FEM = algo.use_FEM;
    use_FBAR = algo.use_FBAR;
  }

  // * Attributes saved on particles of this material. Given variable names for easy mapping
  // * REQUIRED : Variable order matches atttribute order in ParticleBuffer.val_1d(VARIABLE, ...)
  // * e.g. if ParticleBuffer<MATERIAL>.val_1d(4_, ...) is Velocity_X, then set Velocity_X = 4
  // * REQUIRED : Define material's unused base variables after END to avoid errors.
  // TODO : Write unit-test to guarantee all attribs_e have base set of variables.
  enum attribs_e : int {
          EMPTY=-3, // Empty attribute request 
          INVALID_CT=-2, // Invalid compile-time request, e.g. asking for variable after END
          INVALID_RT=-1, // Invalid run-time request e.g. "Speed_X" instead of "Velocity_X"
          START=0, // Values less than or equal to START not held on particle
          Position_X=0, Position_Y=1, Position_Z=2,
          DefGrad_XX, DefGrad_XY, DefGrad_XZ,
          DefGrad_YX, DefGrad_YY, DefGrad_YZ,
          DefGrad_ZX, DefGrad_ZY, DefGrad_ZZ,
          logJp,
          Velocity_X, Velocity_Y, Velocity_Z,
          JBar, DefGrad_Determinant_FBAR=JBar, 
          ID,
          END, // Values greater than or equal to END not held on particle
          // REQUIRED: Put N/A variables for specific material below END
          J, DefGrad_Determinant=J, 
          Volume_FBAR 
  };

  // TODO : Change if/else statement to case/switch. may require compile-time min-max guarantee
  template <attribs_e ATTRIBUTE, typename T>
   __device__ PREC
  getAttribute(const T bin, const T particle_id_in_bin){
    if (ATTRIBUTE < attribs_e::START) return (PREC)ATTRIBUTE;
    else if (ATTRIBUTE >= attribs_e::END) return (PREC)attribs_e::INVALID_CT;
    else return this->ch(std::integral_constant<unsigned, 0>{}, bin).val_1d(std::integral_constant<unsigned, std::min(abs(ATTRIBUTE), attribs_e::END-1)>{}, particle_id_in_bin);
  }

  template <typename T = PREC>
   __device__ void
  getVelocity(const T bin, const T particle_id_in_bin, PREC * velocity) {
    velocity = {
      this->ch(std::integral_constant<unsigned, 0>{}, bin).val_1d(std::integral_constant<unsigned, attribs_e::Velocity_X>{}, particle_id_in_bin),
      this->ch(std::integral_constant<unsigned, 0>{}, bin).val_1d(std::integral_constant<unsigned, attribs_e::Velocity_Y>{}, particle_id_in_bin),
      this->ch(std::integral_constant<unsigned, 0>{}, bin).val_1d(std::integral_constant<unsigned, attribs_e::Velocity_Z>{}, particle_id_in_bin)
    };
  }

  template <typename T>
   __device__ constexpr void
  getDefGrad(T bin, T particle_id_in_bin, PREC * DefGrad) {
    DefGrad[0] = this->ch(std::integral_constant<unsigned, 0>{}, bin).val_1d(std::integral_constant<unsigned, attribs_e::DefGrad_XX>{}, particle_id_in_bin);
    DefGrad[1] = this->ch(std::integral_constant<unsigned, 0>{}, bin).val_1d(std::integral_constant<unsigned, attribs_e::DefGrad_XY>{}, particle_id_in_bin);
    DefGrad[2] = this->ch(std::integral_constant<unsigned, 0>{}, bin).val_1d(std::integral_constant<unsigned, attribs_e::DefGrad_XZ>{}, particle_id_in_bin);
    DefGrad[3] = this->ch(std::integral_constant<unsigned, 0>{}, bin).val_1d(std::integral_constant<unsigned, attribs_e::DefGrad_YX>{}, particle_id_in_bin);
    DefGrad[4] = this->ch(std::integral_constant<unsigned, 0>{}, bin).val_1d(std::integral_constant<unsigned, attribs_e::DefGrad_YY>{}, particle_id_in_bin);
    DefGrad[5] = this->ch(std::integral_constant<unsigned, 0>{}, bin).val_1d(std::integral_constant<unsigned, attribs_e::DefGrad_YZ>{}, particle_id_in_bin);
    DefGrad[6] = this->ch(std::integral_constant<unsigned, 0>{}, bin).val_1d(std::integral_constant<unsigned, attribs_e::DefGrad_ZX>{}, particle_id_in_bin);
    DefGrad[7] = this->ch(std::integral_constant<unsigned, 0>{}, bin).val_1d(std::integral_constant<unsigned, attribs_e::DefGrad_ZY>{}, particle_id_in_bin);
    DefGrad[8] = this->ch(std::integral_constant<unsigned, 0>{}, bin).val_1d(std::integral_constant<unsigned, attribs_e::DefGrad_ZZ>{}, particle_id_in_bin);
  }

  template <typename T = PREC>
   __device__ void
  getStress_Cauchy(vec<T,9>& F, vec<T,9>& PF){
    PREC lj = logJp0;
    compute_stress_sand(volume, mu, lambda, cohesion, beta, yieldSurface, volumeCorrection, lj, F, PF);
  }
  template <typename T = PREC>
   __device__ void
  getStress_Cauchy(T vol, vec<T,9>& F, vec<T,9>& PF){
    PREC lj = logJp0;
    compute_stress_sand(vol, mu, lambda, cohesion, beta, yieldSurface, volumeCorrection, lj, F, PF);
  }
  
  template <typename T = PREC>
   __device__ void
  getStress_PK1(const vec<T,9>& F, vec<T,9>& P){
    PREC lj = logJp0;
    compute_stress_PK1_sand(volume, mu, lambda, cohesion, beta, yieldSurface, volumeCorrection, lj, F, P);
  }
  template <typename T = PREC>
   __device__ void
  getStress_PK1(T vol, const vec<T,9>& F, vec<T,9>& P){
    PREC lj = logJp0;
    compute_stress_PK1_sand(vol, mu, lambda, cohesion, beta, yieldSurface, volumeCorrection, lj,F, P);
  }

  template <typename T = PREC>
   __device__ void
  getEnergy_Strain(const vec<T,9>& F, T& strain_energy, T vol){
    compute_energy_sand(vol, mu, lambda, cohesion, beta, yieldSurface, volumeCorrection, logJp0,F, strain_energy);
  }
  template <typename T = PREC>
   __device__ void
  getEnergy_Strain(const vec<T,9>& F, T& strain_energy){
    compute_energy_sand(volume, mu, lambda, cohesion, beta, yieldSurface, volumeCorrection, logJp0,F, strain_energy);
  }
  template <typename T = PREC>
   __device__ void
  getEnergy_Kinetic(const vec<T,3>& velocity, T& kinetic_energy){  
    kinetic_energy = 0.5 * mass * __fma_rn(velocity[0], velocity[0], __fma_rn(velocity[1], velocity[1], (velocity[2], velocity[2])));
    }

  template <typename Allocator>
  ParticleBuffer(Allocator allocator) : base_t{allocator} {}

  template <typename Allocator>
  ParticleBuffer(Allocator allocator, std::size_t count)
      : base_t{allocator, count} {}
};

template <>
struct ParticleBuffer<material_e::NACC> : ParticleBufferImpl<material_e::NACC> {
  using base_t = ParticleBufferImpl<material_e::NACC>;
  PREC length = DOMAIN_LENGTH; // Domain total length [m] (scales volume, etc.)
  PREC rho = DENSITY;
  PREC volume = DOMAIN_VOLUME * (1.f / (1 << DOMAIN_BITS) / (1 << DOMAIN_BITS) /
                  (1 << DOMAIN_BITS) / MODEL_PPC);
  PREC mass = (volume * DENSITY);
  PREC E = YOUNGS_MODULUS;
  PREC nu = POISSON_RATIO;
  PREC lambda = YOUNGS_MODULUS * POISSON_RATIO /
                 ((1 + POISSON_RATIO) * (1 - 2 * POISSON_RATIO));
  PREC mu = YOUNGS_MODULUS / (2 * (1 + POISSON_RATIO));

  PREC frictionAngle = 45.f;
  PREC bm = 2.f / 3.f * (YOUNGS_MODULUS / (2 * (1 + POISSON_RATIO))) +
             (YOUNGS_MODULUS * POISSON_RATIO /
              ((1 + POISSON_RATIO) *
               (1 - 2 * POISSON_RATIO))); ///< bulk modulus, kappa
  PREC xi = 0.8f;                        ///< hardening factor
  PREC logJp0 = -0.01f;
  PREC beta = 0.5f;
  static constexpr PREC mohrColumbFriction =
      0.503599787772409; //< sqrt((T)2 / (T)3) * (T)2 * sin_phi / ((T)3 -
                         // sin_phi);
  static constexpr PREC M =
      1.850343771924453; ///< mohrColumbFriction * (T)dim / sqrt((T)2 / ((T)6
                         ///< - dim));
  static constexpr PREC Msqr = 3.423772074299613;
  bool hardeningOn = true;
  bool use_ASFLIP = false; //< Use ASFLIP/PIC mixing? Default off.
  PREC alpha = 0.0;  //< FLIP/PIC Mixing Factor [0.1] -> [PIC, FLIP]
  PREC beta_min = 0.0; //< ASFLIP Minimum Position Correction Factor  
  PREC beta_max = 0.0; //< ASFLIP Maximum Position Correction Factor 
  PREC FBAR_ratio = 0.0; //< F-Bar Anti-locking mixing ratio (0 = None, 1 = Full)
  bool use_FEM = false; //< Use Finite Elements? Default off. Must set mesh
  bool use_FBAR = false; //< Use Simple F-Bar anti-locking? Default off.
  void updateParameters(PREC l, config::MaterialConfigs mat, 
                        config::AlgoConfigs algo) {
    length = l;
    rho = mat.rho;
    volume = length*length*length * ( 1.f / (1 << DOMAIN_BITS) / (1 << DOMAIN_BITS) /
                    (1 << DOMAIN_BITS) / mat.ppc);
    mass = volume * mat.rho;
    lambda = mat.E * mat.nu / ((1 + mat.nu) * (1 - 2 * mat.nu));
    mu = mat.E / (2 * (1 + mat.nu));
    bm =
        2.f / 3.f * (mat.E / (2 * (1 + mat.nu))) + (mat.E * mat.nu / ((1 + mat.nu) * (1 - 2 * mat.nu)));
    logJp0 = mat.logJp0;
    frictionAngle = mat.frictionAngle;
    beta = mat.beta;
    xi = mat.xi;
    hardeningOn = mat.hardeningOn;
    alpha = algo.ASFLIP_alpha;
    beta_min = algo.ASFLIP_beta_min;
    beta_max = algo.ASFLIP_beta_max;
    FBAR_ratio = algo.FBAR_ratio;
    use_ASFLIP = algo.use_ASFLIP;
    use_FEM = algo.use_FEM;
    use_FBAR = algo.use_FBAR;
  }

  // * Attributes saved on particles of this material. Given variable names for easy mapping
  // * REQUIRED : Variable order matches atttribute order in ParticleBuffer.val_1d(VARIABLE, ...)
  // * e.g. if ParticleBuffer<MATERIAL>.val_1d(4_, ...) is Velocity_X, then set Velocity_X = 4
  // * REQUIRED : Define material's unused base variables after END to avoid errors.
  // TODO : Write unit-test to guarantee all attribs_e have base set of variables.
  enum attribs_e : int {
          EMPTY=-3, // Empty attribute request 
          INVALID_CT=-2, // Invalid compile-time request, e.g. asking for variable after END
          INVALID_RT=-1, // Invalid run-time request e.g. "Speed_X" instead of "Velocity_X"
          START=0, // Values less than or equal to START not held on particle
          Position_X=0, Position_Y=1, Position_Z=2,
          DefGrad_XX, DefGrad_XY, DefGrad_XZ,
          DefGrad_YX, DefGrad_YY, DefGrad_YZ,
          DefGrad_ZX, DefGrad_ZY, DefGrad_ZZ,
          logJp,
          Velocity_X, Velocity_Y, Velocity_Z,
          JBar, DefGrad_Determinant_FBAR=JBar, 
          ID,
          END, // Values greater than or equal to END not held on particle
          // REQUIRED: Put N/A variables for specific material below END
          J, DefGrad_Determinant=J, 
          Volume_FBAR 
  };

  // TODO : Change if/else statement to case/switch. may require compile-time min-max guarantee
  template <attribs_e ATTRIBUTE, typename T>
   __device__ PREC
  getAttribute(const T bin, const T particle_id_in_bin){
    if (ATTRIBUTE < attribs_e::START) return (PREC)ATTRIBUTE;
    else if (ATTRIBUTE >= attribs_e::END) return (PREC)attribs_e::INVALID_CT;
    else return this->ch(std::integral_constant<unsigned, 0>{}, bin).val_1d(std::integral_constant<unsigned, std::min(abs(ATTRIBUTE), attribs_e::END-1)>{}, particle_id_in_bin);
  }

  template <typename T = PREC>
   __device__ void
  getVelocity(const T bin, const T particle_id_in_bin, PREC * velocity) {
    velocity = {
      this->ch(std::integral_constant<unsigned, 0>{}, bin).val_1d(std::integral_constant<unsigned, attribs_e::Velocity_X>{}, particle_id_in_bin),
      this->ch(std::integral_constant<unsigned, 0>{}, bin).val_1d(std::integral_constant<unsigned, attribs_e::Velocity_Y>{}, particle_id_in_bin),
      this->ch(std::integral_constant<unsigned, 0>{}, bin).val_1d(std::integral_constant<unsigned, attribs_e::Velocity_Z>{}, particle_id_in_bin)
    };
  }

  template <typename T>
   __device__ constexpr void
  getDefGrad(T bin, T particle_id_in_bin, PREC * DefGrad) {
    DefGrad[0] = this->ch(std::integral_constant<unsigned, 0>{}, bin).val_1d(std::integral_constant<unsigned, attribs_e::DefGrad_XX>{}, particle_id_in_bin);
    DefGrad[1] = this->ch(std::integral_constant<unsigned, 0>{}, bin).val_1d(std::integral_constant<unsigned, attribs_e::DefGrad_XY>{}, particle_id_in_bin);
    DefGrad[2] = this->ch(std::integral_constant<unsigned, 0>{}, bin).val_1d(std::integral_constant<unsigned, attribs_e::DefGrad_XZ>{}, particle_id_in_bin);
    DefGrad[3] = this->ch(std::integral_constant<unsigned, 0>{}, bin).val_1d(std::integral_constant<unsigned, attribs_e::DefGrad_YX>{}, particle_id_in_bin);
    DefGrad[4] = this->ch(std::integral_constant<unsigned, 0>{}, bin).val_1d(std::integral_constant<unsigned, attribs_e::DefGrad_YY>{}, particle_id_in_bin);
    DefGrad[5] = this->ch(std::integral_constant<unsigned, 0>{}, bin).val_1d(std::integral_constant<unsigned, attribs_e::DefGrad_YZ>{}, particle_id_in_bin);
    DefGrad[6] = this->ch(std::integral_constant<unsigned, 0>{}, bin).val_1d(std::integral_constant<unsigned, attribs_e::DefGrad_ZX>{}, particle_id_in_bin);
    DefGrad[7] = this->ch(std::integral_constant<unsigned, 0>{}, bin).val_1d(std::integral_constant<unsigned, attribs_e::DefGrad_ZY>{}, particle_id_in_bin);
    DefGrad[8] = this->ch(std::integral_constant<unsigned, 0>{}, bin).val_1d(std::integral_constant<unsigned, attribs_e::DefGrad_ZZ>{}, particle_id_in_bin);
  }

  // TODO: Change logp0 to use particle held value, not initial
  template <typename T = PREC>
   __device__ constexpr void
  getStress_Cauchy(vec<T,9>& F, vec<T,9>& PF){
    PREC lj = logJp0;
    compute_stress_nacc(volume, mu, lambda, bm, xi, beta, Msqr, hardeningOn, lj, F, PF);
  }
  template <typename T = PREC>
   __device__ constexpr void
  getStress_Cauchy(T vol, vec<T,9>& F, vec<T,9>& PF){
    PREC lj = logJp0;
    compute_stress_nacc(vol, mu, lambda, bm, xi, beta, Msqr, hardeningOn, lj, F, PF);
  }
  
  template <typename T = PREC>
   __device__ void
  getStress_PK1(vec<T,9>& F, vec<T,9>& P){
    PREC lj = logJp0;
    compute_stress_PK1_nacc(volume, mu, lambda, bm, xi, beta, Msqr, hardeningOn, lj, F, P);
  }
  template <typename T = PREC>
   __device__ void
  getStress_PK1(T vol, vec<T,9>& F, vec<T,9>& P){
    PREC lj = logJp0;
    compute_stress_PK1_nacc(vol, mu, lambda, bm, xi, beta, Msqr, hardeningOn, lj, F, P);
  }

  template <typename T = PREC>
   __device__ void
  getEnergy_Strain(const vec<T,9>& F, T& strain_energy, T vol){
    strain_energy = 0.;
  }
  template <typename T = PREC>
   __device__ void
  getEnergy_Strain(const vec<T,9>& F, T& strain_energy){
    strain_energy = 0.;
  }
  template <typename T = PREC>
   __device__ void
  getEnergy_Kinetic(const vec<T,3>& velocity, T& kinetic_energy){  
    kinetic_energy = 0.5 * mass * __fma_rn(velocity[0], velocity[0], __fma_rn(velocity[1], velocity[1], (velocity[2], velocity[2])));
    }

  template <typename Allocator>
  ParticleBuffer(Allocator allocator) : base_t{allocator} {}

  template <typename Allocator>
  ParticleBuffer(Allocator allocator, std::size_t count)
      : base_t{allocator, count} {}
};


template <>
struct ParticleBuffer<material_e::Meshed>
    : ParticleBufferImpl<material_e::Meshed> {
  using base_t = ParticleBufferImpl<material_e::Meshed>;
  PREC length = DOMAIN_LENGTH; // Domain total length [m] (scales volume, etc.)
  PREC rho = DENSITY;
  PREC volume = DOMAIN_VOLUME * (1.f / (1 << DOMAIN_BITS) / (1 << DOMAIN_BITS) /
                  (1 << DOMAIN_BITS) / MODEL_PPC);
  PREC mass = (volume * DENSITY);
  PREC E = YOUNGS_MODULUS;
  PREC nu = POISSON_RATIO;
  PREC lambda = YOUNGS_MODULUS * POISSON_RATIO /
                 ((1 + POISSON_RATIO) * (1 - 2 * POISSON_RATIO));
  PREC mu = YOUNGS_MODULUS / (2 * (1 + POISSON_RATIO));
  bool use_ASFLIP = false; //< Use ASFLIP/PIC mixing? Default off.
  PREC alpha = 0.0;  //< FLIP/PIC Mixing Factor [0.1] -> [PIC, FLIP]
  PREC beta_min = 0.0; //< ASFLIP Minimum Position Correction Factor  
  PREC beta_max = 0.0; //< ASFLIP Maximum Position Correction Factor 
  PREC FBAR_ratio = 0.0; //< F-Bar Anti-locking mixing ratio (0 = None, 1 = Full)
  bool use_FEM = false; //< Use Finite Elements? Default off. Must set mesh
  bool use_FBAR = false; //< Use Simple F-Bar anti-locking? Default off.
  void updateParameters(PREC l, config::MaterialConfigs mat, 
                        config::AlgoConfigs algo) {
    length = l;
    rho = mat.rho;
    volume = length*length*length * ( 1.f / (1 << DOMAIN_BITS) / (1 << DOMAIN_BITS) /
                    (1 << DOMAIN_BITS) / mat.ppc);
    E = mat.E;
    nu = mat.nu;
    mass = volume * mat.rho;
    lambda = mat.E * mat.nu / ((1 + mat.nu) * (1 - 2 * mat.nu));
    mu = mat.E / (2 * (1 + mat.nu));
    alpha = algo.ASFLIP_alpha;
    beta_min = algo.ASFLIP_beta_min;
    beta_max = algo.ASFLIP_beta_max;
    FBAR_ratio = algo.FBAR_ratio;
    use_ASFLIP = algo.use_ASFLIP;
    use_FEM = algo.use_FEM;
    use_FBAR = algo.use_FBAR;
  }

  enum attribs_e : int {
          EMPTY=-3, // Empty attribute request 
          INVALID_CT=-2, // Invalid compile-time request, e.g. asking for variable after END
          INVALID_RT=-1, // Invalid run-time request e.g. "Speed_X" instead of "Velocity_X"
          START=0, // Values less than or equal to START not held on particle
          Position_X=0, Position_Y=1, Position_Z=2,
          ID,
          Velocity_X, Velocity_Y, Velocity_Z,
          J, DefGrad_Determinant=J, 
          JBar, DefGrad_Determinant_FBAR=JBar, 
          Pressure, VonMisesStress,
          END, // Values greater than or equal to END not held on particle
          // REQUIRED: Put N/A variables for specific material below END
          DefGrad_XX, DefGrad_XY, DefGrad_XZ,
          DefGrad_YX, DefGrad_YY, DefGrad_YZ,
          DefGrad_ZX, DefGrad_ZY, DefGrad_ZZ,
          Volume_FBAR, 
          logJp
  };

  // TODO : Change if/else statement to case/switch. may require compile-time min-max guarantee
  template <attribs_e ATTRIBUTE, typename T>
   __device__ PREC
  getAttribute(const T bin, const T particle_id_in_bin){
    if (ATTRIBUTE < attribs_e::START) return (PREC)ATTRIBUTE;
    else if (ATTRIBUTE >= attribs_e::END) return (PREC)attribs_e::INVALID_CT;
    else return this->ch(std::integral_constant<unsigned, 0>{}, bin).val_1d(std::integral_constant<unsigned, std::min(abs(ATTRIBUTE), attribs_e::END-1)>{}, particle_id_in_bin);
  }

  template <typename T = PREC>
   __device__ void
  getVelocity(const T bin, const T particle_id_in_bin, PREC * velocity) {
    velocity = {
      this->ch(std::integral_constant<unsigned, 0>{}, bin).val_1d(std::integral_constant<unsigned, attribs_e::Velocity_X>{}, particle_id_in_bin),
      this->ch(std::integral_constant<unsigned, 0>{}, bin).val_1d(std::integral_constant<unsigned, attribs_e::Velocity_Y>{}, particle_id_in_bin),
      this->ch(std::integral_constant<unsigned, 0>{}, bin).val_1d(std::integral_constant<unsigned, attribs_e::Velocity_Z>{}, particle_id_in_bin)
    };
  }

  template <typename T>
   __device__ constexpr void
  getDefGrad(T bin, T particle_id_in_bin, PREC * DefGrad) {
    for (int d=0; d<9; d++) DefGrad[d] = 0.;
    DefGrad[0] = DefGrad[4] = DefGrad[8] = 1.;
  }

  template <typename T = PREC>
   __device__ void
  getStress_Cauchy(const vec<T,9>& F, vec<T,9>& PF){
    compute_stress_fixedcorotated(volume, mu, lambda, F, PF);
  }
  template <typename T = PREC>
   __device__ void
  getStress_Cauchy(T vol, const vec<T,9>& F, vec<T,9>& PF){
    compute_stress_fixedcorotated(vol, mu, lambda, F, PF);
  }
  
  
  template <typename Allocator>
  ParticleBuffer(Allocator allocator) : base_t{allocator} {}

  template <typename Allocator>
  ParticleBuffer(Allocator allocator, std::size_t count)
      : base_t{allocator, count} {}
};

/// Reference: http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2018/p0608r3.html
/// * Make sure to add new materials to this
using particle_buffer_t =
    variant<ParticleBuffer<material_e::JFluid>,
            ParticleBuffer<material_e::JFluid_ASFLIP>,
            ParticleBuffer<material_e::JFluid_FBAR>,
            ParticleBuffer<material_e::JBarFluid>,
            ParticleBuffer<material_e::FixedCorotated>,
            ParticleBuffer<material_e::FixedCorotated_ASFLIP>,
            ParticleBuffer<material_e::FixedCorotated_ASFLIP_FBAR>,
            ParticleBuffer<material_e::NeoHookean_ASFLIP_FBAR>,
            ParticleBuffer<material_e::Sand>, 
            ParticleBuffer<material_e::NACC>,
            ParticleBuffer<material_e::Meshed>>;

using particle_array_ =
    structural<structural_type::dynamic,
               decorator<structural_allocation_policy::full_allocation,
                         structural_padding_policy::compact>,
               ParticleArrayDomain, attrib_layout::aos, f_, f_, f_>;
using particle_array_0_ =
    structural<structural_type::dynamic,
               decorator<structural_allocation_policy::full_allocation,
                         structural_padding_policy::compact>,
               ParticleArrayDomain, attrib_layout::aos, empty_>;
using particle_array_1_ =
    structural<structural_type::dynamic,
               decorator<structural_allocation_policy::full_allocation,
                         structural_padding_policy::compact>,
               ParticleArrayDomain, attrib_layout::aos, f_>;
using particle_array_2_ =
    structural<structural_type::dynamic,
               decorator<structural_allocation_policy::full_allocation,
                         structural_padding_policy::compact>,
               ParticleArrayDomain, attrib_layout::aos, f_, f_>;
using particle_array_3_ =
    structural<structural_type::dynamic,
               decorator<structural_allocation_policy::full_allocation,
                         structural_padding_policy::compact>,
               ParticleArrayDomain, attrib_layout::aos, f_, f_, f_>;
using particle_array_4_ =
    structural<structural_type::dynamic,
               decorator<structural_allocation_policy::full_allocation,
                         structural_padding_policy::compact>,
               ParticleArrayDomain, attrib_layout::aos, f_, f_, f_, f_>;
using particle_array_5_ =
    structural<structural_type::dynamic,
               decorator<structural_allocation_policy::full_allocation,
                         structural_padding_policy::compact>,
               ParticleArrayDomain, attrib_layout::aos, f_, f_, f_, f_, f_>;
using particle_array_6_ =
    structural<structural_type::dynamic,
               decorator<structural_allocation_policy::full_allocation,
                         structural_padding_policy::compact>,
               ParticleArrayDomain, attrib_layout::aos, f_, f_, f_, f_, f_, f_>;
using particle_array_7_ =
    structural<structural_type::dynamic,
               decorator<structural_allocation_policy::full_allocation,
                         structural_padding_policy::compact>,
               ParticleArrayDomain, attrib_layout::aos, f_, f_, f_, f_, f_, f_, f_>;
using particle_array_8_ =
    structural<structural_type::dynamic,
               decorator<structural_allocation_policy::full_allocation,
                         structural_padding_policy::compact>,
               ParticleArrayDomain, attrib_layout::aos, f_, f_, f_, f_, f_, f_, f_, f_>;
using particle_array_9_ =
    structural<structural_type::dynamic,
               decorator<structural_allocation_policy::full_allocation,
                         structural_padding_policy::compact>,
               ParticleArrayDomain, attrib_layout::aos, f_, f_, f_, f_, f_, f_, f_, f_, f_>;
using particle_array_10_ =
    structural<structural_type::dynamic,
               decorator<structural_allocation_policy::full_allocation,
                         structural_padding_policy::compact>,
               ParticleArrayDomain, attrib_layout::aos, f_, f_, f_, f_, f_, f_, 
               f_, f_, f_, f_>;
using particle_array_11_ =
    structural<structural_type::dynamic,
               decorator<structural_allocation_policy::full_allocation,
                         structural_padding_policy::compact>,
               ParticleArrayDomain, attrib_layout::aos, f_, f_, f_, f_, f_, f_, 
               f_, f_, f_, f_, f_>;
using particle_array_12_ =
    structural<structural_type::dynamic,
               decorator<structural_allocation_policy::full_allocation,
                         structural_padding_policy::compact>,
               ParticleArrayDomain, attrib_layout::aos, f_, f_, f_, f_, f_, f_, 
               f_, f_, f_, f_, f_, f_>;
using particle_array_13_ =
    structural<structural_type::dynamic,
               decorator<structural_allocation_policy::full_allocation,
                         structural_padding_policy::compact>,
               ParticleArrayDomain, attrib_layout::aos, f_, f_, f_, f_, f_, f_, 
               f_, f_, f_, f_, f_, f_, f_>;
using particle_array_14_ =
    structural<structural_type::dynamic,
               decorator<structural_allocation_policy::full_allocation,
                         structural_padding_policy::compact>,
               ParticleArrayDomain, attrib_layout::aos, f_, f_, f_, f_, f_, f_, 
               f_, f_, f_, f_, f_, f_, f_, f_>;
using particle_array_15_ =
    structural<structural_type::dynamic,
               decorator<structural_allocation_policy::full_allocation,
                         structural_padding_policy::compact>,
               ParticleArrayDomain, attrib_layout::aos, f_, f_, f_, f_, f_, f_, 
               f_, f_, f_, f_, f_, f_, f_, f_, f_>;
using particle_array_16_ =
    structural<structural_type::dynamic,
               decorator<structural_allocation_policy::full_allocation,
                         structural_padding_policy::compact>,
               ParticleArrayDomain, attrib_layout::aos, f_, f_, f_, f_, f_, f_, 
               f_, f_, f_, f_, f_, f_, f_, f_, f_, f_>;
using particle_array_18_ =
    structural<structural_type::dynamic,
               decorator<structural_allocation_policy::full_allocation,
                         structural_padding_policy::compact>,
               ParticleArrayDomain, attrib_layout::aos, f_, f_, f_, f_, f_, f_, 
               f_, f_, f_, f_, f_, f_, f_, f_, f_, f_, f_, f_>;
using particle_array_24_ =
    structural<structural_type::dynamic,
               decorator<structural_allocation_policy::full_allocation,
                         structural_padding_policy::compact>,
               ParticleArrayDomain, attrib_layout::aos, f_, f_, f_, f_, f_, f_, 
               f_, f_, f_, f_, f_, f_, f_, f_, f_,
               f_, f_, f_, f_, f_, f_, f_, f_, f_>;
using particle_array_32_ =
    structural<structural_type::dynamic,
               decorator<structural_allocation_policy::full_allocation,
                         structural_padding_policy::compact>,
               ParticleArrayDomain, attrib_layout::aos, f_, f_, f_, f_, f_, f_, 
               f_, f_, f_, f_, f_, f_, f_, f_, f_,
               f_, f_, f_, f_, f_, f_, f_, f_, f_,
               f_, f_, f_, f_, f_, f_, f_, f_, f_>;

struct ParticleArray : Instance<particle_array_> {
  using base_t = Instance<particle_array_>;
  ParticleArray &operator=(base_t &&instance) {
    static_cast<base_t &>(*this) = instance;
    return *this;
  }
  //ParticleArray(base_t &&instance) { static_cast<base_t &>(*this) = instance; }
};

template <num_attribs_e N> struct particle_attrib_;
template <> struct particle_attrib_<num_attribs_e::Zero> : particle_array_0_ {};
template <> struct particle_attrib_<num_attribs_e::One> : particle_array_1_ {};
template <> struct particle_attrib_<num_attribs_e::Two> : particle_array_2_ {};
template <> struct particle_attrib_<num_attribs_e::Three> : particle_array_3_ {};
template <> struct particle_attrib_<num_attribs_e::Four> : particle_array_4_ {}; //
template <> struct particle_attrib_<num_attribs_e::Five> : particle_array_5_ {}; //
template <> struct particle_attrib_<num_attribs_e::Six> : particle_array_6_ {};
template <> struct particle_attrib_<num_attribs_e::Seven> : particle_array_7_ {}; //
template <> struct particle_attrib_<num_attribs_e::Eight> : particle_array_8_ {}; //
template <> struct particle_attrib_<num_attribs_e::Nine> : particle_array_9_ {};
template <> struct particle_attrib_<num_attribs_e::Ten> : particle_array_10_ {}; //
template <> struct particle_attrib_<num_attribs_e::Eleven> : particle_array_11_ {}; //
template <> struct particle_attrib_<num_attribs_e::Twelve> : particle_array_12_ {};
template <> struct particle_attrib_<num_attribs_e::Thirteen> : particle_array_13_ {};
// template <> struct particle_attrib_<num_attribs_e::Fourteen> : particle_array_14_ {};
// template <> struct particle_attrib_<num_attribs_e::Fifteen> : particle_array_15_ {};
// template <> struct particle_attrib_<num_attribs_e::Sixteen> : particle_array_16_ {};
// template <> struct particle_attrib_<num_attribs_e::Eighteen> : particle_array_18_ {};
// template <> struct particle_attrib_<num_attribs_e::Twentyfour> : particle_array_24_ {};
// template <> struct particle_attrib_<num_attribs_e::Thirtytwo> : particle_array_32_ {};

template<num_attribs_e N=num_attribs_e::Three>
struct ParticleAttrib: Instance<particle_attrib_<N>> {
  static constexpr unsigned numAttributes = static_cast<unsigned>(N);
  using base_t = Instance<particle_attrib_<N>>;
  
  template <typename Allocator>
  ParticleAttrib(Allocator allocator) : base_t{spawn<particle_attrib_<N>, orphan_signature>(allocator)} { 
      std::cout << "ParticleAttrib constructor with an allocator." << "\n";
    }
  ParticleAttrib &operator=(base_t &&instance) {
    std::cout << "ParticleAttrib move assignment operator called." << "\n";
    static_cast<base_t &>(*this) = instance;
    return *this;
  }
  ParticleAttrib(base_t &&instance) { 
    std::cout << "ParticleAttrib move constructor called." << "\n";
    static_cast<base_t &>(*this) = instance; 
    }

  template <num_attribs_e ATTRIBUTE, typename T>
   __device__ PREC
  getAttribute(T parid){
    if (ATTRIBUTE >= N) return (PREC)-1;
    else return this->val(std::integral_constant<unsigned, std::min(ATTRIBUTE, N)>{}, parid);
  }
  template <num_attribs_e ATTRIBUTE, typename T>
   __device__ void
  getAttribute(T parid, PREC & val){
    if (ATTRIBUTE >= N) val = (PREC)-1;
    else val = this->val(std::integral_constant<unsigned, std::min(ATTRIBUTE, N)>{}, parid);
  }
  // template <num_attribs_e ATTRIBUTE, typename T>
  //  __device__ constexpr void
  // setAttribute(const T parid, const PREC val){
  //   if (ATTRIBUTE >= this.numAttributes) return;
  //   else{
  //     this->val(std::integral_constant<unsigned, std::min(ATTRIBUTE, this.numAttributes)>{}, parid) = val;
  //     return;
  //   }
  // }
};

using particle_attrib_t =
    variant<ParticleAttrib<num_attribs_e::One>,
            ParticleAttrib<num_attribs_e::Two>,
            ParticleAttrib<num_attribs_e::Three>,
            ParticleAttrib<num_attribs_e::Four>,
            ParticleAttrib<num_attribs_e::Five>,
            ParticleAttrib<num_attribs_e::Six>,
            ParticleAttrib<num_attribs_e::Seven>,
            ParticleAttrib<num_attribs_e::Eight>,
            ParticleAttrib<num_attribs_e::Nine>,
            ParticleAttrib<num_attribs_e::Ten>,
            ParticleAttrib<num_attribs_e::Eleven>,
            ParticleAttrib<num_attribs_e::Twelve>,
            ParticleAttrib<num_attribs_e::Thirteen>
            // ParticleAttrib<num_attribs_e::Fourteen>,
            // ParticleAttrib<num_attribs_e::Fifteen>,
            // ParticleAttrib<num_attribs_e::Sixteen>,
            // ParticleAttrib<num_attribs_e::Eighteen>,
            // ParticleAttrib<num_attribs_e::Twentyfour>,
            // ParticleAttrib<num_attribs_e::Thirtytwo>
            >;


using particle_target_ =
    structural<structural_type::dynamic,
               decorator<structural_allocation_policy::full_allocation,
                         structural_padding_policy::compact>,
               ParticleTargetDomain, attrib_layout::aos, f_, f_, f_, f_, f_, f_, f_, f_, f_, f_>;

/// * ParticleTarget structure for device instantiation 
struct ParticleTarget : Instance<particle_target_> {
  using base_t = Instance<particle_target_>;
  ParticleTarget &operator=(base_t &&instance) {
    std::cout << "ParticleTarget move assignment operator." << "\n";
    static_cast<base_t &>(*this) = instance;
    return *this;
  }
  ParticleTarget(base_t &&instance) { 
    std::cout << "ParticleTarget move constructor." << "\n";
    static_cast<base_t &>(*this) = instance; 
    }
};

} // namespace mn

#endif