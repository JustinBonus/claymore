#include "mgsp_benchmark.cuh"
#include "partition_domain.h"
#include <MnBase/Geometry/GeometrySampler.h>
#include <MnBase/Math/Vec.h>
#include <MnSystem/Cuda/Cuda.h>
#include <MnSystem/IO/IO.h>
#include <MnSystem/IO/ParticleIO.hpp>


#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
typedef std::vector<std::array<float, 3>> WaveHolder;


// dragon_particles.bin, 775196
// cube256_2.bin, 1048576
// two_dragons.bin, 388950

decltype(auto) load_model(std::size_t pcnt, std::string filename) {
  std::vector<std::array<float, 3>> rawpos(pcnt);
  auto addr_str = std::string(AssetDirPath) + "MpmParticles/";
  auto f = fopen((addr_str + filename).c_str(), "rb");
  auto res = std::fread((float *)rawpos.data(), sizeof(float), rawpos.size() * 3, f);
  std::fclose(f);
  return rawpos;
}

void load_waveMaker(const std::string& filename, char sep, WaveHolder& fields){
  auto addr_str = std::string(AssetDirPath) + "WaveMaker/";
  std::ifstream in((addr_str + filename).c_str());
  if (in) {
      std::string line;
      while (getline(in, line)) {
          std::stringstream sep(line);
          std::string field;
          int col = 0;
          std::array<float, 3> arr;
          while (getline(sep, field, ',')) {
              if (col >= 3) break;
              arr[col] = stof(field);
              col++;
          }
          fields.push_back(arr);
      }
  }
  // for (auto row : fields) {
  //     for (auto field : row) {
  //         std::cout << field << ' ';
  //     }
  //     std::cout << '\n';
  // }
}

// load from analytic levelset
// init models
void init_models(
    std::vector<std::array<float, 3>> models[mn::config::g_device_cnt],
    int opt = 0) {
  using namespace mn;
  using namespace config;
  switch (opt) {
  case 0:
    models[0] = load_model(775196, "dragon_particles.bin");
    models[1] = read_sdf(std::string{"two_dragons.sdf"}, 8.f, g_dx,
                         vec<float, 3>{0.5f, 0.5f, 0.5f},
                         vec<float, 3>{0.2f, 0.2f, 0.2f});
    break;
  case 1:
    models[0] = load_model(775196, "dragon_particles.bin");
    models[1] = load_model(775196, "dragon_particles.bin");
    for (auto &pt : models[1])
      pt[1] -= 0.3f;
    break;
  case 2: {
    constexpr auto LEN = 54;
    constexpr auto STRIDE = 56;
    constexpr auto MODEL_CNT = 1;
    for (int did = 0; did < g_device_cnt; ++did) {
      models[did].clear();
      std::vector<std::array<float, 3>> model;
      for (int i = 0; i < MODEL_CNT; ++i) {
        auto idx = (did * MODEL_CNT + i);
        model = sample_uniform_box(
            g_dx, ivec3{18 + (idx & 1 ? STRIDE : 0), 18, 18},
            ivec3{18 + (idx & 1 ? STRIDE : 0) + LEN, 18 + LEN, 18 + LEN});
        models[did].insert(models[did].end(), model.begin(), model.end());
      }
    }
  } break;
  case 3: {
    constexpr auto LEN = 32; // 54;
    constexpr auto STRIDE = (g_domain_size / 4);
    constexpr auto MODEL_CNT = 1;
    for (int did = 0; did < g_device_cnt; ++did) {
      models[did].clear();
      std::vector<std::array<float, 3>> model;
      for (int i = 0; i < MODEL_CNT; ++i) {
        auto idx = (did * MODEL_CNT + i);
        // model = sample_uniform_box(
        //     g_dx,
        //     ivec3{18 + (idx & 1 ? STRIDE : 0), 18, 18 + (idx & 2 ? STRIDE : 0)},
        //     ivec3{18 + (idx & 1 ? STRIDE : 0) + LEN, 18 + LEN / 3,
        //           18 + (idx & 2 ? STRIDE : 0) + LEN});
        model = sample_uniform_box(
            g_dx,
            ivec3{8 + (idx & 1 ? STRIDE : 0), 8, 8},
            ivec3{8 + (idx & 1 ? STRIDE : 0) + LEN, 8 + LEN / 3,
                  8 + LEN});
                  
        models[did].insert(models[did].end(), model.begin(), model.end());
      }
    }
  } break;
  case 4: {
      float off = 8.f * g_dx;
      float f = 1.f;
      float water_ppc = MODEL_PPC;

      if (g_device_cnt == 1) {
        vec<float, 3> water_offset{0.f, 0.f, 0.f};
        water_offset /= g_length;
        water_offset = water_offset + off;
        vec<float, 3> water_lengths{12.f, 0.2f, 0.9f};
        water_lengths /= g_length;
        models[0] = read_sdf(std::string{"Water/Water_x12_y0.2_z0.9_dx0.02_pad1.sdf"}, 
                          water_ppc, mn::config::g_dx, mn::config::g_domain_size,
                          water_offset, water_lengths);

      } else if (g_device_cnt == 2) {
        vec<float, 3> water_offset{0.f, 0.f, 0.f};
        water_offset /= g_length;
        water_offset = water_offset + off;
        vec<float, 3> water_lengths{12.f, 0.2f, 0.9f};
        water_lengths /= g_length;
        models[0] = read_sdf(std::string{"Water/Water_x12_y0.2_z0.9_dx0.02_pad1.sdf"}, 
                          water_ppc, mn::config::g_dx, mn::config::g_domain_size,
                          water_offset, water_lengths);

        vec<float, 3> debris_offset{42.f, 2.f, 0.7538f};
        debris_offset /= g_length;
        debris_offset = debris_offset + off;
        vec<float, 3> debris_lengths{0.558f, 0.051f, 2.15f};
        debris_lengths /= g_length;
        float debris_ppc = MODEL_PPC_FC;
        models[1] = read_sdf(std::string{"Debris/OSU_AT162_spacing_5cm_dx0.005_pad1.sdf"}, 
                          debris_ppc, mn::config::g_dx, mn::config::g_domain_size,
                          debris_offset, debris_lengths);
      }
  }
  default:
    break;
  }
}

int main() {
  using namespace mn;
  using namespace config;
  Cuda::startup();

  std::vector<std::array<float, 3>> models[g_device_cnt];
  std::vector<std::array<float, 10>> h_gridTarget(mn::config::g_target_cells, 
                                                  std::array<float, 10>{0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f});
  float off = 8.f * g_dx;
  
  vec<float, 3> h_point_a;
  vec<float, 3> h_point_b;
  h_point_a[0] = 12.f / g_length + (2.f * g_dx);
  h_point_a[1] = 0.f / g_length;
  h_point_a[2] = (0.9f - 0.254f) / 2.f / g_length;
  h_point_a = h_point_a + off;
  h_point_b[0] = h_point_a[0] + (1.01f * g_dx);
  h_point_b[1] = h_point_a[1] + 0.500f / g_length;
  h_point_b[2] = h_point_a[2] + 0.254f / g_length;

  /// Initialize
  auto benchmark = std::make_unique<mgsp_benchmark>();
  init_models(models, 4);

  /// Loop through GPU devices
  for (int did = 0; did < g_device_cnt; ++did) {
    benchmark->initModel(did, models[did]);
    benchmark->initGridTarget(did, h_gridTarget, h_point_a, h_point_b, 1.f);
  }
  benchmark->main_loop();
  ///
  IO::flush();
  benchmark.reset();
  ///
  Cuda::shutdown();
  return 0;
}