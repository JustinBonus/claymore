#include "mgsp_benchmark.cuh"
#include "read_scene_input.h"
#include "partition_domain.h"

#include <MnBase/Geometry/GeometrySampler.h>
#include <MnBase/Math/Vec.h>
#include <MnSystem/Cuda/Cuda.h>
#include <MnSystem/IO/IO.h>
#include <MnSystem/IO/ParticleIO.hpp>

#include <cxxopts.hpp>
#include <fmt/color.h>
#include <fmt/core.h>

#include <string>
#include <vector>
#include <array>
//#include <thread>

#if 0
#include <ghc/filesystem.hpp>
namespace fs = ghc::filesystem;
#else
#include <experimental/filesystem>
namespace fs = std::experimental::filesystem;
#endif

// #include <rapidjson/document.h>
// #include <rapidjson/stringbuffer.h>
// #include <rapidjson/writer.h>
// namespace rj = rapidjson;


/// @brief Multi-GPU MPM for Engineers. Builds on the original, open-source Claymore MPM, all rights reserved. Claymore MPM for Engineers: https://github.com/JustinBonus/claymore . Original Claymore MPM : https://github.com/penn-graphics-research/claymore . For an executable [test] with scene file [scene_test.json]
/// @param argv  For an executable [test] with desired scene file [scene_test.json], use Command-line: ./test --file = scene_test.json  (NOTE : defaults to scene.json if not specified)
int main(int argc, char *argv[]) {
  using namespace mn;
  using namespace config;
  
  IO;

  Cuda::startup(); //< Start CUDA GPUs if available.
  // ---------------- Read JSON input file for simulation ---------------- 
  cxxopts::Options options("Scene_Loader", "Read simulation scene");
  options.add_options()("f,file", "Scene Configuration File",
      cxxopts::value<std::string>()->default_value("scene.json")); //< scene.json is default
  auto results = options.parse(argc, argv);
  auto fn = results["file"].as<std::string>();
  fmt::print(fg(fmt::color::green),"Find scene file from command-line option --file={}\n", fn);
  
  // ---------------- Initialize the simulation ---------------- 
  std::unique_ptr<mn::mgsp_benchmark> simulator; //< Simulation object pointer
  std::vector<std::array<PREC, 3>> models[g_model_cnt]; //< Initial particle positions
  parse_scene(fn, simulator, models); //< Initialize from input scene file  
  fmt::print(fg(fmt::color::green),"Finished scene initialization.\n");

  // ---------------- Run Simulation
  fmt::print(fg(fmt::color::cyan),"Starting simulation...\n");
  if (g_log_level > 1) { fmt::print(fg(fmt::color::blue),"Press ENTER... \n"); getchar(); }
  simulator->main_loop();
  if (simulator == nullptr) fmt::print(fg(fmt::color::green),"Simulator nullptr after scope.\n"); 
  
  fmt::print(fg(fmt::color::green), "Finished simulation.\n");

  // ---------------- Clear
  IO::flush(); 
  fmt::print(fg(fmt::color::green),"Cleared I/O.\n");
  simulator.reset();
  fmt::print(fg(fmt::color::green),"Reset simulator.\n");
  if(simulator == nullptr) fmt::print(fg(fmt::color::green),"Simulation nullptr after reset.\n");
  
  // ---------------- Shutdown GPU / CUDA
  Cuda::shutdown();
  fmt::print(fg(fmt::color::green),"Shut-down CUDA GPU communication.\n");
  cudaDeviceReset();
  fmt::print(fg(fmt::color::green),"Reset CUDA GPUs.\n");

  // ---------------- Finish application
  fmt::print(fg(fmt::color::green),"Application finished.\n");
  return 0;
}