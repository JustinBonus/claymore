#ifndef __GMPM_SIMULATOR_CUH_
#define __GMPM_SIMULATOR_CUH_
#include "grid_buffer.cuh"
#include "hash_table.cuh"
#include "mgmpm_kernels.cuh"
#include "particle_buffer.cuh"
#include "settings.h"
#include <MnBase/Concurrency/Concurrency.h>
#include <MnBase/Meta/ControlFlow.h>
#include <MnBase/Meta/TupleMeta.h>
#include <MnBase/Profile/CppTimers.hpp>
#include <MnBase/Profile/CudaTimers.cuh>
#include <MnSystem/Cuda/Cuda.h>
#include <MnSystem/IO/IO.h>
#include <MnSystem/IO/ParticleIO.hpp>
#include <array>
#include <fmt/color.h>
#include <fmt/core.h>
#include <vector>

namespace mn {

struct GmpmSimulator {
  using streamIdx = Cuda::StreamIndex;
  using eventIdx = Cuda::EventIndex;
  using host_allocator = heap_allocator;
  struct device_allocator { // hide the global one
    void *allocate(std::size_t bytes) {
      void *ret;
      checkCudaErrors(cudaMalloc(&ret, bytes));
      return ret;
    }
    void deallocate(void *p, std::size_t) { checkCudaErrors(cudaFree(p)); }
  };
  void initialize() {
    auto &cuDev = Cuda::ref_cuda_context(gpuid);
    cuDev.setContext();
    tmps.alloc(config::g_max_active_block);
    for (int copyid = 0; copyid < 2; copyid++) {
      gridBlocks.emplace_back(device_allocator{});
      partitions.emplace_back(device_allocator{}, config::g_max_active_block);
      checkedCnts[copyid] = 0;
    }
    cuDev.syncStream<streamIdx::Compute>();
    curNumActiveBlocks = config::g_max_active_block;
  }
  GmpmSimulator(int gpu = 0, float dt = 1e-4, int fp = 24, int frames = 60)
      : gpuid{gpu}, dtDefault{dt}, curTime{0.f}, rollid{0},
        curFrame{(uint64_t)0}, curStep{(uint64_t)0}, fps{fp}, nframes{frames} {
    // data
    initialize();
  }
  ~GmpmSimulator() {}


  /// Initialize grid object from host (CPU) setting (&graph), output as *.bgeo (JB)
  /// Still inefficient for debugging purposes
  void initGrid(const std::vector<std::array<float, 7>> &graph) {
    auto &cuDev = Cuda::ref_cuda_context(gpuid);

    fmt::print("Just entered initGrid in gmpm_simulator.cuh!\n");

    /// Populate nodes (device) with appropiate grid_ structure and memory size (JB)
    nodes.emplace_back(
        std::move(GridArray{spawn<grid_array_, orphan_signature>(
            device_allocator{}, sizeof(float) * 7 * graph.size())}));
    
    fmt::print("Created nodes structure in initGrid gmpm_simulator.cuh!\n");

    /// Set size
    node_cnt.emplace_back(graph.size());


    /// Populate nodes (device) with data from graph (host) (JB)
    cudaMemcpyAsync((void *)&nodes.back().val_1d(_0, 0), graph.data(),
                    sizeof(std::array<float, 7>) * graph.size(),
                    cudaMemcpyDefault, cuDev.stream_compute());
    cuDev.syncStream<streamIdx::Compute>();

    fmt::print("Populated nodes with graph in initGrid gmpm_simulator.cuh!\n");


    /// Write graph data to a *.bgeo output file using Partio  
    std::string fn = std::string{"grid"} + "_frame[0].bgeo";
    IO::insert_job([fn, graph]() { write_partio_grid<float, 7>(fn, graph); });
    IO::flush();
    
    fmt::print("Exiting initGrid in gmpm_simulator.cuh!\n");
  }

  /// Initialize (host --> device) particle position objects (&model --> particles) 
  /// and velocity vector (&v0 --> vel0) 
  /// Also initialize particle attributes
  /// Output initial positions as *.bgeo file
  template <material_e m>
  void initModel(const std::vector<std::array<float, 3>> &model,
                 const mn::vec<float, 3> &v0) {
    auto &cuDev = Cuda::ref_cuda_context(gpuid);
    /// Establish double-buffered particle structures (device)
    /// Set proper material 
    for (int copyid = 0; copyid < 2; ++copyid) {
      particleBins[copyid].emplace_back(ParticleBuffer<m>(
          device_allocator{},
          model.size() / config::g_bin_capacity + config::g_max_active_block));
      match(particleBins[copyid].back())([&](auto &pb) {
        pb.reserveBuckets(device_allocator{}, config::g_max_active_block);
      });
    }
    /// Set initial velocity vector (host) (host --> host)
    vel0.emplace_back();
    for (int i = 0; i < 3; ++i)
      vel0.back()[i] = v0[i];

    /// Allocate memory for particles (device) and pattribs (device) (JB)
    particles.emplace_back(
        std::move(ParticleArray{spawn<particle_array_, orphan_signature>(
            device_allocator{}, sizeof(float) * 3 * model.size())}));
    pattribs.emplace_back(
        std::move(ParticleArray{spawn<particle_array_, orphan_signature>(
            device_allocator{}, sizeof(float) * 3 * model.size())}));

    /// Set-up particle bin parameters
    curNumActiveBins.emplace_back(config::g_max_particle_bin);
    bincnt.emplace_back(0);
    checkedBinCnts.emplace_back(0);

    /// Set particle count (host) and print
    pcnt.emplace_back(model.size());
    fmt::print("init {}-th model with {} particles\n",
               particleBins[0].size() - 1, pcnt.back());

    /// Copy particles (device) position data from model (host), asynchronous
    cudaMemcpyAsync((void *)&particles.back().val_1d(_0, 0), model.data(),
                    sizeof(std::array<float, 3>) * model.size(),
                    cudaMemcpyDefault, cuDev.stream_compute());
    cuDev.syncStream<streamIdx::Compute>();

    /// Copy pattribs (device) attribute data from model (host), asynchronous (JB)
    cudaMemcpyAsync((void *)&pattribs.back().val_1d(_0, 0), model.data(),
                    sizeof(std::array<float, 3>) * model.size(),
                    cudaMemcpyDefault, cuDev.stream_compute());
    cuDev.syncStream<streamIdx::Compute>();

    /// Write model data (host) to *.bgeo binary output (disk) using Partio
    /// Functions found in Library/MnSystem/IO/ParticleIO.hpp (JB)
    std::string fn = std::string{"model"} + "_id[" +
                     std::to_string(particleBins[0].size() - 1) +
                     "]_frame[0].bgeo";
    IO::insert_job([fn, model]() { write_partio<float, 3>(fn, model); });
    IO::flush();
  }
  void updateFRParameters(float rho, float vol, float ym, float pr) {
    match(particleBins[0].back())(
        [&](auto &pb) {},
        [&](ParticleBuffer<material_e::FixedCorotated> &pb) {
          pb.updateParameters(rho, vol, ym, pr);
        });
    match(particleBins[1].back())(
        [&](auto &pb) {},
        [&](ParticleBuffer<material_e::FixedCorotated> &pb) {
          pb.updateParameters(rho, vol, ym, pr);
        });
  }
  void updateJFluidParameters(float rho, float vol, float bulk, float gamma,
                              float visco) {
    match(particleBins[0].back())([&](auto &pb) {},
                                  [&](ParticleBuffer<material_e::JFluid> &pb) {
                                    pb.updateParameters(rho, vol, bulk, gamma,
                                                        visco);
                                  });
    match(particleBins[1].back())([&](auto &pb) {},
                                  [&](ParticleBuffer<material_e::JFluid> &pb) {
                                    pb.updateParameters(rho, vol, bulk, gamma,
                                                        visco);
                                  });
  }
  void updateNACCParameters(float rho, float vol, float ym, float pr,
                            float beta, float xi) {
    match(particleBins[0].back())([&](auto &pb) {},
                                  [&](ParticleBuffer<material_e::NACC> &pb) {
                                    pb.updateParameters(rho, vol, ym, pr, beta,
                                                        xi);
                                  });
    match(particleBins[1].back())([&](auto &pb) {},
                                  [&](ParticleBuffer<material_e::NACC> &pb) {
                                    pb.updateParameters(rho, vol, ym, pr, beta,
                                                        xi);
                                  });
  }
  void updateGridParameters(float gravity, float length, float cfl, float atm){
    gridBlocks[0].updateParameters(gravity, length, cfl, atm);
    gridBlocks[1].updateParameters(gravity, length, cfl, atm);
  }
  template <typename CudaContext>
  void exclScan(std::size_t cnt, int const *const in, int *out,
                CudaContext &cuDev) {
#if 1
    auto policy = thrust::cuda::par.on((cudaStream_t)cuDev.stream_compute());
    thrust::exclusive_scan(policy, getDevicePtr(in), getDevicePtr(in) + cnt,
                           getDevicePtr(out));
#else
    std::size_t temp_storage_bytes = 0;
    auto plus_op = [] __device__(const int &a, const int &b) { return a + b; };
    checkCudaErrors(cub::DeviceScan::ExclusiveScan(nullptr, temp_storage_bytes,
                                                   in, out, plus_op, 0, cnt,
                                                   cuDev.stream_compute()));
    void *d_tmp = tmps[cuDev.getDevId()].d_tmp;
    checkCudaErrors(cub::DeviceScan::ExclusiveScan(d_tmp, temp_storage_bytes,
                                                   in, out, plus_op, 0, cnt,
                                                   cuDev.stream_compute()));
#endif
  }
  float getMass(int id = 0) {
    return match(particleBins[rollid][id])(
        [&](const auto &particleBuffer) { return particleBuffer.mass; });
  }
  int getModelCnt() const noexcept { return (int)particleBins[0].size(); }
  void checkCapacity() {
    if (ebcnt > curNumActiveBlocks * 3 / 4 && checkedCnts[0] == 0) {
      curNumActiveBlocks = curNumActiveBlocks * 3 / 2;
      checkedCnts[0] = 2;
      fmt::print(fmt::emphasis::bold, "resizing blocks {} -> {}\n", ebcnt,
                 curNumActiveBlocks);
    }
    for (int i = 0; i < getModelCnt(); ++i)
      if (bincnt[i] > curNumActiveBins[i] * 3 / 4 && checkedBinCnts[i] == 0) {
        curNumActiveBins[i] = curNumActiveBins[i] * 3 / 2;
        checkedBinCnts[i] = 2;
        fmt::print(fmt::emphasis::bold, "resizing bins {} -> {}\n", bincnt[i],
                   curNumActiveBins[i]);
      }
  }
  void main_loop() {
    /// initial
    float nextTime = 1.f / fps;
    {
      float maxVel = 0.f;
      for (int i = 0; i < getModelCnt(); ++i) {
        float velNorm = std::sqrt(vel0[i].l2NormSqr());
        if (velNorm > maxVel)
          maxVel = velNorm;
      }
      dt = compute_dt(maxVel, curTime, nextTime, dtDefault);
    }
    fmt::print(fmt::emphasis::bold, "{} --{}--> {}, defaultDt: {}\n", curTime,
               dt, nextTime, dtDefault);
    initial_setup();
    curTime = dt;
    for (curFrame = 1; curFrame <= nframes; ++curFrame) {
      for (; curTime < nextTime; curTime += dt, curStep++) {
        /// max grid vel
        {
          auto &cuDev = Cuda::ref_cuda_context(gpuid);
          /// check capacity
          checkCapacity();
          float *d_maxVel = tmps.d_maxVel;
          CudaTimer timer{cuDev.stream_compute()};
          timer.tick();
          checkCudaErrors(cudaMemsetAsync(d_maxVel, 0, sizeof(float),
                                          cuDev.stream_compute()));
          cuDev.compute_launch({(nbcnt + g_num_grid_blocks_per_cuda_block - 1) /
                                    g_num_grid_blocks_per_cuda_block,
                                g_num_warps_per_cuda_block * 32,
                                g_num_warps_per_cuda_block},
                               update_grid_velocity_query_max, (uint32_t)nbcnt,
                               gridBlocks[0], partitions[rollid], dt, d_maxVel);
          checkCudaErrors(cudaMemcpyAsync(&maxVels, d_maxVel, sizeof(float),
                                          cudaMemcpyDefault,
                                          cuDev.stream_compute()));
          timer.tock(fmt::format("GPU[{}] frame {} step {} grid_update_query",
                                 gpuid, curFrame, curStep));
        }

        /// host: compute maxvel & next dt
        float maxVel = maxVels;
        // if (maxVels > maxVel)
        //  maxVel = maxVels[id];
        maxVel = std::sqrt(maxVel); // this is a bug, should insert this line
        nextDt = compute_dt(maxVel, curTime, nextTime, dtDefault);
        fmt::print(fmt::emphasis::bold,
                   "{} --{}--> {}, defaultDt: {}, maxVel: {}\n", curTime,
                   nextDt, nextTime, dtDefault, maxVel);

        /// g2p2g
        {
          auto &cuDev = Cuda::ref_cuda_context(gpuid);
          CudaTimer timer{cuDev.stream_compute()};

          /// check capacity
          for (int i = 0; i < getModelCnt(); ++i)
            if (checkedBinCnts[i] > 0) {
              match(particleBins[rollid ^ 1][i])([&](auto &pb) {
                pb.resize(device_allocator{}, curNumActiveBins[i]);
              });
              checkedBinCnts[i]--;
            }

          timer.tick();
          // grid
          gridBlocks[1].reset(nbcnt, cuDev);
          // adv map
          for (int i = 0; i < getModelCnt(); ++i) {
            match(particleBins[rollid ^ 1][i])([&](auto &pb) {
              checkCudaErrors(cudaMemsetAsync(
                  pb._ppcs, 0, sizeof(int) * ebcnt * g_blockvolume,
                  cuDev.stream_compute()));
            });
            // g2p2g
            match(particleBins[rollid][i])([&](const auto &pb) {
              cuDev.compute_launch({pbcnt, 128, (512 * 3 * 4) + (512 * 4 * 4)},
                                   g2p2g, dt, nextDt, pb,
                                   get<typename std::decay_t<decltype(pb)>>(
                                       particleBins[rollid ^ 1][i]),
                                   partitions[rollid ^ 1], partitions[rollid],
                                   gridBlocks[0], gridBlocks[1]);
            });
          }
          cuDev.syncStream<streamIdx::Compute>();
          timer.tock(fmt::format("GPU[{}] frame {} step {} g2p2g", gpuid,
                                 curFrame, curStep));
          if (checkedCnts[0] > 0) {
            partitions[rollid ^ 1].resizePartition(device_allocator{},
                                                   curNumActiveBlocks);
            for (int i = 0; i < getModelCnt(); ++i)
              match(particleBins[rollid][i])([&](auto &pb) {
                pb.reserveBuckets(device_allocator{}, curNumActiveBlocks);
              });
            checkedCnts[0]--;
          }
        }

        /// update partition
        {
          auto &cuDev = Cuda::ref_cuda_context(gpuid);
          CudaTimer timer{cuDev.stream_compute()};
          timer.tick();
          /// mark particle blocks
          for (int i = 0; i < getModelCnt(); ++i)
            match(particleBins[rollid ^ 1][i])([&](auto &pb) {
              checkCudaErrors(cudaMemsetAsync(pb._ppbs, 0,
                                              sizeof(int) * (ebcnt + 1),
                                              cuDev.stream_compute()));
              cuDev.compute_launch({ebcnt, config::g_blockvolume},
                                   cell_bucket_to_block, pb._ppcs,
                                   pb._cellbuckets, pb._ppbs, pb._blockbuckets);
              // partitions[rollid].buildParticleBuckets(cuDev, ebcnt);
            });

          int *activeBlockMarks = tmps.activeBlockMarks,
              *destinations = tmps.destinations, *sources = tmps.sources;
          checkCudaErrors(cudaMemsetAsync(activeBlockMarks, 0,
                                          sizeof(int) * nbcnt,
                                          cuDev.stream_compute()));
          /// mark grid blocks
          cuDev.compute_launch({(nbcnt * g_blockvolume + 127) / 128, 128},
                               mark_active_grid_blocks, (uint32_t)nbcnt,
                               gridBlocks[1], activeBlockMarks);
          /// mark particle blocks
          checkCudaErrors(cudaMemsetAsync(sources, 0, sizeof(int) * (ebcnt + 1),
                                          cuDev.stream_compute()));
          for (int i = 0; i < getModelCnt(); ++i)
            match(particleBins[rollid ^ 1][i])([&](auto &pb) {
              cuDev.compute_launch({(ebcnt + 1 + 127) / 128, 128},
                                   mark_active_particle_blocks, ebcnt + 1,
                                   pb._ppbs, sources);
            });
          exclScan(ebcnt + 1, sources, destinations, cuDev);
          /// building new partition
          // block count
          checkCudaErrors(cudaMemcpyAsync(
              partitions[rollid ^ 1]._cnt, destinations + ebcnt, sizeof(int),
              cudaMemcpyDefault, cuDev.stream_compute()));
          checkCudaErrors(cudaMemcpyAsync(&pbcnt, destinations + ebcnt,
                                          sizeof(int), cudaMemcpyDefault,
                                          cuDev.stream_compute()));
          cuDev.compute_launch({(ebcnt + 255) / 256, 256},
                               exclusive_scan_inverse, ebcnt,
                               (const int *)destinations, sources);
          // indextable, activeKeys, ppb, buckets
          partitions[rollid ^ 1].resetTable(cuDev.stream_compute());
          cuDev.syncStream<streamIdx::Compute>();
          cuDev.compute_launch({(pbcnt + 127) / 128, 128}, update_partition,
                               (uint32_t)pbcnt, (const int *)sources,
                               partitions[rollid], partitions[rollid ^ 1]);
          for (int i = 0; i < getModelCnt(); ++i)
            match(particleBins[rollid ^ 1][i])([&](auto &pb) {
              auto &next_pb = get<typename std::decay_t<decltype(pb)>>(
                  particleBins[rollid][i]);
              cuDev.compute_launch({pbcnt, 128}, update_buckets,
                                   (uint32_t)pbcnt, (const int *)sources, pb,
                                   next_pb);
            });
          // binsts
          int *binpbs = tmps.binpbs;
          for (int i = 0; i < getModelCnt(); ++i) {
            match(particleBins[rollid][i])([&](auto &pb) {
              cuDev.compute_launch({(pbcnt + 1 + 127) / 128, 128},
                                   compute_bin_capacity, pbcnt + 1,
                                   (const int *)pb._ppbs, binpbs);
              exclScan(pbcnt + 1, binpbs, pb._binsts, cuDev);
              checkCudaErrors(cudaMemcpyAsync(&bincnt[i], pb._binsts + pbcnt,
                                              sizeof(int), cudaMemcpyDefault,
                                              cuDev.stream_compute()));
              cuDev.syncStream<streamIdx::Compute>();
            });
          }
          timer.tock(fmt::format("GPU[{}] frame {} step {} update_partition",
                                 gpuid, curFrame, curStep));

          /// neighboring blocks
          timer.tick();
          cuDev.compute_launch({(pbcnt + 127) / 128, 128},
                               register_neighbor_blocks, (uint32_t)pbcnt,
                               partitions[rollid ^ 1]);
          auto prev_nbcnt = nbcnt;
          checkCudaErrors(cudaMemcpyAsync(&nbcnt, partitions[rollid ^ 1]._cnt,
                                          sizeof(int), cudaMemcpyDefault,
                                          cuDev.stream_compute()));
          cuDev.syncStream<streamIdx::Compute>();
          timer.tock(
              fmt::format("GPU[{}] frame {} step {} build_partition_for_grid",
                          gpuid, curFrame, curStep));

          /// check capacity
          if (checkedCnts[0] > 0) {
            gridBlocks[0].resize(device_allocator{}, curNumActiveBlocks);
          }
          /// rearrange grid blocks
          timer.tick();
          gridBlocks[0].reset(ebcnt, cuDev);
          cuDev.compute_launch(
              {prev_nbcnt, g_blockvolume}, copy_selected_grid_blocks,
              (const ivec3 *)partitions[rollid]._activeKeys,
              partitions[rollid ^ 1], (const int *)activeBlockMarks,
              gridBlocks[1], gridBlocks[0]);
          cuDev.syncStream<streamIdx::Compute>();
          timer.tock(fmt::format("GPU[{}] frame {} step {} copy_grid_blocks",
                                 gpuid, curFrame, curStep));
          /// check capacity
          if (checkedCnts[0] > 0) {
            gridBlocks[1].resize(device_allocator{}, curNumActiveBlocks);
            tmps.resize(curNumActiveBlocks);
          }
        }

        {
          auto &cuDev = Cuda::ref_cuda_context(gpuid);
          CudaTimer timer{cuDev.stream_compute()};

          timer.tick();
          /// exterior blocks
          cuDev.compute_launch({(pbcnt + 127) / 128, 128},
                               register_exterior_blocks, (uint32_t)pbcnt,
                               partitions[rollid ^ 1]);
          checkCudaErrors(cudaMemcpyAsync(&ebcnt, partitions[rollid ^ 1]._cnt,
                                          sizeof(int), cudaMemcpyDefault,
                                          cuDev.stream_compute()));
          cuDev.syncStream<streamIdx::Compute>();

          fmt::print(fmt::emphasis::bold | fg(fmt::color::yellow),
                     "block count on device {}: {}, {}, {} [{}]\n", gpuid,
                     pbcnt, nbcnt, ebcnt, curNumActiveBlocks);
          for (int i = 0; i < getModelCnt(); ++i) {
            fmt::print(fmt::emphasis::bold | fg(fmt::color::yellow),
                       "bin count on device {}: model {}: {} [{}]\n", gpuid, i,
                       bincnt[i], curNumActiveBins[i]);
          }
          timer.tock(fmt::format(
              "GPU[{}] frame {} step {} build_partition_for_particles", gpuid,
              curFrame, curStep));
        }
        rollid ^= 1;
        dt = nextDt;
      }
      // Restart frame's output scheme
      IO::flush();

      // Output material point position and attribute data (JB)
      output_model();
      
      // Output grid block data (JB)
      output_grid();

      // Step forward, terminal print
      nextTime = 1.f * (curFrame + 1) / fps;
      fmt::print(fmt::emphasis::bold | fg(fmt::color::red),
                 "-----------------------------------------------------------"
                 "-----\n");
    }
  }
  void output_model() {
    auto &cuDev = Cuda::ref_cuda_context(gpuid);
    CudaTimer timer{cuDev.stream_compute()};
    timer.tick();
    /// Iterate through particle models
    for (int i = 0; i < getModelCnt(); ++i) {
      /// Set-up particle counter on host and device
      int parcnt, *d_parcnt = (int *)cuDev.borrow(sizeof(int));
      checkCudaErrors(
          cudaMemsetAsync(d_parcnt, 0, sizeof(int), cuDev.stream_compute()));
      
      /// Copy (device --> device ) particle position and attributes (JB)
      /// Moves from time-step struct (pbuffer) to frame struct (particles, pattribs)
      match(particleBins[rollid][i])([&](const auto &pb) {
        cuDev.compute_launch({pbcnt, 128}, retrieve_particle_buffer_attributes,
                             partitions[rollid], partitions[rollid ^ 1], pb,
                             get<typename std::decay_t<decltype(pb)>>(
                                 particleBins[rollid ^ 1][i]),
                             particles[i], pattribs[i], d_parcnt);
      });
      checkCudaErrors(cudaMemcpyAsync(&parcnt, d_parcnt, sizeof(int),
                                      cudaMemcpyDefault,
                                      cuDev.stream_compute()));
      cuDev.syncStream<streamIdx::Compute>();

      /// Terminal output
      fmt::print(fg(fmt::color::red), "total number of particles {}\n", parcnt);
      
      /// Copy (device --> host) particle position (x,y,z) data, asynch.
      model.resize(parcnt);
      checkCudaErrors(
          cudaMemcpyAsync(model.data(), (void *)&particles[i].val_1d(_0, 0),
                          sizeof(std::array<float, 3>) * (parcnt),
                          cudaMemcpyDefault, cuDev.stream_compute()));
      cuDev.syncStream<streamIdx::Compute>();

      /// Copy (device --> host) particle attribute (.,.,.) data, asynch. (JB)
      attribs.resize(parcnt);
      checkCudaErrors(
          cudaMemcpyAsync(attribs.data(), (void *)&pattribs[i].val_1d(_0, 0),
                          sizeof(std::array<float, 3>) * (parcnt),
                          cudaMemcpyDefault, cuDev.stream_compute()));
      cuDev.syncStream<streamIdx::Compute>();
      
      /// Output filename
      std::string fn = std::string{"model"} + "_id[" + std::to_string(i) +
                       "]_frame[" + std::to_string(curFrame) + "].bgeo";
      
      /// Write to combined binary *.bgeo file, seperate by model and frame (JB)
      IO::insert_job([fn, m = model, a = attribs]() { write_partio_particles<float, 3>(fn, m, a); });
    }
    timer.tock(fmt::format("GPU[{}] frame {} step {} retrieve_particles", gpuid,
                           curFrame, curStep));
  }


  /// Output data from grid blocks (mass, momentum) to *.bgeo (JB)
  void output_grid() {
    auto &cuDev = Cuda::ref_cuda_context(gpuid);
    CudaTimer timer{cuDev.stream_compute()};
    timer.tick();

    int node_cnt, *d_node_cnt = (int *)cuDev.borrow(sizeof(int));
    
    /// Reserve memory
    checkCudaErrors(
        cudaMemsetAsync(d_node_cnt, 0, sizeof(int), cuDev.stream_compute()));
    
    /// Use previously set active block marks for efficiency
    int *activeBlockMarks = tmps.activeBlockMarks;

    /// Copy down-sampled grid value from buffer (device) to nodes (device)
    cuDev.compute_launch(
              {nbcnt, g_blockvolume}, retrieve_selected_grid_blocks,
              (const ivec3 *)partitions[rollid]._activeKeys,
              partitions[rollid ^ 1], (const int *)activeBlockMarks,
              gridBlocks[1], nodes[0]);
    cuDev.syncStream<streamIdx::Compute>();
    
    /// Should change to use a d_node_cnt synch for better active count
    /// For now set to neighboring block count
    node_cnt = nbcnt;

    fmt::print(fg(fmt::color::red), "total number of nodes {}\n", node_cnt);
    
    /// graph defined at bottom as std::vector<std::array<float,7>>
    graph.resize(node_cnt);

    // Asynchronously copy data from nodes (device) to graph (host)
    checkCudaErrors(
        cudaMemcpyAsync(graph.data(), (void *)&nodes[0].val_1d(_0, 0),
                        sizeof(std::array<float, 7>) * (node_cnt),
                        cudaMemcpyDefault, cuDev.stream_compute()));
    cuDev.syncStream<streamIdx::Compute>();

    /// Output to Partio as 'grid_frame[i].bgeo'
    std::string fn = std::string{"grid"} + "_frame[" + std::to_string(curFrame) + "].bgeo";
    IO::insert_job([fn, m = graph]() { write_partio_grid<float, 7>(fn, m); });

    timer.tock(fmt::format("GPU[{}] frame {} step {} retrieve_nodes", gpuid,
                           curFrame, curStep));
  }


  void initial_setup() {
    {
      auto &cuDev = Cuda::ref_cuda_context(gpuid);
      CudaTimer timer{cuDev.stream_compute()};

      timer.tick();
      for (int i = 0; i < getModelCnt(); ++i) {
        cuDev.compute_launch({(pcnt[i] + 255) / 256, 256}, activate_blocks,
                             pcnt[i], particles[i], partitions[rollid ^ 1]);
      }

      checkCudaErrors(cudaMemcpyAsync(&pbcnt, partitions[rollid ^ 1]._cnt,
                                      sizeof(int), cudaMemcpyDefault,
                                      cuDev.stream_compute()));
      timer.tock(fmt::format("GPU[{}] step {} init_table", gpuid, curStep));

      timer.tick();
      cuDev.resetMem();
      // particle block
      for (int i = 0; i < getModelCnt(); ++i) {
        match(particleBins[rollid][i])([&](auto &pb) {
          cuDev.compute_launch({(pcnt[i] + 255) / 256, 256},
                               build_particle_cell_buckets, pcnt[i],
                               particles[i], pb, partitions[rollid ^ 1]);
        });
      }
      cuDev.syncStream<streamIdx::Compute>();

      // bucket, binsts
      for (int i = 0; i < getModelCnt(); ++i)
        match(particleBins[rollid][i])([&](auto &pb) {
          checkCudaErrors(cudaMemsetAsync(
              pb._ppbs, 0, sizeof(int) * (pbcnt + 1), cuDev.stream_compute()));
          cuDev.compute_launch({pbcnt, config::g_blockvolume},
                               cell_bucket_to_block, pb._ppcs, pb._cellbuckets,
                               pb._ppbs, pb._blockbuckets);
          // partitions[rollid ^ 1].buildParticleBuckets(cuDev, pbcnt);
        });
      int *binpbs = tmps.binpbs;
      for (int i = 0; i < getModelCnt(); ++i) {
        match(particleBins[rollid][i])([&](auto &pb) {
          cuDev.compute_launch({(pbcnt + 1 + 127) / 128, 128},
                               compute_bin_capacity, pbcnt + 1,
                               (const int *)pb._ppbs, binpbs);
          exclScan(pbcnt + 1, binpbs, pb._binsts, cuDev);
          checkCudaErrors(cudaMemcpyAsync(&bincnt[i], pb._binsts + pbcnt,
                                          sizeof(int), cudaMemcpyDefault,
                                          cuDev.stream_compute()));
          cuDev.syncStream<streamIdx::Compute>();
          cuDev.compute_launch({pbcnt, 128}, array_to_buffer, particles[i], pb);
        });
      }
      // grid block
      cuDev.compute_launch({(pbcnt + 127) / 128, 128}, register_neighbor_blocks,
                           (uint32_t)pbcnt, partitions[rollid ^ 1]);
      checkCudaErrors(cudaMemcpyAsync(&nbcnt, partitions[rollid ^ 1]._cnt,
                                      sizeof(int), cudaMemcpyDefault,
                                      cuDev.stream_compute()));
      cuDev.syncStream<streamIdx::Compute>();
      cuDev.compute_launch({(pbcnt + 127) / 128, 128}, register_exterior_blocks,
                           (uint32_t)pbcnt, partitions[rollid ^ 1]);
      checkCudaErrors(cudaMemcpyAsync(&ebcnt, partitions[rollid ^ 1]._cnt,
                                      sizeof(int), cudaMemcpyDefault,
                                      cuDev.stream_compute()));
      cuDev.syncStream<streamIdx::Compute>();
      timer.tock(fmt::format("GPU[{}] step {} init_partition", gpuid, curStep));

      fmt::print(fmt::emphasis::bold | fg(fmt::color::yellow),
                 "block count on device {}: {}, {}, {} [{}]\n", gpuid, pbcnt,
                 nbcnt, ebcnt, curNumActiveBlocks);
      for (int i = 0; i < getModelCnt(); ++i) {
        fmt::print(fmt::emphasis::bold | fg(fmt::color::yellow),
                   "bin count on device {}: model {}: {} [{}]\n", gpuid, i,
                   bincnt[i], curNumActiveBins[i]);
      }
    }

    {
      auto &cuDev = Cuda::ref_cuda_context(gpuid);
      CudaTimer timer{cuDev.stream_compute()};

      partitions[rollid ^ 1].copy_to(partitions[rollid], ebcnt,
                                     cuDev.stream_compute());
      checkCudaErrors(cudaMemcpyAsync(
          partitions[rollid]._activeKeys, partitions[rollid ^ 1]._activeKeys,
          sizeof(ivec3) * ebcnt, cudaMemcpyDefault, cuDev.stream_compute()));

      for (int i = 0; i < getModelCnt(); ++i)
        match(particleBins[rollid][i])([&](const auto &pb) {
          // binsts, ppbs
          pb.copy_to(get<typename std::decay_t<decltype(pb)>>(
                         particleBins[rollid ^ 1][i]),
                     pbcnt, cuDev.stream_compute());
        });
      cuDev.syncStream<streamIdx::Compute>();

      timer.tick();
      gridBlocks[0].reset(nbcnt, cuDev);
      for (int i = 0; i < getModelCnt(); ++i) {
        cuDev.compute_launch({(pcnt[i] + 255) / 256, 256}, rasterize, pcnt[i],
                             particles[i], gridBlocks[0], partitions[rollid],
                             dt, getMass(i), vel0[i]);
        match(particleBins[rollid ^ 1][i])([&](auto &pb) {
          cuDev.compute_launch({pbcnt, 128}, init_adv_bucket,
                               (const int *)pb._ppbs, pb._blockbuckets);
        });
      }
      cuDev.syncStream<streamIdx::Compute>();
      timer.tock(fmt::format("GPU[{}] step {} init_grid", gpuid, curStep));
    }
  }

  /// Basic simulation settings
  int gpuid, nframes, fps;

  /// Basic physical settings
  float gravity, length, cfl;

  /// Timing variables
  float dt, nextDt, dtDefault, curTime, maxVel;
  uint64_t curFrame, curStep;

  /// Data on device, double-buffering scheme
  std::vector<GridBuffer> gridBlocks;
  std::vector<particle_buffer_t> particleBins[2];
  std::vector<Partition<1>> partitions; ///< with halo info  
  std::vector<GridArray>     nodes;     ///< Node array structure on device, 4+ f32 (ID+, mass, mx,my,mz) (JB)
  std::vector<ParticleArray> particles; ///< Particle array structure on device,  three f32 (x,y,z) (JB)
  std::vector<ParticleArray> pattribs;  ///< Particle atrrib structure on device, three f32 (.,.,.) (JB)  
  struct Intermediates {
    void *base;
    float *d_maxVel;
    int *d_tmp;
    int *activeBlockMarks;
    int *destinations;
    int *sources;
    int *binpbs;
    void alloc(int maxBlockCnt) {
      checkCudaErrors(cudaMalloc(&base, sizeof(int) * (maxBlockCnt * 5 + 1)));
      d_maxVel = (float *)((char *)base + sizeof(int) * maxBlockCnt * 5);
      d_tmp = (int *)((uintptr_t)base);
      activeBlockMarks = (int *)((char *)base + sizeof(int) * maxBlockCnt);
      destinations = (int *)((char *)base + sizeof(int) * maxBlockCnt * 2);
      sources = (int *)((char *)base + sizeof(int) * maxBlockCnt * 3);
      binpbs = (int *)((char *)base + sizeof(int) * maxBlockCnt * 4);
    }
    void dealloc() {
      cudaDeviceSynchronize();
      checkCudaErrors(cudaFree(base));
    }
    void resize(int maxBlockCnt) {
      dealloc();
      alloc(maxBlockCnt);
    }
  };
  Intermediates tmps;

  /// data on host
  static_assert(std::is_same<GridBufferDomain::index_type, int>::value,
                "block index type is not int");
  char rollid; ///< ID to switch between n and n+1
  std::size_t curNumActiveBlocks;            ///< num active grid-blocks
  std::vector<std::size_t> curNumActiveBins; ///< num active particle? bins
  std::array<std::size_t, 2> checkedCnts;
  std::vector<std::size_t> checkedBinCnts;
  float maxVels;
  int pbcnt, nbcnt, ebcnt;        ///< Number of particle, neighbor, and exterior blocks
  std::vector<int> bincnt;        ///< Number of particle bins
  std::vector<uint32_t> pcnt;     ///< Number of particles
  std::vector<uint32_t> node_cnt; ///< Number of grid nodes (JB)
  std::vector<std::array<float, 3>> model;   ///< Particle info (x,y,z) on host (JB)
  std::vector<std::array<float, 3>> attribs; ///< Particle attributes on host (JB)
  std::vector<std::array<float, 7>> graph;   ///< Grid info (x,y,z,m,mx,my,mz) on host (JB)
  std::vector<vec3> vel0; ///< Initial velocity vector on host
};

struct Simulator {};

struct SimulatorBuilder {
  Simulator &simulator;
};

// thread pool -> executor -> simulator (built from builder)

} // namespace mn

#endif