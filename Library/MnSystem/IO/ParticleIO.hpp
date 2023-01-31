#ifndef __PARTICLE_IO_HPP_
#define __PARTICLE_IO_HPP_
#include "PoissonDisk/SampleGenerator.h"
#include <MnBase/Math/Vec.h>
#include <Partio.h>
#include <array>
#include <string>
#include <vector>
#include <math.h>

namespace mn {


template <typename T, std::size_t dim>
void write_partio(std::string filename,
                  const std::vector<std::array<T, dim>> &data,
                  std::string tag = std::string{"position"}) {
  Partio::ParticlesDataMutable *parts = Partio::create();

  Partio::ParticleAttribute attrib =
      parts->addAttribute(tag.c_str(), Partio::VECTOR, dim);

  parts->addParticles(data.size());
  for (int idx = 0; idx < (int)data.size(); ++idx) {
    float *val = parts->dataWrite<float>(attrib, idx);
    for (int k = 0; k < dim; k++)
      val[k] = data[idx][k];
  }
  Partio::write(filename.c_str(), *parts);
  parts->release();
}

template <typename T, std::size_t dim>
void write_partio_general(std::string filename,
                  const std::vector<std::array<T, 3>>  &positions, 
                  const std::vector<std::array<T,dim>> &attributes,
                  const std::vector<std::string> &tags,
                  const std::vector<int> &sets) {
  Partio::ParticlesDataMutable *parts = Partio::create();
  
  Partio::FixedAttribute Cd = parts->addFixedAttribute("color", Partio::VECTOR, 3);
  float* c = parts->fixedDataWrite<float>(Cd);
  c[0] = 0; c[1] = 191; c[2] = 255;
  Partio::ParticleAttribute pos = parts->addAttribute("position", Partio::VECTOR, 3);
  for(int i=0; i < (int)positions.size(); ++i) {
    // Create new particle with two write-input vectors/arrays
    int idx  = parts->addParticle();
    float* p = parts->dataWrite<float>(pos,    idx);

    // Add position data for particle
    for(int k=0; k<3; ++k) p[k] = positions[i][k];

    // Add extra attributes for particle
    int shift = 0; // Shift position in attributes as we go through
    for (int j = 0; j< (int)sets.size(); ++j){
      Partio::ParticleAttribute attrib =
          parts->addAttribute(tags[j].c_str(), Partio::VECTOR, sets[j]);
      float* a = parts->dataWrite<float>(attrib, idx);

      for(int k=0; k < sets[j]; ++k) a[k] = attributes[i][(k+shift)];
      shift += sets[j];
    }
  }
  Partio::write(filename.c_str(), *parts);
  parts->release();
}

template <typename T, std::size_t dim>
void read_partio_general(std::string filename,
                  std::vector<std::array<T, 3>>  &positions, 
                  std::vector<std::array<T,dim>> &attributes,
                  const std::vector<std::string> &labels) {
  Partio::ParticlesData *parts = Partio::read(filename.c_str());
  if(!parts) printf("ERROR: Failed to open file with PartIO.");

  std::cout<<"PartIO reading number of particles: "<<parts->numParticles()<<std::endl;
  for(int i=0;i<parts->numAttributes();i++){
      Partio::ParticleAttribute attr;
      parts->attributeInfo(i,attr);
      std::cout<<"Attribute["<<i<<"] is "<<attr.name<<std::endl;
  }

  // Position processing
  Partio::ParticleAttribute posAttr;
  if(!parts->attributeInfo("position",posAttr) || posAttr.type != Partio::VECTOR || posAttr.count != 3)
      printf("ERROR: PartIO failed to get position as VECTOR of size 3");

  // Generic attribute processing
  Partio::ParticleAttribute genericAttr[dim];
  int d = 0;
  for (auto label : labels) {
    if(!parts->attributeInfo(label.c_str(), genericAttr[d]) || genericAttr[d].type !=  Partio::FLOAT || genericAttr[d].count != 1)
        printf("ERROR: PartIO failed to read value as FLOAT of size 1");
    d++;
  }

  // Read particle position
  std::array<T,3> pos = {0,0,0};
  for(int i=0; i < parts->numParticles(); i++) {
    auto val= parts->data<float>(posAttr, i);
    for(int k=0; k < 3; k++) 
        pos[k] = (T)val[k];
    positions.push_back(pos);
  }

  // Read particle generic attirbutes
  for(int i=0; i < parts->numParticles(); i++) {   
    std::array<T,dim> generics;
    int d = 0;
    for (auto label : labels) {
      if (d >= dim) break;
      auto val = parts->data<float>(genericAttr[d], i);
      generics[d] = (T)val[0];
      d++;
    }
    attributes.push_back(generics);
  }
  parts->release();
}


// Write combined particle position (x,y,z) and attribute (...) data (JB)
template <typename T, std::size_t dim>
void write_partio_particles(std::string filename,
                  const std::vector<std::array<T, 3>>  &positions, 
                  const std::vector<std::array<T,dim>> &attributes,
                  const std::vector<std::string> &labels) {
  // Create a mutable Partio structure pointer
  Partio::ParticlesDataMutable*       parts = Partio::create();

  // Add positions and attributes to the pointer by arrow operator
  Partio::FixedAttribute Cd      = parts->addFixedAttribute("color", Partio::VECTOR, 3);
  float* c = parts->fixedDataWrite<float>(Cd);
  c[0] = 0; c[1] = 191; c[2] = 255;
  Partio::ParticleAttribute pos     = parts->addAttribute("position", Partio::VECTOR, 3);
  Partio::ParticleAttribute attrib[dim];
  for (int d = 0; d < dim; d++){
    attrib[d] = parts->addAttribute(labels[d].c_str(), Partio::FLOAT, (int)1);
  }

  for(int i=0; i < (int)positions.size(); ++i) {
    // Create new particle with two write-input vectors/arrays
    int idx  = parts->addParticle();
    float* p = parts->dataWrite<float>(pos,    idx);
    // Add position data for particle
    for(int k=0; k<3; ++k) {
      p[k] = positions[i][k];
    }

    // Add extra attributes for particle
    for(int k=0; k<(int)dim; ++k) {
      float* a = parts->dataWrite<float>(attrib[k], idx);
      a[0] = attributes[i][k];
    }
  }

  // Write
  Partio::write(filename.c_str(), *parts);

  // Release (scope-dependent)
  parts->release();
}

// Write combined particle position (x,y,z) and attribute (...) data (JB)
template <typename T>
void write_partio_finite_elements(std::string filename,
                  const std::vector<std::array<T, 6>>  &attributes) {
  // Create a mutable Partio structure pointer
  Partio::ParticlesDataMutable*       parts = Partio::create();

  // Add positions and attributes to the pointer by arrow operator
  Partio::ParticleAttribute pos     = parts->addAttribute("position", Partio::VECTOR, 3);
  Partio::ParticleAttribute attrib  = parts->addAttribute("attributes", Partio::FLOAT, 3);

  for(int i=0; i < (int)attributes.size(); ++i)
  {
    // Create new particle with two write-input vectors/arrays
    int idx  = parts->addParticle();
    float* p = parts->dataWrite<float>(pos,    idx);
    float* a = parts->dataWrite<float>(attrib, idx);

    // Add position data for particle
    for(int k=0; k<3; ++k)
    {
      p[k] = attributes[i][k];
    }

    // Add extra attributes for particle
    for(int k=0; k<3; ++k)
    {
      int j  = k+3;
      a[k] = attributes[i][j];
    }
  }

  // Write
  Partio::write(filename.c_str(), *parts);

  // Release (scope-dependent)
  parts->release();
}
/// Write grid data (m, mvx, mvy, mvz) on host to disk as *.bgeo (JB) 
template <typename T, std::size_t dim>
void write_partio_grid(std::string filename,
		       const std::vector<std::array<T, dim>> &data) {
  /// Set mutable particle structure, add attributes
  Partio::ParticlesDataMutable* parts = Partio::create();
  Partio::ParticleAttribute position  = parts->addAttribute("position", Partio::VECTOR, 3); /// Block ID
  Partio::ParticleAttribute mass      = parts->addAttribute("mass",     Partio::FLOAT, 1);  /// Mass
  Partio::ParticleAttribute momentum  = parts->addAttribute("momentum", Partio::VECTOR, 3); /// Momentum
    
  /// Loop over grid-blocks, set values in Partio structure
  for(int i=0; i < (int)data.size(); ++i)
    {
      int idx   = parts->addParticle();
      float* p  = parts->dataWrite<float>(position,idx);
      float* m  = parts->dataWrite<float>(mass,idx);
      float* mv = parts->dataWrite<float>(momentum,idx);

      p[0]  = data[i][0];
      p[1]  = data[i][1];
      p[2]  = data[i][2];
      m[0]  = data[i][3];
      mv[0] = data[i][4];
      mv[1] = data[i][5];
      mv[2] = data[i][6];
    }
  /// Output as *.bgeo
  Partio::write(filename.c_str(), *parts);
  parts->release();
}


/// Write grid data (m, mvx, mvy, mvz) on host to disk as *.bgeo (JB) 
template <typename T, std::size_t dim>
void write_partio_gridTarget(std::string filename,
		       const std::vector<std::array<T, dim>> &data) {
  /// Set mutable particle structure, add attributes
  Partio::ParticlesDataMutable* parts = Partio::create();
  Partio::ParticleAttribute position  = parts->addAttribute("position", Partio::VECTOR, 3); /// Block ID
  Partio::ParticleAttribute mass      = parts->addAttribute("mass",     Partio::FLOAT, 1);  /// Mass
  Partio::ParticleAttribute momentum  = parts->addAttribute("velocity", Partio::VECTOR, 3); /// Momentum
  Partio::ParticleAttribute force     = parts->addAttribute("force",    Partio::VECTOR, 3); /// Force

  /// Loop over grid-blocks, set values in Partio structure
  for(int i=0; i < (int)data.size(); ++i)
    {
      int idx   = parts->addParticle();
      float* p  = parts->dataWrite<float>(position,idx);
      float* m  = parts->dataWrite<float>(mass,idx);
      float* mv = parts->dataWrite<float>(momentum,idx);
      float* f  = parts->dataWrite<float>(force,idx);

      p[0]  = data[i][0];
      p[1]  = data[i][1];
      p[2]  = data[i][2];
      m[0]  = data[i][3];
      mv[0] = data[i][4];
      mv[1] = data[i][5];
      mv[2] = data[i][6];
      f[0]  = data[i][7];
      f[1]  = data[i][8];
      f[2]  = data[i][9];
    }
  /// Output as *.bgeo
  Partio::write(filename.c_str(), *parts);
  parts->release();
}



/// Write grid data (m, mvx, mvy, mvz) on host to disk as *.bgeo (JB) 
template <typename T, std::size_t dim>
void write_partio_particleTarget(std::string filename,
		       const std::vector<std::array<T, dim>> &data) {
  /// Set mutable particle structure, add attributes
  Partio::ParticlesDataMutable* parts = Partio::create();
  Partio::ParticleAttribute position  = parts->addAttribute("position", Partio::VECTOR, 3); /// Block ID
  Partio::ParticleAttribute mass      = parts->addAttribute("aggregate",     Partio::FLOAT, 1);  /// Mass
  // Partio::ParticleAttribute momentum  = parts->addAttribute("momentum", Partio::VECTOR, 3); /// Momentum
  // Partio::ParticleAttribute force     = parts->addAttribute("force",    Partio::VECTOR, 3); /// Force

  /// Loop over grid-blocks, set values in Partio structure
  for(int i=0; i < (int)data.size(); ++i)
    {
      int idx   = parts->addParticle();
      float* p  = parts->dataWrite<float>(position,idx);
      float* m  = parts->dataWrite<float>(mass,idx);
      // float* mv = parts->dataWrite<float>(momentum,idx);
      // float* f  = parts->dataWrite<float>(force,idx);

      p[0]  = data[i][0];
      p[1]  = data[i][1];
      p[2]  = data[i][2];
      m[0]  = data[i][3];

    }
  /// Output as *.bgeo
  Partio::write(filename.c_str(), *parts);
  parts->release();
}

/// have issues
auto read_sdf(std::string fn, float ppc, float dx, vec<float, 3> offset,
              vec<float, 3> lengths) {
  std::vector<std::array<float, 3>> data;
  std::string fileName = std::string(AssetDirPath) + "MpmParticles/" + fn;

  float levelsetDx;
  SampleGenerator pd;
  std::vector<float> samples;
  vec<float, 3> mins, maxs, scales;
  vec<int, 3> maxns;
  pd.LoadSDF(fileName, levelsetDx, mins[0], mins[1], mins[2], maxns[0],
             maxns[1], maxns[2]);
  maxs = maxns.cast<float>() * levelsetDx;
  scales = lengths / (maxs - mins);
  float scale = scales[0] < scales[1] ? scales[0] : scales[1];
  scale = scales[2] < scale ? scales[2] : scale;

  float samplePerLevelsetCell = ppc * levelsetDx / dx * scale;

  if (0) pd.GenerateUniformSamples(samplePerLevelsetCell, samples);
  if (1) pd.GenerateCartesianSamples(samplePerLevelsetCell, samples);

  for (int i = 0, size = samples.size() / 3; i < size; i++) {
    vec<float, 3> p{samples[i * 3 + 0], samples[i * 3 + 1], samples[i * 3 + 2]};
    p = (p - mins) * scale + offset;
    // particle[0] = ((samples[i * 3 + 0]) + offset[0]);
    // particle[1] = ((samples[i * 3 + 1]) + offset[1]);
    // particle[2] = ((samples[i * 3 + 2]) + offset[2]);
    data.push_back(std::array<float, 3>{p[0], p[1], p[2]});
  }
  printf("[%f, %f, %f] - [%f, %f, %f], scale %f, parcnt %d, lsdx %f, dx %f\n",
         mins[0], mins[1], mins[2], maxs[0], maxs[1], maxs[2], scale,
         (int)data.size(), levelsetDx, dx);
  return data;
}


// Read SDF file, cartesian/uniform sample position data into an array (JB)
template <typename T>
auto read_sdf(std::string fn, std::vector<std::array<T,3>>& data, T ppc, T g_dx, int domainsize,
              vec<T, 3> offset, T length, 
              vec<T, 3> point_a, vec<T, 3> point_b, T scaling_factor=1.0, int pad=1) {
  //std::vector<std::array<T, 3>> data;
  std::string fileName = fn;

  // Create SampleGenerator class
  SampleGenerator pd;
  //int pad = 1;
  float fpad = (float)pad; //< Padding cells in a direction on SDF file 
  float levelsetDx; //< dx in SDF file
  float dx = length / domainsize; //< User-desired grid-dx for simulation
  std::vector<float> samples;
  vec<float, 3> mins, maxs;
  vec<float, 3> lengthRatios;
  vec<int, 3> maxns;

  // Load sdf into pd, update levelsetDx, mins, maxns
  pd.LoadSDF(fileName, levelsetDx, mins[0], mins[1], mins[2], maxns[0],
             maxns[1], maxns[2]);
  levelsetDx *= (float)scaling_factor;
  maxs = maxns.cast<float>() * levelsetDx;

  // Adjust *.sdf extents to simulation domainsize, select smallest ratio
  float ppl;
  ppl = 1.f / cbrtf(ppc);
  //ppl = ppl * dx; //
  //float samplePerLevelsetCell = 1.f / (ppl*ppl*ppl);
  float samplePerLevelsetCell = (levelsetDx / dx)*(levelsetDx / dx)*(levelsetDx / dx) * ppc;

  lengthRatios[0] = 1.0 / ((float)maxns[0] - fpad);
  lengthRatios[1] = 1.0 / ((float)maxns[1] - fpad);
  lengthRatios[2] = 1.0 / ((float)maxns[2] - fpad);
  float lengthRatio = lengthRatios[0] > lengthRatios[1] ? lengthRatios[0] : lengthRatios[1];
  lengthRatio = lengthRatios[2] > lengthRatio ? lengthRatios[2] : lengthRatio;


  // Output uniformly sampled sdf into samples
  if (0) pd.GenerateUniformSamples(samplePerLevelsetCell, samples);
  if (1) pd.GenerateCartesianSamples(samplePerLevelsetCell, samples);

  printf("SDF Samples: %f , lengthRatio %f %f %f %f Max-Cells: %f %f %f PPL: %f , samplePerLevelsetCell: %f \n", (float)(samples.size()/3.0), lengthRatio, lengthRatios[0],lengthRatios[1],lengthRatios[2], (float)maxns[0], (float)maxns[1], (float)maxns[2], ppl, samplePerLevelsetCell);

  // Loop through samples
  for (int i = 0, size = samples.size() / 3; i < size; i++) {
    // Group x,y,z position data
    vec<T, 3> p{samples[i * 3 + 0], samples[i * 3 + 1], samples[i * 3 + 2]};
    
    // Scale positions, add-in offset from JSON
    //for (int d = 0; d<3; d++) p[d] = ((p[d] - fpad) * levelsetDx - mins[d]) / length + offset[d]; 
    for (int d = 0; d<3; d++) p[d] = ((p[d] - fpad) * levelsetDx) / length + offset[d]; 

    if (p[0] >= point_a[0] && p[0] < point_b[0]) {
      if (p[1] >= point_a[1] && p[1] < point_b[1]) {
        if (p[2] >= point_a[2] && p[2] < point_b[2]) {
            //for (int d = 0; d<3; d++) p[d] = p[d] + offset[d];

            // Add (x,y,z) to data in order
            data.push_back(std::array<T, 3>{p[0], p[1], p[2]});
        }
      }
    }
  }
  printf("[%f, %f, %f] - [%f, %f, %f], scale %f, parcnt %d, lsdx %f, dx %f\n",
         (float)mins[0], (float)mins[1], (float)mins[2], (float)maxs[0], (float)maxs[1], (float)maxs[2], (float)scaling_factor,
         (int)data.size(), (float)levelsetDx, (float)dx);
}


// Read SDF file, cartesian/uniform sample position data into an array (JB)
template <typename T>
auto read_sdf(std::string fn, float ppc, float dx, int domainsize,
              vec<T, 3> offset, vec<T, 3> lengths, 
              vec<T, 3> point_a, vec<T, 3> point_b,
              vec<T, 3> inter_a, vec<T, 3> inter_b) {
  std::vector<std::array<T, 3>> data;
  std::string fileName = std::string(AssetDirPath) + "MpmParticles/" + fn;

  // Create SampleGenerator class
  SampleGenerator pd;
  int pad = 1;
  float fpad = (float)pad;
  float levelsetDx;
  std::vector<float> samples;
  vec<float, 3> mins, maxs, scales;
  vec<float, 3> lengthRatios;
  vec<int, 3> maxns;

  // Load sdf into pd, update levelsetDx, mins, maxns
  pd.LoadSDF(fileName, levelsetDx, mins[0], mins[1], mins[2], maxns[0],
             maxns[1], maxns[2]);
  maxs = maxns.cast<float>() * levelsetDx;

  // Adjust *.sdf extents to simulation domainsize, select smallest ratio
  float ppl;
  ppl = rcbrtf(ppc);

  lengthRatios[0] = lengths[0] / (maxs[0] - fpad*levelsetDx);
  lengthRatios[1] = lengths[1] / (maxs[1] - fpad*levelsetDx);
  lengthRatios[2] = lengths[2] / (maxs[2] - fpad*levelsetDx);
  lengthRatios[0] = lengths[0] / ((float)maxns[0] - fpad);
  lengthRatios[1] = lengths[1] / ((float)maxns[1] - fpad);
  lengthRatios[2] = lengths[2] / ((float)maxns[2] - fpad);
  float lengthRatio = lengthRatios[0] > lengthRatios[1] ? lengthRatios[0] : lengthRatios[1];
  lengthRatio = lengthRatios[2] > lengthRatio ? lengthRatios[2] : lengthRatio;

  scales[0] = powf((dx / lengthRatios[0]), 3);
  scales[1] = powf((dx / lengthRatios[1]), 3);
  scales[2] = powf((dx / lengthRatios[2]), 3);
  float scale = scales[0] > scales[1] ? scales[0] : scales[1];
  scale = scales[2] > scale ? scales[2] : scale;
  printf("scale %f lengthRatio %f %f %f %f %f %f %f %f\n", scale, lengthRatio, lengths[0],lengths[1],lengths[2],(float)maxns[0],(float)maxns[1],(float)maxns[2], (float)domainsize);

  ppl = ppl * g_dx / lengthRatio;

  float samplePerLevelsetCell;
  if (1) samplePerLevelsetCell = 1.0 / (ppl*ppl*ppl);
  if (0) samplePerLevelsetCell = 1.0 * (int)(ppc * scale + 0.5);


  // Output uniformly sampled sdf into samples
  if (0) pd.GenerateUniformSamples(samplePerLevelsetCell, samples);
  if (1) pd.GenerateCartesianSamples(samplePerLevelsetCell, samples);

  // Adjust lengths to extents of the *.sdf, select smallest ratio
  scales[0] = lengths[0] / ((float)maxns[0] - fpad);
  scales[1] = lengths[1] / ((float)maxns[1] - fpad);
  scales[2] = lengths[2] / ((float)maxns[2] - fpad);
  scale = scales[0] > scales[1] ? scales[0] : scales[1];
  scale = scales[2] > scale ? scales[2] : scale;

  // Loop through samples
  for (int i = 0, size = samples.size() / 3; i < size; i++) {
    // Group x,y,z position data
    vec<T, 3> p{(T)samples[i * 3 + 0], (T)samples[i * 3 + 1], (T)samples[i * 3 + 2]};
    
    // Scale positions, add-in offset from JSON
    //p = (p - mins) * scale + offset;
    p = (p - fpad) * scale; //+ offset;


    if (p[0] >= point_a[0] && p[0] < point_b[0]) {
      if (p[1] >= point_a[1] && p[1] < point_b[1]) {
        if (p[2] >= point_a[2] && p[2] < point_b[2]) {

          if (p[0] > inter_b[0] || p[0] < inter_a[0] || p[1] > inter_b[1] || p[1] < inter_a[1] || p[2] > inter_b[2] || p[2] < inter_a[2]) {
            p = p + offset;
            // Add (x,y,z) to data in order
            data.push_back(std::array<T, 3>{p[0], p[1], p[2]});
          }
        }
      }
    }
  }
  printf("[%f, %f, %f] - [%f, %f, %f], scale %f, parcnt %d, lsdx %f, dx %f\n",
         mins[0], mins[1], mins[2], maxs[0], maxs[1], maxs[2], scale,
         (int)data.size(), levelsetDx, dx);
  return data;
}

} // namespace mn

#endif
