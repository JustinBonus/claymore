#ifndef __PARTICLE_IO_HPP_
#define __PARTICLE_IO_HPP_
#include "PoissonDisk/SampleGenerator.h"
#include <MnBase/Math/Vec.cuh>
#include <Partio.h>
#include <array>
#include <string>
#include <vector>
#include <math.h>
#include <memory.h>

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
                  const std::vector<int> &sets, const std::array<int,3> color = {{0, 191, 255}}) {
  Partio::ParticlesDataMutable *parts = Partio::create();
  
  Partio::FixedAttribute Cd = parts->addFixedAttribute("Cd", Partio::VECTOR, 3);
  float* c = parts->fixedDataWrite<float>(Cd);
  c[0] = color[0]; c[1] = color[1]; c[2] = color[2];
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

template <typename T>
void read_partio_general(std::string filename,
                  std::vector<std::array<T, 3>>  &positions, 
                  std::vector<std::vector<T>> &attributes,
                  const int dim_in,
                  const std::vector<std::string> &labels, bool verbose = false) {
  Partio::ParticlesData *parts = Partio::read(filename.c_str());
  if(!parts) std::cout << "ERROR: Failed to open file with PartIO." << std::endl;

  if (verbose) std::cout<<"PartIO reading number of particles: "<<parts->numParticles()<<std::endl;
  for(int i=0;i<parts->numAttributes();i++){
      Partio::ParticleAttribute attr;
      parts->attributeInfo(i,attr);
      if (verbose) std::cout<<"Attribute["<<i<<"] is ["<<attr.name<<"]"<<std::endl;
  }

  // Position processing
  Partio::ParticleAttribute posAttr;
  if(!parts->attributeInfo("position",posAttr) || posAttr.type != Partio::VECTOR || posAttr.count != 3)
      printf("ERROR: PartIO failed to get position as VECTOR of size 3");

  // Generic attribute processing
  // Partio::ParticleAttribute genericAttr[dim_in];
  auto genericAttr = std::make_unique<Partio::ParticleAttribute[]>(dim_in); // fix for MSVC which doesn't like above
  int d = 0;
  for (auto label : labels) {
    if(!parts->attributeInfo(label.c_str(), genericAttr[d]) || genericAttr[d].type !=  Partio::FLOAT || genericAttr[d].count != 1)
        printf("ERROR: PartIO failed to read value as FLOAT of size 1");

    if (verbose) std::cout << "PartIO: Input label[" << label << "] found in index[" << d << "] of file." << std::endl;
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
    std::vector<T> generics;
    generics.resize(dim_in);
    int d = 0;
    for (auto label : labels) {
      if (d >= dim_in) break;
      auto val = parts->data<float>(genericAttr[d], i);
      generics[d] = (T)val[0];
      d++;
    }
    attributes.push_back(generics);
  }
  parts->release();
}


// Write combined particle position (x,y,z) and attribute (...) data (JB)
template <typename T>
void write_partio_particles(std::string filename,
                  const std::vector<std::array<T, 3>>  &positions, 
                  const std::vector<std::vector<T>> &attributes,
                  //const int dim_out,
                  const std::vector<std::string> &labels, 
                  const std::array<int,3> color = {{0, 191, 255}}) {
  // Create a mutable Partio structure pointer
  Partio::ParticlesDataMutable*       parts = Partio::create();
  const int dim_out = attributes[0].size();
  // Add positions and attributes to the pointer by arrow operator
  Partio::FixedAttribute Cd      = parts->addFixedAttribute("Cd", Partio::VECTOR, 3);
  float* c = parts->fixedDataWrite<float>(Cd);
  c[0] = color[0]; c[1] = color[1]; c[2] = color[2];
  Partio::ParticleAttribute pos     = parts->addAttribute("position", Partio::VECTOR, 3);
  Partio::ParticleAttribute attrib[dim_out];
  for (int d = 0; d < dim_out; d++){
    attrib[d] = parts->addAttribute(labels[d].c_str(), Partio::FLOAT, (int)1);
  }

  // Loop through particles to add positions and attributes values
  // TODO: I think this can be done as SIMD more efficiently
  for(int i=0; i < (int)positions.size(); ++i) {
    // Create new particle with two write-input vectors/arrays
    int idx  = parts->addParticle();
    float* p = parts->dataWrite<float>(pos,    idx);
    // Add position data for particle
    for(int k=0; k<3; ++k) {
      p[k] = positions[i][k];
    }

    // Add extra attributes for particle
    for(int k=0; k<(int)dim_out; ++k) {
      float* a = parts->dataWrite<float>(attrib[k], idx);
      a[0] = attributes[i][k];
    }

  }
  Partio::write(filename.c_str(), *parts);
  parts->release();
}


// Write combined particle position (x,y,z) and attribute (...) data (JB)
template <typename T>
void write_partio_particles(std::string filename,
                  const std::vector<std::array<T, 3>>  &positions, 
                  const std::vector<T> &attributes,
                  //const int dim_out,
                  const std::vector<std::string> &labels, 
                  const std::array<int,3> color = {{0, 191, 255}}) {
  const int dim_out = attributes.size() / positions.size(); //< Output attributes per particle
  Partio::ParticlesDataMutable* parts = Partio::create();
  Partio::FixedAttribute Cd      = parts->addFixedAttribute("Cd", Partio::VECTOR, 3);
  float* c = parts->fixedDataWrite<float>(Cd);
  c[0] = color[0]; c[1] = color[1]; c[2] = color[2];
  // Add positions and attributes to the pointer by arrow operator
  Partio::ParticleAttribute pos     = parts->addAttribute("position", Partio::VECTOR, 3);
  // Partio::ParticleAttribute attrib[dim_out];
  auto attrib = std::make_unique<Partio::ParticleAttribute[]>(dim_out); // fix for MSVC which doesn't like above
  for (int d = 0; d < dim_out; d++)
    attrib[d] = parts->addAttribute(labels[d].c_str(), Partio::FLOAT, (int)1);
  
  for(int i=0; i < (int)positions.size(); ++i) {
    // Create new particle with two write-input vectors/arrays
    int idx  = parts->addParticle();
    float* p = parts->dataWrite<float>(pos,    idx);
    // Add position data for particle
    for(int k=0; k<3; ++k) 
      p[k] = positions[i][k];
    
    // Add extra attributes for particle
    for(int k=0; k<(int)dim_out; ++k) {
      float* a = parts->dataWrite<float>(attrib[k], idx);
      a[0] = attributes[i*dim_out + k];
    }

  }
  Partio::write(filename.c_str(), *parts);
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
  Partio::write(filename.c_str(), *parts);
  parts->release();
}



/// Write grid data (m, mvx, mvy, mvz) on host to disk as *.bgeo (JB) 
template <typename T, std::size_t dim>
void write_partio_particleTarget(std::string filename,
		       const std::vector<std::array<T, dim>> &data, std::size_t num_aggregate_values = 1) {
  /// Set mutable particle structure, add attributes
  Partio::ParticlesDataMutable* parts = Partio::create();
  Partio::ParticleAttribute position  = parts->addAttribute("position", Partio::VECTOR, 3); /// Block ID
  Partio::ParticleAttribute agg      = parts->addAttribute("aggregate", Partio::FLOAT, num_aggregate_values);  /// Aggregate value of particle target (e.g. max elevation)
  /// Loop over particles, set values in Partio structure
  for(int i=0; i < (int)data.size(); ++i) {
      int idx   = parts->addParticle();
      float* p  = parts->dataWrite<float>(position,idx);
      float* a  = parts->dataWrite<float>(agg,idx);

      for (int d=0; d<3; d++) {
        p[d] = data[i][d];
      }
      for (int j=0; j<num_aggregate_values; j++) {
          a[j] = data[i][3+j];
      }

    }
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



template <typename T>
void translate_rotate_translate_point_IO(const mn::vec<T,3> &fulcrum, const mn::vec<T,3,3> &rotate, mn::vec<T,3>& point) {
  mn::vec<T,3> tmp_point;
  for (int d=0;d<3;d++) tmp_point[d] = point[d] - fulcrum[d]; // Translate to rotation fulcrum
  mn::matrixVectorMultiplication3d(rotate.data(), tmp_point.data(), point.data()); // Rotate
  for (int d=0;d<3;d++) point[d] += fulcrum[d]; // Translate back.
}

template <typename T>
void translate_rotate_translate_point_IO(const mn::vec<T,3> &fulcrum, const mn::vec<T,3,3> &rotate, std::array<T,3>& point) {
  std::array<T,3> tmp_point;
  for (int d=0;d<3;d++) tmp_point[d] = point[d] - fulcrum[d]; // Translate to rotation fulcrum
  mn::matrixVectorMultiplication3d(rotate.data(), tmp_point.data(), point.data()); // Rotate
  for (int d=0;d<3;d++) point[d] += fulcrum[d]; // Translate back.
}

// Read SDF file, cartesian/uniform sample position data into an array (JB)
template <typename T>
auto read_sdf(std::string fn, std::vector<std::array<T,3>>& data, T ppc, T g_dx, int domainsize,
              vec<T, 3> offset, T length, 
              vec<T, 3> point_a, vec<T, 3> point_b, vec<T, 3, 3>& rotation, vec<T, 3>& fulcrum,
              T scaling_factor=1.0, int pad=1) {
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

  // Check if non-identity rotation matrix, if so, must rotate points
  bool must_rotate = false;
  if (rotation(0,0) != 1.0 || rotation(1,1) != 1.0 || rotation(2,2) != 1.0 || rotation(0,1) != 0 || rotation(0,2) != 0 || rotation(1,0) != 0 || rotation(1,2) != 0 || rotation(2,0) != 0 || rotation(2,1) != 0) {
      printf("Must rotate points\n");
      must_rotate = true;
  }
  // Loop through samples
  for (int i = 0, size = samples.size() / 3; i < size; i++) {
    // Group x,y,z position data
    vec<T, 3> p{samples[i * 3 + 0], samples[i * 3 + 1], samples[i * 3 + 2]};
    
    // Scale positions, add-in offset from JSON
    //for (int d = 0; d<3; d++) p[d] = ((p[d] - fpad) * levelsetDx - mins[d]) / length + offset[d]; 
    for (int d = 0; d<3; d++) p[d] = ((p[d] - fpad) * levelsetDx) / length + offset[d]; 
    if (must_rotate) translate_rotate_translate_point_IO<T>(fulcrum, rotation, p);

    // Check if point is within domain
    if (p[0] >= point_a[0] && p[0] < point_b[0]) {
      if (p[1] >= point_a[1] && p[1] < point_b[1]) {
        if (p[2] >= point_a[2] && p[2] < point_b[2]) {
            //for (int d = 0; d<3; d++) p[d] = p[d] + offset[d];

            // Add (x,y,z) to data vector in order
            data.push_back(std::array<T, 3>{p[0], p[1], p[2]});
        }
      }
    }
  }
  printf("SDF Info: mins[%f, %f, %f] - maxs[%f, %f, %f], scale %f, parcnt %d, lsdx %f, dx %f\n",
         (float)mins[0], (float)mins[1], (float)mins[2], (float)maxs[0], (float)maxs[1], (float)maxs[2], (float)scaling_factor,
         (int)data.size(), (float)levelsetDx, (float)dx);
}


} // namespace mn

#endif
