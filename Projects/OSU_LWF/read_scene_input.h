#ifndef __READ_SCENE_INPUT_H_
#define __READ_SCENE_INPUT_H_
#include "mgsp_benchmark.cuh"
#include "partition_domain.h"
#include <MnBase/Math/Vec.h>
#include <MnBase/Geometry/GeometrySampler.h>
#include <MnSystem/IO/IO.h>
#include <MnSystem/IO/ParticleIO.hpp>

#include <cxxopts.hpp>
#include <fmt/color.h>
#include <fmt/core.h>

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <array>
#include <cassert>

#if CLUSTER_COMM_STYLE == 1
#include <mpi.h>
#endif


#if 0
#include <ghc/filesystem.hpp>
namespace fs = ghc::filesystem;
#else
#include <experimental/filesystem>
namespace fs = std::experimental::filesystem;
#endif

// #if 1
// #include <ghc/filesystem.hpp>
// namespace fs = ghc::filesystem;
// #else
// #include <experimental/filesystem>
// namespace fs = std::experimental::filesystem;
// #endif

// Above had issues, probably C++ version related. Maybe windows/linux 
// Took this from build/_deps/filesystem-src/include/ghc/filesystem.hpp verbatim:
// #if defined(__cplusplus) && __cplusplus >= 201703L && defined(__has_include) && __has_include(<filesystem>)
// #include <filesystem>
// namespace fs = std::filesystem;
// #else
// #include <ghc/filesystem.hpp>
// namespace fs = ghc::filesystem;
// #endif

#include <rapidjson/document.h>
#include <rapidjson/stringbuffer.h>
#include <rapidjson/writer.h>
namespace rj = rapidjson;
static const char *kTypeNames[] = {"Null",  "False",  "True",  "Object",
                                   "Array", "String", "Number"};
static const auto red = fmt::color::red;
static const auto blue = fmt::color::blue;
static const auto green = fmt::color::green;
static const auto yellow = fmt::color::yellow;
static const auto orange = fmt::color::orange;
static const auto cyan = fmt::color::cyan;
static const auto white = fmt::color::white;

typedef std::vector<std::array<PREC, 3>> PositionHolder;
typedef std::vector<std::array<PREC, 13>> VerticeHolder;
typedef std::vector<std::array<int, 4>> ElementHolder;
typedef std::vector<std::array<PREC, 6>> ElementAttribsHolder;
typedef std::vector<std::array<PREC_G, 3>> MotionHolder;

PREC o = mn::config::g_offset; //< Grid-cell buffer size (see off-by-2, Xinlei Wang)
PREC l = mn::config::g_length; //< Domain max length default
PREC dx = mn::config::g_dx; //< Grid-cell length [1x1x1] default
std::string save_suffix; //< File-format to save particles with
float verbose = 0;


// Outputs a box as an OBJ file. Used to output user box boundaries for rendering.
template <typename T>
void writeBoxOBJ(const std::string &filename, const mn::vec<T, 3> &origin,
                 const mn::vec<T, 3> &lengths) {
  std::ofstream file(filename);
  if (!file.is_open()) {
    std::cerr << "ERROR: Could not open file " << filename << " for writing."
              << std::endl;
    return;
  }
  // Box's vertex ID diagram:
  // 3-------------7
  // |\            |\ 
  // | \           | \ 
  // |  \          |  \ 
  // |   4---------|---8
  // 1---|---------5   |
  //  \  |          \  |
  //   \ |           \ |
  //    \|            \|
  //     2-------------6 
  // Y
  // |
  // o---X
  //  \  
  //   Z

  // Write vertices
  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 2; ++j) {
      for (int k = 0; k < 2; ++k) {
        file << "v " << (origin[0]-o)*l + i * (lengths[0])*l << " "
             << (origin[1]-o)*l + j * (lengths[1])*l << " " << (origin[2]-o)*l + k * (lengths[2])*l
             << std::endl;
      }
    }
  }
  // Write box faces as triangles. Two triangles per face, six total faces for box. No overlap of triangles on same face.
  // Face 1
  file << "f 1 2 4" << std::endl;
  file << "f 1 4 3" << std::endl;
  // Face 2
  file << "f 5 6 8" << std::endl;
  file << "f 5 8 7" << std::endl;
  // Face 3
  file << "f 1 2 6" << std::endl;
  file << "f 1 6 5" << std::endl;
  // Face 4
  file << "f 3 4 8" << std::endl;
  file << "f 3 8 7" << std::endl;
  // Face 5
  file << "f 1 3 7" << std::endl;
  file << "f 1 7 5" << std::endl;
  // Face 6
  file << "f 2 4 8" << std::endl;
  file << "f 2 8 6" << std::endl;
  file.close();
}

// write OBJ output for cylinder. User specifies origin, radius, and length.
// User also specifies number of radial segments and number of length segments.
// Radial segments are the number of triangles used to approximate the cylinder's
// surface. Length segments are the number of triangles used to approximate the
// cylinder's length.
template <typename T>
void writeCylinderOBJ(const std::string &filename, const mn::vec<T, 3> &origin,
                      T radius, T length, int radial_segments,
                      int length_segments) {
  std::ofstream file(filename);
  if (!file.is_open()) {
    std::cerr << "ERROR: Could not open file " << filename << " for writing."
              << std::endl;
    return;
  }
  // Write vertices
  for (int i = 0; i <= length_segments; ++i) {
    for (int j = 0; j <= radial_segments; ++j) {
      file << "v " << origin[0] + radius * cos(j * 2 * M_PI / radial_segments)
           << " " << origin[1] + radius * sin(j * 2 * M_PI / radial_segments)
           << " " << origin[2] + i * length / length_segments << std::endl;
    }
  }
  // Write cylinder faces as triangles. Two triangles per face. No overlap of triangles on same face.
  for (int i = 0; i < length_segments; ++i) {
    for (int j = 0; j < radial_segments; ++j) {
      // First triangle
      file << "f " << i * (radial_segments + 1) + j + 1 << " "
           << i * (radial_segments + 1) + j + 2 << " "
           << (i + 1) * (radial_segments + 1) + j + 2 << std::endl;
      // Second triangle
      file << "f " << i * (radial_segments + 1) + j + 1 << " "
           << (i + 1) * (radial_segments + 1) + j + 2 << " "
           << (i + 1) * (radial_segments + 1) + j + 1 << std::endl;
    }
  }

  file.close();
}

/// @brief Make 3D rotation matrix that can rotate a point around origin
/// @brief Order: [Z Rot.]->[Y Rot.]->[X Rot.]=[ZYX Rot.]. (X Rot. 1st).
/// @brief Rot. on fixed axis (X,Y,Z) (i.e. NOT Euler or Quat. rotation)
/// @param angles Rotation angles [a,b,c] (degrees) for [X,Y,Z] axis.
/// @param R Original rotation matrix, use 3x3 Identity mat. if none.
/// @returns Multiply new rotation matrix into R.
template <typename T>
void elementaryToRotationMatrix(const mn::vec<T,3> &angles, mn::vec<T,3,3> &R) {
    // TODO: Can be way more efficient, made this quickly
    mn::vec<T,3,3> prev_R;
    for (int i=0; i<3; i++)
      for (int j=0; j<3; j++)
        prev_R(i,j) = R(i,j);
    mn::vec<T,3,3> tmp_R; tmp_R.set(0.0);
    // X-Axis Rotation
    tmp_R(0, 0) = 1;
    tmp_R(1, 1) = tmp_R(2, 2) = cos(angles[0] * PI_AS_A_DOUBLE / 180.);
    tmp_R(2, 1) = tmp_R(1, 2) = sin(angles[0] * PI_AS_A_DOUBLE / 180.);
    tmp_R(1, 2) = -tmp_R(1, 2);
    // mn::matrixMatrixMultiplication3d(tmp_R.data(), prev_R.data(), R.data());
    for (int i=0; i<3; i++) for (int j=0; j<3; j++) prev_R(i,j) = tmp_R(i,j); 
    tmp_R.set(0.0);
    // Z-Axis * Y-Axis * X-Axis Rotation
    tmp_R(1, 1) = 1;
    tmp_R(0, 0) = tmp_R(2, 2) = cos(angles[1] * PI_AS_A_DOUBLE / 180.);
    tmp_R(2, 0) = tmp_R(0, 2) = sin(angles[1] * PI_AS_A_DOUBLE / 180.);
    tmp_R(2, 0) = -tmp_R(2, 0);
    mn::matrixMatrixMultiplication3d(tmp_R.data(), prev_R.data(), R.data());
    for (int i=0; i<3; i++) for (int j=0; j<3; j++) prev_R(i,j) = R(i,j); 
    tmp_R.set(0.0);
    // Z-Axis * Y-Axis * X-Axis Rotation
    tmp_R(2, 2) = 1;
    tmp_R(0, 0) = tmp_R(1, 1) = cos(angles[2] * PI_AS_A_DOUBLE / 180.);
    tmp_R(1, 0) = tmp_R(0, 1) = sin(angles[2] * PI_AS_A_DOUBLE / 180.);
    tmp_R(0, 1) = -tmp_R(0, 1);
    mn::matrixMatrixMultiplication3d(tmp_R.data(), prev_R.data(), R.data());
  }

//https://danceswithcode.net/engineeringnotes/rotations_in_3d/rotations_in_3d_part1.html
// Convert elementary angles of rotation around fixed axis to euler angles of rotation around axis that move with previous rotation,
// input elementary angles as an array of three doubles in degrees, output euler angles as an array of three doubles in degrees
// elementary angles are in the order of a, b, c
// euler angles are in the order of z, y, x
void elementaryToEulerAngles(double *elementary, double *euler) {
  double a = elementary[0] * PI_AS_A_DOUBLE / 180.;
  double b = elementary[1] * PI_AS_A_DOUBLE / 180.;
  double c = elementary[2] * PI_AS_A_DOUBLE / 180.;
  double z = atan2(sin(c) * cos(b) * cos(a) - sin(a) * sin(b), cos(c) * cos(b)) * 180. / PI_AS_A_DOUBLE;
  double y = atan2(sin(c) * sin(a) + cos(c) * cos(a) * sin(b), cos(a) * cos(b)) * 180. / PI_AS_A_DOUBLE;
  double x = atan2(sin(c) * cos(a) + cos(c) * sin(a) * sin(b), cos(c) * cos(b)) * 180. / PI_AS_A_DOUBLE;
  euler[0] = z;
  euler[1] = y;
  euler[2] = x;
}

// Cnvert euler angles of rotation around axis that move with previous rotation to elementary angles of rotation around fixed axis,
// input euler angles as an array of three doubles in degrees, output elementary angles as an array of three doubles in degrees
// euler angles are in the order of z, y, x
// elementary angles are in the order of a, b, c
void eulerAnglesToElementary(double *euler, double *elementary) {
  double z = euler[0] * PI_AS_A_DOUBLE / 180.;
  double y = euler[1] * PI_AS_A_DOUBLE / 180.;
  double x = euler[2] * PI_AS_A_DOUBLE / 180.;
  double a = atan2(sin(z) * cos(y) * cos(x) + sin(x) * sin(y), cos(z) * cos(y)) * 180. / PI_AS_A_DOUBLE;
  double b = atan2(sin(z) * sin(x) - cos(z) * cos(x) * sin(y), cos(x) * cos(y)) * 180. / PI_AS_A_DOUBLE;
  double c = atan2(sin(z) * cos(x) - cos(z) * sin(x) * sin(y), cos(z) * cos(y)) * 180. / PI_AS_A_DOUBLE;
  elementary[0] = a;
  elementary[1] = b;
  elementary[2] = c;
}

// Convert euler angles of rotation around axis that move with previous rotation to rotation matrix,
// input euler angles as an array of three doubles in degrees, output rotation matrix as an array of nine doubles
// euler angles are in the order of z, y, x
void eulerAnglesToRotationMatrix(mn::vec<PREC,3> &euler, mn::vec<PREC,3,3> &matrix) {
  // convert euler angles to elementary angles
  mn::vec<PREC,3> elementary;
  eulerAnglesToElementary(euler.data(), elementary.data());

  // convert elementary angles to rotation matrix
  elementaryToRotationMatrix(elementary, matrix);
}

// Rotate a point around a fulcrum point using euler angles
// The point is an array of three doubles, the fulcrum is an array of three doubles, the euler angles are an array of three doubles in degrees
// Order of rotation is z, y, x
void translate_rotate_euler_translate_point(double *point, double *fulcrum, double *euler) {
  // Translate to rotation fulcrum
  double tmp_point[3];
  for (int d=0;d<3;d++) tmp_point[d] = point[d] - fulcrum[d];

  // Rotate with euler angles, convert to radians
  double x = euler[0] * PI_AS_A_DOUBLE / 180.;
  double y = euler[1] * PI_AS_A_DOUBLE / 180.;
  double z = euler[2] * PI_AS_A_DOUBLE / 180.;
  double tmp_x = ((tmp_point[0]) * cos(y) + (tmp_point[1] * sin(x) + tmp_point[2] * cos(x)) * sin(y)) * cos(z) - (tmp_point[1] * cos(x) - tmp_point[0] * sin(x)) * sin(z) ;
  double tmp_y =  ((tmp_point[0]) * cos(y) + (tmp_point[1] * sin(x) + tmp_point[2] * cos(x)) * sin(y)) * sin(z) +  (tmp_point[1] * cos(x) - tmp_point[0] * sin(x)) * cos(z);
  double tmp_z = ( (-tmp_point[0]) * sin(y) + (tmp_point[1] * sin(x) + tmp_point[2] * cos(x)) * cos(y) ) ;
  tmp_point[0] = tmp_x;
  tmp_point[1] = tmp_y;
  tmp_point[2] = tmp_z;

  // Translate back
  for (int d=0;d<3;d++) point[d] = tmp_point[d] + fulcrum[d];
}


template <typename T>
void translate_rotate_translate_point(const mn::vec<T,3> &fulcrum, const mn::vec<T,3,3> &rotate, mn::vec<T,3>& point) {
  mn::vec<T,3> tmp_point;
  for (int d=0;d<3;d++) tmp_point[d] = point[d] - fulcrum[d]; // Translate to rotation fulcrum
  mn::matrixVectorMultiplication3d(rotate.data(), tmp_point.data(), point.data()); // Rotate
  for (int d=0;d<3;d++) point[d] += fulcrum[d]; // Translate back.
}

template <typename T>
void translate_rotate_translate_point(const mn::vec<T,3> &fulcrum, const mn::vec<T,3,3> &rotate, std::array<T,3>& point) {
  std::array<T,3> tmp_point;
  for (int d=0;d<3;d++) tmp_point[d] = point[d] - fulcrum[d]; // Translate to rotation fulcrum
  mn::matrixVectorMultiplication3d(rotate.data(), tmp_point.data(), point.data()); // Rotate
  for (int d=0;d<3;d++) point[d] += fulcrum[d]; // Translate back.
}
/// @brief Check if a particle is inside a box partition.
/// @tparam Data-type, e.g. float or double. 
/// @param arr Array of [x,y,z] position to check for being inside partition.
/// @param partition_start Starting corner of box partition.
/// @param partition_end Far corner of box partition.
/// @return Returns true if an inputted position is inside the specified partition. False otherwise.
template <typename T>
bool inside_partition(std::array<T,3> arr, mn::vec<T,3> partition_start, mn::vec<T,3> partition_end) {
  if (arr[0] >= partition_start[0] && arr[0] < partition_end[0])
    if (arr[1] >= partition_start[1] && arr[1] < partition_end[1])
      if (arr[2] >= partition_start[2] && arr[2] < partition_end[2])
        return true;
  return false;
}

/// @brief  Load in binary file (*.bin) as particles. Assumes data is sequential particles [x,y,z] positions.
/// @tparam T Data-type in file, e.g. 'int', 'float' or 'double'.
/// @param pcnt Particle count in file. If too low, not all particles will be read.
/// @param filename Path to file (e.g. MpmParticles/my_particles.bin), starting from AssetDirectory (e.g. claymore/Data/).
/// @return Positions of particles returned as vector of arrays, containing [x,y,z] per particle
template <typename T>
decltype(auto) load_binary_particles(std::size_t pcnt, std::string filename) {
  std::vector<std::array<T, 3>> fields(pcnt);
  auto f = fopen(filename.c_str(), "rb");
  auto res = std::fread((T *)fields.data(), sizeof(T), fields.size() * 3, f);
  std::fclose(f);
  int i = 0;
  fmt::print(fg(fmt::color::white), "Particle count[{}] read from file[{}].\n", filename, fields.size()); 
  fmt::print(fg(fmt::color::white), "Printing first/last 4 particle positions (NOTE: Not scaled): \n");
  for (auto row : fields) {
    if ((i >= 4) && (i < fields.size() - 4))  {i++; continue;}
    std::cout << "Particle["<< i << "] (x, y, z): ";
    for (auto field : row) std::cout << " "<< field << ", ";
    std::cout << '\n';
    i++;
  }
  return fields;
}

/// @brief Load a comma-delimited *.csv file as particles. Reads in [x, y, z] positions per row. Outputs particles into fields.
/// @brief Assume offset, partition_start/end are already scaled to 1x1x1 domain. Does not assumed input file is scaled to 1x1x1, this function will scale it for you.
/// @param filename Path to file (e.g. MpmParticles/my_particles.csv), starting from AssetDirectory (e.g. claymore/Data/).
/// @param sep Delimiter of data. Typically a comma (',') for CSV files.
/// @param fields Vector of array to output data into
/// @param offset Offsets starting corner of particle object from origin. Assume simulation's grid buffer is already included (offset += g_offset, e.g. 8*g_dx).
/// @param partition_start Start corner of particle GPU partition, cut out everything before.
/// @param partition_end End corner of particle GPU partition, cut out everything beyond.
void load_csv_particles(const std::string& filename, char sep, 
                        std::vector<std::array<PREC, 3>>& fields, 
                        mn::vec<PREC, 3> offset, mn::vec<PREC,3> partition_start, mn::vec<PREC,3> partition_end, mn::vec<PREC, 3, 3>& rotation, mn::vec<PREC, 3>& fulcrum) {
  std::ifstream in(filename.c_str());
  if (in) {
    std::string line;
    while (getline(in, line)) {
      std::stringstream sep(line);
      std::string field;
      const int el = 3; // 3 = x, y, z - Default,
      int col = 0;
      std::array<PREC, 3> arr;
      while (getline(sep, field, ',')) 
      {
        if (col >= el) break;
        arr[col] = stof(field) / l + offset[col];
        col++;
      }
      if (inside_partition(arr, partition_start, partition_end)){
        translate_rotate_translate_point(fulcrum, rotation, arr);
        fields.push_back(arr);
      }
    }
  }
  fmt::print(fg(fmt::color::white), "Particle count[{}] read from file[{}].\n", fields.size(), filename); 
  fmt::print(fg(fmt::color::white), "Printing first/last 4 particle positions (NOTE: scaled to 1x1x1 domain): \n");
  int i = 0;
  for (auto row : fields) {
    if ((i >= 4) && (i < fields.size() - 4))  {i++; continue;}
    std::cout << "Particle["<< i << "] (x, y, z): ";
    for (auto field : row) std::cout << " "<< field << ", ";
    std::cout << '\n';
    i++;
  }
}

/// @brief Load a comma-delimited *.csv file as particles. Reads in [x, y, z] positions per row. Outputs particles into fields.
/// @brief Assume offset already scaled to 1x1x1 domain. Does not assumed input file is scaled to 1x1x1, this function will scale it for you.
/// @param filename Path to file (e.g. MpmParticles/my_particles.csv), starting from AssetDirectory (e.g. claymore/Data/)
/// @param sep Delimiter of data. Typically a comma (',') for CSV files.
/// @param fields Vector of array to output data into
/// @param offset Offsets starting corner of particle object from origin. Assume simulation's grid buffer is already included (offset += g_offset, e.g. 8*g_dx).
void load_csv_particles(const std::string& filename, char sep, 
                        std::vector<std::array<PREC, 3>>& fields, 
                        mn::vec<PREC, 3> offset) {
  std::ifstream in(filename.c_str());
  if (in) {
    std::string line;
    while (getline(in, line)) {
      std::stringstream sep(line);
      std::string field;
      const int el = 3; // 3 = x, y, z - Default,
      int col = 0;
      std::array<PREC, 3> arr;
      while (getline(sep, field, ',')) 
      {
        if (col >= el) break;
        arr[col] = stof(field) / l + offset[col];
        col++;
      }
      fields.push_back(arr);
    }
  }
  fmt::print(fg(fmt::color::white), "Particle count[{}] read from file[{}].\n", fields.size(), filename); 
  fmt::print(fg(fmt::color::white), "Printing first/last 4 particle positions (NOTE: scaled to 1x1x1 domain): \n");
  int i = 0;
  for (auto row : fields) {
    if ((i >= 4) && (i < fields.size() - 4))  {i++; continue;}
    std::cout << "Particle["<< i << "] (x, y, z): ";
    for (auto field : row) std::cout << " " << field << ", ";
    std::cout << '\n';
    i++;
  }
}

/// @brief Make box as particles, write to fields as [x,y,z] data.
/// @brief Assume span, offset, radius, partition_start/end are already scaled to 1x1x1 domain. 
/// @param fields Vector of arrays to write particle position [x,y,z] data into
/// @param span Sets max span to look in (for efficiency). Erases particles if too low.
/// @param offset Offsets starting corner of particle object from origin. Assume simulation's grid buffer is already included (offset += g_offset, e.g. 8*g_dx).
/// @param ppc Particle-per-cell, number of particles to sample per 3D grid-cell (e.g. 8).
/// @param partition_start Start corner of particle GPU partition, cut out everything before.
/// @param partition_end End corner of particle GPU partition, cut out everything beyond.
void make_box(std::vector<std::array<PREC, 3>>& fields, 
                        mn::vec<PREC, 3> span, mn::vec<PREC, 3> offset, PREC ppc, mn::vec<PREC,3> partition_start, mn::vec<PREC,3> partition_end, mn::vec<PREC, 3, 3>& rotation, mn::vec<PREC, 3>& fulcrum) {
  // Make a rectangular prism of particles, write to fields
  // Span sets dimensions, offset is starting corner, ppc is particles-per-cell
  // Assumes span and offset are pre-adjusted to 1x1x1 domain with 8*g_dx offset
  PREC ppl_dx = dx / cbrt(ppc); // Linear spacing of particles [1x1x1]
  int i_lim, j_lim, k_lim; // Number of par. per span direction
  i_lim = (int)((span[0]) / ppl_dx + 1.0); 
  j_lim = (int)((span[1]) / ppl_dx + 1.0); 
  k_lim = (int)((span[2]) / ppl_dx + 1.0); 

  for (int i = 0; i < i_lim ; i++){
    for (int j = 0; j < j_lim ; j++){
      for (int k = 0; k < k_lim ; k++){
          std::array<PREC, 3> arr;
          arr[0] = (i + 0.5) * ppl_dx + offset[0];
          arr[1] = (j + 0.5) * ppl_dx + offset[1];
          arr[2] = (k + 0.5) * ppl_dx + offset[2];
          PREC x, y, z;
          x = ((arr[0] - offset[0]) * l);
          y = ((arr[1] - offset[1]) * l);
          z = ((arr[2] - offset[2]) * l);
          if (arr[0] < (span[0] + offset[0]) && arr[1] < (span[1] + offset[1]) && arr[2] < (span[2] + offset[2])) {
            if (0)
            {
              PREC m, b, surf;
              
              m = -0.2/3.2;
              b = (3.2 + 0.2/2.0);
              surf =  m * x + b;
              
              if (y <= surf)
                if (inside_partition(arr, partition_start, partition_end)){
                  translate_rotate_translate_point(fulcrum, rotation, arr);
                  fields.push_back(arr);
                }
            }  
            else if (0)
            {
              PREC m, b, surf_one, surf_two;
              
              m = -(0.1-0.00001)/0.5;
              b = 0.1;
              surf_one =  m * (x) + b;
              m = (0.1-0.00001)/0.5;
              b = 0.1;
              surf_two =  m * (x) + b;

              if (z >= surf_one && z <= surf_two)
                if (inside_partition(arr, partition_start, partition_end)){
                  translate_rotate_translate_point(fulcrum, rotation, arr);
                  fields.push_back(arr);
                }
            }
            else 
            {
            if (inside_partition(arr, partition_start, partition_end)){
              translate_rotate_translate_point(fulcrum, rotation, arr);
              fields.push_back(arr);
            }
          }
        }
      }
    }
  } 
}
/// @brief Make cylinder as particles, write to fields as [x,y,z] data.
/// @brief Assume span, offset, radius, partition_start/end are already scaled to 1x1x1 domain. 
/// @param fields Vector of arrays to write particle position [x,y,z] data into
/// @param span Sets max span to look in (for efficiency). Erases particles if too low.
/// @param offset Offsets starting corner of particle object from origin. Assume simulation's grid buffer is already included (offset += g_offset, e.g. 8*g_dx).
/// @param ppc Particle-per-cell, number of particles to sample per 3D grid-cell (e.g. 8).
/// @param radius Radius of cylinder.
/// @param axis Longitudinal axis of cylinder, e.g. std::string{"X"} for X oriented cylinder.
/// @param partition_start Start corner of particle GPU partition, cut out everything before.
/// @param partition_end End corner of particle GPU partition, cut out everything beyond.
void make_cylinder(std::vector<std::array<PREC, 3>>& fields, 
                        mn::vec<PREC, 3> span, mn::vec<PREC, 3> offset,
                        PREC ppc, PREC radius, std::string axis, mn::vec<PREC,3> partition_start, mn::vec<PREC,3> partition_end, mn::vec<PREC, 3, 3>& rotation, mn::vec<PREC, 3>& fulcrum) {
  PREC ppl_dx = dx / cbrt(ppc); // Linear spacing of particles [1x1x1]
  int i_lim, j_lim, k_lim; // Number of par. per span direction
  i_lim = (int)((span[0]) / ppl_dx + 1.0); 
  j_lim = (int)((span[1]) / ppl_dx + 1.0); 
  k_lim = (int)((span[2]) / ppl_dx + 1.0); 

  for (int i = 0; i < i_lim ; i++) {
    for (int j = 0; j < j_lim ; j++) {
      for (int k = 0; k < k_lim ; k++) {
        std::array<PREC, 3> arr;
        arr[0] = (i + 0.5) * ppl_dx + offset[0];
        arr[1] = (j + 0.5) * ppl_dx + offset[1];
        arr[2] = (k + 0.5) * ppl_dx + offset[2];
        PREC x, y, z;
        x = ((arr[0] - offset[0]) * l);
        y = ((arr[1] - offset[1]) * l);
        z = ((arr[2] - offset[2]) * l);
        if (arr[0] < (span[0] + offset[0]) && arr[1] < (span[1] + offset[1]) && arr[2] < (span[2] + offset[2])) {
          PREC xo, yo, zo;
          xo = yo = zo = radius; 
          PREC r;
          if (axis == "x" || axis == "X") 
            r = std::sqrt((y-yo)*(y-yo) + (z-zo)*(z-zo));
          else if (axis == "y" || axis == "Y") 
            r = std::sqrt((x-xo)*(x-xo) + (z-zo)*(z-zo));
          else if (axis == "z" || axis == "Z") 
            r = std::sqrt((x-xo)*(x-xo) + (y-yo)*(y-yo));
          else 
          {
            r = 0;
            fmt::print(fg(red), "ERROR: Value of axis[{}] is not applicable for a Cylinder. Use X, Y, or Z.", axis);
          }
          if (r <= radius)
            if (inside_partition(arr, partition_start, partition_end)){
              translate_rotate_translate_point(fulcrum, rotation, arr);
              fields.push_back(arr);
            }
        }
      }
    }
  } 
}
/// @brief Make sphere as particles, write to fields as [x,y,z] data.
/// @brief Assume span, offset, radius, partition_start/end are already scaled to 1x1x1 domain. 
/// @param fields Vector of arrays to write particle position [x,y,z] data into
/// @param span Sets max span to look in (for efficiency). Erases particles if too low.
/// @param offset Offsets starting corner of particle object from origin. Assume simulation's grid buffer is already included (offset += g_offset, e.g. 8*g_dx).
/// @param ppc Particle-per-cell, number of particles to sample per 3D grid-cell (e.g. 8).
/// @param radius Radius of cylinder.
/// @param partition_start Start corner of particle GPU partition, cut out everything before.
/// @param partition_end End corner of particle GPU partition, cut out everything beyond.
void make_sphere(std::vector<std::array<PREC, 3>>& fields, 
                        mn::vec<PREC, 3> span, mn::vec<PREC, 3> offset,
                        PREC ppc, PREC radius, mn::vec<PREC,3> partition_start, mn::vec<PREC,3> partition_end, mn::vec<PREC, 3, 3>& rotation, mn::vec<PREC, 3>& fulcrum) {
  PREC ppl_dx = dx / cbrt(ppc); // Linear spacing of particles [1x1x1]
  int i_lim, j_lim, k_lim; // Number of par. per span direction
  i_lim = (int)((span[0]) / ppl_dx + 1.0); 
  j_lim = (int)((span[1]) / ppl_dx + 1.0); 
  k_lim = (int)((span[2]) / ppl_dx + 1.0); 

  for (int i = 0; i < i_lim ; i++) {
    for (int j = 0; j < j_lim ; j++) {
      for (int k = 0; k < k_lim ; k++) {
          std::array<PREC, 3> arr;
          arr[0] = (i + 0.5) * ppl_dx + offset[0];
          arr[1] = (j + 0.5) * ppl_dx + offset[1];
          arr[2] = (k + 0.5) * ppl_dx + offset[2];
          PREC x, y, z;
          x = ((arr[0] - offset[0]) * l);
          y = ((arr[1] - offset[1]) * l);
          z = ((arr[2] - offset[2]) * l);
          if (arr[0] < (span[0] + offset[0]) && arr[1] < (span[1] + offset[1]) && arr[2] < (span[2] + offset[2])) {
            PREC xo, yo, zo;
            xo = yo = zo = radius; 
            PREC r;
            r = std::sqrt((x-xo)*(x-xo) + (y-yo)*(y-yo) + (z-zo)*(z-zo));
            if (r <= radius) 
              if (inside_partition(arr, partition_start, partition_end)){
                translate_rotate_translate_point(fulcrum, rotation, arr);
                fields.push_back(arr);
              }
        }
      }
    }
  } 
}


/// @brief Make bathymetry flume fluid as particles, write to fields as [x,y,z] data.
/// @brief Assume span, offset, radius, partition_start/end are already scaled to 1x1x1 domain. 
/// @param fields Vector of arrays to write particle position [x,y,z] data into
/// @param span Sets max span to look in (for efficiency). Erases particles if too low.
/// @param offset Offsets starting corner of particle object from origin. Assume simulation's grid buffer is already included (offset += g_offset, e.g. 8*g_dx).
/// @param ppc Particle-per-cell, number of particles to sample per 3D grid-cell (e.g. 8).
/// @param partition_start Start corner of particle GPU partition, cut out everything before.
/// @param partition_end End corner of particle GPU partition, cut out everything beyond.
void make_bathymetry(std::vector<std::array<PREC, 3>>& fields, 
                        mn::vec<PREC, 3> span, mn::vec<PREC, 3> offset,
                        PREC ppc, mn::vec<PREC,3> partition_start, mn::vec<PREC,3> partition_end, mn::vec<PREC, 3, 3>& rotation, mn::vec<PREC, 3>& fulcrum, mn::vec<PREC, 2> bathymetry_points[mn::config::g_max_bathymetry_points], int num_bathymetry_points, double fr_scale = 1.0) {
  PREC ppl_dx = dx / cbrt(ppc); // Linear spacing of particles [1x1x1]
  int i_lim, j_lim, k_lim; // Number of par. per span direction
  i_lim = (int)((span[0]) / ppl_dx + 1.0); 
  j_lim = (int)((span[1]) / ppl_dx + 1.0); 
  k_lim = (int)((span[2]) / ppl_dx + 1.0); 
  
  for (int i = 0; i < i_lim ; i++) {
    for (int j = 0; j < j_lim ; j++) {
      for (int k = 0; k < k_lim ; k++) {
        std::array<PREC, 3> arr;
        arr[0] = (i + 0.5) * ppl_dx + offset[0];
        arr[1] = (j + 0.5) * ppl_dx + offset[1];
        arr[2] = (k + 0.5) * ppl_dx + offset[2];
        PREC x, y;
        x = ((arr[0] - offset[0]) * l);
        y = ((arr[1] - offset[1]) * l);
        if (arr[0] < (span[0] + offset[0]) && arr[1] < (span[1] + offset[1]) && arr[2] < (span[2] + offset[2])) {
          // Start ramp segment definition 
          for (int d = 1; d < num_bathymetry_points; d++)
          {
            if (x < bathymetry_points[d][0])
            {
              PREC slope = (bathymetry_points[d][1] - bathymetry_points[d-1][1]) / (bathymetry_points[d][0] - bathymetry_points[d-1][0]);
              if (y >= ( slope * (x - bathymetry_points[d-1][0]) + bathymetry_points[d-1][1]) )
              {
                if (inside_partition(arr, partition_start, partition_end)){
                  translate_rotate_translate_point(fulcrum, rotation, arr);
                  fields.push_back(arr);
                }
                break;
              }
              else break;
            }
          }
        }
      }
    }
  } 
}


/// @brief Make OSU LWF flume fluid as particles, write to fields as [x,y,z] data.
/// @brief Assume span, offset, radius, partition_start/end are already scaled to 1x1x1 domain. 
/// @param fields Vector of arrays to write particle position [x,y,z] data into
/// @param span Sets max span to look in (for efficiency). Erases particles if too low.
/// @param offset Offsets starting corner of particle object from origin. Assume simulation's grid buffer is already included (offset += g_offset, e.g. 8*g_dx).
/// @param ppc Particle-per-cell, number of particles to sample per 3D grid-cell (e.g. 8).
/// @param partition_start Start corner of particle GPU partition, cut out everything before.
/// @param partition_end End corner of particle GPU partition, cut out everything beyond.
void make_OSU_LWF(std::vector<std::array<PREC, 3>>& fields, 
                        mn::vec<PREC, 3> span, mn::vec<PREC, 3> offset,
                        PREC ppc, mn::vec<PREC,3> partition_start, mn::vec<PREC,3> partition_end, mn::vec<PREC, 3, 3>& rotation, mn::vec<PREC, 3>& fulcrum, double fr_scale = 1.0) {
  PREC ppl_dx = dx / cbrt(ppc); // Linear spacing of particles [1x1x1]
  int i_lim, j_lim, k_lim; // Number of par. per span direction
  i_lim = (int)((span[0]) / ppl_dx + 1.0); 
  j_lim = (int)((span[1]) / ppl_dx + 1.0); 
  k_lim = (int)((span[2]) / ppl_dx + 1.0); 

  // Assume JSON input offsets model 2.085 meters forward in X
  PREC bathx[7];
  PREC bathy[7];
  PREC bath_slope[7];

  bathx[0] = 0.0; //- wave_maker_neutral; // Start of bathymetry X direction
  bathx[0] += 0.1; // Turns out wave-maker starts -2.085m relative to neutral, adjust 8.5cm
  bathx[1] = 14.275 + bathx[0];
  bathx[2] = 3.65 + bathx[1];
  bathx[3] = 10.975 + bathx[2];
  bathx[4] = 14.625 + bathx[3];
  bathx[5] = 36.575 + bathx[4];
  bathx[6] = 7.35 + bathx[5];

  bathy[0] = 0.0;
  bathy[1] = 0.225 + bathy[0]; // Bathymetry slab raised ~0.15m, 0.076m thick
  bathy[2] = 0.0 + bathy[1];
  bathy[3] = 1.15;
  bathy[4] = 1.75; //(14.63f / 24.f) + bathy[3];
  bathy[5] = 0.0 + bathy[4];
  bathy[6] = (7.35 / 12.25) + bathy[5]; 
  // Adjust OSU LWF bathymetry for Froude similarity scaling
  for (int d=0; d<7; ++d) {
    bathx[d] *= fr_scale;
    bathy[d] *= fr_scale;
  }

  bath_slope[0] = 0;
  bath_slope[1] = 0;
  bath_slope[2] = 0;
  bath_slope[3] = 1.0 / 11.864861; // 1/12
  bath_slope[4] = 1.0 / 24.375;  // 1/24
  bath_slope[5] = 0;
  bath_slope[6] = 1.0 / 12.25;

  PREC buffer_y = 0.025 * fr_scale;
  
  for (int i = 0; i < i_lim ; i++) {
    for (int j = 0; j < j_lim ; j++) {
      for (int k = 0; k < k_lim ; k++) {
        std::array<PREC, 3> arr;
        arr[0] = (i + 0.5) * ppl_dx + offset[0];
        arr[1] = (j + 0.5) * ppl_dx + offset[1];
        arr[2] = (k + 0.5) * ppl_dx + offset[2];
        PREC x, y;
        x = ((arr[0] - offset[0]) * l);
        y = ((arr[1] - offset[1]) * l);
        if (arr[0] < (span[0] + offset[0]) && arr[1] < (span[1] + offset[1]) && arr[2] < (span[2] + offset[2])) {
          // Start ramp segment definition for OSU flume
          // Based on bathymetry diagram, February
          for (int d = 1; d < 7; d++)
          {
            if (x < bathx[d])
            {
              if (y >= ( bath_slope[d] * (x - bathx[d-1]) + bathy[d-1] - buffer_y) )
              {
                if (inside_partition(arr, partition_start, partition_end)){
                  translate_rotate_translate_point(fulcrum, rotation, arr);
                  fields.push_back(arr);
                }
                break;
              }
              else break;
            }
          }
        }
      }
    }
  } 
}



/// @brief Make OSU TWB flume fluid as particles, write to fields as [x,y,z] data.
/// @brief Assume span, offset, radius, partition_start/end are already scaled to 1x1x1 domain. 
/// @param fields Vector of arrays to write particle position [x,y,z] data into
/// @param span Sets max span to look in (for efficiency). Erases particles if too low.
/// @param offset Offsets starting corner of particle object from origin. Assume simulation's grid buffer is already included (offset += g_offset, e.g. 8*g_dx).
/// @param ppc Particle-per-cell, number of particles to sample per 3D grid-cell (e.g. 8).
/// @param partition_start Start corner of particle GPU partition, cut out everything before.
/// @param partition_end End corner of particle GPU partition, cut out everything beyond.
void make_OSU_TWB(std::vector<std::array<PREC, 3>>& fields, 
                        mn::vec<PREC, 3> span, mn::vec<PREC, 3> offset,
                        PREC ppc, mn::vec<PREC,3> partition_start, mn::vec<PREC,3> partition_end, mn::vec<PREC, 3, 3>& rotation, mn::vec<PREC, 3>& fulcrum, double fr_scale = 1.0) {
  PREC ppl_dx = dx / cbrt(ppc); // Linear spacing of particles [1x1x1]
  int i_lim, j_lim, k_lim; // Number of par. per span direction
  i_lim = (int)((span[0]) / ppl_dx + 1.0); 
  j_lim = (int)((span[1]) / ppl_dx + 1.0); 
  k_lim = (int)((span[2]) / ppl_dx + 1.0); 

  // Assume JSON input offsets model 2 meters forward in X
  PREC bathx[4];
  PREC bathy[4];
  PREC bath_slope[4];

  bathx[0] = 0.0; //- wave_maker_neutral; // Start of bathymetry X direction
  bathx[1] = 11.3 + bathx[0];
  bathx[2] = 20.0 + bathx[1];
  bathx[3] = 10.0 + bathx[2];

  bathy[0] = 0.0;
  bathy[1] = (0.0) + bathy[0]; // Bathymetry slab raised ~0.15m, 0.076m thick
  bathy[2] = (20.0 / 20.0) + bathy[1];
  bathy[3] = (0.0) + bathy[2];

  // Adjust OSU TWB bathymetry for Froude similarity scaling
  for (int d=0; d<4; ++d) {
    bathx[d] *= fr_scale;
    bathy[d] *= fr_scale;
  }

  bath_slope[0] = 0.0;
  bath_slope[1] = 0.0;
  bath_slope[2] = 1.0 / 20.0;
  bath_slope[3] = 0.0;


  for (int i = 0; i < i_lim ; i++) {
    for (int j = 0; j < j_lim ; j++) {
      for (int k = 0; k < k_lim ; k++) {
        std::array<PREC, 3> arr;
        arr[0] = (i + 0.5) * ppl_dx + offset[0];
        arr[1] = (j + 0.5) * ppl_dx + offset[1];
        arr[2] = (k + 0.5) * ppl_dx + offset[2];
        PREC x, y;
        x = ((arr[0] - offset[0]) * l);
        y = ((arr[1] - offset[1]) * l);
        if (arr[0] < (span[0] + offset[0]) && arr[1] < (span[1] + offset[1]) && arr[2] < (span[2] + offset[2])) {
          // Start ramp segment definition for OSU flume
          // Based on bathymetry diagram, February
          for (int d = 1; d < 4; d++)
          {
            if (x < bathx[d])
            {
              if (y >= ( bath_slope[d] * (x - bathx[d-1]) + bathy[d-1]) )
              {
                if (inside_partition(arr, partition_start, partition_end)){
                  translate_rotate_translate_point(fulcrum, rotation, arr);
                  fields.push_back(arr);
                }
                break;
              }
              else break;
            }
          }
        }
      }
    }
  } 
}

template <typename T>
bool inside_box(std::array<T,3>& arr, mn::vec<T,3> span, mn::vec<T,3> offset) 
{
  if (arr[0] >= offset[0] && arr[0] < span[0] + offset[0])
    if (arr[1] >= offset[1] && arr[1] < span[1] + offset[1])
      if (arr[2] >= offset[2] && arr[2] < span[2] + offset[2])
        return true;
  return false;
}

template <typename T>
bool inside_cylinder(std::array<T,3>& arr, T radius, std::string axis, mn::vec<T,3> span, mn::vec<T,3> offset) 
{
  std::array<T,3> center;
  for (int d=0; d<3; d++) center[d] = offset[d] + radius;

  if (axis == "X" || axis == "x")
  { 
    PREC r = std::sqrt((arr[1]-center[1])*(arr[1]-center[1]) + (arr[2]-center[2])*(arr[2]-center[2]));
    if (r <= radius)
       if (arr[0] >= offset[0] && arr[0] < offset[0] + span[0])
          return true;
  }
  else if (axis == "Y" || axis == "Y")
  { 
    PREC r = std::sqrt((arr[0]-center[0])*(arr[0]-center[0]) + (arr[2]-center[2])*(arr[2]-center[2]));
    if (r <= radius)
       if (arr[1] >= offset[1] && arr[1] < offset[1] + span[1])
          return true;
  }
  else if (axis == "Z" || axis == "z")
  { 
    PREC r = std::sqrt((arr[1]-center[1])*(arr[1]-center[1]) + (arr[0]-center[0])*(arr[0]-center[0]));
    if (r <= radius)
       if (arr[2] >= offset[2] && arr[2] < offset[2] + span[2])
          return true;
  }
  return false;
}

template <typename T>
bool inside_sphere(std::array<T,3>& arr, T radius, mn::vec<T,3> offset) 
{
  std::array<T,3> center;
  for (int d=0; d<3; d++) center[d] = offset[d] + radius;
  PREC r = std::sqrt((arr[0]-center[0])*(arr[0]-center[0]) + (arr[1]-center[1])*(arr[1]-center[1]) + (arr[2]-center[2])*(arr[2]-center[2]));
  if (r <= radius)
      return true;
  return false;
}

template <typename T>
bool inside_bathymetry(std::array<T, 3>& arr, mn::vec<T, 2> bathymetry_points[mn::config::g_max_bathymetry_points], int num_bathymetry_points) {
  T x = arr[0];
  T y = arr[1];
  for (int d = 1; d < num_bathymetry_points; d++) {
    if (d >= mn::config::g_max_bathymetry_points) break;
    if (x < bathymetry_points[d][0]) {
      T slope = (bathymetry_points[d][1] - bathymetry_points[d-1][1]) / (bathymetry_points[d][0] - bathymetry_points[d-1][0]);
      T y_bath = slope * (x - bathymetry_points[d-1][0]) + bathymetry_points[d-1][1];
      if (y <= y_bath) {
        return true;
      }
      break;
    }
  }
  return false;
}

template <typename T>
void subtract_box(std::vector<std::array<T,3>>& particles, mn::vec<T,3> span, mn::vec<T,3> offset) {
  fmt::print("Previous particle count: {}\n", particles.size());
  particles.erase(std::remove_if(particles.begin(), particles.end(),
                              [&](std::array<T,3> x){ return inside_box(x, span, offset); }), particles.end());
  fmt::print("Updated particle count: {}\n", particles.size());
}

template <typename T>
void subtract_cylinder(std::vector<std::array<T,3>>& particles, T radius, std::string axis, mn::vec<T,3> span, mn::vec<T,3> offset) {
  fmt::print("Previous particle count: {}\n", particles.size());
  particles.erase(std::remove_if(particles.begin(), particles.end(),
                              [&](std::array<T,3> x){ return inside_cylinder(x, radius / l, axis, span, offset); }), particles.end());
  fmt::print("Updated particle count: {}\n", particles.size());
}


template <typename T>
void subtract_sphere(std::vector<std::array<T,3>>& particles, T radius, mn::vec<T,3> offset) {
  fmt::print("Previous particle count: {}\n", particles.size());
  particles.erase(std::remove_if(particles.begin(), particles.end(),
                              [&](std::array<T,3> x){ return inside_sphere(x, radius / l, offset); }), particles.end());
  fmt::print("Updated particle count: {}\n", particles.size());
}

template <typename T>
void subtract_bathymetry(std::vector<std::array<T,3>>& particles, mn::vec<T,3> span, mn::vec<T,3> offset, mn::vec<T,2> bathymetry_points[mn::config::g_max_bathymetry_points], int num_bathymetry_points) {
  fmt::print("Previous particle count: {}\n", particles.size());
  particles.erase(std::remove_if(particles.begin(), particles.end(),
                              [&](std::array<T,3> x){ return inside_bathymetry(x, bathymetry_points, num_bathymetry_points); }), particles.end());
  fmt::print("Updated particle count: {}\n", particles.size());
}


void load_FEM_Vertices(const std::string& filename, char sep, 
                       VerticeHolder& fields, 
                       mn::vec<PREC, 3> offset){
  std::ifstream in(filename.c_str());
  if (in) {
      std::string line;
      while (getline(in, line)) {
          std::stringstream sep(line);
          std::string field;
          const int el = 3; // x, y, z - Default
          std::array<PREC, 13> arr;
          int col = 0;
          while (getline(sep, field, ',')) {
              if (col >= el) break;
              arr[col] = stof(field) / l + offset[col];
              arr[col+el] = arr[col]; 
              col++;
          }
          arr[3] = (PREC)0.;
          arr[4] = (PREC)0.;
          arr[5] = (PREC)0.;
          arr[6] = (PREC)0.;
          arr[7] = (PREC)0.;
          arr[8] = (PREC)0.;
          arr[9] = (PREC)0.;
          arr[10] = (PREC)0.;
          arr[11] = (PREC)0.;
          arr[12] = (PREC)0.;
          fields.push_back(arr);
      }
  }
  if (verbose) {
    for (auto row : fields) {
        for (auto field : row) std::cout << field << ' ';
        std::cout << '\n';
    }
  }
}

void load_FEM_Elements(const std::string& filename, char sep, 
                       ElementHolder& fields){
  std::ifstream in(filename.c_str());
  if (in) {
      std::string line;
      while (getline(in, line)) {
          std::stringstream sep(line);
          std::string field;
          int col = 0;
          // Elements hold integer IDs of vertices
          const int el = 4; // 4-node Tetrahedron
          std::array<int, el> arr;
          while (getline(sep, field, ',')) {
              if (col >= el) break;
              arr[col] = stoi(field); // string to integer
              col++;
          }
          fields.push_back(arr);
      }
  }
  if (verbose) {
    for (auto row : fields) for (auto field : row) std::cout << field << ' ' << '\n';
  }
}


void load_motionPath(const std::string& filename, char sep, MotionHolder& fields, int rate=1, double fr_scale = 1.0){
  std::ifstream in((filename).c_str());
  if (in) {
      int iter = 0;
      std::string line;
      while (getline(in, line)) {
          std::stringstream sep(line);
          std::string field;
          int col = 0;
          std::array<float, 3> arr;
          while (getline(sep, field, ',')) {
              if (col >= 3) break;
              if ((iter % rate) == 0) {
		arr[col] = stof(field);
		if (col == 0) arr[col] *= sqrt(fr_scale); // Adjust time for froude scaling
		else if (col == 1) arr[col] *= fr_scale; // Adjust X position for froude scaling
		else if (col == 2) arr[col] *= sqrt(fr_scale); // Adjust X velocity for froude scaling
	      }
	      col++;
          }
          if ((iter % rate) == 0) fields.push_back(arr);
          iter++;
      }
  }
  if (verbose) {
    for (auto row : fields) {
      for (auto field : row) std::cout << field << ' '; 
      std::cout << '\n';
    }
    std::cout << '\n';
  }
}


/// @brief Check if JSON value at 'key' is (i) in JSON script and (ii) is a string.
/// Return retrieved string from JSON or return backup value if not found/is not a string.
std::string CheckString(rapidjson::Value &object, const std::string &key, std::string backup) {
  auto check = object.FindMember(key.c_str());
  if (check != object.MemberEnd())
  { 
    if (!object[key.c_str()].IsString()){
      fmt::print(fg(red), "ERROR: Input [{}] not a string! Fix and retry. Current type: {}.\n", key, object[key.c_str()].GetType());
    }
    else {
      fmt::print(fg(green), "Input [{}] found: {}.\n", key, object[key.c_str()].GetString());
      return object[key.c_str()].GetString();
    }
  }
  else {
    fmt::print(fg(red), "ERROR: Input [{}] does not exist in scene file!\n ", key);
  }
  fmt::print(fg(orange), "WARNING: Press ENTER to use default value for [{}]: {}.\n", key, backup);
  if (mn::config::g_log_level >= 3) getchar();
  return backup;
}

/// @brief Check if JSON value at 'key' is (i) in JSON script and (ii) is a number array.
/// Return retrieved double-precision floating point array from JSON or return backup value if not found/is not a number array.
std::vector<std::string> CheckStringArray(rapidjson::Value &object, const std::string &key, std::vector<std::string> backup) {
  auto check = object.FindMember(key.c_str());
  if (check != object.MemberEnd())
  { 
    if (object[key.c_str()].IsArray())
    {
      int dim = object[key.c_str()].GetArray().Size();
      if (dim > 0) {
        if (!object[key.c_str()].GetArray()[0].IsString()){
          fmt::print(fg(red), "ERROR: Input [{}] not an array of strings! Fix and retry. Current type: {}.\n", key, kTypeNames[object[key.c_str()].GetArray()[0].GetType()]);
        }
        else {
          std::vector<std::string> arr;
          // for (int d=0; d<dim; d++) assert(object[key.c_str()].GetArray()[d].GetString());
          for (int d=0; d<dim; d++) arr.push_back(object[key.c_str()].GetArray()[d].GetString());
          fmt::print(fg(green), "Input [{}] found: ", key);
          fmt::print(fg(green), " [ ");
          for (int d=0; d<dim; d++) fmt::print(fg(green), " {}, ", arr[d]);
          fmt::print(fg(green), "]\n");
          return arr;
        }
      } else {
        fmt::print(fg(red), "ERROR: Input [{}] is an Array! Populate and retry.\n", key);
      }
    }
    else 
    {
      fmt::print(fg(red), "ERROR: Input [{}] not an Array! Fix and retry. Current type: {}.\n", key, object[key.c_str()].GetType());
    }
  }
  else {
    fmt::print(fg(red), "ERROR: Input [{}] not in scene file! \n ", key);
    fmt::print(fg(orange), "WARNING: Using default value: [ ");
    for (int d=0; d<backup.size(); d++) fmt::print(fg(orange), " {}, ", backup[d]);
    fmt::print(fg(orange), "]\n");
    return backup;
  }
  fmt::print(fg(orange), "WARNING: Press ENTER to use default value for [{}]: ", key);
  fmt::print(fg(orange), " [ ");
  for (int d=0; d<backup.size(); d++) fmt::print(fg(orange), " {}, ", backup[d]);
  fmt::print(fg(orange), "]\n");
  if (mn::config::g_log_level >= 3) getchar();
  return backup;
}

/// @brief Check if JSON value at 'key' is (i) in JSON script and (ii) is a number array.
/// Return retrieved double-precision floating point array from JSON or return backup value if not found/is not a number array.
template<int dim=3>
mn::vec<PREC, dim> CheckDoubleArray(rapidjson::Value &object, const std::string &key, mn::vec<PREC,dim> backup) {
  auto check = object.FindMember(key.c_str());
  if (check != object.MemberEnd())
  { 
    if (object[key.c_str()].IsArray())
    {
      // assert(object[key.c_str()].IsArray());
      mn::vec<PREC,dim> arr; 
      if (object[key.c_str()].GetArray().Size() != dim) 
      {
        fmt::print(fg(red), "ERROR: Input [{}] must be an array of size [{}]! Fix and retry. Current size: {}.\n", key, dim, object[key.c_str()].GetArray().Size());
      }
      else {
        if (!object[key.c_str()].GetArray()[0].IsNumber()){
          fmt::print(fg(red), "ERROR: Input [{}] not an array of numbers! Fix and retry. Current type: {}.\n", key, kTypeNames[object[key.c_str()].GetArray()[0].GetType()]);
        }
        else {
          // for (int d=0; d<dim; d++) assert(object[key.c_str()].GetArray()[d].GetDouble());
          for (int d=0; d<dim; d++) arr[d] = object[key.c_str()].GetArray()[d].GetDouble();
          fmt::print(fg(green), "Input [{}] found: [{}, {}, {}].\n", key, arr[0],arr[1],arr[2]);
          return arr;
        }
      }
    }
    else 
    {
      fmt::print(fg(red), "ERROR: Input [{}] not an Array! Fix and retry. Current type: {}.\n", key, object[key.c_str()].GetType());
    }
  }
  else {
    fmt::print(fg(red), "ERROR: Input [{}] not in scene file! \n ", key);
    fmt::print(fg(orange), "WARNING: Using default value: [ ");
    for (int d=0; d<dim; d++) fmt::print(fg(orange), " {}, ", backup[d]);
    fmt::print(fg(orange), "]\n");
    return backup;
  }
  fmt::print(fg(orange), "WARNING: Press ENTER to use default value for [{}]: ", key);
  fmt::print(fg(orange), " [ ");
  for (int d=0; d<dim; d++) fmt::print(fg(orange), " {}, ", backup[d]);
  fmt::print(fg(orange), "]\n");
  if (mn::config::g_log_level >= 3) getchar();
  return backup;
}

/// @brief Check if JSON value at 'key' is (i) in JSON script and (ii) is a number array.
/// Return retrieved double-precision floating point array from JSON or return backup value if not found/is not a number array.
template<int dim=3>
mn::vec<float, dim> CheckFloatArray(rapidjson::Value &object, const std::string &key, mn::vec<float,dim> backup) {
  auto check = object.FindMember(key.c_str());
  if (check != object.MemberEnd())
  { 
    if (object[key.c_str()].IsArray())
    {
      // assert(object[key.c_str()].IsArray());
      mn::vec<float,dim> arr; 
      if (object[key.c_str()].GetArray().Size() != dim) 
      {
        fmt::print(fg(red), "ERROR: Input [{}] must be an array of size [{}]! Fix and retry. Current size: {}.\n", key, dim, object[key.c_str()].GetArray().Size());
      }
      else {
        if (!object[key.c_str()].GetArray()[0].IsNumber()){
          fmt::print(fg(red), "ERROR: Input [{}] not an array of numbers! Fix and retry. Current type: {}.\n", key, kTypeNames[object[key.c_str()].GetArray()[0].GetType()]);
        }
        else {
          // for (int d=0; d<dim; d++) assert(object[key.c_str()].GetArray()[d].GetDouble());
          for (int d=0; d<dim; d++) arr[d] = object[key.c_str()].GetArray()[d].GetFloat();
          fmt::print(fg(green), "Input [{}] found: [{}, {}, {}].\n", key, arr[0],arr[1],arr[2]);
          return arr;
        }
      }
    }
    else 
    {
      fmt::print(fg(red), "ERROR: Input [{}] not an Array! Fix and retry. Current type: {}.\n", key, object[key.c_str()].GetType());
    }
  }
  else {
    fmt::print(fg(red), "ERROR: Input [{}] not in scene file! \n ", key);
    fmt::print(fg(orange), "WARNING: Using default value: [ ");
    for (int d=0; d<dim; d++) fmt::print(fg(orange), " {}, ", backup[d]);
    fmt::print(fg(orange), "]\n");
    return backup;
  }
  fmt::print(fg(orange), "WARNING: Press ENTER to use default value for [{}]: ", key);
  fmt::print(fg(orange), " [ ");
  for (int d=0; d<dim; d++) fmt::print(fg(orange), " {}, ", backup[d]);
  fmt::print(fg(orange), "]\n");
  if (mn::config::g_log_level >= 3) getchar();
  return backup;
}
/// @brief Check if JSON value at 'key' is (i) in JSON script and (ii) is an integer array.
/// Return retrieved integer array from JSON or return backup value if not found/is not a double.
template<int dim>
mn::vec<int, dim> CheckIntArray(rapidjson::Value &object, const std::string &key, mn::vec<int,dim> backup) {
  auto check = object.FindMember(key.c_str());
  if (check != object.MemberEnd()) { 
    if (object[key.c_str()].IsArray()) {
      mn::vec<int,dim> arr; 
      if (object[key.c_str()].GetArray().Size() != dim) {
        fmt::print(fg(red), "ERROR: Input [{}] must be an array of size [{}]! Fix and retry. Current size: {}.\n", key, dim, object[key.c_str()].GetArray().Size());
      }
      else {
        if (!object[key.c_str()].GetArray()[0].IsInt()) {
          fmt::print(fg(red), "ERROR: Input [{}] not an array of integers! Fix and retry. Current type: {}.\n", key, kTypeNames[object[key.c_str()].GetArray()[0].GetType()]);
        }
        else {
          // for (int d=0; d<dim; d++) assert(object[key.c_str()].GetArray()[d].GetInt());
          for (int d=0; d<dim; d++) arr[d] = object[key.c_str()].GetArray()[d].GetInt();
          fmt::print(fg(green), "Input [{}] found: [{}, {}, {}].\n", key, arr[0],arr[1],arr[2]);
          return arr;
        }
      }
    }
    else {
      fmt::print(fg(red), "ERROR: Input [{}] not an Array! Fix and retry. Current type: {}.\n", key, object[key.c_str()].GetType());
    }
  }
  else {
    fmt::print(fg(red), "ERROR: Input [{}] not in scene file! \n ", key);
    fmt::print(fg(orange), "WARNING: Using default value: [ ");
    for (int d=0; d<dim; d++) fmt::print(fg(orange), " {}, ", backup[d]);
    fmt::print(fg(orange), "]\n");
    return backup;
  }
  fmt::print(fg(orange), "WARNING: Press ENTER to use default value for [{}]: ", key);
  fmt::print(fg(orange), " [ ");
  for (int d=0; d<dim; d++) fmt::print(fg(orange), " {}, ", backup[d]);
  fmt::print(fg(orange), "]\n");
  if (mn::config::g_log_level >= 3) getchar();
  return backup;
}

/// @brief Check if JSON value at 'key' is (i) in JSON script and (ii) is an integer array.
/// Return retrieved integer array from JSON or return backup value if not found/is not a double.
std::vector<int> CheckIntArray(rapidjson::Value &object, const std::string &key, std::vector<int> backup) {
  auto check = object.FindMember(key.c_str());
  if (check != object.MemberEnd()) { 
    if (object[key.c_str()].IsArray()) {
      int dim = object[key.c_str()].GetArray().Size();
      std::vector<int> arr; 
      if (!object[key.c_str()].GetArray()[0].IsInt()) {
        fmt::print(fg(red), "ERROR: Input [{}] not an array of integers! Fix and retry. Current type: {}.\n", key, kTypeNames[object[key.c_str()].GetArray()[0].GetType()]);
      }
      else {
        // for (int d=0; d<dim; d++) assert(object[key.c_str()].GetArray()[d].GetInt());
        for (int d=0; d<dim; d++) {
          arr.push_back(object[key.c_str()].GetArray()[d].GetInt());
          fmt::print(fg(green), "Input [{}] found: [{}].\n", key, arr[d]);
        }
        return arr;
      }
      
    }
    else {
      fmt::print(fg(red), "ERROR: Input [{}] not an Array! Fix and retry. Current type: {}.\n", key, object[key.c_str()].GetType());
    }
  }
  else {
    fmt::print(fg(red), "ERROR: Input [{}] not in scene file! \n ", key);
    fmt::print(fg(orange), "WARNING: Using default value: [ ");
    for (int d=0; d<backup.size(); d++) fmt::print(fg(orange), " {}, ", backup[d]);
    fmt::print(fg(orange), "]\n");
    return backup;
  }
  fmt::print(fg(orange), "WARNING: Press ENTER to use default value for [{}]: ", key);
  fmt::print(fg(orange), " [ ");
  for (int d=0; d<backup.size(); d++) fmt::print(fg(orange), " {}, ", backup[d]);
  fmt::print(fg(orange), "]\n");
  if (mn::config::g_log_level >= 3) getchar();
  return backup;
}

/// @brief Check if JSON value at 'key' is (i) in JSON script and (ii) is a double.
/// Return retrieved double from JSON or return backup value if not found/is not a double.
double CheckDouble(rapidjson::Value &object, const std::string &key, double backup) {
  auto check = object.FindMember(key.c_str());
  if (check != object.MemberEnd())
  { 
    if (!object[key.c_str()].IsDouble() && !object[key.c_str()].IsNumber()){
      fmt::print(fg(red), "ERROR: Input [{}] not a number! Fix and retry. Current type: {}.\n", key, object[key.c_str()].GetType());
    }
    else {
      fmt::print(fg(green), "Input [{}] found: {}.\n", key, object[key.c_str()].GetDouble());
      return object[key.c_str()].GetDouble();
    }
  }
  else {
    fmt::print(fg(red), "ERROR: Input [{}] does not exist in scene file!\n ", key);
  }
  fmt::print(fg(orange), "WARNING: Press ENTER to use default value for [{}]: {}.\n", key, backup);
  if (mn::config::g_log_level >= 3) getchar();
  return backup;
}

/// @brief Check if JSON value at 'key' is (i) in JSON script and (ii) is a booelean.
/// Return retrieved boolean from JSON or return backup value if not found/is not a boolean.
bool CheckBool(rapidjson::Value &object, const std::string &key, bool backup) {
  auto check = object.FindMember(key.c_str());
  if (check != object.MemberEnd())
  { 
    if (!object[key.c_str()].IsBool()){
      fmt::print(fg(red), "ERROR: Input [{}] not a boolean (true/false! Fix and retry. Current type: {}.\n", key, object[key.c_str()].GetType());
    }
    else {
      fmt::print(fg(green), "Input [{}] found: {}.\n", key, object[key.c_str()].GetBool());
      return object[key.c_str()].GetBool();
    }
  }
  else {
    fmt::print(fg(red), "ERROR: Input [{}] does not exist in scene file!\n ", key);
  }
  fmt::print(fg(orange), "WARNING: Press ENTER to use default value for [{}]: {}.\n", key, backup);
  if (mn::config::g_log_level >= 3) getchar();
  return backup;
}

/// @brief Check if JSON value at 'key' is (i) in JSON script and (ii) is a float.
/// Return retrieved float from JSON or return backup value if not found/is not a float.
float CheckFloat(rapidjson::Value &object, const std::string &key, float backup) {
  auto check = object.FindMember(key.c_str());
  if (check != object.MemberEnd())
  { 
    if (!object[key.c_str()].IsFloat() && !object[key.c_str()].IsNumber()){
      fmt::print(fg(red), "ERROR: Input [{}] not a number! Fix and retry. Current type: {}.\n", key, object[key.c_str()].GetType());
    }
    else {
      fmt::print(fg(green), "Input [{}] found: {}.\n", key, object[key.c_str()].GetFloat());
      return object[key.c_str()].GetFloat();
    }
  }
  else {
    fmt::print(fg(red), "ERROR: Input [{}] does not exist in scene file!\n ", key);
  }
  fmt::print(fg(orange), "WARNING: Press ENTER to use default value for [{}]: {}.\n", key, backup);
  if (mn::config::g_log_level >= 3) getchar();
  return backup;
}

/// @brief Check if JSON value at 'key' is (i) in JSON script and (ii) is an int.
/// Return retrieved int from JSON or return backup value if not found/is not an int.
int CheckInt(rapidjson::Value &object, const std::string &key, int backup) {
  auto check = object.FindMember(key.c_str());
  if (check != object.MemberEnd())
  { 
    if (!object[key.c_str()].IsNumber()){
      fmt::print(fg(red), "ERROR: Input [{}] not a number! Fix and retry. Current type: {}.\n", key, object[key.c_str()].GetType());
    }
    else {
      fmt::print(fg(green), "Input [{}] found: {}.\n", key, object[key.c_str()].GetInt());
      return object[key.c_str()].GetInt();
    }
  }
  else {
    fmt::print(fg(red), "ERROR: Input [{}] does not exist in scene file!\n ", key);
  }
  fmt::print(fg(orange), "WARNING: Press ENTER to use default value for [{}]: {}.\n", key, backup);
  if (mn::config::g_log_level >= 3) getchar();
  return backup;
}

/// @brief Check if JSON value at 'key' is (i) in JSON script and (ii) is an uint32.
/// Return retrieved int from JSON or return backup value if not found/is not an uint32.
uint32_t CheckUint(rapidjson::Value &object, const std::string &key, uint32_t backup) {
  auto check = object.FindMember(key.c_str());
  if (check != object.MemberEnd())
  { 
    if (!object[key.c_str()].IsNumber() && !object[key.c_str()].IsUint() ){
      fmt::print(fg(red), "ERROR: Input [{}] not an 32-bit Unsigned Integer! Fix and retry. Current type: {}.\n", key, object[key.c_str()].GetType());
    }
    else {
      fmt::print(fg(green), "Input [{}] found: {}.\n", key, object[key.c_str()].GetUint());
      return object[key.c_str()].GetUint();
    }
  }
  else {
    fmt::print(fg(red), "ERROR: Input [{}] does not exist in scene file!\n ", key);
  }
  fmt::print(fg(orange), "WARNING: Press ENTER to use default value for [{}]: {}.\n", key, backup);
  if (mn::config::g_log_level >= 3) getchar();
  return backup;
}

/// @brief Check if JSON value at 'key' is (i) in JSON script and (ii) is an uint64.
/// Return retrieved int from JSON or return backup value if not found/is not an uint64.
uint64_t CheckUint64(rapidjson::Value &object, const std::string &key, uint64_t backup) {
  auto check = object.FindMember(key.c_str());
  if (check != object.MemberEnd())
  { 
    if (!object[key.c_str()].IsNumber() && !object[key.c_str()].IsUint64() ){
      fmt::print(fg(red), "ERROR: Input [{}] not a 64-bit Unsigned Integer! Fix and retry. Current type: {}.\n", key, object[key.c_str()].GetType());
    }
    else {
      fmt::print(fg(green), "Input [{}] found: {}.\n", key, object[key.c_str()].GetUint64());
      return object[key.c_str()].GetUint64();
    }
  }
  else {
    fmt::print(fg(red), "ERROR: Input [{}] does not exist in scene file!\n ", key);
  }
  fmt::print(fg(orange), "WARNING: Press ENTER to use default value for [{}]: {}.\n", key, backup);
  if (mn::config::g_log_level >= 3) getchar();
  return backup;
}

/// @brief Parses an input JSON script to set-up a Multi-GPU simulation.
/// @param fn Filename of input JSON script. Default: scene.json in current working directory
/// @param benchmark Simulation object to initalize. Calls GPU/Host functions and manages memory.
/// @param models Contains initial particle positions for simulation. One per GPU.
void parse_scene(std::string fn,
                 std::unique_ptr<mn::mgsp_benchmark> &benchmark,
                 std::vector<std::array<PREC, 3>> models[mn::config::g_model_cnt]) {
  fs::path p{fn};
  if (p.empty()) fmt::print(fg(red), "ERROR: Input file[{}] does not exist.\n", fn);
  else {
    std::size_t size=0;
    try{ size = fs::file_size(p); } 
    catch(fs::filesystem_error& e) { std::cout << e.what() << '\n'; }
    std::string configs;
    configs.resize(size);
    std::ifstream istrm(fn);
    if (!istrm.is_open())  fmt::print(fg(red), "ERROR: Cannot open file[{}]\n", fn);
    else istrm.read(const_cast<char *>(configs.data()), configs.size());
    istrm.close();
    fmt::print(fg(green), "Opened scene file[{}] of size [{}] kilobytes.\n", fn, configs.size());
    fmt::print(fg(white), "Scanning JSON scheme in file[{}]...\n", fn);


    rj::Document doc;
    doc.Parse(configs.data());
    for (rj::Value::ConstMemberIterator itr = doc.MemberBegin();
         itr != doc.MemberEnd(); ++itr) {
      fmt::print("Scene member {} is type {}. \n", itr->name.GetString(),
                 kTypeNames[itr->value.GetType()]);
    }
    mn::vec<PREC, 3> domain; // Domain size [meters] for whole 3D simulation
    mn::vec<double, 2> time; // Time range [seconds] for simulation

    double froude_scaling = 1.0; // Froude length scaling to apply. Keeps Fr = U / sqrt(gL) constant while increasing lengths.

    {
      auto it = doc.FindMember("simulation");
      if (it != doc.MemberEnd()) {
        auto &sim = it->value;
        if (sim.IsObject()) {
          fmt::print(fmt::emphasis::bold,
              "-----------------------------------------------------------"
              "-----\n");
          froude_scaling = CheckDouble(sim, "froude_scaling", 1.0);

          PREC sim_default_dx = CheckDouble(sim, "default_dx", 0.1) * froude_scaling;
          double sim_default_dt = CheckDouble(sim, "default_dt", sim_default_dx/100.) * sqrt(froude_scaling);
          uint64_t sim_fps = CheckUint64(sim, "fps", 60);
          uint64_t sim_frames = ceil(CheckUint64(sim, "frames", 60) * sqrt(froude_scaling));
          uint64_t sim_info_rate = CheckInt(sim, "info_rate", mn::config::g_info_rate); // TODO : Implement
          mn::pvec3 sim_gravity = CheckDoubleArray(sim, "gravity", mn::pvec3{0.,-9.81,0.});
          domain = CheckDoubleArray(sim, "domain", mn::pvec3{1.,1.,1.});
          for (int d=0; d<3 ; ++d) domain[d] *= froude_scaling;

          time[0] = CheckDouble(sim, "time", 0.0) * sqrt(froude_scaling); //< Initial time [sec]
          time[1] = time[0] + (double) sim_frames / (double) sim_fps; //< End time [sec]
          std::string save_suffix = CheckString(sim, "save_suffix", std::string{".bgeo"});
          
          bool particles_output_exterior_only = CheckBool(sim, "particles_output_exterior_only", mn::config::g_particles_output_exterior_only);

          l = sim_default_dx * mn::config::g_dx_inv_d; 
          double lx = l * mn::config::g_grid_ratio_x;
          double ly = l * mn::config::g_grid_ratio_y;
          double lz = l * mn::config::g_grid_ratio_z;
          if (domain[0] > (lx-mn::config::g_bc*mn::config::g_blocksize*sim_default_dx) || domain[1] > (ly-mn::config::g_bc*mn::config::g_blocksize*sim_default_dx) || domain[2] > (lz-mn::config::g_bc*mn::config::g_blocksize*sim_default_dx)) {
            fmt::print(fg(red), "ERROR: Simulation domain[{},{},{}] exceeds max domain length[{}, {}, {}]\n", domain[0], domain[1], domain[2], (lx-mn::config::g_bc*mn::config::g_blocksize*sim_default_dx),(ly-mn::config::g_bc*mn::config::g_blocksize*sim_default_dx),(lz-mn::config::g_bc*mn::config::g_blocksize*sim_default_dx));
            fmt::print(fg(yellow), "TIP: Shrink domain, grow default_dx, and/or increase DOMAIN_BITS (settings.h) and recompile. Press Enter to continue...\n" ); 
            if (mn::config::g_log_level >= 3) getchar();
            std::exit(EXIT_FAILURE);
          } 
          // Only saves memory when reducing X domain size currently, Y and Z only reduce if X is already smaller than they are. Fix later 
          uint64_t domainBlockCnt = static_cast<uint64_t>(std::ceil((domain[0] + mn::config::g_bc * mn::config::g_blocksize * sim_default_dx)  / lx * (mn::config::g_grid_size_x))) * static_cast<uint64_t>(std::ceil( (mn::config::g_grid_size_y))) * static_cast<uint64_t>(std::ceil( (mn::config::g_grid_size_z)));
          double reduction = 100. * ( 1. - domainBlockCnt / (mn::config::g_grid_size_x * mn::config::g_grid_size_y * mn::config::g_grid_size_z));
          domainBlockCnt = (mn::config::g_grid_size_x * mn::config::g_grid_size_y * mn::config::g_grid_size_z); // Force full domain, fix later
          fmt::print(fg(yellow),"Partitions _indexTable data-structure: Saved [{}] percent memory of preallocated partition _indexTable by reducing domainBlockCnt from [{}] to run-time of [{}] using domain input relative to DOMAIN_BITS and default_dx.\n", reduction, mn::config::g_grid_size_x * mn::config::g_grid_size_y * mn::config::g_grid_size_z, domainBlockCnt);
          fmt::print(fg(cyan),
              "Scene simulation parameters: Domain Length [{}], domainBlockCnt [{}], default_dx[{}], default_dt[{}], init_time[{}], fps[{}], frames[{}], gravity[{}, {}, {}], save_suffix[{}], froude_scaling[{}], particles_output_exterior_only[{}]\n", 
              l, domainBlockCnt, sim_default_dx, sim_default_dt, time[0],
              sim_fps, sim_frames, sim_gravity[0], sim_gravity[1], sim_gravity[2], save_suffix, froude_scaling, particles_output_exterior_only);
          benchmark = std::make_unique<mn::mgsp_benchmark>(
              l, domainBlockCnt, sim_default_dt, time[0],
              sim_fps, sim_frames, sim_gravity, froude_scaling, save_suffix, particles_output_exterior_only); //< Initialize simulation object
          fmt::print(fmt::emphasis::bold,
              "-----------------------------------------------------------"
              "-----\n");
        }
      }
    } ///< End basic simulation scene parsing
    {
      auto it = doc.FindMember("meshes");
      if (it != doc.MemberEnd()) {
        if (it->value.IsArray()) {
          fmt::print(fg(cyan),"Scene file has [{}] Finite Element meshes.\n", it->value.Size());
          for (auto &model : it->value.GetArray()) {
            if (model["gpu"].GetInt() >= mn::config::g_device_cnt) {
              fmt::print(fg(red),
                       "ERROR! Mesh model GPU[{}] exceeds global device count (settings.h)! Skipping mesh.\n", 
                       model["gpu"].GetInt());
              continue;
            }
            std::string constitutive{model["constitutive"].GetString()};
            fmt::print(fg(green),
                       "Mesh model using constitutive[{}], file_elements[{}], file_vertices[{}].\n", constitutive,
                       model["file_elements"].GetString(), model["file_vertices"].GetString());
            fs::path p{model["file"].GetString()};
            
            std::vector<std::string> output_attribs;
            for (int d = 0; d < model["output_attribs"].GetArray().Size(); ++d) output_attribs.emplace_back(model["output_attribs"].GetArray()[d].GetString());
            std::cout <<"Output Attributes: [ " << output_attribs[0] << ", " << output_attribs[1] << ", " << output_attribs[2] << " ]"<<'\n';
            
            // TODO: Fix particle tracker for FEM
            std::vector<std::string> track_attribs;
            track_attribs = CheckStringArray(model, "track_attribs", std::vector<std::string>{{"Position_Y"}});
            std::vector<int> track_particle_ids;
            track_particle_ids = CheckIntArray(model, "track_particle_id", std::vector<int>{0});


            mn::config::AlgoConfigs algoConfigs;
            algoConfigs.use_FEM = CheckBool(model,std::string{"use_FEM"}, false);
            algoConfigs.use_ASFLIP = CheckBool(model, std::string{"use_ASFLIP"}, true);
            algoConfigs.ASFLIP_alpha = CheckDouble(model, std::string{"alpha"}, 0.);
            algoConfigs.ASFLIP_beta_min = CheckDouble(model, std::string{"beta_min"}, 0.);
            algoConfigs.ASFLIP_beta_max = CheckDouble(model, std::string{"beta_max"}, 0.);
            algoConfigs.use_FBAR = CheckBool(model, std::string{"use_FBAR"}, true);
            algoConfigs.FBAR_ratio = CheckDouble(model, std::string{"FBAR_ratio"}, 0.);
            algoConfigs.FBAR_fused_kernel = CheckBool(model, "FBAR_fused_kernel", false); 


            mn::config::MaterialConfigs materialConfigs;
            materialConfigs.ppc = model["ppc"].GetDouble();
            materialConfigs.rho = model["rho"].GetDouble();

            auto initModel = [&](auto &positions, auto &velocity, auto &vertices, auto &elements, auto &attribs)
            {
              if (constitutive == "Meshed") 
              {
                materialConfigs.E = model["youngs_modulus"].GetDouble(); 
                materialConfigs.nu = model["poisson_ratio"].GetDouble();

                // Initialize FEM model on GPU arrays
                if (model["use_FBAR"].GetBool() == true)
                {
                  std::cout << "Initialize FEM FBAR Model." << '\n';
                  benchmark->initFEM<mn::fem_e::Tetrahedron_FBar>(model["gpu"].GetInt(), vertices, elements, attribs);
                  benchmark->initModel<mn::material_e::Meshed>(model["gpu"].GetInt(), 0, positions, velocity, track_particle_ids, track_attribs); //< Initalize particle model

                  std::cout << "Initialize Mesh Parameters." << '\n';
                  benchmark->updateMeshedFBARParameters(
                    model["gpu"].GetInt(),
                    materialConfigs, algoConfigs,
                    output_attribs); //< Update particle material with run-time inputs
                  fmt::print(fg(green),"GPU[{}] Particle material[{}] model updated.\n", model["gpu"].GetInt(), constitutive);
                }
                else if (model["use_FBAR"].GetBool() == false)
                {
                  std::cout << "Initialize FEM Model." << '\n';
                  benchmark->initFEM<mn::fem_e::Tetrahedron>(model["gpu"].GetInt(), vertices, elements, attribs);
                  benchmark->initModel<mn::material_e::Meshed>(model["gpu"].GetInt(), 0, positions, velocity, track_particle_ids, track_attribs); //< Initalize particle model

                  std::cout << "Initialize Mesh Parameters." << '\n';
                  benchmark->updateMeshedParameters(
                    model["gpu"].GetInt(),
                    materialConfigs, algoConfigs,
                    output_attribs); //< Update particle material with run-time inputs
                  fmt::print(fg(green),"GPU[{}] Mesh material[{}] model updated.\n", model["gpu"].GetInt(), constitutive);
                }
                else 
                {
                  fmt::print(fg(red),
                       "ERROR: GPU[{}] Improper/undefined settings for material [{}] with: use_ASFLIP[{}], use_FEM[{}], and use_FBAR[{}]! \n", 
                       model["gpu"].GetInt(), constitutive,
                       model["use_ASFLIP"].GetBool(), model["use_FEM"].GetBool(), model["use_FBAR"].GetBool());
                  fmt::print(fg(red), "Press Enter to continue...");
                  if (mn::config::g_log_level >= 3) getchar();
                }
              }
              else 
              {
                fmt::print(fg(red),
                      "ERROR: GPU[{}] No material [{}] implemented for finite element meshes! \n", 
                      model["gpu"].GetInt(), constitutive);
                fmt::print(fg(red), "Press Enter to continue...");
                if (mn::config::g_log_level >= 3) getchar();
              }
            };
            
            ElementHolder h_FEM_Elements; //< Declare Host elements
            VerticeHolder h_FEM_Vertices; //< Declare Host vertices

            mn::vec<PREC, 3> offset, velocity;
            for (int d = 0; d < 3; ++d) {
              offset[d]   = model["offset"].GetArray()[d].GetDouble() / l + o;
              velocity[d] = model["velocity"].GetArray()[d].GetDouble() / l; 
            }


            // * NOTE : Assumes geometry "file" specified by scene.json is in AssetDirPath/, e.g. ~/claymore/Data/file
            // std::string elements_fn = std::string(AssetDirPath) + model["file_elements"].GetString();
            // std::string vertices_fn = std::string(AssetDirPath) + model["file_vertices"].GetString();
            std::string elements_fn = std::string("./") + model["file_elements"].GetString();
            std::string vertices_fn = std::string("./") + model["file_vertices"].GetString();

            fmt::print(fg(cyan),"GPU[{}] Load FEM elements file[{}]...", model["gpu"].GetInt(),  elements_fn);
            load_FEM_Elements(elements_fn, ',', h_FEM_Elements);
            
            std::vector<std::array<PREC, 6>> h_FEM_Element_Attribs(mn::config::g_max_fem_element_num, 
                                                            std::array<PREC, 6>{0., 0., 0., 0., 0., 0.});
            
            fmt::print(fg(cyan),"GPU[{}] Load FEM vertices file[{}]...", model["gpu"].GetInt(),  vertices_fn);
            load_FEM_Vertices(vertices_fn, ',', h_FEM_Vertices,
                              offset);
            fmt::print(fg(cyan),"GPU[{}] Load FEM-MPM particle file[{}]...", model["gpu"].GetInt(),  vertices_fn);
            load_csv_particles(model["file_vertices"].GetString(), ',', 
                                models[model["gpu"].GetInt()], 
                                offset);

            // Initialize particle and finite element model
            initModel(models[model["gpu"].GetInt()], velocity, h_FEM_Vertices, h_FEM_Elements, h_FEM_Element_Attribs);
            fmt::print(fmt::emphasis::bold,
                      "-----------------------------------------------------------"
                      "-----\n");
          }
        }
      }
    } ///< end mesh parsing
    {
      auto it = doc.FindMember("bodies"); //< Used to be "models", updated for HydroUQ unified interface
      int rank = 0; 
      // int num_ranks = 1;
#if CLUSTER_COMM_STYLE == 1
        // Obtain our rank (Node ID) and the total number of ranks (Num Nodes)
        // Assume we launch MPI with one rank per GPU Node
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);
        fmt::print(fg(fmt::color::cyan),"MPI Rank {} of {}.\n", rank, num_ranks);
#endif
      if (it != doc.MemberEnd()) {
        if (it->value.IsArray()) {
          fmt::print(fg(cyan), "Scene file has [{}] material bodies. \n", it->value.Size());
          for (auto &model : it->value.GetArray()) {
            int gpu_id = CheckInt(model, "gpu", 0);
            if (gpu_id >= rank * mn::config::g_device_cnt && gpu_id < (rank + 1) * mn::config::g_device_cnt)
              gpu_id = gpu_id % mn::config::g_device_cnt;
            int model_id = CheckInt(model, "model", 0);
            int total_id = model_id + gpu_id * mn::config::g_models_per_gpu;
            int node_id = rank;
            fmt::print(fg(cyan), "NODE[{}] GPU[{}] MODEL[{}] Begin reading...\n", node_id, gpu_id, model_id);
            if (gpu_id >= mn::config::g_device_cnt) {
              fmt::print(fg(red), "ERROR! Particle model[{}] on gpu[{}] exceeds GPUs reserved by g_device_cnt[{}] (settings.h)! Skipping model. Increase g_device_cnt and recompile. \n", model_id, gpu_id, mn::config::g_device_cnt);
              continue;
            } else if (gpu_id < 0) {
              fmt::print(fg(red), "ERROR! GPU[{}] MODEL[{}] GPU ID cannot be negative. \n", gpu_id, model_id);
              if (mn::config::g_log_level >= 3) { fmt::print(fg(red), "Press ENTER to continue..."); getchar(); } continue;
            } 
            if (model_id >= mn::config::g_models_per_gpu) {
              fmt::print(fg(red), "ERROR! Particle model[{}] on gpu[{}] exceeds models reserved by g_models_per_gpu[{}] (settings.h)! Skipping model. Increase g_models_per_gpu and recompile. \n", model_id, gpu_id, mn::config::g_models_per_gpu);
              continue;
            } else if (model_id < 0) {
              fmt::print(fg(red), "ERROR! GPU[{}] MODEL[{}] Model ID cannot be negative. \n", gpu_id, model_id);
              if (mn::config::g_log_level >= 3) { fmt::print(fg(red), "Press ENTER to continue..."); getchar(); } continue;
            }

            const bool use_HydroUQ_interface = true; //< temporary flag to switch between new HydroUQ and ClaymoreUW legacy user-interface

            std::vector<std::string> output_attribs;
            std::vector<std::string> input_attribs;
            bool has_attributes = false;
            std::vector<std::vector<PREC>> attributes; //< Initial attributes (not incl. position)
            std::vector<std::string> target_attribs;
            std::vector<std::string> track_attribs;
            std::vector<int> track_particle_ids;

            mn::config::AlgoConfigs algoConfigs;
            mn::config::MaterialConfigs materialConfigs;

            if (use_HydroUQ_interface) { 
              auto findAlgo = model.FindMember("algorithm");
              if (findAlgo != model.MemberEnd()) {
                auto &algo = findAlgo->value;
                if (algo.IsObject()) {
                  fmt::print(fmt::emphasis::bold,
                      "-----------------------------------------------------------"
                      "-----\n");
                  materialConfigs.ppc = CheckDouble(algo, "ppc", 8.0); // particles per cell , TODO: Move into bodies:algorithm / numerical representation object
                  // algoConfigs.ppc = CheckDouble(algo, "ppc", 8.0); // particles per cell 
                  // algoConfigs.type = CheckString(algo, "type", std::string{"particles"});
                  algoConfigs.use_FEM = CheckBool(algo, "use_FEM", false);
                  algoConfigs.use_ASFLIP = CheckBool(algo, "use_ASFLIP", true);
                  algoConfigs.ASFLIP_alpha = CheckDouble(algo, "ASFLIP_alpha", 0.);
                  algoConfigs.ASFLIP_beta_min = CheckDouble(algo, "ASFLIP_beta_min", 0.);
                  algoConfigs.ASFLIP_beta_max = CheckDouble(algo, "ASFLIP_beta_max", 0.);
                  algoConfigs.use_FBAR = CheckBool(algo, "use_FBAR", true);
                  algoConfigs.FBAR_ratio = CheckDouble(algo, "FBAR_psi", 0.);
                  // algoConfigs.FBAR_ratio = CheckDouble(algo, "FBAR_ratio", 0.);
                  algoConfigs.FBAR_fused_kernel = CheckBool(algo, "FBAR_fused_kernel", false);
                  // algoConfigs.ASFLIP_alpha = CheckDouble(algo, "ASFLIP_velocity_ratio", 0.);
                  // algoConfigs.ASFLIP_beta_min = CheckDouble(algo, "ASFLIP_position_ratio_min", 0.);
                  // algoConfigs.ASFLIP_beta_max = CheckDouble(algo, "ASFLIP_position_ratio_max", 0.);
                  // algoConfigs.FBAR_ratio = CheckDouble(algo, "FBAR_antilocking_ratio", 0.);
                  fmt::print(fg(green), "GPU[{}] Read algorithm parameters.\n", gpu_id);
                  
                }
              }
            }
            else {
              algoConfigs.use_FEM = CheckBool(model, "use_FEM", false);
              algoConfigs.use_ASFLIP = CheckBool(model, "use_ASFLIP", true);
              algoConfigs.ASFLIP_alpha = CheckDouble(model, "alpha", 0.);
              algoConfigs.ASFLIP_beta_min = CheckDouble(model, "beta_min", 0.);
              algoConfigs.ASFLIP_beta_max = CheckDouble(model, "beta_max", 0.);
              algoConfigs.use_FBAR = CheckBool(model, "use_FBAR", true);
              algoConfigs.FBAR_ratio = CheckDouble(model, "FBAR_ratio", 0.); 
              algoConfigs.FBAR_fused_kernel = CheckBool(model, "FBAR_fused_kernel", false); 
            }

            std::string constitutive; // Material's constitutive law name, bodies:material:constitutive
            if (use_HydroUQ_interface) { 
              // Begin bodies:material
              auto findMat = model.FindMember("material");
              if (findMat != model.MemberEnd()) {
                auto &mat = findMat->value;
                if (mat.IsObject()) {
                  fmt::print(fmt::emphasis::bold,
                      "-----------------------------------------------------------"
                      "-----\n");
                  constitutive = CheckString(mat, "constitutive", std::string{"JFluid"});
                  fmt::print(fg(green), "GPU[{}] Read model constitutive[{}].\n", gpu_id, constitutive);

                  // Basic material properties
                  // materialConfigs.ppc = CheckDouble(mat, "ppc", 8.0); // particles per cell , TODO: Move into bodies:algorithm / numerical representation object
                  materialConfigs.rho = CheckDouble(mat, "rho", 1e3); // density
                  materialConfigs.CFL = CheckDouble(mat, "CFL", 0.5); // CFL number

                  if (constitutive == "JFluid" || constitutive == "J-Fluid" || constitutive == "J_Fluid" || constitutive == "J Fluid" ||  constitutive == "jfluid" || constitutive == "j-fluid" || constitutive == "j_fluid" || constitutive == "j fluid" || constitutive == "Fluid" || constitutive == "fluid" || constitutive == "Water" || constitutive == "Liquid") {
                    materialConfigs.bulk = CheckDouble(mat, "bulk_modulus", 2e7); // bulk modulus
                    materialConfigs.gamma = CheckDouble(mat, "gamma", 7.1); // deriv bulk wrt pressure
                    materialConfigs.visco = CheckDouble(mat, "viscosity", 0.001); // viscosity
                  }  
                  else if (constitutive == "FixedCorotated" || constitutive == "Fixed_Corotated" || constitutive == "Fixed-Corotated" || constitutive == "Fixed Corotated" || constitutive == "fixedcorotated" || constitutive == "fixed_corotated" || constitutive == "fixed-corotated"|| constitutive == "fixed corotated") {
                    materialConfigs.E = CheckDouble(mat, "youngs_modulus", 1e7);
                    materialConfigs.nu = CheckDouble(mat, "poisson_ratio", 0.2);
                    if (materialConfigs.nu >= 0.5) { 
                      materialConfigs.nu = 0.4999; 
                      fmt::print(fg(red), "GPU[{}] MODEL[{}] Poisson ratio[{}] is invalid. Set to 0.4999.\n", gpu_id, model_id, materialConfigs.nu);
                    } else if (materialConfigs.nu <= -0.5) { 
                      materialConfigs.nu = -0.4999; 
                      fmt::print(fg(red), "GPU[{}] MODEL[{}] Poisson ratio[{}] is invalid. Set to -0.4999.\n", gpu_id, model_id, materialConfigs.nu); 
                    }
                  }
                  else if (constitutive == "NeoHookean" || constitutive == "neohookean" || 
                          constitutive == "Neo-Hookean" || constitutive == "neo-hookean") {
                    materialConfigs.E = CheckDouble(mat, "youngs_modulus", 1e7); 
                    materialConfigs.nu = CheckDouble(mat, "poisson_ratio", 0.2);
                    if (materialConfigs.nu >= 0.5) { 
                      materialConfigs.nu = 0.4999; 
                      fmt::print(fg(red), "GPU[{}] MODEL[{}] Poisson ratio[{}] is invalid. Set to 0.4999.\n", gpu_id, model_id, materialConfigs.nu);
                    } else if (materialConfigs.nu <= -0.5) { 
                      materialConfigs.nu = -0.4999; 
                      fmt::print(fg(red), "GPU[{}] MODEL[{}] Poisson ratio[{}] is invalid. Set to -0.4999.\n", gpu_id, model_id, materialConfigs.nu); 
                    }
                  }
                  else if (constitutive == "Sand" || constitutive == "sand" || constitutive == "DruckerPrager" || constitutive == "Drucker_Prager" || constitutive == "Drucker-Prager" || constitutive == "Drucker Prager") { 
                    materialConfigs.E = CheckDouble(mat, "youngs_modulus", 1e7); 
                    materialConfigs.nu = CheckDouble(mat, "poisson_ratio", 0.2);
                    materialConfigs.logJp0 = CheckDouble(mat, "logJp0", 0.0); // TODO: Rename
                    materialConfigs.frictionAngle = CheckDouble(mat, "friction_angle", 30.0);
                    materialConfigs.cohesion = CheckDouble(mat, "cohesion", 0.0); // TODO: Rename
                    materialConfigs.beta = CheckDouble(mat, "beta", 0.5); // TODO: Rename
                    materialConfigs.volumeCorrection = CheckBool(mat, "SandVolCorrection", true); 
                    // materialConfigs.dilationAngle = CheckDouble(moat "dilation_angle", 30.0);
                    if (materialConfigs.nu >= 0.5) { 
                      materialConfigs.nu = 0.4999; 
                      fmt::print(fg(red), "GPU[{}] MODEL[{}] Poisson ratio[{}] is invalid. Set to 0.4999.\n", gpu_id, model_id, materialConfigs.nu);
                    } else if (materialConfigs.nu <= -0.5) { 
                      materialConfigs.nu = -0.4999; 
                      fmt::print(fg(red), "GPU[{}] MODEL[{}] Poisson ratio[{}] is invalid. Set to -0.4999.\n", gpu_id, model_id, materialConfigs.nu); 
                    }
                  }
                  else if (constitutive == "CoupledUP" || constitutive == "coupled" || constitutive == "up" || constitutive == "UP" || constitutive == "coupledup" || constitutive == "undrained") { 
                    materialConfigs.E = CheckDouble(mat, "youngs_modulus", 1e7); 
                    materialConfigs.nu = CheckDouble(mat, "poisson_ratio", 0.2);
                    materialConfigs.logJp0 = CheckDouble(mat, "logJp0", 0.0);
                    materialConfigs.frictionAngle = CheckDouble(mat, "friction_angle", 30.0);
                    materialConfigs.cohesion = CheckDouble(mat, "cohesion", 0.0);
                    materialConfigs.beta = CheckDouble(mat, "beta", 0.5);
                    materialConfigs.volumeCorrection = CheckBool(mat, "SandVolCorrection", true); 
                    materialConfigs.rhow = CheckDouble(mat, "rhow", 1e3); // Density of water [kg/m3]
                    materialConfigs.alpha1 = CheckDouble(mat, "alpha1", 1.0); // Biot coefficient
                    materialConfigs.poro = CheckDouble(mat, "poro", 0.9); // Porosity
                    materialConfigs.Kf = CheckDouble(mat, "Kf", 1.0e7); // Bulk modulus of fluid [Pa]
                    materialConfigs.Ks = CheckDouble(mat, "Ks", 2.2e7); // Bulk modulus of solid [Pa]
                    materialConfigs.Kperm = CheckDouble(mat, "Kperm", 1.0e-5); // Permeability
                    if (materialConfigs.nu >= 0.5) { 
                      materialConfigs.nu = 0.4999; 
                      fmt::print(fg(red), "GPU[{}] MODEL[{}] Poisson ratio[{}] is invalid. Set to 0.4999.\n", gpu_id, model_id, materialConfigs.nu);
                    } else if (materialConfigs.nu <= -0.5) { 
                      materialConfigs.nu = -0.4999; 
                      fmt::print(fg(red), "GPU[{}] MODEL[{}] Poisson ratio[{}] is invalid. Set to -0.4999.\n", gpu_id, model_id, materialConfigs.nu); 
                    }
                  }
                  else if (constitutive == "NACC" || constitutive == "nacc" || constitutive == "CamClay" || constitutive == "Cam_Clay" || constitutive == "Cam-Clay" || constitutive == "Cam Clay") {
                    materialConfigs.E = CheckDouble(mat, "youngs_modulus", 1e7); 
                    materialConfigs.nu = CheckDouble(mat, "poisson_ratio", 0.2);
                    materialConfigs.logJp0 = CheckDouble(mat, "logJp0", 0.0);
                    materialConfigs.frictionAngle = CheckDouble(mat, "friction_angle", 30.0);
                    materialConfigs.xi = CheckDouble(mat, "xi", 0.8);
                    materialConfigs.beta = CheckDouble(mat, "beta", 0.5);
                    materialConfigs.hardeningOn = CheckBool(mat, "hardeningOn", true); 
                    // materialConfigs.Msqr = CheckDouble(mat, "Msqr", 0.0); // mohr_friction squared
                    // materialConfigs.hardeningRatio = CheckDouble(mat, "hardening_ratio", 0.0);
                    // materialConfigs.cohesionRatio = CheckDouble(mat, "cohesion_ratio", 0.0);
                    // materialConfigs.mohrFriction = CheckDouble(mat, "mohr_friction", 0.0);
                    if (materialConfigs.nu >= 0.5) { 
                      materialConfigs.nu = 0.4999; 
                      fmt::print(fg(red), "GPU[{}] MODEL[{}] Poisson ratio[{}] is invalid. Set to 0.4999.\n", gpu_id, model_id, materialConfigs.nu);
                    } else if (materialConfigs.nu <= -0.5) { 
                      materialConfigs.nu = -0.4999; 
                      fmt::print(fg(red), "GPU[{}] MODEL[{}] Poisson ratio[{}] is invalid. Set to -0.4999.\n", gpu_id, model_id, materialConfigs.nu); 
                    }
                  }
                  else if (constitutive == "VonMises" || constitutive == "Von_Mises" || constitutive == "Von-Mises" || constitutive == "Von Mises" || constitutive == "vonmises" || constitutive == "von_mises" || constitutive == "von-mises"|| constitutive == "von mises") {
                    materialConfigs.E = CheckDouble(mat, "youngs_modulus", 1e7);
                    materialConfigs.nu = CheckDouble(mat, "poisson_ratio", 0.2);
                    materialConfigs.tensile_yield_strength = CheckDouble(mat, "tensile_yield_strength", 2.0e6);
                    materialConfigs.hardening_modulus = CheckDouble(mat, "hardening_modulus", 1.0e6);
                    if (materialConfigs.nu >= 0.5) { 
                      materialConfigs.nu = 0.4999; 
                      fmt::print(fg(red), "GPU[{}] MODEL[{}] Poisson ratio[{}] is invalid. Set to 0.4999.\n", gpu_id, model_id, materialConfigs.nu);
                    } else if (materialConfigs.nu <= -0.5) { 
                      materialConfigs.nu = -0.4999; 
                      fmt::print(fg(red), "GPU[{}] MODEL[{}] Poisson ratio[{}] is invalid. Set to -0.4999.\n", gpu_id, model_id, materialConfigs.nu); 
                    }
                    if (materialConfigs.tensile_yield_strength <= 0.0) {
                      materialConfigs.tensile_yield_strength = 1e-6;
                      fmt::print(fg(red), "GPU[{}] MODEL[{}] Tensile yield strength[{}] is invalid. Set to 1e-6.\n", gpu_id, model_id, materialConfigs.tensile_yield_strength);
                    }
                  }
                }
              }
            } // end bodies:material
            else {
              constitutive = CheckString(model, "constitutive", std::string{"JFluid"});
              fmt::print(fg(green), "GPU[{}] Read model constitutive[{}].\n", gpu_id, constitutive);

              materialConfigs.ppc = CheckDouble(model, "ppc", 8.0); // particles per cell
              materialConfigs.rho = CheckDouble(model, "rho", 1e3); // density
              materialConfigs.CFL = CheckDouble(model, "CFL", 0.5); // CFL number
            }

            auto initModel = [&](auto &positions, auto &velocity) {
              bool algo_error = false, mat_error  = false;
              if (constitutive == "JFluid" || constitutive == "J-Fluid" || constitutive == "J_Fluid" || constitutive == "J Fluid" ||  constitutive == "jfluid" || constitutive == "j-fluid" || constitutive == "j_fluid" || constitutive == "j fluid" || constitutive == "Fluid" || constitutive == "fluid" || constitutive == "Water" || constitutive == "Liquid") {
                if (use_HydroUQ_interface == false) {
                  materialConfigs.bulk = CheckDouble(model, "bulk_modulus", 2e7); // bulk modulus
                  materialConfigs.gamma = CheckDouble(model, "gamma", 7.1); // deriv bulk wrt pressure
                  materialConfigs.visco = CheckDouble(model, "viscosity", 0.001); // viscosity
                }
                // Update time-step for material properties: dt = dx / v_pwave * CFL
                PREC pwave_velocity = std::sqrt(materialConfigs.bulk / materialConfigs.rho);
                PREC max_dt = (l / mn::config::g_dx_inv_d) / pwave_velocity * materialConfigs.CFL;
                benchmark->set_time_step(max_dt); //< Set time-step
                fmt::print(fg(yellow), "GPU[{}] MODEL[{}] Max time-step[{}] of material.\n", gpu_id, model_id, max_dt);

                if(!algoConfigs.use_ASFLIP && !algoConfigs.use_FBAR && !algoConfigs.use_FEM)
                {
                  benchmark->initModel<mn::material_e::JFluid>(gpu_id, model_id, positions, velocity, track_particle_ids, track_attribs);                    
                  benchmark->updateParameters<mn::material_e::JFluid>( 
                      gpu_id, model_id, materialConfigs, algoConfigs,
                      output_attribs, track_particle_ids, track_attribs, target_attribs);
                }
                else if (algoConfigs.use_ASFLIP && !algoConfigs.use_FBAR && !algoConfigs.use_FEM)
                {
                  benchmark->initModel<mn::material_e::JFluid_ASFLIP>(gpu_id, model_id, positions, velocity, track_particle_ids, track_attribs);                    
                  benchmark->updateParameters<mn::material_e::JFluid_ASFLIP>( 
                      gpu_id, model_id, materialConfigs, algoConfigs,
                      output_attribs, track_particle_ids, track_attribs, target_attribs);
                }
                else if (algoConfigs.use_ASFLIP && algoConfigs.use_FBAR && !algoConfigs.use_FEM)
                {
                  benchmark->initModel<mn::material_e::JBarFluid>(gpu_id, model_id, positions, velocity, track_particle_ids, track_attribs);
                  benchmark->updateParameters<mn::material_e::JBarFluid>( 
                      gpu_id, model_id, materialConfigs, algoConfigs,
                      output_attribs, track_particle_ids, track_attribs, target_attribs);
                } 
                else if (!algoConfigs.use_ASFLIP && algoConfigs.use_FBAR && !algoConfigs.use_FEM)
                {
                  benchmark->initModel<mn::material_e::JFluid_FBAR>(gpu_id, model_id, positions, velocity, track_particle_ids, track_attribs);
                  benchmark->updateParameters<mn::material_e::JFluid_FBAR>( 
                      gpu_id, model_id, materialConfigs, algoConfigs,
                      output_attribs, track_particle_ids, track_attribs, target_attribs);
                }
                else { algo_error = true; }
              } 
              else if (constitutive == "FixedCorotated" || constitutive == "Fixed_Corotated" || constitutive == "Fixed-Corotated" || constitutive == "Fixed Corotated" || constitutive == "fixedcorotated" || constitutive == "fixed_corotated" || constitutive == "fixed-corotated"|| constitutive == "fixed corotated") {
                if (use_HydroUQ_interface == false) {
                  materialConfigs.E = CheckDouble(model, "youngs_modulus", 1e7);
                  materialConfigs.nu = CheckDouble(model, "poisson_ratio", 0.2);
                  if (materialConfigs.nu >= 0.5) { 
                    materialConfigs.nu = 0.4999; 
                    fmt::print(fg(red), "GPU[{}] MODEL[{}] Poisson ratio[{}] is invalid. Set to 0.4999.\n", gpu_id, model_id, materialConfigs.nu);
                  } else if (materialConfigs.nu <= -0.5) { 
                    materialConfigs.nu = -0.4999; 
                    fmt::print(fg(red), "GPU[{}] MODEL[{}] Poisson ratio[{}] is invalid. Set to -0.4999.\n", gpu_id, model_id, materialConfigs.nu); 
                  }
                }
                // Update time-step for material properties: dt = dx / v_pwave * CFL
                PREC pwave_velocity = std::sqrt((materialConfigs.E / (3.0 * (1.0 - 2.0 * materialConfigs.nu))) / materialConfigs.rho);
                PREC max_dt = (l / mn::config::g_dx_inv_d) / pwave_velocity * materialConfigs.CFL;
                benchmark->set_time_step(max_dt); //< Set time-step
                fmt::print(fg(yellow), "GPU[{}] MODEL[{}] Max time-step[{}] of material.\n", gpu_id, model_id, max_dt);

                if(!algoConfigs.use_ASFLIP && !algoConfigs.use_FBAR && !algoConfigs.use_FEM)
                {
                  benchmark->initModel<mn::material_e::FixedCorotated>(gpu_id, model_id, positions, velocity, track_particle_ids, track_attribs);
                  benchmark->updateParameters<mn::material_e::FixedCorotated>(
                      gpu_id, model_id, materialConfigs, algoConfigs,
                      output_attribs, track_particle_ids, track_attribs, target_attribs);
                }
                else if (algoConfigs.use_ASFLIP && !algoConfigs.use_FBAR && !algoConfigs.use_FEM)
                {
                  benchmark->initModel<mn::material_e::FixedCorotated_ASFLIP>(gpu_id, model_id, positions, velocity, track_particle_ids, track_attribs);
                  benchmark->updateParameters<mn::material_e::FixedCorotated_ASFLIP>(
                      gpu_id, model_id, materialConfigs, algoConfigs,
                      output_attribs, track_particle_ids, track_attribs, target_attribs);
                }
                else if (!algoConfigs.use_ASFLIP && algoConfigs.use_FBAR && !algoConfigs.use_FEM)
                {
                  benchmark->initModel<mn::material_e::FixedCorotated_FBAR>(gpu_id, model_id, positions, velocity, track_particle_ids, track_attribs);
                  benchmark->updateParameters<mn::material_e::FixedCorotated_FBAR>(
                      gpu_id, model_id, materialConfigs, algoConfigs,
                      output_attribs, track_particle_ids, track_attribs, target_attribs);
                }
                else if (algoConfigs.use_ASFLIP && algoConfigs.use_FBAR && !algoConfigs.use_FEM)
                {
                  benchmark->initModel<mn::material_e::FixedCorotated_ASFLIP_FBAR>(gpu_id, model_id, positions, velocity, track_particle_ids, track_attribs);
                  benchmark->updateParameters<mn::material_e::FixedCorotated_ASFLIP_FBAR>(
                      gpu_id, model_id, materialConfigs, algoConfigs,
                      output_attribs, track_particle_ids, track_attribs, target_attribs);
                }
                else { algo_error = true; }
              } 
              else if (constitutive == "VonMises" || constitutive == "Von_Mises" || constitutive == "Von-Mises" || constitutive == "Von Mises" || constitutive == "vonmises" || constitutive == "von_mises" || constitutive == "von-mises"|| constitutive == "von mises") {
                if (use_HydroUQ_interface == false) {
                  materialConfigs.E = CheckDouble(model, "youngs_modulus", 1e7);
                  materialConfigs.nu = CheckDouble(model, "poisson_ratio", 0.2);
                  materialConfigs.tensile_yield_strength = CheckDouble(model, "tensile_yield_strength", 2.0e6);
                  materialConfigs.hardening_modulus = CheckDouble(model, "hardening_modulus", 1.0e6);
                  if (materialConfigs.nu >= 0.5) { 
                    materialConfigs.nu = 0.4999; 
                    fmt::print(fg(red), "GPU[{}] MODEL[{}] Poisson ratio[{}] is invalid. Set to 0.4999.\n", gpu_id, model_id, materialConfigs.nu);
                  } else if (materialConfigs.nu <= -0.5) { 
                    materialConfigs.nu = -0.4999; 
                    fmt::print(fg(red), "GPU[{}] MODEL[{}] Poisson ratio[{}] is invalid. Set to -0.4999.\n", gpu_id, model_id, materialConfigs.nu); 
                  }
                  if (materialConfigs.tensile_yield_strength <= 0.0) {
                    materialConfigs.tensile_yield_strength = 1e-6;
                    fmt::print(fg(red), "GPU[{}] MODEL[{}] Tensile yield strength[{}] is invalid. Set to 1e-6.\n", gpu_id, model_id, materialConfigs.tensile_yield_strength);
                  }
                }
                // Update time-step for material properties: dt = dx / v_pwave * CFL
                PREC pwave_velocity = std::sqrt((materialConfigs.E / (3.0 * (1.0 - 2.0 * materialConfigs.nu))) / materialConfigs.rho);
                PREC max_dt = (l / mn::config::g_dx_inv_d) / pwave_velocity * materialConfigs.CFL;
                benchmark->set_time_step(max_dt); //< Set time-step
                fmt::print(fg(yellow), "GPU[{}] MODEL[{}] Max time-step[{}] of material.\n", gpu_id, model_id, max_dt);

                if(!algoConfigs.use_ASFLIP && !algoConfigs.use_FBAR && !algoConfigs.use_FEM)
                {
                  benchmark->initModel<mn::material_e::VonMises>(gpu_id, model_id, positions, velocity, track_particle_ids, track_attribs);
                  benchmark->updateParameters<mn::material_e::VonMises>(
                      gpu_id, model_id, materialConfigs, algoConfigs,
                      output_attribs, track_particle_ids, track_attribs, target_attribs);
                }
                else { algo_error = true; }
              } 
              else if (constitutive == "NeoHookean" || constitutive == "neohookean" || 
                      constitutive == "Neo-Hookean" || constitutive == "neo-hookean") {
                if (use_HydroUQ_interface == false) {
                  materialConfigs.E = CheckDouble(model, "youngs_modulus", 1e7); 
                  materialConfigs.nu = CheckDouble(model, "poisson_ratio", 0.2);
                  if (materialConfigs.nu >= 0.5) { 
                    materialConfigs.nu = 0.4999; 
                    fmt::print(fg(red), "GPU[{}] MODEL[{}] Poisson ratio[{}] is invalid. Set to 0.4999.\n", gpu_id, model_id, materialConfigs.nu);
                  } else if (materialConfigs.nu <= -0.5) { 
                    materialConfigs.nu = -0.4999; 
                    fmt::print(fg(red), "GPU[{}] MODEL[{}] Poisson ratio[{}] is invalid. Set to -0.4999.\n", gpu_id, model_id, materialConfigs.nu); 
                  }
                }
                // Update time-step for material properties: dt = dx / v_pwave * CFL
                PREC pwave_velocity = std::sqrt((materialConfigs.E / (3.0 * (1.0 - 2.0 * materialConfigs.nu))) / materialConfigs.rho);
                PREC max_dt = (l / mn::config::g_dx_inv_d) / pwave_velocity * materialConfigs.CFL;
                benchmark->set_time_step(max_dt); //< Set time-step
                fmt::print(fg(yellow), "GPU[{}] MODEL[{}] Max time-step[{}] of material.\n", gpu_id, model_id, max_dt);

                if (algoConfigs.use_ASFLIP && algoConfigs.use_FBAR && !algoConfigs.use_FEM)
                {
                  benchmark->initModel<mn::material_e::NeoHookean_ASFLIP_FBAR>(gpu_id, model_id, positions, velocity, track_particle_ids, track_attribs);
                  benchmark->updateParameters<mn::material_e::NeoHookean_ASFLIP_FBAR>( 
                      gpu_id, model_id, materialConfigs, algoConfigs,
                      output_attribs, track_particle_ids, track_attribs, target_attribs);
                }
                else if (algoConfigs.use_ASFLIP && !algoConfigs.use_FBAR && !algoConfigs.use_FEM)
                {
                  benchmark->initModel<mn::material_e::NeoHookean_ASFLIP>(gpu_id, model_id, positions, velocity, track_particle_ids, track_attribs);
                  benchmark->updateParameters<mn::material_e::NeoHookean_ASFLIP>( 
                      gpu_id, model_id, materialConfigs, algoConfigs,
                      output_attribs, track_particle_ids, track_attribs, target_attribs);
                }
                else if (!algoConfigs.use_ASFLIP && algoConfigs.use_FBAR && !algoConfigs.use_FEM)
                {
                  benchmark->initModel<mn::material_e::NeoHookean_FBAR>(gpu_id, model_id, positions, velocity, track_particle_ids, track_attribs);
                  benchmark->updateParameters<mn::material_e::NeoHookean_FBAR>( 
                      gpu_id, model_id, materialConfigs, algoConfigs,
                      output_attribs, track_particle_ids, track_attribs, target_attribs);
                }
                else if (!algoConfigs.use_ASFLIP && !algoConfigs.use_FBAR && !algoConfigs.use_FEM)
                {
                  benchmark->initModel<mn::material_e::NeoHookean>(gpu_id, model_id, positions, velocity, track_particle_ids, track_attribs);
                  benchmark->updateParameters<mn::material_e::NeoHookean>( 
                      gpu_id, model_id, materialConfigs, algoConfigs,
                      output_attribs, track_particle_ids, track_attribs, target_attribs);
                }
                else { algo_error = true; }
              } 
              else if (constitutive == "Sand" || constitutive == "sand" || constitutive == "DruckerPrager" || constitutive == "Drucker_Prager" || constitutive == "Drucker-Prager" || constitutive == "Drucker Prager") { 
                if (use_HydroUQ_interface == false) {
                  materialConfigs.E = CheckDouble(model, "youngs_modulus", 1e7); 
                  materialConfigs.nu = CheckDouble(model, "poisson_ratio", 0.2);
                  materialConfigs.logJp0 = CheckDouble(model, "logJp0", 0.0);
                  materialConfigs.frictionAngle = CheckDouble(model, "friction_angle", 30.0);
                  materialConfigs.cohesion = CheckDouble(model, "cohesion", 0.0);
                  materialConfigs.beta = CheckDouble(model, "beta", 0.5);
                  materialConfigs.volumeCorrection = CheckBool(model, "SandVolCorrection", true); 
                  if (materialConfigs.nu >= 0.5) { 
                    materialConfigs.nu = 0.4999; 
                    fmt::print(fg(red), "GPU[{}] MODEL[{}] Poisson ratio[{}] is invalid. Set to 0.4999.\n", gpu_id, model_id, materialConfigs.nu);
                  } else if (materialConfigs.nu <= -0.5) { 
                    materialConfigs.nu = -0.4999; 
                    fmt::print(fg(red), "GPU[{}] MODEL[{}] Poisson ratio[{}] is invalid. Set to -0.4999.\n", gpu_id, model_id, materialConfigs.nu); 
                  }
                }
                
                // Update time-step for material properties: dt = dx / v_pwave * CFL
                PREC pwave_velocity = std::sqrt((materialConfigs.E / (3.0 * (1.0 - 2.0 * materialConfigs.nu))) / materialConfigs.rho);
                PREC max_dt = (l / mn::config::g_dx_inv_d) / pwave_velocity * materialConfigs.CFL;
                benchmark->set_time_step(max_dt); //< Set time-step
                fmt::print(fg(yellow), "GPU[{}] MODEL[{}] Max time-step[{}] of material.\n", gpu_id, model_id, max_dt);
                
                if (algoConfigs.use_ASFLIP && algoConfigs.use_FBAR && !algoConfigs.use_FEM)
                {
                  benchmark->initModel<mn::material_e::Sand_ASFLIP_FBAR>(gpu_id, model_id, positions, velocity, track_particle_ids, track_attribs); 
                  benchmark->updateParameters<mn::material_e::Sand_ASFLIP_FBAR>( 
                        gpu_id, model_id, materialConfigs, algoConfigs,
                        output_attribs, track_particle_ids, track_attribs, target_attribs);
                }
                else if (algoConfigs.use_ASFLIP && !algoConfigs.use_FBAR && !algoConfigs.use_FEM)
                {
                  benchmark->initModel<mn::material_e::Sand_ASFLIP>(gpu_id, model_id, positions, velocity, track_particle_ids, track_attribs); 
                  benchmark->updateParameters<mn::material_e::Sand_ASFLIP>( 
                        gpu_id, model_id, materialConfigs, algoConfigs,
                        output_attribs, track_particle_ids, track_attribs, target_attribs);
                }
                else if (!algoConfigs.use_ASFLIP && algoConfigs.use_FBAR && !algoConfigs.use_FEM)
                {
                  benchmark->initModel<mn::material_e::Sand_FBAR>(gpu_id, model_id, positions, velocity, track_particle_ids, track_attribs); 
                  benchmark->updateParameters<mn::material_e::Sand_FBAR>( 
                        gpu_id, model_id, materialConfigs, algoConfigs,
                        output_attribs, track_particle_ids, track_attribs, target_attribs);
                }
                else if (!algoConfigs.use_ASFLIP && !algoConfigs.use_FBAR && !algoConfigs.use_FEM)
                {
                  benchmark->initModel<mn::material_e::Sand>(gpu_id, model_id, positions, velocity, track_particle_ids, track_attribs); 
                  benchmark->updateParameters<mn::material_e::Sand>( 
                        gpu_id, model_id, materialConfigs, algoConfigs,
                        output_attribs, track_particle_ids, track_attribs, target_attribs);
                }
                else { algo_error = true; }
              } 
              else if (constitutive == "CoupledUP" || constitutive == "coupled" || constitutive == "up" || constitutive == "UP" || constitutive == "coupledup") { 
                if (use_HydroUQ_interface == false) {
                  materialConfigs.E = CheckDouble(model, "youngs_modulus", 1e7); 
                  materialConfigs.nu = CheckDouble(model, "poisson_ratio", 0.2);
                  materialConfigs.logJp0 = CheckDouble(model, "logJp0", 0.0);
                  materialConfigs.frictionAngle = CheckDouble(model, "friction_angle", 30.0);
                  materialConfigs.cohesion = CheckDouble(model, "cohesion", 0.0);
                  materialConfigs.beta = CheckDouble(model, "beta", 0.5);
                  materialConfigs.volumeCorrection = CheckBool(model, "SandVolCorrection", true); 
                  materialConfigs.rhow = CheckDouble(model, "rhow", 1e3); // Density of water [kg/m3]
                  materialConfigs.alpha1 = CheckDouble(model, "alpha1", 1.0); // Biot coefficient
                  materialConfigs.poro = CheckDouble(model, "poro", 0.9); // Porosity
                  materialConfigs.Kf = CheckDouble(model, "Kf", 1.0e7); // Bulk modulus of fluid [Pa]
                  materialConfigs.Ks = CheckDouble(model, "Ks", 2.2e7); // Bulk modulus of solid [Pa]
                  materialConfigs.Kperm = CheckDouble(model, "Kperm", 1.0e-5); // Permeability
                  if (materialConfigs.nu >= 0.5) { 
                    materialConfigs.nu = 0.4999; 
                    fmt::print(fg(red), "GPU[{}] MODEL[{}] Poisson ratio[{}] is invalid. Set to 0.4999.\n", gpu_id, model_id, materialConfigs.nu);
                  } else if (materialConfigs.nu <= -0.5) { 
                    materialConfigs.nu = -0.4999; 
                    fmt::print(fg(red), "GPU[{}] MODEL[{}] Poisson ratio[{}] is invalid. Set to -0.4999.\n", gpu_id, model_id, materialConfigs.nu); 
                  }
                }
                
                // ! TODO: Update time-step calculation for coupled model
                // Update time-step for material properties: dt = dx / v_pwave * CFL
                PREC pwave_velocity = std::sqrt((materialConfigs.E / (3.0 * (1.0 - 2.0 * materialConfigs.nu))) / materialConfigs.rho);
                PREC max_dt = (l / mn::config::g_dx_inv_d) / pwave_velocity * materialConfigs.CFL;
                benchmark->set_time_step(max_dt); //< Set time-step
                fmt::print(fg(yellow), "GPU[{}] MODEL[{}] Max time-step[{}] of material.\n", gpu_id, model_id, max_dt);
                
                if (algoConfigs.use_ASFLIP && algoConfigs.use_FBAR && !algoConfigs.use_FEM)
                {
                  benchmark->initModel<mn::material_e::CoupledUP>(gpu_id, model_id, positions, velocity, track_particle_ids, track_attribs); 
                  benchmark->updateParameters<mn::material_e::CoupledUP>( 
                        gpu_id, model_id, materialConfigs, algoConfigs,
                        output_attribs, track_particle_ids, track_attribs, target_attribs);
                }
                else { algo_error = true; }
              } 
              else if (constitutive == "NACC" || constitutive == "nacc" || constitutive == "CamClay" || constitutive == "Cam_Clay" || constitutive == "Cam-Clay" || constitutive == "Cam Clay") {
                if (use_HydroUQ_interface == false) {
                  materialConfigs.E = CheckDouble(model, "youngs_modulus", 1e7); 
                  materialConfigs.nu = CheckDouble(model, "poisson_ratio", 0.2);
                  materialConfigs.logJp0 = CheckDouble(model, "logJp0", 0.0);
                  materialConfigs.xi = CheckDouble(model, "xi", 0.8);
                  materialConfigs.frictionAngle = CheckDouble(model, "friction_angle", 30.0);
                  materialConfigs.beta = CheckDouble(model, "beta", 0.5);
                  materialConfigs.hardeningOn = CheckBool(model, "hardeningOn", true); 
                  if (materialConfigs.nu >= 0.5) { 
                    materialConfigs.nu = 0.4999; 
                    fmt::print(fg(red), "GPU[{}] MODEL[{}] Poisson ratio[{}] is invalid. Set to 0.4999.\n", gpu_id, model_id, materialConfigs.nu);
                  } else if (materialConfigs.nu <= -0.5) { 
                    materialConfigs.nu = -0.4999; 
                    fmt::print(fg(red), "GPU[{}] MODEL[{}] Poisson ratio[{}] is invalid. Set to -0.4999.\n", gpu_id, model_id, materialConfigs.nu); 
                  }
                }
                // Update time-step for material properties: dt = dx / v_pwave * CFL
                PREC pwave_velocity = std::sqrt((materialConfigs.E / (3.0 * (1.0 - 2.0 * materialConfigs.nu))) / materialConfigs.rho);
                PREC max_dt = (l / mn::config::g_dx_inv_d) / pwave_velocity * materialConfigs.CFL;
                benchmark->set_time_step(max_dt); //< Set time-step
                fmt::print(fg(yellow), "GPU[{}] MODEL[{}] Max time-step[{}] of material.\n", gpu_id, model_id, max_dt);
                
                if (algoConfigs.use_ASFLIP && algoConfigs.use_FBAR && !algoConfigs.use_FEM)
                {
                  benchmark->initModel<mn::material_e::NACC_ASFLIP_FBAR>(gpu_id, model_id, positions, velocity, track_particle_ids, track_attribs);
                  benchmark->updateParameters<mn::material_e::NACC_ASFLIP_FBAR>( 
                        gpu_id, model_id, materialConfigs, algoConfigs,
                        output_attribs, track_particle_ids, track_attribs, target_attribs);
                }
                else if (!algoConfigs.use_ASFLIP && algoConfigs.use_FBAR && !algoConfigs.use_FEM)
                {
                  benchmark->initModel<mn::material_e::NACC_FBAR>(gpu_id, model_id, positions, velocity, track_particle_ids, track_attribs);
                  benchmark->updateParameters<mn::material_e::NACC_FBAR>( 
                        gpu_id, model_id, materialConfigs, algoConfigs,
                        output_attribs, track_particle_ids, track_attribs, target_attribs);
                }
                else if (algoConfigs.use_ASFLIP && !algoConfigs.use_FBAR && !algoConfigs.use_FEM)
                {
                  benchmark->initModel<mn::material_e::NACC_ASFLIP>(gpu_id, model_id, positions, velocity, track_particle_ids, track_attribs);
                  benchmark->updateParameters<mn::material_e::NACC_ASFLIP>( 
                        gpu_id, model_id, materialConfigs, algoConfigs,
                        output_attribs, track_particle_ids, track_attribs, target_attribs);
                }
                else if (!algoConfigs.use_ASFLIP && !algoConfigs.use_FBAR && !algoConfigs.use_FEM)
                {
                  benchmark->initModel<mn::material_e::NACC>(gpu_id, model_id, positions, velocity, track_particle_ids, track_attribs);
                  benchmark->updateParameters<mn::material_e::NACC>( 
                        gpu_id, model_id, materialConfigs, algoConfigs,
                        output_attribs, track_particle_ids, track_attribs, target_attribs);
                }
                else { algo_error = true; }
              } 
              else { mat_error = true; } //< Requested material doesn't exist in code
              if (mat_error) {
                fmt::print(fg(red),"ERROR: GPU[{}] constititive[{}] does not exist! Press ENTER to continue...\n",  gpu_id, constitutive); if (mn::config::g_log_level >= 3) getchar(); return; 
              }
              if (algo_error) {
                fmt::print(fg(red), "ERROR: GPU[{}] Undefined algorithm use for material [{}]: use_ASFLIP[{}], use_FEM[{}], use_FBAR[{}]! Press ENTER to continue...\n", gpu_id, constitutive, algoConfigs.use_ASFLIP, algoConfigs.use_FEM, algoConfigs.use_FBAR); if (mn::config::g_log_level >= 3) getchar(); return; 
              }
              fmt::print(fg(green),"GPU[{}] Material[{}] model set and updated.\n", gpu_id, constitutive);
            };

            mn::vec<PREC, 3> velocity, partition_start, partition_end;
            velocity = CheckDoubleArray(model, "velocity", mn::pvec3{0.,0.,0.});
            partition_start = CheckDoubleArray(model, "partition_start", mn::pvec3{0.,0.,0.});
            partition_end = CheckDoubleArray(model, "partition_end", domain);
            for (int d = 0; d < 3; ++d) {
              velocity[d] *= sqrt(froude_scaling);
              velocity[d] = velocity[d] / l;
              partition_start[d] *= froude_scaling;
              partition_start[d] = partition_start[d] / l + o;
              // Messy check to avoid froude scaling a partition that was already set to a default value of the froude scaled domain
              if (partition_end[d] != domain[d]) partition_end[d] *= froude_scaling;
              partition_end[d]   = partition_end[d] / l + o;
              if (partition_start[d] > partition_end[d]) {
                fmt::print(fg(red), "GPU[{}] ERROR: Inverted partition (Element of partition_start > partition_end). Fix and Retry.", gpu_id); if (mn::config::g_log_level >= 3) getchar();
                std::exit(EXIT_FAILURE);
              } else if (partition_end[d] == partition_start[d]) {
                fmt::print(fg(red), "GPU[{}] ERROR: Zero volume partition (Element of partition_end == partition_start). Fix and Retry.", gpu_id); if (mn::config::g_log_level >= 3) getchar();
                std::exit(EXIT_FAILURE);
              }
            }
            output_attribs = CheckStringArray(model, "output_attribs", std::vector<std::string> {{"ID"}});
            track_attribs = CheckStringArray(model, "track_attribs", std::vector<std::string> {{"Position_Y"}});
            track_particle_ids = CheckIntArray(model, "track_particle_id", std::vector<int>{0});
            target_attribs = CheckStringArray(model, "target_attribs", std::vector<std::string> {{"Position_Y"}});
            if (output_attribs.size() > mn::config::g_max_particle_attribs) { fmt::print(fg(red), "ERROR: GPU[{}] Only [{}] output_attribs value supported.\n", gpu_id, mn::config::g_max_particle_attribs); }
            if (track_attribs.size() > mn::config::g_max_particle_tracker_attribs) { fmt::print(fg(red), "ERROR: GPU[{}] Only [{}] track_attribs value supported currently.\n", gpu_id, mn::config::g_max_particle_tracker_attribs); }
            if (track_particle_ids.size() > mn::config::g_max_particle_trackers) { fmt::print(fg(red), "ERROR: Only [{}] track_particle_id value supported currently.\n", mn::config::g_max_particle_trackers); }
            
            if (track_particle_ids.size() != 0 && track_attribs.size() == 0) 
              fmt::print(fg(red),"ERROR: GPU[{}] MODEL[{}] track_particle_ids provided but no track_attribs provided. \n", gpu_id, model_id);
            if (track_particle_ids.size() == 0 && track_attribs.size() != 0) 
              fmt::print(fg(red),"ERROR: GPU[{}] MODEL[{}] track_particle_ids is not provided but track_attribs is provided. \n", gpu_id, model_id);
            if (track_particle_ids.size() != 0 && track_attribs.size() != 0) 
              fmt::print("GPU[{}] Track attribute count[{}] on particle count[{}].\n", gpu_id, track_attribs.size(), track_particle_ids.size());
            if (target_attribs.size() > 1) { fmt::print(fg(red), "ERROR: GPU[{}] Only [1] target_attribs value supported currently.\n", gpu_id); }

            // * Begin particle geometry construction 
            auto geo = model.FindMember("geometry");
            // auto geos = model.FindMember("geometries"); // TODO: Use this instead or have schema accept either (but not both?)
            if (geo != model.MemberEnd()) {
              if (geo->value.IsArray()) {
                fmt::print(fg(cyan),"GPU[{}] MODEL[{}] has [{}] particle geometry operations to perform. \n", gpu_id, model_id, geo->value.Size());
                for (auto &geometry : geo->value.GetArray()) {
                  std::string operation = CheckString(geometry, "operation", std::string{"add"});
                  std::string type = CheckString(geometry, "object", std::string{"box"});
                  fmt::print(fg(white), "GPU[{}] MODEL[{}] Begin operation[{}] with object[{}]... \n", gpu_id, model_id, operation, type);

                  mn::vec<PREC, 3> geometry_offset, geometry_span, geometry_spacing;
                  mn::vec<int, 3> geometry_array;
                  mn::vec<PREC, 3> geometry_rotate, geometry_fulcrum;
                  geometry_span = CheckDoubleArray(geometry, "span", mn::pvec3{1.,1.,1.});
                  geometry_offset = CheckDoubleArray(geometry, "offset", mn::pvec3{0.,0.,0.});
                  geometry_array = CheckIntArray(geometry, "array", mn::ivec3{1,1,1});
                  geometry_spacing = CheckDoubleArray(geometry, "spacing", mn::pvec3{0.,0.,0.});
                  geometry_rotate = CheckDoubleArray(geometry, "rotate", mn::pvec3{0.,0.,0.});
                  geometry_fulcrum = CheckDoubleArray(geometry, "fulcrum", mn::pvec3{0.,0.,0.});           
                  for (int d = 0; d < 3; ++d) {
                    geometry_span[d]    *= froude_scaling;
                    geometry_span[d]    = geometry_span[d]    / l;
                    geometry_offset[d]  *= froude_scaling;
                    geometry_offset[d]  = geometry_offset[d]  / l + o;
                    geometry_spacing[d] *= froude_scaling;
                    geometry_spacing[d] = geometry_spacing[d] / l;
                    geometry_fulcrum[d] *= froude_scaling;                    
                    geometry_fulcrum[d] = geometry_fulcrum[d] / l + o;
                  }
                  mn::pvec3x3 rotation_matrix; rotation_matrix.set(0.0);
                  rotation_matrix(0,0) = rotation_matrix(1,1) = rotation_matrix(2,2) = 1;
                  elementaryToRotationMatrix(geometry_rotate, rotation_matrix);
                  fmt::print("Rotation Matrix: \n");
                  for (int i=0;i<3;i++) {for (int j=0;j<3;j++) fmt::print("{} ", rotation_matrix(i,j)); fmt::print("\n");}


                  int keep_track_of_array = 0; // Current index in array operation
                  int keep_track_of_particles = 0; // Particles per single array operation
                  mn::vec<PREC, 3> geometry_offset_updated;
                  geometry_offset_updated[0] = geometry_offset[0];
                  for (int i = 0; i < geometry_array[0]; i++)
                  {
                  geometry_offset_updated[1] = geometry_offset[1];
                  for (int j = 0; j < geometry_array[1]; j++)
                  {
                  geometry_offset_updated[2] = geometry_offset[2];
                  for (int k = 0; k < geometry_array[2]; k++)
                  {
                  std::vector<int> geo_track_particle_ids; //< Particle IDs to track for this geometry instance
                  geo_track_particle_ids = CheckIntArray(geometry, "track_particle_id", std::vector<int>{});
                  // * Shift geometry local particle IDs to track global IDs
                  if (keep_track_of_array == 1)
                    keep_track_of_particles = models[total_id].size();
                  // int shift_idx = keep_track_of_particles * keep_track_of_array;

                  for (int geo_idx = 0; geo_idx < geo_track_particle_ids.size(); ++geo_idx) {
                    geo_track_particle_ids[geo_idx] += keep_track_of_particles * keep_track_of_array; // ! May need to adjust for previous geometry particles, e.g. + model.size()
                  }

                  track_particle_ids.insert(track_particle_ids.end(), geo_track_particle_ids.begin(), geo_track_particle_ids.end()); //< Append to global track_particle_ids
                  if (track_particle_ids.size() > mn::config::g_max_particle_trackers) { fmt::print(fg(red), "ERROR: Only [{}] track_particle_id value supported currently.\n", mn::config::g_max_particle_trackers); }

                  if (type == "Box" || type == "box")
                  {
                    if (operation == "Add" || operation == "add") {
                      make_box(models[total_id], geometry_span, geometry_offset_updated, materialConfigs.ppc, partition_start, partition_end, rotation_matrix, geometry_fulcrum); }
                    else if (operation == "Subtract" || operation == "subtract") {
                      subtract_box(models[total_id], geometry_span, geometry_offset_updated); }
                    else if (operation == "Union" || operation == "union") { fmt::print(fg(red),"Operation not implemented...\n");}
                    else if (operation == "Intersect" || operation == "intersect") { fmt::print(fg(red),"Operation not implemented...\n");}
                    else if (operation == "Difference" || operation == "difference") { fmt::print(fg(red),"Operation not implemented...\n");}
                    else { fmt::print(fg(red), "ERROR: GPU[{}] MODEL[{}] geometry operation[{}] invalid! \n", gpu_id, model_id, operation); 
                      if (mn::config::g_log_level >= 3) getchar(); 
                    }
                  }
                  else if (type == "Cylinder" || type == "cylinder")
                  {
                    PREC geometry_radius = CheckDouble(geometry, "radius", 0.) * froude_scaling;
                    std::string geometry_axis = CheckString(geometry, "axis", std::string{"X"});

                    if (operation == "Add" || operation == "add") {
                      make_cylinder(models[total_id], geometry_span, geometry_offset_updated, materialConfigs.ppc, geometry_radius, geometry_axis, partition_start, partition_end, rotation_matrix, geometry_fulcrum); }
                    else if (operation == "Subtract" || operation == "subtract") {             subtract_cylinder(models[total_id], geometry_radius, geometry_axis, geometry_span, geometry_offset_updated); }
                    else { fmt::print(fg(red), "ERROR: GPU[{}] geometry operation[{}] invalid! \n", gpu_id, operation); if (mn::config::g_log_level >= 3) getchar(); }
                  }
                  else if (type == "Sphere" || type == "sphere")
                  {
                    PREC geometry_radius = CheckDouble(geometry, "radius", 0.) * froude_scaling;
                    if (operation == "Add" || operation == "add") {
                      make_sphere(models[total_id], geometry_span, geometry_offset_updated, materialConfigs.ppc, geometry_radius, partition_start, partition_end, rotation_matrix, geometry_fulcrum); }
                    else if (operation == "Subtract" || operation == "subtract") {
                      subtract_sphere(models[total_id], geometry_radius, geometry_offset_updated); }
                    else {  fmt::print(fg(red), "ERROR: GPU[{}] geometry operation[{}] invalid! \n", gpu_id, operation); if (mn::config::g_log_level >= 3) getchar(); }
                  }
                  else if (type == "OSU LWF" || type == "OSU_LWF_WATER")
                  {
                    auto use_custom_bathymetry = CheckBool(geometry, "use_custom_bathymetry", false);
                    if (use_custom_bathymetry) {
                      fmt::print(fg(cyan),"GPU[{}] MODEL[{}] Using custom bathymetry for geometry object[{}]. Loading... \n", gpu_id, model_id, type);
                      // Check for bathymetry coordinates (optional)
                      int num_bathymetry_points = 0;
                      mn::vec<PREC, 2> bathymetry_points[mn::config::g_max_bathymetry_points];
                      
                      auto bathymetry = geometry.FindMember("bathymetry"); // [[x1,y1],[x2,y2],...]
                      if (bathymetry != geometry.MemberEnd()) {
                        fmt::print(fg(cyan),"GPU[{}] Found bathymetry for geometry object[{}]. Loading... \n", gpu_id, type);
                        // vec<PREC_G, 2> bathymetry_coords[mn::config::g_max_bathymetry_points];
                        int bathymetry_size = static_cast<int>(geometry["bathymetry"].GetArray().Size());
                        if (bathymetry_size > static_cast<int>(mn::config::g_max_bathymetry_points)) {
                          fmt::print(fg(red),"ERROR: GPU[{}] Too many bathymetry coordinates[{}]. Max allowed is [{}].\n", gpu_id, bathymetry_size, mn::config::g_max_bathymetry_points);
                          if (mn::config::g_log_level >= 3) getchar();
                        } else if (bathymetry_size == 0) {
                          fmt::print(fg(red),"ERROR: GPU[{}] No bathymetry coordinates set!\n", gpu_id);
                        }
                        num_bathymetry_points = static_cast<int>(bathymetry_size);
                        fmt::print(fg(cyan),"INFO: GPU[{}] Found [{}] bathymetry coordinates! Max allowable: [{}] \n", gpu_id, num_bathymetry_points, mn::config::g_max_bathymetry_points); 
                      
                        int bathymetry_i = 0;
                        for (auto &coord : geometry["bathymetry"].GetArray()) {
                          if (bathymetry_i >= mn::config::g_max_bathymetry_points) break;
                          mn::vec<PREC_G, 2> xy;
                          for (int d = 0; d < 2; ++d) {
                            xy[d] = coord.GetArray()[d].GetDouble();
                          }

                          bathymetry_points[bathymetry_i][0] = xy[0] * froude_scaling / l + o;
                          bathymetry_points[bathymetry_i][1] = xy[1] * froude_scaling / l + o;
                          fmt::print(fg(green),"INFO: GPU[{}] Bathymetry[{}/{}]: Input [{}, {}], Scaled [{}, {}]\n", gpu_id, bathymetry_i, num_bathymetry_points, xy[0], xy[1], bathymetry_points[bathymetry_i][0], bathymetry_points[bathymetry_i][1]);
                          bathymetry_i++;
                        }

                        if (operation == "Add" || operation == "add") {
                          make_bathymetry(models[total_id], geometry_span, geometry_offset_updated, materialConfigs.ppc, partition_start, partition_end, rotation_matrix, geometry_fulcrum, bathymetry_points, num_bathymetry_points); 
                        } else if (operation == "Subtract" || operation == "subtract") { 
                          subtract_bathymetry(models[total_id], geometry_span, geometry_offset_updated, bathymetry_points, num_bathymetry_points);
                        } else { fmt::print(fg(red), "ERROR: GPU[{}] geometry [{}] operation[{}] invalid! \n", gpu_id, type, operation); if (mn::config::g_log_level >= 3) getchar(); }
                        
                      } else {
                        fmt::print(fg(red),"ERROR: GPU[{}] No custom bathymetry points set!\n", gpu_id);
                      }
                    }
                    else {
                      if (operation == "Add" || operation == "add") {
                        make_OSU_LWF(models[total_id], geometry_span, geometry_offset_updated, materialConfigs.ppc, partition_start, partition_end, rotation_matrix, geometry_fulcrum, froude_scaling); }
                      else if (operation == "Subtract" || operation == "subtract") { fmt::print(fg(red),"Operation not implemented yet...\n"); }
                      else { fmt::print(fg(red), "ERROR: GPU[{}] geometry operation[{}] invalid! \n", gpu_id, operation); if (mn::config::g_log_level >= 3) getchar(); }
                    }
                  }
                  else if (type == "OSU TWB" || type == "OSU_TWB_WATER")
                  {
                    if (operation == "Add" || operation == "add") {
                      make_OSU_TWB(models[total_id], geometry_span, geometry_offset_updated, materialConfigs.ppc, partition_start, partition_end, rotation_matrix, geometry_fulcrum, froude_scaling); }
                    else if (operation == "Subtract" || operation == "subtract") { fmt::print(fg(red),"Operation not implemented yet...\n"); }
                    else { fmt::print(fg(red), "ERROR: GPU[{}] geometry operation[{}] invalid! \n", gpu_id, operation); if (mn::config::g_log_level >= 3) getchar(); }
                  }
                  else if (type == "Bathymetry" || type == "bathymetry")
                  {
                    // Check for bathymetry coordinates (optional)
                    int num_bathymetry_points = 0;
                    mn::vec<PREC, 2> bathymetry_points[mn::config::g_max_bathymetry_points];
                    
                    auto bathymetry = geometry.FindMember("bathymetry"); // [[x1,y1],[x2,y2],...]
                    if (bathymetry != geometry.MemberEnd()) {
                      fmt::print(fg(cyan),"GPU[{}] Found bathymetry for geometry object[{}]. Loading... \n", gpu_id, type);
                      // vec<PREC_G, 2> bathymetry_coords[mn::config::g_max_bathymetry_points];
                      int bathymetry_size = static_cast<int>(geometry["bathymetry"].GetArray().Size());
                      if (bathymetry_size > static_cast<int>(mn::config::g_max_bathymetry_points)) {
                        fmt::print(fg(red),"ERROR: GPU[{}] Too many bathymetry coordinates[{}]. Max allowed is [{}].\n", gpu_id, bathymetry_size, mn::config::g_max_bathymetry_points);
                        if (mn::config::g_log_level >= 3) getchar();
                      } else if (bathymetry_size == 0) {
                        fmt::print(fg(red),"ERROR: GPU[{}] No bathymetry coordinates set!\n", gpu_id);
                      }
                      num_bathymetry_points = static_cast<int>(bathymetry_size);
                      fmt::print(fg(cyan),"INFO: GPU[{}] Found [{}] bathymetry coordinates! Max allowable: [{}] \n", gpu_id, num_bathymetry_points, mn::config::g_max_bathymetry_points); 
                    
                      int bathymetry_i = 0;
                      for (auto &coord : geometry["bathymetry"].GetArray()) {
                        if (bathymetry_i >= mn::config::g_max_bathymetry_points) break;
                        mn::vec<PREC_G, 2> xy;
                        for (int d = 0; d < 2; ++d) {
                          xy[d] = coord.GetArray()[d].GetDouble();
                        }

                        bathymetry_points[bathymetry_i][0] = xy[0] * froude_scaling / l + o;
                        bathymetry_points[bathymetry_i][1] = xy[1] * froude_scaling / l + o;
                        fmt::print(fg(green),"INFO: GPU[{}] Bathymetry[{}/{}]: Input [{}, {}], Scaled [{}, {}]\n", gpu_id, bathymetry_i, num_bathymetry_points, xy[0], xy[1], bathymetry_points[bathymetry_i][0], bathymetry_points[bathymetry_i][1]);
                        bathymetry_i++;
                      }

                      if (operation == "Add" || operation == "add") {
                        make_bathymetry(models[total_id], geometry_span, geometry_offset_updated, materialConfigs.ppc, partition_start, partition_end, rotation_matrix, geometry_fulcrum, bathymetry_points, num_bathymetry_points); 
                      } else if (operation == "Subtract" || operation == "subtract") { 
                        subtract_bathymetry(models[total_id], geometry_span, geometry_offset_updated, bathymetry_points, num_bathymetry_points);
                      } else { fmt::print(fg(red), "ERROR: GPU[{}] geometry [{}] operation[{}] invalid! \n", gpu_id, type, operation); if (mn::config::g_log_level >= 3) getchar(); }
                      
                    } else {
                      fmt::print(fg(red),"ERROR: GPU[{}] No custom bathymetry points set!\n", gpu_id);
                    }
                  }
                  else if (type == "File" || type == "file") 
                  {
                    // * NOTE : Assumes geometry "file" specified by scene.json is in  AssetDirPath/, e.g. for AssetDirPath = ~/claymore/Data/, then use ~/claymore/Data/file
                    std::string geometry_file = CheckString(geometry, "file", std::string{"MpmParticles/yoda.sdf"});
                    // std::string geometry_fn = std::string(AssetDirPath) + geometry_file;
                    std::string geometry_fn = std::string("./") + geometry_file;
                    fs::path geometry_file_path{geometry_fn};
                    if (geometry_file_path.empty()) fmt::print(fg(red), "ERROR: Input file[{}] does not exist.\n", geometry_fn);
                    else {
                      std::ifstream istrm(geometry_fn);
                      if (!istrm.is_open())  fmt::print(fg(red), "ERROR: Cannot open file[{}]\n", geometry_fn);
                      istrm.close();
                    }
                    if (operation == "Add" || operation == "add") {
                      if (geometry_file_path.extension() == ".sdf") 
                      {
                        PREC geometry_scaling_factor = CheckDouble(geometry, "scaling_factor", 1) * froude_scaling;
                        int geometry_padding = CheckInt(geometry, "padding", 1);
                        if (geometry_scaling_factor <= 0) {
                          fmt::print(fg(red), "ERROR: [scaling_factor] must be greater than [0] for SDF file load (e.g. [2] doubles size, [0] erases size). Fix and Retry.\n"); if (mn::config::g_log_level >= 3) getchar(); }
                        if (geometry_padding < 1) {
                          fmt::print(fg(red), "ERROR: Signed-Distance-Field (.sdf) files require [padding] of atleast [1] (padding is empty exterior cells on sides of model, allows surface definition). Fix and Retry.");fmt::print(fg(yellow), "TIP: Use open-source SDFGen to create *.sdf from *.obj files.\n"); if (mn::config::g_log_level >= 3) getchar();}
                        mn::read_sdf(geometry_fn, models[total_id], materialConfigs.ppc,
                            (PREC)dx, mn::config::g_domain_size, geometry_offset_updated, l,
                            partition_start, partition_end, rotation_matrix, geometry_fulcrum, geometry_scaling_factor, geometry_padding);
                      }
                      else if (geometry_file_path.extension() == ".csv") 
                      {
                        load_csv_particles(geometry_fn, ',', 
                                            models[total_id], geometry_offset_updated, 
                                            partition_start, partition_end, rotation_matrix, geometry_fulcrum);
                      }
                      else if (geometry_file_path.extension() == ".bgeo" ||
                          geometry_file_path.extension() == ".geo" ||
                          geometry_file_path.extension() == ".pdb" ||
                          geometry_file_path.extension() == ".ptc") 
                      {
                        // Can enable / disable Froude scaling when loading in bgeo, etc. files.
                        bool use_froude_scaling = CheckBool(geometry, "use_froude_scaling", true);

                        if (keep_track_of_array == 0) {
                          has_attributes = CheckBool(geometry, "has_attributes", false);
                          input_attribs = CheckStringArray(geometry, "input_attribs", std::vector<std::string> {{"ID"}});
                          fmt::print(fg(white),"GPU[{}] Try to read pre-existing particle attributes into model? has_attributes[{}].\n", gpu_id, has_attributes);
                          if (input_attribs.size() > mn::config::g_max_particle_attribs) {
                            fmt::print(fg(red), "ERROR: GPU[{}] MODEL[{}] suppports max of [{}] input_attribs, but [{}] are specified. Press ENTER to continue...\n", gpu_id, model_id, mn::config::g_max_particle_attribs, input_attribs.size()); 
                            if (mn::config::g_log_level >= 3) getchar();
                          }
                          attributes.resize(0, std::vector<PREC>(input_attribs.size()));
                        }
                        // Read in particle positions and attributes from file
                        mn::read_partio_general<PREC>(geometry_fn, models[total_id], attributes, input_attribs.size(), input_attribs); 

                        if (keep_track_of_array == 0) {
                          attributes.reserve(attributes.size() * geometry_array[0] * geometry_array[1] * geometry_array[2]);
                          keep_track_of_particles = models[total_id].size();
                          fmt::print("Size of attributes after reading in initial data: Particles[{}], Attributes[{}]\n", attributes.size(), attributes[0].size());
                          for (int i = 0; i < attributes[0].size(); i++) {
                              fmt::print("Input Attribute[{}][{}]:  Label[{}] Value[{}]\n", 0, i, input_attribs[i], attributes[0][i]);
                          }
                        }

                        int shift_idx = keep_track_of_particles * keep_track_of_array;
                        // Scale particle positions to 1x1x1 simulation
                        for (int part=0; part < keep_track_of_particles; part++) {
                          for (int d = 0; d<3; d++) {
                            if (use_froude_scaling) models[total_id][part + shift_idx][d] = models[total_id][part + shift_idx][d] * froude_scaling;
                            models[total_id][part + shift_idx][d] = models[total_id][part + shift_idx][d] / l + geometry_offset_updated[d];
                          }
                          // Scale velocity based attributes to 1x1x1 simulation 
                          for (int d = 0; d < input_attribs.size(); d++) {
                            if (input_attribs[d] == "Velocity_X" || input_attribs[d] == "Velocity_Y" || input_attribs[d] == "Velocity_Z" ) {
                              if (use_froude_scaling) attributes[part + shift_idx][d] = attributes[part + shift_idx][d] * sqrt(froude_scaling);
                              attributes[part + shift_idx][d] = attributes[part + shift_idx][d] / l; 
                            }
                          }
                          // Scale force based attributes to 1x1x1 simulation 
                          for (int d = 0; d < input_attribs.size(); d++) {
                            if (input_attribs[d] == "Force_X" || input_attribs[d] == "Force_Y" || input_attribs[d] == "Force_Z" ) {
                              if (use_froude_scaling) attributes[part + shift_idx][d] = attributes[part + shift_idx][d] * froude_scaling*froude_scaling*froude_scaling;
                              attributes[part + shift_idx][d] = attributes[part + shift_idx][d] / l; 
                            }
                          }
                          // Scale deformation based attributes by froude_scaling
                          // ! Assumes J = sJ and JBar = sJBar currrently.
                          // TODO: Fix J to not be sJ (1-J), 
                          for (int d = 0; d < input_attribs.size(); d++) {
                            if (input_attribs[d] == "J" || input_attribs[d] == "JBar" || input_attribs[d] == "sJ"|| input_attribs[d] == "sJBar" )
                              if (use_froude_scaling) attributes[part + shift_idx][d] = attributes[part + shift_idx][d] * froude_scaling;  //< Need to recheck validity, probs not accurate scaling for det | deformation gradient |
                          }

                        }
                      }
                    }
                    else if (operation == "Subtract" || operation == "subtract") { fmt::print(fg(red),"Operation not implemented...\n"); }
                    else if (operation == "Union" || operation == "union") {fmt::print(fg(red),"Operation not implemented...\n");}
                    else if (operation == "Intersect" || operation == "intersect") {fmt::print(fg(red),"Operation not implemented...\n");}
                    else if (operation == "Difference" || operation == "difference") {fmt::print(fg(red),"Operation not implemented...\n");}
                    else { fmt::print(fg(red), "ERROR: GPU[{}] geometry operation[{}] invalid! Press ENTER to continue...\n", gpu_id, operation); 
                    if (mn::config::g_log_level >= 3) getchar(); 
                    }
                  }
                  else  { fmt::print(fg(red), "GPU[{}] ERROR: Geometry object[{}] does not exist! Press ENTER to continue...\n", gpu_id, type); 
                    if (mn::config::g_log_level >= 3) getchar();
                  } 
                  keep_track_of_array++; // * Keep track of how many times we've iterated through the array
                  geometry_offset_updated[2] += geometry_spacing[2];
                  } 
                  geometry_offset_updated[1] += geometry_spacing[1];
                  } 
                  geometry_offset_updated[0] += geometry_spacing[0];
                  }
                }
              }
            } //< End geometry
            else {
              fmt::print(fg(red), "ERROR: NODE[{}] GPU[{}] MODEL[{}] No geometry object! Neccesary to create particles.\n", node_id, gpu_id, model_id);
              fmt::print(fg(red), "Press enter to continue...\n"); if (mn::config::g_log_level >= 3) getchar();
            }
              
            auto positions = models[total_id];
            mn::IO::insert_job([&]() {
              mn::write_partio<PREC,3>(std::string{p.stem()} + save_suffix,positions); });              
            mn::IO::flush();
            fmt::print(fg(green), "NODE[{}] GPU[{}] MODEL[{}] Saved particles to [{}].\n", node_id, gpu_id, model_id, std::string{p.stem()} + save_suffix);
            
            if (positions.size() > mn::config::g_max_particle_num) {
              fmt::print(fg(red), "ERROR: NODE[{}] GPU[{}] MODEL[{}] Particle count [{}] exceeds g_max_particle_num in settings.h! Increase and recompile to avoid problems. \n", node_id, gpu_id, model_id, positions.size());
              fmt::print(fg(red), "Press ENTER to continue anyways... \n");
              if (mn::config::g_log_level >= 3) getchar();
            }

            // * Initialize particle positions in simulator and on GPU
            initModel(positions, velocity);

            // ! Hard-coded available attribute count per particle for input, output
            // ! Better optimized run-time binding for GPU Taichi-esque data-structures, but could definitely be improved using Thrust data-structures, etc. 

            // * Initialize particle attributes in simulator and on GPU
            if (!has_attributes) attributes = std::vector<std::vector<PREC> >(positions.size(), std::vector<PREC>(input_attribs.size(), 0.)); //< Zero initial attribs if none
            if (input_attribs.size() == 1){
              constexpr mn::num_attribs_e N = static_cast<mn::num_attribs_e>(1);
              benchmark->initInitialAttribs<N>(gpu_id, model_id, attributes, has_attributes); 
            } else if (input_attribs.size() == 2){
              constexpr mn::num_attribs_e N = static_cast<mn::num_attribs_e>(2);
              benchmark->initInitialAttribs<N>(gpu_id, model_id, attributes, has_attributes); 
            } else if (input_attribs.size() == 3){
              constexpr mn::num_attribs_e N = static_cast<mn::num_attribs_e>(3);
              benchmark->initInitialAttribs<N>(gpu_id, model_id, attributes, has_attributes); 
            } else if (input_attribs.size() == 4){
              constexpr mn::num_attribs_e N = static_cast<mn::num_attribs_e>(4);
              benchmark->initInitialAttribs<N>(gpu_id, model_id, attributes, has_attributes); 
            } else if (input_attribs.size() == 5){
              constexpr mn::num_attribs_e N = static_cast<mn::num_attribs_e>(5);
              benchmark->initInitialAttribs<N>(gpu_id, model_id, attributes, has_attributes); 
            } else if (input_attribs.size() == 6){
              constexpr mn::num_attribs_e N = static_cast<mn::num_attribs_e>(6);
              benchmark->initInitialAttribs<N>(gpu_id, model_id, attributes, has_attributes); 
            } else if (input_attribs.size() == 7){
              constexpr mn::num_attribs_e N = static_cast<mn::num_attribs_e>(7);
              benchmark->initInitialAttribs<N>(gpu_id, model_id, attributes, has_attributes); 
            } else if (input_attribs.size() == 8){
              constexpr mn::num_attribs_e N = static_cast<mn::num_attribs_e>(8);
              benchmark->initInitialAttribs<N>(gpu_id, model_id, attributes, has_attributes); 
            } else if (input_attribs.size() == 9){
              constexpr mn::num_attribs_e N = static_cast<mn::num_attribs_e>(9);
              benchmark->initInitialAttribs<N>(gpu_id, model_id, attributes, has_attributes); 
            } else if (input_attribs.size() > mn::config::g_max_particle_attribs){
              fmt::print("More than [{}] input_attribs not implemented. You requested [{}].", mn::config::g_max_particle_attribs, input_attribs.size());
              if (mn::config::g_log_level >= 3) getchar();
            } else {
              constexpr mn::num_attribs_e N = static_cast<mn::num_attribs_e>(1);
              benchmark->initInitialAttribs<N>(gpu_id, model_id, attributes, has_attributes); 
            }
            
            // * Initialize output particle attributes in simulator and on GPU
            attributes = std::vector<std::vector<PREC> >(positions.size(), std::vector<PREC>(output_attribs.size(), 0.));
            if (output_attribs.size() == 1){
              constexpr mn::num_attribs_e N = static_cast<mn::num_attribs_e>(1);
              benchmark->initOutputAttribs<N>(gpu_id, model_id, attributes); 
            } else if (output_attribs.size() == 2){
              constexpr mn::num_attribs_e N = static_cast<mn::num_attribs_e>(2);
              benchmark->initOutputAttribs<N>(gpu_id, model_id, attributes); 
            } else if (output_attribs.size() == 3){
              constexpr mn::num_attribs_e N = static_cast<mn::num_attribs_e>(3);
              benchmark->initOutputAttribs<N>(gpu_id, model_id, attributes); 
            } else if (output_attribs.size() == 4){
              constexpr mn::num_attribs_e N = static_cast<mn::num_attribs_e>(4);
              benchmark->initOutputAttribs<N>(gpu_id, model_id, attributes); 
            } else if (output_attribs.size() == 5){
              constexpr mn::num_attribs_e N = static_cast<mn::num_attribs_e>(5);
              benchmark->initOutputAttribs<N>(gpu_id, model_id, attributes); 
            } else if (output_attribs.size() == 6){
              constexpr mn::num_attribs_e N = static_cast<mn::num_attribs_e>(6);
              benchmark->initOutputAttribs<N>(gpu_id, model_id, attributes); 
            } else if (output_attribs.size() == 7){
              constexpr mn::num_attribs_e N = static_cast<mn::num_attribs_e>(7);
              benchmark->initOutputAttribs<N>(gpu_id, model_id, attributes); 
            } else if (output_attribs.size() == 8){
              constexpr mn::num_attribs_e N = static_cast<mn::num_attribs_e>(8);
              benchmark->initOutputAttribs<N>(gpu_id, model_id, attributes); 
            } else if (output_attribs.size() == 9){
              constexpr mn::num_attribs_e N = static_cast<mn::num_attribs_e>(9);
              benchmark->initOutputAttribs<N>(gpu_id, model_id, attributes); 
            } else if (output_attribs.size() > mn::config::g_max_particle_attribs){
              fmt::print(fg(red), "ERROR: GPU[{}] MODEL[{}] More than [{}] output_attribs not valid. Requested: [{}]. Truncating...", gpu_id, model_id, mn::config::g_max_particle_attribs, output_attribs.size()); 
              constexpr mn::num_attribs_e N = static_cast<mn::num_attribs_e>(mn::config::g_max_particle_attribs);
              benchmark->initOutputAttribs<N>(gpu_id, model_id, attributes); 
            } else {
              fmt::print(fg(orange), "WARNING: GPU[{}] MODEL[{}] output_attribs not found. Using [1] element default", gpu_id, model_id );
              constexpr mn::num_attribs_e N = static_cast<mn::num_attribs_e>(1);
              benchmark->initOutputAttribs<N>(gpu_id, model_id, attributes); 
            }
            fmt::print(fmt::emphasis::bold,
                      "-----------------------------------------------------------"
                      "-----\n");
          }
        }
      }
    } ///< end models parsing
    {
      auto it = doc.FindMember("grid-sensors");
      if (it != doc.MemberEnd()) {
        if (it->value.IsArray()) {
          fmt::print(fg(cyan),"Scene has [{}] grid-sensors.\n", it->value.Size());
          int target_ID = 0;
          for (auto &model : it->value.GetArray()) {
          
            std::vector<std::array<PREC_G, mn::config::g_grid_target_attribs>> h_gridTarget(mn::config::g_grid_target_cells, std::array<PREC_G, mn::config::g_grid_target_attribs> {0.0});
            mn::vec<PREC_G, 7> target; // TODO : Make structure for grid-target data

            
            std::string direction = CheckString(model,"direction", std::string{"X"});
            if      (direction == "X"  || direction == "x")  target[0] = 0;
            else if (direction == "X-" || direction == "x-") target[0] = 1;
            else if (direction == "X+" || direction == "x+") target[0] = 2;
            else if (direction == "Y"  || direction == "y")  target[0] = 3;
            else if (direction == "Y-" || direction == "y-") target[0] = 4;
            else if (direction == "Y+" || direction == "y+") target[0] = 5;
            else if (direction == "Z"  || direction == "z")  target[0] = 6;
            else if (direction == "Z-" || direction == "z-") target[0] = 7;
            else if (direction == "Z+" || direction == "z+") target[0] = 8;
            else {
              target[0] = -1;
              fmt::print(fg(red), "ERROR: gridTarget[{}] has invalid direction[{}].\n", target_ID, direction);
              if (mn::config::g_log_level >= 3) getchar();
            }

            std::string attribute = CheckString(model,"attribute", std::string{"force"});
            if      (attribute == "Force"  || attribute == "force") target[0] += 0*9;
            else if (attribute == "Velocity" || attribute == "velocity") target[0] += 1*9;
            else if (attribute == "Momentum" || attribute == "momentum") target[0] += 2*9;
            else if (attribute == "Mass"  || attribute == "mass") target[0] += 3*9;
            else if (attribute == "Volume" || attribute == "volume") target[0] += 4*9;
            else if (attribute == "JBar" || attribute == "J Bar") target[0] += 5*9;
            else if (attribute == "MassWater" || attribute == "masw") target[0] += 6*9;
            else if (attribute == "PorePressure" || attribute == "pore_pressure") target[0] += 7*9;
            else {
              target[0] = -1;
              fmt::print(fg(red), "ERROR: gridTarget[{}] has invalid attribute[{}].\n", target_ID, attribute);
            }

            std::string operation = CheckString(model, "operation", std::string{"sum"});
            bool averages = false;
            if (operation == "sum" || operation == "Sum" || operation == "add" || operation == "Add" || operation == "addition" || operation == "Addition") {
              averages = false;
            }
            else if (operation == "average" || operation == "averages" || operation == "mean" || operation == "Average" || operation == "Mean" || operation == "Averages")
            {
              averages = true;
            }
            // * Load and scale target domain
            mn::vec<PREC_G,3> domain_start, domain_end;
            for (int d = 0; d < 3; ++d) 
            {
              target[d+1] = (model["domain_start"].GetArray()[d].GetFloat() * froude_scaling) / l + o;
              target[d+4] = (model["domain_end"].GetArray()[d].GetFloat() * froude_scaling) / l + o;
              domain_start[d] = (model["domain_start"].GetArray()[d].GetFloat()  * froude_scaling) / l + o;
              domain_end[d] = (model["domain_end"].GetArray()[d].GetFloat()  * froude_scaling) / l + o;
            }

            // * NOTE: Checks for zero length target dimensions, grows by 1 grid-cell if so
            for (int d=0; d < 3; ++d)
              if (target[d+1] == target[d+4]) target[d+4] = target[d+4] + dx;         
            
            // TODO: Should we froude scale the sensor frequency?
            PREC_G freq = CheckDouble(model, "output_frequency", 60.);

            // Write to OBJ file
            std::string fn_gb = "gridTarget[" + std::to_string(target_ID) + "].obj";
            writeBoxOBJ(fn_gb, domain_start, domain_end - domain_start);
            
            // mn::config::GridTargetConfigs gridTargetConfigs((int)target[6], (int)target[6], (int)target[6], make_float3((float)target[1], (float)target[2], (float)target[3]), make_float3((float)target[4], (float)target[5], (float)target[6]), (float)freq);

            // * Loop through GPU devices to initialzie
            for (int did = 0; did < mn::config::g_device_cnt; ++did) {
              benchmark->initGridTarget(did, h_gridTarget, target, 
                freq, averages); // TODO : Allow more than one frequency for grid-targets
            fmt::print(fg(green), "GPU[{}] gridTarget[{}] Initialized.\n", did, target_ID);
            }
            target_ID += 1;
            fmt::print(fmt::emphasis::bold,
                      "-----------------------------------------------------------"
                      "-----\n");
          }
        }
      }
    } ///< End grid-target parsing
    {
      auto it = doc.FindMember("particle-sensors"); // "sensors" used to be "targets", updated for HydroUQ
      if (it != doc.MemberEnd()) {
        if (it->value.IsArray()) {
          fmt::print(fg(cyan),"Scene has [{}] particle-sensors.\n", it->value.Size());
          int target_ID = 0;
          for (auto &model : it->value.GetArray()) {
          
            std::vector<std::array<PREC, mn::config::g_particle_target_attribs>> h_particleTarget(mn::config::g_particle_target_cells,std::array<PREC,          mn::config::g_particle_target_attribs>{0.f});
            mn::vec<PREC, 7> target; // TODO : Make structure for particle-target data
            // TODO : Implement attribute selection for particle-targets (only elevation currently)
            std::string operation = CheckString(model,"operation", std::string{"max"});
            if      (operation == "Maximum" || operation == "maximum" || operation == "Max" || operation == "max") target[0] = 0;
            else if (operation == "Minimum" || operation == "minimum" || operation == "Min" || operation == "min") target[0] = 1;
            else if (operation == "Add" || operation == "add" || operation == "Sum" || operation == "sum") target[0] = 2;
            else if (operation == "Subtract" || operation == "subtract") target[0] = 3;
            else if (operation == "Average" || operation == "average" ||  operation == "Mean" || operation == "mean") target[0] = 4;
            else if (operation == "Variance" || operation == "variance") target[0] = 5;
            else if (operation == "Standard Deviation" || operation == "stdev" || operation == "STDEV") target[0] = 6;
            else {
              target[0] = -1;
              fmt::print(fg(red), "ERROR: particleTarget[{}] has invalid operation[{}].\n", target_ID, operation);
              if (mn::config::g_log_level >= 3) getchar();
            }
            // Load and scale target domain to 1 x 1 x 1 domain + off-by-2 offset
            mn::vec<PREC, 3> domain_start, domain_end;
            for (int d = 0; d < 3; ++d) 
            {
              target[d+1] = (model["domain_start"].GetArray()[d].GetDouble() * froude_scaling) / l + o;
              target[d+4] = (model["domain_end"].GetArray()[d].GetDouble() * froude_scaling) / l + o;
              domain_start[d] = (model["domain_start"].GetArray()[d].GetFloat() * froude_scaling) / l + o;
              domain_end[d] = (model["domain_end"].GetArray()[d].GetFloat() * froude_scaling) / l + o;
            }
            // TODO: Should we froude scale sensor frequency?
            PREC freq = CheckDouble(model, "output_frequency", 60.);

            // mn::config::ParticleTargetConfigs particleTargetConfigs((int)target[6], (int)target[6], (int)target[6], {(float)target[1], (float)target[2], (float)target[3]}, {(float)target[4], (float)target[5], (float)target[6]}, (float)freq);

            // Write to OBJ file
            std::string fn_gb = "particleTarget[" + std::to_string(target_ID) + "].obj";
            writeBoxOBJ(fn_gb, domain_start, domain_end - domain_start);
            
            // Initialize on GPUs
            for (int did = 0; did < mn::config::g_device_cnt; ++did) {
              for (int mid = 0; mid < mn::config::g_models_per_gpu; ++mid) {
                benchmark->initParticleTarget(did, mid, h_particleTarget, target, 
                  freq);
                fmt::print(fg(green), "GPU[{}] particleTarget[{}] Initialized.\n", did, target_ID);
              }
            }
            target_ID += 1; // TODO : Count targets using static variable in a structure
            fmt::print(fmt::emphasis::bold,
                      "-----------------------------------------------------------"
                      "-----\n");
          }
        }
      }
    } ///< End particle-target parsing
    {
      auto it = doc.FindMember("boundaries"); // "boundaries" used to be "grid-boundaries", updated for HydroUQ
      if (it != doc.MemberEnd()) {
        int boundary_ID = 0;
        if (it->value.IsArray()) {
          fmt::print(fg(cyan), "Scene has [{}] boundaries.\n", it->value.Size());
          for (auto &model : it->value.GetArray()) {
            
            if (boundary_ID >= mn::config::g_max_grid_boundaries) {
              fmt::print(fg(red), "ERROR: Grid-boundary ID[{}] exceeds max[{}]. Increase g_max_grid_boundaries in settings.h and recompile.\n", boundary_ID, mn::config::g_max_grid_boundaries);
              if (mn::config::g_log_level >= 3) { 
                getchar();
              }
              continue;
            }

            mn::vec<float, 7> h_boundary;
            mn::GridBoundaryConfigs h_gridBoundary;
            // Load and scale target domain to 1 x 1 x 1 domain + off-by-2 offset
            for (int d = 0; d < 3; ++d) {
              h_boundary[d] = (model["domain_start"].GetArray()[d].GetFloat() * froude_scaling) / l + o;
              h_boundary[d+3] = (model["domain_end"].GetArray()[d].GetFloat() * froude_scaling) / l + o;
            }

            h_gridBoundary._ID = boundary_ID;
            h_gridBoundary._domain_start = CheckFloatArray(model, "domain_start", mn::vec<float, 3>{0.f,0.f,0.f});
            h_gridBoundary._domain_end = CheckFloatArray(model, "domain_end", mn::vec<float, 3>{1.f,1.f,1.f});
            h_gridBoundary._velocity = CheckFloatArray(model, "velocity", mn::vec<float, 3>{0.f,0.f,0.f});
            h_gridBoundary._array = CheckIntArray(model, "array", mn::vec<int, 3>{1,1,1});
            h_gridBoundary._spacing = CheckFloatArray(model, "spacing", mn::vec<float, 3>{0.f,0.f,0.f});
            for (int d = 0; d < 3; ++d) {
              h_gridBoundary._domain_start[d] *= froude_scaling;
              h_gridBoundary._domain_start[d] = h_gridBoundary._domain_start[d] / l + o;
              h_gridBoundary._domain_end[d] *= froude_scaling;
              h_gridBoundary._domain_end[d] = h_gridBoundary._domain_end[d] / l + o;
              h_gridBoundary._velocity[d] *= sqrt(froude_scaling);
              h_gridBoundary._velocity[d] = h_gridBoundary._velocity[d] / l;
              h_gridBoundary._spacing[d] *= froude_scaling;
              h_gridBoundary._spacing[d] = h_gridBoundary._spacing[d] / l;
            }
            
            // Time range the boundary is active
            // ! WARNING: Last index of _time is deprecated but still required, should fix later.
            // e.g. [0, 1, 0] is valid but [0, 1] is not
            mn::vec<float, 3> float_time;
            float_time[0] = (float)time[0];
            float_time[1] = (float)time[1];
            float_time[2] = (float)(-1);
            h_gridBoundary._time = CheckFloatArray(model, "time", float_time);
            // Adjust for froude scaling if not using default
            // TODO: check if "time" is in scene file for grid-boundary instead of comparing against default value
            if ((h_gridBoundary._time[0] != float_time[0]) || (h_gridBoundary._time[1] != float_time[1])) {
              for (int d=0; d<2; ++d) h_gridBoundary._time[d] *= sqrt(froude_scaling);
            }
            // Boundary object and contact type
            std::string object = CheckString(model, "object", std::string{"box"});
            std::string contact = CheckString(model, "contact", std::string{"Sticky"});

            if ((contact == "Sticky") || (contact == "sticky") || (contact == "Stick") || (contact == "stick") || (contact == "Rigid") || (contact == "rigid") || (contact == "No-Slip") || (contact == "no-slip") || (contact == "Fixed") || (contact == "fixed") || (contact == "Fix") || (contact == "fix"))
              h_gridBoundary._contact = mn::config::boundary_contact_t::Sticky;
            else if ((contact == "Slip") || (contact == "slip") || (contact == "Roller") || (contact == "roller")) 
              h_gridBoundary._contact = mn::config::boundary_contact_t::Slip;
            else if ( (contact == "Separable" || contact == "separable") || (contact == "Seperable" || contact == "seperable") || (contact == "Separate" || contact == "separate") || (contact == "Separatable" || contact == "separatable") )
              h_gridBoundary._contact = mn::config::boundary_contact_t::Separate;
            else {
              fmt::print(fg(red),"ERROR: Invalid contact[{}] set for grid-boundary[{}]! Try sticky, slip, or separable... \n", contact, boundary_ID); 
              if ((mn::config::g_log_level < 3) && (boundary_ID > 0)) {
                // We check boundary_ID > 0 as it is absolutely required that the first boundary be correctly set-up, no skipping.
                fmt::print(fg(orange),"WARNING: Skipping grid-boundary[{}]... \n", boundary_ID);
                continue;
              } 
              fmt::print(fg(orange),"WARNING: Press ENTER to continue anyways... \n");
              if (boundary_ID == 0) fmt::print(fg(red),"ERROR: First grid-boundary[ID = {}] must be correctly set-up in input scene file or simulation crashes! Important! \n", boundary_ID);
              getchar();
            }
            
            // Set boundary object type if valid
            if ((object == "Wall") || (object == "wall") || (object == "Walls") || (object == "walls")) {
              h_gridBoundary._object = mn::config::boundary_object_t::Walls;
              if (h_gridBoundary._contact == mn::config::boundary_contact_t::Sticky) h_boundary[6] = 0;
              else if (h_gridBoundary._contact == mn::config::boundary_contact_t::Slip) h_boundary[6] = 1;
              else if (h_gridBoundary._contact == mn::config::boundary_contact_t::Separate) h_boundary[6] = 2;
              else h_boundary[6] = -1; // TODO: Throw error here. -1 value signifies invalid contact type but is deprecated.

              // Write to OBJ file
              std::string fn_gb = "gridBoundary[" + std::to_string(boundary_ID) + "].obj";
              writeBoxOBJ(fn_gb, h_gridBoundary._domain_start, h_gridBoundary._domain_end - h_gridBoundary._domain_start);
            }
            else if ((object == "Box") || (object == "box") || (object == "Cube") || (object == "cube")) {
              h_gridBoundary._object = mn::config::boundary_object_t::Box;
              if (h_gridBoundary._contact == mn::config::boundary_contact_t::Sticky) h_boundary[6] = 3;
              else if (h_gridBoundary._contact == mn::config::boundary_contact_t::Slip) h_boundary[6] = 4;
              else if (h_gridBoundary._contact == mn::config::boundary_contact_t::Separate) h_boundary[6] = 5;
              else h_boundary[6] = -1;

              // Write to OBJ file. Easily visualized in Blender/Houdini/etc.
              std::string fn_gb = "gridBoundary[" + std::to_string(boundary_ID) + "].obj";
              writeBoxOBJ(fn_gb, h_gridBoundary._domain_start, h_gridBoundary._domain_end - h_gridBoundary._domain_start);
            }
            else if ((object == "Sphere") || (object == "sphere") || (object == "Ball") || (object == "ball")) {
              h_gridBoundary._object = mn::config::boundary_object_t::Sphere;
              if (h_gridBoundary._contact == mn::config::boundary_contact_t::Sticky) h_boundary[6] = 6;
              else if (h_gridBoundary._contact == mn::config::boundary_contact_t::Slip) h_boundary[6] = 7;
              else if (h_gridBoundary._contact == mn::config::boundary_contact_t::Separate) h_boundary[6] = 8;
              else h_boundary[6] = -1;
            }
            else if ((object == "OSU LWF") || (object == "OSU Flume") || (object == "OSU_LWF") || (object == "OSU_LWF_RAMP") || (object == "OSU LWF Ramp") || (object == "OSU LWF Bathymetry")) {
              h_gridBoundary._object = mn::config::boundary_object_t::OSU_LWF_RAMP;
              if (h_gridBoundary._contact == mn::config::boundary_contact_t::Sticky) h_boundary[6] = 9;
              else if (h_gridBoundary._contact == mn::config::boundary_contact_t::Slip) h_boundary[6] = 10;
              else if (h_gridBoundary._contact == mn::config::boundary_contact_t::Separate) h_boundary[6] = 11;
              else h_boundary[6] = -1;
            }
            else if ((object == "OSU Paddle") || (object == "OSU LWF Paddle") || (object == "OSU LWF Piston") || (object == "OSU Wave Maker") || (object == "OSU_LWF_PADDLE")) {
              h_gridBoundary._object = mn::config::boundary_object_t::OSU_LWF_PADDLE;
              if (h_gridBoundary._contact == mn::config::boundary_contact_t::Sticky) h_boundary[6] = 12;
              else if (h_gridBoundary._contact == mn::config::boundary_contact_t::Slip) h_boundary[6] = 13;
              else if (h_gridBoundary._contact == mn::config::boundary_contact_t::Separate) h_boundary[6] = 14;
              else h_boundary[6] = -1;
            }            
            else if ((object == "Cylinder") || (object == "cylinder")) {
              h_gridBoundary._object = mn::config::boundary_object_t::Cylinder;
              if (h_gridBoundary._contact == mn::config::boundary_contact_t::Sticky) h_boundary[6] = 15;
              else if (h_gridBoundary._contact == mn::config::boundary_contact_t::Slip) h_boundary[6] = 16;
              else if (h_gridBoundary._contact == mn::config::boundary_contact_t::Separate) h_boundary[6] = 17;
              // Default to a Z+ oriented cylinder
              else h_boundary[6] = -1;
            }
            else if ((object == "Plane") || (object == "plane") || (object == "Surface") || (object == "surface")) {
              h_gridBoundary._object = mn::config::boundary_object_t::Plane;
              if (h_gridBoundary._contact == mn::config::boundary_contact_t::Sticky) h_boundary[6] = 18;
              else if (h_gridBoundary._contact == mn::config::boundary_contact_t::Slip) h_boundary[6] = 19;
              else if (h_gridBoundary._contact == mn::config::boundary_contact_t::Separate) h_boundary[6] = 20;
              else h_boundary[6] = -1;
            }
            else if ((object == "Bathymetry") || (object == "bathymetry") || (object == "bath") || (object == "Bath") || (object == "BATH")) {
              h_gridBoundary._object = mn::config::boundary_object_t::BATHYMETRY;
              h_boundary[6] = 94;
            }
            else if ((object == "USGS Ramp") || (object == "USGS Flume") || (object == "USGS DFF Ramp") ||  (object == "USGS_RAMP")) {
              h_gridBoundary._object = mn::config::boundary_object_t::USGS_RAMP;
              h_boundary[6] = 95; 
            }
            else if ((object == "USGS Gate") || (object == "USGS Gates") || (object == "USGS DFF Gate") || (object == "USGS DFF Gates") || (object == "USGS GATE") || (object == "USGS_GATE")) {
              h_gridBoundary._object = mn::config::boundary_object_t::USGS_GATE;
              h_boundary[6] = 96; 
            }
            else if ((object == "OSU TWB Ramp") || (object == "OSU TWB Bathymetry") || (object == "OSU_TWB_RAMP") || (object == "OSU TWB")) {
              h_gridBoundary._object = mn::config::boundary_object_t::OSU_TWB_RAMP;
              h_boundary[6] = 97; 
            }
            else if ((object == "OSU TWB Paddle") || (object == "OSU TWB Piston") || (object == "OSU TWB Wave Maker") || (object == "OSU_TWB_PADDLE") || (object == "OSU DWB Paddle")) {
              h_gridBoundary._object = mn::config::boundary_object_t::OSU_TWB_PADDLE;
              h_boundary[6] = 98; 
            }       
            else if ((object == "WASIRF Pump") || (object == "UW WASIRF Pump") ||  (object == "WASIRF_PUMP")) {
              h_gridBoundary._object = mn::config::boundary_object_t::WASIRF_PUMP;
              h_boundary[6] = 99; 
            }
            else if ((object == "TOKYO Harbor") || (object == "WU TWB Harbor") || (object == "Waseda Harbor") || (object == "TOKYO_HARBOR")) {
              h_gridBoundary._object = mn::config::boundary_object_t::TOKYO_HARBOR;
              h_boundary[6] = 100;
            } else if ((object == "Velocity" ) || (object == "velocity")) {
              // TODO : Implement both constant and moving velocity grid-boundaries
              h_gridBoundary._object = mn::config::boundary_object_t::VELOCITY_BOUNDARY;
              h_boundary[6] = 101;
            } else if ((object == "Floor") || (object == "floor")) {
              h_gridBoundary._object = mn::config::boundary_object_t::FLOOR;
              h_boundary[6] = 102;
            } else {
              h_boundary[6] = -1;
              fmt::print(fg(red), "ERROR: gridBoundary[{}] object[{}][{}] or contact[{}][{}] is not valid! hb6[{}] \n", boundary_ID, object,  h_gridBoundary._object, contact, h_gridBoundary._contact, h_boundary[6]);
            }

            // Set up grid-boundary friction
            {
              h_gridBoundary._friction_static = CheckFloat(model,"friction_static", static_cast<PREC_G>(0.0));
              PREC_G temp_friction = std::max(h_gridBoundary._friction_static, static_cast<PREC_G>(0.0));
              h_gridBoundary._friction_dynamic = CheckFloat(model,"friction_dynamic", temp_friction);
            }

            // Check for bathymetry coordinates (optional)
            auto use_custom_bathymetry = CheckBool(geometry, "use_custom_bathymetry", false);
            if (use_custom_bathymetry || (object == "Bathymetry") || (object == "bathymetry")) {
              auto bathymetry = model.FindMember("bathymetry"); // [[x1,y1],[x2,y2],...]
              if (bathymetry != model.MemberEnd()) {
                fmt::print(fg(cyan),"INFO: Found bathymetry key for grid-boundary[{}]. Value should be nested array of form: [ [x0,y0], [x1,y1], ..., [xn,yn] ]. Loading... \n", boundary_ID);
                int bathymetry_size = static_cast<int>(model["bathymetry"].GetArray().Size());
                if (bathymetry_size > mn::config::g_max_bathymetry_points) {
                  fmt::print(fg(red),"ERROR: Too many bathymetry coordinates[{}] for grid-boundary[{}]. Max allowed is [{}].\n", bathymetry_size, boundary_ID, mn::config::g_max_bathymetry_points);
                  if (mn::config::g_log_level >= 3) getchar();
                }
                h_gridBoundary._num_bathymetry_points = static_cast<int>(bathymetry_size);
                fmt::print(fg(cyan),"INFO: Found [{}] bathymetry coordinates for grid-boundary[{}]! Max allowable: [32] \n", h_gridBoundary._num_bathymetry_points, boundary_ID); 

                int bathymetry_i = 0;
                for (auto &coord : model["bathymetry"].GetArray()) {
                  if (bathymetry_i >= mn::config::g_max_bathymetry_points) break;
                  mn::vec<PREC_G, 2> xy;
                  for (int d = 0; d < 2; ++d) {
                    xy[d] = coord.GetArray()[d].GetFloat();
                  }
                  h_gridBoundary._bathymetry_points[bathymetry_i][0] = xy[0] * froude_scaling / l + o;
                  h_gridBoundary._bathymetry_points[bathymetry_i][1] = xy[1] * froude_scaling / l + o;
                  fmt::print(fg(green),"Bathymetry[{}/{}]: Input [{}, {}], Scaled [{}, {}]\n", bathymetry_i, h_gridBoundary._num_bathymetry_points, xy[0], xy[1], h_gridBoundary._bathymetry_points[bathymetry_i][0], h_gridBoundary._bathymetry_points[bathymetry_i][1]);
                  bathymetry_i++;
                }
                // h_gridBoundary._bathymetry_points = bathymetry_coords;
              } else {
                h_gridBoundary._num_bathymetry_points = 0;
                fmt::print(fg(yellow),"NOTE: No custom bathymetry points set for grid-boundary[{}].\n", boundary_ID);
              }
            }

            // Set up moving grid-boundary if applicable
            auto motion_file = model.FindMember("file"); // Check for motion file
            auto motion_velocity = model.FindMember("velocity"); // Check for velocity
            if (motion_file != model.MemberEnd() && motion_velocity == model.MemberEnd()) 
            {
              fmt::print(fg(cyan),"Found motion file for grid-boundary[{}]. Loading... \n", boundary_ID);
              MotionHolder motionPath;
              // std::string motion_fn = std::string(AssetDirPath) + model["file"].GetString();
              std::string motion_fn = std::string("./") + model["file"].GetString();
              fs::path motion_file_path{motion_fn};
              if (motion_file_path.empty()) fmt::print(fg(red), "ERROR: Input file[{}] does not exist.\n", motion_fn);
              else {
                std::ifstream istrm(motion_fn);
                if (!istrm.is_open())  fmt::print(fg(red), "ERROR: Cannot open file[{}]\n", motion_fn);
                istrm.close();
              }


              PREC_G mp_freq = CheckFloat(model, "output_frequency", 1.0);
              mp_freq *= 1 / sqrt(froude_scaling);

              load_motionPath(motion_fn, ',', motionPath,  1, froude_scaling);

              for (int did = 0; did < mn::config::g_device_cnt; ++did) {
                benchmark->initMotionPath(did, motionPath, mp_freq);
                fmt::print(fg(green),"GPU[{}] gridBoundary[{}] motion file[{}] initialized with frequency[{}].\n", did, boundary_ID, model["file"].GetString(), mp_freq);
              }
            }
            // TODO : Fully implement constant velocity grid boundaries
            else if (motion_velocity != model.MemberEnd() && motion_file == model.MemberEnd()) {
              // Currently loading in velocity and adjusting it for scaling near the top of the gridBoundary loop
              fmt::print(fg(cyan),"Found initial velocity for grid-boundary[{}]. Loading...\n", boundary_ID);
              // mn::vec<PREC_G, 3> velocity;
              // h_gridBoundary._velocity = CheckFloatArray(model, "velocity", mn::vec<float, 3>{0.f,0.f,0.f});
              // for (int d=0; d<3; d++) velocity[d] = (model["velocity"].GetArray()[d].GetDouble() * sqrt(froude_scaling)) / l; // Already did this earlier
              // for (int d=0; d<3; d++) h_gridBoundary._velocity[d] = velocity[d];
            }
            else 
              fmt::print(fg(yellow),"NOTE: No velocity or motion-file set for grid-boundary[{}]. Assuming static...\n", boundary_ID);
            
            // ----------------  Initialize grid-boundaries ---------------- 
            benchmark->initGridBoundaries(0, h_boundary, h_gridBoundary, boundary_ID);
            fmt::print(fg(green), "Initialized gridBoundary[{}]: object[{}], contact[{}].\n", boundary_ID, object, contact);
            fmt::print(fmt::emphasis::bold,
                      "----------------------------------------------------------------\n");
            boundary_ID += 1;
          }
        }
      }
    }
  }
} ///< End scene file parsing

#endif
