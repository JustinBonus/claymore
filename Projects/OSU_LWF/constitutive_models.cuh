#ifndef __CONSTITUTIVE_MODELS_CUH_
#define __CONSTITUTIVE_MODELS_CUH_
#include <MnBase/Math/Matrix/svd.cuh>
#include <MnBase/Math/Matrix/MatrixUtils.h>
#include <MnBase/Math/Vec.h>
#include "settings.h"

namespace mn {

// Note: the following Von Mises yield criterion is not at all validated, nor is it what you probably expect
// Basically it was a quick hack to allow easily controllable yielding based on the Von Mises yield circle
// If the deviatoric stress exceeds the yield strength on the Pi plane, it is scaled back to the yield circle.
// This is not physically what is standard, but it is a quick way to get something working.
// Typically you would want a bilateral model with a tangent modulus, etc., not hard to implement if you have the time.
// Here we throttle stress to disallow transgression of the yield circle using the tensile_yield_strength as user-input.
// Because stress is throttle, the forces produced aren't able to resist all deformation so we get plasticity. 
template <typename T = double>
__forceinline__ __device__ void
compute_stress_vonmises(T volume, T mu, T lambda, T tensile_yield_strength, const vec<T, 9> &F,
                            vec<T, 9> &PF) {

  // Von Mises Cauchy stress. https://en.wikipedia.org/wiki/Von_Mises_yield_criterion
  // Von mises strength, sigma_v = Tensile yield strength, sigma_y = (yield stress in pure shear, k) * sqrt(3) = sqrt(J2, the second invariant of the deviatoric stress tensor) * sqrt(3)
  // yield stress in pure shear, k = sqrt(second invariant of the deviatoric stress tensor, J2)
  // (von mises strength)^2 = (1/2) * [(sigma11 - sigma22)^2 + (sigma22 - sigma33)^2 + (sigma33 - sigma11)^2 + 6 * (sigma12^2 + sigma23^2 + sigma31^2)] = (3/2) sij sij, where s is the deviatoric stress
  // Radius of yield circle on deviatoric plane = sqrt(3/2) * (tensile yield strength, sigma_y) = sqrt(2) * (yield stress in pure shear, k)
  vec<T, 9> Finv;
  matrixInverse(F.data(), Finv.data());

  T J = matrixDeterminant3d(F.data());
  T logJ = log(J);

  // P  = mu * (F - F^-T) + lambda * log(J) * F^-T
  vec<T, 9> P;
  P[0] = mu * (F[0] - Finv[0]) + lambda * (J - 1) * J * Finv[0];
  P[1] = mu * (F[1] - Finv[3]) + lambda * (J - 1) * J * Finv[3];
  P[2] = mu * (F[2] - Finv[6]) + lambda * (J - 1) * J * Finv[6];
  P[3] = mu * (F[3] - Finv[1]) + lambda * (J - 1) * J * Finv[1];
  P[4] = mu * (F[4] - Finv[4]) + lambda * (J - 1) * J * Finv[4];
  P[5] = mu * (F[5] - Finv[7]) + lambda * (J - 1) * J * Finv[7];
  P[6] = mu * (F[6] - Finv[2]) + lambda * (J - 1) * J * Finv[2];
  P[7] = mu * (F[7] - Finv[5]) + lambda * (J - 1) * J * Finv[5];
  P[8] = mu * (F[8] - Finv[8]) + lambda * (J - 1) * J * Finv[8];

  /// Cauchy = J^-1 PK1 F^T
  PF[0] = (P[0] * F[0] + P[3] * F[3] + P[6] * F[6]); // May need to check J
  PF[1] = (P[1] * F[0] + P[4] * F[3] + P[7] * F[6]);
  PF[2] = (P[2] * F[0] + P[5] * F[3] + P[8] * F[6]);
  PF[3] = (P[0] * F[1] + P[3] * F[4] + P[6] * F[7]);
  PF[4] = (P[1] * F[1] + P[4] * F[4] + P[7] * F[7]);
  PF[5] = (P[2] * F[1] + P[5] * F[4] + P[8] * F[7]);
  PF[6] = (P[0] * F[2] + P[3] * F[5] + P[6] * F[8]);
  PF[7] = (P[1] * F[2] + P[4] * F[5] + P[7] * F[8]);
  PF[8] = (P[2] * F[2] + P[5] * F[5] + P[8] * F[8]);

  // Ensure does not exceed yield strength
  T yield_strength = sqrt(3.0 / 2.0) * tensile_yield_strength;
  T stress_squared = (0.5) * ( (PF[0] - PF[4]) * (PF[0] - PF[4]) +
                              (PF[4] - PF[8]) * (PF[4] - PF[8]) +
                              (PF[8] - PF[0]) * (PF[8] - PF[0]) +
                              6.0 * (PF[1] * PF[1] + PF[2] * PF[2] + PF[5] * PF[5]));
  
  T deviatoric_stress_magnitude = stress_squared > 0.0 ? sqrt(stress_squared) : 0.0;

  T scale = volume;
  if (deviatoric_stress_magnitude > yield_strength) {
    // Scale back to yield strength
    // Note: maybe just apply to deviatoric stress, not total stress. But this material model isn't intended to be permanent regardless. To be replaced
    scale *= yield_strength / deviatoric_stress_magnitude;
  }
  PF[0] *= scale;
  PF[1] *= scale;
  PF[2] *= scale;
  PF[3] *= scale;
  PF[4] *= scale;
  PF[5] *= scale;
  PF[6] *= scale;
  PF[7] *= scale;
  PF[8] *= scale;

}

// TODO: This is for FixedCorotated, not VonMises
template <typename T = double>
__forceinline__ __device__ void
compute_stress_PK1_vonmises(T volume, T mu, T lambda, const vec<T, 9> &F,
                              vec<T, 9> &P) {
  T U[9], S[3], V[9];
  math::svd(F[0], F[3], F[6], F[1], F[4], F[7], F[2], F[5], F[8], U[0], U[3],
            U[6], U[1], U[4], U[7], U[2], U[5], U[8], S[0], S[1], S[2], V[0],
            V[3], V[6], V[1], V[4], V[7], V[2], V[5], V[8]);
  T J = S[0] * S[1] * S[2];
  T scaled_mu = 2.0 * mu;
  T scaled_lambda = lambda * (J - 1.0);
  vec<T, 3> P_hat;
  P_hat[0] = scaled_mu * (S[0] - 1.0) + scaled_lambda * (S[1] * S[2]);
  P_hat[1] = scaled_mu * (S[1] - 1.0) + scaled_lambda * (S[0] * S[2]);
  P_hat[2] = scaled_mu * (S[2] - 1.0) + scaled_lambda * (S[0] * S[1]);

  P[0] =
      (P_hat[0] * U[0] * V[0] + P_hat[1] * U[3] * V[3] + P_hat[2] * U[6] * V[6]) * volume;
  P[1] =
      (P_hat[0] * U[1] * V[0] + P_hat[1] * U[4] * V[3] + P_hat[2] * U[7] * V[6]) * volume;
  P[2] =
      (P_hat[0] * U[2] * V[0] + P_hat[1] * U[5] * V[3] + P_hat[2] * U[8] * V[6]) * volume;
  P[3] =
      (P_hat[0] * U[0] * V[1] + P_hat[1] * U[3] * V[4] + P_hat[2] * U[6] * V[7]) * volume;
  P[4] =
      (P_hat[0] * U[1] * V[1] + P_hat[1] * U[4] * V[4] + P_hat[2] * U[7] * V[7]) * volume;
  P[5] =
      (P_hat[0] * U[2] * V[1] + P_hat[1] * U[5] * V[4] + P_hat[2] * U[8] * V[7]) * volume;
  P[6] =
      (P_hat[0] * U[0] * V[2] + P_hat[1] * U[3] * V[5] + P_hat[2] * U[6] * V[8]) * volume;
  P[7] =
      (P_hat[0] * U[1] * V[2] + P_hat[1] * U[4] * V[5] + P_hat[2] * U[7] * V[8]) * volume;
  P[8] =
      (P_hat[0] * U[2] * V[2] + P_hat[1] * U[5] * V[5] + P_hat[2] * U[8] * V[8]) * volume;
}

// TODO: This is for FixedCorotated, not VonMises
template <typename T = double>
__forceinline__ __device__ void
compute_energy_vonmises(T volume, T mu, T lambda, const vec<T, 9> &F,
                              T &strain_energy) {
  T U[9], S[3], V[9];
  math::svd(F[0], F[3], F[6], F[1], F[4], F[7], F[2], F[5], F[8], U[0], U[3],
            U[6], U[1], U[4], U[7], U[2], U[5], U[8], S[0], S[1], S[2], V[0],
            V[3], V[6], V[1], V[4], V[7], V[2], V[5], V[8]);
  T J = S[0] * S[1] * S[2];

  // Fixed-corotated potential strain energy. Page 90 UCLA MPM course Jiang et al.
  strain_energy = (mu * ((S[0] - 1.0)*(S[0] - 1.0) + (S[1] - 1.0)*(S[1] - 1.0) +
                        (S[2] - 1.0)*(S[2] - 1.0)) + 0.5 * lambda*(J - 1.0)*(J - 1.0)) * volume;
}

// template <typename T = double>
// __forceinline__ __device__ void
// compute_stress_PK1_vonmises(T volume, T mu, T lambda, const vec<T, 9> &F,
//                             vec<T, 9> &P) {
//   // Von Mises Cauchy stress. Page 90 UCLA MPM course Jiang et al.
//   vec<T, 9> Finv;
//   matrixInverse(F.data(), Finv.data());

//   T J = matrixDeterminant3d(F.data());
//   T logJ = log(J);

//   // P  = mu * (F - F^-T) + lambda * log(J) * F^-T
//   P[0] = mu * (F[0] - Finv[0]) + lambda * logJ * Finv[0] * volume;
//   P[1] = mu * (F[1] - Finv[3]) + lambda * logJ * Finv[3] * volume;
//   P[2] = mu * (F[2] - Finv[6]) + lambda * logJ * Finv[6] * volume;
//   P[3] = mu * (F[3] - Finv[1]) + lambda * logJ * Finv[1] * volume;
//   P[4] = mu * (F[4] - Finv[4]) + lambda * logJ * Finv[4] * volume;
//   P[5] = mu * (F[5] - Finv[7]) + lambda * logJ * Finv[7] * volume;
//   P[6] = mu * (F[6] - Finv[2]) + lambda * logJ * Finv[2] * volume;
//   P[7] = mu * (F[7] - Finv[5]) + lambda * logJ * Finv[5] * volume;
//   P[8] = mu * (F[8] - Finv[8]) + lambda * logJ * Finv[8] * volume;
// }

// template <typename T = double>
// __forceinline__ __device__ void
// compute_energy_vonmises(T volume, T mu, T lambda, const vec<T, 9> &F,
//                         T &strain_energy) {
//   // Von Mises strain potential energy. Page 90 UCLA MPM course Jiang et al.
//   // ABAQUS uses a slightly different formulation.
//   T J = matrixDeterminant3d(F.data());
//   T logJ = log(J);

//   T C[3]; //< Left Cauchy Green Diagonal, F^T F
//   // tr(F'F)
//   C[0] = (F[0] * F[0] + F[1] * F[1] + F[2] * F[2]) ;
//   C[1] = (F[3] * F[3] + F[4] * F[4] + F[5] * F[5]) ;
//   C[2] = (F[6] * F[6] + F[7] * F[7] + F[8] * F[8]) ;
//   strain_energy = (mu * (0.5*((C[0] + C[1] + C[2]) - 3) - logJ) + 0.5*lambda*logJ*logJ) * volume;
// }


/// * J-Fluid - Pressure, Stress, and Energy
/// * Isotropic Tait-Murnaghan fluid, uses the deformation gradient determinant J
template <typename T = double>
__forceinline__ __device__ void
compute_pressure_jfluid(T volume, T bulk, T bulk_wrt_pressure, T sJ, T &pressure)
{
  //pressure =  (bulk / bulk_wrt_pressure) * (  pow(J, -bulk_wrt_pressure) - 1.0 );
  pressure = (bulk / bulk_wrt_pressure) * expm1(-bulk_wrt_pressure*log1p(-sJ));
}
template <>
__forceinline__ __device__ void
compute_pressure_jfluid(float volume, float bulk, float bulk_wrt_pressure, float J, float &pressure)
{
  pressure =  (bulk / bulk_wrt_pressure) * (  powf(J, -bulk_wrt_pressure) - 1.f );
}
template <typename T = double>
__forceinline__ __device__ void
compute_stress_jfluid(T volume, T bulk, T bulk_wrt_pressure, T Dp_inv, T viscosity, T J, const vec<T, 9> &C, vec<T, 9> &PF)
{
  T pressure =  (bulk / bulk_wrt_pressure) * (  pow(J, -bulk_wrt_pressure) - 1.0 );
  {
    PF[0] = ((C[0] + C[0]) * Dp_inv * viscosity - pressure) * volume;
    PF[1] = (C[1] + C[3]) * Dp_inv * viscosity * volume;
    PF[2] = (C[2] + C[6]) * Dp_inv * viscosity * volume;
    PF[3] = (C[3] + C[1]) * Dp_inv * viscosity * volume;
    PF[4] = ((C[4] + C[4]) * Dp_inv * viscosity - pressure) * volume;
    PF[5] = (C[5] + C[7]) * Dp_inv * viscosity * volume;
    PF[6] = (C[6] + C[2]) * Dp_inv * viscosity * volume;
    PF[7] = (C[7] + C[5]) * Dp_inv * viscosity * volume;
    PF[8] = ((C[8] + C[8]) * Dp_inv * viscosity - pressure) * volume;
  }
}

template <typename T = double>
__forceinline__ __device__ void
compute_stress_jfluid(T volume, T bulk, T bulk_wrt_pressure, T Dp_inv, T viscosity, T J, const vec<T, 9> &C, vec<T, 6> &PF)
{
  // Use symmetry of Cauchy stress to reduce size.
  // Array index 0, 1, 2, 3, 4, 5 -> 
  // | 11, 12, 13, |
  // |     22. 23, |
  // |         33  |
  T pressure =  (bulk / bulk_wrt_pressure) * (  pow(J, -bulk_wrt_pressure) - 1.0 );
  {
    PF[0] = ((C[0] + C[0]) * Dp_inv * viscosity - pressure) * volume;

    PF[1] = (C[3] + C[1]) * Dp_inv * viscosity * volume;
    PF[2] = ((C[4] + C[4]) * Dp_inv * viscosity - pressure) * volume;

    PF[3] = (C[6] + C[2]) * Dp_inv * viscosity * volume;
    PF[4] = (C[7] + C[5]) * Dp_inv * viscosity * volume;
    PF[5] = ((C[8] + C[8]) * Dp_inv * viscosity - pressure) * volume;
  }
}

template <typename T = double>
__forceinline__ __device__ void
compute_energy_jfluid( T volume, T bulk, T bulk_wrt_pressure, T J, T& strain_energy)
{
  // Based on Pradhana 2017 Multi-Species MPM paper. Modified pressure constant term for gamma
  // Energy(J) = -(bulk/bulk_wrt_pressure) * ( ( J^(1 - bulk_wrt_pressure)  / (1 - bulk_wrt_pressure) ) - J) * volume?
  // J = det | Deformation Gradient | = V / Vo = Ratio of volume change of particle
  // bulk = bulk modulus [Pa] (2.2e9 Pa for water at 20 deg C, sea-level, no salinity)
  // bulk_wrt_pressure = bulk modulus derivative with respect to pressure at atmospheric conditions (7.1 for water)
  // strain_energy = [Joules]
  // A better Tait-Murnaghan energy formulation is probabaly:
  // E = Eo + k * volume * ( (1 / (g(g-1)))*J^(1-g) + (1/g)*J - (1/(1-g)) )
  // Note that assuming Eo = 0 (i.e. no strain energy at ambient pressure without deformation)
  // for J = 1 (no volume change) there is no energy (good)
  // gamma cannot be 0 or 1 in this model (1 is common in graphics). Maybe impossible to go under 5/3 (Thomas-Fermi limit)
//   strain_energy = - volume * (bulk / bulk_wrt_pressure) * 
                    // ( ( pow(J, (1.0 - bulk_wrt_pressure))  / (1.0 - bulk_wrt_pressure) ) - J );
  T one_minus_bwp = 1.0 - bulk_wrt_pressure;
  strain_energy = volume * bulk * 
                    ((1.0/(bulk_wrt_pressure*(bulk_wrt_pressure-1.0))) * pow(J, one_minus_bwp) + (1.0/bulk_wrt_pressure)*J - (1.0/(bulk_wrt_pressure-1.0)));
}

template <>
__forceinline__ __device__ void
compute_energy_jfluid(float volume, float bulk, float bulk_wrt_pressure, float J, float& strain_energy)
{
  float one_minus_bwp = 1.f - bulk_wrt_pressure;
  strain_energy = volume * bulk * 
                    ((1.f/(bulk_wrt_pressure*(bulk_wrt_pressure-1.0))) * powf(J, one_minus_bwp) + (1.f/bulk_wrt_pressure)*J - (1.f/(bulk_wrt_pressure-1.f)));
}


/// * Fixed-Corotated - Stress and Energy
/// * Hyperelastic solid model. Similar to NeoHookean, popular in graphics.
/// TODO : Force derivative
template <typename T = double>
__forceinline__ __device__ void
compute_stress_fixedcorotated(T volume, T mu, T lambda, const vec<T, 9> &F,
                              vec<T, 9> &PF) {
  T U[9], S[3], V[9];
  math::svd(F[0], F[3], F[6], F[1], F[4], F[7], F[2], F[5], F[8], U[0], U[3],
            U[6], U[1], U[4], U[7], U[2], U[5], U[8], S[0], S[1], S[2], V[0],
            V[3], V[6], V[1], V[4], V[7], V[2], V[5], V[8]);
  T J = S[0] * S[1] * S[2];
  T scaled_mu = 2.0 * mu;
  T scaled_lambda = lambda * (J - 1.0);
  vec<T, 3> P_hat;
  P_hat[0] = scaled_mu * (S[0] - 1.0) + scaled_lambda * (S[1] * S[2]);
  P_hat[1] = scaled_mu * (S[1] - 1.0) + scaled_lambda * (S[0] * S[2]);
  P_hat[2] = scaled_mu * (S[2] - 1.0) + scaled_lambda * (S[0] * S[1]);

  vec<T, 9> P;
  P[0] =
      P_hat[0] * U[0] * V[0] + P_hat[1] * U[3] * V[3] + P_hat[2] * U[6] * V[6];
  P[1] =
      P_hat[0] * U[1] * V[0] + P_hat[1] * U[4] * V[3] + P_hat[2] * U[7] * V[6];
  P[2] =
      P_hat[0] * U[2] * V[0] + P_hat[1] * U[5] * V[3] + P_hat[2] * U[8] * V[6];
  P[3] =
      P_hat[0] * U[0] * V[1] + P_hat[1] * U[3] * V[4] + P_hat[2] * U[6] * V[7];
  P[4] =
      P_hat[0] * U[1] * V[1] + P_hat[1] * U[4] * V[4] + P_hat[2] * U[7] * V[7];
  P[5] =
      P_hat[0] * U[2] * V[1] + P_hat[1] * U[5] * V[4] + P_hat[2] * U[8] * V[7];
  P[6] =
      P_hat[0] * U[0] * V[2] + P_hat[1] * U[3] * V[5] + P_hat[2] * U[6] * V[8];
  P[7] =
      P_hat[0] * U[1] * V[2] + P_hat[1] * U[4] * V[5] + P_hat[2] * U[7] * V[8];
  P[8] =
      P_hat[0] * U[2] * V[2] + P_hat[1] * U[5] * V[5] + P_hat[2] * U[8] * V[8];

  /// PF'
  PF[0] = (P[0] * F[0] + P[3] * F[3] + P[6] * F[6]) * volume;
  PF[1] = (P[1] * F[0] + P[4] * F[3] + P[7] * F[6]) * volume;
  PF[2] = (P[2] * F[0] + P[5] * F[3] + P[8] * F[6]) * volume;
  PF[3] = (P[0] * F[1] + P[3] * F[4] + P[6] * F[7]) * volume;
  PF[4] = (P[1] * F[1] + P[4] * F[4] + P[7] * F[7]) * volume;
  PF[5] = (P[2] * F[1] + P[5] * F[4] + P[8] * F[7]) * volume;
  PF[6] = (P[0] * F[2] + P[3] * F[5] + P[6] * F[8]) * volume;
  PF[7] = (P[1] * F[2] + P[4] * F[5] + P[7] * F[8]) * volume;
  PF[8] = (P[2] * F[2] + P[5] * F[5] + P[8] * F[8]) * volume;
}

template <typename T = double>
__forceinline__ __device__ void
compute_stress_PK1_fixedcorotated(T volume, T mu, T lambda, const vec<T, 9> &F,
                              vec<T, 9> &P) {
  T U[9], S[3], V[9];
  math::svd(F[0], F[3], F[6], F[1], F[4], F[7], F[2], F[5], F[8], U[0], U[3],
            U[6], U[1], U[4], U[7], U[2], U[5], U[8], S[0], S[1], S[2], V[0],
            V[3], V[6], V[1], V[4], V[7], V[2], V[5], V[8]);
  T J = S[0] * S[1] * S[2];
  T scaled_mu = 2.0 * mu;
  T scaled_lambda = lambda * (J - 1.0);
  vec<T, 3> P_hat;
  P_hat[0] = scaled_mu * (S[0] - 1.0) + scaled_lambda * (S[1] * S[2]);
  P_hat[1] = scaled_mu * (S[1] - 1.0) + scaled_lambda * (S[0] * S[2]);
  P_hat[2] = scaled_mu * (S[2] - 1.0) + scaled_lambda * (S[0] * S[1]);

  P[0] =
      (P_hat[0] * U[0] * V[0] + P_hat[1] * U[3] * V[3] + P_hat[2] * U[6] * V[6]) * volume;
  P[1] =
      (P_hat[0] * U[1] * V[0] + P_hat[1] * U[4] * V[3] + P_hat[2] * U[7] * V[6]) * volume;
  P[2] =
      (P_hat[0] * U[2] * V[0] + P_hat[1] * U[5] * V[3] + P_hat[2] * U[8] * V[6]) * volume;
  P[3] =
      (P_hat[0] * U[0] * V[1] + P_hat[1] * U[3] * V[4] + P_hat[2] * U[6] * V[7]) * volume;
  P[4] =
      (P_hat[0] * U[1] * V[1] + P_hat[1] * U[4] * V[4] + P_hat[2] * U[7] * V[7]) * volume;
  P[5] =
      (P_hat[0] * U[2] * V[1] + P_hat[1] * U[5] * V[4] + P_hat[2] * U[8] * V[7]) * volume;
  P[6] =
      (P_hat[0] * U[0] * V[2] + P_hat[1] * U[3] * V[5] + P_hat[2] * U[6] * V[8]) * volume;
  P[7] =
      (P_hat[0] * U[1] * V[2] + P_hat[1] * U[4] * V[5] + P_hat[2] * U[7] * V[8]) * volume;
  P[8] =
      (P_hat[0] * U[2] * V[2] + P_hat[1] * U[5] * V[5] + P_hat[2] * U[8] * V[8]) * volume;
}

template <typename T = double>
__forceinline__ __device__ void
compute_energy_fixedcorotated(T volume, T mu, T lambda, const vec<T, 9> &F,
                              T &strain_energy) {
  T U[9], S[3], V[9];
  math::svd(F[0], F[3], F[6], F[1], F[4], F[7], F[2], F[5], F[8], U[0], U[3],
            U[6], U[1], U[4], U[7], U[2], U[5], U[8], S[0], S[1], S[2], V[0],
            V[3], V[6], V[1], V[4], V[7], V[2], V[5], V[8]);
  T J = S[0] * S[1] * S[2];

  // Fixed-corotated potential strain energy. Page 90 UCLA MPM course Jiang et al.
  strain_energy = (mu * ((S[0] - 1.0)*(S[0] - 1.0) + (S[1] - 1.0)*(S[1] - 1.0) +
                        (S[2] - 1.0)*(S[2] - 1.0)) + 0.5 * lambda*(J - 1.0)*(J - 1.0)) * volume;
}
template <>
__forceinline__ __device__ void
compute_energy_fixedcorotated(float volume, float mu, float lambda, const vec<float, 9> &F,
                              float &strain_energy) {
  float U[9], S[3], V[9];
  math::svd(F[0], F[3], F[6], F[1], F[4], F[7], F[2], F[5], F[8], U[0], U[3],
            U[6], U[1], U[4], U[7], U[2], U[5], U[8], S[0], S[1], S[2], V[0],
            V[3], V[6], V[1], V[4], V[7], V[2], V[5], V[8]);
  float J = S[0] * S[1] * S[2];

  // Fixed-corotated potential strain energy. Page 90 UCLA MPM course Jiang et al.
  strain_energy = (mu * ((S[0] - 1.f)*(S[0] - 1.f) + (S[1] - 1.f)*(S[1] - 1.f) +
                        (S[2] - 1.f)*(S[2] - 1.f)) + 0.5f * lambda*(J - 1.f)*(J - 1.f)) * volume;
}


/// * Neo-Hookean - Stress and Energy
/// * Hyperelastic model for solids. Abaqus uses a slighly different formulation.
template <typename T = double>
__forceinline__ __device__ void
compute_stress_neohookean(T volume, T mu, T lambda, const vec<T, 9> &F,
                             vec<T, 9> &PF)
{
  // Neo-hooken Cauchy stress. Page 90 UCLA MPM course Jiang et al.
  vec<T, 9> Finv;
  matrixInverse(F.data(), Finv.data());

  T J = matrixDeterminant3d(F.data());
  T logJ = log(J);

  // P  = mu * (F - F^-T) + lambda * log(J) * F^-T ??
  // First Piola Kirchoff Stress = (mu * (F - F^-T) + lambda * (J - 1) * J * F^-T)
  vec<T, 9> P;
  P[0] = mu * (F[0] - Finv[0]) + lambda * (J - 1) * J * Finv[0];
  P[1] = mu * (F[1] - Finv[3]) + lambda * (J - 1) * J * Finv[3];
  P[2] = mu * (F[2] - Finv[6]) + lambda * (J - 1) * J * Finv[6];
  P[3] = mu * (F[3] - Finv[1]) + lambda * (J - 1) * J * Finv[1];
  P[4] = mu * (F[4] - Finv[4]) + lambda * (J - 1) * J * Finv[4];
  P[5] = mu * (F[5] - Finv[7]) + lambda * (J - 1) * J * Finv[7];
  P[6] = mu * (F[6] - Finv[2]) + lambda * (J - 1) * J * Finv[2];
  P[7] = mu * (F[7] - Finv[5]) + lambda * (J - 1) * J * Finv[5];
  P[8] = mu * (F[8] - Finv[8]) + lambda * (J - 1) * J * Finv[8];

  /// Cauchy = J^-1 PK1 F^T

  PF[0] = (P[0] * F[0] + P[3] * F[3] + P[6] * F[6]) * volume; // May need to check J
  PF[1] = (P[1] * F[0] + P[4] * F[3] + P[7] * F[6]) * volume;
  PF[2] = (P[2] * F[0] + P[5] * F[3] + P[8] * F[6]) * volume;
  PF[3] = (P[0] * F[1] + P[3] * F[4] + P[6] * F[7]) * volume;
  PF[4] = (P[1] * F[1] + P[4] * F[4] + P[7] * F[7]) * volume;
  PF[5] = (P[2] * F[1] + P[5] * F[4] + P[8] * F[7]) * volume;
  PF[6] = (P[0] * F[2] + P[3] * F[5] + P[6] * F[8]) * volume;
  PF[7] = (P[1] * F[2] + P[4] * F[5] + P[7] * F[8]) * volume;
  PF[8] = (P[2] * F[2] + P[5] * F[5] + P[8] * F[8]) * volume;
}

template <typename T = double>
__forceinline__ __device__ void
compute_stress_PK1_neohookean(T volume, T mu, T lambda, const vec<T, 9> &F,
                             vec<T, 9> &P)
{
  // Neo-hooken Cauchy stress. Page 90 UCLA MPM course Jiang et al.
  vec<T, 9> Finv;
  matrixInverse(F.data(), Finv.data());

  T J = matrixDeterminant3d(F.data());
  T logJ = log(J);

  // P  = mu * (F - F^-T) + lambda * log(J) * F^-T
  P[0] = mu * (F[0] - Finv[0]) + lambda * logJ * Finv[0] * volume;
  P[1] = mu * (F[1] - Finv[3]) + lambda * logJ * Finv[3] * volume;
  P[2] = mu * (F[2] - Finv[6]) + lambda * logJ * Finv[6] * volume;
  P[3] = mu * (F[3] - Finv[1]) + lambda * logJ * Finv[1] * volume;
  P[4] = mu * (F[4] - Finv[4]) + lambda * logJ * Finv[4] * volume;
  P[5] = mu * (F[5] - Finv[7]) + lambda * logJ * Finv[7] * volume;
  P[6] = mu * (F[6] - Finv[2]) + lambda * logJ * Finv[2] * volume;
  P[7] = mu * (F[7] - Finv[5]) + lambda * logJ * Finv[5] * volume;
  P[8] = mu * (F[8] - Finv[8]) + lambda * logJ * Finv[8] * volume;
}

template <typename T = double>
__forceinline__ __device__ void
compute_energy_neohookean(T volume, T mu, T lambda, const vec<T, 9> &F,
                             T &strain_energy)
{
  // Neo-hooken strain potential energy. Page 90 UCLA MPM course Jiang et al.
  // ABAQUS uses a slightly different formulation.
  T J = matrixDeterminant3d(F.data());
  T logJ = log(J);

  T C[3]; //< Left Cauchy Green Diagonal, F^T F
  // tr(F'F)
  C[0] = (F[0] * F[0] + F[1] * F[1] + F[2] * F[2]) ;
  C[1] = (F[3] * F[3] + F[4] * F[4] + F[5] * F[5]) ;
  C[2] = (F[6] * F[6] + F[7] * F[7] + F[8] * F[8]) ;
  strain_energy = (mu * (0.5*((C[0] + C[1] + C[2]) - 3) - logJ) + 0.5*lambda*logJ*logJ) * volume;
}

/// Drucker-Prager - Stress and Energy
/// Granular materials.
template <typename T = double>
__forceinline__ __device__ void
compute_energy_sand(T volume, T mu, T lambda, T cohesion, T beta,
                    T yieldSurface, bool volCorrection, T logJp, vec<T, 9> &F, T &strain_energy) {
  T U[9], S[3], V[9];
  // ψs(Fs) = ψ˜s (ϵ) = µtr(ϵ^2) + λ/2 tr(ϵ) 
  // Modified code from stress graphics version, need to recheck it
  // https://www.math.ucla.edu/~jteran/papers/PGKFTJM17.pdf  T U[9], S[3], V[9];
  math::svd(F[0], F[3], F[6], F[1], F[4], F[7], F[2], F[5], F[8], U[0], U[3],
            U[6], U[1], U[4], U[7], U[2], U[5], U[8], S[0], S[1], S[2], V[0],
            V[3], V[6], V[1], V[4], V[7], V[2], V[5], V[8]);

  T epsilon[3]; ///< logarithmic strain
  // 'Cohesion' uses a strange definition to account for wet sand tensile effects
#pragma unroll 3
  for (int i = 0; i < 3; i++) 
  {
    T abs_S = S[i] > 0 ? S[i] : -S[i];
    abs_S = abs_S > 1e-4 ? abs_S : 1e-4;
    epsilon[i] = log(abs_S) - cohesion;
  }
  T sum_epsilon = epsilon[0] + epsilon[1] + epsilon[2];
  //T sum_epsilon_squared = epsilon[0]*epsilon[0] + epsilon[1]*epsilon[1] + epsilon[2]*epsilon[2];
  T trace_epsilon = sum_epsilon + logJp;
  T trace_epsilon_squared = sum_epsilon + logJp*logJp;
  strain_energy = (mu * trace_epsilon_squared + lambda * 0.5 * trace_epsilon) * volume;
}                  



template <typename T = double>
__forceinline__ __device__ void
compute_stress_sand(T volume, T mu, T lambda, T cohesion, T beta,
                    T yieldSurface, bool volCorrection, T &logJp, vec<T, 9> &F,
                    vec<T, 9> &PF) {
  T U[9], S[3], V[9];
  math::svd(F[0], F[3], F[6], F[1], F[4], F[7], F[2], F[5], F[8], U[0], U[3],
            U[6], U[1], U[4], U[7], U[2], U[5], U[8], S[0], S[1], S[2], V[0],
            V[3], V[6], V[1], V[4], V[7], V[2], V[5], V[8]);
  T scaled_mu = T(2) * mu;

  T epsilon[3], New_S[3]; ///< helper
  T New_F[9];

  constexpr T small = 1e-9;
#pragma unroll 3
  for (int i = 0; i < 3; i++) {
    T abs_S = S[i] > 0 ? S[i] : -S[i];
    abs_S = abs_S > small ? abs_S : small;
    epsilon[i] = log(abs_S) - cohesion;
  }
  T sum_epsilon = epsilon[0] + epsilon[1] + epsilon[2];
  T trace_epsilon = sum_epsilon + logJp;

  T epsilon_hat[3];
#pragma unroll 3
  for (int i = 0; i < 3; i++)
    epsilon_hat[i] = epsilon[i] - (trace_epsilon / (T)3);

  T epsilon_hat_norm =
      sqrt(epsilon_hat[0] * epsilon_hat[0] + epsilon_hat[1] * epsilon_hat[1] +
            epsilon_hat[2] * epsilon_hat[2]);

  /* Calculate Plasticiy */
  if (trace_epsilon >= (T)0) { ///< case II: project to the cone tip
    New_S[0] = New_S[1] = New_S[2] = exp(cohesion);
    matmul_mat_diag_matT_3D(New_F, U, New_S, V); // new F_e
                                                 /* Update F */
#pragma unroll 9
    for (int i = 0; i < 9; i++)
      F[i] = New_F[i];
    if (volCorrection) {
      logJp = beta * sum_epsilon + logJp;
    }
  } else if (mu != 0) {
    logJp = 0; // TODO: Should this be an else if statement for volCorrection == false above
    T delta_gamma = epsilon_hat_norm + ((T)3 * lambda + scaled_mu) / scaled_mu *
                                           trace_epsilon * yieldSurface;
    T H[3];
    if (delta_gamma <= 0) { ///< case I: inside the yield surface cone
#pragma unroll 3
      for (int i = 0; i < 3; i++)
        H[i] = epsilon[i] + cohesion;
    } else { ///< case III: project to the cone surface
#pragma unroll 3
      for (int i = 0; i < 3; i++)
        H[i] = epsilon[i] - (delta_gamma / epsilon_hat_norm) * epsilon_hat[i] +
               cohesion;
    }
#pragma unroll 3
    for (int i = 0; i < 3; i++)
      New_S[i] = exp(H[i]);
    matmul_mat_diag_matT_3D(New_F, U, New_S, V); // new F_e
                                                 /* Update F */
#pragma unroll 9
    for (int i = 0; i < 9; i++)
      F[i] = New_F[i];
  }

  /* Elasticity -- Calculate Coefficient */
  T New_S_log[3] = {log(New_S[0]), log(New_S[1]), log(New_S[2])};
  T P_hat[3];

  // T S_inverse[3] = {1.f/S[0], 1.f/S[1], 1.f/S[2]};  // TO CHECK
  // T S_inverse[3] = {1.f / New_S[0], 1.f / New_S[1], 1.f / New_S[2]}; // TO
  // CHECK
  T trace_log_S = New_S_log[0] + New_S_log[1] + New_S_log[2];
#pragma unroll 3
  for (int i = 0; i < 3; i++)
    P_hat[i] = (scaled_mu * New_S_log[i] + lambda * trace_log_S) / New_S[i];

  T P[9];
  matmul_mat_diag_matT_3D(P, U, P_hat, V);
  ///< |f| = P * F^T * Volume
  PF[0] = (P[0] * F[0] + P[3] * F[3] + P[6] * F[6]) * volume;
  PF[1] = (P[1] * F[0] + P[4] * F[3] + P[7] * F[6]) * volume;
  PF[2] = (P[2] * F[0] + P[5] * F[3] + P[8] * F[6]) * volume;
  PF[3] = (P[0] * F[1] + P[3] * F[4] + P[6] * F[7]) * volume;
  PF[4] = (P[1] * F[1] + P[4] * F[4] + P[7] * F[7]) * volume;
  PF[5] = (P[2] * F[1] + P[5] * F[4] + P[8] * F[7]) * volume;
  PF[6] = (P[0] * F[2] + P[3] * F[5] + P[6] * F[8]) * volume;
  PF[7] = (P[1] * F[2] + P[4] * F[5] + P[7] * F[8]) * volume;
  PF[8] = (P[2] * F[2] + P[5] * F[5] + P[8] * F[8]) * volume;
}



/// U-P, Drucker-Prager - Stress and Energy (JLM)
/// Granular materials.
template <typename T = double>
__forceinline__ __device__ void
compute_energy_CoupledUP(T volume, T mu, T lambda, T cohesion, T beta,
                    T yieldSurface, bool volCorrection, T logJp, vec<T, 9> &F, T &strain_energy) {
  T U[9], S[3], V[9];
  // ψs(Fs) = ψ˜s (ϵ) = µtr(ϵ^2) + λ/2 tr(ϵ) 
  // Modified code from stress graphics version, need to recheck it
  // https://www.math.ucla.edu/~jteran/papers/PGKFTJM17.pdf  T U[9], S[3], V[9];
  math::svd(F[0], F[3], F[6], F[1], F[4], F[7], F[2], F[5], F[8], U[0], U[3],
            U[6], U[1], U[4], U[7], U[2], U[5], U[8], S[0], S[1], S[2], V[0],
            V[3], V[6], V[1], V[4], V[7], V[2], V[5], V[8]);

  T epsilon[3]; ///< logarithmic strain
  // 'Cohesion' uses a strange definition to account for wet sand tensile effects
#pragma unroll 3
  for (int i = 0; i < 3; i++) 
  {
    T abs_S = S[i] > 0 ? S[i] : -S[i];
    abs_S = abs_S > 1e-4 ? abs_S : 1e-4;
    epsilon[i] = log(abs_S) - cohesion;
  }
  T sum_epsilon = epsilon[0] + epsilon[1] + epsilon[2];
  //T sum_epsilon_squared = epsilon[0]*epsilon[0] + epsilon[1]*epsilon[1] + epsilon[2]*epsilon[2];
  T trace_epsilon = sum_epsilon + logJp;
  T trace_epsilon_squared = sum_epsilon + logJp*logJp;
  strain_energy = (mu * trace_epsilon_squared + lambda * 0.5 * trace_epsilon) * volume;
}          


template <typename T = double>
__forceinline__ __device__ void
compute_stress_CoupledUP(T volume, T mu, T lambda, T cohesion, T beta,
                    T yieldSurface, bool volCorrection, T &logJp, T &pw, 
                    vec<T, 9> &F, vec<T, 9> &PF) {
  T U[9], S[3], V[9];
  math::svd(F[0], F[3], F[6], F[1], F[4], F[7], F[2], F[5], F[8], U[0], U[3],
            U[6], U[1], U[4], U[7], U[2], U[5], U[8], S[0], S[1], S[2], V[0],
            V[3], V[6], V[1], V[4], V[7], V[2], V[5], V[8]);
  T scaled_mu = T(2) * mu;

  T epsilon[3], New_S[3]; ///< helper
  T New_F[9];

#pragma unroll 3
  for (int i = 0; i < 3; i++) {
    T abs_S = S[i] > 0 ? S[i] : -S[i];
    abs_S = abs_S > 1e-4 ? abs_S : 1e-4;
    epsilon[i] = log(abs_S) - cohesion;
  }
  T sum_epsilon = epsilon[0] + epsilon[1] + epsilon[2];
  T trace_epsilon = sum_epsilon + logJp;

  T epsilon_hat[3];
#pragma unroll 3
  for (int i = 0; i < 3; i++)
    epsilon_hat[i] = epsilon[i] - (trace_epsilon / (T)3);

  T epsilon_hat_norm =
      sqrt(epsilon_hat[0] * epsilon_hat[0] + epsilon_hat[1] * epsilon_hat[1] +
            epsilon_hat[2] * epsilon_hat[2]);

  /* Calculate Plasticiy */
  if (trace_epsilon >= (T)0) { ///< case II: project to the cone tip
    New_S[0] = New_S[1] = New_S[2] = exp(cohesion);
    matmul_mat_diag_matT_3D(New_F, U, New_S, V); // new F_e
                                                 /* Update F */
#pragma unroll 9
    for (int i = 0; i < 9; i++)
      F[i] = New_F[i];
    if (volCorrection) {
      logJp = beta * sum_epsilon + logJp;
    }
  } else if (mu != 0) {
    logJp = 0;
    T delta_gamma = epsilon_hat_norm + ((T)3 * lambda + scaled_mu) / scaled_mu *
                                           trace_epsilon * yieldSurface;
    T H[3];
    if (delta_gamma <= 0) { ///< case I: inside the yield surface cone
#pragma unroll 3
      for (int i = 0; i < 3; i++)
        H[i] = epsilon[i] + cohesion;
    } else { ///< case III: project to the cone surface
#pragma unroll 3
      for (int i = 0; i < 3; i++)
        H[i] = epsilon[i] - (delta_gamma / epsilon_hat_norm) * epsilon_hat[i] +
               cohesion;
    }
#pragma unroll 3
    for (int i = 0; i < 3; i++)
      New_S[i] = exp(H[i]);
    matmul_mat_diag_matT_3D(New_F, U, New_S, V); // new F_e
                                                 /* Update F */
#pragma unroll 9
    for (int i = 0; i < 9; i++)
      F[i] = New_F[i];
  }

  /* Elasticity -- Calculate Coefficient */
  T New_S_log[3] = {log(New_S[0]), log(New_S[1]), log(New_S[2])};
  T P_hat[3];

  // T S_inverse[3] = {1.f/S[0], 1.f/S[1], 1.f/S[2]};  // TO CHECK
  // T S_inverse[3] = {1.f / New_S[0], 1.f / New_S[1], 1.f / New_S[2]}; // TO
  // CHECK
  T trace_log_S = New_S_log[0] + New_S_log[1] + New_S_log[2];
#pragma unroll 3
  for (int i = 0; i < 3; i++)
    P_hat[i] = (scaled_mu * New_S_log[i] + lambda * trace_log_S) / New_S[i];

  T P[9];
  matmul_mat_diag_matT_3D(P, U, P_hat, V);
  ///< |f| = P * F^T * Volume 
  PF[0] = (P[0] * F[0] + P[3] * F[3] + P[6] * F[6] ) * volume;
  PF[1] = (P[1] * F[0] + P[4] * F[3] + P[7] * F[6] ) * volume;
  PF[2] = (P[2] * F[0] + P[5] * F[3] + P[8] * F[6] ) * volume;
  PF[3] = (P[0] * F[1] + P[3] * F[4] + P[6] * F[7]) * volume;
  PF[4] = (P[1] * F[1] + P[4] * F[4] + P[7] * F[7] ) * volume;
  PF[5] = (P[2] * F[1] + P[5] * F[4] + P[8] * F[7]) * volume;
  PF[6] = (P[0] * F[2] + P[3] * F[5] + P[6] * F[8]) * volume;
  PF[7] = (P[1] * F[2] + P[4] * F[5] + P[7] * F[8]) * volume;
  PF[8] = (P[2] * F[2] + P[5] * F[5] + P[8] * F[8] ) * volume;
}

/// * Non-Associative Cam-Clay - Stress and Energy
/// * Elasto-plastic model. Good for snow, clay, concrete, etc.
template <typename T = float>
__forceinline__ __device__ void
compute_stress_nacc(T volume, T mu, T lambda, T bm, T xi, T beta, T Msqr,
                    bool hardeningOn, T &logJp, vec<T, 9> &F, vec<T, 9> &PF) {
  T U[9], S[3], V[9];
  math::svd(F[0], F[3], F[6], F[1], F[4], F[7], F[2], F[5], F[8], U[0], U[3],
            U[6], U[1], U[4], U[7], U[2], U[5], U[8], S[0], S[1], S[2], V[0],
            V[3], V[6], V[1], V[4], V[7], V[2], V[5], V[8]);
  T p0 = bm * (T(0.00001) + sinh(xi * (-logJp > 0 ? -logJp : 0)));
  T p_min = -beta * p0;

  T Je_trial = S[0] * S[1] * S[2];

  ///< 0). Calculate Yield Surface
  T B_hat_trial[3] = {S[0] * S[0], S[1] * S[1], S[2] * S[2]};
  T trace_B_hat_trial_divdim =
      (B_hat_trial[0] + B_hat_trial[1] + B_hat_trial[2]) / 3.f;
  T J_power_neg_2_d_mulmu =
      mu * powf(Je_trial, -2.f / 3.f); ///< J^(-2/dim) * mu
  T s_hat_trial[3] = {
      J_power_neg_2_d_mulmu * (B_hat_trial[0] - trace_B_hat_trial_divdim),
      J_power_neg_2_d_mulmu * (B_hat_trial[1] - trace_B_hat_trial_divdim),
      J_power_neg_2_d_mulmu * (B_hat_trial[2] - trace_B_hat_trial_divdim)};
  T psi_kappa_partial_J = bm * 0.5f * (Je_trial - 1.f / Je_trial);
  T p_trial = -psi_kappa_partial_J * Je_trial;

  T y_s_half_coeff = 3.f / 2.f * (1 + 2.f * beta); ///< a
  T y_p_half = (Msqr * (p_trial - p_min) * (p_trial - p0));
  T s_hat_trial_sqrnorm = s_hat_trial[0] * s_hat_trial[0] +
                          s_hat_trial[1] * s_hat_trial[1] +
                          s_hat_trial[2] * s_hat_trial[2];
  T y = (y_s_half_coeff * s_hat_trial_sqrnorm) + y_p_half;

  //< 1). update strain and hardening alpha(in logJp)

  ///< case 1, project to max tip of YS
  if (p_trial > p0) {
    T Je_new = sqrtf(-2.f * p0 / bm + 1.f);
    S[0] = S[1] = S[2] = powf(Je_new, 1.f / 3.f);
    T New_F[9];
    matmul_mat_diag_matT_3D(New_F, U, S, V);
#pragma unroll 9
    for (int i = 0; i < 9; i++)
      F[i] = New_F[i];
    if (hardeningOn)
      logJp += logf(Je_trial / Je_new);
  } ///< case 1 -- end

  /// case 2, project to min tip of YS
  else if (p_trial < p_min) {
    T Je_new = sqrtf(-2.f * p_min / bm + 1.f);
    S[0] = S[1] = S[2] = powf(Je_new, 1.f / 3.f);
    T New_F[9];
    matmul_mat_diag_matT_3D(New_F, U, S, V);
#pragma unroll 9
    for (int i = 0; i < 9; i++)
      F[i] = New_F[i];
    if (hardeningOn)
      logJp += logf(Je_trial / Je_new);
  } ///< case 2 -- end

  /// case 3, keep or project to YS by hardening
  else {
    ///< outside YS
    if (y >= 1e-4) {
      ////< yield surface projection
      T B_s_coeff = powf(Je_trial, 2.f / 3.f) / mu *
                    sqrtf(-y_p_half / y_s_half_coeff) /
                    sqrtf(s_hat_trial_sqrnorm);
#pragma unroll 3
      for (int i = 0; i < 3; i++)
        S[i] = sqrt(s_hat_trial[i] * B_s_coeff + trace_B_hat_trial_divdim);
      T New_F[9];
      matmul_mat_diag_matT_3D(New_F, U, S, V);
#pragma unroll 9
      for (int i = 0; i < 9; i++)
        F[i] = New_F[i];

      ////< hardening
      if (hardeningOn && p0 > 1e-4 && p_trial < p0 - 1e-4 &&
          p_trial > 1e-4 + p_min) {
        T p_center = ((T)1 - beta) * p0 / 2;
#if 1 /// solve in 19 Josh Fracture paper
        T q_trial = sqrt(1.5f * s_hat_trial_sqrnorm);
        T direction[2] = {p_center - p_trial, -q_trial};
        T direction_norm =
            sqrt(direction[0] * direction[0] + direction[1] * direction[1]);
        direction[0] /= direction_norm;
        direction[1] /= direction_norm;

        T C = Msqr * (p_center - p_min) * (p_center - p0);
        T B = Msqr * direction[0] * (2 * p_center - p0 - p_min);
        T A = Msqr * direction[0] * direction[0] +
              (1 + 2 * beta) * direction[1] * direction[1];

        T l1 = (-B + sqrt(B * B - 4 * A * C)) / (2 * A);
        T l2 = (-B - sqrt(B * B - 4 * A * C)) / (2 * A);

        T p1 = p_center + l1 * direction[0];
        T p2 = p_center + l2 * direction[0];
#else /// solve in ziran - Compare_With_Physbam
        T aa = Msqr * powf(p_trial - p_center, 2) /
               (y_s_half_coeff * s_hat_trial_sqrnorm);
        T dd = 1 + aa;
        T ff = aa * beta * p0 - aa * p0 - 2 * p_center;
        T gg = (p_center * p_center) - aa * beta * (p0 * p0);
        T zz = sqrtf(fabsf(ff * ff - 4 * dd * gg));
        T p1 = (-ff + zz) / (2 * dd);
        T p2 = (-ff - zz) / (2 * dd);
#endif

        T p_fake = (p_trial - p_center) * (p1 - p_center) > 0 ? p1 : p2;
        T tmp_Je_sqr = (-2 * p_fake / bm + 1);
        T Je_new_fake = sqrtf(tmp_Je_sqr > 0 ? tmp_Je_sqr : -tmp_Je_sqr);
        if (Je_new_fake > 1e-4)
          logJp += logf(Je_trial / Je_new_fake);
      }
    } ///< outside YS -- end
  }   ///< case 3 --end

  //< 2). elasticity
  ///< known: F(renewed), U, V, S(renewed)
  ///< unknown: J, dev(FF^T)
  T J = S[0] * S[1] * S[2];
  T b_dev[9], b[9];
  matrixMatrixTranposeMultiplication3d(F.data(), b);
  matrixDeviatoric3d(b, b_dev);

  ///< |f| = P * F^T * Volume
  T dev_b_coeff = mu * powf(J, -2.f / 3.f);
  T i_coeff = bm * .5f * (J * J - 1.f);
  PF[0] = (dev_b_coeff * b_dev[0] + i_coeff) * volume;
  PF[1] = (dev_b_coeff * b_dev[1]) * volume;
  PF[2] = (dev_b_coeff * b_dev[2]) * volume;
  PF[3] = (dev_b_coeff * b_dev[3]) * volume;
  PF[4] = (dev_b_coeff * b_dev[4] + i_coeff) * volume;
  PF[5] = (dev_b_coeff * b_dev[5]) * volume;
  PF[6] = (dev_b_coeff * b_dev[6]) * volume;
  PF[7] = (dev_b_coeff * b_dev[7]) * volume;
  PF[8] = (dev_b_coeff * b_dev[8] + i_coeff) * volume;
}

template <>
__forceinline__ __device__ void
compute_stress_nacc(double volume, double mu, double lambda, double bm, double xi, double beta, double Msqr,
                    bool hardeningOn, double &logJp, vec<double, 9> &F, vec<double, 9> &PF) {
  using T = double;
  T U[9], S[3], V[9];
  math::svd(F[0], F[3], F[6], F[1], F[4], F[7], F[2], F[5], F[8], U[0], U[3],
            U[6], U[1], U[4], U[7], U[2], U[5], U[8], S[0], S[1], S[2], V[0],
            V[3], V[6], V[1], V[4], V[7], V[2], V[5], V[8]); 
  T p0 = bm * (T(0.00001) + sinh(xi * (-logJp > 0 ? -logJp : 0)));
  T p_min = -beta * p0;

  T Je_trial = S[0] * S[1] * S[2];

  ///< 0). calculate YS
  T B_hat_trial[3] = {S[0] * S[0], S[1] * S[1], S[2] * S[2]};
  T trace_B_hat_trial_divdim =
      (B_hat_trial[0] + B_hat_trial[1] + B_hat_trial[2]) / 3.0;
  T J_power_neg_2_d_mulmu =
      mu * rcbrt(Je_trial*Je_trial); ///< J^(-2/dim) * mu
  T s_hat_trial[3] = {
      J_power_neg_2_d_mulmu * (B_hat_trial[0] - trace_B_hat_trial_divdim),
      J_power_neg_2_d_mulmu * (B_hat_trial[1] - trace_B_hat_trial_divdim),
      J_power_neg_2_d_mulmu * (B_hat_trial[2] - trace_B_hat_trial_divdim)};
  T psi_kappa_partial_J = bm * 0.5 * (Je_trial - 1.0 / Je_trial);
  T p_trial = -psi_kappa_partial_J * Je_trial;

  T y_s_half_coeff = 1.5 * (1 + 2.0 * beta); ///< a
  T y_p_half = (Msqr * (p_trial - p_min) * (p_trial - p0));
  T s_hat_trial_sqrnorm = s_hat_trial[0] * s_hat_trial[0] +
                          s_hat_trial[1] * s_hat_trial[1] +
                          s_hat_trial[2] * s_hat_trial[2];
  T y = (y_s_half_coeff * s_hat_trial_sqrnorm) + y_p_half;

  //< 1). update strain and hardening alpha(in logJp)

  ///< case 1, project to max tip of YS
  if (p_trial > p0) {
    T Je_new = sqrt(-2.0 * p0 / bm + 1.0);
    S[0] = S[1] = S[2] = cbrt(Je_new);
    T New_F[9];
    matmul_mat_diag_matT_3D(New_F, U, S, V);
#pragma unroll 9
    for (int i = 0; i < 9; i++)
      F[i] = New_F[i];
    if (hardeningOn)
      logJp += log(Je_trial / Je_new);
  } ///< case 1 -- end

  /// case 2, project to min tip of YS
  else if (p_trial < p_min) {
    T Je_new = sqrt(-2.0 * p_min / bm + 1.0);
    S[0] = S[1] = S[2] = cbrt(Je_new);
    T New_F[9];
    matmul_mat_diag_matT_3D(New_F, U, S, V);
#pragma unroll 9
    for (int i = 0; i < 9; i++)
      F[i] = New_F[i];
    if (hardeningOn)
      logJp += log(Je_trial / Je_new);
  } ///< case 2 -- end

  /// case 3, keep or project to YS by hardening
  else {
    ///< outside YS
    if (y >= 1e-4) {
      ////< yield surface projection
      T B_s_coeff = cbrt(Je_trial*Je_trial) / mu *
                    sqrt(-y_p_half / y_s_half_coeff) /
                    sqrt(s_hat_trial_sqrnorm);
#pragma unroll 3
      for (int i = 0; i < 3; i++)
        S[i] = sqrt(s_hat_trial[i] * B_s_coeff + trace_B_hat_trial_divdim);
      T New_F[9];
      matmul_mat_diag_matT_3D(New_F, U, S, V);
#pragma unroll 9
      for (int i = 0; i < 9; i++)
        F[i] = New_F[i];

      ////< hardening
      if (hardeningOn && p0 > 1e-4 && p_trial < p0 - 1e-4 &&
          p_trial > 1e-4 + p_min) {
        T p_center = ((T)1 - beta) * p0 / 2;
#if 1 /// solve in 19 Josh Fracture paper
        T q_trial = sqrt(1.5 * s_hat_trial_sqrnorm);
        T direction[2] = {p_center - p_trial, -q_trial};
        T direction_norm =
            sqrt(direction[0] * direction[0] + direction[1] * direction[1]);
        direction[0] /= direction_norm;
        direction[1] /= direction_norm;

        T C = Msqr * (p_center - p_min) * (p_center - p0);
        T B = Msqr * direction[0] * (2 * p_center - p0 - p_min);
        T A = Msqr * direction[0] * direction[0] +
              (1 + 2 * beta) * direction[1] * direction[1];

        T l1 = (-B + sqrt(B * B - 4 * A * C)) / (2 * A);
        T l2 = (-B - sqrt(B * B - 4 * A * C)) / (2 * A);

        T p1 = p_center + l1 * direction[0];
        T p2 = p_center + l2 * direction[0];
#else /// solve in ziran - Compare_With_Physbam
        T aa = Msqr * (p_trial - p_center) * (p_trial - p_center)  /
               (y_s_half_coeff * s_hat_trial_sqrnorm);
        T dd = 1 + aa;
        T ff = aa * beta * p0 - aa * p0 - 2 * p_center;
        T gg = (p_center * p_center) - aa * beta * (p0 * p0);
        T zz = sqrt(abs(ff * ff - 4 * dd * gg));
        T p1 = (-ff + zz) / (2 * dd);
        T p2 = (-ff - zz) / (2 * dd);
#endif

        T p_fake = (p_trial - p_center) * (p1 - p_center) > 0 ? p1 : p2;
        T tmp_Je_sqr = (-2 * p_fake / bm + 1);
        T Je_new_fake = sqrt(tmp_Je_sqr > 0 ? tmp_Je_sqr : -tmp_Je_sqr);
        if (Je_new_fake > 1e-4)
          logJp += log(Je_trial / Je_new_fake);
      }
    } ///< outside YS -- end
  }   ///< case 3 --end

  //< 2). elasticity
  ///< known: F(renewed), U, V, S(renewed)
  ///< unknown: J, dev(FF^T)
  T J = S[0] * S[1] * S[2];
  T b_dev[9], b[9];
  matrixMatrixTranposeMultiplication3d(F.data(), b);
  matrixDeviatoric3d(b, b_dev);

  ///< |f| = P * F^T * Volume
  T dev_b_coeff = mu * rcbrt(J*J);
  T i_coeff = bm * 0.5 * (J * J - 1.0);
  PF[0] = (dev_b_coeff * b_dev[0] + i_coeff) * volume;
  PF[1] = (dev_b_coeff * b_dev[1]) * volume;
  PF[2] = (dev_b_coeff * b_dev[2]) * volume;
  PF[3] = (dev_b_coeff * b_dev[3]) * volume;
  PF[4] = (dev_b_coeff * b_dev[4] + i_coeff) * volume;
  PF[5] = (dev_b_coeff * b_dev[5]) * volume;
  PF[6] = (dev_b_coeff * b_dev[6]) * volume;
  PF[7] = (dev_b_coeff * b_dev[7]) * volume;
  PF[8] = (dev_b_coeff * b_dev[8] + i_coeff) * volume;
}


/// * Various continuum mechanics / matrix functions

template <typename T = double>
__forceinline__ __device__ void
compute_SVD_DefGrad(const vec<T, 9> &F,
                              vec<T, 9> &U, vec<T, 3> &S, vec<T, 9> &V) 
{
  math::svd(F[0], F[3], F[6], F[1], F[4], F[7], F[2], F[5], F[8], U[0], U[3],
            U[6], U[1], U[4], U[7], U[2], U[5], U[8], S[0], S[1], S[2], V[0],
            V[3], V[6], V[1], V[4], V[7], V[2], V[5], V[8]);
}


template <typename T = double>
__forceinline__ __device__ void
compute_DefGradRate_from_DefGrad_and_VelocityGrad(const vec<T, 9> &F,
                               const vec<T, 9> &C, vec<T, 9> &Fdot) 
{
  matrixTransposeMatrixMultiplication3d(F.data(), C.data(), Fdot.data());
}

template <typename T = double>
__forceinline__ __device__ void
compute_StressCauchy_from_DefGrad_and_StressPK1(const vec<T, 9> &F,
                               const vec<T, 9> &P, vec<T, 9> &C) 
{
  T J = matrixDeterminant3d(F.data());
  matrixMatrixTransposeMultiplication3d(P.data(), F.data(), C.data());
#pragma unroll 9
  for (int i = 0; i < 9; i++) C[i] = C[i] / J;
}

} // namespace mn

#endif