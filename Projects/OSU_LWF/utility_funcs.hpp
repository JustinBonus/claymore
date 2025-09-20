#ifndef __UTILITY_FUNCS_HPP_
#define __UTILITY_FUNCS_HPP_
#include "settings.cuh"

namespace mn {

/// assume p is already within kernel range [-1.5, 1.5]
template <typename T>
constexpr vec<T, 3> bspline_weight(T p) {
  vec<T, 3> dw{0.0, 0.0, 0.0};
  T d = p * config::g_dx_inv; ///< normalized offset
  dw[0] = 0.5 * (1.5 - d) * (1.5 - d);
  d -= 1.0;
  dw[1] = 0.75 - d * d;
  d = 0.5 + d;
  dw[2] = 0.5 * d * d;
  return dw;
}

constexpr vec3 bspline_weight(float p) {
	vec3 dw {0.0f, 0.0f, 0.0f};
	float d = p * config::g_dx_inv;///< normalized offset
	dw[0]	= 0.5f * (1.5f - d) * (1.5f - d);
	d -= 1.0f;
	dw[1] = 0.75f - d * d;
	d	  = 0.5f + d;
	dw[2] = 0.5f * d * d;
	return dw;
}

constexpr int dir_offset(ivec3 d) {
  return (d[0] + 1) * 9 + (d[1] + 1) * 3 + d[2] + 1;
}

constexpr int dir_offset(const std::array<int, 3>& d) {
	return (d[0] + 1) * 9 + (d[1] + 1) * 3 + d[2] + 1;
}

constexpr void dir_components(int dir, ivec3 &d) {
  d[2] = (dir % 3) - 1;
  d[1] = ((dir / 3) % 3) - 1;
  d[0] = ((dir / 9) % 3) - 1;
}

// constexpr void dir_components(int dir, std::array<int, 3>& d) {
// 	d[2] = (dir % 3) - 1;
// 	d[1] = ((dir / 3) % 3) - 1; 
// 	d[0] = ((dir / 9) % 3) - 1;
// }
//NOLINTEND(readability-magic-numbers) Magic numbers are formula-specific

template <typename T>
constexpr ivec3 get_block_id(const std::array<T, 3>& position) {
	return ivec3(static_cast<int>(std::lround(position[0] * static_cast<T>(config::g_dx_inv))), static_cast<int>(std::lround(position[1] * static_cast<T>(config::g_dx_inv))), static_cast<int>(std::lround(position[2] * static_cast<T>(config::g_dx_inv))));
}

constexpr ivec3 get_block_id(const std::array<double, 3>& position) {
	return ivec3(static_cast<int>(std::lround(position[0] * config::g_dx_inv_d)), static_cast<int>(std::lround(position[1] * config::g_dx_inv_d)), static_cast<int>(std::lround(position[2] * config::g_dx_inv_d)));
}

constexpr ivec3 get_block_id(const std::array<float, 3>& position) {
	return ivec3(static_cast<int>(std::lround(position[0] * config::g_dx_inv)), static_cast<int>(std::lround(position[1] * config::g_dx_inv)), static_cast<int>(std::lround(position[2] * config::g_dx_inv)));
}

constexpr float compute_dt(float maxVel, const float cur, const float next,
                           const float dtDefault) noexcept {
  if (next < cur)
    return 0.f;
  float dt = dtDefault;
  if (maxVel > 0. && (maxVel = config::g_dx * config::g_cfl / maxVel) < dtDefault)
    dt = maxVel;

  if (cur + dt >= next)
    dt = next - cur;
  else if ((maxVel = (next - cur) * 0.51) < dt)
    dt = maxVel;

  return dt;
}

//NOLINTBEGIN(readability-magic-numbers) Magic numbers are formula-specific
// constexpr Duration compute_dt(float max_vel, const Duration cur_time, const Duration next_time, const Duration dt_default) noexcept {
// 	//Choose dt such that particles with maximum velocity cannot move more than G_DX * CFL
// 	//This ensures CFL condition is satisfied
// 	Duration dt = dt_default;
// 	if(max_vel > 0.0f) {
// 		const Duration new_dt(config::g_dx * config::g_cfl / max_vel);
// 		dt = std::min(new_dt, dt);
// 	}

// 	//If next_time - cur_time is smaller as current dt, use this.
// 	dt = std::min(dt, next_time - cur_time);

// 	return dt;
// }
//NOLINTEND(readability-magic-numbers) Magic numbers are formula-specific

} // namespace mn

#endif