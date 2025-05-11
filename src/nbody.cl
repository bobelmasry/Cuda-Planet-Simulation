#pragma OPENCL EXTENSION cl_khr_fp64 : enable

__kernel void update_positions(
    __global double2* positions,      // x,y for each body
    __global double2* velocities,     // vx,vy for each body
    __global const double* masses,    // mass per body
    const double G,                   // gravitational constant
    const double timestep,            // Δt
    const int num_bodies
) {
    int i = get_global_id(0);
    double2 pos_i = positions[i];
    double2 vel_i = velocities[i];
    double2 acc = (double2)(0.0, 0.0);

    // Compute net acceleration on body i
    for (int j = 0; j < num_bodies; ++j) {
        if (i == j) continue;
        double2 r = positions[j] - pos_i;
        double dist_sqr = r.x * r.x + r.y * r.y + 1e-10;
        double inv_dist = native_rsqrt(dist_sqr);
        double force = G * masses[j] * inv_dist * inv_dist;
        acc += (force / masses[i]) * (r * inv_dist);
    }

    // Update velocity and position
    vel_i += acc * timestep;
    pos_i += vel_i * timestep;

    // Write back
    velocities[i] = vel_i;
    positions[i]  = pos_i;
}
