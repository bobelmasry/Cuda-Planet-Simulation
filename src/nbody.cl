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

    double2 total_force = (double2)(0.0, 0.0);

    // Compute net acceleration on body i
    for (int j = 0; j < num_bodies; ++j) {
        if (i == j) continue;
        // Compute the force exerted on body i by body j
        double2 r = positions[j] - pos_i;
        double dist_sqr = r.x * r.x + r.y * r.y;
        double force = G * masses[j] * masses[i] / dist_sqr;
        double angle = atan2(r.y, r.x);
        double fx = force * cos(angle);
        double fy = force * sin(angle);
        total_force.x += fx;
        total_force.y += fy;
    }

    // Update velocity and position
    vel_i += total_force / masses[i] * timestep;
    pos_i += vel_i * timestep;

    // Write back
    velocities[i] = vel_i;
    positions[i]  = pos_i;
}
