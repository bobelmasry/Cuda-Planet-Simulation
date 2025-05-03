# Planet Trajectory Simulation Using Linear Algebra and CUDA

## Overview

This project is a two-dimensional simulation of planetary motion based on Newtonian mechanics, developed as part of a university linear algebra course. It leverages GPU acceleration through **CUDA** to efficiently compute gravitational forces and simulate the interactions of multiple celestial bodies in real time.

The simulation applies core principles from linear algebra to calculate vector-based motion, while CUDA enables the parallel processing of large-scale force computations across bodies.

## Objectives

* Demonstrate the application of **vector algebra** in modeling gravitational systems.
* Leverage **CUDA** to parallelize force and motion computations for improved performance.
* Build a minimal custom physics engine without relying on high-level math or physics libraries.
* Visualize planetary motion using Python and GPU-computed results.

## Key Features

* GPU-accelerated computation of gravitational forces via CUDA
* Real-time simulation of multiple bodies interacting through gravity
* Accurate modeling using linear algebra operations (magnitude, normalization, dot products)
* Object-oriented design for scalability and maintainability
* Visualization with `pygame`

## Technologies Used

* Python 3.x
* CUDA (via `numba.cuda` or custom CUDA kernels in C++)
* `pygame` for real-time visualization
* Custom vector math library

## Team Members

* \[Aly Youssef]
* \[Amr Abdou]
* \[Omar Rabeh]

## Installation and Setup

1. **Clone the repository**

   ```bash
   git clone https://github.com/yourusername/planet-trajectory-cuda.git
   cd planet-trajectory-cuda
   ```

2. **Install dependencies**

   ```bash
   pip install pygame numba
   ```

3. **Ensure CUDA is installed**

   * NVIDIA GPU with CUDA support
   * CUDA Toolkit installed and correctly configured

4. **Run the simulation**

   ```bash
   python main.py
   ```

## File Structure

* `main.py`: Entry point; handles initialization and rendering
* `cuda_kernels.py`: CUDA-accelerated functions for computing gravitational forces
* `vector.py`: Custom implementation of 2D vector operations
* `planet.py`: Planet class containing properties and motion logic
* `constants.py`: Physical constants and simulation parameters

## Performance Notes

* Force calculations are offloaded to the GPU using CUDA, allowing for real-time performance with a large number of bodies.
* CPU-based fallback implementation is available for debugging or non-CUDA environments.

## Future Extensions

* Extend to 3D simulation with OpenGL-based visualization
* Use advanced integrators (e.g., Runge-Kutta) for improved numerical stability
* Implement collision detection and elastic body interaction

## References

* Tech with Tim, "Creating a Solar System Simulation with Python," YouTube, [Link](https://www.youtube.com/watch?v=WTLPmUHTPqo)
* CUDA Programming Guide, NVIDIA
