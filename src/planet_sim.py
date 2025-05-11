# planet_sim_opencl.py

import math
import pygame
import numpy as np
import pyopencl as cl

pygame.init()

# Window setup
WIDTH, HEIGHT = 1000, 1000
WIN = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Planet Simulation (with OpenCL)")

# Colors
WHITE     = (255, 255, 255)
YELLOW    = (255, 255,   0)
BLUE      = (100,149,237)
RED       = (188,  39,  50)
DARK_GREY = ( 80,  78,  81)
GREEN     = ( 34,139, 34)

# Astronomical units
AU = 149.6e6 * 1000
FONT = pygame.font.SysFont("comicsans", 16)

def format_meters_to_AU(meters):
    au = meters / AU
    return f"{au:.3f} AU"

class Planet:
    G = 6.67428e-11
    SCALE = 200 / AU      # 1 AU = 200 px
    TIMESTEP = 1200 * 24  # ~1 day

    def __init__(self, x, y, radius, color, mass, name):
        self.x, self.y       = x, y
        self.radius          = radius
        self.color           = color
        self.mass            = mass
        self.name            = name
        self.sun             = False
        self.orbit           = []
        self.perigee         = float('inf')
        self.apogee          = 0.0
        self.eccentricity    = 0.0
        self.x_vel = 0.0
        self.y_vel = 0.0

    def draw(self, win):
        x = self.x * self.SCALE + WIDTH/2
        y = self.y * self.SCALE + HEIGHT/2

        if len(self.orbit) > 2:
            pts = [
                (px * self.SCALE + WIDTH/2,
                 py * self.SCALE + HEIGHT/2)
                for px, py in self.orbit
            ]
            pygame.draw.lines(win, self.color, False, pts, 2)

        pygame.draw.circle(win, self.color, (int(x), int(y)), self.radius)
        if not self.sun:
            txt = FONT.render(
                f"{self.name} [peri: {format_meters_to_AU(self.perigee)}, apo: {format_meters_to_AU(self.apogee)}, e={self.eccentricity:.3f}]",
                True, GREEN
            )
            win.blit(txt, (x - txt.get_width()/2, y - txt.get_height()/2))

# ——— OpenCL setup ———
platform = cl.get_platforms()[0]
device   = platform.get_devices()[0]
ctx      = cl.Context([device])
queue    = cl.CommandQueue(ctx)

with open("nbody.cl", "r") as f:
    program = cl.Program(ctx, f.read()).build()

def opencl_update(planets):
    n = len(planets)
    # Prepare NumPy arrays
    pos  = np.array([[p.x, p.y] for p in planets], dtype=np.float64)
    vel  = np.array([[p.x_vel, p.y_vel] for p in planets], dtype=np.float64)
    mass = np.array([p.mass for p in planets], dtype=np.float64)

    # Create buffers
    mf = cl.mem_flags
    pos_buf  = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=pos)
    vel_buf  = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=vel)
    m_buf    = cl.Buffer(ctx, mf.READ_ONLY  | mf.COPY_HOST_PTR, hostbuf=mass)

    # Launch kernel
    program.update_positions(
        queue, (n,), None,
        pos_buf, vel_buf, m_buf,
        np.float64(Planet.G),
        np.float64(Planet.TIMESTEP),
        np.int32(n)
    )

    # Read results back
    cl.enqueue_copy(queue, pos, pos_buf)
    cl.enqueue_copy(queue, vel, vel_buf)
    queue.finish()

    # Update Python objects
    for i, p in enumerate(planets):
        p.x, p.y       = pos[i]
        p.x_vel, p.y_vel = vel[i]
        p.orbit.append((p.x, p.y))

# ——— Main loop ———
def main():
    clock = pygame.time.Clock()
    run   = True

    # Create bodies
    sun = Planet(0, 0, 30, YELLOW, 1.98892e30, 'Sun')
    sun.sun = True

    mercury = Planet(0.387 * AU, 0, 4, DARK_GREY, 3.30e23, 'Mercury')
    mercury.y_vel = -47.4e3

    venus = Planet(0.723 * AU, 0, 7, WHITE, 4.8685e24, 'Venus')
    venus.y_vel = -35.02e3

    earth = Planet(-1 * AU, 0, 8, BLUE, 5.9742e24, 'Earth')
    earth.y_vel = 29.783e3

    mars = Planet(-1.524 * AU, 0, 5, RED, 6.39e23, 'Mars')
    mars.y_vel = 24.077e3

    asteroid = Planet(0.72 * AU, 0, 5, GREEN, 4.8685, 'Asteroid')
    asteroid.y_vel = -22e3

    planets = [sun, mercury, venus, earth, mars, asteroid]

    while run:
        clock.tick(60)
        WIN.fill((0,0,0))

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False

        # Offload the heavy N-body step to OpenCL
        opencl_update(planets)

        for p in planets:
            p.draw(WIN)

        pygame.display.update()

    pygame.quit()

if __name__ == "__main__":
    main()
