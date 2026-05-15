"""
Generate initial conditions for the N-body simulation and write them to a
binary file readable by main.cu.

File format:
  [int32:  n]
  [n * 7 * float32:  x  y  z  vx  vy  vz  mass]

Usage:
  python generate_ic.py [options]

  --dist      distribution: sphere | cube | disk  (default: sphere)
  -n          number of bodies                     (default: 10000)
  -R          cluster radius / half-side           (default: 0.3)
  --cx/cy/cz  cluster center                       (default: 0.5 0.5 0.5)
  --thickness disk thickness as fraction of R      (default: 0.05, disk only)
  --seed      random seed                          (default: 42)
  -o          output file                          (default: ic.bin)
"""

import argparse
import struct
import random
import math

def sample_in_unit_ball(rng: random.Random):
    while True:
        x = 2.0 * rng.random() - 1.0
        y = 2.0 * rng.random() - 1.0
        z = 2.0 * rng.random() - 1.0
        if x*x + y*y + z*z <= 1.0:
            return x, y, z


def generate_uniform_sphere(n, R, center, seed):
    rng = random.Random(seed)
    cx, cy, cz = center
    mass = 1.0 / n
    bodies = []
    for _ in range(n):
        ux, uy, uz = sample_in_unit_ball(rng)
        bodies.append((cx + R*ux, cy + R*uy, cz + R*uz,
                       0.0, 0.0, 0.0, mass))
    return bodies


def generate_uniform_cube(n, R, center, seed):
    """Uniform distribution inside a cube of half-side R."""
    rng = random.Random(seed)
    cx, cy, cz = center
    mass = 1.0 / n
    bodies = []
    for _ in range(n):
        ux = 2.0 * rng.random() - 1.0
        uy = 2.0 * rng.random() - 1.0
        uz = 2.0 * rng.random() - 1.0
        bodies.append((cx + R*ux, cy + R*uy, cz + R*uz,
                       0.0, 0.0, 0.0, mass))
    return bodies


def generate_disk(n, R, center, seed, thickness=0.05):
    """
    Thin rotating disk (galaxy-like) in the x-y plane.

    Positions: uniform in area up to radius R, small Gaussian spread in z
               (thickness * R is the vertical sigma).
    Velocities: circular velocity consistent with a uniform disk,
                v_circ(r) = sqrt(M_enclosed / r) = sqrt(r) / R,
                directed tangentially (counter-clockwise when viewed from +z).
    """
    rng = random.Random(seed)
    cx, cy, cz = center
    mass = 1.0 / n
    bodies = []
    for _ in range(n):
        # Uniform in area: r = R * sqrt(U) gives constant surface density
        r   = R * math.sqrt(rng.random())
        phi = 2.0 * math.pi * rng.random()
        z   = rng.gauss(0.0, thickness * R)

        x = cx + r * math.cos(phi)
        y = cy + r * math.sin(phi)

        # Circular speed for a uniform disk: v^2 = G*M_enclosed/r = r/R^2 (G=1, M_total=1)
        v_circ = math.sqrt(r) / R if r > 0 else 0.0
        vx = -v_circ * math.sin(phi)
        vy =  v_circ * math.cos(phi)

        bodies.append((x, y, cz + z, vx, vy, 0.0, mass))
    return bodies


def write_ic(filename, bodies):
    n = len(bodies)
    with open(filename, "wb") as f:
        f.write(struct.pack("i", n))
        for body in bodies:
            f.write(struct.pack("7f", *body))
    print(f"Wrote {n} bodies to '{filename}'")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Generate N-body initial conditions")
    p.add_argument("--dist",      type=str,   default="sphere",
                   choices=["sphere", "cube", "disk"],
                   help="initial distribution (default: sphere)")
    p.add_argument("-n",          type=int,   default=10000, help="number of bodies")
    p.add_argument("-R",          type=float, default=0.3,   help="cluster radius / half-side")
    p.add_argument("--cx",        type=float, default=0.5,   help="cluster center x")
    p.add_argument("--cy",        type=float, default=0.5,   help="cluster center y")
    p.add_argument("--cz",        type=float, default=0.5,   help="cluster center z")
    p.add_argument("--thickness", type=float, default=0.05,
                   help="disk vertical thickness as fraction of R (disk only, default: 0.05)")
    p.add_argument("--seed",      type=int,   default=42,    help="random seed")
    p.add_argument("-o",          type=str,   default="ic.bin", help="output file")
    args = p.parse_args()

    center = (args.cx, args.cy, args.cz)
    if args.dist == "disk":
        bodies = generate_disk(args.n, args.R, center, args.seed, args.thickness)
    elif args.dist == "cube":
        bodies = generate_uniform_cube(args.n, args.R, center, args.seed)
    else:
        bodies = generate_uniform_sphere(args.n, args.R, center, args.seed)
    write_ic(args.o, bodies)
