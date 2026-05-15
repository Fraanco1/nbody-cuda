"""
Generate initial conditions for the N-body simulation and write them to a
binary file readable by main.cu.

File format:
  [int32:  n]
  [n * 7 * float32:  x  y  z  vx  vy  vz  mass]

Usage:
  python generate_ic.py [options]

  --dist   distribution: sphere | cube  (default: sphere)
  -n       number of bodies             (default: 10000)
  -R       cluster radius / half-side   (default: 0.3)
  --cx/cy/cz  cluster center            (default: 0.5 0.5 0.5)
  --seed   random seed                  (default: 42)
  -o       output file                  (default: ic.bin)
"""

import argparse
import struct
import random

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


def write_ic(filename, bodies):
    n = len(bodies)
    with open(filename, "wb") as f:
        f.write(struct.pack("i", n))
        for body in bodies:
            f.write(struct.pack("7f", *body))
    print(f"Wrote {n} bodies to '{filename}'")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Generate N-body initial conditions")
    p.add_argument("--dist", type=str,   default="sphere", choices=["sphere", "cube"],
                   help="initial distribution (default: sphere)")
    p.add_argument("-n",     type=int,   default=10000, help="number of bodies")
    p.add_argument("-R",     type=float, default=0.3,   help="cluster radius / cube half-side")
    p.add_argument("--cx",   type=float, default=0.5,   help="cluster center x")
    p.add_argument("--cy",   type=float, default=0.5,   help="cluster center y")
    p.add_argument("--cz",   type=float, default=0.5,   help="cluster center z")
    p.add_argument("--seed", type=int,   default=42,    help="random seed")
    p.add_argument("-o",     type=str,   default="ic.bin", help="output file")
    args = p.parse_args()

    generators = {"sphere": generate_uniform_sphere, "cube": generate_uniform_cube}
    bodies = generators[args.dist](
        n      = args.n,
        R      = args.R,
        center = (args.cx, args.cy, args.cz),
        seed   = args.seed,
    )
    write_ic(args.o, bodies)
