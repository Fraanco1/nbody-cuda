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
  -R          outer radius / half-side             (default: 0.5)
  --r-in      disk inner radius (annulus hole)     (default: 0.05)
  --cx/cy/cz  center                               (default: 0.5 0.5 0.5)
  --thickness  disk vertical thickness as fraction of (R - r_in)  (default: 0.05)
  --halo-mass  Hernquist halo mass (G=1)                          (default: 5.0)
  --halo-scale Hernquist halo scale radius                        (default: 0.2)
  --seed       random seed                                        (default: 42)
  -o           output file                                        (default: ic.bin)

Disk notes:
  Positions are drawn with uniform surface density on the annulus [r_in, R].
  Circular velocity: v(r) = sqrt(M_enc/r + M_h*r/(r+a)^2)
    M_enc = (r²-r_in²)/(R²-r_in²)  (self-gravity, G=M_disk=1)
    Hernquist halo term stabilises the disk against escape.
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
        bodies.append((cx + R*ux, cy + R*uy, cz + R*uz, 0.0, 0.0, 0.0, mass))
    return bodies


def generate_uniform_cube(n, R, center, seed):
    rng = random.Random(seed)
    cx, cy, cz = center
    mass = 1.0 / n
    bodies = []
    for _ in range(n):
        ux = 2.0 * rng.random() - 1.0
        uy = 2.0 * rng.random() - 1.0
        uz = 2.0 * rng.random() - 1.0
        bodies.append((cx + R*ux, cy + R*uy, cz + R*uz, 0.0, 0.0, 0.0, mass))
    return bodies


def generate_disk(n, r_out, center, seed, r_in=0.05, thickness=0.05,
                  halo_mass=5.0, halo_scale=0.2):
    """
    Uniform annulus [r_in, r_out] with circular velocities from self-gravity +
    a static Hernquist dark-matter halo.

    v_circ(r) = sqrt(M_enc/r + M_h*r/(r+a)^2)
      M_enc = (r²-r_in²)/(r_out²-r_in²)  (self-gravity, uniform surface density)
      Hernquist halo: M_h, scale a  (G=1)
    """
    rng = random.Random(seed)
    cx, cy, cz = center
    mass = 1.0 / n
    width = r_out - r_in
    bodies = []
    for _ in range(n):
        r   = math.sqrt(r_in*r_in + rng.random() * (r_out*r_out - r_in*r_in))
        phi = 2.0 * math.pi * rng.random()
        z   = rng.gauss(0.0, thickness * width)

        x = cx + r * math.cos(phi)
        y = cy + r * math.sin(phi)

        m_enc   = (r*r - r_in*r_in) / (r_out*r_out - r_in*r_in)
        v_halo2 = halo_mass * r / (r + halo_scale)**2
        v_circ  = math.sqrt(m_enc / r + v_halo2)
        vx = -v_circ * math.sin(phi)
        vy =  v_circ * math.cos(phi)

        bodies.append((x, y, cz + z, vx, vy, 0.0, mass))
    return bodies


# ── I/O ───────────────────────────────────────────────────────────────────────

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
    p.add_argument("-n",          type=int,   default=10000,
                   help="number of bodies (default: 10000)")
    p.add_argument("-R",          type=float, default=0.5,
                   help="outer radius / half-side (default: 0.5)")
    p.add_argument("--r-in",      type=float, default=0.05,
                   help="disk inner radius / annulus hole (default: 0.05)")
    p.add_argument("--cx",        type=float, default=0.5, help="center x")
    p.add_argument("--cy",        type=float, default=0.5, help="center y")
    p.add_argument("--cz",        type=float, default=0.5, help="center z")
    p.add_argument("--thickness", type=float, default=0.05,
                   help="disk vertical thickness as fraction of annulus width (default: 0.05)")
    p.add_argument("--halo-mass",  type=float, default=5.0,
                   help="Hernquist halo mass (default: 5.0)")
    p.add_argument("--halo-scale", type=float, default=0.2,
                   help="Hernquist halo scale radius (default: 0.2)")
    p.add_argument("--seed",      type=int,   default=42,  help="random seed")
    p.add_argument("-o",          type=str,   default="ic.bin", help="output file")
    args = p.parse_args()

    center = (args.cx, args.cy, args.cz)

    if args.dist == "sphere":
        bodies = generate_uniform_sphere(args.n, args.R, center, args.seed)
    elif args.dist == "cube":
        bodies = generate_uniform_cube(args.n, args.R, center, args.seed)
    else:
        bodies = generate_disk(args.n, args.R, center, args.seed,
                               r_in=args.r_in,
                               thickness=args.thickness,
                               halo_mass=args.halo_mass,
                               halo_scale=args.halo_scale)

    write_ic(args.o, bodies)
