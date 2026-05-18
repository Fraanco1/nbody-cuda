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
  --eps       gravitational softening (disk only)  (default: 0.1)
              should match --eps used in the simulation
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


def _disk_rotation_curve(R, eps, n_r=64, n_phi=128):
    """
    Numerically integrate the radial gravitational force of a uniform softened
    disk (G=1, M_total=1) at n_r sample radii, returning (r_table, vcirc_table).

    The shell theorem does not hold for a disk, so the rotation curve must be
    computed by integrating over the full mass distribution instead of using
    M_enclosed as a point mass.
    """
    sigma = 1.0 / (math.pi * R * R)
    dr    = R / n_r
    dphi  = 2.0 * math.pi / n_phi

    r_tab = [(i + 0.5) * dr for i in range(n_r)]
    v_tab = []

    for r in r_tab:
        F_r = 0.0
        for j in range(n_r):
            rp = (j + 0.5) * dr
            for k in range(n_phi):
                phi = (k + 0.5) * dphi
                ex  = rp * math.cos(phi) - r
                ey  = rp * math.sin(phi)
                d2  = ex*ex + ey*ey + eps*eps
                F_r += sigma * rp * ex / (d2 * math.sqrt(d2)) * dphi * dr
        # F_r < 0 (inward); v_circ^2 = r * |F_r|
        v_tab.append(math.sqrt(max(0.0, -r * F_r)))

    return r_tab, v_tab


def _interp(r_tab, v_tab, r):
    if r <= r_tab[0]:
        return v_tab[0] * r / r_tab[0]
    if r >= r_tab[-1]:
        return v_tab[-1]
    for i in range(len(r_tab) - 1):
        if r_tab[i] <= r < r_tab[i + 1]:
            t = (r - r_tab[i]) / (r_tab[i + 1] - r_tab[i])
            return v_tab[i] + t * (v_tab[i + 1] - v_tab[i])
    return v_tab[-1]


def generate_disk(n, R, center, seed, thickness=0.05, eps=0.1):
    """Thin rotating disk in the x-y plane with self-consistent circular velocities."""
    print("Computing rotation curve...", flush=True)
    r_tab, v_tab = _disk_rotation_curve(R, eps)

    rng = random.Random(seed)
    cx, cy, cz = center
    mass = 1.0 / n
    bodies = []
    for _ in range(n):
        r   = R * math.sqrt(rng.random())
        phi = 2.0 * math.pi * rng.random()
        z   = rng.gauss(0.0, thickness * R)

        x = cx + r * math.cos(phi)
        y = cy + r * math.sin(phi)

        v_circ = _interp(r_tab, v_tab, r)
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
    p.add_argument("--eps",       type=float, default=0.1,
                   help="gravitational softening used for disk circular velocities; "
                        "must match the simulation's --eps (default: 0.1)")
    p.add_argument("--thickness", type=float, default=0.05,
                   help="disk vertical thickness as fraction of R (disk only, default: 0.05)")
    p.add_argument("--seed",      type=int,   default=42,    help="random seed")
    p.add_argument("-o",          type=str,   default="ic.bin", help="output file")
    args = p.parse_args()

    center = (args.cx, args.cy, args.cz)

    if args.dist == "sphere":
        bodies = generate_uniform_sphere(args.n, args.R, center, args.seed)
    elif args.dist == "cube":
        bodies = generate_uniform_cube(args.n, args.R, center, args.seed)
    else:
        bodies = generate_disk(args.n, args.R, center, args.seed,
                               args.thickness, args.eps)

    write_ic(args.o, bodies)
