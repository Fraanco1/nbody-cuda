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
  -R          outer radius (sphere/cube half-side, disk outer edge)  (default: 0.5)
  --Rd        disk scale radius                    (default: R/5)
  --r-in      disk inner radius — creates a central hole  (default: 0.0)
  --cx/cy/cz  center                               (default: 0.5 0.5 0.5)
  --eps       gravitational softening (disk only)  (default: 0.01)
              must match --eps used in the simulation
  --thickness disk vertical thickness as fraction of Rd  (default: 0.1)
  --seed      random seed                          (default: 42)
  -o          output file                          (default: ic.bin)

Disk notes:
  Positions follow an exponential surface density Sigma(r) ~ exp(-r/Rd),
  truncated to [r_in, R].  Circular velocities are derived by numerically
  integrating the actual radial force from this mass distribution (no shell-
  theorem approximation).  eps must match the simulation's softening so the
  forces are consistent.
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


# ── Disk helpers ──────────────────────────────────────────────────────────────

def _rotation_curve(r_in, r_out, sigma_func, eps, n_r=64, n_phi=128):
    """
    Numerically integrate the radial gravitational force of a disk with
    surface density sigma_func(r) on [r_in, r_out], softening eps (G=1).

    The shell theorem does not hold for disks, so this integral over the full
    mass distribution is necessary for self-consistent circular velocities.

    Returns (r_table, vcirc_table).
    """
    dr   = (r_out - r_in) / n_r
    dphi = 2.0 * math.pi / n_phi

    r_tab = [r_in + (i + 0.5) * dr for i in range(n_r)]
    v_tab = []

    for r in r_tab:
        F_r = 0.0
        for j in range(n_r):
            rp  = r_in + (j + 0.5) * dr
            sig = sigma_func(rp)
            for k in range(n_phi):
                phi = (k + 0.5) * dphi
                ex  = rp * math.cos(phi) - r
                ey  = rp * math.sin(phi)
                d2  = ex*ex + ey*ey + eps*eps
                F_r += sig * rp * ex / (d2 * math.sqrt(d2)) * dphi * dr
        # F_r < 0 (inward); v_circ^2 = r * |F_r|
        v_tab.append(math.sqrt(max(0.0, -r * F_r)))

    return r_tab, v_tab


def _interp(r_tab, v_tab, r):
    """Linear interpolation; extrapolates linearly to 0 below table, flat above."""
    if r <= r_tab[0]:
        return v_tab[0] * r / r_tab[0]
    if r >= r_tab[-1]:
        return v_tab[-1]
    for i in range(len(r_tab) - 1):
        if r_tab[i] <= r < r_tab[i + 1]:
            t = (r - r_tab[i]) / (r_tab[i + 1] - r_tab[i])
            return v_tab[i] + t * (v_tab[i + 1] - v_tab[i])
    return v_tab[-1]


def _sample_exp_annulus(rng, R_d, r_in, r_out):
    """
    Sample r from p(r) ∝ r·exp(-r/R_d) on [r_in, r_out].

    Uses the exact CDF  F(r) = -R_d·(r+R_d)·exp(-r/R_d)  via bisection
    (52 iterations → double-precision accuracy).
    """
    def F(r):
        return -R_d * (r + R_d) * math.exp(-r / R_d)

    F_in, F_out = F(r_in), F(r_out)
    target = F_in + rng.random() * (F_out - F_in)

    lo, hi = r_in, r_out
    for _ in range(52):
        mid = 0.5 * (lo + hi)
        if F(mid) < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def generate_disk(n, r_out, R_d, center, seed, r_in=0.0, thickness=0.1, eps=0.01):
    """
    Exponential disk on [r_in, r_out] with self-consistent circular velocities.

    Surface density: Sigma(r) = C · exp(-r/R_d), normalised so total mass = 1.
    Circular velocities come from numerically integrating the actual radial
    force (no shell-theorem approximation).
    """
    # Normalisation constant: integral_{r_in}^{r_out} 2*pi*r*Sigma dr = 1
    def F_cdf(r):
        return -R_d * (r + R_d) * math.exp(-r / R_d)

    norm = 2.0 * math.pi * (F_cdf(r_out) - F_cdf(r_in))
    sigma_func = lambda r: math.exp(-r / R_d) / norm

    print(f"Computing rotation curve for exponential disk "
          f"(r_in={r_in:.3f}, r_out={r_out:.3f}, Rd={R_d:.3f}, eps={eps:.4f})...",
          flush=True)
    r_tab, v_tab = _rotation_curve(r_in, r_out, sigma_func, eps)
    print(f"  Peak v_circ = {max(v_tab):.4f} at r = "
          f"{r_tab[v_tab.index(max(v_tab))]:.3f}", flush=True)

    rng = random.Random(seed)
    cx, cy, cz = center
    mass = 1.0 / n
    bodies = []
    for _ in range(n):
        r   = _sample_exp_annulus(rng, R_d, r_in, r_out)
        phi = 2.0 * math.pi * rng.random()
        z   = rng.gauss(0.0, thickness * R_d)

        x = cx + r * math.cos(phi)
        y = cy + r * math.sin(phi)

        v_circ = _interp(r_tab, v_tab, r)
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
    p.add_argument("--Rd",        type=float, default=None,
                   help="disk scale radius (default: R/5)")
    p.add_argument("--r-in",      type=float, default=0.0,
                   help="disk inner radius — leaves a central hole (default: 0)")
    p.add_argument("--cx",        type=float, default=0.5, help="center x")
    p.add_argument("--cy",        type=float, default=0.5, help="center y")
    p.add_argument("--cz",        type=float, default=0.5, help="center z")
    p.add_argument("--eps",       type=float, default=0.01,
                   help="gravitational softening for disk velocities; "
                        "must match simulation --eps (default: 0.01)")
    p.add_argument("--thickness", type=float, default=0.1,
                   help="disk vertical thickness as fraction of Rd (default: 0.1)")
    p.add_argument("--seed",      type=int,   default=42,  help="random seed")
    p.add_argument("-o",          type=str,   default="ic.bin", help="output file")
    args = p.parse_args()

    center = (args.cx, args.cy, args.cz)

    if args.dist == "sphere":
        bodies = generate_uniform_sphere(args.n, args.R, center, args.seed)
    elif args.dist == "cube":
        bodies = generate_uniform_cube(args.n, args.R, center, args.seed)
    else:
        R_d = args.Rd if args.Rd is not None else args.R / 5.0
        bodies = generate_disk(args.n, args.R, R_d, center, args.seed,
                               r_in=args.r_in,
                               thickness=args.thickness,
                               eps=args.eps)

    write_ic(args.o, bodies)
