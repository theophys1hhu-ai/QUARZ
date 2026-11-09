#!/usr/bin/env python3
"""Analysis of the QED example decks: QUARZ output vs the theory of the QED module paper.

usage: python3 analyze.py MODE DECK OUTDIR [OUTDIR2] [key=value ...]
  larmor        larmor.in, larmor_spiral.in: <gamma>(t) vs the synchrotron loss with g(chi); orbit radius
  spectrum      spectrum.in: dN/du vs N_e T C0 w(u, chi)                                  (Fig. 2)
  spindiff      spin_dependence.in OUT_plus OUT_minus: dN+/du - dN-/du vs 2 N_e T C0 g  (Fig. 3)
  spin          sokolov_ternov.in: <s2>(t) vs eq. 14                                       (Fig. 4)
  polarization  photon_polarization.in: <xi_3> vs eq. 22, xi_3(u) vs eq. 21                 (Fig. 5)
The key=value arguments repeat the command-line overrides of the run (e.g. "fields.external_B=0.41492 0 0"
e.gamma=3e6 "e.spin=1 0 0"). Prints the comparison and writes OUTDIR/<mode>.png (with matplotlib).
The theory functions (scipy, independent of the C++ code) come from validation/qed_check.py.
"""
import os
import sys

import numpy as np
import scipy.integrate as si

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "tools"))
sys.path.insert(0, os.path.join(HERE, "..", "..", "validation"))
import quarz_read as qr  # noqa: E402
from qed_check import ALPHA, coef, integrals, uint  # noqa: E402


def deck(path, overrides):
    """key -> value of the deck, with the overrides applied"""
    d = {}
    for line in list(open(path)) + list(overrides):
        line = line.split("#")[0].strip()
        if "=" in line:
            k, v = line.split("=", 1)
            d[k.strip()] = v.strip().strip('"')
    return d


class Run:
    """the physical parameters of a run"""
    def __init__(self, d):
        self.n0 = float(d["units.n0_cm3"])
        self.eta = 3.8615926796e-11 * 5.64146e4 * np.sqrt(self.n0) / 2.99792458e10   # hbar omega_p / m c^2
        self.gamma = float(d["e.gamma"])
        self.B = np.array([float(v) for v in d.get("fields.external_B", "0 0 0").split()])
        self.dt, self.steps = float(d["time.dt"]), int(d["time.steps"])
        self.p_perp = self.gamma * np.hypot(float(d.get("e.xp0", 0)), float(d.get("e.yp0", 0)))
        sv = [float(v) for v in d.get("e.spin", "0 0 0").split()]
        self.s2 = sv[0] / max(np.linalg.norm(sv), 1e-300) if any(sv) else 0.0   # e2 = +x for electrons along z in B_x
        self.C0 = ALPHA / (np.sqrt(3) * np.pi * self.eta * self.gamma)
        self.chi = self.eta * self.gamma * np.hypot(self.B[0], self.B[1])   # beam along z, transverse field


def plot(out, name, draw):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig = draw(plt)
    fig.tight_layout()
    fig.savefig(os.path.join(out, name + ".png"), dpi=120)
    print("  plot: %s" % os.path.join(out, name + ".png"))


def photons_u(out, run):
    p = qr.read_photons(os.path.join(out, "photons_e.bin"))
    return p, p["energy"] / run.gamma   # no recoil: the electron energy stays gamma


def n_electrons_time(out, run):
    q = qr.read_qedlog(os.path.join(out, "qed.txt"))["e"]
    b = qr.read_beamlog(os.path.join(out, "beams.txt"))["e"]
    return b["alive"][0], len(q["step"]) * run.dt


def larmor(d, out):
    run = Run(d)
    b = qr.read_beamlog(os.path.join(out, "beams.txt"))["e"]
    t, g = b["t"], b["gamma_mean"]
    Bz = run.B[2]
    # the recoil keeps the direction: p_perp / gamma stays constant, chi = eta p_perp B
    chi = lambda gam: run.eta * run.p_perp * gam / run.gamma * Bz
    chis = np.geomspace(1e-5, chi(run.gamma) * 1.01, 40)
    gc = np.array([uint(lambda u, c=c: u * coef(u, c)["w"]) / (2 * np.pi / np.sqrt(3) * c * c) for c in chis])
    gfun = lambda c: np.interp(np.log(c), np.log(chis), gc)
    rhs = lambda tt, y: [-(2 * ALPHA / (3 * run.eta)) * chi(y[0]) ** 2 * gfun(chi(y[0]))]
    sol = si.solve_ivp(rhs, (0, t[-1]), [run.gamma], t_eval=t, rtol=1e-9)
    r0, r1 = run.p_perp / Bz, run.p_perp / Bz * g[-1] / run.gamma
    print("  chi_0 = %.3g: <gamma> %.6g -> %.6g at t = %.4g (synchrotron loss with g(chi): %.6g; gamma_rms %.4g)"
          % (chi(run.gamma), g[0], g[-1], t[-1], sol.y[0][-1], b["gamma_rms"][-1]))
    print("  orbit radius p_perp / B: %.4g -> %.4g; centroid at the end (%.3g, %.3g) (the bunch is spread over"
          " the circle once gamma differs between particles)" % (r0, r1, b["x_mean"][-1], b["y_mean"][-1]))

    def draw(plt):
        import glob
        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        dumps = sorted(glob.glob(os.path.join(out, "beam_e_*.bin")))
        xs, ys = [], []
        for f in dumps:   # one particle (the first one written) at every dump: its orbit
            p = qr.read_beam(f)
            if len(p["x"]):
                xs.append(p["x"][0]); ys.append(p["y"][0])
        ax[0].plot(xs, ys, "-")
        ax[0].set_aspect("equal"); ax[0].set_xlabel("x [c/omega_p]"); ax[0].set_ylabel("y [c/omega_p]")
        ax[0].set_title("chi_0 = %.2g" % chi(run.gamma))
        ax[1].plot(t, g, label="QUARZ <gamma>")
        ax[1].plot(t, sol.y[0], "r--", label="d gamma/dt = -(2 alpha/3 eta) chi^2 g(chi)")
        ax[1].set_xlabel("t [1/omega_p]"); ax[1].set_ylabel("gamma"); ax[1].legend()
        return fig
    plot(out, "larmor", draw)


EDGES = np.linspace(0, 1, 41)


def spectrum(d, out):
    run = Run(d)
    p, u = photons_u(out, run)
    Ne, T = n_electrons_time(out, run)
    h = np.histogram(u, EDGES)[0]
    du = np.diff(EDGES)
    th = np.array([uint(lambda x: coef(x, run.chi)["w"] + run.s2 * coef(x, run.chi)["g"], a, b)
                   for a, b in zip(EDGES[:-1], EDGES[1:])]) * Ne * T * run.C0
    I = integrals(run.chi)
    print("  chi = %.3g, s2 = %+.2g: %d photons from %d electrons in t = %.4g; expected N_e T C0 (W + G s2) = %.1f"
          % (run.chi, run.s2, len(u), Ne, T, Ne * T * run.C0 * (I["w"] + run.s2 * I["g"])))
    print("   u-bin    dN/du (QUARZ)   dN/du (theory)")
    for k in range(0, len(h), 4):
        print("  %5.3f    %12.5g   %12.5g" % (0.5 * (EDGES[k] + EDGES[k + 1]), h[k] / du[k], th[k] / du[k]))

    def draw(plt):
        fig, ax = plt.subplots(figsize=(6, 4))
        c = 0.5 * (EDGES[1:] + EDGES[:-1])
        ax.semilogy(c, h / du, "x", label="QUARZ")
        ax.semilogy(c, th / du, "--", label="N_e T C0 w(u, chi)")
        ax.set_xlabel("u = photon energy / electron energy"); ax.set_ylabel("dN/du")
        ax.set_title("chi = %.3g" % run.chi); ax.legend()
        return fig
    plot(out, "spectrum", draw)


def spindiff(d, outp, outm):
    run = Run(d)
    up, um = photons_u(outp, run)[1], photons_u(outm, run)[1]
    Ne, T = n_electrons_time(outp, run)
    I = integrals(run.chi)
    hp, hm = np.histogram(up, EDGES)[0], np.histogram(um, EDGES)[0]
    th = np.array([uint(lambda x: coef(x, run.chi)["g"], a, b) for a, b in zip(EDGES[:-1], EDGES[1:])]) * 2 * Ne * T * run.C0
    r = len(up) / len(um)
    print("  chi = %.3g: photons s2 = +1: %d, s2 = -1: %d; ratio %.4f +- %.4f (theory (W + G)/(W - G) = %.4f)"
          % (run.chi, len(up), len(um), r, r * np.sqrt(1 / len(up) + 1 / len(um)), (I["w"] + I["g"]) / (I["w"] - I["g"])))
    e_gev = 0.5 * (EDGES[1:] + EDGES[:-1]) * run.gamma * 0.51099895e-3

    def draw(plt):
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(e_gev, hp - hm, "-", label="QUARZ: N(s2 = +1) - N(s2 = -1)")
        ax.plot(e_gev, th, "--", label="2 N_e T C0 g du")
        ax.set_xlabel("photon energy [GeV]"); ax.set_ylabel("photons per bin"); ax.legend()
        return fig
    plot(outp, "spindiff", draw)


def spin(d, out):
    run = Run(d)
    s = qr.read_spinlog(os.path.join(out, "spin.txt"))["e"]
    t, sx = s["t"], s["sx"]
    I = integrals(run.chi)
    a, b = run.C0 * (I["fs"] - I["w"]), run.C0 * (I["f2"] - I["g"])
    Sinf = -b / a
    ref = (sx[0] - Sinf) * np.exp(a * t) + Sinf
    print("  chi = %.3g: <s2> %.4f -> %.4f at t = %.4g; eq. 14: %.4f (limit %.4f, relaxation time %.4g); max |diff| %.4f"
          % (run.chi, sx[0], sx[-1], t[-1], ref[-1], Sinf, -1 / a, np.max(np.abs(sx - ref))))

    def draw(plt):
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.semilogx(t[1:], sx[1:], "x", label="QUARZ")
        ax.semilogx(t[1:], ref[1:], "--", label="eq. 14")
        ax.axhline(-8 * np.sqrt(3) / 15, color="gray", ls=":", label="Sokolov-Ternov -8 sqrt(3)/15")
        ax.set_xlabel("t [1/omega_p]"); ax.set_ylabel("<s2>"); ax.set_title("chi = %.3g" % run.chi); ax.legend()
        return fig
    plot(out, "spin", draw)


def polarization(d, out):
    run = Run(d)
    p, u = photons_u(out, run)
    c, S = run.chi, run.s2
    ref = uint(lambda x: coef(x, c)["x3"] + S * coef(x, c)["x3s"]) / uint(lambda x: coef(x, c)["w"] + S * coef(x, c)["g"])
    n = len(u)
    print("  gamma = %.3g, chi = %.3g, s2 = %+.2g: <xi_3> = %.5f +- %.5f (eq. 22: %.5f); <xi_1> = %.2g, <xi_2> = %.2g"
          % (run.gamma, c, S, np.mean(p["xi3"]), np.std(p["xi3"]) / np.sqrt(n), ref, np.mean(p["xi1"]), np.mean(p["xi2"])))
    e = np.linspace(0, 1, 21)
    cen, mx, th = [], [], []
    for a, b in zip(e[:-1], e[1:]):
        sel = (u >= a) & (u < b)
        if sel.sum() < 20:
            continue
        cen.append(0.5 * (a + b)); mx.append(np.mean(p["xi3"][sel]))
        th.append(uint(lambda x: coef(x, c)["x3"] + S * coef(x, c)["x3s"], a, b) / uint(lambda x: coef(x, c)["w"] + S * coef(x, c)["g"], a, b))

    def draw(plt):
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(cen, mx, "x", label="QUARZ")
        ax.plot(cen, th, "--", label="eq. 21 (rate-weighted in the bin)")
        ax.set_xlabel("u"); ax.set_ylabel("xi_3"); ax.set_title("gamma = %.3g, s2 = %+.0f" % (run.gamma, S)); ax.legend()
        return fig
    plot(out, "polarization", draw)


if __name__ == "__main__":
    if len(sys.argv) < 4:
        sys.exit(__doc__)
    mode, deckfile = sys.argv[1], sys.argv[2]
    outs = [a for a in sys.argv[3:] if "=" not in a]
    d = deck(deckfile, [a for a in sys.argv[3:] if "=" in a])
    {"larmor": lambda: larmor(d, outs[0]), "spectrum": lambda: spectrum(d, outs[0]),
     "spindiff": lambda: spindiff(d, outs[0], outs[1]), "spin": lambda: spin(d, outs[0]),
     "polarization": lambda: polarization(d, outs[0])}[mode]()
