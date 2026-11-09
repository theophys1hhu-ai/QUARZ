#!/usr/bin/env python3
"""Analysis of the QED example decks: QUARZ output vs the theory of the QED module paper.

usage: python3 analyze.py MODE DECK OUTDIR [OUTDIR2] [key=value ...]
  larmor        larmor.in, larmor_spiral.in: <gamma>(t) vs the synchrotron loss with g(chi); orbit radius
  spectrum      spectrum.in: dN/du vs N_e T C0 w(u, chi)                                  (Fig. 2)
  spindiff      spin_dependence.in OUT_plus OUT_minus: dN+/du - dN-/du vs 2 N_e T C0 g  (Fig. 3)
  spin          sokolov_ternov.in: <s2>(t) vs eq. 14                                       (Fig. 4)
  polarization  photon_polarization.in: <xi_3> vs eq. 22, xi_3(u) vs eq. 21                 (Fig. 5)
  betatron      betatron.in: <gamma>(t) vs the orbit-averaged loss with A ~ gamma^(1/8)      (eqs. 25-27)
  benchmark     pwfa_benchmark.in OUT_base OUT_sync OUT_qed: witness energy, spectra, wake (Figs. 6-8)
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


def gaunt_interp(chi_max):
    """g(chi) = int u w du / ((2 pi/sqrt 3) chi^2) on a log grid up to chi_max, as a function"""
    chis = np.geomspace(1e-6, chi_max * 1.01, 40)
    gc = np.array([uint(lambda u, c=c: u * coef(u, c)["w"]) / (2 * np.pi / np.sqrt(3) * c * c) for c in chis])
    return lambda c: np.interp(np.log(np.maximum(c, 1e-6)), np.log(chis), gc)


def betatron(d, out):
    """pure ion channel: E_r = r/2 (immobile ions of density 1), electrons start at x_0 with p_perp = 0"""
    import glob
    run = Run(d)
    A0 = float(d.get("e.x0", 1))
    b = qr.read_beamlog(os.path.join(out, "beams.txt"))["e"]
    t, g = b["t"], b["gamma_mean"]
    k = 2 * ALPHA / (3 * run.eta)
    gfun = gaunt_interp(run.eta * run.gamma * A0 / 2)
    def ode(expo):   # d gamma/dt = -k chi_eff^2 g(chi_eff), chi_eff = eta gamma A / (2 sqrt 2), A = A0 (gamma/gamma0)^expo
        def rhs(tt, y):
            A = A0 * (y[0] / run.gamma) ** expo
            c = run.eta * y[0] * A / (2 * np.sqrt(2))
            return [-k * c * c * gfun(c)]
        return si.solve_ivp(rhs, (0, t[-1]), [run.gamma], t_eval=t, rtol=1e-9).y[0]
    g8, g0 = ode(1 / 8), ode(0.0)
    # amplitude of every particle: A^2 = x^2 + (x'/omega_beta)^2 = x^2 + 2 p_x^2 / gamma  (omega_beta^2 = 1/(2 gamma))
    ta, Am = [], []
    for f in sorted(glob.glob(os.path.join(out, "beam_e_*.bin"))):
        p = qr.read_beam(f)
        gam = np.sqrt(1 + p["px"] ** 2 + p["py"] ** 2 + p["pz"] ** 2)
        ta.append(int(f.split("_")[-1].split(".")[0]) * run.dt)
        Am.append(np.mean(np.sqrt(p["x"] ** 2 + 2 * p["px"] ** 2 / gam)))
    ta, Am = np.array(ta), np.array(Am)
    gi = np.interp(ta, t, g)
    print("  gamma_0 = %.3g, A_0 = %.3g, chi_0 = %.3g (turning points), %.1f betatron periods" % (
        run.gamma, A0, run.eta * run.gamma * A0 / 2, t[-1] / (2 * np.pi * np.sqrt(2 * run.gamma))))
    print("  <gamma> at the end: QUARZ %.6g; orbit-averaged ODE with A ~ gamma^(1/8): %.6g (%+.2f %%); with constant A: %.6g (%+.2f %%)"
          % (g[-1], g8[-1], 100 * (g[-1] / g8[-1] - 1), g0[-1], 100 * (g[-1] / g0[-1] - 1)))
    print("  amplitude <A> at the end: %.5f; A_0 (gamma/gamma_0)^(1/8) = %.5f; A_0 (gamma/gamma_0)^(1/4) = %.5f"
          % (Am[-1], A0 * (gi[-1] / run.gamma) ** 0.125, A0 * (gi[-1] / run.gamma) ** 0.25))

    def draw(plt):
        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        ax[0].plot(t, g, label="QUARZ <gamma>")
        ax[0].plot(t, g8, "r--", label="eq. 26 with A ~ gamma^(1/8)")
        ax[0].plot(t, g0, "k:", label="eq. 26 with A = A_0")
        ax[0].set_xlabel("t [1/omega_p]"); ax[0].set_ylabel("gamma"); ax[0].legend()
        ax[1].plot(ta, Am / A0, "x", label="QUARZ <A>/A_0")
        ax[1].plot(ta, (gi / run.gamma) ** 0.125, "r--", label="(gamma/gamma_0)^(1/8)")
        ax[1].set_xlabel("t [1/omega_p]"); ax[1].set_ylabel("A / A_0"); ax[1].legend()
        return fig
    plot(out, "betatron", draw)


MC2_GEV = 0.51099895e-3


def benchmark(d, base, sync, qed):
    import glob
    def wlog(o):
        return qr.read_beamlog(os.path.join(o, "beams.txt"))["witness"]
    wb, ws, wq = wlog(base), wlog(sync), wlog(qed)
    E = lambda w: w["gamma_mean"] * MC2_GEV
    print("  witness energy: start %.4g GeV; base (no radiation) gain %+.4g GeV at t = %.4g" % (E(wb)[0], E(wb)[-1] - E(wb)[0], wb["t"][-1]))
    print("  with radiation at t = %.4g: synchrotron (continuous, g(chi)) %.5g GeV, QED %.5g GeV (rms %.4g GeV)"
          % (ws["t"][-1], E(ws)[-1], E(wq)[-1], wq["gamma_rms"][-1] * MC2_GEV))
    last = lambda o: sorted(glob.glob(os.path.join(o, "beam_witness_*.bin")))[-1]
    es, eq = [np.sqrt(1 + p["px"] ** 2 + p["py"] ** 2 + p["pz"] ** 2) * MC2_GEV for p in (qr.read_beam(last(sync)), qr.read_beam(last(qed)))]
    ph = qr.read_photons(os.path.join(qed, "photons_witness.bin"))
    print("  final witness energy spread (rms): synchrotron %.4g GeV, QED %.4g GeV; QED energy range %.4g ... %.4g GeV"
          % (np.std(es), np.std(eq), eq.min(), eq.max()))
    print("  photons (QED): %d macro-photons, max energy %.4g GeV, emitter chi: mean %.3g, max %.3g"
          % (len(ph["energy"]), ph["energy"].max() * MC2_GEV, np.average(ph["chi"]), ph["chi"].max()))

    def draw(plt):
        fig, ax = plt.subplots(2, 2, figsize=(11, 8))
        ax[0, 0].plot(wb["t"], E(wb) - E(wb)[0], label="base: energy gain")
        ax[0, 0].set_xlabel("t [1/omega_p]"); ax[0, 0].set_ylabel("Delta E [GeV]"); ax[0, 0].legend()
        ax[0, 1].plot(wq["t"], E(wq), label="QED"); ax[0, 1].plot(ws["t"], E(ws), "--", label="synchrotron")
        ax[0, 1].set_xlabel("t [1/omega_p]"); ax[0, 1].set_ylabel("E [GeV]"); ax[0, 1].legend()
        bins = np.linspace(0, max(eq.max(), es.max()) * 1.02, 120)
        ax[1, 1].semilogy(0.5 * (bins[1:] + bins[:-1]), np.histogram(eq, bins)[0], label="QED")
        ax[1, 1].semilogy(0.5 * (bins[1:] + bins[:-1]), np.histogram(es, bins)[0], "--", label="synchrotron")
        ax[1, 1].set_xlabel("E_e [GeV]"); ax[1, 1].set_ylabel("witness macro-particles per bin"); ax[1, 1].legend()
        pb = np.linspace(0, ph["energy"].max() * MC2_GEV, 120)
        ax[1, 0].semilogy(0.5 * (pb[1:] + pb[:-1]), np.histogram(ph["energy"] * MC2_GEV, pb, weights=ph["w"])[0] / np.diff(pb))
        ax[1, 0].set_xlabel("E_gamma [GeV]"); ax[1, 0].set_ylabel("dN/dE (weights)")
        return fig
    plot(qed, "benchmark", draw)

    fs = sorted(glob.glob(os.path.join(qed, "fields_*.bin")))
    fmid = [f for f in fs if f.endswith("_000140.bin")]
    if fmid:
        F = qr.read_fields(fmid[0])
        def draw2(plt):
            fig, ax = plt.subplots(1, 2, figsize=(11, 4))
            ext = [F["xi"][0], F["xi"][-1], 0, F["r"][-1]]
            ax[0].imshow(np.minimum(F["ne"].T, 10), origin="lower", aspect="auto", extent=ext, cmap="gray_r")
            ax[0].contour(F["xi"], F["r"], np.abs(F["rhob"]).T, levels=[0.5, 2, 5], colors="orange", linewidths=0.8)
            ax[0].set_xlabel("xi [1/k_p]"); ax[0].set_ylabel("r [1/k_p]"); ax[0].set_title("n_e / n_0 (beams: contours), t = %.0f" % F["t"])
            ax[1].imshow(F["ez"].T, origin="lower", aspect="auto", extent=ext, cmap="RdBu_r", vmin=-5, vmax=5)
            ax[1].set_xlabel("xi [1/k_p]"); ax[1].set_title("E_z / E_0")
            return fig
        plot(qed, "benchmark_wake", draw2)


if __name__ == "__main__":
    if len(sys.argv) < 4:
        sys.exit(__doc__)
    mode, deckfile = sys.argv[1], sys.argv[2]
    outs = [a for a in sys.argv[3:] if "=" not in a]
    d = deck(deckfile, [a for a in sys.argv[3:] if "=" in a])
    {"larmor": lambda: larmor(d, outs[0]), "spectrum": lambda: spectrum(d, outs[0]),
     "spindiff": lambda: spindiff(d, outs[0], outs[1]), "spin": lambda: spin(d, outs[0]),
     "polarization": lambda: polarization(d, outs[0]), "betatron": lambda: betatron(d, outs[0]),
     "benchmark": lambda: benchmark(d, outs[0], outs[1], outs[2])}[mode]()
