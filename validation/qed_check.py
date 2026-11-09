#!/usr/bin/env python3
"""Validation of the photon emission (section 26): QUARZ output vs references computed here with scipy,
independently of the C++ code (Bessel functions scipy.special.kv, integrals scipy.integrate.quad).

Setup (qed.in): electrons along z in a constant field B along x, no plasma fields, chi = eta gamma B;
for electrons e2 = +x (along B), so s2 = <s_x>.
usage: qed_check.py spectrum DIR S            photon number and spectrum vs C0 (w + g S) (no recoil)
       qed_check.py spindiff DIR_plus DIR_minus   spectrum difference (S = +1) - (S = -1) vs 2 C0 g
       qed_check.py spin DIR S0               <s2>(t) vs eq. 14 (no recoil)
       qed_check.py polar DIR S               mean Stokes vector vs eq. 22, xi_3(u) vs eq. 21 (fixed spin)
       qed_check.py energy DIR [model]        <gamma>(t) vs the ODE d gamma/dt = -P(gamma); energy balance
                                              (model continuous | classical: deterministic, g = 1 for classical)
       qed_check.py fixed DIR                 momentum_push = 0, spin_precession = 0: p and s unchanged
       qed_check.py compare DIR_A DIR_B label final <gamma>, gamma_rms of two runs (statistics)
       qed_check.py same DIR_A DIR_B [step0]  photon records and qed.txt identical (from step0 on)
       qed_check.py gyration DIR              no QED: rotation of p in B by omega t = B t / gamma
Statistical checks append "<-- above tolerance" when they fail (4 sigma, 4.5 for the maximum over a time
series, or chi^2/dof > 2).
"""
import os
import sys

import numpy as np
import scipy.integrate as si
import scipy.special as sp
import warnings

warnings.filterwarnings("ignore", category=si.IntegrationWarning)

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools"))
import quarz_read as qr  # noqa: E402

ALPHA = 7.2973525693e-3


def deck(d):
    """parameters of a run: the defaults of qed.in, overridden by the arguments in d.args"""
    p = {"n0": 1e18, "gamma": 1e5, "B": 137.614, "dt": 1.0}
    if os.path.exists(d + ".args"):
        for kv in open(d + ".args").read().splitlines():   # one key=value per line
            if "=" not in kv:
                continue
            k, v = kv.split("=", 1)
            if k == "fields.external_B":
                p["B"] = float(v.split()[0])
            elif k == "e.gamma":
                p["gamma"] = float(v)
            elif k == "time.dt":
                p["dt"] = float(v)
    p["eta"] = 3.8615926796e-11 * 5.64146e4 * np.sqrt(p["n0"]) / 2.99792458e10
    p["chi"] = p["eta"] * p["gamma"] * p["B"]
    p["C0"] = ALPHA / (np.sqrt(3) * np.pi * p["eta"] * p["gamma"])
    return p


# ---- the coefficients of eq. 4 (QEDRates.hpp), written out again
def bessel(u, chi):
    y = 2 * u / (3 * (1 - u) * chi)
    if y > 600:
        return 0.0, 0.0, 0.0
    ik = si.quad(lambda t: sp.kv(1 / 3, t), y, np.inf, limit=200)[0]
    return sp.kv(1 / 3, y), sp.kv(2 / 3, y), ik


def coef(u, chi):
    k1, k2, ik = bessel(u, chi)
    r = u / (1 - u)
    return dict(w=((1 - u) + 1 / (1 - u)) * k2 - ik, g=-u * k1, fs=2 * k2 - ik, f2=-r * k1, x3=k2, x3s=-r * k1)


def uint(f, lo=0.0, hi=1.0):
    """int_lo^hi f(u) du in r = u^{1/3}"""
    a, b = np.cbrt(lo), np.cbrt(hi)
    return si.quad(lambda r: 3 * r * r * f(r ** 3) if 0 < r < 1 else 0.0, a, b, limit=400, epsabs=0, epsrel=1e-10)[0]


def integrals(chi):
    return {k: uint(lambda u, k=k: coef(u, chi)[k]) for k in ("w", "g", "fs", "f2")}


def mark(bad):
    return "  <-- above tolerance" if bad else ""


def npush(d):
    return len(qr.read_qedlog(d + "/qed.txt")["e"]["step"])


def nlive(d):
    return qr.read_beamlog(d + "/beams.txt")["e"]["alive"][0]


EDGES = np.linspace(0, 1, 21)   # 20 bins in r = u^{1/3}


def expected_bins(chi, S):
    return np.array([uint(lambda u: coef(u, chi)["w"] + S * coef(u, chi)["g"], a ** 3, b ** 3)
                     for a, b in zip(EDGES[:-1], EDGES[1:])])


def spectrum(d, S):
    p = deck(d)
    ph = qr.read_photons(d + "/photons_e.bin")
    u = ph["energy"] / p["gamma"]
    T = npush(d) * p["dt"]
    Ne = nlive(d)
    eb = expected_bins(p["chi"], S) * Ne * T * p["C0"]
    E = eb.sum()
    N = len(u)
    sig = (N - E) / np.sqrt(E)
    h = np.histogram(np.cbrt(u), EDGES)[0]
    ok = eb > 5
    c2 = np.sum((h[ok] - eb[ok]) ** 2 / eb[ok]) / ok.sum()
    um = np.mean(u)
    Wu = uint(lambda x: x * (coef(x, p["chi"])["w"] + S * coef(x, p["chi"])["g"]))
    ue = Wu * Ne * T * p["C0"] / E
    print("  chi = %.3g, s2 = %+.0f: %d photons, expected %.1f (%+.2f sigma); spectrum chi^2/dof = %.2f (%d bins);"
          " <u> = %.5f (expected %.5f)%s" % (p["chi"], S, N, E, sig, c2, ok.sum(), um, ue, mark(abs(sig) > 4 or c2 > 2)))


def spindiff(dp, dm):
    p = deck(dp)
    up = qr.read_photons(dp + "/photons_e.bin")["energy"] / p["gamma"]
    um = qr.read_photons(dm + "/photons_e.bin")["energy"] / p["gamma"]
    T = npush(dp) * p["dt"]
    Ne = nlive(dp)
    I = integrals(p["chi"])
    ratio, ratio_e = len(up) / len(um), (I["w"] + I["g"]) / (I["w"] - I["g"])
    err = ratio * np.sqrt(1 / len(up) + 1 / len(um))
    hp, hm = np.histogram(np.cbrt(up), EDGES)[0], np.histogram(np.cbrt(um), EDGES)[0]
    g = np.array([uint(lambda u: coef(u, p["chi"])["g"], a ** 3, b ** 3) for a, b in zip(EDGES[:-1], EDGES[1:])])
    de = 2 * g * Ne * T * p["C0"]
    var = hp + hm
    ok = var > 10
    c2 = np.sum((hp - hm - de)[ok] ** 2 / var[ok]) / ok.sum()
    print("  chi = %.3g: photon number ratio s2 = +1 / -1: %.4f +- %.4f, expected (W + G)/(W - G) = %.4f (%+.2f sigma);"
          " (dN+ - dN-)/du vs 2 C0 g: chi^2/dof = %.2f%s" % (p["chi"], ratio, err, ratio_e, (ratio - ratio_e) / err, c2,
                                                             mark(abs(ratio - ratio_e) > 4 * err or c2 > 2)))


def spin(d, S0):
    p = deck(d)
    s = qr.read_spinlog(d + "/spin.txt")["e"]
    t, sx = s["t"], s["sx"]
    I = integrals(p["chi"])
    # eq. 14: dS/dt = C0 [(Xi - W) S + K - G]
    a, b = p["C0"] * (I["fs"] - I["w"]), p["C0"] * (I["f2"] - I["g"])
    Sinf = -b / a
    ref = (S0 - Sinf) * np.exp(a * t) + Sinf
    # statistical error of the mean over N particles: at most that of pure states, sqrt(1 - <s>^2) / sqrt(N)
    N = nlive(d)
    sig = np.sqrt(np.maximum(1 - ref ** 2, 0.02) / N)
    z = np.abs(sx - ref) / sig
    print("  chi = %.3g, S0 = %+.0f: <s2>(t) vs eq. 14 over %.1f relaxation times: max |diff| %.4f (%.1f sigma);"
          " final %.4f (eq. 14: %.4f, limit S_inf = %.4f)%s" % (p["chi"], S0, -a * t[-1], np.max(np.abs(sx - ref)),
                                                                 np.max(z), sx[-1], ref[-1], Sinf, mark(np.max(z) > 4.5)))


def polar(d, S):
    p = deck(d)
    ph = qr.read_photons(d + "/photons_e.bin")
    chi = p["chi"]
    num3 = uint(lambda u: coef(u, chi)["x3"] + S * coef(u, chi)["x3s"])
    den = uint(lambda u: coef(u, chi)["w"] + S * coef(u, chi)["g"])
    m = [np.mean(ph[k]) for k in ("xi1", "xi2", "xi3")]
    n = len(ph["xi3"])
    se = np.std(ph["xi3"]) / np.sqrt(n)
    # xi_3(u) in 10 bins of r vs the rate-weighted eq. 21
    u = ph["energy"] / p["gamma"]
    e = np.linspace(0, 1, 11)
    c2, nb = 0.0, 0
    for a, b in zip(e[:-1], e[1:]):
        sel = (np.cbrt(u) >= a) & (np.cbrt(u) < b)
        if sel.sum() < 50:
            continue
        ref = uint(lambda x: coef(x, chi)["x3"] + S * coef(x, chi)["x3s"], a ** 3, b ** 3) / \
            uint(lambda x: coef(x, chi)["w"] + S * coef(x, chi)["g"], a ** 3, b ** 3)
        sd = np.std(ph["xi3"][sel]) / np.sqrt(sel.sum())
        c2 += ((np.mean(ph["xi3"][sel]) - ref) / max(sd, 1e-12)) ** 2
        nb += 1
    c2 /= max(nb, 1)
    print("  chi = %.3g, s2 = %+.0f: <xi_3> = %.5f +- %.5f, eq. 22: %.5f (%+.2f sigma); <xi_1>, <xi_2> = %.1e, %.1e (0);"
          " xi_3(u) vs eq. 21: chi^2/dof = %.2f (%d bins)%s" % (chi, S, m[2], se, num3 / den, (m[2] - num3 / den) / se,
                                                                m[0], m[1], c2, nb,
                                                                mark(abs(m[2] - num3 / den) > 4 * se or c2 > 2 or
                                                                     max(abs(m[0]), abs(m[1])) > 1e-12)))


def energy(d, model="stochastic"):
    p = deck(d)
    b = qr.read_beamlog(d + "/beams.txt")["e"]
    t, gm = b["t"], b["gamma_mean"]
    # ODE: d gamma/dt = -C0(gamma) gamma int u w du = -alpha/(sqrt3 pi eta) Wu(eta gamma B)
    k = ALPHA / (np.sqrt(3) * np.pi * p["eta"])
    chis = np.geomspace(1e-3, p["chi"] * 1.01, 60)
    if model == "classical":   # g = 1: int u w du -> (2 pi / sqrt 3) chi^2
        Wu = 2 * np.pi / np.sqrt(3) * chis ** 2
    else:
        Wu = np.array([uint(lambda u, c=c: u * coef(u, c)["w"]) for c in chis])
    lw = np.polyfit(np.log(chis), np.log(Wu), 9)
    rhs = lambda tt, g: [-k * np.exp(np.polyval(lw, np.log(p["eta"] * g[0] * p["B"])))]
    sol = si.solve_ivp(rhs, (0, t[-1]), [p["gamma"]], t_eval=t, rtol=1e-10, atol=1e-6)
    dev = np.max(np.abs(gm / sol.y[0] - 1))
    if model != "stochastic":   # deterministic radiation reaction: no photons, every electron the same
        print("  qed.model = %s, chi_0 = %.3g: <gamma>(t)/gamma_ODE - 1 over %d steps (gamma %.4g -> %.4g): max %.2e;"
              " gamma_rms %.1e%s" % (model, p["chi"], len(t), gm[0], gm[-1], dev, b["gamma_rms"][-1],
                                     mark(dev > 1e-4 or b["gamma_rms"][-1] > 1e-6 * gm[-1])))
        return
    # energy balance: sum over photons of w k = sum over electrons of w (gamma_0 - gamma(T))
    ph = qr.read_photons(d + "/photons_e.bin")
    nT = b["step"][-1]
    sel = ph["step"] < nT
    Ne = b["alive"][-1]
    lost_e = Ne * (gm[0] - gm[-1])
    rad = np.sum(ph["energy"][sel])   # equal weights: in units of one electron
    print("  chi_0 = %.3g: <gamma>(t)/gamma_ODE - 1 over %d steps (gamma %.4g -> %.4g): max %.2e;"
          " energy balance (photons - electron loss)/loss = %.1e%s" % (p["chi"], len(t), gm[0], gm[-1], dev,
                                                                       rad / lost_e - 1, mark(dev > 0.01 or abs(rad / lost_e - 1) > 1e-6)))


def compare(da, db, label):
    a, b = qr.read_beamlog(da + "/beams.txt")["e"], qr.read_beamlog(db + "/beams.txt")["e"]
    n = a["alive"][-1]
    ga, gb, ra, rb = a["gamma_mean"][-1], b["gamma_mean"][-1], a["gamma_rms"][-1], b["gamma_rms"][-1]
    se = np.sqrt((ra ** 2 + rb ** 2) / n)
    print("  %s: final <gamma> %.6g vs %.6g (%+.2f sigma), gamma_rms %.5g vs %.5g%s" % (
        label, ga, gb, (ga - gb) / se, ra, rb, mark(abs(ga - gb) > 4 * se or abs(ra / rb - 1) > 0.15)))


def same(da, db, step0=0):
    """photon records and qed.txt lines of the steps both runs have (a run that writes a final checkpoint
    pushes once more in its last step)"""
    pa, pb = qr.read_photons(da + "/photons_e.bin"), qr.read_photons(db + "/photons_e.bin")
    la = [l for l in open(da + "/qed.txt") if not l.startswith("#")]
    lb = [l for l in open(db + "/qed.txt") if not l.startswith("#")]
    step1 = min(max(int(l.split()[1]) for l in la), max(int(l.split()[1]) for l in lb))
    sa = (pa["step"] >= step0) & (pa["step"] <= step1)
    sb = (pb["step"] >= step0) & (pb["step"] <= step1)
    ok = sa.sum() == sb.sum() and all(np.array_equal(pa[k][sa], pb[k][sb]) for k in qr.PHOTON_FIELDS)
    la = [l for l in la if step0 <= int(l.split()[1]) <= step1]
    lb = [l for l in lb if step0 <= int(l.split()[1]) <= step1]
    print("  steps %d-%d: photon records (%d) %s, qed.txt %s%s" % (step0, step1, sa.sum(), "identical" if ok else "DIFFER",
                                                                   "identical" if la == lb else "DIFFERS", mark(not ok or la != lb)))


def fixed(d):
    """<beam>.momentum_push = 0, <beam>.spin_precession = 0: momenta and spins of the first and last dump equal"""
    import glob
    fs = sorted(glob.glob(d + "/beam_e_*.bin"))
    a, b = qr.read_beam(fs[0]), qr.read_beam(fs[-1])
    dp = max(np.max(np.abs(a[k] - b[k])) for k in ("px", "py", "pz"))
    ds = max(np.max(np.abs(a[k] - b[k])) for k in ("sx", "sy", "sz")) if "sx" in a else 0.0
    moved = np.max(np.abs(a["xi"] - b["xi"])) + np.max(np.abs(a["x"] - b["x"])) + np.max(np.abs(a["y"] - b["y"]))
    print("  momentum_push = 0, spin_precession = 0 in B (no QED), %d steps: max |p change| %.1e, max |s change| %.1e"
          " (positions drift: %.2g)%s" % (int(fs[-1].split("_")[-1].split(".")[0]), dp, ds, moved, mark(dp != 0 or ds != 0)))


def gyration(d):
    p = deck(d)
    import glob
    f = sorted(glob.glob(d + "/beam_e_*.bin"))[-1]
    bm = qr.read_beam(f)
    step = int(f.split("_")[-1].split(".")[0])
    pp = np.sqrt(bm["px"] ** 2 + bm["py"] ** 2 + bm["pz"] ** 2)
    g = np.sqrt(1 + pp ** 2)
    th = np.arctan2(-bm["py"], bm["pz"])   # rotation about +x for an electron: p_y < 0
    th_e = p["B"] / g * (step - 0.5) * p["dt"]   # leapfrog: the momentum of step n is at t_n - dt/2
    print("  no QED, gamma = %.3g, B = %.4g, %d steps: rotation angle / (B t / gamma) - 1 = %.1e; |p| change %.1e%s" % (
        p["gamma"], p["B"], step, np.max(np.abs(th / th_e - 1)), np.max(np.abs(pp / pp[0] - 1)),
        mark(np.max(np.abs(th / th_e - 1)) > 1e-6)))


if __name__ == "__main__":
    c, a = sys.argv[1], sys.argv[2:]
    if c == "spectrum": spectrum(a[0], float(a[1]))
    elif c == "spindiff": spindiff(a[0], a[1])
    elif c == "spin": spin(a[0], float(a[1]))
    elif c == "polar": polar(a[0], float(a[1]))
    elif c == "energy": energy(a[0], a[1] if len(a) > 1 else "stochastic")
    elif c == "fixed": fixed(a[0])
    elif c == "compare": compare(a[0], a[1], a[2])
    elif c == "same": same(a[0], a[1], int(a[2]) if len(a) > 2 else 0)
    elif c == "gyration": gyration(a[0])
    else: sys.exit(__doc__)
