#!/usr/bin/env python3
"""Reference values for tests/test_bessel.cpp and tests/test_qed_rates.cpp (mpmath, 40 digits).

Independent of the C++ code: K_nu from mpmath.besselk, IntK_{1/3} from the Airy integral
IntK_{1/3}(x) = pi/sqrt(3) - pi sqrt(3) int_0^z Ai(s) ds,  z = (3x/2)^{2/3}  (computed with 120 digits),
the coefficients of eq. 4 (QEDRates.hpp) written out directly and the u-integrals by mpmath.quad.
usage: python3 tests/qed_reference.py [bessel|rates]      (rates: ~15 min)
"""
import sys
import mpmath as mp

mp.mp.dps = 40
NU1, NU2 = mp.mpf(1) / 3, mp.mpf(2) / 3


def intk13(x):
    x = mp.mpf(x)
    if x > 20:   # the Airy form cancels as e^-x: integrate K_{1/3} directly
        return mp.quad(lambda t: mp.besselk(NU1, t), [x, x + 1, x + 5, x + 30, mp.inf])
    with mp.workdps(mp.mp.dps + 15):
        z = (mp.mpf(3) * x / 2) ** (mp.mpf(2) / 3)
        v = mp.pi / mp.sqrt(3) - mp.pi * mp.sqrt(3) * mp.airyai(z, derivative=-1)
    return +v


def coef(u, chi):
    u, chi = mp.mpf(u), mp.mpf(chi)
    y = 2 * u / (3 * (1 - u) * chi)
    if y > 300:   # e^-300: negligible in the u-integrals
        return dict(w=0, g=0, f2=0, fs=0, fv=0, x1=0, x2=0, x3=0, x3s=0)
    k1, k2, ik = mp.besselk(NU1, y), mp.besselk(NU2, y), intk13(y)
    return dict(w=(u * u - 2 * u + 2) / (1 - u) * k2 - ik, g=-u * k1,
                f2=-u / (1 - u) * k1, fs=2 * k2 - ik, fv=u * u / (1 - u) * (k2 - ik),
                x1=u / (1 - u) * k1, x2=(2 * u - u * u) / (1 - u) * k2 - u * ik,
                x3=k2, x3s=-u / (1 - u) * k1)


def uint(f, chi):
    # int_0^1 f(u) du in r = u^{1/3}; break points where the exponential cut-off sets in
    rc = [mp.cbrt(min(mp.mpf(1) / 2, t * chi)) for t in (1, 10, 100)]
    pts = sorted(set([mp.mpf(0)] + rc + [mp.mpf(1) - mp.mpf(10) ** -k for k in (1, 2, 3)] + [mp.mpf(1)]))
    return mp.quad(lambda r: 3 * r * r * f(r ** 3) if r < 1 else mp.mpf(0), pts)


def bessel():
    for s in ['1e-8', '1e-5', '1e-3', '0.01', '0.1', '0.3', '0.7', '1', '1.5', '2', '2.4', '2.5', '2.6', '2.9',
              '3', '3.1', '4', '6', '10', '20', '50', '100', '300', '700']:
        a = mp.mpf(s)
        print("    {%s, %s, %s, %s}," % (s, mp.nstr(mp.besselk(NU1, a) * mp.exp(a), 20),
                                       mp.nstr(mp.besselk(NU2, a) * mp.exp(a), 20), mp.nstr(intk13(a) * mp.exp(a), 20)))


def rates():
    mp.mp.dps = 25   # 18 printed digits; the C++ functions are accurate to ~1e-12
    print("// u, chi, w, g, f2, fs, fv, x1, x2, x3, x3s")
    for u, chi in [('0.001', '0.1'), ('0.05', '0.1'), ('0.3', '0.1'), ('0.1', '1'), ('0.5', '1'), ('0.9', '1'),
                   ('0.2', '10'), ('0.9', '10'), ('0.999', '10'), ('0.5', '1000')]:
        c = coef(u, chi)
        print("    {%s, %s, %s}," % (u, chi, ", ".join(mp.nstr(c[k], 18) for k in
                                                       ('w', 'g', 'f2', 'fs', 'fv', 'x1', 'x2', 'x3', 'x3s'))))
    print("// chi, W, G, Xi, K, Wu, Gu, gaunt, polarization limit")
    for chi in ['1e-3', '0.01', '0.1', '1', '10', '100']:
        x = mp.mpf(chi)
        I = {k: uint(lambda u, k=k: coef(u, x)[k], x) for k in ('w', 'g', 'fs', 'f2')}
        Wu = uint(lambda u: u * coef(u, x)['w'], x)
        Gu = uint(lambda u: u * coef(u, x)['g'], x)
        y = lambda u: 2 * u / (3 * (1 - u) * x)
        A = uint(lambda u: 0 if y(u) > 300 else u * u / (1 - u) * mp.besselk(NU1, y(u)), x)
        B = uint(lambda u: 0 if y(u) > 300 else u * u / (1 - u) * mp.besselk(NU2, y(u)), x)
        gaunt = Wu / (2 * mp.pi / mp.sqrt(3) * x * x)
        print("    {%s, %s}," % (chi, ", ".join(mp.nstr(v, 18) for v in
                                               (I['w'], I['g'], I['fs'], I['f2'], Wu, Gu, gaunt, -A / B))))


if __name__ == "__main__":
    {'bessel': bessel, 'rates': rates}[sys.argv[1] if len(sys.argv) > 1 else 'rates']()
