// Strong-field QED: spin- and polarization-resolved nonlinear Compton (photon emission) rate in the
// locally-constant-field approximation, for an electron or positron (|q| = e, m = m_e).
// Reference: O. Mathiak, A. Pukhov et al., QED module for QV3D (eqs. 2-8), following Li et al.,
// PRL 122, 154801 (2019) and PRL 124, 014801 (2020).
//
// Code units: time 1/omega_p, momentum m c, fields m c omega_p / e. One dimensionless constant,
//   eta = hbar omega_p / (m c^2)        (eta_from_n0_cm3),
// gives the quantum parameter and the rate prefactor:
//   chi = eta sqrt((gamma E + u x B)^2 - (u.E)^2)                   (eq. 2, u = gamma v = p / m c)
//   C0  = alpha / (sqrt(3) pi eta gamma)  [omega_p]                  (rate_prefactor)
//
// Fully resolved emission probability per unit time and photon energy fraction u = eps_gamma / eps_e:
//   d2W / du dt = C0/4 (w + <g, S_i> + <p_f, S_f> + <p_xi, xi>)      (eq. 3)
// summed over the final spin S_f and photon polarization (Stokes vector xi):  C0 (w + <g, S_i>).
// With y = 2u / (3 (1-u) chi) and K = K(y) the Bessel functions of Bessel.hpp (standard normalization;
// the 1/(sqrt(3) pi) is in C0), the coefficients of eq. 4 are
//   w   = (u^2 - 2u + 2)/(1-u) K_{2/3} - IntK_{1/3}                                    (4a)
//   g   = -u K_{1/3} e2                                                                (4b)
//   p_f = -u/(1-u) K_{1/3} e2 + [2 K_{2/3} - IntK_{1/3}] S_i
//         + u^2/(1-u) [K_{2/3} - IntK_{1/3}] <S_i, e_v> e_v                             (4c-e)
//   p_xi = ( u/(1-u) K_{1/3} <S_i, e1>,
//            [(2u - u^2)/(1-u) K_{2/3} - u IntK_{1/3}] <S_i, e_v>,
//            K_{2/3} - u/(1-u) K_{1/3} <S_i, e2> )                                        (4f-h)
// in the orthonormal basis (e1, e2, e_v): e_v along the momentum, e1 along the transverse part of the
// Lorentz force q (E + v x B) (so the charge sign enters through e1), e2 = e_v x e1. Stokes vector
// (xi_1, xi_2, xi_3): linear polarization at 45 degrees, circular, linear along e1 (+1) / e2 (-1).
// Sokolov-Ternov check: an electron in B = B e_z moving along x has e1 = +y, e2 = +z, and the spin
// relaxes to S_2 -> -8 sqrt(3)/15 (antiparallel to B); for a positron e2 = -z (parallel to B).
#pragma once

#include "Bessel.hpp"
#include "LogTable.hpp"
#include "Pushers.hpp"   // V3

#include <cmath>
#include <vector>

namespace quarz {
namespace qed {

using push::V3;

constexpr double ALPHA = 7.2973525693e-3;              // fine-structure constant (CODATA 2018)
constexpr double LAMBDA_C_CM = 3.8615926796e-11;        // reduced Compton wavelength hbar / (m_e c), cm
constexpr double Y_MAX = 700;
constexpr double RMAX_SAFETY = 1.01;                    // margin of the tabulated bound Rmax (RateTable)                           // y beyond which all K(y) are set to 0 (e^-700)

// eta = hbar omega_p / (m_e c^2) = lambda_C k_p for the plasma density n0 (cm^-3)
inline double eta_from_n0_cm3(double n0) {
    const double kp = 5.64146e4 * std::sqrt(n0) / 2.99792458e10;   // 1/cm (as in Beam.cpp)
    return LAMBDA_C_CM * kp;
}

// C0 in units of omega_p: the rate is C0 (w + <g, S>) per unit u
KOKKOS_INLINE_FUNCTION Real rate_prefactor(Real gamma, Real eta) {
    return Real(ALPHA * bessel::INV_PI_SQRT3) / (eta * gamma);
}

// quantum parameter chi of a particle with u = p / m c in the fields E, B (code units)
KOKKOS_INLINE_FUNCTION Real chi(const V3& u, const V3& E, const V3& B, Real eta) {
    const Real g = Kokkos::sqrt(Real(1) + push::dot(u, u));
    const V3 F = g * E + push::cross(u, B);
    const Real ue = push::dot(u, E);
    const Real c2 = push::dot(F, F) - ue * ue;
    return c2 > Real(0) ? eta * Kokkos::sqrt(c2) : Real(0);
}

struct Basis {
    V3 e1, e2, ev;
};

// chi and the basis (e1, e2, e_v) for a particle of charge sign q (+1 positron, -1 electron).
// Without a transverse force (chi = 0) e1, e2 are an arbitrary orthonormal pair.
KOKKOS_INLINE_FUNCTION Real chi_basis(const V3& u, const V3& E, const V3& B, Real q, Real eta, Basis& b) {
    const Real pu = Kokkos::sqrt(push::dot(u, u));
    b.ev = pu > Real(0) ? (Real(1) / pu) * u : V3{Real(0), Real(0), Real(1)};
    const Real g = Kokkos::sqrt(Real(1) + pu * pu);
    const V3 F = q * (g * E + push::cross(u, B));             // gamma x Lorentz force
    const V3 Ft = F - push::dot(F, b.ev) * b.ev;
    const Real ft = Kokkos::sqrt(push::dot(Ft, Ft));
    if (ft > Real(0)) {
        b.e1 = (Real(1) / ft) * Ft;
    } else {   // any unit vector perpendicular to e_v
        const V3 a = Kokkos::fabs(b.ev.x) < Real(0.9) ? V3{Real(1), Real(0), Real(0)} : V3{Real(0), Real(1), Real(0)};
        const V3 t = a - push::dot(a, b.ev) * b.ev;
        b.e1 = (Real(1) / Kokkos::sqrt(push::dot(t, t))) * t;
    }
    b.e2 = push::cross(b.ev, b.e1);
    return chi(u, E, B, eta);
}

// Bessel argument of the photon emission rate
KOKKOS_INLINE_FUNCTION Real y_arg(Real u, Real chi) {
    return Real(2) * u / (Real(3) * (Real(1) - u) * chi);
}

// The scalar coefficients of eq. 4 at (u, chi):
//   w (4a); g: g_rad = g e2 (4b);
//   p_f  = f2 e2 + fs S + fv <S, e_v> e_v                                (4c-e)
//   p_xi = (x1 <S, e1>, x2 <S, e_v>, x3 + x3s <S, e2>)                   (4f-h)
struct Coef {
    Real w = 0, g = 0;
    Real f2 = 0, fs = 0, fv = 0;
    Real x1 = 0, x2 = 0, x3 = 0, x3s = 0;
};

// The coefficients from the Bessel functions K13 = K_{1/3}(y), K23 = K_{2/3}(y), IK13 = IntK_{1/3}(y)
// at y = y_arg(u, chi), 0 < u < 1
KOKKOS_INLINE_FUNCTION Coef coefficients_from_K(Real u, Real K13, Real K23, Real IK13) {
    Coef c;
    const Real omu = Real(1) - u, r = u / omu;   // u / (1 - u)
    c.w = (omu + Real(1) / omu) * K23 - IK13;    // (u^2 - 2u + 2)/(1-u) = (1-u) + 1/(1-u)
    c.g = -u * K13;
    c.f2 = -r * K13;
    c.fs = Real(2) * K23 - IK13;
    c.fv = u * r * (K23 - IK13);
    c.x1 = r * K13;
    c.x2 = (Real(2) - u) * r * K23 - u * IK13;   // (2u - u^2)/(1-u) = (2 - u) u/(1-u)
    c.x3 = K23;
    c.x3s = -r * K13;
    return c;
}

namespace detail {
// the cases without Bessel functions: true if c is final, otherwise y is set
KOKKOS_INLINE_FUNCTION bool coefficients_edge(Real u, Real chi, Real& y, Coef& c) {
    if (!(chi > Real(0)) || u >= Real(1)) return true;
    if (u <= Real(0)) {   // y = 0: K_{2/3} = inf, all terms with a factor u vanish
        const Real inf = Kokkos::Experimental::infinity<Real>::value;
        c.w = inf; c.fs = inf; c.x3 = inf;
        return true;
    }
    y = y_arg(u, chi);
    return y >= Real(Y_MAX);
}
} // namespace detail

// Valid for 0 < u < 1, chi > 0; u <= 0 gives the (divergent) u -> 0 limits, u >= 1 or chi <= 0 zero.
KOKKOS_INLINE_FUNCTION Coef coefficients(Real u, Real chi) {
    Coef c;
    Real y = 0;
    if (detail::coefficients_edge(u, chi, y, c)) return c;
    Real K13, K23, IK13;
    bessel::synchrotron_functions(y, K13, K23, IK13);
    return coefficients_from_K(u, K13, K23, IK13);
}

// spin-averaged w_rad(u, chi) and the e2 component of g_rad(u, chi)
KOKKOS_INLINE_FUNCTION Real w_rad(Real u, Real chi) { return coefficients(u, chi).w; }
KOKKOS_INLINE_FUNCTION Real g_rad(Real u, Real chi) { return coefficients(u, chi).g; }

// emission rate per unit u and C0, summed over final states:  w + <g_rad, S>
KOKKOS_INLINE_FUNCTION Real rate(const Coef& c, const Basis& b, const V3& S) {
    return c.w + c.g * push::dot(S, b.e2);
}
// spin quantization vector p_f (eqs. 4c-e)
KOKKOS_INLINE_FUNCTION V3 p_f(const Coef& c, const Basis& b, const V3& S) {
    return c.f2 * b.e2 + c.fs * S + (c.fv * push::dot(S, b.ev)) * b.ev;
}
// photon polarization vector p_xi (eqs. 4f-h), components in the Stokes basis
KOKKOS_INLINE_FUNCTION V3 p_xi(const Coef& c, const Basis& b, const V3& S) {
    return V3{c.x1 * push::dot(S, b.e1), c.x2 * push::dot(S, b.ev), c.x3 + c.x3s * push::dot(S, b.e2)};
}
// fully resolved rate per unit u and C0 (eq. 3): (w + <g, S_i> + <p_f, S_f> + <p_xi, xi>) / 4
KOKKOS_INLINE_FUNCTION Real rate_resolved(const Coef& c, const Basis& b, const V3& Si, const V3& Sf, const V3& xi) {
    return Real(0.25) * (rate(c, b, Si) + push::dot(p_f(c, b, Si), Sf) + push::dot(p_xi(c, b, Si), xi));
}
// mixed-state final spin (eq. 7) and photon Stokes vector (eq. 20) of an emission
KOKKOS_INLINE_FUNCTION V3 spin_after_emission(const Coef& c, const Basis& b, const V3& S) {
    return (Real(1) / rate(c, b, S)) * p_f(c, b, S);
}
KOKKOS_INLINE_FUNCTION V3 stokes_of_photon(const Coef& c, const Basis& b, const V3& S) {
    return (Real(1) / rate(c, b, S)) * p_xi(c, b, S);
}

// Sampling density in r = u^{1/3} (Algorithm 1: u = r^3, du = 3 r^2 dr), per C0:
//   3 r^2 (w(r^3) + g(r^3) s2),  s2 = <S, e2>.
// Finite at r = 0: 3 Gamma(2/3) (3 chi)^{2/3}. Its integral over r in [0, 1] is W + G s2.
KOKKOS_INLINE_FUNCTION Real rate_r(Real r, Real chi, Real s2) {
    if (!(chi > Real(0))) return Real(0);
    if (r <= Real(0)) return Real(3 * 1.3541179394264004169) * Kokkos::cbrt(Real(9) * chi * chi);   // Gamma(2/3)
    const Coef c = coefficients(r * r * r, chi);
    return Real(3) * r * r * (c.w + c.g * s2);
}

// ---- look-up tables (logarithmic axes, LogTable.hpp)

// K_{1/3}, K_{2/3}, IntK_{1/3} on log y in [y_min, y_max]; stored as ln(e^y K), which is smooth in ln y
// over the whole range (power law ~ y^{-nu} for small y, ~ y^{-1/2} for large y). Outside the range the
// direct functions of Bessel.hpp are used. Accuracy: tests/test_qed_tables.cpp.
struct BesselTable {
    LogTable t;
    Real y_min = 0, y_max = 0;
    KOKKOS_INLINE_FUNCTION void operator()(Real y, Real& K13, Real& K23, Real& IK13) const {
        if (y >= y_min && y <= y_max) {
            Real F[3];
            t.eval_raw(y, F);
            K13 = Kokkos::exp(F[0] - y);
            K23 = Kokkos::exp(F[1] - y);
            IK13 = Kokkos::exp(F[2] - y);
        } else {
            bessel::synchrotron_functions(y, K13, K23, IK13);
        }
    }
};
BesselTable make_bessel_table(double y_min = 1e-6, double y_max = Y_MAX, double points_per_decade = 32);

// coefficients with the Bessel functions from the table
KOKKOS_INLINE_FUNCTION Coef coefficients(Real u, Real chi, const BesselTable& T) {
    Coef c;
    Real y = 0;
    if (detail::coefficients_edge(u, chi, y, c)) return c;
    Real K13, K23, IK13;
    T(y, K13, K23, IK13);
    return coefficients_from_K(u, K13, K23, IK13);
}

// u-integrals of Integrals (below) and an upper bound Rmax of the sampling density over r and the spin,
// max_r 3 r^2 (w + |g|) <= Rmax <= 1.02 max_r 3 r^2 (w + |g|)  (so that P <= C0 dt Rmax per step).
// The maximum sits at r = 0 (3 Gamma(2/3) (3 chi)^{2/3}) for chi < ~25 and near u -> 1 above, so it has
// a kink at chi ~ 25 that the cubic interpolation cannot follow (0.4 %): the table stores
// RMAX_SAFETY max_r(...) and the evaluation takes the maximum with the exact r = 0 value.
struct RateValues {
    Real W = 0, G = 0, Xi = 0, K = 0, Wu = 0, Gu = 0, Rmax = 0;
};
// on log chi in [chi_min, chi_max], all columns logarithmic (power laws: W ~ chi, G ~ chi^2, ... for
// small chi). Outside the range: power-law extrapolation of the end interval (approximate).
struct RateTable {
    LogTable t;
    KOKKOS_INLINE_FUNCTION RateValues operator()(Real chi) const {
        RateValues r;
        if (!(chi > Real(0))) return r;
        Real v[7];
        t.eval(chi, v);
        r.W = v[0]; r.G = v[1]; r.Xi = v[2]; r.K = v[3]; r.Wu = v[4]; r.Gu = v[5];
        const Real R0 = Real(RMAX_SAFETY) * rate_r(Real(0), chi, Real(0));
        r.Rmax = v[6] > R0 ? v[6] : R0;
        return r;
    }
};
RateTable make_rate_table(double chi_min = 1e-5, double chi_max = 1e4, double points_per_decade = 32);

// ---- table files (plain text, LogTable format, "*.lut" by convention; exact round trip).
// The files carry meta entries: table (qed.bessel / qed.rate), version (TABLE_VERSION, raised whenever
// the tabulated functions change), the effective points_per_decade, and for the rate table rmax_safety.
constexpr int TABLE_VERSION = 1;
void save_bessel_table(const BesselTable& T, const std::string& path);
void save_rate_table(const RateTable& T, const std::string& path);
// throw if the file is not a table of that kind and version
BesselTable load_bessel_table(const std::string& path);
RateTable load_rate_table(const std::string& path);
// The table for these parameters: loaded from path if that file exists and holds a table of this kind,
// version and parameters (same axis ends and number of points); otherwise built, and saved to path if write (with MPI: write on one rank only).
// An empty path just builds. status (optional) receives "loaded", "built", "built (saved)" or
// "built (<reason>)".
BesselTable bessel_table(const std::string& path, double y_min = 1e-6, double y_max = Y_MAX,
                         double points_per_decade = 32, bool write = true, std::string* status = nullptr);
RateTable rate_table(const std::string& path, double chi_min = 1e-5, double chi_max = 1e4,
                     double points_per_decade = 32, bool write = true, std::string* status = nullptr);

// ---- u-integrals (host; quadrature). All per C0, i.e. the rates are C0 times these.
struct Integrals {
    double W = 0;    // int w du                     total emission rate (spin-averaged)
    double G = 0;    // int g du                     spin-dependent rate: C0 (W + G <S, e2>)
    double Xi = 0;   // int (2 K_{2/3} - IntK_{1/3}) du    (coefficient fs of S in p_f)
    double K = 0;    // int -u/(1-u) K_{1/3} du             (coefficient f2 of e2 in p_f)
    double Wu = 0;   // int u w du                   radiated power: P = eps_e C0 (Wu + Gu <S, e2>)
    double Gu = 0;   // int u g du
};
// relative accuracy ~1e-12 (adaptive Gauss-Kronrod in r = u^{1/3})
Integrals integrals(double chi);
// quantum correction of the radiated power, P = P_cl g(chi), P_cl = (2/3) alpha m^2 c^4 chi^2 / hbar:
// g = Wu / ((2 pi / sqrt 3) chi^2)  ->  1 for chi -> 0
double gaunt(double chi);
// equilibrium polarization along e2 in a constant field (eq. 14): -(K - G)/(Xi - W) -> -8 sqrt(3)/15
double polarization_limit(double chi);
// max over r in [0, 1] of the sampling density for the least favourable spin, 3 r^2 (w + |g|)
// (grid of 400 points and golden-section refinement, relative accuracy ~1e-10)
double max_rate_r(double chi);

// adaptive Gauss-Kronrod (7-15) quadrature of f over [a, b] to relative accuracy rtol (host):
// the interval with the largest error estimate |K15 - G7| is bisected until the sum of the
// estimates is below rtol |I| (at most max_intervals intervals)
template <class F>
double integrate(F f, double a, double b, double rtol = 1e-13, int max_intervals = 4000) {
    static constexpr double xk[8] = {0.991455371120812639206854697526329, 0.949107912342758524526189684047851,
                                     0.864864423359769072789712788640926, 0.741531185599394439863864773280788,
                                     0.586087235467691130294144845693013, 0.405845151377397166906606412076961,
                                     0.207784955007898467600689403773245, 0.0};
    static constexpr double wk[8] = {0.022935322010529224963732008058970, 0.063092092629978553290700663189204,
                                     0.104790010322250183839876322541518, 0.140653259715525918745189590510238,
                                     0.169004726639267902826583426598550, 0.190350578064785409913256402421014,
                                     0.204432940075298892414161999234649, 0.209482141084727828012999174891714};
    static constexpr double wg[4] = {0.129484966168869693270611432679082, 0.279705391489276667901467771423780,
                                     0.381830050505118944950369775488975, 0.417959183673469387755102040816327};
    struct Seg { double a, b, I, e; };
    auto gk = [&](double lo, double hi) {
        const double c = 0.5 * (lo + hi), h = 0.5 * (hi - lo);
        const double fc = f(c);
        double K = wk[7] * fc, G = wg[3] * fc;
        for (int i = 0; i < 7; ++i) {
            const double s = f(c - h * xk[i]) + f(c + h * xk[i]);
            K += wk[i] * s;
            if (i % 2 == 1) G += wg[i / 2] * s;
        }
        return Seg{lo, hi, K * h, std::abs((K - G) * h)};
    };
    std::vector<Seg> segs{gk(a, b)};
    for (;;) {
        double I = 0, E = 0;
        size_t imax = 0;
        for (size_t i = 0; i < segs.size(); ++i) {
            I += segs[i].I;
            E += segs[i].e;
            if (segs[i].e > segs[imax].e) imax = i;
        }
        if (E <= rtol * std::abs(I) || static_cast<int>(segs.size()) >= max_intervals) return I;
        const Seg s = segs[imax];
        const double m = 0.5 * (s.a + s.b);
        segs[imax] = gk(s.a, m);
        segs.push_back(gk(m, s.b));
    }
}

} // namespace qed
} // namespace quarz
