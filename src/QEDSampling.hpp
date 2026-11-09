// Photon emission samplers for nonlinear Compton scattering (QEDRates.hpp).
//
// A lepton with quantum parameter chi and spin component s2 = <S, e2> emits photons of energy
// fraction u = eps_gamma / eps_e at the rate (per unit time and u)
//
//     dN / dt du = C0 (w(u) + g(u) s2)        (eq. 3, summed over final spin and polarization)
//
// Two ways to draw the emission times and the photon energies from this rate:
//
// 1. REJECTION sampler (Algorithm 1 of the QED module): every (sub-)step draws a candidate u and
//    accepts it with the probability of emitting exactly this photon in this step.
//    Needs only the Bessel functions. Requires a step short enough that the probability is <= 1.
//
// 2. INVERSION sampler (optical depth): every particle carries an optical depth tau that decreases
//    at the total rate C0 (W + G s2); when it reaches 0 the particle emits, and the photon energy is
//    drawn by inverting the cumulative spectrum. Needs the rate table and the spectrum table below.
//
// The samplers take their uniform random numbers as arguments (counter-based generators of the
// caller), so they are deterministic functions of their inputs.
#pragma once

#include "QEDRates.hpp"

#include <string>

namespace quarz {
namespace qed {

// =====================================================================================================
// 1. Rejection sampler
// =====================================================================================================
//
// In a step dt the probability to emit a photon in [u, u + du] is  dP = C0 dt (w + g s2) du.
// With u = r^3 (du = 3 r^2 dr) the density in r is finite everywhere (w ~ u^{-2/3} for u -> 0):
//
//     dP = C0 dt * rate_r(r) dr,     rate_r(r) = 3 r^2 (w(r^3) + g(r^3) s2).
//
// Draw r uniformly in [0, 1) and accept with probability P = C0 dt rate_r(r). Then the accepted u are
// distributed exactly as the spectrum, and the probability of emitting at all is C0 dt (W + G s2), the
// exact mean number of photons per step. This needs P <= 1 for every r, i.e. C0 dt Rmax <= 1 (Rmax from
// the RateTable): a longer step is split into sub-steps. Only one photon per sub-step can be emitted;
// the mean number of photons stays exact, only the variance is slightly reduced (by the probability
// p_max of two emissions in one sub-step).

// Number of sub-steps of a step with C0 dt (C0 = rate_prefactor) so that every acceptance
// probability stays below p_max (0 < p_max <= 1).
KOKKOS_INLINE_FUNCTION int rejection_substeps(Real C0_dt, Real Rmax, Real p_max) {
    const Real n = Kokkos::ceil(C0_dt * Rmax / p_max);
    return n < Real(1) ? 1 : static_cast<int>(n);
}

// One trial of the rejection sampler in a (sub-)step with C0 dt.
//   random_r, random_accept: independent uniform numbers in [0, 1).
// Returns the energy fraction u of the emitted photon, or 0 if no photon is emitted.
KOKKOS_INLINE_FUNCTION Real rejection_trial(Real chi, Real s2, Real C0_dt, Real random_r, Real random_accept,
                                            const BesselTable& bessel) {
    // candidate photon energy fraction
    const Real r = random_r;
    const Real u = r * r * r;
    // probability of emitting exactly this photon in this step
    const Coef c = coefficients(u, chi, bessel);
    const Real density_r = r > Real(0) ? Real(3) * r * r * (c.w + c.g * s2) : rate_r(Real(0), chi, s2);
    const Real P = C0_dt * density_r;
    return random_accept < P ? u : Real(0);
}

// =====================================================================================================
// 2. Inversion sampler (optical depth)
// =====================================================================================================
//
// Emission is a Poisson process with the total rate  R = C0 (W + G s2)  (RateTable). Its waiting time
// is drawn once as an optical depth tau = -ln(xi) (xi uniform in (0, 1]), which then decreases as
// d tau / dt = -R. The photon is emitted when tau reaches 0, and tau is redrawn.
//
// The energy fraction u of the photon solves  F(u) = xi' F(1)  for another uniform xi', with the
// cumulative spectrum
//
//     F(u) = Wc(u) + s2 Gc(u),   Wc(u) = int_0^u w du',   Gc(u) = int_0^u g du'
//
// (F is increasing since w + g s2 > 0 for |s2| <= 1). Wc and Gc are tabulated in SpectrumTable.

// a fresh optical depth from a uniform number xi in [0, 1)
KOKKOS_INLINE_FUNCTION Real new_optical_depth(Real xi) {
    return -Kokkos::log(Real(1) - xi);   // 1 - xi in (0, 1]
}

// Advance the optical depth tau by up to dt at the total rate R.
// Returns the time until emission if tau reaches 0 within dt (tau is then 0: the caller emits the photon
// and draws a new tau), otherwise dt (no emission, tau reduced by R dt).
KOKKOS_INLINE_FUNCTION Real advance_optical_depth(Real& tau, Real R, Real dt) {
    const Real decrease = R * dt;
    if (decrease < tau) {
        tau -= decrease;
        return dt;
    }
    const Real t_emit = tau / R;
    tau = Real(0);
    return t_emit;
}
KOKKOS_INLINE_FUNCTION bool emits(Real t_advanced, Real dt) { return t_advanced < dt; }

// Cumulative spectrum Wc(u; chi), Gc(u; chi) on two logarithmic axes:
//   chi:  logarithmic (a LogTable over chi, cubic interpolation in ln chi);
//   u:    the logit L = ln(u / (1 - u)) on a uniform grid L_0 ... L_{nu-1}, i.e. logarithmic in u for
//         u -> 0 and logarithmic in 1 - u for u -> 1 (where the spectrum ends for large chi).
// The LogTable has 2 nu columns: Wc(u_j) for j = 0 ... nu-1, then Gc(u_j).
// Below u_0 the leading power laws are used (w ~ u^{-2/3}, g ~ u^{2/3}, so Wc ~ u^{1/3}, Gc ~ u^{5/3});
// above u_{nu-1} (1 - u < 1e-8 by default) the spectrum is taken as zero.
class SpectrumTable {
public:
    LogTable t;      // over chi; columns Wc(u_j), Gc(u_j)
    int nu = 0;      // number of u nodes
    Real L0 = 0;     // logit of the first u node
    Real dL = 1;     // spacing in L

    KOKKOS_INLINE_FUNCTION static Real u_of_L(Real L) { return Real(1) / (Real(1) + Kokkos::exp(-L)); }
    KOKKOS_INLINE_FUNCTION Real L_node(int j) const { return L0 + j * dL; }

    // F(u_j) = Wc(u_j) + s2 Gc(u_j) at the chi of the stencil
    KOKKOS_INLINE_FUNCTION Real F_node(const LogTable::Stencil& at_chi, Real s2, int j) const {
        return t.eval(at_chi, j) + s2 * t.eval(at_chi, nu + j);
    }

    // The photon energy fraction u with F(u) = xi F(1), xi uniform in [0, 1).
    // bessel: for the slope dF/du = w + g s2 used by the cubic interpolation between the u nodes.
    KOKKOS_INLINE_FUNCTION Real sample_u(Real chi, Real s2, Real xi, const BesselTable& bessel) const {
        const LogTable::Stencil at_chi = t.stencil_at(chi);
        const Real target = xi * F_node(at_chi, s2, nu - 1);

        // (a) below the first node: invert the power laws  Wc_0 z + s2 Gc_0 z^5 = target, z = (u/u_0)^{1/3}
        const Real F0 = F_node(at_chi, s2, 0);
        if (target <= F0) return u_below_first_node(at_chi, s2, target);

        // (b) find the interval [j, j + 1] of u nodes with F_j <= target < F_{j+1} (bisection)
        int lo = 0, hi = nu - 1;
        while (hi - lo > 1) {
            const int mid = (lo + hi) / 2;
            if (F_node(at_chi, s2, mid) <= target) lo = mid;
            else hi = mid;
        }

        // (c) inside the interval: F is a cubic in L (Hermite: values and slopes at both ends);
        //     solve F(L) = target by Newton's method, safeguarded by bisection
        return u_in_interval(at_chi, chi, s2, target, lo, bessel);
    }

private:
    KOKKOS_INLINE_FUNCTION Real u_below_first_node(const LogTable::Stencil& at_chi, Real s2, Real target) const {
        const Real W0 = t.eval(at_chi, 0), G0 = s2 * t.eval(at_chi, nu);
        Real z = target / (W0 + G0);   // the solution if G0 = 0
        for (int it = 0; it < 4; ++it) {   // G0 z^5 is a small correction (y^{4/3} at u_0): converges at once
            const Real z4 = z * z * z * z;
            z -= (W0 * z + G0 * z4 * z - target) / (W0 + Real(5) * G0 * z4);
        }
        const Real u0 = u_of_L(L0);
        return u0 * z * z * z;
    }

    KOKKOS_INLINE_FUNCTION Real u_in_interval(const LogTable::Stencil& at_chi, Real chi, Real s2, Real target, int j,
                                              const BesselTable& bessel) const {
        // end points: values F and slopes dF/dL = (w + g s2) du/dL, du/dL = u (1 - u)
        const Real La = L_node(j), Lb = L_node(j + 1);
        const Real Fa = F_node(at_chi, s2, j), Fb = F_node(at_chi, s2, j + 1);
        auto slope = [&](Real L) {
            const Real u = u_of_L(L);
            const Coef c = coefficients(u, chi, bessel);
            return (c.w + c.g * s2) * u * (Real(1) - u);
        };
        const Real Da = slope(La) * dL, Db = slope(Lb) * dL;   // slopes per unit p = (L - La) / dL
        // cubic Hermite F(p), p in [0, 1]
        auto F = [&](Real p) {
            const Real p2 = p * p, p3 = p2 * p;
            return (Real(2) * p3 - Real(3) * p2 + Real(1)) * Fa + (p3 - Real(2) * p2 + p) * Da +
                   (Real(3) * p2 - Real(2) * p3) * Fb + (p3 - p2) * Db;
        };
        auto dF = [&](Real p) {
            const Real p2 = p * p;
            return (Real(6) * p2 - Real(6) * p) * Fa + (Real(3) * p2 - Real(4) * p + Real(1)) * Da +
                   (Real(6) * p - Real(6) * p2) * Fb + (Real(3) * p2 - Real(2) * p) * Db;
        };
        // Newton from the linear interpolation, kept inside the bracket [a, b]
        Real a = 0, b = 1;
        Real p = (Fb > Fa) ? (target - Fa) / (Fb - Fa) : Real(0.5);
        for (int it = 0; it < 30; ++it) {
            const Real f = F(p) - target;
            if (f > Real(0)) b = p; else a = p;
            const Real d = dF(p);
            Real next = d > Real(0) ? p - f / d : Real(-1);
            if (!(next > a && next < b)) next = Real(0.5) * (a + b);   // outside the bracket: bisect
            if (Kokkos::fabs(next - p) < Real(1e-14)) { p = next; break; }
            p = next;
        }
        return u_of_L(La + p * dL);
    }
};

// Table parameters. The u axis runs from u_min to 1 - one_minus_u_min in the logit L.
struct SpectrumTableParams {
    double chi_min = 1e-5, chi_max = 1e4, chi_points_per_decade = 16;
    double u_min = 1e-12, one_minus_u_min = 1e-8, u_points_per_decade = 24;   // per decade of u or 1 - u
};
SpectrumTable make_spectrum_table(const SpectrumTableParams& p = SpectrumTableParams());
void save_spectrum_table(const SpectrumTable& T, const std::string& path);
SpectrumTable load_spectrum_table(const std::string& path);   // throws if not a spectrum table of this version
// loaded from path if it holds a table with these parameters, otherwise built (and saved if write);
// status as for rate_table()
SpectrumTable spectrum_table(const std::string& path, const SpectrumTableParams& p = SpectrumTableParams(),
                             bool write = true, std::string* status = nullptr);

} // namespace qed
} // namespace quarz
