// Photon emission (nonlinear Compton scattering) inside the beam push: what happens to one beam
// electron or positron during one sub-step of the push (Beam::advance).
//
// Per sub-step of length h, after the Lorentz kick (momentum u, spin S, fields E, B at the particle):
//   1. chi and the basis (e1, e2, e_v) from u, E, B; the rate prefactor C0 = alpha / (sqrt(3) pi eta gamma)
//   2. emission, with one of the samplers of QEDSampling.hpp:
//        inversion (default): the optical depth tau decreases at the total rate C0 (W + G s2); each time
//                             it reaches 0 a photon is emitted (several per sub-step are possible)
//        rejection:           one trial: candidate u = r^3, accepted with probability C0 h 3 r^2 (w + g s2)
//   3. for every photon: recoil (the lepton keeps its direction, eps -> (1 - u) eps), the new spin
//      (eqs. 5-7), the photon's Stokes vector (eq. 20), and a record for the output
//   4. without emission: the spin changes by the selection bias: states that emit less become more
//      likely. Rejection (a trial per sub-step): eq. 8, conditioned on no emission in the sub-step.
//      Inversion (continuous time): the exact solution of the same process in continuous time
//      (spin_without_emission_continuous); eq. 8 there would overstate the change by 1/p0.
// Spin models: mixed (default) - the spin vector of a macro-particle is an average, |S| <= 1, and takes
// the expectation values S_f = p_f / (w + <g, S>); pure - |S| = 1, the spin collapses to +-p_f/|p_f|
// with the probabilities (1 +- |p_f|/(w + <g, S>)) / 2 (the same rule for the photon polarization and,
// with the rejection sampler, for the spin change without emission; in continuous time a pure state
// stays pure without a collapse).
// Beams without spin use the spin-averaged rates (s2 = 0) and no spin update.
#pragma once

#include "Ionization.hpp"   // ion_rand: counter-based random numbers
#include "QEDSampling.hpp"

namespace quarz {
namespace qed {

enum Sampler : int { INVERSION = 0, REJECTION = 1 };
enum SpinModel : int { MIXED = 0, PURE = 1 };
// stochastic: photon emission (the samplers); continuous / classical: deterministic radiation reaction,
// d gamma/dt = -P with the mean power of the stochastic emission P = alpha/(sqrt3 pi eta) int u (w + g s2) du
// (synchrotron loss x g(chi)), or the classical P = (2 alpha / 3 eta) chi^2 (g = 1); no photons, the spin
// keeps only its precession
enum Model : int { STOCHASTIC = 0, CONTINUOUS = 1, CLASSICAL = 2 };

// everything the push needs: parameters and the tables (device-copyable)
struct Emission {
    int sampler = INVERSION;
    int spin_model = MIXED;
    int model = STOCHASTIC;
    Real eta = 0;              // hbar omega_p / (m c^2)
    Real charge = -1;          // charge sign of the lepton (enters the basis through e1)
    Real p_max = 0.1;          // largest emission probability per sub-step (sets the number of sub-steps)
    int max_substeps = 1000;
    bool recoil = true;        // testing: false keeps the momentum (fixed chi)
    bool spin_change = true;   // testing: false keeps the spin
    BesselTable bessel;
    RateTable rates;
    SpectrumTable spectrum;
};

// uniform random numbers of one particle in one step: ion_rand(key, particle, 0), (.., 1), ...
struct ParticleRandom {
    uint64_t key, particle, draw = 0;
    KOKKOS_INLINE_FUNCTION Real operator()() { return ion_rand(key, particle, draw++); }
};

// an emitted photon
struct Photon {
    Real t;        // time of emission, from the start of the sub-step
    Real u;        // energy fraction of the emitter
    V3 k;          // momentum (m c), along the emitter's momentum
    V3 stokes;     // (xi_1, xi_2, xi_3) in the basis of the emitter
    Real chi;      // chi of the emitter
};

// Number of sub-steps of the particle push in a step dt, so that the emission probability in a
// sub-step stays below p_max (rejection_substeps with the bound Rmax of the rate table).
KOKKOS_INLINE_FUNCTION int substeps(const Emission& Q, const V3& u, const V3& E, const V3& B, Real dt) {
    const Real chi_now = chi(u, E, B, Q.eta);
    if (!(chi_now > Real(0))) return 1;
    const Real gamma = Kokkos::sqrt(Real(1) + push::dot(u, u));
    const int n = rejection_substeps(rate_prefactor(gamma, Q.eta) * dt, Q.rates(chi_now).Rmax, Q.p_max);
    return n < Q.max_substeps ? n : Q.max_substeps;
}

// A mixed-state vector m (|m| <= 1) measured as a pure state: +m/|m| with probability (1 + |m|)/2,
// otherwise -m/|m| (the mean stays m).
KOKKOS_INLINE_FUNCTION V3 collapse(const V3& m, Real random) {
    const Real len = Kokkos::sqrt(push::dot(m, m));
    if (!(len > Real(0))) return m;
    const Real sign = random < Real(0.5) * (Real(1) + len) ? Real(1) : Real(-1);
    return (sign / len) * m;
}

// Spin after a time t without emission (eq. 8): p_NR = S (1 - C0 t W) - C0 t G e2, normalized by the
// probability of no emission, p0 = 1 - C0 t (W + G s2). (|p_NR| <= p0, so the length stays <= 1.)
KOKKOS_INLINE_FUNCTION V3 spin_without_emission(const V3& S, const Basis& b, Real C0_t, const RateValues& r) {
    const Real s2 = push::dot(S, b.e2);
    const V3 p_NR = (Real(1) - C0_t * r.W) * S - (C0_t * r.G) * b.e2;
    const Real p0 = Real(1) - C0_t * (r.W + r.G * s2);
    return (Real(1) / p0) * p_NR;
}

// The same in continuous time, for a time t without emission (inversion sampler). The unnormalized
// state (q0, q) = (survival probability, survival probability x S) obeys
//   dq0/dt = -C0 (W q0 + G q.e2),   dq/dt = -C0 (W q + G q0 e2),
// whose solution (the W terms only give a common factor e^{-C0 W t}) with c = C0 G t is
//   q0 = cosh c - s2 sinh c,   q.e2 = s2 cosh c - sinh c,   q_perp = S_perp.
// The new spin is q / q0; first order: dS/dt = C0 G (s2 S - e2). A pure state stays pure (|S| = 1).
KOKKOS_INLINE_FUNCTION V3 spin_without_emission_continuous(const V3& S, const Basis& b, Real C0_t, const RateValues& r) {
    const Real c = C0_t * r.G;
    const Real ch = Kokkos::cosh(c), sh = Kokkos::sinh(c);
    const Real s2 = push::dot(S, b.e2);
    const V3 S_perp = S - s2 * b.e2;
    const Real q0 = ch - s2 * sh;
    return (Real(1) / q0) * (S_perp + (s2 * ch - sh) * b.e2);
}

// One emission of a photon with energy fraction u_ph: changes the momentum p and the spin S,
// returns the photon (time t from the start of the sub-step).
KOKKOS_INLINE_FUNCTION Photon emit(const Emission& Q, bool has_spin, Real u_ph, Real chi_e, const Basis& b, Real t,
                                   V3& p, V3& S, ParticleRandom& random) {
    const Real gamma = Kokkos::sqrt(Real(1) + push::dot(p, p));
    const Coef c = coefficients(u_ph, chi_e, Q.bessel);
    const V3 S0 = has_spin ? S : V3{Real(0), Real(0), Real(0)};
    const Real rate_here = rate(c, b, S0);   // w + <g, S>: normalizes the final states

    Photon ph;
    ph.t = t;
    ph.u = u_ph;
    ph.chi = chi_e;
    ph.k = (u_ph * gamma) * b.ev;
    // photon polarization: expectation value (mixed) or a pure state
    const V3 stokes = (Real(1) / rate_here) * p_xi(c, b, S0);
    ph.stokes = Q.spin_model == PURE ? collapse(stokes, random()) : stokes;
    // lepton spin after the emission
    if (has_spin && Q.spin_change) {
        const V3 Sf = (Real(1) / rate_here) * p_f(c, b, S0);
        S = Q.spin_model == PURE ? collapse(Sf, random()) : Sf;
    }
    // recoil: energy (1 - u) eps, same direction
    if (Q.recoil) {
        const Real g_after = (Real(1) - u_ph) * gamma;
        const Real p_after = g_after > Real(1) ? Kokkos::sqrt(g_after * g_after - Real(1)) : Real(0);
        p = p_after * b.ev;
    }
    return ph;
}

// Deterministic radiation reaction over a time h (qed.model = continuous | classical): the lepton keeps
// its direction, d gamma/dt = -P(chi), chi ~ gamma in the fields of the sub-step; second order (midpoint).
KOKKOS_INLINE_FUNCTION void radiate_continuously(const Emission& Q, bool has_spin, const V3& E, const V3& B, Real h,
                                                 V3& p, const V3& S) {
    Basis b;
    const Real chi0 = chi_basis(p, E, B, Q.charge, Q.eta, b);
    if (!(chi0 > Real(0))) return;
    const Real gamma0 = Kokkos::sqrt(Real(1) + push::dot(p, p));
    const Real s2 = has_spin ? push::dot(S, b.e2) : Real(0);
    const Real k = Real(ALPHA * bessel::INV_PI_SQRT3) / Q.eta;   // gamma C0: P = gamma C0 int u (w + g s2) du
    auto loss_rate = [&](Real gamma) {   // d gamma/dt at energy gamma (chi scales with gamma)
        const Real chi_g = chi0 * gamma / gamma0;
        if (Q.model == CLASSICAL) return Real(2.0 / 3.0 * ALPHA) / Q.eta * chi_g * chi_g;
        const RateValues r = Q.rates(chi_g);
        return k * (r.Wu + r.Gu * s2);
    };
    const Real g_half = gamma0 - Real(0.5) * h * loss_rate(gamma0);
    Real g1 = gamma0 - h * loss_rate(g_half > Real(1) ? g_half : Real(1));
    if (g1 < Real(1)) g1 = Real(1);
    p = Kokkos::sqrt(g1 * g1 - Real(1)) * b.ev;
}

// QED during one sub-step of length h for a particle with momentum p, spin S (has_spin) and optical
// depth tau (inversion sampler) in the fields E, B. record(photon) is called for every emitted photon.
template <class Record>
KOKKOS_INLINE_FUNCTION void substep(const Emission& Q, bool has_spin, const V3& E, const V3& B, Real h, V3& p, V3& S,
                                    Real& tau, ParticleRandom& random, Record record) {
    if (!(h > Real(0))) return;
    if (Q.model != STOCHASTIC) {
        if (Q.recoil) radiate_continuously(Q, has_spin, E, B, h, p, S);
        return;
    }

    if (Q.sampler == INVERSION) {
        Real t_done = 0;
        for (int n = 0; n < 1000; ++n) {   // (bound only against pathological input)
            Basis b;
            const Real chi_e = chi_basis(p, E, B, Q.charge, Q.eta, b);
            if (!(chi_e > Real(0))) return;
            const Real gamma = Kokkos::sqrt(Real(1) + push::dot(p, p));
            const Real C0 = rate_prefactor(gamma, Q.eta);
            const RateValues r = Q.rates(chi_e);
            const Real s2 = has_spin ? push::dot(S, b.e2) : Real(0);
            const Real total_rate = C0 * (r.W + r.G * s2);
            // advance the optical depth over the rest of the sub-step
            const Real left = h - t_done;
            const Real t = advance_optical_depth(tau, total_rate, left);
            // spin change during the time t without emission (continuous time; keeps a pure state pure)
            if (has_spin && Q.spin_change && t > Real(0)) S = spin_without_emission_continuous(S, b, C0 * t, r);
            if (!emits(t, left)) return;
            // emission: photon energy from the inverted cumulative spectrum, then a new optical depth
            t_done += t;
            const Real s2_now = has_spin ? push::dot(S, b.e2) : Real(0);
            const Real u_ph = Q.spectrum.sample_u(chi_e, s2_now, random(), Q.bessel);
            record(emit(Q, has_spin, u_ph, chi_e, b, t_done, p, S, random));
            tau = new_optical_depth(random());   // (a lepton stopped by the recoil has chi = 0: loop ends)
        }
    } else {   // REJECTION: one trial in the sub-step
        Basis b;
        const Real chi_e = chi_basis(p, E, B, Q.charge, Q.eta, b);
        if (!(chi_e > Real(0))) return;
        const Real gamma = Kokkos::sqrt(Real(1) + push::dot(p, p));
        const Real C0 = rate_prefactor(gamma, Q.eta);
        const Real s2 = has_spin ? push::dot(S, b.e2) : Real(0);
        const Real r1 = random(), r2 = random();
        const Real u_ph = rejection_trial(chi_e, s2, C0 * h, r1, r2, Q.bessel);
        if (u_ph > Real(0)) {
            record(emit(Q, has_spin, u_ph, chi_e, b, Real(0.5) * h, p, S, random));
        } else if (has_spin && Q.spin_change) {   // no emission in this trial: eq. 8 (a pure state is measured)
            const V3 m = spin_without_emission(S, b, C0 * h, Q.rates(chi_e));
            S = Q.spin_model == PURE ? collapse(m, random()) : m;
        }
    }
}

} // namespace qed
} // namespace quarz
