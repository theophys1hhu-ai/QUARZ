// Unit test of the photon emission samplers (src/QEDSampling.*).
//  1. spectrum table: cumulative Wc, Gc at random chi vs quadrature; Wc(1), Gc(1) = W, G
//  2. inversion: the sampled u solves F(u) = xi F(1): |F(u)/F(1) - xi| for chi = 1e-4 ... 300 (off the
//     grid), s2 = -1, 0, 1 and xi from 1e-9 to 1 - 1e-9 (F by independent quadrature)
//  3. spectra: histograms of 4e6 rejection trials and 1e6 inversion samples vs C0 (w + g s2): chi^2
//  4. emission statistics over a time T with many steps: mean number of photons = C0 T (W + G s2) for
//     both samplers (optical depth: exponential waiting times), spin dependence, sub-step bound P <= p_max
//  5. files: save/load bit-identical, load-or-build
// Random numbers: the counter-based ion_rand of Ionization.hpp; samples drawn in Kokkos kernels.
#include "Ionization.hpp"
#include "QEDSampling.hpp"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <vector>

using namespace quarz;
using namespace quarz::qed;

static int failures = 0;
static void check(bool ok, const char* what) {
    std::printf("  [%s] %s\n", ok ? "PASS" : "FAIL", what);
    if (!ok) ++failures;
}

// exact cumulative spectrum F(u) = int_0^u (w + g s2) du' by quadrature in r = u^{1/3}
static double F_exact(double u, double chi, double s2) {
    return integrate([&](double r) { const Coef c = coefficients(r * r * r, chi); return 3 * r * r * (c.w + c.g * s2); },
                     0.0, std::cbrt(u), 1e-12);
}

// sample n photon energies with the inversion sampler in a kernel (xi from ion_rand or given)
static std::vector<double> sample_inversion(const SpectrumTable& S, const BesselTable& B, double chi, double s2,
                                            const std::vector<double>& xi) {
    const int n = static_cast<int>(xi.size());
    Kokkos::View<double*> d_xi("xi", n), d_u("u", n);
    Kokkos::deep_copy(d_xi, Kokkos::View<const double*, Kokkos::HostSpace>(xi.data(), n));
    Kokkos::parallel_for("inv", Kokkos::RangePolicy<>(0, n), KOKKOS_LAMBDA(int i) { d_u(i) = S.sample_u(chi, s2, d_xi(i), B); });
    std::vector<double> u(n);
    Kokkos::deep_copy(Kokkos::View<double*, Kokkos::HostSpace>(u.data(), n), d_u);
    return u;
}

static void spectrum_table_values(const SpectrumTable& S) {
    std::printf("1. Spectrum table (%d chi x %d u nodes)\n", S.t.size(), S.nu);
    double e = 0, eW = 0;
    for (double chi : {3.3e-5, 0.0123, 0.77, 4.6, 523.0}) {
        for (int j = 0; j < S.nu; j += 37) {   // (eval_host: the table itself lives in device memory)
            const double u = SpectrumTable::u_of_L(S.L_node(j));
            const double Wc = S.t.eval_host(j, chi), Gc = S.t.eval_host(S.nu + j, chi);
            const double scale = F_exact(1.0, chi, 0.0);   // errors relative to the total W
            e = std::fmax(e, std::fmax(std::abs(Wc - F_exact(u, chi, 0.0)), std::abs(Gc - (F_exact(u, chi, 1.0) - F_exact(u, chi, 0.0)))) / scale);
        }
        const Integrals I = integrals(chi);
        eW = std::fmax(eW, std::fmax(std::abs(S.t.eval_host(S.nu - 1, chi) / I.W - 1), std::abs(S.t.eval_host(2 * S.nu - 1, chi) / I.G - 1)));
    }
    char m[200];
    std::snprintf(m, sizeof m, "Wc(u_j), Gc(u_j) at 5 chi off the grid vs quadrature: max error %.1e of W (< 1e-6)", e);
    check(e < 1e-6, m);
    std::snprintf(m, sizeof m, "Wc, Gc at the last node = W, G (integrals()): max rel. %.1e (< 1e-6)", eW);
    check(eW < 1e-6, m);
}

static void inversion_accuracy(const SpectrumTable& S, const BesselTable& B) {
    std::printf("2. Inversion: F(u)/F(1) = xi\n");
    std::vector<double> xi;
    for (double x : {1e-9, 1e-6, 1e-4, 1e-3}) xi.push_back(x);
    for (int k = 1; k < 100; ++k) xi.push_back(k / 100.0);
    for (double x : {0.999, 1 - 1e-6, 1 - 1e-9}) xi.push_back(x);
    double e = 0, worst_chi = 0, worst_xi = 0;
    for (double chi : {1.3e-4, 0.0317, 1.0, 7.3, 300.0})
        for (double s2 : {-1.0, 0.0, 1.0}) {
            const std::vector<double> u = sample_inversion(S, B, chi, s2, xi);
            const double Ftot = F_exact(1.0, chi, s2);
            for (size_t k = 0; k < xi.size(); ++k) {
                const double d = std::abs(F_exact(u[k], chi, s2) / Ftot - xi[k]);
                if (d > e) { e = d; worst_chi = chi; worst_xi = xi[k]; }
            }
        }
    char m[200];
    // dominated by the cubic interpolation in ln chi (16 points per decade) of the cut-off region of the
    // spectrum at small chi; far below the statistical resolution of any Monte-Carlo sample (~1e12 photons)
    std::snprintf(m, sizeof m, "15 (chi, s2) x 106 xi: max |F(u)/F(1) - xi| = %.1e (chi %g, xi %g) (< 2e-6)", e, worst_chi, worst_xi);
    check(e < 2e-6, m);
    const std::vector<double> u = sample_inversion(S, B, 1.0, 0.0, {0.0, 0.5});
    check(u[0] == 0 && u[1] > 0 && u[1] < 1, "xi = 0 gives u = 0; u in (0, 1)");
}

// chi^2 of a histogram against expected counts
static double chi2(const std::vector<double>& obs, const std::vector<double>& expd) {
    double c = 0;
    for (size_t i = 0; i < obs.size(); ++i) c += (obs[i] - expd[i]) * (obs[i] - expd[i]) / expd[i];
    return c;
}

static void spectra(const SpectrumTable& S, const BesselTable& B, const RateTable& R) {
    std::printf("3. Spectra of the samplers vs C0 (w + g s2)  (chi = 1, s2 = 0.5; 20 bins in r = u^{1/3})\n");
    const double chi = 1.0, s2 = 0.5;
    const int nb = 20;
    // expected fraction per bin of r
    std::vector<double> frac(nb);
    const double Ftot = F_exact(1.0, chi, s2);
    for (int b = 0; b < nb; ++b)
        frac[b] = (F_exact(std::pow((b + 1.0) / nb, 3), chi, s2) - F_exact(std::pow(double(b) / nb, 3), chi, s2)) / Ftot;
    // rejection: N trials with C0 dt such that P <= 0.9
    const int N = 4000000;
    const double C0dt = 0.9 / R.host(chi).Rmax;
    Kokkos::View<double*> d_u("u", N);
    Kokkos::parallel_for("rej", Kokkos::RangePolicy<>(0, N), KOKKOS_LAMBDA(int i) {
        d_u(i) = rejection_trial(chi, s2, C0dt, ion_rand(7, i, 0), ion_rand(7, i, 1), B);
    });
    auto hu = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), d_u);
    std::vector<double> h(nb, 0.0);
    double nacc = 0;
    for (int i = 0; i < N; ++i)
        if (hu(i) > 0) { ++nacc; h[std::min(nb - 1, static_cast<int>(std::cbrt(hu(i)) * nb))] += 1; }
    const double mean = N * C0dt * Ftot;   // expected number of photons
    std::vector<double> e(nb);
    for (int b = 0; b < nb; ++b) e[b] = mean * frac[b];
    const double c1 = chi2(h, e);
    const double sig = (nacc - mean) / std::sqrt(mean * (1 - C0dt * Ftot));
    char m[200];
    std::snprintf(m, sizeof m, "rejection, 4e6 trials: %.0f photons, expected %.0f (%+.2f sigma); spectrum chi^2 = %.1f (20 bins)", nacc, mean, sig, c1);
    check(std::abs(sig) < 4 && c1 < 45, m);   // chi^2 > 45 for 20 dof: p < 0.001
    // inversion: 1e6 samples
    const int M = 1000000;
    std::vector<double> xi(M);
    for (int i = 0; i < M; ++i) xi[i] = ion_rand(9, i, 0);
    const std::vector<double> u = sample_inversion(S, B, chi, s2, xi);
    std::vector<double> h2(nb, 0.0), e2(nb);
    for (double v : u) h2[std::min(nb - 1, static_cast<int>(std::cbrt(v) * nb))] += 1;
    for (int b = 0; b < nb; ++b) e2[b] = M * frac[b];
    const double c2 = chi2(h2, e2);
    std::snprintf(m, sizeof m, "inversion, 1e6 samples: spectrum chi^2 = %.1f (20 bins)", c2);
    check(c2 < 45, m);
}

// Emission over a time T: n_particles particles at fixed chi and spin (no recoil), steps dt.
// Returns the total number of photons for the rejection (mode 0) or the inversion (mode 1) sampler.
static double count_photons(int mode, double chi, double s2, double C0, double T, double dt, int n_particles,
                            const RateTable& R, const BesselTable& B, double p_max) {
    const int nsteps = static_cast<int>(T / dt + 0.5);
    const RateValues rv = R.host(chi);
    const double rate = C0 * (rv.W + rv.G * s2);   // total emission rate
    double total = 0;
    Kokkos::parallel_reduce("count", Kokkos::RangePolicy<>(0, n_particles), KOKKOS_LAMBDA(int p, double& count) {
        if (mode == 0) {
            // rejection: sub-steps with P <= p_max, one trial each
            const int nsub = rejection_substeps(C0 * dt, rv.Rmax, p_max);
            const double C0dt_sub = C0 * dt / nsub;
            for (int n = 0; n < nsteps; ++n)
                for (int k = 0; k < nsub; ++k) {
                    const uint64_t draw = 2 * (static_cast<uint64_t>(n) * nsub + k);
                    if (rejection_trial(chi, s2, C0dt_sub, ion_rand(11, p, draw), ion_rand(11, p, draw + 1), B) > 0) count += 1;
                }
        } else {
            // inversion: optical depth, possibly several photons per step
            uint64_t draw = 0;
            double tau = new_optical_depth(ion_rand(13, p, draw++));
            for (int n = 0; n < nsteps; ++n) {
                double left = dt;
                for (;;) {
                    const double t = advance_optical_depth(tau, rate, left);
                    if (!emits(t, left)) break;
                    count += 1;   // (the photon energy would be drawn here with sample_u)
                    tau = new_optical_depth(ion_rand(13, p, draw++));
                    left -= t;
                }
            }
        }
    }, total);
    return total;
}

static void emission_statistics(const RateTable& R, const BesselTable& B) {
    std::printf("4. Mean number of photons over T = 1000 dt (chi = 0.5; 2000 particles; C0 chosen per case)\n");
    const double chi = 0.5, T = 1000, dt = 1;
    char m[200];
    for (double s2 : {-1.0, 1.0}) {
        const RateValues rv = R.host(chi);
        for (double photons_per_step : {0.01, 0.5, 3.0}) {   // mean number of photons per particle and step
            const double C0 = photons_per_step / (rv.W + rv.G * s2) / dt;
            const double expect = 2000 * C0 * T * (rv.W + rv.G * s2);
            const double nr = count_photons(0, chi, s2, C0, T, dt, 2000, R, B, 0.1);
            const double ni = count_photons(1, chi, s2, C0, T, dt, 2000, R, B, 0.1);
            const double sr = (nr - expect) / std::sqrt(expect), si = (ni - expect) / std::sqrt(expect);
            std::snprintf(m, sizeof m, "s2 = %+.0f, %.2f photons/step: expected %.0f, rejection %.0f (%+.1f sigma), inversion %.0f (%+.1f sigma)",
                          s2, photons_per_step, expect, nr, sr, ni, si);
            check(std::abs(sr) < 4 && std::abs(si) < 4, m);
        }
    }
    const RateValues rv = R.host(chi);
    std::snprintf(m, sizeof m, "spin dependence: rate(s2 = +1) / rate(s2 = -1) = (W + G)/(W - G) = %.4f", (rv.W + rv.G) / (rv.W - rv.G));
    check((rv.W + rv.G) / (rv.W - rv.G) < 1, m);
    // the sub-step bound: C0 dt_sub max_r 3 r^2 (w + |g|) <= p_max
    double worst = 0;
    for (double chi2v : {1e-4, 0.03, 1.0, 30.0, 3000.0})
        for (double C0dt : {1e-3, 0.7, 50.0}) {
            const int n = rejection_substeps(C0dt, R.host(chi2v).Rmax, 0.1);
            worst = std::fmax(worst, C0dt / n * max_rate_r(chi2v) / 0.1);
        }
    std::snprintf(m, sizeof m, "rejection_substeps: max P / p_max = %.3f (<= 1)", worst);
    check(worst <= 1, m);
}

static void files(const SpectrumTable& S, const BesselTable& B) {
    std::printf("5. Spectrum table files\n");
    const std::string f = "test_qed_sampling_spectrum.lut";
    const auto t0 = std::chrono::steady_clock::now();
    save_spectrum_table(S, f);
    const auto t1 = std::chrono::steady_clock::now();
    const SpectrumTable L = load_spectrum_table(f);
    const auto t2 = std::chrono::steady_clock::now();
    std::vector<double> xi;
    for (int i = 0; i < 2000; ++i) xi.push_back(ion_rand(3, i, 0));
    bool same = L.nu == S.nu && L.L0 == S.L0 && L.dL == S.dL;
    double du = 0;   // samples of the built and the loaded table (same data; equal up to round-off of the kernels)
    for (double chi : {2e-5, 0.4, 9000.0}) {
        const std::vector<double> a = sample_inversion(S, B, chi, 0.3, xi), b = sample_inversion(L, B, chi, 0.3, xi);
        for (size_t i = 0; i < a.size(); ++i) du = std::fmax(du, std::abs(a[i] - b[i]) / std::fmax(std::abs(a[i]), 1e-300));
    }
    same = same && du <= 1e-12;
    char m[200];
    std::snprintf(m, sizeof m, "save (%.0f ms) / load (%.0f ms): axis identical, samples equal (1e-12)",
                  std::chrono::duration<double, std::milli>(t1 - t0).count(), std::chrono::duration<double, std::milli>(t2 - t1).count());
    check(same, m);
    std::string s1, s2, s3;
    SpectrumTableParams p;
    spectrum_table(f, p, false, &s1);
    p.u_points_per_decade = 12;
    spectrum_table(f, p, false, &s2);
    bool rejected = false;
    try { load_rate_table(f); } catch (const std::exception&) { rejected = true; }
    std::printf("    same parameters: %s; other u resolution: %s\n", s1.c_str(), s2.c_str());
    check(s1 == "loaded" && s2.find("different") != std::string::npos && rejected, "load-or-build; a spectrum file is not a rate table");
    std::remove(f.c_str());
}

int main(int argc, char** argv) {
    Kokkos::initialize(argc, argv);
    {
        const auto t0 = std::chrono::steady_clock::now();
        const SpectrumTable S = make_spectrum_table();
        const double tb = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        std::printf("spectrum table built in %.2f s\n", tb);
        const BesselTable B = make_bessel_table();
        const RateTable R = make_rate_table();
        spectrum_table_values(S);
        inversion_accuracy(S, B);
        spectra(S, B, R);
        emission_statistics(R, B);
        files(S, B);
    }
    Kokkos::finalize();
    std::printf("%s (%d failures)\n", failures ? "FAILED" : "all tests passed", failures);
    return failures ? 1 : 0;
}
