// Unit test of the look-up tables: src/LogTable.hpp (generic) and the QED tables of src/QEDRates.*.
//  1. LogTable: exact for cubics in t = ln x (Linear) and for exp(cubic in t) (Log, incl. negative
//     columns); fourth-order convergence; power-law extrapolation; parameter / sign errors
//  2. Bessel table (log y) vs the direct functions: accuracy for 16, 32, 64 points per decade;
//     fall-back outside the range; coefficients(u, chi, table) vs coefficients(u, chi)
//  3. Rate table (log chi) vs the quadratures integrals(); Rmax an upper bound within 2 %; max_rate_r vs a scan
//  4. device kernel = host; timing of table vs direct evaluation (informational)
#include "QEDRates.hpp"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <random>

using namespace quarz;
using namespace quarz::qed;

static int failures = 0;
static void check(bool ok, const char* what) {
    std::printf("  [%s] %s\n", ok ? "PASS" : "FAIL", what);
    if (!ok) ++failures;
}
static double rel(double a, double b) { return std::abs(a - b) / std::abs(b); }

// host evaluation of a device table (the views live in the default memory space)
static double host_eval(const LogTable& T, int c, double x) {
    double v = 0;
    Kokkos::parallel_reduce("tab.eval", Kokkos::RangePolicy<>(0, 1), KOKKOS_LAMBDA(int, double& s) { s = T.eval(c, x); }, v);
    return v;
}

static void generic() {
    std::printf("1. LogTable\n");
    // column 0: Linear, cubic in t; column 1: Log, exp(cubic); column 2: Log, negative power law
    auto f = [](double x, double* o) {
        const double t = std::log(x);
        o[0] = t * t * t - 2 * t + 1;
        o[1] = std::exp(0.1 * t * t * t - 0.5 * t);
        o[2] = -3 * std::pow(x, -1.7);
    };
    const LogTable T = LogTable::build(1e-3, 1e2, 8, 3, f, {LogTable::Linear, LogTable::Log, LogTable::Log}, "test");
    double e = 0;
    for (int i = 0; i <= 1000; ++i) {
        const double x = std::pow(10.0, -3 + 5.0 * i / 1000);
        double r[3];
        f(x, r);
        for (int c = 0; c < 3; ++c) e = std::fmax(e, rel(host_eval(T, c, x), r[c]));
    }
    char m[200];
    std::snprintf(m, sizeof m, "exact for cubics in ln x (Linear) and exp(cubic) (Log, negative column): max rel. %.1e", e);
    check(e < 1e-12, m);
    // power law: extrapolation outside the range is exact for the Log column
    const double ex = std::fmax(rel(host_eval(T, 2, 1e-5), -3 * std::pow(1e-5, -1.7)), rel(host_eval(T, 2, 1e4), -3 * std::pow(1e4, -1.7)));
    std::snprintf(m, sizeof m, "power law extrapolated two decades beyond both ends: rel. %.1e", ex);
    check(ex < 1e-12, m);
    // convergence for f = exp(sin(2t)): error ratio 16 per doubling of the density
    auto g = [](double x, double* o) { o[0] = std::exp(std::sin(2 * std::log(x))); };
    double err[3];
    int k = 0;
    for (double ppd : {10.0, 20.0, 40.0}) {
        const LogTable G = LogTable::build(1e-2, 1e2, ppd, 1, g, {LogTable::Log});
        double em = 0;
        for (int i = 0; i <= 2000; ++i) {
            const double x = std::pow(10.0, -2 + 4.0 * i / 2000);
            double r;
            g(x, &r);
            em = std::fmax(em, rel(host_eval(G, 0, x), r));
        }
        err[k++] = em;
    }
    const double o1 = std::log2(err[0] / err[1]), o2 = std::log2(err[1] / err[2]);
    std::snprintf(m, sizeof m, "exp(sin(2 ln x)), 10/20/40 per decade: errors %.1e %.1e %.1e, order %.2f %.2f (4)", err[0], err[1], err[2], o1, o2);
    check(o1 > 3.7 && o2 > 3.7, m);
    bool thrown = false;
    try { LogTable::build(1.0, 10.0, 10, 1, [](double x, double* o) { o[0] = x - 2; }, {LogTable::Log}); }
    catch (const std::exception&) { thrown = true; }
    bool thrown2 = false;
    try { LogTable::build(0.0, 10.0, 10, 1, [](double x, double* o) { o[0] = x; }, {LogTable::Linear}); }
    catch (const std::exception&) { thrown2 = true; }
    check(thrown && thrown2, "errors: Log column changing sign; x_min <= 0");
    check(T.size() == 41 && T.ncomp() == 3, "5 decades at 8 per decade: 41 points, 3 columns");
}

static void bessel_table() {
    std::printf("2. Bessel table, y = 1e-6 ... 700 (ln(e^y K) vs ln y)\n");
    std::mt19937_64 rng(1);
    std::uniform_real_distribution<double> U(0, 1);
    char m[200];
    double e32 = 0;
    for (double ppd : {16.0, 32.0, 64.0}) {
        const BesselTable T = make_bessel_table(1e-6, Y_MAX, ppd);
        // relative error over 1e5 log-uniform y in the range, all three functions (device kernel)
        double em = 0;
        const int N = 100000;
        Kokkos::parallel_reduce("btab.err", Kokkos::RangePolicy<>(0, N), KOKKOS_LAMBDA(int i, double& d) {
            const double y = Kokkos::pow(10.0, -6.0 + 8.845 * (i + 0.5) / N);
            double a, b, c, A, B, C;
            T(y, a, b, c);
            bessel::synchrotron_functions(y, A, B, C);
            const double e = Kokkos::fmax(Kokkos::fmax(Kokkos::fabs(a / A - 1), Kokkos::fabs(b / B - 1)), Kokkos::fabs(c / C - 1));
            if (e > d) d = e;
        }, Kokkos::Max<double>(em));
        if (ppd == 32) e32 = em;
        std::printf("    %2.0f points per decade (%3d points): max rel. error %.1e\n", ppd, T.t.size(), em);
    }
    std::snprintf(m, sizeof m, "default 32 points per decade: max rel. error %.1e (< 1e-6)", e32);
    check(e32 < 1e-6, m);
    const BesselTable T = make_bessel_table();
    // outside the range: the direct functions, bit-identical
    bool same = true;
    for (double y : {1e-9, 5e-7, 701.0, 1e4}) {
        double a, b, c, A, B, C;
        Kokkos::parallel_reduce("btab.out", Kokkos::RangePolicy<>(0, 1), KOKKOS_LAMBDA(int, double& s) {
            double p, q, r; T(y, p, q, r); s = p; }, a);
        Kokkos::parallel_reduce("btab.out2", Kokkos::RangePolicy<>(0, 1), KOKKOS_LAMBDA(int, double& s) {
            double p, q, r; T(y, p, q, r); s = q + 0 * p + 0 * r; }, b);
        Kokkos::parallel_reduce("btab.out3", Kokkos::RangePolicy<>(0, 1), KOKKOS_LAMBDA(int, double& s) {
            double p, q, r; T(y, p, q, r); s = r + 0 * p + 0 * q; }, c);
        bessel::synchrotron_functions(y, A, B, C);
        same = same && a == A && b == B && c == C;
    }
    check(same, "outside [1e-6, 700]: the direct functions (bit-identical)");
    // coefficients through the table
    double ec = 0;
    for (int i = 0; i < 20000; ++i) {
        const double r = U(rng), u = r * r * r, chi = std::pow(10.0, -3 + 6 * U(rng));
        const Coef a = coefficients(u, chi, T), b = coefficients(u, chi);
        if (!(b.w > 0)) continue;
        // relative to the scale of the spin-dependent terms (w), as they enter the rates
        const double d[9] = {a.w - b.w, a.g - b.g, a.f2 - b.f2, a.fs - b.fs, a.fv - b.fv,
                             a.x1 - b.x1, a.x2 - b.x2, a.x3 - b.x3, a.x3s - b.x3s};
        for (double v : d) ec = std::fmax(ec, std::abs(v) / b.w);
    }
    std::snprintf(m, sizeof m, "coefficients(u, chi, table) vs direct, 2e4 random (u, chi): max |diff|/w %.1e (< 1e-6)", ec);
    check(ec < 1e-6, m);
}

static void rate_table() {
    std::printf("3. Rate table, chi = 1e-5 ... 1e4\n");
    const auto t0 = std::chrono::steady_clock::now();
    const RateTable T = make_rate_table();
    const double tb = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::printf("    built in %.2f s (%d points)\n", tb, T.t.size());
    std::mt19937_64 rng(2);
    std::uniform_real_distribution<double> U(0, 1);
    double e[6] = {0};
    for (int i = 0; i < 300; ++i) {
        const double chi = std::pow(10.0, -5 + 9 * U(rng));
        double W = 0;
        Kokkos::parallel_reduce("rtab", Kokkos::RangePolicy<>(0, 1), KOKKOS_LAMBDA(int, double& s) { s = T(chi).W; }, W);
        double vv[6];
        for (int c = 0; c < 6; ++c) vv[c] = host_eval(T.t, c, chi);
        const Integrals I = integrals(chi);
        const double r[6] = {I.W, I.G, I.Xi, I.K, I.Wu, I.Gu};
        for (int c = 0; c < 6; ++c) e[c] = std::fmax(e[c], rel(vv[c], r[c]));
        e[0] = std::fmax(e[0], rel(W, I.W));
    }
    const char* names[6] = {"W", "G", "Xi", "K", "Wu", "Gu"};
    for (int c = 0; c < 6; ++c) {
        char m[160];
        std::snprintf(m, sizeof m, "%-4s 300 random chi: max rel. error %.1e (< 1e-6)", names[c], e[c]);
        check(e[c] < 1e-6, m);
    }
    // Rmax: an upper bound within 2 %, on a dense grid through the kink at chi ~ 25
    double lo = 1e300, hi = 0;
    for (int i = 0; i <= 2000; ++i) {
        const double chi = std::pow(10.0, -5 + 9.0 * i / 2000);
        double R = 0;
        Kokkos::parallel_reduce("rtab.r", Kokkos::RangePolicy<>(0, 1), KOKKOS_LAMBDA(int, double& s) { s = T(chi).Rmax; }, R);
        const double q = R / max_rate_r(chi);
        lo = std::fmin(lo, q);
        hi = std::fmax(hi, q);
    }
    {
        char m[160];
        std::snprintf(m, sizeof m, "Rmax / max_r 3r^2(w + |g|) in [%.4f, %.4f] for 2001 chi (bound: in [1, 1.02])", lo, hi);
        check(lo >= 1 && hi <= 1.02, m);
    }
    // Rmax vs brute force (1e5 points) and the bound P <= C0 dt Rmax for every r and spin
    double eb = 0;
    bool bound = true;
    for (double chi : {1e-4, 0.01, 0.3, 1.0, 5.0, 100.0}) {
        double mx = 0;
        const double M = max_rate_r(chi) * (1 + 1e-12);
        for (int k = 0; k <= 100000; ++k) {
            const double r = k / 100000.0, vm = rate_r(r, chi, -1.0);
            mx = std::fmax(mx, vm);
            bound = bound && rate_r(r, chi, 1.0) <= M && vm <= M;
        }
        eb = std::fmax(eb, (M / (1 + 1e-12) - mx) / mx);
    }
    char m[200];
    std::snprintf(m, sizeof m, "max_rate_r vs 1e5-point scan: %.1e relative (>= 0, < 1e-6); bound holds for s2 = +-1", eb);
    check(eb >= -1e-14 && eb < 1e-6 && bound, m);
    check(T(0.0).W == 0 && T(-1.0).Rmax == 0, "chi <= 0: all zero");
}

static void timing() {
    std::printf("4. Timing (%s, informational): coefficients for 1e7 random (u, chi)\n", Kokkos::DefaultExecutionSpace::name());
    const BesselTable T = make_bessel_table();
    const int N = 10000000;
    for (int mode = 0; mode < 2; ++mode) {
        double s = 0;
        const auto t0 = std::chrono::steady_clock::now();
        Kokkos::parallel_reduce("time", Kokkos::RangePolicy<>(0, N), KOKKOS_LAMBDA(int i, double& a) {
            const double r = (i % 9973 + 0.5) / 9973, u = r * r * r;
            const double chi = Kokkos::pow(10.0, -2.0 + 4.0 * (i % 101) / 100.0);
            const Coef c = mode ? coefficients(u, chi, T) : coefficients(u, chi);
            a += c.w + c.g + c.x2;
        }, s);
        Kokkos::fence();
        const double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        std::printf("    %-6s %.1f ns per evaluation (checksum %.6e)\n", mode ? "table" : "direct", dt / N * 1e9, s);
    }
}

int main(int argc, char** argv) {
    Kokkos::initialize(argc, argv);
    generic();
    bessel_table();
    rate_table();
    timing();
    Kokkos::finalize();
    std::printf("%s (%d failures)\n", failures ? "FAILED" : "all tests passed", failures);
    return failures ? 1 : 0;
}
