// Unit test of the look-up tables: src/LogTable.hpp (generic) and the QED tables of src/QEDRates.*.
//  1. LogTable: exact for cubics in t = ln x (Linear) and for exp(cubic in t) (Log, incl. negative
//     columns); fourth-order convergence; power-law extrapolation; parameter / sign errors
//  2. Bessel table (log y) vs the direct functions: accuracy for 16, 32, 64 points per decade;
//     fall-back outside the range; coefficients(u, chi, table) vs coefficients(u, chi)
//  3. Rate table (log chi) vs the quadratures integrals(); Rmax an upper bound within 2 %; max_rate_r vs a scan
//  4. device kernel = host; timing of table vs direct evaluation (informational)
//  5. files: save/load round trip (bit-identical), header, corrupted files, QED meta checks, load-or-build
#include "QEDRates.hpp"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <map>
#include <random>
#include <string>

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
        // device kernel vs host: the same functions up to round-off (y = 1e4: both 0)
        auto close = [](double p, double q) { return p == q || std::abs(p - q) <= 1e-12 * std::abs(q); };
        same = same && close(a, A) && close(b, B) && close(c, C);
        double hA, hB, hC;
        T.host(y, hA, hB, hC);
        same = same && hA == A && hB == B && hC == C;   // host path of the table: the direct functions
    }
    check(same, "outside [1e-6, 700]: the direct functions (device 1e-12, host path exactly)");
    {   // host path inside the range = device path
        double eh = 0;
        for (double y : {1e-5, 0.02, 0.7, 3.3, 40.0, 500.0}) {
            double a = 0, hA, hB, hC;
            Kokkos::parallel_reduce("btab.in", Kokkos::RangePolicy<>(0, 1), KOKKOS_LAMBDA(int, double& s) {
                double p, q, r; T(y, p, q, r); s = p + q + r; }, a);
            T.host(y, hA, hB, hC);
            eh = std::fmax(eh, std::abs(a / (hA + hB + hC) - 1));
        }
        std::snprintf(m, sizeof m, "table evaluated on the host (BesselTable::host) vs in a kernel: max rel. %.1e (< 1e-12)", eh);
        check(eh < 1e-12, m);
    }
    // coefficients through the table
    double ec = 0;
    const int NC = 20000;
    Kokkos::View<double*[2]> uc("uc", NC);
    {
        auto h = Kokkos::create_mirror_view(uc);
        for (int i = 0; i < NC; ++i) {
            const double r = U(rng);
            h(i, 0) = r * r * r;
            h(i, 1) = std::pow(10.0, -3 + 6 * U(rng));
        }
        Kokkos::deep_copy(uc, h);
    }
    Kokkos::parallel_reduce("btab.coef", Kokkos::RangePolicy<>(0, NC), KOKKOS_LAMBDA(int i, double& dm) {
        const Coef a = coefficients(uc(i, 0), uc(i, 1), T), b = coefficients(uc(i, 0), uc(i, 1));
        if (!(b.w > 0)) return;
        // relative to the scale of the spin-dependent terms (w), as they enter the rates
        const double d[9] = {a.w - b.w, a.g - b.g, a.f2 - b.f2, a.fs - b.fs, a.fv - b.fv,
                             a.x1 - b.x1, a.x2 - b.x2, a.x3 - b.x3, a.x3s - b.x3s};
        for (int k = 0; k < 9; ++k) {
            const double e = Kokkos::fabs(d[k]) / b.w;
            if (e > dm) dm = e;
        }
    }, Kokkos::Max<double>(ec));
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
    double e[9] = {0}, ehost = 0;
    for (int i = 0; i < 300; ++i) {
        const double chi = std::pow(10.0, -5 + 9 * U(rng));
        double W = 0;
        Kokkos::parallel_reduce("rtab", Kokkos::RangePolicy<>(0, 1), KOKKOS_LAMBDA(int, double& s) { s = T(chi).W; }, W);
        ehost = std::fmax(ehost, std::abs(T.host(chi).W / W - 1));
        const int col[9] = {0, 1, 2, 3, 4, 5, 7, 8, 9};   // (column 6 is Rmax)
        const Integrals I = integrals(chi);
        const double r[9] = {I.W, I.G, I.Xi, I.K, I.Wu, I.Gu, I.SB, I.SA, I.SL};
        for (int c = 0; c < 9; ++c) e[c] = std::fmax(e[c], rel(host_eval(T.t, col[c], chi), r[c]));
        e[0] = std::fmax(e[0], rel(W, I.W));
    }
    const char* names[9] = {"W", "G", "Xi", "K", "Wu", "Gu", "SB", "SA", "SL"};
    for (int c = 0; c < 9; ++c) {
        char m[160];
        std::snprintf(m, sizeof m, "%-4s 300 random chi: max rel. error %.1e (< 1e-6)", names[c], e[c]);
        check(e[c] < 1e-6, m);
    }
    {
        char m[160];
        std::snprintf(m, sizeof m, "RateTable::host() vs the kernel evaluation: max rel. %.1e (< 1e-12)", ehost);
        check(ehost < 1e-12, m);
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
    check(T(0.0).W == 0 && T(-1.0).Rmax == 0 && T.host(0.0).W == 0, "chi <= 0: all zero");
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

// device evaluation of all columns at many x: tables with identical data give the same results
// (to 1e-13: the two evaluations sit at different places of a kernel and need not round identically)
static bool same_eval(const LogTable& A, const LogTable& B, double lo, double hi) {
    const int N = 20000, nc = A.ncomp();
    int bad = 0;
    Kokkos::parallel_reduce("tab.same", Kokkos::RangePolicy<>(0, N), KOKKOS_LAMBDA(int i, int& b) {
        const double x = lo * Kokkos::pow(hi / lo, (i + 0.37) / N);
        for (int c = 0; c < nc; ++c) {
            const double p = A.eval(c, x), q = B.eval(c, x);
            if (!(Kokkos::fabs(p - q) <= 1e-13 * Kokkos::fabs(q))) ++b;
        }
    }, bad);
    return bad == 0;
}

static std::string slurp(const std::string& path) {
    std::string s;
    if (FILE* fp = std::fopen(path.c_str(), "r")) {
        char b[4096];
        size_t n;
        while ((n = std::fread(b, 1, sizeof b, fp)) > 0) s.append(b, n);
        std::fclose(fp);
    }
    return s;
}
static void spit(const std::string& path, const std::string& s) {
    FILE* fp = std::fopen(path.c_str(), "w");
    std::fputs(s.c_str(), fp);
    std::fclose(fp);
}
template <class F>
static bool throws(F f) {
    try { f(); } catch (const std::exception&) { return true; }
    return false;
}

static void files() {
    std::printf("5. Table files (plain text *.lut)\n");
    const std::string f1 = "test_qed_tables_generic.lut", f2 = "test_qed_tables_rate.lut", f3 = "test_qed_tables_cache.lut";
    auto f = [](double x, double* o) { o[0] = std::sin(x); o[1] = std::exp(-x) * x; o[2] = -1 / (1 + x * x); };
    const LogTable T = LogTable::build(1e-3, 30, 20, 3, f, {LogTable::Linear, LogTable::Log, LogTable::Log}, "gen");
    T.save(f1, {"sin", "x_exp(-x)", "-1/(1+x^2)"}, {{"origin", "unit test"}, {"answer", "42"}});
    std::vector<std::string> lab;
    std::map<std::string, std::string> meta;
    const LogTable L = LogTable::load(f1, &lab, &meta);
    bool vals = L.size() == T.size() && L.ncomp() == 3 && L.axis_min() == T.axis_min() && L.axis_max() == T.axis_max();
    for (int i = 0; vals && i < T.size(); ++i)
        for (int c = 0; c < 3; ++c) vals = vals && L.value(i, c) == T.value(i, c) && L.mode(c) == T.mode(c);
    check(vals && same_eval(T, L, 1e-4, 100), "save/load: stored values, modes, axis bit-identical; evaluation (incl. extrapolation) equal to 1e-13");
    check(lab.size() == 3 && lab[1] == "x_exp(-x)" && meta["origin"] == "unit test" && meta["answer"] == "42" && L.name() == "gen",
          "labels, meta entries (values with spaces) and name restored");
    const std::string txt = slurp(f1);
    check(txt.rfind("# QUARZ LogTable 1\n# name gen\n", 0) == 0 && txt.find("# modes lin log log") != std::string::npos &&
          txt.find("# x sin x_exp(-x) -1/(1+x^2)") != std::string::npos, "human-readable header (magic, modes, column labels)");
    // corrupted files
    spit(f3, "# not a table\n");
    const bool e1 = throws([&] { LogTable::load(f3); });
    spit(f3, txt.substr(0, txt.rfind('\n', txt.size() - 2) + 1));   // last data line missing
    const bool e2 = throws([&] { LogTable::load(f3); });
    std::string bad = txt;
    const size_t p = bad.find("\n0.001") + 1;   // first data line: x moved off the grid
    bad.replace(p, 5, "0.002");
    spit(f3, bad);
    const bool e3 = throws([&] { LogTable::load(f3); });
    const bool e4 = throws([&] { LogTable::load("does_not_exist.lut"); });
    const bool e5 = throws([&] { T.save(f3, {"two words", "b", "c"}); });
    check(e1 && e2 && e3 && e4 && e5, "errors: wrong magic, missing data line, x off the grid, no file, label with a space");
    // QED tables
    const auto t0 = std::chrono::steady_clock::now();
    const RateTable R = make_rate_table();
    const auto t1 = std::chrono::steady_clock::now();
    save_rate_table(R, f2);
    const auto t2 = std::chrono::steady_clock::now();
    const RateTable RL = load_rate_table(f2);
    const auto t3 = std::chrono::steady_clock::now();
    auto ms = [](auto a, auto b) { return std::chrono::duration<double, std::milli>(b - a).count(); };
    char m[200];
    std::snprintf(m, sizeof m, "rate table: loaded = built (build %.1f ms, save %.2f ms, load %.2f ms, %zu bytes)",
                  ms(t0, t1), ms(t1, t2), ms(t2, t3), slurp(f2).size());
    check(same_eval(R.t, RL.t, 1e-6, 1e5), m);
    check(throws([&] { load_bessel_table(f2); }) && throws([&] { load_rate_table(f1); }),
          "load_bessel_table on a rate table, load_rate_table on a generic table: rejected");
    // load or build
    std::remove(f3.c_str());
    std::string st1, st2, st3, st4, st5;
    const BesselTable B1 = bessel_table(f3, 1e-6, Y_MAX, 32, true, &st1);
    const BesselTable B2 = bessel_table(f3, 1e-6, Y_MAX, 32, true, &st2);
    check(st1 == "built (saved)" && st2 == "loaded" && same_eval(B1.t, B2.t, 1e-6, Y_MAX) && B2.y_min == 1e-6 && B2.y_max == Y_MAX,
          ("bessel_table(): first call " + st1 + ", second " + st2 + ", identical").c_str());
    bessel_table(f3, 1e-6, Y_MAX, 16, false, &st3);
    std::string v = slurp(f3);
    const std::string ver = "# meta version " + std::to_string(TABLE_VERSION);
    v.replace(v.find(ver), ver.size(), "# meta version 0");
    spit(f3, v);
    bessel_table(f3, 1e-6, Y_MAX, 32, false, &st4);
    rate_table(f3, 1e-5, 1e4, 32, false, &st5);
    std::printf("    other parameters: %s\n    old version: %s\n    other kind: %s\n", st3.c_str(), st4.c_str(), st5.c_str());
    check(st3.find("different range or points") != std::string::npos && st4.find("table version 0, expected") != std::string::npos &&
          st5.find("not a qed.rate") != std::string::npos, "other parameters, old version, other kind: rebuilt (write = false: file kept)");
    std::string st6;
    std::remove("test_qed_tables_none.lut");
    rate_table("test_qed_tables_none.lut", 1e-5, 1e4, 32, false, &st6);
    check(st6 == "built (no file)" && slurp("test_qed_tables_none.lut").empty(), "write = false: no file written");
    for (const auto& fn : {f1, f2, f3}) std::remove(fn.c_str());
}

int main(int argc, char** argv) {
    Kokkos::initialize(argc, argv);
    generic();
    bessel_table();
    rate_table();
    timing();
    files();
    Kokkos::finalize();
    std::printf("%s (%d failures)\n", failures ? "FAILED" : "all tests passed", failures);
    return failures ? 1 : 0;
}
