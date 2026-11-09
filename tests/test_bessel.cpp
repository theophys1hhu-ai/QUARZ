// Unit test of src/Bessel.hpp: K_{1/3}, K_{2/3} and IntK_{1/3} = int_x^inf K_{1/3}.
//  1. 40-digit references (mpmath 1.4, besselk and quad) at 24 points 1e-8 ... 700
//  2. series / rational approximation switch points: continuity
//  3. differential identities on a dense grid (independent of any reference values):
//       K_{1/3}' = -K_{2/3} - K_{1/3}/(3x),  K_{2/3}' = -K_{1/3} - 2 K_{2/3}/(3x),  IntK_{1/3}' = -K_{1/3}
//  4. limits: x -> 0 (Gamma(nu)/2 (2/x)^nu, IntK(0) = pi/sqrt(3)), x -> inf (Hankel expansion)
//  5. synchrotron_functions() == the single functions; float instantiation; device kernel
#include "Bessel.hpp"

#include <cmath>
#include <cstdio>

using namespace quarz::bessel;

static int failures = 0;
static void check(bool ok, const char* what) {
    std::printf("  [%s] %s\n", ok ? "PASS" : "FAIL", what);
    if (!ok) ++failures;
}
static double rel(double a, double b) { return std::abs(a - b) / std::abs(b); }

// x, e^x K_{1/3}(x), e^x K_{2/3}(x), e^x IntK_{1/3}(x)   (mpmath, 40 digits, printed to 20)
static const double REF[][4] = {
    {1e-8, 783.32291407631484038, 231550.91284788991474, 1.8137876325024509612},
    {1e-5, 78.298688299998541272, 2315.5316732623922534, 1.812642761496704208},
    {1e-3, 16.731770343763983363, 107.57135307602333236, 1.7903953904403226009},
    {0.01, 7.5614624751786901242, 23.330216878838688278, 1.7159785448354665931},
    {0.1, 3.2048055519509183322, 5.2528361260054906958, 1.4633236646945201637},
    {0.3, 2.0370893728739777417, 2.6816552772193073888, 1.2340678458473308602},
    {0.7, 1.4026392951746998809, 1.6407737667140328988, 1.0149962903366814093},
    {1, 1.1917780239239114492, 1.3441225759440271958, 0.91749435026393661062},
    {1.5, 0.98667831422392373008, 1.0766830097882951411, 0.8069458885240835863},
    {2, 0.86115725706506641007, 0.92244183854011569203, 0.73048137327237683909},
    {2.4, 0.7893949156999788108, 0.83722692244294820482, 0.68345843569996440492},
    {2.5, 0.7741065787900635385, 0.81933755618096768415, 0.67311545412385249682},
    {2.6, 0.75967810916373489006, 0.80253739922557955672, 0.66324738661888402302},
    {2.9, 0.7208183069662927153, 0.75768327656907216842, 0.63614788748988073352},
    {3, 0.70913805651893852022, 0.74431123803842460882, 0.62785203759588604528},
    {3.1, 0.69801013326042908064, 0.731617896080117973, 0.61988317558298107413},
    {4, 0.61695623299401684673, 0.6404800579323970804, 0.55989058317215388794},
    {6, 0.50619638696759403406, 0.51941217974990554956, 0.47227049618036494943},
    {10, 0.3937144476779695558, 0.40002677679332597122, 0.37662059827904158572},
    {20, 0.27930124965494636538, 0.28158252563076923758, 0.27282680752749497151},
    {50, 0.17700179133972530423, 0.17758697602797954769, 0.17528694329658913507},
    {100, 0.12524483873592936154, 0.12545271894473553554, 0.12462861730294816648},
    {300, 0.072343405180851898538, 0.072383540326102224484, 0.072223485821034815405},
    {700, 0.047366125840912323729, 0.047377396786604622901, 0.047332371810172318688},
};

static void references() {
    std::printf("1. 40-digit references, 24 points x = 1e-8 ... 700\n");
    double ek = 0, eik = 0, es = 0;
    for (const auto& r : REF) {
        const double x = r[0], ex = std::exp(-x);
        if (x <= 300) {   // unscaled (e^-700 K is still representable, but test the scaled ones there)
            ek = std::fmax(ek, std::fmax(rel(k13(x), r[1] * ex), rel(k23(x), r[2] * ex)));
            eik = std::fmax(eik, rel(ik13(x), r[3] * ex));
        }
        es = std::fmax(es, std::fmax(std::fmax(rel(k13e(x), r[1]), rel(k23e(x), r[2])), rel(ik13e(x), r[3])));
    }
    char m[160];
    std::snprintf(m, sizeof m, "K_{1/3}, K_{2/3}: max rel. error %.1e (< 1e-12)", ek);
    check(ek < 1e-12, m);
    std::snprintf(m, sizeof m, "IntK_{1/3}:       max rel. error %.1e (< 5e-12)", eik);
    check(eik < 5e-12, m);
    std::snprintf(m, sizeof m, "scaled e^x K, e^x IntK up to x = 700: max rel. error %.1e (< 1e-11)", es);
    check(es < 1e-11, m);
}

static void switch_points() {
    std::printf("2. Continuity at the series / rational switch points\n");
    double d = 0;
    for (double s : {K_SWITCH, IK_SWITCH}) {
        const double a = std::nextafter(s, 0.0), b = std::nextafter(s, 10.0);
        d = std::fmax(d, std::fmax(std::fmax(rel(k13(a), k13(b)), rel(k23(a), k23(b))), rel(ik13(a), ik13(b))));
    }
    char m[160];
    std::snprintf(m, sizeof m, "jump at x = 2.5 and 3: max rel. %.1e (< 1e-12)", d);
    check(d < 1e-12, m);
}

// 4th-order central difference; h small against both x and the decay length 1 of e^-x
template <class F>
static double deriv(F f, double x) {
    const double h = 1e-3 * std::fmin(x, 2.0);
    return (8 * (f(x + h) - f(x - h)) - (f(x + 2 * h) - f(x - 2 * h))) / (12 * h);
}

static void identities() {
    std::printf("3. Differential identities, 2000 points x = 1e-6 ... 60 (4th-order differences, round-off ~1e-9)\n");
    double e1 = 0, e2 = 0, e3 = 0;
    for (int i = 0; i < 2000; ++i) {
        const double x = std::pow(10.0, -6.0 + 7.778 * i / 1999.0);
        const double K1 = k13(x), K2 = k23(x);
        e1 = std::fmax(e1, rel(deriv(k13<double>, x), -K2 - K1 / (3 * x)));
        e2 = std::fmax(e2, rel(deriv(k23<double>, x), -K1 - 2 * K2 / (3 * x)));
        e3 = std::fmax(e3, rel(deriv(ik13<double>, x), -K1));
    }
    char m[160];
    std::snprintf(m, sizeof m, "K_{1/3}' = -K_{2/3} - K_{1/3}/(3x):    max rel. %.1e (< 1e-8)", e1);
    check(e1 < 1e-8, m);
    std::snprintf(m, sizeof m, "K_{2/3}' = -K_{1/3} - 2K_{2/3}/(3x):   max rel. %.1e (< 1e-8)", e2);
    check(e2 < 1e-8, m);
    std::snprintf(m, sizeof m, "IntK_{1/3}' = -K_{1/3}:                max rel. %.1e (< 1e-8)", e3);
    check(e3 < 1e-8, m);
}

static void limits() {
    std::printf("4. Limits\n");
    const double x0 = 1e-12;   // K_nu ~ Gamma(nu)/2 (2/x)^nu (1 + O(x^{2 nu}))
    const double s1 = std::tgamma(1.0 / 3) / 2 * std::pow(2 / x0, 1.0 / 3);
    const double s2 = std::tgamma(2.0 / 3) / 2 * std::pow(2 / x0, 2.0 / 3);
    char m[160];
    std::snprintf(m, sizeof m, "x = 1e-12: K_{1/3}, K_{2/3} vs Gamma(nu)/2 (2/x)^nu: rel. %.1e, %.1e (< 1e-6)",
                  rel(k13(x0), s1), rel(k23(x0), s2));
    check(rel(k13(x0), s1) < 1e-6 && rel(k23(x0), s2) < 1e-6, m);
    std::snprintf(m, sizeof m, "IntK_{1/3}(0) = pi/sqrt(3), IntK(1e-300) the same: rel. %.1e",
                  rel(ik13(1e-300), M_PI / std::sqrt(3.0)));
    check(ik13(0.0) == PI_OVER_SQRT3 && rel(ik13(1e-300), M_PI / std::sqrt(3.0)) < 1e-15, m);
    check(std::isinf(k13(0.0)) && std::isinf(k23(0.0)), "K_{1/3}(0) = K_{2/3}(0) = +inf");
    // Hankel: e^x K_nu ~ sqrt(pi/2x) (1 + (4nu^2-1)/(8x) + (4nu^2-1)(4nu^2-9)/(2 (8x)^2));
    // e^x IntK_nu ~ sqrt(pi/2x) (1 + (a1 - 1/2)/x + (a2 - 3a1/2 + 3/4)/x^2), a1, a2 the two coefficients above
    // (integrated term by term)
    const double X = 1e4, h = std::sqrt(M_PI / (2 * X));
    auto hk = [&](double nu) { const double m2 = 4 * nu * nu; return h * (1 + (m2 - 1) / (8 * X) + (m2 - 1) * (m2 - 9) / (128 * X * X)); };
    const double a1 = (4.0 / 9 - 1) / 8, a2 = (4.0 / 9 - 1) * (4.0 / 9 - 9) / 128;
    const double hi = h * (1 + (a1 - 0.5) / X + (a2 - 1.5 * a1 + 0.75) / (X * X));
    const double e = std::fmax(std::fmax(rel(k13e(X), hk(1.0 / 3)), rel(k23e(X), hk(2.0 / 3))), rel(ik13e(X), hi));
    std::snprintf(m, sizeof m, "x = 1e4: scaled functions vs Hankel expansion: max rel. %.1e (< 1e-11)", e);
    check(e < 1e-11, m);
    check(k13(800.0) == 0.0 && k13e(800.0) > 0, "x = 800: K underflows to 0, e^x K stays finite");
}

static void variants() {
    std::printf("5. Combined evaluation, float, device kernel\n");
    bool same = true;
    for (int i = 0; i < 4000; ++i) {
        const double x = std::pow(10.0, -7.0 + 9.7 * i / 3999.0);
        double a, b, c;
        synchrotron_functions(x, a, b, c);
        same = same && a == k13(x) && b == k23(x) && c == ik13(x);
    }
    check(same, "synchrotron_functions() bit-identical to k13, k23, ik13 (4000 points)");
    double ef = 0;
    for (const auto& r : REF) {
        if (r[0] > 50) continue;
        const float x = static_cast<float>(r[0]);
        const double xd = x;   // compare at the float argument
        ef = std::fmax(ef, std::fmax(std::fmax(rel(k13(x), k13(xd)), rel(k23(x), k23(xd))), rel(ik13(x), ik13(xd))));
    }
    char m[160];
    std::snprintf(m, sizeof m, "float instantiation up to x = 50: max rel. %.1e (< 1e-4)", ef);
    check(ef < 1e-4, m);
    // (float: the Airy series cancel near the switch points, ~2 digits)
    // device kernel: same results as on the host (identical code; on GPUs up to fma contraction)
    const int N = 100000;
    double dmax = 0;
    Kokkos::parallel_reduce("bessel.test", Kokkos::RangePolicy<>(0, N), KOKKOS_LAMBDA(int i, double& d) {
        const double x = Kokkos::pow(10.0, -7.0 + 9.5 * i / (N - 1.0));
        double a, b, c;
        synchrotron_functions(x, a, b, c);
        const double e = Kokkos::fmax(Kokkos::fmax(Kokkos::fabs(a / k13(x) - 1), Kokkos::fabs(b / k23(x) - 1)),
                                      Kokkos::fabs(c / ik13(x) - 1));
        if (e > d) d = e;
    }, Kokkos::Max<double>(dmax));
    std::snprintf(m, sizeof m, "Kokkos kernel (%s), 1e5 points: combined vs single functions %.1e",
                  Kokkos::DefaultExecutionSpace::name(), dmax);
    check(dmax < 1e-14, m);
}

int main(int argc, char** argv) {
    Kokkos::initialize(argc, argv);
    references();
    switch_points();
    identities();
    limits();
    variants();
    Kokkos::finalize();
    std::printf("%s (%d failures)\n", failures ? "FAILED" : "all tests passed", failures);
    return failures ? 1 : 0;
}
