// Unit test of src/QEDRates.*: nonlinear Compton coefficients (eq. 4), chi, basis, u-integrals.
//  1. coefficients w, g, p_f, p_xi at 10 points (u, chi) vs mpmath (tests/qed_reference.py rates)
//  2. u-integrals W, G, Xi, K, Wu, Gu, gaunt g(chi), polarization limit at chi = 1e-3 ... 100 vs mpmath
//  3. small-chi limits: W -> 5 pi chi / 2, G -> -3 pi chi^2 / 4, g(chi) -> 1, S_inf -> -8 sqrt(3)/15
//  4. physical admissibility (1e6 random u, chi, |S_i| <= 1): every resolved probability >= 0,
//     i.e. w + <g, S_i> >= |p_f| and >= |p_xi| (final spin and Stokes vector of length <= 1).
//     The margin is small for S_i near -e2 at large y (3e-4 at y = 370, mpmath), so this is sharp.
//  5. sum over final spins and polarizations of the resolved rate = w + <g, S_i>
//  6. chi and the basis (e1, e2, e_v) in simple fields, electron and positron
//  7. sampling density in r = u^{1/3}: limit at r = 0 and normalization; device kernel
#include "QEDRates.hpp"

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
static double norm(V3 a) { return std::sqrt(push::dot(a, a)); }

// u, chi, w, g, f2, fs, fv, x1, x2, x3, x3s   (mpmath, tests/qed_reference.py rates)
static const double REF_COEF[][11] = {
    {0.001, 0.1, 58.8297802887033186, -0.00866048711375864579, -0.00866915627002867446, 58.8297499807254542, 0.0000285806607549642254, 0.00866915627002867446, 0.0588600579585898348, 30.277669886516193, -0.00866915627002867446},
    {0.05, 0.1, 2.60443340972923127, -0.0670354222661907577, -0.0705636023854639555, 2.59990476089241823, 0.00231320579711437785, 0.0705636023854639555, 0.13452388688143395, 1.72088655798895465, -0.0705636023854639555},
    {0.3, 0.1, 0.0566082846381973846, -0.0125089065944574748, -0.0178698665635106783, 0.0509694403942154787, 0.000914369520988655598, 0.0178698665635106783, 0.0209296763622465496, 0.0438576774531926018, -0.0178698665635106783},
    {0.1, 1, 10.4511748200237279, -0.334892379684210863, -0.372102644093567626, 10.3856726953393137, 0.0498942385971337393, 0.372102644093567626, 1.10406939421834556, 5.89519122159727715, -0.372102644093567626},
    {0.5, 1, 1.63705882817329942, -0.368089948041023153, -0.736179896082046306, 1.20406653640786036, 0.169040976438491127, 0.736179896082046306, 1.03502555996936924, 0.865984583530878109, -0.736179896082046306},
    {0.9, 1, 0.0118330485981527909, -0.00112926185641491298, -0.0112926185641491298, 0.00140434662190181167, 0.000946505661153695289, 0.0112926185641491298, 0.0116926139359626097, 0.0012874940711420962, -0.0112926185641491298},
    {0.2, 10, 31.9516333277497382, -1.23913927931681935, -1.54892409914602419, 31.13198929736912, 0.736955434487837744, 1.54892409914602419, 7.04604188985444225, 16.3928806076123651, -1.54892409914602419},
    {0.9, 10, 9.34671905989843235, -0.742584372546525217, -7.42584372546525217, 1.3857924332290725, 3.26399208248612737, 7.42584372546525217, 9.2081398165755251, 0.982830447736958007, -7.42584372546525217},
    {0.999, 10, 1.83024739470836697e-27, -1.82568700000721948e-30, -1.82568700000721948e-27, 1.84999995075711591e-30, 1.79044060979425842e-29, 1.82568700000721948e-27, 1.83024554470841621e-27, 1.83205968206205189e-30, -1.82568700000721948e-27},
    {0.5, 1000, 350.266270495326403, -9.58882950990400603, -19.1776590198080121, 279.854106138131257, 69.5148887118704831, 19.1776590198080121, 210.339217426260774, 140.824328714390291, -19.1776590198080121},
};

// chi, W, G, Xi, K, Wu, Gu, gaunt, polarization limit   (mpmath, tests/qed_reference.py rates)
static const double REF_INT[][9] = {
#include "qed_reference_integrals.inc"
};

static void pointwise() {
    std::printf("1. Coefficients of eq. 4 at 10 points (u, chi), u = 0.001 ... 0.999, chi = 0.1 ... 1000\n");
    double e = 0;
    for (const auto& r : REF_COEF) {
        const Coef c = coefficients(r[0], r[1]);
        const double v[9] = {c.w, c.g, c.f2, c.fs, c.fv, c.x1, c.x2, c.x3, c.x3s};
        for (int k = 0; k < 9; ++k) e = std::fmax(e, rel(v[k], r[2 + k]));
    }
    char m[160];
    std::snprintf(m, sizeof m, "w, g, p_f, p_xi coefficients vs mpmath: max rel. error %.1e (< 1e-10)", e);
    check(e < 1e-10, m);   // K_{2/3} - IntK_{1/3} cancels at y ~ 70 (u = 0.999)
    check(w_rad(0.5, 1.0) == coefficients(0.5, 1.0).w && g_rad(0.5, 1.0) == coefficients(0.5, 1.0).g,
          "w_rad(), g_rad() == coefficients()");
    const Coef z1 = coefficients(1.0, 1.0), z2 = coefficients(0.5, 0.0), z3 = coefficients(0.5, 1e-6);
    check(z1.w == 0 && z2.w == 0 && z3.w == 0 && z3.x3s == 0, "u = 1, chi = 0 and y > 700 (chi = 1e-6, u = 0.5): all zero");
    const Coef c0 = coefficients(0.0, 1.0);
    check(std::isinf(c0.w) && std::isinf(c0.fs) && std::isinf(c0.x3) && c0.g == 0 && c0.x1 == 0,
          "u = 0: w, fs, x3 = +inf (K_{2/3}(0)), the terms with a factor u = 0");
}

static void integral_values() {
    std::printf("2. u-integrals at chi = 1e-3 ... 100 vs mpmath\n");
    double e[8] = {0};
    for (const auto& r : REF_INT) {
        const Integrals I = integrals(r[0]);
        const double v[6] = {I.W, I.G, I.Xi, I.K, I.Wu, I.Gu};
        for (int k = 0; k < 6; ++k) e[k] = std::fmax(e[k], rel(v[k], r[1 + k]));
        e[6] = std::fmax(e[6], rel(gaunt(r[0]), r[7]));
        e[7] = std::fmax(e[7], rel(polarization_limit(r[0]), r[8]));
    }
    const char* names[8] = {"W", "G", "Xi", "K", "Wu", "Gu", "gaunt g(chi)", "polarization limit"};
    for (int k = 0; k < 8; ++k) {
        char m[160];
        std::snprintf(m, sizeof m, "%-18s max rel. error %.1e (< 1e-12)", names[k], e[k]);
        check(e[k] < 1e-12, m);
    }
}

static void limits() {
    std::printf("3. Small-chi limits (chi = 1e-5; corrections O(chi))\n");
    const double x = 1e-5;
    const Integrals I = integrals(x);
    const double eW = rel(I.W, 2.5 * M_PI * x), eG = rel(I.G, -0.75 * M_PI * x * x), eX = rel(I.Xi, 2.5 * M_PI * x);
    const double eg = std::abs(gaunt(x) - 1), eS = rel(polarization_limit(x), -8 * std::sqrt(3.0) / 15);
    char m[200];
    std::snprintf(m, sizeof m, "W -> 5 pi chi/2: %.1e, Xi -> 5 pi chi/2: %.1e, G -> -3 pi chi^2/4: %.1e (< 1e-4)", eW, eX, eG);
    check(eW < 1e-4 && eX < 1e-4 && eG < 1e-4, m);
    std::snprintf(m, sizeof m, "Gaunt factor g -> 1: |g - 1| = %.1e (55 sqrt(3)/16 chi = %.1e)", eg, 55 * std::sqrt(3.0) / 16 * x);
    check(eg < 1e-4, m);
    std::snprintf(m, sizeof m, "S_inf -> -8 sqrt(3)/15 (Sokolov-Ternov): rel. %.1e (< 1e-4)", eS);
    check(eS < 1e-4, m);
    // spin relaxation integrals: limits, and the identities with the other integrals at chi = 1
    const double x3 = x * x * x;
    const double eB = rel(I.SB, 15 * M_PI / 8 * x3), eA = rel(I.SA, std::sqrt(3.0) * M_PI * x3), eL = rel(I.SL, 35 * M_PI / 24 * x3);
    std::snprintf(m, sizeof m, "SB -> 15 pi chi^3/8: %.1e, SA -> sqrt(3) pi chi^3: %.1e, SL -> 35 pi chi^3/24: %.1e (< 2e-4)", eB, eA, eL);
    check(eB < 2e-4 && eA < 2e-4 && eL < 2e-4, m);
    const Integrals J = integrals(1.0);
    const double V = integrate([&](double r) { return 3 * r * r * coefficients(r * r * r, 1.0).fv; }, 0.0, 1.0);
    const double i1 = rel(J.SB, J.W - J.Xi), i2 = rel(J.SA, J.G - J.K), i3 = rel(J.SL, J.SB - V), i4 = rel(-J.SA / J.SB, polarization_limit(1.0));
    std::snprintf(m, sizeof m, "chi = 1: SB = W - Xi %.1e, SA = G - K %.1e, SL = SB - int fv %.1e, -SA/SB = polarization limit %.1e (< 1e-11)", i1, i2, i3, i4);
    check(i1 < 1e-11 && i2 < 1e-11 && i3 < 1e-11 && i4 < 1e-11, m);
}

static V3 random_unit(std::mt19937_64& rng) {
    std::normal_distribution<double> N(0, 1);
    V3 v{N(rng), N(rng), N(rng)};
    return (1 / norm(v)) * v;
}

static void admissibility() {
    std::printf("4. Admissibility, 1e6 random (u, chi = 1e-3 ... 1e3, |S_i| <= 1, random basis)\n");
    std::mt19937_64 rng(11);
    std::uniform_real_distribution<double> U(0, 1);
    double mf = 1, mx = 1, mw = 1;   // min of (rate - |p_f|)/rate, (rate - |p_xi|)/rate, (w - |g|)/w
    long n = 0;
    for (int i = 0; i < 1000000; ++i) {
        const double r = U(rng), u = r * r * r, chi = std::pow(10.0, -3 + 6 * U(rng));
        Coef c = coefficients(u, chi);
        if (!(c.w > 0)) continue;
        ++n;
        // scale to w = 1: at large y all coefficients are ~1e-300 and their squares underflow
        const double sc = 1 / c.w;
        for (Real* f : {&c.w, &c.g, &c.f2, &c.fs, &c.fv, &c.x1, &c.x2, &c.x3, &c.x3s}) *f *= sc;
        Basis b;
        b.ev = random_unit(rng);
        const V3 t = random_unit(rng);
        const V3 e1 = t - push::dot(t, b.ev) * b.ev;
        b.e1 = (1 / norm(e1)) * e1;
        b.e2 = push::cross(b.ev, b.e1);
        const double len = i % 4 == 0 ? 1.0 : std::cbrt(U(rng));   // pure states and the unit ball
        const V3 S = len * random_unit(rng);
        const double R = rate(c, b, S);
        mf = std::fmin(mf, (R - norm(p_f(c, b, S))) / R);
        mx = std::fmin(mx, (R - norm(p_xi(c, b, S))) / R);
        mw = std::fmin(mw, (c.w - std::abs(c.g)) / c.w);
    }
    char m[200];
    std::snprintf(m, sizeof m, "%ld samples: min (w - |g|)/w = %.2e >= 0 (rate > 0 for every spin)", n, mw);
    check(mw >= 0, m);
    std::snprintf(m, sizeof m, "min (w + <g,S> - |p_f|)/(w + <g,S>) = %.2e >= -1e-12 (P(S_f = +-n) >= 0, |S_f| <= 1)", mf);
    check(mf >= -1e-12, m);
    std::snprintf(m, sizeof m, "min (w + <g,S> - |p_xi|)/(w + <g,S>) = %.2e >= -1e-12 (|xi| <= 1)", mx);
    check(mx >= -1e-12, m);
}

static void sum_rule() {
    std::printf("5. Sum over final states\n");
    std::mt19937_64 rng(5);
    double e = 0;
    for (int i = 0; i < 10000; ++i) {
        Basis b;
        b.ev = random_unit(rng);
        const V3 t = random_unit(rng), e1 = t - push::dot(t, b.ev) * b.ev;
        b.e1 = (1 / norm(e1)) * e1;
        b.e2 = push::cross(b.ev, b.e1);
        const Coef c = coefficients(0.05 + 0.9 * i / 10000.0, 0.5);
        const V3 S = random_unit(rng), n = random_unit(rng), m = random_unit(rng);
        double s = 0;
        for (int a : {-1, 1})
            for (int d : {-1, 1}) s += rate_resolved(c, b, S, double(a) * n, double(d) * m);
        e = std::fmax(e, rel(s, rate(c, b, S)));
    }
    char m[160];
    std::snprintf(m, sizeof m, "sum over S_f = +-n, xi = +-m of eq. 3 = w + <g, S_i>: max rel. %.1e", e);
    check(e < 1e-14, m);
}

static void fields() {
    std::printf("6. chi and basis in simple fields (eta = 1e-6)\n");
    const double eta = 1e-6, g = 1e4, pu = std::sqrt(g * g - 1);
    Basis b;
    // electron along x in B = B z: Lorentz force -e v x B = +y
    double c = chi_basis({pu, 0, 0}, {0, 0, 0}, {0, 0, 2.0}, -1.0, eta, b);
    const bool be = norm(b.e1 - V3{0, 1, 0}) < 1e-15 && norm(b.e2 - V3{0, 0, 1}) < 1e-15 && norm(b.ev - V3{1, 0, 0}) < 1e-15;
    char m[200];
    std::snprintf(m, sizeof m, "electron, u = u x, B = B z: e1 = y, e2 = z (along B), chi = eta |u| B (rel. %.0e)", rel(c, eta * pu * 2));
    check(be && rel(c, eta * pu * 2) < 1e-15, m);
    c = chi_basis({pu, 0, 0}, {0, 0, 0}, {0, 0, 2.0}, 1.0, eta, b);
    check(norm(b.e1 - V3{0, -1, 0}) < 1e-15 && norm(b.e2 - V3{0, 0, -1}) < 1e-15, "positron: e1 = -y, e2 = -z");
    c = chi_basis({0, 0, pu}, {0, 0, 3.0}, {0, 0, 0}, -1.0, eta, b);
    std::snprintf(m, sizeof m, "E parallel to u: chi = eta |E| (rel. %.0e), e1 perpendicular to e_v", rel(c, eta * 3));
    check(rel(c, eta * 3) < 1e-10 && std::abs(push::dot(b.e1, b.ev)) < 1e-15, m);
    c = chi(V3{pu, 0, 0}, V3{0, 3.0, 0}, V3{0, 0, 0}, eta);
    std::snprintf(m, sizeof m, "E perpendicular to u: chi = eta gamma |E| (rel. %.0e)", rel(c, eta * g * 3));
    check(rel(c, eta * g * 3) < 1e-15, m);
    // force-free: E = -v x B, u = gamma v
    const V3 v{0, 0, pu / g}, B{0, 5.0, 0}, E = -1.0 * push::cross(v, B);
    c = chi({0, 0, pu}, E, B, eta);
    std::snprintf(m, sizeof m, "force-free E + v x B = 0: chi = %.1e (vs eta gamma |E| = %.1e)", c, eta * g * norm(E));
    check(c < 1e-12 * eta * g * norm(E), m);
    // orthonormal basis for random states
    std::mt19937_64 rng(3);
    double eo = 0;
    for (int i = 0; i < 1000; ++i) {
        chi_basis(1e3 * random_unit(rng), random_unit(rng), random_unit(rng), -1.0, eta, b);
        eo = std::fmax(eo, std::fmax(std::abs(push::dot(b.e1, b.e2)), std::abs(push::dot(b.e1, b.ev))));
        eo = std::fmax(eo, std::fmax(std::abs(norm(b.e1) - 1), std::abs(norm(b.e2) - 1)));
        eo = std::fmax(eo, norm(push::cross(b.e1, b.e2) - b.ev));
    }
    std::snprintf(m, sizeof m, "1000 random states: (e1, e2, e_v) right-handed orthonormal to %.0e", eo);
    check(eo < 1e-14, m);
    std::snprintf(m, sizeof m, "eta(n0 = 1e18 cm^-3) = %.6e (hbar omega_p / m c^2 = 0.037134 eV / 0.51100 MeV = 7.2669e-8)",
                  eta_from_n0_cm3(1e18));
    check(rel(eta_from_n0_cm3(1e18), 7.2669e-8) < 1e-4, m);
}

static void sampling() {
    std::printf("7. Sampling density 3 r^2 (w + g s2), device kernel\n");
    double e0 = 0, en = 0;
    for (double x : {1e-3, 0.1, 1.0, 10.0, 100.0}) {
        e0 = std::fmax(e0, rel(rate_r(1e-6, x, 0.7), rate_r(0.0, x, 0.7)));
        const Integrals I = integrals(x);
        for (double s2 : {-1.0, 0.3, 1.0})
            en = std::fmax(en, rel(integrate([&](double r) { return rate_r(r, x, s2); }, 0.0, 1.0), I.W + I.G * s2));
    }
    char m[200];
    std::snprintf(m, sizeof m, "r = 0 limit 3 Gamma(2/3) (3 chi)^{2/3} vs r = 1e-6: max rel. %.1e (< 1e-6)", e0);
    check(e0 < 1e-6, m);
    std::snprintf(m, sizeof m, "int_0^1 dr = W + G s2: max rel. %.1e (< 1e-11)", en);
    check(en < 1e-11, m);
    const int N = 100000;
    double d = 0;
    Kokkos::parallel_reduce("qed.test", Kokkos::RangePolicy<>(0, N), KOKKOS_LAMBDA(int i, double& dm) {
        const Real u = (i + Real(0.5)) / N, x = Kokkos::pow(Real(10), Real(-2) + Real(4) * (i % 97) / Real(96));
        const Coef c = coefficients(u, x);
        const Real e = Kokkos::fabs(c.w / w_rad(u, x) - 1) + Kokkos::fabs(c.x3 / coefficients(u, x).x3 - 1);
        if (e > dm) dm = e;
    }, Kokkos::Max<double>(d));
    // device kernel vs the host: same functions, but compilers may contract a*b+c differently
    const double HOSTW0 = w_rad(0.01, 1.0), HOSTW1 = w_rad(0.3, 1.0), HOSTW2 = w_rad(0.6, 1.0), HOSTW3 = w_rad(0.95, 1.0);
    double dh = 0;
    Kokkos::parallel_reduce("qed.test2", Kokkos::RangePolicy<>(0, 4), KOKKOS_LAMBDA(int i, double& dm) {
        const Real us[4] = {0.01, 0.3, 0.6, 0.95}, ws[4] = {HOSTW0, HOSTW1, HOSTW2, HOSTW3};
        const Real e = Kokkos::fabs(w_rad(us[i], Real(1)) / ws[i] - 1);
        if (e > dm) dm = e;
    }, Kokkos::Max<double>(dh));
    std::snprintf(m, sizeof m, "Kokkos kernel (%s): coefficients consistent within the kernel %.1e, vs the host %.1e (< 1e-12)",
                  Kokkos::DefaultExecutionSpace::name(), d, dh);
    check(d < 1e-12 && dh < 1e-12, m);
}

int main(int argc, char** argv) {
    Kokkos::initialize(argc, argv);
    pointwise();
    integral_values();
    limits();
    admissibility();
    sum_rule();
    fields();
    sampling();
    Kokkos::finalize();
    std::printf("%s (%d failures)\n", failures ? "FAILED" : "all tests passed", failures);
    return failures ? 1 : 0;
}
