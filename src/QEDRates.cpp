// u-integrals of the nonlinear Compton coefficients (see QEDRates.hpp), by quadrature in r = u^{1/3}
// (du = 3 r^2 dr removes the u^{-2/3} singularity of w at u -> 0; the integrands are smooth in r).
#include "QEDRates.hpp"

namespace quarz {
namespace qed {

namespace {
constexpr double RTOL = 1e-13;
// int_0^1 f(u) du = int_0^1 f(r^3) 3 r^2 dr
template <class F>
double u_integral(F f) {
    return integrate([&](double r) { return 3 * r * r * f(r * r * r); }, 0.0, 1.0, RTOL);
}
} // namespace

Integrals integrals(double chi) {
    Integrals I;
    if (!(chi > 0)) return I;
    I.W = u_integral([&](double u) { return coefficients(u, chi).w; });
    I.G = u_integral([&](double u) { return coefficients(u, chi).g; });
    I.Xi = u_integral([&](double u) { return coefficients(u, chi).fs; });
    I.K = u_integral([&](double u) { return coefficients(u, chi).f2; });
    I.Wu = u_integral([&](double u) { return u * coefficients(u, chi).w; });
    I.Gu = u_integral([&](double u) { return u * coefficients(u, chi).g; });
    return I;
}

double gaunt(double chi) {
    if (!(chi > 0)) return 1.0;
    const double Wu = u_integral([&](double u) { return u * coefficients(u, chi).w; });
    return Wu / (2 * M_PI / std::sqrt(3.0) * chi * chi);
}

double polarization_limit(double chi) {
    // K - G = -int u^2/(1-u) K_{1/3} du,  Xi - W = -int u^2/(1-u) K_{2/3} du: integrated directly
    // (the differences of the separate integrals cancel to O(chi^2) for small chi)
    if (!(chi > 0)) return -8 * std::sqrt(3.0) / 15;
    auto kern = [&](double u, bool k13) {
        if (u >= 1) return 0.0;
        const double y = y_arg(u, chi);
        if (y >= Y_MAX) return 0.0;
        double K13, K23, IK13;
        bessel::synchrotron_functions(y, K13, K23, IK13);
        return u * u / (1 - u) * (k13 ? K13 : K23);
    };
    const double A = u_integral([&](double u) { return kern(u, true); });
    const double B = u_integral([&](double u) { return kern(u, false); });
    return -A / B;
}

double max_rate_r(double chi) {
    if (!(chi > 0)) return 0.0;
    auto f = [&](double r) { return rate_r(r, chi, -1.0); };   // g < 0: s2 = -1 gives w + |g|
    const int N = 400;
    int kbest = 0;
    double best = f(0.0);
    for (int k = 1; k <= N; ++k) {
        const double v = f(static_cast<double>(k) / N);
        if (v > best) { best = v; kbest = k; }
    }
    double a = std::max(0, kbest - 1) / static_cast<double>(N), b = std::min(N, kbest + 1) / static_cast<double>(N);
    const double gr = 0.5 * (std::sqrt(5.0) - 1);
    double c = b - gr * (b - a), d = a + gr * (b - a), fc = f(c), fd = f(d);
    for (int it = 0; it < 80 && b - a > 1e-12; ++it) {
        if (fc > fd) { b = d; d = c; fd = fc; c = b - gr * (b - a); fc = f(c); }
        else { a = c; c = d; fc = fd; d = a + gr * (b - a); fd = f(d); }
    }
    return std::max(best, std::max(fc, fd));
}

BesselTable make_bessel_table(double y_min, double y_max, double ppd) {
    BesselTable T;
    T.t = LogTable::build(y_min, y_max, ppd, 3, [](double y, double* o) {
        o[0] = bessel::k13e(y); o[1] = bessel::k23e(y); o[2] = bessel::ik13e(y);
    }, {LogTable::Log, LogTable::Log, LogTable::Log}, "qed.bessel_table");
    T.y_min = y_min;
    T.y_max = y_max;
    return T;
}

RateTable make_rate_table(double chi_min, double chi_max, double ppd) {
    RateTable T;
    T.t = LogTable::build(chi_min, chi_max, ppd, 7, [](double chi, double* o) {
        const Integrals I = integrals(chi);
        o[0] = I.W; o[1] = I.G; o[2] = I.Xi; o[3] = I.K; o[4] = I.Wu; o[5] = I.Gu;
        o[6] = RMAX_SAFETY * max_rate_r(chi);
    }, std::vector<int>(7, LogTable::Log), "qed.rate_table");
    return T;
}

} // namespace qed
} // namespace quarz
