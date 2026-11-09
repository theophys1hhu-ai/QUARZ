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

} // namespace qed
} // namespace quarz
