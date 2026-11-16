// u-integrals of the nonlinear Compton coefficients, the look-up tables built from them and their files
// (see QEDRates.hpp). Quadratures in r = u^{1/3}: du = 3 r^2 dr removes the u^{-2/3} singularity of w at
// u -> 0, and the integrands are smooth in r.
#include "QEDRates.hpp"

#include <stdexcept>

namespace quarz {
namespace qed {

namespace {
constexpr double RTOL = 1e-13;

// int_0^1 f(u) du = int_0^1 f(r^3) 3 r^2 dr
template <class F>
double u_integral(F f) {
    return integrate([&](double r) { return 3 * r * r * f(r * r * r); }, 0.0, 1.0, RTOL);
}

// the spin relaxation integrals int_0^1 u^2/(1-u) K du with K = K_{2/3} (SB), K_{1/3} (SA), IntK_{1/3} (SL).
// They are integrated directly: as differences of the other integrals (W - Xi, G - K) they would cancel
// to O(chi^3) for small chi.
enum SpinKernel { K23_KERNEL, K13_KERNEL, IK13_KERNEL };
double spin_integral(double chi, SpinKernel which) {
    return u_integral([&](double u) {
        if (u >= 1) return 0.0;
        const double y = y_arg(u, chi);
        if (y >= Y_MAX) return 0.0;
        double K13, K23, IK13;
        bessel::synchrotron_functions(y, K13, K23, IK13);
        return u * u / (1 - u) * (which == K23_KERNEL ? K23 : which == K13_KERNEL ? K13 : IK13);
    });
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
    I.SB = spin_integral(chi, K23_KERNEL);
    I.SA = spin_integral(chi, K13_KERNEL);
    I.SL = spin_integral(chi, IK13_KERNEL);
    return I;
}

double gaunt(double chi) {
    if (!(chi > 0)) return 1.0;
    const double Wu = u_integral([&](double u) { return u * coefficients(u, chi).w; });
    return Wu / (2 * M_PI / std::sqrt(3.0) * chi * chi);
}

double polarization_limit(double chi) {
    if (!(chi > 0)) return -8 * std::sqrt(3.0) / 15;
    return -spin_integral(chi, K13_KERNEL) / spin_integral(chi, K23_KERNEL);
}

double max_rate_r(double chi) {
    if (!(chi > 0)) return 0.0;
    auto f = [&](double r) { return rate_r(r, chi, -1.0); };   // g < 0: s2 = -1 gives w + |g|
    // coarse scan, then golden-section search around the best grid point
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

// ---- tables

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
    T.t = LogTable::build(chi_min, chi_max, ppd, RateTable::RATE_COLUMNS, [](double chi, double* o) {
        const Integrals I = integrals(chi);
        o[0] = I.W; o[1] = I.G; o[2] = I.Xi; o[3] = I.K; o[4] = I.Wu; o[5] = I.Gu;
        o[6] = RMAX_SAFETY * max_rate_r(chi);
        o[7] = I.SB; o[8] = I.SA; o[9] = I.SL;
    }, std::vector<int>(RateTable::RATE_COLUMNS, LogTable::Log), "qed.rate_table");
    return T;
}

// ---- table files

namespace {
const std::vector<std::string> BESSEL_LABELS{"exp(y)K_1/3", "exp(y)K_2/3", "exp(y)IntK_1/3"};
const std::vector<std::string> RATE_LABELS{"W", "G", "Xi", "K", "Wu", "Gu", "Rmax", "SB", "SA", "SL"};

// meta entries of a Bessel or rate table file; points_per_decade is the effective (n - 1) / decades
std::map<std::string, std::string> table_meta(const std::string& kind, const LogTable& t) {
    std::map<std::string, std::string> m{{"table", kind}, {"version", std::to_string(TABLE_VERSION)},
                                         {"points_per_decade", detail::exact((t.size() - 1) / std::log10(t.axis_max() / t.axis_min()))}};
    if (kind == "qed.rate") m["rmax_safety"] = detail::exact(RMAX_SAFETY);
    return m;
}

// the table of a file, checked for kind, version and number of columns (throws otherwise)
LogTable load_checked(const std::string& path, const std::string& kind, int ncomp) {
    std::map<std::string, std::string> m;
    LogTable t = LogTable::load(path, nullptr, &m);
    std::string why = detail::check_meta(m, kind);
    if (why.empty() && kind == "qed.rate" && detail::meta_get(m, "rmax_safety") != detail::exact(RMAX_SAFETY))
        why = "different rmax_safety";
    if (why.empty() && t.ncomp() != ncomp) why = std::to_string(ncomp) + " columns expected";
    if (!why.empty()) throw std::runtime_error(path + ": " + why);
    return t;
}

// the axis of t is the one LogTable::build makes for these parameters
bool same_axis(const LogTable& t, double lo, double hi, double ppd) {
    return t.axis_min() == lo && t.axis_max() == hi && t.size() == LogTable::grid_size(lo, hi, ppd);
}
} // namespace

void save_bessel_table(const BesselTable& T, const std::string& path) {
    T.t.save(path, BESSEL_LABELS, table_meta("qed.bessel", T.t));
}
void save_rate_table(const RateTable& T, const std::string& path) {
    T.t.save(path, RATE_LABELS, table_meta("qed.rate", T.t));
}

BesselTable load_bessel_table(const std::string& path) {
    BesselTable T;
    T.t = load_checked(path, "qed.bessel", 3);
    T.y_min = T.t.axis_min();
    T.y_max = T.t.axis_max();
    return T;
}
RateTable load_rate_table(const std::string& path) {
    RateTable T;
    T.t = load_checked(path, "qed.rate", RateTable::RATE_COLUMNS);
    return T;
}

BesselTable bessel_table(const std::string& path, double y_min, double y_max, double ppd, bool write, std::string* status) {
    return detail::load_or_build(path, write, status, load_bessel_table,
                                 [&](const BesselTable& T) { return same_axis(T.t, y_min, y_max, ppd); },
                                 [&] { return make_bessel_table(y_min, y_max, ppd); }, save_bessel_table);
}
RateTable rate_table(const std::string& path, double chi_min, double chi_max, double ppd, bool write, std::string* status) {
    return detail::load_or_build(path, write, status, load_rate_table,
                                 [&](const RateTable& T) { return same_axis(T.t, chi_min, chi_max, ppd); },
                                 [&] { return make_rate_table(chi_min, chi_max, ppd); }, save_rate_table);
}

} // namespace qed
} // namespace quarz
