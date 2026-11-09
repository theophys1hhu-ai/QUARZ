// u-integrals of the nonlinear Compton coefficients (see QEDRates.hpp), by quadrature in r = u^{1/3}
// (du = 3 r^2 dr removes the u^{-2/3} singularity of w at u -> 0; the integrands are smooth in r).
#include "QEDRates.hpp"

#include <cstdio>
#include <map>
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

namespace {
std::string num(double v) {
    char b[32];
    std::snprintf(b, sizeof b, "%.17g", v);
    return b;
}
// meta entries of a table file; points_per_decade is the effective (n - 1) / decades (informational)
std::map<std::string, std::string> table_meta(const char* kind, const LogTable& t) {
    std::map<std::string, std::string> m{{"table", kind}, {"version", std::to_string(TABLE_VERSION)},
                                         {"points_per_decade", num((t.size() - 1) / std::log10(t.axis_max() / t.axis_min()))}};
    if (std::string(kind) == "qed.rate") m["rmax_safety"] = num(RMAX_SAFETY);
    return m;
}
// number of grid points LogTable::build makes for these parameters
int grid_points(double lo, double hi, double ppd) {
    return std::max(4, static_cast<int>(std::ceil(std::log10(hi / lo) * ppd)) + 1);
}
// "" if the file's meta identifies a table of this kind and version, else the reason
std::string check_meta(const char* kind, const std::map<std::string, std::string>& m) {
    auto get = [&](const char* k) { auto it = m.find(k); return it == m.end() ? std::string() : it->second; };
    if (get("table") != kind) return "not a " + std::string(kind) + " table";
    if (get("version") != std::to_string(TABLE_VERSION))
        return "table version " + (get("version").empty() ? std::string("?") : get("version")) + ", expected " + std::to_string(TABLE_VERSION);
    if (std::string(kind) == "qed.rate" && get("rmax_safety") != num(RMAX_SAFETY)) return "different rmax_safety";
    return "";
}
const std::vector<std::string> BESSEL_LABELS{"exp(y)K_1/3", "exp(y)K_2/3", "exp(y)IntK_1/3"};
const std::vector<std::string> RATE_LABELS{"W", "G", "Xi", "K", "Wu", "Gu", "Rmax"};

template <class Tab, class Load, class Build>
Tab load_or_build(const char* kind, const std::string& path, double lo, double hi, double ppd, bool write,
                  std::string* status, Load load, Build build, void (*save)(const Tab&, const std::string&)) {
    std::string why;
    if (!path.empty()) {
        if (FILE* fp = std::fopen(path.c_str(), "r")) {
            std::fclose(fp);
            try {
                std::vector<std::string> lab;
                std::map<std::string, std::string> m;
                const LogTable t = LogTable::load(path, &lab, &m);
                why = check_meta(kind, m);
                if (why.empty() && (t.axis_min() != lo || t.axis_max() != hi || t.size() != grid_points(lo, hi, ppd)))
                    why = "different range or points per decade";
                if (why.empty()) {
                    if (status) *status = "loaded";
                    return load(t);
                }
            } catch (const std::exception& e) {
                why = e.what();
            }
        } else {
            why = "no file";
        }
    }
    Tab T = build();
    std::string st = "built";
    if (!path.empty() && write) {
        save(T, path);
        st += why == "no file" ? " (saved)" : " (" + why + "; saved)";
    } else if (!path.empty()) {
        st += " (" + why + ")";
    }
    if (status) *status = st;
    return T;
}
} // namespace

void save_bessel_table(const BesselTable& T, const std::string& path) {
    T.t.save(path, BESSEL_LABELS, table_meta("qed.bessel", T.t));
}
void save_rate_table(const RateTable& T, const std::string& path) {
    T.t.save(path, RATE_LABELS, table_meta("qed.rate", T.t));
}

namespace {
BesselTable bessel_from(const LogTable& t) {
    BesselTable T;
    T.t = t;
    T.y_min = t.axis_min();
    T.y_max = t.axis_max();
    return T;
}
RateTable rate_from(const LogTable& t) {
    RateTable T;
    T.t = t;
    return T;
}
} // namespace

BesselTable load_bessel_table(const std::string& path) {
    std::map<std::string, std::string> m;
    const LogTable t = LogTable::load(path, nullptr, &m);
    const std::string why = check_meta("qed.bessel", m);
    if (!why.empty() || t.ncomp() != 3) throw std::runtime_error("load_bessel_table: " + path + ": " + (why.empty() ? "3 columns expected" : why));
    return bessel_from(t);
}
RateTable load_rate_table(const std::string& path) {
    std::map<std::string, std::string> m;
    const LogTable t = LogTable::load(path, nullptr, &m);
    const std::string why = check_meta("qed.rate", m);
    if (!why.empty() || t.ncomp() != 7) throw std::runtime_error("load_rate_table: " + path + ": " + (why.empty() ? "7 columns expected" : why));
    return rate_from(t);
}

BesselTable bessel_table(const std::string& path, double y_min, double y_max, double ppd, bool write, std::string* status) {
    return load_or_build<BesselTable>("qed.bessel", path, y_min, y_max, ppd, write, status,
        [&](const LogTable& t) { if (t.ncomp() != 3) throw std::runtime_error("3 columns expected"); return bessel_from(t); },
        [&] { return make_bessel_table(y_min, y_max, ppd); }, save_bessel_table);
}
RateTable rate_table(const std::string& path, double chi_min, double chi_max, double ppd, bool write, std::string* status) {
    return load_or_build<RateTable>("qed.rate", path, chi_min, chi_max, ppd, write, status,
        [&](const LogTable& t) { if (t.ncomp() != 7) throw std::runtime_error("7 columns expected"); return rate_from(t); },
        [&] { return make_rate_table(chi_min, chi_max, ppd); }, save_rate_table);
}

} // namespace qed
} // namespace quarz
