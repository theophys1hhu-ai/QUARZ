// Cumulative photon spectrum table for the inversion sampler (QEDSampling.hpp).
#include "QEDSampling.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <stdexcept>
#include <vector>

namespace quarz {
namespace qed {

namespace {

double logit(double u) { return std::log(u / (1 - u)); }

std::string num(double v) {
    char b[32];
    std::snprintf(b, sizeof b, "%.17g", v);
    return b;
}

// the u axis of the table: nu nodes, uniform in L = logit(u) with dL = ln(10) / points_per_decade
struct UAxis {
    int nu;
    double L0, dL;
};
UAxis u_axis(const SpectrumTableParams& p) {
    const double La = logit(p.u_min), Lb = logit(1 - p.one_minus_u_min);
    const double dL_wanted = std::log(10.0) / p.u_points_per_decade;
    UAxis a;
    a.nu = static_cast<int>(std::ceil((Lb - La) / dL_wanted)) + 1;
    a.L0 = La;
    a.dL = (Lb - La) / (a.nu - 1);
    return a;
}

// Wc(u_j), Gc(u_j) for j = 0 ... nu-1 at one chi, by quadrature (direct Bessel functions):
// [0, u_0] in r = u^{1/3} (smooth integrand), then each interval [L_j, L_{j+1}] in L (du = u (1 - u) dL)
void cumulative_spectrum(double chi, const UAxis& a, double* Wc, double* Gc) {
    auto w = [&](double u) { return coefficients(u, chi).w; };
    auto g = [&](double u) { return coefficients(u, chi).g; };
    const double r0 = std::cbrt(SpectrumTable::u_of_L(a.L0));
    Wc[0] = integrate([&](double r) { return 3 * r * r * w(r * r * r); }, 0.0, r0, 1e-13);
    Gc[0] = integrate([&](double r) { return 3 * r * r * g(r * r * r); }, 0.0, r0, 1e-13);
    for (int j = 1; j < a.nu; ++j) {
        const double La = a.L0 + (j - 1) * a.dL, Lb = a.L0 + j * a.dL;
        const double atol = 1e-16 * Wc[j - 1];   // beyond the end of the spectrum the integrals vanish
        auto in_L = [&](auto f) {
            return integrate([&](double L) { const double u = SpectrumTable::u_of_L(L); return f(u) * u * (1 - u); },
                             La, Lb, 1e-13, 4000, atol);
        };
        Wc[j] = Wc[j - 1] + in_L(w);
        Gc[j] = Gc[j - 1] + in_L(g);
    }
}

// meta entries of a spectrum table file: kind, version and the u axis (nu, L0, dL exactly; u_min and
// 1 - u_max for reading)
std::map<std::string, std::string> spectrum_meta(const SpectrumTable& T) {
    return {{"table", "qed.spectrum"},
            {"version", std::to_string(TABLE_VERSION)},
            {"nu", std::to_string(T.nu)},
            {"L0", num(T.L0)},
            {"dL", num(T.dL)},
            {"u_min", num(SpectrumTable::u_of_L(T.L0))},
            {"one_minus_u_max", num(1 / (1 + std::exp(T.L0 + (T.nu - 1) * T.dL)))}};
}

std::string meta_get(const std::map<std::string, std::string>& m, const char* k) {
    auto it = m.find(k);
    return it == m.end() ? std::string() : it->second;
}

} // namespace

SpectrumTable make_spectrum_table(const SpectrumTableParams& p) {
    if (!(p.u_min > 0 && p.u_min < 0.5 && p.one_minus_u_min > 0 && p.one_minus_u_min < 0.5 && p.u_points_per_decade >= 1))
        throw std::runtime_error("qed spectrum table: invalid u axis parameters");
    const UAxis a = u_axis(p);
    SpectrumTable T;
    T.nu = a.nu;
    T.L0 = a.L0;
    T.dL = a.dL;
    std::vector<int> modes(2 * a.nu, LogTable::Log);   // Wc > 0, Gc < 0
    T.t = LogTable::build(p.chi_min, p.chi_max, p.chi_points_per_decade, 2 * a.nu,
                          [&](double chi, double* out) { cumulative_spectrum(chi, a, out, out + a.nu); }, modes,
                          "qed.spectrum_table");
    return T;
}

void save_spectrum_table(const SpectrumTable& T, const std::string& path) {
    T.t.save(path, {}, spectrum_meta(T));
}

SpectrumTable load_spectrum_table(const std::string& path) {
    std::map<std::string, std::string> m;
    const LogTable t = LogTable::load(path, nullptr, &m);
    if (meta_get(m, "table") != "qed.spectrum")
        throw std::runtime_error("load_spectrum_table: " + path + ": not a qed.spectrum table");
    if (meta_get(m, "version") != std::to_string(TABLE_VERSION))
        throw std::runtime_error("load_spectrum_table: " + path + ": table version " + meta_get(m, "version") +
                                 ", expected " + std::to_string(TABLE_VERSION));
    SpectrumTable T;
    T.t = t;
    T.nu = std::atoi(meta_get(m, "nu").c_str());
    T.L0 = std::strtod(meta_get(m, "L0").c_str(), nullptr);
    T.dL = std::strtod(meta_get(m, "dL").c_str(), nullptr);
    if (T.nu < 2 || t.ncomp() != 2 * T.nu || !(T.dL > 0))
        throw std::runtime_error("load_spectrum_table: " + path + ": inconsistent u axis");
    return T;
}

SpectrumTable spectrum_table(const std::string& path, const SpectrumTableParams& p, bool write, std::string* status) {
    std::string why = "no file";
    if (!path.empty()) {
        if (FILE* fp = std::fopen(path.c_str(), "r")) {
            std::fclose(fp);
            try {
                SpectrumTable T = load_spectrum_table(path);
                const UAxis a = u_axis(p);
                const int nchi = std::max(4, static_cast<int>(std::ceil(std::log10(p.chi_max / p.chi_min) * p.chi_points_per_decade)) + 1);
                if (T.t.axis_min() == p.chi_min && T.t.axis_max() == p.chi_max && T.t.size() == nchi && T.nu == a.nu &&
                    T.L0 == a.L0 && T.dL == a.dL) {
                    if (status) *status = "loaded";
                    return T;
                }
                why = "different range or points per decade";
            } catch (const std::exception& e) {
                why = e.what();
            }
        }
    }
    SpectrumTable T = make_spectrum_table(p);
    std::string st = "built";
    if (!path.empty() && write) {
        save_spectrum_table(T, path);
        st += why == "no file" ? " (saved)" : " (" + why + "; saved)";
    } else if (!path.empty()) {
        st += " (" + why + ")";
    }
    if (status) *status = st;
    return T;
}

} // namespace qed
} // namespace quarz
