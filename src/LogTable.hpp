// Look-up table on a logarithmic axis: nc columns f_c(x) tabulated at x_i = x_min (x_max/x_min)^{i/(n-1)},
// interpolated in t = ln x with 4-point (cubic) Lagrange stencils, i.e. fourth order in the spacing
// h = ln(x_max/x_min)/(n-1). Each column is stored either linearly (f) or logarithmically (ln|f|, with
// its constant sign): the logarithmic form is exact for power laws and smooth for exponentials.
// Outside [x_min, x_max] the end interval is continued linearly in t (power-law extrapolation for
// logarithmic columns); callers that need the exact value there should test in_range().
// Device-copyable (Kokkos views); built on the host from any callable f(x, double* out).
//
// save() / load(): plain text ("*.lut" by convention), exact round trip (17 significant digits; the
// loaded table is bit-identical to the saved one). Format:
//   # QUARZ LogTable 1
//   # name <name>
//   # points <n>   columns <nc>
//   # x_min <x_min>   x_max <x_max>
//   # modes <lin|log> ...            (interpolation of each column: f or ln|f|)
//   # labels <label> ...             (optional)
//   # meta <key> <value>             (any number of lines; used by the QED tables to identify themselves)
//   # x f_0 f_1 ...
//   <n lines: x_i and the function values f_c(x_i)>
// The x column is for reading only; load() checks it against the grid given by x_min, x_max, n.
#pragma once

#include "Types.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace quarz {

class LogTable {
public:
    enum Mode : int { Linear = 0, Log = 1 };

    LogTable() = default;

    // x_min > 0, x_max > x_min, points_per_decade >= 1; modes[c] = Linear or Log for column c
    // (Log columns must not change sign or vanish on the grid). f(x, out) fills out[0 .. nc-1].
    template <class F>
    static LogTable build(double x_min, double x_max, double points_per_decade, int nc, F f,
                          const std::vector<int>& modes, const std::string& name = "LogTable") {
        if (!(x_min > 0) || !(x_max > x_min) || !(points_per_decade >= 1) || nc < 1)
            throw std::runtime_error(name + ": invalid table parameters");
        const int n = std::max(4, static_cast<int>(std::ceil(std::log10(x_max / x_min) * points_per_decade)) + 1);
        HostValues V("LogTable.values", n, nc);
        std::vector<double> out(nc);
        for (int i = 0; i < n; ++i) {
            f(grid_x(x_min, x_max, n, i), out.data());
            for (int c = 0; c < nc; ++c) V(i, c) = out[c];
        }
        return from_values(x_min, x_max, V, modes, name);
    }

    // node i of the grid with n points (the end points exactly x_min, x_max)
    static double grid_x(double x_min, double x_max, int n, int i) {
        if (i == 0) return x_min;
        if (i == n - 1) return x_max;
        const double t0 = std::log(x_min), dt = (std::log(x_max) - t0) / (n - 1);
        return std::exp(t0 + i * dt);
    }

    // ---- files (host)
    void save(const std::string& path, const std::vector<std::string>& labels = {},
              const std::map<std::string, std::string>& meta = {}) const {
        if (empty()) throw std::runtime_error("LogTable::save: empty table");
        FILE* fp = std::fopen(path.c_str(), "w");
        if (!fp) throw std::runtime_error("LogTable::save: cannot write " + path);
        std::fprintf(fp, "# QUARZ LogTable 1\n# name %s\n# points %d columns %d\n# x_min %.17g x_max %.17g\n# modes",
                     name_ok(name_), n_, nc_, xmin_, xmax_);
        for (int c = 0; c < nc_; ++c) std::fprintf(fp, " %s", mode_h_(c) == Log ? "log" : "lin");
        std::fprintf(fp, "\n");
        if (!labels.empty()) {
            if (static_cast<int>(labels.size()) != nc_) { std::fclose(fp); throw std::runtime_error("LogTable::save: wrong number of labels"); }
            std::fprintf(fp, "# labels");
            for (const auto& l : labels) std::fprintf(fp, " %s", name_ok(l));
            std::fprintf(fp, "\n");
        }
        for (const auto& kv : meta) {
            if (kv.second.find_first_of("\n\r") != std::string::npos) { std::fclose(fp); throw std::runtime_error("LogTable::save: line break in meta value"); }
            std::fprintf(fp, "# meta %s %s\n", name_ok(kv.first), kv.second.c_str());
        }
        std::fprintf(fp, "# x");
        for (int c = 0; c < nc_; ++c) std::fprintf(fp, " %s", labels.empty() ? ("f" + std::to_string(c)).c_str() : labels[c].c_str());
        std::fprintf(fp, "\n");
        char buf[32];
        std::string line;
        for (int i = 0; i < n_; ++i) {
            line.clear();
            std::snprintf(buf, sizeof buf, "%.17g", grid_x(xmin_, xmax_, n_, i));
            line += buf;
            for (int c = 0; c < nc_; ++c) {
                std::snprintf(buf, sizeof buf, " %.17g", V_(i, c));
                line += buf;
            }
            line += '\n';
            std::fputs(line.c_str(), fp);
        }
        const bool ok = std::fflush(fp) == 0;
        std::fclose(fp);
        if (!ok) throw std::runtime_error("LogTable::save: write error on " + path);
    }

    // labels / meta (optional) receive the file's labels and meta entries
    static LogTable load(const std::string& path, std::vector<std::string>* labels = nullptr,
                         std::map<std::string, std::string>* meta = nullptr) {
        FILE* fp = std::fopen(path.c_str(), "r");
        if (!fp) throw std::runtime_error("LogTable::load: cannot open " + path);
        auto fail = [&](const std::string& why) {
            std::fclose(fp);
            throw std::runtime_error("LogTable::load: " + path + ": " + why);
        };
        std::string name = "LogTable";
        int n = -1, nc = -1;
        double x_min = 0, x_max = 0;
        std::vector<int> modes;
        std::vector<std::string> lab;
        std::map<std::string, std::string> mt;
        std::vector<char> buf(1 << 16);
        auto getline = [&](std::string& l) {
            l.clear();
            while (std::fgets(buf.data(), static_cast<int>(buf.size()), fp)) {
                l += buf.data();
                if (!l.empty() && l.back() == '\n') { l.pop_back(); return true; }
            }
            return !l.empty();
        };
        auto words = [](const std::string& l) {
            std::vector<std::string> w;
            size_t i = 0;
            while (i < l.size()) {
                while (i < l.size() && (l[i] == ' ' || l[i] == '\t' || l[i] == '\r')) ++i;
                size_t j = i;
                while (j < l.size() && l[j] != ' ' && l[j] != '\t' && l[j] != '\r') ++j;
                if (j > i) w.push_back(l.substr(i, j - i));
                i = j;
            }
            return w;
        };
        std::string l;
        if (!getline(l) || l.rfind("# QUARZ LogTable 1", 0) != 0) fail("not a QUARZ LogTable (version 1) file");
        HostValues V;
        int row = 0;
        while (getline(l)) {
            if (l.empty()) continue;
            if (l[0] == '#') {
                const auto w = words(l.substr(1));
                if (w.empty()) continue;
                if (w[0] == "name" && w.size() >= 2) name = w[1];
                else if (w[0] == "points" && w.size() >= 4) { n = std::atoi(w[1].c_str()); nc = std::atoi(w[3].c_str()); }
                else if (w[0] == "x_min" && w.size() >= 4) { x_min = std::strtod(w[1].c_str(), nullptr); x_max = std::strtod(w[3].c_str(), nullptr); }
                else if (w[0] == "modes") {
                    for (size_t k = 1; k < w.size(); ++k) {
                        if (w[k] != "lin" && w[k] != "log") fail("unknown mode '" + w[k] + "'");
                        modes.push_back(w[k] == "log" ? Log : Linear);
                    }
                } else if (w[0] == "labels") lab.assign(w.begin() + 1, w.end());
                else if (w[0] == "meta" && w.size() >= 2) {
                    const size_t p = l.find(w[1]) + w[1].size();
                    const size_t q = l.find_first_not_of(" \t", p);
                    mt[w[1]] = q == std::string::npos ? "" : l.substr(q);
                }
                continue;
            }
            if (n < 4 || nc < 1 || static_cast<int>(modes.size()) != nc || !(x_min > 0) || !(x_max > x_min))
                fail("incomplete or invalid header");
            if (V.extent(0) == 0) V = HostValues("LogTable.values", n, nc);
            if (row >= n) fail("more than " + std::to_string(n) + " data lines");
            const char* p = l.c_str();
            char* e = nullptr;
            const double x = std::strtod(p, &e);
            if (e == p) fail("bad number in data line " + std::to_string(row + 1));
            const double xg = grid_x(x_min, x_max, n, row);
            if (!(std::abs(x - xg) <= 1e-12 * xg)) fail("x of data line " + std::to_string(row + 1) + " is not on the grid");
            for (int c = 0; c < nc; ++c) {
                p = e;
                V(row, c) = std::strtod(p, &e);
                if (e == p) fail("too few values in data line " + std::to_string(row + 1));
            }
            ++row;
        }
        std::fclose(fp);
        if (row != n) throw std::runtime_error("LogTable::load: " + path + ": " + std::to_string(row) + " data lines, expected " + std::to_string(n));
        if (labels) *labels = lab;
        if (meta) *meta = mt;
        return from_values(x_min, x_max, V, modes, name);
    }

    KOKKOS_INLINE_FUNCTION Real x_min() const { return Kokkos::exp(t0_); }
    KOKKOS_INLINE_FUNCTION Real x_max() const { return Kokkos::exp(t1_); }
    KOKKOS_INLINE_FUNCTION bool in_range(Real x) const {
        const Real t = Kokkos::log(x);
        return t >= t0_ && t <= t1_;
    }
    int size() const { return n_; }
    int ncomp() const { return nc_; }
    bool empty() const { return n_ == 0; }
    std::string name() const { return name_; }
    // function value f_c(x_i) at node i as built or loaded (host)
    double value(int i, int c) const { return V_(i, c); }
    int mode(int c) const { return mode_h_(c); }
    double axis_min() const { return xmin_; }   // x_min, x_max exactly (host)
    double axis_max() const { return xmax_; }

    // stored (transformed) values at x: ln|f_c| for Log columns, f_c for Linear ones
    KOKKOS_INLINE_FUNCTION void eval_raw(Real x, Real* out) const {
        int j;
        Real w[4];
        stencil(Kokkos::log(x), j, w);
        for (int c = 0; c < nc_; ++c)
            out[c] = w[0] * F_(j, c) + w[1] * F_(j + 1, c) + w[2] * F_(j + 2, c) + w[3] * F_(j + 3, c);
    }
    // all columns at x
    KOKKOS_INLINE_FUNCTION void eval(Real x, Real* out) const {
        eval_raw(x, out);
        for (int c = 0; c < nc_; ++c)
            if (sign_(c) != Real(0)) out[c] = sign_(c) * Kokkos::exp(out[c]);
    }
    // one column at x
    KOKKOS_INLINE_FUNCTION Real eval(int c, Real x) const { return eval(stencil_at(x), c); }

    // For many columns at the same x: compute the interpolation stencil once, then evaluate columns.
    struct Stencil {
        int j;        // first of the 4 nodes
        Real w[4];    // their weights
    };
    KOKKOS_INLINE_FUNCTION Stencil stencil_at(Real x) const {
        Stencil s;
        stencil(Kokkos::log(x), s.j, s.w);
        return s;
    }
    KOKKOS_INLINE_FUNCTION Real eval(const Stencil& s, int c) const {
        const Real v = s.w[0] * F_(s.j, c) + s.w[1] * F_(s.j + 1, c) + s.w[2] * F_(s.j + 2, c) + s.w[3] * F_(s.j + 3, c);
        return sign_(c) != Real(0) ? sign_(c) * Kokkos::exp(v) : v;
    }

private:
    using HostValues = Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::HostSpace>;

    // the table from the function values V(i, c) at the n = V.extent(0) grid points
    static LogTable from_values(double x_min, double x_max, const HostValues& V, const std::vector<int>& modes,
                                const std::string& name) {
        const int n = static_cast<int>(V.extent(0)), nc = static_cast<int>(V.extent(1));
        if (n < 4 || nc < 1 || static_cast<int>(modes.size()) != nc || !(x_min > 0) || !(x_max > x_min))
            throw std::runtime_error(name + ": invalid table parameters");
        LogTable T;
        T.nc_ = nc;
        T.n_ = n;
        T.xmin_ = x_min;
        T.xmax_ = x_max;
        T.t0_ = std::log(x_min);
        T.t1_ = std::log(x_max);
        T.idt_ = (n - 1) / (T.t1_ - T.t0_);
        T.F_ = View2D(name + ".F", n, nc);
        T.sign_ = View1D(name + ".sign", nc);
        T.V_ = V;
        T.mode_h_ = Kokkos::View<int*, Kokkos::HostSpace>("LogTable.modes", nc);
        std::snprintf(T.name_, sizeof T.name_, "%s", name.c_str());
        auto hF = Kokkos::create_mirror_view(T.F_);
        auto hs = Kokkos::create_mirror_view(T.sign_);
        for (int c = 0; c < nc; ++c) {
            if (modes[c] != Linear && modes[c] != Log) throw std::runtime_error(name + ": unknown column mode");
            T.mode_h_(c) = modes[c];
            hs(c) = 0;
            for (int i = 0; i < n; ++i) {
                const double v = V(i, c);
                if (!std::isfinite(v)) throw std::runtime_error(name + ": non-finite value in column " + std::to_string(c));
                if (modes[c] == Log) {
                    const double sg = v > 0 ? 1.0 : (v < 0 ? -1.0 : 0.0);
                    if (i == 0) hs(c) = sg;
                    if (sg == 0 || sg != hs(c))
                        throw std::runtime_error(name + ": logarithmic column " + std::to_string(c) + " vanishes or changes sign");
                    hF(i, c) = std::log(std::abs(v));
                } else {
                    hF(i, c) = v;
                }
            }
        }
        Kokkos::deep_copy(T.F_, hF);
        Kokkos::deep_copy(T.sign_, hs);
        return T;
    }
    // names and labels are single words in the file
    static const char* name_ok(const std::string& s) {
        if (s.empty() || s.find_first_of(" \t\n\r") != std::string::npos)
            throw std::runtime_error("LogTable::save: names, labels and meta keys must be single words: '" + s + "'");
        return s.c_str();
    }

    // first node j of the 4-point window and the weights at t. Inside: the window around the interval
    // [i, i+1] containing t (shifted inwards at the ends), cubic Lagrange weights at p = s - j.
    // Outside: linear continuation of the end interval (weights on two nodes).
    KOKKOS_INLINE_FUNCTION void stencil(Real t, int& j, Real* w) const {
        const Real s = (t - t0_) * idt_;
        if (!(s >= Real(0))) {   // below x_min (or NaN)
            j = 0;
            w[0] = Real(1) - s; w[1] = s; w[2] = w[3] = Real(0);
            return;
        }
        if (s > Real(n_ - 1)) {
            j = n_ - 4;
            const Real q = s - Real(n_ - 1);
            w[0] = w[1] = Real(0); w[2] = -q; w[3] = Real(1) + q;
            return;
        }
        int i = static_cast<int>(s);
        j = i - 1;
        j = j < 0 ? 0 : (j > n_ - 4 ? n_ - 4 : j);
        const Real p = s - Real(j), p1 = p - Real(1), p2 = p - Real(2), p3 = p - Real(3);
        w[0] = -p1 * p2 * p3 / Real(6);
        w[1] = p * p2 * p3 / Real(2);
        w[2] = -p * p1 * p3 / Real(2);
        w[3] = p * p1 * p2 / Real(6);
    }

    View2D F_;        // (n, nc) stored values
    View1D sign_;     // per column: 0 = Linear, +-1 = Log with that sign
    Real t0_ = 0, t1_ = 0, idt_ = 1;
    int n_ = 0, nc_ = 0;
    // host side: the function values as built or loaded (saved exactly), modes, axis ends, name
    HostValues V_;
    Kokkos::View<int*, Kokkos::HostSpace> mode_h_;
    double xmin_ = 0, xmax_ = 0;
    char name_[64] = "LogTable";   // not std::string: the class is copied into device kernels
};

} // namespace quarz
