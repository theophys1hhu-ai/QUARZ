// Look-up table on a logarithmic axis: nc columns f_c(x) tabulated at x_i = x_min (x_max/x_min)^{i/(n-1)},
// interpolated in t = ln x with 4-point (cubic) Lagrange stencils, i.e. fourth order in the spacing
// h = ln(x_max/x_min)/(n-1). Each column is stored either linearly (f) or logarithmically (ln|f|, with
// its constant sign): the logarithmic form is exact for power laws and smooth for exponentials.
// Outside [x_min, x_max] the end interval is continued linearly in t (power-law extrapolation for
// logarithmic columns); callers that need the exact value there should test in_range().
// Device-copyable (Kokkos views); built on the host from any callable f(x, double* out).
#pragma once

#include "Types.hpp"

#include <cmath>
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
        if (!(x_min > 0) || !(x_max > x_min) || !(points_per_decade >= 1) || nc < 1 ||
            static_cast<int>(modes.size()) != nc)
            throw std::runtime_error(name + ": invalid table parameters");
        LogTable T;
        T.nc_ = nc;
        T.n_ = std::max(4, static_cast<int>(std::ceil(std::log10(x_max / x_min) * points_per_decade)) + 1);
        T.t0_ = std::log(x_min);
        T.t1_ = std::log(x_max);
        const double dt = (T.t1_ - T.t0_) / (T.n_ - 1);
        T.idt_ = 1 / dt;
        T.F_ = View2D(name + ".F", T.n_, nc);
        T.sign_ = View1D(name + ".sign", nc);
        auto hF = Kokkos::create_mirror_view(T.F_);
        auto hs = Kokkos::create_mirror_view(T.sign_);
        std::vector<double> out(nc);
        for (int i = 0; i < T.n_; ++i) {
            const double x = i == T.n_ - 1 ? x_max : std::exp(T.t0_ + i * dt);
            f(i == 0 ? x_min : x, out.data());
            for (int c = 0; c < nc; ++c) {
                const double v = out[c];
                if (!std::isfinite(v)) throw std::runtime_error(name + ": non-finite value in column " + std::to_string(c));
                if (modes[c] == Log) {
                    const double s = v > 0 ? 1.0 : (v < 0 ? -1.0 : 0.0);
                    if (i == 0) hs(c) = s;
                    if (s == 0 || s != hs(c))
                        throw std::runtime_error(name + ": logarithmic column " + std::to_string(c) + " vanishes or changes sign");
                    hF(i, c) = std::log(std::abs(v));
                } else {
                    if (i == 0) hs(c) = 0;
                    hF(i, c) = v;
                }
            }
        }
        Kokkos::deep_copy(T.F_, hF);
        Kokkos::deep_copy(T.sign_, hs);
        return T;
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
    KOKKOS_INLINE_FUNCTION Real eval(int c, Real x) const {
        int j;
        Real w[4];
        stencil(Kokkos::log(x), j, w);
        const Real v = w[0] * F_(j, c) + w[1] * F_(j + 1, c) + w[2] * F_(j + 2, c) + w[3] * F_(j + 3, c);
        return sign_(c) != Real(0) ? sign_(c) * Kokkos::exp(v) : v;
    }

private:
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
};

} // namespace quarz
