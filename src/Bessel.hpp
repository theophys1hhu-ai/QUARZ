// Modified Bessel functions of the second kind of orders 1/3 and 2/3, and the integral
//   IntK_{1/3}(x) = int_x^inf K_{1/3}(t) dt,
// as they enter the strong-field QED rates in the locally-constant-field approximation
// (nonlinear Compton / synchrotron emission, Breit-Wheeler pair creation).
// Device functions (Kokkos), any floating-point type T (the coefficients are double).
//
// Small x: ascending series of the Airy functions, with z = (3x/2)^{2/3},
//   K_{1/3}(x) = pi sqrt(3/z) Ai(z),   K_{2/3}(x) = -pi sqrt(3) Ai'(z) / z,
//   IntK_{1/3}(x) = pi/sqrt(3) - pi sqrt(3) int_0^z Ai(s) ds,
//   Ai = c1 f - c2 g,  Ai' = c1 f' - c2 g',  c1 = Ai(0), c2 = -Ai'(0)  (Abramowitz & Stegun 10.4.2-3).
// Large x: K(x) = pi sqrt(3) e^{-x} x^{-1/2} R(x), with [7/7] rational approximations R(x) of
//   sqrt(x) e^x K(x) / (pi sqrt(3)) (R -> sqrt(pi/2) / (pi sqrt(3)) for x -> inf).
// The series are used for x <= 2.5 (K) and x <= 3 (IntK); the switch points and the rational
// coefficients are those of the SONIQ implementation (O. Mathiak).
//
// Accuracy in double (tests/test_bessel.cpp, against 40-digit references): relative error
//   K_{1/3}, K_{2/3} <= 3e-13 for all x (largest at the switch point);
//   IntK_{1/3} <= 5e-13 for x < 100, rising to 3e-12 at x ~ 700 (the rational approximation's
//   limit, 4e-12 for x -> inf). In float ~5e-5 near the switch points (Airy series cancel).
// Domain: x >= 0; K_{1/3}(0) = K_{2/3}(0) = +inf, IntK_{1/3}(0) = pi/sqrt(3).
#pragma once

#include <Kokkos_Core.hpp>

namespace quarz {
namespace bessel {

constexpr double AI0 = 0.35502805388781723926;          // Ai(0)
constexpr double AIP0 = 0.25881940379280679840;         // -Ai'(0)
constexpr double PI_SQRT3 = 5.4413980927026535518;      // pi sqrt(3)
constexpr double INV_PI_SQRT3 = 0.18377629847393068317;  // 1 / (pi sqrt(3))
constexpr double PI_OVER_SQRT3 = 1.8137993642342178506;  // pi / sqrt(3) = IntK_{1/3}(0)
constexpr double K_SWITCH = 2.5;    // series below, rational approximation above (K_{1/3}, K_{2/3})
constexpr double IK_SWITCH = 3.0;   // the same for IntK_{1/3}

namespace detail {

// Horner evaluation of c[0] x^N + c[1] x^{N-1} + ... + c[N]
template <class T, int N>
KOKKOS_INLINE_FUNCTION T horner(T x, const double (&c)[N + 1]) {
    T a = T(c[0]);
    for (int i = 1; i <= N; ++i) a = a * x + T(c[i]);
    return a;
}

// sum_k a_k with a_0 = a0, a_k = a_{k-1} * q * ratio(k), until |a_k| < eps |sum|
template <class T, class F>
KOKKOS_INLINE_FUNCTION T series(T a0, T q, F ratio) {
    const T eps = T(1e-16);
    T s = a0, a = a0;
    for (int k = 1; k < 40; ++k) {
        a *= q * ratio(k);
        s += a;
        if (Kokkos::fabs(a) <= eps * Kokkos::fabs(s)) break;
    }
    return s;
}

// rational parts R(x) = sqrt(x) e^x K(x) / (pi sqrt(3))  (x >= K_SWITCH, IK_SWITCH)
template <class T>
KOKKOS_INLINE_FUNCTION T r13(T x) {
    constexpr double n[8] = {0.0989661656685458577, 1.62995613416167718, 9.74966013031451196, 26.6449010920692951,
                             34.3755121519465788, 19.4781803818936145, 3.90702508233342435, 0.146561905058396895};
    constexpr double d[8] = {0.429672249819486846, 7.10646817200790996, 42.8067473158570568, 118.406830547692049,
                             156.123341266696941, 92.2782163609703901, 20.3042041738952909, 1.00000000000000000};
    return horner<T, 7>(x, n) / horner<T, 7>(x, d);
}
template <class T>
KOKKOS_INLINE_FUNCTION T r23(T x) {
    constexpr double n[8] = {0.161873154141704363, 2.65140015419094228, 15.8114068189474134, 43.3266946701903233,
                             56.7654565486395420, 33.6269901561447377, 7.57901141734622247, 0.404290798098008398};
    constexpr double d[8] = {0.702789704497446700, 11.4430120906251616, 67.5652497466353831, 182.010994488395593,
                             231.281058799004099, 129.257905007374597, 25.7897028824420341, 1.00000000000000000};
    return horner<T, 7>(x, n) / horner<T, 7>(x, d);
}
template <class T>
KOKKOS_INLINE_FUNCTION T ri13(T x) {
    constexpr double n[8] = {0.00287437794496329036, 0.0738540275842796942, 0.680022676534713602, 2.81186434652663117,
                             5.35607602167495162, 4.30755230707350558, 1.13259475038349653, 0.0410195965344244903};
    constexpr double d[8] = {0.0124794209223556174, 0.327751538989704101, 3.12790495522497312, 13.7253358799859016,
                             28.9249560798889103, 27.8595559232117210, 10.4590432272811964, 1.00000000000000000};
    return horner<T, 7>(x, n) / horner<T, 7>(x, d);
}

// series parts, all with z = (3x/2)^{2/3}, q = z^3 = (3x/2)^2; results divided by pi sqrt(3)
template <class T>
KOKKOS_INLINE_FUNCTION T k13_series(T sz, T z, T q) {   // Ai(z) / sqrt(z),  sz = sqrt(z)
    const T f = series(T(1), q, [](int k) { return T(1) / (T(3 * k) * T(3 * k - 1)); });
    const T g = series(z, q, [](int k) { return T(1) / (T(3 * k) * T(3 * k + 1)); });
    return (T(AI0) * f - T(AIP0) * g) / sz;
}
template <class T>
KOKKOS_INLINE_FUNCTION T k23_series(T z, T q) {         // -Ai'(z) / z
    const T fp = series(T(0.5) * z * z, q, [](int k) { return T(1) / (T(3 * k) * T(3 * k + 2)); });
    const T gp = series(T(1), q, [](int k) { return T(1) / (T(3 * k) * T(3 * k - 2)); });
    return -(T(AI0) * fp - T(AIP0) * gp) / z;
}
template <class T>
KOKKOS_INLINE_FUNCTION T ik13_series(T z, T q) {        // 1/3 - int_0^z Ai(s) ds
    const T F = series(z, q, [](int k) { return T(3 * k - 2) / (T(3 * k + 1) * T(3 * k) * T(3 * k - 1)); });
    const T G = series(T(0.5) * z * z, q, [](int k) { return T(3 * k - 1) / (T(3 * k + 2) * T(3 * k) * T(3 * k + 1)); });
    return T(1.0 / 3.0) - (T(AI0) * F - T(AIP0) * G);
}

} // namespace detail

// K_{1/3}(x)
template <class T>
KOKKOS_INLINE_FUNCTION T k13(T x) {
    if (x <= T(K_SWITCH)) {
        const T sz = Kokkos::cbrt(T(1.5) * x), z = sz * sz;
        return T(PI_SQRT3) * detail::k13_series(sz, z, T(2.25) * x * x);
    }
    return T(PI_SQRT3) * Kokkos::exp(-x) / Kokkos::sqrt(x) * detail::r13(x);
}

// K_{2/3}(x)
template <class T>
KOKKOS_INLINE_FUNCTION T k23(T x) {
    if (x <= T(K_SWITCH)) {
        const T sz = Kokkos::cbrt(T(1.5) * x);
        return T(PI_SQRT3) * detail::k23_series(sz * sz, T(2.25) * x * x);
    }
    return T(PI_SQRT3) * Kokkos::exp(-x) / Kokkos::sqrt(x) * detail::r23(x);
}

// IntK_{1/3}(x) = int_x^inf K_{1/3}(t) dt
template <class T>
KOKKOS_INLINE_FUNCTION T ik13(T x) {
    if (x <= T(0)) return T(PI_OVER_SQRT3);
    if (x <= T(IK_SWITCH)) {
        const T sz = Kokkos::cbrt(T(1.5) * x);
        return T(PI_SQRT3) * detail::ik13_series(sz * sz, T(2.25) * x * x);
    }
    return T(PI_SQRT3) * Kokkos::exp(-x) / Kokkos::sqrt(x) * detail::ri13(x);
}

// Exponentially scaled versions e^x K(x), e^x IntK(x): finite for large x where K underflows
// (x > ~700 in double)
template <class T>
KOKKOS_INLINE_FUNCTION T k13e(T x) {
    return x <= T(K_SWITCH) ? Kokkos::exp(x) * k13(x) : T(PI_SQRT3) / Kokkos::sqrt(x) * detail::r13(x);
}
template <class T>
KOKKOS_INLINE_FUNCTION T k23e(T x) {
    return x <= T(K_SWITCH) ? Kokkos::exp(x) * k23(x) : T(PI_SQRT3) / Kokkos::sqrt(x) * detail::r23(x);
}
template <class T>
KOKKOS_INLINE_FUNCTION T ik13e(T x) {
    return x <= T(IK_SWITCH) ? Kokkos::exp(x) * ik13(x) : T(PI_SQRT3) / Kokkos::sqrt(x) * detail::ri13(x);
}

// All three at once (the rates always need them together): shares z, q and the exponential
template <class T>
KOKKOS_INLINE_FUNCTION void synchrotron_functions(T x, T& K13, T& K23, T& IK13) {
    if (x > T(IK_SWITCH)) {
        const T e = T(PI_SQRT3) * Kokkos::exp(-x) / Kokkos::sqrt(x);
        K13 = e * detail::r13(x);
        K23 = e * detail::r23(x);
        IK13 = e * detail::ri13(x);
        return;
    }
    if (x <= T(0)) {
        K13 = K23 = Kokkos::Experimental::infinity<T>::value;
        IK13 = T(PI_OVER_SQRT3);
        return;
    }
    const T sz = Kokkos::cbrt(T(1.5) * x), z = sz * sz, q = T(2.25) * x * x;
    IK13 = T(PI_SQRT3) * detail::ik13_series(z, q);
    if (x <= T(K_SWITCH)) {
        K13 = T(PI_SQRT3) * detail::k13_series(sz, z, q);
        K23 = T(PI_SQRT3) * detail::k23_series(z, q);
    } else {
        const T e = T(PI_SQRT3) * Kokkos::exp(-x) / Kokkos::sqrt(x);
        K13 = e * detail::r13(x);
        K23 = e * detail::r23(x);
    }
}

} // namespace bessel
} // namespace quarz
