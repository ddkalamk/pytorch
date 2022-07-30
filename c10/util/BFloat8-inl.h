#pragma once

#include <c10/macros/Macros.h>
#include <limits>

C10_CLANG_DIAGNOSTIC_PUSH()
#if C10_CLANG_HAS_WARNING("-Wimplicit-int-float-conversion")
C10_CLANG_DIAGNOSTIC_IGNORE("-Wimplicit-int-float-conversion")
#endif

namespace c10 {

/// Constructors
inline C10_HOST_DEVICE BFloat8::BFloat8(float value) {
#if defined(CUDA_VERSION) && CUDA_VERSION >= 11000 && \
    defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  x = __bfloat8_as_ushort(__float2bfloat8(value));
#else
  // RNE by default
  x = detail::round_to_nearest_even_uint8(value);
#endif
}

/// Implicit conversions
inline C10_HOST_DEVICE BFloat8::operator float() const {
#if defined(CUDA_VERSION) && CUDA_VERSION >= 11000
  return __bfloat82float(*reinterpret_cast<const __nv_bfloat8*>(&x));
#else
  return detail::f32_from_bits(x);
#endif
}

#if defined(CUDA_VERSION) && CUDA_VERSION >= 11000
inline C10_HOST_DEVICE BFloat8::BFloat8(const __nv_bfloat8& value) {
  x = *reinterpret_cast<const unsigned short*>(&value);
}
inline C10_HOST_DEVICE BFloat8::operator __nv_bfloat8() const {
  return *reinterpret_cast<const __nv_bfloat8*>(&x);
}
#endif

// CUDA intrinsics

#if defined(__CUDACC__) || defined(__HIPCC__)
inline C10_DEVICE BFloat8 __ldg(const BFloat8* ptr) {
#if defined(CUDA_VERSION) && CUDA_VERSION >= 11000 && \
    defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  return __ldg(reinterpret_cast<const __nv_bfloat8*>(ptr));
#else
  return *ptr;
#endif
}
#endif

/// Arithmetic

inline C10_HOST_DEVICE BFloat8
operator+(const BFloat8& a, const BFloat8& b) {
  return static_cast<float>(a) + static_cast<float>(b);
}

inline C10_HOST_DEVICE BFloat8
operator-(const BFloat8& a, const BFloat8& b) {
  return static_cast<float>(a) - static_cast<float>(b);
}

inline C10_HOST_DEVICE BFloat8
operator*(const BFloat8& a, const BFloat8& b) {
  return static_cast<float>(a) * static_cast<float>(b);
}

inline C10_HOST_DEVICE BFloat8 operator/(const BFloat8& a, const BFloat8& b)
    __ubsan_ignore_float_divide_by_zero__ {
  return static_cast<float>(a) / static_cast<float>(b);
}

inline C10_HOST_DEVICE BFloat8 operator-(const BFloat8& a) {
  return -static_cast<float>(a);
}

inline C10_HOST_DEVICE BFloat8& operator+=(BFloat8& a, const BFloat8& b) {
  a = a + b;
  return a;
}

inline C10_HOST_DEVICE BFloat8& operator-=(BFloat8& a, const BFloat8& b) {
  a = a - b;
  return a;
}

inline C10_HOST_DEVICE BFloat8& operator*=(BFloat8& a, const BFloat8& b) {
  a = a * b;
  return a;
}

inline C10_HOST_DEVICE BFloat8& operator/=(BFloat8& a, const BFloat8& b) {
  a = a / b;
  return a;
}

inline C10_HOST_DEVICE BFloat8& operator|(BFloat8& a, const BFloat8& b) {
  a.x = a.x | b.x;
  return a;
}

inline C10_HOST_DEVICE BFloat8& operator^(BFloat8& a, const BFloat8& b) {
  a.x = a.x ^ b.x;
  return a;
}

inline C10_HOST_DEVICE BFloat8& operator&(BFloat8& a, const BFloat8& b) {
  a.x = a.x & b.x;
  return a;
}

/// Arithmetic with floats

inline C10_HOST_DEVICE float operator+(BFloat8 a, float b) {
  return static_cast<float>(a) + b;
}
inline C10_HOST_DEVICE float operator-(BFloat8 a, float b) {
  return static_cast<float>(a) - b;
}
inline C10_HOST_DEVICE float operator*(BFloat8 a, float b) {
  return static_cast<float>(a) * b;
}
inline C10_HOST_DEVICE float operator/(BFloat8 a, float b) {
  return static_cast<float>(a) / b;
}

inline C10_HOST_DEVICE float operator+(float a, BFloat8 b) {
  return a + static_cast<float>(b);
}
inline C10_HOST_DEVICE float operator-(float a, BFloat8 b) {
  return a - static_cast<float>(b);
}
inline C10_HOST_DEVICE float operator*(float a, BFloat8 b) {
  return a * static_cast<float>(b);
}
inline C10_HOST_DEVICE float operator/(float a, BFloat8 b) {
  return a / static_cast<float>(b);
}

inline C10_HOST_DEVICE float& operator+=(float& a, const BFloat8& b) {
  return a += static_cast<float>(b);
}
inline C10_HOST_DEVICE float& operator-=(float& a, const BFloat8& b) {
  return a -= static_cast<float>(b);
}
inline C10_HOST_DEVICE float& operator*=(float& a, const BFloat8& b) {
  return a *= static_cast<float>(b);
}
inline C10_HOST_DEVICE float& operator/=(float& a, const BFloat8& b) {
  return a /= static_cast<float>(b);
}

/// Arithmetic with doubles

inline C10_HOST_DEVICE double operator+(BFloat8 a, double b) {
  return static_cast<double>(a) + b;
}
inline C10_HOST_DEVICE double operator-(BFloat8 a, double b) {
  return static_cast<double>(a) - b;
}
inline C10_HOST_DEVICE double operator*(BFloat8 a, double b) {
  return static_cast<double>(a) * b;
}
inline C10_HOST_DEVICE double operator/(BFloat8 a, double b) {
  return static_cast<double>(a) / b;
}

inline C10_HOST_DEVICE double operator+(double a, BFloat8 b) {
  return a + static_cast<double>(b);
}
inline C10_HOST_DEVICE double operator-(double a, BFloat8 b) {
  return a - static_cast<double>(b);
}
inline C10_HOST_DEVICE double operator*(double a, BFloat8 b) {
  return a * static_cast<double>(b);
}
inline C10_HOST_DEVICE double operator/(double a, BFloat8 b) {
  return a / static_cast<double>(b);
}

/// Arithmetic with ints

inline C10_HOST_DEVICE BFloat8 operator+(BFloat8 a, int b) {
  return a + static_cast<BFloat8>(b);
}
inline C10_HOST_DEVICE BFloat8 operator-(BFloat8 a, int b) {
  return a - static_cast<BFloat8>(b);
}
inline C10_HOST_DEVICE BFloat8 operator*(BFloat8 a, int b) {
  return a * static_cast<BFloat8>(b);
}
inline C10_HOST_DEVICE BFloat8 operator/(BFloat8 a, int b) {
  return a / static_cast<BFloat8>(b);
}

inline C10_HOST_DEVICE BFloat8 operator+(int a, BFloat8 b) {
  return static_cast<BFloat8>(a) + b;
}
inline C10_HOST_DEVICE BFloat8 operator-(int a, BFloat8 b) {
  return static_cast<BFloat8>(a) - b;
}
inline C10_HOST_DEVICE BFloat8 operator*(int a, BFloat8 b) {
  return static_cast<BFloat8>(a) * b;
}
inline C10_HOST_DEVICE BFloat8 operator/(int a, BFloat8 b) {
  return static_cast<BFloat8>(a) / b;
}

//// Arithmetic with int64_t

inline C10_HOST_DEVICE BFloat8 operator+(BFloat8 a, int64_t b) {
  return a + static_cast<BFloat8>(b);
}
inline C10_HOST_DEVICE BFloat8 operator-(BFloat8 a, int64_t b) {
  return a - static_cast<BFloat8>(b);
}
inline C10_HOST_DEVICE BFloat8 operator*(BFloat8 a, int64_t b) {
  return a * static_cast<BFloat8>(b);
}
inline C10_HOST_DEVICE BFloat8 operator/(BFloat8 a, int64_t b) {
  return a / static_cast<BFloat8>(b);
}

inline C10_HOST_DEVICE BFloat8 operator+(int64_t a, BFloat8 b) {
  return static_cast<BFloat8>(a) + b;
}
inline C10_HOST_DEVICE BFloat8 operator-(int64_t a, BFloat8 b) {
  return static_cast<BFloat8>(a) - b;
}
inline C10_HOST_DEVICE BFloat8 operator*(int64_t a, BFloat8 b) {
  return static_cast<BFloat8>(a) * b;
}
inline C10_HOST_DEVICE BFloat8 operator/(int64_t a, BFloat8 b) {
  return static_cast<BFloat8>(a) / b;
}

// Overloading < and > operators, because std::max and std::min use them.

inline C10_HOST_DEVICE bool operator>(BFloat8& lhs, BFloat8& rhs) {
  return float(lhs) > float(rhs);
}

inline C10_HOST_DEVICE bool operator<(BFloat8& lhs, BFloat8& rhs) {
  return float(lhs) < float(rhs);
}

} // namespace c10

namespace std {

template <>
class numeric_limits<c10::BFloat8> {
 public:
  static constexpr bool is_signed = true;
  static constexpr bool is_specialized = true;
  static constexpr bool is_integer = false;
  static constexpr bool is_exact = false;
  static constexpr bool has_infinity = true;
  static constexpr bool has_quiet_NaN = true;
  static constexpr bool has_signaling_NaN = true;
  static constexpr auto has_denorm = numeric_limits<float>::has_denorm;
  static constexpr auto has_denorm_loss =
      numeric_limits<float>::has_denorm_loss;
  static constexpr auto round_style = numeric_limits<float>::round_style;
  static constexpr bool is_iec559 = false;
  static constexpr bool is_bounded = true;
  static constexpr bool is_modulo = false;
  static constexpr int digits = 3;
  static constexpr int digits10 = 1;
  static constexpr int max_digits10 = 5;
  static constexpr int radix = 2;
  static constexpr int min_exponent = -13;
  static constexpr int min_exponent10 = -4;
  static constexpr int max_exponent = 16;
  static constexpr int max_exponent10 = 4;
  static constexpr auto traps = numeric_limits<float>::traps;
  static constexpr auto tinyness_before =
      numeric_limits<float>::tinyness_before;

  static constexpr c10::BFloat8 min() {
    return c10::BFloat8(0x04, c10::BFloat8::from_bits());
  }
  static constexpr c10::BFloat8 lowest() {
    return c10::BFloat8(0xFB, c10::BFloat8::from_bits());
  }
  static constexpr c10::BFloat8 max() {
    return c10::BFloat8(0x7B, c10::BFloat8::from_bits());
  }
  static constexpr c10::BFloat8 epsilon() {
    return c10::BFloat8(0x14, c10::BFloat8::from_bits());
  }
  static constexpr c10::BFloat8 round_error() {
    return c10::BFloat8(0x38, c10::BFloat8::from_bits());
  }
  static constexpr c10::BFloat8 infinity() {
    return c10::BFloat8(0x7C, c10::BFloat8::from_bits());
  }
  static constexpr c10::BFloat8 quiet_NaN() {
    return c10::BFloat8(0x7E, c10::BFloat8::from_bits());
  }
  static constexpr c10::BFloat8 signaling_NaN() {
    return c10::BFloat8(0x7D, c10::BFloat8::from_bits());
  }
  static constexpr c10::BFloat8 denorm_min() {
    return c10::BFloat8(0x01, c10::BFloat8::from_bits());
  }
};

} // namespace std

C10_CLANG_DIAGNOSTIC_POP()
