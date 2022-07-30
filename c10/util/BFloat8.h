#pragma once
#include <c10/macros/Macros.h>
#include <cmath>
#include <cstring>

#include <c10/util/Half.h>

namespace c10 {

namespace detail {
inline C10_HOST_DEVICE float f32_from_bits(uint8_t src) {
  uint16_t tmp = src;
  tmp <<= 8;
  return fp16_ieee_to_fp32_value(tmp);
}

inline C10_HOST_DEVICE uint8_t bits_from_f32_to_uint8(float src) {
  uint16_t res = fp16_ieee_from_fp32_value(src);

  return res >> 8;
}

inline C10_HOST_DEVICE uint8_t round_to_nearest_even_uint8(float src) {
  if (std::isnan(src)) {
    return UINT8_C(0x7C);
  } else {
    union {
      uint16_t U16;
      Half F16;
    };

    F16 = Half(src);
    uint16_t rounding_bias = ((U16 >> 8) & 1) + UINT16_C(0x007F);
    return static_cast<uint8_t>((U16 + rounding_bias) >> 8);
  }
}
} // namespace detail


/**
 * quint8 is for unsigned 8 bit quantized Tensors
 */
struct alignas(1) BFloat8 {
  uint8_t x;
  BFloat8() = default;
  struct from_bits_t {};
  static constexpr C10_HOST_DEVICE from_bits_t from_bits() {
    return from_bits_t();
  }

  constexpr C10_HOST_DEVICE BFloat8(uint8_t bits, from_bits_t)
      : x(bits){};

  inline C10_HOST_DEVICE BFloat8(float value); // { x = detail::round_to_nearest_even_uint8(value); }
  inline C10_HOST_DEVICE operator float() const; // { return detail::f32_from_bits(x); }
};

} // namespace c10

#include <c10/util/BFloat8-inl.h> // IWYU pragma: keep
