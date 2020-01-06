// Implements the math functions for CPU.
// The implementation in this file allows us to route the underlying numerical
// computation library to different compiler options (-mno-avx2 or -mavx2).

#include <immintrin.h>
#include <cmath>
#include <cstdint>

namespace caffe2 {

namespace memory {

static inline void float_memory_copy_block64(
    float* input_data,
    float* output_data) {
  __m256 tmp_values0 = _mm256_loadu_ps(&input_data[0]);
  __m256 tmp_values1 = _mm256_loadu_ps(&input_data[8]);
  __m256 tmp_values2 = _mm256_loadu_ps(&input_data[16]);
  __m256 tmp_values3 = _mm256_loadu_ps(&input_data[24]);
  __m256 tmp_values4 = _mm256_loadu_ps(&input_data[32]);
  __m256 tmp_values5 = _mm256_loadu_ps(&input_data[40]);
  __m256 tmp_values6 = _mm256_loadu_ps(&input_data[48]);
  __m256 tmp_values7 = _mm256_loadu_ps(&input_data[56]);
  _mm256_storeu_ps(&output_data[0], tmp_values0);
  _mm256_storeu_ps(&output_data[8], tmp_values1);
  _mm256_storeu_ps(&output_data[16], tmp_values2);
  _mm256_storeu_ps(&output_data[24], tmp_values3);
  _mm256_storeu_ps(&output_data[32], tmp_values4);
  _mm256_storeu_ps(&output_data[40], tmp_values5);
  _mm256_storeu_ps(&output_data[48], tmp_values6);
  _mm256_storeu_ps(&output_data[56], tmp_values7);
}

void float_memory_region_select_copy__avx2(
    int64_t one_region_size,
    int64_t select_start,
    int64_t select_end,
    float* input_data,
    float* output_data)
{
  int64_t aligned_size = (one_region_size >> 6) << 6;
  int n = one_region_size - aligned_size;
  for(auto s = select_start; s < select_end; s++) {
    auto output_region = output_data + one_region_size * s;
    for (auto d = 0; d < aligned_size; d += 64) {
      float_memory_copy_block64((float*)(input_data + d), (float*)(output_region + d));
    }
    if (n > 0) {
      for (int64_t dn = aligned_size; dn < one_region_size; dn++) {
        output_region[dn] = input_data[dn];
      }
    }
  }
}

} // namespace math
} // namespace caffe2
