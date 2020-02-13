#pragma once

#include <cstdint>

namespace caffe2 {

namespace memory {

void float_memory_region_select_copy(
    int64_t one_region_size,
    int64_t select_start,
    int64_t select_end,
    float* input_data,
    float* output_data);

} // namespace memory
} // namespace caffe2
