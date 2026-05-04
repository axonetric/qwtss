#pragma once
#include <cstddef>

// The Global Configuration
namespace QwtssConfig {
    // The maximum number of distinct tiles in any possible alphabet utilized by this application.
    inline constexpr size_t MAX_TILES = 64;

}

// The QWTSS reference constraint values
namespace QwtssReference {
    inline constexpr uint8_t version = 1;
    inline constexpr int grid_size = 64;
    // 0.0f to 1.0f percentage of the border constraints to keep (border retention).
    inline constexpr float keep_boundary_percentage = 0.70f;
    // Tunable frustration: this must be an even number
    inline constexpr int boundary_num_splice_segments = 14;

    inline constexpr int total_defects_lower_bound = 160;
    inline constexpr int total_defects_upper_bound = 170;

    inline constexpr int alien_tiles_lower_bound = 1500;
    inline constexpr int core32_lower_bound = 20;
    inline constexpr int line_max_upper_bound = 12;

    inline constexpr bool do_greedy_prequench = true;

    inline static bool is_valid_defect_count(int defect_count){
        return (defect_count <= total_defects_upper_bound && defect_count >= total_defects_lower_bound);
    }
}

namespace QwtssZkPoWTestConstraints {
    inline constexpr int total_defects_upper_bound = 190; // 170; // 130; // 110;
    inline constexpr int alien_tiles_lower_bound = 1500;
    inline constexpr int core48_lower_bound = -1; // 60;
    inline constexpr int core32_lower_bound = 20;
    inline constexpr int frame6_upper_bound = -1; // 135;
    inline constexpr int line_max_upper_bound = 12;
    inline constexpr int block8_max_upper_bound = -1; // 10;
}
