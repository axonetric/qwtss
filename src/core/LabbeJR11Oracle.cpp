#include "LabbeJR11Oracle.h"
#include <boost/multiprecision/cpp_bin_float.hpp>

using namespace boost::multiprecision;


// Use Boost's built-in 100-decimal-digit type (approx 332 bits)
// Keeping the alias 'float256' to avoid code changes
using float256 = cpp_bin_float_100;

struct Point256 {
    float256 x, y;
};

// 256-bit precision is required to match SageMath's RealField(200)
struct LabbePolygon {
    int tile_id;
    std::vector<Point256> vertices;
};

// The 11 Polygons defining the tiles
// Extracted from: Labbé, S. (2025). slabbe: Sébastien Labbé's Research code (Version 0.8.0) [SageMath package]. GitLab. https://gitlab.com/seblabbe/slabbe
// Reference: S. Labbé, "A Markov partition for the Jeandel-Rao Wang shift", arXiv:1903.06137
const std::vector<LabbePolygon> LABBE_POLYGONS = {
    {0, {{float256("0.000000000000000000000000000000000000000000000000000000000000"), float256("0.000000000000000000000000000000000000000000000000000000000000") }, {float256("0.618033988749894848204586834365638117720309179805762862135449"), float256("0.000000000000000000000000000000000000000000000000000000000000") }, {float256("0.618033988749894848204586834365638117720309179805762862135449"), float256("1.00000000000000000000000000000000000000000000000000000000000") }}},
    {0, {{float256("0.618033988749894848204586834365638117720309179805762862135449"), float256("0.000000000000000000000000000000000000000000000000000000000000") }, {float256("1.00000000000000000000000000000000000000000000000000000000000"), float256("0.000000000000000000000000000000000000000000000000000000000000") }, {float256("1.00000000000000000000000000000000000000000000000000000000000"), float256("1.00000000000000000000000000000000000000000000000000000000000") }}},
    {0, {{float256("1.00000000000000000000000000000000000000000000000000000000000"), float256("0.000000000000000000000000000000000000000000000000000000000000") }, {float256("1.61803398874989484820458683436563811772030917980576286213545"), float256("0.000000000000000000000000000000000000000000000000000000000000") }, {float256("1.61803398874989484820458683436563811772030917980576286213545"), float256("1.00000000000000000000000000000000000000000000000000000000000") }}},
    {1, {{float256("0.000000000000000000000000000000000000000000000000000000000000"), float256("0.000000000000000000000000000000000000000000000000000000000000") }, {float256("0.000000000000000000000000000000000000000000000000000000000000"), float256("1.00000000000000000000000000000000000000000000000000000000000") }, {float256("0.618033988749894848204586834365638117720309179805762862135449"), float256("1.00000000000000000000000000000000000000000000000000000000000") }}},
    {1, {{float256("0.618033988749894848204586834365638117720309179805762862135449"), float256("0.000000000000000000000000000000000000000000000000000000000000") }, {float256("0.618033988749894848204586834365638117720309179805762862135449"), float256("1.00000000000000000000000000000000000000000000000000000000000") }, {float256("1.00000000000000000000000000000000000000000000000000000000000"), float256("1.00000000000000000000000000000000000000000000000000000000000") }}},
    {1, {{float256("1.00000000000000000000000000000000000000000000000000000000000"), float256("0.000000000000000000000000000000000000000000000000000000000000") }, {float256("1.00000000000000000000000000000000000000000000000000000000000"), float256("1.00000000000000000000000000000000000000000000000000000000000") }, {float256("1.61803398874989484820458683436563811772030917980576286213545"), float256("1.00000000000000000000000000000000000000000000000000000000000") }}},
    {2, {{float256("0.000000000000000000000000000000000000000000000000000000000000"), float256("2.00000000000000000000000000000000000000000000000000000000000") }, {float256("0.000000000000000000000000000000000000000000000000000000000000"), float256("2.61803398874989484820458683436563811772030917980576286213545") }, {float256("0.618033988749894848204586834365638117720309179805762862135449"), float256("3.61803398874989484820458683436563811772030917980576286213545") }}},
    {3, {{float256("1.00000000000000000000000000000000000000000000000000000000000"), float256("2.00000000000000000000000000000000000000000000000000000000000") }, {float256("1.00000000000000000000000000000000000000000000000000000000000"), float256("1.00000000000000000000000000000000000000000000000000000000000") }, {float256("1.61803398874989484820458683436563811772030917980576286213545"), float256("2.00000000000000000000000000000000000000000000000000000000000") }, {float256("1.61803398874989484820458683436563811772030917980576286213545"), float256("3.61803398874989484820458683436563811772030917980576286213545") }}},
    {4, {{float256("0.000000000000000000000000000000000000000000000000000000000000"), float256("2.00000000000000000000000000000000000000000000000000000000000") }, {float256("0.618033988749894848204586834365638117720309179805762862135449"), float256("3.61803398874989484820458683436563811772030917980576286213545") }, {float256("1.00000000000000000000000000000000000000000000000000000000000"), float256("3.61803398874989484820458683436563811772030917980576286213545") }}},
    {4, {{float256("0.618033988749894848204586834365638117720309179805762862135449"), float256("3.61803398874989484820458683436563811772030917980576286213545") }, {float256("1.00000000000000000000000000000000000000000000000000000000000"), float256("3.61803398874989484820458683436563811772030917980576286213545") }, {float256("1.00000000000000000000000000000000000000000000000000000000000"), float256("4.61803398874989484820458683436563811772030917980576286213545") }}},
    {5, {{float256("0.000000000000000000000000000000000000000000000000000000000000"), float256("2.61803398874989484820458683436563811772030917980576286213545") }, {float256("0.000000000000000000000000000000000000000000000000000000000000"), float256("3.61803398874989484820458683436563811772030917980576286213545") }, {float256("0.618033988749894848204586834365638117720309179805762862135449"), float256("3.61803398874989484820458683436563811772030917980576286213545") }}},
    {5, {{float256("0.000000000000000000000000000000000000000000000000000000000000"), float256("3.61803398874989484820458683436563811772030917980576286213545") }, {float256("0.381966011250105151795413165634361882279690820194237137864551"), float256("3.61803398874989484820458683436563811772030917980576286213545") }, {float256("0.381966011250105151795413165634361882279690820194237137864551"), float256("4.61803398874989484820458683436563811772030917980576286213545") }}},
    {5, {{float256("0.381966011250105151795413165634361882279690820194237137864551"), float256("3.61803398874989484820458683436563811772030917980576286213545") }, {float256("0.618033988749894848204586834365638117720309179805762862135449"), float256("3.61803398874989484820458683436563811772030917980576286213545") }, {float256("1.00000000000000000000000000000000000000000000000000000000000"), float256("4.61803398874989484820458683436563811772030917980576286213545") }}},
    {6, {{float256("0.000000000000000000000000000000000000000000000000000000000000"), float256("3.61803398874989484820458683436563811772030917980576286213545") }, {float256("0.000000000000000000000000000000000000000000000000000000000000"), float256("4.61803398874989484820458683436563811772030917980576286213545") }, {float256("0.381966011250105151795413165634361882279690820194237137864551"), float256("4.61803398874989484820458683436563811772030917980576286213545") }}},
    {6, {{float256("0.381966011250105151795413165634361882279690820194237137864551"), float256("3.61803398874989484820458683436563811772030917980576286213545") }, {float256("0.381966011250105151795413165634361882279690820194237137864551"), float256("4.61803398874989484820458683436563811772030917980576286213545") }, {float256("1.00000000000000000000000000000000000000000000000000000000000"), float256("4.61803398874989484820458683436563811772030917980576286213545") }}},
    {6, {{float256("1.00000000000000000000000000000000000000000000000000000000000"), float256("3.61803398874989484820458683436563811772030917980576286213545") }, {float256("1.00000000000000000000000000000000000000000000000000000000000"), float256("4.61803398874989484820458683436563811772030917980576286213545") }, {float256("1.61803398874989484820458683436563811772030917980576286213545"), float256("4.61803398874989484820458683436563811772030917980576286213545") }}},
    {7, {{float256("1.00000000000000000000000000000000000000000000000000000000000"), float256("2.61803398874989484820458683436563811772030917980576286213545") }, {float256("1.00000000000000000000000000000000000000000000000000000000000"), float256("3.61803398874989484820458683436563811772030917980576286213545") }, {float256("1.61803398874989484820458683436563811772030917980576286213545"), float256("3.61803398874989484820458683436563811772030917980576286213545") }}},
    {7, {{float256("1.00000000000000000000000000000000000000000000000000000000000"), float256("3.61803398874989484820458683436563811772030917980576286213545") }, {float256("1.61803398874989484820458683436563811772030917980576286213545"), float256("3.61803398874989484820458683436563811772030917980576286213545") }, {float256("1.61803398874989484820458683436563811772030917980576286213545"), float256("4.61803398874989484820458683436563811772030917980576286213545") }}},
    {7, {{float256("0.000000000000000000000000000000000000000000000000000000000000"), float256("1.00000000000000000000000000000000000000000000000000000000000") }, {float256("0.000000000000000000000000000000000000000000000000000000000000"), float256("2.00000000000000000000000000000000000000000000000000000000000") }, {float256("1.00000000000000000000000000000000000000000000000000000000000"), float256("3.61803398874989484820458683436563811772030917980576286213545") }}},
    {8, {{float256("0.618033988749894848204586834365638117720309179805762862135449"), float256("1.00000000000000000000000000000000000000000000000000000000000") }, {float256("0.618033988749894848204586834365638117720309179805762862135449"), float256("2.00000000000000000000000000000000000000000000000000000000000") }, {float256("1.61803398874989484820458683436563811772030917980576286213545"), float256("3.61803398874989484820458683436563811772030917980576286213545") }}},
    {9, {{float256("0.000000000000000000000000000000000000000000000000000000000000"), float256("1.00000000000000000000000000000000000000000000000000000000000") }, {float256("0.618033988749894848204586834365638117720309179805762862135449"), float256("1.00000000000000000000000000000000000000000000000000000000000") }, {float256("0.618033988749894848204586834365638117720309179805762862135449"), float256("2.00000000000000000000000000000000000000000000000000000000000") }}},
    {9, {{float256("0.618033988749894848204586834365638117720309179805762862135449"), float256("1.00000000000000000000000000000000000000000000000000000000000") }, {float256("1.00000000000000000000000000000000000000000000000000000000000"), float256("1.00000000000000000000000000000000000000000000000000000000000") }, {float256("1.00000000000000000000000000000000000000000000000000000000000"), float256("2.00000000000000000000000000000000000000000000000000000000000") }}},
    {9, {{float256("1.00000000000000000000000000000000000000000000000000000000000"), float256("1.00000000000000000000000000000000000000000000000000000000000") }, {float256("1.61803398874989484820458683436563811772030917980576286213545"), float256("1.00000000000000000000000000000000000000000000000000000000000") }, {float256("1.61803398874989484820458683436563811772030917980576286213545"), float256("2.00000000000000000000000000000000000000000000000000000000000") }}},
    {10, {{float256("0.000000000000000000000000000000000000000000000000000000000000"), float256("1.00000000000000000000000000000000000000000000000000000000000") }, {float256("1.00000000000000000000000000000000000000000000000000000000000"), float256("2.61803398874989484820458683436563811772030917980576286213545") }, {float256("1.00000000000000000000000000000000000000000000000000000000000"), float256("3.61803398874989484820458683436563811772030917980576286213545") }}},
};


// The internal implementation struct
struct LabbeJR11Oracle::Impl {
    // Constants derived from Labbé's Markov Partition
    float256 phi;
    float256 v1_x, v2_x, v2_y;

    // The statically deduced color permutation from Labbé's partition to our JR-11 tileset
    const int LABBE_TO_JR_MAP[11] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

    Impl() {
        phi = (1 + sqrt(float256(5))) / 2;
        v1_x = phi;
        v2_x = 1;
        v2_y = phi + 3;
    }

        // Exact Torus Reduction: reduce_mod_gamma
    Point256 reduce_mod_gamma(float256 x, float256 y) {
        float256 c2 = floor(y / v2_y);
        float256 v = y - (c2 * v2_y);
        
        float256 x_shifted = x - (c2 * v2_x);
        float256 c1 = floor(x_shifted / v1_x);
        float256 u = x_shifted - (c1 * v1_x);
        
        return {u, v};
    }

    // Ray-Casting Point-in-Polygon (PIP) with epsilon tie-breaking
    bool is_inside(float256 u, float256 v, const std::vector<Point256>& poly) {
        bool inside = false;
        for (size_t i = 0, j = poly.size() - 1; i < poly.size(); j = i++) {
            if (((poly[i].y > v) != (poly[j].y > v)) &&
                (u < (poly[j].x - poly[i].x) * (v - poly[i].y) / (poly[j].y - poly[i].y) + poly[i].x)) {
                inside = !inside;
            }
        }
        return inside;
    }

    int get_tile_at(float256 x, float256 y) {
        Point256 uv = reduce_mod_gamma(x, y);
        
        // Iterate through all the polygons, regardless of how many there are
        for (const auto& poly_struct : LABBE_POLYGONS) {
            
            // Pass the struct's specific vertices to the ray-caster
            if (is_inside(uv.x, uv.y, poly_struct.vertices)) {
                // Return the correct tile ID associated with the matched polygon
                return poly_struct.tile_id;
            }
        }

        // Deterministic Tie-Breaker: If on a border, nudge and re-check
        // This handles the "measure zero" edge cases exactly
        return get_tile_at(x + float256("1e-40"), y + float256("1e-40"));
    }

    /**
     * @brief Generates an NxN Jeandel-Rao grid using the O(1) continuous torus map formalized by Sébastien Labbé.
     * @param start_x The 256-bit X-coordinate on the continuous plane (Left-most bound)
     * @param start_y The 256-bit Y-coordinate on the continuous plane (Top-most bound)
     * @param grid_size The NxN dimension of the desired grid (e.g., 64)
     * @return std::vector<int> A 1D array representing the row-major NxN grid
     */
    std::vector<int> generate_jr11_grid(float256 start_x, float256 start_y, int grid_size) {
        if (grid_size <= 0) throw std::invalid_argument("Grid size must be positive.");

        std::vector<int> grid(grid_size * grid_size, -1);

        for (int r = 0; r < grid_size; ++r) {
            for (int c = 0; c < grid_size; ++c) {
                
                // Apply the deduced geometric mapping (Flip X: 1, Flip Y: 0)
                // c goes right (C++), so X goes left (Labbé)
                // r goes down (C++), so Y goes up (Labbé)
                float256 current_x = start_x - float256(c);
                float256 current_y = start_y + float256(r);

                // 1. Get the raw topological tile ID from the Markov Partition (0 to 10)
                int raw_labbe_id = get_tile_at(current_x, current_y);

                // 2. Map it to the C++ alphabet color orientation
                int jr_tile_id = LABBE_TO_JR_MAP[raw_labbe_id];

                // 3. Store in the row-major grid
                grid[r * grid_size + c] = jr_tile_id;
            }
        }

        return grid;
    }
};

LabbeJR11Oracle::LabbeJR11Oracle() : pimpl(std::make_unique<Impl>()) {}

LabbeJR11Oracle::~LabbeJR11Oracle() = default;

/**
 * @brief Generates an NxN Jeandel-Rao grid using the O(1) continuous torus map formalized by Sébastien Labbé.
 * @param start_x The X-coordinate on the continuous plane (Left-most bound)
 * @param start_y The Y-coordinate on the continuous plane (Top-most bound)
 * @param grid_size The NxN dimension of the desired grid (e.g., 64)
 * @return std::vector<int> A 1D array representing the row-major NxN grid
 */
std::vector<int> LabbeJR11Oracle::generate_jr11_grid(double start_x, double start_y, int grid_size){
    // 1. Safely upcast the standard IEEE-754 doubles to Boost's 256-bit floats
    float256 boost_x(start_x);
    float256 boost_y(start_y);

    // 2. Delegate the actual heavy lifting to the hidden implementation struct
    return pimpl->generate_jr11_grid(boost_x, boost_y, grid_size);
}

/**
 * @brief Generates a Jeandel-Rao Public Key Mask in O(1) time using Deterministic Alternating Splicing.
 * @param num_segments Tunable topological frustration. This must be an even number to prevent parity merging between the first and last segment.
 * @return The returned vector of tile ids (0-10)  is a flat 1D array of size grid_size * grid_size.
 * The Border: Contains values 0 through 10. The Bulk (Interior): Contains -1.
 */
LabbeOraclePublicKey LabbeJR11Oracle::generate_spliced_public_key(int grid_size, int num_segments, ChaCha20PRNG& rng) {
    if (num_segments < 10) throw std::invalid_argument("num_segments must be 10+");
    if (num_segments % 2 != 0) throw std::invalid_argument("num_segments must be an even number");

    // 1. Generate four 64-bit random seeds directly from the RNG
    uint64_t seed_xA = rng.next_u64();
    uint64_t seed_yA = rng.next_u64();
    uint64_t seed_xB = rng.next_u64();
    uint64_t seed_yB = rng.next_u64();

    // 3. Map seeds into the deep continuous Torus space (approx 10^29)
    float256 xA = float256("100000000000000000000000000000.0") + float256(seed_xA);
    float256 yA = float256("200000000000000000000000000000.0") + float256(seed_yA);
    float256 xB = float256("300000000000000000000000000000.0") + float256(seed_xB);
    float256 yB = float256("400000000000000000000000000000.0") + float256(seed_yB);

    // 4. Generate the two zero-defect quasiperiodic planes
    std::vector<int> plane_A = pimpl->generate_jr11_grid(xA, yA, grid_size);
    std::vector<int> plane_B = pimpl->generate_jr11_grid(xB, yB, grid_size);

    // 5. Unwrap perimeters into 1D loops
    int perimeter_len = grid_size * 4 - 4;
    std::vector<int> pA(perimeter_len);
    std::vector<int> pB(perimeter_len);
    
    auto get_perimeter = [&](const std::vector<int>& grid, std::vector<int>& p) {
        int idx = 0;
        for(int c = 0; c < grid_size; c++) p[idx++] = grid[0 * grid_size + c];
        for(int r = 1; r < grid_size; r++) p[idx++] = grid[r * grid_size + (grid_size - 1)];
        for(int c = grid_size - 2; c >= 0; c--) p[idx++] = grid[(grid_size - 1) * grid_size + c];
        for(int r = grid_size - 2; r >= 1; r--) p[idx++] = grid[r * grid_size + 0];
    };

    get_perimeter(plane_A, pA);
    get_perimeter(plane_B, pB);

    // 6. The Deterministic Splicing Logic
    std::vector<int> pMaster(perimeter_len);

    // 80% min segment length
    int min_seg_len = static_cast<int>(std::round(0.80f * (float)perimeter_len / (float)num_segments));

    // Set the minimum seg length for each 
    std::vector<int> segment_lengths(num_segments, min_seg_len); 
    int total_assigned = num_segments * min_seg_len;
    
    // Randomly distribute the remaining tiles across the segments
    int remaining = perimeter_len - total_assigned;
    for (int i = 0; i < remaining; i++) {
        segment_lengths[rng.next_u32_range(0, num_segments - 1)]++;
    }
    
    // Random Start Offset to prevent fixed-point attacks
    // The start_idx shifts the seams (the splice points) around the perimeter, defeating fixed-point
    // attacks without breaking the underlying geometry.
    int start_idx = rng.next_u32_range(0, perimeter_len - 1);

    // Map them onto the perimeter
    int tiles_filled = 0;
    for (int s = 0; s < num_segments; s++) {
        // Because num_segments is always even, this alternates A, B, A, B... and
        // ensures the last segment (B) will never merge with the first (A)
        bool use_A = (s % 2 == 0);

        for (int i = 0; i < segment_lengths[s]; i++) {
            int current_idx = (start_idx + tiles_filled + i) % perimeter_len;
            pMaster[current_idx] = use_A ? pA[current_idx] : pB[current_idx];
        }
        tiles_filled += segment_lengths[s];
    }

    // 7. Create the Public Key Mask (-1 for interior tiles that need to be solved)
    std::vector<int> public_key_boundary(grid_size * grid_size, -1);
    
    // Re-wrap the spliced perimeter back onto the main grid
    int idx = 0;
    for(int c = 0; c < grid_size; c++) public_key_boundary[0 * grid_size + c] = pMaster[idx++];
    for(int r = 1; r < grid_size; r++) public_key_boundary[r * grid_size + (grid_size - 1)] = pMaster[idx++];
    for(int c = grid_size - 2; c >= 0; c--) public_key_boundary[(grid_size - 1) * grid_size + c] = pMaster[idx++];
    for(int r = grid_size - 2; r >= 1; r--) public_key_boundary[r * grid_size + 0] = pMaster[idx++];

    return { public_key_boundary, plane_A, plane_B };
}

bool LabbeJR11Oracle::run_labbe_oracle_unit_test() {
    std::cout << "\n---------------------------------------------------------" << std::endl;
    std::cout << "   O(1) Labbé Oracle Generation Unit Test\n" << std::endl;

    JeandelRaoTileSet jr_tileset;
    std::vector<Tile> alphabet = jr_tileset.get_tiles();

    int grid_size = 64;

    // Direct Pimpl instantiation for the static test
    Impl oracle_impl;

    // 1. Define arbitrary, high-precision coordinates (Simulating a SHA-256 Hash Seed)
    // High-magnitude coordinates prove that the 100-decimal-digit precision prevents aperiodic drift.
    float256 start_x("123456789012345678901234567890.12345678901234567890");
    float256 start_y("987654321098765432109876543210.98765432109876");

    std::cout << "[*] Seed Coordinate X: " << start_x.str(20) << "..." << std::endl;
    std::cout << "[*] Seed Coordinate Y: " << start_y.str(20) << "..." << std::endl;
    std::cout << "[*] Generating " << grid_size << "x" << grid_size 
            << " mathematical plane from continuous torus ..." << std::endl;

    // 2. Generate the "Ground Truth" mathematical plane
    std::vector<int> nums_grid = oracle_impl.generate_jr11_grid(start_x, start_y, grid_size);

    // 3. Prove the Toral Z^2-Rotation Math and Static Permutation are Flawless
    std::cout << "[*] Scanning " << (grid_size * grid_size) << " tiles for topological defects..." << std::endl;
    
    int oracle_defects = count_grid_defects(nums_grid.data(), grid_size, alphabet);
    std::cout << "[*] Oracle Defect Count: " << oracle_defects << std::endl;

    if (oracle_defects > 0) {
        std::cout << "\n[UNIT TEST FAILED] The O(1) generated grid contains topological defects." << std::endl;
        return false;
    }
    else {
        std::cout << "\n[UNIT TEST PASSED] The continuous generation is mathematically perfect." << std::endl;
        return true;
    }
}
