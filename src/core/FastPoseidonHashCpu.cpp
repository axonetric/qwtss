#include "FastPoseidonHashCpu.h"
#include "PoseidonConstants.h"
#include <cstdint>

// Standalone Fast Poseidon Hash using GMP (GNU Multiple Precision Arithmetic Library)

// The 252-bit StarkWare Prime (0x800000000000011000000000000000000000000000000000000000000000001)
const mpz_class STARK_PRIME("3618502788666131213697322783095070105623107215331596699973092056135872020481");

// Helper: Fast Modular Addition
inline void add_mod(mpz_class& res, const mpz_class& a, const mpz_class& b) {
    res = a + b;
    if (res >= STARK_PRIME) {
        res -= STARK_PRIME;
    }
}

// Helper: Fast Modular Multiplication
inline void mul_mod(mpz_class& res, const mpz_class& a, const mpz_class& b) {
    res = (a * b) % STARK_PRIME;
}

// Helper to safely map the 4 Little-Endian limbs into GMP
inline mpz_class u64_array_to_mpz(const uint64_t limbs[4]) {
    mpz_class res = limbs[3]; // Most significant
    res = (res << 64) | limbs[2];
    res = (res << 64) | limbs[1];
    res = (res << 64) | limbs[0]; // Least significant
    return res;
}

// Thread-safe static initialization: Runs once at startup.
// Pre-convert all 16,384 dynamic round constants to GMP format (done once thread-safely)
static const std::vector<std::array<mpz_class, 4>> GMP_FAST_ROUND_CONSTANTS = []() {
    std::vector<std::array<mpz_class, 4>> res(4096);
    for (size_t i = 0; i < 4096; ++i) {
        res[i][0] = u64_array_to_mpz(PoseidonConstants::ROUND_CONSTANTS_EXTENDED[i * 4 + 0]);
        res[i][1] = u64_array_to_mpz(PoseidonConstants::ROUND_CONSTANTS_EXTENDED[i * 4 + 1]);
        res[i][2] = u64_array_to_mpz(PoseidonConstants::ROUND_CONSTANTS_EXTENDED[i * 4 + 2]);
        res[i][3] = u64_array_to_mpz(PoseidonConstants::ROUND_CONSTANTS_EXTENDED[i * 4 + 3]);
    }
    return res;
}();

// Initialize the static Cauchy MDS GMP constants (Full 3x3 Matrix)
// Local static variables are initialized exactly once in a thread-safe manner.
static const std::array<std::array<mpz_class, 4>, 4> GMP_FAST_MDS = []() {
    std::array<std::array<mpz_class, 4>, 4> res;
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            res[i][j] = u64_array_to_mpz(PoseidonConstants::MDS[i][j]);
        }
    }
    return res;
}();

// The CPU-Optimized Poseidon Hash Function (Full Sponge Compliance)
// Note: current_state made const reference to prevent ABI Segfaults
std::array<mpz_class, 4> fast_poseidon_hash_step_cpu(
    const std::array<mpz_class, 4>& current_state, 
    const mpz_class& tile_id, 
    int step_index
) {
    // Create an aligned local working copy
    std::array<mpz_class, 4> next_state = current_state;

    // 1. Sponge Absorption (Add Tile ID to Rate 1, Step Index to Rate 2)
    // Elements 2 and 3 (Capacity) are left alone during absorption
    add_mod(next_state[0], next_state[0], tile_id);
    add_mod(next_state[1], next_state[1], mpz_class(step_index));

    // 2. Add Round Constants
    add_mod(next_state[0], next_state[0], GMP_FAST_ROUND_CONSTANTS[step_index][0]);
    add_mod(next_state[1], next_state[1], GMP_FAST_ROUND_CONSTANTS[step_index][1]);
    add_mod(next_state[2], next_state[2], GMP_FAST_ROUND_CONSTANTS[step_index][2]);
    add_mod(next_state[3], next_state[3], GMP_FAST_ROUND_CONSTANTS[step_index][3]);

    // 3. Non-Linear Layer (S-Box x^3)
    mpz_class x2;
    for (int i = 0; i < 4; i++) {
        mul_mod(x2, next_state[i], next_state[i]); // x^2
        mul_mod(next_state[i], x2, next_state[i]); // x^3
    }

    // 4. Linear Layer (Full 4x4 MDS Matrix Multiplication)
    std::array<mpz_class, 4> final_state;
    mpz_class m0, m1, m2, m3;

    for (int i = 0; i < 4; i++) {
        mul_mod(m0, next_state[0], GMP_FAST_MDS[i][0]);
        mul_mod(m1, next_state[1], GMP_FAST_MDS[i][1]);
        mul_mod(m2, next_state[2], GMP_FAST_MDS[i][2]);
        mul_mod(m3, next_state[3], GMP_FAST_MDS[i][3]);

        add_mod(final_state[i], m0, m1);
        add_mod(final_state[i], final_state[i], m2);
        add_mod(final_state[i], final_state[i], m3);
    }

    return final_state;
}

// Function to compute the full digest over a grid
std::array<mpz_class, 2> compute_grid_poseidon_hash_cpu(const int* grid, int grid_size) {

    // The STARK trace initializes the hash accumulator at 0 for all 4 elements
    std::array<mpz_class, 4> state = {mpz_class(0), mpz_class(0), mpz_class(0), mpz_class(0)};

    // Evaluate the grid identically to the execution trace
    for (int step = 0; step < grid_size * grid_size; step++) {
        mpz_class tile_id = grid[step];

        // Step the hash chain using the dynamic constants for this specific step
        state = fast_poseidon_hash_step_cpu(state, tile_id, step);
    }

    // Squeeze the sponge: The final 504-bit digest is extracted from both Rate elements (0 and 1)
    return {state[0], state[1]};
}

FieldElement256 mpz_to_field_element256(const mpz_class& val) {
    // Zero-initialize the array. If the mpz_class value is 0, 
    // mpz_export writes nothing, so this ensures it returns {0,0,0,0}.
    FieldElement256 result = {0, 0, 0, 0};
    size_t count = 0;

    // mpz_export parameters:
    // rop:     result.data() (destination array pointer)
    // countp:  &count (records how many words were actually written)
    // order:   -1 (Least significant word first -> result[0] gets the lowest 64 bits)
    // size:    sizeof(uint64_t) (8 bytes per word)
    // endian:  0 (Native byte order for the system's architecture)
    // nails:   0 (Use all 64 bits of the word, no padding)
    // op:      val.get_mpz_t() (the raw GMP struct)
    mpz_export(result.data(), &count, -1, sizeof(uint64_t), 0, 0, val.get_mpz_t());

    // A valid 256-bit prime field element should never exceed four 64-bit words.
    if (count > 4) {
        throw std::overflow_error("GMP value exceeds 256 bits; cannot safely convert to FieldElement256.");
    }

    return result;
}

// ------------------------------------------------------------------------
// Hash Grinding Optimization Functions
// ------------------------------------------------------------------------

std::array<mpz_class, 4> get_poseidon_state_at_step_cpu(
    const int* grid, int size, int target_step) {
    
    std::array<mpz_class, 4> state = {mpz_class(0), mpz_class(0), mpz_class(0), mpz_class(0)};

    for (int step = 0; step < target_step; step++) {
        mpz_class tile_id = grid[step];
        state = fast_poseidon_hash_step_cpu(state, tile_id, step);
    }
    
    return state;
}

std::array<mpz_class, 2> resume_grid_poseidon_hash_cpu(
    const int* grid, int size, 
    const std::array<mpz_class, 4>& cached_state, int start_step) {

    // Create local aligned copy
    std::array<mpz_class, 4> state = cached_state;
    int total_steps = size * size;

    for (int step = start_step; step < total_steps; step++) {
        mpz_class tile_id = grid[step];
        state = fast_poseidon_hash_step_cpu(state, tile_id, step);
    }

    // Squeeze the sponge: The final 504-bit digest is extracted from both Rate elements (0 and 1)
    return {state[0], state[1]};
}
