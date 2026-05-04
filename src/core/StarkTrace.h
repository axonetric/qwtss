#pragma once
#include "Tiles.h"
#include "QwtssAirTypes.h"
#include <vector>
#include <cstdint>
#include <array>


std::array<StoneField, 4> poseidon_hash_step(std::array<StoneField, 4> current_state, StoneField tile_id, size_t step_index);
void test_poseidon_math();

// The Algebraic Execution Trace (the prover's secret)
// CRITICAL: Private key data. Do not transmit.
struct ExecutionTrace {
    // 4096 for a single 64x64 grid
    size_t valid_steps = 0;
    // Random noise injected at these final steps
    size_t blinding_steps;
    // Next power of 2 for FFTs; this is a zk-STARK requirement
    size_t trace_length = 0;

    // Tile Definition (Derived strictly from booleans to minimize trace size)
    // Dynamic One-Hot encoded tile booleans for STARK alphabet constraints
    // This scales automatically to any ITileSet (JR-11, A-16, etc.)
    std::vector<std::vector<FieldElement>> tile_booleans;

    // Defect Flags (Prover asserts these are exactly 0 or 1)
    std::vector<FieldElement> is_h_defect;  // 1 if Right doesn't match next Left
    std::vector<FieldElement> is_v_defect;  // 1 if Bottom doesn't match next Top

    // Orthogonal Seam Slack Accumulators (Line_Max Enforcement)
    std::vector<FieldElement> h_seam_acc; 
    std::vector<FieldElement> v_seam_acc;

    // Slack variable for the <= defect_upper_bound proof
    std::vector<FieldElement> dummy_defect;

    // Accumulator for Alien Tiles.
    std::vector<FieldElement> alien_acc;
    // Helper column to prove Inequality without one-hot bloat
    std::vector<FieldElement256> alien_inv;

    // Core 32x32 defect accumulator
    std::vector<FieldElement> core32_acc;

    // The Constraint Counter
    std::vector<FieldElement256> defect_accumulator;
    // The running algebraic hash state
    std::vector<FieldElement256> hash_state_0;  // The state of the hash Capacity at the very beginning of the step, before any math happens.
    std::vector<FieldElement256> hash_state_1;
    std::vector<FieldElement256> hash_state_2;
    std::vector<FieldElement256> hash_state_3;

    // Intermediate columns to drop the max AIR degree down to 4 (trace flattening)
    std::vector<FieldElement> half_check;
    std::vector<FieldElement> core_defect;
    std::vector<FieldElement256> diff_A_B;

    // Tight Packing Flattening
    std::vector<FieldElement> is_alien;
    std::vector<FieldElement256> expected_hash_0;   // This represents what the hash will be on the next row (after all math has happened).
    std::vector<FieldElement256> expected_hash_1;
    std::vector<FieldElement256> expected_hash_2;
    std::vector<FieldElement256> expected_hash_3;
};


// The Trace Builder with CSPRNG Blinding (Accepts K+1 grids)
ExecutionTrace build_execution_trace(const std::vector<QwtssPrivateKey>& private_keys, int grid_size,
    const std::vector<Tile>& alphabet, bool throw_preemptively = true, bool do_logging = false);

std::vector<std::byte> generate_stark_signature(
    const ExecutionTrace& raw_trace, 
    const QwtssPublicInputs& public_inputs,
    const std::vector<uint8_t>& message_to_sign,
    const std::vector<Tile>& alphabet,
    bool do_logging = true
);

bool verify_stark_signature(
    const std::vector<std::byte>& signature, 
    const QwtssPublicInputs& public_inputs,
    const std::vector<uint8_t>& message_to_sign,
    const std::vector<Tile>& alphabet,
    bool do_logging = true
);
