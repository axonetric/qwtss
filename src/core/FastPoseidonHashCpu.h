#pragma once
#include "Tiles.h"
#include "QwtssAirTypes.h"
#include <array>
#include <vector>
#include <gmp.h>
#include <gmpxx.h>

// Core Hash Step (Full 4-Element State: 2 Rate, 2 Capacity)
std::array<mpz_class, 4> fast_poseidon_hash_step_cpu(
    const std::array<mpz_class, 4>& current_state, 
    const mpz_class& tile_id, 
    int step_index
);

// Function to compute the full digest over a grid (Returns a 504-bit digest via two Rate elements)
std::array<mpz_class, 2> compute_grid_poseidon_hash_cpu(const int* grid, int grid_size);

FieldElement256 mpz_to_field_element256(const mpz_class& val);

// Fetches the Poseidon state exactly *before* processing the target_step.
// Used to cache the "clean" state before a localized micro-grinding solver mutates the grid.
std::array<mpz_class, 4> get_poseidon_state_at_step_cpu(
    const int* grid, 
    int size, 
    int target_step
);

// Resumes the Poseidon hash chain from a cached state. 
// Used to rapidly hash only the mutated tail-end of the grid for zk-PoW checking.
std::array<mpz_class, 2> resume_grid_poseidon_hash_cpu(
    const int* grid, 
    int size, 
    const std::array<mpz_class, 4>& cached_state, 
    int start_step
);
