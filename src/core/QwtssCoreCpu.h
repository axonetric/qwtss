#pragma once
#include "Tiles.h"
#include "QwtssCoreShared.h"
#include "CryptoUtils.h"

int count_grid_defects_cpu(const int* grid, int grid_size, const std::vector<Tile>& alphabet);

GridAnnealResult grid_anneal_cpu(
    int* grid, 
    int grid_size,
    ITileSet* tileset, 
    GridAnnealParams params,
    ChaCha20PRNG& rng,
    const std::vector<uint8_t>& external_lock_mask = {},
    int target_threads = -1,
    bool do_logging = false
);

GridAnnealResult grid_anneal_cpu_adiabatic(
    int* grid, 
    int grid_size,
    ITileSet* tileset, 
    GridAnnealParams params,
    ChaCha20PRNG& rng,
    const std::vector<uint8_t>& external_lock_mask,
    int target_threads = -1,
    bool do_logging = false
);

GridAnnealResult grid_anneal_cpu_optimized(
    int* grid, 
    int grid_size,
    ITileSet* tileset, 
    GridAnnealParams params,
    ChaCha20PRNG& rng,
    const std::vector<uint8_t>& external_lock_mask,
    int target_threads, 
    bool do_logging
);
