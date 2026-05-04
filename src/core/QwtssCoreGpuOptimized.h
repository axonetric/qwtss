#pragma once
#include "CryptoUtils.h"
#include "QwtssCoreShared.h"
#include "Tiles.h"

void initialize_gpu_constants_from_core_gpu_opt(const ITileSet* tileset);

GridAnnealResult grid_anneal_gpu_optimized(
    int* h_grid, 
    int grid_size,
    ITileSet* tileset, 
    GridAnnealParams params,
    ChaCha20PRNG& rng,
    const std::vector<uint8_t>& external_lock_mask
);
