#pragma once
#include "Tiles.h"
#include "CryptoUtils.h"
#include <vector>
#include <random>
#include <memory>

struct LabbeOraclePublicKey{
    // The returned vector of tile ids (0-10)  is a flat 1D array of size grid_size * grid_size. The Border: Contains values 0 through 10. The Bulk (Interior): Contains -1.
    std::vector<int> pk_mask;
    // The oracle grid A used in the PK splicing. Used for "alien tile" calcs.
    std::vector<int> plane_A;
    // The oracle grid B used in the PK splicing. Used for "alien tile" calcs.
    std::vector<int> plane_B;
};


class LabbeJR11Oracle {
private:
    // Pimpl Idiom: Forward declare an internal implementation struct.
    // This hides all Boost and multiprecision types from the CUDA compiler.
    struct Impl;
    std::unique_ptr<Impl> pimpl;

public:
    LabbeJR11Oracle();
    ~LabbeJR11Oracle();

    /**
     * @brief Generates a Jeandel-Rao grid in O(1) time using double-precision coordinates.
     * Note: Subject to standard IEEE-754 floating-point precision loss.
     */
    std::vector<int> generate_jr11_grid(double start_x, double start_y, int grid_size);

    void batch_export_jr11_grids_to_csv(int num_grids, int grid_size, double temperature, const std::string& filename = "");

    LabbeOraclePublicKey generate_spliced_public_key(int grid_size, int num_segments, ChaCha20PRNG& rng);

    static bool run_labbe_oracle_unit_test();
};
