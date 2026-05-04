#pragma once
#include "Tiles.h"
#include "QwtssConfig.h"
#include "CryptoUtils.h"
#include "QwtssAirTypes.h" // For FieldElement256
#include <random>

// Guard CUDA headers and macros from standard C++ compilers
#ifdef __CUDACC__

#include <curand_kernel.h>
#include <device_launch_parameters.h>


#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d code=%d(%s)\n", __FILE__, __LINE__, err, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)
#endif

#ifdef __CUDACC__
// GPU constants are hidden to guide the init pipeline; if we try to do it manually/locally, then __constant__ 
// internal linkage will cause an unintended "Shadow Constant Memory" bug.
namespace qwtss_core_shared_device {
    // Declared as arrays in constant memory
    // Hidden from host scope; each .cu file instantiates an independent physical copy
    __constant__ Tile d_tiles_const[QwtssConfig::MAX_TILES];
    __constant__ int d_num_tiles_const;
}
#endif

// Declare explicit initialization functions for each caller module. The actual implementations must be
// inside those individual modules (where the device kernels actually live) to prevent the "Shadow Constant Memory" bug.
void initialize_gpu_constants_from_core_shared(const ITileSet* tileset);
void initialize_gpu_constants_from_crypto_core(const ITileSet* tileset);
void initialize_gpu_constants_from_core_gpu_opt(const ITileSet* tileset);

// ---------------------------------------------------------
// GPU KERNELS
// ---------------------------------------------------------

#ifdef __CUDACC__
// Version for NxN square grids
__device__ inline int calculate_local_energy_gpu(int *grid, int grid_size, int row, int col, int proposed_tile_idx) {
    int energy = 0;
    Tile proposed = qwtss_core_shared_device::d_tiles_const[proposed_tile_idx];

    // North
    if (row > 0) {
        Tile neighbor = qwtss_core_shared_device::d_tiles_const[grid[(row - 1) * grid_size + col]];
        if (proposed.top != neighbor.bottom) energy++;
    }
    // South
    if (row < grid_size - 1) {
        Tile neighbor = qwtss_core_shared_device::d_tiles_const[grid[(row + 1) * grid_size + col]];
        if (proposed.bottom != neighbor.top) energy++;
    }
    // West
    if (col > 0) {
        Tile neighbor = qwtss_core_shared_device::d_tiles_const[grid[row * grid_size + (col - 1)]];
        if (proposed.left != neighbor.right) energy++;
    }
    // East
    if (col < grid_size - 1) {
        Tile neighbor = qwtss_core_shared_device::d_tiles_const[grid[row * grid_size + (col + 1)]];
        if (proposed.right != neighbor.left) energy++;
    }
    return energy;
}

__global__ inline void init_curand_states(curandStatePhilox4_32_10_t *state, const unsigned long long *secure_host_seeds, int grid_size) {
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id < grid_size * grid_size) {
        // curand_init args for Philox: (seed, subsequence, offset, state)
        // By giving every thread a truly random seed from the host, the global entropy is massively secure.
        curand_init(secure_host_seeds[id], id, 0, &state[id]);
    }
}


__global__ void simulated_annealing_kernel(int *grid, int grid_size, float temperature, curandStatePhilox4_32_10_t* state, int is_black_phase);

__global__ void heat_bath_kernel_unified(int* d_grid, int grid_size, float temp, curandStatePhilox4_32_10_t* states, int is_black_phase, const uint8_t* d_locked_mask);

int locked_ais_mcmc_step_gpu(
    int* h_grid, int grid_size, int* d_grid, curandStatePhilox4_32_10_t* d_states, ITileSet* tileset, 
    float current_temp, int steps, const uint8_t* d_locked_mask
);

#endif

struct DefectStats {
    int total_defective_tiles = 0;
    int num_holes = 0;
    int max_hole_size = 0;
    float avg_hole_size = 0.0f;
};

std::vector<uint8_t> generate_partial_boundary_mask(int grid_size, float keep_boundary_percentage, ChaCha20PRNG& rng,
    const std::vector<uint8_t>& existing_mask = {});

void generate_boundary_conditioned_random_seed_grid(const std::vector<int>& ground_truth_grid, int grid_size,
    std::vector<int>& h_grid, int run_target_defects, std::vector<Tile>& alphabet, ChaCha20PRNG& rng,
    const std::vector<uint8_t>& public_key_mask = {});

void confirm_mask_constraints_honored(const std::vector<int> h_grid, const std::vector<uint8_t> boundary_mask, int grid_size,
    const std::vector<int> ground_truth, const std::vector<Tile>& alphabet, bool do_logging);

std::vector<uint8_t> generate_quenched_disorder_mask(int grid_size, int num_pins, double min_radius, ChaCha20PRNG& rng);

DefectStats analyze_defect_topology(int* h_grid, int grid_size, const std::vector<Tile>& tiles);

// Struct to hold the spatial distribution metrics
struct DefectDistributionStats {
    int min_defects;
    int max_defects;
    double avg_defects;
    double std_dev_defects;
    int total_defects; // The sum across all sub-grids (should match the global count)
};

DefectDistributionStats calculate_defect_spatial_distribution_subgrids(
    const std::vector<int>& h_grid, 
    int grid_size, 
    const std::vector<Tile>& alphabet, 
    int sub_grid_size
);

DefectDistributionStats calculate_defect_spatial_distribution_sliding(
    const std::vector<int>& h_grid, 
    int grid_size, 
    const std::vector<Tile>& alphabet, 
    int sliding_grid_size
);

DefectDistributionStats calculate_defect_spatial_distribution_lines(
    const std::vector<int>& h_grid, 
    int grid_size, 
    const std::vector<Tile>& alphabet
);

int calculate_defect_spatial_distribution_frame(
    const std::vector<int>& h_grid,
    int grid_size,
    const std::vector<Tile>& alphabet,
    int frame_depth
);

int calculate_alien_frame_count(
    const std::vector<int>& grid,
    const std::vector<int>& plane_A,
    const std::vector<int>& plane_B,
    int grid_size,
    int frame_depth
);

int calculate_core_defects(
    const std::vector<int>& h_grid,
    int grid_size,
    const std::vector<Tile>& alphabet,
    int core_size
);

std::vector<int> calculate_local_energy_histogram(
    const std::vector<int>& h_grid,
    int grid_size,
    const std::vector<Tile>& alphabet
);

int calculate_alien_tiles(
    const std::vector<int>& grid, 
    const std::vector<int>& plane_A, 
    const std::vector<int>& plane_B, 
    int grid_size
);

struct GridAnnealResult {
    // The final defect count, or -1 if failed.
    int final_defects = -1;
    float final_temp;
    // The computational work performed (how many times heat_bath_kernel was called on the grid). This value is returned even on failures.
    int steps;
};

struct GridAnnealParams {
    float initial_temp = 5.0f;      // High initial temperature allows escaping local minima
    float cooling_rate = 0.99992f;  // Best between 0.99992 to 0.99996
    float reheat_temp = 0.39f;      // Between 0.35 and 0.43 is best
    int stagnation_patience = 2;    // Clear winner
    int check_interval = 2200;      // 2000 to 2400 is best for low target defect counts, but should be much lower for high defect counts
    int max_steps = 250000;         // Based on Las Vegas Restart Strategy optimal stats
    bool do_greedy_prequench = true;
    bool do_pinned_defect_doping = false;

    int max_allowed_defects = 350;
    int min_allowed_defects = 0;    // Default to 0 to not alter existing code

    // This ctor also dynamically adjusts check_interval for the specified max_allowed_defects value.
    GridAnnealParams(int max_allowed_defects = 170, int min_allowed_defects = 0, int max_steps = 250000) :
        max_allowed_defects(max_allowed_defects), min_allowed_defects(min_allowed_defects), max_steps(max_steps) {
        assign_optimal_check_interval(max_allowed_defects);
    }

    // Assigns new int max_allowed_defects and the optimal check_interval value for it.
    void assign_optimal_check_interval(int max_allowed_defects){
        this->max_allowed_defects = max_allowed_defects;

        check_interval = 2200;      // 2000 to 2400 is best for low target defect counts
        if (max_allowed_defects >= 1500) check_interval = 5;
        else if (max_allowed_defects >= 1000) check_interval = 30;
        else if (max_allowed_defects >= 800) check_interval = 80;
        else if (max_allowed_defects >= 600) check_interval = 200;
        else if (max_allowed_defects >= 400) check_interval = 500;
        else if (max_allowed_defects >= 200) check_interval = 1000;
        else if (max_allowed_defects >= 120) check_interval = 1500;
    }
};

void export_grid_to_ppm(const int* grid, int grid_size, ITileSet* tileset, const std::string& filename);
void export_defect_edges_to_ppm(const int* grid, int grid_size, ITileSet* tileset, const std::string& filename);
void export_alien_tiles_to_ppm(const int* grid, int grid_size, const std::vector<int>& plane_A, const std::vector<int>& plane_B, const std::string& filename);

struct RebarPin {
    int r;
    int c;
    int tile_id;
};
