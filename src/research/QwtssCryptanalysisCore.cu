#include "QwtssCryptanalysisCore.h"
#include "QwtssCoreShared.h"
#include "QwtssConfig.h"
#include "PrivateKeyDatabase.h"
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <cstdint>
#include <fstream>
#include <cstring>
#include <queue>
#include <algorithm>
#include <unordered_set>

// Forward declarations
void optimize_attack_A_maximize_aliens(std::vector<int>& forged_grid, int grid_size,
    const std::vector<Tile>& alphabet, const std::vector<int>& plane_A, const std::vector<int>& plane_B, int defect_limit,
    const std::vector<uint8_t>& boundary_mask, 
    const std::vector<int>& public_key_grid);
void optimize_attack_B_minimize_defects(std::vector<int>& forged_grid, int grid_size,
    const std::vector<Tile>& alphabet, const std::vector<int>& plane_A, const std::vector<int>& plane_B, int alien_limit,
    const std::vector<uint8_t>& boundary_mask, 
    const std::vector<int>& public_key_grid);
void optimize_attack_C_maximize_frame_aliens(
    std::vector<int>& forged_grid, int grid_size, const std::vector<Tile>& alphabet,
    const std::vector<int>& plane_A, const std::vector<int>& plane_B,
    int defect_limit, int alien_limit, int frame_depth,
    const std::vector<uint8_t>& boundary_mask, 
    const std::vector<int>& public_key_grid);
void optimize_attack_D_spatial_trap(
    std::vector<int>& forged_grid, int grid_size, const std::vector<Tile>& alphabet,
    const std::vector<int>& plane_A, const std::vector<int>& plane_B,
    int defect_limit, int alien_limit,
    const std::vector<uint8_t>& boundary_mask, 
    const std::vector<int>& public_key_grid);
void optimize_attack_E_universal_trap(
    std::vector<int>& forged_grid, int grid_size, const std::vector<Tile>& alphabet,
    const std::vector<int>& plane_A, const std::vector<int>& plane_B,
    int defect_limit, int alien_limit,
    int target_core32, int target_core48, int target_frame6, int target_line, int target_block,
    bool use_diagonal_exploit,
    const std::vector<uint8_t>& boundary_mask, 
    const std::vector<int>& public_key_grid);
void optimize_attack_E_universal_trap_tcv(
    std::vector<int>& forged_grid, int grid_size, const std::vector<Tile>& alphabet,
    const std::vector<int>& plane_A, const std::vector<int>& plane_B,
    int defect_limit, int alien_limit,
    int target_core32, int target_core48, int target_frame6, int target_line, int target_block,
    bool use_diagonal_exploit,
    const std::vector<uint8_t>& boundary_mask, 
    const std::vector<int>& public_key_grid);


// ---------------------------------------------------------
// GPU KERNELS
// ---------------------------------------------------------

// Version for NxN square grids using an attacker-constrained candidate bitmask + physical boundary constraints
// @param d_locked_mask The physical constraints (0=Free, 1=Colors, 2=ID).
// @param d_attacker_mask The attacker's predictions (0=Unknown, >0=Candidate Bits of allowed Tile IDs).
__global__ void heat_bath_kernel_candidate_constrained(
    int* d_grid, int grid_size, float temp, curandStatePhilox4_32_10_t* states, int is_black_phase, 
    const uint8_t* d_locked_mask,      // The physical constraints (0=Free, 1=Colors, 2=ID)
    const uint16_t* d_attacker_mask    // The attacker's predictions (0=Unknown, >0=Candidate Bits)
) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col >= grid_size || row >= grid_size) return;

    if (((row + col) % 2) == is_black_phase) {
        int id = row * grid_size + col;
        
        // 1. Evaluate Structural Boundary Constraints
        bool enforce_color = false;
        if (d_locked_mask) {
            uint8_t mask_val = d_locked_mask[id];
            if (mask_val == 2) return; // Rigidly pinned Tile ID (e.g., internal doping). Skip entirely.
            if (mask_val == 1) enforce_color = true; // Pin the published outward colors.
        }

        // 2. Fetch the Attacker's Winnowed Candidate Mask
        uint16_t allowed_mask = 0;
        if (d_attacker_mask) {
            allowed_mask = d_attacker_mask[id];
        }

        // Fast-Path: If the attacker perfectly proved exactly 1 tile is valid, lock it in
        // Skip this optimization on the perimeter to protect the Public Key colors.
        if (!enforce_color && allowed_mask != 0 && __popc(allowed_mask) == 1) {
            d_grid[id] = __ffs(allowed_mask) - 1; 
            return;
        }

        curandStatePhilox4_32_10_t local_state = states[id]; 
        float weights[QwtssConfig::MAX_TILES];
        float max_weight = -1e20f;
        int num_tiles = qwtss_core_shared_device::d_num_tiles_const;

        // Read required perimeter colors if geometrically constrained
        int req_top = -1, req_bottom = -1, req_left = -1, req_right = -1;
        if (enforce_color) {
            Tile t_current = qwtss_core_shared_device::d_tiles_const[d_grid[id]];
            if (row == 0) req_top = t_current.top;
            if (row == grid_size - 1) req_bottom = t_current.bottom;
            if (col == 0) req_left = t_current.left;
            if (col == grid_size - 1) req_right = t_current.right;
        }

        // 3. Calculate local energy only for tiles satisfying *both* constraints
        for (int i = 0; i < num_tiles; i++) {
            
            // FILTER A: Attacker's Candidate Mask
            if (allowed_mask != 0 && ((allowed_mask & (1 << i)) == 0)) {
                weights[i] = -1e20f; // Infinite penalty (0% probability)
                continue;             
            }

            // FILTER B: Public Key Boundary Colors
            if (enforce_color) {
                Tile cand = qwtss_core_shared_device::d_tiles_const[i];
                if ((req_top != -1 && cand.top != req_top) ||
                    (req_bottom != -1 && cand.bottom != req_bottom) ||
                    (req_left != -1 && cand.left != req_left) ||
                    (req_right != -1 && cand.right != req_right)) {
                    
                    weights[i] = -1e20f;
                    continue;
                }
            }

            int e = calculate_local_energy_gpu(d_grid, grid_size, row, col, i);
            weights[i] = - (float)e / temp;
            if (weights[i] > max_weight) max_weight = weights[i];
        }

        // 4. Softmax / Boltzmann Distribution
        float sum = 0.0f;
        for (int i = 0; i < num_tiles; i++) {
            weights[i] = expf(weights[i] - max_weight); 
            sum += weights[i];
        }

        // Safety check: ensure the stacked masks didn't create a mathematically impossible state.
        // If max_weight is still -1e20f, NO tiles passed the filters, and expf() will hallucinate 1.0 weights.
        if (max_weight > -1e19f) {
            float r = curand_uniform(&local_state) * sum;
            float cumulative = 0.0f;
            // CRITICAL: Fallback to the current tile (mathematically guaranteed 
            // to be valid) if we experience floating-point fall-through.
            int selected = d_grid[id];
            for (int i = 0; i < num_tiles; i++) {
                cumulative += weights[i];
                // ADDED SAFETY: Explicitly ensure 0-weight options can never be selected, 
                // even if 'r' lands exactly on a boundary edge.
                if (weights[i] > 0.0f && r <= cumulative) {
                    selected = i;
                    break;
                }
            }
            d_grid[id] = selected;
        }
        states[id] = local_state;
    }
}

void initialize_gpu_constants_from_crypto_core(const ITileSet* tileset) {
    std::vector<Tile> alphabet = tileset->get_tiles();
    int num_tiles = tileset->get_size();
    // Load constant memory
    CUDA_CHECK(cudaMemcpyToSymbol(qwtss_core_shared_device::d_tiles_const, alphabet.data(), num_tiles * sizeof(Tile)));
    CUDA_CHECK(cudaMemcpyToSymbol(qwtss_core_shared_device::d_num_tiles_const, &num_tiles, sizeof(int)));
}


// Returns the final defect count, or -1 if failed.
// Memory management is hoisted to the caller for massive ensemble performance.
GridAnnealResult grid_anneal_gpu_hoisted(
    int* h_grid,
    int grid_size,
    int* d_grid,
    curandStatePhilox4_32_10_t* d_states,
    ITileSet* tileset, 
    GridAnnealParams params,
    ChaCha20PRNG& rng,
    const std::vector<uint8_t>& external_lock_mask = {}
) {
    int grid_cells = grid_size * grid_size;
    std::vector<Tile> alphabet = tileset->get_tiles();
    int grid_bytes = grid_cells * sizeof(int);

    // Copy the CPU grid (which already has the boundaries locked) to the GPU
    CUDA_CHECK(cudaMemcpy(d_grid, h_grid, grid_bytes, cudaMemcpyHostToDevice));

    uint8_t* d_locked_mask = nullptr;
    if (!external_lock_mask.empty()) {
        CUDA_CHECK(cudaMalloc((void**)&d_locked_mask, external_lock_mask.size() * sizeof(uint8_t)));
        CUDA_CHECK(cudaMemcpy(d_locked_mask, external_lock_mask.data(), external_lock_mask.size() * sizeof(uint8_t), cudaMemcpyHostToDevice));
    } else if (params.do_pinned_defect_doping){
        int num_pins = std::min(params.max_allowed_defects, std::max(10, params.max_allowed_defects / 10));
        // Calculate a statistically safe min_radius for dart throw sampling
        double min_radius = std::sqrt((grid_cells) / (num_pins * M_PI)) * 0.8;

        std::vector<uint8_t> pinned_mask = generate_quenched_disorder_mask(grid_size, num_pins, min_radius, rng);

        // Allocate and copy to GPU
        CUDA_CHECK(cudaMalloc((void**)&d_locked_mask, pinned_mask.size() * sizeof(uint8_t)));
        CUDA_CHECK(cudaMemcpy(d_locked_mask, pinned_mask.data(), pinned_mask.size() * sizeof(uint8_t), cudaMemcpyHostToDevice));
    }

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((grid_size + 15) / 16, (grid_size + 15) / 16);

    // Setup Philox cuRAND States with Host Entropy Injection
    // Generate a cryptographically robust array of seeds for every cell in the grid
    std::vector<unsigned long long> h_seeds(grid_cells);
    std::uniform_int_distribution<unsigned long long> dist; 
    for (int i = 0; i < grid_cells; ++i) {
        h_seeds[i] = dist(rng);
    }

    // Push the seeds to the device
    unsigned long long* d_seeds;
    CUDA_CHECK(cudaMalloc(&d_seeds, grid_cells * sizeof(unsigned long long)));
    CUDA_CHECK(cudaMemcpy(d_seeds, h_seeds.data(), grid_cells * sizeof(unsigned long long), cudaMemcpyHostToDevice));

    // Initialize the PRNG states with the unique dynamic seeds
    init_curand_states<<<(grid_cells + 255) / 256, 256>>>(d_states, d_seeds, grid_size);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Free the seed array from VRAM; Philox doesn't need it anymore
    CUDA_CHECK(cudaFree(d_seeds));

    if (params.do_greedy_prequench)
    {
        // --- GREEDY PRE-QUENCH ---
        float quench_temp = 0.05f; 
        for (int step = 0; step < 50; step++) {
            heat_bath_kernel_unified<<<numBlocks, threadsPerBlock>>>(d_grid, grid_size, quench_temp, d_states, 0, d_locked_mask);
            heat_bath_kernel_unified<<<numBlocks, threadsPerBlock>>>(d_grid, grid_size, quench_temp, d_states, 1, d_locked_mask);
        }
    }

    // Start at reheat_temp because the grid is now heavily quenched
    float current_temp = (params.do_greedy_prequench ? params.reheat_temp : params.initial_temp);
    int last_defects = INT32_MAX;
    int stagnation_counter = 0;

    // --- MAIN OPTIMIZATION LOOP ---
    for (int step = 1; step <= params.max_steps; step++) { // <-- Start loop at 1
        // Only check AFTER a full interval
        if (step % params.check_interval == 0) {
            CUDA_CHECK(cudaMemcpy(h_grid, d_grid, grid_bytes, cudaMemcpyDeviceToHost));
            int defects = count_grid_defects(h_grid, grid_size, alphabet);

            // Goal condition met: The interior has settled into a valid configuration with tolerance
            if (defects <= params.max_allowed_defects && defects >= params.min_allowed_defects) {
                if (d_locked_mask) CUDA_CHECK(cudaFree(d_locked_mask));
                return  { defects, current_temp, step };
            }

            // TOO FEW DEFECTS: We are stuck below the floor
            if (defects < params.min_allowed_defects) {
                // Grid energy dropped below the floor. Spike the temperature to "re-melt" the crystal.
                //std::cout << "Bouncing off " << params.min_allowed_defects << " defects floor!" << std::endl;

                // Thermodynamic Rewind: Spike the temperature just enough so that after the next
                // check_interval, it decays back to exactly the current_temp ("micro-melt")
                float rewind_factor = std::pow(params.cooling_rate, -(float)params.check_interval);

                // Multiply the interval by 1.5 or 2.0 for slightly more "kick" 
                // to escape the local minimum, but 1.0 is the mathematically pure rewind.
                current_temp = current_temp * rewind_factor * 2.0f;

                // Safety clamp: Never over-melt the grid by exceeding the standard reheat ceiling
                if (current_temp > params.reheat_temp) {
                    current_temp = params.reheat_temp;
                }

                stagnation_counter = 0;
            }
            // STANDARD STAGNATION: We are stuck above the ceiling.
            else if (defects == last_defects) {
                stagnation_counter++;
            } else {
                stagnation_counter = 0;
            }
            last_defects = defects;

            // Thermal Reheat Injection
            if (stagnation_counter >= params.stagnation_patience) {
                //float orig_temp = current_temp;
                if (current_temp >= params.reheat_temp)
                    current_temp *= 2.0f;
                else
                    current_temp = std::max(params.reheat_temp, 1.2f * current_temp);
                // But cap at 10.0f
                if (current_temp > 10.0f) current_temp = 10.0f;
                stagnation_counter = 0;
                // std::cout << "Stagnation detected w/ defect count " << last_defects << "; will update temp from " << orig_temp
                //           << " to " << current_temp << " for re-heating" << std::endl;
            }
        }

        // Call the LOCKED kernels so the boundaries are never mutated
        heat_bath_kernel_unified<<<numBlocks, threadsPerBlock>>>(d_grid, grid_size, current_temp, d_states, 0, d_locked_mask);
        heat_bath_kernel_unified<<<numBlocks, threadsPerBlock>>>(d_grid, grid_size, current_temp, d_states, 1, d_locked_mask);
        // Check for kernel errors
        // cudaError_t err = cudaGetLastError();
        // if (err != cudaSuccess) {
        //     std::cout << "GPU Kernel Failed at Temp " << current_temp << ": " << cudaGetErrorString(err) << std::endl;
        // }

        current_temp *= params.cooling_rate;
    }

    if (d_locked_mask) CUDA_CHECK(cudaFree(d_locked_mask));

    //std::cout << "Failed with last defects: " << last_defects << std::endl;
    return  { -1, 0.0f, params.max_steps }; // FAILED
}

MarginalEntropyMetrics calculate_ensemble_marginal_entropy_metrics(
    const std::vector<int>& ground_truth_grid, 
    int grid_size,
    int ground_truth_defects,
    int defect_count_tolerance,
    int samples_required,
    ITileSet* tileset,
    ChaCha20PRNG& rng,
    bool do_greedy_prequench,
    const std::vector<uint8_t>& public_key_mask
) {
    if (public_key_mask.empty()) throw std::invalid_argument("public_key_mask should not be empty for this ensemble method");

    std::vector<Tile> alphabet = tileset->get_tiles();
    int num_tiles = alphabet.size();

    size_t grid_count = grid_size * grid_size;
    int grid_bytes = grid_count * sizeof(int);

    // 1. Hoist GPU Allocations
    int *d_grid;
    CUDA_CHECK(cudaMalloc((void**)&d_grid, grid_bytes));
    curandStatePhilox4_32_10_t* d_states;
    CUDA_CHECK(cudaMalloc(&d_states, grid_count * sizeof(curandStatePhilox4_32_10_t)));

    // Init constant memory using proper function to prevent the "Shadow Constant Memory" bug.
    // We initialize both here in case we end up calling kernels from either module
    initialize_gpu_constants_from_crypto_core(tileset);
    initialize_gpu_constants_from_core_shared(tileset);

    std::uniform_int_distribution<int> dist_tile(0, num_tiles - 1);

    int max_steps = 250000;
    GridAnnealParams params((ground_truth_defects + defect_count_tolerance),
        (ground_truth_defects - defect_count_tolerance), max_steps);
    params.do_greedy_prequench = do_greedy_prequench;

    std::uniform_real_distribution<double> dist_prob(0.0, 1.0);
    // Used for proper probabilistic edge tile defect seeding
    std::uniform_int_distribution<int> dist_target(ground_truth_defects - defect_count_tolerance, ground_truth_defects + defect_count_tolerance);

    // 2. Setup the Master Frequency Trackers
    // Dimensions: [Tile Index * num_tiles + Tile ID]
    std::vector<int> frequencies = std::vector<int>(grid_count * num_tiles, 0); // Tile frequencies per grid position

    std::vector<int> h_grid(grid_count);
    int successful_grids = 0;

    std::cout << "Collecting eligible grid samples..." << std::endl;

    while (successful_grids < samples_required) {
        // Pick a random thermodynamic target for THIS specific seed grid (for sake of boundary defect rate only)
        int run_target_defects = dist_target(rng);

        // Prepare the starting grid
        generate_boundary_conditioned_random_seed_grid(ground_truth_grid, grid_size, h_grid, run_target_defects,
            alphabet, rng, public_key_mask);

        // Run the Annealer
        GridAnnealResult result = grid_anneal_gpu_hoisted(
            h_grid.data(), grid_size, d_grid, d_states, tileset,
            params, rng,
            public_key_mask
        );

        int final_defects = result.final_defects;

        // Only accept grids in our specific tolerance band; otherwise we distort stats be forcing a mismatch between
        //  interior grid defect dynamics and the already locked in boundary for a different defect count.
        if (final_defects != -1 && final_defects >= (ground_truth_defects - defect_count_tolerance) &&
            final_defects <= (ground_truth_defects + defect_count_tolerance)) {

            successful_grids++;
            // Tally the results
            for (int idx = 0; idx < grid_count; idx++) {
                int settled_tile = h_grid[idx];
                frequencies[idx * num_tiles + settled_tile]++;
            }
            std::cout << "Successful grids captured: " << successful_grids << " / " << samples_required << "\r" << std::flush;
        }
    }
    std::cout << "Ensemble capture complete. " << successful_grids << " valid grids found." << std::endl;

    // 3. Cleanup GPU Memory
    CUDA_CHECK(cudaFree(d_grid));
    CUDA_CHECK(cudaFree(d_states));

    if (successful_grids == 0) return MarginalEntropyMetrics{};

    // 4. Calculate Final Marginal Entropy Curve, Metrics, and Heatmap
    double bucket_entropy = 0.0;
    double sum_sq_entropy = 0.0;
    double max_local_entropy = 0.0;
    std::vector<double> local_entropies(grid_count, 0.0);

    for (int i = 0; i < grid_count; i++) {
        double local_entropy = 0.0;
        for (int t = 0; t < num_tiles; t++) {
            int frequency = frequencies[i * num_tiles + t];
            if (frequency > 0) {
                double p = (double)frequency / successful_grids;
                local_entropy -= p * std::log2(p);
            }
        }

        bucket_entropy += local_entropy;
        sum_sq_entropy += (local_entropy * local_entropy);
        local_entropies[i] = local_entropy;

        if (local_entropy > max_local_entropy) {
            max_local_entropy = local_entropy;
        }
    }

    // Calculate Standard Deviation
    double mean_entropy = bucket_entropy / grid_count;
    double variance = (sum_sq_entropy / grid_count) - (mean_entropy * mean_entropy);
    // std::max(0.0) prevents NaN on tiny floating-point precision drifts near zero
    double std_dev_entropy = std::sqrt(std::max(0.0, variance));
    double max_theoretical_entropy = std::log2((float)num_tiles);

    // 4.5 Geometric Banding and Fault Line Analysis

    // Excludes outer boundary tiles so the cryptographic metrics are 
    // not skewed by the naturally low entropy of the Public Key constraints.
    int buffer = 2; // Exclude boundary (0, 63) AND the heavily correlated abutting tiles (1, 62)
    int start_idx = buffer;
    int end_idx = grid_size - buffer;
    int interior_span = end_idx - start_idx;

    std::vector<double> row_means(grid_size, 0.0);
    std::vector<double> col_means(grid_size, 0.0);

    // Accumulate strictly the interior entropy (ignoring edge buffer columns)
    for (int r = start_idx; r < end_idx; r++) {
        for (int c = start_idx; c < end_idx; c++) {
            double e = local_entropies[r * grid_size + c];
            row_means[r] += e;
            col_means[c] += e;
        }
    }

    // Convert to means and find the worst-case structural fault lines
    double min_row_entropy = max_theoretical_entropy;
    double min_col_entropy = max_theoretical_entropy;

    // Only search for fault lines inside the bulk
    for (int i = start_idx; i < end_idx; i++) {
        row_means[i] /= interior_span;
        col_means[i] /= interior_span;
        
        if (row_means[i] < min_row_entropy) min_row_entropy = row_means[i];
        if (col_means[i] < min_col_entropy) min_col_entropy = col_means[i];
    }

    // Metric 1: Fault Line Vulnerability (Lowest average entropy of any single axis)
    double fault_line_entropy = std::min(min_row_entropy, min_col_entropy);

    // Calculate Variance of the Row and Column means relative to the *interior* global mean
    double interior_mean = 0.0;
    for (int i = start_idx; i < end_idx; i++) {
        interior_mean += row_means[i];
    }
    interior_mean /= interior_span;

    double var_row = 0.0;
    double var_col = 0.0;
    for (int i = start_idx; i < end_idx; i++) {
        var_row += (row_means[i] - interior_mean) * (row_means[i] - interior_mean);
        var_col += (col_means[i] - interior_mean) * (col_means[i] - interior_mean);
    }
    var_row /= interior_span;
    var_col /= interior_span;
    // Safety clamp to prevent floating-point underflow
    var_row = std::max(0.0, var_row);
    var_col = std::max(0.0, var_col);

    // Metric 2: Banding Intensity (The raw variance of the dominant banding axis)
    double banding_intensity = std::max(var_row, var_col);

    // Metric 3: Directional Bias (Normalized ratio from -1.0 to 1.0)
    //  > 0.0 = Horizontal Banding dominance
    //  < 0.0 = Vertical Banding dominance
    //    0.0 = Isotropic (Symmetric)
    double directional_bias = 0.0;
    if (var_row + var_col > 1e-9) { // Prevent division by zero on perfectly uniform grids
        directional_bias = (var_row - var_col) / (var_row + var_col);
    }

    std::cout << "Bucket Center Defect Count: " << std::setw(3) << ground_truth_defects << " | Samples: " << std::setw(3) << successful_grids
                << " | Entropy: " << bucket_entropy << " bits\n";
    std::cout << "  -> Local Max Entropy: " << max_local_entropy << " bits (Theoretical max: " << std::fixed << std::setprecision(3) << max_theoretical_entropy << ")\n";
    std::cout << "  -> Local Std Dev:     " << std_dev_entropy << " bits\n";
    std::cout << "  -> Fault Line Entropy:" << std::setw(6) << std::setprecision(3) << fault_line_entropy << " bits (Worst-case continuous line)\n";
    std::cout << "  -> Banding Intensity: " << std::setw(6) << std::setprecision(4) << banding_intensity << " (Max structural variance)\n";
    std::cout << "  -> Directional Bias:  " << std::setw(6) << std::setprecision(3) << directional_bias << " (-1.0 Vertical <-> +1.0 Horizontal)\n";

    // Export 2D Heatmap to PPM (Portable Pixmap format)
    std::string ppm_filename = "";
    if (public_key_mask.empty()) {
        ppm_filename = "entropy_heatmap_" + std::to_string(ground_truth_defects) + "_defects.ppm";
    } else {
        double border_pct = get_border_kept_pct(public_key_mask, grid_size);
        std::stringstream ss;
        ss << "entropy_heatmap_" << ground_truth_defects << "_defects (" 
            << std::fixed << std::setprecision(1) << border_pct << "% border).ppm";
        ppm_filename = ss.str();
    }
    std::ofstream ppm_file("entropy heatmaps/" + ppm_filename);
    if (ppm_file.is_open()) {
        // P3 = ASCII RGB, followed by width, height, and max color value
        ppm_file << "P3\n" << grid_size << " " << grid_size << "\n255\n";

        for (int i = 0; i < grid_count; i++) {
            // Scale 0.0 -> 3.4594 into 0 -> 255 intensity
            int pixel_val = (int)((local_entropies[i] / max_theoretical_entropy) * 255.0);
            pixel_val = std::max(0, std::min(255, pixel_val)); // Clamp for safety
            
            // Write R G B (all same for grayscale)
            ppm_file << pixel_val << " " << pixel_val << " " << pixel_val << " ";
            
            // Newline at the end of every row
            if ((i + 1) % grid_size == 0) ppm_file << "\n";
        }
        ppm_file.close();
        std::cout << "  -> Exported heatmap to " << ppm_filename << "\n";
    }

    return MarginalEntropyMetrics { ground_truth_defects, successful_grids, bucket_entropy, max_local_entropy, std_dev_entropy,
        fault_line_entropy, banding_intensity, directional_bias
    };
}

AISMetrics calculate_ais_joint_entropy(
    const std::vector<int>& ground_truth_grid, 
    int grid_size,
    int ground_truth_defects,
    int defect_count_tolerance,
    int num_ais_chains,       // Number of parallel chains to run
    ITileSet* tileset,
    ChaCha20PRNG& rng,
    const std::vector<uint8_t>& public_key_mask
) {
    if (public_key_mask.empty()) throw std::invalid_argument("public_key_mask should not be empty for this ensemble method");

    std::vector<Tile> alphabet = tileset->get_tiles();
    int num_tiles = alphabet.size();
    size_t grid_count = grid_size * grid_size;

    int final_ceil = ground_truth_defects + defect_count_tolerance;
    int final_floor = ground_truth_defects - defect_count_tolerance;

    // Generate the Inverse Temperature (Beta) Schedule
    const int num_beta_steps = 12000; // 5000;  // The more steps, the more exact the Joint Entropy
    // beta_start must be exactly zero to reflect perfectly random start state, otherwise the integral math will be wrong
    const double beta_start = 0.0;    // High Initial Temp
    // fixed_beta_end must be be fixed at 11.0 for the JR-11 interpolation table to work correctly
    const double fixed_beta_end = 11.0;      // Very cold (Forces the grid to ~100 defects)
    const int sweeps_multiplier = 2;    // This must be a 1+ value. Higher values means better thermalization.

    std::uniform_int_distribution<int> dist_tile(0, num_tiles - 1);

    int grid_bytes = grid_count * sizeof(int);
    int *d_grid;
    CUDA_CHECK(cudaMalloc((void**)&d_grid, grid_bytes));

    uint8_t *d_locked_mask = nullptr;
    if (!public_key_mask.empty()) {
        CUDA_CHECK(cudaMalloc((void**)&d_locked_mask, public_key_mask.size() * sizeof(uint8_t)));
        CUDA_CHECK(cudaMemcpy(d_locked_mask, public_key_mask.data(), public_key_mask.size() * sizeof(uint8_t), cudaMemcpyHostToDevice));
    }

    // Setup Philox cuRAND States with Host Entropy Injection
    // Generate a cryptographically robust array of seeds for every cell in the grid
    std::vector<unsigned long long> h_seeds(grid_count);
    std::uniform_int_distribution<unsigned long long> dist; 
    for (int i = 0; i < grid_count; ++i) {
        h_seeds[i] = dist(rng);
    }

    // Push the seeds to the device
    unsigned long long* d_seeds;
    CUDA_CHECK(cudaMalloc(&d_seeds, grid_count * sizeof(unsigned long long)));
    CUDA_CHECK(cudaMemcpy(d_seeds, h_seeds.data(), grid_count * sizeof(unsigned long long), cudaMemcpyHostToDevice));

    // Allocate the Philox states and initialize them
    curandStatePhilox4_32_10_t* d_states;
    CUDA_CHECK(cudaMalloc(&d_states, grid_count * sizeof(curandStatePhilox4_32_10_t)));

    // Initialize the PRNG states with the unique dynamic seeds
    init_curand_states<<<(grid_count + 255) / 256, 256>>>(d_states, d_seeds, grid_size);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Free the seed array from VRAM; Philox doesn't need it anymore
    CUDA_CHECK(cudaFree(d_seeds));

    // Init constant memory using proper function to prevent the "Shadow Constant Memory" bug.
    // We initialize both here in case we end up calling kernels from either module
    initialize_gpu_constants_from_crypto_core(tileset);
    initialize_gpu_constants_from_core_shared(tileset);

    std::cout << "--- Starting AIS Joint Entropy Estimation ---" << std::endl;
    std::cout << "Target Defect Band: [" << final_floor << ", " << final_ceil << "]" << std::endl;
    std::cout << "Parallel Chains: " << num_ais_chains << std::endl;

    // Run the Scout Chain 8x to accurately estimate the exact empirical Beta end target
    double empirical_beta_end_estimates[8];
    int num_scout_chains = std::size(empirical_beta_end_estimates);
    for (int p = 0; p < num_scout_chains; ++p){
        std::vector<double> beta_schedule;
        for (int i = 0; i < num_beta_steps; i++) {
            // Geometric/Logarithmic spacing for Beta
            double fraction = (double)i / (num_beta_steps - 1);
            //double b = beta_start + (fixed_beta_end - beta_start) * std::pow(fraction, 3.0);
            // Using 0.5 (square root) forces large beta jumps in the liquid phase 
            // and tiny, high-resolution beta jumps in the glassy phase
            double b = beta_start + (fixed_beta_end - beta_start) * std::pow(fraction, 0.5);

            beta_schedule.push_back(b);
        }

        std::vector<int> scout_grid(grid_count);
        // Initialize Scout Grid Randomly at T=Infinity
        generate_boundary_conditioned_random_seed_grid(
            ground_truth_grid, grid_size, scout_grid, 0, alphabet, rng, public_key_mask
        );
        CUDA_CHECK(cudaMemcpy(d_grid, scout_grid.data(), grid_bytes, cudaMemcpyHostToDevice));

        // Target slightly below the center
        int scout_target_defects = ground_truth_defects - (defect_count_tolerance / 2);

        int scout_defects = count_grid_defects(scout_grid.data(), grid_size, alphabet);

        // Walk down the thermodynamic schedule
        bool worked = false;
        for (size_t step = 0; step < beta_schedule.size() - 1; ++step) {
            double current_beta = beta_schedule[step];
            double next_beta = beta_schedule[step + 1];

            // THERMALIZE THE GRID TO THE NEW VOLUME
            // Convert Beta back to Temperature (T = 1 / Beta). 
            // If Beta is 0, Temp is arbitrarily high (e.g. 1000.0)
            float step_temp = (next_beta == 0.0) ? 1000.0f : (float)(1.0 / next_beta);

            // AIS allows the sweeps to change based on temperature (just no early or conditional exits are allowed).
            // In AIS, the number of sweeps we perform at each temperature step does not need to be uniform. The mathematical
            // proof (Jarzynski's Equality) only requires that the transition kernel obeys Detailed Balance for the current
            // temperature. It does not care how many times we apply that kernel.
            int sweeps = 80;
            if (step_temp > 5.0f) sweeps = 1;       // Completely random selection
            else if (step_temp > 2.0f) sweeps = 5;  // Boiling
            else if (step_temp > 1.0f) sweeps = 10;
            else if (step_temp > 0.5f) sweeps = 40;
            else if (step_temp > 0.2f) sweeps = 60;
            else sweeps = 80;
            // Apply the global multiplier
            sweeps *= sweeps_multiplier;

            // Run N sweeps to reach the new equilibrium. If the beta steps are tiny, less sweeps will be needed.
            scout_defects = locked_ais_mcmc_step_gpu(
                scout_grid.data(), grid_size, d_grid, d_states, tileset, 
                step_temp, sweeps, d_locked_mask
            );
            CUDA_CHECK(cudaGetLastError());

            std::cout << "Scout Chain Progress: " << step << " / " << (beta_schedule.size() - 2)
                    << " | Scout Beta: " << std::fixed << std::setprecision(4) << current_beta
                    << " | Defects: " << scout_defects << "    \r" << std::flush;

            if (scout_defects <= scout_target_defects) {
                confirm_mask_constraints_honored(scout_grid, public_key_mask, grid_size, ground_truth_grid, alphabet, false);

                empirical_beta_end_estimates[p] = next_beta; //current_beta;
                std::cout << "Scout Chain successfully anchored target beta at: " << empirical_beta_end_estimates[p] << " (for "
                        << scout_defects << " defects)                               " << std::endl;
                worked = true;
                break;
            }
        }
        if (!worked) {
            std::cout << "Scout Chain failed to converge; retrying ...                                                " << std::endl;
            p--;
            continue;
        }
    }
    // Take the average of all the scout results
    double empirical_beta_end = std::accumulate(std::begin(empirical_beta_end_estimates), std::end(empirical_beta_end_estimates), 0.0) /
                                    (double)num_scout_chains;
    std::cout << "Final average Scout Chain target beta: " << empirical_beta_end << " (averaged over " << num_scout_chains << " chains)" << std::endl;

    // Generate the fixed, thermodynamic schedule (nothing can alter this; all physics are downstream)
    // We use a geometric spacing to put most of the steps in the critical cold region
    std::vector<double> beta_schedule;
    for (int i = 0; i < num_beta_steps; i++) {
        // Geometric/Logarithmic spacing for Beta
        double fraction = (double)i / (num_beta_steps - 1);
        //double b = beta_start + (fixed_beta_end - beta_start) * std::pow(fraction, 3.0);
        // Using 0.5 (square root) forces large beta jumps in the liquid phase 
        // and tiny, high-resolution beta jumps in the glassy phase
        double b = beta_start + (fixed_beta_end - beta_start) * std::pow(fraction, 0.5);

        // Truncate the schedule to meet the target defect count (so we don't compute too many wasted chains)
        // Only append steps that are mathematically before the target equilibrium point.
        if (b < empirical_beta_end) {
            beta_schedule.push_back(b);
        } else {
            // We crossed the threshold! Cap the final step at the exact target beta and stop.
            beta_schedule.push_back(empirical_beta_end);
            break;
        }
    }
    std::cout << "Cooling Steps per chain: " << beta_schedule.size() << "\n" << std::endl;

    // Accurately calculate the Unconstrained Entropy (Z_0) Prior
    // Because boundary tiles are now color-constrained (not tile-constrained), they often have 
    // multiple valid states at T=Infinity. We must sum the exact log2(states) of every position.
    double log_Z_0 = 0.0;
    for (int r = 0; r < grid_size; r++) {
        for (int c = 0; c < grid_size; c++) {
            int idx = r * grid_size + c;
            bool is_geometric_boundary = (r == 0 || r == grid_size - 1 || c == 0 || c == grid_size - 1);

            // Determine the constraint type: 0 = Free, 1 = Pinned Outward Colors, 2 = Pinned Tile ID
            uint8_t constraint_type = 0;
            if (!public_key_mask.empty()) {
                constraint_type = public_key_mask[idx];
            } else if (is_geometric_boundary) {
                constraint_type = 1; // Legacy fallback: treat geometric boundaries as Pinned Colors
            }

            if (constraint_type == 2) {
                // Pinned Tile ID: Exactly 1 valid state. log2(1) = 0.
                // Do nothing; it contributes 0 to the log_Z_0 prior.
            } else if (constraint_type == 1) {
                int true_tile = ground_truth_grid[idx];
                int valid_count = 0;
                // Count exactly how many tiles in the alphabet satisfy the Public Key colors here
                for (int t = 0; t < num_tiles; t++) {
                    bool preserves_pub_key = true;
                    if (r == 0 && alphabet[t].top != alphabet[true_tile].top) preserves_pub_key = false;
                    if (r == grid_size - 1 && alphabet[t].bottom != alphabet[true_tile].bottom) preserves_pub_key = false;
                    if (c == 0 && alphabet[t].left != alphabet[true_tile].left) preserves_pub_key = false;
                    if (c == grid_size - 1 && alphabet[t].right != alphabet[true_tile].right) preserves_pub_key = false;

                    if (preserves_pub_key) {
                        valid_count++;
                    }
                }
                if (valid_count > 0) {
                    log_Z_0 += std::log2((double)valid_count);
                }
            } else if (constraint_type == 0) {
                // Pure interior or unlocked boundary tile
                // Free tile: All tiles in the alphabet are valid at T=Infinity
                log_Z_0 += std::log2((double)num_tiles);
            } else {
                throw std::invalid_argument("Unexpected public_key_mask constraint type value.");
            }
        }
    }

    std::vector<double> log_weights(num_ais_chains, 0.0);

    // Track the total defects at each step across all chains to map the phase transition
    std::vector<long long> step_defect_totals(beta_schedule.size() - 1, 0);
    std::vector<bool> valid_chains(num_ais_chains, false);
    std::vector<int> final_chain_defects(num_ais_chains, 0);
    int num_valid_chains = 0;

    for (int chain = 0; chain < num_ais_chains; ++chain) {
        std::vector<int> h_grid(grid_count);

        // Initialize Chain Randomly at T=Infinity
        // Pass 0 for 'run_target_defects' for pure uniform sampling at T=Infinity 
        // across all valid boundary tiles, without any artificial defect skewing.
        generate_boundary_conditioned_random_seed_grid(
            ground_truth_grid, 
            grid_size, 
            h_grid, 
            0,            // 0 forced boundary defects for a pure T=Infinity prior
            alphabet, 
            rng, 
            public_key_mask // Possible partial boundary mask
        );
        CUDA_CHECK(cudaMemcpy(d_grid, h_grid.data(), grid_bytes, cudaMemcpyHostToDevice));

        double chain_log_weight = 0.0;
        int current_defects = count_grid_defects(h_grid.data(), grid_size, alphabet);
        // std::cout << "Chain " << std::setw(3) << chain + 1 << "/" << num_ais_chains << " | Initial Defects: "
        //           << current_defects << std::endl;

        // Walk down the thermodynamic schedule
        for (size_t step = 0; step < beta_schedule.size() - 1; ++step) {
            double current_beta = beta_schedule[step];
            double next_beta = beta_schedule[step + 1];

            // CALCULATE THE EXACT THERMODYNAMIC VOLUME REDUCTION
            // d(ln Z) = -E * d(beta)
            double delta_beta = next_beta - current_beta;
            double penalty_bits = -(delta_beta * current_defects) * 1.44269504; // convert to base-2 bits
            
            chain_log_weight += penalty_bits;

            // THERMALIZE THE GRID TO THE NEW VOLUME
            // Convert Beta back to Temperature (T = 1 / Beta). 
            // If Beta is 0, Temp is arbitrarily high (e.g. 1000.0)
            float step_temp = (next_beta == 0.0) ? 1000.0f : (float)(1.0 / next_beta);

            // AIS allows the sweeps to change based on temperature (just no early or conditional exits are allowed).
            // In AIS, the number of sweeps we perform at each temperature step does not need to be uniform. The mathematical
            // proof (Jarzynski's Equality) only requires that the transition kernel obeys Detailed Balance for the current
            // temperature. It does not care how many times we apply that kernel.
            int sweeps = 80;
            if (step_temp > 5.0f) sweeps = 1;       // Completely random selection
            else if (step_temp > 2.0f) sweeps = 5;  // Boiling
            else if (step_temp > 1.0f) sweeps = 10;
            else if (step_temp > 0.5f) sweeps = 40;
            else if (step_temp > 0.2f) sweeps = 60;
            else sweeps = 80;
            // Apply the global multiplier
            sweeps *= sweeps_multiplier;

            // Run N sweeps to reach the new equilibrium. If the beta steps are tiny, less sweeps will be needed.
            current_defects = locked_ais_mcmc_step_gpu(
                h_grid.data(), grid_size, d_grid, d_states, tileset, 
                step_temp, sweeps, d_locked_mask
            );
            CUDA_CHECK(cudaGetLastError());

            // Add to our global telemetry tracker
            step_defect_totals[step] += current_defects;

            std::cout << "Chain " << std::setw(3) << chain + 1 << "/" << num_ais_chains 
                      << " | Step " << std::setw(5) << step + 1 << "/" << beta_schedule.size() - 1
                      << " | Beta: " << std::fixed << std::setprecision(4) << std::setw(6) << next_beta
                      << " | Temp: " << std::setw(7) << std::setprecision(2) << step_temp
                      << " | Defects: " << std::setw(5) << current_defects 
                      << " | Weight: " << std::setprecision(2) << chain_log_weight
                      << " | Free Energy: " << (log_Z_0 + chain_log_weight) << " bits"
                      << "      \r" << std::flush;
        }

        // Check if the chain successfully reached the target band naturally
        bool chain_is_valid = false;
        if (current_defects > final_ceil || current_defects < final_floor) {
            valid_chains[chain] = false; // Chain missed the target band
        } else {
            confirm_mask_constraints_honored(h_grid, public_key_mask, grid_size, ground_truth_grid, alphabet, false);
            chain_is_valid = true;
            valid_chains[chain] = true;
            num_valid_chains++;
        }

        // Save the final defects for this specific chain
        final_chain_defects[chain] = current_defects;

        log_weights[chain] = chain_log_weight;
        
        // --- PROGRESS LOGGING (End of Chain) ---
        // Overwrite the inner loop log with a clean chain completion message
        std::cout << "Chain " << std::setw(3) << chain + 1 << "/" << num_ais_chains 
                  << " | Final Defects: " << current_defects << (chain_is_valid ? "*" : " ")
                  << " | Final Log Weight: " << std::fixed << std::setprecision(2) << chain_log_weight
                  << " | Final Free Energy: " << (log_Z_0 + chain_log_weight) << " bits"
                  << "                                          \n"; 
    }

    CUDA_CHECK(cudaFree(d_grid));
    CUDA_CHECK(cudaFree(d_states));
    if (d_locked_mask) CUDA_CHECK(cudaFree(d_locked_mask));

    // Export the Thermodynamic Cooling Curve (Energy vs. Beta Phase Transition Map)
    std::string curve_filename = "ais_cooling_curve_" + std::to_string(ground_truth_defects) + "_defects.csv";
    std::ofstream curve_file(curve_filename);
    if (curve_file.is_open()) {
        curve_file << "Step,Beta,Temperature,Avg_Defects\n";
        for (size_t i = 0; i < beta_schedule.size() - 1; ++i) {
            double avg_defects = (double)step_defect_totals[i] / num_ais_chains;
            float temp = (beta_schedule[i+1] == 0.0) ? 1000.0f : (float)(1.0 / beta_schedule[i+1]);
            curve_file << i << "," << std::fixed << std::setprecision(6) << beta_schedule[i+1] << "," 
                       << std::setprecision(2) << temp << "," 
                       << std::setprecision(2) << avg_defects << "\n";
        }
        curve_file.close();
    }

    // SAFETY CATCH: If all chains failed, abort the math entirely
    if (num_valid_chains == 0) {
        std::cout << "\n=== AIS Results ===" << std::endl;
        std::cout << "Starting Unconstrained Entropy (Z_0): " << log_Z_0 << " bits" << std::endl;
        std::cout << "Effective Sample Size (ESS): 0.00 / " << num_ais_chains << " chains" << std::endl;
        std::cout << "True Joint Entropy at " << ground_truth_defects << " defects: N/A (No chains survived)" << std::endl;
        return AISMetrics{ground_truth_defects, NAN, log_Z_0, 0.0, 0.0, 0};
    }

    // 5. Aggregate the weights using Log-Sum-Exp (This is the Log Partition Function, ln Z)

    // Find max log weight strictly among valid chains to prevent Log-Sum-Exp baseline shifting
    double max_log_weight = -1e300; 
    for (int i = 0; i < num_ais_chains; ++i) {
        if (valid_chains[i] && log_weights[i] > max_log_weight) {
            max_log_weight = log_weights[i];
        }
    }

    double sum_exp = 0.0;
    for (int i = 0; i < num_ais_chains; ++i) {
        if (valid_chains[i]) {
            sum_exp += std::pow(2.0, log_weights[i] - max_log_weight);
        }
    }

    // Note: We still divide by total num_ais_chains. This correctly punishes the aggregate
    // probability if many chains died (Jarzynski Equality)
    double log_average_weight = max_log_weight + std::log2(sum_exp) - std::log2((double)num_ais_chains);
    double log_Z_final_bits = log_Z_0 + log_average_weight;

    // 6. CALCULATE TRUE ENTROPY (S = ln Z + Beta * E)
    // We must add the Internal Energy back to get the actual structural entropy.
    // Calculate the average final defect count across the surviving chains.
    // Only average the internal energy of chains that converged within the target band.
    double avg_final_defects = 0.0;
    for (int i = 0; i < num_ais_chains; ++i) {
        if (valid_chains[i]) {
            avg_final_defects += final_chain_defects[i];
        }
    }
    avg_final_defects /= (double)num_valid_chains;

    double final_beta = beta_schedule.back(); // The exact beta we ended at
    
    // Internal Energy = Beta * E (multiplied by log2(e) to convert to bits)
    double internal_energy_bits = (final_beta * avg_final_defects) * 1.44269504;

    // S = ln Z + Beta * E
    // S = Free Energy + Internal Energy
    double final_joint_entropy = log_Z_final_bits + internal_energy_bits;

    // Calculate Effective Sample Size (ESS) strictly for valid chains
    double sum_w = 0.0;
    double sum_w_sq = 0.0;
    for (int i = 0; i < num_ais_chains; ++i) {
        if (valid_chains[i]) {
            double w = std::pow(2.0, log_weights[i] - max_log_weight);
            sum_w += w;
            sum_w_sq += (w * w);
        }
    }
    double ess = (sum_w * sum_w) / sum_w_sq;

    std::cout << "\n=== AIS Results ===" << std::endl;
    std::cout << "Starting Unconstrained Entropy (Z_0): " << log_Z_0 << " bits" << std::endl;
    std::cout << "Valid Survived Chains: " << num_valid_chains << " / " << num_ais_chains << std::endl;
    std::cout << "Effective Sample Size (ESS): " << std::fixed << std::setprecision(2) << ess << std::endl;
    std::cout << "True Joint Entropy at " << avg_final_defects << " defects: " << final_joint_entropy << " bits" << std::endl;

    return AISMetrics{(int)round(avg_final_defects), final_joint_entropy, log_Z_0, ess, max_log_weight, num_valid_chains};
}

TopologyMetrics calculate_grid_topology(const std::vector<int>& grid, int grid_size,
    const std::vector<Tile>& alphabet,
    const std::vector<uint8_t>& public_key_mask = {}) {
    int N = grid_size * grid_size;
    if (grid.size() != N) throw std::invalid_argument("Sanity check failed: (grid.size() != N)");
    std::vector<std::vector<int>> adj(N);

    // ------------------------------------------------------------------
    // 1. Build the Adjacency List (The Constraint Graph)
    // Edges added strictly for matching tile boundaries (zero-defect).
    // ------------------------------------------------------------------
    for (int r = 0; r < grid_size; ++r) {
        for (int c = 0; c < grid_size; ++c) {
            int u = r * grid_size + c;
            int tile_u = grid[u];

            // Check Right Neighbor
            if (c < grid_size - 1) {
                int v = r * grid_size + (c + 1);
                int tile_v = grid[v];
                if (alphabet[tile_u].right == alphabet[tile_v].left) {
                    adj[u].push_back(v);
                    adj[v].push_back(u);
                }
            }
            // Check Bottom Neighbor
            if (r < grid_size - 1) {
                int v = (r + 1) * grid_size + c;
                int tile_v = grid[v];
                if (alphabet[tile_u].bottom == alphabet[tile_v].top) {
                    adj[u].push_back(v);
                    adj[v].push_back(u);
                }
            }
        }
    }

    // ------------------------------------------------------------------
    // 2. Extract GCC and Cyclomatic Complexity via BFS
    // ------------------------------------------------------------------
    std::vector<bool> visited(N, false);
    int max_component_nodes = 0;
    int edges_in_gcc = 0;

    for (int i = 0; i < N; ++i) {
        if (!visited[i]) {
            int current_component_nodes = 0;
            int sum_of_degrees = 0;
            std::queue<int> q;
            
            q.push(i);
            visited[i] = true;

            while (!q.empty()) {
                int curr = q.front();
                q.pop();
                current_component_nodes++;
                
                // The degree inside the graph represents the connected edges
                sum_of_degrees += adj[curr].size();

                for (int neighbor : adj[curr]) {
                    if (!visited[neighbor]) {
                        visited[neighbor] = true;
                        q.push(neighbor);
                    }
                }
            }

            // Because every undirected edge is counted twice in sum_of_degrees
            int current_component_edges = sum_of_degrees / 2;

            // Track the largest component (The GCC)
            if (current_component_nodes > max_component_nodes) {
                max_component_nodes = current_component_nodes;
                edges_in_gcc = current_component_edges;
            }
        }
    }

    double gcc_mass = (double)max_component_nodes / N;

    // Cyclomatic Complexity: C = E - V + 1
    int cyclomatic_complexity = edges_in_gcc - max_component_nodes + 1;
    if (cyclomatic_complexity < 0) cyclomatic_complexity = 0; // Safety clamp for trees

    // Normalize C. A perfect grid of size L has (L-1)*(L-1) internal loops.
    int max_possible_loops = (grid_size - 1) * (grid_size - 1);
    double normalized_c = (double)cyclomatic_complexity / max_possible_loops;

    // ------------------------------------------------------------------
    // 3. Extract the 3-Core Rigidity Mass via Topological Pruning
    // ------------------------------------------------------------------
    std::vector<int> degrees(N);
    std::vector<bool> deleted(N, false);
    std::queue<int> prune_queue;
    int deleted_count = 0;

    // Initialize degrees WITH VIRTUAL PERIMETER ANCHORS and enqueue fragile tiles (< 3 connections)
    // Dirichlet boundary conditions ("virtual edges") prevent runaway topological peeling from the outside inward
    for (int r = 0; r < grid_size; ++r) {
        for (int c = 0; c < grid_size; ++c) {
            int i = r * grid_size + c;
            
            // Start with the true interior/defective connections
            int actual_connections = adj[i].size();
            
            // Determine constraint type (0 = Free, 1 = Pinned Colors, 2 = Pinned Tile ID)
            // Empty mask defaults entirely to 0 (Free)
            uint8_t constraint_type = 0;
            if (!public_key_mask.empty()) {
                constraint_type = public_key_mask[i];
            }

            // Add virtual connections based on the cryptographic constraint
            int virtual_connections = 0;
            if (constraint_type == 1) {
                // Pinned Outward Colors: Anchor the published geometric boundaries
                if (r == 0) virtual_connections++;               // North Color published
                if (r == grid_size - 1) virtual_connections++;   // South Color published
                if (c == 0) virtual_connections++;               // West Color published
                if (c == grid_size - 1) virtual_connections++;   // East Color published
            } else if (constraint_type == 2) {
                // Pinned Tile ID: Rigidly locked anchor. 
                // Pad its virtual degree to guarantee it survives 3-Core topological pruning.
                virtual_connections += 4;
            }

            // The tile must survive based strictly on its native connections + known colors
            degrees[i] = actual_connections + virtual_connections;

            // Enqueue truly fragile tiles
            if (degrees[i] < 3) {
                prune_queue.push(i);
                deleted[i] = true;
                deleted_count++;
            }
        }
    }

    // Propagate the destruction: if removing a tile causes its neighbor 
    // to drop below 3 connections, the neighbor collapses too.
    while (!prune_queue.empty()) {
        int u = prune_queue.front();
        prune_queue.pop();

        for (int v : adj[u]) {
            if (!deleted[v]) {
                degrees[v]--;
                if (degrees[v] < 3) {
                    prune_queue.push(v);
                    deleted[v] = true;
                    deleted_count++;
                }
            }
        }
    }

    int core3_nodes = N - deleted_count;
    double core3_mass = (double)core3_nodes / N;

    return TopologyMetrics{gcc_mass, core3_mass, normalized_c};
}

TopologyEnsembleTracker calculate_ensemble_topological_metrics(
    const std::vector<int>& ground_truth_grid, 
    int grid_size,
    int ground_truth_defects,
    int defect_count_tolerance,
    int samples_required,
    ITileSet* tileset,
    ChaCha20PRNG& rng,
    bool do_greedy_prequench,
    const std::vector<uint8_t>& public_key_mask
) {
    if (public_key_mask.empty()) throw std::invalid_argument("public_key_mask should not be empty for this ensemble method");

    std::vector<Tile> alphabet = tileset->get_tiles();
    int num_tiles = alphabet.size();

    size_t grid_count = grid_size * grid_size;
    int grid_bytes = grid_count * sizeof(int);

    // 1. Hoist GPU Allocations
    int *d_grid;
    CUDA_CHECK(cudaMalloc((void**)&d_grid, grid_bytes));
    // Allocate the Philox states (Initialization is automatically handled by grid_anneal_gpu_hoisted)
    curandStatePhilox4_32_10_t* d_states;
    CUDA_CHECK(cudaMalloc(&d_states, grid_count * sizeof(curandStatePhilox4_32_10_t)));

    // Init constant memory using proper function to prevent the "Shadow Constant Memory" bug.
    // We initialize both here in case we end up calling kernels from either module
    initialize_gpu_constants_from_crypto_core(tileset);
    initialize_gpu_constants_from_core_shared(tileset);

    int max_steps = 250000;
    GridAnnealParams params(ground_truth_defects + defect_count_tolerance, ground_truth_defects - defect_count_tolerance, max_steps);
    params.do_greedy_prequench = do_greedy_prequench;

    // Used for proper probabilistic edge tile defect seeding
    std::uniform_int_distribution<int> dist_target(ground_truth_defects - defect_count_tolerance, ground_truth_defects + defect_count_tolerance);

    // Estimate the Topological Metrics at this Defect Count using ensemble
    TopologyEnsembleTracker topo_tracker;
    // Just to prove uniqueness of ensemble samples
    std::unordered_set<uint64_t> seen_hashes;
    seen_hashes.reserve(samples_required);

    std::vector<int> h_grid(grid_count);
    int successful_grids = 0;

    std::cout << "Collecting eligible grid samples..." << std::endl;

    while (successful_grids < samples_required) {
        // Pick a random thermodynamic target for this specific sample (for edge defect probability)
        int run_target_defects = dist_target(rng);

        // Prepare the boundary-conditioned starting grid
        generate_boundary_conditioned_random_seed_grid(ground_truth_grid, grid_size, h_grid,
            run_target_defects, alphabet, rng, public_key_mask);

        // Run the Annealer
        GridAnnealResult result = grid_anneal_gpu_hoisted(
            h_grid.data(), grid_size, d_grid, d_states, tileset, 
            params, rng,
            public_key_mask
        );
        int final_defects = result.final_defects;
        if (final_defects == -1) {
            continue;
        }

        // Hash the newly generated grid
        uint64_t current_hash = hash_grid_state(h_grid.data(), grid_size);
        // Ensure uniqueness
        auto hash_result = seen_hashes.insert(current_hash);
        if (!hash_result.second) {
            // result.second is FALSE: The hash was already in the set
            std::cout << "Warning: Duplicate grid detected. This should almost never occur!" << std::endl;
            //throw std::runtime_error("Duplicate grid detected. This should almost never occur!");
        }

        // Only accept grids in our specific tolerance band; otherwise we distort stats due to forcing a mismatch between
        // interior grid defect dynamics and the already locked in boundary for a different defect count.
        if (final_defects != -1 && final_defects >= (ground_truth_defects - defect_count_tolerance) &&
            final_defects <= (ground_truth_defects + defect_count_tolerance)) {

            successful_grids++;

            // Extract the metrics for this single grid and accumulate the correct Dirichlet boundary physics
            TopologyMetrics current_topo_metrics = calculate_grid_topology(h_grid, grid_size, alphabet, public_key_mask);
            // Push to the accumulator
            topo_tracker.push(current_topo_metrics);

            std::cout << "Successful grids captured: " << successful_grids << " / " << samples_required << "\r" << std::flush;
        }
    }
    std::cout << "Ensemble capture complete. " << successful_grids << " valid grids found." << std::endl;

    // 3. Cleanup GPU Memory
    CUDA_CHECK(cudaFree(d_grid));
    CUDA_CHECK(cudaFree(d_states));

    if (successful_grids == 0) throw std::runtime_error("Should never happen.");

    return topo_tracker;
}

// Returns empty on failure
std::vector<int> generate_boundary_conditioned_private_key_hoisted(const std::vector<int>& ground_truth_grid, 
    int ground_truth_defects, const std::vector<uint8_t>& public_key_mask, int grid_size, GridAnnealParams params,
    int *d_grid, curandStatePhilox4_32_10_t* d_states, ITileSet* tileset, ChaCha20PRNG& rng){

    std::vector<Tile> alphabet = tileset->get_tiles();

    std::vector<int> h_grid(grid_size * grid_size);
    // Prepare the boundary-conditioned starting grid
    generate_boundary_conditioned_random_seed_grid(ground_truth_grid, grid_size, h_grid,
        ground_truth_defects, alphabet, rng, public_key_mask);

    // Run the Annealer
    GridAnnealResult result = grid_anneal_gpu_hoisted(
        h_grid.data(), grid_size, d_grid, d_states, tileset, 
        params, rng,
        public_key_mask
    );
    int final_defects = result.final_defects;
    if (final_defects == -1) {
        return {};
    }

    // Only accept grids in our specific tolerance band; otherwise we distort stats due to forcing a mismatch between
    //  interior grid defect dynamics and the already locked in boundary for a different defect count.
    if (final_defects != -1 && final_defects >= params.min_allowed_defects && final_defects <= params.max_allowed_defects) {
        return h_grid;
    } else {
        return {};
    }
}

// ---------- ATTACKER-ANALYSIS CODE ----------

AISMetrics calculate_ais_joint_entropy_attacker_constrained(
    const std::vector<int>& ground_truth_grid, 
    int grid_size,
    int ground_truth_defects,
    int defect_count_tolerance,
    int num_ais_chains,
    ITileSet* tileset,
    ChaCha20PRNG& rng,
    const std::vector<uint8_t>& structural_mask,
    const std::vector<uint16_t>& attacker_mask
) {
    if (structural_mask.empty() || attacker_mask.empty()) {
        throw std::invalid_argument("Both masks must be specified for attacker-constrained AIS.");
    }

    std::vector<Tile> alphabet = tileset->get_tiles();
    int num_tiles = alphabet.size();
    size_t grid_count = grid_size * grid_size;

    // The exact probability that any given tile in the grid contains a defect
    double p_dropout = (double)ground_truth_defects / (double)grid_count;
    std::uniform_real_distribution<double> dist_dropout(0.0, 1.0);

    int final_ceil = ground_truth_defects + defect_count_tolerance;
    int final_floor = ground_truth_defects - defect_count_tolerance;

    const int num_beta_steps = 5000;
    const double beta_start = 0.0;
    const double fixed_beta_end = 11.0;

    std::uniform_int_distribution<int> dist_tile(0, num_tiles - 1);

    int grid_bytes = grid_count * sizeof(int);
    int *d_grid;
    uint8_t *d_locked_mask = nullptr;
    uint16_t *d_attacker_mask = nullptr;

    CUDA_CHECK(cudaMalloc((void**)&d_grid, grid_bytes));
    
    // Copy the structural physics mask
    CUDA_CHECK(cudaMalloc((void**)&d_locked_mask, structural_mask.size() * sizeof(uint8_t)));
    CUDA_CHECK(cudaMemcpy(d_locked_mask, structural_mask.data(), structural_mask.size() * sizeof(uint8_t), cudaMemcpyHostToDevice));

    // Copy the attacker's prediction mask
    CUDA_CHECK(cudaMalloc((void**)&d_attacker_mask, attacker_mask.size() * sizeof(uint16_t)));

    // Setup Philox cuRAND States with Host Entropy Injection
    // Generate a cryptographically robust array of seeds for every cell in the grid
    std::vector<unsigned long long> h_seeds(grid_count);
    std::uniform_int_distribution<unsigned long long> dist; 
    for (int i = 0; i < grid_count; ++i) {
        h_seeds[i] = dist(rng);
    }

    // Push the seeds to the device
    unsigned long long* d_seeds;
    CUDA_CHECK(cudaMalloc(&d_seeds, grid_count * sizeof(unsigned long long)));
    CUDA_CHECK(cudaMemcpy(d_seeds, h_seeds.data(), grid_count * sizeof(unsigned long long), cudaMemcpyHostToDevice));

    // Allocate the Philox states and initialize them
    curandStatePhilox4_32_10_t* d_states;
    CUDA_CHECK(cudaMalloc(&d_states, grid_count * sizeof(curandStatePhilox4_32_10_t)));

    // Initialize the PRNG states with the unique dynamic seeds
    init_curand_states<<<(grid_count + 255) / 256, 256>>>(d_states, d_seeds, grid_size);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Free the seed array from VRAM; Philox doesn't need it anymore
    CUDA_CHECK(cudaFree(d_seeds));

    // Init constant memory using proper function to prevent the "Shadow Constant Memory" bug.
    // We initialize both here in case we end up calling kernels from either module
    initialize_gpu_constants_from_crypto_core(tileset);
    initialize_gpu_constants_from_core_shared(tileset);

    std::cout << "--- Starting Candidate-Constrained AIS Joint Entropy Estimation ---" << std::endl;
    std::cout << "Target Defect Band: [" << final_floor << ", " << final_ceil << "]" << std::endl;
    std::cout << "Parallel Chains: " << num_ais_chains << std::endl;

    // Prepare scout chain mask
    std::vector<uint16_t> scout_attacker_mask = attacker_mask;
    for (int i = 0; i < grid_count; ++i) {
        if (scout_attacker_mask[i] != 0 && dist_dropout(rng) < p_dropout) {
            scout_attacker_mask[i] = 0; 
        }
    }
    CUDA_CHECK(cudaMemcpy(d_attacker_mask, scout_attacker_mask.data(), 
                          scout_attacker_mask.size() * sizeof(uint16_t), cudaMemcpyHostToDevice));

    // Run the Scout Chain 3x to estimate the exact empirical Beta end target
    double empirical_beta_end_estimates[3];
    for (int p = 0; p < 3; ++p){
        std::vector<double> beta_schedule;
        for (int i = 0; i < num_beta_steps; i++) {
            double fraction = (double)i / (num_beta_steps - 1);
            double b = beta_start + (fixed_beta_end - beta_start) * std::pow(fraction, 3.0);
            beta_schedule.push_back(b);
        }

        std::vector<int> scout_grid(grid_count);
        // We use the structural mask to seed the geometry, but the attacker mask handles the rest
        generate_boundary_conditioned_random_seed_grid(
            ground_truth_grid, grid_size, scout_grid, 0, alphabet, rng, structural_mask
        );
        CUDA_CHECK(cudaMemcpy(d_grid, scout_grid.data(), grid_bytes, cudaMemcpyHostToDevice));

        int scout_target_defects = ground_truth_defects - (defect_count_tolerance / 2);
        int scout_defects = count_grid_defects(scout_grid.data(), grid_size, alphabet);

        bool worked = false;
        for (size_t step = 0; step < beta_schedule.size() - 1; ++step) {
            double current_beta = beta_schedule[step];
            double next_beta = beta_schedule[step + 1];
            float step_temp = (next_beta == 0.0) ? 1000.0f : (float)(1.0 / next_beta);

            int sweeps = 80;
            if (step_temp > 5.0f) sweeps = 1;
            else if (step_temp > 2.0f) sweeps = 5;
            else if (step_temp > 1.0f) sweeps = 10;
            else if (step_temp > 0.5f) sweeps = 40;
            else if (step_temp > 0.2f) sweeps = 60;
            else sweeps = 80;

            // ============================================================
            // INVOCATION OF THE NEW ATTACKER-CONSTRAINED KERNEL
            // ============================================================
            dim3 threadsPerBlock(16, 16);
            dim3 numBlocks((grid_size + 15) / 16, (grid_size + 15) / 16);
            
            for(int sw = 0; sw < sweeps; sw++) {
                heat_bath_kernel_candidate_constrained<<<numBlocks, threadsPerBlock>>>(
                    d_grid, grid_size, step_temp, d_states, 0, d_locked_mask, d_attacker_mask
                );
                heat_bath_kernel_candidate_constrained<<<numBlocks, threadsPerBlock>>>(
                    d_grid, grid_size, step_temp, d_states, 1, d_locked_mask, d_attacker_mask
                );
            }
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaMemcpy(scout_grid.data(), d_grid, grid_bytes, cudaMemcpyDeviceToHost));
            scout_defects = count_grid_defects(scout_grid.data(), grid_size, alphabet);

            std::cout << "Scout Chain Progress: " << step << " / " << (beta_schedule.size() - 2)
                    << " | Scout Beta: " << std::fixed << std::setprecision(4) << current_beta
                    << " | Defects: " << scout_defects << "    \r" << std::flush;

            if (scout_defects <= scout_target_defects) {
                empirical_beta_end_estimates[p] = next_beta;
                std::cout << "Scout Chain successfully anchored target beta at: " << empirical_beta_end_estimates[p] << " (for "
                        << scout_defects << " defects)                               " << std::endl;
                worked = true;
                break;
            }
        }
        if (!worked) {
            std::cout << "Scout Chain failed to converge; retrying ...                                                " << std::endl;
            p--;
            continue;
        }
    }

    double empirical_beta_end = (empirical_beta_end_estimates[0] + empirical_beta_end_estimates[1] + empirical_beta_end_estimates[2]) / 3.0;
    std::cout << "Final average Scout Chain target beta: " << empirical_beta_end << std::endl;

    std::vector<double> beta_schedule;
    for (int i = 0; i < num_beta_steps; i++) {
        double fraction = (double)i / (num_beta_steps - 1);
        double b = beta_start + (fixed_beta_end - beta_start) * std::pow(fraction, 3.0);
        if (b < empirical_beta_end) {
            beta_schedule.push_back(b);
        } else {
            beta_schedule.push_back(empirical_beta_end);
            break;
        }
    }

    std::vector<double> log_weights(num_ais_chains, 0.0);
    std::vector<long long> step_defect_totals(beta_schedule.size() - 1, 0);
    std::vector<bool> valid_chains(num_ais_chains, false);
    std::vector<int> final_chain_defects(num_ais_chains, 0);
    int num_valid_chains = 0;

    for (int chain = 0; chain < num_ais_chains; ++chain) {
        // ----------------------------------------------------------------
        // A. STOCHASTIC MASK DROPOUT
        // ----------------------------------------------------------------
        std::vector<uint16_t> chain_specific_attacker_mask = attacker_mask;
        for (int i = 0; i < grid_count; ++i) {
            if (chain_specific_attacker_mask[i] != 0) {
                // Drop the constraint with probability equal to the global defect rate
                if (dist_dropout(rng) < p_dropout) {
                    chain_specific_attacker_mask[i] = 0; // Unconstrained!
                }
            }
        }
        
        // Upload this chain's specific relaxed mask to the GPU
        CUDA_CHECK(cudaMemcpy(d_attacker_mask, chain_specific_attacker_mask.data(), 
                              chain_specific_attacker_mask.size() * sizeof(uint16_t), cudaMemcpyHostToDevice));

        // ----------------------------------------------------------------
        // B. CALCULATE EXACT T=INFINITY PRIOR FOR THIS SPECIFIC CHAIN
        // ----------------------------------------------------------------
        double chain_log_Z_0 = 0.0;
        for (int r = 0; r < grid_size; r++) {
            for (int c = 0; c < grid_size; c++) {
                int idx = r * grid_size + c;
                uint8_t struct_mask = structural_mask[idx];
                uint16_t att_mask = chain_specific_attacker_mask[idx];
                int true_tile = ground_truth_grid[idx];

                if (struct_mask == 2) continue; 

                int valid_count = 0;
                for (int t = 0; t < num_tiles; t++) {
                    if (att_mask != 0 && ((att_mask & (1 << t)) == 0)) continue;
                    
                    if (struct_mask == 1) {
                        if (r == 0 && alphabet[t].top != alphabet[true_tile].top) continue;
                        if (r == grid_size - 1 && alphabet[t].bottom != alphabet[true_tile].bottom) continue;
                        if (c == 0 && alphabet[t].left != alphabet[true_tile].left) continue;
                        if (c == grid_size - 1 && alphabet[t].right != alphabet[true_tile].right) continue;
                    }
                    valid_count++;
                }

                if (valid_count > 0) {
                    chain_log_Z_0 += std::log2((double)valid_count);
                } else {
                    int backup_count = 0;
                    for (int t = 0; t < num_tiles; t++) {
                        if (struct_mask == 1) {
                            if (r == 0 && alphabet[t].top != alphabet[true_tile].top) continue;
                            if (r == grid_size - 1 && alphabet[t].bottom != alphabet[true_tile].bottom) continue;
                            if (c == 0 && alphabet[t].left != alphabet[true_tile].left) continue;
                            if (c == grid_size - 1 && alphabet[t].right != alphabet[true_tile].right) continue;
                        }
                        backup_count++;
                    }
                    chain_log_Z_0 += std::log2((double)backup_count);
                }
            }
        }

        // ----------------------------------------------------------------
        // C. EXECUTE THE AIS CHAIN
        // ----------------------------------------------------------------
        std::vector<int> h_grid(grid_count);

        generate_boundary_conditioned_random_seed_grid(
            ground_truth_grid, grid_size, h_grid, 0, alphabet, rng, structural_mask
        );
        CUDA_CHECK(cudaMemcpy(d_grid, h_grid.data(), grid_bytes, cudaMemcpyHostToDevice));

        double chain_log_weight = 0.0;
        int current_defects = count_grid_defects(h_grid.data(), grid_size, alphabet);

        for (size_t step = 0; step < beta_schedule.size() - 1; ++step) {
            double current_beta = beta_schedule[step];
            double next_beta = beta_schedule[step + 1];

            double delta_beta = next_beta - current_beta;
            double penalty_bits = -(delta_beta * current_defects) * 1.44269504; 
            
            chain_log_weight += penalty_bits;

            float step_temp = (next_beta == 0.0) ? 1000.0f : (float)(1.0 / next_beta);

            int sweeps = 80;
            if (step_temp > 5.0f) sweeps = 1;
            else if (step_temp > 2.0f) sweeps = 5;
            else if (step_temp > 1.0f) sweeps = 10;
            else if (step_temp > 0.5f) sweeps = 40;
            else if (step_temp > 0.2f) sweeps = 60;
            else sweeps = 80;

            // ============================================================
            // INVOCATION OF THE NEW ATTACKER-CONSTRAINED KERNEL
            // ============================================================
            dim3 threadsPerBlock(16, 16);
            dim3 numBlocks((grid_size + 15) / 16, (grid_size + 15) / 16);
            
            for(int sw = 0; sw < sweeps; sw++) {
                heat_bath_kernel_candidate_constrained<<<numBlocks, threadsPerBlock>>>(
                    d_grid, grid_size, step_temp, d_states, 0, d_locked_mask, d_attacker_mask
                );
                heat_bath_kernel_candidate_constrained<<<numBlocks, threadsPerBlock>>>(
                    d_grid, grid_size, step_temp, d_states, 1, d_locked_mask, d_attacker_mask
                );
            }
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaMemcpy(h_grid.data(), d_grid, grid_bytes, cudaMemcpyDeviceToHost));
            current_defects = count_grid_defects(h_grid.data(), grid_size, alphabet);

            step_defect_totals[step] += current_defects;

            std::cout << "Chain " << std::setw(3) << chain + 1 << "/" << num_ais_chains 
                      << " | Step " << std::setw(5) << step + 1 << "/" << beta_schedule.size() - 1
                      << " | Temp: " << std::setw(7) << std::setprecision(2) << step_temp
                      << " | Defects: " << std::setw(5) << current_defects 
                      << " | Free Energy: " << (chain_log_Z_0 + chain_log_weight) << " bits"
                      << "      \r" << std::flush;
        }

        bool chain_is_valid = false;
        if (current_defects > final_ceil || current_defects < final_floor) {
            valid_chains[chain] = false; 
        } else {
            chain_is_valid = true;
            valid_chains[chain] = true;
            num_valid_chains++;
        }

        final_chain_defects[chain] = current_defects;
        // The weight of this chain includes its specific starting volume
        log_weights[chain] = chain_log_Z_0 + chain_log_weight;
        
        std::cout << "Chain " << std::setw(3) << chain + 1 << "/" << num_ais_chains 
                  << " | Final Defects: " << current_defects << (chain_is_valid ? "*" : " ")
                  << " | Final Free Energy: " << (chain_log_Z_0 + chain_log_weight) << " bits"
                  << "                                          \n"; 
    }

    CUDA_CHECK(cudaFree(d_grid));
    CUDA_CHECK(cudaFree(d_states));
    CUDA_CHECK(cudaFree(d_locked_mask));
    CUDA_CHECK(cudaFree(d_attacker_mask));

    if (num_valid_chains == 0) {
        // log_Z_0 is now dynamic per chain, so we set to zero here
        return AISMetrics{ground_truth_defects, NAN, 0.0, 0.0, 0.0, 0};
    }

    double max_log_weight = -1e300; 
    for (int i = 0; i < num_ais_chains; ++i) {
        if (valid_chains[i] && log_weights[i] > max_log_weight) {
            max_log_weight = log_weights[i];
        }
    }

    double sum_exp = 0.0;
    for (int i = 0; i < num_ais_chains; ++i) {
        if (valid_chains[i]) {
            sum_exp += std::pow(2.0, log_weights[i] - max_log_weight);
        }
    }

    double log_average_weight = max_log_weight + std::log2(sum_exp) - std::log2((double)num_ais_chains);
    double log_Z_final_bits = log_average_weight;

    double avg_final_defects = 0.0;
    for (int i = 0; i < num_ais_chains; ++i) {
        if (valid_chains[i]) {
            avg_final_defects += final_chain_defects[i];
        }
    }
    avg_final_defects /= (double)num_valid_chains;

    double final_beta = beta_schedule.back();
    double internal_energy_bits = (final_beta * avg_final_defects) * 1.44269504;
    double final_joint_entropy = log_Z_final_bits + internal_energy_bits;

    double sum_w = 0.0, sum_w_sq = 0.0;
    for (int i = 0; i < num_ais_chains; ++i) {
        if (valid_chains[i]) {
            double w = std::pow(2.0, log_weights[i] - max_log_weight);
            sum_w += w;
            sum_w_sq += (w * w);
        }
    }
    double ess = (sum_w * sum_w) / sum_w_sq;

    // log_Z_0 is now dynamic per chain, so we set to zero here
    return AISMetrics{(int)round(avg_final_defects), final_joint_entropy, 0.0, ess, max_log_weight, num_valid_chains};
}


struct AttackerTopologyResult {
    std::string topology_name;
    int total_defects;
    int alien_tiles;
    int frame_alien_tiles_2;

    int min_subgrid_defects;
    int max_subgrid_defects;

    int min_sliding_defects;
    int max_sliding_defects;

    int min_line_defects_seam;
    int max_line_defects_seam;

    int max_line_defects_tiles;      // Holds the max of (Row, Col)
    int max_diagonal_defects_tiles;  // Holds the max of (Left-Diag, Right-Diag)

    int frame_defects_4;
    int frame_defects_6;
    int frame_defects_8;

    int core_32;
    int core_48;

    // Stats for after greedy attacks. -1 if no greedy attack executed.
    int post_greedy_alien_tiles = -1;
    int post_greedy_total_defects = -1;
    int post_greedy_frame_alien_tiles_2 = -1;
    int post_greedy_core_48 = -1;
    int post_greedy_frame_defects_6 = -1;

    bool is_viable_threat;
};

int calculate_max_diagonal_defects(const std::vector<int>& grid, int grid_size, const std::vector<Tile>& alphabet) {
    int max_diag = 0;
    
    // There are 2*N - 1 diagonals in a square grid
    int num_diagonals = 2 * grid_size - 1;
    std::vector<int> left_diag_counts(num_diagonals, 0);
    std::vector<int> right_diag_counts(num_diagonals, 0);

    for (int r = 0; r < grid_size; r++) {
        for (int c = 0; c < grid_size; c++) {
            int tile_id = grid[r * grid_size + c];
            Tile t = alphabet[tile_id];
            
            // Re-calculate local energy/defects for this specific tile
            bool is_defective = false;
            if (r > 0 && t.top != alphabet[grid[(r - 1) * grid_size + c]].bottom) is_defective = true;
            if (r < grid_size - 1 && t.bottom != alphabet[grid[(r + 1) * grid_size + c]].top) is_defective = true;
            if (c > 0 && t.left != alphabet[grid[r * grid_size + (c - 1)]].right) is_defective = true;
            if (c < grid_size - 1 && t.right != alphabet[grid[r * grid_size + (c + 1)]].left) is_defective = true;

            if (is_defective) {
                // Map (r,c) to its specific Left (\) and Right (/) diagonal index
                int left_idx = r - c + (grid_size - 1); 
                int right_idx = r + c;
                
                left_diag_counts[left_idx]++;
                right_diag_counts[right_idx]++;
            }
        }
    }

    for (int count : left_diag_counts) max_diag = std::max(max_diag, count);
    for (int count : right_diag_counts) max_diag = std::max(max_diag, count);

    return max_diag;
}

void calculate_all_axis_defects(const std::vector<int>& grid, int grid_size, const std::vector<Tile>& alphabet, 
                                int& max_line, int& max_diag) {
    std::vector<int> row_counts(grid_size, 0);
    std::vector<int> col_counts(grid_size, 0);
    std::vector<int> left_diag_counts(2 * grid_size - 1, 0);
    std::vector<int> right_diag_counts(2 * grid_size - 1, 0);

    for (int r = 0; r < grid_size; r++) {
        for (int c = 0; c < grid_size; c++) {
            int tile_id = grid[r * grid_size + c];
            Tile t = alphabet[tile_id];
            
            // Mark tile if ANY of its 4 edges are mismatched
            bool is_defective = false;
            if (r > 0 && t.top != alphabet[grid[(r - 1) * grid_size + c]].bottom) is_defective = true;
            if (r < grid_size - 1 && t.bottom != alphabet[grid[(r + 1) * grid_size + c]].top) is_defective = true;
            if (c > 0 && t.left != alphabet[grid[r * grid_size + (c - 1)]].right) is_defective = true;
            if (c < grid_size - 1 && t.right != alphabet[grid[r * grid_size + (c + 1)]].left) is_defective = true;

            if (is_defective) {
                row_counts[r]++;
                col_counts[c]++;
                
                int left_idx = r - c + (grid_size - 1); 
                int right_idx = r + c;
                left_diag_counts[left_idx]++;
                right_diag_counts[right_idx]++;
            }
        }
    }

    max_line = 0;
    for (int count : row_counts) max_line = std::max(max_line, count);
    for (int count : col_counts) max_line = std::max(max_line, count);

    max_diag = 0;
    for (int count : left_diag_counts) max_diag = std::max(max_diag, count);
    for (int count : right_diag_counts) max_diag = std::max(max_diag, count);
}

void run_custom_topology_attack_sweep(
    const std::vector<int>& plane_A,
    const std::vector<int>& plane_B,
    const std::vector<int>& plane_C,
    const std::vector<int>& boundary_source_map, // length: 4*grid_size - 4, values: 0 for A, 1 for B
    int grid_size,
    ITileSet* tileset,
    ChaCha20PRNG& rng) 
{
    std::vector<Tile> alphabet = tileset->get_tiles();

    std::vector<AttackerTopologyResult> results;
    int perimeter_len = grid_size * 4 - 4;

    // Map 1D perimeter index back to 2D grid coordinates
    auto get_perimeter_coords = [&](int idx, int& r, int& c) {
        if (idx < grid_size) { r = 0; c = idx; }
        else if (idx < 2 * grid_size - 1) { r = idx - grid_size + 1; c = grid_size - 1; }
        else if (idx < 3 * grid_size - 2) { r = grid_size - 1; c = (grid_size - 1) - (idx - (2 * grid_size - 2)); }
        else { r = (grid_size - 1) - (idx - (3 * grid_size - 3)); c = 0; }
    };

    std::vector<uint8_t> boundary_mask = generate_partial_boundary_mask(grid_size, 0.70f, rng);

    // Cache the true Public Key grid so the attacker knows the required edge colors
    std::vector<int> public_key_grid(grid_size * grid_size, 0);
    for (int p = 0; p < perimeter_len; p++) {
        int pr, pc;
        get_perimeter_coords(p, pr, pc);
        if (boundary_source_map[p] == 0) public_key_grid[pr * grid_size + pc] = plane_A[pr * grid_size + pc];
        else public_key_grid[pr * grid_size + pc] = plane_B[pr * grid_size + pc];
    }

    int current_evaluation = 0;
    int total_evaluations = 94;

    // Evaluates any pre-built forged grid
    auto evaluate_grid = [&](const std::string& name, std::vector<int>& forged_grid) {
        current_evaluation++;
        std::cout << "\r[*] Progress: " << current_evaluation << " / " << total_evaluations 
                  << " | Evaluating: " << name 
                  << "                                        " << std::flush;

        // ENFORCE THE FIXED PUBLIC KEY BOUNDARY
        for (int p = 0; p < perimeter_len; p++) {
            int pr, pc;
            get_perimeter_coords(p, pr, pc);
            if (boundary_source_map[p] == 0) {
                forged_grid[pr * grid_size + pc] = plane_A[pr * grid_size + pc];
            } else {
                forged_grid[pr * grid_size + pc] = plane_B[pr * grid_size + pc];
            }
        }

        AttackerTopologyResult res;
        res.topology_name = name;
        res.total_defects = count_grid_defects(forged_grid.data(), grid_size, alphabet);
        res.alien_tiles = calculate_alien_tiles(forged_grid, plane_A, plane_B, grid_size);
        res.frame_alien_tiles_2 = calculate_alien_frame_count(forged_grid, plane_A, plane_B, grid_size, 2);

        res.frame_defects_4 = calculate_defect_spatial_distribution_frame(forged_grid, grid_size, alphabet, 4);
        res.frame_defects_6 = calculate_defect_spatial_distribution_frame(forged_grid, grid_size, alphabet, 6);
        res.frame_defects_8 = calculate_defect_spatial_distribution_frame(forged_grid, grid_size, alphabet, 8);

        res.core_32 = calculate_core_defects(forged_grid, grid_size, alphabet, 32);
        res.core_48 = calculate_core_defects(forged_grid, grid_size, alphabet, 48);

        // Execute greedy optimizer attacks
        bool did_greedy_attack = false;

        // If it's a low-defect attack, try to pump the Alien count up
        if (QwtssZkPoWTestConstraints::total_defects_upper_bound != -1 && res.total_defects <= QwtssZkPoWTestConstraints::total_defects_upper_bound) {
            optimize_attack_A_maximize_aliens(forged_grid, grid_size, alphabet, plane_A, plane_B, QwtssZkPoWTestConstraints::total_defects_upper_bound,
                boundary_mask, public_key_grid);
            did_greedy_attack = true;
        }
        // If it's a high-alien attack, try to heal the defects down
        if (QwtssZkPoWTestConstraints::alien_tiles_lower_bound != -1 && res.alien_tiles >= QwtssZkPoWTestConstraints::alien_tiles_lower_bound) {
            optimize_attack_B_minimize_defects(forged_grid, grid_size, alphabet, plane_A, plane_B, QwtssZkPoWTestConstraints::alien_tiles_lower_bound,
                boundary_mask, public_key_grid);
            did_greedy_attack = true;
        }

        int intermediate_defects = count_grid_defects(forged_grid.data(), grid_size, alphabet);
        int intermediate_aliens = calculate_alien_tiles(forged_grid, plane_A, plane_B, grid_size);

        /*
        // Only trigger Attack C if the grid is currently a viable "Sneak-Through"
        if (intermediate_defects <= defect_count_upper_bound && intermediate_aliens >= alien_tiles_lower_bound) {
            optimize_attack_C_maximize_frame_aliens(forged_grid, grid_size, alphabet, plane_A, plane_B, defect_count_upper_bound,
                alien_tiles_lower_bound, 2, boundary_mask, public_key_grid);
            did_greedy_attack = true;
        }
        */

        /*
        // Only trigger Attack D if the grid is currently a viable "Sneak-Through"
        if (intermediate_defects <= defect_count_upper_bound && intermediate_aliens >= alien_tiles_lower_bound) {
            optimize_attack_D_spatial_trap(forged_grid, grid_size, alphabet, plane_A, plane_B, defect_count_upper_bound,
                alien_tiles_lower_bound, boundary_mask, public_key_grid);
            did_greedy_attack = true;
        }
        */

        bool use_diagonal_exploit = true;
        if ((QwtssZkPoWTestConstraints::total_defects_upper_bound == -1 || intermediate_defects <= QwtssZkPoWTestConstraints::total_defects_upper_bound) &&
            (QwtssZkPoWTestConstraints::alien_tiles_lower_bound == -1 || intermediate_aliens >= QwtssZkPoWTestConstraints::alien_tiles_lower_bound)) {
            // Note: The optimize_attack_E_universal_trap_tcv() version is a theoretically stronger attacker than the non-TCV version
            //optimize_attack_E_universal_trap(
            optimize_attack_E_universal_trap_tcv(
                forged_grid, grid_size, alphabet, plane_A, plane_B, 
                QwtssZkPoWTestConstraints::total_defects_upper_bound,
                QwtssZkPoWTestConstraints::alien_tiles_lower_bound,
                QwtssZkPoWTestConstraints::core32_lower_bound,
                QwtssZkPoWTestConstraints::core48_lower_bound, // 45
                QwtssZkPoWTestConstraints::frame6_upper_bound, // 140
                QwtssZkPoWTestConstraints::line_max_upper_bound, 
                QwtssZkPoWTestConstraints::block8_max_upper_bound,
                use_diagonal_exploit, boundary_mask, public_key_grid);
            did_greedy_attack = true;
        }

        if (did_greedy_attack){
            // Recalculate various metrics that could be affected by the greedy optimizers changing the grid
            res.post_greedy_total_defects = count_grid_defects(forged_grid.data(), grid_size, alphabet);
            res.post_greedy_alien_tiles = calculate_alien_tiles(forged_grid, plane_A, plane_B, grid_size);
            res.post_greedy_frame_alien_tiles_2 = calculate_alien_frame_count(forged_grid, plane_A, plane_B, grid_size, 2);
            res.post_greedy_core_48 = calculate_core_defects(forged_grid, grid_size, alphabet, 48);
            res.post_greedy_frame_defects_6 = calculate_defect_spatial_distribution_frame(forged_grid, grid_size, alphabet, 6);
        }

        int sub_grid_size = 8;
        DefectDistributionStats subgrid_stats = calculate_defect_spatial_distribution_subgrids(
            forged_grid, grid_size, alphabet, sub_grid_size);
        res.min_subgrid_defects = subgrid_stats.min_defects;
        res.max_subgrid_defects = subgrid_stats.max_defects;

        // 2. Local Sparsity (3x3 Sliding Window)
        int sliding_window_size = 3;
        DefectDistributionStats sliding_stats = calculate_defect_spatial_distribution_sliding(
            forged_grid, grid_size, alphabet, sliding_window_size);
        res.min_sliding_defects = sliding_stats.min_defects;
        res.max_sliding_defects = sliding_stats.max_defects;

        DefectDistributionStats lines_stats = calculate_defect_spatial_distribution_lines(forged_grid, grid_size, alphabet);
        //if (res.total_defects != lines_stats.total_defects) throw std::runtime_error("(res.total_defects != lines_stats.total_defects)");
        res.min_line_defects_seam = lines_stats.min_defects;
        res.max_line_defects_seam = lines_stats.max_defects;

        // Use the unified tile-density counter
        calculate_all_axis_defects(forged_grid, grid_size, alphabet, res.max_line_defects_tiles, res.max_diagonal_defects_tiles);

        // Attack is viable if the pre-greedy stats or the post-greedy stats slip through the threshold (Sneak-Through)
        res.is_viable_threat = (((QwtssZkPoWTestConstraints::total_defects_upper_bound != -1 && res.total_defects <= QwtssZkPoWTestConstraints::total_defects_upper_bound)
                                && (QwtssZkPoWTestConstraints::alien_tiles_lower_bound != -1 && res.alien_tiles >= QwtssZkPoWTestConstraints::alien_tiles_lower_bound))
                            || ((QwtssZkPoWTestConstraints::total_defects_upper_bound != -1 && res.post_greedy_total_defects <= QwtssZkPoWTestConstraints::total_defects_upper_bound)
                                && (QwtssZkPoWTestConstraints::alien_tiles_lower_bound != -1 && res.post_greedy_alien_tiles >= QwtssZkPoWTestConstraints::alien_tiles_lower_bound)));
        results.push_back(res);

        // Export specific topological grids to PPM
        //if (name.find("Voronoi_Chebyshev_Square") != std::string::npos)
        //if (false && res.is_viable_threat)
        int expected_defects = (res.post_greedy_total_defects != -1 ? res.post_greedy_total_defects : res.total_defects);
        if (false && (expected_defects <= 55 || name.find("Plane_C_Grid_Flood") != std::string::npos)) {
            std::cout << "\n  -> Exporting visual PPM image for " << name << "..." << std::endl;
            export_defect_edges_to_ppm(forged_grid.data(), grid_size, tileset, name + " - " + std::to_string(expected_defects) + " defects.ppm");
            if (name.find("Plane_C_Grid_Flood") != std::string::npos){
                export_alien_tiles_to_ppm(forged_grid.data(), grid_size, plane_A, plane_B, 
                    name + " - " + std::to_string(expected_defects) + " alien tiles.ppm");
            }
        }
    };

    // Builds the 2-plane grid and passes it to the evaluator
    auto test_mask = [&](const std::string& name, const std::vector<bool>& mask) {
        std::vector<int> forged_grid(grid_size * grid_size);
        for (int i = 0; i < grid_size * grid_size; i++) {
            forged_grid[i] = mask[i] ? plane_A[i] : plane_B[i];
        }
        evaluate_grid(name, forged_grid);
    };

    // =========================================================================
    // TOPOLOGY 1: Voronoi Distance Fields (L1 / L2 / Chebyshev)
    // This perfectly models "diagonal lines coming from the splice points"
    // connecting into a central diamond or shape.
    // =========================================================================
    for (int metric = 0; metric < 3; metric++) {
        // Bias parameter to expand/shrink the core A/B regions
        for (double bias = -5.0; bias <= 5.0; bias += 1.0) {
            std::vector<bool> mask(grid_size * grid_size);
            for (int r = 0; r < grid_size; r++) {
                for (int c = 0; c < grid_size; c++) {
                    double dist_A = 99999.0, dist_B = 99999.0;
                    
                    for (int p = 0; p < perimeter_len; p++) {
                        int pr, pc;
                        get_perimeter_coords(p, pr, pc);
                        
                        double d = 0;
                        if (metric == 0) d = std::abs(r - pr) + std::abs(c - pc); // L1 (Diamond/Diagonal)
                        else if (metric == 1) d = std::sqrt((r - pr)*(r - pr) + (c - pc)*(c - pc)); // L2 (Circular)
                        else d = std::max(std::abs(r - pr), std::abs(c - pc)); // Chebyshev (Square)
                        
                        if (boundary_source_map[p] == 0) dist_A = std::min(dist_A, d);
                        else dist_B = std::min(dist_B, d);
                    }
                    
                    // NEW: Smoothly taper the bias to 0 as it approaches the boundary.
                    // This guarantees the seams mathematically anchor to the exact splice points 
                    // without causing disjointed gaps when the perimeter is enforced!
                    int min_dist_to_boundary = std::min({r, c, grid_size - 1 - r, grid_size - 1 - c});
                    double effective_bias = bias * ((double)min_dist_to_boundary / (grid_size / 2.0));

                    mask[r * grid_size + c] = (dist_A + effective_bias <= dist_B);
                }
            }
            std::string m_name = (metric == 0) ? "L1_Diamond" : (metric == 1) ? "L2_Circular" : "Chebyshev_Square";
            test_mask("Voronoi_" + m_name + "_Bias_" + std::to_string(static_cast<int>(bias)), mask);
        }
    }

    // =========================================================================
    // TOPOLOGY 2: Central Diamond with Radial Rays
    // =========================================================================
    for (int radius = 10; radius <= 30; radius += 5) {
        // Toggle: 0 for Original Native Core (Plane A), 1 for Alien Core (Plane C)
        for (int use_plane_c = 0; use_plane_c <= 1; use_plane_c++) {
            std::vector<int> forged_grid(grid_size * grid_size);
            
            for (int r = 0; r < grid_size; r++) {
                for (int c = 0; c < grid_size; c++) {
                    int center_r = grid_size / 2;
                    int center_c = grid_size / 2;
                    int idx = r * grid_size + c;
                    
                    if (std::abs(r - center_r) + std::abs(c - center_c) <= radius) {
                        // Core of the diamond
                        if (use_plane_c) {
                            forged_grid[idx] = plane_C[idx]; // NEW: Flood with Alien Tiles
                        } else {
                            forged_grid[idx] = plane_A[idx]; // OLD: Native Plane A (replicates test_mask true)
                        }
                    } else {
                        // Outer radial rays connecting to perimeter
                        double min_d = 99999.0;
                        int best_source = 0;
                        for (int p = 0; p < perimeter_len; p++) {
                            int pr, pc;
                            get_perimeter_coords(p, pr, pc);
                            double d = std::sqrt((r - pr)*(r - pr) + (c - pc)*(c - pc));
                            if (d < min_d) {
                                min_d = d;
                                best_source = boundary_source_map[p];
                            }
                        }
                        forged_grid[idx] = (best_source == 0) ? plane_A[idx] : plane_B[idx];
                    }
                }
            }
            std::string name = "Central_Diamond_Radius_" + std::to_string(radius) + (use_plane_c ? "_PlaneC" : "_Native");
            evaluate_grid(name, forged_grid);
        }
    }

    // =========================================================================
    // TOPOLOGY 3 & 4: Inner Square (Plane C Core) with Straight or Jagged Seams
    // This pushes a solid Plane C square deep into the grid, forcing native seams 
    // to connect the perimeter splice points directly to the walls of the alien core.
    // =========================================================================
    for (int depth = 1; depth <= 17; depth += 1) {
        for (int jagged = 0; jagged <= 1; jagged++) {
            std::vector<int> forged_grid(grid_size * grid_size);
            
            for (int r = 0; r < grid_size; r++) {
                for (int c = 0; c < grid_size; c++) {
                    int r_eff = r;
                    int c_eff = c;
                    
                    if (jagged) {
                        // Global Spatial Shear
                        int dr = r - (grid_size / 2);
                        int dc = c - (grid_size / 2);
                        r_eff = r + (dc / 7);
                        c_eff = c + (dr / 7);
                    }

                    int min_dist_to_boundary = std::min({r_eff, c_eff, grid_size - 1 - r_eff, grid_size - 1 - c_eff});
                    
                    if (min_dist_to_boundary >= depth) {
                        // The Core Inner Square is now populated by Plane C
                        forged_grid[r * grid_size + c] = plane_C[r * grid_size + c]; 
                    } else {
                        // The Outer Extruding Zone (Assigns to closest perimeter splice point)
                        double min_d = 99999.0;
                        int best_source = 0;
                        
                        for (int p = 0; p < perimeter_len; p++) {
                            int pr, pc;
                            get_perimeter_coords(p, pr, pc);
                            
                            int pr_eff = pr;
                            int pc_eff = pc;
                            
                            if (jagged) {
                                int dpr = pr - (grid_size / 2);
                                int dpc = pc - (grid_size / 2);
                                pr_eff = pr + (dpc / 7);
                                pc_eff = pc + (dpr / 7);
                            }
                            
                            double d = std::abs(r_eff - pr_eff) + std::abs(c_eff - pc_eff);
                            
                            if (d < min_d) {
                                min_d = d;
                                best_source = boundary_source_map[p];
                            }
                        }
                        // Assign Plane A or Plane B based on the Voronoi best_source
                        forged_grid[r * grid_size + c] = (best_source == 0) ? plane_A[r * grid_size + c] : plane_B[r * grid_size + c];
                    }
                }
            }
            std::string name = "Inner_Square_Depth_" + std::to_string(depth) + (jagged ? "_Jagged" : "_Straight");
            
            // Pass the directly constructed grid to evaluate_grid
            evaluate_grid(name, forged_grid);
        }
    }

    // =========================================================================
    // TOPOLOGY 5: 3rd Plane Grid Flood (Inner Core is Plane C)
    // Connects a random valid Labbé grid core directly to the perimeter 
    // splice points using the shortest straight paths (Voronoi projection).
    // =========================================================================
    for (int inner_size = 30; inner_size <= 62; inner_size += 2) {
        std::vector<int> forged_grid(grid_size * grid_size);
        int depth = (grid_size - inner_size) / 2;

        for (int r = 0; r < grid_size; r++) {
            for (int c = 0; c < grid_size; c++) {
                int min_dist_to_boundary = std::min({r, c, grid_size - 1 - r, grid_size - 1 - c});
                
                if (min_dist_to_boundary >= depth) {
                    // Core Flood: Fill the center with the 3rd Labbé phase
                    forged_grid[r * grid_size + c] = plane_C[r * grid_size + c];
                } else {
                    // Outer Extruding Zone: Connect to the closest splice point
                    double min_d = 99999.0;
                    int best_source = 0;
                    
                    for (int p = 0; p < perimeter_len; p++) {
                        int pr, pc;
                        get_perimeter_coords(p, pr, pc);
                        
                        // L1 (Manhattan) shortest straight path
                        double d = std::abs(r - pr) + std::abs(c - pc);
                        if (d < min_d) {
                            min_d = d;
                            best_source = boundary_source_map[p];
                        }
                    }
                    forged_grid[r * grid_size + c] = (best_source == 0) ? plane_A[r * grid_size + c] : plane_B[r * grid_size + c];
                }
            }
        }
        std::string name = "Plane_C_Grid_Flood_Size_" + std::to_string(inner_size);
        evaluate_grid(name, forged_grid);
    }

    std::cout << "\n";
    std::cout << "\n--- Attacker Custom Topology Sweep Results ---\n";
    
    int lowest_defects = 99999;
    std::string lowest_name = "";

    // Open the file in append mode
    std::ofstream csv_file("attacker_custom_topology_stats.csv", std::ios::app);
    // Only write the header if the file is completely empty (newly created)
    if (csv_file.tellp() == 0) {
        csv_file << "Attack_Name,Actual_Defects,"
                << "Block8_Min,Block8_Max,"
                << "Sliding3_Min,Sliding3_Max,"
                << "Line_Min,Line_Max,"
                << "Frame4,Frame6,Frame8,"
                << "Core32,Core48,"
                << "Alien_Tiles,Frame2_Alien_Tiles,Post_Greedy_Defect_Count,"
                << "Post_Greedy_Alien_Tiles,Post_Greedy_Frame2_Alien_Tiles,"
                << "Post_Greedy_Frame6,Post_Greedy_Core48,Is_SneakThrough\n";
    }

    for (const auto& res : results) {
        if (res.is_viable_threat) {
            std::cout << res.topology_name << ":\n";
            std::cout << "  Pre-Greedy Total Defects: " << res.total_defects << "\n";
            std::cout << "  Pre-Greedy Alien Tiles:   " << res.alien_tiles << "\n";
            std::cout << "  Pre-Greedy Frame2 Aliens: " << res.frame_alien_tiles_2 << "\n";
            if (res.post_greedy_total_defects != -1) {
                std::cout << "  Post-Greedy Defects:      " << res.post_greedy_total_defects << "\n";
                std::cout << "  Post-Greedy Aliens:       " << res.post_greedy_alien_tiles << "\n";
                std::cout << "  Post-Greedy Frame2 Alien: " << res.post_greedy_frame_alien_tiles_2 << "\n";
                std::cout << "  Post-Greedy Core48:       " << res.post_greedy_core_48 << "\n";
                std::cout << "  Post-Greedy Frame6:       " << res.post_greedy_frame_defects_6 << "\n";
            }
            std::cout << "  Max Line Defects (Edges): " << res.max_line_defects_seam << "\n";
            std::cout << "  Max Line Defects (Tiles): " << res.max_line_defects_tiles << "\n";
            std::cout << "  Max Diag Defects (Tiles): " << res.max_diagonal_defects_tiles << "\n";
            std::cout << "  Frame Defects (4):        " << res.frame_defects_4 << "\n";
            std::cout << "  Pre-Greedy Frame6:        " << res.frame_defects_6 << "\n";
            std::cout << "  Frame Defects (8):        " << res.frame_defects_8 << "\n";
            std::cout << "  Viable Threat?            " << (res.is_viable_threat ? "YES" : "NO") << "\n\n";
        }

        if (res.total_defects < lowest_defects) {
            lowest_defects = res.total_defects;
            lowest_name = res.topology_name;
        }

        csv_file << res.topology_name << "," << res.total_defects << ","
                 << res.min_subgrid_defects << "," << res.max_subgrid_defects << ","
                 << res.min_sliding_defects << "," << res.max_sliding_defects << ","
                 << res.min_line_defects_seam << "," << res.max_line_defects_seam << ","
                 << res.frame_defects_4 << "," << res.frame_defects_6 << "," << res.frame_defects_8 << ","
                 << res.core_32 << "," << res.core_48 << ","
                 << res.alien_tiles << "," << res.frame_alien_tiles_2 << "," << res.post_greedy_total_defects << ","
                 << res.post_greedy_alien_tiles << "," << res.post_greedy_frame_alien_tiles_2 << ","
                 << res.post_greedy_frame_defects_6 << "," << res.post_greedy_core_48 << ","
                 << (res.is_viable_threat ? "1" : "0") << "\n";
        csv_file.flush();
    }

    csv_file.close();

    std::cout << "Lowest defects: " << lowest_name << " with " << lowest_defects << " defects.\n";
}

void optimize_attack_A_maximize_aliens(
    std::vector<int>& forged_grid, int grid_size, const std::vector<Tile>& alphabet,
    const std::vector<int>& plane_A, const std::vector<int>& plane_B,
    int defect_limit,
    const std::vector<uint8_t>& boundary_mask, 
    const std::vector<int>& public_key_grid) 
{
    auto get_local_defects = [&](const std::vector<int>& g, int r, int c) {
        int def = 0;
        int idx = r * grid_size + c;
        Tile t = alphabet[g[idx]];
        if (r > 0 && t.top != alphabet[g[(r - 1) * grid_size + c]].bottom) def++;
        if (r < grid_size - 1 && t.bottom != alphabet[g[(r + 1) * grid_size + c]].top) def++;
        if (c > 0 && t.left != alphabet[g[r * grid_size + (c - 1)]].right) def++;
        if (c < grid_size - 1 && t.right != alphabet[g[r * grid_size + (c + 1)]].left) def++;
        return def;
    };

    int current_defects = count_grid_defects(forged_grid.data(), grid_size, alphabet);
    std::vector<int> best_grid_state = forged_grid;

    bool improved = true;
    while (improved && current_defects < defect_limit) {
        improved = false;
        
        int best_r = -1, best_c = -1, best_tile_id = -1;
        int lowest_defect_cost = 999; 

        // Loop now covers the entire grid including the boundary
        for (int r = 0; r < grid_size; r++) {     
            for (int c = 0; c < grid_size; c++) {
                int idx = r * grid_size + c;
                
                if (forged_grid[idx] == plane_A[idx] || forged_grid[idx] == plane_B[idx]) {
                    int original_tile = forged_grid[idx];
                    int old_local_defects = get_local_defects(forged_grid, r, c);

                    for (int new_t = 0; new_t < alphabet.size(); new_t++) {
                        if (new_t != plane_A[idx] && new_t != plane_B[idx]) {
                            
                            // --- STRICT PUBLIC KEY BOUNDARY ENFORCEMENT ---
                            bool pk_violation = false;
                            if (r == 0 && boundary_mask[idx] == 1 && alphabet[new_t].top != alphabet[public_key_grid[idx]].top) pk_violation = true;
                            if (r == grid_size - 1 && boundary_mask[idx] == 1 && alphabet[new_t].bottom != alphabet[public_key_grid[idx]].bottom) pk_violation = true;
                            if (c == 0 && boundary_mask[idx] == 1 && alphabet[new_t].left != alphabet[public_key_grid[idx]].left) pk_violation = true;
                            if (c == grid_size - 1 && boundary_mask[idx] == 1 && alphabet[new_t].right != alphabet[public_key_grid[idx]].right) pk_violation = true;
                            if (pk_violation) continue;

                            forged_grid[idx] = new_t; 
                            int new_local_defects = get_local_defects(forged_grid, r, c);
                            
                            int defect_cost = new_local_defects - old_local_defects;
                            
                            if (defect_cost < lowest_defect_cost) {
                                lowest_defect_cost = defect_cost;
                                best_r = r; best_c = c; best_tile_id = new_t;
                            }
                        }
                    }
                    forged_grid[idx] = original_tile; 
                }
            }
        }

        if (best_r != -1 && (current_defects + lowest_defect_cost <= defect_limit)) {
            forged_grid[best_r * grid_size + best_c] = best_tile_id;
            current_defects += lowest_defect_cost;
            best_grid_state = forged_grid; 
            improved = true;
        } else {
            break; 
        }
    }
    
    forged_grid = best_grid_state;
}

void optimize_attack_B_minimize_defects(
    std::vector<int>& forged_grid, int grid_size, const std::vector<Tile>& alphabet,
    const std::vector<int>& plane_A, const std::vector<int>& plane_B,
    int alien_limit,
    const std::vector<uint8_t>& boundary_mask, 
    const std::vector<int>& public_key_grid)
{
    auto count_aliens = [&]() {
        int a = 0;
        for (int i = 0; i < grid_size * grid_size; i++) 
            if (forged_grid[i] != plane_A[i] && forged_grid[i] != plane_B[i]) a++;
        return a;
    };

    auto get_local_defects = [&](const std::vector<int>& g, int r, int c) {
        int def = 0;
        int idx = r * grid_size + c;
        Tile t = alphabet[g[idx]];
        if (r > 0 && t.top != alphabet[g[(r - 1) * grid_size + c]].bottom) def++;
        if (r < grid_size - 1 && t.bottom != alphabet[g[(r + 1) * grid_size + c]].top) def++;
        if (c > 0 && t.left != alphabet[g[r * grid_size + (c - 1)]].right) def++;
        if (c < grid_size - 1 && t.right != alphabet[g[r * grid_size + (c + 1)]].left) def++;
        return def;
    };

    int current_aliens = count_aliens();
    int current_defects = count_grid_defects(forged_grid.data(), grid_size, alphabet);
    
    int min_defects_seen = current_defects;
    std::vector<int> best_grid_state = forged_grid;
    
    bool improved = true;
    while (improved && current_aliens > alien_limit) {
        improved = false;
        
        int best_r = -1, best_c = -1, best_tile_id = -1;
        int best_healing_amount = -999; 

        // Loop now covers the entire grid including the boundary
        for (int r = 0; r < grid_size; r++) {
            for (int c = 0; c < grid_size; c++) {
                int idx = r * grid_size + c;
                
                if (forged_grid[idx] != plane_A[idx] && forged_grid[idx] != plane_B[idx]) {
                    int original_tile = forged_grid[idx];
                    int old_local_defects = get_local_defects(forged_grid, r, c);

                    int native_options[2] = {plane_A[idx], plane_B[idx]};
                    for (int native_t : native_options) {
                        
                        // --- STRICT PUBLIC KEY BOUNDARY ENFORCEMENT ---
                        bool pk_violation = false;
                        if (r == 0 && boundary_mask[idx] == 1 && alphabet[native_t].top != alphabet[public_key_grid[idx]].top) pk_violation = true;
                        if (r == grid_size - 1 && boundary_mask[idx] == 1 && alphabet[native_t].bottom != alphabet[public_key_grid[idx]].bottom) pk_violation = true;
                        if (c == 0 && boundary_mask[idx] == 1 && alphabet[native_t].left != alphabet[public_key_grid[idx]].left) pk_violation = true;
                        if (c == grid_size - 1 && boundary_mask[idx] == 1 && alphabet[native_t].right != alphabet[public_key_grid[idx]].right) pk_violation = true;
                        if (pk_violation) continue;

                        forged_grid[idx] = native_t;
                        int new_local_defects = get_local_defects(forged_grid, r, c);
                        
                        int defects_healed = old_local_defects - new_local_defects;
                        
                        if (defects_healed > best_healing_amount) {
                            best_healing_amount = defects_healed;
                            best_r = r; best_c = c; best_tile_id = native_t;
                        }
                    }
                    forged_grid[idx] = original_tile; 
                }
            }
        }

        if (best_r != -1) {
            forged_grid[best_r * grid_size + best_c] = best_tile_id;
            current_aliens--;
            current_defects -= best_healing_amount; 
            
            if (current_defects < min_defects_seen) {
                min_defects_seen = current_defects;
                best_grid_state = forged_grid; 
            }
            
            improved = true;
        } else {
            break; 
        }
    }
    
    forged_grid = best_grid_state;
}

void optimize_attack_C_maximize_frame_aliens(
    std::vector<int>& forged_grid, int grid_size, const std::vector<Tile>& alphabet,
    const std::vector<int>& plane_A, const std::vector<int>& plane_B,
    int defect_limit, int alien_limit, int frame_depth,
    const std::vector<uint8_t>& boundary_mask, 
    const std::vector<int>& public_key_grid) 
{
    auto get_local_defects = [&](const std::vector<int>& g, int r, int c) {
        int def = 0;
        int idx = r * grid_size + c;
        Tile t = alphabet[g[idx]];
        
        // This only calculates *internal* grid defects.
        // Outward PK boundary constraints are enforced strictly via the boundary_mask below.
        if (r > 0 && t.top != alphabet[g[(r - 1) * grid_size + c]].bottom) def++;
        if (r < grid_size - 1 && t.bottom != alphabet[g[(r + 1) * grid_size + c]].top) def++;
        if (c > 0 && t.left != alphabet[g[r * grid_size + (c - 1)]].right) def++;
        if (c < grid_size - 1 && t.right != alphabet[g[r * grid_size + (c + 1)]].left) def++;
        return def;
    };

    int current_defects = count_grid_defects(forged_grid.data(), grid_size, alphabet);
    
    int current_total_aliens = 0;
    for (int i = 0; i < grid_size * grid_size; i++) {
        if (forged_grid[i] != plane_A[i] && forged_grid[i] != plane_B[i]) current_total_aliens++;
    }

    int current_frame_aliens = calculate_alien_frame_count(forged_grid, plane_A, plane_B, grid_size, frame_depth);
    
    // SNAPSHOT TRACKERS
    std::vector<int> best_grid_state = forged_grid;
    int max_frame_aliens_seen = current_frame_aliens;

    bool improved = true;
    while (improved && current_defects <= defect_limit) {
        improved = false;
        
        int best_r = -1, best_c = -1, best_tile_id = -1;
        int lowest_defect_cost = 999; 

        // Loop covers all tiles (r=0 and c=0) so the attacker can exploit Frame 1 Wildcards
        for (int r = 0; r < grid_size; r++) {     
            for (int c = 0; c < grid_size; c++) {
                
                if (r < frame_depth || r >= grid_size - frame_depth || 
                    c < frame_depth || c >= grid_size - frame_depth) {
                    
                    int idx = r * grid_size + c;
                    
                    // We can only convert Native A/B tiles into Alien tiles
                    if (forged_grid[idx] == plane_A[idx] || forged_grid[idx] == plane_B[idx]) {
                        int original_tile = forged_grid[idx];
                        int old_local_defects = get_local_defects(forged_grid, r, c);

                        for (int new_t = 0; new_t < alphabet.size(); new_t++) {
                            if (new_t != plane_A[idx] && new_t != plane_B[idx]) {
                                
                                // --- STRICT PUBLIC KEY BOUNDARY ENFORCEMENT ---
                                // If the tile is on the perimeter and the mask says it's pinned, 
                                // the outward edge color MUST match the public key perfectly.
                                bool pk_violation = false;
                                if (r == 0 && boundary_mask[idx] == 1 && alphabet[new_t].top != alphabet[public_key_grid[idx]].top) pk_violation = true;
                                if (r == grid_size - 1 && boundary_mask[idx] == 1 && alphabet[new_t].bottom != alphabet[public_key_grid[idx]].bottom) pk_violation = true;
                                if (c == 0 && boundary_mask[idx] == 1 && alphabet[new_t].left != alphabet[public_key_grid[idx]].left) pk_violation = true;
                                if (c == grid_size - 1 && boundary_mask[idx] == 1 && alphabet[new_t].right != alphabet[public_key_grid[idx]].right) pk_violation = true;
                                
                                if (pk_violation) continue; // Reject this tile substitution entirely
                                // ---------------------------------------------------

                                forged_grid[idx] = new_t; 
                                int new_local_defects = get_local_defects(forged_grid, r, c);
                                
                                int defect_cost = new_local_defects - old_local_defects;
                                
                                if (defect_cost < lowest_defect_cost) {
                                    lowest_defect_cost = defect_cost;
                                    best_r = r; best_c = c; best_tile_id = new_t;
                                }
                            }
                        }
                        forged_grid[idx] = original_tile; 
                    }
                }
            }
        }

        if (best_r != -1 && (current_defects + lowest_defect_cost <= defect_limit)) {
            forged_grid[best_r * grid_size + best_c] = best_tile_id;
            current_defects += lowest_defect_cost;
            current_total_aliens++; 
            current_frame_aliens++; 
            
            if (current_total_aliens >= alien_limit && current_frame_aliens > max_frame_aliens_seen) {
                max_frame_aliens_seen = current_frame_aliens;
                best_grid_state = forged_grid; 
            }
            improved = true;
        } else {
            break; 
        }
    }
    
    // RESTORE OPTIMAL STATE
    forged_grid = best_grid_state;
}

void optimize_attack_D_spatial_trap(
    std::vector<int>& forged_grid, int grid_size, const std::vector<Tile>& alphabet,
    const std::vector<int>& plane_A, const std::vector<int>& plane_B,
    int defect_limit, int alien_limit,
    const std::vector<uint8_t>& boundary_mask, 
    const std::vector<int>& public_key_grid) 
{
    // Helper to count raw defective edges (for the Total Defects budget)
    auto get_local_edges = [&](const std::vector<int>& g, int r, int c) {
        int def = 0;
        int idx = r * grid_size + c;
        Tile t = alphabet[g[idx]];
        if (r > 0 && t.top != alphabet[g[(r - 1) * grid_size + c]].bottom) def++;
        if (r < grid_size - 1 && t.bottom != alphabet[g[(r + 1) * grid_size + c]].top) def++;
        if (c > 0 && t.left != alphabet[g[r * grid_size + (c - 1)]].right) def++;
        if (c < grid_size - 1 && t.right != alphabet[g[r * grid_size + (c + 1)]].left) def++;
        return def;
    };

    // Helper to check if a specific tile has ANY defective edges
    auto is_tile_defective = [&](const std::vector<int>& g, int r, int c) {
        if (r < 0 || r >= grid_size || c < 0 || c >= grid_size) return false;
        return get_local_edges(g, r, c) > 0;
    };

    // Helper to calculate exact region membership
    int core_depth = (grid_size - 48) / 2;
    int frame_depth = 6;
    auto in_core48 = [&](int r, int c) { return (r >= core_depth && r < grid_size - core_depth && c >= core_depth && c < grid_size - core_depth); };
    auto in_frame6 = [&](int r, int c) { return (r < frame_depth || r >= grid_size - frame_depth || c < frame_depth || c >= grid_size - frame_depth); };

    // Helper to dynamically calculate the spatial metrics for a tile and its 4 neighbors
    auto get_region_defects = [&](const std::vector<int>& g, int r, int c, int& core48_count, int& frame6_count) {
        core48_count = 0; frame6_count = 0;
        int dr[] = {0, -1, 1, 0, 0};
        int dc[] = {0, 0, 0, -1, 1};
        for (int i = 0; i < 5; i++) {
            int nr = r + dr[i], nc = c + dc[i];
            if (nr >= 0 && nr < grid_size && nc >= 0 && nc < grid_size) {
                if (is_tile_defective(g, nr, nc)) {
                    if (in_core48(nr, nc)) core48_count++;
                    if (in_frame6(nr, nc)) frame6_count++;
                }
            }
        }
    };

    int current_defects = count_grid_defects(forged_grid.data(), grid_size, alphabet);
    int current_total_aliens = 0;
    for (int i = 0; i < grid_size * grid_size; i++) {
        if (forged_grid[i] != plane_A[i] && forged_grid[i] != plane_B[i]) current_total_aliens++;
    }

    struct Move {
        int r, c, tile_id;
        int delta_score;  // Primary: (delta_core48 - delta_frame6). Higher is better.
        int delta_edges;  // Secondary: Total defects cost. Lower is better.
        
        bool operator<(const Move& other) const {
            if (delta_score != other.delta_score) return delta_score < other.delta_score;
            return delta_edges > other.delta_edges; // If scores tie, prefer the cheaper move
        }
    };

    bool improved = true;
    while (improved) {
        improved = false;
        Move best_move = {-1, -1, -1, -9999, 9999};
        bool found_move = false;

        for (int r = 0; r < grid_size; r++) {     
            for (int c = 0; c < grid_size; c++) {
                int idx = r * grid_size + c;
                int original_tile = forged_grid[idx];
                
                int old_edges = get_local_edges(forged_grid, r, c);
                int old_core48 = 0, old_frame6 = 0;
                get_region_defects(forged_grid, r, c, old_core48, old_frame6);
                bool old_alien = (original_tile != plane_A[idx] && original_tile != plane_B[idx]);

                for (int new_t = 0; new_t < alphabet.size(); new_t++) {
                    if (new_t == original_tile) continue;
                    
                    // --- STRICT PUBLIC KEY BOUNDARY ENFORCEMENT ---
                    bool pk_violation = false;
                    if (r == 0 && boundary_mask[idx] == 1 && alphabet[new_t].top != alphabet[public_key_grid[idx]].top) pk_violation = true;
                    if (r == grid_size - 1 && boundary_mask[idx] == 1 && alphabet[new_t].bottom != alphabet[public_key_grid[idx]].bottom) pk_violation = true;
                    if (c == 0 && boundary_mask[idx] == 1 && alphabet[new_t].left != alphabet[public_key_grid[idx]].left) pk_violation = true;
                    if (c == grid_size - 1 && boundary_mask[idx] == 1 && alphabet[new_t].right != alphabet[public_key_grid[idx]].right) pk_violation = true;
                    if (pk_violation) continue;

                    // Apply hypothetical swap
                    forged_grid[idx] = new_t; 
                    
                    int new_edges = get_local_edges(forged_grid, r, c);
                    int new_core48 = 0, new_frame6 = 0;
                    get_region_defects(forged_grid, r, c, new_core48, new_frame6);
                    bool new_alien = (new_t != plane_A[idx] && new_t != plane_B[idx]);

                    int d_edges = new_edges - old_edges;
                    int d_core48 = new_core48 - old_core48;
                    int d_frame6 = new_frame6 - old_frame6;
                    int d_alien = new_alien - old_alien;

                    // Revert swap
                    forged_grid[idx] = original_tile; 

                    // Check budget and alien limits
                    if (current_defects + d_edges > defect_limit) continue;
                    if (current_total_aliens + d_alien < alien_limit) continue;

                    // Objective Function: +1 for adding a Core defect, +1 for healing a Frame defect
                    int score = d_core48 - d_frame6;
                    
                    // A move is an improvement if it actively helps the spatial trap, 
                    // OR if it heals total defects without hurting the spatial trap (free budget!)
                    bool is_improvement = (score > 0) || (score == 0 && d_edges < 0);

                    if (is_improvement) {
                        Move m = {r, c, new_t, score, d_edges};
                        if (!found_move || best_move < m) {
                            best_move = m;
                            found_move = true;
                        }
                    }
                }
            }
        }

        if (found_move) {
            // Apply the best optimized move
            int idx = best_move.r * grid_size + best_move.c;
            bool was_alien = (forged_grid[idx] != plane_A[idx] && forged_grid[idx] != plane_B[idx]);
            bool is_alien = (best_move.tile_id != plane_A[idx] && best_move.tile_id != plane_B[idx]);

            forged_grid[idx] = best_move.tile_id;
            current_defects += best_move.delta_edges;
            current_total_aliens += (is_alien - was_alien);

            improved = true;
        }
    }
}

void optimize_attack_E_universal_trap(
    std::vector<int>& forged_grid, int grid_size, const std::vector<Tile>& alphabet,
    const std::vector<int>& plane_A, const std::vector<int>& plane_B,
    int defect_limit, int alien_limit,
    int target_core32, int target_core48, int target_frame6, int target_line, int target_block,
    bool use_diagonal_exploit,
    const std::vector<uint8_t>& boundary_mask, 
    const std::vector<int>& public_key_grid) 
{
    auto get_local_edges = [&](const std::vector<int>& g, int r, int c) {
        int def = 0;
        int idx = r * grid_size + c;
        Tile t = alphabet[g[idx]];
        if (r > 0 && t.top != alphabet[g[(r - 1) * grid_size + c]].bottom) def++;
        if (r < grid_size - 1 && t.bottom != alphabet[g[(r + 1) * grid_size + c]].top) def++;
        if (c > 0 && t.left != alphabet[g[r * grid_size + (c - 1)]].right) def++;
        if (c < grid_size - 1 && t.right != alphabet[g[r * grid_size + (c + 1)]].left) def++;
        return def;
    };

    auto is_tile_defective = [&](const std::vector<int>& g, int r, int c) {
        if (r < 0 || r >= grid_size || c < 0 || c >= grid_size) return false;
        return get_local_edges(g, r, c) > 0;
    };

    int core32_depth = (grid_size - 32) / 2;
    int core_depth = (grid_size - 48) / 2;
    int frame_depth = 6;
    auto in_core32 = [&](int r, int c) { return (r >= core32_depth && r < grid_size - core32_depth && c >= core32_depth && c < grid_size - core32_depth); };
    auto in_core48 = [&](int r, int c) { return (r >= core_depth && r < grid_size - core_depth && c >= core_depth && c < grid_size - core_depth); };
    auto in_frame6 = [&](int r, int c) { return (r < frame_depth || r >= grid_size - frame_depth || c < frame_depth || c >= grid_size - frame_depth); };

    // Initialize all global spatial trackers
    std::vector<int> h_seam_defects(grid_size - 1, 0);
    std::vector<int> v_seam_defects(grid_size - 1, 0);
    std::vector<std::vector<int>> block_defects(8, std::vector<int>(8, 0));
    int current_core32 = 0, current_core48 = 0, current_frame6 = 0;

    // Pre-calculate true seam defects
    for (int r = 0; r < grid_size - 1; r++) {
        for (int c = 0; c < grid_size; c++) {
            if (alphabet[forged_grid[r * grid_size + c]].bottom != alphabet[forged_grid[(r + 1) * grid_size + c]].top) h_seam_defects[r]++;
        }
    }
    for (int c = 0; c < grid_size - 1; c++) {
        for (int r = 0; r < grid_size; r++) {
            if (alphabet[forged_grid[r * grid_size + c]].right != alphabet[forged_grid[r * grid_size + (c + 1)]].left) v_seam_defects[c]++;
        }
    }

    for (int r = 0; r < grid_size; r++) {
        for (int c = 0; c < grid_size; c++) {
            if (is_tile_defective(forged_grid, r, c)) {
                block_defects[r / 8][c / 8]++;
                if (in_core32(r, c)) current_core32++;
                if (in_core48(r, c)) current_core48++;
                if (in_frame6(r, c)) current_frame6++;
            }
        }
    }

    int current_defects = count_grid_defects(forged_grid.data(), grid_size, alphabet);
    int current_total_aliens = 0;
    for (int i = 0; i < grid_size * grid_size; i++) {
        if (forged_grid[i] != plane_A[i] && forged_grid[i] != plane_B[i]) current_total_aliens++;
    }

    struct TileState { int r, c; bool defective; };
    struct Move {
        int r, c, tile_id;
        int delta_score;  
        int delta_edges;
        int delta_aliens;
        std::vector<TileState> affected_tiles_old;
        
        bool operator>(const Move& other) const {
            // 1. Highest Score Wins
            if (delta_score != other.delta_score) return delta_score > other.delta_score;
            // 2. Lowest Edge Cost Wins (Strict Hierarchy 1)
            if (delta_edges != other.delta_edges) return delta_edges < other.delta_edges;
            // 3. Most Aliens Gained Wins (Strict Hierarchy 2)
            return delta_aliens > other.delta_aliens; 
        }
    };

    int max_optimization_passes = 1000; // Watchdog limit for rare multi-tile loops
    int current_pass = 0;

    bool improved = true;
    while (improved && current_pass < max_optimization_passes) {
        improved = false;
        current_pass++;
        
        Move best_move = {-1, -1, -1, -9999, 9999, 0, {}};
        bool found_move = false;

        for (int r = 0; r < grid_size; r++) {     
            for (int c = 0; c < grid_size; c++) {
                int idx = r * grid_size + c;
                int original_tile = forged_grid[idx];
                
                int old_edges = get_local_edges(forged_grid, r, c);
                bool old_alien = (original_tile != plane_A[idx] && original_tile != plane_B[idx]);

                // Snapshot the 5 affected tiles
                std::vector<TileState> affected_tiles;
                int dr[] = {0, -1, 1, 0, 0};
                int dc[] = {0, 0, 0, -1, 1};
                for(int i = 0; i < 5; i++) {
                    int nr = r + dr[i], nc = c + dc[i];
                    if(nr >= 0 && nr < grid_size && nc >= 0 && nc < grid_size) {
                        affected_tiles.push_back({nr, nc, is_tile_defective(forged_grid, nr, nc)});
                    }
                }

                for (int new_t = 0; new_t < alphabet.size(); new_t++) {
                    if (new_t == original_tile) continue;
                    
                    bool pk_violation = false;
                    if (r == 0 && boundary_mask[idx] == 1 && alphabet[new_t].top != alphabet[public_key_grid[idx]].top) pk_violation = true;
                    if (r == grid_size - 1 && boundary_mask[idx] == 1 && alphabet[new_t].bottom != alphabet[public_key_grid[idx]].bottom) pk_violation = true;
                    if (c == 0 && boundary_mask[idx] == 1 && alphabet[new_t].left != alphabet[public_key_grid[idx]].left) pk_violation = true;
                    if (c == grid_size - 1 && boundary_mask[idx] == 1 && alphabet[new_t].right != alphabet[public_key_grid[idx]].right) pk_violation = true;
                    if (pk_violation) continue;

                    forged_grid[idx] = new_t; 
                    int new_edges = get_local_edges(forged_grid, r, c);
                    bool new_alien = (new_t != plane_A[idx] && new_t != plane_B[idx]);

                    int d_edges = new_edges - old_edges;
                    int d_alien = new_alien - old_alien;

                    if (defect_limit != -1 && current_defects + d_edges > defect_limit) { forged_grid[idx] = original_tile; continue; }
                    if (alien_limit != -1 && current_total_aliens + d_alien < alien_limit) { forged_grid[idx] = original_tile; continue; }

                    // Safely Aggregate Deltas Before Scoring
                    int d_core32 = 0, d_core48 = 0, d_frame6 = 0;
                    
                    // Fast, zero-allocation inline trackers for the 5 affected tiles
                    int b_deltas[5];
                    std::pair<int, int> unique_b[5];
                    int num_b = 0;

                    for (const auto& at : affected_tiles) {
                        bool new_defective = is_tile_defective(forged_grid, at.r, at.c);
                        if (at.defective != new_defective) {
                            int delta = new_defective ? 1 : -1;

                            if (in_core32(at.r, at.c)) d_core32 += delta;
                            if (in_core48(at.r, at.c)) d_core48 += delta;
                            if (in_frame6(at.r, at.c)) d_frame6 += delta;

                            // Aggregate Block Deltas
                            bool found_b = false;
                            for (int i = 0; i < num_b; i++) { if (unique_b[i].first == at.r/8 && unique_b[i].second == at.c/8) { b_deltas[i] += delta; found_b = true; break; } }
                            if (!found_b) { unique_b[num_b] = {at.r/8, at.c/8}; b_deltas[num_b] = delta; num_b++; }
                        }
                    }

                    // Calculate exact seam differences
                    int old_h_above = 0, new_h_above = 0, old_h_below = 0, new_h_below = 0;
                    int old_v_left = 0, new_v_left = 0, old_v_right = 0, new_v_right = 0;

                    if (r > 0) {
                        old_h_above = (alphabet[original_tile].top != alphabet[forged_grid[(r - 1) * grid_size + c]].bottom) ? 1 : 0;
                        new_h_above = (alphabet[new_t].top != alphabet[forged_grid[(r - 1) * grid_size + c]].bottom) ? 1 : 0;
                    }
                    if (r < grid_size - 1) {
                        old_h_below = (alphabet[original_tile].bottom != alphabet[forged_grid[(r + 1) * grid_size + c]].top) ? 1 : 0;
                        new_h_below = (alphabet[new_t].bottom != alphabet[forged_grid[(r + 1) * grid_size + c]].top) ? 1 : 0;
                    }
                    if (c > 0) {
                        old_v_left = (alphabet[original_tile].left != alphabet[forged_grid[r * grid_size + (c - 1)]].right) ? 1 : 0;
                        new_v_left = (alphabet[new_t].left != alphabet[forged_grid[r * grid_size + (c - 1)]].right) ? 1 : 0;
                    }
                    if (c < grid_size - 1) {
                        old_v_right = (alphabet[original_tile].right != alphabet[forged_grid[r * grid_size + (c + 1)]].left) ? 1 : 0;
                        new_v_right = (alphabet[new_t].right != alphabet[forged_grid[r * grid_size + (c + 1)]].left) ? 1 : 0;
                    }

                    int dist_improvement = 0;

                    // Now calculate the true violation distance using the exact seam deltas
                    if (target_line != -1) {
                        if (r > 0) {
                            dist_improvement += (std::max(0, h_seam_defects[r - 1] - target_line) - std::max(0, h_seam_defects[r - 1] - old_h_above + new_h_above - target_line)) * 50;
                        }
                        if (r < grid_size - 1) {
                            dist_improvement += (std::max(0, h_seam_defects[r] - target_line) - std::max(0, h_seam_defects[r] - old_h_below + new_h_below - target_line)) * 50;
                        }
                        if (c > 0) {
                            dist_improvement += (std::max(0, v_seam_defects[c - 1] - target_line) - std::max(0, v_seam_defects[c - 1] - old_v_left + new_v_left - target_line)) * 50;
                        }
                        if (c < grid_size - 1) {
                            dist_improvement += (std::max(0, v_seam_defects[c] - target_line) - std::max(0, v_seam_defects[c] - old_v_right + new_v_right - target_line)) * 50;
                        }
                    }

                    for (int i = 0; i < num_b; i++) {
                        int b_cnt = block_defects[unique_b[i].first][unique_b[i].second];
                        if (target_block != -1) {
                            dist_improvement += (std::max(0, b_cnt - target_block) - std::max(0, b_cnt + b_deltas[i] - target_block)) * 50;
                        }
                    }

                    // Deep Core32 constraint (Weight: 10)
                    if (target_core32 != -1) {
                        int old_core32_viol = std::max(0, target_core32 - current_core32);
                        int new_core32_viol = std::max(0, target_core32 - (current_core32 + d_core32));
                        dist_improvement += (old_core32_viol - new_core32_viol) * 10;
                    }

                    // Core48 constraint (Weight: 10)
                    if (target_core48 != -1) {
                        int old_core48_viol = std::max(0, target_core48 - current_core48);
                        int new_core48_viol = std::max(0, target_core48 - (current_core48 + d_core48));
                        dist_improvement += (old_core48_viol - new_core48_viol) * 10;
                    }

                    // Frame constraint (Weight: 10)
                    if (target_frame6 != -1) {
                        int old_frame_viol = std::max(0, current_frame6 - target_frame6);
                        int new_frame_viol = std::max(0, (current_frame6 + d_frame6) - target_frame6);
                        dist_improvement += (old_frame_viol - new_frame_viol) * 10;
                    }

                    // THE SMART DIAGONAL EXPLOIT HEURISTIC
                    if (use_diagonal_exploit) {
                        for (const auto& at : affected_tiles) {
                            bool new_defective = is_tile_defective(forged_grid, at.r, at.c);
                            if (at.defective != new_defective) {
                                int delta = new_defective ? 1 : -1;
                                
                                // Check the 4 diagonal neighbors (Top-Left, Top-Right, Bot-Left, Bot-Right)
                                int ddr[] = {-1, -1, 1, 1};
                                int ddc[] = {-1, 1, -1, 1};
                                for (int j = 0; j < 4; j++) {
                                    int nr = at.r + ddr[j];
                                    int nc = at.c + ddc[j];
                                    if (nr >= 0 && nr < grid_size && nc >= 0 && nc < grid_size) {
                                        // If the diagonal neighbor is defective, reward the connection
                                        if (is_tile_defective(forged_grid, nr, nc)) {
                                            // Apply a massive +15 point reward to incentivize diagonal clumping
                                            // over orthogonal clumping or simple defect healing.
                                            dist_improvement += (delta * 15); 
                                        }
                                    }
                                }
                            }
                        }
                    }

                    // Valid move if it improves constraints, OR is constraint-neutral but banks defect budget
                    // NEW STABLE LOGIC
                    bool is_improvement = false;

                    if (dist_improvement > 0) {
                        // Priority 1: Actively solving the spatial/core constraints
                        is_improvement = true;
                    } else if (dist_improvement == 0) {
                        // Priority 2: Only accept neutral moves if they strictly bank defect budget
                        // This prevents "Tug-of-War" with alien tile counts
                        if (d_edges < 0) {
                            is_improvement = true;
                        } 
                        // Optimization: Only allow alien gains if they cost ZERO defects
                        else if (d_edges == 0 && d_alien > 0) {
                            is_improvement = true;
                        }
                    }

                    if (is_improvement) {
                        Move m = {r, c, new_t, dist_improvement, d_edges, d_alien, affected_tiles};
                        if (!found_move || m > best_move) {
                            best_move = m;
                            found_move = true;
                        }
                    }
                    
                    forged_grid[idx] = original_tile; // Revert
                }
            }
        }

        // Apply the mathematically best move and update live trackers
        if (found_move) {
            int app_r = best_move.r;
            int app_c = best_move.c;
            int old_app_t = forged_grid[app_r * grid_size + app_c];
            int new_app_t = best_move.tile_id;

            // Update Seam Trackers
            if (app_r > 0) {
                if (alphabet[old_app_t].top != alphabet[forged_grid[(app_r - 1) * grid_size + app_c]].bottom) h_seam_defects[app_r - 1]--;
                if (alphabet[new_app_t].top != alphabet[forged_grid[(app_r - 1) * grid_size + app_c]].bottom) h_seam_defects[app_r - 1]++;
            }
            if (app_r < grid_size - 1) {
                if (alphabet[old_app_t].bottom != alphabet[forged_grid[(app_r + 1) * grid_size + app_c]].top) h_seam_defects[app_r]--;
                if (alphabet[new_app_t].bottom != alphabet[forged_grid[(app_r + 1) * grid_size + app_c]].top) h_seam_defects[app_r]++;
            }
            if (app_c > 0) {
                if (alphabet[old_app_t].left != alphabet[forged_grid[app_r * grid_size + (app_c - 1)]].right) v_seam_defects[app_c - 1]--;
                if (alphabet[new_app_t].left != alphabet[forged_grid[app_r * grid_size + (app_c - 1)]].right) v_seam_defects[app_c - 1]++;
            }
            if (app_c < grid_size - 1) {
                if (alphabet[old_app_t].right != alphabet[forged_grid[app_r * grid_size + (app_c + 1)]].left) v_seam_defects[app_c]--;
                if (alphabet[new_app_t].right != alphabet[forged_grid[app_r * grid_size + (app_c + 1)]].left) v_seam_defects[app_c]++;
            }

            // Now apply the tile
            forged_grid[app_r * grid_size + app_c] = new_app_t;
            current_defects += best_move.delta_edges;
            current_total_aliens += best_move.delta_aliens;
            
            for (const auto& at : best_move.affected_tiles_old) {
                bool new_defective = is_tile_defective(forged_grid, at.r, at.c);
                if (at.defective != new_defective) {
                    int delta = new_defective ? 1 : -1;
                    block_defects[at.r / 8][at.c / 8] += delta;
                    if (in_core32(at.r, at.c)) current_core32 += delta;
                    if (in_core48(at.r, at.c)) current_core48 += delta;
                    if (in_frame6(at.r, at.c)) current_frame6 += delta;
                }
            }
            improved = true;
        }
    }
}

// Stronger Total Constraint Violation version
void optimize_attack_E_universal_trap_tcv(
    std::vector<int>& forged_grid, int grid_size, const std::vector<Tile>& alphabet,
    const std::vector<int>& plane_A, const std::vector<int>& plane_B,
    int defect_limit, int alien_limit,
    int target_core32, int target_core48, int target_frame6, int target_line, int target_block,
    bool use_diagonal_exploit,
    const std::vector<uint8_t>& boundary_mask, 
    const std::vector<int>& public_key_grid) 
{
    auto get_local_edges = [&](const std::vector<int>& g, int r, int c) {
        int def = 0;
        int idx = r * grid_size + c;
        Tile t = alphabet[g[idx]];
        if (r > 0 && t.top != alphabet[g[(r - 1) * grid_size + c]].bottom) def++;
        if (r < grid_size - 1 && t.bottom != alphabet[g[(r + 1) * grid_size + c]].top) def++;
        if (c > 0 && t.left != alphabet[g[r * grid_size + (c - 1)]].right) def++;
        if (c < grid_size - 1 && t.right != alphabet[g[r * grid_size + (c + 1)]].left) def++;
        return def;
    };

    auto is_tile_defective = [&](const std::vector<int>& g, int r, int c) {
        if (r < 0 || r >= grid_size || c < 0 || c >= grid_size) return false;
        return get_local_edges(g, r, c) > 0;
    };

    int core32_depth = (grid_size - 32) / 2;
    int core_depth = (grid_size - 48) / 2;
    int frame_depth = 6;
    auto in_core32 = [&](int r, int c) { return (r >= core32_depth && r < grid_size - core32_depth && c >= core32_depth && c < grid_size - core32_depth); };
    auto in_core48 = [&](int r, int c) { return (r >= core_depth && r < grid_size - core_depth && c >= core_depth && c < grid_size - core_depth); };
    auto in_frame6 = [&](int r, int c) { return (r < frame_depth || r >= grid_size - frame_depth || c < frame_depth || c >= grid_size - frame_depth); };

    // Initialize all global spatial trackers
    std::vector<int> h_seam_defects(grid_size - 1, 0);
    std::vector<int> v_seam_defects(grid_size - 1, 0);
    std::vector<std::vector<int>> block_defects(8, std::vector<int>(8, 0));
    int current_core32 = 0, current_core48 = 0, current_frame6 = 0;

    // Pre-calculate true seam defects
    for (int r = 0; r < grid_size - 1; r++) {
        for (int c = 0; c < grid_size; c++) {
            if (alphabet[forged_grid[r * grid_size + c]].bottom != alphabet[forged_grid[(r + 1) * grid_size + c]].top) h_seam_defects[r]++;
        }
    }
    for (int c = 0; c < grid_size - 1; c++) {
        for (int r = 0; r < grid_size; r++) {
            if (alphabet[forged_grid[r * grid_size + c]].right != alphabet[forged_grid[r * grid_size + (c + 1)]].left) v_seam_defects[c]++;
        }
    }

    for (int r = 0; r < grid_size; r++) {
        for (int c = 0; c < grid_size; c++) {
            if (is_tile_defective(forged_grid, r, c)) {
                block_defects[r / 8][c / 8]++;
                if (in_core32(r, c)) current_core32++;
                if (in_core48(r, c)) current_core48++;
                if (in_frame6(r, c)) current_frame6++;
            }
        }
    }

    int current_defects = count_grid_defects(forged_grid.data(), grid_size, alphabet);
    int current_total_aliens = 0;
    for (int i = 0; i < grid_size * grid_size; i++) {
        if (forged_grid[i] != plane_A[i] && forged_grid[i] != plane_B[i]) current_total_aliens++;
    }

    struct TileState { int r, c; bool defective; };
    struct Move {
        int r, c, tile_id;
        int delta_tcv;    // LOWER is better
        int delta_edges;  // LOWER is better (tie breaker)
        int delta_aliens; // HIGHER is better (tie breaker)
        std::vector<TileState> affected_tiles_old;
    };

    bool improved = true;
    while (improved) {
        improved = false;
        Move best_move = {-1, -1, -1, -9999, 9999, 0, {}};
        bool found_move = false;

        for (int r = 0; r < grid_size; r++) {     
            for (int c = 0; c < grid_size; c++) {
                int idx = r * grid_size + c;
                int original_tile = forged_grid[idx];
                
                int old_edges = get_local_edges(forged_grid, r, c);
                bool old_alien = (original_tile != plane_A[idx] && original_tile != plane_B[idx]);

                // Snapshot the 5 affected tiles
                std::vector<TileState> affected_tiles;
                int dr[] = {0, -1, 1, 0, 0};
                int dc[] = {0, 0, 0, -1, 1};
                for(int i = 0; i < 5; i++) {
                    int nr = r + dr[i], nc = c + dc[i];
                    if(nr >= 0 && nr < grid_size && nc >= 0 && nc < grid_size) {
                        affected_tiles.push_back({nr, nc, is_tile_defective(forged_grid, nr, nc)});
                    }
                }

                for (int new_t = 0; new_t < alphabet.size(); new_t++) {
                    if (new_t == original_tile) continue;
                    
                    bool pk_violation = false;
                    if (r == 0 && boundary_mask[idx] == 1 && alphabet[new_t].top != alphabet[public_key_grid[idx]].top) pk_violation = true;
                    if (r == grid_size - 1 && boundary_mask[idx] == 1 && alphabet[new_t].bottom != alphabet[public_key_grid[idx]].bottom) pk_violation = true;
                    if (c == 0 && boundary_mask[idx] == 1 && alphabet[new_t].left != alphabet[public_key_grid[idx]].left) pk_violation = true;
                    if (c == grid_size - 1 && boundary_mask[idx] == 1 && alphabet[new_t].right != alphabet[public_key_grid[idx]].right) pk_violation = true;
                    if (pk_violation) continue;

                    forged_grid[idx] = new_t; 
                    int new_edges = get_local_edges(forged_grid, r, c);
                    bool new_alien = (new_t != plane_A[idx] && new_t != plane_B[idx]);

                    int d_edges = new_edges - old_edges;
                    int d_alien = new_alien - old_alien;

                    // Safely Aggregate Deltas Before Scoring
                    int d_core32 = 0, d_core48 = 0, d_frame6 = 0;
                    
                    // Fast, zero-allocation inline trackers for the 5 affected tiles
                    int b_deltas[5];
                    std::pair<int, int> unique_b[5];
                    int num_b = 0;

                    for (const auto& at : affected_tiles) {
                        bool new_defective = is_tile_defective(forged_grid, at.r, at.c);
                        if (at.defective != new_defective) {
                            int delta = new_defective ? 1 : -1;

                            if (in_core32(at.r, at.c)) d_core32 += delta;
                            if (in_core48(at.r, at.c)) d_core48 += delta;
                            if (in_frame6(at.r, at.c)) d_frame6 += delta;

                            // Aggregate Block Deltas
                            bool found_b = false;
                            for (int i = 0; i < num_b; i++) { if (unique_b[i].first == at.r/8 && unique_b[i].second == at.c/8) { b_deltas[i] += delta; found_b = true; break; } }
                            if (!found_b) { unique_b[num_b] = {at.r/8, at.c/8}; b_deltas[num_b] = delta; num_b++; }
                        }
                    }

                    // Calculate exact seam differences
                    int old_h_above = 0, new_h_above = 0, old_h_below = 0, new_h_below = 0;
                    int old_v_left = 0, new_v_left = 0, old_v_right = 0, new_v_right = 0;

                    if (r > 0) {
                        old_h_above = (alphabet[original_tile].top != alphabet[forged_grid[(r - 1) * grid_size + c]].bottom) ? 1 : 0;
                        new_h_above = (alphabet[new_t].top != alphabet[forged_grid[(r - 1) * grid_size + c]].bottom) ? 1 : 0;
                    }
                    if (r < grid_size - 1) {
                        old_h_below = (alphabet[original_tile].bottom != alphabet[forged_grid[(r + 1) * grid_size + c]].top) ? 1 : 0;
                        new_h_below = (alphabet[new_t].bottom != alphabet[forged_grid[(r + 1) * grid_size + c]].top) ? 1 : 0;
                    }
                    if (c > 0) {
                        old_v_left = (alphabet[original_tile].left != alphabet[forged_grid[r * grid_size + (c - 1)]].right) ? 1 : 0;
                        new_v_left = (alphabet[new_t].left != alphabet[forged_grid[r * grid_size + (c - 1)]].right) ? 1 : 0;
                    }
                    if (c < grid_size - 1) {
                        old_v_right = (alphabet[original_tile].right != alphabet[forged_grid[r * grid_size + (c + 1)]].left) ? 1 : 0;
                        new_v_right = (alphabet[new_t].right != alphabet[forged_grid[r * grid_size + (c + 1)]].left) ? 1 : 0;
                    }

                    // 1. Calculate Spatial TCV components BEFORE and AFTER swap
                    int old_line_viol = 0, new_line_viol = 0;
                    if (target_line != -1) {
                        if (r > 0) {
                            old_line_viol += std::max(0, h_seam_defects[r - 1] - target_line);
                            new_line_viol += std::max(0, h_seam_defects[r - 1] - old_h_above + new_h_above - target_line);
                        }
                        if (r < grid_size - 1) {
                            old_line_viol += std::max(0, h_seam_defects[r] - target_line);
                            new_line_viol += std::max(0, h_seam_defects[r] - old_h_below + new_h_below - target_line);
                        }
                        if (c > 0) {
                            old_line_viol += std::max(0, v_seam_defects[c - 1] - target_line);
                            new_line_viol += std::max(0, v_seam_defects[c - 1] - old_v_left + new_v_left - target_line);
                        }
                        if (c < grid_size - 1) {
                            old_line_viol += std::max(0, v_seam_defects[c] - target_line);
                            new_line_viol += std::max(0, v_seam_defects[c] - old_v_right + new_v_right - target_line);
                        }
                    }

                    int old_block_viol = 0, new_block_viol = 0;
                    if (target_block != -1) {
                        for (int i = 0; i < num_b; i++) {
                            old_block_viol += std::max(0, block_defects[unique_b[i].first][unique_b[i].second] - target_block);
                            new_block_viol += std::max(0, block_defects[unique_b[i].first][unique_b[i].second] + b_deltas[i] - target_block);
                        }
                    }

                    // 2. Calculate Global TCV Before and After
                    int old_TCV = old_line_viol + old_block_viol;
                    if (defect_limit != -1)  old_TCV += std::max(0, current_defects - defect_limit);
                    if (alien_limit != -1)   old_TCV += std::max(0, alien_limit - current_total_aliens);
                    if (target_core32 != -1) old_TCV += std::max(0, target_core32 - current_core32);
                    if (target_core48 != -1) old_TCV += std::max(0, target_core48 - current_core48);
                    if (target_frame6 != -1) old_TCV += std::max(0, current_frame6 - target_frame6);

                    int new_TCV = new_line_viol + new_block_viol;
                    if (defect_limit != -1)  new_TCV += std::max(0, (current_defects + d_edges) - defect_limit);
                    if (alien_limit != -1)   new_TCV += std::max(0, alien_limit - (current_total_aliens + d_alien));
                    if (target_core32 != -1) new_TCV += std::max(0, target_core32 - (current_core32 + d_core32));
                    if (target_core48 != -1) new_TCV += std::max(0, target_core48 - (current_core48 + d_core48));
                    if (target_frame6 != -1) new_TCV += std::max(0, (current_frame6 + d_frame6) - target_frame6);

                    int delta_TCV = new_TCV - old_TCV;

                    // 3. The Decision: Strict Hierarchy
                    // We must enforce a strict priority hierarchy when the TCV is neutral. If a move doesn't improve 
                    // the global TCV, the optimizer should only be allowed to spend defects if it actively fixes a constraint.
                    bool is_improvement = false;
                    
                    if (delta_TCV < 0) {
                        // Always accept mathematically superior grids
                        is_improvement = true;
                    } else if (delta_TCV == 0) {
                        // Strict Hierarchy for Neutral Moves
                        if (d_edges < 0) {
                            // Priority 1: Bank defects without hurting TCV
                            is_improvement = true;
                        } else if (d_edges == 0 && d_alien > 0) {
                            // Priority 2: Gain aliens ONLY IF it costs exactly 0 defects
                            is_improvement = true;
                        }
                        // Notice: If d_edges > 0, we reject the move, even if it gains an alien.
                    }

                    if (is_improvement) {
                        Move m = {r, c, new_t, delta_TCV, d_edges, d_alien, affected_tiles};
                        
                        bool is_better = false;
                        if (!found_move) {
                            is_better = true;
                        } else if (m.delta_tcv < best_move.delta_tcv) {
                            is_better = true;
                        } else if (m.delta_tcv == best_move.delta_tcv) {
                            // Strict Hierarchy Tie-breakers
                            if (m.delta_edges < best_move.delta_edges) {
                                is_better = true;
                            } else if (m.delta_edges == best_move.delta_edges && m.delta_aliens > best_move.delta_aliens) {
                                is_better = true;
                            }
                        }

                        if (is_better) {
                            best_move = m;
                            found_move = true;
                        }
                    }

                    forged_grid[idx] = original_tile; // Revert
                }
            }
        }

        // Apply the mathematically best move and update live trackers
        if (found_move) {
            int app_r = best_move.r;
            int app_c = best_move.c;
            int old_app_t = forged_grid[app_r * grid_size + app_c];
            int new_app_t = best_move.tile_id;

            // Update Seam Trackers
            if (app_r > 0) {
                if (alphabet[old_app_t].top != alphabet[forged_grid[(app_r - 1) * grid_size + app_c]].bottom) h_seam_defects[app_r - 1]--;
                if (alphabet[new_app_t].top != alphabet[forged_grid[(app_r - 1) * grid_size + app_c]].bottom) h_seam_defects[app_r - 1]++;
            }
            if (app_r < grid_size - 1) {
                if (alphabet[old_app_t].bottom != alphabet[forged_grid[(app_r + 1) * grid_size + app_c]].top) h_seam_defects[app_r]--;
                if (alphabet[new_app_t].bottom != alphabet[forged_grid[(app_r + 1) * grid_size + app_c]].top) h_seam_defects[app_r]++;
            }
            if (app_c > 0) {
                if (alphabet[old_app_t].left != alphabet[forged_grid[app_r * grid_size + (app_c - 1)]].right) v_seam_defects[app_c - 1]--;
                if (alphabet[new_app_t].left != alphabet[forged_grid[app_r * grid_size + (app_c - 1)]].right) v_seam_defects[app_c - 1]++;
            }
            if (app_c < grid_size - 1) {
                if (alphabet[old_app_t].right != alphabet[forged_grid[app_r * grid_size + (app_c + 1)]].left) v_seam_defects[app_c]--;
                if (alphabet[new_app_t].right != alphabet[forged_grid[app_r * grid_size + (app_c + 1)]].left) v_seam_defects[app_c]++;
            }

            // Now apply the tile
            forged_grid[app_r * grid_size + app_c] = new_app_t;
            current_defects += best_move.delta_edges;
            current_total_aliens += best_move.delta_aliens;
            
            for (const auto& at : best_move.affected_tiles_old) {
                bool new_defective = is_tile_defective(forged_grid, at.r, at.c);
                if (at.defective != new_defective) {
                    int delta = new_defective ? 1 : -1;
                    block_defects[at.r / 8][at.c / 8] += delta;
                    if (in_core32(at.r, at.c)) current_core32 += delta;
                    if (in_core48(at.r, at.c)) current_core48 += delta;
                    if (in_frame6(at.r, at.c)) current_frame6 += delta;
                }
            }
            improved = true;
        }
    }
}
