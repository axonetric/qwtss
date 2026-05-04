#include "QwtssCoreCpu.h"
#include "QwtssConfig.h"
#include <vector>
#include <random>
#include <cmath>
#include <iostream>
#include <array>
#include <cstdint>
#include <cmath>

#ifdef _OPENMP
#include <omp.h>
#endif

// ----------------------------------------------------------------------------------------
// Local Energy Calculation
// ----------------------------------------------------------------------------------------
// Note: this function is now deprecated in favor of the faster cache-based logic inline in sweep_checkerboard_cpu
inline int calculate_local_energy_cpu_slow(
    const int* grid, int grid_size, int r, int c, int proposed_tile_idx, const std::vector<Tile>& alphabet) 
{
    int energy = 0;
    Tile proposed = alphabet[proposed_tile_idx];

    if (r > 0 && proposed.top != alphabet[grid[(r - 1) * grid_size + c]].bottom) energy++;
    if (r < grid_size - 1 && proposed.bottom != alphabet[grid[(r + 1) * grid_size + c]].top) energy++;
    if (c > 0 && proposed.left != alphabet[grid[r * grid_size + (c - 1)]].right) energy++;
    if (c < grid_size - 1 && proposed.right != alphabet[grid[r * grid_size + (c + 1)]].left) energy++;

    return energy;
}

// ----------------------------------------------------------------------------------------
// Global Defect Counter
// ----------------------------------------------------------------------------------------
int count_grid_defects_cpu(const int* grid, int grid_size, const std::vector<Tile>& alphabet) 
{
    int defects = 0;
    for (int r = 0; r < grid_size; r++) {
        for (int c = 0; c < grid_size; c++) {
            Tile t = alphabet[grid[r * grid_size + c]];
            
            // Only check right and bottom edges to avoid double-counting internal collisions
            if (c < grid_size - 1 && t.right != alphabet[grid[r * grid_size + (c + 1)]].left) defects++;
            if (r < grid_size - 1 && t.bottom != alphabet[grid[(r + 1) * grid_size + c]].top) defects++;
        }
    }
    return defects;
}

// ----------------------------------------------------------------------------------------
// Global Defect Counter (OpenMP Reduction Optimized)
// ----------------------------------------------------------------------------------------
int count_grid_defects_cpu_with_omp(const int* grid, int grid_size, const std::vector<Tile>& alphabet) 
{
    int defects = 0;
    
    // Leverage the existing thread pool to evaluate the grid simultaneously.
    // The reduction clause safely aggregates the local counts into the global 'defects' variable.
    #pragma omp parallel for reduction(+:defects)
    for (int r = 0; r < grid_size; r++) {
        for (int c = 0; c < grid_size; c++) {
            Tile t = alphabet[grid[r * grid_size + c]];
            
            // Only check right and bottom edges to avoid double-counting
            if (c < grid_size - 1 && t.right != alphabet[grid[r * grid_size + (c + 1)]].left) defects++;
            if (r < grid_size - 1 && t.bottom != alphabet[grid[(r + 1) * grid_size + c]].top) defects++;
        }
    }
    
    return defects;
}

// ----------------------------------------------------------------------------------------
// Thread-Safe Checkerboard Sweep
// ----------------------------------------------------------------------------------------
void sweep_checkerboard_cpu(
    int* grid, int grid_size, const std::vector<Tile>& alphabet,
    float current_temp, int checkerboard_offset, 
    std::vector<ChaCha20PRNG>& thread_gens, bool use_omp,
    const std::vector<uint8_t>& external_lock_mask) 
{
    int num_tiles = alphabet.size();
    int total_cells = grid_size * grid_size;

    // 1. Open the parallel region FIRST to allow thread-local instantiations
    #pragma omp parallel if(use_omp)
    {
        // Get thread-local RNG and Distribution exactly once per thread
        int thread_id = 0;
        #ifdef _OPENMP
        if (use_omp) thread_id = omp_get_thread_num();
        #endif
        ChaCha20PRNG& gen = thread_gens[thread_id];
        std::uniform_real_distribution<float> prob_dist(0.0f, 1.0f);

        // 2. Now distribute the loop iterations among the threads
        #pragma omp for schedule(static)
        for (int i = 0; i < total_cells; i++) {
            
            bool enforce_color = false;

            // 1. Evaluate Constraints
            if (!external_lock_mask.empty()) {
                uint8_t mask_val = external_lock_mask[i];
                if (mask_val == 1) {
                    enforce_color = true; // Pin the outward-facing colors
                } else if (mask_val == 2) {
                    continue; // Rigidly pin the Tile ID (Skip completely)
                }
            }

            int r = i / grid_size;
            int c = i % grid_size;

            if ((r + c) % 2 == checkerboard_offset) {
                
                // Dynamically sized to config limit to match GPU, preventing buffer overflow
                float weights[QwtssConfig::MAX_TILES]; 
                float max_weight = -1e20f;

                // 2. Read required perimeter colors if constrained
                int req_top = -1, req_bottom = -1, req_left = -1, req_right = -1;
                if (enforce_color) {
                    Tile t_current = alphabet[grid[i]];
                    if (r == 0) req_top = t_current.top;
                    if (r == grid_size - 1) req_bottom = t_current.bottom;
                    if (c == 0) req_left = t_current.left;
                    if (c == grid_size - 1) req_right = t_current.right;
                }

                // Cache neighbor colors ONCE per cell
                int n_top = (r > 0) ? alphabet[grid[i - grid_size]].bottom : -1;
                int n_bottom = (r < grid_size - 1) ? alphabet[grid[i + grid_size]].top : -1;
                int n_left = (c > 0) ? alphabet[grid[i - 1]].right : -1;
                int n_right = (c < grid_size - 1) ? alphabet[grid[i + 1]].left : -1;

                // 3. Calculate local energy for ALL possible tiles
                for (int t = 0; t < num_tiles; t++) {
                    
                    // Filter illegal boundary colors BEFORE calculating energy
                    if (enforce_color) {
                        Tile cand = alphabet[t];
                        if ((req_top != -1 && cand.top != req_top) ||
                            (req_bottom != -1 && cand.bottom != req_bottom) ||
                            (req_left != -1 && cand.left != req_left) ||
                            (req_right != -1 && cand.right != req_right)) {
                            
                            weights[t] = -1e20f; 
                            continue; 
                        }
                    }

                    // Direct cached evaluation for local energy
                    int e = 0;
                    Tile cand = alphabet[t];
                    if (n_top != -1 && cand.top != n_top) e++;
                    if (n_bottom != -1 && cand.bottom != n_bottom) e++;
                    if (n_left != -1 && cand.left != n_left) e++;
                    if (n_right != -1 && cand.right != n_right) e++;

                    weights[t] = -(float)e / current_temp;
                    if (weights[t] > max_weight) max_weight = weights[t];
                }

                // 4. Softmax / Boltzmann Distribution
                float sum = 0.0f;
                for (int t = 0; t < num_tiles; t++) {
                    weights[t] = expf(weights[t] - max_weight); // Single precision to match GPU
                    sum += weights[t];
                }

                // 5. Sample from the distribution
                float rand_val = prob_dist(gen) * sum;
                float cumulative = 0.0f;
                // Fallback to the current tile (which is mathematically guaranteed 
                // to be valid) in case we encounter the rare floating-point roulette wheel bug.
                int selected_tile = grid[i];
                for (int t = 0; t < num_tiles; t++) {
                    cumulative += weights[t];
                    // Explicitly ensure 0-weight options can never be selected
                    if (weights[t] > 0.0f && rand_val <= cumulative) {
                        selected_tile = t;
                        break;
                    }
                }

                grid[i] = selected_tile;
            }
        }
    }
}

// ----------------------------------------------------------------------------------------
// Main CPU Entry Point (Fast Annealing)
// ----------------------------------------------------------------------------------------
GridAnnealResult grid_anneal_cpu(
    int* grid, 
    int grid_size,
    ITileSet* tileset, 
    GridAnnealParams params,
    ChaCha20PRNG& rng,
    const std::vector<uint8_t>& external_lock_mask,
    int target_threads, // Default: -1 (AUTO: 70% of available threads)
    bool do_logging)    // Default: False
{
    std::vector<Tile> alphabet = tileset->get_tiles();
    int num_tiles = tileset->get_size();

    if (num_tiles > QwtssConfig::MAX_TILES) {
        throw std::runtime_error("Alphabet size exceeds QwtssConfig::MAX_TILES allocation.");
    }
    
    // Process Pinned Masks exactly like the GPU
    std::vector<uint8_t> active_lock_mask = external_lock_mask;
    
    if (active_lock_mask.empty() && params.do_pinned_defect_doping) {
        int num_pins = std::min(params.max_allowed_defects, std::max(10, params.max_allowed_defects / 10));
        double min_radius = std::sqrt((grid_size * grid_size) / (num_pins * M_PI)) * 0.8;
        active_lock_mask = generate_quenched_disorder_mask(grid_size, num_pins, min_radius, rng);
    }

    // Determine thread count
    bool use_omp = true;
    int num_threads = 1;
    std::string max_threads_details = "";

    #ifdef _OPENMP
    if (target_threads == 1) {
        use_omp = false; // Bypass OpenMP overhead entirely for single-thread requests
    } else {
        int max_threads = omp_get_max_threads();
        max_threads_details = " / " + std::to_string(max_threads);
        
        if (target_threads > 1) {
            // User requested a specific number of threads
            num_threads = std::min(target_threads, max_threads);
        } else {
            // AUTO Mode (-1): Target ~70% of available hardware threads
            num_threads = std::max(1, static_cast<int>((double)max_threads * 0.70));
        }

        if (num_threads < 2){
            use_omp = false;
            num_threads = 1;
        }
        else omp_set_num_threads(num_threads); // Explicitly cap the OMP worker pool
    }
    #else
    use_omp = false;
    #endif

    if (do_logging){
        std::cout << "CPU Key Generation (multithreading " << (use_omp ? "enabled" : "disabled") << "): "
                << num_threads << max_threads_details << " threads will be used" << std::endl;
    }

    // Seed thread-local RNGs
    std::vector<ChaCha20PRNG> thread_gens;
    for (int i = 0; i < num_threads; i++) {
        // Seed thread-local generators to ensure deterministic multi-threading 
        // Securely cascade 256 bits of entropy into each thread's constructor
        std::array<uint32_t, 8> thread_seed;
        for (int j = 0; j < 4; j++) {
            uint64_t chunk = rng.next_u64();
            thread_seed[j * 2] = static_cast<uint32_t>(chunk);
            thread_seed[j * 2 + 1] = static_cast<uint32_t>(chunk >> 32);
        }
        thread_gens.emplace_back(thread_seed);
    }

    // --- ISOTROPIC GREEDY PRE-QUENCH ---
    if (params.do_greedy_prequench) {
        float quench_temp = 0.05f; 
        for (int step = 0; step < 50; step++) {
            sweep_checkerboard_cpu(grid, grid_size, alphabet, quench_temp, 0, thread_gens, use_omp, active_lock_mask);
            sweep_checkerboard_cpu(grid, grid_size, alphabet, quench_temp, 1, thread_gens, use_omp, active_lock_mask);
        }
    }

    // --- THERMODYNAMIC STATE TRACKING ---
    float current_temp = (params.do_greedy_prequench ? params.reheat_temp : params.initial_temp);
    int last_defects = INT32_MAX;
    int stagnation_counter = 0;

    // --- MAIN OPTIMIZATION LOOP ---
    for (int step = 1; step <= params.max_steps; step++) {
        
        // 1. Evaluate State
        if (step % params.check_interval == 0) {
            int defects = count_grid_defects_cpu(grid, grid_size, alphabet);
            
            // Goal condition met: The interior has settled into a valid configuration with tolerance
            if (defects <= params.max_allowed_defects && defects >= params.min_allowed_defects) {
                return  { defects, current_temp, step }; // SUCCESS
            }

            // TOO FEW DEFECTS PENALTY: We are stuck below the floor
            if (defects < params.min_allowed_defects) {
                // Thermodynamic Rewind: Spike the temperature to "re-melt" the crystal
                float rewind_factor = std::pow(params.cooling_rate, -(float)params.check_interval);
                current_temp = current_temp * rewind_factor * 2.0f;

                // Safety clamp
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
                if (current_temp >= params.reheat_temp) {
                    current_temp *= 2.0f;
                } else {
                    current_temp = std::max(params.reheat_temp, 1.2f * current_temp);
                }
                // Cap at 10.0f to match GPU
                if (current_temp > 10.0f) current_temp = 10.0f;
                stagnation_counter = 0;
            }
        }

        // 2. MCMC Heat Bath Phase (Checkerboard updates avoid race conditions)
        sweep_checkerboard_cpu(grid, grid_size, alphabet, current_temp, 0, thread_gens, use_omp, active_lock_mask);
        sweep_checkerboard_cpu(grid, grid_size, alphabet, current_temp, 1, thread_gens, use_omp, active_lock_mask);

        // 3. Geometric Cooling
        current_temp *= params.cooling_rate;
    }

    // Failed to hit target within params.max_steps
    return  { -1, 0.0f, params.max_steps }; // FAILED
}

// ----------------------------------------------------------------------------------------
// // Main CPU Entry Point (Adiabatic Annealing)
// ----------------------------------------------------------------------------------------
GridAnnealResult grid_anneal_cpu_adiabatic(
    int* grid, 
    int grid_size,
    ITileSet* tileset, 
    GridAnnealParams params,
    ChaCha20PRNG& rng,
    const std::vector<uint8_t>& external_lock_mask,
    int target_threads, // Default: -1 (AUTO: 70% of available threads)
    bool do_logging)    // Default: false
{
    std::vector<Tile> alphabet = tileset->get_tiles();
    int num_tiles = tileset->get_size();

    if (num_tiles > QwtssConfig::MAX_TILES) {
        throw std::runtime_error("Alphabet size exceeds QwtssConfig::MAX_TILES allocation.");
    }
    
    // Process Pinned Masks exactly like the standard CPU/GPU routines
    std::vector<uint8_t> active_lock_mask = external_lock_mask;
    
    if (active_lock_mask.empty() && params.do_pinned_defect_doping) {
        int num_pins = std::min(params.max_allowed_defects, std::max(10, params.max_allowed_defects / 10));
        double min_radius = std::sqrt((grid_size * grid_size) / (num_pins * M_PI)) * 0.8;
        active_lock_mask = generate_quenched_disorder_mask(grid_size, num_pins, min_radius, rng);
    }

    // Determine thread count
    bool use_omp = true;
    int num_threads = 1;
    std::string max_threads_details = "";

    #ifdef _OPENMP
    if (target_threads == 1) {
        use_omp = false; // Bypass OpenMP overhead entirely for single-thread requests
    } else {
        int max_threads = omp_get_max_threads();
        max_threads_details = " / " + std::to_string(max_threads);
        
        if (target_threads > 1) {
            // User requested a specific number of threads
            num_threads = std::min(target_threads, max_threads);
        } else {
            // AUTO Mode (-1): Target ~70% of available hardware threads
            num_threads = std::max(1, static_cast<int>((double)max_threads * 0.70));
        }

        if (num_threads < 2){
            use_omp = false;
            num_threads = 1;
        }
        else omp_set_num_threads(num_threads); // Explicitly cap the OMP worker pool
    }
    #else
    use_omp = false;
    #endif

    if (do_logging) {
        std::cout << "CPU Key Generation (Adiabatic, multithreading " << (use_omp ? "enabled" : "disabled") << "): "
                  << num_threads << max_threads_details << " threads will be used" << std::endl;
    }

    // Seed thread-local RNGs
    std::vector<ChaCha20PRNG> thread_gens;
    for (int i = 0; i < num_threads; i++) {
        // Seed thread-local generators to ensure deterministic multi-threading 
        // Securely cascade 256 bits of entropy into each thread's constructor
        std::array<uint32_t, 8> thread_seed;
        for (int j = 0; j < 4; j++) {
            uint64_t chunk = rng.next_u64();
            thread_seed[j * 2] = static_cast<uint32_t>(chunk);
            thread_seed[j * 2 + 1] = static_cast<uint32_t>(chunk >> 32);
        }
        thread_gens.emplace_back(thread_seed);
    }

    // --- ISOTROPIC GREEDY PRE-QUENCH ---
    if (params.do_greedy_prequench) {
        float quench_temp = 0.05f; 
        for (int step = 0; step < 50; step++) {
            sweep_checkerboard_cpu(grid, grid_size, alphabet, quench_temp, 0, thread_gens, use_omp, active_lock_mask);
            sweep_checkerboard_cpu(grid, grid_size, alphabet, quench_temp, 1, thread_gens, use_omp, active_lock_mask);
        }
    }

    // --- ADIABATIC THERMODYNAMIC SCHEDULE ---
    const int num_beta_steps = 5000; 
    const double beta_start = 0.20; // No point at starting at even lower beta; 0.20 is already max boiling
    const double fixed_beta_end = 11.0;

    int total_sweeps_executed = 0;

    // Walk down the exact thermodynamic schedule from the AIS scout chains
    for (int step = 0; step < num_beta_steps - 1; ++step) {
        
        // Calculate the cubic Beta progression
        double next_fraction = (double)(step + 1) / (num_beta_steps - 1);
        double next_beta = beta_start + (fixed_beta_end - beta_start) * std::pow(next_fraction, 3.0);

        // Convert Beta to Temperature
        float step_temp = (next_beta == 0.0) ? 1000.0f : (float)(1.0 / next_beta);

        // Dynamically scale the sweep volume based on the temperature regime
        int sweeps = 80;
        if (step_temp > 5.0f) sweeps = 1;      
        else if (step_temp > 2.0f) sweeps = 5;  
        else if (step_temp > 1.0f) sweeps = 10;
        else if (step_temp > 0.5f) sweeps = 40;
        else if (step_temp > 0.2f) sweeps = 60;
        else sweeps = 80;

        // Execute the MCMC Heat Bath Phase
        for (int i = 0; i < sweeps; i++) {
            sweep_checkerboard_cpu(grid, grid_size, alphabet, step_temp, 0, thread_gens, use_omp, active_lock_mask);
            sweep_checkerboard_cpu(grid, grid_size, alphabet, step_temp, 1, thread_gens, use_omp, active_lock_mask);
        }
        total_sweeps_executed += sweeps;

        // Check if the grid has organically entered the target defect band
        int defects = count_grid_defects_cpu(grid, grid_size, alphabet);

        if (defects <= params.max_allowed_defects) {
            if (defects >= params.min_allowed_defects) {
                return { defects, step_temp, total_sweeps_executed }; // SUCCESS
            } else {
                // Overshot the target.
                // Adiabatic mode strictly prevents returns to higher E; fast-fail to preserve compute cycles
                return { -1, 0.0f, total_sweeps_executed }; // FAILED
            }
        }
    }

    // Cleanup if the schedule completed without hitting the target band
    return { -1, 0.0f, total_sweeps_executed }; // FAILED
}

// -------------------- OPTIMIZED LOGIC ---------------------------------------------------

// ----------------------------------------------------------------------------------------
// CPU Fixed-Point MCMC Optimization
// ----------------------------------------------------------------------------------------
struct FixedPointWeightsCpu {
    uint32_t w[5];
};

inline FixedPointWeightsCpu calc_fixed_point_weights_cpu(float temp) {
    FixedPointWeightsCpu fw;
    for (int delta_e = 0; delta_e <= 4; delta_e++) {
        double prob = std::exp(-(double)delta_e / (double)temp);
        fw.w[delta_e] = std::max((uint32_t)1, static_cast<uint32_t>(prob * 16777216.0));
    }
    return fw;
}

// Thread-Safe Checkerboard Sweep (Fixed Point Integer Optimized & Branchless)
void sweep_checkerboard_cpu_optimized(
    int* grid, int grid_size, const std::vector<Tile>& alphabet,
    const FixedPointWeightsCpu& fw, int checkerboard_offset, 
    std::vector<ChaCha20PRNG>& thread_gens, bool use_omp,
    const std::vector<uint8_t>& external_lock_mask) 
{
    int num_tiles = alphabet.size();
    const Tile* alphabet_ptr = alphabet.data();

    #pragma omp parallel if(use_omp)
    {
        int thread_id = 0;
        #ifdef _OPENMP
        if (use_omp) thread_id = omp_get_thread_num();
        #endif
        ChaCha20PRNG& gen = thread_gens[thread_id];

        // Distribute the rows among the threads
        #pragma omp for schedule(static)
        for (int r = 0; r < grid_size; r++) {
            
            // Calculate the starting column for this row's color
            int start_c = ((r % 2) == checkerboard_offset) ? 0 : 1;
            
            // Increment by 2 to ONLY hit the active checkerboard cells
            for (int c = start_c; c < grid_size; c += 2) {
                int i = r * grid_size + c;
                
                bool enforce_color = false;

                if (!external_lock_mask.empty()) {
                    uint8_t mask_val = external_lock_mask[i];
                    if (mask_val == 1) enforce_color = true; 
                    else if (mask_val == 2) continue; 
                }

                int req_top = -1, req_bottom = -1, req_left = -1, req_right = -1;
                if (enforce_color) {
                    Tile t_current = alphabet_ptr[grid[i]];
                    if (r == 0) req_top = t_current.top;
                    if (r == grid_size - 1) req_bottom = t_current.bottom;
                    if (c == 0) req_left = t_current.left;
                    if (c == grid_size - 1) req_right = t_current.right;
                }

                int n_top = (r > 0) ? alphabet_ptr[grid[i - grid_size]].bottom : -1;
                int n_bottom = (r < grid_size - 1) ? alphabet_ptr[grid[i + grid_size]].top : -1;
                int n_left = (c > 0) ? alphabet_ptr[grid[i - 1]].right : -1;
                int n_right = (c < grid_size - 1) ? alphabet_ptr[grid[i + 1]].left : -1;

                int energies[QwtssConfig::MAX_TILES];
                int min_energy = 999;

                for (int t = 0; t < num_tiles; t++) {
                    if (enforce_color) {
                        Tile cand = alphabet_ptr[t];
                        if ((req_top != -1 && cand.top != req_top) ||
                            (req_bottom != -1 && cand.bottom != req_bottom) ||
                            (req_left != -1 && cand.left != req_left) ||
                            (req_right != -1 && cand.right != req_right)) {
                            
                            energies[t] = 999; 
                            continue; 
                        }
                    }

                    Tile cand = alphabet_ptr[t];
                    
                    // Branchless energy calculation (allows compiler auto-vectorization)
                    int e = 0;
                    e += (n_top != -1 && cand.top != n_top);
                    e += (n_bottom != -1 && cand.bottom != n_bottom);
                    e += (n_left != -1 && cand.left != n_left);
                    e += (n_right != -1 && cand.right != n_right);

                    energies[t] = e;
                    if (e < min_energy) min_energy = e;
                }

                uint32_t weights[QwtssConfig::MAX_TILES];
                uint32_t sum = 0;
                for (int t = 0; t < num_tiles; t++) {
                    if (energies[t] == 999) {
                        weights[t] = 0;
                    } else {
                        int delta_e = energies[t] - min_energy;
                        if (delta_e > 4) delta_e = 4;
                        weights[t] = fw.w[delta_e];
                        sum += weights[t];
                    }
                }

                if (sum == 0) continue; 

                // Entropy Recycling handled natively by the ChaCha20PRNG class
                uint32_t rand_val = gen.next_u32(); 
                uint32_t r_val = (uint32_t)(((uint64_t)rand_val * (uint64_t)sum) >> 32);

                uint32_t cumulative = 0;
                int selected_tile = grid[i];
                for (int t = 0; t < num_tiles; t++) {
                    if (weights[t] > 0) {
                        cumulative += weights[t];
                        if (r_val < cumulative) {
                            selected_tile = t;
                            break;
                        }
                    }
                }

                grid[i] = selected_tile;
            }
        }
    }
}

// ----------------------------------------------------------------------------------------
// Main CPU Entry Point (Optimized Integer Fast Annealing)
// ----------------------------------------------------------------------------------------
GridAnnealResult grid_anneal_cpu_optimized(
    int* grid, 
    int grid_size,
    ITileSet* tileset, 
    GridAnnealParams params,
    ChaCha20PRNG& rng,
    const std::vector<uint8_t>& external_lock_mask,
    int target_threads, 
    bool do_logging)
{
    std::vector<Tile> alphabet = tileset->get_tiles();
    int num_tiles = tileset->get_size();

    if (num_tiles > QwtssConfig::MAX_TILES) {
        throw std::runtime_error("Alphabet size exceeds QwtssConfig::MAX_TILES allocation.");
    }
    
    std::vector<uint8_t> active_lock_mask = external_lock_mask;
    
    if (active_lock_mask.empty() && params.do_pinned_defect_doping) {
        int num_pins = std::min(params.max_allowed_defects, std::max(10, params.max_allowed_defects / 10));
        double min_radius = std::sqrt((grid_size * grid_size) / (num_pins * M_PI)) * 0.8;
        active_lock_mask = generate_quenched_disorder_mask(grid_size, num_pins, min_radius, rng);
    }

    bool use_omp = true;
    int num_threads = 1;
    std::string max_threads_details = "";

    #ifdef _OPENMP
    if (target_threads == 1) {
        use_omp = false; 
    } else {
        int max_threads = omp_get_max_threads();
        max_threads_details = " / " + std::to_string(max_threads);
        
        if (target_threads > 1) {
            num_threads = std::min(target_threads, max_threads);
        } else {
            num_threads = std::max(1, static_cast<int>((double)max_threads * 0.70));
        }

        if (num_threads < 2){
            use_omp = false;
            num_threads = 1;
        }
        else omp_set_num_threads(num_threads); 
    }
    #else
    use_omp = false;
    #endif

    if (do_logging){
        std::cout << "CPU Key Generation (Optimized, multithreading " << (use_omp ? "enabled" : "disabled") << "): "
                << num_threads << max_threads_details << " threads will be used" << std::endl;
    }

    std::vector<ChaCha20PRNG> thread_gens;
    for (int i = 0; i < num_threads; i++) {
        std::array<uint32_t, 8> thread_seed;
        for (int j = 0; j < 4; j++) {
            uint64_t chunk = rng.next_u64();
            thread_seed[j * 2] = static_cast<uint32_t>(chunk);
            thread_seed[j * 2 + 1] = static_cast<uint32_t>(chunk >> 32);
        }
        thread_gens.emplace_back(thread_seed);
    }

    // --- ISOTROPIC GREEDY PRE-QUENCH ---
    if (params.do_greedy_prequench) {
        float quench_temp = 0.05f; 
        FixedPointWeightsCpu quench_fw = calc_fixed_point_weights_cpu(quench_temp);
        for (int step = 0; step < 50; step++) {
            sweep_checkerboard_cpu_optimized(grid, grid_size, alphabet, quench_fw, 0, thread_gens, use_omp, active_lock_mask);
            sweep_checkerboard_cpu_optimized(grid, grid_size, alphabet, quench_fw, 1, thread_gens, use_omp, active_lock_mask);
        }
    }

    // --- THERMODYNAMIC STATE TRACKING ---
    float current_temp = (params.do_greedy_prequench ? params.reheat_temp : params.initial_temp);
    FixedPointWeightsCpu current_fw = calc_fixed_point_weights_cpu(current_temp);
    int last_defects = INT32_MAX;
    int stagnation_counter = 0;

    // --- MAIN OPTIMIZATION LOOP ---
    for (int step = 1; step <= params.max_steps; step++) {
        
        if (step % params.check_interval == 0) {
            int defects = count_grid_defects_cpu_with_omp(grid, grid_size, alphabet);
            //int defects = count_grid_defects_cpu(grid, grid_size, alphabet);
            
            if (defects <= params.max_allowed_defects && defects >= params.min_allowed_defects) {
                return  { defects, current_temp, step }; 
            }

            bool temp_changed = false;

            if (defects < params.min_allowed_defects) {
                float rewind_factor = std::pow(params.cooling_rate, -(float)params.check_interval);
                current_temp = current_temp * rewind_factor * 2.0f;
                if (current_temp > params.reheat_temp) {
                    current_temp = params.reheat_temp;
                }
                stagnation_counter = 0;
                temp_changed = true;
            }
            else if (defects == last_defects) {
                stagnation_counter++;
            } else {
                stagnation_counter = 0;
            }
            last_defects = defects;

            if (stagnation_counter >= params.stagnation_patience) {
                if (current_temp >= params.reheat_temp) {
                    current_temp *= 2.0f;
                } else {
                    current_temp = std::max(params.reheat_temp, 1.2f * current_temp);
                }
                if (current_temp > 10.0f) current_temp = 10.0f;
                stagnation_counter = 0;
                temp_changed = true;
            }

            if (temp_changed) {
                current_fw = calc_fixed_point_weights_cpu(current_temp);
            }
        }

        // 2. MCMC Heat Bath Phase 
        sweep_checkerboard_cpu_optimized(grid, grid_size, alphabet, current_fw, 0, thread_gens, use_omp, active_lock_mask);
        sweep_checkerboard_cpu_optimized(grid, grid_size, alphabet, current_fw, 1, thread_gens, use_omp, active_lock_mask);

        // 3. Geometric Cooling
        current_temp *= params.cooling_rate;
        current_fw = calc_fixed_point_weights_cpu(current_temp);
    }

    return  { -1, 0.0f, params.max_steps };
}
