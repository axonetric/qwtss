#include "QwtssCore.h"
#include "QwtssCoreShared.h"
#include "StarkTrace.h"
#include "QwtssConfig.h"
#include "PrivateKeyDatabase.h"
#include "CryptoUtils.h"
#include "QwtssAir.h"
#include "QwtssCoreCpu.h"
#include "QwtssCoreGpuOptimized.h"
#include <string>
#include <iostream>
#include <stdexcept>
#include <cstdint>
#include <chrono>
#include <fstream>
#include <cstring>
#include <random>
#include <numeric>
#include <algorithm>
#include <cmath>
#include <stdio.h>
#include <curand_kernel.h>
#include <device_launch_parameters.h>
#include <sys/resource.h>   // For RAM profiling


// ---------------------------------------------------------
// HELPER FUNCTIONS
// ---------------------------------------------------------

// Converts the raw grid fingerprint into perfectly formatted STARK hex strings for JSON storage.
std::array<std::string, 2> serialize_fingerprint(const std::array<FieldElement256, 2>& grid_fingerprint) {
    return {
        from_fe256(grid_fingerprint[0]).ToString(),
        from_fe256(grid_fingerprint[1]).ToString()
    };
}

// Parses the JSON hex strings securely back into the raw FieldElement256 struct for verification/signing.
std::array<FieldElement256, 2> deserialize_fingerprint(const std::array<std::string, 2>& fingerprint_limbs) {
    return {
        to_fe256(StoneField::FromString(fingerprint_limbs[0])),
        to_fe256(StoneField::FromString(fingerprint_limbs[1]))
    };
}

// Encodes the 4096-element grid of JR-11 tiles (0-10) directly into a 4096-character hex string.
std::string encode_grid_to_hex(const std::vector<int>& private_grid) {
    if (private_grid.empty()) throw std::invalid_argument("private_grid cannot be empty");
    if (private_grid.size() != (QwtssReference::grid_size * QwtssReference::grid_size)) throw std::invalid_argument("private_grid is not the expected size");

    std::string hex_str;
    hex_str.reserve(private_grid.size());
    const char* hex_chars = "0123456789ABCDEF";
    
    for (int tile_id : private_grid) {
        if (tile_id < 0 || tile_id > 15) throw std::runtime_error("tile id must be a 4-bit non-negative value for packing");

        // Bitwise AND with 0x0F is a safety guard to ensure it's strictly a 4-bit value
        hex_str.push_back(hex_chars[tile_id & 0x0F]);
    }
    return hex_str;
}

std::vector<int> decode_grid_from_hex(const std::string& hex_str) {
    std::vector<int> grid;
    grid.reserve(hex_str.size());
    for (char c : hex_str) {
        if (c >= '0' && c <= '9') grid.push_back(c - '0');
        else if (c >= 'A' && c <= 'F') grid.push_back(c - 'A' + 10);
        else if (c >= 'a' && c <= 'f') grid.push_back(c - 'a' + 10);
    }
    return grid;
}

// ---------------------------------------------------------
// HOST FUNCTIONS
// ---------------------------------------------------------

// Returns -1 if the maximum number of steps is reached without success.
// @param external_lock_mask Grid tile allowed behavior mask: 0 = Free, 1 = Pinned Outward Colors, 2 = Pinned Tile ID.
GridAnnealResult grid_anneal_gpu(
    int* h_grid, 
    int grid_size,
    ITileSet* tileset, 
    GridAnnealParams params,
    ChaCha20PRNG& rng,
    const std::vector<uint8_t>& external_lock_mask
) {
    std::vector<Tile> alphabet = tileset->get_tiles();
    int num_tiles = tileset->get_size();

    if (num_tiles > QwtssConfig::MAX_TILES) {
        throw std::runtime_error("Alphabet size exceeds QwtssConfig::MAX_TILES allocation.");
    }

    // Init constant memory using proper function to prevent the "Shadow Constant Memory" bug.
    initialize_gpu_constants_from_core_shared(tileset);

    int grid_cells = grid_size * grid_size;
    int grid_bytes = grid_cells * sizeof(int);
    int *d_grid;

    CUDA_CHECK(cudaMalloc((void**)&d_grid, grid_bytes));
    CUDA_CHECK(cudaMemcpy(d_grid, h_grid, grid_bytes, cudaMemcpyHostToDevice));

    uint8_t* d_locked_mask = nullptr;
    if (!external_lock_mask.empty()) {
        CUDA_CHECK(cudaMalloc((void**)&d_locked_mask, external_lock_mask.size() * sizeof(uint8_t)));
        CUDA_CHECK(cudaMemcpy(d_locked_mask, external_lock_mask.data(), external_lock_mask.size() * sizeof(uint8_t), cudaMemcpyHostToDevice));
    } else if (params.do_pinned_defect_doping){
        int num_pins = std::min(params.max_allowed_defects, std::max(10, params.max_allowed_defects / 10));
        // Calculate a statistically safe min_radius for Poisson Disk Sampling
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

    // Allocate the Philox states and initialize them
    curandStatePhilox4_32_10_t* d_states;
    CUDA_CHECK(cudaMalloc(&d_states, grid_cells * sizeof(curandStatePhilox4_32_10_t)));

    // Initialize the PRNG states with the unique dynamic seeds
    init_curand_states<<<(grid_cells + 255) / 256, 256>>>(d_states, d_seeds, grid_size);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Free the seed array from VRAM; Philox doesn't need it anymore
    CUDA_CHECK(cudaFree(d_seeds));

    if (params.do_greedy_prequench){
        // Optional: for reporting
        CUDA_CHECK(cudaMemcpy(h_grid, d_grid, grid_bytes, cudaMemcpyDeviceToHost));
        int pre_quench_defects = count_grid_defects(h_grid, grid_size, alphabet);

        // --- ISOTROPIC GREEDY PRE-QUENCH --- Force multiple nucleation sites
        // This improves the speed of valid solutions, as well as lower defect counts and better defect distribution statistics.
        // Run exactly 50 steps at near-zero temperature. This forces all 4096 threads to act 
        // as purely greedy agents, eliminating 90% of defects without directional scarring.
        float quench_temp = 0.05f; 
        for (int step = 0; step < 50; step++) {
            heat_bath_kernel_unified<<<numBlocks, threadsPerBlock>>>(d_grid, grid_size, quench_temp, d_states, 0, d_locked_mask);
            heat_bath_kernel_unified<<<numBlocks, threadsPerBlock>>>(d_grid, grid_size, quench_temp, d_states, 1, d_locked_mask);
        }

        // Optional: Print the results of the quench
        CUDA_CHECK(cudaMemcpy(h_grid, d_grid, grid_bytes, cudaMemcpyDeviceToHost));
        int post_quench_defects = count_grid_defects(h_grid, grid_size, alphabet);
        //std::cout << "Greedy Quench Complete. Pre-quench defects: " << pre_quench_defects << ", Post-quench defects: " << post_quench_defects << std::endl;
    }

    // --- THERMODYNAMIC STATE TRACKING ---
    float current_temp = (params.do_greedy_prequench ? params.reheat_temp : params.initial_temp);
    int last_defects = INT32_MAX;
    int stagnation_counter = 0;

    // --- MAIN OPTIMIZATION LOOP ---
    for (int step = 1; step <= params.max_steps; step++) {  // <-- Start loop at 1
        
        // Only check AFTER a full interval
        if (step % params.check_interval == 0) {
            CUDA_CHECK(cudaMemcpy(h_grid, d_grid, grid_bytes, cudaMemcpyDeviceToHost));
            int defects = count_grid_defects(h_grid, grid_size, alphabet);

            // Goal condition met: The interior has settled into a valid configuration with tolerance
            if (defects <= params.max_allowed_defects && defects >= params.min_allowed_defects) {
                CUDA_CHECK(cudaFree(d_grid));
                CUDA_CHECK(cudaFree(d_states));
                if (d_locked_mask) CUDA_CHECK(cudaFree(d_locked_mask));
                return  { defects, current_temp, step }; // SUCCESS
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

        // 2. MCMC Heat Bath Phase (Checkerboard updates avoid race conditions)
        heat_bath_kernel_unified<<<numBlocks, threadsPerBlock>>>(d_grid, grid_size, current_temp, d_states, 0, d_locked_mask);
        heat_bath_kernel_unified<<<numBlocks, threadsPerBlock>>>(d_grid, grid_size, current_temp, d_states, 1, d_locked_mask);

        // 3. Geometric Cooling
        current_temp *= params.cooling_rate;
    }

    // Cleanup if the goal was never reached
    CUDA_CHECK(cudaFree(d_grid));
    CUDA_CHECK(cudaFree(d_states));
    if (d_locked_mask) CUDA_CHECK(cudaFree(d_locked_mask));

    return  { -1, 0.0f, params.max_steps }; // FAILED
}

// Returns -1 if the maximum number of steps is reached without success.
// Uses an adiabatic cubic Beta cooling schedule to gently navigate highly frustrated grids (like Rebar).
// params.check_interval, params.cooling_rate, and params.max_steps are all ignored in this method. There
// is no geometric decay and the stagnation reheat logic is also removed.
// @param external_lock_mask Grid tile allowed behavior mask: 0 = Free, 1 = Pinned Outward Colors, 2 = Pinned Tile ID.
GridAnnealResult grid_anneal_gpu_adiabatic(
    int* h_grid, 
    int grid_size,
    ITileSet* tileset, 
    GridAnnealParams params,
    ChaCha20PRNG& rng,
    const std::vector<uint8_t>& external_lock_mask
) {
    std::vector<Tile> alphabet = tileset->get_tiles();
    int num_tiles = tileset->get_size();

    if (num_tiles > QwtssConfig::MAX_TILES) {
        throw std::runtime_error("Alphabet size exceeds QwtssConfig::MAX_TILES allocation.");
    }

    // Init constant memory using proper function to prevent the "Shadow Constant Memory" bug.
    initialize_gpu_constants_from_core_shared(tileset);

    int grid_cells = grid_size * grid_size;
    int grid_bytes = grid_cells * sizeof(int);
    int *d_grid;

    CUDA_CHECK(cudaMalloc((void**)&d_grid, grid_bytes));
    CUDA_CHECK(cudaMemcpy(d_grid, h_grid, grid_bytes, cudaMemcpyHostToDevice));

    uint8_t* d_locked_mask = nullptr;
    if (!external_lock_mask.empty()) {
        CUDA_CHECK(cudaMalloc((void**)&d_locked_mask, external_lock_mask.size() * sizeof(uint8_t)));
        CUDA_CHECK(cudaMemcpy(d_locked_mask, external_lock_mask.data(), external_lock_mask.size() * sizeof(uint8_t), cudaMemcpyHostToDevice));
    } else if (params.do_pinned_defect_doping){
        int num_pins = std::min(params.max_allowed_defects, std::max(10, params.max_allowed_defects / 10));
        // Calculate a statistically safe min_radius for Poisson Disk Sampling
        double min_radius = std::sqrt((grid_cells) / (num_pins * M_PI)) * 0.8;

        std::vector<uint8_t> pinned_mask = generate_quenched_disorder_mask(grid_size, num_pins, min_radius, rng);

        // Allocate and copy to GPU
        CUDA_CHECK(cudaMalloc((void**)&d_locked_mask, pinned_mask.size() * sizeof(uint8_t)));
        CUDA_CHECK(cudaMemcpy(d_locked_mask, pinned_mask.data(), pinned_mask.size() * sizeof(uint8_t), cudaMemcpyHostToDevice));
    }

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((grid_size + 15) / 16, (grid_size + 15) / 16);

    // Setup Philox cuRAND States with Host Entropy Injection
    std::vector<unsigned long long> h_seeds(grid_cells);
    std::uniform_int_distribution<unsigned long long> dist; 
    for (int i = 0; i < grid_cells; ++i) {
        h_seeds[i] = dist(rng);
    }

    unsigned long long* d_seeds;
    CUDA_CHECK(cudaMalloc(&d_seeds, grid_cells * sizeof(unsigned long long)));
    CUDA_CHECK(cudaMemcpy(d_seeds, h_seeds.data(), grid_cells * sizeof(unsigned long long), cudaMemcpyHostToDevice));

    curandStatePhilox4_32_10_t* d_states;
    CUDA_CHECK(cudaMalloc(&d_states, grid_cells * sizeof(curandStatePhilox4_32_10_t)));

    init_curand_states<<<(grid_cells + 255) / 256, 256>>>(d_states, d_seeds, grid_size);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaFree(d_seeds));

    if (params.do_greedy_prequench){
        float quench_temp = 0.05f; 
        for (int step = 0; step < 50; step++) {
            heat_bath_kernel_unified<<<numBlocks, threadsPerBlock>>>(d_grid, grid_size, quench_temp, d_states, 0, d_locked_mask);
            heat_bath_kernel_unified<<<numBlocks, threadsPerBlock>>>(d_grid, grid_size, quench_temp, d_states, 1, d_locked_mask);
        }
    }

    // --- ADIABATIC THERMODYNAMIC SCHEDULE ---
    const int num_beta_steps = 5000; 
    const double beta_start = 0.20; // No point at starting at even lower beta; 0.20 is already max boiling
    const double fixed_beta_end = 11.0;

    int total_sweeps_executed = 0;

    // TMP:
    //std::cout << "Step, Temp, Defects, Angle, Hash_Bucket" << std::endl;

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
            heat_bath_kernel_unified<<<numBlocks, threadsPerBlock>>>(d_grid, grid_size, step_temp, d_states, 0, d_locked_mask);
            heat_bath_kernel_unified<<<numBlocks, threadsPerBlock>>>(d_grid, grid_size, step_temp, d_states, 1, d_locked_mask);
        }
        total_sweeps_executed += sweeps;

        // Check if the grid has organically entered the target defect band
        CUDA_CHECK(cudaMemcpy(h_grid, d_grid, grid_bytes, cudaMemcpyDeviceToHost));
        int defects = count_grid_defects(h_grid, grid_size, alphabet);

        // TMP:
        /*
        GlobalHashMetrics global_metrics = compute_global_hashes(h_grid, grid_size);
        double raw_angle_deg = std::atan2(global_metrics.quad_C, global_metrics.quad_A) * (180.0 / M_PI);
        uint8_t similarity_hash = compute_uniform_similarity_hash(global_metrics);
        std::cout << step << ", " << std::fixed << std::setprecision(2) << step_temp
                  << ", " << defects << ", " << std::fixed << std::setprecision(1) << std::setw(6) << raw_angle_deg
                  << ", " << (int)similarity_hash << std::endl;
        */

        if (defects <= params.max_allowed_defects) {
            CUDA_CHECK(cudaFree(d_grid));
            CUDA_CHECK(cudaFree(d_states));
            if (d_locked_mask) CUDA_CHECK(cudaFree(d_locked_mask));

            if (defects >= params.min_allowed_defects) {
                return { defects, step_temp, total_sweeps_executed }; // SUCCESS
            } else {
                // Overshot the target.
                //printf("overshot, step temp: %f\n", step_temp);
                // Adiabatic mode strictly prevents returns to higher E; fast-fail to preserve compute cycles
                return { -1, 0.0f, total_sweeps_executed }; // FAILED
            }
        }
    }

    // Cleanup if the schedule completed without hitting the target band
    CUDA_CHECK(cudaFree(d_grid));
    CUDA_CHECK(cudaFree(d_states));
    if (d_locked_mask) CUDA_CHECK(cudaFree(d_locked_mask));

    return { -1, 0.0f, total_sweeps_executed }; // FAILED
}

bool detect_eligible_nvidia_gpu(bool do_logging) {
    int device_count = 0;
    
    // Catch systems without NVIDIA drivers gracefully without crashing
    cudaError_t error = cudaGetDeviceCount(&device_count);
    if (error != cudaSuccess || device_count == 0) {
        return false; 
    }

    // Defaulting to the primary device (Device 0). On 99% of single-GPU systems, this is the discrete graphics card.
    int selected_device = 0;
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, selected_device);

    double vram_gb = static_cast<double>(prop.totalGlobalMem) / (1024.0 * 1024.0 * 1024.0);

    if (do_logging) {
        std::cout << "Auto-detected NVIDIA GPU: " << prop.name 
                  << " (Compute " << prop.major << "." << prop.minor 
                  << ", " << std::fixed << std::setprecision(1) << vram_gb << " GB VRAM)\n";
    }
    
    // Lock the runtime context to this device
    cudaSetDevice(selected_device); 
    return true;
}

PkDerivedFields get_pk_derived_fields(const std::string& username, uint64_t identity_nonce, uint8_t version,
    int grid_size, float keep_boundary_percentage, bool do_logging){
    if (username.empty()) throw std::invalid_argument("username must be specified");
    if (grid_size < 32) throw std::invalid_argument("grid_size is expected to be 64 for QWTSS");
    if (keep_boundary_percentage < 0.0f || keep_boundary_percentage > 1.0f) throw std::invalid_argument("keep_boundary_percentage must be 0.0f to 1.0f");

    // Generate the 64-character Hex string using Username + Nonce + Version
    // Prevent any canonicalization attacks with strict delimiters
    std::string hash_input = username + "|" + std::to_string(identity_nonce) + "|" + std::to_string(version);
    std::string user_hash_hex = SHA256::hash_string(hash_input);
    // Sanity check
    if (user_hash_hex.length() != 64) {
            throw std::runtime_error("Sanity check failed: (user_hash_hex.length() != 64)");
    }

    if (do_logging){
        std::cout << "Username:      " << username << std::endl;
        std::cout << "Nonce:         " << identity_nonce << std::endl;
        // Output: e.g., "b1f8...3c9a"
        std::cout << "Identity Hash: " << user_hash_hex << std::endl;
    }

    // Extract the 64-char hex string into an array of eight 32-bit integers
    std::array<uint32_t, 8> chacha_key;
    for (int i = 0; i < 8; i++) {
        // Extract 8 hex chars (32 bits) at a time
        std::string chunk = user_hash_hex.substr(i * 8, 8);
        // Using stringstream with std::hex to extract 8-character chunks without
        // any endianness or casting undefined behavior.
        std::stringstream ss;
        ss << std::hex << chunk;
        ss >> chacha_key[i];
    }

    // Initialize a strictly deterministic CSPRNG using the User Hash
    ChaCha20PRNG user_hash_rng(chacha_key);

    // Instantiate Labbé Oracle generator and build QWTSS-style valid public key
    LabbeJR11Oracle oracle;
    LabbeOraclePublicKey oracle_pk = oracle.generate_spliced_public_key(grid_size, QwtssReference::boundary_num_splice_segments, user_hash_rng);

    // User's hash determines the determinisitic partial boundary mask (which grid edge constraints drop out)
    std::vector<uint8_t> boundary_mask = generate_partial_boundary_mask(grid_size, keep_boundary_percentage, user_hash_rng);

    return { oracle_pk, boundary_mask };
}

/**
 * @brief Generates a valid QWTSS private key using the Jeandel-Rao 11-tile set and QWTSS reference constraint values.
 * @param username The user identity (e.g., username).
 * @param identity_nonce A public salt/nonce for key rotation.
 * @param target_cpu_threads Number of CPU threads to use (CPU mode only). Default: -1 (AUTO: 70% of cores).
 * @return A valid QWTSS private key that meets all protocol constraints, and the associated
 * boundary mask. Otherwise returns empty, if key gen failed.
 */
QwtssPrivateKey build_qwtss_private_key(std::string username, uint64_t identity_nonce, AnnealMode anneal_mode,
    AnnealDevice anneal_device, int target_cpu_threads, bool do_logging)
{
    return build_qwtss_private_key(username, identity_nonce,
        QwtssReference::version,
        QwtssReference::grid_size,
        QwtssReference::total_defects_lower_bound,
        QwtssReference::total_defects_upper_bound,
        QwtssReference::keep_boundary_percentage,
        QwtssReference::do_greedy_prequench,
        anneal_mode, anneal_device, target_cpu_threads, do_logging
    );
}

/**
 * @brief Generates a valid QWTSS private key using the Jeandel-Rao 11-tile set and QWTSS reference constraint values.
 * @param username The user identity (e.g., username).
 * @param identity_nonce A public salt/nonce for key rotation.
 * @param target_cpu_threads Number of CPU threads to use (CPU mode only). Default: -1 (AUTO: 70% of cores).
 * @return A valid QWTSS private key that meets all protocol constraints, and the associated
 * boundary mask. Otherwise returns empty, if key gen failed.
 */
QwtssPrivateKey build_qwtss_private_key(std::string username, uint64_t identity_nonce, int min_defect_count,
    int max_defect_count, AnnealMode anneal_mode, AnnealDevice anneal_device, int target_cpu_threads, bool do_logging)
{
    return build_qwtss_private_key(username, identity_nonce,
        QwtssReference::version,
        QwtssReference::grid_size,
        min_defect_count,
        max_defect_count,
        QwtssReference::keep_boundary_percentage,
        QwtssReference::do_greedy_prequench,
        anneal_mode, anneal_device, target_cpu_threads, do_logging
    );
}

/**
 * @brief Generates a valid QWTSS private key using the Jeandel-Rao 11-tile set.
 * @param username The user identity (e.g., username).
 * @param identity_nonce A public salt/nonce for key rotation.
 * @param keep_boundary_percentage 0.0f to 1.0f percentage of the border constraints to keep (border retention).
 * @param target_cpu_threads Number of CPU threads to use (CPU mode only). Default: -1 (AUTO: 70% of cores).
 * @return A valid QWTSS private key that meets all protocol constraints, and the associated
 * boundary mask. Otherwise returns empty, if key gen failed.
 */
QwtssPrivateKey build_qwtss_private_key(std::string username, uint64_t identity_nonce, uint8_t version,
    int grid_size, int min_defect_count, int max_defect_count, float keep_boundary_percentage,
    bool do_greedy_prequench, AnnealMode anneal_mode, AnnealDevice anneal_device, int target_cpu_threads, bool do_logging)
{
    if (username.empty()) throw std::invalid_argument("username must be specified");
    if (grid_size < 32) throw std::invalid_argument("grid_size is expected to be 64 for QWTSS");
    if (min_defect_count <= 0) throw std::invalid_argument("min_defect_count must be 1+");
    if (max_defect_count < min_defect_count) throw std::invalid_argument("max_defect_count cannot be less than min_defect_count");
    if (keep_boundary_percentage < 0.0f || keep_boundary_percentage > 1.0f) throw std::invalid_argument("keep_boundary_percentage must be 0.0f to 1.0f");
    if (anneal_device == AnnealDevice::AUTO){
        bool gpu_exists = detect_eligible_nvidia_gpu();
        anneal_device = (gpu_exists ? AnnealDevice::GPU : AnnealDevice::CPU);
    }
    if (anneal_device != AnnealDevice::GPU && anneal_device != AnnealDevice::CPU) throw std::invalid_argument("anneal_device must be CPU or GPU");
    if (target_cpu_threads != -1){
        if (target_cpu_threads < 1) throw std::invalid_argument("target_cpu_threads value must be 1+");
        if (anneal_device != AnnealDevice::CPU) throw std::invalid_argument("target_cpu_threads can only be defined with the CPU device target");
    }

    auto start = std::chrono::high_resolution_clock::now();

    JeandelRaoTileSet jr_tileset;
    int num_tiles = jr_tileset.get_size();
    std::vector<Tile> alphabet = jr_tileset.get_tiles();

    PkDerivedFields pk_derived = get_pk_derived_fields(username, identity_nonce, version, grid_size, keep_boundary_percentage, do_logging);

    // New rng to decouple any local private key seeds from the username hash
    ChaCha20PRNG local_rng;

    std::uniform_int_distribution<int> target_defects(min_defect_count, max_defect_count);
    int target_defects_for_edge_prob = target_defects(local_rng);

    // Private key seed grid
    std::vector<int> h_grid(grid_size * grid_size);
    std::vector<int> ground_truth = pk_derived.oracle_pk.pk_mask; // Tile ids
    std::vector<uint8_t> boundary_mask = pk_derived.boundary_mask;
    generate_boundary_conditioned_random_seed_grid(ground_truth, grid_size, h_grid, target_defects_for_edge_prob, alphabet, local_rng, boundary_mask);

    // Las Vegas Restart Strategy:
    // The mathematically optimal strategy to minimize user wait time is to set a hard timeout at roughly 250,000 steps.
    GridAnnealParams params(max_defect_count, min_defect_count, 320000 /* 250000 * 2 */);
    params.do_greedy_prequench = do_greedy_prequench;

    // Perform the annealing process
    GridAnnealResult result;
    if (anneal_mode == ADIABATIC){
        params.do_greedy_prequench = false; // Set this to false to honor the adiabatic process
        if (anneal_device == AnnealDevice::GPU){
            result = grid_anneal_gpu_adiabatic(h_grid.data(), grid_size, &jr_tileset, params, local_rng, boundary_mask);
        } else {
            result = grid_anneal_cpu_adiabatic(h_grid.data(), grid_size, &jr_tileset, params, local_rng, boundary_mask, target_cpu_threads, do_logging);
        }
    } else if (anneal_mode == ORIGINAL) {
        if (anneal_device == AnnealDevice::GPU){
            result = grid_anneal_gpu(h_grid.data(), grid_size, &jr_tileset, params, local_rng, boundary_mask);
        } else {
            result = grid_anneal_cpu(h_grid.data(), grid_size, &jr_tileset, params, local_rng, boundary_mask, target_cpu_threads, do_logging);
        }
    } else {
        // anneal_mode == FAST
        if (anneal_device == AnnealDevice::GPU){
            result = grid_anneal_gpu_optimized(h_grid.data(), grid_size, &jr_tileset, params, local_rng, boundary_mask);
        } else {
            result = grid_anneal_cpu_optimized(h_grid.data(), grid_size, &jr_tileset, params, local_rng, boundary_mask, target_cpu_threads, do_logging);
        }
    }
    int final_defects = result.final_defects;

    if (final_defects != -1 && final_defects <= max_defect_count && final_defects >= min_defect_count) {
        // Final sanity check to ensure no PK boundary constraints were violated
        confirm_mask_constraints_honored(h_grid, boundary_mask, grid_size, ground_truth, alphabet, do_logging);

        // Check all additional QWTSS constraints here

        // Alien tiles metric
        int alien_tiles = calculate_alien_tiles(h_grid, pk_derived.oracle_pk.plane_A, pk_derived.oracle_pk.plane_B, grid_size);
        //std::cout << "Private key alien tiles: " << alien_tiles << " / 4096 (higher is better)" << std::endl;
        //export_alien_tiles_to_ppm(h_grid.data(), grid_size, oracle_pk.plane_A, oracle_pk.plane_B, "alien tiles.ppm");
        if (alien_tiles < QwtssReference::alien_tiles_lower_bound){
            if (do_logging) std::cout << "[FAILED] Key gen could not reach target alien tiles count.\n";

            QwtssPrivateKey qpk;
            qpk.work_steps = result.steps;
            return qpk;
        }

        // Calculate core32 metric
        int core_32 = calculate_core_defects(h_grid, grid_size, alphabet, 32);
        if (core_32 < QwtssReference::core32_lower_bound){
            if (do_logging) std::cout << "[FAILED] Key gen could not reach target core32 count.\n";

            QwtssPrivateKey qpk;
            qpk.work_steps = result.steps;
            return qpk;
        }

        // Calculate line_max metric
        DefectDistributionStats lines_stats = calculate_defect_spatial_distribution_lines(h_grid, grid_size, alphabet);
        int max_line_defects_seam = lines_stats.max_defects;
        if (max_line_defects_seam > QwtssReference::line_max_upper_bound){
            if (do_logging) std::cout << "[FAILED] Key gen could not reach target line_max value.\n";

            QwtssPrivateKey qpk;
            qpk.work_steps = result.steps;
            return qpk;
        }

        auto end = std::chrono::high_resolution_clock::now();
        double elapsed_sec = std::chrono::duration<double>(end - start).count();

        if (do_logging) std::cout << "[SUCCESS] Valid QWTSS Key with " << final_defects << " defects generated in " << result.steps << " steps and "
            << std::fixed << std::setprecision(2) << elapsed_sec << " seconds.\n";

        QwtssPrivateKey qpk;
        qpk.private_key = h_grid;
        qpk.boundary_mask = boundary_mask;
        qpk.plane_A = pk_derived.oracle_pk.plane_A;
        qpk.plane_B = pk_derived.oracle_pk.plane_B;
        qpk.defect_count = final_defects;
        qpk.alien_tile_count = alien_tiles;
        qpk.core32_defect_tile_count = core_32;
        qpk.line_max_defect_count = max_line_defects_seam;
        qpk.work_steps = result.steps;
        return qpk;
    } else {
        if (do_logging) std::cout << "[FAILED] Key gen could not reach target defects range.\n";

        QwtssPrivateKey qpk;
        qpk.work_steps = result.steps;
        return qpk;
    }
}

std::array<std::string, 2> get_serialized_grid_fingerprint(const QwtssPrivateKey& private_key){
    if (private_key.private_key.empty()) throw std::invalid_argument("private_key cannot be empty");

    JeandelRaoTileSet jr_tileset;
    std::vector<Tile> alphabet = jr_tileset.get_tiles();

    // Wrap the single identity grid in a vector
    std::vector<QwtssPrivateKey> private_keys = { private_key };
    bool throw_preemptively = true;
    ExecutionTrace trace = build_execution_trace(private_keys, QwtssReference::grid_size, alphabet, throw_preemptively);

    // Extract the 504-bit fingerprint from the flattened S-Box math *after* the blanking rounds
    std::array<FieldElement256, 2> grid_fingerprint = {
        trace.expected_hash_0[trace.valid_steps + 63], 
        trace.expected_hash_1[trace.valid_steps + 63]
    };

    return serialize_fingerprint(grid_fingerprint);
}

/**
 * @brief Runs the full QWTSS reference pipeline for standard signature (non-PoW).
 * @param username The user identity (e.g., username).
 * @param identity_nonce A public salt/nonce for key rotation.
 * @return True on success; otherwise, False on failure.
  */
bool run_qwtss_full_pipeline(std::string username, uint64_t identity_nonce) {

    int grid_size = QwtssReference::grid_size;

    // Instantiate the Jeandel-Rao Tile Set
    JeandelRaoTileSet jr_tileset;
    std::vector<Tile> alphabet = jr_tileset.get_tiles();

    std::cout << "\nGenerating private key ..." << std::endl;
    QwtssPrivateKey private_key{};
    // Try up to 5 times
    auto start = std::chrono::high_resolution_clock::now();
    int line_stats_max_defects = -1;
    int t = 0;
    for (t = 0; t < 5; t++){
        bool do_logging = (t == 0);
        private_key = build_qwtss_private_key(username, identity_nonce, FAST, AUTO, -1, do_logging);
        //private_key = build_qwtss_private_key(username, nonce, 64, 171, 172, 0.7f, true, true); // Failing key

        if (private_key.private_key.empty()){
            std::cout << "\nTry " << (t+1) << " failed: private key convergence failed" << std::endl;
            continue;
        }

        // Check the max hole size
        DefectStats topo_stats = analyze_defect_topology(private_key.private_key.data(), grid_size, alphabet);
        if (topo_stats.max_hole_size > 20){
            std::cout << "\nTry " << (t+1) << " failed: private key defect distribution out of bounds (this happens on occasion)" << std::endl;
            continue;
        }

        // Check 1D defect lines (all rows and columns) for quality control
        DefectDistributionStats line_stats = calculate_defect_spatial_distribution_lines(
            private_key.private_key, grid_size, alphabet);
        if (line_stats.max_defects > 99){
            std::cout << "\nTry " << (t+1) << " failed: private key defect distribution out of bounds (this happens rarely on occasion)" << std::endl;
            continue;
        } else {
            // Apparent success
            line_stats_max_defects = line_stats.max_defects;
            break;
        }
    }
    auto end = std::chrono::high_resolution_clock::now();
    double elapsed_sec = std::chrono::duration<double>(end - start).count();

    if (private_key.private_key.empty()){
        std::cout << "Error: All attempts at private key convergence failed (" << elapsed_sec << " seconds total)" << std::endl;
        return false;
    }

    /*
    // Double-confirm a valid tiling defect count
    int defect_count = count_grid_defects(private_key.private_key.data(), grid_size, alphabet);
    if (QwtssReference::is_valid_defect_count(defect_count)) {
        std::cout << "\nValid tiling confirmed with " << defect_count << " defects; private key is ready after "
            << (t+1) << " attempts (" << std::fixed << std::setprecision(2) << elapsed_sec << " seconds total)" << std::endl;
    } else {
        std::cout << "Error: private key creation failed; tiling had invalid defect count (" << elapsed_sec << " seconds total)" << std::endl;
        return false;
    }
    */

    // Topological scan
    DefectStats stats = analyze_defect_topology(private_key.private_key.data(), grid_size, alphabet);
    std::cout << "\n--- Topological Defect Scan ---" << std::endl;
    std::cout << "Total Defective Tiles: " << stats.total_defective_tiles << std::endl;
    std::cout << "Isolated Defect Clusters (Holes): " << stats.num_holes << std::endl;
    std::cout << "Average Hole Size: " << std::fixed << std::setprecision(2) << stats.avg_hole_size << " tiles" << std::endl;
    std::cout << "Max Hole Size: " << stats.max_hole_size << " tiles" << std::endl;
    std::cout << "Line Stats (Max Defects): " << line_stats_max_defects << std::endl;
    
    // Safety check against very rare topological failures in annealed key
    if (stats.max_hole_size > 20 || line_stats_max_defects > 12 || line_stats_max_defects <= 0) {
        std::cout << "Error: private key failed defect distribution quality control" << std::endl;
        return false;
    }

    std::cout << "\nPrivate key generation complete: key annealed successfully.\n" << std::endl;
    // Debugging
    std::cout << "\tAlien Tiles: " << private_key.alien_tile_count << std::endl;
    std::cout << "\tCore32:      " << private_key.core32_defect_tile_count << std::endl;
    std::cout << "\tLine_Max:    " << private_key.line_max_defect_count << std::endl;

    // OS Telemetry for Peak RAM Usage
    struct rusage pre_stark_usage;
    getrusage(RUSAGE_SELF, &pre_stark_usage);

    // Linux returns ru_maxrss in Kilobytes. 
    double pre_peak_ram_mb = (double)pre_stark_usage.ru_maxrss / 1024.0;

    // STARK integration logic
    std::cout << "Building Algebraic Execution Trace..." << std::endl;
    // Wrap the single identity grid in a vector
    std::vector<QwtssPrivateKey> private_keys = { private_key };
    bool throw_preemptively = true;
    start = std::chrono::high_resolution_clock::now();
    ExecutionTrace trace = build_execution_trace(private_keys, grid_size, alphabet, throw_preemptively);

    // Extract the Pinned Perimeter (Public Key Constraints) directly from the private key
    std::cout << "Extracting QWTSS Public Inputs..." << std::endl;
    // Extract the 504-bit fingerprint from the flattened S-Box math *after* the blanking rounds
    std::array<FieldElement256, 2> grid_fingerprint = {
        trace.expected_hash_0[trace.valid_steps + 63], 
        trace.expected_hash_1[trace.valid_steps + 63]
    };
    QwtssPublicInputs public_inputs(username, identity_nonce, QwtssReference::version, private_key,
        grid_size, grid_fingerprint, alphabet);

    /*
    // Validate the trace mathematically
    std::cout << "Validating Trace... " << std::flush;
    QwtssAir validator(public_inputs, trace.trace_length, alphabet);
    bool validated = validator.validate_trace(trace);
    if (validated) std::cout << "OK" << std::endl;
    else throw std::runtime_error("The trace failed to be validated against the AIR");
    */

    // The message we are signing
    std::vector<uint8_t> message = {'H', 'E', 'L', 'L', 'O'};

    std::cout << "Generating QTWSS Signature..." << std::endl;
    std::vector<std::byte> final_sig = generate_stark_signature(trace, public_inputs, message, alphabet);
    end = std::chrono::high_resolution_clock::now();
    elapsed_sec = std::chrono::duration<double>(end - start).count();

    // OS Telemetry for Peak RAM Usage
    struct rusage post_stark_usage;
    getrusage(RUSAGE_SELF, &post_stark_usage);
    
    // Linux returns ru_maxrss in Kilobytes. 
    double post_peak_ram_mb = (double)post_stark_usage.ru_maxrss / 1024.0; 

    size_t final_sig_size = final_sig.size();
    float final_sig_size_kb = (float)final_sig_size / 1024.0f;
    std::cout << "Signature generated in " << elapsed_sec << " seconds. Size: " << final_sig_size << " bytes (" << std::fixed
                << std::setprecision(1) << final_sig_size_kb << " KB)." << std::endl;

    std::cout << "Pre-STARK Peak RAM Usage: " << std::fixed << std::setprecision(2) 
              << pre_peak_ram_mb << " MB" << std::endl;
    std::cout << "Post-STARK Peak RAM Usage: " << std::fixed << std::setprecision(2) 
              << post_peak_ram_mb << " MB (STARK signature generation required " << (post_peak_ram_mb - pre_peak_ram_mb) << " MB)" << std::endl;

    // --- VERIFIER SIMULATION ---
    std::cout << "\n\nSimulating Verifier..." << std::endl;

    bool is_valid = verify_stark_signature(final_sig, public_inputs, message, alphabet);
    if (!is_valid) {
        std::cout << "Error: Generated signature failed verification!" << std::endl;
        return false;
    } else {
        std::cout << "\nSuccess: Generated signature passed verification!" << std::endl;
        std::cout << "Pipeline Complete." << std::endl;
        return true;
    }
}

// The Systematic Optimizer
void run_hyperparameter_optimization(int grid_size) {
    JeandelRaoTileSet jr_tileset;

    ChaCha20PRNG gen;

    // Define parameter ranges
    std::uniform_real_distribution<float> dist_cooling(0.99985f, 0.99999f);
    std::uniform_real_distribution<float> dist_reheat(0.25f, 0.75f);
    std::uniform_int_distribution<int> dist_patience(1, 4);
    std::uniform_int_distribution<int> dist_interval(2000, 4000);

    std::ofstream log_file("optimizer_results.csv");
    log_file << "CoolingRate,ReheatTemp,Patience,CheckInterval,StepsTaken,ElapsedSecs\n";

    std::cout << "Starting Random Search Optimizer..." << std::endl;

    GridAnnealParams params;

    // Run 1000 random combinations
    for (int run = 0; run < 1000; run++) {
        params.cooling_rate = dist_cooling(gen);
        params.reheat_temp = dist_reheat(gen);
        params.stagnation_patience = dist_patience(gen);
        params.check_interval = dist_interval(gen);
        
        // Randomize initial grid for a fair test
        int h_grid[grid_size * grid_size];
        for (int i = 0; i < grid_size * grid_size; ++i) {
            h_grid[i] = gen() % jr_tileset.get_size();
        }

        std::cout << "Test " << run << " | Cool: " << std::fixed << std::setprecision(5) << params.cooling_rate
                    << " | Reheat: " << std::setprecision(3) << params.reheat_temp
                    << " | Patience: " << params.stagnation_patience
                    << " | Check Interval: " << params.check_interval << " | " << std::flush;

        auto start = std::chrono::high_resolution_clock::now();
        auto result = grid_anneal_gpu(h_grid, grid_size, &jr_tileset, params, gen);
        auto end = std::chrono::high_resolution_clock::now();
        double elapsed_sec = std::chrono::duration<double>(end - start).count();

        if (result.final_defects != -1) {
            std::cout << "SUCCESS in " <<  result.steps << " steps and " << std::fixed << std::setprecision(2) << elapsed_sec << " secs" << std::endl;
        } else {
            std::cout << "FAILED" << std::endl;
        }

        log_file << std::fixed << std::setprecision(5) << params.cooling_rate << "," << params.reheat_temp << "," << params.stagnation_patience
                << "," << params.check_interval << "," << result.steps << "," << std::setprecision(2) << elapsed_sec << "\n";
        log_file.flush();
    }
}

void build_private_key_db(ITileSet* tileset, int min_defect_count, int max_defect_count, bool do_greedy_prequench,
    int grid_size, PrivateKeyGenMode key_gen_mode, float keep_boundary_percentage, std::string filename) {
    if (keep_boundary_percentage < 0.0f || keep_boundary_percentage > 1.0f) {
        throw std::invalid_argument("keep_boundary_percentage must be 0.0f to 1.0f");
    }

    // 1. Initialize and load existing history
    PrivateKeyDatabase key_db;
    std::ifstream in(filename, std::ios::binary);
    if (!in) {
        std::cout << "Private key database file not found. Starting fresh.\n";
    } else {
        key_db.load_from_disk(filename);
        std::cout << "Loaded existing private key database: " << filename << ", Keys: " << key_db.get_total_key_count() << "\n";
        //key_db.print_summary();
        //return;
    }

    int num_tiles = tileset->get_size();
    std::vector<Tile> alphabet = tileset->get_tiles();

    ChaCha20PRNG gen;
    std::uniform_int_distribution<> dis_grid_start(0, num_tiles - 1);
    std::uniform_int_distribution<> dis_defect_count(min_defect_count, max_defect_count);

    // 3. Define Simulated Annealing parameters

    // Las Vegas Restart Strategy:
    // The mathematically optimal strategy to minimize user wait time is to set a hard timeout at roughly 3.0 seconds (or 250,000 steps).
    int max_steps = 250000;
    GridAnnealParams params(max_defect_count, min_defect_count, max_steps);
    params.do_greedy_prequench = do_greedy_prequench;

    int max_vegas_restarts = 150;

    int defect_count = INT32_MAX;
    int keys_added = 0;

    // Initialize the host grid (64x64)
    std::vector<int> h_grid(grid_size * grid_size);
    std::vector<uint8_t> h_mask(grid_size * grid_size, 0);

    // Open the file in append mode
    std::ofstream csv("pk_db_creation_stats.csv", std::ios::app);
    // Only write the header if the file is completely empty (newly created)
    if (csv.tellp() == 0) {
        csv << "Actual_Defects,Alien_Tile_Count,Core32_Defect_Tile_Count,Line_Max_Defect_Count\n";
    }

    std::string exception_key_db_filename = "qwtss_exception_key.db";
    PrivateKeyDatabase exception_key_db;
    std::ifstream in2(exception_key_db_filename, std::ios::binary);
    if (!in2) {
        std::cout << "Exception key private key database file not found. Starting fresh.\n";
    } else {
        exception_key_db.load_from_disk(exception_key_db_filename, true);
        std::cout << "Loaded existing exception key private key database: " << exception_key_db_filename
                  << ", Keys: " << exception_key_db.get_total_key_count() << "\n";
    }

    auto start_global = std::chrono::steady_clock::now();

    for (int k = 0; k < 2000; k++){
        bool valid_tiling_found = false;
        auto start = std::chrono::high_resolution_clock::now();

        // Up to N separate Las Vegas Restart attempts
        for (int t = 0; t < max_vegas_restarts; t++){
            // This must vary run to run to properly exercise the different early abort dynamics
            int use_max_defect_count = dis_defect_count(gen);

            if (key_gen_mode == UNCONSTRAINED){
                // Re-randomize the grid from scratch for every attempt
                for (int i = 0; i < grid_size * grid_size; ++i) {
                    h_grid[i] = dis_grid_start(gen);
                }

                params.assign_optimal_check_interval(use_max_defect_count);

                std::cout << "\nGenerating private key with defect count in range [" << min_defect_count << ", " << max_defect_count
                        << "], current target: " << use_max_defect_count << std::endl;

                // Call the GPU generation function (no boundary mask argument in unconstrained mode)
                grid_anneal_gpu(
                    h_grid.data(),      // Pointer to the initialized host grid
                    grid_size,
                    tileset,            // The selected tile set implementation
                    params,
                    gen
                );

                h_mask = generate_partial_boundary_mask(grid_size, keep_boundary_percentage, gen);
            } else if (key_gen_mode == QWTSS_STANDARD){
                uint64_t nonce = 0;
                QwtssPrivateKey qwtss_private_key_result = build_qwtss_private_key(generate_random_username(gen), nonce, 0,
                    grid_size, min_defect_count, use_max_defect_count, keep_boundary_percentage, do_greedy_prequench);

                if (qwtss_private_key_result.private_key.empty() || qwtss_private_key_result.boundary_mask.empty()) continue; // Key gen failed
                else {
                    h_grid = qwtss_private_key_result.private_key;
                    h_mask = qwtss_private_key_result.boundary_mask;

                    csv << qwtss_private_key_result.defect_count << "," << qwtss_private_key_result.alien_tile_count
                        << "," << qwtss_private_key_result.core32_defect_tile_count << ","
                        << qwtss_private_key_result.line_max_defect_count << "\n";
                    csv.flush();

                    // Export defect edges .ppm file?
                    // if (k < 5) {
                    //     std::string defect_ppm = "[" + std::to_string(k) + "] " + std::to_string(qwtss_private_key_result.defect_count) + " defect edges.ppm";
                    //     export_defect_edges_to_ppm(qwtss_private_key_result.private_key.data(), grid_size, tileset, defect_ppm);
                    // }

                    bool add_exception_key = false;
                    std::string exception_name = "";
                    if (qwtss_private_key_result.alien_tile_count < QwtssReference::alien_tiles_lower_bound){
                        exception_name = "alien tile count too low";
                        // Skipping these because they are most common
                        //add_exception_key = true;
                    }
                    else if (qwtss_private_key_result.core32_defect_tile_count < QwtssReference::core32_lower_bound){
                        exception_name = "core32 count too low";
                        add_exception_key = true;
                    }
                    else if (qwtss_private_key_result.line_max_defect_count > QwtssReference::line_max_upper_bound){
                        exception_name = "Line_Max count too high";
                        add_exception_key = true;
                    }

                    if (add_exception_key){
                        std::cout << "Key Exception Detected: " << exception_name << std::endl;

                        PrivateKey pk;
                        pk.defect_count = qwtss_private_key_result.defect_count;
                        pk.grid_size = grid_size;
                        pk.grid_data = qwtss_private_key_result.private_key;
                        pk.boundary_mask = qwtss_private_key_result.boundary_mask;
                        // Extended data
                        pk.plane_A = qwtss_private_key_result.plane_A;
                        pk.plane_B = qwtss_private_key_result.plane_B;
                        pk.name = exception_name;
                        exception_key_db.add_key(pk);

                        exception_key_db.save_to_disk(exception_key_db_filename, true);
                        std::cout << "Exception key saved. Press Enter to continue..." << std::endl;
                        std::cin.get();
                    }
                }
            } else throw std::invalid_argument("Unexpected 'key_gen_mode' value");

            // Confirm a valid tiling
            defect_count = count_grid_defects(h_grid.data(), grid_size, alphabet);
            if (defect_count >= min_defect_count && defect_count <= max_defect_count) {
                auto end = std::chrono::high_resolution_clock::now();
                double elapsed_sec = std::chrono::duration<double>(end - start).count();
                std::cout << "Valid tiling confirmed with " << defect_count << " defects; private key is ready after "
                          << (t+1) << " attempts (" << std::fixed << std::setprecision(2) << elapsed_sec << " seconds total)" << std::endl;
                valid_tiling_found = true;
                break;
            } else {
                std::cout << "Try " << (t+1) << " failed: tiling was invalid (had " << defect_count << " defects)" << std::endl;
            }
        }

        if (!valid_tiling_found){
            std::cout << "Error: No valid tiling found with defects in specified range; continuing..." << std::endl;
            continue;
        }

        key_db.add_key(defect_count, grid_size, h_grid, h_mask);
        keys_added++;
        // Optionally flush to disk immediately so it is safe
        key_db.save_to_disk(filename);

        auto end_global = std::chrono::steady_clock::now();
        std::chrono::duration<double, std::milli> elapsed = end_global - start_global;
        double time_per_key_ms = elapsed.count() / (double)keys_added;

        std::cout << "[+] Harvested new key with " << defect_count << " defects! " << keys_added << " keys added so far. "
                  << std::fixed << std::setprecision(2) << time_per_key_ms << " ms/key average (non-hoisted generation)\n" << std::endl;
    }
    csv.close();
}

void run_spatial_distribution_analysis(int min_defect_count, int max_defect_count, int grid_size) {
    //std::string filename = "../external/qwtss_private_keys (225) 70% boundary - 140 to 150 defects.db";
    std::string filename = "../external/qwtss_private_keys (187) 70% boundary - 100 to 110 defects.db";

    JeandelRaoTileSet tileset;
    std::vector<Tile> alphabet = tileset.get_tiles();

    PrivateKeyDatabase key_db;
    std::ifstream in(filename, std::ios::binary);
    if (!in) {
        std::cout << "Private key database file not found. Exiting...\n";
        return;
    }
    in.close();
    key_db.load_from_disk(filename);
    std::cout << "Loaded private key database. Total Keys: " << key_db.get_total_key_count() << "\n";
    //key_db.print_summary();
    //return;

    // Open CSV file and write headers
    std::ofstream csv_metrics("spatial_distribution_metrics.csv");
    csv_metrics << "Actual_Defects,"
                << "Block8_Min,Block8_Max,Block8_Avg,Block8_StdDev,"
                << "Sliding3_Min,Sliding3_Max,Sliding3_Avg,Sliding3_StdDev,"
                << "Line_Min,Line_Max,Line_Avg,Line_StdDev,"
                << "Frame4,Frame6,Frame8,"
                << "Core32,Core48\n";

    int keys_evaluated = 0;
    std::vector<int> all_line_max_defects;

    for (const PrivateKey& key : key_db) {
        if (key.defect_count < min_defect_count || key.defect_count > max_defect_count) continue;
        if (key.grid_size != grid_size) throw std::runtime_error("(key.grid_size != grid_size)");

        std::cout << "\nEvaluating Spatial Distribution for private key with " << key.defect_count
                  << " defects" << std::endl;

        // 1. Static 8x8 Sub-grid Blocks
        int sub_grid_size = 8;
        DefectDistributionStats block_stats = calculate_defect_spatial_distribution_subgrids(
            key.grid_data, grid_size, alphabet, sub_grid_size);

        // 2. Local Sparsity (3x3 Sliding Window)
        int sliding_window_size = 3;
        DefectDistributionStats sliding_stats = calculate_defect_spatial_distribution_sliding(
            key.grid_data, grid_size, alphabet, sliding_window_size);

        // 3. 1D Fault Lines / Shearing Defects (All Rows and Columns)
        DefectDistributionStats line_stats = calculate_defect_spatial_distribution_lines(
            key.grid_data, grid_size, alphabet);
        all_line_max_defects.push_back(line_stats.max_defects);

        int frame4 = calculate_defect_spatial_distribution_frame(key.grid_data, grid_size, alphabet, 4);
        int frame6 = calculate_defect_spatial_distribution_frame(key.grid_data, grid_size, alphabet, 6);
        int frame8 = calculate_defect_spatial_distribution_frame(key.grid_data, grid_size, alphabet, 8);

        int core_32 = calculate_core_defects(key.grid_data, grid_size, alphabet, 32);
        int core_48 = calculate_core_defects(key.grid_data, grid_size, alphabet, 48);

        keys_evaluated++;

        // Write to Main Metrics CSV
        csv_metrics << key.defect_count << ","
                    << block_stats.min_defects << "," << block_stats.max_defects << "," 
                    << block_stats.avg_defects << "," << block_stats.std_dev_defects << ","
                    << sliding_stats.min_defects << "," << sliding_stats.max_defects << "," 
                    << sliding_stats.avg_defects << "," << sliding_stats.std_dev_defects << ","
                    << line_stats.min_defects << "," << line_stats.max_defects << "," 
                    << line_stats.avg_defects << "," << line_stats.std_dev_defects << ","
                    << frame4 << "," << frame6 << "," << frame8 << ","
                    << core_32 << "," << core_48 << "\n";

        csv_metrics.flush();

        // Console Output for immediate feedback
        std::cout << "   [8x8 Blocks] Max: " << block_stats.max_defects 
                  << " | Avg: " << block_stats.avg_defects << "\n";
        std::cout << "   [3x3 Window] Max: " << sliding_stats.max_defects 
                  << " | Avg: " << sliding_stats.avg_defects << "\n";
        std::cout << "   [Rows/Cols ] Max: " << line_stats.max_defects 
                  << " | Avg: " << line_stats.avg_defects << std::endl;
    }
    
    csv_metrics.close();

    if (!all_line_max_defects.empty()) {
        // Sort ascending to easily grab the max and calculate percentiles
        std::sort(all_line_max_defects.begin(), all_line_max_defects.end());
        size_t n = all_line_max_defects.size();
        
        // Avg value encountered
        double avg_val = std::accumulate(all_line_max_defects.begin(), all_line_max_defects.end(), 0.0) / n;
        
        // Max value encountered
        int max_val = all_line_max_defects.back();
        
        // Helper lambda to fetch nearest-rank percentiles
        auto get_percentile = [&](double p) {
            size_t idx = static_cast<size_t>(std::ceil((p / 100.0) * n)) - 1;
            return all_line_max_defects[std::max<size_t>(0, std::min(idx, n - 1))]; // Clamped for safety
        };

        std::cout << "\n=== Aggregate Line Max Defects (" << n << " keys) ===\n";
        std::cout << "1) Avg Value: " << avg_val << "\n";
        std::cout << "2) Max Value: " << max_val << "\n";
        std::cout << "3) Percentiles:\n";
        std::cout << "   80%: " << get_percentile(80.0) << "\n";
        std::cout << "   85%: " << get_percentile(85.0) << "\n";
        std::cout << "   90%: " << get_percentile(90.0) << "\n";
        std::cout << "   95%: " << get_percentile(95.0) << "\n";
        std::cout << "   99%: " << get_percentile(99.0) << "\n";
        std::cout << "===============================================\n";
    }

    std::cout << "\n[SUCCESS] Spatial Distribution Analysis complete. " << keys_evaluated
              << " keys evaluated. Data saved to spatial_distribution_metrics.csv.\n";
}
