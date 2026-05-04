#include "QwtssCoreGpuOptimized.h"
#include <cmath>
#include <cstdint>


void initialize_gpu_constants_from_core_gpu_opt(const ITileSet* tileset) {
    std::vector<Tile> alphabet = tileset->get_tiles();
    int num_tiles = tileset->get_size();
    // Load constant memory
    CUDA_CHECK(cudaMemcpyToSymbol(qwtss_core_shared_device::d_tiles_const, alphabet.data(), num_tiles * sizeof(Tile)));
    CUDA_CHECK(cudaMemcpyToSymbol(qwtss_core_shared_device::d_num_tiles_const, &num_tiles, sizeof(int)));
}

// ---------------------------------------------------------
// Fixed-Point Weight Structure and Helper
// ---------------------------------------------------------
// This struct is trivially copyable and will be passed by value 
// directly into the kernel, residing in ultra-fast constant memory/registers.
struct FixedPointWeights {
    uint32_t w[5];
};

// Computes the 5 possible scaled Boltzmann weights for the current temperature
inline FixedPointWeights calc_fixed_point_weights(float temp) {
    FixedPointWeights fw;
    for (int delta_e = 0; delta_e <= 4; delta_e++) {
        double prob = std::exp(-(double)delta_e / (double)temp);
        
        // Use 2^24 (16777216.0) for the scaling factor.
        // This safely allows us to sum the weights of all tiles in the alphabet 
        // without overflowing the uint32_t 'sum' variable in the GPU kernel.
        // We retain the std::max(1) to prevent absolute zero probabilities.
        fw.w[delta_e] = std::max((uint32_t)1, static_cast<uint32_t>(prob * 16777216.0));
    }
    return fw;
}

// Version for NxN square grids (Shared Memory Optimized)
__device__ inline int calculate_local_energy_gpu_optimized(int *grid, int grid_size, int row, int col, int proposed_tile_idx, const Tile* s_alphabet) {
    int energy = 0;
    Tile proposed = s_alphabet[proposed_tile_idx];

    // North
    if (row > 0) {
        Tile neighbor = s_alphabet[grid[(row - 1) * grid_size + col]];
        if (proposed.top != neighbor.bottom) energy++;
    }
    // South
    if (row < grid_size - 1) {
        Tile neighbor = s_alphabet[grid[(row + 1) * grid_size + col]];
        if (proposed.bottom != neighbor.top) energy++;
    }
    // West
    if (col > 0) {
        Tile neighbor = s_alphabet[grid[row * grid_size + (col - 1)]];
        if (proposed.left != neighbor.right) energy++;
    }
    // East
    if (col < grid_size - 1) {
        Tile neighbor = s_alphabet[grid[row * grid_size + (col + 1)]];
        if (proposed.right != neighbor.left) energy++;
    }
    return energy;
}

// ---------------------------------------------------------
// Pure Integer GPU Kernel (Very Fast)
// ---------------------------------------------------------
__global__ void heat_bath_kernel_unified_fixed_point(
    int* d_grid, 
    int grid_size, 
    float temp, 
    FixedPointWeights fw, 
    curandStatePhilox4_32_10_t* states, 
    int is_black_phase, 
    const uint8_t* d_locked_mask
) {
    // Cache the alphabet into Shared Memory first to prevent syncthreads deadlocks
    __shared__ Tile s_alphabet[QwtssConfig::MAX_TILES];
    if (threadIdx.y == 0 && threadIdx.x < qwtss_core_shared_device::d_num_tiles_const) {
        s_alphabet[threadIdx.x] = qwtss_core_shared_device::d_tiles_const[threadIdx.x];
    }
    __syncthreads();

    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col >= grid_size || row >= grid_size) return;

    int id = row * grid_size + col;
    bool enforce_color = false;

    // 1. Evaluate Constraints
    if (d_locked_mask) {
        uint8_t mask_val = d_locked_mask[id];
        if (mask_val == 1) {
            enforce_color = true; 
        } else if (mask_val == 2) {
            return; 
        }
    }

    if (((row + col) % 2) == is_black_phase) {
        curandStatePhilox4_32_10_t local_state = states[id]; 
        int num_tiles = qwtss_core_shared_device::d_num_tiles_const;

        // 2. Read required perimeter colors if constrained
        int req_top = -1, req_bottom = -1, req_left = -1, req_right = -1;
        if (enforce_color) {
            Tile t_current = s_alphabet[d_grid[id]];
            if (row == 0) req_top = t_current.top;
            if (row == grid_size - 1) req_bottom = t_current.bottom;
            if (col == 0) req_left = t_current.left;
            if (col == grid_size - 1) req_right = t_current.right;
        }

        // 3. Calculate local energy for ALL possible tiles 
        int energies[QwtssConfig::MAX_TILES];
        int min_energy = 999; // Arbitrary high value to find local minimum

        for (int i = 0; i < num_tiles; i++) {
            if (enforce_color) {
                Tile cand = s_alphabet[i];
                if ((req_top != -1 && cand.top != req_top) ||
                    (req_bottom != -1 && cand.bottom != req_bottom) ||
                    (req_left != -1 && cand.left != req_left) ||
                    (req_right != -1 && cand.right != req_right)) {
                    
                    energies[i] = 999; // Marker for illegal tile
                    continue; 
                }
            }

            int e = calculate_local_energy_gpu_optimized(d_grid, grid_size, row, col, i, s_alphabet);
            energies[i] = e;
            if (e < min_energy) min_energy = e;
        }

        // 4. Map to Fixed-Point Distribution via Host LUT
        uint32_t weights[QwtssConfig::MAX_TILES];
        uint32_t sum = 0;
        
        for (int i = 0; i < num_tiles; i++) {
            if (energies[i] == 999) {
                weights[i] = 0;
            } else {
                int delta_e = energies[i] - min_energy;
                if (delta_e > 4) delta_e = 4; // Safety clamp (Mathematically guaranteed <= 4)
                weights[i] = fw.w[delta_e];
                sum += weights[i];
            }
        }

        // 5. Sample from the distribution purely via integer math
        // Instead of float conversion, use the raw 32-bit random output
        uint32_t rand_val = curand(&local_state); 
        // Fast range reduction maps full 32-bit random to [0, sum - 1] 
        // without expensive modulo division overhead
        uint32_t r = (uint32_t)(((uint64_t)rand_val * (uint64_t)sum) >> 32);

        uint32_t cumulative = 0;
        int selected = d_grid[id];
        
        for (int i = 0; i < num_tiles; i++) {
            if (weights[i] > 0) {
                cumulative += weights[i];
                if (r < cumulative) {
                    selected = i;
                    break;
                }
            }
        }

        d_grid[id] = selected;
        states[id] = local_state;
    }
}

// ---------------------------------------------------------
// Device-Side Defect Counting (Reduction Kernel)
// ---------------------------------------------------------
__global__ void count_defects_kernel(const int* d_grid, int grid_size, int* d_total_defects) {
    __shared__ int shared_defects[256]; // Matches 16x16 threadsPerBlock
    __shared__ Tile s_alphabet[QwtssConfig::MAX_TILES];
    
    // Cache the alphabet into Shared Memory
    if (threadIdx.y == 0 && threadIdx.x < qwtss_core_shared_device::d_num_tiles_const) {
        s_alphabet[threadIdx.x] = qwtss_core_shared_device::d_tiles_const[threadIdx.x];
    }
    __syncthreads();
    
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int tid = threadIdx.y * blockDim.x + threadIdx.x;

    int local_defects = 0;
    
    // Each thread evaluates the right and bottom edges of its assigned tile
    if (row < grid_size && col < grid_size) {
        int id = row * grid_size + col;
        Tile t = s_alphabet[d_grid[id]];
        
        if (col < grid_size - 1) {
            Tile t_right = s_alphabet[d_grid[id + 1]];
            if (t.right != t_right.left) local_defects++;
        }
        if (row < grid_size - 1) {
            Tile t_bottom = s_alphabet[d_grid[id + grid_size]];
            if (t.bottom != t_bottom.top) local_defects++;
        }
    }
    
    shared_defects[tid] = local_defects;
    __syncthreads();

    // Parallel Block Reduction
    for (int s = (blockDim.x * blockDim.y) / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_defects[tid] += shared_defects[tid + s];
        }
        __syncthreads();
    }

    // Thread 0 of each block atomically adds its block total to the global counter
    if (tid == 0) {
        atomicAdd(d_total_defects, shared_defects[0]);
    }
}

// ---------------------------------------------------------
// Host-Side Key Generation Wrapper
// ---------------------------------------------------------
// This version integrates the Fixed-Point Lookup Table (LUT) architecture. It extracts the expensive expf()
// calculation out of the GPU entirely, processing it just once per temperature step on the host CPU. Inside
// the CUDA kernel, it replaces floating-point math and decimal PRNGs with lightning-fast uint32_t integer 
// comparisons and a bitshift range reduction.
GridAnnealResult grid_anneal_gpu_optimized(
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
    // We initialize both here in case we end up calling kernels from either module
    initialize_gpu_constants_from_core_gpu_opt(tileset);
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
        double min_radius = std::sqrt((grid_cells) / (num_pins * M_PI)) * 0.8;

        std::vector<uint8_t> pinned_mask = generate_quenched_disorder_mask(grid_size, num_pins, min_radius, rng);

        CUDA_CHECK(cudaMalloc((void**)&d_locked_mask, pinned_mask.size() * sizeof(uint8_t)));
        CUDA_CHECK(cudaMemcpy(d_locked_mask, pinned_mask.data(), pinned_mask.size() * sizeof(uint8_t), cudaMemcpyHostToDevice));
    }

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((grid_size + 15) / 16, (grid_size + 15) / 16);

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

    // Allocate device counter for fast defect checking
    int* d_total_defects;
    CUDA_CHECK(cudaMalloc((void**)&d_total_defects, sizeof(int)));

    if (params.do_greedy_prequench){
        CUDA_CHECK(cudaMemcpy(h_grid, d_grid, grid_bytes, cudaMemcpyDeviceToHost));
        int pre_quench_defects = count_grid_defects(h_grid, grid_size, alphabet);

        float quench_temp = 0.05f; 
        FixedPointWeights quench_fw = calc_fixed_point_weights(quench_temp);
        
        for (int step = 0; step < 50; step++) {
            heat_bath_kernel_unified_fixed_point<<<numBlocks, threadsPerBlock>>>(d_grid, grid_size, quench_temp, quench_fw, d_states, 0, d_locked_mask);
            heat_bath_kernel_unified_fixed_point<<<numBlocks, threadsPerBlock>>>(d_grid, grid_size, quench_temp, quench_fw, d_states, 1, d_locked_mask);
        }

        CUDA_CHECK(cudaMemcpy(h_grid, d_grid, grid_bytes, cudaMemcpyDeviceToHost));
        int post_quench_defects = count_grid_defects(h_grid, grid_size, alphabet);
    }

    // --- THERMODYNAMIC STATE TRACKING ---
    float current_temp = (params.do_greedy_prequench ? params.reheat_temp : params.initial_temp);
    FixedPointWeights current_fw = calc_fixed_point_weights(current_temp);
    
    int last_defects = INT32_MAX;
    int stagnation_counter = 0;

    // --- MAIN OPTIMIZATION LOOP ---
    for (int step = 1; step <= params.max_steps; step++) {  
        
        if (step % params.check_interval == 0) {
            // Slower, host-side defect counting
            //CUDA_CHECK(cudaMemcpy(h_grid, d_grid, grid_bytes, cudaMemcpyDeviceToHost));
            //int defects = count_grid_defects(h_grid, grid_size, alphabet);

            // Device-side defect counting
            // Reset the device counter to 0
            int zero = 0;
            CUDA_CHECK(cudaMemcpyAsync(d_total_defects, &zero, sizeof(int), cudaMemcpyHostToDevice));
            
            // Launch the reduction kernel
            count_defects_kernel<<<numBlocks, threadsPerBlock>>>(d_grid, grid_size, d_total_defects);
            
            // Pull only the single 4-byte integer back to the CPU
            int defects = 0;
            CUDA_CHECK(cudaMemcpy(&defects, d_total_defects, sizeof(int), cudaMemcpyDeviceToHost));

            //printf("steps %d: defects: %d\n", step, defects);

            if (defects <= params.max_allowed_defects && defects >= params.min_allowed_defects) {
                // Found a valid key: Copy the finalized grid from VRAM back to the CPU one last time.
                CUDA_CHECK(cudaMemcpy(h_grid, d_grid, grid_bytes, cudaMemcpyDeviceToHost));

                CUDA_CHECK(cudaFree(d_grid));
                CUDA_CHECK(cudaFree(d_states));
                CUDA_CHECK(cudaFree(d_total_defects));
                if (d_locked_mask) CUDA_CHECK(cudaFree(d_locked_mask));
                return  { defects, current_temp, step }; // SUCCESS
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
                if (current_temp >= params.reheat_temp)
                    current_temp *= 2.0f;
                else
                    current_temp = std::max(params.reheat_temp, 1.2f * current_temp);
                    
                if (current_temp > 10.0f) current_temp = 10.0f;
                stagnation_counter = 0;
                temp_changed = true;
            }

            if (temp_changed) {
                current_fw = calc_fixed_point_weights(current_temp);
            }
        }

        // 2. MCMC Heat Bath Phase 
        heat_bath_kernel_unified_fixed_point<<<numBlocks, threadsPerBlock>>>(d_grid, grid_size, current_temp, current_fw, d_states, 0, d_locked_mask);
        heat_bath_kernel_unified_fixed_point<<<numBlocks, threadsPerBlock>>>(d_grid, grid_size, current_temp, current_fw, d_states, 1, d_locked_mask);

        // 3. Geometric Cooling
        current_temp *= params.cooling_rate;
        // Re-calculate Fixed Point LUT for the next micro-step
        current_fw = calc_fixed_point_weights(current_temp);
    }

    CUDA_CHECK(cudaFree(d_grid));
    CUDA_CHECK(cudaFree(d_states));
    CUDA_CHECK(cudaFree(d_total_defects));
    if (d_locked_mask) CUDA_CHECK(cudaFree(d_locked_mask));

    return  { -1, 0.0f, params.max_steps }; // FAILED
}
