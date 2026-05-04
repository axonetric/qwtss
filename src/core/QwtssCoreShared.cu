#include "QwtssCoreShared.h"
#include <queue>
#include <random>
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <fstream>
#include <iostream>

// ---------------------------------------------------------
// GPU KERNELS
// ---------------------------------------------------------

// Updated kernel signature: removed Tile* d_tiles as it's in constant memory
__global__ void simulated_annealing_kernel(int *grid, int grid_size, float temperature, curandStatePhilox4_32_10_t* state, int is_black_phase) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < grid_size && col < grid_size) {
        if (((row + col) % 2) == is_black_phase) {
            int id = row * grid_size + col;
            curandStatePhilox4_32_10_t local_state = state[id]; 

            int current_tile = grid[id];
            int current_energy = calculate_local_energy_gpu(grid, grid_size, row, col, current_tile);

            // Use the constant memory variable for tile count
            int proposed_tile = curand(&local_state) % qwtss_core_shared_device::d_num_tiles_const;
            int proposed_energy = calculate_local_energy_gpu(grid, grid_size, row, col, proposed_tile);

            int delta_energy = proposed_energy - current_energy;

            if (delta_energy <= 0) {
                grid[id] = proposed_tile;
            } else {
                float rand_val = curand_uniform(&local_state);
                // Boltzmann distribution for acceptance
                if (rand_val <= expf(- (float)delta_energy / temperature)) {
                    grid[id] = proposed_tile;
                }
            }
            state[id] = local_state;
        }
    }
}

// Version for NxN square grids
// @param d_locked_mask Grid tile allowed behavior mask: 0 = Free, 1 = Pinned Outward Colors, 2 = Pinned Tile ID. If empty, defaults to all tiles Free.
__global__ void heat_bath_kernel_unified(int* d_grid, int grid_size, float temp, curandStatePhilox4_32_10_t* states, int is_black_phase, const uint8_t* d_locked_mask) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col >= grid_size || row >= grid_size) return;

    int id = row * grid_size + col;
    bool enforce_color = false;

    // 1. Evaluate Constraints
    if (d_locked_mask) {
        uint8_t mask_val = d_locked_mask[id];
        if (mask_val == 1) {
            enforce_color = true; // Pin the outward-facing colors
        } else if (mask_val == 2) {
            return; // Rigidly pin the Tile ID (Skip completely)
        }
    }

    if (((row + col) % 2) == is_black_phase) {
        curandStatePhilox4_32_10_t local_state = states[id]; 
        
        float weights[QwtssConfig::MAX_TILES];
        float max_weight = -1e20f;
        int num_tiles = qwtss_core_shared_device::d_num_tiles_const;

        // 2. Read required perimeter colors if constrained
        int req_top = -1, req_bottom = -1, req_left = -1, req_right = -1;
        if (enforce_color) {
            // Instead of allocating and copying a separate d_public_key_colors array to the GPU, since the CPU
            // properly initialized the boundary with valid colors, the current tile sitting in memory already holds
            // the authoritative Public Key colors on its outward edges.
            Tile t_current = qwtss_core_shared_device::d_tiles_const[d_grid[id]];
            if (row == 0) req_top = t_current.top;
            if (row == grid_size - 1) req_bottom = t_current.bottom;
            if (col == 0) req_left = t_current.left;
            if (col == grid_size - 1) req_right = t_current.right;
        }

        // 3. Calculate local energy for ALL possible tiles
        for (int i = 0; i < num_tiles; i++) {
            
            // OPTIMIZATION: Filter illegal boundary colors before calculating energy
            if (enforce_color) {
                Tile cand = qwtss_core_shared_device::d_tiles_const[i];
                if ((req_top != -1 && cand.top != req_top) ||
                    (req_bottom != -1 && cand.bottom != req_bottom) ||
                    (req_left != -1 && cand.left != req_left) ||
                    (req_right != -1 && cand.right != req_right)) {
                    
                    weights[i] = -1e20f; // Infinite penalty (0% probability)
                    continue; // Skip the expensive Global Memory reads in calculate_local_energy_gpu
                }
            }

            int e = calculate_local_energy_gpu(d_grid, grid_size, row, col, i);
            weights[i] = - (float)e / temp;
            if (weights[i] > max_weight) max_weight = weights[i];
        }

        // 4. Softmax / Boltzmann Distribution
        float sum = 0.0f;
        for (int i = 0; i < num_tiles; i++) {
            weights[i] = expf(weights[i] - max_weight); // Math gracefully handles -1e20f -> 0.0f
            sum += weights[i];
        }

        // 5. Sample from the distribution
        float r = curand_uniform(&local_state) * sum;
        float cumulative = 0.0f;
        // CRITICAL: Fallback to the current tile (mathematically guaranteed 
        // to be valid) if we experience floating-point fall-through.
        int selected = d_grid[id];
        for (int i = 0; i < num_tiles; i++) {
            cumulative += weights[i];
            // ADDED SAFETY: Explicitly ensure 0-weight options can never be selected, 
            // even if 'r' lands exactly on a boundary edge.
            if (weights[i] > 0.0f && r <= cumulative){
                selected = i;
                break;
            }
        }

        /*
        // ==========================================================
        // CUDA KERNEL TRACE LOGIC
        // Only trigger for the exact center tile when Temp is very high
        // ==========================================================
        if (row == QwtssConfig::GRID_SIZE / 2 && col == QwtssConfig::GRID_SIZE / 2 && temp >= 9.0f) {
            printf("--- GPU TRACE (Row %d, Col %d) ---\n", row, col);
            printf("Temp: %.2f | Max Weight: %.2f | Sum: %.4f | PRNG 'r': %.4f\n", temp, max_weight, sum, r);
            printf("Current Tile: %d | Selected Tile: %d\n", d_grid[id], selected);
            printf("Tile 0 Energy: %d (exp_w: %.4f)\n", calculate_local_energy_gpu(d_grid, row, col, 0), weights[0]);
            printf("Tile 1 Energy: %d (exp_w: %.4f)\n", calculate_local_energy_gpu(d_grid, row, col, 1), weights[1]);
            printf("----------------------------------\n");
        }
        // ==========================================================
        */

        d_grid[id] = selected;
        states[id] = local_state;
    }
}

// A physical heat-bath. No logic, no early exits. The boundary edge colors are locked in place (unless otherwise specified via d_locked_mask).
int locked_ais_mcmc_step_gpu(
    int* h_grid, int grid_size, int* d_grid, curandStatePhilox4_32_10_t* d_states, ITileSet* tileset, 
    float current_temp, int steps, const uint8_t* d_locked_mask
){
    if (!d_locked_mask) throw std::invalid_argument("d_locked_mask must be specified for the boundary condition");

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((grid_size + 15) / 16, (grid_size + 15) / 16);
    int grid_bytes = grid_size * grid_size * sizeof(int);

    // Run the requested number of thermodynamic sweeps at the exact requested temperature
    // The unified kernel handles legacy 100% borders, partial borders, and pinned defects dynamically.
    for (int i = 0; i < steps; i++) {
        heat_bath_kernel_unified<<<numBlocks, threadsPerBlock>>>(d_grid, grid_size, current_temp, d_states, 0, d_locked_mask);
        heat_bath_kernel_unified<<<numBlocks, threadsPerBlock>>>(d_grid, grid_size, current_temp, d_states, 1, d_locked_mask);
    }

    CUDA_CHECK(cudaMemcpy(h_grid, d_grid, grid_bytes, cudaMemcpyDeviceToHost));
    return count_grid_defects(h_grid, grid_size, tileset->get_tiles());
}

void initialize_gpu_constants_from_core_shared(const ITileSet* tileset) {
    std::vector<Tile> alphabet = tileset->get_tiles();
    int num_tiles = tileset->get_size();
    // Load constant memory
    CUDA_CHECK(cudaMemcpyToSymbol(qwtss_core_shared_device::d_tiles_const, alphabet.data(), num_tiles * sizeof(Tile)));
    CUDA_CHECK(cudaMemcpyToSymbol(qwtss_core_shared_device::d_num_tiles_const, &num_tiles, sizeof(int)));
}

// Generates a boundary mask (of 1 = Pinned Outward Colors) keeping only a specific percentage of the perimeter, using uniform random dropout.
// @param keep_boundary_percentage 0.0f to 1.0f
// @param existing_mask Pass the original mask if we need to preserve any pinned tiles (value=2) for cryptographic rebar
std::vector<uint8_t> generate_partial_boundary_mask(int grid_size, float keep_boundary_percentage, ChaCha20PRNG& rng,
    const std::vector<uint8_t>& existing_mask)
{
    if (keep_boundary_percentage < 0.0f || keep_boundary_percentage > 1.0f) throw std::invalid_argument("'keep_boundary_percentage' must be 0.0f to 1.0f");

    int grid_count = grid_size * grid_size;

    // Either clone the existing mask to preserve Rebar, or start fresh
    std::vector<uint8_t> mask;
    if (!existing_mask.empty()) {
        mask = existing_mask;
        // Wipe ONLY the perimeter (1s) so we can recalculate the new density below
        for (int i = 0; i < grid_count; i++) {
            if (mask[i] == 1) mask[i] = 0;
        }
    } else {
        mask.assign(grid_count, 0); // Everything defaults to unlocked (0 = Free)
    }

    // Collect the 1D indices of all outer boundary tiles
    std::vector<int> boundary_indices;
    boundary_indices.reserve(grid_size * 4 - 4);
    
    for (int r = 0; r < grid_size; r++) {
        for (int c = 0; c < grid_size; c++) {
            if (r == 0 || r == grid_size - 1 || c == 0 || c == grid_size - 1) {
                boundary_indices.push_back(r * grid_size + c);
            }
        }
    }
    
    // Calculate exactly how many boundary tiles to keep locked
    int total_boundary_tiles = boundary_indices.size();
    int keep_count = std::round(total_boundary_tiles * keep_boundary_percentage);
    
    // Safety clamps
    if (keep_count < 0) keep_count = 0;
    if (keep_count > total_boundary_tiles) keep_count = total_boundary_tiles;

    // Randomly select which boundary tiles survive
    // Applies a 100% cross-platform deterministic, cryptographically unbiased Fisher-Yates Shuffle
    rng.deterministic_shuffle(boundary_indices);
    
    // Lock only the surviving tiles in the mask
    for (int i = 0; i < keep_count; i++) {
        mask[boundary_indices[i]] = 1;         // 1 = Pinned Outward Colors
    }
    
    return mask;
}

// @param run_target_defects This is only used to seed a proper uniform probability of expected defects in the edge tiles.
void generate_boundary_conditioned_random_seed_grid(const std::vector<int>& ground_truth_grid, int grid_size,
    std::vector<int>& h_grid, int run_target_defects, std::vector<Tile>& alphabet, ChaCha20PRNG& rng,
    const std::vector<uint8_t>& public_key_mask){

    int num_tiles = alphabet.size();
    std::uniform_int_distribution<int> dist_tile(0, num_tiles - 1);

    std::uniform_real_distribution<double> dist_prob(0.0, 1.0);

    // Generalized internal edge count for any N x N grid: 2 * N * (N - 1)
    double total_internal_edges = 2.0 * grid_size * (grid_size - 1);
    // Calculate edge tile defect probability based on the run target
    // Divide by 2 because mutating one edge tile usually breaks two tile edges
    double p_defect = ((double)run_target_defects / total_internal_edges) / 2.0;

    for (int r = 0; r < grid_size; r++) {
        for (int c = 0; c < grid_size; c++) {
            int idx = r * grid_size + c;
            bool is_geometric_boundary = (r == 0 || r == grid_size - 1 || c == 0 || c == grid_size - 1);

            // Determine the constraint type: 0 = Free, 1 = Pinned Outward Colors, 2 = Pinned Tile ID
            uint8_t constraint_type = 0;
            if (!public_key_mask.empty()) {
                constraint_type = public_key_mask[idx];
            } else if (is_geometric_boundary) {
                constraint_type = 1; // Legacy fallback: treat geometric boundaries as Public Key colors
            }

            if (constraint_type == 2) {
                // 2 = Pinned Tile ID: Strictly pin the exact Tile ID (No mutations allowed)
                h_grid[idx] = ground_truth_grid[idx];
            } else if (constraint_type == 1) {
                // 1 = Pinned Outward Colors
                int true_tile = ground_truth_grid[idx];

                std::vector<int> valid_any;
                std::vector<int> valid_mutations;
                
                // Find ALL replacement tiles that STILL satisfy the Public Key boundary colors
                for (int t = 0; t < num_tiles; t++) {
                    bool preserves_pub_key = true;
                    if (r == 0 && alphabet[t].top != alphabet[true_tile].top) preserves_pub_key = false;
                    if (r == grid_size - 1 && alphabet[t].bottom != alphabet[true_tile].bottom) preserves_pub_key = false;
                    if (c == 0 && alphabet[t].left != alphabet[true_tile].left) preserves_pub_key = false;
                    if (c == grid_size - 1 && alphabet[t].right != alphabet[true_tile].right) preserves_pub_key = false;

                    if (preserves_pub_key) {
                        valid_any.push_back(t); // Any tile that fits the edge colors
                        if (t != true_tile) valid_mutations.push_back(t); // Strict mutations for forced defects
                    }
                }

                // Probabilistically determine if this boundary tile requires a forced defect seed
                if (dist_prob(rng) < p_defect && !valid_mutations.empty()) {
                    std::uniform_int_distribution<int> dist_mut(0, valid_mutations.size() - 1);
                    h_grid[idx] = valid_mutations[dist_mut(rng)];
                } else if (!valid_any.empty()) {
                    // Standard seed: Sample uniformly from all valid eligible options
                    std::uniform_int_distribution<int> dist_any(0, valid_any.size() - 1);
                    h_grid[idx] = valid_any[dist_any(rng)];
                } else {
                    // Corner case where no other tile in the alphabet shares the required colors
                    h_grid[idx] = true_tile;
                }
            } else {
                // Randomize the interior (or any tiles marked 0 = Free)
                h_grid[idx] = dist_tile(rng);
            }
        }
    }
}

// This function throws if an expected boundary edge color or pinned tile constraint is violated
void confirm_mask_constraints_honored(const std::vector<int> h_grid, const std::vector<uint8_t> boundary_mask, int grid_size,
    const std::vector<int> ground_truth, const std::vector<Tile>& alphabet, bool do_logging){

    // DIAGNOSTICS TRACKERS
    int tiles_same = 0, tiles_changed = 0;
    int tiles_changed_pinned_2 = 0;     // Track Tile ID violations

    int colors_same = 0, colors_changed = 0;
    int colors_changed_pinned_1 = 0;    // Track pinned edge color violations

    for (int r = 0; r < grid_size; r++) {
        for (int c = 0; c < grid_size; c++) {
            int idx = r * grid_size + c;
            int orig_tile = ground_truth[idx];
            int final_tile = h_grid[idx];
            uint8_t constraint = boundary_mask.empty() ? 0 : boundary_mask[idx];

            // Tile ID Diagnostics
            if (orig_tile == final_tile) {
                tiles_same++;
            } else {
                tiles_changed++;
                if (constraint == 2) tiles_changed_pinned_2++; // Track illegal tile mutations
            }

            // Outward Color Diagnostics (Only applies to geometric boundaries)
            bool is_boundary = (r == 0 || r == grid_size - 1 || c == 0 || c == grid_size - 1);
            if (is_boundary) {
                auto check_color = [&](int orig_color, int final_color) {
                    if (orig_color == final_color) {
                        colors_same++;
                    } else {
                        colors_changed++;
                        if (constraint == 1) colors_changed_pinned_1++;
                    }
                };

                if (r == 0) check_color(alphabet[orig_tile].top, alphabet[final_tile].top);
                if (r == grid_size - 1) check_color(alphabet[orig_tile].bottom, alphabet[final_tile].bottom);
                if (c == 0) check_color(alphabet[orig_tile].left, alphabet[final_tile].left);
                if (c == grid_size - 1) check_color(alphabet[orig_tile].right, alphabet[final_tile].right);
            }
        }
    }

    /*
    if (do_logging){
        std::cout << "\n--- Constraint Diagnostics ---\n";
        std::cout << "[Tiles]  Same ID: " << tiles_same << " | Changed ID: " << tiles_changed << "\n";
        std::cout << "         -> ILLEGAL Pinned ID (2) Mutations:     " << tiles_changed_pinned_2 << "\n";
        std::cout << "[Colors] Same:    " << colors_same << " | Changed:    " << colors_changed << "\n";
        std::cout << "         -> ILLEGAL Pinned Color (1) Mutations:  " << colors_changed_pinned_1 << "\n";
        std::cout << "----------------------------\n";
    }
    */

    // Failure Conditions
    if (colors_changed_pinned_1 != 0) {
        throw std::runtime_error("CRITICAL Constraint Failure: Pinned outward boundary colors (Mask=1) were illegally mutated!");
    }
    if (tiles_changed_pinned_2 != 0) {
        throw std::runtime_error("CRITICAL Constraint Failure: Rigidly pinned interior Tile IDs (Mask=2) were illegally mutated!");
    }

    if (do_logging) std::cout << "Valid PK/SK identity confirmed." << std::endl;
}

// Generates a 2 = Pinned Tile ID mask of locked tiles spread out by a minimum radius using Dart Throwing
std::vector<uint8_t> generate_quenched_disorder_mask(
    int grid_size, int num_pins, double min_radius, ChaCha20PRNG& rng
) {
    throw std::runtime_error("Defect doping is expected to be disabled currently.");

    std::vector<uint8_t> is_locked(grid_size * grid_size, 0); // 0 = Free
    std::vector<std::pair<int, int>> pinned_coords;
    
    std::uniform_int_distribution<int> dist_pos(1, grid_size - 2); // Keep off the public boundary

    // Precalculate squared radius to avoid expensive std::sqrt() in the inner loop
    double min_radius_sq = min_radius * min_radius;
    
    // Give each individual pin up to 2000 attempts to find a valid home
    int max_attempts_per_pin = 2000; 

    while (pinned_coords.size() < num_pins) {
        bool pin_placed = false;

        for (int attempts = 0; attempts < max_attempts_per_pin; attempts++) {
            int r = dist_pos(rng);
            int c = dist_pos(rng);

            // Perform rejection sampling
            bool too_close = false;
            for (const auto& pin : pinned_coords) {
                // Fast squared Euclidean distance
                double dist_sq = (r - pin.first) * (r - pin.first) + (c - pin.second) * (c - pin.second);
                if (dist_sq < min_radius_sq) {
                    too_close = true;
                    break;
                }
            }

            if (!too_close) {
                pinned_coords.push_back({r, c});
                is_locked[r * grid_size + c] = 2; // 2 = Pinned Tile ID
                pin_placed = true;
                break; // Break out of the attempts loop and move to the next pin
            }
        }

        // If a pin fails all 2000 attempts, the grid is completely saturated for this radius
        if (!pin_placed) {
            std::cout << "[WARNING] Grid saturation reached. Could only fit " 
                      << pinned_coords.size() << " / " << num_pins 
                      << " pins with a radius of " << min_radius << std::endl;
            break; 
        }
    }

    return is_locked;
}

// Scans the grid to find connected clusters of defective tiles
DefectStats analyze_defect_topology(int* h_grid, int grid_size, const std::vector<Tile>& tiles) {
    std::vector<std::vector<bool>> is_defective(grid_size, std::vector<bool>(grid_size, false));
    int total_defective = 0;

    // Identify all tiles involved in a defect
    for (int r = 0; r < grid_size; r++) {
        for (int c = 0; c < grid_size; c++) {
            int id = r * grid_size + c;
            Tile current = tiles[h_grid[id]];
            bool has_defect = false;
            
            // Check North, South, West, East
            if (r > 0 && current.top != tiles[h_grid[(r-1)*grid_size + c]].bottom) has_defect = true;
            if (r < grid_size - 1 && current.bottom != tiles[h_grid[(r+1)*grid_size + c]].top) has_defect = true;
            if (c > 0 && current.left != tiles[h_grid[r*grid_size + (c-1)]].right) has_defect = true;
            if (c < grid_size - 1 && current.right != tiles[h_grid[r*grid_size + (c+1)]].left) has_defect = true;

            if (has_defect) {
                is_defective[r][c] = true;
                total_defective++;
            }
        }
    }

    // Find connected components (BFS)
    std::vector<std::vector<bool>> visited(grid_size, std::vector<bool>(grid_size, false));
    DefectStats stats;
    stats.total_defective_tiles = total_defective;

    int dr[] = {-1, 1, 0, 0};
    int dc[] = {0, 0, -1, 1};

    for (int r = 0; r < grid_size; r++) {
        for (int c = 0; c < grid_size; c++) {
            if (is_defective[r][c] && !visited[r][c]) {
                // Start a new BFS for a discovered hole
                stats.num_holes++;
                int current_hole_size = 0;
                
                std::queue<std::pair<int, int>> q;
                q.push({r, c});
                visited[r][c] = true;

                while (!q.empty()) {
                    auto [curr_r, curr_c] = q.front();
                    q.pop();
                    current_hole_size++;

                    // Traverse adjacent tiles (North, South, West, East)
                    for (int i = 0; i < 4; i++) {
                        int nr = curr_r + dr[i];
                        int nc = curr_c + dc[i];

                        // If neighbor is within bounds, defective, and unvisited, add to current hole
                        if (nr >= 0 && nr < grid_size && nc >= 0 && nc < grid_size) {
                            if (is_defective[nr][nc] && !visited[nr][nc]) {
                                visited[nr][nc] = true;
                                q.push({nr, nc});
                            }
                        }
                    }
                }

                if (current_hole_size > stats.max_hole_size) {
                    stats.max_hole_size = current_hole_size;
                }
            }
        }
    }

    if (stats.num_holes > 0) {
        stats.avg_hole_size = (float)total_defective / stats.num_holes;
    }

    return stats;
}

DefectDistributionStats calculate_defect_spatial_distribution_subgrids(
    const std::vector<int>& h_grid, 
    int grid_size, 
    const std::vector<Tile>& alphabet, 
    int sub_grid_size
) {
    if (grid_size % sub_grid_size != 0) {
        throw std::invalid_argument("grid_size must be perfectly divisible by sub_grid_size.");
    }

    int num_blocks = grid_size / sub_grid_size;
    int total_subgrids = num_blocks * num_blocks;
    std::vector<int> subgrid_counts(total_subgrids, 0);

    // Iterate through each sub-grid block
    for (int block_r = 0; block_r < num_blocks; ++block_r) {
        for (int block_c = 0; block_c < num_blocks; ++block_c) {
            
            int block_defects = 0;
            
            // Scan the specific tiles within this sub-grid
            for (int r = 0; r < sub_grid_size; ++r) {
                for (int c = 0; c < sub_grid_size; ++c) {
                    int global_r = block_r * sub_grid_size + r;
                    int global_c = block_c * sub_grid_size + c;
                    
                    int u = global_r * grid_size + global_c;
                    int tile_u = h_grid[u];

                    // Check Right Neighbor (Skip if at the absolute right edge of the full grid)
                    if (global_c < grid_size - 1) {
                        int v = global_r * grid_size + (global_c + 1);
                        int tile_v = h_grid[v];
                        if (alphabet[tile_u].right != alphabet[tile_v].left) {
                            block_defects++;
                        }
                    }

                    // Check Bottom Neighbor (Skip if at the absolute bottom edge of the full grid)
                    if (global_r < grid_size - 1) {
                        int v = (global_r + 1) * grid_size + global_c;
                        int tile_v = h_grid[v];
                        if (alphabet[tile_u].bottom != alphabet[tile_v].top) {
                            block_defects++;
                        }
                    }
                }
            }
            // Store the final count for this sub-grid
            subgrid_counts[block_r * num_blocks + block_c] = block_defects;
        }
    }

    // ------------------------------------------------------------------
    // Calculate the Statistical Observables
    // ------------------------------------------------------------------
    DefectDistributionStats stats;
    stats.min_defects = *std::min_element(subgrid_counts.begin(), subgrid_counts.end());
    stats.max_defects = *std::max_element(subgrid_counts.begin(), subgrid_counts.end());
    
    double sum = 0.0;
    double sum_sq = 0.0;
    for (int count : subgrid_counts) {
        sum += count;
        sum_sq += ((double)count * count);
    }
    
    stats.total_defects = static_cast<int>(sum);
    stats.avg_defects = sum / total_subgrids;
    
    // Population Variance = E[X^2] - (E[X])^2
    double variance = (sum_sq / total_subgrids) - (stats.avg_defects * stats.avg_defects);
    
    // Clamp at 0.0 to prevent NaN from floating-point inaccuracies on perfectly uniform grids
    stats.std_dev_defects = std::sqrt(std::max(0.0, variance));

    return stats;
}

// If needed, we can optimize this to O(N^2) instead of O(N^2⋅K^2) by applying a 2D Prefix Sum array over the
// grid's horizontal and vertical defect maps before sweeping the window.
DefectDistributionStats calculate_defect_spatial_distribution_sliding(
    const std::vector<int>& h_grid, 
    int grid_size, 
    const std::vector<Tile>& alphabet, 
    int sliding_grid_size
) {
    if (sliding_grid_size <= 0 || sliding_grid_size > grid_size) {
        throw std::invalid_argument("sliding_grid_size must be > 0 and <= grid_size.");
    }

    int max_wr = grid_size - sliding_grid_size;
    int max_wc = grid_size - sliding_grid_size;
    int total_windows = (max_wr + 1) * (max_wc + 1);
    
    std::vector<int> window_counts(total_windows, 0);

    // Slide the window across the grid
    for (int wr = 0; wr <= max_wr; ++wr) {
        for (int wc = 0; wc <= max_wc; ++wc) {
            
            int window_defects = 0;

            // Scan the specific tiles within this sliding window
            for (int r = 0; r < sliding_grid_size; ++r) {
                for (int c = 0; c < sliding_grid_size; ++c) {
                    int global_r = wr + r;
                    int global_c = wc + c;
                    
                    int u = global_r * grid_size + global_c;
                    int tile_u = h_grid[u];

                    // Check Right Neighbor (Strictly within the sliding window)
                    if (c < sliding_grid_size - 1) {
                        int v = global_r * grid_size + (global_c + 1);
                        int tile_v = h_grid[v];
                        if (alphabet[tile_u].right != alphabet[tile_v].left) {
                            window_defects++;
                        }
                    }

                    // Check Bottom Neighbor (Strictly within the sliding window)
                    if (r < sliding_grid_size - 1) {
                        int v = (global_r + 1) * grid_size + global_c;
                        int tile_v = h_grid[v];
                        if (alphabet[tile_u].bottom != alphabet[tile_v].top) {
                            window_defects++;
                        }
                    }
                }
            }
            // Store the final count for this specific sliding window
            window_counts[wr * (max_wc + 1) + wc] = window_defects;
        }
    }

    // ------------------------------------------------------------------
    // Calculate the Statistical Observables
    // ------------------------------------------------------------------
    DefectDistributionStats stats;
    stats.min_defects = *std::min_element(window_counts.begin(), window_counts.end());
    stats.max_defects = *std::max_element(window_counts.begin(), window_counts.end());
    
    double sum = 0.0;
    double sum_sq = 0.0;
    for (int count : window_counts) {
        sum += count;
        sum_sq += ((double)count * count);
    }

    stats.total_defects = 0; // Not applicable for overlapping windows
    stats.avg_defects = sum / total_windows;

    // Population Variance = E[X^2] - (E[X])^2
    double variance = (sum_sq / total_windows) - (stats.avg_defects * stats.avg_defects);

    // Clamp at 0.0 to prevent NaN from floating-point inaccuracies
    stats.std_dev_defects = std::sqrt(std::max(0.0, variance));

    return stats;
}

// Returns stats on the sums of perpendicular shearing defects for each horizontal and vertical seam (independently).
DefectDistributionStats calculate_defect_spatial_distribution_lines(
    const std::vector<int>& h_grid, 
    int grid_size, 
    const std::vector<Tile>& alphabet
) {
    // 64x64 grid has 63 horizontal and 63 vertical interior seams
    int num_seams_per_axis = grid_size - 1; 
    int total_seams = num_seams_per_axis * 2; 

    std::vector<int> seam_counts;
    seam_counts.reserve(total_seams);

    // 1. Horizontal Seams (Boundary between Row r and Row r+1)
    // There are exactly (grid_size - 1) such boundaries.
    for (int r = 0; r < num_seams_per_axis; ++r) {
        int defects = 0;
        for (int c = 0; c < grid_size; ++c) { 
            int top = r * grid_size + c;
            int bottom = (r + 1) * grid_size + c;
            
            // Check the North/South connection between the two rows
            if (alphabet[h_grid[top]].bottom != alphabet[h_grid[bottom]].top) {
                defects++;
            }
        }
        seam_counts.push_back(defects);
    }

    // 2. Vertical Seams (Boundary between Column c and Column c+1)
    // There are exactly (grid_size - 1) such boundaries.
    for (int c = 0; c < num_seams_per_axis; ++c) {
        int defects = 0;
        for (int r = 0; r < grid_size; ++r) { 
            int left = r * grid_size + c;
            int right = r * grid_size + (c + 1);
            
            // Check the East/West connection between the two columns
            if (alphabet[h_grid[left]].right != alphabet[h_grid[right]].left) {
                defects++;
            }
        }
        seam_counts.push_back(defects);
    }

    // Stats calculation
    DefectDistributionStats stats;
    stats.max_defects = *std::max_element(seam_counts.begin(), seam_counts.end());
    stats.min_defects = *std::min_element(seam_counts.begin(), seam_counts.end());
    
    double sum = std::accumulate(seam_counts.begin(), seam_counts.end(), 0.0);
    stats.total_defects = static_cast<int>(sum); // This now matches count_grid_defects() perfectly
    stats.avg_defects = sum / total_seams;
    
    // Variance/StdDev
    double sum_sq = 0;
    for (int c : seam_counts) sum_sq += (c * c);
    double variance = (sum_sq / total_seams) - (stats.avg_defects * stats.avg_defects);
    stats.std_dev_defects = std::sqrt(std::max(0.0, variance));

    return stats;
}

// Returns the defective tile count within a given size "picture frame" border region (cumulative).
int calculate_defect_spatial_distribution_frame(const std::vector<int>& h_grid, int grid_size, const std::vector<Tile>& alphabet, int frame_depth) {
    int frame_defect_count = 0;

    for (int r = 0; r < grid_size; r++) {
        for (int c = 0; c < grid_size; c++) {
            // Check if this tile is within the "frame_depth" of any border
            if (r < frame_depth || r >= grid_size - frame_depth || 
                c < frame_depth || c >= grid_size - frame_depth) {
                
                int tile_id = h_grid[r * grid_size + c];
                Tile t = alphabet[tile_id];
                
                bool is_defective = false;
                if (r > 0 && t.top != alphabet[h_grid[(r - 1) * grid_size + c]].bottom) is_defective = true;
                if (r < grid_size - 1 && t.bottom != alphabet[h_grid[(r + 1) * grid_size + c]].top) is_defective = true;
                if (c > 0 && t.left != alphabet[h_grid[r * grid_size + (c - 1)]].right) is_defective = true;
                if (c < grid_size - 1 && t.right != alphabet[h_grid[r * grid_size + (c + 1)]].left) is_defective = true;

                if (is_defective) {
                    frame_defect_count++;
                }
            }
        }
    }
    return frame_defect_count;
}

// Returns the alien tile count within a given size "picture frame" border region (cumulative).
int calculate_alien_frame_count(
    const std::vector<int>& grid, 
    const std::vector<int>& plane_A, 
    const std::vector<int>& plane_B, 
    int grid_size, 
    int frame_depth)
{
    int alien_count = 0;
    for (int r = 0; r < grid_size; r++) {
        for (int c = 0; c < grid_size; c++) {
            if (r < frame_depth || r >= grid_size - frame_depth || 
                c < frame_depth || c >= grid_size - frame_depth) {
                
                int idx = r * grid_size + c;
                if (grid[idx] != plane_A[idx] && grid[idx] != plane_B[idx]) {
                    alien_count++;
                }
            }
        }
    }
    return alien_count;
}

// Returns the number of core tiles with 1+ defects
int calculate_core_defects(const std::vector<int>& h_grid, int grid_size, const std::vector<Tile>& alphabet, int core_size) {
    int core_defect_count = 0;
    int offset = (grid_size - core_size) / 2; // For 32, offset is 16

    for (int r = offset; r < offset + core_size; r++) {
        for (int c = offset; c < offset + core_size; c++) {
            int tile_id = h_grid[r * grid_size + c];
            Tile t = alphabet[tile_id];
            
            bool is_defective = false;
            if (r > 0 && t.top != alphabet[h_grid[(r - 1) * grid_size + c]].bottom) is_defective = true;
            if (r < grid_size - 1 && t.bottom != alphabet[h_grid[(r + 1) * grid_size + c]].top) is_defective = true;
            if (c > 0 && t.left != alphabet[h_grid[r * grid_size + (c - 1)]].right) is_defective = true;
            if (c < grid_size - 1 && t.right != alphabet[h_grid[r * grid_size + (c + 1)]].left) is_defective = true;

            if (is_defective) {
                core_defect_count++;
            }
        }
    }
    return core_defect_count;
}

std::vector<int> calculate_local_energy_histogram(const std::vector<int>& h_grid, int grid_size, const std::vector<Tile>& alphabet) {
    // Initialize a histogram with 5 buckets (for 0, 1, 2, 3, and 4 defects), all set to 0.
    std::vector<int> histogram(5, 0);

    for (int r = 0; r < grid_size; r++) {
        for (int c = 0; c < grid_size; c++) {
            int tile_id = h_grid[r * grid_size + c];
            Tile t = alphabet[tile_id];
            
            int local_defects = 0;

            // Check Top
            if (r > 0 && t.top != alphabet[h_grid[(r - 1) * grid_size + c]].bottom) {
                local_defects++;
            }
            // Check Bottom
            if (r < grid_size - 1 && t.bottom != alphabet[h_grid[(r + 1) * grid_size + c]].top) {
                local_defects++;
            }
            // Check Left
            if (c > 0 && t.left != alphabet[h_grid[r * grid_size + (c - 1)]].right) {
                local_defects++;
            }
            // Check Right
            if (c < grid_size - 1 && t.right != alphabet[h_grid[r * grid_size + (c + 1)]].left) {
                local_defects++;
            }

            // Increment the appropriate bucket (H0 through H4)
            histogram[local_defects]++;
        }
    }
    
    return histogram;
}

int calculate_alien_tiles(const std::vector<int>& grid, 
                          const std::vector<int>& plane_A, 
                          const std::vector<int>& plane_B, 
                          int grid_size)
{
    int alien_count = 0;
    int num_tiles = grid_size * grid_size;

    for (int i = 0; i < num_tiles; i++) {
        // A tile is "Alien" if it does not match the Oracle A tile *and* does not match the Oracle B tile
        if (grid[i] != plane_A[i] && grid[i] != plane_B[i]) {
            alien_count++;
        }
    }

    return alien_count;
}


void export_grid_to_ppm(const int* grid, int grid_size, ITileSet* tileset, const std::string& filename) {
    std::vector<Tile> alphabet = tileset->get_tiles();
    
    std::ofstream out(filename);
    if (!out) {
        std::cerr << "Error: Could not open " << filename << " for writing.\n";
        return;
    }

    // Image dimensions are doubled because each tile is a 2x2 pixel block
    int img_size = grid_size * 2;
    
    // Write the P3 PPM Header (ASCII RGB, Width, Height, Max Color Value)
    out << "P3\n" << img_size << " " << img_size << "\n255\n";

    // High-contrast 16-color palette (RGB)
    const int palette[16][3] = {
        {255, 0, 0},     // 0: Red
        {0, 255, 0},     // 1: Green
        {0, 0, 255},     // 2: Blue
        {255, 255, 0},   // 3: Yellow
        {0, 255, 255},   // 4: Cyan
        {255, 0, 255},   // 5: Magenta
        {255, 128, 0},   // 6: Orange
        {128, 0, 255},   // 7: Purple
        {0, 255, 128},   // 8: Spring Green
        {255, 0, 128},   // 9: Pink
        {128, 255, 0},   // 10: Lime
        {0, 128, 255},   // 11: Light Blue
        {128, 0, 0},     // 12: Maroon
        {0, 128, 0},     // 13: Dark Green
        {0, 0, 128},     // 14: Navy
        {64, 64, 64}     // 15: Dark Gray
    };

    for (int r = 0; r < grid_size; r++) {
        // --- PIXEL ROW 1: The Top and Right colors of the tile ---
        for (int c = 0; c < grid_size; c++) {
            int tile_id = grid[r * grid_size + c];
            Tile t = alphabet[tile_id];
            
            // Safe modulo to prevent crashes if the tileset has >16 colors
            int c_top = t.top % 16;
            int c_right = t.right % 16;

            // [0][0]: Top Color
            out << palette[c_top][0] << " " << palette[c_top][1] << " " << palette[c_top][2] << " ";
            // [0][1]: Right Color
            out << palette[c_right][0] << " " << palette[c_right][1] << " " << palette[c_right][2] << " ";
        }
        out << "\n";

        // --- PIXEL ROW 2: The Left and Bottom colors of the tile ---
        for (int c = 0; c < grid_size; c++) {
            int tile_id = grid[r * grid_size + c];
            Tile t = alphabet[tile_id];
            
            int c_left = t.left % 16;
            int c_bottom = t.bottom % 16;

            // [1][0]: Left Color
            out << palette[c_left][0] << " " << palette[c_left][1] << " " << palette[c_left][2] << " ";
            // [1][1]: Bottom Color
            out << palette[c_bottom][0] << " " << palette[c_bottom][1] << " " << palette[c_bottom][2] << " ";
        }
        out << "\n";
    }

    out.close();
    std::cout << "[*] Exported visual grid to " << filename << "\n";
}

void export_defect_edges_to_ppm(const int* grid, int grid_size, ITileSet* tileset, const std::string& filename) {
    std::vector<Tile> alphabet = tileset->get_tiles();
    
    std::ofstream out(filename);
    if (!out) {
        std::cerr << "Error: Could not open " << filename << " for writing.\n";
        return;
    }

    // A grid of N tiles has N+1 boundaries (including outer edges).
    // By mapping tiles to a 2x2 structure + 1 closing pixel, we get perfectly separated edge lines.
    int img_size = 2 * grid_size + 1;
    
    // Write the P3 PPM Header
    out << "P3\n" << img_size << " " << img_size << "\n255\n";

    for (int ir = 0; ir < img_size; ir++) {
        for (int ic = 0; ic < img_size; ic++) {
            int r = 0, g = 0, b = 0; // Default background is Black

            bool is_horizontal_edge = (ir % 2 == 0) && (ic % 2 != 0);
            bool is_vertical_edge   = (ir % 2 != 0) && (ic % 2 == 0);
            bool is_intersection    = (ir % 2 == 0) && (ic % 2 == 0);

            // Grid coordinates corresponding to the pixel
            int gr = ir / 2;
            int gc = ic / 2;

            if (is_horizontal_edge) {
                if (gr == 0 || gr == grid_size) {
                    r = 128; g = 128; b = 128; // Perimeter
                } else {
                    Tile t_above = alphabet[grid[(gr - 1) * grid_size + gc]];
                    Tile t_below = alphabet[grid[gr * grid_size + gc]];
                    if (t_above.bottom != t_below.top) {
                        r = 255; g = 255; b = 255; // Defect!
                    }
                }
            } 
            else if (is_vertical_edge) {
                if (gc == 0 || gc == grid_size) {
                    r = 128; g = 128; b = 128; // Perimeter
                } else {
                    Tile t_left = alphabet[grid[gr * grid_size + (gc - 1)]];
                    Tile t_right = alphabet[grid[gr * grid_size + gc]];
                    if (t_left.right != t_right.left) {
                        r = 255; g = 255; b = 255; // Defect!
                    }
                }
            } 
            else if (is_intersection) {
                if (gr == 0 || gr == grid_size || gc == 0 || gc == grid_size) {
                    r = 128; g = 128; b = 128; // Perimeter Corner
                } else {
                    // Check if any of the 4 connecting interior edges are defects
                    // so the white lines connect cleanly at the corners
                    bool defect_touching = false;
                    
                    // Edge Up
                    if (alphabet[grid[(gr - 1) * grid_size + gc - 1]].right != alphabet[grid[(gr - 1) * grid_size + gc]].left) defect_touching = true;
                    // Edge Down
                    if (alphabet[grid[gr * grid_size + gc - 1]].right != alphabet[grid[gr * grid_size + gc]].left) defect_touching = true;
                    // Edge Left
                    if (alphabet[grid[(gr - 1) * grid_size + gc - 1]].bottom != alphabet[grid[gr * grid_size + gc - 1]].top) defect_touching = true;
                    // Edge Right
                    if (alphabet[grid[(gr - 1) * grid_size + gc]].bottom != alphabet[grid[gr * grid_size + gc]].top) defect_touching = true;

                    if (defect_touching) {
                        r = 255; g = 255; b = 255;
                    }
                }
            }

            // Write pixel
            out << r << " " << g << " " << b << " ";
        }
        out << "\n";
    }

    out.close();
    std::cout << "[*] Exported defect visualization to " << filename << "\n";
}

void export_alien_tiles_to_ppm(
    const int* grid, 
    int grid_size, 
    const std::vector<int>& plane_A, 
    const std::vector<int>& plane_B, 
    const std::string& filename)
{
    std::ofstream out(filename);
    if (!out) {
        std::cerr << "Error: Could not open " << filename << " for writing.\n";
        return;
    }

    int img_size = 2 * grid_size + 1;
    out << "P3\n" << img_size << " " << img_size << "\n255\n";

    auto is_alien = [&](int r, int c) {
        if (r < 0 || r >= grid_size || c < 0 || c >= grid_size) return false;
        int idx = r * grid_size + c;
        return (grid[idx] != plane_A[idx] && grid[idx] != plane_B[idx]);
    };

    for (int ir = 0; ir < img_size; ir++) {
        for (int ic = 0; ic < img_size; ic++) {
            int r = 0, g = 0, b = 0; 

            // 1. Handle Perimeter First
            if (ir == 0 || ir == img_size - 1 || ic == 0 || ic == img_size - 1) {
                r = 128; g = 128; b = 128; 
            } else {
                // 2. Determine if we are on a Tile Center or a Boundary
                bool row_is_odd = (ir % 2 != 0);
                bool col_is_odd = (ic % 2 != 0);

                if (row_is_odd && col_is_odd) {
                    // TILE CENTER
                    if (is_alien(ir / 2, ic / 2)) {
                        r = 255; g = 255; b = 255;
                    }
                } 
                else if (!row_is_odd && col_is_odd) {
                    // HORIZONTAL EDGE (Between tile above and tile below)
                    int tile_above_r = (ir / 2) - 1;
                    int tile_below_r = (ir / 2);
                    int col = ic / 2;
                    if (is_alien(tile_above_r, col) && is_alien(tile_below_r, col)) {
                        r = 255; g = 255; b = 255;
                    }
                } 
                else if (row_is_odd && !col_is_odd) {
                    // VERTICAL EDGE (Between tile left and tile right)
                    int row = ir / 2;
                    int tile_left_c = (ic / 2) - 1;
                    int tile_right_c = (ic / 2);
                    if (is_alien(row, tile_left_c) && is_alien(row, tile_right_c)) {
                        r = 255; g = 255; b = 255;
                    }
                } 
                else {
                    // INTERSECTION (Corner point)
                    // Only fill if all 4 surrounding tiles are alien
                    int tr = (ir / 2) - 1;
                    int br = (ir / 2);
                    int lc = (ic / 2) - 1;
                    int rc = (ic / 2);
                    if (is_alien(tr, lc) && is_alien(tr, rc) && 
                        is_alien(br, lc) && is_alien(br, rc)) {
                        r = 255; g = 255; b = 255;
                    }
                }
            }
            out << r << " " << g << " " << b << " ";
        }
        out << "\n";
    }

    out.close();
    std::cout << "[*] Exported corrected alien visualization to " << filename << "\n";
}
