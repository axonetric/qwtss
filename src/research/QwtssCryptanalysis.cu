#include "QwtssCryptanalysis.h"
#include "QwtssCryptanalysisCore.h"
#include "QwtssCoreShared.h"
#include "JeandelRaoOracleGrid.h"
#include "Amman16OracleGrid.h"
#include "LabbeJR11Oracle.h"
#include "PrivateKeyDatabase.h"
#include "QwtssCore.h"
#include "CryptoUtils.h"
#include "FastPoseidonHashCpu.h"
#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <set>
#include <unordered_set>
#include <cstdint>
#include <functional>
#include <random>
#include <map>
#include <chrono>
#include <fstream>
#include <algorithm>
#include <utility>
#include <cuda_runtime.h>
#include <omp.h>


struct Point {
    double x, y;
};

// CUDA SLIDING WINDOW KERNEL
// This kernel sweeps a single edge of the public key across the entire Oracle ground
// truth looking for unbroken color sequences that exceed the 'min_match_length'.
__global__ void sliding_window_kernel(
    const int* oracle, int oracle_size,
    const uint64_t* boundary, int boundary_len,
    int min_match_length,
    BoundaryAnchor::Edge edge_type,
    BoundaryAnchor* d_results,
    int* d_result_count,
    int max_results,
    const Tile* alphabet
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_cells = oracle_size * oracle_size;
    
    if (idx >= total_cells) return;

    int r = idx / oracle_size;
    int c = idx % oracle_size;

    // True color extraction logic based on the edge we are currently scanning
    auto get_color = [&](int tile_id, BoundaryAnchor::Edge e) -> uint64_t {
        Tile t = alphabet[tile_id];
        if (e == BoundaryAnchor::NORTH) return t.top;
        if (e == BoundaryAnchor::SOUTH) return t.bottom;
        if (e == BoundaryAnchor::EAST)  return t.right;
        if (e == BoundaryAnchor::WEST)  return t.left;
        return 0;
    };

    // Check that the window fits within the boundary limits
    for (int pk_offset = 0; pk_offset <= boundary_len - min_match_length; ++pk_offset) {
        int match_len = 0;
        
        // Scan horizontally or vertically based on the edge type
        while (pk_offset + match_len < boundary_len) {
            int check_r = r + (edge_type == BoundaryAnchor::EAST || edge_type == BoundaryAnchor::WEST ? match_len : 0);
            int check_c = c + (edge_type == BoundaryAnchor::NORTH || edge_type == BoundaryAnchor::SOUTH ? match_len : 0);
            
            if (check_r >= oracle_size || check_c >= oracle_size) break;

            uint64_t oracle_color = get_color(oracle[check_r * oracle_size + check_c], edge_type);
            uint64_t pk_color = boundary[pk_offset + match_len];

            // Treat WILDCARD_COLOR as a wildcard that automatically matches any oracle color
            // This is for proper modeling of variable boundary edge color pinning (% boundary retention)
            if (pk_color == QwtssPublicInputs::WILDCARD_COLOR || oracle_color == pk_color) {
                match_len++;
            } else {
                break;
            }
        }

        // If we found a sequence long enough, record it atomically
        if (match_len >= min_match_length) {
            int res_idx = atomicAdd(d_result_count, 1);
            if (res_idx < max_results) {
                d_results[res_idx].oracle_x = c;
                d_results[res_idx].oracle_y = r;
                d_results[res_idx].pk_offset = pk_offset;
                d_results[res_idx].match_length = match_len;
                d_results[res_idx].edge = edge_type;
            }
        }
    }
}

// Host dispatcher for the CUDA kernel
std::vector<BoundaryAnchor> launch_cuda_anchor_search(
    const int* h_oracle, int oracle_size, 
    const FieldElement* h_boundary, int boundary_len, 
    int min_match_length, BoundaryAnchor::Edge edge_type,
    const std::vector<Tile>& alphabet
) {
    int* d_oracle;
    uint64_t* d_boundary;
    BoundaryAnchor* d_results;
    int* d_result_count;

    size_t oracle_bytes = oracle_size * oracle_size * sizeof(int);
    size_t boundary_bytes = boundary_len * sizeof(uint64_t);
    //int max_results = 10000;
    int max_results = 500000; // Allow half a million anchors per edge
    
    CUDA_CHECK(cudaMalloc(&d_oracle, oracle_bytes));
    CUDA_CHECK(cudaMalloc(&d_boundary, boundary_bytes));
    CUDA_CHECK(cudaMalloc(&d_results, max_results * sizeof(BoundaryAnchor)));
    CUDA_CHECK(cudaMalloc(&d_result_count, sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_oracle, h_oracle, oracle_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_boundary, h_boundary, boundary_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_result_count, 0, sizeof(int)));


    Tile* d_alphabet;
    size_t alphabet_bytes = alphabet.size() * sizeof(Tile);
    CUDA_CHECK(cudaMalloc(&d_alphabet, alphabet_bytes));
    CUDA_CHECK(cudaMemcpy(d_alphabet, alphabet.data(), alphabet_bytes, cudaMemcpyHostToDevice));

    int threads = 256;
    int blocks = (oracle_size * oracle_size + threads - 1) / threads;

    sliding_window_kernel<<<blocks, threads>>>(
        d_oracle, oracle_size, d_boundary, boundary_len, 
        min_match_length, edge_type, d_results, d_result_count, max_results,
        d_alphabet
    );

    int h_result_count = 0;
    CUDA_CHECK(cudaMemcpy(&h_result_count, d_result_count, sizeof(int), cudaMemcpyDeviceToHost));
    
    int fetch_count = std::min(h_result_count, max_results);
    std::vector<BoundaryAnchor> anchors(fetch_count);
    CUDA_CHECK(cudaMemcpy(anchors.data(), d_results, fetch_count * sizeof(BoundaryAnchor), cudaMemcpyDeviceToHost));

    cudaFree(d_oracle);
    cudaFree(d_boundary);
    cudaFree(d_results);
    cudaFree(d_result_count);
    cudaFree(d_alphabet);

    return anchors;
}

OracleEngine::OracleEngine(int target_size, const std::vector<Tile>& alphabet)
    : oracle_size(target_size), alphabet(alphabet), alphabet_size(alphabet.size()){
    oracle_grid.resize(oracle_size * oracle_size, -1);
}

void OracleEngine::generate_random_plane() {
    std::cout << "Generating Random Oracle (" << oracle_size << "x" << oracle_size << ")...\n";
    // Use a 2D non-linear hash to eliminate cyclic modulo aliasing
    for (int r = 0; r < oracle_size; r++) {
        for (int c = 0; c < oracle_size; c++) {
            unsigned int h = (c * 19349663U) ^ (r * 83492791U);
            oracle_grid[r * oracle_size + c] = h % alphabet_size;
        }
    }
}

void OracleEngine::generate_perfect_plane_jeandel_rao() {
    if (alphabet_size != 11) throw std::runtime_error("This must be called with the Jeandel-Rao 11-tile set.");

    std::cout << "Loading 1024x1024 Oracle Ground Truth from Sébastien Labbé's SageMath extraction...\n";
    
    if (oracle_size != JR11_ORACLE_SIZE) {
        throw std::runtime_error("Engine size does not match extracted JeandelRaoOracleGrid.h size.");
    }

    // Read the Python-exported grid. It contains the True Labbé topological adjacencies,
    // but the colors are scrambled by a permutation.
    std::vector<int> raw_grid(JR11_ORACLE_GROUND_TRUTH, JR11_ORACLE_GROUND_TRUTH + oracle_size * oracle_size);

    std::cout << "[*] Auto-deducing geometric orientation and true color permutation..." << std::endl;

    // Test all 4 geometric orientations because the mathematical Y-axis in
    // the SageMath package may be inverted compared to a C++ array's row-major layout.
    for (int flip_y = 0; flip_y <= 1; flip_y++) {
        for (int flip_x = 0; flip_x <= 1; flip_x++) {
            
            // Extract global unique adjacencies using a set to guarantee all tiles are evaluated
            std::set<std::pair<int, int>> h_rules_set;
            std::set<std::pair<int, int>> v_rules_set;
            
            auto get_val = [&](int r, int c) {
                int read_r = flip_y ? (oracle_size - 1 - r) : r;
                int read_c = flip_x ? (oracle_size - 1 - c) : c;
                return raw_grid[read_r * oracle_size + read_c];
            };

            for(int r = 0; r < oracle_size; r++) {
                for(int c = 0; c < oracle_size; c++) {
                    int current = get_val(r, c);
                    if (c > 0) h_rules_set.insert({get_val(r, c - 1), current});
                    if (r > 0) v_rules_set.insert({get_val(r - 1, c), current});
                }
            }

            std::vector<std::pair<int, int>> h_rules(h_rules_set.begin(), h_rules_set.end());
            std::vector<std::pair<int, int>> v_rules(v_rules_set.begin(), v_rules_set.end());

            std::vector<int> mapped_to_jr(11, -1);
            std::vector<bool> used_jr(11, false);

            // Backtracking solver
            std::function<bool(int)> solve_mapping = [&](int depth) -> bool {
                if (depth == 11) return true; // Successfully mapped all 11 tiles
                
                int symbol_id = depth;
                for (int jr_id = 0; jr_id < 11; jr_id++) {
                    if (used_jr[jr_id]) continue;
                    
                    mapped_to_jr[symbol_id] = jr_id;
                    used_jr[jr_id] = true;
                    bool valid = true;
                    
                    // Check Horizontal Adjacencies
                    for(auto& rule : h_rules) {
                        int L = rule.first, R = rule.second;
                        if (L <= symbol_id && R <= symbol_id) {
                            if (alphabet[mapped_to_jr[L]].right != alphabet[mapped_to_jr[R]].left) { valid = false; break; }
                        }
                    }
                    // Check Vertical Adjacencies
                    if (valid) {
                        for(auto& rule : v_rules) {
                            int T = rule.first, B = rule.second;
                            if (T <= symbol_id && B <= symbol_id) {
                                if (alphabet[mapped_to_jr[T]].bottom != alphabet[mapped_to_jr[B]].top) { valid = false; break; }
                            }
                        }
                    }
                    
                    if (valid && solve_mapping(depth + 1)) return true;
                    
                    // Prune and backtrack
                    used_jr[jr_id] = false;
                    mapped_to_jr[symbol_id] = -1;
                }
                return false;
            };

            // Backtracker survival guarantees the correct geometric orientation and permutation
            if (solve_mapping(0)) {
                std::cout << "[SUCCESS] Found valid geometry! (Flip Y: " << flip_y << ", Flip X: " << flip_x << ")\n";
                
                // Apply the exact transformation and true color mapping to the entire oracle
                for (int r = 0; r < oracle_size; r++) {
                    for (int c = 0; c < oracle_size; c++) {
                        oracle_grid[r * oracle_size + c] = mapped_to_jr[get_val(r, c)];
                    }
                }
                return; // Mapping complete
            }
        }
    }

    throw std::runtime_error("Auto-mapping failed for all 4 orientations! The extraction output is completely invalid.");
}

void OracleEngine::generate_perfect_plane_ammann16() {
    if (alphabet_size != 16) throw std::runtime_error("This must be called with the Ammann 16-tile set.");

    std::cout << "Loading 1024x1024 Oracle Ground Truth from Sébastien Labbé's SageMath extraction...\n";
    
    if (oracle_size != A16_ORACLE_SIZE) {
        throw std::runtime_error("Engine size does not match extracted Amman16OracleGrid.h size.");
    }

    // Read the Python-exported grid. It contains the True Labbé topological adjacencies,
    // but the colors are scrambled by a rotation and permutation.
    std::vector<int> raw_grid(A16_ORACLE_GROUND_TRUTH, A16_ORACLE_GROUND_TRUTH + oracle_size * oracle_size);

    // --- EXACT ISOMORPHISM MAPPING ---
    // Maps Labbé Discovery ID (0-15) -> MINE Tileset Index (0-15)
    // This perfectly encapsulates the underlying 6-color permutation and the 90-degree CW piece rotation.
    const int labbe_to_mine_map[16] = {
        10, 4, 9, 5, 2, 8, 1, 0, 13, 14, 12, 15, 3, 6, 11, 7
    };

    int best_transform = 4;

    /* No need to re-run this each time; we know transformation=4 is correct

    // We use the D4 Auto-Resolver to dynamically find the exact spatial grid transposition 
    // that matches the 90-degree CW piece rotation. 
    int best_transform = -1;
    
    // Test on a 16x16 window for instantaneous spatial resolution
    for (int t = 0; t < 8; t++) {
        int defects = 0;
        for (int r = 0; r < 15; r++) {
            for (int c = 0; c < 15; c++) {
                
                auto get_mapped_tile = [&](int rr, int cc) {
                    int labbe_r = rr, labbe_c = cc;
                    if (t == 1) { labbe_r = oracle_size - 1 - rr; } 
                    else if (t == 2) { labbe_c = oracle_size - 1 - cc; } 
                    else if (t == 3) { labbe_r = oracle_size - 1 - rr; labbe_c = oracle_size - 1 - cc; } 
                    else if (t == 4) { labbe_r = cc; labbe_c = rr; } // Transpose
                    else if (t == 5) { labbe_r = oracle_size - 1 - cc; labbe_c = rr; } // Rot 90 CW
                    else if (t == 6) { labbe_r = cc; labbe_c = oracle_size - 1 - rr; } // Rot 90 CCW
                    else if (t == 7) { labbe_r = oracle_size - 1 - cc; labbe_c = oracle_size - 1 - rr; } // Anti-Transpose

                    int idx = labbe_r * oracle_size + labbe_c;
                    return alphabet[labbe_to_mine_map[raw_grid[idx]]];
                };

                Tile current = get_mapped_tile(r, c);
                Tile right   = get_mapped_tile(r, c + 1);
                Tile bottom  = get_mapped_tile(r + 1, c);

                if (current.right != right.left) defects++;
                if (current.bottom != bottom.top) defects++;
            }
        }
        
        if (defects == 0) {
            best_transform = t;
            break;
        }
    }
    */

    if (best_transform == -1) {
        throw std::runtime_error("CRITICAL FAILURE: No spatial orientation resolved the grid.");
    }

    std::cout << "[*] Auto-Resolver locked onto Spatial Transformation: " << best_transform << "\n";

    // Apply the winning geometric transformation to the entire oracle
    for (int r = 0; r < oracle_size; r++) {
        for (int c = 0; c < oracle_size; c++) {
            int labbe_r = r, labbe_c = c;
            int t = best_transform;
            
            if (t == 1) { labbe_r = oracle_size - 1 - r; }
            else if (t == 2) { labbe_c = oracle_size - 1 - c; }
            else if (t == 3) { labbe_r = oracle_size - 1 - r; labbe_c = oracle_size - 1 - c; }
            else if (t == 4) { labbe_r = c; labbe_c = r; }
            else if (t == 5) { labbe_r = oracle_size - 1 - c; labbe_c = r; }
            else if (t == 6) { labbe_r = c; labbe_c = oracle_size - 1 - r; }
            else if (t == 7) { labbe_r = oracle_size - 1 - c; labbe_c = oracle_size - 1 - r; }

            int labbe_idx = labbe_r * oracle_size + labbe_c;
            int mine_idx = r * oracle_size + c;
            
            oracle_grid[mine_idx] = labbe_to_mine_map[raw_grid[labbe_idx]];
        }
    }
}

int OracleEngine::get_oracle_defects_count() {
    return count_grid_defects(oracle_grid.data(), oracle_size, alphabet);
}

std::vector<BoundaryAnchor> OracleEngine::find_anchors(const QwtssPublicInputs& pub_key, int min_match_length) {
    std::vector<BoundaryAnchor> all_anchors;

    // Launch CUDA search for all 4 edges of the public key boundary
    auto north = launch_cuda_anchor_search(oracle_grid.data(), oracle_size, pub_key.north.data(), 64, min_match_length, BoundaryAnchor::NORTH, alphabet);
    auto south = launch_cuda_anchor_search(oracle_grid.data(), oracle_size, pub_key.south.data(), 64, min_match_length, BoundaryAnchor::SOUTH, alphabet);
    auto east  = launch_cuda_anchor_search(oracle_grid.data(), oracle_size, pub_key.east.data(), 64, min_match_length, BoundaryAnchor::EAST, alphabet);
    auto west  = launch_cuda_anchor_search(oracle_grid.data(), oracle_size, pub_key.west.data(), 64, min_match_length, BoundaryAnchor::WEST, alphabet);

    all_anchors.insert(all_anchors.end(), north.begin(), north.end());
    all_anchors.insert(all_anchors.end(), south.begin(), south.end());
    all_anchors.insert(all_anchors.end(), east.begin(), east.end());
    all_anchors.insert(all_anchors.end(), west.begin(), west.end());

    return all_anchors;
}

std::vector<uint16_t> OracleEngine::project_and_intersect(const std::vector<BoundaryAnchor>& anchors) {
    if (alphabet.size() > 16) throw std::runtime_error("Does not support alphabet sizes > 16 yet (due to uint16_t).");

    // Because the alphabet tile count <= 16, we can use a single 16-bit integer (uint16_t)
    // as a Bitmask to track every prediction from every anchor simultaneously
    std::vector<uint16_t> predicted_grid(64 * 64, 0); // 0 means no predictions yet

    for (const auto& anchor : anchors) {
        int top_left_x = anchor.oracle_x;
        int top_left_y = anchor.oracle_y;

        if (anchor.edge == BoundaryAnchor::NORTH) {
            top_left_x -= anchor.pk_offset;
        } else if (anchor.edge == BoundaryAnchor::SOUTH) {
            top_left_x -= anchor.pk_offset;
            top_left_y -= 63;
        } else if (anchor.edge == BoundaryAnchor::WEST) {
            top_left_y -= anchor.pk_offset;
        } else if (anchor.edge == BoundaryAnchor::EAST) {
            top_left_y -= anchor.pk_offset;
            top_left_x -= 63;
        } 

        // Flood the interior prediction for this specific anchor
        for (int r = 0; r < 64; r++) {
            for (int c = 0; c < 64; c++) {
                int ox = top_left_x + c;
                int oy = top_left_y + r;
                if (ox >= 0 && ox < oracle_size && oy >= 0 && oy < oracle_size) {
                    int predicted_tile = oracle_grid[oy * oracle_size + ox];
                    // Flip the bit for this tile in the master grid!
                    predicted_grid[r * 64 + c] |= (1 << predicted_tile);
                }
            }
        }
    }
    return predicted_grid;
}

// Evaluate the Shannon Entropy. Checks the bitmask against the actual private key tile and calculates
// the exact mathematical bits of entropy removed from the brute-force search space.
// @param predicted_grid A uint16_t Bitmask that tracks every prediction from every anchor simultaneously
CryptanalysisMetrics OracleEngine::evaluate_vulnerability(
    const std::vector<uint16_t>& predicted_grid, 
    const std::vector<int>& true_private_grid
) {
    CryptanalysisMetrics metrics = {};
    int total_tiles = 64 * 64;
    
    // Total brute-force entropy of a 64x64 grid with N tiles
    double total_initial_entropy = total_tiles * std::log2((float)alphabet_size);
    double total_remaining_entropy = 0.0;

    for (int i = 0; i < total_tiles; i++) {
        uint16_t mask = predicted_grid[i];
        int true_tile = true_private_grid[i];
        
        if (mask == 0) {
            // No anchors reached this tile. Full N-tile entropy remains.
            total_remaining_entropy += std::log2((float)alphabet_size);
        } 
        else if ((mask & (1 << true_tile)) != 0) {
            // The true tile is present in the candidate set.
            // Count how many bits (predictions) are in the mask
            int candidates = 0;
            for(int b = 0; b < alphabet_size; b++) {
                if(mask & (1 << b)) candidates++;
            }
            
            total_remaining_entropy += std::log2((double)candidates);
            
            if (candidates == 1) {
                // The math predicted exactly 1 tile, and it was right.
                metrics.tiles_correctly_predicted++; 
            }
        } 
        else {
            // Anchor predictions missed the true tile.
            // The attacker's heuristic failed here, reverting to brute force.
            total_remaining_entropy += std::log2((float)alphabet_size);
        }
    }

    double bits_reduced = total_initial_entropy - total_remaining_entropy;
    metrics.entropy_reduction_bits = bits_reduced;
    metrics.entropy_reduction_percent = (bits_reduced / total_initial_entropy) * 100.0;
    metrics.remaining_entropy_bits = total_remaining_entropy;
    
    return metrics;
}

// Note: this reports the reduction in *marginal* entropy, which is very misleading. Only the joint entropy really matters.
CryptanalysisMetrics evaluate_private_key_vulnerability(
    OracleEngine& engine, 
    const PrivateKey &private_key, 
    int min_match_length, 
    bool do_logging = true
) {
    int grid_size = private_key.grid_size;
    int defect_count = private_key.defect_count;

    if (do_logging){
        std::cout << "\n=========================================================" << std::endl;
        std::cout << "   Evaluating Key (Defects: " << defect_count 
                << ", Min Match: " << min_match_length << ")" << std::endl;
        std::cout << "=========================================================\n" << std::endl;
    }

    QwtssPublicInputs pub_inputs("username", 0, 0, private_key.to_QwtssPrivateKey(), grid_size, ZERO_FINGERPRINT, engine.get_alphabet());

    // Launch CUDA Sliding Window Search against the perfect Oracle
    auto anchors = engine.find_anchors(pub_inputs, min_match_length);
    if (do_logging) std::cout << "[*] CUDA Engine found " << anchors.size() << " valid mathematical anchors." << std::endl;

    if (anchors.empty()) {
        if (do_logging){
            std::cout << "[SECURE] No anchors found. The defects successfully obfuscated the phase." << std::endl;
            std::cout << "Entropy Reduction: 0.0%\n" << std::endl;
        }
        return CryptanalysisMetrics{};
    }

    // =======================================================
    // STRATEGY A: UNFILTERED (Better for High Defects)
    // =======================================================

    // 3. Attacker propagates the interior using the geometry of the anchors
    std::vector<uint16_t> grid_unfiltered = engine.project_and_intersect(anchors);
    // 4. Measure Vulnerability
    CryptanalysisMetrics metrics_unfiltered = engine.evaluate_vulnerability(grid_unfiltered, private_key.grid_data);
    metrics_unfiltered.is_spatially_filtered = false;

    // =======================================================
    // STRATEGY B: SPATIALLY FILTERED "CORNER PINNING" (Better for Low Defects)
    // =======================================================
    std::map<std::pair<int, int>, uint8_t> origin_edge_mask;
    for (const auto& anchor : anchors) {
        int tx = anchor.oracle_x;
        int ty = anchor.oracle_y;
        if (anchor.edge == BoundaryAnchor::NORTH) { tx -= anchor.pk_offset; }
        else if (anchor.edge == BoundaryAnchor::SOUTH) { tx -= anchor.pk_offset; ty -= 63; }
        else if (anchor.edge == BoundaryAnchor::WEST) { ty -= anchor.pk_offset; }
        else if (anchor.edge == BoundaryAnchor::EAST) { ty -= anchor.pk_offset; tx -= 63; }
        
        // Use a bitmask to record WHICH edge voted (1=N, 2=S, 4=E, 8=W)
        origin_edge_mask[{tx, ty}] |= (1 << anchor.edge);
    }

    // Helper to count how many unique edges voted
    auto count_unique_edges = [](uint8_t mask) {
        int count = 0;
        while(mask) { count += mask & 1; mask >>= 1; }
        return count;
    };

    std::vector<BoundaryAnchor> filtered_anchors;
    for (const auto& anchor : anchors) {
        int tx = anchor.oracle_x;
        int ty = anchor.oracle_y;
        if (anchor.edge == BoundaryAnchor::NORTH) { tx -= anchor.pk_offset; }
        else if (anchor.edge == BoundaryAnchor::SOUTH) { tx -= anchor.pk_offset; ty -= 63; }
        else if (anchor.edge == BoundaryAnchor::WEST) { ty -= anchor.pk_offset; }
        else if (anchor.edge == BoundaryAnchor::EAST) { ty -= anchor.pk_offset; tx -= 63; }
        
        // Require at least 2 DISTINCT edges to geometrically intersect
        if (count_unique_edges(origin_edge_mask[{tx, ty}]) >= 2) { 
            filtered_anchors.push_back(anchor);
        }
    }

    CryptanalysisMetrics metrics_filtered = {};
    if (!filtered_anchors.empty()) {
        std::vector<uint16_t> grid_filtered = engine.project_and_intersect(filtered_anchors);
        metrics_filtered = engine.evaluate_vulnerability(grid_filtered, private_key.grid_data);
        metrics_filtered.is_spatially_filtered = true;
    }

    // =======================================================
    // TAKE THE MAXIMUM VULNERABILITY
    // =======================================================
    CryptanalysisMetrics best_local_metrics;
    if (metrics_filtered.entropy_reduction_percent > metrics_unfiltered.entropy_reduction_percent){
        best_local_metrics = metrics_filtered;
        best_local_metrics.anchors_found = filtered_anchors.size();
    } else {
        best_local_metrics = metrics_unfiltered;
        best_local_metrics.anchors_found = anchors.size();
    }
    best_local_metrics.true_defect_count = defect_count;
    best_local_metrics.min_match_length = min_match_length;

    if (do_logging){
        std::cout << "--- CRYPTANALYSIS RESULTS ---" << std::endl;
        std::cout << "Defect Count:             " << best_local_metrics.true_defect_count << std::endl;
        std::cout << "Min Match Length:         " << best_local_metrics.min_match_length << std::endl;
        std::cout << "Anchors Found:            " << best_local_metrics.anchors_found << std::endl;
        std::cout << "Tiles Correctly Guessed:  " << best_local_metrics.tiles_correctly_predicted << " / 4096" << std::endl;
        std::cout << "Entropy Reduction:        " << best_local_metrics.entropy_reduction_bits << " bits ("
                  << best_local_metrics.entropy_reduction_percent << "%)" << std::endl;
        std::cout << "Remaining Security Bits:  " << best_local_metrics.remaining_entropy_bits << " bits" << std::endl;
        std::cout << "Anchor Filtering Mode:    " << (best_local_metrics.is_spatially_filtered ? "Spatial" : "None") << std::endl;
        
        if (best_local_metrics.entropy_reduction_percent >= 99.0) {
            std::cout << "\n[CRITICAL] This key is completely vulnerable to phase recovery!" << std::endl;
        } else if (best_local_metrics.entropy_reduction_percent > 30.0) {
            std::cout << "\n[WARNING] Significant entropy loss detected. Heuristic partially successful." << std::endl;
        } else {
            std::cout << "\n[SECURE] Heuristic failure. Key heavily resists phase recovery." << std::endl;
        }
    }

    return best_local_metrics;
}

struct JointEntropyVulnerabilityResult {
    double base_joint_entropy;
    double constrained_joint_entropy;
    double joint_entropy_lost_bits;
    double marginal_entropy_lost_bits; // For comparison
    int anchors_found;
    bool attack_failed;
};

// Computes the Labbe oracle attack joint entropy reduction
JointEntropyVulnerabilityResult compute_joint_entropy_vulnerability(
    OracleEngine& engine,
    ITileSet* tileset,
    const PrivateKey& private_key,
    int defect_count_tolerance,
    int min_match_length,
    int num_ais_chains,
    ChaCha20PRNG& rng
) {
    int grid_size = private_key.grid_size;
    JointEntropyVulnerabilityResult result = {0};
    
    // ---------------------------------------------------------
    // 1. SETUP THE STRUCTURAL MASK (Physical Boundaries)
    // ---------------------------------------------------------
    std::vector<uint8_t> structural_mask = private_key.boundary_mask;
    if (structural_mask.empty()) {
        structural_mask.assign(grid_size * grid_size, 0); // Default to Free if empty
    }

    // ---------------------------------------------------------
    // 2. EXTRACT PUBLIC KEY & LAUNCH ORACLE ATTACK
    // ---------------------------------------------------------
    QwtssPublicInputs pub_inputs("username", 0, 0, private_key.to_QwtssPrivateKey(), grid_size, ZERO_FINGERPRINT, engine.get_alphabet());

    auto anchors = engine.find_anchors(pub_inputs, min_match_length);
    result.anchors_found = anchors.size();

    // The mask to feed into the constrained AIS (0 means no constraint)
    std::vector<uint16_t> best_attacker_mask(grid_size * grid_size, 0);

    if (anchors.empty()) {
        std::cout << "[*] Attack failed to find anchors with min match length: " << min_match_length << ". Joint entropy loss is 0." << std::endl;
        result.attack_failed = true;
        return result;
    } else {
        std::cout << "[*] Attack found " << anchors.size() << " anchors with min match length: " << min_match_length << "." << std::endl;

        // --- Replicate the Filtering Logic to get the Best Mask ---
        std::vector<uint16_t> grid_unfiltered = engine.project_and_intersect(anchors);
        CryptanalysisMetrics metrics_unfiltered = engine.evaluate_vulnerability(grid_unfiltered, private_key.grid_data);

        // Spatial filtering (Corner Pinning)
        std::map<std::pair<int, int>, uint8_t> origin_edge_mask;
        for (const auto& anchor : anchors) {
            int tx = anchor.oracle_x, ty = anchor.oracle_y;
            if (anchor.edge == BoundaryAnchor::NORTH) { tx -= anchor.pk_offset; }
            else if (anchor.edge == BoundaryAnchor::SOUTH) { tx -= anchor.pk_offset; ty -= (grid_size - 1); }
            else if (anchor.edge == BoundaryAnchor::WEST) { ty -= anchor.pk_offset; }
            else if (anchor.edge == BoundaryAnchor::EAST) { ty -= anchor.pk_offset; tx -= (grid_size - 1); }
            origin_edge_mask[{tx, ty}] |= (1 << anchor.edge);
        }

        std::vector<BoundaryAnchor> filtered_anchors;
        auto count_unique_edges = [](uint8_t mask) {
            int count = 0; while(mask) { count += mask & 1; mask >>= 1; } return count;
        };
        for (const auto& anchor : anchors) {
            int tx = anchor.oracle_x, ty = anchor.oracle_y;
            if (anchor.edge == BoundaryAnchor::NORTH) { tx -= anchor.pk_offset; }
            else if (anchor.edge == BoundaryAnchor::SOUTH) { tx -= anchor.pk_offset; ty -= (grid_size - 1); }
            else if (anchor.edge == BoundaryAnchor::WEST) { ty -= anchor.pk_offset; }
            else if (anchor.edge == BoundaryAnchor::EAST) { ty -= anchor.pk_offset; tx -= (grid_size - 1); }
            if (count_unique_edges(origin_edge_mask[{tx, ty}]) >= 2) { 
                filtered_anchors.push_back(anchor);
            }
        }

        std::vector<uint16_t> grid_filtered;
        CryptanalysisMetrics metrics_filtered = {};
        if (!filtered_anchors.empty()) {
            grid_filtered = engine.project_and_intersect(filtered_anchors);
            metrics_filtered = engine.evaluate_vulnerability(grid_filtered, private_key.grid_data);
        }

        // Choose the mask that provided the deepest marginal entropy reduction
        if (!filtered_anchors.empty() && metrics_filtered.entropy_reduction_percent > metrics_unfiltered.entropy_reduction_percent) {
            best_attacker_mask = grid_filtered;
            result.marginal_entropy_lost_bits = metrics_filtered.entropy_reduction_bits;
            std::cout << "[*] Using SPATIALLY FILTERED attacker mask.\n";
        } else {
            best_attacker_mask = grid_unfiltered;
            result.marginal_entropy_lost_bits = metrics_unfiltered.entropy_reduction_bits;
            std::cout << "[*] Using UNFILTERED attacker mask.\n";
        }
    }

    // ---------------------------------------------------------
    // 3. RUN BASE AIS (No Attacker Constraints)
    // ---------------------------------------------------------
    std::cout << "\n>>> Phase 1: Computing Base Joint Entropy (No Attack) <<<\n";
    std::vector<uint16_t> empty_attacker_mask(grid_size * grid_size, 0); 
    
    AISMetrics base_metrics = calculate_ais_joint_entropy_attacker_constrained(
        private_key.grid_data, private_key.grid_size, private_key.defect_count, defect_count_tolerance, num_ais_chains,
        tileset, rng, structural_mask, empty_attacker_mask
    );
    result.base_joint_entropy = base_metrics.joint_entropy;

    // ---------------------------------------------------------
    // 4. RUN CONSTRAINED AIS (With Attacker Mask)
    // ---------------------------------------------------------
    if (result.attack_failed) {
        result.constrained_joint_entropy = result.base_joint_entropy;
        result.joint_entropy_lost_bits = 0.0;
    } else {
        std::cout << "\n>>> Phase 2: Computing Attacker-Constrained Joint Entropy <<<\n";
        AISMetrics constrained_metrics = calculate_ais_joint_entropy_attacker_constrained(
            private_key.grid_data, private_key.grid_size, private_key.defect_count, defect_count_tolerance, num_ais_chains,
            tileset, rng, structural_mask, best_attacker_mask
        );
        
        // Handle case where attacker mask is mathematically contradictory to the true phase
        if (constrained_metrics.survived_chains == 0 || std::isnan(constrained_metrics.joint_entropy)) {
            std::cout << "[!] Attacker constraints overly restricted the phase space (0 valid chains). "
                      << "Heuristic failed dynamically.\n";
            result.constrained_joint_entropy = result.base_joint_entropy;
            result.joint_entropy_lost_bits = 0.0;
        } else {
            result.constrained_joint_entropy = constrained_metrics.joint_entropy;
            // The true thermodynamic vulnerability is the difference between the two
            result.joint_entropy_lost_bits = result.base_joint_entropy - result.constrained_joint_entropy;
        }
    }

    // ---------------------------------------------------------
    // 5. SUMMARY
    // ---------------------------------------------------------
    std::cout << "\n=========================================================\n";
    std::cout << "             JOINT ENTROPY VULNERABILITY REPORT          \n";
    std::cout << "=========================================================\n";
    std::cout << "Base Joint Entropy:          " << result.base_joint_entropy << " bits\n";
    std::cout << "Constrained Joint Entropy:   " << result.constrained_joint_entropy << " bits\n";
    std::cout << "---------------------------------------------------------\n";
    std::cout << "Marginal Bits Lost (Naive):  " << result.marginal_entropy_lost_bits << " bits\n";
    std::cout << "Joint Bits Lost (True):      " << result.joint_entropy_lost_bits << " bits\n";
    std::cout << "Remaining True Security:     " << result.base_joint_entropy - result.joint_entropy_lost_bits << " bits\n";
    std::cout << "=========================================================\n";

    return result;
}

// @param mode 1=marginal entropy (very misleading); 2=joint entropy
void run_private_key_vulnerability_analysis(int mode){
    //std::string filename = "../external/qwtss_standard_private_keys (481) 70% boundary - 100 to 600.db";
    //std::string filename = "../external/qwtss_private_keys (45) 70% boundary - 50 to 100.db";
    std::string filename = "../external/qwtss_private_keys (422) 100% boundary - 60 to 600 .db";

    PrivateKeyDatabase key_db;
    std::ifstream in(filename, std::ios::binary);
    if (!in) {
        std::cout << "Private key database file not found. Exiting...\n";
        return;
    } else {
        in.close();
        key_db.load_from_disk(filename);
        std::cout << "Loaded existing private key database: " << filename << ", Keys: " << key_db.get_total_key_count() << "\n";
        //key_db.print_summary();
        //return;
    }

    JeandelRaoTileSet jr_tileset;
    std::vector<Tile> alphabet = jr_tileset.get_tiles();

    // Load 1024x1024 oracle
    int oracle_size = 1024;
    OracleEngine engine(oracle_size, alphabet);
    engine.generate_perfect_plane_jeandel_rao();
    // Confirm the oracle ground truth grid is perfect
    int oracle_defects = engine.get_oracle_defects_count();
    if (oracle_defects > 0) throw std::runtime_error("Oracle ground truth grid has defects.");
    else std::cout << "Oracle ground truth grid defects: " << oracle_defects << "\n" << std::endl;

    ChaCha20PRNG rng;

    // Open CSV for writing
    std::ofstream csv_file("key_vulnerability_analysis (mode " + std::to_string(mode) + ").csv");
    if (mode == 1) {
        csv_file << "Defect_Count,Max_Entropy_Reduction_Percent,Entropy_Reduction_Bits,Optimal_Anchors_Found,Optimal_Min_Match_Len,Tiles_Guessed_Correct,Anchor_Filter_Mode,Remaining_Security_Bits\n";
    } else if (mode == 2){
        csv_file << "Defect_Count,Avg_Base_Joint_Entropy,Optimal_Constrained_Joint_Entropy,Optimal_Anchors_found,Optimal_Min_Match_Len,Optimal_Marginal_Entropy_Lost_Bits\n";
    } else throw std::invalid_argument("Unsupported 'mode' argument value.");

    int defect_count_tolerance = 5;
    bool do_logging = false;

    for (int target_defect_count = 500; target_defect_count <= 600; target_defect_count += 20) {
        auto key = key_db.get_nearest_key_by_defect_count(target_defect_count);
        if (key.grid_data.empty()) continue;
        if (std::abs(key.defect_count - target_defect_count) > defect_count_tolerance){
            throw std::runtime_error("Private key db did not contain a key within defect_count_tolerance of target " + std::to_string(target_defect_count));
        }

        double border_kept_pct = get_border_kept_pct(key.boundary_mask, key.grid_size);
        std::cout << "\n---------------------------------------------------------" << std::endl;
        std::cout << "Testing private key with " << key.defect_count << " defects (for target defect count: " << target_defect_count
                  << ") and " << std::fixed << std::setprecision(1) << border_kept_pct << "% border kept mask" << std::endl;

        if (mode == 1){
            // Marginal entropy analysis

            //int min_match_len = 32;
            CryptanalysisMetrics max_vulnerability{};
            for (int min_match_len = 40; min_match_len >= 6; --min_match_len){
                CryptanalysisMetrics metrics = evaluate_private_key_vulnerability(engine, key, min_match_len, do_logging);
                if (metrics.entropy_reduction_percent > max_vulnerability.entropy_reduction_percent){
                    max_vulnerability = metrics;
                }
            }

            {
                // Log max vulnerability results
                std::cout << "--- MAX VULNERABILITY CRYPTANALYSIS RESULTS ---" << std::endl;
                std::cout << "Defect Count:             " << max_vulnerability.true_defect_count << std::endl;
                std::cout << "Min Match Length:         " << max_vulnerability.min_match_length << std::endl;
                std::cout << "Anchors Found:            " << max_vulnerability.anchors_found << std::endl;
                std::cout << "Tiles Correctly Guessed:  " << max_vulnerability.tiles_correctly_predicted << " / 4096" << std::endl;
                std::cout << "Entropy Reduction:        " << max_vulnerability.entropy_reduction_bits << " bits ("
                        << max_vulnerability.entropy_reduction_percent << "%)" << std::endl;
                std::cout << "Remaining Security Bits:  " << max_vulnerability.remaining_entropy_bits << " bits" << std::endl;
                std::cout << "Anchor Filtering Mode:    " << (max_vulnerability.is_spatially_filtered ? "Spatial" : "None") << std::endl;
            }

            // Save max vulnerability results to .csv file
            csv_file << max_vulnerability.true_defect_count << "," 
                    << max_vulnerability.entropy_reduction_percent << "," 
                    << max_vulnerability.entropy_reduction_bits << "," 
                    << max_vulnerability.anchors_found << ","
                    << max_vulnerability.min_match_length << ","
                    << max_vulnerability.tiles_correctly_predicted << ","
                    << (max_vulnerability.is_spatially_filtered ? "Spatial" : "None") << ","
                    << max_vulnerability.remaining_entropy_bits << "\n";
            csv_file.flush();
        } else if (mode == 2){
            // Joint entropy analysis

            int num_ais_chains = 80;

            JointEntropyVulnerabilityResult max_vulnerability{};
            max_vulnerability.constrained_joint_entropy = 1e30;
            double base_joint_entropy_sum = 0.0;
            int base_joint_entropy_sum_cnt = 0;
            int optimal_min_match_len = 0;

            for (int min_match_len = 40; min_match_len >= 6; --min_match_len){
                JointEntropyVulnerabilityResult jevr = compute_joint_entropy_vulnerability(
                    engine, &jr_tileset,
                    key, defect_count_tolerance, 
                    min_match_len, num_ais_chains, rng);
                if (jevr.attack_failed || std::isnan(jevr.base_joint_entropy)) continue;

                base_joint_entropy_sum += jevr.base_joint_entropy;
                base_joint_entropy_sum_cnt++;
                if (jevr.constrained_joint_entropy < max_vulnerability.constrained_joint_entropy){
                    max_vulnerability = jevr;
                    optimal_min_match_len = min_match_len;
                }
            }

            // This is more accurate than any single AIS run
            double avg_base_joint_entropy = base_joint_entropy_sum / (double)base_joint_entropy_sum_cnt;

            // Write results to .csv file
            csv_file << key.defect_count << "," 
                    << std::fixed << std::setprecision(2) << avg_base_joint_entropy << "," 
                    << max_vulnerability.constrained_joint_entropy << "," 
                    << max_vulnerability.anchors_found << ","
                    << optimal_min_match_len << ","
                    //<< max_vulnerability.joint_entropy_lost_bits << "," // We can derive this using the more accurate 'avg_base_joint_entropy'
                    << max_vulnerability.marginal_entropy_lost_bits << "\n";
            csv_file.flush();
        }

        // Step through manually?
        //std::cin.get();
    }
    csv_file.close();
}

void run_marginal_entropy_analysis(int sample_rank, int grid_size) {
    //std::string filename = "../external/qwtss jr11 private keys (1146) no doping.db";
    //std::string filename = "../external/qwtss_a16_private_keys (556).db";
    std::string filename = "../external/jr11 qwtss_private_keys (538) (spliced).db";

    bool do_partial_boundary_sweep = true;

    bool do_greedy_prequench = true;

    JeandelRaoTileSet tileset;
    //Ammann16TileSet tileset;
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
    // key_db.print_summary();
    // return;

    ChaCha20PRNG rng;

    int defect_count_tolerance = 5;
    // This has to be high enough to overcome the negative bias of the maximum likelihood estimator for Shannon entropy
    int samples_required = 200;// 300; // 200;

    int total_points_sampled = 0;
    std::ofstream csv_file("marginal_entropy_curve.csv");
    if (do_partial_boundary_sweep)
        csv_file << "Defect_Count,Grids_Sampled,Marginal_Entropy_Bits,Max_Local_Entropy,Std_Dev_Local_Entropy,Fault_Line_Entropy,Banding_Intensity,Directional_Bias,Keep_Boundary_Pct\n";
    else
        csv_file << "Defect_Count,Grids_Sampled,Marginal_Entropy_Bits,Max_Local_Entropy,Std_Dev_Local_Entropy,Fault_Line_Entropy,Banding_Intensity,Directional_Bias\n";

    for (int defect_count_center = 306; defect_count_center >= 306; defect_count_center -= (defect_count_tolerance * 2))
    //for (int defect_count_center = 100; defect_count_center <= 300; defect_count_center += (defect_count_tolerance * 2))
    //for (int defect_count_center = 520; defect_count_center <= 610; defect_count_center += (defect_count_tolerance * 2))
    {
        PrivateKey key = key_db.get_nearest_key_by_defect_count(defect_count_center, sample_rank);
        if (abs(key.defect_count - defect_count_center) > 4)
            continue;
        if (key.grid_size != grid_size) throw std::runtime_error("(key.grid_size != grid_size)");

        std::cout << "\nEstimating marginal entropy for private key with " << key.defect_count
                  << " defects (for defect count center: " << defect_count_center << ")" << std::endl;

        if (do_partial_boundary_sweep){
            // Sweep from 100% down to 10% in increments of 10%
            for (int pct = 100; pct >= 0; pct -= 10) {
                float keep_rate = pct / 100.0f;

                std::cout << "\n\n>>> Testing boundary density: " << pct << "% <<<\n";

                // Generate the random partial mask for this density
                std::vector<uint8_t> partial_mask = generate_partial_boundary_mask(grid_size, keep_rate, rng);

                // Estimate the Marginal Entropy at this Defect Count & Partial Boundary %
                MarginalEntropyMetrics metrics = calculate_ensemble_marginal_entropy_metrics(key.grid_data, key.grid_size,
                    defect_count_center /* key.defect_count */,
                    defect_count_tolerance, samples_required, &tileset, rng, do_greedy_prequench, partial_mask);

                csv_file << defect_count_center << "," << samples_required << "," << metrics.entropy_bits
                        << "," << metrics.max_local_entropy << "," << metrics.std_dev_local_entropy
                        << "," << metrics.fault_line_entropy << "," << metrics.banding_intensity
                        << "," << metrics.directional_bias << "," << pct << "\n";
                csv_file.flush();

                std::cout << "Density: " << pct << "% | Kept Boundary Tiles: " << std::round(252 * keep_rate) << " / 252\n";
                std::cout << "Marginal Entropy: " << metrics.entropy_bits << " bits\n";
            }
        } else {
            // Use the key's pre-assigned boundary mask

            // Estimate the Marginal Entropy at this Defect Count
            MarginalEntropyMetrics metrics = calculate_ensemble_marginal_entropy_metrics(key.grid_data, key.grid_size,
                defect_count_center /* key.defect_count */,
                defect_count_tolerance, samples_required, &tileset, rng, do_greedy_prequench, key.boundary_mask);

            csv_file << defect_count_center << "," << samples_required << "," << metrics.entropy_bits
                    << "," << metrics.max_local_entropy << "," << metrics.std_dev_local_entropy
                    << "," << metrics.fault_line_entropy << "," << metrics.banding_intensity
                    << "," << metrics.directional_bias << "\n";
            csv_file.flush();
        }

        total_points_sampled++;
    }
    csv_file.close();

    std::cout << "\n[SUCCESS] Analysis complete. " << total_points_sampled << " total defect count points were sampled.\n";
}

void run_ais_joint_entropy_analysis(bool do_partial_boundary_sweep, int sample_rank, int grid_size) {
    bool has_extended_data = false;
    //std::string filename = "../external/qwtss_private_keys (422) 100% boundary - 60 to 600 .db";
    //std::string filename = "../external/a16 qwtss private keys (1075) --100 to 1000 (no doping).db";

    std::string filename = "../external/qwtss_private_keys (1836) 70% boundary  - 160 to 170 defects.db";
    //std::string filename = "../external/qwtss_standard_private_keys (481) 70% boundary - 100 to 600.db";

    // has_extended_data = true;
    // std::string filename = "../external/qwtss rebar_keys (40) - 160 to 170 defects (extended data format).db";


    JeandelRaoTileSet tileset;
    //Ammann16TileSet tileset;
    std::vector<Tile> alphabet = tileset.get_tiles();

    PrivateKeyDatabase key_db;
    std::ifstream in(filename, std::ios::binary);
    if (!in) {
        std::cout << "Private key database file not found. Exiting...\n";
        return;
    }
    in.close();
    key_db.load_from_disk(filename, has_extended_data);
    std::cout << "Loaded private key database. Total Keys: " << key_db.get_total_key_count() << "\n";
    //key_db.print_summary();
    //return;

    ChaCha20PRNG rng;

    int defect_count_tolerance = 5;
    int num_ais_chains = 500; // 80; // 40; // 50;

    int total_points_sampled = 0;
    std::ofstream csv_file("ais_joint_entropy_curve.csv");
    if (do_partial_boundary_sweep)
        csv_file << "Defect_Count,Joint_Entropy_Bits,Effective_Sample_Space,Marginal_Bound,Max_Weight,Valid_Chain_Count,Keep_Boundary_Pct\n";
    else
        csv_file << "Defect_Count,Joint_Entropy_Bits,Effective_Sample_Space,Marginal_Bound,Max_Weight,Valid_Chain_Count\n";

    // auto filtered = key_db.filter_by_name("rebar=66");
    // std::cout << filtered.size() << " filtered keys found" << std::endl;

    //for (int defect_count_center = 600; defect_count_center >= 100; defect_count_center -= (defect_count_tolerance * 2))
    for (int defect_count_center = 160; defect_count_center <= 170; defect_count_center += (defect_count_tolerance * 2))
    //for (int defect_count_center = 100; defect_count_center <= 600; defect_count_center += (defect_count_tolerance * 2))
    //for (int i = 0; i < 2; ++i)
    {
        // TMP: set tolerance to 5% for large sweeps
        //defect_count_tolerance = std::max(defect_count_center / 20, 5);

        PrivateKey key = key_db.get_nearest_key_by_defect_count(defect_count_center, sample_rank);
        if (abs(key.defect_count - defect_count_center) >= defect_count_tolerance)
            continue;   // Too far away to be an eligible border

        // TMP: to use the same key for each level
        //PrivateKey key = key_db.get_nearest_key_by_defect_count(100, sample_rank);

        // PrivateKey key = filtered[i];
        // int defect_count_center = key.defect_count;

        if (key.grid_size != grid_size) throw std::runtime_error("(key.grid_size != grid_size)");

        if (do_partial_boundary_sweep){
            std::cout << "\nEstimating AIS joint entropy for private key with " << key.defect_count
                      << " defects (for defect count center: " << defect_count_center << ")" << std::endl;
            if (!key.name.empty()) std::cout << "\tKey name: \"" << key.name << "\"" << std::endl;

            // Sweep from 100% down to 10% in increments of 10%
            for (int pct = 100; pct >= 0; pct -= 10) {
                float keep_rate = pct / 100.0f;

                std::cout << "\n\n>>> Testing boundary density: " << pct << "% <<<\n";

                // Generate the random partial mask for this density
                std::vector<uint8_t> partial_mask = generate_partial_boundary_mask(grid_size, keep_rate, rng, key.boundary_mask);

                // Estimate the AIS Joint Entropy at this Defect Count & Partial Boundary %
                AISMetrics metrics = calculate_ais_joint_entropy(key.grid_data, key.grid_size,
                    defect_count_center /* key.defect_count */,
                    defect_count_tolerance, num_ais_chains, &tileset, rng, partial_mask);

                csv_file << metrics.target_defects << "," << metrics.joint_entropy << "," << metrics.ess
                    << "," << metrics.marginal_bound << "," << metrics.max_weight << "," << metrics.survived_chains
                    << "," << pct << "\n";
                csv_file.flush();

                std::cout << "Density: " << pct << "% | Kept Boundary Tiles: " << std::round(252 * keep_rate) << " / 252\n";
                std::cout << "Joint Entropy: " << metrics.joint_entropy << " bits\n";
            }
        } else {
            // Use the key's pre-assigned boundary mask
            double border_kept_pct = get_border_kept_pct(key.boundary_mask, key.grid_size);
            int rebar_tile_count = get_pinned_tile_count(key.boundary_mask);
            std::cout << "\nEstimating AIS joint entropy for private key with " << key.defect_count
                      << " defects (for defect count center: " << defect_count_center << ") and "
                      << std::fixed << std::setprecision(1) << border_kept_pct << "% border kept mask" << std::endl;
            if (rebar_tile_count > 0) std::cout << "\tRebar tile count: " << rebar_tile_count << std::endl;
            if (!key.name.empty()) std::cout << "\tKey name: \"" << key.name << "\"" << std::endl;

            // Estimate the AIS Joint Entropy at this Defect Count
            AISMetrics metrics = calculate_ais_joint_entropy(key.grid_data, key.grid_size,
                defect_count_center /* key.defect_count */,
                defect_count_tolerance, num_ais_chains, &tileset, rng, key.boundary_mask);

            csv_file << metrics.target_defects << "," << metrics.joint_entropy << "," << metrics.ess
                    << "," << metrics.marginal_bound << "," << metrics.max_weight << "," << metrics.survived_chains << "\n";
            csv_file.flush();
        }

        total_points_sampled++;
    }
    csv_file.close();

    std::cout << "\n[SUCCESS] Analysis complete. " << total_points_sampled << " total defect count points were sampled.\n";
}

void run_3metric_topological_analysis(int sample_rank, int grid_size) {
    //std::string filename = "../external/qwtss_private_keys (422) 100% boundary - 60 to 600.db";
    std::string filename = "../external/qwtss_private_keys (1836) 70% boundary  - 160 to 170 defects.db";

    JeandelRaoTileSet tileset;
    //Ammann16TileSet tileset;
    std::vector<Tile> alphabet = tileset.get_tiles();

    bool do_partial_boundary_sweep = true;

    bool do_greedy_prequench = true;

    PrivateKeyDatabase key_db;
    std::ifstream in(filename, std::ios::binary);
    if (!in) {
        std::cout << "Private key database file not found. Exiting...\n";
        return;
    }
    in.close();
    key_db.load_from_disk(filename);
    std::cout << "Loaded private key database. Total Keys: " << key_db.get_total_key_count() << "\n";
    // key_db.print_summary();
    // return;

    ChaCha20PRNG rng;

    int defect_count_tolerance = 5;
    int samples_required = 200; // 300; // 200;

    int total_points_sampled = 0;

    std::ofstream csv_file("defect_topological_curve.csv");
    if (do_partial_boundary_sweep)
        csv_file << "Defect_Count,Grids_Sampled,GCC_Mean,GCC_StdDev,Core3_Mean,Core3_StdDev,Cyclomatic_Mean,Cyclomatic_StdDev,Millisec_Per_Key,Keep_Boundary_Pct\n";
    else
        csv_file << "Defect_Count,Grids_Sampled,GCC_Mean,GCC_StdDev,Core3_Mean,Core3_StdDev,Cyclomatic_Mean,Cyclomatic_StdDev,Millisec_Per_Key\n";

    for (int defect_count_center = 160; defect_count_center <= 170; defect_count_center += (defect_count_tolerance * 2))
    //for (int defect_count_center = 600; defect_count_center >= 100; defect_count_center -= (defect_count_tolerance * 2))
    {
        PrivateKey key = key_db.get_nearest_key_by_defect_count(defect_count_center, sample_rank);
        if (abs(key.defect_count - defect_count_center) >= defect_count_tolerance)
            continue;
        if (key.grid_size != grid_size) throw std::runtime_error("(key.grid_size != grid_size)");
        // if (key.boundary_mask.empty()) throw std::runtime_error("(key.boundary_mask.empty())");
        // auto [min_it, max_it] = std::minmax_element(key.boundary_mask.begin(), key.boundary_mask.end());
        // if (*min_it < 0 || *max_it > 1) throw std::runtime_error("key.boundary_mask contains values outside the expected range");

        std::cout << "\nEstimating defects topology for private key with " << key.defect_count
                  << " defects (for defect count center: " << defect_count_center << ")" << std::endl;

        if (do_partial_boundary_sweep){
            // Sweep from 100% down to 0% in increments of 10%
            for (int pct = 100; pct >= 0; pct -= 10) {
                float keep_rate = pct / 100.0f;

                std::cout << "\n\n>>> Testing boundary density: " << pct << "% <<<\n";

                // Generate the partial dropout mask for this density
                std::vector<uint8_t> partial_mask = generate_partial_boundary_mask(grid_size, keep_rate, rng);

                auto start = std::chrono::steady_clock::now();

                // Estimate the Topological Metrics at this Defect Count & Partial Boundary % using ensemble
                TopologyEnsembleTracker topo_results = calculate_ensemble_topological_metrics(key.grid_data, key.grid_size,
                    defect_count_center /* key.defect_count */,
                    defect_count_tolerance, samples_required, &tileset, rng, do_greedy_prequench, partial_mask);

                auto end = std::chrono::steady_clock::now();
                std::chrono::duration<double, std::milli> elapsed = end - start;
                double time_per_key_ms = elapsed.count() / (double)samples_required;

                csv_file << defect_count_center << ","
                        << samples_required << ","
                        << topo_results.gcc.get_mean() << "," 
                        << topo_results.gcc.get_std_dev() << ","
                        << topo_results.core3.get_mean() << "," 
                        << topo_results.core3.get_std_dev() << ","
                        << topo_results.cyclomatic.get_mean() << "," 
                        << topo_results.cyclomatic.get_std_dev() << "," 
                        << std::fixed << std::setprecision(2) << time_per_key_ms << "," 
                        << pct << "\n";
                csv_file.flush();

                std::cout << "Density: " << pct << "% | Kept Boundary Tiles: " << std::round(252 * keep_rate) << " / 252" << std::endl;
                std::cout << "\nAnalysis complete for defect count: " << defect_count_center << ", GCC: " << topo_results.gcc.get_mean()
                        << " (" << topo_results.gcc.get_std_dev() << "), Core3: " << topo_results.core3.get_mean()
                        << " (" << topo_results.core3.get_std_dev() << "), Cyclomatic: " << topo_results.cyclomatic.get_mean()
                        << " (" << topo_results.cyclomatic.get_std_dev() << "), Time Per Key: "
                        << std::fixed << std::setprecision(2) << time_per_key_ms << " ms" << std::endl;
            }
        } else {
            // Use the key's pre-assigned boundary mask

            auto start = std::chrono::steady_clock::now();

            // Estimate the Topological Metrics at this Defect Count using ensemble
            TopologyEnsembleTracker topo_results = calculate_ensemble_topological_metrics(key.grid_data, key.grid_size,
                defect_count_center /* key.defect_count */,
                defect_count_tolerance, samples_required, &tileset, rng, do_greedy_prequench, key.boundary_mask);

            auto end = std::chrono::steady_clock::now();
            std::chrono::duration<double, std::milli> elapsed = end - start;
            double time_per_key_ms = elapsed.count() / (double)samples_required;

            csv_file << defect_count_center << ","
                    << samples_required << ","
                    << topo_results.gcc.get_mean() << "," 
                    << topo_results.gcc.get_std_dev() << ","
                    << topo_results.core3.get_mean() << "," 
                    << topo_results.core3.get_std_dev() << ","
                    << topo_results.cyclomatic.get_mean() << "," 
                    << topo_results.cyclomatic.get_std_dev() << ","
                    << std::fixed << std::setprecision(2) << time_per_key_ms << "\n";
            csv_file.flush();

            std::cout << "\nAnalysis complete for defect count: " << defect_count_center << ", GCC: " << topo_results.gcc.get_mean()
                    << " (" << topo_results.gcc.get_std_dev() << "), Core3: " << topo_results.core3.get_mean()
                    << " (" << topo_results.core3.get_std_dev() << "), Cyclomatic: " << topo_results.cyclomatic.get_mean()
                    << " (" << topo_results.cyclomatic.get_std_dev() << "), Time Per Key: "
                    << std::fixed << std::setprecision(2) << time_per_key_ms << " ms" << std::endl;
        }

        total_points_sampled++;
    }
    csv_file.close();

    std::cout << "\n[SUCCESS] Analysis complete. " << total_points_sampled << " total defect count points were sampled.\n";
}

// Helper function to safely strip the file extension (e.g., "db1.sqlite" -> "db1")
std::string strip_extension(const std::string& filename) {
    size_t last_dot = filename.find_last_of(".");
    if (last_dot == std::string::npos) return filename;
    return filename.substr(0, last_dot);
}

// Extracts the nearest key from two databases and writes them to high-contrast PPM images
// @param mode 1=Show tiles, 2=Show defect edges
void export_comparable_db_keys_to_ppm(const std::string& db_file1, const std::string& db_file2, 
                              int target_defects, int mode, ITileSet* tileset) {
    if (mode < 1 || mode > 2) throw std::invalid_argument("Mode value is invalid.");

    std::cout << "\nInitiating Key Extraction (Target: " << target_defects << " defects)...\n";

    // Lambda to handle the extraction and export for a single database
    auto process_database = [&](const std::string& filename) {
        std::cout << "  -> Accessing " << filename << std::endl;

        PrivateKeyDatabase db;
        db.load_from_disk(filename);
        std::cout << "    -> " << db.get_total_key_count() << " total keys loaded" << std::endl;
        
        // Extract the nearest record. 
        auto record = db.get_nearest_key_by_defect_count(target_defects);

        // Dynamically deduce the grid size (e.g., 1048576 size -> 1024 grid_size)
        int grid_size = record.grid_size;

        // Construct the output filename: "original_name-[actual defect count] defects.ppm"
        std::string base_name = strip_extension(filename);

        // Export
        if (mode == 1) {
            std::string out_filename = base_name + "-" + std::to_string(record.defect_count) + " defects (grid).ppm";
            export_grid_to_ppm(record.grid_data.data(), grid_size, tileset, out_filename);
        } else {
            std::string out_filename = base_name + "-" + std::to_string(record.defect_count) + " defects (defect edges).ppm";
            export_defect_edges_to_ppm(record.grid_data.data(), grid_size, tileset, out_filename);
        }
    };

    // Process both files
    process_database(db_file1);
    process_database(db_file2);

    std::cout << "[SUCCESS] Both keys successfully exported to disk.\n";
}

// Extracts the keys from the database and writes them to high-contrast PPM images
// @param mode 1=Show tiles, 2=Show defect edges
void export_db_keys_to_ppm(const std::string& db_file, int target_defects, int max_count,
    int mode, ITileSet* tileset, bool has_extended_data) {
    if (mode < 1 || mode > 2) throw std::invalid_argument("Mode value is invalid.");

    std::cout << "\nInitiating Key Extraction (Target: " << target_defects << " defects)...\n";
    std::cout << "  -> Accessing " << db_file << std::endl;

    PrivateKeyDatabase db;
    db.load_from_disk(db_file, has_extended_data);
    int total_key_count = db.get_total_key_count();
    std::cout << "    -> " << total_key_count << " total keys loaded" << std::endl;

    auto filtered = db.filter_by_name("rebar=66");

    int exported_imgs = 0;
    for (int i = 0; i < max_count; ++i){
        // Extract the nearest record by rank
        //auto record = db.get_nearest_key_by_defect_count(target_defects, i+1);
        auto record = filtered[filtered.size() - (i+1)];

        // Dynamically deduce the grid size (e.g., 1048576 size -> 1024 grid_size)
        int grid_size = record.grid_size;
        if (grid_size < 1 || record.grid_data.empty()) continue;

        // Construct the output filename: "original_name-[actual defect count] defects.ppm"
        std::string base_name = strip_extension(db_file);

        // Export
        std::string name_details = (record.name.empty() ? "" : " (" + record.name + ")");
        std::string out_filename = base_name + " #" + std::to_string(i+1) + " - " + std::to_string(record.defect_count) + " defects" + name_details;
        if (mode == 1) {
            export_grid_to_ppm(record.grid_data.data(), grid_size, tileset, out_filename + " (grid).ppm");
        } else {
            export_defect_edges_to_ppm(record.grid_data.data(), grid_size, tileset, out_filename + " (defect edges).ppm");
        }
        exported_imgs++;

        if (!record.name.empty()) std::cout << "\tFor key name: \"" << record.name << "\"" << std::endl;
    }

    std::cout << "[SUCCESS] Exported " << exported_imgs << " key images to disk.\n";
}


void run_local_rigidity_sampling_test() {
    /*
--- Minimum Induced Defects Histogram (10000 Samples) ---
Best swap caused 0 defects: 0 times
Best swap caused 1 defects: 1050 times
Best swap caused 2 defects: 8950 times
Best swap caused 3 defects: 0 times
Best swap caused 4 defects: 0 times

Average minimum cost to perturb 1 tile: 1.895 defects.
    */
    std::cout << "\n=========================================================" << std::endl;
    std::cout << "   Local Rigidity & Single-Tile Swap Analysis (JR-11)" << std::endl;
    std::cout << "=========================================================\n" << std::endl;

    JeandelRaoTileSet tileset;
    std::vector<Tile> alphabet = tileset.get_tiles();
    int alphabet_size = alphabet.size();

    // 1. Initialize the Oracle and generate the 1024x1024 ground truth
    OracleEngine engine(1024, alphabet);
    engine.generate_perfect_plane_jeandel_rao();
    
    std::vector<int> grid = engine.get_oracle_grid();
    int grid_size = 1024; // The internal dimension generated by the oracle
    
    // 2. Setup the Monte Carlo sampler
    int num_samples = 10000;
    std::mt19937 rng(1337);
    
    // Sample strictly from the interior to guarantee all 4 neighbors exist
    std::uniform_int_distribution<int> dist(1, grid_size - 2);

    // Histogram to track how many times the *best* alternative tile caused N defects
    std::vector<int> histogram(5, 0); 

    for (int i = 0; i < num_samples; ++i) {
        // Pick a random interior coordinate
        int r = dist(rng);
        int c = dist(rng);

        int orig_id = grid[r * grid_size + c];
        
        // Fetch the perfectly matched neighbors
        Tile top_tile    = engine.get_alphabet_tile(grid[(r - 1) * grid_size + c]);
        Tile bottom_tile = engine.get_alphabet_tile(grid[(r + 1) * grid_size + c]);
        Tile left_tile   = engine.get_alphabet_tile(grid[r * grid_size + (c - 1)]);
        Tile right_tile  = engine.get_alphabet_tile(grid[r * grid_size + (c + 1)]);

        int min_induced_defects = 4; // Worst-case scenario bounds

        // Evaluate swapping out the original tile for every other tile in the alphabet
        for (int cand_id = 0; cand_id < alphabet_size; ++cand_id) {
            if (cand_id == orig_id) continue; // Skip the ground truth tile

            Tile cand_tile = engine.get_alphabet_tile(cand_id);
            int defects = 0;

            // Count mismatches against the frozen neighbors
            if (cand_tile.top != top_tile.bottom) defects++;
            if (cand_tile.bottom != bottom_tile.top) defects++;
            if (cand_tile.left != left_tile.right) defects++;
            if (cand_tile.right != right_tile.left) defects++;

            if (defects < min_induced_defects) {
                min_induced_defects = defects;
            }
        }

        histogram[min_induced_defects]++;
    }

    // 3. Print the results
    std::cout << "--- Minimum Induced Defects Histogram (" << num_samples << " Samples) ---\n";
    for (int i = 0; i < 5; ++i) {
        std::cout << "Best swap caused " << i << " defects: " << histogram[i] << " times\n";
    }
    
    // Calculate the expected minimum energy barrier
    double avg_min_defects = 0.0;
    for (int i = 0; i < 5; ++i) {
        avg_min_defects += (i * histogram[i]);
    }
    avg_min_defects /= num_samples;
    
    std::cout << "\nAverage minimum cost to perturb 1 tile: " << avg_min_defects << " defects.\n";
}


// SIMD-Optimized Hamming Distance
// Compares two 64-byte arrays. The compiler will auto-vectorize this loop.
inline int calculate_hamming_distance(const std::vector<uint8_t>& g1, const std::vector<uint8_t>& g2, int grid_area) {
    int distance = 0;
    for (int i = 0; i < grid_area; ++i) {
        if (g1[i] != g2[i]) {
            distance++;
        }
    }
    return distance;
}

void run_streaming_evt_analysis_deep(int grid_size) {
    const int GRID_AREA = grid_size * grid_size; // 4096 tiles if grid_size = 64
    // For the Extreme Value Theory (EVT) Weibull fit:
    // The Probe Depth
    constexpr int ENSEMBLE_SIZE =   100000;   // 100k
    // The Statistical Power
    constexpr int TOTAL_SAMPLES =   300000;   // 300k

    JeandelRaoTileSet tileset;
    auto alphabet = tileset.get_tiles();

    std::vector<std::vector<uint8_t>> reference_ensemble;
    reference_ensemble.reserve(ENSEMBLE_SIZE);

    std::unordered_set<uint64_t> unique_hashes;
    unique_hashes.reserve(TOTAL_SAMPLES + ENSEMBLE_SIZE);

    int min_defect_count = 160;
    int max_defect_count = 170;
    bool do_greedy_prequench = true;
    GridAnnealParams params(max_defect_count, min_defect_count);
    params.do_greedy_prequench = do_greedy_prequench;

    ChaCha20PRNG rng;

    size_t grid_count = GRID_AREA;
    int grid_bytes = grid_count * sizeof(int);

    // 1. Hoist GPU Allocations
    int *d_grid;
    CUDA_CHECK(cudaMalloc((void**)&d_grid, grid_bytes));

    // Allocate the Philox states (initialization occurs inside generate_boundary_conditioned_private_key_hoisted() )
    curandStatePhilox4_32_10_t* d_states;
    CUDA_CHECK(cudaMalloc(&d_states, GRID_AREA * sizeof(curandStatePhilox4_32_10_t)));

    // Init constant memory using proper function to prevent the "Shadow Constant Memory" bug.
    // We initialize both here in case we end up calling kernels from either module
    initialize_gpu_constants_from_crypto_core(&tileset);
    initialize_gpu_constants_from_core_shared(&tileset);

    std::cout << "Phase 1: Building Reference Ensemble..." << std::endl;

    // Generate the master key, of which all other candidates below are boundary-conditioned upon
    std::uniform_int_distribution<uint64_t> nonce_dist;
    uint64_t rnd_nonce = nonce_dist(rng);
    QwtssPrivateKey master_key = build_qwtss_private_key(generate_random_username(rng), rnd_nonce, 0,
        grid_size, min_defect_count, max_defect_count, 0.7f, do_greedy_prequench);
    int master_key_defects = count_grid_defects(master_key.private_key.data(), grid_size, alphabet);
    std::cout << "Master key generated with " << master_key_defects << " defects" << std::endl;

    while (reference_ensemble.size() < ENSEMBLE_SIZE) {
        std::vector<int> candidate = generate_boundary_conditioned_private_key_hoisted(master_key.private_key, master_key_defects,
            master_key.boundary_mask, grid_size, params, d_grid, d_states, &tileset, rng);
        if (candidate.empty()) {
            //std::cout << "Candidate key gen failed" << std::endl;
            continue;
        }

        std::vector<uint8_t> byte_private_key = downcast_private_key(candidate);
        uint64_t h = hash_grid_state(byte_private_key.data(), grid_size);

        // Only keep strictly unique grids
        if (unique_hashes.insert(h).second) {
            reference_ensemble.push_back(std::move(byte_private_key));

            if (reference_ensemble.size() % 20 == 0){
                std::cout << std::setw(10) << reference_ensemble.size() << " / " << ENSEMBLE_SIZE << "              \r" << std::flush;
                // Final sanity check to ensure no PK boundary constraints were violated
                confirm_mask_constraints_honored(candidate, master_key.boundary_mask, grid_size, master_key.private_key, alphabet, false);
            }
        } else {
            // Collision occurred
            std::cout << "Collision occurred during ensemble population" << std::endl;
        }
    }

    std::cout << "Phase 2: Streaming Analysis Started...                         " << std::endl;

    // --- Analytics & Threading Setup ---
    int global_min_distance = GRID_AREA;
    int global_max_distance = 0;
    long long total_comparisons = 0;
    int valid_samples_found = 0;
    long long total_collisions = 0;

    // These shouldn't overflow as long as we do < 1 trillion comparisons
    uint64_t global_sum = 0;
    uint64_t global_sq_sum = 0;

    // Calculate ~60% of available CPU cores
    int target_threads = std::max(1, omp_get_num_procs() * 6 / 10);
    std::cout << "Using " << target_threads << " threads for comparison loop." << std::endl;

    // Setup CSV File
    std::ofstream csv_file("streaming_evt_deep_" + std::to_string(min_defect_count) + "_to_" + std::to_string(max_defect_count) + ".csv");
    csv_file << "Samples,Total_Comparisons,Global_Min,Global_Max,Global_Mean,Global_StdDev,Collisions\n";

    auto start_time = std::chrono::high_resolution_clock::now();

    while (valid_samples_found < TOTAL_SAMPLES) {
        std::vector<int> candidate = generate_boundary_conditioned_private_key_hoisted(master_key.private_key, master_key_defects,
            master_key.boundary_mask, grid_size, params, d_grid, d_states, &tileset, rng);
        if (candidate.empty()) {
            //std::cout << "Candidate key gen failed" << std::endl;
            continue;
        }

        std::vector<uint8_t> byte_private_key = downcast_private_key(candidate);
        uint64_t h = hash_grid_state(byte_private_key.data(), grid_size);

        // Check uniqueness
        if (unique_hashes.insert(h).second) {
            long long local_distance_sum = 0;
            long long local_distance_sq_sum = 0; // Track sum of squares
            int local_min = GRID_AREA;
            int local_max = 0;
            
            // --- OpenMP Parallelized Compare Loop ---
            #pragma omp parallel for num_threads(target_threads) \
                reduction(+:local_distance_sum, local_distance_sq_sum) \
                reduction(min:local_min) reduction(max:local_max)
            for (int i = 0; i < ENSEMBLE_SIZE; ++i) {
                long long dist = calculate_hamming_distance(byte_private_key, reference_ensemble[i], GRID_AREA);
                local_distance_sum += dist;
                local_distance_sq_sum += (dist * dist);
                if (dist < local_min) local_min = dist;
                if (dist > local_max) local_max = dist;
            }

            // Update Global Metrics
            if (local_min < global_min_distance) global_min_distance = local_min;
            if (local_max > global_max_distance) global_max_distance = local_max;

            global_sum += local_distance_sum;
            global_sq_sum += local_distance_sq_sum;

            total_comparisons += ENSEMBLE_SIZE;
            valid_samples_found++;

            // Print and Log running statistics every 1,000 samples
            if (valid_samples_found % 1000 == 0) {
                // Standard Variance Formula: Var = (SumSq / N) - (Mean^2)
                double current_mean = (double)global_sum / total_comparisons;
                double mean_sq = current_mean * current_mean;
                double current_variance = ((double)global_sq_sum / total_comparisons) - mean_sq;
                double current_stddev = std::sqrt(current_variance);

                std::cout << "Samples: " << std::setw(7) << valid_samples_found 
                          << " | Collisions: " << total_collisions
                          << " | Min: " << global_min_distance
                          << " | Max: " << global_max_distance
                          << " | Avg: " << std::fixed << std::setprecision(4) << current_mean 
                          << " | StdDev: " << std::fixed << std::setprecision(4) << current_stddev
                          << " / " << GRID_AREA << " tiles" << std::endl;

                // Flush to CSV to ensure data isn't lost
                csv_file << valid_samples_found << "," 
                         << total_comparisons << "," 
                         << global_min_distance << "," 
                         << global_max_distance << "," 
                         << current_mean << "," 
                         << current_stddev << ","
                         << total_collisions << "\n";
                csv_file.flush();
            }

            // Confirm we're still generating valid keys conditioned on the PK boundary constraints
            if (valid_samples_found % 200 == 0){
                confirm_mask_constraints_honored(candidate, master_key.boundary_mask, grid_size, master_key.private_key, alphabet, false);
            }

        } else {
            // Collision occurred: The hash was already in the set
            total_collisions++; 
        }
    }

    csv_file.close();

    CUDA_CHECK(cudaFree(d_grid));
    CUDA_CHECK(cudaFree(d_states));

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;

    std::cout << "\nAnalysis Complete in " << elapsed.count() << " seconds." << std::endl;
    std::cout << "Total Comparisons: " << total_comparisons << std::endl;
    std::cout << "Total Collisions: " << total_collisions << std::endl;
}

void run_streaming_evt_analysis_scout(int grid_size, int num_trajectories) {
    const int GRID_AREA = grid_size * grid_size; 

    int ENSEMBLE_SIZE = 0, TOTAL_SAMPLES = 0;
    int log_interval = 100; // Log more frequently for shorter scout runs
    if (!true) {
        // Scout shallow depths
        std::cout << "Scouting shallow depths ..." << std::endl;
        ENSEMBLE_SIZE = 1000;   // 1k reference grids
        TOTAL_SAMPLES = 5000;   // 5k streaming samples
    } else if (!true) {
        // Scout Intermediate A depths
        std::cout << "Scouting Intermediate A depths ..." << std::endl;
        ENSEMBLE_SIZE = 3000;   // 3k reference grids
        TOTAL_SAMPLES = 15000;  // 15k streaming samples
    } else if (!true) {
        // Scout Intermediate B depths
        std::cout << "Scouting Intermediate B depths ..." << std::endl;
        ENSEMBLE_SIZE = 10000;  // 10k reference grids
        TOTAL_SAMPLES = 40000;  // 40k streaming samples
        log_interval = 500;
    } else {
        // Scout Intermediate C depths
        std::cout << "Scouting Intermediate C depths ..." << std::endl;
        ENSEMBLE_SIZE =  30000; // 30k reference grids
        TOTAL_SAMPLES = 115000; // 115k streaming samples
        log_interval = 1000;
    }

    JeandelRaoTileSet tileset;
    auto alphabet = tileset.get_tiles();

    int min_defect_count = 160;
    int max_defect_count = 170;
    bool do_greedy_prequench = true;
    GridAnnealParams params(max_defect_count, min_defect_count);
    params.do_greedy_prequench = do_greedy_prequench;

    ChaCha20PRNG rng;

    // 1. Hoist GPU Allocations ONCE for all trajectories
    int grid_bytes = GRID_AREA * sizeof(int);
    int *d_grid;
    CUDA_CHECK(cudaMalloc((void**)&d_grid, grid_bytes));
    curandStatePhilox4_32_10_t* d_states;
    CUDA_CHECK(cudaMalloc(&d_states, GRID_AREA * sizeof(curandStatePhilox4_32_10_t)));

    initialize_gpu_constants_from_crypto_core(&tileset);
    initialize_gpu_constants_from_core_shared(&tileset);

    int target_threads = std::max(1, omp_get_num_procs() * 6 / 10);
    std::cout << "Using " << target_threads << " threads for comparison loop." << std::endl;

    // Setup CSV File ONCE, adding the Trajectory_ID column (Open in Append Mode)
    std::ofstream csv_file("streaming_evt_scout_" + std::to_string(min_defect_count) + "_to_" + std::to_string(max_defect_count) + ".csv", std::ios::app);
    if (csv_file.tellp() == 0) {
        csv_file << "Trajectory_ID,Samples,Total_Comparisons,Global_Min,Global_Max,Global_Mean,Global_StdDev,Collisions\n";
        std::cout << "Created new .csv file." << std::endl;
    } else {
        std::cout << "Will append to existing .csv file." << std::endl;
    }

    auto start_time = std::chrono::high_resolution_clock::now();

    // ---------------------------------------------------------
    // TRAJECTORY LOOP
    // ---------------------------------------------------------
    for (int traj = 0; traj < num_trajectories; ++traj) {
        std::cout << "\n\n=========================================================\n";
        std::cout << " Starting Scout Trajectory " << traj + 1 << " / " << num_trajectories << "\n";
        std::cout << "=========================================================\n";

        // By declaring these inside the loop, memory is automatically freed and reset per trajectory
        std::vector<std::vector<uint8_t>> reference_ensemble;
        reference_ensemble.reserve(ENSEMBLE_SIZE);

        std::unordered_set<uint64_t> unique_hashes;
        unique_hashes.reserve(TOTAL_SAMPLES + ENSEMBLE_SIZE);

        // Generate the master key for THIS specific trajectory
        std::uniform_int_distribution<uint64_t> nonce_dist;
        uint64_t rnd_nonce = nonce_dist(rng);
        QwtssPrivateKey master_key = build_qwtss_private_key(generate_random_username(rng), rnd_nonce, 0,
            grid_size, min_defect_count, max_defect_count, 0.7f, do_greedy_prequench);
        int master_key_defects = count_grid_defects(master_key.private_key.data(), grid_size, alphabet);
        std::cout << "Master key generated with " << master_key_defects << " defects" << std::endl;

        // --- Populate Ensemble ---
        while (reference_ensemble.size() < ENSEMBLE_SIZE) {
            std::vector<int> candidate = generate_boundary_conditioned_private_key_hoisted(master_key.private_key, master_key_defects,
                master_key.boundary_mask, grid_size, params, d_grid, d_states, &tileset, rng);
            if (candidate.empty()) continue;

            std::vector<uint8_t> byte_private_key = downcast_private_key(candidate);
            uint64_t h = hash_grid_state(byte_private_key.data(), grid_size);

            if (unique_hashes.insert(h).second) {
                reference_ensemble.push_back(std::move(byte_private_key));
                if (reference_ensemble.size() % 10 == 0){
                    std::cout << std::setw(10) << reference_ensemble.size() << " / " << ENSEMBLE_SIZE << "              \r" << std::flush;
                }
            }
        }

        std::cout << "Phase 2: Streaming Analysis Started...                         " << std::endl;

        // Analytics Setup for this Trajectory
        int global_min_distance = GRID_AREA;
        int global_max_distance = 0;
        long long total_comparisons = 0;
        int valid_samples_found = 0;
        long long total_collisions = 0;
        uint64_t global_sum = 0;
        uint64_t global_sq_sum = 0;

        // --- Stream Samples ---
        while (valid_samples_found < TOTAL_SAMPLES) {
            std::vector<int> candidate = generate_boundary_conditioned_private_key_hoisted(master_key.private_key, master_key_defects,
                master_key.boundary_mask, grid_size, params, d_grid, d_states, &tileset, rng);
            if (candidate.empty()) continue;

            std::vector<uint8_t> byte_private_key = downcast_private_key(candidate);
            uint64_t h = hash_grid_state(byte_private_key.data(), grid_size);

            if (unique_hashes.insert(h).second) {
                long long local_distance_sum = 0;
                long long local_distance_sq_sum = 0; 
                int local_min = GRID_AREA;
                int local_max = 0;
                
                #pragma omp parallel for num_threads(target_threads) \
                    reduction(+:local_distance_sum, local_distance_sq_sum) \
                    reduction(min:local_min) reduction(max:local_max)
                for (int i = 0; i < ENSEMBLE_SIZE; ++i) {
                    long long dist = calculate_hamming_distance(byte_private_key, reference_ensemble[i], GRID_AREA);
                    local_distance_sum += dist;
                    local_distance_sq_sum += (dist * dist);
                    if (dist < local_min) local_min = dist;
                    if (dist > local_max) local_max = dist;
                }

                if (local_min < global_min_distance) global_min_distance = local_min;
                if (local_max > global_max_distance) global_max_distance = local_max;

                global_sum += local_distance_sum;
                global_sq_sum += local_distance_sq_sum;
                total_comparisons += ENSEMBLE_SIZE;
                valid_samples_found++;

                // Check for loggin event
                if (valid_samples_found % log_interval == 0) {
                    double current_mean = (double)global_sum / total_comparisons;
                    double mean_sq = current_mean * current_mean;
                    double current_variance = ((double)global_sq_sum / total_comparisons) - mean_sq;
                    double current_stddev = std::sqrt(current_variance);

                    // Write to CSV with Trajectory_ID prepended
                    csv_file << traj << "," 
                             << valid_samples_found << "," 
                             << total_comparisons << "," 
                             << global_min_distance << "," 
                             << global_max_distance << "," 
                             << current_mean << "," 
                             << current_stddev << ","
                             << total_collisions << "\n";

                    std::cout << std::setw(10) << valid_samples_found << " / " << TOTAL_SAMPLES << "              \r" << std::flush;
                }

                // Sanity check
                if (valid_samples_found % 200 == 0){
                    confirm_mask_constraints_honored(candidate, master_key.boundary_mask, grid_size, master_key.private_key, alphabet, false);
                }

            } else {
                total_collisions++; 
            }
        }
        
        // Force flush to disk at the end of each trajectory
        csv_file.flush();
    }

    csv_file.close();
    CUDA_CHECK(cudaFree(d_grid));
    CUDA_CHECK(cudaFree(d_states));

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;

    std::cout << "\n=========================================================\n";
    std::cout << "Scout Analysis Complete in " << elapsed.count() << " seconds." << std::endl;
}

// Helper function to calculate defects for a single tile against its 4 neighbors
inline int get_local_tile_defects(const std::vector<int>& grid, int idx, int tile_id, const std::vector<Tile>& alphabet, int grid_size) {
    int r = idx / grid_size;
    int c = idx % grid_size;
    int defects = 0;
    const Tile& t = alphabet[tile_id];

    if (r > 0) {
        const Tile& n = alphabet[grid[idx - grid_size]];
        if (t.top != n.bottom) defects++;
    }
    if (r < grid_size - 1) {
        const Tile& s = alphabet[grid[idx + grid_size]];
        if (t.bottom != s.top) defects++;
    }
    if (c > 0) {
        const Tile& w = alphabet[grid[idx - 1]];
        if (t.left != w.right) defects++;
    }
    if (c < grid_size - 1) {
        const Tile& e = alphabet[grid[idx + 1]];
        if (t.right != e.left) defects++;
    }
    return defects;
}

// Executes the Greedy Walk and returns the step-by-step defect trajectory
// @param mode 0=steepest descent, 1=flat-path mode, 2=radial, 3=stochastic
std::vector<int> execute_greedy_walk(const std::vector<int>& grid_X, const std::vector<int>& grid_Y, const std::vector<Tile>& alphabet,
    int grid_size, int mode = 0, ChaCha20PRNG* rng = nullptr, int top_k_pool = 5) {
    std::vector<int> current_grid = grid_X;
    std::vector<int> diff_indices;
    
    // Find all tiles that differ between X and Y
    for (int i = 0; i < grid_size * grid_size; ++i) {
        if (current_grid[i] != grid_Y[i]) {
            diff_indices.push_back(i);
        }
    }

    int current_defects = count_grid_defects(current_grid.data(), grid_size, alphabet);
    int start_defects = current_defects;

    std::vector<int> trajectory;
    trajectory.reserve(diff_indices.size() + 1);
    trajectory.push_back(current_defects);

    std::vector<std::pair<int, int>> evaluated_moves;
    if (mode == 3) evaluated_moves.reserve(diff_indices.size());

    // Greedily resolve differences one by one
    while (!diff_indices.empty()) {
        int best_score = 999999;
        int best_delta = 0;
        int best_diff_idx_pos = -1;

        if (mode == 3) evaluated_moves.clear(); // Clear it each step

        // Scan all remaining differences to find the "cheapest" flip
        for (size_t p = 0; p < diff_indices.size(); ++p) {
            int idx = diff_indices[p];

            // Calculate how the defect count changes if we swap this specific tile to match Y
            int old_local_d = get_local_tile_defects(current_grid, idx, current_grid[idx], alphabet, grid_size);
            int new_local_d = get_local_tile_defects(current_grid, idx, grid_Y[idx], alphabet, grid_size);
            int delta = new_local_d - old_local_d;

            // Mode Selection Logic:
            // Mode 0 uses steepest descent. Modes 1, 2, and 3 use the flat-path absolute deviation score.
            int score = (mode != 0) ? std::abs((current_defects + delta) - start_defects) : delta;

            if (mode == 3) {
                evaluated_moves.push_back({score, static_cast<int>(p)});
            } else if (score < best_score) {
                best_score = score;
                best_delta = delta;     // Save the actual delta for applying later
                best_diff_idx_pos = p;
            }
        }

        // Mode 3 Stochastic Selection
        if (mode == 3) {
            std::sort(evaluated_moves.begin(), evaluated_moves.end());
            int pool_size = std::min(top_k_pool, static_cast<int>(evaluated_moves.size()));
            std::uniform_int_distribution<int> dist(0, pool_size - 1);
            best_diff_idx_pos = evaluated_moves[dist(*rng)].second;
            
            // Recalculate delta for the randomly selected move
            int idx = diff_indices[best_diff_idx_pos];
            best_delta = get_local_tile_defects(current_grid, idx, grid_Y[idx], alphabet, grid_size) - 
                         get_local_tile_defects(current_grid, idx, current_grid[idx], alphabet, grid_size);
        }

        // Apply the best flip
        int best_idx = diff_indices[best_diff_idx_pos];
        current_grid[best_idx] = grid_Y[best_idx];
        current_defects += best_delta;
        
        trajectory.push_back(current_defects);

        // Fast removal from the un-matched list
        diff_indices[best_diff_idx_pos] = diff_indices.back();
        diff_indices.pop_back();
    }

    return trajectory;
}

// Executes the Energy Barrier Analysis (and associated visualizations) and returns the step-by-step defect trajectory
// @param mode 0=steepest descent, 1=flat-path mode, 2=radial, 3=stochastic
void run_energy_barrier_analysis(int mode, int grid_size) {
    /*
Mode 2: The Radial Map (1-to-N Topology)
-Data Generation: Instead of testing 100 completely different phase spaces, Mode 2 locks onto a single Master Key (Grid A) as the absolute center of the universe. The algorithm then finds 100 different valid candidate keys (Grid Bs) located on the perimeter of that local phase space. It executes 100 independent flat-path greedy walks radiating outward from Grid A to each of the 100 targets.
-Visualization: This generates a 3D Topological Surface Map. Grid A sits at the origin center (r=0), and the 100 paths fan out radially like the spokes of a wheel. The height (Z-axis) represents the energy (defect count).

Mode 3: The Stochastic Heatmap (1-to-1)
-Data Generation: Mode 3 maps the specific "corridor" between one exact pair (Grid A and Grid B). Instead of taking the single absolute best path, the solver runs 1,000 independent times between the two grids. At every single step, it evaluates all possible moves, ranks them using the flat-path logic, and randomly selects a move from the Top 5. This forces the solver to explore the width and fragility of the connecting paths, rather than just the single mathematically optimal "tightrope".
-Visualization: This generates a 2D Path Density Heatmap. The X-axis is the transition progress, the Y-axis is the defect count, and the color intensity (e.g., glowing red vs. dark blue) shows how many of the 1,000 paths passed through that exact coordinate.
    */
    int PAIR_COUNT = (mode >= 2) ? 1 : 100;     // Radial/Heatmap only need 1 central phase space
    int CANDIDATES_COUNT = 1000;                // Ensemble of 1000 keys to find the nearest pairwise distance
    int PATHS_PER_PAIR = (mode == 3) ? 1000 : ((mode == 2) ? 100 : 1);

    int min_defect_count = 160;
    int max_defect_count = 170;

    JeandelRaoTileSet tileset;
    std::vector<Tile> alphabet = tileset.get_tiles();

    std::string mode_details;
    if (mode == 0) mode_details = " (mode 0=steepest descent)";
    else if (mode == 1) mode_details = " (mode 1=flat-path mode)";
    else if (mode == 2) mode_details = " (mode 2=radial 1-to-N)";
    else if (mode == 3) mode_details = " (mode 3=stochastic heatmap)";

    std::ofstream csv_file("energy_barrier_trajectories" + mode_details + ".csv");
    // Normalized_Step will go from 0.0 to 1.0 so Python can easily interpolate and average them
    csv_file << "Pair_ID,Step_Index,Normalized_Step,Defect_Count\n";

    std::ofstream stats_file("energy_barrier_stats" + mode_details + ".csv");
    stats_file << "Pair_ID,Hamming_Distance,Start_Defects,Peak_Defects,Min_Defects,Activation_Energy,True_Climb,Peak_Normalized_Step\n";

    std::cout << "Starting Energy Barrier Analysis (" << PAIR_COUNT << " pairs) " << mode_details << "...\n";

    ChaCha20PRNG rng;

    size_t grid_count = grid_size * grid_size;
    int grid_bytes = grid_count * sizeof(int);

    // 1. Hoist GPU Allocations
    int *d_grid;
    CUDA_CHECK(cudaMalloc((void**)&d_grid, grid_bytes));

    // Allocate the Philox states (initialization occurs inside generate_boundary_conditioned_private_key_hoisted() )
    curandStatePhilox4_32_10_t* d_states;
    CUDA_CHECK(cudaMalloc(&d_states, grid_count * sizeof(curandStatePhilox4_32_10_t)));

    // Init constant memory using proper function to prevent the "Shadow Constant Memory" bug.
    // We initialize both here in case we end up calling kernels from either module
    initialize_gpu_constants_from_crypto_core(&tileset);
    initialize_gpu_constants_from_core_shared(&tileset);

    std::uniform_int_distribution<uint64_t> nonce_dist;

    GridAnnealParams params(max_defect_count, min_defect_count);

    for (int pair_id = 0; pair_id < PAIR_COUNT; ++pair_id) {
        // 1. Generate a Master Key (to fix the boundary)
        uint64_t rnd_nonce = nonce_dist(rng);
        QwtssPrivateKey master_key = build_qwtss_private_key(generate_random_username(rng), rnd_nonce, 0,
            grid_size, min_defect_count, max_defect_count, 0.7f, true);
        int master_key_defects = count_grid_defects(master_key.private_key.data(), grid_size, alphabet);
        if (master_key_defects >= min_defect_count && master_key_defects <= max_defect_count){
            std::cout << "\n0. Master key generated with " << master_key_defects << " defects" << std::endl;
        } else {
            std::cout << "Error: Master key generation failed; retrying..." << std::endl;
            pair_id--;
            continue;
        }

        std::vector<std::vector<int>> candidates;
        candidates.reserve(CANDIDATES_COUNT);

        std::unordered_set<uint64_t> unique_hashes;
        unique_hashes.reserve(CANDIDATES_COUNT + 1);
        // Add the master key hash first
        std::vector<uint8_t> byte_private_key = downcast_private_key(master_key.private_key);
        uint64_t h = hash_grid_state(byte_private_key.data(), grid_size);
        unique_hashes.insert(h);

        std::cout << "1. Generating candidate ensemble for pair id " << pair_id << std::endl;

        if (mode == 2 || mode == 3){
            // A tighter tolerance for these visualization modes
            params.min_allowed_defects = std::max(min_defect_count, master_key_defects - 2);
            params.max_allowed_defects = std::min(max_defect_count, master_key_defects + 2);
        }

        while (candidates.size() < CANDIDATES_COUNT) {
            std::vector<int> candidate = generate_boundary_conditioned_private_key_hoisted(master_key.private_key, master_key_defects,
                master_key.boundary_mask, grid_size, params, d_grid, d_states, &tileset, rng);
            if (candidate.empty()) {
                std::cout << "Candidate key gen failed" << std::endl;
                continue;
            }

            std::vector<uint8_t> byte_private_key = downcast_private_key(candidate);
            uint64_t h = hash_grid_state(byte_private_key.data(), grid_size);

            // Only keep strictly unique grids
            if (unique_hashes.insert(h).second) {
                if (candidates.size() % 20 == 0){
                    std::cout << std::setw(10) << candidates.size() << " / " << CANDIDATES_COUNT << "              \r" << std::flush;
                    // Final sanity check to ensure no PK boundary constraints were violated
                    confirm_mask_constraints_honored(candidate, master_key.boundary_mask, grid_size, master_key.private_key, alphabet, false);
                }

                // Add candidate
                candidates.push_back(std::move(candidate));
            } else {
                // Collision occurred
                throw std::runtime_error("Collision occurred during ensemble population; the chances of this occurring should be statistically zero");
            }
        }

        int best_i = 0;
        int best_j = 1; // Default fallbacks
        int global_min_dist = grid_size * grid_size + 1;

        if (mode != 2) {
            std::cout << "2. Finding nearest pair for pair id " << pair_id << ": " << std::flush;

            // --- OpenMP Nearest Pair Search ---
            int target_threads = std::max(1, omp_get_num_procs() * 6 / 10);

            #pragma omp parallel num_threads(target_threads)
            {
                // Thread-local trackers
                int local_min_dist = grid_size * grid_size + 1;
                int local_best_i = -1;
                int local_best_j = -1;

                // Dynamic schedule balances the triangular loop workload perfectly
                #pragma omp for schedule(dynamic, 10)
                for (int i = 0; i < CANDIDATES_COUNT; ++i) {
                    for (int j = i + 1; j < CANDIDATES_COUNT; ++j) {
                        
                        // Fast inline hamming distance for std::vector<int>
                        int dist = 0;
                        for (size_t k = 0; k < grid_count; ++k) {
                            if (candidates[i][k] != candidates[j][k]) dist++;
                        }

                        if (dist < local_min_dist) {
                            local_min_dist = dist;
                            local_best_i = i;
                            local_best_j = j;
                        }
                    }
                }

                // Safely merge thread-local minimums into the global minimum
                #pragma omp critical
                {
                    if (local_min_dist < global_min_dist) {
                        global_min_dist = local_min_dist;
                        best_i = local_best_i;
                        best_j = local_best_j;
                    }
                }
            }

            std::cout << "Nearest pair found! Distance: " << global_min_dist << " tiles." << std::endl;
        } else {
            std::cout << "2. Mode 2 selected: Skipping nearest pair search (using master key as origin)." << std::endl;
        }

        std::vector<int> grid_X = (mode == 2) ? master_key.private_key : candidates[best_i];
        confirm_mask_constraints_honored(grid_X, master_key.boundary_mask, grid_size, master_key.private_key, alphabet, false);

        std::vector<std::vector<int>> target_Y_list;
        
        // Setup the datasets based on the requested mode
        if (mode <= 1) {
            target_Y_list.push_back(candidates[best_j]);
        } else if (mode == 2) {
            for(int i = 0; i < PATHS_PER_PAIR; ++i) target_Y_list.push_back(candidates[i]); // 100 radial spokes
        } else if (mode == 3) {
            for(int i = 0; i < PATHS_PER_PAIR; ++i) target_Y_list.push_back(candidates[best_j]); // 1000 stochastic paths to same target
        }

        std::cout << "3. Executing pathfinding for phase space " << pair_id << " (" << PATHS_PER_PAIR << " paths)..." << std::endl;

        for (int path_idx = 0; path_idx < PATHS_PER_PAIR; ++path_idx) {
            std::vector<int> grid_Y = target_Y_list[path_idx];
            confirm_mask_constraints_honored(grid_Y, master_key.boundary_mask, grid_size, master_key.private_key, alphabet, false);

            // Execute pathfinding (Pass the rng for mode 3. Mode 2 uses standard greedy logic)
            std::vector<int> trajectory = execute_greedy_walk(grid_X, grid_Y, alphabet, grid_size, mode, &rng, 5);
            // To force Top-All stochastic mode ("Blind Shortest Paths"), uncomment this line and run with mode 3=stochastic
            //std::vector<int> trajectory = execute_greedy_walk(grid_X, grid_Y, alphabet, grid_size, 3, &rng, 10000000);

            int effective_id = (mode >= 2) ? path_idx : pair_id; // Ensures CSV groups correctly by path for 2 & 3
            int total_steps = trajectory.size() - 1; // Equals the Hamming Distance
            int peak_defects = 0;
            int min_defects = 999999;   // Also track the lowest point
            int peak_step = 0;

            // Log the trajectory
            for (int step = 0; step <= total_steps; ++step) {
                double normalized_step = static_cast<double>(step) / total_steps;
                int defects = trajectory[step];
                
                csv_file << effective_id << "," << step << "," << normalized_step << "," << defects << "\n";
                csv_file.flush();
                
                if (defects > peak_defects) {
                    peak_defects = defects;
                    peak_step = step;
                }
                if (defects < min_defects) {
                    min_defects = defects;
                }
            }

            // Log the macro statistics
            int start_defects = trajectory.front();
            int activation_energy = peak_defects - start_defects;
            int true_climb = peak_defects - min_defects; 
            double peak_normalized = static_cast<double>(peak_step) / total_steps;

            stats_file << effective_id << "," 
                       << total_steps << "," 
                       << start_defects << "," 
                       << peak_defects << "," 
                       << min_defects << ","
                       << activation_energy << "," 
                       << true_climb << ","
                       << peak_normalized << "\n";
            stats_file.flush();

            // Keep console output clean for massive loops
            if (path_idx % 20 == 0 || path_idx == PATHS_PER_PAIR - 1) {
                std::cout << "Path " << path_idx << " processed. Act Energy: " << activation_energy 
                          << " True Climb: " << true_climb << " \r" << std::flush;
            }
        }
        std::cout << "\nPhase space fully mapped." << std::endl;
    }

    csv_file.close();
    stats_file.close();

    CUDA_CHECK(cudaFree(d_grid));
    CUDA_CHECK(cudaFree(d_states));

    std::cout << "\nEnergy Barrier Analysis complete.\n";
}


void execute_topology_attack_simulation(int loop_iterations, int grid_size) {
    JeandelRaoTileSet tileset;
    std::vector<Tile> alphabet = tileset.get_tiles();

    ChaCha20PRNG rng;

    int perimeter_len = grid_size * 4 - 4; // 252 for a 64x64 grid
    std::uniform_real_distribution<double> dist_coords(0.0, 1000000.0);

    LabbeJR11Oracle oracle;

    for (int l = 0; l < loop_iterations; ++l){
        std::cout << "\nAttack Simulation #" << (l+1) << std::endl;
        std::cout << "[1] Generating Ground Truth Quasiperiodic Planes..." << std::endl;

        // Fetch continuous Torus coordinates to generate two independent, 0-defect planes
        std::vector<int> plane_A = oracle.generate_jr11_grid(dist_coords(rng), dist_coords(rng), grid_size);
        std::vector<int> plane_B = oracle.generate_jr11_grid(dist_coords(rng), dist_coords(rng), grid_size);
        std::vector<int> plane_C = oracle.generate_jr11_grid(dist_coords(rng), dist_coords(rng), grid_size);

        std::cout << "[2] Building Spliced Boundary Map (Alternating Sequential Splicing)..." << std::endl;
        std::vector<int> boundary_source_map(perimeter_len, 0); 
        
        // 1. Generate the segment lengths
        // This must be an even number to prevent parity merging between the first and last segment
        int num_segments = 14;  // Must match the reference Public Key generation function's setting
        // 80% min segment length
        int min_seg_len = std::round(0.80f * (float)perimeter_len / num_segments);

        // Set the minimum seg length for each 
        std::vector<int> segment_lengths(num_segments, min_seg_len); 
        int total_assigned = num_segments * min_seg_len;
        
        // 2. Randomly distribute the remaining tiles across the segments
        int remaining = perimeter_len - total_assigned;
        std::uniform_int_distribution<int> dist_idx(0, num_segments - 1);
        for (int i = 0; i < remaining; i++) {
            segment_lengths[dist_idx(rng)]++;
        }
        
        // 3. Random Start Offset to prevent fixed-point attacks
        std::uniform_int_distribution<int> dist_start(0, perimeter_len - 1);
        int start_idx = dist_start(rng);
        
        // 4. Map them onto the perimeter
        int tiles_filled = 0;
        for (int s = 0; s < num_segments; s++) {
            // Because num_segments is exactly 10, this 
            // alternates A, B, A, B... and ensures 
            // the last segment (B) will never merge with the first (A)
            bool use_A = (s % 2 == 0); 
            
            for (int i = 0; i < segment_lengths[s]; i++) {
                int current_idx = (start_idx + tiles_filled + i) % perimeter_len;
                boundary_source_map[current_idx] = use_A ? 0 : 1;
            }
            tiles_filled += segment_lengths[s];
        }
        
        std::cout << "[3] Commencing Custom Topology Attack Sweep..." << std::endl;
        
        // Fire the testing sweep
        run_custom_topology_attack_sweep(plane_A, plane_B, plane_C, boundary_source_map, grid_size, &tileset, rng);
    }
    
    std::cout << "\nSimulation Complete." << std::endl;
}

// ----- Labbé Oracle & Labbé Oracle Vulnerability Unit Tests ------------------------------------------------------

bool run_jr11_oracle_ground_truth_unit_test() {
    std::cout << "\n---------------------------------------------------------" << std::endl;
    std::cout << "   JR-11 Oracle Ground Truth Unit Tests\n" << std::endl;

    JeandelRaoTileSet jr_tileset;
    std::vector<Tile> alphabet = jr_tileset.get_tiles();

    int oracle_size = 1024;
    OracleEngine engine(oracle_size, alphabet);

    // Generate the "Ground Truth" mathematical plane
    engine.generate_perfect_plane_jeandel_rao();

    // Prove the Toral Z^2-Rotation Math is Flawless
    std::cout << "[*] Scanning " << (oracle_size * oracle_size) << " tiles for topological defects..." << std::endl;
    int oracle_defects = engine.get_oracle_defects_count();
    
    std::cout << "[*] Oracle Defect Count: " << oracle_defects << std::endl;
    if (oracle_defects > 0) {
        std::cout << "\n[UNIT TEST FAILED] The oracle ground truth grid is invalid." << std::endl;
        return false;
    }
    else {
        std::cout << "\n[UNIT TEST PASSED] The oracle ground truth grid is defect free." << std::endl;
        return true;
    }
}

bool run_ammann16_oracle_ground_truth_unit_test() {
    std::cout << "\n---------------------------------------------------------" << std::endl;
    std::cout << "   Ammann-16 Oracle Ground Truth Unit Tests\n" << std::endl;

    Ammann16TileSet tileset;
    std::vector<Tile> alphabet = tileset.get_tiles();

    int oracle_size = 1024;
    OracleEngine engine(oracle_size, alphabet);

    // Generate the "Ground Truth" mathematical plane
    engine.generate_perfect_plane_ammann16();

    // Prove the Ground Truth is Flawless
    std::cout << "[*] Scanning " << (oracle_size * oracle_size) << " tiles for topological defects..." << std::endl;
    int oracle_defects = engine.get_oracle_defects_count();
    std::cout << "[*] Oracle Defect Count: " << oracle_defects << std::endl;

    if (oracle_defects > 0) {
        std::cout << "\n[UNIT TEST FAILED] The oracle ground truth grid is invalid." << std::endl;
        return false;
    }
    else {
        std::cout << "\n[UNIT TEST PASSED] The oracle ground truth grid is defect free." << std::endl;
        return true;
    }
}

bool run_anchor_and_vulnerability_unit_tests() {
    std::cout << "\n---------------------------------------------------------" << std::endl;
    std::cout << "   Anchor and Vulnerability Unit Tests\n" << std::endl;

    JeandelRaoTileSet jr_tileset;
    std::vector<Tile> alphabet = jr_tileset.get_tiles();

    // Change to the true Oracle size
    int oracle_size = 1024;
    OracleEngine engine(oracle_size, alphabet);

    // 1. Generate the true mathematical plane (not random)
    engine.generate_perfect_plane_jeandel_rao();

    // 2. Extract a "perfect" 64x64 Wang grid from deep inside the true Oracle
    int secret_x = 100;
    int secret_y = 100;
    
    std::vector<int> true_private_grid(64 * 64);
    QwtssPublicInputs mock_pub_inputs("username", 0, 0, 64);

    // Copy the private tiles directly from the engine's validated memory
    for (int r = 0; r < 64; r++) {
        for (int c = 0; c < 64; c++) {
            int x = secret_x + c;
            int y = secret_y + r;
            
            // Extract the real Labbé tile ID
            int oracle_val = engine.get_tile_id(x, y);
            true_private_grid[r * 64 + c] = oracle_val;

            // Build Boundaries mapping strictly to grid edges using true alphabet colors
            if (r == 0)  mock_pub_inputs.north[c] = alphabet[oracle_val].top;
            if (r == 63) mock_pub_inputs.south[c] = alphabet[oracle_val].bottom;
            if (c == 0)  mock_pub_inputs.west[r]  = alphabet[oracle_val].left;
            if (c == 63) mock_pub_inputs.east[r]  = alphabet[oracle_val].right;
        }
    }

    std::cout << "[*] True 64x64 Private Grid and Public Key extracted at offset (" 
              << secret_x << ", " << secret_y << ")." << std::endl;
    std::cout << "[*] Launching CUDA Sliding Window Search (Min Match: 32 colors)..." << std::endl;

    // 3. Run the Attacker's Search
    int min_match_length = 64;  // Perfect matches should be possible
    auto anchors = engine.find_anchors(mock_pub_inputs, min_match_length);
    
    std::cout << "[*] CUDA Engine found " << anchors.size() << " valid mathematical anchors." << std::endl;

    if (anchors.empty()) {
        std::cout << "\n[UNIT TEST FAILED] No mathematical anchors were found!" << std::endl;
        return false;
    }

    if (true){
        // --- SPATIAL INTERSECTION FILTER ---
        // 3.5 Attacker Heuristic: "Corner Pinning"
        // Group all 1D anchors by their implied 2D top-left origin coordinate
        std::map<std::pair<int, int>, int> origin_votes;
        for (const auto& anchor : anchors) {
            int tx = anchor.oracle_x;
            int ty = anchor.oracle_y;
            if (anchor.edge == BoundaryAnchor::NORTH) { tx -= anchor.pk_offset; }
            else if (anchor.edge == BoundaryAnchor::SOUTH) { tx -= anchor.pk_offset; ty -= 63; }
            else if (anchor.edge == BoundaryAnchor::WEST) { ty -= anchor.pk_offset; }
            else if (anchor.edge == BoundaryAnchor::EAST) { ty -= anchor.pk_offset; tx -= 63; }
            
            origin_votes[{tx, ty}]++;
        }

        // Only keep anchors that geometrically intersect with at least one other edge
        std::vector<BoundaryAnchor> filtered_anchors;
        for (const auto& anchor : anchors) {
            int tx = anchor.oracle_x;
            int ty = anchor.oracle_y;
            if (anchor.edge == BoundaryAnchor::NORTH) { tx -= anchor.pk_offset; }
            else if (anchor.edge == BoundaryAnchor::SOUTH) { tx -= anchor.pk_offset; ty -= 63; }
            else if (anchor.edge == BoundaryAnchor::WEST) { ty -= anchor.pk_offset; }
            else if (anchor.edge == BoundaryAnchor::EAST) { ty -= anchor.pk_offset; tx -= 63; }
            
            // For a perfect 0-defect grid, we demand all 4 edges perfectly align at the exact same coordinate
            if (origin_votes[{tx, ty}] == 4) { 
                filtered_anchors.push_back(anchor);
            }
        }

        std::cout << "[*] Spatial Intersection isolated " << (filtered_anchors.size() / 4) 
                << " true 2D origin patches." << std::endl;
        anchors = filtered_anchors;
    }

    // 4. Attacker floods the interior using the anchored coordinate geometry
    std::vector<uint16_t> predicted_grid = engine.project_and_intersect(anchors);

    // 5. Measure the Vulnerability
    CryptanalysisMetrics metrics = engine.evaluate_vulnerability(predicted_grid, true_private_grid);

    int true_defect_count = count_grid_defects(true_private_grid.data(), 64, alphabet);

    std::cout << "\n--- CRYPTANALYSIS RESULTS ---" << std::endl;
    std::cout << "True Defect Count:        " << true_defect_count << std::endl;
    std::cout << "Anchors Found:            " << anchors.size() << std::endl;
    std::cout << "Tiles Correctly Guessed:  " << metrics.tiles_correctly_predicted << " / 4096" << std::endl;
    std::cout << "Entropy Reduction:        " << metrics.entropy_reduction_percent << "%" << std::endl;
    std::cout << "Remaining Security Bits:  " << metrics.remaining_entropy_bits << " bits" << std::endl;

    // Confirm the function returns same result
    ChaCha20PRNG rng;
    PrivateKey private_key;
    private_key.defect_count = true_defect_count;
    private_key.grid_size = 64;
    private_key.grid_data = true_private_grid;
    private_key.boundary_mask = generate_partial_boundary_mask(64, 1.0f, rng);
    auto func_result = evaluate_private_key_vulnerability(engine, private_key, 64);

    // Note: the evaluate_private_key_vulnerability() uses 2 edge votes for spatial filtering, so it is slightly less than 100%
    if (metrics.entropy_reduction_percent == 100.0 && func_result.entropy_reduction_percent > 95.0) {
        std::cout << "\n[UNIT TEST PASSED] The framework successfully achieved 100% phase recovery on the Labbé Oracle." << std::endl;
        return true;
    } else {
        std::cout << "\n[UNIT TEST FAILED] The framework failed to fully recover the grid." << std::endl;
        return false;
    }
}

bool run_tiling_cryptanalysis_unit_tests() {
    std::cout << "\n=========================================================" << std::endl;
    std::cout << "   Algebraic Cryptanalysis Unit Tests" << std::endl;
    std::cout << "=========================================================\n" << std::endl;

    bool all_passed = true;

    all_passed &= run_jr11_oracle_ground_truth_unit_test();
    all_passed &= run_ammann16_oracle_ground_truth_unit_test();
    all_passed &= run_anchor_and_vulnerability_unit_tests();

    if (all_passed) std::cout << "\n[PASSED] All unit tests passed." << std::endl;
    else {
        std::cout << "\n[FAILED] One or more unit tests failed." << std::endl;
        throw std::runtime_error("One or more unit tests failed.");
    }

    return all_passed;
}


// ----- Solution Cluster Characterization Routines ----------------------------------------------------------------

// The ray-trace result
struct StarburstResult {
    std::vector<int> terminal_grid;
    int length;
    int final_defects;
    double raw_branching_bits;          // The pure thickness of the dendrite (without D! over-correction)
    double log2_local_commutativity;    // Tracks how many moves commute locally
    std::vector<int> branching_path;    // Records 'b' (valid moves) at each step N
    std::vector<int> in_degree_path;    // The accurate commutative penalty
};

// Computes the exact in-degree for the most accurate volume estimation
StarburstResult execute_radial_starburst_walk_with_exact_correction(
    const std::vector<int>& origin_grid,
    const std::vector<Tile>& alphabet,
    int grid_size,
    int min_defect_floor,
    int max_defect_ceiling,
    ChaCha20PRNG& rng,
    bool record_path = false
) {
    std::vector<int> current_grid = origin_grid;
    int current_defects = count_grid_defects(current_grid.data(), grid_size, alphabet);
    int distance = 0;
    
    double raw_branching_bits = 0.0;
    double log2_local_commutativity = 0.0;
    std::vector<int> branching_path;
    std::vector<int> in_degree_path;

    // --- MEMORY OPTIMIZATION ---
    // Hoisted outside the loop to prevent expensive heap allocations
    struct OutwardMove { int tile_idx; int new_tile_id; int delta; };
    std::vector<OutwardMove> valid_moves;
    // Pre-allocate to avoid resizing
    valid_moves.reserve(4096);

    while (true) {
        valid_moves.clear();

        // Scan for all valid outward moves
        for (int i = 0; i < grid_size * grid_size; ++i) {
            if (current_grid[i] == origin_grid[i]) {
                int old_local_d = get_local_tile_defects(current_grid, i, current_grid[i], alphabet, grid_size);
                for (int a = 0; a < alphabet.size(); ++a) {
                    if (a == current_grid[i]) continue;
                    int next_defects = current_defects + (get_local_tile_defects(current_grid, i, a, alphabet, grid_size) - old_local_d);
                    if (next_defects >= min_defect_floor && next_defects <= max_defect_ceiling) {
                        valid_moves.push_back({i, a, next_defects - current_defects});
                    }
                }
            }
        }

        if (valid_moves.empty()) break;

        // Record the branching factor for this specific step
        if (record_path) {
            branching_path.push_back(valid_moves.size());
        }

        // RAW THICKNESS: How many directions can we grow? (Out-Degree)
        raw_branching_bits += std::log2(static_cast<double>(valid_moves.size()));

        // 1. EXECUTE THE MOVE FIRST (Transition to the next node)
        std::uniform_int_distribution<int> dist(0, valid_moves.size() - 1);
        OutwardMove selected_move = valid_moves[dist(rng)];

        current_grid[selected_move.tile_idx] = selected_move.new_tile_id;
        current_defects += selected_move.delta;
        distance++;

        // 2. EXACT LOCAL COMMUTATIVITY (IN-DEGREE PROBE)
        // Probe the In-Degree of the new node we just arrived at
        int true_in_degree = 0;
        
        for (int i = 0; i < grid_size * grid_size; ++i) {
            if (current_grid[i] != origin_grid[i]) {
                int old_local_d = get_local_tile_defects(current_grid, i, current_grid[i], alphabet, grid_size);
                int new_local_d = get_local_tile_defects(current_grid, i, origin_grid[i], alphabet, grid_size);
                int parent_defects = current_defects + (new_local_d - old_local_d);
                
                if (parent_defects >= min_defect_floor && parent_defects <= max_defect_ceiling) {
                    true_in_degree++;
                }
            }
        }

        // Apply the exact mathematical penalty for the node we are now standing on
        int exact_in_degree = std::max(1, true_in_degree);
        log2_local_commutativity += std::log2(static_cast<double>(exact_in_degree));

        // Save the exact penalty for step-by-step entropy decay profiling
        if (record_path) {
            in_degree_path.push_back(exact_in_degree);
        }
    }

    return {current_grid, distance, current_defects, raw_branching_bits, log2_local_commutativity, branching_path, in_degree_path};
}

struct DeepProbeResult {
    // Index 0 = Survivors at Level 1, Index 1 = Survivors at Level 2, etc.
    // These values will always be strictly monotonically decreasing and <= 4096
    std::vector<int> surviving_parents; 
};

// Returns the maximum backward depth this specific parent can reach (up to max_depth)
int get_max_backward_depth(
    std::vector<int>& current_grid,
    int current_defects,
    const std::vector<int>& origin_grid,
    const std::vector<Tile>& alphabet,
    int grid_size,
    int min_defect_floor,
    int max_defect_ceiling,
    int current_depth,
    int max_depth
) {
    // If we've already reached the target depth, we can stop searching this branch
    if (current_depth == max_depth) return max_depth;
    
    int max_found = current_depth;

    for (int i = 0; i < grid_size * grid_size; ++i) {
        if (current_grid[i] != origin_grid[i]) {
            
            int old_local_d = get_local_tile_defects(current_grid, i, current_grid[i], alphabet, grid_size);
            int new_local_d = get_local_tile_defects(current_grid, i, origin_grid[i], alphabet, grid_size);
            int parent_defects = current_defects + (new_local_d - old_local_d);
            
            if (parent_defects >= min_defect_floor && parent_defects <= max_defect_ceiling) {
                
                // We found a valid step deeper
                int temp_tile = current_grid[i];
                current_grid[i] = origin_grid[i];
                
                int branch_depth = get_max_backward_depth(
                    current_grid, parent_defects, origin_grid, alphabet, 
                    grid_size, min_defect_floor, max_defect_ceiling, 
                    current_depth + 1, max_depth
                );
                
                current_grid[i] = temp_tile;

                if (branch_depth > max_found) {
                    max_found = branch_depth;
                }
                
                // EARLY EXIT: If we found any path that reaches the target max_depth, 
                // we don't need to search any other branches for this parent. It survives.
                if (max_found == max_depth) {
                    break; 
                }
            }
        }
    }
    
    return max_found;
}

// The main interface for the Survival Probe
DeepProbeResult execute_survival_backward_probe(
    const std::vector<int>& current_grid_state,
    int current_defects,
    const std::vector<int>& origin_grid,
    const std::vector<Tile>& alphabet,
    int grid_size,
    int min_defect_floor,
    int max_defect_ceiling,
    int max_probe_levels
) {
    DeepProbeResult result;
    if (max_probe_levels <= 0) return result;
    
    result.surviving_parents.resize(max_probe_levels, 0);
    std::vector<int> working_grid = current_grid_state;
    
    // We iterate strictly over the Level-1 parents
    for (int i = 0; i < grid_size * grid_size; ++i) {
        if (working_grid[i] != origin_grid[i]) {
            
            int old_local_d = get_local_tile_defects(working_grid, i, working_grid[i], alphabet, grid_size);
            int new_local_d = get_local_tile_defects(working_grid, i, origin_grid[i], alphabet, grid_size);
            int parent_defects = current_defects + (new_local_d - old_local_d);
            
            if (parent_defects >= min_defect_floor && parent_defects <= max_defect_ceiling) {
                
                // This is a valid Level-1 parent
                result.surviving_parents[0]++;
                
                // Test how far back this specific parent can go
                if (max_probe_levels > 1) {
                    int temp_tile = working_grid[i];
                    working_grid[i] = origin_grid[i];
                    
                    int max_reached = get_max_backward_depth(
                        working_grid, parent_defects, origin_grid, alphabet, 
                        grid_size, min_defect_floor, max_defect_ceiling, 
                        1, max_probe_levels
                    );
                    
                    working_grid[i] = temp_tile;
                    
                    // Increment the survival counters based on how deep it made it
                    for (int lvl = 1; lvl < max_probe_levels; ++lvl) {
                        if (max_reached >= lvl + 1) {
                            result.surviving_parents[lvl]++;
                        }
                    }
                }
            }
        }
    }
    
    return result;
}

struct ProbeStepRecord {
    int step_n;
    int out_degree;
    int in_level_1;
    int in_level_2;
    int in_level_3;
};

struct ProfilingResult {
    int terminal_length;
    std::vector<ProbeStepRecord> path_data;
};

// Custom walk function optimized for data caching and deep DFS probing
ProfilingResult execute_profiling_walk(
    const std::vector<int>& origin_grid,
    const std::vector<Tile>& alphabet,
    int grid_size,
    int min_defect_floor,
    int max_defect_ceiling,
    int max_probe_levels,
    ChaCha20PRNG& rng
) {
    std::vector<int> current_grid = origin_grid;
    int current_defects = count_grid_defects(current_grid.data(), grid_size, alphabet);
    int distance = 0;
    
    ProfilingResult result;
    result.path_data.reserve(500); // Pre-allocate cache to prevent reallocation
    
    struct OutwardMove { int tile_idx; int new_tile_id; int delta; };
    std::vector<OutwardMove> valid_moves;
    valid_moves.reserve(4096);

    while (true) {
        valid_moves.clear();
        
        for (int i = 0; i < grid_size * grid_size; ++i) {
            if (current_grid[i] == origin_grid[i]) {
                int old_local_d = get_local_tile_defects(current_grid, i, current_grid[i], alphabet, grid_size);
                for (int a = 0; a < alphabet.size(); ++a) {
                    if (a == current_grid[i]) continue;
                    int next_defects = current_defects + (get_local_tile_defects(current_grid, i, a, alphabet, grid_size) - old_local_d);
                    if (next_defects >= min_defect_floor && next_defects <= max_defect_ceiling) {
                        valid_moves.push_back({i, a, next_defects - current_defects});
                    }
                }
            }
        }

        if (valid_moves.empty()) break;

        std::uniform_int_distribution<int> dist(0, valid_moves.size() - 1);
        OutwardMove selected_move = valid_moves[dist(rng)];

        current_grid[selected_move.tile_idx] = selected_move.new_tile_id;
        current_defects += selected_move.delta;
        distance++;

        // Set this greater than 'max_probe_levels' for extra confidence in the results
        int DEEP_VERIFICATION_LEVEL = 10; // Deep enough for rigorous proof, shallow enough to survive a brute-force failure
        if (DEEP_VERIFICATION_LEVEL < max_probe_levels) DEEP_VERIFICATION_LEVEL = max_probe_levels;

        // HORIZON FIX: We cannot probe deeper than our current distance from the origin.
        // We probe up to our target canary depth, cleanly capped by the physical boundary.
        int actual_probe_depth = std::min(distance, DEEP_VERIFICATION_LEVEL);

        // Execute the deep DFS topological survival probe
        DeepProbeResult deep_probe = execute_survival_backward_probe(
            current_grid, current_defects, origin_grid, alphabet, grid_size,
            min_defect_floor, max_defect_ceiling,
            //max_probe_levels
            actual_probe_depth
        );

        // Safely extract the levels, defaulting to 0 if the vector is smaller than 3
        int in_1 = deep_probe.surviving_parents.size() > 0 ? deep_probe.surviving_parents[0] : 0;
        int in_2 = deep_probe.surviving_parents.size() > 1 ? deep_probe.surviving_parents[1] : 0;
        int in_3 = deep_probe.surviving_parents.size() > 2 ? deep_probe.surviving_parents[2] : 0;

        // Extract the absolute deepest verification level actually physically available right now
        int in_max = deep_probe.surviving_parents.empty() ? 0 : deep_probe.surviving_parents.back();

        // Trigger on any unexpected ghost path: 
        // If a valid immediate parent fails to survive to the deepest level, the topology has ghost paths that lead to bias
        if (in_1 > 0 && in_1 != in_max) {
            #pragma omp critical
            {
                std::cout << "\n[TOPOLOGY ALERT] Ghost Path Detected! Source of bias identified at Step " << distance 
                          << " | In_1: " << in_1 << " | In_" << DEEP_VERIFICATION_LEVEL << ": " << in_max 
                          << " | Thread ID: " << omp_get_thread_num() << "\n";
            }
        }

        // Cache the step data in RAM (Only writing 1, 2, and 3 to keep the CSV clean)
        result.path_data.push_back({
            distance, 
            (int)valid_moves.size(), 
            in_1, 
            in_2, 
            in_3
        });
    }
    
    result.terminal_length = distance;
    return result;
}

// Orchestrator for the Surrogate Model Data Collection
// This collects a large dataset of in-degree values of increasing resolution (increasing recursion depth)
void run_deep_surrogate_profiling(int grid_size, int cluster_num) {
    JeandelRaoTileSet tileset;
    std::vector<Tile> alphabet = tileset.get_tiles();
    ChaCha20PRNG master_rng;

    int TARGET_DEFECTS = 165;
    int MIN_FLOOR = 160;     
    int MAX_CEILING = 170;

    // Strict subset. O(D^3) scaling dictates a smaller ray pool.
    int RAYS_TO_CAST_PROFILING = 1000;
    int MAX_PROBE_LEVELS = 3;

    std::cout << "--- Initializing Deep Surrogate Profiling For Cluster #" << cluster_num << " ---" << std::endl;

    std::uniform_int_distribution<uint64_t> nonce_dist;
    QwtssPrivateKey master_key;
    int master_key_defects;
    bool mk_success = false;
    
    while (!mk_success) {
        uint64_t rnd_nonce = nonce_dist(master_rng);
        master_key = build_qwtss_private_key(generate_random_username(master_rng), rnd_nonce, 0,
            grid_size, TARGET_DEFECTS - 3, TARGET_DEFECTS + 3, 0.7f, true);
        master_key_defects = count_grid_defects(master_key.private_key.data(), grid_size, alphabet);
        if (master_key_defects >= MIN_FLOOR && master_key_defects <= MAX_CEILING) mk_success = true;
    }
    std::vector<int> G_Origin = master_key.private_key;
    
    std::cout << "Origin Anchor established with " << master_key_defects << " defects. Firing " << RAYS_TO_CAST_PROFILING << " profiling rays...\n";

    // Generate a strictly unique 256-bit seed (8x 32-bit words) for every single ray before threading
    std::vector<std::array<uint32_t, 8>> thread_seeds(RAYS_TO_CAST_PROFILING);
    std::uniform_int_distribution<uint32_t> seed_dist;
    
    for (int i = 0; i < RAYS_TO_CAST_PROFILING; ++i) {
        for (int j = 0; j < 8; ++j) {
            thread_seeds[i][j] = seed_dist(master_rng); 
        }
    }

    std::ofstream csv("deep_surrogate_profiling_" + std::to_string(cluster_num) + ".csv");
    csv << "Ray_ID,Step_N,Normalized_N_hat,Out_Degree,In_Level_1,In_Level_2,In_Level_3\n";

    int target_threads = std::max(1, omp_get_num_procs() * 8 / 10);
    
    // Dynamic schedule chunk set to 1 due to high variance in O(D^3) ray execution times
    #pragma omp parallel for num_threads(target_threads) schedule(dynamic, 1)
    for (int i = 0; i < RAYS_TO_CAST_PROFILING; ++i) {
        // Initialize the local PRNG with the pre-calculated, thread-safe unique 256-bit seed
        ChaCha20PRNG local_rng(thread_seeds[i]);

        ProfilingResult profile = execute_profiling_walk(
            G_Origin, alphabet, grid_size, MIN_FLOOR, MAX_CEILING, MAX_PROBE_LEVELS, local_rng
        );

        #pragma omp critical
        {
            // Calculate N_hat post-termination and write cached trajectory to disk
            for (const auto& step : profile.path_data) {
                double n_hat = (profile.terminal_length > 0) ? ((double)step.step_n / profile.terminal_length) : 0.0;
                csv << i << "," 
                    << step.step_n << "," 
                    << std::fixed << std::setprecision(6) << n_hat << ","
                    << step.out_degree << ","
                    << step.in_level_1 << ","
                    << step.in_level_2 << ","
                    << step.in_level_3 << "\n";
            }

            if (i % 10 == 0 || i == RAYS_TO_CAST_PROFILING - 1) {
                std::cout << "Profiling Rays Cast: " << std::setw(4) << (i + 1) << " / " << RAYS_TO_CAST_PROFILING << "\r" << std::flush;
            }
        }
    }

    std::cout << "\n\n================ PROFILING COMPLETE ================\n";
    std::cout << "Data saved to deep_surrogate_profiling_" << cluster_num << ".csv\n";
    std::cout << "====================================================\n";

    csv.close();
}

void run_starburst_volume_estimation(int grid_size, int cluster_num) {
    JeandelRaoTileSet tileset;
    std::vector<Tile> alphabet = tileset.get_tiles();
    ChaCha20PRNG master_rng;

    int TARGET_DEFECTS = 165;
    int MIN_FLOOR = 160;     // Strict Defect Band
    int MAX_CEILING = 170;
    // Note: for any form of Importance Sampling, the ray walks must be purely stochastic
    // with uniform random path sampling to produce correct results
    int RAYS_TO_CAST_STOCHASTIC = 50000;
    //RAYS_TO_CAST_STOCHASTIC = 10000;

    std::cout << "--- Initializing Strict Starburst Volume Estimation For Cluster #" << cluster_num << " ---" << std::endl;

    // 1. Generate the Origin Anchor
    std::uniform_int_distribution<uint64_t> nonce_dist;
    QwtssPrivateKey master_key;
    int master_key_defects;
    bool mk_success = false;
    
    while (!mk_success) {
        uint64_t rnd_nonce = nonce_dist(master_rng);
        master_key = build_qwtss_private_key(generate_random_username(master_rng), rnd_nonce, 0,
            grid_size, TARGET_DEFECTS - 2, TARGET_DEFECTS + 2, 0.7f, true);
        master_key_defects = count_grid_defects(master_key.private_key.data(), grid_size, alphabet);
        if (master_key_defects >= MIN_FLOOR && master_key_defects <= MAX_CEILING) mk_success = true;
    }
    std::vector<int> G_Origin = master_key.private_key;
    std::cout << "Origin Anchor established with " << master_key_defects << " defects. Firing " << RAYS_TO_CAST_STOCHASTIC << " strict rays into the phase space...\n";

    std::ofstream csv("starburst_volume_map_" + std::to_string(cluster_num) + ".csv");
    csv << "Ray_ID,Terminal_Distance,Terminal_Defects,Raw_Thickness_Bits,Local_Commutativity_Bits\n";

    long long total_dendrite_length = 0;
    int max_length_seen = 0;
    double total_real_volume_accumulator = 0.0;

    // Generate a strictly unique 256-bit seed (8x 32-bit words) for every single ray before threading
    std::vector<std::array<uint32_t, 8>> thread_seeds(RAYS_TO_CAST_STOCHASTIC);
    std::uniform_int_distribution<uint32_t> seed_dist;
    
    for (int i = 0; i < RAYS_TO_CAST_STOCHASTIC; ++i) {
        for (int j = 0; j < 8; ++j) {
            thread_seeds[i][j] = seed_dist(master_rng); 
        }
    }

    // 2. Fire the Rays
    int target_threads = std::max(1, omp_get_num_procs() * 8 / 10);
    #pragma omp parallel for num_threads(target_threads) schedule(dynamic, 100)
    for (int i = 0; i < RAYS_TO_CAST_STOCHASTIC; ++i) {
        // Initialize the local PRNG with the pre-calculated, thread-safe unique 256-bit seed
        ChaCha20PRNG local_rng(thread_seeds[i]);

        // Implements the exact Rosenbluth Markov Chain for a self-avoiding walk
        StarburstResult tip = execute_radial_starburst_walk_with_exact_correction(G_Origin, alphabet, grid_size, MIN_FLOOR, MAX_CEILING, local_rng);

        #pragma omp critical
        {
            total_dendrite_length += tip.length;
            if (tip.length > max_length_seen) max_length_seen = tip.length;

            // Calculate the net entropy for this ray
            double net_bits = tip.raw_branching_bits - tip.log2_local_commutativity;
            
            // Accumulate in real-space (Importance Sampling)
            total_real_volume_accumulator += std::pow(2.0, net_bits);

            csv << i << "," << tip.length << "," << tip.final_defects << "," 
                << std::fixed << std::setprecision(4) << tip.raw_branching_bits << ","
                << std::fixed << std::setprecision(4) << tip.log2_local_commutativity << "\n";

            if (i % 200 == 0 || i == RAYS_TO_CAST_STOCHASTIC - 1) {
                // Compute the current log-average estimate
                int current_m = i + 1;
                double current_est_bits = std::log2(total_real_volume_accumulator / current_m);

                std::cout << "Rays Cast: " << std::setw(6) << i 
                          << " | Max Depth: " << std::setw(4) << max_length_seen 
                          << " | Raw Branching Bits: " << std::setw(6) << std::fixed << std::setprecision(2) << tip.raw_branching_bits
                          << " | Log2 Local Comm: " << tip.log2_local_commutativity
                          << " | Est. Cluster Size: 2^" << std::fixed << std::setprecision(2) << std::setw(6) << current_est_bits << " bits"
                          << "         \r" << std::flush;
            }
        }
    }

    double final_est_bits = std::log2(total_real_volume_accumulator / RAYS_TO_CAST_STOCHASTIC);

    std::cout << "\n\n================ CENSUS RESULTS FOR CLUSTER " << cluster_num << " ================\n";
    std::cout << "Total Rays Cast:        " << RAYS_TO_CAST_STOCHASTIC << "\n";
    std::cout << "Average Ray Depth:      " << ((double)total_dendrite_length / RAYS_TO_CAST_STOCHASTIC) << " tiles\n";
    std::cout << "Maximum Ray Depth:      " << max_length_seen << " tiles\n";
    std::cout << "Final Estimated Cluster Entropy:  " << std::fixed << std::setprecision(2) << final_est_bits << " bits\n";
    std::cout << "Data saved to starburst_volume_map.csv\n";
    std::cout << "================================================\n";

    csv.close();
}

struct RaySummary {
    int id;
    int length;
    double net_bits;
    bool operator<(const RaySummary& other) const {
        return net_bits > other.net_bits; // Sort descending
    }
};

void run_cluster_volume_est_ensemble(int grid_size, int cluster_num, int num_anchors_in_ensemble) {
    JeandelRaoTileSet tileset;
    std::vector<Tile> alphabet = tileset.get_tiles();
    ChaCha20PRNG master_rng;

    int TARGET_DEFECTS = 165;
    int MIN_FLOOR = 160;
    int MAX_CEILING = 170;
    int RAYS_PER_ANCHOR = 30000;        // Rays per origin anchor

    // Note: Entropic gravity will naturally pull us into more central cluster locations; however,
    // diffusion is notoriously slow. For an unguided random walk in a 1D-like corridor (e.g., our 
    // fractal dendrites, given their low branching factor), the expected displacement D after t steps
    // scales with the square root of time: D ≈ sqrt(t)​. So we need a very large value here to ensure
    // adaquate diffusion.
    int MCMC_STEPS_BETWEEN_ANCHORS = 120000; // How far to walk to ensure anchor independence

    std::cout << "--- Initializing Cluster Volume Estimation Stability Analysis For Cluster #" << cluster_num << " ---\n";

    // 1. Generate the Origin Master Anchor
    std::uniform_int_distribution<uint64_t> nonce_dist;
    QwtssPrivateKey master_key;
    int master_key_defects;
    bool mk_success = false;
    
    while (!mk_success) {
        uint64_t rnd_nonce = nonce_dist(master_rng);
        master_key = build_qwtss_private_key(generate_random_username(master_rng), rnd_nonce, 0,
            grid_size, TARGET_DEFECTS - 2, TARGET_DEFECTS + 2, 0.7f, true);
        if (master_key.private_key.empty()) continue;   // Key gen failed; retry
        master_key_defects = count_grid_defects(master_key.private_key.data(), grid_size, alphabet);
        if (master_key_defects >= MIN_FLOOR && master_key_defects <= MAX_CEILING) mk_success = true;
    }
    
    std::cout << "Master Anchor established. Gathering " << num_anchors_in_ensemble - 1 
              << " independent anchors via intra-cluster random walk...\n";

    // 2. Gather independent anchors via MCMC walk
    std::vector<std::vector<int>> cluster_anchors;
    cluster_anchors.push_back(master_key.private_key);

    std::vector<int> current_grid = master_key.private_key;
    int current_defects = master_key_defects;

    // Strictly target the grid interior to preserve PK boundary constraints
    std::uniform_int_distribution<int> interior_dist(1, grid_size - 2);
    std::uniform_int_distribution<int> color_dist(0, alphabet.size() - 1);

    for (int a = 1; a < num_anchors_in_ensemble; ++a) {
        int walk_steps = 0;
        while (walk_steps < MCMC_STEPS_BETWEEN_ANCHORS) {
            // Allow mutation of grid interior tiles only (to preserve PK boundary)
            int r = interior_dist(master_rng);
            int c = interior_dist(master_rng);
            int idx = r * grid_size + c;

            int new_color = color_dist(master_rng);
            if (current_grid[idx] == new_color) continue;

            int old_d = get_local_tile_defects(current_grid, idx, current_grid[idx], alphabet, grid_size);
            int new_d = get_local_tile_defects(current_grid, idx, new_color, alphabet, grid_size);
            int next_defects = current_defects + (new_d - old_d);

            if (next_defects >= MIN_FLOOR && next_defects <= MAX_CEILING) {
                current_grid[idx] = new_color;
                current_defects = next_defects;
                walk_steps++;
            }
        }
        // Sanity check
        int anchor_grid_defects = count_grid_defects(current_grid.data(), grid_size, alphabet);
        if (anchor_grid_defects < MIN_FLOOR || anchor_grid_defects > MAX_CEILING) {
            std::cout << "Error: sampled anchor grid was not within the operational band!" << std::endl;
            throw std::runtime_error("Error: sampled anchor grid was not within the operational band!");
        }

        cluster_anchors.push_back(current_grid);
    }

    std::cout << "\n--- Intra-Cluster Anchor Distance Analysis ---\n";
    for (int a = 0; a < num_anchors_in_ensemble; ++a) {
        int min_hd = grid_size * grid_size + 1;
        int max_hd = -1;
        int closest_idx = -1;
        int farthest_idx = -1;

        for (int b = 0; b < num_anchors_in_ensemble; ++b) {
            if (a == b) continue;
            int hd = 0;
            for (size_t k = 0; k < cluster_anchors[a].size(); ++k) {
                if (cluster_anchors[a][k] != cluster_anchors[b][k]) hd++;
            }
            if (hd < min_hd) { min_hd = hd; closest_idx = b; }
            if (hd > max_hd) { max_hd = hd; farthest_idx = b; }
        }
        std::cout << "  -> Anchor " << a << " acquired. Closest to Anchor " << closest_idx 
                  << " (HD = " << min_hd << ") and farthest from Anchor " << farthest_idx 
                  << " (HD = " << max_hd << ").\n";
    }

    std::vector<double> volume_estimates;
    std::vector<double> volume_accumulators; // Store accumulators for later Pareto calculation
    std::vector<std::vector<RaySummary>> top_rays_per_anchor; // Store top rays
    std::vector<int> max_depths; // Store max depths for final stats
    int target_threads = std::max(1, omp_get_num_procs() * 8 / 10);

    // 3. Perform the Rosenbluth-Knuth Starburst for each Anchor
    for (int a = 0; a < num_anchors_in_ensemble; ++a) {
        std::cout << "\nFiring " << RAYS_PER_ANCHOR << " rays from Anchor " << a << "...\n";

        // Pre-generate unique seeds for thread safety
        std::vector<std::array<uint32_t, 8>> thread_seeds(RAYS_PER_ANCHOR);
        std::uniform_int_distribution<uint32_t> seed_dist;
        for (int i = 0; i < RAYS_PER_ANCHOR; ++i) {
            for (int j = 0; j < 8; ++j) thread_seeds[i][j] = seed_dist(master_rng); 
        }

        double total_real_volume_accumulator = 0.0;
        long long total_length = 0;
        int max_len = 0;
        int rays_completed = 0; // Shared counter for OpenMP progress tracking

        std::vector<RaySummary> all_rays(RAYS_PER_ANCHOR);

        #pragma omp parallel for num_threads(target_threads) schedule(dynamic, 100)
        for (int i = 0; i < RAYS_PER_ANCHOR; ++i) {
            ChaCha20PRNG local_rng(thread_seeds[i]);
            
            StarburstResult tip = execute_radial_starburst_walk_with_exact_correction(
                cluster_anchors[a], alphabet, grid_size, MIN_FLOOR, MAX_CEILING, local_rng);

            double net_bits = tip.raw_branching_bits - tip.log2_local_commutativity;

            // Store the ray data for Pareto sorting later
            all_rays[i] = {i, tip.length, net_bits};

            #pragma omp critical
            {
                total_length += tip.length;
                if (tip.length > max_len) max_len = tip.length;
                total_real_volume_accumulator += std::pow(2.0, net_bits);

                // Real-time progress output (throttled to avoid I/O bottlenecks)
                rays_completed++;
                if (rays_completed % 200 == 0 || rays_completed == RAYS_PER_ANCHOR) {
                    double current_est_bits = std::log2(total_real_volume_accumulator / rays_completed);
                    std::cout << "    Progress: " << std::setw(5) << rays_completed << " / " << RAYS_PER_ANCHOR 
                              << " rays | Current Est: 2^" << std::fixed << std::setprecision(2) << current_est_bits 
                              << " bits      \r" << std::flush;
                }
            }
        }

        // Lock in the final output line so the Pareto analysis doesn't overwrite it
        std::cout << "\n";

        // Sort rays by their entropic volume contribution (descending)
        std::sort(all_rays.begin(), all_rays.end());

        // Save the top 10 rays and accumulator for this anchor instead of printing them now
        std::vector<RaySummary> top_10;
        for (int k = 0; k < 10 && k < RAYS_PER_ANCHOR; ++k) {
            top_10.push_back(all_rays[k]);
        }
        top_rays_per_anchor.push_back(top_10);
        volume_accumulators.push_back(total_real_volume_accumulator);

        double est_bits = std::log2(total_real_volume_accumulator / RAYS_PER_ANCHOR);
        volume_estimates.push_back(est_bits);
        max_depths.push_back(max_len); // Save the maximum depth
        std::cout << "  => Anchor " << a << " Estimate: 2^" << std::fixed << std::setprecision(2) << est_bits << " bits "
                  << "(Max Depth: " << max_len << ")\n";
    }

    // 4. Calculate Stability Metrics
    double sum = std::accumulate(volume_estimates.begin(), volume_estimates.end(), 0.0);
    double mean = sum / num_anchors_in_ensemble;

    double variance = 0.0;
    for (double v : volume_estimates) {
        variance += (v - mean) * (v - mean);
    }
    double stddev = std::sqrt(variance / num_anchors_in_ensemble);
    
    auto minmax = std::minmax_element(volume_estimates.begin(), volume_estimates.end());
    int max_anchor_idx = std::distance(volume_estimates.begin(), minmax.second); // Find the absolute max

    // Calculate "Max Depth" Statistics
    double depth_sum = std::accumulate(max_depths.begin(), max_depths.end(), 0.0);
    double depth_mean = depth_sum / num_anchors_in_ensemble;
    double depth_variance = 0.0;
    for (int d : max_depths) {
        depth_variance += (d - depth_mean) * (d - depth_mean);
    }
    double depth_stddev = std::sqrt(depth_variance / num_anchors_in_ensemble);
    auto depth_minmax = std::minmax_element(max_depths.begin(), max_depths.end());

    std::cout << "\n================ STABILITY RESULTS FOR CLUSTER " << cluster_num << " ================\n";
    std::cout << "Anchors Tested:       " << num_anchors_in_ensemble << "\n";
    std::cout << "Mean Entropic Volume: " << std::fixed << std::setprecision(2) << mean << " bits\n";
    std::cout << "Standard Deviation:   " << std::fixed << std::setprecision(2) << stddev << " bits\n";
    std::cout << "Spread (Min to Max):  " << *minmax.first << " to " << *minmax.second 
              << " (" << (*minmax.second - *minmax.first) << " bit diff)\n";
    std::cout << "-----------------------------------------------------------------\n";
    std::cout << "Mean Max Depth:       " << std::fixed << std::setprecision(2) << depth_mean << " tiles\n";
    std::cout << "Max Depth Std. Dev:   " << std::fixed << std::setprecision(2) << depth_stddev << " tiles\n";
    std::cout << "Max Depth Spread:     " << *depth_minmax.first << " to " << *depth_minmax.second << " tiles\n";
    std::cout << "=================================================================\n";

    // 5. Deferred Pareto Anisotropy Output
    std::cout << "\n================ PARETO ANISOTROPY ANALYSIS ================\n";
    for (int a = 0; a < num_anchors_in_ensemble; ++a) {
        std::cout << "\n  --- Pareto Analysis for Anchor " << a;
        if (a == max_anchor_idx) std::cout << " **MAX VOL ESTIMATE**";
        std::cout << " ---\n";
        
        double cumulative_real_volume = 0.0;
        for (size_t k = 0; k < top_rays_per_anchor[a].size(); ++k) {
            double ray_real_vol = std::pow(2.0, top_rays_per_anchor[a][k].net_bits);
            cumulative_real_volume += ray_real_vol;
            
            double percent_of_total = (ray_real_vol / volume_accumulators[a]) * 100.0;
            double cumulative_percent = (cumulative_real_volume / volume_accumulators[a]) * 100.0;
            
            std::cout << "    Top Ray #" << (k+1) << " (ID: " << std::setw(5) << top_rays_per_anchor[a][k].id << ") "
                    << "| Depth: " << std::setw(4) << top_rays_per_anchor[a][k].length << " "
                    << "| Vol: 2^" << std::fixed << std::setprecision(2) << std::setw(6) << top_rays_per_anchor[a][k].net_bits << " "
                    << "| Contrib: " << std::setw(5) << std::fixed << std::setprecision(2) << percent_of_total << "% "
                    << "(Cumulative: " << cumulative_percent << "%)\n";
        }
    }

    // Average Anisotropy of Peripheral (Non-Winner) Anchors
    if (num_anchors_in_ensemble > 1) {
        std::cout << "\n  --- Averaged Pareto Anisotropy for Peripheral Anchors (Non-Winners) ---\n";
        double cumulative_avg_percent = 0.0;
        int non_winner_count = num_anchors_in_ensemble - 1;

        for (size_t k = 0; k < 10 && k < RAYS_PER_ANCHOR; ++k) {
            double sum_percent = 0.0;
            double sum_real_vol = 0.0;

            for (int a = 0; a < num_anchors_in_ensemble; ++a) {
                if (a == max_anchor_idx) continue; // Skip the winner
                
                if (k < top_rays_per_anchor[a].size()) {
                    double ray_real_vol = std::pow(2.0, top_rays_per_anchor[a][k].net_bits);
                    sum_real_vol += ray_real_vol;
                    sum_percent += (ray_real_vol / volume_accumulators[a]) * 100.0;
                }
            }

            double avg_percent = sum_percent / non_winner_count;
            cumulative_avg_percent += avg_percent;
            double avg_vol_bits = std::log2(sum_real_vol / non_winner_count);

            std::cout << "    Average Ray #" << (k+1) << " "
                      << "| Avg Vol: 2^" << std::fixed << std::setprecision(2) << std::setw(6) << avg_vol_bits << " "
                      << "| Avg Contrib: " << std::setw(5) << std::fixed << std::setprecision(2) << avg_percent << "% "
                      << "(Cumulative: " << cumulative_avg_percent << "%)\n";
        }
    }

    std::cout << "============================================================\n";
}

void run_starburst_branching_decay_estimation(int grid_size, int cluster_num) {
    JeandelRaoTileSet tileset;
    std::vector<Tile> alphabet = tileset.get_tiles();
    ChaCha20PRNG master_rng;

    int TARGET_DEFECTS = 165;
    int MIN_FLOOR = 160;
    int MAX_CEILING = 170;
    int RAYS_TO_CAST = 5000; // 1,000 is plenty for a high-res density heatmap

    std::cout << "--- Initializing Local Branching Decay Experiment for Cluster #" << cluster_num << " ---" << std::endl;

    // 1. Generate the Origin Anchor
    std::uniform_int_distribution<uint64_t> nonce_dist;
    QwtssPrivateKey master_key;
    int master_key_defects;
    bool mk_success = false;
    
    while (!mk_success) {
        uint64_t rnd_nonce = nonce_dist(master_rng);
        try {
            master_key = build_qwtss_private_key(generate_random_username(master_rng), rnd_nonce, 0,
                grid_size, TARGET_DEFECTS - 2, TARGET_DEFECTS + 2, 0.7f, true);
        } catch (const std::exception& e) {
            std::cout << "[EXCEPTION] Master key gen failed:\n" << e.what() << std::endl;
            continue;
        }
        master_key_defects = count_grid_defects(master_key.private_key.data(), grid_size, alphabet);
        if (master_key_defects >= MIN_FLOOR && master_key_defects <= MAX_CEILING) mk_success = true;
    }
    std::vector<int> G_Origin = master_key.private_key;
    std::cout << "Origin Anchor established with " << master_key_defects << " defects. Firing " << RAYS_TO_CAST << " path-recording rays...\n";

    std::ofstream csv("branching_decay_map_extended_" + std::to_string(cluster_num) + ".csv");
    csv << "Ray_ID,Step_N,Valid_Moves_b,Raw_Bits,Comm_Bits,Net_Bits\n";

    double total_real_volume_accumulator = 0.0;

    // Generate a strictly unique 256-bit seed (8x 32-bit words) for every single ray before threading
    std::vector<std::array<uint32_t, 8>> thread_seeds(RAYS_TO_CAST);
    std::uniform_int_distribution<uint32_t> seed_dist;
    
    for (int i = 0; i < RAYS_TO_CAST; ++i) {
        for (int j = 0; j < 8; ++j) {
            thread_seeds[i][j] = seed_dist(master_rng); 
        }
    }

    int target_threads = std::max(1, omp_get_num_procs() * 8 / 10);
    
    // 2. Fire the Rays
    #pragma omp parallel for num_threads(target_threads) schedule(dynamic, 10)
    for (int i = 0; i < RAYS_TO_CAST; ++i) {
        ChaCha20PRNG local_rng(thread_seeds[i]);

        // Implements the exact Rosenbluth Markov Chain for a self-avoiding walk
        // Execute with record_path = true
        StarburstResult tip = execute_radial_starburst_walk_with_exact_correction(
            G_Origin, alphabet, grid_size, MIN_FLOOR, MAX_CEILING, local_rng, true
        );

        #pragma omp critical
        {
            // Dump the entire step-by-step history of this ray
            for (size_t n = 0; n < tip.branching_path.size(); ++n) {
                int d_out = tip.branching_path[n];
                int d_in  = tip.in_degree_path[n];
                
                // Calculate the exact local topological dynamics
                double raw_bits = (d_out > 0) ? std::log2(static_cast<double>(d_out)) : 0.0;
                double comm_bits = std::log2(static_cast<double>(d_in));
                double net_bits = raw_bits - comm_bits;

                csv << i << "," 
                    << n << "," 
                    << d_out << "," 
                    << std::fixed << std::setprecision(6) << raw_bits << "," 
                    << comm_bits << "," 
                    << net_bits << "\n";
            }

            // Accumulate terminal volume for the running LogSumExp estimate
            double ray_total_net_bits = tip.raw_branching_bits - tip.log2_local_commutativity;
            total_real_volume_accumulator += std::pow(2.0, ray_total_net_bits);

            if (i % 20 == 0 || i == RAYS_TO_CAST - 1) {
                int current_m = i + 1;
                double current_est_bits = std::log2(total_real_volume_accumulator / current_m);

                std::cout << "Rays Cast: " << std::setw(4) << current_m 
                          << " / " << RAYS_TO_CAST 
                          << " | Last Ray Depth: " << std::setw(3) << tip.length
                          << " | Est. Cluster Size: 2^" << std::fixed << std::setprecision(2) << current_est_bits << " bits    \r" << std::flush;
            }
        }
    }

    std::cout << "\n\nExperiment complete. Data saved to branching_decay_map_" << cluster_num << ".csv\n";
    csv.close();
}

void run_empirical_cluster_diameter_estimation(int grid_size, int cluster_num, int target_defects, int target_defects_tolerance) {
    JeandelRaoTileSet tileset;
    std::vector<Tile> alphabet = tileset.get_tiles();
    ChaCha20PRNG master_rng;

    int MIN_FLOOR = 160;
    int MAX_CEILING = 170;
    if (target_defects < MIN_FLOOR || target_defects > MAX_CEILING) throw std::invalid_argument("target_defects is outside allowed range");
    int master_key_defect_min = std::max(MIN_FLOOR, target_defects - target_defects_tolerance);
    int master_key_defect_max = std::min(MAX_CEILING, target_defects + target_defects_tolerance);

    // 20,000 is enough to get an accurate volume estimate and find the deepest branches
    int RAYS_TO_CAST_STOCHASTIC = 20000; // 5000; // 10000;
    // Note: purely stochastic walks find deeper branches than greedy heuristics do

    // How many of the absolute longest terminal tips to preserve for pairwise comparisons
    int TOP_K_TIPS = 300;

    std::cout << "--- Initializing Empirical Diameter (R*) Estimation For Cluster #" << cluster_num << " ---" << std::endl;

    // 1. Generate the Origin Anchor
    std::uniform_int_distribution<uint64_t> nonce_dist;
    QwtssPrivateKey master_key;
    int master_key_defects;
    bool mk_success = false;
    
    while (!mk_success) {
        uint64_t rnd_nonce = nonce_dist(master_rng);
        try {
            master_key = build_qwtss_private_key(generate_random_username(master_rng), rnd_nonce, 0,
                grid_size, master_key_defect_min, master_key_defect_max, 0.7f, true);
        } catch (const std::exception& e) {
            std::cout << "[EXCEPTION] Master key gen failed:\n" << e.what() << std::endl;
            continue;
        }
        if (master_key.private_key.empty()) continue;   // Key gen failed; retry
        master_key_defects = master_key.defect_count;
        if (master_key_defects >= MIN_FLOOR && master_key_defects <= MAX_CEILING) mk_success = true;
    }
    std::vector<int> G_Origin = master_key.private_key;

    std::cout << "Origin Anchor established with " << master_key_defects << " defects. Firing "
              << RAYS_TO_CAST_STOCHASTIC << " rays to locate deep branches...\n";

    // Accumulators
    double stoc_accum = 0.0;
    int stoc_max = 0;
    std::vector<StarburstResult> top_stoc_rays;

    // 2. Fire the Rays
    int target_threads = std::max(1, omp_get_num_procs() * 8 / 10);
    #pragma omp parallel for num_threads(target_threads) schedule(dynamic, 100)
    for (int i = 0; i < RAYS_TO_CAST_STOCHASTIC; ++i) {
        try {
            ChaCha20PRNG local_rng; 

            // Thread-Local Sandbox: Force a local copy of the grid so concurrent walks don't shred the shared heap
            std::vector<int> thread_local_grid = G_Origin;

            // Execute walk (record_path = false to save memory)
            StarburstResult tip = execute_radial_starburst_walk_with_exact_correction(thread_local_grid, alphabet,
                grid_size, MIN_FLOOR, MAX_CEILING, local_rng, false);

            #pragma omp critical
            {
                // Calculate Volume
                double net_bits = tip.raw_branching_bits - tip.log2_local_commutativity;
                double real_vol = std::pow(2.0, net_bits);
                
                {
                    stoc_accum += real_vol;
                    if (tip.length > stoc_max) stoc_max = tip.length;

                    // Dynamically maintain Top K Stochastic rays
                    if (top_stoc_rays.size() < TOP_K_TIPS) {
                        top_stoc_rays.push_back(tip);
                        if (top_stoc_rays.size() == TOP_K_TIPS) {
                            std::sort(top_stoc_rays.begin(), top_stoc_rays.end(), [](const StarburstResult& a, const StarburstResult& b) {
                                return a.length > b.length;
                            });
                        }
                    } else if (!top_stoc_rays.empty() && tip.length > top_stoc_rays.back().length) {
                        top_stoc_rays.pop_back();
                        top_stoc_rays.push_back(tip);
                        std::sort(top_stoc_rays.begin(), top_stoc_rays.end(), [](const StarburstResult& a, const StarburstResult& b) {
                            return a.length > b.length;
                        });
                    }
                }

                // Progress Bar
                if ((i + 1) % 100 == 0 || i == RAYS_TO_CAST_STOCHASTIC - 1) {
                    std::cout << "Rays Cast: " << std::setw(5) << (i + 1) << " / " << RAYS_TO_CAST_STOCHASTIC << "\r" << std::flush;
                }
            } // End of #pragma omp critical
        } catch (const std::exception& e) {
            #pragma omp critical
            {
                std::cerr << "\n[THREAD FATAL ERROR] Ray " << i << " crashed: " << e.what() << "\n";
            }
        } catch (...) {
            #pragma omp critical
            {
                std::cerr << "\n[UNKNOWN THREAD ERROR] Ray " << i << " segfaulted\n";
            }
        }
    }

    // 3. Post-Process the Global Cluster Data
    double stoc_est_bits = std::log2(stoc_accum / RAYS_TO_CAST_STOCHASTIC);

    // 4. Pairwise Hamming Analysis on the Top Spines (Isolated by Category)
    int max_d_stoc = 0, min_d_stoc = grid_size * grid_size;
    
    double sum_stoc = 0;
    int count_stoc = 0;
    std::vector<int> dists_stoc;

    // Stochastic-vs-Stochastic
    if (RAYS_TO_CAST_STOCHASTIC > 0){
        for (size_t i = 0; i < top_stoc_rays.size(); ++i) {
            if (top_stoc_rays[i].terminal_grid.size() != grid_size * grid_size){
                std::cout << "Warning: top_stoc_rays contained an improper grid size" << std::endl;
                continue;
            }

            for (size_t j = i + 1; j < top_stoc_rays.size(); ++j) {
                if (top_stoc_rays[j].terminal_grid.size() != grid_size * grid_size){
                    std::cout << "Warning: top_stoc_rays contained an improper grid size" << std::endl;
                    continue;
                }

                int dist = 0;
                for (int t = 0; t < grid_size * grid_size; ++t) {
                    if (top_stoc_rays[i].terminal_grid[t] != top_stoc_rays[j].terminal_grid[t]) dist++;
                }
                dists_stoc.push_back(dist);
                if (dist > max_d_stoc) max_d_stoc = dist;
                if (dist < min_d_stoc) min_d_stoc = dist;
                sum_stoc += dist;
                count_stoc++;
            }
        }
    }

    if (count_stoc == 0) min_d_stoc = 0;

    double avg_stoc = count_stoc > 0 ? sum_stoc / count_stoc : 0;

    double var_stoc = 0;
    for (int d : dists_stoc) var_stoc += (d - avg_stoc) * (d - avg_stoc);
    
    double std_stoc = count_stoc > 0 ? std::sqrt(var_stoc / count_stoc) : 0;

    // 5. Export Data to CSV
    std::string filename = "cluster_" + std::to_string(cluster_num) + "_stoch_terminal_dists.csv";
    std::ofstream csv(filename);
    csv << "Ray_A_Length,Ray_B_Length,Hamming_Distance\n";
    
    int idx = 0;
    // Write Stochastic Pairs
    for (size_t i = 0; i < top_stoc_rays.size(); ++i) {
        for (size_t j = i + 1; j < top_stoc_rays.size(); ++j) {
            csv << top_stoc_rays[i].length << "," << top_stoc_rays[j].length << "," 
                << dists_stoc[idx++] << "\n";
        }
    }
    csv.close();

    // 6. Print Console Report
    std::cout << "\n\n========================= CLUSTER " << cluster_num << " DIAMETER ESTIMATION =========================\n";
    
    std::cout << std::left << std::setw(30) << "Metric" 
              << std::right << std::setw(20) << "Stochastic" << "\n";
    std::cout << std::string(50, '-') << "\n";

    std::cout << std::left << std::setw(30) << "Est. Volume (bits)" << std::fixed << std::setprecision(2)
              << std::right << std::setw(20) << stoc_est_bits << "\n";
    std::cout << std::left << std::setw(30) << "Max Ray Radius (N)" 
              << std::right << std::setw(20) << stoc_max << "\n";
    std::cout << std::left << std::setw(30) << "Valid Tip Pairs Evaluated" 
              << std::right << std::setw(20) << count_stoc << "\n";
              
    std::cout << std::string(50, '-') << "\n";
    std::cout << "Pairwise Hamming Stats (R* Diameter Intra-Cluster):\n";
    std::cout << std::left << std::setw(30) << "Max Distance" 
              << std::right << std::setw(20) << max_d_stoc << "\n";
    std::cout << std::left << std::setw(30) << "Min Distance" 
              << std::right << std::setw(20) << min_d_stoc << "\n";
    std::cout << std::left << std::setw(30) << "Average Distance" << std::fixed << std::setprecision(2)
              << std::right << std::setw(20) << avg_stoc << "\n";
    std::cout << std::left << std::setw(30) << "Std Dev" 
              << std::right << std::setw(20) << std_stoc << "\n";
    std::cout << "======================================================================\n";

    // 7. Append Meta-Data to Shared Global CSV
    std::string macro_filename = "global_cluster_stoch_macro_stats.csv";
    std::ifstream check_file(macro_filename);
    bool write_header = !check_file.good();
    check_file.close();

    std::ofstream macro_csv(macro_filename, std::ios::app);
    if (write_header) {
        macro_csv << "Cluster_ID,Cluster_Defect_Count,"
                  << "Total_Rays_Stoc,"
                  << "Top_Spines_Stoc,"
                  << "Pairs_Stoc,"
                  << "Vol_Bits_Stoc,"
                  << "Max_Rad_Stoc,"
                  << "Max_Dia_Stoc,"
                  << "Min_Dia_Stoc,"
                  << "Avg_Dia_Stoc,"
                  << "Std_Dev_Stoc\n";
    }

    macro_csv << cluster_num << "," << master_key_defects << ","
              << RAYS_TO_CAST_STOCHASTIC << ","
              << top_stoc_rays.size() << ","
              << count_stoc << ","
              << std::fixed << std::setprecision(4) << stoc_est_bits << ","
              << stoc_max << ","
              << max_d_stoc << ","
              << min_d_stoc << ","
              << avg_stoc << ","
              << std_stoc << "\n";

    macro_csv.close();
}
