#pragma once
#include "Tiles.h"
#include "StarkTrace.h"
#include <vector>
#include <cstdint>
#include <stdexcept>

// Forward Declarations
bool run_tiling_cryptanalysis_unit_tests();
void run_private_key_vulnerability_analysis(int mode);
void run_marginal_entropy_analysis(int sample_rank = 1, int grid_size = 64);
void run_ais_joint_entropy_analysis(bool do_partial_boundary_sweep, int sample_rank = 1, int grid_size = 64);
void run_3metric_topological_analysis(int sample_rank = 1, int grid_size = 64);

void export_comparable_db_keys_to_ppm(const std::string& db_file1, const std::string& db_file2, 
    int target_defects, int mode, ITileSet* tileset);
void export_db_keys_to_ppm(const std::string& db_file, int target_defects, int max_count,
    int mode, ITileSet* tileset, bool has_extended_data = false);

void run_local_rigidity_sampling_test();
void run_streaming_evt_analysis_deep(int grid_size = 64);
void run_streaming_evt_analysis_scout(int grid_size = 64, int num_trajectories = 100);
void run_energy_barrier_analysis(int mode = 0, int grid_size = 64);


// Defines an "Anchor" where a fragment of the Public Key perfectly matches the Oracle
struct BoundaryAnchor {
    int oracle_x;
    int oracle_y;
    int pk_offset;       // Where on the 64-length boundary this match starts
    int match_length;    // How many consecutive colors matched perfectly
    enum Edge { NORTH, SOUTH, EAST, WEST } edge;
};

// Represents the statistical results of a single phase-recovery attack
struct CryptanalysisMetrics {
    int true_defect_count = 0;
    int tiles_correctly_predicted = 0;
    double entropy_reduction_bits = 0;
    double entropy_reduction_percent = 0;
    double remaining_entropy_bits = 0;
    int anchors_found = 0;
    int min_match_length = 0;
    bool is_spatially_filtered = false;
};

class OracleEngine {
private:
    std::vector<int> oracle_grid; // The massive, perfect 1D-flattened grid (e.g., 1024 x 1024)
    int oracle_size;              // N for the NxN grid
    const std::vector<Tile>& alphabet;
    const int alphabet_size;

public:
    OracleEngine(int target_size, const std::vector<Tile>& alphabet);

    // Retrieves the raw tile ID from the generated Oracle grid
    int get_tile_id(int x, int y) const {
        return oracle_grid[y * oracle_size + x];
    }

    Tile get_alphabet_tile(int tile_id) const {
        if (tile_id >= 0 && tile_id < alphabet_size) {
            return alphabet[tile_id];
        }
        throw std::runtime_error("Invalid tile id: " + std::to_string(tile_id));
    }

    const std::vector<Tile>& get_alphabet() const {
        return alphabet;
    }

    std::vector<int> get_oracle_grid() {
        return oracle_grid;
    }

    // Phase 1: Inflate the Jeandel-Rao perfect plane using Labbé's 2D substitution rules
    void generate_perfect_plane_jeandel_rao();

    void generate_perfect_plane_ammann16();

    void generate_random_plane();

    // Verify the generated Oracle contains exactly 0 topological defects
    int get_oracle_defects_count();

    // Phase 2: CPU orchestration for the GPU boundary search
    std::vector<BoundaryAnchor> find_anchors(const QwtssPublicInputs& pub_key, int min_match_length);

    // Phase 3: Flood the interior based on the anchors until collisions occur
    std::vector<uint16_t> project_and_intersect(const std::vector<BoundaryAnchor>& anchors);

    // Phase 4: Compare against the private key to get the security curve data
    CryptanalysisMetrics evaluate_vulnerability(const std::vector<uint16_t>& predicted_grid, const std::vector<int>& true_private_grid);
};

// CUDA Kernel Dispatcher for the sliding window search
std::vector<BoundaryAnchor> launch_cuda_anchor_search(
    const int* d_oracle, int oracle_size, 
    const FieldElement* h_boundary, int boundary_len, 
    int min_match_length, BoundaryAnchor::Edge edge_type,
    const std::vector<Tile>& alphabet
);

void execute_topology_attack_simulation(int loop_iterations, int grid_size = 64);

void run_deep_surrogate_profiling(int grid_size, int cluster_num);
void run_starburst_volume_estimation(int grid_size, int cluster_num);
void run_cluster_volume_est_ensemble(int grid_size, int cluster_num, int num_anchors_in_ensemble = 10);
void run_starburst_branching_decay_estimation(int grid_size, int cluster_num);
void run_empirical_cluster_diameter_estimation(int grid_size, int cluster_num, int target_defects = 165, int target_defects_tolerance = 2);
