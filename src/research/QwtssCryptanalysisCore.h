#include "Tiles.h"
#include "QwtssCoreShared.h"
#include <iomanip>
#include <random>
#include <cmath>

// Hide device-specific includes from the standard C++ compiler
#ifdef __CUDACC__
#include <curand_kernel.h>
#include <device_launch_parameters.h>
#endif

struct MarginalEntropyMetrics {
    int defect_count = 0;
    int num_samples = 0;
    double entropy_bits = 0.0;
    double max_local_entropy = 0.0;
    double std_dev_local_entropy = 0.0;
    // Calculates the average marginal entropy per row and column individually to isolate the absolute minimum.
    double fault_line_entropy;
    // The variance of the 64 row means (Vrow​) and the 64 column means (Vcol​). max(V_row, V_col) epresents the severity of the structural banding.
    double banding_intensity;
    // Anisotropy Index: +1.0 = Pure Horizontal Banding, -1.0 = Pure Vertical Banding, 0.0 = Perfectly Isotropic/Symmetric
    double directional_bias;
};

MarginalEntropyMetrics calculate_ensemble_marginal_entropy_metrics(
    const std::vector<int>& ground_truth_grid,
    int grid_size,
    int ground_truth_defects,
    int defect_count_tolerance,
    int samples_required,
    ITileSet* tileset,
    ChaCha20PRNG& rng,
    bool do_greedy_prequench = true,
    const std::vector<uint8_t>& public_key_mask = {}
);

// Annealed Importance Sampling (AIS) for True Joint Entropy
struct AISMetrics {
    int target_defects;
    double joint_entropy;   // The estimated log2 of the Partition Function (True Joint Entropy)
    double marginal_bound;  // Unconstrained starting entropy for comparison
    double ess;             // Effective Sample Size
    double max_weight;      // Min-Entropy indicator
    int survived_chains;    // How many chains survived/contributed?
};

// Annealed Importance Sampling (AIS) for True Joint Entropy
AISMetrics calculate_ais_joint_entropy(
    const std::vector<int>& ground_truth_grid,
    int grid_size,
    int ground_truth_defects,
    int defect_count_tolerance,
    int num_ais_chains,         // Number of parallel chains to run
    ITileSet* tileset,
    ChaCha20PRNG& rng,
    const std::vector<uint8_t>& public_key_mask = {}
);

// Tracks the online arithmetic mean and variance for a single variable.
class WelfordAccumulator {
private:
    int count = 0;
    double mean = 0.0;
    double M2 = 0.0;

public:
    void add_sample(double value) {
        count++;
        double delta = value - mean;
        mean += delta / count;
        double delta2 = value - mean;
        M2 += delta * delta2;
    }

    double get_mean() const { return mean; }
    
    // Uses Bessel's correction (N-1) for an unbiased sample variance
    double get_variance() const {
        if (count < 2) return 0.0;
        return M2 / (count - 1);
    }
    
    double get_std_dev() const {
        return std::sqrt(get_variance());
    }
    
    int get_count() const { return count; }
};

struct TopologyMetrics {
    double gcc_mass_fraction;
    double core3_mass_fraction;
    double normalized_cyclomatic_complexity;
};

// The unified tracker for ensemble data
struct TopologyEnsembleTracker {
    WelfordAccumulator gcc;
    WelfordAccumulator core3;
    WelfordAccumulator cyclomatic;

    // Feed it the raw metrics from a single grid
    void push(const TopologyMetrics& m) {
        gcc.add_sample(m.gcc_mass_fraction);
        core3.add_sample(m.core3_mass_fraction);
        cyclomatic.add_sample(m.normalized_cyclomatic_complexity);
    }
};

TopologyEnsembleTracker calculate_ensemble_topological_metrics(
    const std::vector<int>& ground_truth_grid,
    int grid_size,
    int ground_truth_defects,
    int defect_count_tolerance,
    int samples_required,
    ITileSet* tileset,
    ChaCha20PRNG& rng,
    bool do_greedy_prequench,
    const std::vector<uint8_t>& public_key_mask = {}
);

#ifdef __CUDACC__
    typedef curandStatePhilox4_32_10_t prng_state_t;
#else
    typedef void prng_state_t; 
#endif

std::vector<int> generate_boundary_conditioned_private_key_hoisted(const std::vector<int>& ground_truth_grid, 
    int ground_truth_defects, const std::vector<uint8_t>& public_key_mask, int grid_size, GridAnnealParams params,
    int *d_grid, prng_state_t* d_states, ITileSet* tileset, ChaCha20PRNG& rng);

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
);

void run_custom_topology_attack_sweep(
    const std::vector<int>& plane_A,
    const std::vector<int>& plane_B,
    const std::vector<int>& plane_C,
    const std::vector<int>& boundary_source_map, // length: 4*grid_size - 4, values: 0 for A, 1 for B
    int grid_size,
    ITileSet* tileset,
    ChaCha20PRNG& rng
);
