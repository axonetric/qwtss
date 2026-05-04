#pragma once
#include "Tiles.h"
#include "QwtssCoreShared.h"    // Needed only for GridAnnealResult definition
#include "QwtssAirTypes.h"
#include "LabbeJR11Oracle.h"
#include <iomanip>
#include <random>
#include <array>


struct QwtssPrivateKey {
    // Will be empty if key gen failed.
    std::vector<int> private_key;
    // The allowed behavior mask: 0 = Free, 1 = Pinned Outward Colors, 2 = Pinned Tile ID. If empty, defaults to all tiles Free.
    std::vector<uint8_t> boundary_mask;

    // The oracle grid A used in the PK splicing. Used for "alien tile" calcs.
    std::vector<int> plane_A;
    // The oracle grid B used in the PK splicing. Used for "alien tile" calcs.
    std::vector<int> plane_B;

    int defect_count = 0;
    int alien_tile_count = 0;   // -1 if not determined yet
    int core32_defect_tile_count = 0;
    int line_max_defect_count = 0;

    // The computational work performed (how many times heat_bath_kernel was called on the grid). This value is returned even on failures.
    int work_steps = 0;
};

struct PkDerivedFields {
    LabbeOraclePublicKey oracle_pk;
    // The allowed behavior mask: 0 = Free, 1 = Pinned Outward Colors, 2 = Pinned Tile ID. If empty, defaults to all tiles Free.
    std::vector<uint8_t> boundary_mask;
};

enum AnnealMode {
    // Fixed point integer MCMC math. Replaces expf() with a dynamic LUT and utilizes branchless heuristics.
    // Recommended. Guarantees 100% cross-platform deterministic key generation and achieves the highest performance.
    FAST,
    // Legacy floating-point MCMC math with heuristic shortcuts. Faster than adiabatic mode.
    // Subject to floating-point non-determinism across different hardware architectures.
    ORIGINAL,
    // Floating-point MCMC math following a strict, unaccelerated adiabatic cooling schedule.
    // No heuristic shortcuts. Slowest key generation time and subject to floating-point non-determinism.
    ADIABATIC
};

inline std::string anneal_mode_to_string(AnnealMode mode){
    if (mode == AnnealMode::FAST) return "FAST";
    else if (mode == AnnealMode::ADIABATIC) return "ADIABATIC";
    else return "?";
}

enum AnnealDevice {
    CPU,
    GPU,
    AUTO
};

inline std::string anneal_device_to_string(AnnealDevice dev){
    if (dev == AnnealDevice::CPU) return "CPU";
    else if (dev == AnnealDevice::GPU) return "GPU";
    else if (dev == AnnealDevice::AUTO) return "AUTO";
    else return "?";
}

bool detect_eligible_nvidia_gpu(bool do_logging = false);

PkDerivedFields get_pk_derived_fields(const std::string& username, uint64_t identity_nonce, uint8_t version,
    int grid_size, float keep_boundary_percentage, bool do_logging
);

QwtssPrivateKey build_qwtss_private_key(std::string username, uint64_t identity_nonce,
    AnnealMode anneal_mode = FAST, AnnealDevice anneal_device = AUTO, int target_cpu_threads = -1, bool do_logging = false
);

QwtssPrivateKey build_qwtss_private_key(std::string username, uint64_t identity_nonce, int min_defect_count,
    int max_defect_count, AnnealMode anneal_mode = FAST, AnnealDevice anneal_device = AUTO, int target_cpu_threads = -1, bool do_logging = false
);

QwtssPrivateKey build_qwtss_private_key(std::string username, uint64_t identity_nonce, uint8_t version, int grid_size,
    int min_defect_count, int max_defect_count, float keep_boundary_percentage, bool do_greedy_prequench,
    AnnealMode anneal_mode = FAST, AnnealDevice anneal_device = AUTO, int target_cpu_threads = -1, bool do_logging = false
);

std::array<std::string, 2> get_serialized_grid_fingerprint(const QwtssPrivateKey& private_key);

bool run_qwtss_full_pipeline(std::string username, uint64_t identity_nonce);

GridAnnealResult grid_anneal_gpu(
    int* h_grid, 
    int grid_size,
    ITileSet* tileset, 
    GridAnnealParams params,
    ChaCha20PRNG& rng,
    const std::vector<uint8_t>& external_lock_mask = {}
);

GridAnnealResult grid_anneal_gpu_adiabatic(
    int* h_grid, 
    int grid_size,
    ITileSet* tileset, 
    GridAnnealParams params,
    ChaCha20PRNG& rng,
    const std::vector<uint8_t>& external_lock_mask
);


// Fast inline converter
inline std::vector<uint8_t> downcast_private_key(const std::vector<int>& private_key) {
    std::vector<uint8_t> byte_private_key(private_key.size());
    for (size_t i = 0; i < private_key.size(); ++i) {
        // Safe downcast since JR-11 tile IDs are just 0-10
        byte_private_key[i] = static_cast<uint8_t>(private_key[i]); 
    }
    return byte_private_key;
}

void run_spatial_distribution_analysis(int min_defect_count, int max_defect_count, int grid_size);

void run_hyperparameter_optimization(int grid_size = 64);

enum PrivateKeyGenMode {
    UNCONSTRAINED,
    QWTSS_STANDARD,
};

void build_private_key_db(ITileSet* tileset, int min_defect_count, int max_defect_count,
    bool do_greedy_prequench, int grid_size, PrivateKeyGenMode key_gen_mode = QWTSS_STANDARD,
    float keep_boundary_percentage = 1.0f, std::string filename = "qwtss_private_keys.db"
);

std::array<std::string, 2> serialize_fingerprint(const std::array<FieldElement256, 2>& grid_fingerprint);
std::array<FieldElement256, 2> deserialize_fingerprint(const std::array<std::string, 2>& fingerprint_limbs);
std::string encode_grid_to_hex(const std::vector<int>& private_grid);
std::vector<int> decode_grid_from_hex(const std::string& hex_str);
