#include "QwtssCore.h"
#include "QwtssCoreShared.h"
#include "QwtssCryptanalysis.h"
#include "QwtssCryptanalysisCore.h"
#include "QwtssExtra.h"
#include "LabbeJR11Oracle.h"
#include "PrivateKeyDatabase.h"
#include <iostream>
#include "gflags/gflags.h"
#include <glog/logging.h>


// Forward declarations
int differentiate_attacker_vs_real();


int main(int argc, char** argv) {
    // Initialize Google Logging to silence the WARNING message
    google::InitGoogleLogging(argv[0]); 
    // Optional: Ensure logs still print to the console instead of a file
    FLAGS_logtostderr = true;

    setenv("OMP_STACKSIZE", "64M", 1);

    std::cout << "============================== QWTSS RESEARCH ==============================" << std::endl;

    JeandelRaoTileSet tileset;
    //Ammann16TileSet tileset;
    ChaCha20PRNG rng;

    {
        LabbeJR11Oracle oracle;
        oracle.batch_export_jr11_grids_to_csv(1000, 32, 0.25);
        return 0;
    }

    // Isolated private key gen
    if (false)
    {
        int min_defect_count = 80;
        int max_defect_count = 85;
        AnnealMode anneal_mode = ADIABATIC;

        QwtssPrivateKey sk = build_qwtss_private_key(generate_random_username(rng), 0,
            min_defect_count, max_defect_count, anneal_mode);
        if (sk.private_key.empty()) {
            std::cout << "Failed to generate private key." << std::endl;
            return 0;
        }
    }

    // Build private key db
    {
        //std::string filename = "../external/qwtss_standard_private_keys xxx.db";
        //build_private_key_db(&tileset, 160, 170, true, 64, QWTSS_STANDARD, 0.70f);
    }

    // Basic empirical analysis
    {
        //run_local_rigidity_sampling_test();
        //run_marginal_entropy_analysis(1);
        //run_ais_joint_entropy_analysis(false, 1);

        //int sample_rank = 2;
        //run_3metric_topological_analysis(sample_rank);
    }

    // State space analysis
    {
        // EVT analysis experiments
        //run_streaming_evt_analysis_scout();
        //run_streaming_evt_analysis_deep();

        // Energy Barrier experiments
        //run_energy_barrier_analysis(3);
    }

    // Solution cluster characterizations
    {
        //for (int cluster_num = 1; cluster_num <= 30; cluster_num++)
        {
            //run_deep_surrogate_profiling(64, cluster_num);
            //run_starburst_volume_estimation(64, cluster_num);
            //run_cluster_volume_est_ensemble(64, cluster_num, 10);
            //run_starburst_branching_decay_estimation(64, cluster_num);

            // int target_defects = 160 + ((cluster_num - 1) % 11);
            // run_empirical_cluster_diameter_estimation(64, cluster_num, target_defects, 1);
        }
    }

    // Various crytanalysis / vulnerability testing
    {
        //run_tiling_cryptanalysis_unit_tests();
        //run_private_key_vulnerability_analysis(2);
    }

    // Experiments for differentiation between synthetic attacker grids vs. true grids
    {
        //run_spatial_distribution_analysis(100, 340, 64);
        //execute_topology_attack_simulation(1000);
        //differentiate_attacker_vs_real();
    }

    // Grid analysis and visualizations
    {
        //QwtssPrivateKey private_key = build_qwtss_private_key(generate_random_username(rng), 0, difficulty_parameter_W, true);
        //export_defect_edges_to_ppm(private_key.private_key.data(), 64, &tileset, "private key edges.ppm");

        //std::string filename1 = "../external/jr11 qwtss private keys (1146) -- 50 to 600 (no doping).db";
        //std::string filename2 = "../external/jr11 qwtss_private_keys (538) (spliced).db";
        //export_comparable_db_keys_to_ppm(filename1, filename2, 305, 2, &tileset);

        //std::string filename = "../external/qwtss rebar_keys (40) - 160 to 170 defects (extended data format).db";
        //export_db_keys_to_ppm(filename, 66, 2, 2, &tileset, true);

        // std::string filename = "../external/qwtss_private_keys (1836) 70% boundary  - 160 to 170 defects.db";
        // analyze_tile_frequencies(filename);
        // return 0;
    }

    // Misc
    {
        //benchmark_cpu_greedy_search();
        //run_hyperparameter_optimization();

    }

    return 0;
}
