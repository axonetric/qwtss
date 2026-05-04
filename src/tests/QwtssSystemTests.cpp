#include "QwtssSystemTests.h"
#include "QwtssCore.h"
#include "QwtssCoreShared.h"
#include "StarkTrace.h"
#include "QwtssAir.h"
#include "QwtssAirTypes.h"
#include "QwtssConfig.h"
#include "PrivateKeyDatabase.h"
#include "PoseidonConstants.h"
#include "FastPoseidonHashCpu.h"
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
#include <array>
#include <string>
#include <gmp.h>
#include <gmpxx.h>


// Tests a full sabotage suite against the AIR verification.
bool run_qwtss_air_sabotage_test_suite() {

    auto start = std::chrono::high_resolution_clock::now();

    ChaCha20PRNG rng;

    std::string username = "test_user";
    uint64_t identity_nonce = 0;
    uint8_t version = 0;

    int grid_size = QwtssReference::grid_size;

    // Instantiate the Jeandel-Rao Tile Set
    JeandelRaoTileSet jr_tileset;
    std::vector<Tile> alphabet = jr_tileset.get_tiles();

    std::vector<uint8_t> message = {'H', 'E', 'L', 'L', 'O'};

    // Load exception key database (with has_extended_data=true)
    PrivateKeyDatabase exception_key_db;
    //exception_key_db.load_from_disk("/mnt/.../code/QWTSS/external/qwtss_exception_keys (alien tile count, Line_Max violators) - 160 to 170 defects (extended data format).db", true);
    exception_key_db.load_from_disk("../external/qwtss_exception_keys (alien tile count, Line_Max violators) - 160 to 170 defects (extended data format).db", true);
    int exception_key_count = exception_key_db.get_total_key_count();
    if (exception_key_count < 1) throw std::runtime_error("(exception_key_count < 1)");
    else std::cout << "Loaded exception key database: " << exception_key_count << " keys" << std::endl;
    //exception_key_db.print_summary();

    int tests_passed_good_accept_validator = 0;
    int tests_passed_good_reject_validator = 0;
    int tests_passed_good_accept_verifier = 0;
    int tests_passed_good_reject_verifier = 0;
    int tests_performed = 0;
    std::vector<std::string> failed_tests;
    failed_tests.reserve(25);

    int max_test_idx = 66;
    for (int test_idx = 0; test_idx <= (max_test_idx + exception_key_count); test_idx++)
    //for (int test_idx = 53; test_idx <= 57; test_idx++)
    {
        QwtssPrivateKey private_key{};
        bool is_exception_key = false;
        std::string exception_key_name = "";

        // Test all stored exception keys (at the appropriate test_idx values)
        if (test_idx > max_test_idx && test_idx <= (max_test_idx + exception_key_count)){
            is_exception_key = true;
            int exception_key_idx = (test_idx - max_test_idx) - 1;
            PrivateKey exception_key_from_db = exception_key_db[exception_key_idx];
            private_key.private_key = exception_key_from_db.grid_data;
            private_key.defect_count = exception_key_from_db.defect_count;
            private_key.boundary_mask = exception_key_from_db.boundary_mask;
            private_key.plane_A = exception_key_from_db.plane_A;
            private_key.plane_B = exception_key_from_db.plane_B;
            exception_key_name = exception_key_from_db.name;
        } else {
            // Generate new valid key dynamically

            // Try up to 5 times
            for (int t = 0; t < 5; t++){
                std::string username = generate_random_username(rng);
                private_key = build_qwtss_private_key(username, identity_nonce);

                if (private_key.private_key.empty()){
                    std::cout << "\nTry " << (t+1) << " failed: private key convergence failed" << std::endl;
                    continue;
                } else {
                    break;
                }
            }
        }

        if (private_key.private_key.empty()){
            std::cout << "Error: All attempts at private key convergence failed" << std::endl;
            return false;
        }

        std::vector<QwtssPrivateKey> private_keys = { private_key };
        // throw_preemptively = false here to allow attacks to proceed
        ExecutionTrace trace = build_execution_trace(private_keys, grid_size, alphabet, false);

        // Used by certain test cases below
        std::array<FieldElement256, 2> tmp_grid_fingerprint = {
            trace.expected_hash_0[trace.valid_steps + 63], // +63 for the final hash blanking rounds
            trace.expected_hash_1[trace.valid_steps + 63]
        };
        QwtssPublicInputs tmp_pub_inputs(username, identity_nonce, version, private_key,
            grid_size, tmp_grid_fingerprint, alphabet);

        std::string test_name = "";
        bool should_pass = false;
        bool use_tmp_pub_inputs = false;
        if (is_exception_key){
            int exception_key_num = (test_idx - max_test_idx);
            test_name = "Stored Exception Key Test " + std::to_string(exception_key_num) + " / " + std::to_string(exception_key_count) +
                " (" + exception_key_name + ")";
            should_pass = false;
        }
        else if (test_idx == 0){
            test_name = "Baseline Acceptance Test";
            should_pass = true;
        }
        // Category 1: Base State & Accumulator Math
        //These constraints verify that defects are tallied honestly row-by-row.
        else if (test_idx == 1){
            test_name = "c_acc_transition (accumulator step logic) Test";

            auto current_val = from_fe256(trace.defect_accumulator[1000]);
            auto corrupted_val = current_val + StoneField::FromUint(3);
            trace.defect_accumulator[1001] = to_fe256(corrupted_val);
        } else if (test_idx == 2){
            test_name = "c_slack_limit (dummy defects outside the slack zone) Test";

            trace.dummy_defect[10] = 1;
        } else if (test_idx == 3){
            test_name = "c_dummy_bool (dummy defects must be exactly 0 or 1) Test";

            trace.dummy_defect[5] = 2;
        }
        // Category 2: Grid Geometry & Matching
        // These ensure that tiles placed next to each other actually match if no defect is claimed.
        else if (test_idx == 4){
            test_name = "c_h_match (Horizontal edges must match if is_h_defect == 0) Test";
            
            // Sabotage Strategy: Find a perfect horizontal match, break it, but leave the defect flag at 0.
            int target_step = -1;
            int current_tile_id = -1;
            
            for (int i = 0; i < trace.valid_steps - 1; ++i) {
                // Ignore the right edge of the grid where horizontal matches wrap around
                if (i % 64 == 63) continue; 
                
                if (trace.is_h_defect[i] == 0) {
                    target_step = i;
                    // Identify which tile is currently sitting at target_step
                    for(size_t t = 0; t < alphabet.size(); ++t) {
                        if (trace.tile_booleans[t][target_step] == 1) {
                            current_tile_id = t;
                            break;
                        }
                    }
                    break;
                }
            }

            if (target_step != -1) {
                // Find a malicious tile from the alphabet that DOES NOT match the current tile's right edge
                int malicious_tile_id = -1;
                for(size_t t = 0; t < alphabet.size(); ++t) {
                    if (alphabet[t].left != alphabet[current_tile_id].right) {
                        malicious_tile_id = t;
                        break;
                    }
                }

                // Overwrite the next step (target_step + 1) with the malicious tile
                for(size_t t = 0; t < alphabet.size(); ++t) {
                    trace.tile_booleans[t][target_step + 1] = 0; // Wipe the honest tile
                }
                trace.tile_booleans[malicious_tile_id][target_step + 1] = 1; // Inject the malicious tile
                
                // Note: trace.is_h_defect[target_step] remains un-updated to simulate a dishonest prover 
            }

        } else if (test_idx == 5){
            test_name = "c_v_match (Vertical edges must match if is_v_defect == 0) Test";

            // Sabotage Strategy: Find a perfect vertical match, break the tile below it, and lie about the defect.
            int target_step = -1;
            int current_tile_id = -1;
            
            for (int i = 0; i < trace.valid_steps - 64; ++i) {
                if (trace.is_v_defect[i] == 0) {
                    target_step = i;
                    for(size_t t = 0; t < alphabet.size(); ++t) {
                        if (trace.tile_booleans[t][target_step] == 1) {
                            current_tile_id = t;
                            break;
                        }
                    }
                    break;
                }
            }

            if (target_step != -1) {
                // Find a malicious tile that DOES NOT match the current tile's bottom edge
                int malicious_tile_id = -1;
                for(size_t t = 0; t < alphabet.size(); ++t) {
                    if (alphabet[t].top != alphabet[current_tile_id].bottom) {
                        malicious_tile_id = t;
                        break;
                    }
                }

                // Overwrite the step below (target_step + 64) with the malicious tile
                for(size_t t = 0; t < alphabet.size(); ++t) {
                    trace.tile_booleans[t][target_step + 64] = 0; 
                }
                trace.tile_booleans[malicious_tile_id][target_step + 64] = 1; 
                
                // trace.is_v_defect[target_step] remains 0. 
            }

        } else if (test_idx == 6){
            test_name = "c_h_defect_bool (Must be exactly 0 or 1) Test";
            
            // Attacker Benefit: If an attacker can use a fractional or negative defect (like -1), 
            // they can artificially subtract from their total defect accumulator.
            // STARKs enforce booleans mathematically: x * (x - 1) == 0.
            trace.is_h_defect[100] = 3; 

        } else if (test_idx == 7){
            test_name = "c_v_defect_bool (Must be exactly 0 or 1) Test";
            
            trace.is_v_defect[100] = 2; 
        }
        // Category 3: Grid Boundaries (The Borders)
        // These ensure the outer perimeter of the 64x64 grid adheres to the fixed background colors.
        else if (test_idx == 8){
            test_name = "c_bound_north (Top row top-edges match v_north) Test";
            
            // Sabotage: Find a PINNED North edge and change the tile to one with the wrong top color
            QwtssPublicInputs pub_inputs(username, identity_nonce, version, private_key,
                grid_size, ZERO_FINGERPRINT, alphabet);
            int target_step = -1;
            int current_tile_id = -1;

            for (int i = 0; i < 64; ++i) {
                if (pub_inputs.north[i] != QwtssPublicInputs::WILDCARD_COLOR) {
                    target_step = i;
                    for (size_t t = 0; t < alphabet.size(); ++t) {
                        if (trace.tile_booleans[t][target_step] == 1) { current_tile_id = t; break; }
                    }
                    break;
                }
            }

            if (target_step != -1) {
                int malicious_tile_id = -1;
                for (size_t t = 0; t < alphabet.size(); ++t) {
                    if (alphabet[t].top != pub_inputs.north[target_step]) { malicious_tile_id = t; break; }
                }
                for (size_t t = 0; t < alphabet.size(); ++t) trace.tile_booleans[t][target_step] = 0;
                trace.tile_booleans[malicious_tile_id][target_step] = 1; 
            } else {
                // RNG generated no pinned tiles on this border? Trace is unaltered.
                should_pass = true; 
            }

        } else if (test_idx == 9){
            test_name = "c_bound_south (Bottom row bottom-edges match v_south) Test";

            QwtssPublicInputs pub_inputs(username, identity_nonce, version, private_key,
                grid_size, ZERO_FINGERPRINT, alphabet);
            int target_step = -1;
            int current_tile_id = -1;

            for (int i = trace.valid_steps - 64; i < trace.valid_steps; ++i) {
                int col = i % 64;
                if (pub_inputs.south[col] != QwtssPublicInputs::WILDCARD_COLOR) {
                    target_step = i;
                    for (size_t t = 0; t < alphabet.size(); ++t) {
                        if (trace.tile_booleans[t][target_step] == 1) { current_tile_id = t; break; }
                    }
                    break;
                }
            }

            if (target_step != -1) {
                int malicious_tile_id = -1;
                for (size_t t = 0; t < alphabet.size(); ++t) {
                    if (alphabet[t].bottom != pub_inputs.south[target_step % 64]) { malicious_tile_id = t; break; }
                }
                for (size_t t = 0; t < alphabet.size(); ++t) trace.tile_booleans[t][target_step] = 0;
                trace.tile_booleans[malicious_tile_id][target_step] = 1; 
            } else {
                // RNG generated no pinned tiles on this border? Trace is unaltered.
                should_pass = true; 
            }
        } else if (test_idx == 10){
            test_name = "c_bound_east (Right column right-edges match v_east) Test";

            QwtssPublicInputs pub_inputs(username, identity_nonce, version, private_key,
                grid_size, ZERO_FINGERPRINT, alphabet);
            int target_step = -1;
            int current_tile_id = -1;

            for (int i = 63; i < trace.valid_steps; i += 64) {
                int row = i / 64;
                if (pub_inputs.east[row] != QwtssPublicInputs::WILDCARD_COLOR) {
                    target_step = i;
                    for (size_t t = 0; t < alphabet.size(); ++t) {
                        if (trace.tile_booleans[t][target_step] == 1) { current_tile_id = t; break; }
                    }
                    break;
                }
            }

            if (target_step != -1) {
                int malicious_tile_id = -1;
                for (size_t t = 0; t < alphabet.size(); ++t) {
                    if (alphabet[t].right != pub_inputs.east[target_step / 64]) { malicious_tile_id = t; break; }
                }
                for (size_t t = 0; t < alphabet.size(); ++t) trace.tile_booleans[t][target_step] = 0;
                trace.tile_booleans[malicious_tile_id][target_step] = 1; 
            } else {
                should_pass = true; 
            }

        } else if (test_idx == 11){
            test_name = "c_bound_west (Left column left-edges match v_west) Test";

            QwtssPublicInputs pub_inputs(username, identity_nonce, version, private_key,
                grid_size, ZERO_FINGERPRINT, alphabet);
            int target_step = -1;
            int current_tile_id = -1;

            for (int i = 0; i < trace.valid_steps; i += 64) {
                int row = i / 64;
                if (pub_inputs.west[row] != QwtssPublicInputs::WILDCARD_COLOR) {
                    target_step = i;
                    for (size_t t = 0; t < alphabet.size(); ++t) {
                        if (trace.tile_booleans[t][target_step] == 1) { current_tile_id = t; break; }
                    }
                    break;
                }
            }

            if (target_step != -1) {
                int malicious_tile_id = -1;
                for (size_t t = 0; t < alphabet.size(); ++t) {
                    if (alphabet[t].left != pub_inputs.west[target_step / 64]) { malicious_tile_id = t; break; }
                }
                for (size_t t = 0; t < alphabet.size(); ++t) trace.tile_booleans[t][target_step] = 0;
                trace.tile_booleans[malicious_tile_id][target_step] = 1; 
            }
            else {
                should_pass = true; 
            }
        } else if (test_idx == 12){
            test_name = "Unpinned Edge Tolerance Test (should pass if internal geometry is intact)";
            
            // Strategy: Find an UNPINNED North edge. Swap the tile. 
            // Dynamically check if the new tile matches the internal neighbors (West, East, South).
            // If it does, the Verifier MUST accept it. If it doesn't, the Verifier MUST reject it.
            QwtssPublicInputs pub_inputs(username, identity_nonce, version, private_key,
                grid_size, ZERO_FINGERPRINT, alphabet);
            int target_step = -1;
            int current_tile_id = -1;
            int replacement_tile_id = -1;

            for (int i = 0; i < 64; ++i) {
                if (pub_inputs.north[i] == QwtssPublicInputs::WILDCARD_COLOR) {
                    target_step = i;
                    for (size_t t = 0; t < alphabet.size(); ++t) {
                        if (trace.tile_booleans[t][target_step] == 1) { current_tile_id = t; break; }
                    }
                    
                    // Find a replacement tile with a DIFFERENT top color to prove the boundary doesn't restrict it
                    for (size_t t = 0; t < alphabet.size(); ++t) {
                        if (t != current_tile_id && alphabet[t].top != alphabet[current_tile_id].top) { 
                            replacement_tile_id = t; 
                            break; 
                        }
                    }
                    if (replacement_tile_id != -1) break;
                }
            }

            if (target_step != -1) {
                // Apply the swap
                for (size_t t = 0; t < alphabet.size(); ++t) trace.tile_booleans[t][target_step] = 0;
                trace.tile_booleans[replacement_tile_id][target_step] = 1; 

                // Evaluate the internal geometry damage dynamically
                Tile old_tile = alphabet[current_tile_id];
                Tile new_tile = alphabet[replacement_tile_id];
                bool introduced_defect = false;

                // Since we are on the North row (step < 64), we only need to check West, East, and South neighbors
                if (target_step % 64 != 0 && new_tile.left != old_tile.left) introduced_defect = true;
                if (target_step % 64 != 63 && new_tile.right != old_tile.right) introduced_defect = true;
                if (new_tile.bottom != old_tile.bottom) introduced_defect = true;

                // Dynamically override the should_pass expectation
                if (introduced_defect) {
                    should_pass = false; // Internal geometric constraints verify this rejection
                } else {
                    should_pass = true;  // A perfectly legal move. Verifier acceptance required.
                }
            } else {
                should_pass = true; 
            }
        }
        // Category 4: The Start / End Boundary Anchors
        // These lock the start and end of the trace to public inputs or constants.
        else if (test_idx == 13){
            test_name = "c_acc_start (Defect accumulator starts at 0) Test";
            
            // Sabotage: The prover tries to start with a non-zero accumulator to offset later defects.
            // Safely casts the 252-bit Field Element into the 256-bit storage array.
            trace.defect_accumulator[0] = to_fe256(StoneField::FromInt(-1));

        } else if (test_idx == 14){
            test_name = "c_acc_end (Defect accumulator ends exactly at the upper bound) Test";
            
            // Sabotage: The prover submits a grid that fails to reach the required defect target.
            // Note: We target valid_steps - 1 (row 4095) where s_last evaluates.
            trace.defect_accumulator[trace.valid_steps - 1] = to_fe256(StoneField::FromUint(999));

        } else if (test_idx == 15){
            test_name = "c_hash_start (Hash starts at 0) Test";
            
            // Sabotage: Injecting a fake initial state into the Poseidon Capacity chain.
            trace.hash_state_0[0] = to_fe256(StoneField::FromUint(1));

        } else if (test_idx == 16){
            test_name = "c_hash_end (Fingerprint matches the public key) Test";
            
            // Sabotage: The grid is perfectly valid, but it generates the WRONG fingerprint. 
            // Simulated via a single-bit flip in the final 256-bit hash capacity state.
            // Since FieldElement256 is an std::array<uint64_t, 4>, just XOR the first 64-bit chunk.
            // Targets expected_hash_0 at the new s_identity_last index (end of blanking rounds).
            trace.expected_hash_0[trace.valid_steps + 63][0] ^= 1;
        } else if (test_idx == 17){
            test_name = "c_alien_end (Total alien tiles matches the target) Test";
            
            // Sabotage: The prover did not use enough Alien tiles (or lied about the count).
            // Since alien_acc is a standard 64-bit integer array, assign it directly.
            // Target valid_steps - 1 (row 4095) where s_last evaluates.
            trace.alien_acc[trace.valid_steps - 1] = QwtssReference::alien_tiles_lower_bound - 1;
        }
        // Category 5: Cryptography & Hashes
        else if (test_idx == 18){
            test_name = "c_poseidon_hash (The S-Box step logic) Test";
            
            // Sabotage Strategy: The prover attempts to manipulate the Capacity chain mid-execution.
            // Even if the start (step 0) and end (step 4096) are correct, breaking the chain
            // at step 4090 should cause the entire polynomial to fail.
            auto current_hash = from_fe256(trace.hash_state_0[4090]);
            auto corrupted_hash = current_hash + StoneField::FromUint(1);
            trace.hash_state_0[4090] = to_fe256(corrupted_hash);
        }
        // Category 6: Zero-Knowledge Alien Verification
        else if (test_idx == 19){
            test_name = "c_alien_diff (Alien accumulator increments by exactly 0 or 1) Test";
            
            // Sabotage Strategy: Find a step where an alien tile is legally placed, 
            // but jump the accumulator by 2 to artificially inflate the total alien count.
            for (size_t i = 0; i < trace.valid_steps - 1; ++i) {
                if (trace.alien_acc[i + 1] > trace.alien_acc[i]) {
                    // We found an alien placement. Corrupt the increment.
                    trace.alien_acc[i + 1] = trace.alien_acc[i] + 2;
                    break;
                }
            }
        } else if (test_idx == 20){
            test_name = "c_alien_inv (The ZK Math trick enforcing tile inequality) Test";
            
            // Sabotage Strategy: The prover places an alien tile but refuses to provide 
            // the cryptographic inverse that proves the tile ID does not belong to the standard set.
            for (size_t i = 0; i < trace.valid_steps - 1; ++i) {
                if (trace.alien_acc[i + 1] > trace.alien_acc[i]) {
                    // The ZK constraint is: diff * (tile_id - A) * (tile_id - B) * inv == diff
                    // By forcing inv to 0, the left side evaluates to 0, but the right side is 1.
                    // Triggers immediate Verifier rejection.
                    trace.alien_inv[i] = to_fe256(StoneField::Zero());
                    break;
                }
            }
        }
        // Category 7: Dynamic Alphabet Arrays (The Loops)
        // These ensure the one-hot encoding represents exactly one valid tile.
        else if (test_idx == 21){
            test_name = "Dynamic Boolean Check (Every tile boolean is 0 or 1) Test";
            
            // Sabotage: We inject a "2" into the boolean array. 
            // The algebraic constraint x * (x - 1) == 0 will fail.
            trace.tile_booleans[0][500] = 2;

        } else if (test_idx == 22){
            test_name = "Exclusivity Check (b_sum == 1) - Empty Void Test";
            
            // Sabotage: Wipes all tile configurations at step 100.
            // The sum evaluates to 0, meaning an empty void exists on the grid.
            for (size_t t = 0; t < alphabet.size(); ++t) {
                trace.tile_booleans[t][100] = 0;
            }

        } else if (test_idx == 23){
            test_name = "Exclusivity Check (b_sum == 1) - Stacked Tiles Test";
            
            // Sabotage: Forces two tiles to exist on the same square simultaneously.
            // The sum evaluates to 2, causing the constraint to fail.
            for (size_t t = 0; t < alphabet.size(); ++t) trace.tile_booleans[t][100] = 0; // Wipe first
            trace.tile_booleans[0][100] = 1;
            trace.tile_booleans[1][100] = 1;
        }
        // Category 8: Core32 Lookahead Constraints
        // Lookahead offsets (+64) require strict boundary validation to prevent out-of-bounds shadowing.
        else if (test_idx == 24){
            test_name = "Core32 Diff Boolean Test";
            
            // Sabotage: Increment the core accumulator by an invalid amount (e.g., 2 instead of 0 or 1).
            trace.core32_acc[150] = trace.core32_acc[149] + 2;

        } else if (test_idx == 25){
            test_name = "Core32 Validity Check Test";
            
            // Sabotage: Find a legitimate defect inside the Core32 zone and erase the core32_acc increment.
            // This simulates an attacker trying to "hide" a core defect from the final tally.
            bool did_sabotage = false;
            for (size_t i = 0; i < trace.valid_steps - 1; i++) {
                if (trace.core32_acc[i + 1] > trace.core32_acc[i]) {
                    trace.core32_acc[i + 1] = trace.core32_acc[i];
                    did_sabotage = true;
                    break;
                }
            }
            if (!did_sabotage) throw std::runtime_error("No legitimate defect in the core32 zone? Should not be possible.");

        } else if (test_idx == 26){
            test_name = "Core32 Boundary Test";
            
            // Sabotage: The trace is completely valid, but the prover lies about the final total.
            // Target valid_steps - 1 (row 4095) where s_last evaluates.
            trace.core32_acc[trace.valid_steps - 1] = 21;
        }
        // Category 9: Line_Max Seam Restrictions (Defect Wall Prevention)
        // These tests verify that the orthogonal seam accumulators strictly cap 
        // the number of defects per row/column, specifically targeting the edges.
        else if (test_idx == 27){
            test_name = "Line_Max: Horizontal Start Anchor Bypass (Free Defects) Test";
            
            // Sabotage Strategy: We target Row 5, Column 0. It should start at zero.
            int step = 5 * 64 + 0; 
            trace.h_seam_acc[step] = 1;

        } else if (test_idx == 28){
            test_name = "Line_Max: Internal Seam Packing (Transition Lie) Test";
            
            // Sabotage Strategy: The attacker places a defect mid-seam but freezes the 
            // accumulator, hoping the transition constraint misses the increment.
            int step = 100; // Row 1, Column 36
            trace.is_v_defect[step] = 1;
            // Force the next step's accumulator to equal the current one (diff = 0).
            // The AIR expects diff - defect == 0. Here, 0 - 1 = -1. Constraint evaluation will fail.
            trace.h_seam_acc[step + 1] = trace.h_seam_acc[step];

        } else if (test_idx == 29){
            test_name = "Line_Max: Horizontal 64th Tile Overload (c == 63) Test";
            
            // Sabotage Strategy: The attacker legally uses all allowed defects by column 62.
            // At column 63, the accumulator correctly reads MAX. The attacker then places 
            // one more defect on the final 64th tile.
            // Validates the "Inclusive End Anchor" enforcement at the perimeter.
            int row = 10;
            int step = row * 64 + 63;
            trace.h_seam_acc[step] = QwtssReference::line_max_upper_bound;
            trace.is_v_defect[step] = 1; 

        } else if (test_idx == 30){
            test_name = "Line_Max: Vertical 64th Tile Overload (r == 63) Test";
            
            // Sabotage Strategy: Same as above, but targeting the bottom edge of the grid.
            // This ensures vertical defect walls can't be hidden at the bottom of the board.
            int col = 15;
            int step = 63 * 64 + col;
            trace.v_seam_acc[step] = QwtssReference::line_max_upper_bound;
            trace.is_h_defect[step] = 1;

        } else if (test_idx == 31){
            test_name = "Line_Max: Vertical Seam End Limit Bypass Test";
            
            // Sabotage Strategy: The attacker accrues MAX + 1 defects on a vertical seam 
            // and honestly reports it to the end of the line, hoping the final anchor 
            // check isn't strict enough to catch the overflow.
            int col = 20;
            int step = 63 * 64 + col;
            trace.v_seam_acc[step] = QwtssReference::line_max_upper_bound + 1;
            trace.is_h_defect[step] = 0; 
            
        } else if (test_idx == 32){
            test_name = "Line_Max: Internal Seam Packing (The Amnesia Attack) Test";
            
            // Sabotage Strategy: The attacker plays fairly for the first 53 tiles, 
            // racking up defects. Then, they reset the accumulator to 0, wiping out 
            // their history so they can pack more defects into the final tiles 
            // without hitting the Line_Max upper bound.
            int step = 5 * 64 + 54; 
            trace.h_seam_acc[step] = 0;
        }
        // Constraint Shadowing Tests
        // The Dummy Defect Shadows (for Tests 2 & 3)
        // The Physical Defect Shadows (for Tests 6 & 7)
        else if (test_idx == 33){
            test_name = "c_slack_limit (dummy defects outside the slack zone) Constraint Shadowing Test";
            
            // Sabotage: Place a dummy defect at step 10 (which is the first step outside the allowed slack zone).
            int step = 10;
            trace.dummy_defect[step] = 1;
            
            // EQUATION BALANCING: Artificially increments the subsequent accumulator
            // so Constraint 0 passes and allows the validator to reach Constraint 16.
            auto curr_acc = from_fe256(trace.defect_accumulator[step]);
            auto balanced_next_acc = curr_acc + StoneField::FromUint(trace.is_h_defect[step] + trace.is_v_defect[step] + 1);
            trace.defect_accumulator[step + 1] = to_fe256(balanced_next_acc);

        } else if (test_idx == 34){
            test_name = "c_dummy_bool (dummy defects must be exactly 0 or 1) Constraint Shadowing Test";
            
            // Sabotage: Set the dummy defect to an illegal value of 2.
            int step = 100;
            trace.dummy_defect[step] = 2;

            // BALANCE THE EQUATION: Absorb the illegal +2 into the next accumulator.
            auto curr_acc = from_fe256(trace.defect_accumulator[step]);
            auto balanced_next_acc = curr_acc + StoneField::FromUint(trace.is_h_defect[step] + trace.is_v_defect[step] + 2);
            trace.defect_accumulator[step + 1] = to_fe256(balanced_next_acc);
        }
        else if (test_idx == 35){
            test_name = "c_h_defect_bool (Must be exactly 0 or 1) Constraint Shadowing Test";
            
            // Find a step where the geometry perfectly matches (is_h_defect == 0)
            int target_step = 100;
            for(int i = 0; i < trace.valid_steps; i++) {
                if (trace.is_h_defect[i] == 0) { target_step = i; break; }
            }
            
            trace.is_h_defect[target_step] = 2; // Sabotage
            
            // BALANCE THE EQUATION
            auto curr_acc = from_fe256(trace.defect_accumulator[target_step]);
            auto balanced_next_acc = curr_acc + StoneField::FromUint(2 + trace.is_v_defect[target_step] + trace.dummy_defect[target_step]);
            trace.defect_accumulator[target_step + 1] = to_fe256(balanced_next_acc);

        } else if (test_idx == 36){
            test_name = "c_v_defect_bool (Must be exactly 0 or 1) Constraint Shadowing Test";
            
            int target_step = 100;
            for(int i = 0; i < trace.valid_steps; i++) {
                if (trace.is_v_defect[i] == 0) { target_step = i; break; }
            }
            
            trace.is_v_defect[target_step] = 2; // Sabotage
            
            // BALANCE THE EQUATION
            auto curr_acc = from_fe256(trace.defect_accumulator[target_step]);
            auto balanced_next_acc = curr_acc + StoneField::FromUint(trace.is_h_defect[target_step] + 2 + trace.dummy_defect[target_step]);
            trace.defect_accumulator[target_step + 1] = to_fe256(balanced_next_acc);
        }
        // Constraint Shadowing Tests
        // The Line_Max 64th Tile Shadows (for Tests 29 & 30)
        else if (test_idx == 37){
            test_name = "Line_Max: Horizontal 64th Tile Overload (c == 63) Constraint Shadowing Test";
            
            // Sabotage Strategy: Leave the geometry and standard accumulators perfectly intact.
            // Instead, directly force the horizontal seam accumulator to exceed the limit 
            // at the exact step where the boundary anchor evaluates.
            int step = 10 * 64 + 63; // Row 10, Col 63
            trace.h_seam_acc[step] = QwtssReference::line_max_upper_bound + 1;

        } else if (test_idx == 38){
            test_name = "Line_Max: Vertical 64th Tile Overload (r == 63) Constraint Shadowing Test";
            
            int step = 63 * 64 + 15; // Row 63, Col 15
            trace.v_seam_acc[step] = QwtssReference::line_max_upper_bound + 1;
        }
        // Constraint Shadowing Tests
        // The Boundary Shadows (for Tests 8–11)
        else if (test_idx == 39){
            test_name = "c_bound_north (Top row top-edges match v_north) Constraint Shadowing Test";
            
            // Sabotage: The trace is perfect. We corrupt the public key requirement!
            bool sabotaged = false;
            for (int c = 0; c < 64; ++c) {
                if (tmp_pub_inputs.north[c] != QwtssPublicInputs::WILDCARD_COLOR) {
                    tmp_pub_inputs.north[c] = tmp_pub_inputs.north[c] + 1; // Corrupt the target color
                    use_tmp_pub_inputs = true;
                    sabotaged = true;
                    break;
                }
            }
            if (!sabotaged) should_pass = true;

        } else if (test_idx == 40){
            test_name = "c_bound_south (Bottom row bottom-edges match v_south) Constraint Shadowing Test";

            bool sabotaged = false;
            for (int c = 0; c < 64; ++c) {
                if (tmp_pub_inputs.south[c] != QwtssPublicInputs::WILDCARD_COLOR) {
                    tmp_pub_inputs.south[c] = tmp_pub_inputs.south[c] + 1;
                    use_tmp_pub_inputs = true;
                    sabotaged = true;
                    break;
                }
            }
            if (!sabotaged) should_pass = true;

        } else if (test_idx == 41){
            test_name = "c_bound_east (Right column right-edges match v_east) Constraint Shadowing Test";

            bool sabotaged = false;
            for (int r = 0; r < 64; ++r) {
                if (tmp_pub_inputs.east[r] != QwtssPublicInputs::WILDCARD_COLOR) {
                    tmp_pub_inputs.east[r] = tmp_pub_inputs.east[r] + 1;
                    use_tmp_pub_inputs = true;
                    sabotaged = true;
                    break;
                }
            }
            if (!sabotaged) should_pass = true;

        } else if (test_idx == 42){
            test_name = "c_bound_west (Left column left-edges match v_west) Constraint Shadowing Test";

            bool sabotaged = false;
            for (int r = 0; r < 64; ++r) {
                if (tmp_pub_inputs.west[r] != QwtssPublicInputs::WILDCARD_COLOR) {
                    tmp_pub_inputs.west[r] = tmp_pub_inputs.west[r] + 1;
                    use_tmp_pub_inputs = true;
                    sabotaged = true;
                    break;
                }
            }
            if (!sabotaged) should_pass = true;
        }
        // Constraint Shadowing Tests
        // The Alien Inverse Shadow (for Test 20)
        // The Dynamic Boolean Shadows (for Tests 21 & 22)
        else if (test_idx == 43){
            test_name = "c_alien_inv (The ZK Math trick enforcing tile inequality) Constraint Shadowing Test";
            
            // Sabotage: Find the first placed alien tile and wipe its mathematical inverse.
            // This isolates Constraint 18 without touching physical defects or colors.
            for (size_t i = 0; i < trace.valid_steps - 1; ++i) {
                if (trace.alien_acc[i + 1] > trace.alien_acc[i]) {
                    trace.alien_inv[i] = to_fe256(StoneField::Zero());
                    break;
                }
            }
        } else if (test_idx == 44){
            test_name = "Dynamic Boolean Check (Every tile boolean is 0 or 1) Constraint Shadowing Test";
            
            // Sabotage: We exploit the bottom-right corner (Step 4095) where s_h and s_v 
            // mathematically disable geometry checks. The geometry will pass, isolating 
            // the boolean failure!
            int blind_spot_step = 4095;
            trace.tile_booleans[0][blind_spot_step] = 2;

        } else if (test_idx == 45){
            test_name = "Exclusivity Check (b_sum == 1) - Empty Void Constraint Shadowing Test";
            
            // Sabotage: Wipe out all tiles in the bottom-right corner.
            int blind_spot_step = 4095;
            for (size_t t = 0; t < alphabet.size(); ++t) {
                trace.tile_booleans[t][blind_spot_step] = 0;
            }
        }
        // Constraint Shadowing Tests
        // The Hash End Shadow (for Test 16)
        else if (test_idx == 46){
            test_name = "c_hash_end (Fingerprint matches the public key) Constraint Shadowing Test";
            
            // Sabotage: The trace is mathematically perfect. Instead, we corrupt the 
            // Public Key fingerprint. The transition math will pass, isolating the anchor.
            tmp_pub_inputs.grid_fingerprint[0][0] ^= 1;
            use_tmp_pub_inputs = true;
        }
        // Sponge Hash Rate Element Sabotage Test Cases
        else if (test_idx == 47){
            test_name = "c_hash_start_1 (Rate 1 starts at 0) Test";
            
            // Sabotage: Injecting a fake initial state into the first Rate element.
            // This attempts to bypass the zero-initialization of the sponge.
            trace.hash_state_1[0] = to_fe256(StoneField::FromUint(1));

        } else if (test_idx == 48){
            test_name = "c_hash_start_2 (Rate 2 starts at 0) Test";
            
            // Sabotage: Injecting a fake initial state into the second Rate element.
            trace.hash_state_2[0] = to_fe256(StoneField::FromUint(1));

        } else if (test_idx == 49){
            test_name = "c_poseidon_hash_1 (Rate 1 S-Box/MDS logic) Test";
            
            // Sabotage Strategy: The prover attempts to overwrite the Rate 1 element mid-execution,
            // trying to inject fake horizontal defect entropy into the sponge.
            auto current_hash = from_fe256(trace.hash_state_1[4090]);
            auto corrupted_hash = current_hash + StoneField::FromUint(1);
            trace.hash_state_1[4090] = to_fe256(corrupted_hash);

        } else if (test_idx == 50){
            test_name = "c_poseidon_hash_2 (Rate 2 S-Box/MDS logic) Test";

            // Sabotage Strategy: The prover attempts to overwrite the Rate 2 element mid-execution,
            // trying to inject fake vertical defect entropy into the sponge.
            auto current_hash = from_fe256(trace.hash_state_2[4090]);
            auto corrupted_hash = current_hash + StoneField::FromUint(1);
            trace.hash_state_2[4090] = to_fe256(corrupted_hash);
        }
        // Core32 Intermediate Columns Sabotage Tests
        else if (test_idx == 51) {
            test_name = "c_core32_validity (Out-of-Bounds Illusion) Constraint Shadowing Test";
            
            // Sabotage: Attacker claims a defect outside the core, and steals a legitimate defect inside the core
            // to keep the final accumulator perfectly balanced at 20 (bypassing the boundary check).
            
            int step_out = -1; // Target outside the core
            int step_in = -1;  // Legitimate defect inside the core to steal

            for (int i = 0; i < 4000; ++i) {
                int r = i / 64;
                int c = i % 64;
                // Using the exact +64 shifted lookahead logic from the AIR
                bool in_core = (r >= 15 && r < 47 && c >= 16 && c < 48);
                
                if (!in_core && step_out == -1) step_out = i;
                if (in_core && trace.core_defect[i] == 1 && step_in == -1) step_in = i;
            }

            if (step_out != -1 && step_in != -1) {
                // 1. Forge the intermediate column outside the core (Passes Const 1 & 2 because s_core32 = 0)
                trace.core_defect[step_out] = 1;

                // 2. Rebuild the accumulator to balance the books (Bypasses Const 3 & Const 5)
                std::vector<uint64_t> orig_acc = trace.core32_acc;
                uint64_t fake_running_total = 0;
                
                for (int i = 0; i <= 4096; i++) {
                    trace.core32_acc[i] = fake_running_total;
                    if (i < 4096) {
                        bool claim_defect = false;
                        if (i == step_out) claim_defect = true;   // Inject fake defect
                        else if (i == step_in) claim_defect = false; // Steal real defect
                        else {
                            // Maintain original valid transitions
                            if (orig_acc[i+1] > orig_acc[i]) claim_defect = true;
                        }
                        if (claim_defect) fake_running_total++;
                    }
                }
            } else {
                std::cerr << "Test skipped: Could not find valid target steps for Test 34." << std::endl;
            }

        } else if (test_idx == 52) {
            test_name = "c_core32_half_check (Inside-Job Forgery) Constraint Shadowing Test";
            
            // Sabotage: Attacker invents a defect on a perfect tile inside the core.
            // They smartly fake both intermediate columns to bypass the secondary defect check.
            
            int step_perfect = -1;   // Perfect tile inside core
            int step_defective = -1; // Defective tile inside core

            for (int i = 0; i < 4000; ++i) {
                int r = i / 64;
                int c = i % 64;
                bool in_core = (r >= 15 && r < 47 && c >= 16 && c < 48);
                
                if (in_core) {
                    if (trace.core_defect[i] == 0 && step_perfect == -1) step_perfect = i;
                    if (trace.core_defect[i] == 1 && step_defective == -1) step_defective = i;
                }
            }

            if (step_perfect != -1 && step_defective != -1) {
                // 1. Forge the final defect column 
                trace.core_defect[step_perfect] = 1;
                
                // 2. FORGERY: Forge the half_check column to 0.
                // This mathematically bypasses `c_core32_defect_check` because 1 - (0 * R * B) = 1.
                trace.half_check[step_perfect] = 0; 

                // 3. Rebuild the accumulator to balance the books (Bypasses Const 3 & Const 5)
                std::vector<uint64_t> orig_acc = trace.core32_acc;
                uint64_t fake_running_total = 0;
                
                for (int i = 0; i <= 4096; i++) {
                    trace.core32_acc[i] = fake_running_total;
                    if (i < 4096) {
                        bool claim_defect = false;
                        if (i == step_perfect) claim_defect = true;    // Inject fake defect
                        else if (i == step_defective) claim_defect = false; // Steal real defect
                        else {
                            if (orig_acc[i+1] > orig_acc[i]) claim_defect = true;
                        }
                        if (claim_defect) fake_running_total++;
                    }
                }
            } else {
                std::cerr << "Test skipped: Could not find valid target steps for Test 35." << std::endl;
            }
        }
        // Alien Inverse Intermediate Column Sabotage Test
        else if (test_idx == 53) {
            test_name = "c_alien_diff_A_B_check (Oracle Forgery) Constraint Shadowing Test";
            
            // Sabotage: Attacker wants to count an Oracle-matching tile as an "Alien" tile.
            // They fake the intermediate diff_A_B column to a non-zero value (1), allowing them 
            // to provide a fake valid inverse (1). They steal a real alien tile count elsewhere 
            // to keep the final accumulator perfectly balanced at exactly 1500.

            int step_oracle = -1;
            int step_real_alien = -1;

            // Step A: Dynamically scan the grid for the perfect targets
            for (int i = 0; i < 4000; ++i) {
                // Determine the actual Tile ID at this step
                int t_id = -1;
                for (size_t k = 0; k < alphabet.size(); ++k) {
                    if (trace.tile_booleans[k][i] == 1) {
                        t_id = k;
                        break;
                    }
                }

                bool is_oracle = (t_id == tmp_pub_inputs.plane_A[i] || t_id == tmp_pub_inputs.plane_B[i]);
                bool is_counted_alien = (!is_oracle && trace.alien_acc[i+1] > trace.alien_acc[i]);
                
                if (is_oracle && step_oracle == -1) step_oracle = i;
                if (is_counted_alien && step_real_alien == -1) step_real_alien = i;
            }

            if (step_oracle != -1 && step_real_alien != -1) {
                // 1. Forge the intermediate diff_A_B column AND the inverse for the Oracle tile!
                // Since actual_diff is 0, we forge it to 1. We provide an inverse of 1 (since 1 * 1 = 1).
                // This mathematically bypasses c_alien_inv!
                trace.diff_A_B[step_oracle] = to_fe256(StoneField::One());
                trace.alien_inv[step_oracle] = to_fe256(StoneField::One());

                // 2. Rebuild the alien_acc to balance the books (Bypasses c_alien_end and c_alien_diff)
                std::vector<uint64_t> orig_acc = trace.alien_acc;
                uint64_t fake_running_total = 0;
                
                for (int i = 0; i <= 4096; i++) {
                    trace.alien_acc[i] = fake_running_total;
                    if (i < 4096) {
                        bool claim_alien = false;
                        if (i == step_oracle) claim_alien = true;           // Inject fake alien count
                        else if (i == step_real_alien) claim_alien = false; // Steal real alien count to balance
                        else {
                            if (orig_acc[i+1] > orig_acc[i]) claim_alien = true; // Maintain valid transitions
                        }
                        
                        if (claim_alien) fake_running_total++;
                    }
                }
            } else {
                std::cerr << "Test skipped: Could not find valid target steps for Oracle Forgery." << std::endl;
            }
        }
        // c_alien_start Sabotage
        else if (test_idx == 54) {
            test_name = "c_alien_start (Alien accumulator starts at 0) Constraint Shadowing Test";

            // Sabotage: Start the accumulator at 5 to get free alien tiles.
            int free_aliens = 5;
            std::vector<uint64_t> orig_acc = trace.alien_acc;
            uint64_t fake_running_total = free_aliens;
            int stolen_count = 0;

            for (int i = 0; i <= 4096; i++) {
                trace.alien_acc[i] = fake_running_total;
                if (i < 4096) {
                    bool claim_alien = false;
                    // Was there a real alien transition here?
                    if (orig_acc[i+1] > orig_acc[i]) {
                        if (stolen_count < free_aliens) {
                            stolen_count++; // Reallocate legitimate defect
                            trace.is_alien[i] = 0; // Fix the flattened transition column to hide the theft
                        } else {
                            claim_alien = true;
                        }
                    }
                    if (claim_alien) fake_running_total++;
                }
            }
        }
        // c_core32_out_of_core_zeroing Sabotage
        else if (test_idx == 55) {
            test_name = "c_core32_out_of_core_zeroing Constraint Test";

            // Sabotage: Inject a core defect intermediate flag OUTSIDE the 32x32 core.
            int step_out = -1;
            for (int i = 0; i < 4000; ++i) {
                int r = i / 64;
                int c = i % 64;
                bool in_core = (r >= 15 && r < 47 && c >= 16 && c < 48);
                if (!in_core) {
                    step_out = i;
                    break;
                }
            }

            if (step_out != -1) {
                trace.core_defect[step_out] = 1;
            } else {
                should_pass = true;
            }
        }
        // c_core32_start_anchor Sabotage
        else if (test_idx == 56) {
            test_name = "c_core32_start_anchor Constraint Shadowing Test";

            // Sabotage: Start the core32_acc at 2 to get a head start.
            int head_start = 2;
            std::vector<uint64_t> orig_acc = trace.core32_acc;
            uint64_t fake_running_total = head_start;
            int stolen_count = 0;

            for (int i = 0; i <= 4096; i++) {
                trace.core32_acc[i] = fake_running_total;
                if (i < 4096) {
                    bool claim_defect = false;
                    if (orig_acc[i+1] > orig_acc[i]) {
                        if (stolen_count < head_start) {
                            stolen_count++; // Steal the defect to balance the final accumulator
                        } else {
                            claim_defect = true;
                        }
                    }
                    if (claim_defect) fake_running_total++;
                }
            }
        }
        // c_poseidon_math_0 (Flattened S-Box) Sabotage
        else if (test_idx == 57) {
            test_name = "c_poseidon_math_0 (Flattened S-Box Math) Constraint Shadowing Test";

            // Sabotage: Corrupt the flattened expected_hash_0 at step 100.
            int target_step = 100;
            auto corrupted_hash = from_fe256(trace.expected_hash_0[target_step]) + StoneField::One();
            
            trace.expected_hash_0[target_step] = to_fe256(corrupted_hash);
            
            // To prevent c_poseidon_hash_0 (transition) from failing and shadowing the math error, 
            // we set the next state to the exact same corrupted value. 
            // This isolates the mathematical S-Box verification check at step 100!
            trace.hash_state_0[target_step + 1] = to_fe256(corrupted_hash);
        }
        // -------------------------------------------------------------------------
        // t=4 (504-bit) Poseidon Hash Expansion Sabotage Tests
        // -------------------------------------------------------------------------
        else if (test_idx == 58) {
            test_name = "c_hash_start_3 (Capacity 2 starts at 0) Test";
            
            // Sabotage: Injecting a fake initial state into the 4th element (Capacity 2).
            trace.hash_state_3[0] = to_fe256(StoneField::FromUint(1));

        } else if (test_idx == 59) {
            test_name = "c_poseidon_hash_3 (Capacity 2 Transition Logic) Test";

            // Sabotage Strategy: The prover attempts to overwrite the Capacity 2 element mid-execution.
            auto current_hash = from_fe256(trace.hash_state_3[4090]);
            auto corrupted_hash = current_hash + StoneField::FromUint(1);
            trace.hash_state_3[4090] = to_fe256(corrupted_hash);

        } else if (test_idx == 60) {
            test_name = "c_poseidon_math_1 (Step Index / Domain Separator Forgery) Constraint Shadowing Test";

            // Sabotage Strategy: The attacker tries to bypass domain separation by claiming a different 
            // step index during the S-Box absorption in Rate 2.
            int target_step = 100;
            auto corrupted_hash = from_fe256(trace.expected_hash_1[target_step]) + StoneField::One();

            // Forge the expected hash so the S-Box check fails here
            trace.expected_hash_1[target_step] = to_fe256(corrupted_hash);

            // SHADOWING PREVENTION: Forge the next state so the transition constraint (c_poseidon_hash_1)
            // sees a perfect match and passes, explicitly isolating the math check.
            trace.hash_state_1[target_step + 1] = to_fe256(corrupted_hash);

        } else if (test_idx == 61) {
            test_name = "c_poseidon_math_2 (Capacity 1 S-Box Math) Constraint Shadowing Test";

            // Sabotage: Corrupt the flattened expected_hash_2 at step 100.
            int target_step = 100;
            auto corrupted_hash = from_fe256(trace.expected_hash_2[target_step]) + StoneField::One();
            
            trace.expected_hash_2[target_step] = to_fe256(corrupted_hash);
            trace.hash_state_2[target_step + 1] = to_fe256(corrupted_hash); // Shadowing prevention

        } else if (test_idx == 62) {
            test_name = "c_poseidon_math_3 (Capacity 2 S-Box Math) Constraint Shadowing Test";

            // Sabotage: Corrupt the flattened expected_hash_3 at step 100.
            int target_step = 100;
            auto corrupted_hash = from_fe256(trace.expected_hash_3[target_step]) + StoneField::One();
            
            trace.expected_hash_3[target_step] = to_fe256(corrupted_hash);
            trace.hash_state_3[target_step + 1] = to_fe256(corrupted_hash); // Shadowing prevention

        } else if (test_idx == 63) {
            test_name = "c_hash_end_1 (Rate 1 Fingerprint Matches Public Key) Constraint Shadowing Test";

            // Sabotage: The trace is mathematically perfect, but we corrupt the SECOND limb of the
            // 504-bit Public Key fingerprint. The transition math passes, isolating the Rate 1 anchor.
            tmp_pub_inputs.grid_fingerprint[1][0] ^= 1;
            use_tmp_pub_inputs = true;

        } else if (test_idx == 64) {
            test_name = "c_poseidon_math_0 (Adaptive Attacker: Tile ID Absorption Forgery)";

            // Sabotage Strategy: The attacker placed Tile A (say, ID 0) legally in the grid, but 
            // wants the Hash to absorb Tile B (say, ID 1) to artificially spoof a required fingerprint.
            // A highly rational attacker will recalculate the EXACT S-Box math for Tile B and inject it.
            int target_step = 150;
            
            // Identify the true tile placed at this step
            int true_tile_id = -1;
            for(size_t t = 0; t < alphabet.size(); ++t) {
                if (trace.tile_booleans[t][target_step] == 1) { true_tile_id = t; break; }
            }
            int fake_tile_id = (true_tile_id == 0) ? 1 : 0; // Pick a different tile
            
            // Recompute the genuine Hash Step as if fake_tile_id had been placed
            std::array<StoneField, 4> current_state = {
                from_fe256(trace.hash_state_0[target_step]),
                from_fe256(trace.hash_state_1[target_step]),
                from_fe256(trace.hash_state_2[target_step]),
                from_fe256(trace.hash_state_3[target_step])
            };
            
            auto fake_next_state = poseidon_hash_step(current_state, StoneField::FromUint(fake_tile_id), target_step);
            
            // Attacker writes the forged math output into the intermediate flattened columns
            trace.expected_hash_0[target_step] = to_fe256(fake_next_state[0]);
            trace.expected_hash_1[target_step] = to_fe256(fake_next_state[1]);
            trace.expected_hash_2[target_step] = to_fe256(fake_next_state[2]);
            trace.expected_hash_3[target_step] = to_fe256(fake_next_state[3]);
            
            // SHADOWING PREVENTION: The attacker MUST ALSO forge the next row's starting state 
            // so the sequential transition constraints (c_poseidon_hash_0 through 3) don't fail
            trace.hash_state_0[target_step + 1] = to_fe256(fake_next_state[0]);
            trace.hash_state_1[target_step + 1] = to_fe256(fake_next_state[1]);
            trace.hash_state_2[target_step + 1] = to_fe256(fake_next_state[2]);
            trace.hash_state_3[target_step + 1] = to_fe256(fake_next_state[3]);
            
            // Now the ONLY thing that fails is c_poseidon_math_0, because the AIR polynomial 
            // derives the Tile ID securely from the booleans, not from the attacker's fake math
        } else if (test_idx == 65) {
            test_name = "Blanking Round Continuity (Sponge Linkage) Test";
            
            // Sabotage Strategy: The attacker breaks the hash chain transition constraint 
            // directly in the middle of the blanking rounds.
            int target_step = trace.valid_steps + 30; 
            auto current_hash = from_fe256(trace.hash_state_0[target_step]);
            trace.hash_state_0[target_step] = to_fe256(current_hash + StoneField::FromUint(1));

        } else if (test_idx == 66) {
            test_name = "c_blanking_zero_lock (Unconstrained Blanking Round Injection Forgery) Test";

            // Target a step inside the blanking rounds (Grid 1 territory)
            int target_step = trace.valid_steps + 10;
            
            // 1. Inject the fake tile
            // We set Tile 1 to '1'. Since s_exe is 0, the exclusivity check is OFF.
            // But the AIR still derives current_tile_id = sum(b_i * i).
            // To ensure current_tile_id evaluates EXACTLY to 1, we must zero the rest.
            for (size_t t = 0; t < alphabet.size(); ++t) trace.tile_booleans[t][target_step] = 0;
            trace.tile_booleans[1][target_step] = 1; 

            // 2. Forge the downstream chain
            // We must start from the state AT the target step and propagate the forgery
            // to the very end of the 64 blanking rounds.
            for (size_t step = target_step; step < trace.valid_steps + 64; step++) {
                std::array<StoneField, 4> current_state = {
                    from_fe256(trace.hash_state_0[step]), 
                    from_fe256(trace.hash_state_1[step]),
                    from_fe256(trace.hash_state_2[step]), 
                    from_fe256(trace.hash_state_3[step])
                };
                
                // At the target_step, we absorb the fake Tile ID 1. 
                // In subsequent blanking steps, we continue absorbing 0.
                StoneField absorbed_tile = (step == target_step) ? StoneField::FromUint(1) : StoneField::Zero();
                
                // Align domain separator with the AIR's modulo wrap
                auto next_state = poseidon_hash_step(current_state, absorbed_tile, step % 4096); 
                
                // CRITICAL: Overwrite the 'expected' columns to satisfy the Math constraints
                trace.expected_hash_0[step] = to_fe256(next_state[0]);
                trace.expected_hash_1[step] = to_fe256(next_state[1]);
                trace.expected_hash_2[step] = to_fe256(next_state[2]);
                trace.expected_hash_3[step] = to_fe256(next_state[3]);
                
                if (step + 1 < trace.valid_steps + 64) {
                    // CRITICAL: Overwrite the next starting state to satisfy the Transition constraints
                    trace.hash_state_0[step + 1] = to_fe256(next_state[0]);
                    trace.hash_state_1[step + 1] = to_fe256(next_state[1]);
                    trace.hash_state_2[step + 1] = to_fe256(next_state[2]);
                    trace.hash_state_3[step + 1] = to_fe256(next_state[3]);
                } else {
                    // 3. Spoof the public key fingerprint: Update the Public Key to match our forged fingerprint.
                    // This allows the c_hash_end anchor at row 4159 to pass.
                    tmp_pub_inputs.grid_fingerprint[0] = to_fe256(next_state[0]);
                    tmp_pub_inputs.grid_fingerprint[1] = to_fe256(next_state[1]);
                    use_tmp_pub_inputs = true;
                }
            }
            // With both Math and Transition columns forged, the AIR will pass 
            // unless 'c_blanking_zero_lock' is implemented to stop it.
        }

        // NOTE: any further tests go here and remember to increase loop upper bound (max_test_idx) above
        else {
            break; // End of Sabotage Suite
            //throw std::runtime_error("This test case has not been implemented.");
        }

        // Prefix the test name with test idx for easier location
        test_name = "[" + std::to_string(test_idx) + "] " + test_name;

        // Extract the Pinned Perimeter (Public Key Constraints) directly from the private key
        std::array<FieldElement256, 2> grid_fingerprint = {
            trace.expected_hash_0[trace.valid_steps + 63], // +63 for the final hash blanking rounds
            trace.expected_hash_1[trace.valid_steps + 63]
        }; // This always lives at the end of the first (k=0) Identity Grid + final Blanking Rounds
        QwtssPublicInputs public_inputs(username, identity_nonce, version, private_key,
            grid_size, grid_fingerprint, alphabet);
        if (use_tmp_pub_inputs) public_inputs = tmp_pub_inputs;

        bool current_pass = true;

        // Validate the trace mathematically
        QwtssAir validator(public_inputs, trace.trace_length, alphabet);
        bool validated = validator.validate_trace(trace);
        if (validated) {
            if (should_pass){
                tests_passed_good_accept_validator++;
            } else {
                failed_tests.push_back(test_name + " (Validator): Passed when it should have failed");
                current_pass = false;
            }
        }
        else {
            if (should_pass){
                failed_tests.push_back(test_name + " (Validator): Failed when it should have passed");
                current_pass = false;
            } else {
                tests_passed_good_reject_validator++;
            }
        }
        tests_performed++;

        // Verifier tests
        std::vector<std::byte> final_sig = generate_stark_signature(trace, public_inputs, message, alphabet);
        bool verified = verify_stark_signature(final_sig, public_inputs, message, alphabet);
        if (verified) {
            if (should_pass){
                tests_passed_good_accept_verifier++;
            } else {
                failed_tests.push_back(test_name + " (Verifier): Passed when it should have failed");
                current_pass = false;
            }
        }
        else {
            if (should_pass){
                failed_tests.push_back(test_name + " (Verifier): Failed when it should have passed");
                current_pass = false;
            } else {
                tests_passed_good_reject_verifier++;
            }
        }
        tests_performed++;

        std::cout << "\n\n[" << test_idx << "]: \"" << test_name << "\": " << (current_pass ? "PASSED" : "FAILED") << std::endl;
        std::cout << "=============================================================================================================" << std::endl;
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed = end - start;
    auto elapsed_minutes = std::chrono::floor<std::chrono::minutes>(elapsed);
    auto elapsed_rem_secs = std::chrono::duration_cast<std::chrono::seconds>(elapsed % std::chrono::minutes(1));

    std::cout << "\n\nAIR SABOTAGE SUITE RESULTS" << std::endl;
    std::cout << "======================================================================================" << std::endl;
    std::cout << "Validator: Good Accepts: " << tests_passed_good_accept_validator << ", Good Rejects: " << tests_passed_good_reject_validator << std::endl;
    std::cout << "Verifier: Good Accepts: " << tests_passed_good_accept_verifier << ", Good Rejects: " << tests_passed_good_reject_verifier << std::endl;
    int passed_tests = (tests_passed_good_accept_validator + tests_passed_good_accept_verifier + tests_passed_good_reject_validator + tests_passed_good_reject_verifier);
    std::cout << "Passed tests: " << passed_tests << " / " << tests_performed << std::endl;

    int tests_failed = tests_performed - passed_tests;
    if (tests_failed != failed_tests.size()) throw std::runtime_error("(tests_failed != failed_tests.size())");
    if (tests_failed > 0){
        std::cout << "\n\nFailed tests: " << tests_failed << std::endl;
        for (std::string failed_desc : failed_tests){
            std::cout << "\t" << failed_desc << std::endl;
        }
    }
    std::cout << "\nElapsed time: " << elapsed_minutes.count() << " minutes and " << elapsed_rem_secs.count() << " seconds" << std::endl;

    if (tests_failed == 0) {
        std::cout << "*** ALL TESTS PASSED ***" << std::endl;
        return true;
    } else {
        std::cout << "*** ONE OR MORE TESTS FAILED ***" << std::endl;
        return false;
    }
}

void test_poseidon_sync() {
    std::cout << "\n[TEST] Running CPU vs STARK Poseidon Hash Sync (100 Steps)..." << std::endl;

    // 1. Initialize empty states for both implementations (t=4)
    std::array<mpz_class, 4> cpu_state = {0, 0, 0, 0};
    std::array<StoneField, 4> stark_state = {StoneField::Zero(), StoneField::Zero(), StoneField::Zero(), StoneField::Zero()};

    // 2. Run the 100-step simulation loop
    for (size_t step = 0; step < 100; ++step) {
        uint64_t tile_sim = 0;
        if (step == 0){
            // Mimic the KAT test for visual confirmation
            tile_sim = 1;
        } else {
            // Simulate pseudo-random tile IDs (e.g., looping through an 11-tile alphabet)
            tile_sim = step % 11;
        }

        // --- PREPARE CPU INPUTS ---
        mpz_class tile_cpu = tile_sim;

        // --- PREPARE STARK INPUTS ---
        StoneField tile_stark = StoneField::FromUint(tile_sim);

        // --- EXECUTE BOTH IMPLEMENTATIONS ---
        cpu_state = fast_poseidon_hash_step_cpu(cpu_state, tile_cpu, step);
        stark_state = poseidon_hash_step(stark_state, tile_stark, step);

        // --- COMPARE INTEGRITY ---
        for (int i = 0; i < 4; ++i) {
            // Convert the STARK field element to a hex string (usually formats as "0x...")
            std::string stark_hex = stark_state[i].ToString(); 
            
            // Parse the STARK hex string back into a GMP class for flawless endian-safe comparison
            mpz_class stark_mpz(stark_hex); 

            if (cpu_state[i] != stark_mpz) {
                std::cerr << "\n[FATAL ERROR] Hash Mismatch at Step " << step << ", Element " << i << "!" << std::endl;
                std::cerr << " -> CPU (GMP):   0x" << cpu_state[i].get_str(16) << std::endl;
                std::cerr << " -> STARK (C++): " << stark_hex << std::endl;
                throw std::runtime_error("Poseidon hash sync failed. The math does not match!");
            }
        }

        // --- LOG FIRST AND LAST STEPS
        if (step == 0 || step == 99){
            for (int i = 0; i < 4; ++i) {
                // Convert the STARK field element to a hex string (usually formats as "0x...")
                std::string stark_hex = stark_state[i].ToString(); 

                std::cout << std::endl;
                std::cout << "[" << std::setw(3) << step << "][" << i << "]: CPU (GMP):                   0x" << cpu_state[i].get_str(16) << std::endl;
                std::cout << "[" << std::setw(3) << step << "][" << i << "]: STARK (Stone Prover Fields): " << stark_hex << std::endl;
            }
        }
    }
    
    std::cout << "\n[SUCCESS] CPU and STARK Poseidon implementations are perfectly synced" << std::endl;
}

static void print_global_help() {
    std::cout << "QWTSS (Quasiperiodic Wang Tiling zk-STARK Signatures) Tests\n"
                << "Usage: qwtss-tests <command>\n\n"
                << "Commands:\n"
                << "  sabotage-suite    Run the full AIR sabotage test suite\n"
                << "  pipeline          Run the end-to-end pipeline with a single generated key\n"
                << "  poseidon          Verify the Poseidon hash results\n"
                << "  labbe-oracle      Run the Labbé oracle unit test\n\n";
}

int main(int argc, char** argv) {
    if (argc < 2){
        print_global_help();
        return EXIT_FAILURE;
    }

    std::string_view command = argv[1];
    if (command == "pipeline"){
        // End-to-end pipeline with a single generated key
        run_qwtss_full_pipeline("user@example.com", 0);
        return EXIT_SUCCESS;
    } else if (command == "sabotage-suite"){
        // Full digital signature verification/AIR sabotage suite
        bool all_passed = run_qwtss_air_sabotage_test_suite();
        return (all_passed ? EXIT_SUCCESS : EXIT_FAILURE);
    } else if (command == "poseidon"){
        test_poseidon_math();
        test_poseidon_sync();
        return EXIT_SUCCESS;
    } else if (command == "labbe-oracle"){
        bool passed = LabbeJR11Oracle::run_labbe_oracle_unit_test();
        return (passed ? EXIT_SUCCESS : EXIT_FAILURE);
    } else if (command == "misc"){
        // Placeholder
    } else {
        std::cerr << "Unknown command: " << command << "\n";
        print_global_help();
        return EXIT_FAILURE;
    }
}
