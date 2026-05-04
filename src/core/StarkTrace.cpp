#include "StarkTrace.h"
#include "QwtssAir.h"
#include "QwtssCore.h"
#include "Tiles.h"
#include "PoseidonConstants.h"
#include "CryptoUtils.h"
#include "QwtssConfig.h"
#include "starkware/air/trace.h"
#include "starkware/algebra/polymorphic/field_element_vector.h"
#include "starkware/stark/stark.h"
#include "starkware/stark/utils.h"
#include "starkware/utils/json.h"
#include "starkware/statement/statement.h"
#include "starkware/main/prover_main_helper_impl.h"
#include "starkware/main/verifier_main_helper_impl.h"
#include "starkware/air/trace_context.h"
#include <fcntl.h>


// The 504-bit Poseidon Round
std::array<StoneField, 4> poseidon_hash_step(std::array<StoneField, 4> current_state, StoneField tile_id, size_t step_index) {

    // 1. Sponge Absorption (Add Tile ID to Rate 1, Step Index to Rate 2)
    // Elements 2 and 3 (Capacity) are left alone during absorption
    current_state[0] = current_state[0] + tile_id; 
    current_state[1] = current_state[1] + StoneField::FromUint(step_index);

    // 2. Add Round Constants (Using the Magic Static for zero-overhead loading)
    current_state[0] = current_state[0] + PoseidonConstants::STONE_FAST_ROUND_CONSTANTS[step_index][0];
    current_state[1] = current_state[1] + PoseidonConstants::STONE_FAST_ROUND_CONSTANTS[step_index][1];
    current_state[2] = current_state[2] + PoseidonConstants::STONE_FAST_ROUND_CONSTANTS[step_index][2];
    current_state[3] = current_state[3] + PoseidonConstants::STONE_FAST_ROUND_CONSTANTS[step_index][3];

    // 3. Non-Linear Layer (S-Box x^3)
    current_state[0] = current_state[0] * current_state[0] * current_state[0];
    current_state[1] = current_state[1] * current_state[1] * current_state[1];
    current_state[2] = current_state[2] * current_state[2] * current_state[2];
    current_state[3] = current_state[3] * current_state[3] * current_state[3];

    // 4. Linear Layer (Cauchy MDS Matrix Multiplication) using Magic Static
    std::array<StoneField, 4> next_state;
    for (int i = 0; i < 4; i++) {
        next_state[i] = (current_state[0] * PoseidonConstants::STONE_FAST_MDS[i][0]) +
                        (current_state[1] * PoseidonConstants::STONE_FAST_MDS[i][1]) + 
                        (current_state[2] * PoseidonConstants::STONE_FAST_MDS[i][2]) + 
                        (current_state[3] * PoseidonConstants::STONE_FAST_MDS[i][3]);
    }

    return next_state;
}

void test_poseidon_math() {
    std::cout << "\n--- POSEIDON KAT TEST ---" << std::endl;
    // 1. Initialize state to 0
    std::array<StoneField, 4> state = {StoneField::Zero(), StoneField::Zero(), StoneField::Zero(), StoneField::Zero()};
    
    // 2. Run exactly 1 step (step 0) with a fake tile_id of 1
    auto next_state = poseidon_hash_step(state, StoneField::One(), 0);
    
    std::cout << "C++ Output S0: " << next_state[0].ToString() << std::endl;
    std::cout << "C++ Output S1: " << next_state[1].ToString() << std::endl;
    std::cout << "C++ Output S2: " << next_state[2].ToString() << std::endl;
    std::cout << "C++ Output S3: " << next_state[3].ToString() << std::endl;

    /*
    --- POSEIDON KAT TEST ---
    StoneProver Fields Output S0:   0x458263e7af67e9e6af2ec738879feab438248fc5f172734a7cb32048362c132
    StoneProver Fields Output S1:   0x4c11006c903016d9b90994e856b10e2203be3ba79707b2c5c4a392b8050e39a
    StoneProver Fields Output S2:   0x7d68bd046a5a7a9498523c43aad9b37b07ecd66c91c6ccde0b61318f5ee2e72
    StoneProver Fields Output S3:   0x36d0033fa8c9f4d93585d571b2b732d835f50bcba0defbdc593db10ca4e59d2

    CPU (GMP) S0:                   0x458263e7af67e9e6af2ec738879feab438248fc5f172734a7cb32048362c132
    CPU (GMP) S1:                   0x4c11006c903016d9b90994e856b10e2203be3ba79707b2c5c4a392b8050e39a
    CPU (GMP) S2:                   0x7d68bd046a5a7a9498523c43aad9b37b07ecd66c91c6ccde0b61318f5ee2e72
    CPU (GMP) S3:                   0x36d0033fa8c9f4d93585d571b2b732d835f50bcba0defbdc593db10ca4e59d2

    Python Output S0:               0x458263e7af67e9e6af2ec738879feab438248fc5f172734a7cb32048362c132
    Python Output S1:               0x4c11006c903016d9b90994e856b10e2203be3ba79707b2c5c4a392b8050e39a
    Python Output S2:               0x7d68bd046a5a7a9498523c43aad9b37b07ecd66c91c6ccde0b61318f5ee2e72
    Python Output S3:               0x36d0033fa8c9f4d93585d571b2b732d835f50bcba0defbdc593db10ca4e59d2
    */
}

// @param throw_preemptively If True, will throw preemptively on any violated constraint (instead of proceeding
//  to later generate invalid sig); set to False to bypass preemptive exceptions during validation testing.
ExecutionTrace build_execution_trace(const std::vector<QwtssPrivateKey>& private_keys, int grid_size, const std::vector<Tile>& alphabet,
    bool throw_preemptively, bool do_logging) {

    if (private_keys.empty()) throw std::invalid_argument("At least one identity grid must be provided.");
    if (private_keys[0].plane_A.empty() || private_keys[0].plane_B.empty()) throw std::invalid_argument("Missing private key's oracle plane A and/or plane B fields.");

    // Temporarily restrict to k=0 (Identity Grid Only) until the Branched Sponge is implemented.
    // Implementing a "Branched Sponge" in the AIR (Degree 4) will require three distinct periodic columns to control the branched logic:
    //  1. Grid 0 Absorbing Phase (Steps 0 to 4095)
    //  2. Blanking Phase (Steps 4096 to 4159)
    //  3. State Transition Linker (Steps 0 to 4158)
    //  4. The Final Fingerprint Extraction (Step 4159): s_identity_last_vals[4159] = FieldElementT_::One();
    if (private_keys.size() > 1) {
        throw std::invalid_argument("Multi-grid traces are temporarily disabled until the Branched Sponge is implemented to universally handle the 64 hash blanking rounds. Pass only the k=0 identity grid.");
    }

    size_t k_total_grids = private_keys.size();
    size_t valid_steps = k_total_grids * 4096;
    
    // Canonical Calculation: Guarantee at least 256 rows of ZK blinding noise, 
    // then snap to the next power of 2.
    size_t min_required_rows = valid_steps + 256;
    size_t trace_length = 1;
    while (trace_length < min_required_rows) trace_length *= 2;

    ExecutionTrace trace;
    trace.valid_steps = valid_steps;
    trace.blinding_steps = trace_length - valid_steps;
    trace.trace_length = trace_length;

    trace.is_h_defect.resize(trace_length);
    trace.is_v_defect.resize(trace_length);
    trace.dummy_defect.resize(trace_length);
    trace.h_seam_acc.resize(trace_length, 0);
    trace.v_seam_acc.resize(trace_length, 0);
    trace.defect_accumulator.resize(trace_length);
    trace.hash_state_0.resize(trace_length);
    trace.hash_state_1.resize(trace_length);
    trace.hash_state_2.resize(trace_length);
    trace.hash_state_3.resize(trace_length);
    trace.half_check.resize(trace_length);
    trace.core_defect.resize(trace_length);
    trace.diff_A_B.resize(trace_length);
    trace.is_alien.resize(trace_length);
    trace.expected_hash_0.resize(trace_length);
    trace.expected_hash_1.resize(trace_length);
    trace.expected_hash_2.resize(trace_length);
    trace.expected_hash_3.resize(trace_length);

    // Allocate exactly enough boolean columns for the loaded alphabet
    trace.tile_booleans.resize(alphabet.size(), std::vector<FieldElement>(trace_length, 0));

    trace.alien_acc.resize(trace_length, 0);
    trace.alien_inv.resize(trace_length);
    int current_alien = 0; // For debugging only

    int current_defect = 0; // For debugging only
    trace.core32_acc.resize(trace_length, 0);

    std::array<StoneField, 4> current_hash_state = {StoneField::Zero(), StoneField::Zero(), StoneField::Zero(), StoneField::Zero()};

    // --- MAIN TRACE GENERATION LOOP (K+1 Grids) ---
    for (size_t k = 0; k < k_total_grids; k++) {
        const int* h_grid = private_keys[k].private_key.data();
        size_t grid_offset = k * 4096;

        // Reset these for every grid
        current_alien = 0;
        current_defect = 0;

        StoneField current_accumulator = StoneField::Zero();

        // Calculate required slack padding for the 170 target
        int actual_total_defects = count_grid_defects(h_grid, grid_size, alphabet);
        if (actual_total_defects < QwtssReference::total_defects_lower_bound || actual_total_defects > QwtssReference::total_defects_upper_bound) {
            if (throw_preemptively){
                throw std::runtime_error("Trace generation failed: Grid " + std::to_string(k) + " defect count is outside the valid band.");
            }
        }
        int slack_needed = QwtssReference::total_defects_upper_bound - actual_total_defects;
        int max_slack = (QwtssReference::total_defects_upper_bound - QwtssReference::total_defects_lower_bound);

        // --- Pre-Calculate core32_acc ---
        uint64_t current_core32 = 0;

        for (int step = 0; step < 4096; step++) {
            size_t global_step = grid_offset + step;
            trace.core32_acc[global_step] = current_core32;

            int target_step = step + 64;
            if (target_step < 4096) {
                int tr = target_step / grid_size;
                int tc = target_step % grid_size;

                bool is_core = (tr >= 16 && tr < 48 && tc >= 16 && tc < 48);

                if (is_core && current_core32 < QwtssReference::core32_lower_bound) {
                    Tile target_tile = alphabet[h_grid[target_step]];
                    bool is_defective = false;

                    if (tc < grid_size - 1 && target_tile.right != alphabet[h_grid[target_step + 1]].left) is_defective = true;
                    if (tr < grid_size - 1 && target_tile.bottom != alphabet[h_grid[target_step + grid_size]].top) is_defective = true;
                    if (tc > 0 && target_tile.left != alphabet[h_grid[target_step - 1]].right) is_defective = true;
                    if (tr > 0 && target_tile.top != alphabet[h_grid[target_step - grid_size]].bottom) is_defective = true;

                    if (is_defective) {
                        current_core32++;
                    }
                }
            }
        }

        if (current_core32 < QwtssReference::core32_lower_bound) {
            if (throw_preemptively) {
                throw std::runtime_error("Trace generation failed: Core32 count is below the " + std::to_string(QwtssReference::core32_lower_bound) + " threshold.");
            }
        }

        // --- The Valid Execution Domain ---
        for (int r = 0; r < grid_size; r++) {
            for (int c = 0; c < grid_size; c++) {
                int step = r * grid_size + c;
                size_t global_step = grid_offset + step;
                int t_id = h_grid[step];
                Tile current = alphabet[t_id];

                for (size_t i = 0; i < alphabet.size(); ++i) {
                    trace.tile_booleans[i][global_step] = (t_id == i) ? 1 : 0;
                }

                FieldElement h_defect = 0;
                if (c < grid_size - 1) {
                    Tile next_h = alphabet[h_grid[step + 1]];
                    if (current.right != next_h.left) { h_defect = 1; current_defect++; }
                }
                trace.is_h_defect[global_step] = h_defect;

                FieldElement v_defect = 0;
                if (r < grid_size - 1) {
                    Tile next_v = alphabet[h_grid[step + grid_size]];
                    if (current.bottom != next_v.top) { v_defect = 1; current_defect++; }
                }
                trace.is_v_defect[global_step] = v_defect;

                FieldElement dummy = 0;
                if (step < max_slack && slack_needed > 0) {
                    dummy = 1;
                    slack_needed--;
                }
                trace.dummy_defect[global_step] = dummy;

                trace.alien_acc[global_step] = current_alien;
                
                // IMPORTANT: Plucked from Identity Grid (k=0) to keep STARK boundary public constraints uniform across all K grids
                bool is_A = (t_id == private_keys[0].plane_A[step]);
                bool is_B = (t_id == private_keys[0].plane_B[step]);

                trace.alien_inv[global_step] = to_fe256(StoneField::Zero());

                StoneField diff = (StoneField::FromUint(t_id) - StoneField::FromUint(private_keys[0].plane_A[step])) *
                                  (StoneField::FromUint(t_id) - StoneField::FromUint(private_keys[0].plane_B[step]));
                trace.diff_A_B[global_step] = to_fe256(diff);

                FieldElement is_alien_val = 0;
                if (!is_A && !is_B && current_alien < QwtssReference::alien_tiles_lower_bound) {
                    trace.alien_inv[global_step] = to_fe256(StoneField::One() / diff);
                    is_alien_val = 1;
                    current_alien++; 
                }
                trace.is_alien[global_step] = is_alien_val;

                trace.defect_accumulator[global_step] = to_fe256(current_accumulator);
                trace.hash_state_0[global_step] = to_fe256(current_hash_state[0]);
                trace.hash_state_1[global_step] = to_fe256(current_hash_state[1]);
                trace.hash_state_2[global_step] = to_fe256(current_hash_state[2]);
                trace.hash_state_3[global_step] = to_fe256(current_hash_state[3]);

                StoneField sf_h_defect = StoneField::FromUint(h_defect);
                StoneField sf_v_defect = StoneField::FromUint(v_defect);
                StoneField sf_dummy    = StoneField::FromUint(dummy);

                current_hash_state = poseidon_hash_step(current_hash_state, StoneField::FromUint(t_id), step);
                current_accumulator = current_accumulator + sf_h_defect + sf_v_defect + sf_dummy;

                // Flattened Trace: Capture the exact output of the S-Box + MDS math for this specific round
                trace.expected_hash_0[global_step] = to_fe256(current_hash_state[0]);
                trace.expected_hash_1[global_step] = to_fe256(current_hash_state[1]);
                trace.expected_hash_2[global_step] = to_fe256(current_hash_state[2]);
                trace.expected_hash_3[global_step] = to_fe256(current_hash_state[3]);
            }
        }

        if (current_alien < QwtssReference::alien_tiles_lower_bound) {
            if (throw_preemptively) throw std::runtime_error("Trace generation failed: Alien tile threshold.");
        }

        for (int step = 0; step < 4096; step++) {
            size_t global_step = grid_offset + step;
            auto get_h = [&](int s) { return (s < 4096) ? trace.is_h_defect[grid_offset + s] : 0; };
            auto get_v = [&](int s) { return (s < 4096) ? trace.is_v_defect[grid_offset + s] : 0; };

            FieldElement T_val = 1 - get_v(step);
            FieldElement L_val = 1 - get_h(step + 63);
            FieldElement R_val = 1 - get_h(step + 64);
            FieldElement B_val = 1 - get_v(step + 64);

            trace.half_check[global_step] = T_val * L_val;
            
            // Mask the core defect so it is strictly 0 outside the Core32 zone
            int r = step / 64;
            int c = step % 64;
            bool is_core = (r >= 15 && r < 47 && c >= 16 && c < 48);

            if (is_core) {
                trace.core_defect[global_step] = 1 - (trace.half_check[global_step] * R_val * B_val);
            } else {
                trace.core_defect[global_step] = 0;
            }
        }

        int max_line_target = QwtssReference::line_max_upper_bound;

        for (int r = 0; r < 63; ++r) {
            int actual_defects = 0;
            for (int c = 0; c < 64; ++c) actual_defects += trace.is_v_defect[grid_offset + r * 64 + c];
            if (actual_defects > max_line_target && throw_preemptively) throw std::runtime_error("Trace generation failed: Horizontal seam limit.");

            int dummies_needed = max_line_target - actual_defects;
            int current_acc = 0;
            for (int c = 0; c < 64; ++c) {
                size_t global_step = grid_offset + r * 64 + c;
                trace.h_seam_acc[global_step] = current_acc; 
                if (c < 63) { 
                    int dummy = (dummies_needed > 0) ? 1 : 0;
                    if (dummies_needed > 0) dummies_needed--;
                    current_acc += trace.is_v_defect[global_step] + dummy;
                }
            }
        }

        for (int c = 0; c < 63; ++c) {
            int actual_defects = 0;
            for (int r = 0; r < 64; ++r) actual_defects += trace.is_h_defect[grid_offset + r * 64 + c];
            if (actual_defects > max_line_target && throw_preemptively) throw std::runtime_error("Trace generation failed: Vertical seam limit.");

            int dummies_needed = max_line_target - actual_defects;
            int current_acc = 0;
            for (int r = 0; r < 64; ++r) {
                size_t global_step = grid_offset + r * 64 + c;
                trace.v_seam_acc[global_step] = current_acc; 
                if (r < 63) { 
                    int dummy = (dummies_needed > 0) ? 1 : 0;
                    if (dummies_needed > 0) dummies_needed--;
                    current_acc += trace.is_h_defect[global_step] + dummy;
                }
            }
        }
    } // End of K loop

    // 64 final blanking rounds to secure the continuous sponge
    for (int blank_step = 0; blank_step < 64; blank_step++) {
        size_t global_step = valid_steps + blank_step;
        size_t local_step = global_step % 4096;
        
        trace.hash_state_0[global_step] = to_fe256(current_hash_state[0]);
        trace.hash_state_1[global_step] = to_fe256(current_hash_state[1]);
        trace.hash_state_2[global_step] = to_fe256(current_hash_state[2]);
        trace.hash_state_3[global_step] = to_fe256(current_hash_state[3]);
        
        // Hash a Tile ID of 0 using the wrapped local_step to align with the STARK AIR
        current_hash_state = poseidon_hash_step(current_hash_state, StoneField::Zero(), local_step);
        
        trace.expected_hash_0[global_step] = to_fe256(current_hash_state[0]);
        trace.expected_hash_1[global_step] = to_fe256(current_hash_state[1]);
        trace.expected_hash_2[global_step] = to_fe256(current_hash_state[2]);
        trace.expected_hash_3[global_step] = to_fe256(current_hash_state[3]);
    }

    if (do_logging){
        std::cout << "\n[STARK Trace Integrity Audit]" << std::endl;
        std::cout << "  -> Total Grids (K+1):              " << k_total_grids << std::endl;
        std::cout << "  -> Total Trace Length (Rows):      " << trace_length << std::endl;
        std::cout << "  -> Final Grid Defect Count:        " << current_defect << std::endl;
        std::cout << "  -> Final Grid Alien Tile Count:    " << current_alien << std::endl;
    }

    return trace;
}

starkware::Trace convert_to_stone_trace(const ExecutionTrace& raw_trace) {
    // Note: We use a 2D vector of the concrete FieldElements, NOT the polymorphic wrappers
    std::vector<std::vector<StoneField>> concrete_columns;
    // 21 base columns (including 9 flattened intermediates) + N tileset alphabet one-hot booleans
    concrete_columns.reserve(21 + raw_trace.tile_booleans.size());

    // Instantiate PRNG here for the 64-bit column blinding
    ChaCha20PRNG csprng;
    auto generate_252bit_noise = [&csprng]() -> StoneField {
        FieldElement256 raw_noise = {
            csprng.next_u64(), csprng.next_u64(), 
            csprng.next_u64(), csprng.next_u64()
        };
        // Stone's FromBytes defaults to Big-Endian, making byte 0 the Most Significant Byte.
        // StarkWare's prime is 0x080000... so the MSB must be strictly bounded <= 0x07.
        auto* bytes = reinterpret_cast<uint8_t*>(raw_noise.data());

        // Mask the Most Significant Byte in Little-Endian (Index 31 instead of 0). Yields 251 bits of safe entropy
        bytes[31] &= 0x07;

        return from_fe256(raw_noise);
    };

    // Helper lambda to convert a column of uint64_t into 252-bit PrimeFieldElements
    // Note: We inject 252-bit blinding noise directly into the 64-bit columns during conversion, at the appropriate steps
    auto convert_col_64 = [&generate_252bit_noise, &raw_trace](const std::vector<FieldElement>& raw_col) {
        std::vector<StoneField> stone_col;
        stone_col.reserve(raw_col.size());
        for (size_t step = 0; step < raw_col.size(); step++) {
            // Because indices are 0-based, valid_steps is the EXACT index where padding begins
            if (step >= raw_trace.valid_steps) { 
                // Pure blinding noise. This MUST be 252-bit noise to blind the polynomial securely
                stone_col.push_back(generate_252bit_noise());
            } else {
                stone_col.push_back(StoneField::FromUint(raw_col[step])); 
            }
        }
        return stone_col;
    };

    // Helper lambda to convert 256-bit arrays to StoneField
    // Note: We inject 252-bit blinding noise directly into the 256-bit columns during conversion, at the appropriate steps
    auto convert_col_256 = [&generate_252bit_noise, &raw_trace](const std::vector<FieldElement256>& raw_col) {
        std::vector<StoneField> stone_col;
        stone_col.reserve(raw_col.size());
        for (size_t step = 0; step < raw_col.size(); step++) {
            // Because indices are 0-based, valid_steps is the EXACT index where padding begins
            if (step >= raw_trace.valid_steps) {
                // Pure blinding noise. This MUST be 252-bit noise to blind the polynomial securely
                stone_col.push_back(generate_252bit_noise());
            } else {
                stone_col.push_back(from_fe256(raw_col[step])); 
            }
        }
        return stone_col;
    };

    // Special lambda for Hash state columns to preserve the 64 blanking rounds
    auto convert_hash_col_256 = [&generate_252bit_noise, &raw_trace](const std::vector<FieldElement256>& raw_col) {
        std::vector<StoneField> stone_col;
        stone_col.reserve(raw_col.size());
        size_t hash_valid_steps = raw_trace.valid_steps + 64;
        for (size_t step = 0; step < raw_col.size(); step++) {
            if (step >= hash_valid_steps) {
                stone_col.push_back(generate_252bit_noise());
            } else {
                stone_col.push_back(from_fe256(raw_col[step])); 
            }
        }
        return stone_col;
    };

    // Special lambda for 64-bit boolean columns to preserve the 0s during blanking rounds
    auto convert_hash_col_64 = [&generate_252bit_noise, &raw_trace](const std::vector<FieldElement>& raw_col) {
        std::vector<StoneField> stone_col;
        stone_col.reserve(raw_col.size());
        size_t hash_valid_steps = raw_trace.valid_steps + 64;
        for (size_t step = 0; step < raw_col.size(); step++) {
            if (step >= hash_valid_steps) {
                stone_col.push_back(generate_252bit_noise());
            } else {
                stone_col.push_back(StoneField::FromUint(raw_col[step])); 
            }
        }
        return stone_col;
    };

    // Extract exactly in the order defined by QwtssAir::GetMask()
    // Extract using the 64-bit converter with blinding noise injection
    concrete_columns.push_back(convert_col_64(raw_trace.is_h_defect));
    concrete_columns.push_back(convert_col_64(raw_trace.is_v_defect));
    concrete_columns.push_back(convert_col_64(raw_trace.dummy_defect));
    concrete_columns.push_back(convert_col_64(raw_trace.alien_acc));

    // Extract using the 256-bit converter with blinding noise injection
    // MUST extract exactly in the order defined by QwtssAir::GetMask() (Indices 4 through 9)
    concrete_columns.push_back(convert_col_256(raw_trace.defect_accumulator));
    concrete_columns.push_back(convert_hash_col_256(raw_trace.hash_state_0));
    concrete_columns.push_back(convert_hash_col_256(raw_trace.hash_state_1));
    concrete_columns.push_back(convert_hash_col_256(raw_trace.hash_state_2));
    concrete_columns.push_back(convert_hash_col_256(raw_trace.hash_state_3));
    concrete_columns.push_back(convert_col_256(raw_trace.alien_inv));

    // Extract using the 64-bit converter with blinding noise injection
    concrete_columns.push_back(convert_col_64(raw_trace.core32_acc));
    concrete_columns.push_back(convert_col_64(raw_trace.h_seam_acc));
    concrete_columns.push_back(convert_col_64(raw_trace.v_seam_acc));

    // Extract the 3 intermediate columns at indices 13, 14, and 15
    concrete_columns.push_back(convert_col_64(raw_trace.half_check));
    concrete_columns.push_back(convert_col_64(raw_trace.core_defect));
    concrete_columns.push_back(convert_col_256(raw_trace.diff_A_B));

    // Extract the 5 new Tight-Packing flattened intermediate columns at indices 16, 17, 18, 19, and 20
    concrete_columns.push_back(convert_col_64(raw_trace.is_alien));
    concrete_columns.push_back(convert_hash_col_256(raw_trace.expected_hash_0));
    concrete_columns.push_back(convert_hash_col_256(raw_trace.expected_hash_1));
    concrete_columns.push_back(convert_hash_col_256(raw_trace.expected_hash_2));
    concrete_columns.push_back(convert_hash_col_256(raw_trace.expected_hash_3));

    // Dynamically inject all N boolean columns for the tileset with delayed 252-bit blinding noise
    // This allows the tile booleans to safely evaluate to 0 during the blanking rounds
    for (size_t i = 0; i < raw_trace.tile_booleans.size(); ++i) {
        concrete_columns.push_back(convert_hash_col_64(raw_trace.tile_booleans[i]));
    }

    // The Trace constructor will accept the 2D vector and handle the polymorphic wrapping internally
    return starkware::Trace(std::move(concrete_columns));
}

// A lightweight wrapper to hold the Trace for the Prover
class QwtssTraceContext : public starkware::TraceContext {
private:
    starkware::Trace trace_;
public:
    explicit QwtssTraceContext(starkware::Trace&& trace) : trace_(std::move(trace)) {}
    
    starkware::Trace GetTrace() override { 
        return std::move(trace_); 
    }
    
    // Required by the base class, even if we don't use interaction phases
    starkware::Trace GetInteractionTrace() override {
        return starkware::Trace(std::vector<std::vector<StoneField>>());
    }
};

// The STARK Statement interface expected by Stone Prover
class QwtssStatement : public starkware::Statement {
private:
    QwtssAir air_;
    mutable starkware::Trace trace_; // Mutable allows us to std::move it out inside the const GetTraceContext()
    std::vector<std::byte> message_seed_; 

public:
    QwtssStatement(const QwtssPublicInputs& public_inputs, starkware::Trace&& trace, size_t trace_length,
        const std::vector<uint8_t>& message, const std::vector<Tile>& alphabet)
        : starkware::Statement(std::nullopt), air_(public_inputs, trace_length, alphabet), trace_(std::move(trace)) {

        // To prevent the Weak Fiat-Shamir Vulnerability, we must cryptographically bind the message to the STARK
        // proof by using it as the initial Fiat-Shamir hash chain seed, *PLUS* every single element of the Public Statement.
        // One call handles full cryptographic binding:
        message_seed_ = public_inputs.serialize_for_fiat_shamir(message);
    }

    const starkware::Air& GetAir() override { return air_; }
    
    const std::vector<std::byte> GetInitialHashChainSeed() const override { 
        return message_seed_; 
    }
    
    std::unique_ptr<starkware::TraceContext> GetTraceContext() const override {
        return std::make_unique<QwtssTraceContext>(std::move(trace_));
    }

    starkware::JsonValue FixPublicInput() override { 
        // Initialize empty JSON object via FromString
        return starkware::JsonValue::FromString("{}"); 
    }

    std::string GetName() const override { return "QwtssStatement"; }
};

// Deprecated: this version is less efficient in all relevant metrics
starkware::JsonValue get_stark_params_orig(){
    // Build the STARK Security Parameters in memory.
    // Updated for 128x Blowup factor and 16 Queries. 24 bits (PoW) + (16 queries × 7 bits/query) (FRI Queries) = 136 bits of security.
    // Updated proof_of_work_bits from 16 to 24.
    starkware::JsonValue stark_params = starkware::JsonValue::FromString(R"({
        "field": "PrimeField0",
        "use_extension_field": false,
        "stark": {
            "fri": {
                "fri_step_list": [2, 2, 2],
                "last_layer_degree_bound": 128,
                "n_queries": 16,
                "proof_of_work_bits": 24
            },
            "log_n_cosets": 7
        }
    })");

    /*
        ---------------------------------------------------------------------------------
        Signature generated in 2.17 seconds. Size: 94952 bytes (92.7 KB).
        Pre-STARK Peak RAM Usage: 106.16 MB
        Post-STARK Peak RAM Usage: 210.45 MB (STARK signature generation required 104.30 MB)

        ---------------------------------------------------------------------------------
        Signature generated in 2.15 seconds. Size: 94568 bytes (92.4 KB).
        Pre-STARK Peak RAM Usage: 106.33 MB
        Post-STARK Peak RAM Usage: 210.09 MB (STARK signature generation required 103.77 MB)
    */

    return stark_params;
}

// Dynamically adjusts the FRI polynomial folding list to absorb massive K-trace lengths 
// without exploding the final verification layer constraints.
starkware::JsonValue get_stark_params(size_t trace_length) {
    int log2_trace = std::ceil(std::log2(trace_length));
    
    // To cleanly hit last_layer_degree_bound = 64 (2^6), the sum of the steps in fri_step_list 
    // must equal log2_trace - 6.
    int fri_sum_target = log2_trace - 6; 
    
    std::string fri_list = "[";
    if (fri_sum_target <= 0) {
        fri_list += "0";
    } else {
        int current_sum = 0;
        if (fri_sum_target % 2 != 0) {
            fri_list += "1";
            current_sum += 1;
        }
        while (current_sum < fri_sum_target) {
            if (current_sum > 0) fri_list += ", ";
            fri_list += "2";
            current_sum += 2;
        }
    }
    fri_list += "]";

    std::string json_str = R"({
        "field": "PrimeField0",
        "use_extension_field": false,
        "stark": {
            "fri": {
                "fri_step_list": )" + fri_list + R"(,
                "last_layer_degree_bound": 64,
                "n_queries": 18,
                "proof_of_work_bits": 24
            },
            "log_n_cosets": 6
        }
    })";

    return starkware::JsonValue::FromString(json_str);
}

// The Orchestrator Hook
std::vector<std::byte> generate_stark_signature(
    const ExecutionTrace& raw_trace, 
    const QwtssPublicInputs& public_inputs,
    const std::vector<uint8_t>& message_to_sign,
    const std::vector<Tile>& alphabet,
    bool do_logging
) {
    //std::cout << "Converting trace to STARK Algebraic Domain..." << std::endl;
    //auto start_convert_trace = std::chrono::high_resolution_clock::now();
    starkware::Trace stone_trace = convert_to_stone_trace(raw_trace);
    //auto end_convert_trace = std::chrono::high_resolution_clock::now();
    //std::chrono::duration<double, std::milli> convert_trace_ms = end_convert_trace - start_convert_trace;
    //std::cout << "Convert to stone trace: " << convert_trace_ms.count() << " ms" << std::endl;

    // Instantiate the Statement with our bindings
    QwtssStatement statement(public_inputs, std::move(stone_trace), raw_trace.trace_length, message_to_sign, alphabet);

    // Build the STARK Security Parameters dynamically in memory based on the final concatenated trace length.
    // Must be the same for prover and verifier.
    starkware::JsonValue stark_params = get_stark_params(raw_trace.trace_length);

    // Build the Engine Execution Configuration in memory (Flattened!)
    starkware::JsonValue stark_config = starkware::JsonValue::FromString(R"({
        "cached_lde_config": {
            "store_full_lde": false,
            "use_fft_for_eval": false
        },
        "constraint_polynomial_task_size": 256,
        "fri_prover_config": {
            "max_non_chunked_layer_size": 16,
            "n_chunks_between_layers": 1,
            "log_n_max_in_memory_fri_layer_elements": 10
        },
        "n_out_of_memory_merkle_layers": 1,
        "table_prover_n_tasks_per_segment": 32
    })");

    // Build an empty public input object from string
    starkware::JsonValue public_input = starkware::JsonValue::FromString("{}");

    int backup_stderr = -1;
    if (do_logging){
        std::cout << "Executing Stone Prover Core & Generating Proof (this may take a few seconds) ..." << std::endl;
        std::cout << "---------------------------------------------------------------------------------" << std::endl;
    } else {
        // Silence Stone Prover
        // Back up the original stderr and redirect it to /dev/null
        if (!do_logging) {
            fflush(stderr); // Clear any pending C-style errors
            backup_stderr = dup(STDERR_FILENO);
            int dev_null = open("/dev/null", O_WRONLY);
            dup2(dev_null, STDERR_FILENO);
            close(dev_null);
        }
    }

    //auto start_sign = std::chrono::high_resolution_clock::now();
    // Fire the Starkware Prover. The Helper handles the Channel, the Merkle Trees, and the LDE arrays automatically.
    std::vector<std::byte> signature = starkware::ProverMainHelperImpl(
        &statement, 
        stark_params, 
        stark_config, 
        public_input
    );
    //auto end_sign = std::chrono::high_resolution_clock::now();
    //std::chrono::duration<double, std::milli> sign_ms = end_sign - start_sign;
    //std::cout << "Stone prover sign: " << sign_ms.count() << " ms" << std::endl;

    if (do_logging){
        std::cout << "---------------------------------------------------------------------------------" << std::endl;
    } else if (backup_stderr != -1) {
        // Restore stderr
        fflush(stderr);
        dup2(backup_stderr, STDERR_FILENO);
        close(backup_stderr);
    }

    return signature;
}

// The Verifier Statement. This class cannot access the execution trace.
class QwtssVerifierStatement : public starkware::Statement {
private:
    QwtssAir air_;
    std::vector<std::byte> message_seed_;

public:
    QwtssVerifierStatement(const QwtssPublicInputs& public_inputs, size_t trace_length,
        const std::vector<uint8_t>& message, const std::vector<Tile>& alphabet)
        : starkware::Statement(std::nullopt), air_(public_inputs, trace_length, alphabet) {

        // To prevent the Weak Fiat-Shamir Vulnerability, we must cryptographically bind the message to the STARK
        // proof by using it as the initial Fiat-Shamir hash chain seed, *PLUS* every single element of the Public Statement.
        // One call handles full cryptographic binding:
        message_seed_ = public_inputs.serialize_for_fiat_shamir(message);
    }

    const starkware::Air& GetAir() override { return air_; }
    
    const std::vector<std::byte> GetInitialHashChainSeed() const override { 
        return message_seed_; 
    }
    
    std::unique_ptr<starkware::TraceContext> GetTraceContext() const override {
        // If the STARK engine tries to access the trace here, it instantly crashes.
        // Guarantees Verifier operates in strict zero-knowledge.
        throw std::runtime_error("SECURITY VIOLATION: Verifier attempted to access the private execution trace");
        return nullptr;
    }
    
    starkware::JsonValue FixPublicInput() override { 
        return starkware::JsonValue::FromString("{}"); 
    }
    
    std::string GetName() const override { return "QwtssStatement"; }
};

// The Verifier Execution Hook
bool verify_stark_signature(
    const std::vector<std::byte>& signature, 
    const QwtssPublicInputs& public_inputs,
    const std::vector<uint8_t>& message_to_sign,
    const std::vector<Tile>& alphabet,
    bool do_logging
) {
    // The Verifier currently computes the trace length with the assumption of one identity grid (k=0)
    size_t valid_steps = (1) * 4096;

    // Canonical Calculation: Guarantee at least 256 rows of ZK blinding noise,
    // then snap to the next power of 2.
    size_t min_required_rows = valid_steps + 256;
    size_t expected_trace_length = 1;
    while (expected_trace_length < min_required_rows) expected_trace_length *= 2;

    // The Verifier constructs the same AIR from the Public Key using the dynamic trace length
    QwtssVerifierStatement statement(public_inputs, expected_trace_length, message_to_sign, alphabet);

    // Build the STARK Security Parameters dynamically in memory. Must be the same for prover and verifier.
    starkware::JsonValue stark_params = get_stark_params(expected_trace_length);

    if (do_logging) std::cout << "Executing Stone Verifier Core (Checking OOD & FRI Queries)..." << std::endl;

    try {
        // Fire the Starkware Verifier.
        // This runs the Fiat-Shamir heuristic, evaluates the Out-of-Domain constraints, and checks the FRI layers.
        bool is_valid = starkware::VerifierMainHelperImpl(
            &statement,
            signature,
            stark_params,   // The STARK configuration parameters
            "",             // No annotation file needed
            ""              // No extra output file needed
        );

        if (is_valid) {
            if (do_logging) std::cout << "[SUCCESS] STARK Signature Verified" << std::endl;
            return true;
        } else {
            if (do_logging) std::cout << "[REJECTED] STARK Signature Failed Verification" << std::endl;
            return false;
        }
    } catch (const std::exception& e) {
        std::cout << "[FATAL] STARK Verifier Crashed: \n" << e.what() << std::endl;
        return false;
    }
}
