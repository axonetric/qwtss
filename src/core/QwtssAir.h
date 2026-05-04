#pragma once
#include "StarkTrace.h"
#include "QwtssAirTypes.h"
#include "PoseidonConstants.h"
#include "QwtssConfig.h"
#include "starkware/air/air.h"
#include "starkware/composition_polynomial/composition_polynomial.h"
#include "starkware/algebra/fields/prime_field_element.h"
#include "starkware/algebra/fields/fraction_field_element.h"
#include <functional>
#include <vector>
#include <memory>
#include <tuple> // Required for structured bindings


enum BaseConstraints {
    c_acc_transition = 0,
    c_h_match = 1,
    c_v_match = 2,
    c_h_defect_bool = 3,
    c_v_defect_bool = 4,
    c_dummy_bool = 5,
    c_poseidon_hash_0 = 6,
    c_poseidon_hash_1 = 7,
    c_poseidon_hash_2 = 8,
    c_poseidon_hash_3 = 9,
    c_poseidon_math_0 = 10,     // Math checks for flattened S-Box
    c_poseidon_math_1 = 11,
    c_poseidon_math_2 = 12,
    c_poseidon_math_3 = 13,
    c_bound_north = 14,
    c_bound_south = 15,
    c_bound_west = 16,
    c_bound_east = 17,
    c_acc_start = 18,
    c_acc_end = 19,
    c_hash_start_0 = 20,
    c_hash_start_1 = 21,
    c_hash_start_2 = 22,
    c_hash_start_3 = 23,
    c_hash_end_0 = 24,          // Bind 1st Rate Element
    c_hash_end_1 = 25,          // Bind 2nd Rate Element
    c_alien_end = 26,
    c_alien_start = 27,         // Force alien_acc to 0 at s_first
    c_slack_limit = 28,
    c_alien_bool = 29,          // Flattened Alien Tile checks
    c_alien_inv = 30,
    c_alien_transition = 31,
    c_alien_diff_A_B_check = 32, 
    c_h_seam_diff = 33,
    c_h_seam_start = 34,
    c_h_seam_end = 35,
    c_v_seam_diff = 36,
    c_v_seam_start = 37,
    c_v_seam_end = 38,
    c_rebar_pin = 39,           // Cryptographic Rebar Enforcement (vestigial logic)
    c_blanking_zero_lock = 40,  // Prevents Tile Forgery during blanking rounds
    NUM_BASE_CONSTRAINTS = 41
};

enum PeriodicColumns {
    p_s_exe = 0,
    p_s_h = 1,
    p_s_v = 2,
    p_s_core32 = 3,
    p_s_north = 4,
    p_s_south = 5,
    p_s_west = 6,
    p_s_east = 7,
    p_v_north = 8,
    p_v_south = 9,
    p_v_west = 10,
    p_v_east = 11,
    p_s_first = 12,
    p_s_last = 13,
    p_s_slack_limit = 14,
    p_col_A = 15,
    p_col_B = 16,
    p_s_col_0 = 17,
    p_s_row_0 = 18,
    p_c0 = 19,          
    p_c1 = 20,          
    p_c2 = 21,          
    p_c3 = 22,
    p_step = 23,            // Injects domain separation
    p_s_identity_last = 24, // Fires ONLY at row 4095
    p_s_rebar = 25,         // 1 if tile is pinned by Rebar
    p_v_rebar = 26,         // Target Tile ID for the Rebar pin
    p_s_hash_active = 27,       // Continuous sponge math check (Grid 0 + Blanking)
    p_s_hash_transition = 28,   // Continuous sponge transition (Stops 1 row early)
    NUM_PERIODIC_COLUMNS = 29
};

// Maps to the neighbors array (Base trace columns before the dynamic one-hot alphabet)
enum TraceColumns {
    t_is_h_defect = 0,
    t_is_v_defect = 1,
    t_dummy_defect = 2,
    t_alien_acc = 3,
    t_defect_acc = 4,
    t_hash_state_0 = 5,
    t_hash_state_1 = 6,
    t_hash_state_2 = 7,
    t_hash_state_3 = 8,
    t_alien_inv = 9,
    t_core32_acc = 10,
    t_h_seam_acc = 11,
    t_v_seam_acc = 12,
    t_half_check = 13,
    t_core_defect = 14,
    t_diff_A_B = 15,
    t_is_alien = 16,            // Intermediate column for Alien (0 or 1)
    t_expected_hash_0 = 17,     // Intermediate column for Poseidon S-Box Flattening
    t_expected_hash_1 = 18,     // Intermediate column for Poseidon S-Box Flattening
    t_expected_hash_2 = 19,     // Intermediate column for Poseidon S-Box Flattening
    t_expected_hash_3 = 20,     // Intermediate column for Poseidon S-Box Flattening
    NUM_BASE_TRACE_COLUMNS = 21
};


class QwtssAir : public starkware::Air {
private:
    QwtssPublicInputs pub_inputs;
    uint64_t trace_length_param;
    std::vector<Tile> alphabet; // Dynamically loaded TileSet

public:
    // Required by CompositionPolynomialImpl to resolve template types
    // Directs Stone Prover templates to compile using 252-bit prime field arithmetic
    using FieldElementT_ = starkware::PrimeFieldElement<252, 0>;

    // Pass the trace length and tileset to the starkware::Air base class
    QwtssAir(const QwtssPublicInputs& pk, uint64_t length, const std::vector<Tile>& alphabet) 
        : starkware::Air(length), pub_inputs(pk), trace_length_param(length), alphabet(alphabet) {}

    // Implementation of the composition polynomial creator.
    std::unique_ptr<starkware::CompositionPolynomial> CreateCompositionPolynomial(
        const starkware::FieldElement& trace_generator,
        const starkware::ConstFieldElementSpan& random_coefficients) const override {

        using PolyImpl = starkware::CompositionPolynomialImpl<QwtssAir>;
        
        // 1. Initialize the builder with the exact number of periodic columns
        typename PolyImpl::Builder builder(NUM_PERIODIC_COLUMNS);

        // 2. Construct the Periodic Selectors spanning the entire Trace (8192 steps)
        std::vector<FieldElementT_> s_exe_vals(this->TraceLength(), FieldElementT_::Zero());
        std::vector<FieldElementT_> s_h_vals(this->TraceLength(), FieldElementT_::Zero());
        std::vector<FieldElementT_> s_v_vals(this->TraceLength(), FieldElementT_::Zero());
        std::vector<FieldElementT_> s_core32_vals(this->TraceLength(), FieldElementT_::Zero());

        // difficulty_K is stubbed as 0 since we've moved the PoW out of the ZK sig
        size_t valid_steps = (0 + 1) * 4096;

        for (size_t step = 0; step < valid_steps; ++step) {
            size_t local_step = step % 4096;

            s_exe_vals[step] = FieldElementT_::One(); // 1 inside valid grids, 0 in noise

            // 0 on the right edge of the grid, 0 in the noise, 1 everywhere else
            // FieldElement s_h = ((step % 64 == 63) || step >= 4096) ? 0 : 1;
            // s_h: 0 on the right edge of the grid
            if (local_step % 64 != 63) s_h_vals[step] = FieldElementT_::One();
            // 0 on the bottom row of the grid, 0 in the noise, 1 everywhere else
            // FieldElement s_v = (step >= 4032) ? 0 : 1;
            // s_v: 0 on the bottom row of the grid
            if (local_step < 4032) s_v_vals[step] = FieldElementT_::One();

            // Populate Core32:
            // Shift the boolean window up by 1 row (r >= 15 instead of 16) so it aligns with the +64 offset equation
            size_t r = local_step / 64;
            size_t c = local_step % 64;
            // Evaluates to true when the tile AT step + 64 is inside the core
            if (r >= 15 && r < 47 && c >= 16 && c < 48) {
                s_core32_vals[step] = FieldElementT_::One();
            }
        }

        // Unwrap the polymorphic trace generator into the concrete 252-bit prime field element
        FieldElementT_ concrete_trace_gen = trace_generator.As<FieldElementT_>();

        // Instantiate and Add to Builder
        // PeriodicColumn args: (values, group_generator, offset, coset_size, column_step)
        starkware::PeriodicColumn<FieldElementT_> col_s_exe(
            s_exe_vals, concrete_trace_gen, FieldElementT_::One(), this->TraceLength(), 1ULL);
        
        starkware::PeriodicColumn<FieldElementT_> col_s_h(
            s_h_vals, concrete_trace_gen, FieldElementT_::One(), this->TraceLength(), 1ULL);
        
        starkware::PeriodicColumn<FieldElementT_> col_s_v(
            s_v_vals, concrete_trace_gen, FieldElementT_::One(), this->TraceLength(), 1ULL);

        starkware::PeriodicColumn<FieldElementT_> col_s_core32(
            s_core32_vals, concrete_trace_gen, FieldElementT_::One(), this->TraceLength(), 1ULL);

        int periodic_col_idx = 0;
        builder.AddPeriodicColumn(std::move(col_s_exe), periodic_col_idx++);
        builder.AddPeriodicColumn(std::move(col_s_h), periodic_col_idx++);
        builder.AddPeriodicColumn(std::move(col_s_v), periodic_col_idx++);
        builder.AddPeriodicColumn(std::move(col_s_core32), periodic_col_idx++);

        // 3. Boundary Columns
        std::vector<FieldElementT_> s_north(this->TraceLength(), FieldElementT_::Zero());
        std::vector<FieldElementT_> s_south(this->TraceLength(), FieldElementT_::Zero());
        std::vector<FieldElementT_> s_west(this->TraceLength(), FieldElementT_::Zero());
        std::vector<FieldElementT_> s_east(this->TraceLength(), FieldElementT_::Zero());
        std::vector<FieldElementT_> v_north(this->TraceLength(), FieldElementT_::Zero());
        std::vector<FieldElementT_> v_south(this->TraceLength(), FieldElementT_::Zero());
        std::vector<FieldElementT_> v_west(this->TraceLength(), FieldElementT_::Zero());
        std::vector<FieldElementT_> v_east(this->TraceLength(), FieldElementT_::Zero());
        std::vector<FieldElementT_> s_first(this->TraceLength(), FieldElementT_::Zero());
        std::vector<FieldElementT_> s_last(this->TraceLength(), FieldElementT_::Zero());
        // The Slack Limit constraint
        std::vector<FieldElementT_> s_slack_limit(this->TraceLength(), FieldElementT_::Zero());
        std::vector<FieldElementT_> s_identity_last_vals(this->TraceLength(), FieldElementT_::Zero());
        std::vector<FieldElementT_> s_rebar_vals(this->TraceLength(), FieldElementT_::Zero());
        std::vector<FieldElementT_> v_rebar_vals(this->TraceLength(), FieldElementT_::Zero());

        // Fires ONLY at the end of the 64 Blanking Rounds. Used strictly to bind the PK Fingerprint.
        s_identity_last_vals[4095 + 64] = FieldElementT_::One();

        // Continuous Hash Selectors
        std::vector<FieldElementT_> s_hash_active_vals(this->TraceLength(), FieldElementT_::Zero());
        std::vector<FieldElementT_> s_hash_transition_vals(this->TraceLength(), FieldElementT_::Zero());

        // Activate the sponge for Grid 0 PLUS 64 blanking rounds (4096 + 64 = 4160)
        size_t hash_end_step = valid_steps + 64;
        for(size_t step = 0; step < hash_end_step; ++step) {
            s_hash_active_vals[step] = FieldElementT_::One();
            // Stop the transition link one step early so next_state doesn't index out of bounds
            if (step < hash_end_step - 1) { 
                s_hash_transition_vals[step] = FieldElementT_::One();
            }
        }

        // Modulo math inherently arrays the boundaries across all K grids.
        // Capped at valid_steps so s_first/s_last do not evaluate ZK blinding noise.
        for (size_t step = 0; step < valid_steps; ++step) {
            if (step % 4096 == 0) s_first[step] = FieldElementT_::One();
            if (step % 4096 == 4095) s_last[step] = FieldElementT_::One();
        }

        // Call the deterministic generator for cryptographic rebar rotation
        // difficulty_R is stubbed as 0, so no rebar pins are used here
        /*
        std::vector<RebarPin> all_pins = get_crypto_rebar_pins(pub_inputs, alphabet);
        for (const auto& pin : all_pins) {
            size_t global_step = (pin.k * 4096) + (pin.r * 64) + pin.c;
            s_rebar_vals[global_step] = FieldElementT_::One();
            v_rebar_vals[global_step] = FieldElementT_::FromUint(pin.tile_id);
        }
        */

        int max_slack = (QwtssReference::total_defects_upper_bound - QwtssReference::total_defects_lower_bound);

        for (size_t step = 0; step < valid_steps; ++step) {
            size_t local_step = step % 4096;
            size_t r = local_step / 64;
            size_t c = local_step % 64;

            // If the public key contains the wildcard (-1), we leave the boundary 
            // selector as 0. This mathematically disables the constraint for this coordinate.
            if (r == 0 && pub_inputs.north[c] != QwtssPublicInputs::WILDCARD_COLOR) { 
                s_north[step] = FieldElementT_::One(); 
                v_north[step] = FieldElementT_::FromUint(pub_inputs.north[c]); 
            }
            if (r == 63 && pub_inputs.south[c] != QwtssPublicInputs::WILDCARD_COLOR) { 
                s_south[step] = FieldElementT_::One(); 
                v_south[step] = FieldElementT_::FromUint(pub_inputs.south[c]); 
            }
            if (c == 0 && pub_inputs.west[r] != QwtssPublicInputs::WILDCARD_COLOR) { 
                s_west[step]  = FieldElementT_::One(); 
                v_west[step]  = FieldElementT_::FromUint(pub_inputs.west[r]); 
            }
            if (c == 63 && pub_inputs.east[r] != QwtssPublicInputs::WILDCARD_COLOR) { 
                s_east[step]  = FieldElementT_::One(); 
                v_east[step]  = FieldElementT_::FromUint(pub_inputs.east[r]); 
            }

            // 1 for steps 10-4095 (Forces dummy_defect to be 0 after step 9)
            if (local_step >= (size_t)max_slack) { s_slack_limit[step] = FieldElementT_::One(); }
        }

        builder.AddPeriodicColumn(starkware::PeriodicColumn<FieldElementT_>(s_north, concrete_trace_gen, FieldElementT_::One(), this->TraceLength(), 1ULL), periodic_col_idx++);
        builder.AddPeriodicColumn(starkware::PeriodicColumn<FieldElementT_>(s_south, concrete_trace_gen, FieldElementT_::One(), this->TraceLength(), 1ULL), periodic_col_idx++);
        builder.AddPeriodicColumn(starkware::PeriodicColumn<FieldElementT_>(s_west, concrete_trace_gen, FieldElementT_::One(), this->TraceLength(), 1ULL), periodic_col_idx++);
        builder.AddPeriodicColumn(starkware::PeriodicColumn<FieldElementT_>(s_east, concrete_trace_gen, FieldElementT_::One(), this->TraceLength(), 1ULL), periodic_col_idx++);
        builder.AddPeriodicColumn(starkware::PeriodicColumn<FieldElementT_>(v_north, concrete_trace_gen, FieldElementT_::One(), this->TraceLength(), 1ULL), periodic_col_idx++);
        builder.AddPeriodicColumn(starkware::PeriodicColumn<FieldElementT_>(v_south, concrete_trace_gen, FieldElementT_::One(), this->TraceLength(), 1ULL), periodic_col_idx++);
        builder.AddPeriodicColumn(starkware::PeriodicColumn<FieldElementT_>(v_west, concrete_trace_gen, FieldElementT_::One(), this->TraceLength(), 1ULL), periodic_col_idx++);
        builder.AddPeriodicColumn(starkware::PeriodicColumn<FieldElementT_>(v_east, concrete_trace_gen, FieldElementT_::One(), this->TraceLength(), 1ULL), periodic_col_idx++);
        builder.AddPeriodicColumn(starkware::PeriodicColumn<FieldElementT_>(s_first, concrete_trace_gen, FieldElementT_::One(), this->TraceLength(), 1ULL), periodic_col_idx++);
        builder.AddPeriodicColumn(starkware::PeriodicColumn<FieldElementT_>(s_last, concrete_trace_gen, FieldElementT_::One(), this->TraceLength(), 1ULL), periodic_col_idx++);
        builder.AddPeriodicColumn(starkware::PeriodicColumn<FieldElementT_>(s_slack_limit, concrete_trace_gen, FieldElementT_::One(), this->TraceLength(), 1ULL), periodic_col_idx++);

        // 2 integer columns for the oracle planes
        std::vector<FieldElementT_> col_A_vals(this->TraceLength(), FieldElementT_::Zero());
        std::vector<FieldElementT_> col_B_vals(this->TraceLength(), FieldElementT_::Zero());
        for (size_t step = 0; step < valid_steps; ++step) {
            size_t local_step = step % 4096;
            col_A_vals[step] = FieldElementT_::FromUint(pub_inputs.plane_A[local_step]);
            col_B_vals[step] = FieldElementT_::FromUint(pub_inputs.plane_B[local_step]);
        }
        builder.AddPeriodicColumn(starkware::PeriodicColumn<FieldElementT_>(col_A_vals, concrete_trace_gen, FieldElementT_::One(), this->TraceLength(), 1ULL), periodic_col_idx++);
        builder.AddPeriodicColumn(starkware::PeriodicColumn<FieldElementT_>(col_B_vals, concrete_trace_gen, FieldElementT_::One(), this->TraceLength(), 1ULL), periodic_col_idx++);

        std::vector<FieldElementT_> s_col_0_vals(this->TraceLength(), FieldElementT_::Zero());
        std::vector<FieldElementT_> s_row_0_vals(this->TraceLength(), FieldElementT_::Zero());
        for (size_t step = 0; step < valid_steps; ++step) {
            size_t local_step = step % 4096;
            if (local_step % 64 == 0) s_col_0_vals[step] = FieldElementT_::One();   // First column of every row
            if (local_step < 64) s_row_0_vals[step] = FieldElementT_::One();        // First row
        }
        builder.AddPeriodicColumn(starkware::PeriodicColumn<FieldElementT_>(s_col_0_vals, concrete_trace_gen, FieldElementT_::One(), this->TraceLength(), 1ULL), periodic_col_idx++);
        builder.AddPeriodicColumn(starkware::PeriodicColumn<FieldElementT_>(s_row_0_vals, concrete_trace_gen, FieldElementT_::One(), this->TraceLength(), 1ULL), periodic_col_idx++);

        // Add Poseidon Dynamic Round Constants
        // Because the TraceLength is larger than 4096, initializing the vectors with FieldElementT_::Zero() ensures the
        // constants safely default to 0 for all noise rows outside the actual grid execution.
        std::vector<FieldElementT_> c0_vals(this->TraceLength(), FieldElementT_::Zero());
        std::vector<FieldElementT_> c1_vals(this->TraceLength(), FieldElementT_::Zero());
        std::vector<FieldElementT_> c2_vals(this->TraceLength(), FieldElementT_::Zero());
        std::vector<FieldElementT_> c3_vals(this->TraceLength(), FieldElementT_::Zero());
        std::vector<FieldElementT_> step_vals(this->TraceLength(), FieldElementT_::Zero());

        // Note: hash_end_step includes the final 64 blanking rounds
        for (size_t step = 0; step < hash_end_step; ++step) {
            size_t local_step = step % 4096;
            // Direct memory assignment from the pre-instantiated Magic Static.
            c0_vals[step] = PoseidonConstants::STONE_FAST_ROUND_CONSTANTS[local_step][0];
            c1_vals[step] = PoseidonConstants::STONE_FAST_ROUND_CONSTANTS[local_step][1];
            c2_vals[step] = PoseidonConstants::STONE_FAST_ROUND_CONSTANTS[local_step][2];
            c3_vals[step] = PoseidonConstants::STONE_FAST_ROUND_CONSTANTS[local_step][3];
            step_vals[step] = FieldElementT_::FromUint(local_step); // Injects domain separator
        }

        builder.AddPeriodicColumn(starkware::PeriodicColumn<FieldElementT_>(c0_vals, concrete_trace_gen, FieldElementT_::One(), this->TraceLength(), 1ULL), periodic_col_idx++);
        builder.AddPeriodicColumn(starkware::PeriodicColumn<FieldElementT_>(c1_vals, concrete_trace_gen, FieldElementT_::One(), this->TraceLength(), 1ULL), periodic_col_idx++);
        builder.AddPeriodicColumn(starkware::PeriodicColumn<FieldElementT_>(c2_vals, concrete_trace_gen, FieldElementT_::One(), this->TraceLength(), 1ULL), periodic_col_idx++);
        builder.AddPeriodicColumn(starkware::PeriodicColumn<FieldElementT_>(c3_vals, concrete_trace_gen, FieldElementT_::One(), this->TraceLength(), 1ULL), periodic_col_idx++);
        builder.AddPeriodicColumn(starkware::PeriodicColumn<FieldElementT_>(step_vals, concrete_trace_gen, FieldElementT_::One(), this->TraceLength(), 1ULL), periodic_col_idx++);
        builder.AddPeriodicColumn(starkware::PeriodicColumn<FieldElementT_>(s_identity_last_vals, concrete_trace_gen, FieldElementT_::One(), this->TraceLength(), 1ULL), periodic_col_idx++);
        builder.AddPeriodicColumn(starkware::PeriodicColumn<FieldElementT_>(s_rebar_vals, concrete_trace_gen, FieldElementT_::One(), this->TraceLength(), 1ULL), periodic_col_idx++);
        builder.AddPeriodicColumn(starkware::PeriodicColumn<FieldElementT_>(v_rebar_vals, concrete_trace_gen, FieldElementT_::One(), this->TraceLength(), 1ULL), periodic_col_idx++);
        builder.AddPeriodicColumn(starkware::PeriodicColumn<FieldElementT_>(s_hash_active_vals, concrete_trace_gen, FieldElementT_::One(), this->TraceLength(), 1ULL), periodic_col_idx++);
        builder.AddPeriodicColumn(starkware::PeriodicColumn<FieldElementT_>(s_hash_transition_vals, concrete_trace_gen, FieldElementT_::One(), this->TraceLength(), 1ULL), periodic_col_idx++);

        // 4. Wrap 'this' and build
        auto air_ptr = starkware::UseOwned<const QwtssAir>(this);

        return builder.BuildUniquePtr(
            std::move(air_ptr),
            concrete_trace_gen,
            this->TraceLength(),
            // Extract the concrete gsl::span from the polymorphic wrapper
            random_coefficients.As<FieldElementT_>(),
            {}, // point_exponents
            {}  // shifts
        );
    }

    // Tell Stone Prover how many columns are in the ExecutionTrace
    size_t NumColumns() const override {
        // 12 base columns + 3 intermediate columns + N one-hot tile boolean columns
        return NUM_BASE_TRACE_COLUMNS + alphabet.size();
    }

    // The number of coefficients for the linear combination of the constraints
    size_t NumRandomCoefficients() const override {
        // NUM_BASE_CONSTRAINTS (36) + N one-hot tile booleans + 1 exclusivity +
        //  7 Core32 constraints
        return NUM_BASE_CONSTRAINTS + alphabet.size() + 1 + 7;
    }

    // Required by air.h: Define the max degree for the FRI low-degree test
    uint64_t GetCompositionPolynomialDegreeBound() const override {
        // Note that STARK domains MUST be a power of 2.
        return 4 * this->TraceLength();
    }

    // Required by air.h: Provide interaction parameters (return nullopt for no interaction)
    std::optional<InteractionParams> GetInteractionParams() const override {
        return std::nullopt;
    }

    // Required by air.h: Parser for dynamic configuration
    std::vector<uint64_t> ParseDynamicParams(const std::map<std::string, uint64_t>& params) const override {
        return {};
    }

    // Returns a list of pairs (relative_row, col) that define the trace neighbors needed
    std::vector<std::pair<int64_t, uint64_t>> GetMask() const override {
        std::vector<std::pair<int64_t, uint64_t>> mask = {
            {0, 0}, {0, 1}, {0, 2}, {0, 3}, {0, 4}, // curr: h, v, d, alien_acc, acc,
            {0, 5}, {0, 6}, {0, 7}, {0, 8}, // curr hash_state 0, 1, 2, 3
            {0, 9},  // curr alien_inv
            {0, 10}, // curr core32_acc
            {0, 11}, {0, 12}, // curr h_seam_acc, curr v_seam_acc
            {0, 13}, {0, 14}, {0, 15}, // intermediate cols: curr half_check, core_defect, diff_A_B
            {0, 16}, {0, 17}, {0, 18}, {0, 19}, {0, 20} // is_alien, exp_hash0, exp_hash1, exp_hash2, exp_hash3
        };
        // Booleans now start at NUM_BASE_TRACE_COLUMNS
        for (size_t i = 0; i < alphabet.size(); ++i) mask.push_back({0, NUM_BASE_TRACE_COLUMNS + i});

        mask.push_back({1, 3}); // next_alien_acc
        mask.push_back({1, 4}); // next_acc
        mask.push_back({1, 5}); // next_hash_state_0
        mask.push_back({1, 6}); // next_hash_state_1
        mask.push_back({1, 7}); // next_hash_state_2
        mask.push_back({1, 8}); // next_hash_state_3
        mask.push_back({1, 10}); // next_core32_acc

        // Booleans now start at column 12
        for (size_t i = 0; i < alphabet.size(); ++i) mask.push_back({1, NUM_BASE_TRACE_COLUMNS + i});  // Next booleans
        for (size_t i = 0; i < alphabet.size(); ++i) mask.push_back({64, NUM_BASE_TRACE_COLUMNS + i}); // Bottom booleans

        // FORWARD-LOOKING OFFSETS (Since Stone Prover doesn't allow negative masks)
        mask.push_back({63, 0}); // The H-defect of the tile immediately to our bottom-left
        mask.push_back({64, 0}); // The H-defect of the tile immediately below us
        mask.push_back({64, 1}); // The V-defect of the tile immediately below us

        mask.push_back({1, t_h_seam_acc});   // Horizontal lookahead (+1 step)
        mask.push_back({64, t_v_seam_acc});  // Vertical lookahead (+64 steps)

        return mask;
    }

    // The official Stone Prover API for constraint evaluation
    // Use universal templates for the arguments, but strictly define the Fraction return type
    template <typename NeighborsT, typename PeriodicT, typename RandomT, typename PointT, typename ShiftsT, typename DomainEvalsT>
    starkware::FractionFieldElement<FieldElementT_> ConstraintsEval(
        const NeighborsT& neighbors,
        const PeriodicT& periodic_columns,
        const RandomT& random_coefficients,
        const PointT& point,
        const ShiftsT& shifts,
        const DomainEvalsT& domain_evals) const {

        // 1. Extract elements exactly in the order defined by GetMask()
        const auto& is_h_defect  = neighbors[t_is_h_defect]; const auto& is_v_defect  = neighbors[t_is_v_defect];
        const auto& dummy_defect = neighbors[t_dummy_defect]; const auto& alien_acc   = neighbors[t_alien_acc];
        const auto& defect_acc   = neighbors[t_defect_acc];
        const auto& current_state_0 = neighbors[t_hash_state_0];
        const auto& current_state_1 = neighbors[t_hash_state_1];
        const auto& current_state_2 = neighbors[t_hash_state_2];
        const auto& current_state_3 = neighbors[t_hash_state_3];
        const auto& alien_inv    = neighbors[t_alien_inv];
        const auto& core32_acc     = neighbors[t_core32_acc];
        const auto& h_seam_acc   = neighbors[t_h_seam_acc];
        const auto& v_seam_acc   = neighbors[t_v_seam_acc];
        const auto& t_half_check_val = neighbors[t_half_check];
        const auto& t_core_defect_val = neighbors[t_core_defect];
        const auto& t_diff_A_B_val = neighbors[t_diff_A_B];
        const auto& t_is_alien_val = neighbors[t_is_alien];
        const auto& t_exp_hash0_val = neighbors[t_expected_hash_0];
        const auto& t_exp_hash1_val = neighbors[t_expected_hash_1];
        const auto& t_exp_hash2_val = neighbors[t_expected_hash_2];
        const auto& t_exp_hash3_val = neighbors[t_expected_hash_3];

        // Dynamically extract the N one-hot tile boolean columns based on the alphabet size
        using PolyBoolRef = std::reference_wrapper<const std::remove_reference_t<decltype(is_h_defect)>>;
        size_t offset = NUM_BASE_TRACE_COLUMNS;

        std::vector<PolyBoolRef> b_array;
        for (size_t i = 0; i < alphabet.size(); ++i) b_array.push_back(neighbors[offset++]);

        const auto& next_alien_acc = neighbors[offset++];
        const auto& next_acc     = neighbors[offset++]; 
        const auto& next_state_0 = neighbors[offset++];
        const auto& next_state_1 = neighbors[offset++];
        const auto& next_state_2 = neighbors[offset++];
        const auto& next_state_3 = neighbors[offset++];
        const auto& next_core32  = neighbors[offset++];

        // 1.5. Dynamic Color Derivation (TileSet Agnostic)
        std::vector<PolyBoolRef> n_b_array;
        for (size_t i = 0; i < alphabet.size(); ++i) n_b_array.push_back(neighbors[offset++]);

        std::vector<PolyBoolRef> v_b_array;
        for (size_t i = 0; i < alphabet.size(); ++i) v_b_array.push_back(neighbors[offset++]);

        // Extract the new forward-looking offsets from near the end of the array
        // Because -2 and -1 are now correctly occupied by next_h_acc and next64_v_acc respectively.
        const auto& h_defect_bottom_left = neighbors[neighbors.size() - 5];
        const auto& h_defect_below       = neighbors[neighbors.size() - 4];
        const auto& v_defect_below       = neighbors[neighbors.size() - 3];

        // Helper lambda to construct the STARK algebra dynamically from any loaded ITileSet
        auto derive_colors = [this](const std::vector<PolyBoolRef>& bools) {
            auto top_acc = bools[0].get() * FieldElementT_::FromUint(this->alphabet[0].top);
            auto right_acc = bools[0].get() * FieldElementT_::FromUint(this->alphabet[0].right);
            auto bottom_acc = bools[0].get() * FieldElementT_::FromUint(this->alphabet[0].bottom);
            auto left_acc = bools[0].get() * FieldElementT_::FromUint(this->alphabet[0].left);

            for (size_t i = 1; i < this->alphabet.size(); ++i) {
                top_acc = top_acc + bools[i].get() * FieldElementT_::FromUint(this->alphabet[i].top);
                right_acc = right_acc + bools[i].get() * FieldElementT_::FromUint(this->alphabet[i].right);
                bottom_acc = bottom_acc + bools[i].get() * FieldElementT_::FromUint(this->alphabet[i].bottom);
                left_acc = left_acc + bools[i].get() * FieldElementT_::FromUint(this->alphabet[i].left);
            }
            return std::make_tuple(top_acc, right_acc, bottom_acc, left_acc);
        };

        // Unpack the resulting algebraic expressions using C++17 structured bindings
        auto [d_top, d_right, d_bottom, d_left] = derive_colors(b_array);
        auto [next_d_top, next_d_left_unused, next_d_bottom, next_d_left] = derive_colors(n_b_array);
        auto [bot_d_top, bot_d_right, bot_d_bottom, bot_d_left] = derive_colors(v_b_array);

        // 2. Extract Selectors Needed In Advance
        const auto& s_exe = periodic_columns[p_s_exe];
        const auto& s_h   = periodic_columns[p_s_h];
        const auto& s_v   = periodic_columns[p_s_v];
        const auto& s_core32 = periodic_columns[p_s_core32]; // 0:s_exe, 1:s_h, 2:s_v, 3:s_core32
        const auto& s_first = periodic_columns[p_s_first];
        const auto& s_last  = periodic_columns[p_s_last];
        const auto& s_col_0 = periodic_columns[p_s_col_0];
        const auto& s_row_0 = periodic_columns[p_s_row_0];
        const auto& s_hash_active = periodic_columns[p_s_hash_active];
        const auto& s_hash_transition = periodic_columns[p_s_hash_transition];

        // 3. Linearly combine constraints
        FieldElementT_ res = FieldElementT_::Zero();
        
        // SEVER THE TRANSITION AT THE BOUNDARY
        auto not_last = FieldElementT_::One() - s_last;

        res += random_coefficients[c_acc_transition] * (s_exe * not_last * (next_acc - (defect_acc + is_h_defect + is_v_defect + dummy_defect)));
        res += random_coefficients[c_h_match] * (s_h * (d_right - next_d_left) * (FieldElementT_::One() - is_h_defect));
        res += random_coefficients[c_v_match] * (s_v * (d_bottom - bot_d_top) * (FieldElementT_::One() - is_v_defect));
        res += random_coefficients[c_h_defect_bool] * (s_exe * (is_h_defect * (is_h_defect - FieldElementT_::One())));
        res += random_coefficients[c_v_defect_bool] * (s_exe * (is_v_defect * (is_v_defect - FieldElementT_::One())));
        res += random_coefficients[c_dummy_bool] * (s_exe * (dummy_defect * (dummy_defect - FieldElementT_::One())));

        // Alien Selective Counting (Slack Restored)
        // 1. Prover flags exactly 1500 tiles
        res += random_coefficients[c_alien_bool] * (s_exe * (t_is_alien_val * (t_is_alien_val - FieldElementT_::One())));
        // 2. If flagged (1), it MUST be a valid alien (inv proves difference from A/B is non-zero)
        res += random_coefficients[c_alien_inv] * (s_exe * (t_is_alien_val * ((t_diff_A_B_val * alien_inv) - FieldElementT_::One())));
        // 3. Accumulate ONLY the selectively flagged tiles
        res += random_coefficients[c_alien_transition] * (s_exe * not_last * (next_alien_acc - (alien_acc + t_is_alien_val)));

        // --- Horizontal Seam Transition (diff == 0 or 1) ---
        // Active only where r < 63 AND c < 63 (s_v * s_h)
        auto next_h_acc = neighbors[neighbors.size() - 2]; 
        auto h_diff = next_h_acc - h_seam_acc - neighbors[t_is_v_defect];
        res += random_coefficients[c_h_seam_diff] * (s_v * s_h * (h_diff * (h_diff - FieldElementT_::One())));

        // --- Horizontal Seam Start (acc == 0) ---
        // Active only where r < 63 AND c == 0
        res += random_coefficients[c_h_seam_start] * (s_v * s_col_0 * h_seam_acc);

        // --- Horizontal Seam End (Final step padding == 0 or 1) ---
        const auto max_line_target = FieldElementT_::FromUint(QwtssReference::line_max_upper_bound);
        // Active only where r < 63 AND c == 63 (represented by 1 - s_h)
        // This naturally absorbs the 64th defect and the final potential dummy padding!
        auto h_end_diff = max_line_target - h_seam_acc - neighbors[t_is_v_defect];
        res += random_coefficients[c_h_seam_end] * (s_v * (FieldElementT_::One() - s_h) * (h_end_diff * (h_end_diff - FieldElementT_::One())));

        // --- Vertical Seam Transition (diff == 0 or 1) ---
        // Active only where c < 63 AND r < 63 (s_h * s_v)
        // neighbors.size() - 1 is the t_v_seam_acc lookahead we appended to GetMask
        auto next64_v_acc = neighbors[neighbors.size() - 1]; 
        auto v_diff = next64_v_acc - v_seam_acc - neighbors[t_is_h_defect];
        res += random_coefficients[c_v_seam_diff] * (s_h * s_v * (v_diff * (v_diff - FieldElementT_::One())));

        // --- Vertical Seam Start (acc == 0) ---
        // Active only where c < 63 AND r == 0
        res += random_coefficients[c_v_seam_start] * (s_h * s_row_0 * v_seam_acc);

        // --- Vertical Seam End (Final step padding == 0 or 1) ---
        // Active only where c < 63 AND r == 63 (represented by 1 - s_v)
        auto v_end_diff = max_line_target - v_seam_acc - neighbors[t_is_h_defect];
        res += random_coefficients[c_v_seam_end] * (s_h * (FieldElementT_::One() - s_v) * (v_end_diff * (v_end_diff - FieldElementT_::One())));

        // Reconstruct the current Tile ID from the Booleans
        auto current_tile_id = FieldElementT_::Zero();
        for (size_t i = 0; i < alphabet.size(); ++i) {
            current_tile_id = current_tile_id + (b_array[i].get() * FieldElementT_::FromUint(i));
        }

        // Enforce the ZK Inverse Inequality (Optimized to degree 4)
        auto actual_diff_A_B = (current_tile_id - periodic_columns[p_col_A]) * (current_tile_id - periodic_columns[p_col_B]);
        // 1. Enforce the intermediate flattened calculation
        // (Note: The actual c_alien_inv validation using t_is_alien_val is handled securely in the Alien block above)
        res += random_coefficients[c_alien_diff_A_B_check] * (s_exe * (t_diff_A_B_val - actual_diff_A_B));

        // ONE-HOT ALPHABET CONSTRAINTS
        int rc_idx = NUM_BASE_CONSTRAINTS; // Start picking up from next spot: random_coefficients[19]

        // 1. Tile Boolean constraints: each b_i must be exactly 0 or 1.
        for (const auto& b : b_array) {
            res += random_coefficients[rc_idx++] * (s_exe * (b.get() * (b.get() - FieldElementT_::One())));
        }

        // 2. Exclusivity constraint: sum(b_i) == 1 (Only one tile can exist at a location)
        auto b_sum = b_array[0].get(); 
        for (size_t i = 1; i < alphabet.size(); ++i) {
            b_sum = b_sum + b_array[i].get();
        }
        res += random_coefficients[rc_idx++] * (s_exe * (b_sum - FieldElementT_::One()));

        // Add the Core32 Constraints to the polynomial (Expanded to 7 Equations to maintain algebraic degree 4)
        {
            // THE SHIFTED 4-WAY LOOKBACK (1 if any wall is defective, 0 if perfect):
            // We are evaluating whether the tile at (step + 64) is defective.
            // Top Wall of (step+64):    is_v_defect[step] (Current step V-defect)
            // Left Wall of (step+64):   is_h_defect[step + 63] (H-defect bottom-left)
            // Right Wall of (step+64):  is_h_defect[step + 64] (H-defect below)
            // Bottom Wall of (step+64): is_v_defect[step + 64] (V-defect below)
            auto T = FieldElementT_::One() - is_v_defect;
            auto L = FieldElementT_::One() - h_defect_bottom_left;
            auto R = FieldElementT_::One() - h_defect_below;
            auto B = FieldElementT_::One() - v_defect_below;
            //auto tile_has_defect = FieldElementT_::One() - (T * L * R * B);

            // 1. Enforce the first intermediate step: t_half_check == T * L
            res += random_coefficients[rc_idx++] * (s_core32 * (t_half_check_val - (T * L)));

            // 2. Enforce the second intermediate step INSIDE the core
            res += random_coefficients[rc_idx++] * (s_core32 * (t_core_defect_val - (FieldElementT_::One() - (t_half_check_val * R * B))));

            // 3. NEW: Enforce that t_core_defect_val is STRICTLY 0 OUTSIDE the core (Degree 3)
            res += random_coefficients[rc_idx++] * (s_exe * (FieldElementT_::One() - s_core32) * t_core_defect_val);

            // 4. Selective Core32 Accumulation
            auto core_diff = next_core32 - core32_acc;
            
            // Diff must be 0 or 1
            res += random_coefficients[rc_idx++] * (s_exe * not_last * (core_diff * (core_diff - FieldElementT_::One())));
            
            // If Diff is 1, the tile MUST actually be a core defect (Notice s_core32 is removed to keep it Degree 4!)
            res += random_coefficients[rc_idx++] * (s_exe * not_last * (core_diff * (t_core_defect_val - FieldElementT_::One())));
            
            // 5. Start Anchor
            res += random_coefficients[rc_idx++] * (s_first * core32_acc); 

            // 6. Boundary Evaluation
            res += random_coefficients[rc_idx++] * (s_last * (core32_acc - FieldElementT_::FromUint(QwtssReference::core32_lower_bound)));
        }

        // 3. We must unroll the C++ logic from poseidon_hash_step directly into polynomial math. Because Poseidon uses a
        // non-linear S-Box (x^3 in our case), a single step requires constraints of total degree 4.
        {
            // --- UNROLLED POSEIDON HASH CONSTRAINT (FULL SPONGE COMPLIANCE) ---

            // 1. Sponge Absorption (Add Tile ID to Rate 1, Step Index to Rate 2)
            // Elements 2 and 3 (Capacity) are left alone during absorption
            auto s0 = current_state_0 + current_tile_id + periodic_columns[p_c0]; 
            auto s1 = current_state_1 + periodic_columns[p_step] + periodic_columns[p_c1];
            auto s2 = current_state_2 + periodic_columns[p_c2];
            auto s3 = current_state_3 + periodic_columns[p_c3];

            // 2. Non-Linear S-Box (x^3)
            auto s0_2 = s0 * s0; auto s0_3 = s0_2 * s0;
            auto s1_2 = s1 * s1; auto s1_3 = s1_2 * s1;
            auto s2_2 = s2 * s2; auto s2_3 = s2_2 * s2;
            auto s3_2 = s3 * s3; auto s3_3 = s3_2 * s3;

            // 3. Full 4x4 MDS Matrix Multiplication using Magic Static
            auto expected_next_s0 = (s0_3 * PoseidonConstants::STONE_FAST_MDS[0][0]) + (s1_3 * PoseidonConstants::STONE_FAST_MDS[0][1]) + 
                                    (s2_3 * PoseidonConstants::STONE_FAST_MDS[0][2]) + (s3_3 * PoseidonConstants::STONE_FAST_MDS[0][3]);
            auto expected_next_s1 = (s0_3 * PoseidonConstants::STONE_FAST_MDS[1][0]) + (s1_3 * PoseidonConstants::STONE_FAST_MDS[1][1]) + 
                                    (s2_3 * PoseidonConstants::STONE_FAST_MDS[1][2]) + (s3_3 * PoseidonConstants::STONE_FAST_MDS[1][3]);
            auto expected_next_s2 = (s0_3 * PoseidonConstants::STONE_FAST_MDS[2][0]) + (s1_3 * PoseidonConstants::STONE_FAST_MDS[2][1]) + 
                                    (s2_3 * PoseidonConstants::STONE_FAST_MDS[2][2]) + (s3_3 * PoseidonConstants::STONE_FAST_MDS[2][3]);
            auto expected_next_s3 = (s0_3 * PoseidonConstants::STONE_FAST_MDS[3][0]) + (s1_3 * PoseidonConstants::STONE_FAST_MDS[3][1]) + 
                                    (s2_3 * PoseidonConstants::STONE_FAST_MDS[3][2]) + (s3_3 * PoseidonConstants::STONE_FAST_MDS[3][3]);

            /* Logic before the final hash blanking rounds were implemented
            // 4. Flattened Math Verification (Degree 4)
            res += random_coefficients[c_poseidon_math_0] * (s_exe * (expected_next_s0 - t_exp_hash0_val));
            res += random_coefficients[c_poseidon_math_1] * (s_exe * (expected_next_s1 - t_exp_hash1_val));
            res += random_coefficients[c_poseidon_math_2] * (s_exe * (expected_next_s2 - t_exp_hash2_val));
            res += random_coefficients[c_poseidon_math_3] * (s_exe * (expected_next_s3 - t_exp_hash3_val));

            // 5. Severed Transition (Degree 3)
            res += random_coefficients[c_poseidon_hash_0] * (s_exe * not_last * (next_state_0 - t_exp_hash0_val));
            res += random_coefficients[c_poseidon_hash_1] * (s_exe * not_last * (next_state_1 - t_exp_hash1_val));
            res += random_coefficients[c_poseidon_hash_2] * (s_exe * not_last * (next_state_2 - t_exp_hash2_val));
            res += random_coefficients[c_poseidon_hash_3] * (s_exe * not_last * (next_state_3 - t_exp_hash3_val));
            */

            // 4. Flattened Math Verification (Degree 4)
            res += random_coefficients[c_poseidon_math_0] * (s_hash_active * (expected_next_s0 - t_exp_hash0_val));
            res += random_coefficients[c_poseidon_math_1] * (s_hash_active * (expected_next_s1 - t_exp_hash1_val));
            res += random_coefficients[c_poseidon_math_2] * (s_hash_active * (expected_next_s2 - t_exp_hash2_val));
            res += random_coefficients[c_poseidon_math_3] * (s_hash_active * (expected_next_s3 - t_exp_hash3_val));

            // 5. Continuous Sponge Transition (Degree 2)
            res += random_coefficients[c_poseidon_hash_0] * (s_hash_transition * (next_state_0 - t_exp_hash0_val));
            res += random_coefficients[c_poseidon_hash_1] * (s_hash_transition * (next_state_1 - t_exp_hash1_val));
            res += random_coefficients[c_poseidon_hash_2] * (s_hash_transition * (next_state_2 - t_exp_hash2_val));
            res += random_coefficients[c_poseidon_hash_3] * (s_hash_transition * (next_state_3 - t_exp_hash3_val));
        }

        // 4. Extract boundary periodic columns
        const auto& s_north = periodic_columns[p_s_north];
        const auto& s_south = periodic_columns[p_s_south];
        const auto& s_west  = periodic_columns[p_s_west];
        const auto& s_east  = periodic_columns[p_s_east];
        const auto& v_north = periodic_columns[p_v_north];
        const auto& v_south = periodic_columns[p_v_south];
        const auto& v_west  = periodic_columns[p_v_west];
        const auto& v_east  = periodic_columns[p_v_east];

        const auto& s_slack_limit = periodic_columns[p_s_slack_limit];

        // Apply grid edge boundary constraints using dynamically derived colors
        res += random_coefficients[c_bound_north]  * (s_north * (d_top - v_north));
        res += random_coefficients[c_bound_south]  * (s_south * (d_bottom - v_south));
        res += random_coefficients[c_bound_west]  * (s_west  * (d_left - v_west));
        res += random_coefficients[c_bound_east] * (s_east  * (d_right - v_east));

        // Apply Accumulator Start/End (In-Place Evaluation)
        res += random_coefficients[c_acc_start] * (s_first * defect_acc);
        auto final_acc = defect_acc + is_h_defect + is_v_defect + dummy_defect;
        res += random_coefficients[c_acc_end] * (s_last * (final_acc - FieldElementT_::FromUint(QwtssReference::total_defects_upper_bound)));

        // Apply Fingerprint Start/End (Initialize the sponge to all zeros)
        res += random_coefficients[c_hash_start_0] * (s_first * current_state_0);
        res += random_coefficients[c_hash_start_1] * (s_first * current_state_1);
        res += random_coefficients[c_hash_start_2] * (s_first * current_state_2);
        res += random_coefficients[c_hash_start_3] * (s_first * current_state_3);

        // STARK Hash Binding: We ONLY check this on Grid 0 using s_identity_last
        // We use the flattened expected hashes because the trace transition to row 4096 was severed.
        const auto& s_identity_last = periodic_columns[p_s_identity_last];
        res += random_coefficients[c_hash_end_0] * (s_identity_last * (t_exp_hash0_val - ParseFe256(pub_inputs.grid_fingerprint[0])));
        res += random_coefficients[c_hash_end_1] * (s_identity_last * (t_exp_hash1_val - ParseFe256(pub_inputs.grid_fingerprint[1])));

        // Alien Tile Boundaries
        res += random_coefficients[c_alien_start] * (s_first * alien_acc); // FIX: Anchor to 0
        auto final_alien = alien_acc + t_is_alien_val;
        res += random_coefficients[c_alien_end] * (s_last * (final_alien - FieldElementT_::FromUint(QwtssReference::alien_tiles_lower_bound)));

        // CRYPTOGRAPHIC REBAR ENFORCEMENT
        const auto& s_rebar = periodic_columns[p_s_rebar];
        const auto& v_rebar = periodic_columns[p_v_rebar];
        res += random_coefficients[c_rebar_pin] * (s_rebar * (current_tile_id - v_rebar));

        // ZK BLANKING ROUND SECURITY LOCK (Degree 3)
        // Because s_exe == 0 during the hash blanking rounds, the STARK stops checking the Tile Booleans.
        // We MUST mathematically force current_tile_id=0 to prevent hash forgery.
        auto not_in_grid = FieldElementT_::One() - s_exe;
        res += random_coefficients[c_blanking_zero_lock] * (s_hash_active * not_in_grid * current_tile_id);

        // Enforce Slack Limit (dummy must be 0 when s_slack_limit is 1)
        res += random_coefficients[c_slack_limit] * (s_slack_limit * dummy_defect);

        // 5. Calculate the Vanishing Polynomial: Z(x) = x^N - 1
        // Since TraceLength is a power of 2, we can compute x^N using rapid squaring
        auto z_x = point; 
        size_t length_n = this->TraceLength();

        while (length_n > 1) {
            z_x = z_x * z_x;
            length_n /= 2;
        }

        auto denominator = z_x - FieldElementT_::One();

        // 6. Return the algebraic fraction (Numerator = Constraints, Denominator = Z(x))
        // If the trace violates ANY rule, the numerator won't be 0, the division will result 
        // in a rational fraction with a pole (infinite degree), and the verifier will fail.
        return starkware::FractionFieldElement<FieldElementT_>(res, denominator);
    }

    template <typename PointPowersT, typename ShiftsT>
    std::vector<FieldElementT_> DomainEvalsAtPoint(
        const PointPowersT& point_powers,
        const ShiftsT& shifts) const {
        return {};
    }

    template <typename CosetOffsetT, typename TraceGenT, typename PointExpT, typename ShiftsT>
    std::vector<std::vector<FieldElementT_>> PrecomputeDomainEvalsOnCoset(
        const CosetOffsetT& coset_offset,
        const TraceGenT& trace_generator,
        const PointExpT& point_exponents,
        const ShiftsT& shifts) const {
        return {};
    }

    // STARK ALGEBRA UNIT TEST: Evaluates the core domain constraints (Rows 0 to K*4096) directly against a raw trace 
    // to verify mathematical correctness BEFORE launching the Stone Prover.
    // @return True on success; otherwise False on failure.
    bool validate_trace(const ExecutionTrace& trace) const {
        // Ensure the unit test validates the final hash blanking rounds
        size_t test_steps = trace.valid_steps + 64;

        // Pre-compute expected Rebar constraints to avoid running the PRNG inside the execution loop
        std::vector<FieldElementT_> expected_s_rebar(test_steps, FieldElementT_::Zero());
        std::vector<FieldElementT_> expected_v_rebar(test_steps, FieldElementT_::Zero());
        // difficulty_R is stubbed as 0, so no rebar pins are used here
        /*
        auto all_pins = get_crypto_rebar_pins(pub_inputs, alphabet);
        for (const auto& pin : all_pins) {
            size_t global_step = (pin.k * 4096) + (pin.r * 64) + pin.c;
            if (global_step < trace.valid_steps) {
                expected_s_rebar[global_step] = FieldElementT_::One();
                expected_v_rebar[global_step] = FieldElementT_::FromUint(pin.tile_id);
            }
        }
        */

        for (size_t step = 0; step < test_steps; ++step) {
            
            size_t local_step = step % 4096;
            int r = local_step / 64;
            int c = local_step % 64;
            int k = step / 4096; // Which grid are we in?
            
            // True for the identity grid, False for the blanking phase
            bool is_valid_grid_step = (step < trace.valid_steps);

            // Mock the "Periodic Columns" for this step
            std::vector<FieldElementT_> periodic;
            periodic.reserve(NUM_PERIODIC_COLUMNS);

            periodic.push_back(is_valid_grid_step ? FieldElementT_::One() : FieldElementT_::Zero()); // p_s_exe
            periodic.push_back((is_valid_grid_step && c != 63) ? FieldElementT_::One() : FieldElementT_::Zero()); // p_s_h
            periodic.push_back((is_valid_grid_step && local_step < 4032) ? FieldElementT_::One() : FieldElementT_::Zero()); // p_s_v

            // s_core32
            bool in_core = (r >= 15 && r < 47 && c >= 16 && c < 48);
            periodic.push_back((is_valid_grid_step && in_core) ? FieldElementT_::One() : FieldElementT_::Zero());
            
            // --- Boundary Columns ---
            periodic.push_back((is_valid_grid_step && r == 0 && pub_inputs.north[c] != QwtssPublicInputs::WILDCARD_COLOR) ? FieldElementT_::One() : FieldElementT_::Zero());
            periodic.push_back((is_valid_grid_step && r == 63 && pub_inputs.south[c] != QwtssPublicInputs::WILDCARD_COLOR) ? FieldElementT_::One() : FieldElementT_::Zero());
            periodic.push_back((is_valid_grid_step && c == 0 && pub_inputs.west[r] != QwtssPublicInputs::WILDCARD_COLOR) ? FieldElementT_::One() : FieldElementT_::Zero());
            periodic.push_back((is_valid_grid_step && c == 63 && pub_inputs.east[r] != QwtssPublicInputs::WILDCARD_COLOR) ? FieldElementT_::One() : FieldElementT_::Zero());

            periodic.push_back((is_valid_grid_step && r == 0 && pub_inputs.north[c] != QwtssPublicInputs::WILDCARD_COLOR) ? FieldElementT_::FromUint(pub_inputs.north[c]) : FieldElementT_::Zero());
            periodic.push_back((is_valid_grid_step && r == 63 && pub_inputs.south[c] != QwtssPublicInputs::WILDCARD_COLOR) ? FieldElementT_::FromUint(pub_inputs.south[c]) : FieldElementT_::Zero());
            periodic.push_back((is_valid_grid_step && c == 0 && pub_inputs.west[r] != QwtssPublicInputs::WILDCARD_COLOR) ? FieldElementT_::FromUint(pub_inputs.west[r]) : FieldElementT_::Zero());
            periodic.push_back((is_valid_grid_step && c == 63 && pub_inputs.east[r] != QwtssPublicInputs::WILDCARD_COLOR) ? FieldElementT_::FromUint(pub_inputs.east[r]) : FieldElementT_::Zero());

            // --- Positional Anchors ---
            periodic.push_back((is_valid_grid_step && local_step == 0) ? FieldElementT_::One() : FieldElementT_::Zero()); // s_first
            periodic.push_back((is_valid_grid_step && local_step == 4095) ? FieldElementT_::One() : FieldElementT_::Zero()); // s_last

            int max_slack = (QwtssReference::total_defects_upper_bound - QwtssReference::total_defects_lower_bound);
            periodic.push_back((is_valid_grid_step && local_step >= (size_t)max_slack) ? FieldElementT_::One() : FieldElementT_::Zero()); // s_slack
            
            periodic.push_back(is_valid_grid_step ? FieldElementT_::FromUint(pub_inputs.plane_A[local_step]) : FieldElementT_::Zero()); 
            periodic.push_back(is_valid_grid_step ? FieldElementT_::FromUint(pub_inputs.plane_B[local_step]) : FieldElementT_::Zero()); 

            periodic.push_back((is_valid_grid_step && c == 0) ? FieldElementT_::One() : FieldElementT_::Zero()); // p_s_col_0
            periodic.push_back((is_valid_grid_step && r == 0) ? FieldElementT_::One() : FieldElementT_::Zero()); // p_s_row_0

            periodic.push_back(PoseidonConstants::STONE_FAST_ROUND_CONSTANTS[local_step][0]); 
            periodic.push_back(PoseidonConstants::STONE_FAST_ROUND_CONSTANTS[local_step][1]); 
            periodic.push_back(PoseidonConstants::STONE_FAST_ROUND_CONSTANTS[local_step][2]); 
            periodic.push_back(PoseidonConstants::STONE_FAST_ROUND_CONSTANTS[local_step][3]);
            periodic.push_back(FieldElementT_::FromUint(local_step)); // p_step

            periodic.push_back((step == 4159) ? FieldElementT_::One() : FieldElementT_::Zero()); // p_s_identity_last

            // Read the pre-computed Rebar pins for this specific step
            periodic.push_back(expected_s_rebar[step]);
            periodic.push_back(expected_v_rebar[step]);

            // Continuous Sponge Selectors
            periodic.push_back((step < 4160) ? FieldElementT_::One() : FieldElementT_::Zero()); // p_s_hash_active
            periodic.push_back((step < 4159) ? FieldElementT_::One() : FieldElementT_::Zero()); // p_s_hash_transition

            // 3. Mock the "Neighbors" extraction based on GetMask() order
            std::vector<FieldElementT_> neighbors;
            
            // Current State [0-20]
            neighbors.push_back(FieldElementT_::FromUint(trace.is_h_defect[step]));
            neighbors.push_back(FieldElementT_::FromUint(trace.is_v_defect[step]));
            neighbors.push_back(FieldElementT_::FromUint(trace.dummy_defect[step]));
            neighbors.push_back(FieldElementT_::FromUint(trace.alien_acc[step]));
            neighbors.push_back(ParseFe256(trace.defect_accumulator[step]));
            neighbors.push_back(ParseFe256(trace.hash_state_0[step]));
            neighbors.push_back(ParseFe256(trace.hash_state_1[step]));
            neighbors.push_back(ParseFe256(trace.hash_state_2[step]));
            neighbors.push_back(ParseFe256(trace.hash_state_3[step]));
            neighbors.push_back(ParseFe256(trace.alien_inv[step]));
            neighbors.push_back(FieldElementT_::FromUint(trace.core32_acc[step]));
            neighbors.push_back(FieldElementT_::FromUint(trace.h_seam_acc[step]));
            neighbors.push_back(FieldElementT_::FromUint(trace.v_seam_acc[step]));
            neighbors.push_back(FieldElementT_::FromUint(trace.half_check[step]));
            neighbors.push_back(FieldElementT_::FromUint(trace.core_defect[step]));
            neighbors.push_back(ParseFe256(trace.diff_A_B[step]));
            
            // NEW FLATTENED COLUMNS
            neighbors.push_back(FieldElementT_::FromUint(trace.is_alien[step]));
            neighbors.push_back(ParseFe256(trace.expected_hash_0[step]));
            neighbors.push_back(ParseFe256(trace.expected_hash_1[step]));
            neighbors.push_back(ParseFe256(trace.expected_hash_2[step]));
            neighbors.push_back(ParseFe256(trace.expected_hash_3[step]));

            // Current Booleans
            for (size_t i = 0; i < alphabet.size(); ++i) neighbors.push_back(FieldElementT_::FromUint(trace.tile_booleans[i][step]));

            // Next State (Ensure we don't index out of bounds)
            size_t next_idx = (step + 1 < trace.trace_length) ? step + 1 : step;
            neighbors.push_back(FieldElementT_::FromUint(trace.alien_acc[next_idx]));
            neighbors.push_back(ParseFe256(trace.defect_accumulator[next_idx]));
            neighbors.push_back(ParseFe256(trace.hash_state_0[next_idx]));
            neighbors.push_back(ParseFe256(trace.hash_state_1[next_idx]));
            neighbors.push_back(ParseFe256(trace.hash_state_2[next_idx]));
            neighbors.push_back(ParseFe256(trace.hash_state_3[next_idx]));
            neighbors.push_back(FieldElementT_::FromUint(trace.core32_acc[next_idx]));

            // Next Booleans
            for (size_t i = 0; i < alphabet.size(); ++i) neighbors.push_back(FieldElementT_::FromUint(trace.tile_booleans[i][next_idx]));

            // Bottom Booleans
            size_t bot_idx = (step + 64 < trace.trace_length) ? step + 64 : step;
            for (size_t i = 0; i < alphabet.size(); ++i) {
                if (step + 64 < trace.trace_length) neighbors.push_back(FieldElementT_::FromUint(trace.tile_booleans[i][bot_idx]));
                else neighbors.push_back(FieldElementT_::Zero());
            }

            // Mock the lookbacks
            size_t bl_idx = (step + 63 < trace.trace_length) ? step + 63 : step;
            if (step + 63 < trace.trace_length) neighbors.push_back(FieldElementT_::FromUint(trace.is_h_defect[bl_idx]));
            else neighbors.push_back(FieldElementT_::Zero());

            if (step + 64 < trace.trace_length) neighbors.push_back(FieldElementT_::FromUint(trace.is_h_defect[bot_idx]));
            else neighbors.push_back(FieldElementT_::Zero());

            if (step + 64 < trace.trace_length) neighbors.push_back(FieldElementT_::FromUint(trace.is_v_defect[bot_idx]));
            else neighbors.push_back(FieldElementT_::Zero());

            if (step + 1 < trace.trace_length) neighbors.push_back(FieldElementT_::FromUint(trace.h_seam_acc[next_idx]));
            else neighbors.push_back(FieldElementT_::Zero());

            if (step + 64 < trace.trace_length) neighbors.push_back(FieldElementT_::FromUint(trace.v_seam_acc[bot_idx]));
            else neighbors.push_back(FieldElementT_::Zero());

            // 4. Evaluate Constraints Individually
            for (size_t c_idx = 0; c_idx < NumRandomCoefficients(); ++c_idx) {
                std::vector<FieldElementT_> one_hot_rc(NumRandomCoefficients(), FieldElementT_::Zero());
                one_hot_rc[c_idx] = FieldElementT_::One();

                try {
                    auto result = ConstraintsEval(
                        neighbors, periodic, gsl::make_span(one_hot_rc), 
                        FieldElementT_::Zero(), std::vector<FieldElementT_>(), std::vector<FieldElementT_>()  
                    );

                    if (!(result == starkware::FractionFieldElement<FieldElementT_>::Zero())) {
                        std::string c_name = "Unknown";
                        if (c_idx == c_acc_transition) c_name = "Defect Accumulator Transition";
                        else if (c_idx == c_h_match) c_name = "Horizontal Match Check";
                        else if (c_idx == c_v_match) c_name = "Vertical Match Check";
                        else if (c_idx == c_h_defect_bool) c_name = "H-Defect Boolean Check";
                        else if (c_idx == c_v_defect_bool) c_name = "V-Defect Boolean Check";
                        else if (c_idx == c_dummy_bool) c_name = "Dummy Defect Boolean Check";
                        else if (c_idx == c_poseidon_hash_0) c_name = "Poseidon Hash 0 Transition";
                        else if (c_idx == c_poseidon_hash_1) c_name = "Poseidon Hash 1 Transition";
                        else if (c_idx == c_poseidon_hash_2) c_name = "Poseidon Hash 2 Transition";
                        else if (c_idx == c_poseidon_hash_3) c_name = "Poseidon Hash 3 Transition";
                        else if (c_idx == c_poseidon_math_0) c_name = "Poseidon Math 0 S-Box Verification";
                        else if (c_idx == c_poseidon_math_1) c_name = "Poseidon Math 1 S-Box Verification";
                        else if (c_idx == c_poseidon_math_2) c_name = "Poseidon Math 2 S-Box Verification";
                        else if (c_idx == c_poseidon_math_3) c_name = "Poseidon Math 3 S-Box Verification";
                        else if (c_idx == c_bound_north) c_name = "Grid Boundary North Check";
                        else if (c_idx == c_bound_south) c_name = "Grid Boundary South Check";
                        else if (c_idx == c_bound_west) c_name = "Grid Boundary West Check";
                        else if (c_idx == c_bound_east) c_name = "Grid Boundary East Check";
                        else if (c_idx == c_acc_start) c_name = "Accumulator Start Boundary";
                        else if (c_idx == c_acc_end) c_name = "Accumulator End Boundary";
                        else if (c_idx == c_hash_start_0) c_name = "Hash 0 Start Boundary";
                        else if (c_idx == c_hash_start_1) c_name = "Hash 1 Start Boundary";
                        else if (c_idx == c_hash_start_2) c_name = "Hash 2 Start Boundary";
                        else if (c_idx == c_hash_start_3) c_name = "Hash 3 Start Boundary";
                        else if (c_idx == c_hash_end_0) c_name = "Hash End Boundary 0";
                        else if (c_idx == c_hash_end_1) c_name = "Hash End Boundary 1";
                        else if (c_idx == c_alien_end) c_name = "Alien Accumulator End Boundary";
                        else if (c_idx == c_alien_start) c_name = "Alien Accumulator Start Boundary";
                        else if (c_idx == c_slack_limit) c_name = "Slack Limit Enforcement";
                        else if (c_idx == c_alien_bool) c_name = "Alien Tile Boolean Check (0 or 1)";
                        else if (c_idx == c_alien_inv) c_name = "Alien ZK Inverse Inequality Check";
                        else if (c_idx == c_alien_transition) c_name = "Alien Accumulator Transition";
                        else if (c_idx == c_alien_diff_A_B_check) c_name = "Alien Diff A B Value Verification";
                        else if (c_idx == c_h_seam_diff) c_name = "Horizontal Seam Transition (0 or 1)";
                        else if (c_idx == c_h_seam_start) c_name = "Horizontal Seam Start Anchor (0)";
                        else if (c_idx == c_h_seam_end) c_name = "Horizontal Seam End Anchor (Limit)";
                        else if (c_idx == c_v_seam_diff) c_name = "Vertical Seam Transition (0 or 1)";
                        else if (c_idx == c_v_seam_start) c_name = "Vertical Seam Start Anchor (0)";
                        else if (c_idx == c_v_seam_end) c_name = "Vertical Seam End Anchor (Limit)";
                        else if (c_idx == c_rebar_pin) c_name = "Cryptographic Rebar Pin Verification";
                        else if (c_idx == c_blanking_zero_lock) c_name = "Blanking Round Tile Zero Lock";
                        else if (c_idx >= NUM_BASE_CONSTRAINTS && c_idx < NUM_BASE_CONSTRAINTS + alphabet.size()) c_name = "Tile One-Hot Boolean Check";
                        else if (c_idx == NUM_BASE_CONSTRAINTS + alphabet.size()) c_name = "Tile Exclusivity Check (Sum == 1)";
                        else if (c_idx == NUM_BASE_CONSTRAINTS + alphabet.size() + 1) c_name = "Core32 Half Check (T * L)";
                        else if (c_idx == NUM_BASE_CONSTRAINTS + alphabet.size() + 2) c_name = "Core32 Defect Check (Inside Core)";
                        else if (c_idx == NUM_BASE_CONSTRAINTS + alphabet.size() + 3) c_name = "Core32 Out-of-Core Zeroing";
                        else if (c_idx == NUM_BASE_CONSTRAINTS + alphabet.size() + 4) c_name = "Core32 Transition Boolean Check";
                        else if (c_idx == NUM_BASE_CONSTRAINTS + alphabet.size() + 5) c_name = "Core32 Transition Validity Check";
                        else if (c_idx == NUM_BASE_CONSTRAINTS + alphabet.size() + 6) c_name = "Core32 Start Anchor";
                        else if (c_idx == NUM_BASE_CONSTRAINTS + alphabet.size() + 7) c_name = "Core32 Boundary Check";
                        
                        std::cerr << "Error: AIR Constraint " << c_idx << " (\"" << c_name << "\") failed at global trace step " << step << "." << std::endl;
                        std::cerr << "\tGrid " << k << " Coordinate: (" << r << ", " << c << ")" << std::endl;
                        return false;
                    }
                } catch (const std::exception& e) {
                    std::cerr << "Error: AIR Constraint Evaluation threw an exception at trace step " << step << "." << std::endl;
                    throw;
                }
            }
        }
        return true;   
    }

private:
    // Stone Prover needs to know the algebraic degree of each constraint to 
    // allocate the correct amount of memory for the quotient polynomials.
    std::vector<size_t> GetConstraintDegrees() const {
        std::vector<size_t> degrees = {
            3, 3, 3, 3, 3, 3,       // 0-5 (Acc gets +1 from not_last multiplier)
            2, 2, 2, 2,             // 6-9 (Hash Transition: s_hash_transition * (next - exp) is Degree 2)
            4, 4, 4, 4,             // 10-13 (Hash Math checks)
            2, 2, 2, 2,             // 14-17 (Bounds N, S, W, E)
            2, 2,                   // 18-19 (Acc Start/End)
            2, 2, 2, 2,             // 20-23 (Hash Start 0, 1, 2, 3)
            2, 2,                   // 24-25 (Hash End 0, 1: s_identity_last * (t_exp_hash - fingerprint))
            2,                      // 26 (Alien End)
            2,                      // 27 (Alien Start: s_first * alien_acc)
            2,                      // 28 (Slack limit)
            3,                      // 29 (Alien Bool: s_exe * is_alien * (is_alien - 1))
            4,                      // 30 (Alien Inv: s_exe * is_alien * (diff_A_B * inv - 1))
            3,                      // 31 (Alien Transition: s_exe * not_last * (next - (acc + is_alien)))
            3,                      // 32 (Alien Diff A B Check)
            4, 3, 4,                // 33-35 (H Seam Diff, Start, End)
            4, 3, 4,                // 36-38 (V Seam Diff, Start, End)
            2,                      // 39 (Rebar Pin Constraint: s_rebar * (tile - v_rebar))
            3                       // 40 (Blanking Zero Lock: s_hash_active * (1 - s_exe) * current_tile_id)
        };

        for(size_t i = 0; i < alphabet.size(); ++i) degrees.push_back(3); // Booleans: s_exe(1) * b(1) * b(1) = 3
        degrees.push_back(2); // Exclusivity: s_exe(1) * b_sum(1) = 2

        // Core32 Constraints (Flattened to 7 constraints)
        degrees.push_back(3); // Core32 Half Check
        degrees.push_back(4); // Core32 Defect Check
        degrees.push_back(3); // Core32 Out-of-Core Zeroing
        degrees.push_back(4); // Core32 Transition Boolean
        degrees.push_back(4); // Core32 Transition Validity Check (Degree 4)
        degrees.push_back(2); // Core32 Start Anchor
        degrees.push_back(2); // Core32 Boundary

        return degrees;
    }

    // Safely reads Big-Endian trace arrays back into STARK field elements
    FieldElementT_ ParseFe256(const FieldElement256& raw) const {
        auto span = gsl::make_span(reinterpret_cast<const std::byte*>(raw.data()), 32);
        // Symmetric raw memory load
        return FieldElementT_::FromBytes(span, false); 
    }
};
