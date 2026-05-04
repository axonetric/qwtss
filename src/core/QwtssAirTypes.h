#pragma once
#include "Tiles.h"
#include <vector>
#include <cstdint>
#include <cstddef>
#include <array>
#include "starkware/algebra/fields/prime_field_element.h"

// Standard 64-bit data payload for trace elements
using FieldElement = uint64_t; 

// For StarkWare's 252-bit prime field.
using FieldElement256 = std::array<uint64_t, 4>;

// Directs Stone Prover templates to compile using 252-bit prime field arithmetic
using StoneField = starkware::PrimeFieldElement<252, 0>;

// Forward declares
struct QwtssPrivateKey;
struct PkDerivedFields;


// Safe Conversion Helpers
inline FieldElement256 to_fe256(const StoneField& sf) {
    FieldElement256 res = {0};
    auto span = gsl::make_span(reinterpret_cast<std::byte*>(res.data()), 32);
    // Use symmetric raw memory dump (keeps Montgomery form intact)
    sf.ToBytes(span, false); 
    return res;
}

inline StoneField from_fe256(const FieldElement256& raw) {
    auto span = gsl::make_span(reinterpret_cast<const std::byte*>(raw.data()), 32);
    // Symmetric raw memory load
    return StoneField::FromBytes(span, false); 
}

static const std::array<FieldElement256, 2> ZERO_FINGERPRINT = {to_fe256(StoneField::FromUint(0)), to_fe256(StoneField::FromUint(0))};

struct QwtssPublicInputs {
    // Define the wildcard sentinel explicitly as the max 64-bit value
    static constexpr FieldElement WILDCARD_COLOR = -1ULL;

    std::string username;
    uint64_t identity_nonce;
    uint8_t version; // Protocol Version Byte

    // The grid boundary colors (or WILDCARD_COLOR)
    std::vector<FieldElement> north; // 64 colors
    std::vector<FieldElement> south; // 64 colors
    std::vector<FieldElement> east;  // 64 colors
    std::vector<FieldElement> west;  // 64 colors

    // The Poseidon hash digest of the exact grid configuration. This proves identity of the key holder.
    std::array<FieldElement256, 4> grid_fingerprint;

    std::vector<int> plane_A;
    std::vector<int> plane_B;

    // Convenient ctor for signing pipeline
    QwtssPublicInputs(const std::string& username, uint64_t identity_nonce, uint8_t version,
        const QwtssPrivateKey &private_key, int grid_size, const std::array<FieldElement256, 2>& final_grid_hash_state,
        const std::vector<Tile> &alphabet);

    // Convenient ctor for verification pipeline
    QwtssPublicInputs(const std::string& username, uint64_t identity_nonce, uint8_t version,
        const PkDerivedFields& pk_fields, int grid_size, const std::array<FieldElement256, 2>& final_grid_hash_state,
        const std::vector<Tile> &alphabet);

    // Convenient ctor for internal testing
    QwtssPublicInputs(const std::string& username, uint64_t identity_nonce, uint8_t version, int grid_size);

    std::vector<std::byte> serialize_for_fiat_shamir(const std::vector<uint8_t>& message = {}) const;

    // Debug helper to verify deterministic state between Prover and Verifier
    void log_critical_fields_hash(const std::string& context_name = "") const;
};
