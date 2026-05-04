#include "QwtssAirTypes.h"
#include "QwtssCore.h"
#include "CryptoUtils.h"
#include <cstring>
#include <cstdlib>


QwtssPublicInputs::QwtssPublicInputs(const std::string& username, uint64_t identity_nonce, uint8_t version,
        const QwtssPrivateKey &private_key, int grid_size, const std::array<FieldElement256, 2>& final_grid_hash_state,
        const std::vector<Tile> &alphabet)
{
    if (username.empty()) throw std::invalid_argument("username must be specified");
    if (grid_size < 32) throw std::invalid_argument("grid_size is expected to be 64 for QWTSS");
    if (private_key.private_key.size() != grid_size * grid_size) {
        throw std::invalid_argument("private_key.private_key must be specified and the proper size");
    }
    if (private_key.boundary_mask.size() != grid_size * grid_size) {
        throw std::invalid_argument("private_key.boundary_mask must be specified and the proper size");
    }
    if (alphabet.size() != 11) throw std::invalid_argument("alphabet.size() is expected to be 11 for QWTSS");

    this->username = username;
    this->identity_nonce = identity_nonce;
    this->version = version;

    plane_A = private_key.plane_A;
    plane_B = private_key.plane_B;

    north.resize(grid_size);
    south.resize(grid_size);
    east.resize(grid_size);
    west.resize(grid_size);

    for(int i = 0; i < grid_size; i++) {
        // Calculate the absolute 1D grid indices for the four edges
        int idx_north = i;                               // Row 0
        int idx_south = (grid_size - 1) * grid_size + i; // Row 63
        int idx_west  = i * grid_size;                   // Col 0
        int idx_east  = i * grid_size + (grid_size - 1); // Col 63

        // Apply wildcard if the boundary tile is not explicitly pinned (mask == 1)
        north[i] = (private_key.boundary_mask[idx_north] == 1) ? alphabet[private_key.private_key[idx_north]].top    : WILDCARD_COLOR;
        south[i] = (private_key.boundary_mask[idx_south] == 1) ? alphabet[private_key.private_key[idx_south]].bottom : WILDCARD_COLOR;
        west[i]  = (private_key.boundary_mask[idx_west] == 1) ? alphabet[private_key.private_key[idx_west]].left     : WILDCARD_COLOR;
        east[i]  = (private_key.boundary_mask[idx_east] == 1) ? alphabet[private_key.private_key[idx_east]].right    : WILDCARD_COLOR;
    }

    // Populate the first two elements (Rate 0 and Rate 1) of the fingerprint array, zero the rest
    grid_fingerprint[0] = final_grid_hash_state[0];
    grid_fingerprint[1] = final_grid_hash_state[1];
    for(int i = 2; i < 4; i++) {
        grid_fingerprint[i] = {0, 0, 0, 0};
    }
}

QwtssPublicInputs::QwtssPublicInputs(const std::string& username, uint64_t identity_nonce, uint8_t version,
        const PkDerivedFields& pk_fields, int grid_size, const std::array<FieldElement256, 2>& final_grid_hash_state,
        const std::vector<Tile> &alphabet)
{
    if (username.empty()) throw std::invalid_argument("username must be specified");
    if (grid_size < 32) throw std::invalid_argument("grid_size is expected to be 64 for QWTSS");
    if (pk_fields.oracle_pk.pk_mask.size() != grid_size * grid_size) {
        throw std::invalid_argument("pk_fields.oracle_pk.pk_mask must be specified and the proper length");
    }
    if (pk_fields.boundary_mask.size() != grid_size * grid_size) {
        throw std::invalid_argument("pk_fields.boundary_mask must be specified and the proper size");
    }
    if (alphabet.size() != 11) throw std::invalid_argument("alphabet.size() is expected to be 11 for QWTSS");

    this->username = username;
    this->identity_nonce = identity_nonce;
    this->version = version;

    // Extract the quasiperiodic planes from the oracle output
    plane_A = pk_fields.oracle_pk.plane_A;
    plane_B = pk_fields.oracle_pk.plane_B;

    north.resize(grid_size);
    south.resize(grid_size);
    east.resize(grid_size);
    west.resize(grid_size);

    for(int i = 0; i < grid_size; i++) {
        // Calculate the absolute 1D grid indices for the four edges
        int idx_north = i;                               // Row 0
        int idx_south = (grid_size - 1) * grid_size + i; // Row 63
        int idx_west  = i * grid_size;                   // Col 0
        int idx_east  = i * grid_size + (grid_size - 1); // Col 63

        // Apply wildcard if the boundary tile is not explicitly pinned (mask == 1)
        // Extract the required color directly from the public_key_boundary array
        north[i] = (pk_fields.boundary_mask[idx_north] == 1) ? alphabet[pk_fields.oracle_pk.pk_mask[idx_north]].top    : WILDCARD_COLOR;
        south[i] = (pk_fields.boundary_mask[idx_south] == 1) ? alphabet[pk_fields.oracle_pk.pk_mask[idx_south]].bottom : WILDCARD_COLOR;
        west[i]  = (pk_fields.boundary_mask[idx_west]  == 1) ? alphabet[pk_fields.oracle_pk.pk_mask[idx_west]].left     : WILDCARD_COLOR;
        east[i]  = (pk_fields.boundary_mask[idx_east]  == 1) ? alphabet[pk_fields.oracle_pk.pk_mask[idx_east]].right    : WILDCARD_COLOR;
    }

    // Extract fingerprint from the last valid step of the hash column
    // Populate the first two elements (Rate 0 and Rate 1) of the fingerprint array, zero the rest
    grid_fingerprint[0] = final_grid_hash_state[0];
    grid_fingerprint[1] = final_grid_hash_state[1];
    for(int i = 2; i < 4; i++) {
        grid_fingerprint[i] = {0, 0, 0, 0};
    }
}

QwtssPublicInputs::QwtssPublicInputs(const std::string& username, uint64_t identity_nonce, uint8_t version, int grid_size) {
    if (username.empty()) throw std::invalid_argument("username must be specified");
    if (grid_size < 32) throw std::invalid_argument("grid_size is expected to be 64 for QWTSS");

    this->username = username;
    this->identity_nonce = identity_nonce;
    this->version = version;

    north.resize(grid_size);
    south.resize(grid_size);
    east.resize(grid_size);
    west.resize(grid_size);

    // Explicitly zero-initialize the hash state array.
    for(int i = 0; i < 4; i++) {
        grid_fingerprint[i] = {0, 0, 0, 0}; 
    }
}

// Serializes the public inputs and an arbitrary message into a flat byte array
// used to securely initialize the Fiat-Shamir heuristic transcript.
std::vector<std::byte> QwtssPublicInputs::serialize_for_fiat_shamir(const std::vector<uint8_t>& message) const {
    size_t capacity = 0;
    capacity += sizeof(uint64_t); // For the username length prefix
    capacity += username.size();
    capacity += sizeof(identity_nonce);
    capacity += sizeof(version);
    capacity += north.size() * sizeof(FieldElement);
    capacity += south.size() * sizeof(FieldElement);
    capacity += east.size() * sizeof(FieldElement);
    capacity += west.size() * sizeof(FieldElement);
    capacity += sizeof(grid_fingerprint);
    capacity += plane_A.size() * sizeof(int);
    capacity += plane_B.size() * sizeof(int);
    capacity += message.size();

    std::vector<std::byte> seed;
    seed.reserve(capacity);

    // Helper lambda for clean, memory-safe byte appending
    auto append_bytes = [&seed](const auto* data, size_t size_in_bytes) {
        const std::byte* byte_ptr = reinterpret_cast<const std::byte*>(data);
        seed.insert(seed.end(), byte_ptr, byte_ptr + size_in_bytes);
    };

    // Inject the Domain Context (Identity & Parameters)
    uint64_t u_len = username.size();
    append_bytes(&u_len, sizeof(u_len));    // Prefix username length to prevent any Variable-Length Serialization Collision attack
    append_bytes(username.data(), username.size());
    append_bytes(&identity_nonce, sizeof(identity_nonce));
    append_bytes(&version, sizeof(version));

    // Inject the 4 boundary edges
    append_bytes(north.data(), north.size() * sizeof(FieldElement));
    append_bytes(south.data(), south.size() * sizeof(FieldElement));
    append_bytes(east.data(), east.size() * sizeof(FieldElement));
    append_bytes(west.data(), west.size() * sizeof(FieldElement));

    // Inject the zk-PoW fingerprint and the ZK plane commitments
    append_bytes(grid_fingerprint.data(), sizeof(grid_fingerprint));
    append_bytes(plane_A.data(), plane_A.size() * sizeof(int));
    append_bytes(plane_B.data(), plane_B.size() * sizeof(int));

    // Append the arbitrary message being signed
    if (!message.empty()) {
        append_bytes(message.data(), message.size());
    }

    return seed;
}

void QwtssPublicInputs::log_critical_fields_hash(const std::string& context_name) const {
    SHA256 main_sha;
    std::cout << "\n--- [DEBUG] " << context_name << " Public Inputs Breakdown ---\n";

    // Helper lambda to hash a field individually, log it, and add it to the running aggregate
    auto hash_and_log = [&main_sha](const std::string& field_name, const auto* data, size_t size_in_bytes) {
        if (size_in_bytes > 0) {
            // 1. Compute the isolated hash for just this specific field
            SHA256 field_sha;
            field_sha.update(reinterpret_cast<const uint8_t*>(data), size_in_bytes);
            
            std::cout << "  " << std::setw(15) << std::left << field_name 
                      << " : " << field_sha.digest() << "\n";
            
            // 2. Add the raw bytes to the main running stream
            main_sha.update(reinterpret_cast<const uint8_t*>(data), size_in_bytes);
        }
    };

    // 1. Hash the 4 boundary edges
    hash_and_log("North Edge", north.data(), north.size() * sizeof(FieldElement));
    hash_and_log("South Edge", south.data(), south.size() * sizeof(FieldElement));
    hash_and_log("East Edge", east.data(), east.size() * sizeof(FieldElement));
    hash_and_log("West Edge", west.data(), west.size() * sizeof(FieldElement));

    // 2. Hash the fingerprint (std::array is contiguous memory)
    hash_and_log("Fingerprint", grid_fingerprint.data(), sizeof(grid_fingerprint));

    // 3. Hash the Oracle Planes
    hash_and_log("Plane A", plane_A.data(), plane_A.size() * sizeof(int));
    hash_and_log("Plane B", plane_B.data(), plane_B.size() * sizeof(int));

    std::cout << "  " << std::setw(15) << std::left << "AGGREGATE" 
              << " : " << main_sha.digest() << "\n";
    std::cout << "------------------------------------------------------\n";
}
