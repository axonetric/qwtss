#pragma once
#include <string>
#include <vector>
#include <cstdint>
#include <array>
#include <random>
#include <sstream>
#include <iomanip>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <utility> // for std::swap


class SHA256 {
private:
    uint32_t state[8];
    uint8_t data[64];
    uint32_t datalen;
    uint64_t bitlen;

    static constexpr uint32_t K[64] = {
        0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
        0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
        0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
        0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
        0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
        0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
        0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
        0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2
    };

    static inline uint32_t rotr(uint32_t x, uint32_t n) { return (x >> n) | (x << (32 - n)); }
    static inline uint32_t ch(uint32_t x, uint32_t y, uint32_t z) { return (x & y) ^ (~x & z); }
    static inline uint32_t maj(uint32_t x, uint32_t y, uint32_t z) { return (x & y) ^ (x & z) ^ (y & z); }
    static inline uint32_t ep0(uint32_t x) { return rotr(x, 2) ^ rotr(x, 13) ^ rotr(x, 22); }
    static inline uint32_t ep1(uint32_t x) { return rotr(x, 6) ^ rotr(x, 11) ^ rotr(x, 25); }
    static inline uint32_t sig0(uint32_t x) { return rotr(x, 7) ^ rotr(x, 18) ^ (x >> 3); }
    static inline uint32_t sig1(uint32_t x) { return rotr(x, 17) ^ rotr(x, 19) ^ (x >> 10); }

    void transform() {
        uint32_t a, b, c, d, e, f, g, h, i, j, t1, t2, m[64];

        for (i = 0, j = 0; i < 16; ++i, j += 4)
            m[i] = (data[j] << 24) | (data[j + 1] << 16) | (data[j + 2] << 8) | (data[j + 3]);
        for ( ; i < 64; ++i)
            m[i] = sig1(m[i - 2]) + m[i - 7] + sig0(m[i - 15]) + m[i - 16];

        a = state[0]; b = state[1]; c = state[2]; d = state[3];
        e = state[4]; f = state[5]; g = state[6]; h = state[7];

        for (i = 0; i < 64; ++i) {
            t1 = h + ep1(e) + ch(e, f, g) + K[i] + m[i];
            t2 = ep0(a) + maj(a, b, c);
            h = g; g = f; f = e; e = d + t1;
            d = c; c = b; b = a; a = t1 + t2;
        }

        state[0] += a; state[1] += b; state[2] += c; state[3] += d;
        state[4] += e; state[5] += f; state[6] += g; state[7] += h;
    }

public:
    SHA256() {
        state[0] = 0x6a09e667; state[1] = 0xbb67ae85; state[2] = 0x3c6ef372; state[3] = 0xa54ff53a;
        state[4] = 0x510e527f; state[5] = 0x9b05688c; state[6] = 0x1f83d9ab; state[7] = 0x5be0cd19;
        datalen = 0;
        bitlen = 0;
    }

    void update(const uint8_t* data_in, size_t len) {
        for (size_t i = 0; i < len; ++i) {
            data[datalen] = data_in[i];
            datalen++;
            if (datalen == 64) {
                transform();
                bitlen += 512;
                datalen = 0;
            }
        }
    }

    void update(const std::string& data_in) {
        update(reinterpret_cast<const uint8_t*>(data_in.c_str()), data_in.length());
    }

    std::string digest() {
        uint32_t i = datalen;
        
        if (datalen < 56) {
            data[i++] = 0x80;
            while (i < 56) data[i++] = 0x00;
        } else {
            data[i++] = 0x80;
            while (i < 64) data[i++] = 0x00;
            transform();
            memset(data, 0, 56);
        }

        bitlen += datalen * 8;
        data[63] = bitlen;
        data[62] = bitlen >> 8;
        data[61] = bitlen >> 16;
        data[60] = bitlen >> 24;
        data[59] = bitlen >> 32;
        data[58] = bitlen >> 40;
        data[57] = bitlen >> 48;
        data[56] = bitlen >> 56;
        transform();

        std::stringstream ss;
        ss << std::hex << std::setfill('0');
        for (i = 0; i < 4; ++i) {
            ss << std::setw(2) << ((state[0] >> (24 - i * 8)) & 0x000000ff);
            ss << std::setw(2) << ((state[1] >> (24 - i * 8)) & 0x000000ff);
            ss << std::setw(2) << ((state[2] >> (24 - i * 8)) & 0x000000ff);
            ss << std::setw(2) << ((state[3] >> (24 - i * 8)) & 0x000000ff);
            ss << std::setw(2) << ((state[4] >> (24 - i * 8)) & 0x000000ff);
            ss << std::setw(2) << ((state[5] >> (24 - i * 8)) & 0x000000ff);
            ss << std::setw(2) << ((state[6] >> (24 - i * 8)) & 0x000000ff);
            ss << std::setw(2) << ((state[7] >> (24 - i * 8)) & 0x000000ff);
        }
        return ss.str();
    }

    std::vector<uint8_t> digest_bytes() {
        uint32_t i = datalen;
        
        // --- Identical Legacy Padding Logic ---
        if (datalen < 56) {
            data[i++] = 0x80;
            while (i < 56) data[i++] = 0x00;
        } else {
            data[i++] = 0x80;
            while (i < 64) data[i++] = 0x00;
            transform();
            memset(data, 0, 56);
        }

        bitlen += datalen * 8;
        data[63] = bitlen;
        data[62] = bitlen >> 8;
        data[61] = bitlen >> 16;
        data[60] = bitlen >> 24;
        data[59] = bitlen >> 32;
        data[58] = bitlen >> 40;
        data[57] = bitlen >> 48;
        data[56] = bitlen >> 56;
        transform();
        // --------------------------------------

        // Pack the 8x 32-bit state registers into a 32-byte vector (Big-Endian)
        std::vector<uint8_t> hash_bytes(32);
        for (i = 0; i < 8; ++i) {
            hash_bytes[i * 4]     = (state[i] >> 24) & 0xFF;
            hash_bytes[i * 4 + 1] = (state[i] >> 16) & 0xFF;
            hash_bytes[i * 4 + 2] = (state[i] >> 8)  & 0xFF;
            hash_bytes[i * 4 + 3] =  state[i]        & 0xFF;
        }
        
        return hash_bytes;
    }
    
    // Static helper for one-shot string hashing
    static std::string hash_string(const std::string& input) {
        SHA256 sha;
        sha.update(input);
        return sha.digest();
    }

    // Static helper for one-shot raw byte hashing of strings
    static std::vector<uint8_t> hash_string_bytes(const std::string& input) {
        SHA256 sha;
        sha.update(input);
        return sha.digest_bytes();
    }

    // Static helper for hashing files efficiently without loading them entirely into RAM
    static std::string hash_file(const std::string& filepath) {
        std::ifstream file(filepath, std::ios::binary);
        if (!file.is_open()) {
            throw std::runtime_error("SHA256::hash_file error: Could not open file " + filepath);
        }

        SHA256 sha;
        // 64 KB buffer is typically the sweet spot for OS disk caching and I/O performance
        const size_t buffer_size = 65536; 
        std::vector<uint8_t> buffer(buffer_size);

        while (file.good()) {
            file.read(reinterpret_cast<char*>(buffer.data()), buffer_size);
            std::streamsize bytes_read = file.gcount();
            
            if (bytes_read > 0) {
                sha.update(buffer.data(), static_cast<size_t>(bytes_read));
            }
        }

        return sha.digest();
    }

    // Static helper for highly efficient raw byte hashing of files
    static std::vector<uint8_t> hash_file_bytes(const std::string& filepath) {
        std::ifstream file(filepath, std::ios::binary);
        if (!file.is_open()) {
            throw std::runtime_error("SHA256::hash_file_bytes error: Could not open " + filepath);
        }

        SHA256 sha;
        const size_t buffer_size = 65536; // 64 KB chunks
        std::vector<uint8_t> buffer(buffer_size);

        while (file.good()) {
            file.read(reinterpret_cast<char*>(buffer.data()), buffer_size);
            std::streamsize bytes_read = file.gcount();
            
            if (bytes_read > 0) {
                sha.update(buffer.data(), static_cast<size_t>(bytes_read));
            }
        }

        return sha.digest_bytes();
    }
};

class ChaCha20PRNG {
private:
    uint32_t state[16];
    uint64_t buffer[8];
    int buffer_idx = 8; // Forces block generation on first call

    // 32-bit Entropy Cache
    uint32_t cached_u32;
    bool has_cached_u32 = false;

    static inline uint32_t rotl32(uint32_t x, int n) {
        return (x << n) | (x >> (32 - n));
    }

    static inline void quarter_round(uint32_t state[16], int a, int b, int c, int d) {
        state[a] += state[b]; state[d] ^= state[a]; state[d] = rotl32(state[d], 16);
        state[c] += state[d]; state[b] ^= state[c]; state[b] = rotl32(state[b], 12);
        state[a] += state[b]; state[d] ^= state[a]; state[d] = rotl32(state[d], 8);
        state[c] += state[d]; state[b] ^= state[c]; state[b] = rotl32(state[b], 7);
    }

    void generate_block() {
        uint32_t working_state[16];
        for (int i = 0; i < 16; i++) working_state[i] = state[i];

        for (int i = 0; i < 10; i++) {
            // Column rounds
            quarter_round(working_state, 0, 4, 8, 12);
            quarter_round(working_state, 1, 5, 9, 13);
            quarter_round(working_state, 2, 6, 10, 14);
            quarter_round(working_state, 3, 7, 11, 15);
            // Diagonal rounds
            quarter_round(working_state, 0, 5, 10, 15);
            quarter_round(working_state, 1, 6, 11, 12);
            quarter_round(working_state, 2, 7, 8, 13);
            quarter_round(working_state, 3, 4, 9, 14);
        }

        // Add working state back to original state
        for (int i = 0; i < 16; i++) {
            working_state[i] += state[i];
        }
        
        // Assemble 64-bit words safely using bitwise logic to prevent Strict Aliasing UB
        // and guarantee cross-platform Little-Endian adherence.
        for(int i = 0; i < 8; i++) {
            buffer[i] = static_cast<uint64_t>(working_state[2 * i]) | 
                       (static_cast<uint64_t>(working_state[2 * i + 1]) << 32);
        }

        state[12]++; // Increment block counter
        if (state[12] == 0) state[13]++; // Handle counter overflow
        buffer_idx = 0;
    }

public:
    // --- C++ Standard URBG Interface (UniformRandomBitGenerator) ---
    // This allows ChaCha20PRNG to be used natively with std::uniform_int_distribution
    using result_type = uint64_t;

    static constexpr result_type min() {
        return 0;
    }

    static constexpr result_type max() {
        return 0xFFFFFFFFFFFFFFFFull; // UINT64_MAX
    }

    result_type operator()() {
        return next_u64();
    }
    // ---------------------------------------------------------------

    // Construct with a true hardware random seed
    ChaCha20PRNG() {
        std::random_device rd;
        
        // "expand 32-byte k" constants
        state[0] = 0x61707865; state[1] = 0x3320646e;
        state[2] = 0x79622d32; state[3] = 0x6b206574;
        
        // 256-bit Key from OS Entropy
        for (int i = 4; i < 12; i++) state[i] = rd();
        
        // Counter
        state[12] = 0; state[13] = 0;
        
        // 64-bit Nonce from OS Entropy
        state[14] = rd(); state[15] = rd();
    }

    // Construct deterministically from a 256-bit hash (8x 32-bit words)
    ChaCha20PRNG(const std::array<uint32_t, 8>& deterministic_key) {
        // "expand 32-byte k" constants
        state[0] = 0x61707865; state[1] = 0x3320646e;
        state[2] = 0x79622d32; state[3] = 0x6b206574;
        
        // 256-bit Key from the provided deterministic seed
        for (int i = 0; i < 8; i++) {
            state[4 + i] = deterministic_key[i];
        }
        
        // Counter
        state[12] = 0; state[13] = 0;
        
        // 64-bit Nonce (Fixed to 0 since the key encapsulates all entropy)
        state[14] = 0; state[15] = 0;

        buffer_idx = 8; // Force block generation on first call
    }

    // Get the next 64-bit random field element
    uint64_t next_u64() {
        if (buffer_idx >= 8) {
            generate_block();
        }
        return buffer[buffer_idx++];
    }

    // Get the next 32-bit random field element efficiently (caches any leftover 64-bit entropy)
    uint32_t next_u32() {
        if (has_cached_u32) {
            has_cached_u32 = false;
            return cached_u32;
        } else {
            uint64_t val = next_u64();
            cached_u32 = static_cast<uint32_t>(val >> 32); // Cache the upper 32 bits
            has_cached_u32 = true;                         // Mark cache as full
            return static_cast<uint32_t>(val);             // Return the lower 32 bits
        }
    }

    // 100% Cross-Platform Deterministic Float in [0.0, 1.0)
    float next_float() {
        // IEEE 754 single-precision floats have 24 bits of mantissa precision
        uint32_t val = static_cast<uint32_t>(next_u64() & 0xFFFFFF);
        return static_cast<float>(val) / 16777216.0f; // Divide by 2^24
    }

    // ------------------------------------------------------------------
    // Deterministic Cross-Platform Distributions & Shuffles
    // ------------------------------------------------------------------

    // Returns a 100% Cross-Platform Deterministic Bounded Integer: [min, max].
    // Uses unbiased Rejection Sampling to eliminate modulo bias.
    // @param min Inclusive.
    // @param max Inclusive.
    uint32_t next_u32_range(uint32_t min, uint32_t max) {
        if (min >= max) return min;
        uint32_t range = max - min;
        
        // Compute the smallest bitmask that fully covers the range
        uint32_t mask = range;
        mask |= mask >> 1;
        mask |= mask >> 2;
        mask |= mask >> 4;
        mask |= mask >> 8;
        mask |= mask >> 16;

        uint32_t val;
        do {
            // Extract the lower 32 bits from the engine and apply the mask
            val = static_cast<uint32_t>(next_u64() & 0xFFFFFFFF) & mask;
            
            // If the value falls outside the range, reject and draw again.
            // This mathematically guarantees zero bias.
        } while (val > range);

        return min + val;
    }

    // Performs a 100% cross-platform deterministic, cryptographically unbiased Fisher-Yates Shuffle.
    template <typename T>
    void deterministic_shuffle(std::vector<T>& vec) {
        if (vec.empty()) return;
        
        // Iterate from the last element down to the second element
        for (size_t i = vec.size() - 1; i > 0; --i) {
            // Draw a completely unbiased, deterministic index between 0 and i (inclusive)
            uint32_t j = next_u32_range(0, static_cast<uint32_t>(i));
            
            // Swap the elements
            std::swap(vec[i], vec[j]);
        }
    }
};

// Helper function to safely parse a 16-character hex substring into a 64-bit integer
inline uint64_t hex_to_u64(const std::string& hex_str) {
    uint64_t val;
    std::stringstream ss;
    ss << std::hex << hex_str;
    ss >> val;
    return val;
}

inline std::string bytes_to_hex(const std::vector<uint8_t>& bytes) {
    std::stringstream ss;
    ss << std::hex << std::setfill('0');
    for (uint8_t b : bytes) {
        ss << std::setw(2) << static_cast<int>(b);
    }
    return ss.str();
}

inline std::vector<uint8_t> hex_to_bytes(const std::string& hex) {
    if (hex.length() % 2 != 0) throw std::invalid_argument("Hex string must have an even length");
    
    std::vector<uint8_t> bytes;
    bytes.reserve(hex.length() / 2);
    for (size_t i = 0; i < hex.length(); i += 2) {
        std::string byteString = hex.substr(i, 2);
        uint8_t byte = static_cast<uint8_t>(std::strtol(byteString.c_str(), nullptr, 16));
        bytes.push_back(byte);
    }
    return bytes;
}

inline std::string generate_random_username(ChaCha20PRNG& rng, int min_len = 6, int max_len = 20) {
    // Standard valid characters for a username (alphanumeric)
    const std::string charset = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
    
    // 1. Determine a random length between 6 and 20
    std::uniform_int_distribution<int> dist_len(min_len, max_len);
    int length = dist_len(rng);
    
    // 2. Setup the uniform distribution for the character pool
    std::uniform_int_distribution<int> dist_char(0, charset.length() - 1);
    
    std::string username;
    username.reserve(length); // Pre-allocate memory for speed
    
    // 3. Build the string
    for (int i = 0; i < length; ++i) {
        username += charset[dist_char(rng)];
    }
    
    return username;
}

static const std::string base64_chars = 
             "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
             "abcdefghijklmnopqrstuvwxyz"
             "0123456789+/";

static inline bool is_base64(uint8_t c) {
  return (isalnum(c) || (c == '+') || (c == '/'));
}

inline std::string base64_encode(const uint8_t* buf, unsigned int bufLen) {
    std::string ret;
    int i = 0;
    int j = 0;
    uint8_t char_array_3[3];
    uint8_t char_array_4[4];

    ret.reserve((bufLen + 2) / 3 * 4); // Pre-allocate to prevent reallocation

    while (bufLen--) {
        char_array_3[i++] = *(buf++);
        if (i == 3) {
            char_array_4[0] = (char_array_3[0] & 0xfc) >> 2;
            char_array_4[1] = ((char_array_3[0] & 0x03) << 4) + ((char_array_3[1] & 0xf0) >> 4);
            char_array_4[2] = ((char_array_3[1] & 0x0f) << 2) + ((char_array_3[2] & 0xc0) >> 6);
            char_array_4[3] = char_array_3[2] & 0x3f;

            for(i = 0; (i <4) ; i++)
                ret += base64_chars[char_array_4[i]];
            i = 0;
        }
    }

    if (i) {
        for(j = i; j < 3; j++)
            char_array_3[j] = '\0';

        char_array_4[0] = (char_array_3[0] & 0xfc) >> 2;
        char_array_4[1] = ((char_array_3[0] & 0x03) << 4) + ((char_array_3[1] & 0xf0) >> 4);
        char_array_4[2] = ((char_array_3[1] & 0x0f) << 2) + ((char_array_3[2] & 0xc0) >> 6);
        char_array_4[3] = char_array_3[2] & 0x3f;

        for (j = 0; (j < i + 1); j++)
            ret += base64_chars[char_array_4[j]];

        while((i++ < 3))
            ret += '=';
    }

    return ret;
}

inline std::vector<uint8_t> base64_decode(std::string const& encoded_string) {
    int in_len = encoded_string.size();
    int i = 0;
    int j = 0;
    int in_ = 0;
    uint8_t char_array_4[4], char_array_3[3];
    std::vector<uint8_t> ret;
    
    ret.reserve(in_len * 3 / 4);

    while (in_len-- && ( encoded_string[in_] != '=') && is_base64(encoded_string[in_])) {
        char_array_4[i++] = encoded_string[in_]; in_++;
        if (i ==4) {
            for (i = 0; i <4; i++)
                char_array_4[i] = base64_chars.find(char_array_4[i]);

            char_array_3[0] = (char_array_4[0] << 2) + ((char_array_4[1] & 0x30) >> 4);
            char_array_3[1] = ((char_array_4[1] & 0xf) << 4) + ((char_array_4[2] & 0x3c) >> 2);
            char_array_3[2] = ((char_array_4[2] & 0x3) << 6) + char_array_4[3];

            for (i = 0; (i < 3); i++)
                ret.push_back(char_array_3[i]);
            i = 0;
        }
    }

    if (i) {
        for (j = i; j <4; j++)
            char_array_4[j] = 0;

        for (j = 0; j <4; j++)
            char_array_4[j] = base64_chars.find(char_array_4[j]);

        char_array_3[0] = (char_array_4[0] << 2) + ((char_array_4[1] & 0x30) >> 4);
        char_array_3[1] = ((char_array_4[1] & 0xf) << 4) + ((char_array_4[2] & 0x3c) >> 2);
        char_array_3[2] = ((char_array_4[2] & 0x3) << 6) + char_array_4[3];

        for (j = 0; (j < i - 1); j++) ret.push_back(char_array_3[j]);
    }

    return ret;
}
