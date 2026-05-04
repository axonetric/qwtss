class GrainLFSR:
    def __init__(self, field_size_bits, t, R_F, R_P):
        # Poseidon Grain LFSR initialization encoding
        # 1 (GF(p)) | 0 (x^3 S-box) | field_size_bits | t | R_F | R_P | 1s
        sbox_type = 0 # 0 for x^3, 1 for x^5, 2 for x^7
        state_int = (1 << 78) | (sbox_type << 74) | (field_size_bits << 62) | (t << 50) | (R_F << 40) | (R_P << 30) | 0x3FFFFFFF
        self.state = [int(x) for x in f"{state_int:080b}"]
        
        # Initialize by clocking 160 times
        for _ in range(160):
            self.get_bit()

    def get_bit(self):
        # Grain LFSR feedback polynomial
        new_bit = self.state[62] ^ self.state[51] ^ self.state[38] ^ self.state[23] ^ self.state[13] ^ self.state[0]
        self.state.pop(0)
        self.state.append(new_bit)
        return new_bit

    def get_field_element(self, P):
        while True:
            val = 0
            # Generate a 252-bit integer
            for i in range(252):
                val |= (self.get_bit() << (251 - i))
            
            # Rejection sampling (must be strictly less than prime)
            if val < P:
                return val

# Generate the 12,288 constants
lfsr = GrainLFSR(field_size_bits=252, t=4, R_F=4096, R_P=0)
P = 0x800000000000011000000000000000000000000000000000000000000000001

round_constants = []
for _ in range(4096 * 4):
    round_constants.append(lfsr.get_field_element(P))

print(f"Successfully generated {len(round_constants)} constants.")

print(f"Writing {len(round_constants)} constants to PoseidonConstants.h...")

with open("PoseidonConstants.h", "w") as f:
    f.write("#pragma once\n")
    f.write("#include <cstdint>\n\n")
    f.write("namespace PoseidonConstants {\n")
    f.write("    // 16,384 constants (4,096 rounds * 4 elements per round)\n")
    f.write("    // Format: [4] represents the 256-bit field element as four 64-bit Little-Endian chunks\n")
    f.write("    inline constexpr uint64_t ROUND_CONSTANTS_EXTENDED[16384][4] = {\n")
    
    for val in round_constants:
        # Split the 252-bit integer into 4x 64-bit chunks for the Stone Prover
        chunks = [(val >> (64 * k)) & 0xFFFFFFFFFFFFFFFF for k in range(4)]
        f.write(f"        {{0x{chunks[0]:016x}, 0x{chunks[1]:016x}, 0x{chunks[2]:016x}, 0x{chunks[3]:016x}}},\n")
        
    f.write("    };\n")
    f.write("}\n")

print("File 'PoseidonConstants.h' generated successfully.")


