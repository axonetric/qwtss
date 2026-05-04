# STARK Prime: P = 2^251 + 17 * 2^192 + 1
P = 0x800000000000011000000000000000000000000000000000000000000000001

x = [0, 1, 2, 3]
y = [4, 5, 6, 7]

print("constexpr uint64_t MDS[4][4][4] = {") # [4] for the 4x 64-bit STARK chunks
for i in range(4):
    print("    {", end="")
    for j in range(4):
        # Calculate inverse modulo P
        val = pow(x[i] + y[j], -1, P)
        
        # Convert to 4x 64-bit chunks for C++ STARK library (Little Endian arrays)
        chunks = [(val >> (64 * k)) & 0xFFFFFFFFFFFFFFFF for k in range(4)]
        hex_chunks = [f"0x{c:016x}" for c in chunks]
        
        print(f"{{{hex_chunks[0]}, {hex_chunks[1]}, {hex_chunks[2]}, {hex_chunks[3]}}}", end="")
        if j < 3: print(", ", end="")
    print("},")
print("};")

