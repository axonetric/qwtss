# STARK Prime
PRIME = 0x800000000000011000000000000000000000000000000000000000000000001

# Helper function to perfectly mimic C++ BigInt<4> Little-Endian limb loading
def from_limbs(l0, l1, l2, l3):
    return l0 | (l1 << 64) | (l2 << 128) | (l3 << 192)

# --- THE EXTRACTED VALUES ---
# Step 0 Constants (The first 4 rows of the new t=4 ROUND_CONSTANTS_EXTENDED)
C0 = from_limbs(0x7355b6f596d0d757, 0x76c8d50162462aee, 0xa49b4d367a5c4134, 0x011084a3f28820cc)
C1 = from_limbs(0x8d100faad86e401d, 0x3d154d7a7d0ef8da, 0x3e7027f2e51e48ca, 0x070696dd461a64b1)
C2 = from_limbs(0x9dd8e383890efc51, 0x8b629347c07d0c2b, 0x1d9c3f6200f64cef, 0x0527963e4e1beac7)
C3 = from_limbs(0x9646026ec5578b49, 0x4a6e89e9b004e129, 0xbf2806380babaa4e, 0x02c6fc943f327044)

# MDS Matrix Row 0
M00 = from_limbs(0x0000000000000001, 0x0000000000000000, 0xc000000000000000, 0x060000000000000c)
M01 = from_limbs(0x0000000000000001, 0x0000000000000000, 0x0000000000000000, 0x0666666666666674)
M02 = from_limbs(0xaaaaaaaaaaaaaaab, 0xaaaaaaaaaaaaaaaa, 0x2aaaaaaaaaaaaaaa, 0x0155555555555558)
M03 = from_limbs(0x0000000000000001, 0x0000000000000000, 0x0000000000000000, 0x06db6db6db6db6ea)

# MDS Matrix Row 1
M10 = from_limbs(0x0000000000000001, 0x0000000000000000, 0x0000000000000000, 0x0666666666666674)
M11 = from_limbs(0xaaaaaaaaaaaaaaab, 0xaaaaaaaaaaaaaaaa, 0x2aaaaaaaaaaaaaaa, 0x0155555555555558)
M12 = from_limbs(0x0000000000000001, 0x0000000000000000, 0x0000000000000000, 0x06db6db6db6db6ea)
M13 = from_limbs(0x0000000000000001, 0x0000000000000000, 0xe000000000000000, 0x070000000000000e)

# MDS Matrix Row 2
M20 = from_limbs(0xaaaaaaaaaaaaaaab, 0xaaaaaaaaaaaaaaaa, 0x2aaaaaaaaaaaaaaa, 0x0155555555555558)
M21 = from_limbs(0x0000000000000001, 0x0000000000000000, 0x0000000000000000, 0x06db6db6db6db6ea)
M22 = from_limbs(0x0000000000000001, 0x0000000000000000, 0xe000000000000000, 0x070000000000000e)
M23 = from_limbs(0x71c71c71c71c71c8, 0xc71c71c71c71c71c, 0x1c71c71c71c71c71, 0x0638e38e38e38e46)

# MDS Matrix Row 3
M30 = from_limbs(0x0000000000000001, 0x0000000000000000, 0x0000000000000000, 0x06db6db6db6db6ea)
M31 = from_limbs(0x0000000000000001, 0x0000000000000000, 0xe000000000000000, 0x070000000000000e)
M32 = from_limbs(0x71c71c71c71c71c8, 0xc71c71c71c71c71c, 0x1c71c71c71c71c71, 0x0638e38e38e38e46)
M33 = from_limbs(0x0000000000000001, 0x0000000000000000, 0x8000000000000000, 0x0733333333333342)

# --- SPONGE MATH ---
# 1. State + Absorption (State=0, tile_id=1, step=0)
s0 = 0
s1 = 0
s2 = 0
s3 = 0

# Absorb into the two rate elements
s0 = (s0 + 1) % PRIME # tile_id
s1 = (s1 + 0) % PRIME # step

# 2. Add Round Constants
s0 = (s0 + C0) % PRIME
s1 = (s1 + C1) % PRIME
s2 = (s2 + C2) % PRIME
s3 = (s3 + C3) % PRIME

# 3. S-Box (x^3)
s0 = pow(s0, 3, PRIME)
s1 = pow(s1, 3, PRIME)
s2 = pow(s2, 3, PRIME)
s3 = pow(s3, 3, PRIME)

# 4. MDS Multiplication
next_s0 = ((s0 * M00) + (s1 * M01) + (s2 * M02) + (s3 * M03)) % PRIME
next_s1 = ((s0 * M10) + (s1 * M11) + (s2 * M12) + (s3 * M13)) % PRIME
next_s2 = ((s0 * M20) + (s1 * M21) + (s2 * M22) + (s3 * M23)) % PRIME
next_s3 = ((s0 * M30) + (s1 * M31) + (s2 * M32) + (s3 * M33)) % PRIME

print(f"Python Output S0: {hex(next_s0)}")
print(f"Python Output S1: {hex(next_s1)}")
print(f"Python Output S2: {hex(next_s2)}")
print(f"Python Output S3: {hex(next_s3)}")

