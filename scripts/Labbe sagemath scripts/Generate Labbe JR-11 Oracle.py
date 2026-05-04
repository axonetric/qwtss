from sage.all import *
from slabbe.arXiv_1903_06137 import jeandel_rao_wang_shift_partition

print("Loading Labbé Markov Partitions...")
P0 = jeandel_rao_wang_shift_partition()

# Use 200-bit precision to eliminate irrational boundary leaks
R = RealField(200) 
phi = R((1 + sqrt(5))/2)

# Lattice Translation Vectors
v1_x = phi
v1_y = R(0)
v2_x = R(1)
v2_y = phi + R(3)

def reduce_mod_gamma(x, y):
    c2 = floor(y / v2_y)
    v = y - c2 * v2_y
    
    x_shifted = x - c2 * v2_x
    c1 = floor(x_shifted / v1_x)
    u = x_shifted - c1 * v1_x
    
    return u, v
    
print("Generating 1024x1024 Aperiodic Grid (This takes about 60 seconds)...")
size = 1024

# The permutation map deduced earlier baked into the exporter
labbe_to_jr = {0:10, 1:1, 2:6, 3:7, 4:9, 5:5, 6:0, 7:2, 8:8, 9:3, 10:4}

start_x = R("0.12345")
start_y = R("0.67890")

polygons = [(tile_id, poly) for tile_id, poly in P0]

with open("OracleGrid.h", "w") as f:
    f.write("#pragma once\n")
    f.write(f"const int ORACLE_SIZE = {size};\n")
    f.write("const int ORACLE_GROUND_TRUTH[] = {\n")
    
    for r in range(size):
        for c in range(size):
            px = start_x + R(c)
            py = start_y + R(r)
            
            u, v = reduce_mod_gamma(px, py)
            point = vector(R, [u, v])
            
            mapped = -1
            # Sage natively handles checking boundaries with exact mathematics
            for tile_id, poly in polygons:
                if poly.contains(point):
                    mapped = tile_id
                    break
                    
            final_id = labbe_to_jr[mapped]
            f.write(f"{final_id}, ")
            
        f.write("\n")
        if (r + 1) % 64 == 0:
            print(f"  Row {r + 1}/{size} complete...")
            
    f.write("};\n")

print("Success! Saved mathematically perfect array to OracleGrid.h")
