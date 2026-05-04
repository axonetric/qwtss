from sage.all import *
from slabbe.arXiv_2403_03197 import partition, metallic_mean_wang_tile_set
import json

print("--- Loading Labbé's 2024 Golden Mean (Ammann-16) Partitions ---")

# 1. Load the alphabet (n=1 is the Golden Mean / Ammann-16 set)
T1 = metallic_mean_wang_tile_set(1)
T1_tiles = T1.tiles()

# 2. Load the partitions
P0_obj = partition(1)
P0 = [(t_id, poly) for t_id, poly in P0_obj]

print(f"[SUCCESS] Loaded {len(P0)} Markov partition polygons.")

# Ammann-16 is the Golden Ratio (n=1)
R = RealField(200) 
golden_ratio = R(1 + sqrt(5)) / R(2)

print("Generating 1024x1024 Aperiodic Grid...")
size = 1024

def reduce_to_domain(px, py):
    u = px - floor(px)
    v = py - floor(py)
    return u, v

discovery_map = {}
next_id = 0

# Transcendental offsets to completely avoid boundary collisions
start_x = R(pi)
start_y = R(e)

with open("OracleGrid.h", "w") as f:
    f.write("#pragma once\nconst int A16_ORACLE_SIZE = 1024;\nconst int A16_ORACLE_GROUND_TRUTH[] = {\n")
    
    for r in range(size):
        for c in range(size):
            # Cross-Dimensional Translation
            # Physical Vertical (r) -> Internal Horizontal (x)
            # Physical Horizontal (c) -> Internal Vertical (y)
            px = start_x + R(r) * golden_ratio
            py = start_y + R(c) * golden_ratio
            
            u, v = reduce_to_domain(px, py)
            point = vector([u, v])
            
            found_tile_id = None
            for t_id, poly in P0:
                if poly.contains(point):
                    found_tile_id = t_id
                    break
            
            if found_tile_id is None:
                # Boundary nudge (should be extremely rare with pi/e offsets)
                u_n, v_n = reduce_to_domain(px + R("1e-15"), py + R("1e-15"))
                point_n = vector([u_n, v_n])
                for t_id, poly in P0:
                    if poly.contains(point_n):
                        found_tile_id = t_id
                        break

            if found_tile_id is None:
                 raise ValueError(f"Topological fault at {c}, {r}")

            tile_tuple = tuple(T1_tiles[found_tile_id])
            
            if tile_tuple not in discovery_map:
                discovery_map[tile_tuple] = next_id
                print(f"Discovered Tile {next_id}: {tile_tuple}")
                next_id += 1
            
            f.write(f"{discovery_map[tile_tuple]},")
        f.write("\n")
    f.write("};\n")

json_map = {str(k): v for k, v in discovery_map.items()}
with open("discovery_map.json", "w") as jf:
    json.dump(json_map, jf, indent=4)

print(f"\n[SUCCESS] Exported grid with {len(discovery_map)} unique tiles.")
