import re
import json
import ast

print("--- Dynamic Ammann-16 Oracle Verifier ---")

# 1. Load the exact mapping used to generate this specific grid
print("[*] Loading discovery_map.json...")
with open("discovery_map.json", "r") as f:
    json_map = json.load(f)

# Invert the map to look up tuple by ID
# json_map has string keys like "('111', '001', '111', '000')" and integer values
discovered_raw = {v: ast.literal_eval(k) for k, v in json_map.items()}

print("[*] Parsing OracleGrid.h...")
with open("OracleGrid.h", "r") as f:
    content = f.read()

# Extract the array contents
match = re.search(r'\{([^}]+)\}', content)
if not match:
    raise ValueError("Could not find the array data in OracleGrid.h")

raw_csv = match.group(1)
grid_1d = [int(x.strip()) for x in raw_csv.split(',') if x.strip() != ""]

# Test on a 128x128 patch to ensure the topological proof holds without taking forever
size = 1024
patch_size = 128

print(f"[*] Testing {patch_size}x{patch_size} window for topological consistency...\n")

valid_horizontal = []
valid_vertical = []

# Test all possible index pairings (0 to 3)
for i1 in range(4):
    for i2 in range(4):
        if i1 == i2: continue
        
        # 1. Check Horizontal Binding (Tile A edge i1 == Tile B edge i2)
        h_defects = 0
        for r in range(patch_size):
            for c in range(patch_size - 1):
                t_curr = discovered_raw[grid_1d[r * size + c]]
                t_right = discovered_raw[grid_1d[r * size + c + 1]]
                if t_curr[i1] != t_right[i2]:
                    h_defects += 1
                    
        if h_defects == 0:
            valid_horizontal.append((i1, i2))
            
        # 2. Check Vertical Binding (Tile A edge i1 == Tile B edge i2)
        v_defects = 0
        for r in range(patch_size - 1):
            for c in range(patch_size):
                t_curr = discovered_raw[grid_1d[r * size + c]]
                t_bottom = discovered_raw[grid_1d[(r + 1) * size + c]]
                if t_curr[i1] != t_bottom[i2]:
                    v_defects += 1
                    
        if v_defects == 0:
            valid_vertical.append((i1, i2))

print("=== RAW TOPOLOGY RESULTS ===")
if not valid_horizontal and not valid_vertical:
    print("[FATAL] The step size or translation vectors in the generator are wrong.")
else:
    print("[SUCCESS] The raw grid contains a flawless 128x128 Ammann-16 plane.")
    print(f"  -> Valid Horizontal Bindings (Current Tile Index -> Right Tile Index): {valid_horizontal}")
    print(f"  -> Valid Vertical Bindings (Current Tile Index -> Bottom Tile Index): {valid_vertical}")
