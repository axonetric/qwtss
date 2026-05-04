from slabbe.arXiv_1903_06137 import jeandel_rao_wang_shift_partition
from sage.all import n, sqrt
import math

def sort_vertices_ccw(vertices):
    """Sorts vertices in counter-clockwise order around their centroid."""
    if len(vertices) <= 3:
        return vertices # Triangles don't need sorting
        
    # Calculate centroid
    cx = sum(v[0] for v in vertices) / len(vertices)
    cy = sum(v[1] for v in vertices) / len(vertices)
    
    # Sort by angle (atan2) from centroid
    # Cast to standard float just for the angle calculation, the original high-precision vertices are preserved.
    return sorted(vertices, key=lambda v: math.atan2(float(v[1] - cy), float(v[0] - cx)))

# 1. Load Labbé's exact Markov Partition for the 11 tiles
P0 = jeandel_rao_wang_shift_partition()

# Use 60 digits to ensure the float256 constructor gets the full entropy
print(f"const float256 GOLDEN_RATIO = float256(\"{n((1 + sqrt(5))/2, digits=60)}\");")

# Extract Torus dimensions with high precision
width = max(v[0] for v in P0.domain().vertices())
height = P0.domain().vertices()[2][1]

# 2. Extract the Torus Lattice dimensions
# Safely extracts the maximum X-coordinate from the bounding box with 60-digit precision
print(f'const float256 TORUS_WIDTH("{n(max(v[0] for v in P0.domain().vertices()), digits=60)}");')
print(f'const float256 TORUS_HEIGHT("{n(P0.domain().vertices()[2][1], digits=60)}");')

# 3. Dump the 11 Polygons into C++ Structs
print("struct LabbePolygon {")
print("    int tile_id;")
print("    std::vector<Point256> vertices;")
print("};")
print("const std::vector<LabbePolygon> LABBE_POLYGONS = {")

for item in P0:
    tile_index = item[0]
    poly = item[1]
    
    raw_vertices = list(poly.vertices())
    sorted_vertices = sort_vertices_ccw(raw_vertices)
    
    vec_str = ", ".join([f'{{float256("{n(v[0], digits=60)}"), float256("{n(v[1], digits=60)}") }}' for v in sorted_vertices])
    
    # Output the tile_index alongside the geometry
    print(f"    {{{tile_index}, {{{vec_str}}}}},")
    
print("};")

