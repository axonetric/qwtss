#pragma once
#include <vector>
#include <string>
#include <iostream>
#include <stdexcept>
#include <cmath>
#include <map>
#include <cstdint>
#include <algorithm>

struct Tile {
    int top;
    int right;
    int bottom;
    int left;
};

class ITileSet {
public:
    virtual ~ITileSet() = default;
    virtual std::vector<Tile> get_tiles() const = 0;
    virtual int get_size() const = 0;
    virtual int get_color_count() const = 0;

protected:
    virtual void throw_if_invalid(){
        std::vector<Tile> alphabet = get_tiles();
        for (size_t i = 0; i < alphabet.size(); i++) {
            Tile t = alphabet[i];
            if (t.top == t.bottom && t.left == t.right) {
                throw std::runtime_error("FATAL: Monochromatic periodic tile detected at ID " + 
                                        std::to_string(i) + ". This set is degenerate!");
            }
        }
    }
};

class JeandelRaoTileSet : public ITileSet {
private:
    std::vector<Tile> tiles;
public:
    JeandelRaoTileSet() {

        /* 180-degree rotation applied universally:
        tiles = {
            {4, 2, 1, 2}, {2, 2, 0, 2}, {1, 1, 1, 3}, {2, 1, 2, 3},
            {1, 3, 3, 3}, {1, 0, 1, 3}, {0, 0, 1, 0}, {1, 3, 2, 0},
            {2, 0, 2, 1}, {2, 1, 4, 1}, {3, 3, 2, 1}
        };
        */

        // The exact canonical published Jeandel-Rao 11-tile set:
        tiles = {
            {1, 2, 4, 2}, {0, 2, 2, 2}, {1, 3, 1, 1}, {2, 3, 2, 1},
            {3, 3, 1, 3}, {1, 3, 1, 0}, {1, 0, 0, 0}, {2, 0, 1, 3},
            {2, 1, 2, 0}, {4, 1, 2, 1}, {2, 1, 3, 3}
        };

        // Safety check
        throw_if_invalid();
    }
    
    std::vector<Tile> get_tiles() const override { return tiles; }
    int get_size() const override { return 11; }
    int get_color_count() const override { return 5; }
};

class Ammann16TileSet : public ITileSet {
private:
    std::vector<Tile> tiles;
public:
    Ammann16TileSet() {
        // Based on the 6-color parity-baked Ammann A2 mapping.
        // Color Map: 0=Red, 1=Yellow, 2=Green, 3=Cyan, 4=Blue, 5=Purple
        // Format: {top, right, bottom, left}
        tiles = {
            {0, 1, 1, 0}, // Tile 1
            {2, 3, 3, 2}, // Tile 2
            {3, 4, 4, 3}, // Tile 3
            {5, 2, 2, 5}, // Tile 4
            {3, 4, 3, 2}, // Tile 5
            {5, 2, 3, 2}, // Tile 6
            {2, 3, 4, 3}, // Tile 7
            {2, 3, 2, 5}, // Tile 8
            {4, 0, 2, 1}, // Tile 9
            {3, 0, 5, 1}, // Tile 10
            {4, 0, 3, 0}, // Tile 11
            {2, 1, 5, 1}, // Tile 12
            {1, 5, 0, 3}, // Tile 13
            {1, 2, 0, 4}, // Tile 14
            {1, 5, 1, 2}, // Tile 15
            {0, 3, 0, 4}  // Tile 16
        };

        // Safety check
        throw_if_invalid();
    }
    
    std::vector<Tile> get_tiles() const override { return tiles; }
    int get_size() const override { return 16; }
    int get_color_count() const override { return 6; }
};

// Generalized defect counter for any square grid size
inline int count_grid_defects(const int* grid, int size, const std::vector<Tile>& alphabet) {
    int defects = 0;
    for (int r = 0; r < size; r++) {
        for (int c = 0; c < size; c++) {
            Tile current = alphabet[grid[r * size + c]];
            if (r > 0) { // Check North
                if (current.top != alphabet[grid[(r - 1) * size + c]].bottom) defects++;
            }
            if (c > 0) { // Check West
                if (current.left != alphabet[grid[r * size + (c - 1)]].right) defects++;
            }
        }
    }
    return defects;
}

// Returns 0.0 to 100.0, representing the % of border colors kept (pinned).
inline double get_border_kept_pct(const std::vector<uint8_t>& boundary_mask, int grid_size){
    if (boundary_mask.empty()) return 0.0;

    // As a convenience, here we track the percentage of the border *tiles* that are constrained (not edge *colors*),
    // but the two values are very close.
    // Count only mask = 1 (Pinned Outward Colors)
    double border_pct = 100.0 * std::count_if(boundary_mask.begin(), boundary_mask.end(), 
        [](uint8_t m) { return m == 1; }) / (4.0 * grid_size - 4.0);
    
    return border_pct;
}

// Returns the total number of strictly pinned interior Rebar tiles.
inline int get_pinned_tile_count(const std::vector<uint8_t>& boundary_mask) {
    if (boundary_mask.empty()) return 0;

    // Strictly count only mask = 2 (Pinned Tile IDs)
    return std::count(boundary_mask.begin(), boundary_mask.end(), 2);
}

// Fast FNV-1a hash for detecting unique grid states.
// It is strictly position-dependent, so swapping any two tiles completely changes the hash.
inline uint64_t hash_grid_state(const int* grid, int size) {
    uint64_t hash = 14695981039346656037ULL; // 64-bit FNV offset basis
    const uint64_t fnv_prime = 1099511628211ULL; // 64-bit FNV prime
    
    int total_grid_size = size * size;
    
    for (int i = 0; i < total_grid_size; ++i) {
        // XOR the bottom bits of the hash with the current tile ID
        hash ^= static_cast<uint64_t>(grid[i]);
        // Multiply by the prime to avalanche the bits and enforce position-dependence
        hash *= fnv_prime;
    }
    
    return hash;
}

// Fast FNV-1a hash for detecting unique grid states.
// It is strictly position-dependent, so swapping any two tiles completely changes the hash.
inline uint64_t hash_grid_state(const uint8_t* grid, int size) {
    uint64_t hash = 14695981039346656037ULL; // 64-bit FNV offset basis
    const uint64_t fnv_prime = 1099511628211ULL; // 64-bit FNV prime
    
    int total_grid_size = size * size;
    
    for (int i = 0; i < total_grid_size; ++i) {
        // XOR the bottom byte of the hash with the current tile ID
        hash ^= static_cast<uint64_t>(grid[i]);
        // Multiply by the prime to avalanche the bits and enforce position-dependence
        hash *= fnv_prime;
    }
    
    return hash;
}
