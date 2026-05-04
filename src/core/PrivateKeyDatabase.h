#pragma once
#include "QwtssCore.h"
#include <vector>
#include <map>
#include <string>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <cmath>
#include <algorithm>


// Represents a single saved private key grid
struct PrivateKey {
    int defect_count;
    // raw_grid length is expected to be (grid_size * grid_size)
    int grid_size;
    std::vector<int> grid_data;
    // Allowed behavior mask: 0 = Free, 1 = Pinned Outward Colors, 2 = Pinned Tile ID. If empty, defaults to all tiles Free.
    std::vector<uint8_t> boundary_mask;

    // The oracle grid A used in the PK splicing. Used for "alien tile" calcs. Only serialized if explicitly requested.
    std::vector<int> plane_A;
    // The oracle grid B used in the PK splicing. Used for "alien tile" calcs. Only serialized if explicitly requested.
    std::vector<int> plane_B;
    // Only serialized if explicitly requested.
    std::string name;


    QwtssPrivateKey to_QwtssPrivateKey() const {
        return QwtssPrivateKey{grid_data, boundary_mask};
    }

    // Export to CSV for Python Visualization
    void export_to_csv(const std::string& filename) const {
        std::ofstream out(filename);
        if (!out) {
            throw std::runtime_error("Failed to open file for CSV export: " + filename);
        }

        for (int r = 0; r < grid_size; ++r) {
            for (int c = 0; c < grid_size; ++c) {
                out << grid_data[r * grid_size + c];
                // Add a comma after every element except the last one in the row
                if (c < grid_size - 1) {
                    out << ",";
                }
            }
            out << "\n"; // Newline at the end of each row
        }
        out.close();
        std::cout << "Successfully exported " << grid_size << "x" << grid_size 
                  << " grid (Defects: " << defect_count << ") to " << filename << "\n";
    }
};

class PrivateKeyDatabase {
private:
    // Automatically sorts keys by defect count. 
    // The vector holds all discovered keys that share that exact defect count.
    std::map<int, std::vector<PrivateKey>> db;

public:
    // --- CUSTOM FLATTENED ITERATOR ---
    class Iterator {
    private:
        using MapIt = std::map<int, std::vector<PrivateKey>>::const_iterator;
        using VecIt = std::vector<PrivateKey>::const_iterator;

        MapIt map_it;
        MapIt map_end;
        VecIt vec_it;

        // Automatically skips over empty vectors (if any exist) and advances to the next valid key
        void advance_to_valid() {
            while (map_it != map_end && vec_it == map_it->second.end()) {
                ++map_it;
                if (map_it != map_end) {
                    vec_it = map_it->second.begin();
                }
            }
        }

    public:
        Iterator(MapIt mit, MapIt mend, VecIt vit) : map_it(mit), map_end(mend), vec_it(vit) {
            advance_to_valid();
        }

        // Dereference operator returns the actual PrivateKey
        const PrivateKey& operator*() const { return *vec_it; }
        const PrivateKey* operator->() const { return &(*vec_it); }

        // Prefix increment (++it)
        Iterator& operator++() {
            ++vec_it;
            advance_to_valid();
            return *this;
        }

        // Inequality check for the loop condition
        bool operator!=(const Iterator& other) const {
            if (map_it != other.map_it) return true;
            if (map_it == map_end) return false; // Both reached the end
            return vec_it != other.vec_it;
        }
    };

    // Standard begin() and end() hooks for C++ range-based for loops
    Iterator begin() const {
        if (db.empty()) return end();
        return Iterator(db.begin(), db.end(), db.begin()->second.begin());
    }

    Iterator end() const {
        // Pass a default-constructed const_iterator for the vector
        return Iterator(db.end(), db.end(), std::vector<PrivateKey>::const_iterator());
    }
    // --------------------------------

    // Global Indexer for O(K) random access across the flattened database
    const PrivateKey& operator[](size_t global_index) const {
        for (const auto& pair : db) {
            if (global_index < pair.second.size()) {
                return pair.second[global_index];
            }
            global_index -= pair.second.size();
        }
        throw std::out_of_range("Global index out of bounds.");
    }

    // Add a new key to the database in memory
    // @param grid_size raw_grid length is expected to be (grid_size * grid_size)
    void add_key(int defect_count, int grid_size, const std::vector<int>& raw_grid, const std::vector<uint8_t>& boundary_mask) {
        int grid_cells = grid_size * grid_size;
        if (raw_grid.size() != grid_cells) throw std::invalid_argument("(raw_grid.size() != grid_cells)");
        if (boundary_mask.size() != grid_cells) throw std::invalid_argument("(boundary_mask.size() != grid_cells)");

        PrivateKey key;
        key.defect_count = defect_count;
        key.grid_size = grid_size;

        key.grid_data = raw_grid;
        key.boundary_mask = boundary_mask;

        db[defect_count].push_back(key);
    }

    // Add a new key to the database in memory
    // @param grid_size raw_grid length is expected to be (grid_size * grid_size)
    void add_key(const PrivateKey& key) {
        if (key.defect_count <= 0) throw std::invalid_argument("(key.defect_count <= 0)");
        if (key.grid_size <= 0) throw std::invalid_argument("(key.grid_size <= 0)");
        int grid_cells = key.grid_size * key.grid_size;
        if (key.grid_data.size() != grid_cells) throw std::invalid_argument("(key.grid_data.size() != grid_cells)");
        if (key.boundary_mask.size() != grid_cells) throw std::invalid_argument("(key.boundary_mask.size() != grid_cells)");

        db[key.defect_count].push_back(key);
    }

    // Retrieve a specific key using [defect_count, idx]
    const PrivateKey& get_key(int defect_count, size_t idx) const {
        auto it = db.find(defect_count);
        if (it == db.end()) {
            throw std::out_of_range("No keys found for this defect count.");
        }
        if (idx >= it->second.size()) {
            throw std::out_of_range("Index out of bounds. Only " + 
                                    std::to_string(it->second.size()) + " keys exist for this defect count.");
        }
        return it->second[idx];
    }

    // Finds the N-th nearest key by defect count within a +/- 10% tolerance band.
    // If rank=1, returns the absolute closest match (identical to legacy behavior).
    PrivateKey get_nearest_key_by_defect_count(int target_defect_count, int rank = 1) const {
        if (this->get_total_key_count() < 1) {
            throw std::runtime_error("Cannot fetch nearest key: PrivateKeyDatabase is empty.");
        }
        if (rank < 1) {
            throw std::invalid_argument("Rank must be >= 1.");
        }

        // ------------------------------------------------------------------
        // 1. EXACT LEGACY BEHAVIOR FOR RANK 1
        // Ensures 100% backward compatibility with existing codebase
        // ------------------------------------------------------------------
        int min_diff = INT32_MAX;
        int nearest_defect_count = -1;

        for (const auto& key : *this) {
            int diff = std::abs(key.defect_count - target_defect_count);
            if (diff < min_diff) {
                min_diff = diff;
                nearest_defect_count = key.defect_count;
            }
            // Optimization for rank 1: absolute perfect match
            if (diff == 0 && rank == 1) {
                break; 
            }
        }

        if (rank == 1) {
            // Return the idx=0 key for the nearest defect count we found
            return this->get_key(nearest_defect_count, 0);
        }

        // ------------------------------------------------------------------
        // 2. DIVERSITY SEARCH FOR RANK > 1
        // ------------------------------------------------------------------
        
        // Define the +/- 10% tolerance band (minimum 1 defect to avoid 0-width bands on tiny targets)
        int tolerance = std::max(1, static_cast<int>(target_defect_count * 0.10f));
        int lower_bound = target_defect_count - tolerance;
        int upper_bound = target_defect_count + tolerance;

        std::vector<PrivateKey> candidates;

        // Collect all candidates within the tolerance band
        for (const auto& key : *this) {
            if (key.defect_count >= lower_bound && key.defect_count <= upper_bound) {
                candidates.push_back(key);
            }
        }

        // If the 10% tolerance band is somehow completely empty, fallback to the absolute nearest
        if (candidates.empty()) {
            return this->get_key(nearest_defect_count, 0); 
        }

        // Sort candidates by absolute distance to the target.
        // std::stable_sort preserves the original database insertion order for keys with identical distances.
        // std::stable_sort is crucial to have deterministic return results for a given 'rank' argument
        std::stable_sort(candidates.begin(), candidates.end(), 
            [target_defect_count](const PrivateKey& a, const PrivateKey& b) {
                int diff_a = std::abs(a.defect_count - target_defect_count);
                int diff_b = std::abs(b.defect_count - target_defect_count);
                return diff_a < diff_b; // Ascending order
            });

        // Clamp the rank if the requested target exceeds the available candidate pool
        // (e.g., requested the 10th nearest, but only 4 exist in the tolerance band)
        int selected_idx = std::min(rank - 1, static_cast<int>(candidates.size() - 1));

        return candidates[selected_idx];
    }

    // Returns all keys whose name field contains the search string (case-insensitive)
    std::vector<PrivateKey> filter_by_name(const std::string& search_string) const {
        std::vector<PrivateKey> results;

        // Returning none is usually the safer default for a search function.
        if (search_string.empty()) return results;

        // 1. Convert the search query to lowercase once
        std::string query_lower = search_string;
        std::transform(query_lower.begin(), query_lower.end(), query_lower.begin(),
            [](unsigned char c) { return std::tolower(c); });

        // 2. Iterate through all keys using the custom flattened iterator
        for (const auto& key : *this) {
            if (key.name.empty()) continue;

            // Convert the current key's name to lowercase
            std::string name_lower = key.name;
            std::transform(name_lower.begin(), name_lower.end(), name_lower.begin(),
                [](unsigned char c) { return std::tolower(c); });

            // 3. Substring match
            if (name_lower.find(query_lower) != std::string::npos) {
                results.push_back(key);
            }
        }

        return results;
    }

    // Returns the total number of private keys stored across all defect counts
    size_t get_total_key_count() const {
        size_t total = 0;
        for (const auto& pair : db) {
            total += pair.second.size();
        }
        return total;
    }

    // Check how many keys exist for a specific defect count
    size_t get_key_count(int defect_count) const {
        auto it = db.find(defect_count);
        return (it != db.end()) ? it->second.size() : 0;
    }

    // Save the entire database to a fast binary file
    void save_to_disk(const std::string& filename, bool save_extended_data = false) const {
        std::ofstream out(filename, std::ios::binary | std::ios::trunc);
        if (!out) throw std::runtime_error("Cannot open database file for writing.");

        // Calculate total keys
        size_t total_keys = 0;
        for (const auto& pair : db) {
            total_keys += pair.second.size();
        }

        // Write version magic
        size_t version_magic = (1ULL << 20) + 1;
        out.write(reinterpret_cast<const char*>(&version_magic), sizeof(version_magic));

        // Write header (total number of keys)
        out.write(reinterpret_cast<const char*>(&total_keys), sizeof(total_keys));

        // Write payload
        for (const auto& pair : db) {
            for (const auto& key : pair.second) {
                size_t elements = key.grid_size * key.grid_size;
                if (key.grid_data.size() != elements) throw std::runtime_error("(key.grid_data.size() != elements)");

                out.write(reinterpret_cast<const char*>(&key.defect_count), sizeof(key.defect_count));
                out.write(reinterpret_cast<const char*>(&key.grid_size), sizeof(key.grid_size));
                out.write(reinterpret_cast<const char*>(key.grid_data.data()), key.grid_data.size() * sizeof(int));
                out.write(reinterpret_cast<const char*>(key.boundary_mask.data()), key.boundary_mask.size() * sizeof(uint8_t));

                if (save_extended_data){
                    if (key.plane_A.size() != elements) throw std::runtime_error("(key.plane_A.size() != elements)");
                    if (key.plane_B.size() != elements) throw std::runtime_error("(key.plane_B.size() != elements)");

                    out.write(reinterpret_cast<const char*>(key.plane_A.data()), key.plane_A.size() * sizeof(int));
                    out.write(reinterpret_cast<const char*>(key.plane_B.data()), key.plane_B.size() * sizeof(int));

                    // Write the size of the string (number of characters)
                    size_t size = key.name.size();
                    out.write(reinterpret_cast<const char*>(&size), sizeof(size_t));

                    // Write the actual string data
                    // sizeof(char) is strictly 1 in C++; multiplication omitted
                    if (size > 0) {
                        out.write(key.name.data(), size);
                    }
                }
            }
        }
    }

    // Load an existing database from disk into memory
    void load_from_disk(const std::string& filename, bool has_extended_data = false) {
        std::ifstream in(filename, std::ios::binary);
        if (!in) {
            throw std::invalid_argument("Private key database file not found: " + filename);
        }

        db.clear();

        size_t version_magic = 0;
        if (!in.read(reinterpret_cast<char*>(&version_magic), sizeof(version_magic))) return;
        if (version_magic != ((1ULL << 20) + 1)) throw std::runtime_error("The version of the loaded private key db is not supported");

        size_t total_keys = 0;
        if (!in.read(reinterpret_cast<char*>(&total_keys), sizeof(total_keys))) return;

        size_t i = 0;
        for (i = 0; i < total_keys; ++i) {
            PrivateKey key;
            if(!in.read(reinterpret_cast<char*>(&key.defect_count), sizeof(key.defect_count))) break;
            if (!in.read(reinterpret_cast<char*>(&key.grid_size), sizeof(key.grid_size))) break;

            size_t elements = key.grid_size * key.grid_size;
            key.grid_data.resize(elements);
            if (!in.read(reinterpret_cast<char*>(key.grid_data.data()), elements * sizeof(int))) break;

            key.boundary_mask.resize(elements);
            if (!in.read(reinterpret_cast<char*>(key.boundary_mask.data()), elements * sizeof(uint8_t))) break;

            if (has_extended_data){
                key.plane_A.resize(elements);
                if (!in.read(reinterpret_cast<char*>(key.plane_A.data()), elements * sizeof(int))) break;

                key.plane_B.resize(elements);
                if (!in.read(reinterpret_cast<char*>(key.plane_B.data()), elements * sizeof(int))) break;

                // Read the size of the string
                size_t size = 0;
                if (!in.read(reinterpret_cast<char*>(&size), sizeof(size_t))) break;

                // Resize the string to allocate the exact amount of memory needed
                key.name.resize(size);

                // Read the character data directly into the string buffer
                if (size > 0) {
                    if (!in.read(&key.name[0], size)) break;
                }
            }

            db[key.defect_count].push_back(key);
        }
        if (i != total_keys) throw std::runtime_error("Database corruption. Failed to load all expected private keys from the database.");
    }

    // Print terminal summary of the aggregated database
    void print_summary() const {
        std::cout << "\n--- Private Key Database Summary ---\n";
        for (const auto& pair : db) {
            std::cout << "Defects: " << pair.first << " | Total Keys: " << pair.second.size() << "\n";
        }
        std::cout << "------------------------------------\n";
    }
};
