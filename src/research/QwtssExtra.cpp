#include "QwtssExtra.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <algorithm>
#include <map>
#include <cmath>
#include <iomanip>

struct DifferentiationMetrics {
    int actual_defects;
    int block8_max;
    int sliding3_max;
    int line_max;
    int frame4;
    int frame6;
    int frame8;
    int core32;
    int core48;
    int dilated_gcc;
};

// Helper to map CSV headers to column indices safely
int get_col(const std::map<std::string, int>& header_map, const std::string& name) {
    auto it = header_map.find(name);
    if (it != header_map.end()) return it->second;
    return -1;
}

// Robust CSV loader
std::vector<DifferentiationMetrics> load_csv(const std::string& filename) {
    std::vector<DifferentiationMetrics> data;
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Could not open " << filename << "\n";
        return data;
    }

    std::string line;
    if (!std::getline(file, line)) return data;

    if (!line.empty() && line.back() == '\r') line.pop_back();

    std::map<std::string, int> headers;
    std::stringstream ss(line);
    std::string col;
    int idx = 0;
    while (std::getline(ss, col, ',')) {
        headers[col] = idx++;
    }

    int idx_defects = get_col(headers, "Actual_Defects");
    int idx_b8 = get_col(headers, "Block8_Max");
    int idx_s3 = get_col(headers, "Sliding3_Max");
    int idx_line = get_col(headers, "Line_Max");
    int idx_f4 = get_col(headers, "Frame4");
    int idx_f6 = get_col(headers, "Frame6");
    int idx_f8 = get_col(headers, "Frame8");
    int idx_c32 = get_col(headers, "Core32");
    int idx_c48 = get_col(headers, "Core48");
    int idx_d_gcc = get_col(headers, "Dilated_Defective_Max_GCC");

    if (idx_defects == -1 || idx_b8 == -1 || idx_s3 == -1 || idx_line == -1 || idx_f4 == -1 || idx_f6 == -1 || idx_f8 == -1 
        || idx_c32 == -1 || idx_c48 == -1 || idx_d_gcc == -1) {
        std::cerr << "Missing required columns in " << filename << "!\n";
        return data;
    }

    while (std::getline(file, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        std::stringstream ss_row(line);
        std::string val;
        std::vector<std::string> row;
        while (std::getline(ss_row, val, ',')) {
            row.push_back(val);
        }

        if (row.size() <= (size_t)std::max({idx_defects, idx_b8, idx_s3, idx_line, idx_f4, idx_f6, idx_f8, idx_c32, idx_c48, idx_d_gcc})) continue;

        DifferentiationMetrics m;
        m.actual_defects = std::stoi(row[idx_defects]);
        m.block8_max = std::stoi(row[idx_b8]);
        m.sliding3_max = std::stoi(row[idx_s3]);
        m.line_max = std::stoi(row[idx_line]);
        m.frame4 = std::stoi(row[idx_f4]);
        m.frame6 = std::stoi(row[idx_f6]);
        m.frame8 = std::stoi(row[idx_f8]);
        m.core32 = std::stoi(row[idx_c32]);
        m.core48 = std::stoi(row[idx_c48]);
        m.dilated_gcc = std::stoi(row[idx_d_gcc]);
        data.push_back(m);
    }
    return data;
}

// Helper to calculate percentiles
int get_percentile(std::vector<int> vals, double p) {
    std::sort(vals.begin(), vals.end());
    int idx = std::round((vals.size() - 1) * (p / 100.0));
    return vals[idx];
}

int differentiate_attacker_vs_real() {
    std::cout << "Loading data...\n";

    int max_real_defects = 140;
    auto real_grids_all = load_csv("../build2/spatial_distribution_metrics - 140 to 150 defects (70% boundary).csv");
    auto attack_grids_all = load_csv("attacker_custom_topology_stats (340 defects and under).csv");

    //std::vector<double> percentiles = {90.0, 95.0, 98.0, 99.0, 100.0};
    std::vector<double> percentiles = {90.0, 95.0, 100.0};

    // Filter attacks to the viable threat band (<= max_real_defects defects)
    std::vector<DifferentiationMetrics> attack_threats;
    for (const auto& a : attack_grids_all) {
        if (a.actual_defects <= max_real_defects) {
            attack_threats.push_back(a);
        }
    }

    std::cout << "Real Grids Loaded: " << real_grids_all.size() << "\n";
    std::cout << "Viable Attack Grids (<= " << max_real_defects << " defects): " << attack_threats.size() << "\n\n";

    if (real_grids_all.empty() || attack_threats.empty()) return 1;

    // Extract individual feature arrays for percentile calculation
    std::vector<int> r_b8, r_s3, r_line, r_f4, r_f6, r_f8, r_c32, r_c48, r_d_gcc;
    for (const auto& r : real_grids_all) {
        r_b8.push_back(r.block8_max);
        r_s3.push_back(r.sliding3_max);
        r_line.push_back(r.line_max);
        r_f4.push_back(r.frame4);
        r_f6.push_back(r.frame6);
        r_f8.push_back(r.frame8);
        r_c32.push_back(r.core32);
        r_c48.push_back(r.core48);
        r_d_gcc.push_back(r.dilated_gcc);
    }

    double best_pass_rate = -1.0;
    std::vector<int> best_thresholds(9, 0);
    std::vector<double> best_percents(9, 0.0);
    int max_attacks_caught_overall = 0;

    std::cout << "Evaluating " << std::pow(percentiles.size(), 9) << " threshold combinations...\n";

    for (double p_b8 : percentiles) {
        int t_b8 = get_percentile(r_b8, p_b8);
        for (double p_s3 : percentiles) {
            int t_s3 = get_percentile(r_s3, p_s3);
            for (double p_line : percentiles) {
                int t_line = get_percentile(r_line, p_line);
                for (double p_f4 : percentiles) {
                    int t_f4 = get_percentile(r_f4, p_f4);
                    for (double p_f6 : percentiles) {
                        int t_f6 = get_percentile(r_f6, p_f6);
                        for (double p_f8 : percentiles) {
                            int t_f8 = get_percentile(r_f8, p_f8);
                            for (double p_c32 : percentiles) {
                                int t_c32 = get_percentile(r_c32, p_c32);
                                for (double p_c48 : percentiles) {
                                    int t_c48 = get_percentile(r_c48, p_c48);
                                    for (double p_d_gcc : percentiles) {
                                        int t_d_gcc = get_percentile(r_d_gcc, p_d_gcc);

                                        // Count how many attacks are caught (exceed ANY threshold)
                                        int attacks_caught = 0;
                                        for (const auto& a : attack_threats) {
                                            if (a.block8_max > t_b8 || 
                                                a.sliding3_max > t_s3 || 
                                                a.line_max > t_line || 
                                                a.frame4 > t_f4 || 
                                                a.frame6 > t_f6 || 
                                                a.frame8 > t_f8 ||
                                                a.core32 > t_c32 ||
                                                a.core48 > t_c48 ||
                                                a.dilated_gcc > t_d_gcc) {
                                                attacks_caught++;
                                            }
                                        }
                                        
                                        if (attacks_caught > max_attacks_caught_overall) {
                                            max_attacks_caught_overall = attacks_caught;
                                        }

                                        // If we caught 100% of attacks, check the honest grid pass rate
                                        if (attacks_caught == attack_threats.size()) {
                                            std::cout << "+" << std::flush;

                                            int real_passed = 0;
                                            for (const auto& r : real_grids_all) {
                                                if (r.block8_max <= t_b8 && 
                                                    r.sliding3_max <= t_s3 && 
                                                    r.line_max <= t_line && 
                                                    r.frame4 <= t_f4 && 
                                                    r.frame6 <= t_f6 && 
                                                    r.frame8 <= t_f8 &&
                                                    r.core32 <= t_c32 &&
                                                    r.core48 <= t_c48 &&
                                                    r.dilated_gcc <= t_d_gcc) {
                                                    real_passed++;
                                                }
                                            }

                                            double pass_rate = (double)real_passed / real_grids_all.size();
                                            // Only update if this new combination is LESS INVASIVE 
                                            // (allows more real grids to survive)
                                            if (pass_rate > best_pass_rate) {
                                                best_pass_rate = pass_rate;
                                                best_thresholds = {t_b8, t_s3, t_line, t_f4, t_f6, t_f8, t_c32, t_c48, t_d_gcc};
                                                best_percents = {p_b8, p_s3, p_line, p_f4, p_f6, p_f8, p_c32, p_c48, p_d_gcc};
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    std::cout << "\n\n--------------------------------------------------\n";
    if (best_pass_rate >= 0.0) {
        std::cout << "SUCCESS! Found a perfect separator.\n";
        std::cout << "Attacks Caught: 100% (" << attack_threats.size() << " / " << attack_threats.size() << ")\n";
        std::cout << "Real Grid Pass Rate: " << std::fixed << std::setprecision(2) << (best_pass_rate * 100.0) << "%\n\n";
        
        std::cout << "Optimal STARK Thresholds:\n";
        std::cout << "  Block8_Max:   <= " << best_thresholds[0] << "  (Real " << best_percents[0] << "th Percentile)\n";
        std::cout << "  Sliding3_Max: <= " << best_thresholds[1] << "  (Real " << best_percents[1] << "th Percentile)\n";
        std::cout << "  Line_Max:     <= " << best_thresholds[2] << "  (Real " << best_percents[2] << "th Percentile)\n";
        std::cout << "  Frame4:       <= " << best_thresholds[3] << "  (Real " << best_percents[3] << "th Percentile)\n";
        std::cout << "  Frame6:       <= " << best_thresholds[4] << "  (Real " << best_percents[4] << "th Percentile)\n";
        std::cout << "  Frame8:       <= " << best_thresholds[5] << "  (Real " << best_percents[5] << "th Percentile)\n";
        std::cout << "  Core32:       <= " << best_thresholds[6] << "  (Real " << best_percents[6] << "th Percentile)\n";
        std::cout << "  Core48:       <= " << best_thresholds[7] << "  (Real " << best_percents[7] << "th Percentile)\n";
        std::cout << "  Dilated_GCC:  <= " << best_thresholds[8] << "  (Real " << best_percents[8] << "th Percentile)\n";
    } else {
        std::cout << "FAILED to find a perfect 100% separator at this defect band.\n";
        std::cout << "Best attack catch rate achieved: " << max_attacks_caught_overall 
                  << " / " << attack_threats.size() << " (" 
                  << std::fixed << std::setprecision(2) 
                  << ((double)max_attacks_caught_overall / attack_threats.size() * 100.0) << "%)\n";
        std::cout << "\nRecommendation: The target defect band is too hot.\n";
    }

    return 0;
}
