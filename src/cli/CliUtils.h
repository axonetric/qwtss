#pragma once
#include <vector>
#include <iostream>
#include <iomanip>
#include <numeric>
#include <cmath>
#include <algorithm>
#include <stdexcept>


class MetricTracker {
private:
    std::vector<double> samples;
    bool is_sorted = false;

    void ensure_sorted() {
        if (!is_sorted) {
            std::sort(samples.begin(), samples.end());
            is_sorted = true;
        }
    }

public:
    void add(double value) {
        samples.push_back(value);
        is_sorted = false; // Invalidate sort state on new data
    }

    size_t count() const {
        return samples.size();
    }

    double mean() const {
        if (samples.empty()) return 0.0;
        double sum = std::accumulate(samples.begin(), samples.end(), 0.0);
        return sum / samples.size();
    }

    double std_dev() const {
        if (samples.size() <= 1) return 0.0;
        double m = mean();
        double variance = 0.0;
        for (double val : samples) {
            variance += (val - m) * (val - m);
        }
        // Sample standard deviation divides by (N - 1)
        return std::sqrt(variance / (samples.size() - 1));
    }

    double min() {
        if (samples.empty()) return 0.0;
        ensure_sorted();
        return samples.front();
    }

    double max() {
        if (samples.empty()) return 0.0;
        ensure_sorted();
        return samples.back();
    }

    double percentile(double p) {
        if (samples.empty()) return 0.0;
        if (p <= 0.0) return min();
        if (p >= 100.0) return max();
        
        ensure_sorted();
        size_t idx = static_cast<size_t>(std::ceil((p / 100.0) * samples.size())) - 1;
        return samples[idx];
    }
};

struct BenchmarkSuite {
    MetricTracker keygen_time_ms;
    MetricTracker sign_time_ms;
    MetricTracker verify_time_ms;
    
    MetricTracker keygen_peak_ram_mb;
    MetricTracker sign_peak_ram_mb;
    MetricTracker verify_peak_ram_mb;
    
    MetricTracker pk_json_size_kb;
    MetricTracker sk_json_size_kb;
    MetricTracker sig_raw_size_kb;
    MetricTracker sig_json_size_kb;

    void print_report(std::string device_target, std::string cpu_threads, std::string anneal_mode) {
        // Helper lambda to format and print a single row
        auto print_metric = [](const std::string& name, MetricTracker& tracker, const std::string& unit) {
            if (tracker.count() == 0) return; // Skip if metric wasn't tracked
            
            std::cout << "  " << std::left << std::setw(24) << name 
                      << ": " << std::right << std::setw(8) << std::fixed << std::setprecision(2) << tracker.mean() << " " << unit
                      << "  (±" << std::setw(6) << tracker.std_dev() << ")  |"
                      << "  99th: " << std::setw(8) << tracker.percentile(99.0) << " " << unit 
                      << "  |  Max: " << std::setw(8) << tracker.max() << " " << unit << "\n";
        };

        std::cout << "================================================================================================\n";
        std::cout << "                              QWTSS BENCHMARK REPORT (" << keygen_time_ms.count() << " iterations)\n";
        // Subtitle logic
        std::string subtitle = "Device: " + device_target;
        if (device_target == "CPU") {
            if (cpu_threads == "-1") subtitle += " (AUTO threads: 70%)";
            else subtitle += " (" + cpu_threads + " threads)";
        }
        subtitle += "   |   Mode: " + anneal_mode;

        // Center the subtitle based on the 88-character width of the '=' divider
        int padding = (88 - subtitle.length()) / 2;
        if (padding > 0) {
            std::cout << std::string(padding, ' ') << subtitle << "\n";
        } else {
            std::cout << subtitle << "\n";
        }
        std::cout << "================================================================================================\n";
        
        std::cout << "--- Execution Time (Compute) ---\n";
        print_metric("Keygen Time", keygen_time_ms, "ms");
        print_metric("Sign Time", sign_time_ms, "ms");
        print_metric("Verify Time", verify_time_ms, "ms");
        
        std::cout << "\n--- Peak Memory Footprint (RAM) ---\n";
        print_metric("Keygen RAM (Isolated)", keygen_peak_ram_mb, "MB");
        print_metric("Sign RAM (Isolated)", sign_peak_ram_mb, "MB");
        print_metric("Verify RAM (Isolated)", verify_peak_ram_mb, "MB");

        std::cout << "\n--- Data Footprint (Storage / Bandwidth) ---\n";
        print_metric("Public Key (JSON)", pk_json_size_kb, "KB");
        print_metric("Secret Key (JSON)", sk_json_size_kb, "KB");
        print_metric("Signature (Raw Binary)", sig_raw_size_kb, "KB");
        print_metric("Signature (JSON Base64)", sig_json_size_kb, "KB");
        std::cout << "================================================================================================\n";
    }
};
