#include "QwtssCore.h"
#include "QwtssConfig.h"
#include "CryptoUtils.h"
#include "StarkTrace.h"
#include "QwtssAir.h"
#include "CliUtils.h"
#include <iostream>
#include <string>
#include <vector>
#include <iomanip>
#include <string_view>
#include <filesystem>
#include <fstream>
#include <array>
#include <algorithm>
#include <chrono>
#include <stdexcept>
#include <cuda_runtime.h>
#include <sys/wait.h>
#include <sys/resource.h>
#include <unistd.h>
#include "third_party/jsoncpp/json/json.h"



namespace fs = std::filesystem;

/**
 * Sanitizes a string to be used safely as a filename.
 * Only allows alphanumeric characters and underscores.
 */
std::string sanitize_for_filename(std::string_view input) {
    std::string sanitized;
    sanitized.reserve(input.length());
    
    for (char c : input) {
        if (std::isalnum(c) || c == '_' || c == '-') {
            sanitized += c;
        } else {
            // Replace any illegal char with an underscore
            sanitized += '_';
        }
    }

    // Trim length to ensure we don't hit filesystem limits
    if (sanitized.length() > 128) {
        sanitized = sanitized.substr(0, 128);
    }

    return sanitized.empty() ? "" : sanitized;
}

/**
 * Writes the QWTSS Public Key / Private Key pair to disk safely using JsonCpp.
 * @param username The identity string.
 * @param identity_nonce The identity nonce.
 * @param version The protocol version byte.
 * @param fingerprint_limbs The 504-bit hash split into two hex strings.
 */
bool write_qwtss_keypair_to_disk(const std::string& username, uint64_t identity_nonce, uint8_t version, 
                                 const QwtssPrivateKey& private_key, bool do_logging, BenchmarkSuite* benchmark_stats = nullptr) {
    if (username.empty()) throw std::invalid_argument("username must be specified");
    std::string sanitized_username = sanitize_for_filename(username);
    if (sanitized_username.empty()) throw std::invalid_argument("username cannot contain only invalid filename chars");
    if (private_key.private_key.empty()) throw std::invalid_argument("");

    // Construct the safe filenames
    std::string pk_filename = "qwtss_" + sanitized_username + "_" + std::to_string(identity_nonce) + "_pk.json";
    std::string sk_filename = "qwtss_" + sanitized_username + "_" + std::to_string(identity_nonce) + "_sk.json";
    
    // Prevent accidental overwrites
    std::ifstream check_pk(pk_filename), check_sk(sk_filename);
    if (check_pk.good() || check_sk.good()) {
        std::cerr << "Error: " << pk_filename << " or " << sk_filename << " already exists. Aborting to prevent overwrite.\n";
        return false;
    }
    check_pk.close(); check_sk.close();

    std::array<std::string, 2> fingerprint_limbs = get_serialized_grid_fingerprint(private_key);

    // Build the Public Key JSON DOM
    Json::Value pk_root;
    pk_root["scheme"] = "QWTSS-PUBLIC";
    pk_root["version"] = version;
    pk_root["username"] = username;     // JsonCpp automatically escapes malicious characters here
    // Cast to Json::UInt64 to ensure it maps to the correct internal JSON type
    pk_root["identity_nonce"] = static_cast<Json::UInt64>(identity_nonce);

    // Add the fingerprint as a JSON array of two hex strings
    Json::Value fingerprint_array(Json::arrayValue);
    fingerprint_array.append(fingerprint_limbs[0]);
    fingerprint_array.append(fingerprint_limbs[1]);
    pk_root["grid_fingerprint"] = fingerprint_array;

    // Build the Secret Key JSON DOM
    // Just copy the entire PK root, update the scheme, and add the private grid data
    Json::Value sk_root = pk_root; 
    sk_root["scheme"] = "QWTSS-PRIVATE";
    sk_root["grid_data"] = encode_grid_to_hex(private_key.private_key);

    // Write to disk using the modern StreamWriterBuilder
    Json::StreamWriterBuilder builder;
    builder["indentation"] = "  ";  // Pretty print with 2 spaces
    std::unique_ptr<Json::StreamWriter> writer(builder.newStreamWriter());

    // Write PK
    std::ofstream pk_file(pk_filename);
    if (pk_file.is_open()) {
        writer->write(pk_root, &pk_file);
        pk_file << std::endl; // Ensure trailing newline for POSIX compliance
        if (benchmark_stats){
            // Grab the final file size in kilobytes
            double final_file_kb = static_cast<double>(pk_file.tellp()) / 1024.0;
            benchmark_stats->pk_json_size_kb.add(final_file_kb);
        }
        pk_file.close();
        if (do_logging) std::cout << "[SUCCESS] Wrote PK file: " << pk_filename << "\n";
    } else {
        std::cerr << "Error: Could not open " << pk_filename << " for writing.\n";
        return false;
    }

    // Write SK
    std::ofstream sk_file(sk_filename);
    if (sk_file.is_open()) {
        writer->write(sk_root, &sk_file);
        sk_file << std::endl; // Ensure trailing newline for POSIX compliance
        if (benchmark_stats){
            // Grab the final file size in kilobytes
            double final_file_kb = static_cast<double>(sk_file.tellp()) / 1024.0;
            benchmark_stats->sk_json_size_kb.add(final_file_kb);
        }
        sk_file.close();
        if (do_logging) std::cout << "[SUCCESS] Wrote SK file: " << sk_filename << "\n";
    } else {
        std::cerr << "Error: Could not open " << sk_filename << " for writing.\n";
        return false;
    }

    return true;
}

enum CliAction {
    KEYGEN,
    SIGN,
    VERIFY
};

struct BenchmarkPayload {
    double clock_time_ms;
    double pk_json_kb;
    double sk_json_kb;
    double sig_raw_kb;
    double sig_json_kb;
};

class QwtssCLI {
private:
    // Helper to get current RSS of the calling process
    static inline double get_current_rss_mb() {
        struct rusage usage;
        getrusage(RUSAGE_SELF, &usage);
        return static_cast<double>(usage.ru_maxrss) / 1024.0;
    }

    // Sterile benchmark wrapper to cleanly measure peak RAM in MB
    static int run_isolated_cli_action(BenchmarkSuite& benchmark_stats, CliAction action, std::vector<std::string_view> args){
        // Capture the sterile parent's baseline before the fork
        double parent_baseline_mb = get_current_rss_mb(); 

        // Create a POSIX pipe (fd[0] is read, fd[1] is write)
        int fd[2];
        if (pipe(fd) == -1) {
            std::cerr << "Pipe failed\n";
            return EXIT_FAILURE;
        }

        pid_t pid = fork();
        if (pid == 0) {
            // ==========================================
            // CHILD PROCESS (Sterile Execution)
            // ==========================================
            close(fd[0]); // Close the reading end of the pipe

            // This child BenchmarkSuite instance is totally decoupled from the parent process master instance
            BenchmarkSuite local_stats = {};
            BenchmarkPayload payload = {};
            if (action == CliAction::KEYGEN){
                if (handle_keygen(args, &local_stats) != EXIT_SUCCESS) {
                    exit(1); // Fail cleanly
                }

                // Pack the payload struct with the results of the single run
                payload.clock_time_ms = local_stats.keygen_time_ms.mean(); // The single run time
                payload.pk_json_kb = local_stats.pk_json_size_kb.mean();
                payload.sk_json_kb = local_stats.sk_json_size_kb.mean();

            } else if (action == CliAction::SIGN){
                if (handle_sign(args, &local_stats) != EXIT_SUCCESS) {
                    exit(1); // Fail cleanly
                }

                // Pack the payload struct with the results of the single run
                payload.clock_time_ms = local_stats.sign_time_ms.mean(); // The single run time
                payload.sig_raw_kb = local_stats.sig_raw_size_kb.mean();
                payload.sig_json_kb = local_stats.sig_json_size_kb.mean();

            } else if (action == CliAction::VERIFY){
                if (handle_verify(args, &local_stats) != EXIT_SUCCESS) {
                    exit(1); // Fail cleanly
                }

                // Pack the payload struct with the results of the single run
                payload.clock_time_ms = local_stats.verify_time_ms.mean(); // The single run time
            }

            // Send the entire struct across the pipe as raw bytes
            write(fd[1], &payload, sizeof(BenchmarkPayload));

            close(fd[1]); // Close pipe
            exit(0);      // Die, returning all massive heap memory to the OS
        } 
        else if (pid > 0) {
            // ==========================================
            // PARENT PROCESS (Orchestrator)
            // ==========================================
            close(fd[1]); // Close the writing end

            int status;
            struct rusage child_usage;
            
            // Block until the child finishes all math and writes the files
            wait4(pid, &status, 0, &child_usage);

            if (WIFEXITED(status) && WEXITSTATUS(status) == 0) {
                // Harvest the Peak RAM directly from the OS
                double child_peak_mb = static_cast<double>(child_usage.ru_maxrss) / 1024.0;
                double isolated_ram = child_peak_mb - parent_baseline_mb;

                // Read the raw bytes passed through the pipe back into a payload struct
                BenchmarkPayload payload = {};
                read(fd[0], &payload, sizeof(BenchmarkPayload));

                // Add the child's metrics to the parent's master BenchmarkSuite dataset
                if (action == CliAction::KEYGEN){
                    benchmark_stats.keygen_time_ms.add(payload.clock_time_ms);
                    benchmark_stats.pk_json_size_kb.add(payload.pk_json_kb);
                    benchmark_stats.sk_json_size_kb.add(payload.sk_json_kb);
                    benchmark_stats.keygen_peak_ram_mb.add((isolated_ram > 0) ? isolated_ram : 0.0);
                } else if (action == CliAction::SIGN){
                    benchmark_stats.sign_time_ms.add(payload.clock_time_ms);
                    benchmark_stats.sig_raw_size_kb.add(payload.sig_raw_kb);
                    benchmark_stats.sig_json_size_kb.add(payload.sig_json_kb);
                    benchmark_stats.sign_peak_ram_mb.add((isolated_ram > 0) ? isolated_ram : 0.0);
                } else if (action == CliAction::VERIFY){
                    benchmark_stats.verify_time_ms.add(payload.clock_time_ms);
                    benchmark_stats.verify_peak_ram_mb.add((isolated_ram > 0) ? isolated_ram : 0.0);
                }

            } else {
                return EXIT_FAILURE;
            }
            close(fd[0]);

            return EXIT_SUCCESS;
        } else {
            // ==========================================
            // FORK FAILED (pid < 0)
            // ==========================================
            std::cerr << "Error: fork() failed to create child process.\n";
            return EXIT_FAILURE;
        }
    }


public:
    static void print_global_help() {
        std::cout << "QWTSS (Quasiperiodic Wang Tiling zk-STARK Signatures) CLI\n"
                  << "Usage: qwtss-cli <command> [options]\n\n"
                  << "Commands:\n"
                  << "  keygen    Generate a new PK/SK identity using MCMC\n"
                  << "  sign      Sign a message using an existing SK\n"
                  << "  verify    Verify a signature against a PK and message\n"
                  << "  benchmark Run pipeline benchmarking\n\n"
                  << "Run 'qwtss-cli <command> --help' for more information on a command.\n";
    }

    static int handle_keygen(const std::vector<std::string_view>& args, BenchmarkSuite* benchmark_stats = nullptr) {
        std::string username = "";
        uint64_t identity_nonce = 0;
        AnnealDevice anneal_device = AnnealDevice::AUTO;
        int target_cpu_threads = -1; // -1 means AUTO
        AnnealMode anneal_mode = AnnealMode::FAST;
        bool do_logging = true;
        bool zero_logging = false;

        for (size_t i = 0; i < args.size(); ++i) {
            if (args[i] == "--help" || args[i] == "-h") {
                std::cout << "Usage: qwtss-cli keygen -u <username> [-n <nonce>] [-d <CPU|GPU|AUTO>] [-t <# of threads>] [-a <FAST|ORIGINAL|ADIABATIC>] [-q]\n"
                          << "Options:\n"
                          << "  -u, --user    String identifier for the public key username. (Required)\n"
                          << "  -n, --nonce   Uint64 identity nonce for the public key username. Default: 0.\n"
                          << "  -d, --device  Annealing device target <CPU|GPU|AUTO>. Default: AUTO.\n"
                          << "  -t, --threads Number of CPU threads to use (CPU mode only). Default: AUTO (70% of cores).\n"
                          << "  -a, --anneal  Thermodynamic mode <FAST|ORIGINAL|ADIABATIC>. Default: FAST.\n"
                          << "  -q, --quiet   Suppress verbose output logging.\n\n"
                          << "Thermodynamic annealing mode details:\n"
                          << "  FAST\n"
                          << "     Fixed point integer MCMC math. Replaces expf() with a dynamic LUT and utilizes branchless heuristics.\n"
                          << "     Recommended. Guarantees 100% cross-platform deterministic key generation and achieves the highest performance.\n"
                          << "  ORIGINAL\n"
                          << "     Legacy floating-point MCMC math with heuristic shortcuts. Faster than adiabatic mode.\n"
                          << "     Subject to floating-point non-determinism across different hardware architectures.\n"
                          << "  ADIABATIC\n"
                          << "     Floating-point MCMC math following a strict, unaccelerated adiabatic cooling schedule.\n"
                          << "     No heuristic shortcuts. Slowest key generation time and subject to floating-point non-determinism.\n";
                return EXIT_SUCCESS;
            } else if ((args[i] == "-u" || args[i] == "--user") && i + 1 < args.size()) {
                username = args[++i];
            } else if ((args[i] == "-n" || args[i] == "--nonce") && i + 1 < args.size()) {
                identity_nonce = std::stoull(std::string(args[++i]));
            } else if ((args[i] == "-d" || args[i] == "--device") && i + 1 < args.size()) {
                std::string dev_str = std::string(args[++i]);
                std::transform(dev_str.begin(), dev_str.end(), dev_str.begin(), ::toupper);
                
                if (dev_str == "CPU") anneal_device = AnnealDevice::CPU;
                else if (dev_str == "GPU") anneal_device = AnnealDevice::GPU;
                else if (dev_str == "AUTO") anneal_device = AnnealDevice::AUTO;
                else {
                    std::cerr << "Error: Invalid device. Choose CPU, GPU, or AUTO.\n";
                    return EXIT_FAILURE;
                }
            } else if ((args[i] == "-t" || args[i] == "--threads") && i + 1 < args.size()) {
                target_cpu_threads = std::stoi(std::string(args[++i]));
            } else if ((args[i] == "-a" || args[i] == "--anneal") && i + 1 < args.size()) {
                std::string mode_str = std::string(args[++i]);
                std::transform(mode_str.begin(), mode_str.end(), mode_str.begin(), ::toupper);
                
                if (mode_str == "FAST") anneal_mode = AnnealMode::FAST;
                else if (mode_str == "ORIGINAL") anneal_mode = AnnealMode::ORIGINAL;
                else if (mode_str == "ADIABATIC") anneal_mode = AnnealMode::ADIABATIC;
                else {
                    std::cerr << "Error: Invalid anneal mode. Choose FAST, ORIGINAL, or ADIABATIC.\n";
                    return EXIT_FAILURE;
                }
            } else if (args[i] == "-q" || args[i] == "--quiet") {
                do_logging = false;
            } else if (args[i] == "-qq" || args[i] == "--qq") {
                do_logging = false;
                zero_logging = true;
            }
        }

        if (username.empty()) {
            std::cerr << "Error: --user is required.\n";
            return EXIT_FAILURE;
        }

        if (anneal_device == AnnealDevice::GPU || anneal_device == AnnealDevice::AUTO){
            // Ensure GPU is accessible and valid
            bool gpu_exists = detect_eligible_nvidia_gpu(do_logging); // || anneal_device == AnnealDevice::AUTO);

            if (anneal_device == AnnealDevice::GPU && !gpu_exists){
                std::cerr << "Error: no compatible NVIDIA GPU detected; use --device -CPU instead.\n";
                return EXIT_FAILURE;
            }
            if (anneal_device == AnnealDevice::AUTO){
                if (gpu_exists) anneal_device = AnnealDevice::GPU;
                else {
                    anneal_device = AnnealDevice::CPU;
                    if (do_logging) std::cout << "No compatible NVIDIA GPU detected; using CPU device target" << std::endl;
                }
            }
        }
        if (target_cpu_threads != -1){
            if (target_cpu_threads < 1){
                std::cerr << "Error: the target CPU threads value must be 1 or more.\n";
                return EXIT_FAILURE;
            } else if (anneal_device != AnnealDevice::CPU){
                std::cerr << "Error: the target CPU threads value can only be defined with the CPU device target.\n";
                return EXIT_FAILURE;
            }
        }

        QwtssPrivateKey private_key;
        auto start = std::chrono::high_resolution_clock::now();
        // Try up to 5 times (would be rare to fail more than once in FAST mode), but 
        // note that ADIABATIC mode requires more possible tries.
        int max_tries = (anneal_mode == AnnealMode::ADIABATIC ? 20 : 5);
        for (int t = 0; t < max_tries; t++){
            private_key = build_qwtss_private_key(username, identity_nonce, anneal_mode, anneal_device, target_cpu_threads, do_logging);

            if (private_key.private_key.empty()){
                if (do_logging) std::cout << "\nTry " << (t+1) << " failed: private key convergence failed" << std::endl;
                continue;
            } else {
                break;
            }
        }
        auto end = std::chrono::high_resolution_clock::now();

        if (benchmark_stats) {
            std::chrono::duration<double, std::milli> elapsed = end - start;
            benchmark_stats->keygen_time_ms.add(elapsed.count());
        }

        if (private_key.private_key.empty()){
            std::cout << "Error: All " << max_tries << " attempts at private key convergence failed" << std::endl;
            return EXIT_FAILURE;
        }
        if (private_key.private_key.size() != (QwtssReference::grid_size * QwtssReference::grid_size)){
            std::cout << "Error: The private key has an improper length" << std::endl;
            return EXIT_FAILURE;
        }

        // Logging on success occurs internally
        bool wrote_json_files = write_qwtss_keypair_to_disk(username, identity_nonce, QwtssReference::version,
                                    private_key, !zero_logging, benchmark_stats);

        return (wrote_json_files ? EXIT_SUCCESS : EXIT_FAILURE);
    }

    static int handle_sign(const std::vector<std::string_view>& args, BenchmarkSuite* benchmark_stats = nullptr) {
        std::string sk_path = "";
        std::string message = "";
        std::string file_path = "";
        std::string out_path = "";
        bool do_logging = true;
        bool zero_logging = false;

        for (size_t i = 0; i < args.size(); ++i) {
            if (args[i] == "--help" || args[i] == "-h") {
                std::cout << "Usage: qwtss-cli sign -k <sk.json> (-m <message> | -f <file>) [-o <output>] [-q]\n"
                        << "Options:\n"
                        << "  -k, --key     Path to the secret key JSON. (Required)\n"
                        << "  -f, --file    Path to the file to sign.\n"
                        << "  -m, --msg     A raw text string to sign.\n"
                        << "  -o, --out     Output path for the signature. Default: <file>.sig.json\n"
                        << "  -q, --quiet   Suppress verbose output logging.\n";
                return EXIT_SUCCESS;
            } else if ((args[i] == "-k" || args[i] == "--key") && i + 1 < args.size()) {
                sk_path = args[++i];
            } else if ((args[i] == "-f" || args[i] == "--file") && i + 1 < args.size()) {
                file_path = args[++i];
            } else if ((args[i] == "-m" || args[i] == "--msg") && i + 1 < args.size()) {
                message = args[++i];
            } else if ((args[i] == "-o" || args[i] == "--out") && i + 1 < args.size()) {
                out_path = args[++i];
            } else if (args[i] == "-q" || args[i] == "--quiet") {
                do_logging = false;
            } else if (args[i] == "-qq" || args[i] == "--qq") {
                do_logging = false;
                zero_logging = true;
            }
        }

        if (sk_path.empty()) {
            std::cerr << "Error: --key is required.\n";
            return EXIT_FAILURE;
        }

        if (message.empty() && file_path.empty()) {
            std::cerr << "Error: You must provide either a --msg or a --file to sign.\n";
            return EXIT_FAILURE;
        }

        if (!message.empty() && !file_path.empty()) {
            std::cerr << "Error: --msg and --file are mutually exclusive. Choose one only.\n";
            return EXIT_FAILURE;
        }

        // Input Resolution
        std::vector<uint8_t> payload_digest; // Stores the SHA-256 hash
        if (!file_path.empty()) {
            if (!fs::exists(file_path)) {
                std::cerr << "Error: Target file does not exist: " << file_path << "\n";
                return EXIT_FAILURE;
            }

            if (do_logging) std::cout << "Reading and hashing file: " << file_path << "...\n";
            
            // Read file in chunks and compute standard SHA-256
            payload_digest = SHA256::hash_file_bytes(file_path);

            // Resolve default output name if none provided
            if (out_path.empty()) {
                out_path = file_path + ".sig.json";
            }
        } else {
            if (do_logging) std::cout << "Hashing raw message string (" << message.size() << " chars)...\n";
            
            // Compute SHA-256 of the raw string
            payload_digest = SHA256::hash_string_bytes(message);

            if (out_path.empty()) {
                out_path = "message.sig.json";
            }
        }

        if (do_logging) std::cout << "Loading private key from " << sk_path << "...\n";

        // Read and parse the JSON Secret Key
        std::ifstream sk_stream(sk_path);
        if (!sk_stream.is_open()) {
            std::cerr << "Error: Could not open " << sk_path << "\n";
            return EXIT_FAILURE;
        }

        Json::Value sk_root;
        Json::CharReaderBuilder reader;
        std::string errs;
        if (!Json::parseFromStream(reader, sk_stream, &sk_root, &errs)) {
            std::cerr << "Error parsing JSON: " << errs << "\n";
            return EXIT_FAILURE;
        }

        if (sk_root["scheme"].asString() != "QWTSS-PRIVATE") {
            std::cerr << "Error: Invalid key scheme. Expected 'QWTSS-PRIVATE'.\n";
            return EXIT_FAILURE;
        }

        // Extract metadata
        std::string username = sk_root["username"].asString();
        uint64_t identity_nonce = sk_root["identity_nonce"].asUInt64();
        uint8_t version = static_cast<uint8_t>(sk_root["version"].asUInt());

        if (username.empty()){
            std::cerr << "Error: Invalid key scheme. Expected a non-empty username.\n";
            return EXIT_FAILURE;
        }
        if (version != QwtssReference::version){
            std::cerr << "Error: Invalid key scheme. Version '" << std::to_string(version)
                    << "' is not supported by this executable. Expected '" << std::to_string(QwtssReference::version) << "'.\n";
            return EXIT_FAILURE;
        }

        // Deserialize grid data and instantiate the private key
        QwtssPrivateKey private_key;
        private_key.private_key = decode_grid_from_hex(sk_root["grid_data"].asString());
        if (private_key.private_key.size() != (QwtssReference::grid_size * QwtssReference::grid_size)){
            std::cerr << "Error: Invalid key scheme. Grid data length (" << private_key.private_key.size()
                    << ") is invalid. Expected " << (QwtssReference::grid_size * QwtssReference::grid_size) << ".\n";
            return EXIT_FAILURE;
        }

        // Deserialize the expected fingerprint
        std::array<std::string, 2> hex_limbs = {
            sk_root["grid_fingerprint"][0].asString(),
            sk_root["grid_fingerprint"][1].asString()
        };
        std::array<FieldElement256, 2> expected_fingerprint = deserialize_fingerprint(hex_limbs);

        auto start_sign = std::chrono::high_resolution_clock::now();

        // Flesh out the other required fields derived from the PK info
        PkDerivedFields pk_derived = get_pk_derived_fields(username, identity_nonce, version,
            QwtssReference::grid_size, QwtssReference::keep_boundary_percentage, do_logging);
        private_key.boundary_mask = pk_derived.boundary_mask;
        private_key.plane_A = pk_derived.oracle_pk.plane_A;
        private_key.plane_B = pk_derived.oracle_pk.plane_B;

        // Instantiate the tile set
        JeandelRaoTileSet jr_tileset;
        std::vector<Tile> alphabet = jr_tileset.get_tiles();

        // Build Algebraic Execution Trace
        if (do_logging) std::cout << "Building algebraic execution trace...\n";
        std::vector<QwtssPrivateKey> private_keys = { private_key };
        ExecutionTrace trace;
        try {
            trace = build_execution_trace(private_keys, QwtssReference::grid_size, alphabet, true);
        } catch (const std::exception& e) {
            std::cerr << "Error: build_execution_trace() threw an exception: \n" << e.what() << std::endl;
            throw;
        }

        // Extract and Verify Fingerprint
        if (do_logging) std::cout << "Extracting QWTSS public inputs...\n";
        std::array<FieldElement256, 2> trace_fingerprint = {
            trace.expected_hash_0[trace.valid_steps + 63], 
            trace.expected_hash_1[trace.valid_steps + 63]
        };

        if (trace_fingerprint[0] != expected_fingerprint[0] || trace_fingerprint[1] != expected_fingerprint[1]) {
            std::cerr << "Error: Computed trace fingerprint does not match the Secret Key fingerprint. Secret Key could be corrupted, or a possible version mismatch.\n";
            return EXIT_FAILURE;
        }

        QwtssPublicInputs public_inputs(username, identity_nonce, version, private_key,
            QwtssReference::grid_size, trace_fingerprint, alphabet);
        //public_inputs.log_critical_fields_hash("SIGNER");

        // NOTE: the validator.validate_trace() is very expensive (almost 5 seconds) and primarily used for debugging.
        // The production traces have always been valid, so there is no need to pre-validate here.
        // The actual Stone Prover proof generation only takes ~2 seconds anyway, and it will fail if the trace is invalid.
        if (false)
        {
            // Trace Mathematical Validation (Pre-Prover Sanity Check)
            if (do_logging) std::cout << "Pre-validating trace against AIR... " << std::flush;
            QwtssAir validator(public_inputs, trace.trace_length, alphabet);
            if (!validator.validate_trace(trace)) {
                if (do_logging) std::cout << "FAILED\n" << std::flush;
                std::cerr << "Error: The execution trace failed mathematical pre-validation against the AIR\n";
                return EXIT_FAILURE;
            }
            if (do_logging) std::cout << "OK\n" << std::flush;
        }

        // STARK Prover Execution
        if (do_logging) std::cout << "Binding payload digest to Fiat-Shamir transcript and executing Stone Prover...\n";

        // Pass the SHA-256 payload_digest directly as the message to bind to the transcript
        std::vector<std::byte> final_sig = generate_stark_signature(trace, public_inputs, payload_digest, alphabet, do_logging);
        auto end_sign = std::chrono::high_resolution_clock::now();

        if (benchmark_stats){
            std::chrono::duration<double, std::milli> sign_ms = end_sign - start_sign;
            benchmark_stats->sign_time_ms.add(sign_ms.count());
        }

        // Write Signature to Disk as JSON
        Json::Value sig_root;
        sig_root["scheme"] = "QWTSS-SIGNATURE";
        sig_root["version"] = version;
        sig_root["username"] = username;
            // Cast to Json::UInt64 to ensure it maps to the correct internal JSON type
        sig_root["identity_nonce"] = static_cast<Json::UInt64>(identity_nonce);

        // Convert the binary digest to a standard Hex string for the JSON envelope
        sig_root["payload_digest"] = bytes_to_hex(payload_digest);
        // Reinterpret the std::byte vector to uint8_t for the Base64 encoder
        sig_root["signature_base64"] = base64_encode(reinterpret_cast<const uint8_t*>(final_sig.data()), final_sig.size()); 

        std::ofstream out_file(out_path);
        if (!out_file.is_open()) {
            std::cerr << "Error: Could not open output file " << out_path << " for writing.\n";
            return EXIT_FAILURE;
        }

        Json::StreamWriterBuilder builder;
        builder["indentation"] = "  ";
        std::unique_ptr<Json::StreamWriter> writer(builder.newStreamWriter());
        writer->write(sig_root, &out_file);
        out_file << "\n";
        // Grab the final file size in kilobytes
        double final_file_kb = static_cast<double>(out_file.tellp()) / 1024.0;
        out_file.close();

        double raw_kb = static_cast<double>(final_sig.size()) / 1024.0;
        if (benchmark_stats){
            benchmark_stats->sig_raw_size_kb.add(raw_kb);
            benchmark_stats->sig_json_size_kb.add(final_file_kb);
        }

        if (!zero_logging){
            std::cout << "[SUCCESS] Wrote signature to " << out_path << "\n"
                    << "          Raw proof size (binary): " << std::fixed << std::setprecision(1) << raw_kb << " KB\n"
                    << "          Final JSON size (base64): " << std::fixed << std::setprecision(1) << final_file_kb << " KB" << std::endl;
        }

        return EXIT_SUCCESS;
    }

    static int handle_verify(const std::vector<std::string_view>& args, BenchmarkSuite* benchmark_stats = nullptr) {
        std::string pk_path = "";
        std::string sig_path = "";
        std::string message = "";
        std::string file_path = "";
        bool do_logging = true;
        bool zero_logging = false;

        for (size_t i = 0; i < args.size(); ++i) {
            if (args[i] == "--help" || args[i] == "-h") {
                std::cout << "Usage: qwtss-cli verify -p <pk.json> -s <signature.json> (-m <message> | -f <file>) [-q]\n"
                        << "Options:\n"
                        << "  -p, --pk      Path to the public key JSON. (Required)\n"
                        << "  -s, --sig     Path to the signature JSON. (Required)\n"
                        << "  -f, --file    Path to the file to sign.\n"
                        << "  -m, --msg     A raw text string to sign.\n"
                        << "  -q, --quiet   Suppress verbose output logging.\n";
                return EXIT_SUCCESS;
            } else if ((args[i] == "-p" || args[i] == "--pk") && i + 1 < args.size()) {
                pk_path = args[++i];
            } else if ((args[i] == "-s" || args[i] == "--sig") && i + 1 < args.size()) {
                sig_path = args[++i];
            } else if ((args[i] == "-f" || args[i] == "--file") && i + 1 < args.size()) {
                file_path = args[++i];
            } else if ((args[i] == "-m" || args[i] == "--msg") && i + 1 < args.size()) {
                message = args[++i];
            } else if (args[i] == "-q" || args[i] == "--quiet") {
                do_logging = false;
            } else if (args[i] == "-qq" || args[i] == "--qq") {
                do_logging = false;
                zero_logging = true;
            }
        }

        if (pk_path.empty()) {
            std::cerr << "Error: --pk is required.\n";
            return EXIT_FAILURE;
        }
        if (sig_path.empty()) {
            std::cerr << "Error: --sig is required.\n";
            return EXIT_FAILURE;
        }

        if (message.empty() && file_path.empty()) {
            std::cerr << "Error: You must provide either a --msg or a --file to verify.\n";
            return EXIT_FAILURE;
        }

        if (!message.empty() && !file_path.empty()) {
            std::cerr << "Error: --msg and --file are mutually exclusive. Choose one only.\n";
            return EXIT_FAILURE;
        }

        // Input Resolution
        std::vector<uint8_t> payload_digest; // Stores the SHA-256 hash
        std::string payload_type = "?";
        if (!file_path.empty()) {
            if (!fs::exists(file_path)) {
                std::cerr << "Error: Target file does not exist: " << file_path << "\n";
                return EXIT_FAILURE;
            }

            if (do_logging) std::cout << "Reading and hashing file: " << file_path << "...\n";
            
            // Read file in chunks and compute standard SHA-256
            payload_digest = SHA256::hash_file_bytes(file_path);
            payload_type = "file";
        } else {
            if (do_logging) std::cout << "Hashing raw message string (" << message.size() << " chars)...\n";
            
            // Compute SHA-256 of the raw string
            payload_digest = SHA256::hash_string_bytes(message);
            payload_type = "message";
        }

        if (do_logging) std::cout << "Loading signature from " << sig_path << "...\n";

        // Read and parse the JSON Signature
        std::ifstream sig_stream(sig_path);
        if (!sig_stream.is_open()) {
            std::cerr << "Error: Could not open " << sig_path << "\n";
            return EXIT_FAILURE;
        }

        Json::Value sig_root;
        {
            Json::CharReaderBuilder reader;
            std::string errs;
            if (!Json::parseFromStream(reader, sig_stream, &sig_root, &errs)) {
                std::cerr << "Error parsing JSON: " << errs << "\n";
                return EXIT_FAILURE;
            }
        }

        if (sig_root["scheme"].asString() != "QWTSS-SIGNATURE") {
            std::cerr << "Error: Invalid key scheme. Expected 'QWTSS-SIGNATURE'.\n";
            return EXIT_FAILURE;
        }

        // Extract metadata
        std::string sig_username = sig_root["username"].asString();
        uint64_t sig_identity_nonce = sig_root["identity_nonce"].asUInt64();
        uint8_t sig_version = static_cast<uint8_t>(sig_root["version"].asUInt());

        if (sig_username.empty()){
            std::cerr << "Error: Invalid key scheme. Expected a non-empty username.\n";
            return EXIT_FAILURE;
        }
        if (sig_version != QwtssReference::version){
            std::cerr << "Error: Invalid key scheme. Version '" << std::to_string(sig_version)
                    << "' is not supported by this executable. Expected '" << std::to_string(QwtssReference::version) << "'.\n";
            return EXIT_FAILURE;
        }

        // Extract and decode the payload digest
        std::string sig_payload_hex = sig_root["payload_digest"].asString();
        std::vector<uint8_t> sig_payload_digest = hex_to_bytes(sig_payload_hex);
        if (sig_payload_digest != payload_digest){
            std::cerr << "Error: The payload digest in the signature file does not match the digest of the local "
                      << payload_type << " specified here. The payload contents must be identical.\n";
            return EXIT_FAILURE;
        }

        // Extract and decode the STARK Proof
        std::string sig_base64 = sig_root["signature_base64"].asString();
        std::vector<uint8_t> sig_decoded_bytes = base64_decode(sig_base64);
        
        // Re-cast the decoded uint8_t bytes back to std::byte for the Prover API
        std::vector<std::byte> stark_proof;
        stark_proof.reserve(sig_decoded_bytes.size());
        for (uint8_t b : sig_decoded_bytes) {
            stark_proof.push_back(static_cast<std::byte>(b));
        }

        if (do_logging) std::cout << "Loading public key from " << pk_path << "...\n";

        // Read and parse the JSON Public Key
        std::ifstream pk_stream(pk_path);
        if (!pk_stream.is_open()) {
            std::cerr << "Error: Could not open " << pk_path << "\n";
            return EXIT_FAILURE;
        }

        Json::Value pk_root;
        {
            Json::CharReaderBuilder reader;
            std::string errs;
            if (!Json::parseFromStream(reader, pk_stream, &pk_root, &errs)) {
                std::cerr << "Error parsing JSON: " << errs << "\n";
                return EXIT_FAILURE;
            }
        }

        if (pk_root["scheme"].asString() != "QWTSS-PUBLIC") {
            std::cerr << "Error: Invalid key scheme. Expected 'QWTSS-PUBLIC'.\n";
            return EXIT_FAILURE;
        }

        // Extract metadata
        std::string username = pk_root["username"].asString();
        uint64_t identity_nonce = pk_root["identity_nonce"].asUInt64();
        uint8_t version = static_cast<uint8_t>(pk_root["version"].asUInt());

        if (username.empty()){
            std::cerr << "Error: Invalid key scheme. Expected a non-empty username.\n";
            return EXIT_FAILURE;
        }
        if (version != QwtssReference::version){
            std::cerr << "Error: Invalid key scheme. Version '" << std::to_string(version)
                    << "' is not supported by this executable. Expected '" << std::to_string(QwtssReference::version) << "'.\n";
            return EXIT_FAILURE;
        }
        if (username != sig_username) {
            std::cerr << "Error: The signature username does not match the PK username.\n";
            return EXIT_FAILURE;
        }
        if (identity_nonce != sig_identity_nonce) {
            std::cerr << "Error: The signature identity nonce does not match the PK identity nonce.\n";
            return EXIT_FAILURE;
        }
        if (version != sig_version) {
            std::cerr << "Error: The signature version does not match the PK version.\n";
            return EXIT_FAILURE;
        }

        // Deserialize the PK grid fingerprint
        std::array<std::string, 2> hex_limbs = {
            pk_root["grid_fingerprint"][0].asString(),
            pk_root["grid_fingerprint"][1].asString()
        };
        std::array<FieldElement256, 2> grid_fingerprint = deserialize_fingerprint(hex_limbs);

        auto start_verify = std::chrono::high_resolution_clock::now();

        if (do_logging) std::cout << "Extracting QWTSS public inputs...\n";
        // Flesh out the other required fields derived from the PK info
        PkDerivedFields pk_derived = get_pk_derived_fields(username, identity_nonce, version,
            QwtssReference::grid_size, QwtssReference::keep_boundary_percentage, do_logging);

        // Instantiate the tile set
        JeandelRaoTileSet jr_tileset;
        std::vector<Tile> alphabet = jr_tileset.get_tiles();

        QwtssPublicInputs public_inputs(username, identity_nonce, version, pk_derived,
            QwtssReference::grid_size, grid_fingerprint, alphabet);
        //public_inputs.log_critical_fields_hash("VERIFIER");

        if (do_logging) std::cout << "Executing Stone Prover...\n";
        bool is_valid = verify_stark_signature(stark_proof, public_inputs, payload_digest, alphabet, false);
        auto end_verify = std::chrono::high_resolution_clock::now();

        std::chrono::duration<double, std::milli> verify_ms = end_verify - start_verify;
        if (benchmark_stats){
            benchmark_stats->verify_time_ms.add(verify_ms.count());
        }

        if (!zero_logging){
            if (is_valid) {
                std::cout << "[SUCCESS] Valid signature confirmed. Verified in " 
                        << std::fixed << std::setprecision(1) << verify_ms.count() << " ms.\n";
                return EXIT_SUCCESS;
            } else {
                std::cerr << "[FAILURE] Invalid signature: The STARK proof was rejected by the verifier.\n";
                return EXIT_FAILURE;
            }
        }

        return (is_valid ? EXIT_SUCCESS : EXIT_FAILURE);
    }

    static int handle_benchmark(const std::vector<std::string_view>& args) {
        int iterations = 10;
        std::string device_str = "AUTO"; // Store as string to pass to child cleanly
        std::string cpu_threads_str = "-1"; // -1 means AUTO
        std::string anneal_mode_str = "FAST";

        for (size_t i = 0; i < args.size(); ++i) {
            if (args[i] == "--help" || args[i] == "-h") {
                std::cout << "Usage: qwtss-cli keygen [-i <iterations>] [-d <CPU|GPU|AUTO>] [-t <# of threads>] [-a <FAST|ORIGINAL|ADIABATIC>]\n"
                          << "Options:\n"
                          << "  -i, --iter    The 1+ number of pipeline iterations to perform. Default: 10.\n"
                          << "  -d, --device  Annealing device target <CPU|GPU|AUTO>. Default: AUTO.\n"
                          << "  -t, --threads Number of CPU threads to use (CPU mode only). Default: AUTO (70% of cores).\n"
                          << "  -a, --anneal  Thermodynamic mode <FAST|ORIGINAL|ADIABATIC>. Default: FAST.\n\n"
                          << "Thermodynamic annealing mode details:\n"
                          << "  FAST\n"
                          << "     Fixed point integer MCMC math. Replaces expf() with a dynamic LUT and utilizes branchless heuristics.\n"
                          << "     Recommended. Guarantees 100% cross-platform deterministic key generation and achieves the highest performance.\n"
                          << "  ORIGINAL\n"
                          << "     Legacy floating-point MCMC math with heuristic shortcuts. Faster than adiabatic mode.\n"
                          << "     Subject to floating-point non-determinism across different hardware architectures.\n"
                          << "  ADIABATIC\n"
                          << "     Floating-point MCMC math following a strict, unaccelerated adiabatic cooling schedule.\n"
                          << "     No heuristic shortcuts. Slowest key generation time and subject to floating-point non-determinism.\n";
                return EXIT_SUCCESS;
            } else if ((args[i] == "-i" || args[i] == "--iter") && i + 1 < args.size()) {
                iterations = std::stoi(std::string(args[++i]));
            } else if ((args[i] == "-d" || args[i] == "--device") && i + 1 < args.size()) {
                device_str = std::string(args[++i]);
                std::transform(device_str.begin(), device_str.end(), device_str.begin(), ::toupper);
                if (device_str != "CPU" && device_str != "GPU" && device_str != "AUTO") {
                    std::cerr << "Error: Invalid device. Choose CPU, GPU, or AUTO.\n";
                    return EXIT_FAILURE;
                }
            } else if ((args[i] == "-t" || args[i] == "--threads") && i + 1 < args.size()) {
                cpu_threads_str = std::string(args[++i]);
            } else if ((args[i] == "-a" || args[i] == "--anneal") && i + 1 < args.size()) {
                anneal_mode_str = std::string(args[++i]);
                std::transform(anneal_mode_str.begin(), anneal_mode_str.end(), anneal_mode_str.begin(), ::toupper);
                if (anneal_mode_str != "FAST" && anneal_mode_str != "ADIABATIC") {
                    std::cerr << "Error: Invalid anneal mode. Choose FAST or ADIABATIC.\n";
                    return EXIT_FAILURE;
                }
            }
        }

        BenchmarkSuite benchmark_stats = {};

        // Static parameters
        std::string nonce_str = "0";
        std::string sig_file = "test.sig.json";
        std::string message = "Deterministic QWTSS Test Message";

        std::cout << "[*] Starting Benchmarking Loop (" << iterations << " iterations)...\n";

        int success_count = 0;
        auto global_start = std::chrono::high_resolution_clock::now();
        ChaCha20PRNG rng;

        for (int i = 0; i < iterations; ++i) {
            // Generate the dynamic username and filenames for this iteration
            std::string rnd_prefix = generate_random_username(rng, 5, 8);
            std::string username = rnd_prefix + "_test_user";
            
            std::string pk_file = "qwtss_" + username + "_0_pk.json";
            std::string sk_file = "qwtss_" + username + "_0_sk.json";

            // Print enough blank spaces at the end to overwrite longer usernames from previous loops
            std::cout << "\r[*] Running pipeline iteration " << (i + 1) << " / " << iterations 
                      << " for user: " << username << "      " << std::flush;

            // Ensure any existing files with the same name are deleted
            if (fs::exists(pk_file)) fs::remove(pk_file);
            if (fs::exists(sk_file)) fs::remove(sk_file);
            if (fs::exists(sig_file)) fs::remove(sig_file);

            // Simulate: qwtss-cli keygen -u <dynamic_username> -n 0 -qq
            std::vector<std::string_view> keygen_args = { "-u", username, "-n", nonce_str, "-d", device_str,
                "-t", cpu_threads_str, "-a", anneal_mode_str, "-qq" };
            if (run_isolated_cli_action(benchmark_stats, CliAction::KEYGEN, keygen_args) != EXIT_SUCCESS) {
                std::cerr << "\n[!] FAILED at Keygen on iteration " << (i + 1) << " with user " << username << "\n";
                return EXIT_FAILURE;
            }

            // Simulate: qwtss-cli sign -k <sk> -m <msg> -o <sig> -qq
            std::vector<std::string_view> sign_args = { "-k", sk_file, "-m", message, "-o", sig_file, "-qq" };
            if (run_isolated_cli_action(benchmark_stats, CliAction::SIGN, sign_args) != EXIT_SUCCESS) {
                std::cerr << "\n[!] FAILED at Sign on iteration " << (i + 1) << " with user " << username << "\n";
                return EXIT_FAILURE;
            }

            // Simulate: qwtss-cli verify -p <pk> -s <sig> -m <msg> -qq
            std::vector<std::string_view> verify_args = { "-p", pk_file, "-s", sig_file, "-m", message, "-qq" };
            if (run_isolated_cli_action(benchmark_stats, CliAction::VERIFY, verify_args) != EXIT_SUCCESS) {
                std::cerr << "\n[!] FAILED at Verify on iteration " << (i + 1) << " with user " << username << "\n";
                return EXIT_FAILURE;
            }

            // Cleanup for next run to ensure no cross-contamination
            fs::remove(pk_file);
            fs::remove(sk_file);
            fs::remove(sig_file);
            
            success_count++;
        }

        auto global_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = global_end - global_start;

        std::cout << "\n\n================================================================================================\n";
        std::cout << "[+] TEST & BENCHMARKING SUITE PASSED\n";
        std::cout << "[+] Successfully completed " << success_count << " full pipeline loops.\n";
        std::cout << "[+] Total Time: " << std::fixed << std::setprecision(2) << elapsed.count() << " seconds\n";
        benchmark_stats.print_report(device_str, cpu_threads_str, anneal_mode_str);

        return EXIT_SUCCESS;
    }
};

int main(int argc, char* argv[]) {
    if (argc < 2) {
        QwtssCLI::print_global_help();
        return 1;
    }

    // Initialize Google Logging to silence the WARNING message
    google::InitGoogleLogging(argv[0]); 
    // Optional: Ensure logs still print to the console instead of a file
    FLAGS_logtostderr = true;

    std::string_view command = argv[1];
    std::vector<std::string_view> args(argv + 2, argv + argc);

    if (command == "keygen") {
        return QwtssCLI::handle_keygen(args);
    } else if (command == "sign") {
        return QwtssCLI::handle_sign(args);
    } else if (command == "verify") {
        return QwtssCLI::handle_verify(args);
    } else if (command == "benchmark") {
        return QwtssCLI::handle_benchmark(args);
    } else if (command == "--help" || command == "-h") {
        QwtssCLI::print_global_help();
        return 0;
    } else {
        std::cerr << "Unknown command: " << command << "\n";
        QwtssCLI::print_global_help();
        return 1;
    }
}
