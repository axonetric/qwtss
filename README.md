# QWTSS: Quasiperiodic Wang Tiling zk-STARK Signatures

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![C++17](https://img.shields.io/badge/Standard-C++17-blue.svg)](https://en.wikipedia.org/wiki/C%2B%2B17)
[![Paper](https://img.shields.io/badge/Paper-PDF-red.svg)](./Reeves_2026_QWTSS_Whitepaper_v1.0.pdf)

### A high-performance C++/CUDA reference implementation of the QWTSS post-quantum digital signature scheme. 

This repository provides the core cryptographic pipeline, including the thermodynamic key generation solver, the STARK algebraic execution trace builder and AIR, and the integrated command-line interface.

## Reference Paper
**Title:** Quasiperiodic Wang Tiling zk-STARK Signatures (QWTSS): A Post-Quantum Signature Scheme Anchored in the Topologically Frustrated Glassy Phase  
**Authors:** Kevin Reeves  
**Link:** [Read the QWTSS Whitepaper (PDF)](./Reeves_2026_QWTSS_Whitepaper_v1.0.pdf)

### Abstract
The emergence of quantum computing threatens classical public-key cryptography, necessitating the development of robust post-quantum alternatives built upon structurally diverse hardness assumptions. We introduce Quasiperiodic Wang Tiling zk-STARK Signatures (QWTSS), a novel digital signature scheme grounded in the NP-hard combinatorial optimization of bounded aperiodic Wang tile sets. QWTSS generates secret keys by sampling solutions deep within the thermodynamic glassy phase of the configuration space. We empirically demonstrate that this critically constrained regime undergoes solution-space shattering and exhibits the Overlap Gap Property (OGP), fracturing into isolated, multifractal solution clusters. The resulting critically constrained operational regime provides maximum combinatorial backtracking against exact classical solvers, while the OGP inhibits heuristic local-search traversal. Furthermore, we model how the extreme phase-space anisotropy of these clusters exponentially suppresses the quantum overlap integral. By coupling this structural hardness with a zk-STARK, QWTSS enables a prover to demonstrate knowledge of a specific geometric configuration without leaking the underlying witness. This yields a highly secure, non-algebraic post-quantum primitive inextricably bound to non-trivial simulated thermodynamic work.

## Repository Overview
The repository is organized as follows:
| Directory | Description |
| :--- | :--- |
| **/external** | Private key db for testing, etc. |
| **/scripts** | Various supporting Python scripts, including KAT generation and NUMS constants generation scripts for the Poseidon-1 sponge hash. |
| **/src/cli** | The command-line interface target. |
| **/src/core** | The core QWTSS logic shared by all targets. Supports both CPU key generation and GPU-accelerated key generation. The signing and verification operations are CPU only. |
| **/src/research** | Various auxiliary logic supporting findings in the whitepaper, cryptanalysis, statistics and data collection. |
| **/src/tests** | A comprehensive test suite, including 77 unique, targeted sabotage attacks on the AIR constraints. |

## Build Requirements  

### System Dependencies  
* GCC/G++ 11+ (or Clang equivalent)
* NVIDIA CUDA Toolkit 12.x
* Bazel 6.x+ (Required for Stone Prover compilation)
* Required C++ Libraries: `libgmp-dev`, `libgmpxx4ldbl`, `libboost-all-dev`, `libgflags-dev`, `libgoogle-glog-dev`, `libdw-dev`, `libomp-dev`

### Source-Level Dependency
* StarkWare Stone Prover: Before building QWTSS, you must clone and compile the StarkWare Stone Prover from source, and then link the static libraries into the QWTSS build. **[Click here to jump to the detailed Stone Prover Integration instructions](#stone-prover-integration)** at the bottom of this page.

### Hardware & Operating System Requirements  
* **Operating System:** Developed and tested on Ubuntu 24.04 LTS base with C++17. Compilation on other modern Linux distributions is expected to be straightforward.
* **GPU Requirements:** A dedicated NVIDIA GPU is **not required** to execute the primary cryptographic pipeline. The CLI provides a fully functional CPU fallback for Key Generation, Signing, and Verification. 
* **CUDA Dependency:** Even if executing solely on the CPU, the **NVIDIA CUDA Toolkit (12.x)** must be installed on the host machine to successfully compile the shared core library. Note that certain supplementary / analysis modules (e.g., within `/src/research`) strictly require GPU hardware to run.

## Building the Project  
Once the system dependencies are installed and the Stone Prover is compiled (see below), you can build the QWTSS CLI using standard CMake commands from the root of this repository:

```bash
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

## CLI Usage  
The executable provides four primary modes: `keygen`, `sign`, `verify`, and `benchmark`. Full `--help` documentation is available via the CLI for each mode.

**1. Key Generation**  
Generates a new thermodynamically annealed private key grid and associated boundary constraints for the specified username and nonce.
```bash
./qwtss-cli keygen <username> <identity_nonce> [options]
```
*Outputs:* `private_key.json` and `public_key.json`

**2. Signing**  
Constructs the execution trace and generates the zk-STARK signature for the provided message.
```bash
./qwtss-cli sign <private_key_file> <message_string_or_file> [options]
```
*Outputs:* `signature.json`

**3. Verification**  
Validates the STARK proof against the provided message and public key.
```bash
./qwtss-cli verify <public_key_file> <signature_file> <message_string_or_file> [options]
```
*Outputs:* Verification SUCCESS or Verification FAILED

**4. Benchmarking**  
Executes the comprehensive benchmark suite of the full pipeline (key generation via MCMC solver, STARK prover, and verifier).
```bash
./qwtss-cli benchmark [options]
```
*Outputs:* Table of benchmark metrics (computational latency, peak RAM usage, payload file sizes).

## Benchmark Results  
The following benchmarks were generated over 500 iterations in `FAST` thermodynamic mode on a system equipped with an NVIDIA RTX 3090 and an x86-64 Intel Core i9-12900K. Note that the MCMC key generation solver is highly parallelizable, making it an ideal candidate for GPU acceleration. Signing (STARK Proving) and verification are executed on the CPU only.

### Compute & Memory Performance

| Phase | Target Device | Average Time (± Std Dev) | 99th Percentile | Peak RAM (Isolated) |
| :--- | :--- | :--- | :--- | :--- |
| **Keygen** | CPU (70% Threads) | 2101.86 ms (± 2342.67) | 12171.92 ms | 0.47 MB |
| **Keygen** | GPU (CUDA) | 432.45 ms (± 367.60) | 1925.45 ms | 93.56 MB <sup>&dagger;</sup> |
| **Sign** | CPU | 2024.26 ms (± 282.44) | 3004.03 ms | 74.32 MB |
| **Verify** | CPU | 69.84 ms (± 6.33) | 93.16 ms | 20.03 MB |

<sup>&dagger;</sup> RAM usage reflects the host-side CUDA context overhead. The VRAM footprint is ∼0.31 MB.

### Data Footprint

| Cryptographic Artifact | Encoding / Format | Average Size | Max Size |
| :--- | :--- | :--- | :--- |
| **Public Key** | JSON | 0.28 KB | 0.28 KB |
| **Secret Key** | JSON | 4.30 KB | 4.30 KB |
| **Signature** | Raw Binary | 79.68 KB | 81.76 KB |
| **Signature** | Base64 JSON | 106.46 KB | 109.23 KB |

## Running the Test Suite  
To ensure the integrity of the AIR constraints and underlying field arithmetic, this repository includes a comprehensive, standalone testing binary. The test battery includes 77 unique, targeted sabotage attacks on the AIR constraints, as well as one baseline acceptance test as the control. Reviewers are highly encouraged to execute this test suite locally to verify the zk-STARK soundness.

After successfully building the project, execute the testing binary directly from your build directory:
```bash
./qwtss-tests sabotage-suite
```

The console output after a successful run should look like this:
```text
[...]

AIR SABOTAGE SUITE RESULTS
======================================================================================
Validator: Good Accepts: 1, Good Rejects: 77
Verifier: Good Accepts: 1, Good Rejects: 77
Passed tests: 156 / 156

Elapsed time: 6 minutes and 3 seconds
*** ALL TESTS PASSED ***
``` 

## Stone Prover Integration  
Building this project requires integrating the StarkWare Stone Prover, which utilizes the Bazel build system. Because linking Bazel targets against standard C++ compilation pipelines can be complex, we recommend following these steps.

**1. Building the Stone Prover**  
Clone and build the StarkWare Stone Prover locally to generate :
```bash
git clone https://github.com/starkware-libs/stone-prover.git Stone-Prover/stone-prover
cd Stone-Prover/stone-prover
```

Set the Bazel Version to 7.4.0 to avoid certain complex build issues. In the root of the `stone-prover` directory, overwrite your `.bazelversion` file to request the final, highly stable release of the Bazel 7 architecture:
```bash
echo "7.4.0" > .bazelversion
```

Generate the required static libraries for the **prover**:
```bash
bazel query 'kind("cc_library", deps(//src/starkware/main/cpu:cpu_air_prover)) intersect //src/...' | xargs bazel build -c opt --copt="-Isrc" --cxxopt="-std=c++17" --copt="-maes" --copt="-mpclmul"
```

Also generate the required static libraries for the **verifier**:
```bash
bazel query 'kind("cc_library", deps(//src/starkware/main/cpu:cpu_air_verifier)) intersect //src/...' | xargs bazel build -c opt --copt="-Isrc" --cxxopt="-std=c++17" --copt="-maes" --copt="-mpclmul"
```

**2. Linking Stone Prover in QWTSS**  
When compiling QWTSS, you must explicitly link against the compiled Bazel binaries. The provided `CMakeLists.txt` fully automates the Stone Prover integration. Ensure the Stone Prover repository is cloned to `../Stone-Prover/stone-prover` relative to this repository (or update the `STONE_PROVER_DIR` path in `CMakeLists.txt`).

The CMake configuration is designed to:
- Inject the `NDEBUG` definition required by Stone Prover headers to allow default constructors.
- Dynamically query `bazel info bazel-bin` to locate the compiled static libraries.
- Wrap the discovered libraries in GNU linker group flags (`-Wl,--start-group`) to automatically resolve circular dependencies.

(Note: Adjust the exact build flags based on your specific linker configuration and architecture).

## Security Disclaimer  
This repository contains an experimental reference implementation accompanying an academic paper. It has not undergone formal security auditing and should **NOT** be used in production environments to protect sensitive data.

## License  
This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.
