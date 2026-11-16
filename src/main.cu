#include "common.cuh"
#include "harness.cuh"
#include "kernel.cuh"

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>

void launch_naive(const GemmProblem&, cudaStream_t);

static const KernelVariant kKernels[] = {
    {"naive", "block=32x32", launch_naive, Metric::Gflops},
};

static void print_usage(const char* argv0) {
  std::cout
      << "usage: " << argv0 << " [options]\n"
      << "  --list-kernels\n"
      << "  --kernel <name>     default: naive\n"
      << "  --size <n>          square size, default: 1024\n"
      << "  --reps <n>          timed launches, default: 50\n"
      << "  --warmup <n>        warmup launches, default: 5\n"
      << "  --seed <n>          default: 0\n"
      << "  --csv <path>        write a CSV row\n"
      << "  --no-verify         skip the cuBLAS reference\n"
      << "  --help\n";
}

int main(int argc, char** argv) {
  std::string kernel_name = "naive";
  int size = 1024;
  int reps = 50;
  int warmup = 5;
  int seed = 0;
  bool verify = true;
  std::string csv_path;

  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];

    auto value = [&](const char* flag) -> std::string {
      if (i + 1 >= argc) {
        std::cerr << "missing value for " << flag << "\n";
        std::exit(EXIT_FAILURE);
      }
      return argv[++i];
    };

    if (arg == "--help" || arg == "-h") {
      print_usage(argv[0]);
      return 0;
    } else if (arg == "--list-kernels") {
      for (const auto& k : kKernels) {
        const char* metric =
            k.metric == Metric::Gflops ? "gflops"
            : k.metric == Metric::Gbps ? "gbps"
                                       : "none";
        std::cout << k.name << "  [" << k.config << "]  metric=" << metric
                  << "\n";
      }
      return 0;
    } else if (arg == "--kernel") {
      kernel_name = value("--kernel");
    } else if (arg == "--size") {
      size = std::stoi(value("--size"));
    } else if (arg == "--reps") {
      reps = std::stoi(value("--reps"));
    } else if (arg == "--warmup") {
      warmup = std::stoi(value("--warmup"));
    } else if (arg == "--seed") {
      seed = std::stoi(value("--seed"));
    } else if (arg == "--csv") {
      csv_path = value("--csv");
    } else if (arg == "--no-verify") {
      verify = false;
    } else {
      std::cerr << "unknown argument: " << arg << "\n";
      print_usage(argv[0]);
      return 1;
    }
  }

  if (size <= 0 || reps <= 0 || warmup < 0) {
    std::cerr << "size and reps must be positive; warmup must be >= 0\n";
    return 1;
  }

  const KernelVariant* chosen = nullptr;
  for (const auto& k : kKernels) {
    if (kernel_name == k.name) {
      chosen = &k;
      break;
    }
  }

  if (chosen == nullptr) {
    std::cerr << "unknown kernel: " << kernel_name << "\n";
    return 1;
  }

  std::ofstream csv;
  std::ostream* csv_out = nullptr;
  if (!csv_path.empty()) {
    csv.open(csv_path, std::ios::out | std::ios::trunc);
    if (!csv) {
      std::cerr << "cannot open " << csv_path << "\n";
      return 1;
    }
    csv << "schema_version,kernel,config,M,N,K,alpha,beta,warmup,reps,"
           "min_ms,median_ms,mean_ms,std_ms,metric,value,"
           "max_abs_err,max_rel_err,passed\n";
    csv_out = &csv;
  }

  BenchOptions opts;
  opts.reps = reps;
  opts.warmup = warmup;
  opts.seed = seed;
  opts.verify = verify;

  const BenchResult result =
      bench_kernel(*chosen, size, size, size, opts, csv_out);

  return result.passed ? 0 : 2;
}
