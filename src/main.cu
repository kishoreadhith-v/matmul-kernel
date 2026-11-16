#include "common.cuh"
#include "harness.cuh"
#include "kernel.cuh"

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

struct Shape {
  int M, N, K;
};

static bool parse_shape(const std::string& text, Shape& out) {
  int m = 0, n = 0, k = 0;
  if (std::sscanf(text.c_str(), "%d,%d,%d", &m, &n, &k) != 3) return false;
  if (m <= 0 || n <= 0 || k <= 0) return false;
  out = {m, n, k};
  return true;
}

static bool parse_sizes(const std::string& text, std::vector<int>& out) {
  std::stringstream ss(text);
  std::string item;
  while (std::getline(ss, item, ',')) {
    try {
      const int value = std::stoi(item);
      if (value <= 0) return false;
      out.push_back(value);
    } catch (...) {
      return false;
    }
  }
  return !out.empty();
}

void launch_cublas(const GemmProblem&, cudaStream_t);
void launch_naive(const GemmProblem&, cudaStream_t);

static const KernelVariant kKernels[] = {
    {"cublas", "cuBLAS SGEMM", launch_cublas, Metric::Gflops},
    {"naive", "block=32x32", launch_naive, Metric::Gflops},
};

static void print_usage(const char* argv0) {
  std::cout
      << "usage: " << argv0 << " [options]\n"
      << "  --list-kernels\n"
      << "  --kernel <name|all> default: all\n"
      << "  --size <n>          one square problem, default: 1024\n"
      << "  --sizes <n,n,...>   several square problems\n"
      << "  --shape <M,N,K>     one rectangular problem\n"
      << "  --reps <n>          timed launches, default: 50\n"
      << "  --warmup <n>        warmup launches, default: 5\n"
      << "  --seed <n>          default: 0\n"
      << "  --alpha <f>         default: 1.0\n"
      << "  --beta <f>          default: 0.0\n"
      << "  --verify-only       verify but do not time\n"
      << "  --test-suite        verify every kernel on the built-in shapes\n"
      << "  --csv <path>        append benchmark rows\n"
      << "  --no-verify         skip the cuBLAS reference\n"
      << "  --help\n";
}

static const Shape kTestShapes[] = {
    {1, 1, 1},       {2, 2, 2},       {3, 3, 3},
    {7, 7, 7},       {16, 16, 16},    {31, 31, 31},
    {32, 32, 32},    {33, 33, 33},    {63, 63, 63},
    {64, 64, 64},    {65, 65, 65},    {127, 127, 127},
    {128, 128, 128},{129, 129, 129},{255, 255, 255},
    {256, 256, 256},{257, 257, 257},{511, 511, 511},
    {512, 512, 512},{1000, 1000, 1000},
    {17, 19, 23},    {128, 256, 64},  {1000, 999, 1001},
};

int main(int argc, char** argv) {
  std::string kernel_name = "all";
  int size = 1024;
  int reps = 50;
  int warmup = 5;
  int seed = 0;
  bool verify = true;
  std::string csv_path;
  bool verify_only = false;
  bool test_suite = false;
  bool size_set = false;
  bool sizes_set = false;
  bool shape_set = false;
  bool beta_set = false;
  float alpha = 1.0f;
  float beta = 0.0f;
  std::vector<int> size_list;
  Shape one_shape{0, 0, 0};

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
      size_set = true;
    } else if (arg == "--sizes") {
      if (!parse_sizes(value("--sizes"), size_list)) {
        std::cerr << "invalid --sizes\n";
        return 1;
      }
      sizes_set = true;
    } else if (arg == "--shape") {
      if (!parse_shape(value("--shape"), one_shape)) {
        std::cerr << "invalid --shape, expected M,N,K\n";
        return 1;
      }
      shape_set = true;
    } else if (arg == "--alpha") {
      alpha = std::stof(value("--alpha"));
    } else if (arg == "--beta") {
      beta = std::stof(value("--beta"));
      beta_set = true;
    } else if (arg == "--verify-only") {
      verify_only = true;
    } else if (arg == "--test-suite") {
      test_suite = true;
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

  if (verify_only && !verify) {
    std::cerr << "--verify-only requires verification\n";
    return 1;
  }

  if (shape_set && (sizes_set || size_set)) {
    std::cerr << "use either --shape or --size/--sizes, not both\n";
    return 1;
  }
  if (sizes_set && size_set) {
    std::cerr << "use either --size or --sizes, not both\n";
    return 1;
  }

  // The test suite uses the nonzero-alpha/beta path; verify-only defaults to
  // exercising beta unless the caller chose it.
  if (test_suite) {
    alpha = 0.5f;
    beta = 0.5f;
  } else if (verify_only && !beta_set) {
    beta = 0.5f;
  }

  // Select kernels before opening the CSV, so an invalid name does not
  // truncate an existing results file.
  std::vector<const KernelVariant*> selected;

  if (kernel_name == "all") {
    for (const auto& k : kKernels) {
      selected.push_back(&k);
    }
  } else {
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
    selected.push_back(chosen);
  }

  if (selected.empty()) {
    std::cerr << "no kernels selected\n";
    return 1;
  }

  std::vector<Shape> shapes;

  if (test_suite) {
    for (const Shape& s : kTestShapes) shapes.push_back(s);
  } else if (shape_set) {
    shapes.push_back(one_shape);
  } else if (sizes_set) {
    for (int s : size_list) shapes.push_back({s, s, s});
  } else {
    shapes.push_back({size, size, size});
  }

  if (!csv_path.empty() && (verify_only || test_suite)) {
    std::cerr << "note: --csv is ignored for verification-only runs\n";
  }

  std::ofstream csv;
  std::ostream* csv_out = nullptr;

  if (!test_suite && !verify_only && !csv_path.empty()) {
    const std::filesystem::path path(csv_path);
    if (path.has_parent_path()) {
      std::filesystem::create_directories(path.parent_path());
    }

    const bool need_header = !std::filesystem::exists(path) ||
                             std::filesystem::file_size(path) == 0;

    csv.open(path, std::ios::out | std::ios::app);
    if (!csv) {
      std::cerr << "cannot open " << csv_path << "\n";
      return 1;
    }

    if (need_header) {
      csv << "schema_version,kernel,config,M,N,K,alpha,beta,warmup,reps,"
             "min_ms,median_ms,mean_ms,std_ms,metric,value,"
             "max_abs_err,max_rel_err,passed\n";
    }
    csv_out = &csv;
  }

  const bool timing = !(verify_only || test_suite);

  auto run = [&](const KernelVariant& kernel, int M, int N, int K, float a,
                 float b) {
    BenchOptions opts;
    opts.reps = reps;
    opts.warmup = warmup;
    opts.seed = seed;
    opts.verify = verify;
    opts.timing = timing;
    opts.alpha = a;
    opts.beta = b;
    return bench_kernel(kernel, M, N, K, opts, csv_out);
  };

  bool all_passed = true;

  for (const Shape& shape : shapes) {
    for (const KernelVariant* kernel : selected) {
      const BenchResult result =
          run(*kernel, shape.M, shape.N, shape.K, alpha, beta);
      all_passed = all_passed && result.passed;
    }
  }

  // The suite above always uses alpha=beta=0.5. Also cover the benchmark path
  // (beta=0, C never read) at a large non-multiple shape.
  if (test_suite) {
    for (const KernelVariant* kernel : selected) {
      const BenchResult result = run(*kernel, 1000, 999, 1001, 1.0f, 0.0f);
      all_passed = all_passed && result.passed;
    }
  }

  if (test_suite) {
    std::cout << (all_passed ? "test suite PASS\n" : "test suite FAIL\n");
  }

  return all_passed ? 0 : 2;
}
