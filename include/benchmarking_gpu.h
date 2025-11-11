#include "bench_option.h"
//depend on uctensor, cublas, cuquantum

CTL::CTL_stats benchmark_GPU(int ITER, std::string expr, std::vector<int64_t> sizes, std::vector<std::pair<int,int>> contraciton_path, benchOption option);
std::pair<float, float> benchmark_cuquantum(int ITER, std::string expr, std::vector<int64_t> sizes, std::vector<std::pair<int,int>> contraciton_path);

void allowtf32();
void disallowtf32();

