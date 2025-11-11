#ifndef BENCH_OPTION
#define BENCH_OPTION
#include <utility>
#include <cmath>
#include <vector>

enum benchOption
{
    expandNmodelFuse,
    expandNprofiledFuse,
    expandOnly,
    noOpt
};
std::pair<float, float> processing_stats(std::vector<float> timings)
{
    // ---- Compute mean and variance ----
    float sum = 0.0f;
    for (float t : timings)
        sum += t;

    float mean = sum / timings.size();

    // Variance
    float variance = 0.0f;
    for (float t : timings)
        variance += (t - mean) * (t - mean);

    variance /= timings.size();         // population variance
    float stddev = std::sqrt(variance); // standard deviation
    return {mean, stddev};
}
#endif