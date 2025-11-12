#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>
#include <memory>

#include <unordered_map>
#include <filesystem>

#include "reader.h"
#include "esaturation.h"
#include "ETL.h"

using namespace ETL;

void write_csv(const std::string& filename,
               std::vector<float> data) {
    std::ofstream file(filename, std::ios::app);
    if (!file.is_open()) {
        std::cerr << "Failed to open " << filename << "\n";
        return;
    }

    // Write rows
    file << std::fixed << std::setprecision(6);
    for (float t : data)
        file << t << ",";
    file << "\n";

    file.close();
}

void write_stats(const std::string& filename,
               std::vector<std::vector<int>> data) {
    std::ofstream file(filename, std::ios::app);
    if (!file.is_open()) {
        std::cerr << "Failed to open " << filename << "\n";
        return;
    }

    // Write rows
    for (const auto& row : data) {
        for (size_t i = 0; i < row.size(); ++i) {
            file << row[i];
            if (i < row.size() - 1)
                file << ",";
        }
        file << "; ";
    }
    file << "\n";

    file.close();
}



int main(int argc, char* argv[]) {
    std::string filename = "../samples/test.cfg";

    int num_repeats = 10;  // default

    std::string stem; // "test"
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        if (arg == "-f" && i + 1 < argc) {
            filename = argv[++i];
            // Extract just the filename part ("test.cfg")
            std::filesystem::path path(filename);
            std::string base = path.filename().string();

            // Replace extension with "_result.csv"
            stem = path.stem().string(); // "test"
        } else if (arg == "-r" && i + 1 < argc) {
            num_repeats = std::atoi(argv[++i]);
        } else {
            std::cerr << "Unknown or incomplete argument: " << arg << std::endl;
            return 1;
        }
    }

    std::string mean_filename = stem + "_mean_cpu.csv";
    std::string stddev_filename = stem + "_stddev_cpu.csv";
    std::string compile_filename = stem + "_compile_cpu.csv";
    std::string stats_filename = stem + "_stats_cpu.txt";

    if (filename.empty()) {
        std::cerr << "Error: filename must be specified with -f <filename>" << std::endl;
        return 1;
    }

    std::cout << "Filename: " << filename << std::endl;
    std::cout << "Repeats: " << num_repeats << std::endl;


    auto expressions = parseFile(filename);


    //GPU benchmarks
    for(auto exp:expressions){
        std::vector<float> means;
        std::vector<float> stddevs;
        std::vector<float> compiles;
        std::vector<std::vector<int>> nodes_stats; // each entry: [num_perms, num_gemms, num_getts]

        std::cout << "Expr: " << exp.expression << "\nExtents: ";
        for (int x : exp.extents) std::cout << x << " ";
        std::cout << "\nPath: ";
        for (auto& p : exp.path) std::cout << "(" << p.first << "," << p.second << ") ";
        std::cout << "\n";

        auto program = ETL::build_ETL_tree(exp.expression, exp.extents, exp.path, nullptr, ETL::FP32);
        program->print();

        Esat(program);

        std::cout << "Benchmarked: " << exp.expression << "\n\n";
    }

    return 0;
}
