#pragma once
#include <string>
#include <unordered_map>
#include <vector>
#include <iostream>
#include <fstream>
#include <random>
#include <numeric>
#include <deque>

struct GenerationParams {
    size_t length = 2000;
    double temperature = 1.5; // Tweak this! Higher = more creative, lower = more repetitive
    int K = 5;
};


class NgramEngine {
private:
    size_t CONTEXT_LEN;

    // Model Data
    std::unordered_map<std::string, int> token_to_id;
    std::vector<std::string> id_to_token;
    std::unordered_map<std::string, std::unordered_map<int, double>> ngram_distr;

    // Internal Math Helpers

    // Binary Serialization Helpers
    void write_string(std::ofstream& out, const std::string& str) const;
    std::string read_string(std::ifstream& in) const;

public:
    NgramEngine(size_t context_length = 4);

    void observe_text(const std::vector<std::string>& tokens);
    void compile_probabilities();
    std::string generate(const GenerationParams& params) const;

    bool save_model(const std::string& filename) const;
    bool load_model(const std::string& filename);
};
