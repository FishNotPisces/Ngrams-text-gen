#pragma once
#include <string>
#include <unordered_map>
#include <vector>
#include <iostream>
#include <fstream>
#include <random>
#include <numeric>

class NgramEngine {
private:
    size_t CONTEXT_LEN;

    // Model Data
    std::unordered_map<std::string, int> char_to_index;
    std::vector<std::string> index_to_char;
    std::unordered_map<std::string, std::vector<double>> ngram_distr;
    std::unordered_map<std::string, std::vector<std::string>> suffix_map;

    // Internal Math Helpers
    size_t utf8_char_len(const std::string& str, size_t pos) const;
    std::vector<std::string> utf8_split(const std::string& str) const;
    double calculate_similarity(const std::vector<std::string>& query, const std::vector<std::string>& key) const;

    // Binary Serialization Helpers
    void write_string(std::ofstream& out, const std::string& str) const;
    std::string read_string(std::ifstream& in) const;

public:
    NgramEngine(size_t context_length = 8);

    void train(const std::string& text);
    std::string generate(size_t length, double noise_level = 0.1) const;

    bool save_model(const std::string& filename) const;
    bool load_model(const std::string& filename);
};
