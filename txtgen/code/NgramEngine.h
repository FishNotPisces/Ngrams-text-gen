#pragma once
#include <string>
#include <unordered_map>
#include <vector>
#include <iostream>
#include <fstream>
#include <random>
#include <numeric>

struct GenerationParams {
    size_t length = 2000;
    double noise_level = 0.0;
    bool use_attention = false;      // Toggle Attention ON/OFF
    double attention_threshold = 2.0; 
};

struct TrainingParams {
    bool enable_pruning = true;
    // Frequencies represented as decimals (e.g., 0.01 = 1% of the total text)
    double prune_low_freq = 0.00005; // 0.005% 
    double prune_high_freq = 0.01;   // 1.0%
};

class NgramEngine {
private:
    size_t CONTEXT_LEN;

    // Model Data
    std::unordered_map<std::string, int> token_to_id;
    std::vector<std::string> id_to_token;
    std::unordered_map<std::string, std::unordered_map<int, double>> ngram_distr;
    std::unordered_map<std::string, std::vector<std::string>> suffix_map;
    std::unordered_map<int, int> global_token_counts;

    // Internal Math Helpers
    size_t utf8_char_len(const std::string& str, size_t pos) const;
    std::vector<std::string> split_tokens(const std::string& str) const;
    double calculate_similarity(const std::vector<std::string>& query, const std::vector<std::string>& key) const;
    // Private Helper for the modular generation pipeline
    bool apply_attention(const std::string& search_key, 
                         std::unordered_map<int, double>& combined_probs, 
                         double threshold) const;
    // Pruning heuristic
    size_t prune_graph(double low_freq, double high_freq, size_t total_tokens);

    // Binary Serialization Helpers
    void write_string(std::ofstream& out, const std::string& str) const;
    std::string read_string(std::ifstream& in) const;

public:
    NgramEngine(size_t context_length = 8);

    void train(const std::string& text, const TrainingParams& params = TrainingParams());
    std::string generate(const GenerationParams& params) const;

    bool save_model(const std::string& filename) const;
    bool load_model(const std::string& filename);
};
