#include "NgramEngine.h"
#include <cmath>      // For std::exp
#include <algorithm>  // For std::min

// Define our alias locally to keep the code clean
using ProbVector = std::vector<double>;

// --- Constructor ---
NgramEngine::NgramEngine(size_t context_length) : CONTEXT_LEN(context_length) {}

// --- Binary Helpers ---
void NgramEngine::write_string(std::ofstream& out, const std::string& str) const {
    size_t len = str.size();
    out.write(reinterpret_cast<const char*>(&len), sizeof(size_t));
    out.write(str.data(), len);
}

std::string NgramEngine::read_string(std::ifstream& in) const {
    size_t len;
    in.read(reinterpret_cast<char*>(&len), sizeof(size_t));
    std::string str(len, '\0');
    in.read(&str[0], len);
    return str;
}

// --- Fast Binary Save ---
bool NgramEngine::save_model(const std::string& filename) const {
    std::ofstream out(filename, std::ios::binary);
    if (!out) return false;

    size_t A = index_to_char.size();
    out.write(reinterpret_cast<const char*>(&A), sizeof(size_t));
    for (const auto& c : index_to_char) write_string(out, c);

    size_t map_size = char_to_index.size();
    out.write(reinterpret_cast<const char*>(&map_size), sizeof(size_t));
    for (const auto& [k, v] : char_to_index) {
        write_string(out, k);
        out.write(reinterpret_cast<const char*>(&v), sizeof(int));
    }

    size_t n_size = ngram_distr.size();
    out.write(reinterpret_cast<const char*>(&n_size), sizeof(size_t));
    for (const auto& [k, vec] : ngram_distr) {
        write_string(out, k);
        out.write(reinterpret_cast<const char*>(vec.data()), vec.size() * sizeof(double));
    }

    size_t s_size = suffix_map.size();
    out.write(reinterpret_cast<const char*>(&s_size), sizeof(size_t));
    for (const auto& [k, vec] : suffix_map) {
        write_string(out, k);
        size_t vec_len = vec.size();
        out.write(reinterpret_cast<const char*>(&vec_len), sizeof(size_t));
        for (const auto& s : vec) write_string(out, s);
    }

    return true;
}

// --- Fast Binary Load ---
bool NgramEngine::load_model(const std::string& filename) {
    std::ifstream in(filename, std::ios::binary);
    if (!in) return false;

    size_t A;
    in.read(reinterpret_cast<char*>(&A), sizeof(size_t));
    index_to_char.resize(A);
    for (size_t i = 0; i < A; ++i) index_to_char[i] = read_string(in);

    size_t map_size;
    in.read(reinterpret_cast<char*>(&map_size), sizeof(size_t));
    char_to_index.reserve(map_size);
    for (size_t i = 0; i < map_size; ++i) {
        std::string k = read_string(in);
        int v;
        in.read(reinterpret_cast<char*>(&v), sizeof(int));
        char_to_index[k] = v;
    }

    size_t n_size;
    in.read(reinterpret_cast<char*>(&n_size), sizeof(size_t));
    ngram_distr.reserve(n_size);
    for (size_t i = 0; i < n_size; ++i) {
        std::string k = read_string(in);
        std::vector<double> vec(A);
        in.read(reinterpret_cast<char*>(vec.data()), A * sizeof(double));
        ngram_distr[k] = vec;
    }

    size_t s_size;
    in.read(reinterpret_cast<char*>(&s_size), sizeof(size_t));
    suffix_map.reserve(s_size);
    for (size_t i = 0; i < s_size; ++i) {
        std::string k = read_string(in);
        size_t vec_len;
        in.read(reinterpret_cast<char*>(&vec_len), sizeof(size_t));
        std::vector<std::string> vec(vec_len);
        for (size_t j = 0; j < vec_len; ++j) vec[j] = read_string(in);
        suffix_map[k] = vec;
    }

    return true;
}

// --- UTF-8 Helper Functions ---
size_t NgramEngine::utf8_char_len(const std::string& str, size_t pos) const {
    unsigned char c = str[pos];
    if ((c & 0x80) == 0) return 1;
    else if ((c & 0xE0) == 0xC0) return 2;
    else if ((c & 0xF0) == 0xE0) return 3;
    else if ((c & 0xF8) == 0xF0) return 4;
    else return 1;
}

std::vector<std::string> NgramEngine::utf8_split(const std::string& str) const {
    std::vector<std::string> chars;
    for (size_t i = 0; i < str.size(); ) {
        unsigned char c = str[i];
        size_t char_len = 1;
        if ((c & 0xF8) == 0xF0) char_len = 4;
        else if ((c & 0xF0) == 0xE0) char_len = 3;
        else if ((c & 0xE0) == 0xC0) char_len = 2;
        chars.push_back(str.substr(i, char_len));
        i += char_len;
    }
    return chars;
}

// --- Similarity Function ---
double NgramEngine::calculate_similarity(const std::vector<std::string>& query, const std::vector<std::string>& key) const {
    double score = 0.0;
    int min_len = std::min(query.size(), key.size());

    for (int i = 1; i <= min_len; ++i) {
        if (query[query.size() - i] == key[key.size() - i]) {
            score += (min_len - i + 1);
        }
    }
    return score;
}

// --- Training Function (OOP Fixed!) ---
void NgramEngine::train(const std::string& text) {
    auto chars = utf8_split(text);

    for (const auto& c : chars) {
        if (char_to_index.find(c) == char_to_index.end()) {
            char_to_index[c] = index_to_char.size();
            index_to_char.push_back(c);
        }
    }
    size_t A = index_to_char.size();

    for (size_t i = 1; i < chars.size(); ++i) {
        auto it_next = char_to_index.find(chars[i]);
        if (it_next == char_to_index.end()) continue;

        std::string key = "";

        for (int j = i - 1; j >= 0 && (i - j) <= CONTEXT_LEN; --j) {
            key = chars[j] + key;

            if (ngram_distr.find(key) == ngram_distr.end()) {
                ngram_distr[key] = ProbVector(A, 0.0);

                auto key_chars = utf8_split(key);
                if (key_chars.size() >= 2) {
                    std::string suffix = key_chars[key_chars.size() - 2] + key_chars[key_chars.size() - 1];
                    suffix_map[suffix].push_back(key);
                }
            }
            ngram_distr[key][it_next->second] += 1.0;
        }
    }

    for (auto& [key, vec] : ngram_distr) {
        double sum = std::accumulate(vec.begin(), vec.end(), 0.0);
        if (sum > 0)
            for (auto& p : vec) p /= sum;
    }
}

// --- Generation Function (OOP Fixed!) ---
std::string NgramEngine::generate(size_t length, double noise_level) const {
    if (length < 1 || ngram_distr.empty()) return "";

    std::random_device rd;
    std::mt19937 gen(rd());

    size_t A = index_to_char.size();

    std::string c1 = index_to_char[std::uniform_int_distribution<int>(0, A - 1)(gen)];
    std::string result = c1;
    std::string current_context = c1;

    std::uniform_real_distribution<double> chance(0.0, 1.0);

    for (size_t i = 1; i < length; ++i) {
        std::string search_key = current_context;

        while (!search_key.empty() && chance(gen) < noise_level) {
            search_key = search_key.substr(utf8_char_len(search_key, 0));
        }

        ProbVector combined_probs(A, 0.0);
        bool found_match = false;

        auto ctx_chars = utf8_split(search_key);

        if (ctx_chars.size() >= 2) {
            std::string suffix = ctx_chars[ctx_chars.size() - 2] + ctx_chars[ctx_chars.size() - 1];

            if (suffix_map.find(suffix) != suffix_map.end()) {
                const auto& candidates = suffix_map.at(suffix);
                double total_weight = 0.0;

                for (const std::string& candidate_key : candidates) {
                    auto cand_chars = utf8_split(candidate_key);
                    double score = calculate_similarity(ctx_chars, cand_chars);

                    if (score > 2.0) {
                        double weight = std::exp(score);
                        total_weight += weight;

                        const ProbVector& candidate_probs = ngram_distr.at(candidate_key);
                        for (size_t k = 0; k < A; ++k) {
                            combined_probs[k] += (candidate_probs[k] * weight);
                        }
                    }
                }

                if (total_weight > 0.0) {
                    for (size_t k = 0; k < A; ++k) {
                        combined_probs[k] /= total_weight;
                    }
                    found_match = true;
                }
            }
        }

        if (!found_match) {
            while (!search_key.empty()) {
                auto it = ngram_distr.find(search_key);

                if (it != ngram_distr.end()) {
                    combined_probs = it->second;
                    found_match = true;
                    break;
                }
                search_key = search_key.substr(utf8_char_len(search_key, 0));
            }
        }

        if (!found_match) {
            std::uniform_int_distribution<int> random_char(0, A - 1);
            combined_probs[random_char(gen)] = 1.0;
        }

        std::discrete_distribution<int> dist(combined_probs.begin(), combined_probs.end());
        std::string next_char = index_to_char[dist(gen)];

        result += next_char;
        current_context += next_char;

        auto ctx_chars_new = utf8_split(current_context);
        if (ctx_chars_new.size() > CONTEXT_LEN) {
            current_context = current_context.substr(utf8_char_len(current_context, 0));
        }
    }

    return result;
}
