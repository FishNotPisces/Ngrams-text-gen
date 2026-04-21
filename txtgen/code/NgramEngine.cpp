#include "NgramEngine.h"
#include <cmath>      // For std::exp
#include <algorithm>  // For std::min


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

    size_t A = id_to_token.size();
    out.write(reinterpret_cast<const char*>(&A), sizeof(size_t));
    for (const auto& c : id_to_token) write_string(out, c);

    size_t map_size = token_to_id.size();
    out.write(reinterpret_cast<const char*>(&map_size), sizeof(size_t));
    for (const auto& [k, v] : token_to_id) {
        write_string(out, k);
        out.write(reinterpret_cast<const char*>(&v), sizeof(int));
    }

// 3. Save Ngram Distribution (Sparse Version)
    size_t n_size = ngram_distr.size();
    out.write(reinterpret_cast<const char*>(&n_size), sizeof(size_t));
    for (const auto& [k, sparse_map] : ngram_distr) {
        write_string(out, k);
        
        // Save how many tokens are in this sparse map
        size_t sparse_size = sparse_map.size();
        out.write(reinterpret_cast<const char*>(&sparse_size), sizeof(size_t));
        
        // Save each ID and Probability pair
        for (const auto& [token_id, prob] : sparse_map) {
            out.write(reinterpret_cast<const char*>(&token_id), sizeof(int));
            out.write(reinterpret_cast<const char*>(&prob), sizeof(double));
        }
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
    id_to_token.resize(A);
    for (size_t i = 0; i < A; ++i) id_to_token[i] = read_string(in);

    size_t map_size;
    in.read(reinterpret_cast<char*>(&map_size), sizeof(size_t));
    token_to_id.reserve(map_size);
    for (size_t i = 0; i < map_size; ++i) {
        std::string k = read_string(in);
        int v;
        in.read(reinterpret_cast<char*>(&v), sizeof(int));
        token_to_id[k] = v;
    }

// 3. Load Ngram Distribution (Sparse Version)
    size_t n_size;
    in.read(reinterpret_cast<char*>(&n_size), sizeof(size_t));
    ngram_distr.reserve(n_size); 
    for (size_t i = 0; i < n_size; ++i) {
        std::string k = read_string(in);
        
        size_t sparse_size;
        in.read(reinterpret_cast<char*>(&sparse_size), sizeof(size_t));
        
        std::unordered_map<int, double> sparse_map;
        sparse_map.reserve(sparse_size);
        
        for (size_t j = 0; j < sparse_size; ++j) {
            int token_id;
            double prob;
            in.read(reinterpret_cast<char*>(&token_id), sizeof(int));
            in.read(reinterpret_cast<char*>(&prob), sizeof(double));
            sparse_map[token_id] = prob;
        }
        ngram_distr[k] = sparse_map;
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

std::vector<std::string> NgramEngine::split_tokens(const std::string& str) const {
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

size_t NgramEngine::prune_graph(double low_freq, double high_freq, size_t total_tokens) {
    size_t deleted_edges = 0;

    for (auto& [context_key, sparse_map] : ngram_distr) {
        
        auto ctx_tokens = split_tokens(context_key);
        if (ctx_tokens.empty()) continue;
        
        std::string last_token_str = ctx_tokens.back();
        int token_A_id = token_to_id[last_token_str];
        
        // Calculate the relative frequency of Token A
        double freq_A = static_cast<double>(global_token_counts[token_A_id]) / total_tokens;

        for (auto it = sparse_map.begin(); it != sparse_map.end(); ) {
            int token_B_id = it->first;
            double local_interactions = it->second;

            if (local_interactions == 1.0) {
                // Calculate the relative frequency of Token B
                double freq_B = static_cast<double>(global_token_counts[token_B_id]) / total_tokens;

                // Compare frequencies against your thresholds
                bool is_low_low = (freq_A <= low_freq && freq_B <= low_freq);
                bool is_high_high = (freq_A >= high_freq && freq_B >= high_freq);

                if (is_low_low || is_high_high) {
                    it = sparse_map.erase(it);
                    deleted_edges++;
                } else {
                    ++it;
                }
            } else {
                ++it;
            }
        }
    }
    return deleted_edges;
}

// --- Training Function (OOP Fixed!) ---
void NgramEngine::train(const std::string& text, const TrainingParams& params) {
    auto tokens = split_tokens(text);

    // =====================================================================
    // CYCLE 1: VOCABULARY BUILDING
    // Iterates through the raw data once to build the global dictionary.
    // Every unique token (word or character) is assigned a persistent integer ID.
    // =====================================================================
    for (const auto& token : tokens) {
        if (token_to_id.find(token) == token_to_id.end()) {
            token_to_id[token] = id_to_token.size();
            id_to_token.push_back(token);
        }

        int current_token_id = token_to_id[token];
        global_token_counts[current_token_id]++;
    }
    
    // (Optional) Stores the total number of unique tokens in the vocabulary
    size_t vocab_size = id_to_token.size(); 

    // =====================================================================
    // CYCLE 2: MATRIX CONSTRUCTION & SUFFIX INDEXING
    // Slides through the tokens to observe the actual sequential transitions.
    // For every target token, it looks backwards to build varying lengths of context.
    // =====================================================================
    for (size_t i = 1; i < tokens.size(); ++i) {
        auto it_next = token_to_id.find(tokens[i]);
        if (it_next == token_to_id.end()) continue;

        std::string key = "";

        // --- INNER CYCLE: CONTEXT EXPANSION ---
        // Expands the context backwards from length 1 up to CONTEXT_LEN.
        // This populates the N-gram distribution and the Attention suffix map.
        for (int j = i - 1; j >= 0 && (i - j) <= CONTEXT_LEN; --j) {
            key = tokens[j] + key;

            if (ngram_distr.find(key) == ngram_distr.end()) {
                ngram_distr[key] = std::unordered_map<int, double>();

                auto key_tokens = split_tokens(key);
                if (key_tokens.size() >= 2) {
                    // The suffix index relies on the last two tokens of the context
                    std::string suffix = key_tokens[key_tokens.size() - 2] + key_tokens[key_tokens.size() - 1];
                    suffix_map[suffix].push_back(key);
                }
            }
            // Record the raw interaction count (Token ID -> Count)
            ngram_distr[key][it_next->second] += 1.0;
        }
    }

    //----------
    // Pruning
    //----------
    if (params.enable_pruning) {
        size_t total_tokens = tokens.size();
        size_t trimmed = prune_graph(params.prune_low_freq, params.prune_high_freq, total_tokens);
        std::cout << "[Training] Pruned " << trimmed << " redundant or noisy connections.\n";

        // Erase the ghost keys so they aren't saved to disk
        size_t ghosts_removed = 0;
        for (auto it = ngram_distr.begin(); it != ngram_distr.end(); ) {
            if (it->second.empty()) {
                it = ngram_distr.erase(it);
                ghosts_removed++;
            } else {
                ++it;
            }
        }
        std::cout << "[Training] Cleaned up " << ghosts_removed << " empty ghost contexts.\n";
    }

    // =====================================================================
    // CYCLE 3: PROBABILITY NORMALIZATION
    // Converts the raw local interaction counts into percentages (0.0 to 1.0).
    // This prepares the sparse maps for probabilistic sampling during generation.
    // =====================================================================
    for (auto& [key, sparse_map] : ngram_distr) {
        double sum = 0.0;
        
        // Step A: Add up all the raw counts in this specific sparse map
        for (const auto& [token_id, count] : sparse_map) {
            sum += count;
        }
        
        // Step B: Divide the count by the sum to get the percentage (0.0 to 1.0)
        if (sum > 0.0) {
            for (auto& [token_id, count] : sparse_map) {
                count /= sum; // We are explicitly dividing the 'double', ignoring the 'int'
            }
        }
    }
}

// --- Modular Attention Helper ---
bool NgramEngine::apply_attention(const std::string& search_key, std::unordered_map<int, double>& combined_probs, double threshold) const {
    auto ctx_tokens = split_tokens(search_key);
    
    // We need at least 2 tokens to form our suffix index
    if (ctx_tokens.size() < 2) return false;

    // Construct the suffix from the last two tokens
    std::string suffix = ctx_tokens[ctx_tokens.size() - 2] + ctx_tokens[ctx_tokens.size() - 1];

    // Check if we have any candidates that share this suffix
    auto it = suffix_map.find(suffix);
    if (it == suffix_map.end()) return false;

    double total_weight = 0.0;
    const auto& candidates = it->second;

    for (const std::string& candidate_key : candidates) {
        auto cand_tokens = split_tokens(candidate_key);
        
        // Score similarity right-to-left
        double score = calculate_similarity(ctx_tokens, cand_tokens);

        if (score > threshold) {
            double weight = std::exp(score); // Exponentiate so best matches dominate
            total_weight += weight;

            // SPARSE BLENDING: Safely check if the candidate survived pruning and blend its probabilities into the combined map
            auto dist_it = ngram_distr.find(candidate_key);
            if (dist_it != ngram_distr.end()) {
                for (const auto& [token_id, prob] : dist_it->second) {
                    combined_probs[token_id] += (prob * weight);
                }
            }
        }
    }

    // Normalize the sparse map into percentages (0.0 to 1.0)
    if (total_weight > 0.0) {
        for (auto& [token_id, prob] : combined_probs) {
            prob /= total_weight;
        }
        return true; // Success! We successfully blended an attention vector.
    }

    return false; // Failed to pass the threshold score.
}

// --- Generation Function (OOP Fixed!) ---
// --- The Modular Generation Function ---
std::string NgramEngine::generate(const GenerationParams& params) const {
    if (params.length < 1 || ngram_distr.empty()) return "";

    std::random_device rd;
    std::mt19937 gen(rd());

    // Start with a random token
    int start_id = std::uniform_int_distribution<int>(0, id_to_token.size() - 1)(gen);
    std::string result = id_to_token[start_id];
    std::string current_context = result;

    std::uniform_real_distribution<double> chance(0.0, 1.0);

    for (size_t i = 1; i < params.length; ++i) {
        std::string search_key = current_context;

        // ---------------------------------------------------------
        // FEATURE 1: STOCHASTIC TRUNCATION (Toggleable via noise_level)
        // ---------------------------------------------------------
        while (!search_key.empty() && chance(gen) < params.noise_level) {
            search_key = search_key.substr(utf8_char_len(search_key, 0));
        }

        // We now use a sparse map to hold our blended probabilities
        std::unordered_map<int, double> combined_probs;
        bool found_match = false;

        // ---------------------------------------------------------
        // FEATURE 2: THE ATTENTION MECHANISM (Toggleable via boolean)
        // ---------------------------------------------------------
        if (params.use_attention) {
            found_match = apply_attention(search_key, combined_probs, params.attention_threshold);
        }

        // ---------------------------------------------------------
        // CORE LOGIC: ELASTIC BACKOFF (Exact Match)
        // ---------------------------------------------------------
        if (!found_match) {
            while (!search_key.empty()) {
                auto it = ngram_distr.find(search_key);
                if (it != ngram_distr.end() && !it->second.empty()) {
                    combined_probs = it->second; // Copy the sparse map
                    found_match = true;
                    break;
                }
                search_key = search_key.substr(utf8_char_len(search_key, 0));
            }
        }

        // ---------------------------------------------------------
        // FALLBACK: Completely Random Token
        // ---------------------------------------------------------
        if (!found_match) {
            std::uniform_int_distribution<int> random_token(0, id_to_token.size() - 1);
            combined_probs[random_token(gen)] = 1.0;
        }

        // ---------------------------------------------------------
        // SPARSE ROULETTE WHEEL
        // ---------------------------------------------------------
        // Separate the sparse map into candidates and weights
        std::vector<int> candidates;
        std::vector<double> weights;
        for (const auto& [token_id, prob] : combined_probs) {
            candidates.push_back(token_id);
            weights.push_back(prob);
        }

        // Roll the dice based on the weights
        std::discrete_distribution<int> dist(weights.begin(), weights.end());
        int selected_index = dist(gen);
        int next_token_id = candidates[selected_index];
        std::string next_token = id_to_token[next_token_id];

        result += next_token;
        current_context += next_token;

        // Keep context within limits
        auto ctx_tokens = split_tokens(current_context); // Using your generic token splitter
        if (ctx_tokens.size() > CONTEXT_LEN) {
            current_context = current_context.substr(utf8_char_len(current_context, 0));
        }
    }

    return result;
}
