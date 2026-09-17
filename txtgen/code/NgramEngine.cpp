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

    return true;
}

// --- Training Function ---
void NgramEngine::observe_text(const std::vector<std::string>& tokens) {
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
    }

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
            key = tokens[j] + key; // Concatenates perfectly!

            if (ngram_distr.find(key) == ngram_distr.end()) {
                ngram_distr[key] = std::unordered_map<int, double>();
            }
            // Record the raw interaction count
            ngram_distr[key][it_next->second] += 1.0;
        }
    }
}

void NgramEngine::compile_probabilities() {
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


// --- Generation Function (OOP Fixed!) ---
// --- The Modular Generation Function ---
std::string NgramEngine::generate(const GenerationParams& params) const {
    if (params.length < 1 || ngram_distr.empty()) return "";

    double temperature = params.temperature; // Tweak this! Higher = more creative, lower = more repetitive
    int K = params.K;

    size_t CACHE_SIZE = 100;    // Remember the last 100 tokens
    double CACHE_WEIGHT = 0.15; // 15% of the decision comes from the cache, 85% from the N-gram

    std::deque<int> cache_window;
    std::unordered_map<int, double> cache_counts;

    std::random_device rd;
    std::mt19937 gen(rd());

    int start_id = std::uniform_int_distribution<int>(0, id_to_token.size() - 1)(gen);
    std::string result = id_to_token[start_id];
    
    // THE FIX: Context is now managed as an array of whole tokens
    std::vector<std::string> context_tokens;
    context_tokens.push_back(result);

    for (size_t i = 1; i < params.length; ++i) {
        
        // We use a temporary copy of our context array for the back-off loop
        std::vector<std::string> temp_context = context_tokens;


        std::unordered_map<int, double> combined_probs;

        // 3. ELASTIC BACKOFF
        while (!temp_context.empty()) {
            std::string current_key = "";
            for (const auto& t : temp_context) current_key += t;

            auto it = ngram_distr.find(current_key);
            if (it != ngram_distr.end() && !it->second.empty()) {
                combined_probs = it->second;
                break;
            }
            // Drop the oldest token to back-off safely
            temp_context.erase(temp_context.begin()); 
        }

        // 4. FALLBACK
        if (combined_probs.empty()) {
            std::uniform_int_distribution<int> random_token(0, id_to_token.size() - 1);
            combined_probs[random_token(gen)] = 1.0;
        } // after this we assume !combined_probs.empty()

        // 4.5. CACHE BOOST
        if (!cache_window.empty()) {
            double current_cache_size = static_cast<double>(cache_window.size());

            for (auto& [token_id, prob] : combined_probs) {
                double cache_prob = 0.0;
                
                // If this legal next-word is also in our recent memory, calculate its cache frequency
                if (cache_counts.find(token_id) != cache_counts.end()) {
                    cache_prob = cache_counts[token_id] / current_cache_size;
                }
                
                // Linear Interpolation: Blend the two minds together
                prob = ((1.0 - CACHE_WEIGHT) * prob) + (CACHE_WEIGHT * cache_prob);
            }
        }

        // 5. ROULETTE WHEEL (WITH TOP-K AND TEMPERATURE)
        std::string next_token = "";
        std::vector<int> candidates;
        std::vector<double> weights;

        // 1. Move map to a vector of pairs so we can sort them by probability
        std::vector<std::pair<int, double>> sorted_words(combined_probs.begin(), combined_probs.end());
        
        // 2. Sort descending (highest probability first)
        std::sort(sorted_words.begin(), sorted_words.end(), 
                [](const auto& a, const auto& b) { return a.second > b.second; });
                
        // 3. TOP-K FILTER: The Guillotine! Keep only the top K choices.
        if (sorted_words.size() > K) {
            sorted_words.resize(K); 
        }

        // 4. Build your candidates and apply Temperature Math
        for (const auto& pair : sorted_words) {
            candidates.push_back(pair.first);
            double adjusted_weight = std::pow(pair.second, 1.0 / temperature);
            weights.push_back(adjusted_weight);
        }
        
        // 5. Roll the dice safely among the survivors
        std::discrete_distribution<int> dist(weights.begin(), weights.end());
        int next_token_id = candidates[dist(gen)];
        next_token = id_to_token[next_token_id];

        // Append to string ONCE, push to memory ONCE.
        result += next_token;
        context_tokens.push_back(next_token);

        // UPDATE CACHE
        auto cache_it = token_to_id.find(next_token);
        if (cache_it != token_to_id.end()) {
            int final_token_id = cache_it->second;
            
            cache_window.push_back(final_token_id);
            cache_counts[final_token_id] += 1.0;

            // If the cache gets too big, forget the oldest word
            if (cache_window.size() > CACHE_SIZE) {
                int oldest_id = cache_window.front();
                cache_window.pop_front();
                
                cache_counts[oldest_id] -= 1.0;
                if (cache_counts[oldest_id] <= 0.0) {
                    cache_counts.erase(oldest_id); // Keep the map clean and fast
                }
            }
        }

        // 6. KEEP CONTEXT WITHIN LIMITS (Limit by token count, not character count!)
        if (context_tokens.size() > CONTEXT_LEN) {
            context_tokens.erase(context_tokens.begin());
        }
    }

    return result;
}
