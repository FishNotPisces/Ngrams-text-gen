#include <iostream>
#include <fstream>
#include <string>
#include <unordered_map>
#include <vector>
#include <random>
#include <numeric>
#include <cctype>

// Set your N-gram context length here
// For N=8, the context length is 8 (it looks at 8 letters to guess the 9th).
const size_t CONTEXT_LEN = 10;

// NOISE LEVER: 0.1 = 10% chance to randomly drop context and pivot
const double NOISE_LEVEL = 0.5;

const int OUTPUT_LENGTH = 2000;

// Dynamic alphabet storage
std::unordered_map<std::string, int> char_to_index;
std::vector<std::string> index_to_char;

using ProbVector = std::vector<double>;
using NgramMap = std::unordered_map<std::string, ProbVector>;

// --- UTF-8 Helper Functions ---
size_t utf8_char_len(const std::string& str, size_t pos) {
    unsigned char c = str[pos];
    if ((c & 0x80) == 0) return 1;
    else if ((c & 0xE0) == 0xC0) return 2;
    else if ((c & 0xF0) == 0xE0) return 3;
    else if ((c & 0xF8) == 0xF0) return 4;
    else return 1; 
}

std::vector<std::string> utf8_split(const std::string& str) {
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

// --- Training Function ---
void train(NgramMap& ngram_distr, const std::string& text) {
    auto chars = utf8_split(text);
    
    // 1. DYNAMIC ALPHABET BUILDER
    for (const auto& c : chars) {
        if (char_to_index.find(c) == char_to_index.end()) {
            char_to_index[c] = index_to_char.size();
            index_to_char.push_back(c);
        }
    }
    size_t A = index_to_char.size();

    // 2. Build the N-gram context maps
    for (size_t i = 1; i < chars.size(); ++i) {
        auto it_next = char_to_index.find(chars[i]);
        if (it_next == char_to_index.end()) continue;

        std::string key = "";
        
        for (int j = i - 1; j >= 0 && (i - j) <= CONTEXT_LEN; --j) {
            key = chars[j] + key; 
            
            if (ngram_distr.find(key) == ngram_distr.end())
                ngram_distr[key] = ProbVector(A, 0.0);
                
            ngram_distr[key][it_next->second] += 1.0;
        }
    }

    // 3. Normalize probabilities
    for (auto& [key, vec] : ngram_distr) {
        double sum = std::accumulate(vec.begin(), vec.end(), 0.0);
        if (sum > 0)
            for (auto& p : vec) p /= sum;
    }
}

// --- Generation Function ---
std::string generate(const NgramMap& ngram_distr, size_t length) {
    if (length < 1 || ngram_distr.empty()) return "";

    std::random_device rd;
    std::mt19937 gen(rd());
    
    size_t A = index_to_char.size(); // Use dynamic alphabet size

    // Start with a single random character from the alphabet
    std::string c1 = index_to_char[std::uniform_int_distribution<int>(0, A - 1)(gen)];
    std::string result = c1;
    std::string current_context = c1;


    std::uniform_real_distribution<double> chance(0.0, 1.0);

    for (size_t i = 1; i < length; ++i) {
        std::string search_key = current_context;

        // 1. STOCHASTIC TRUNCATION (The ADHD Machine)
        while (!search_key.empty() && chance(gen) < NOISE_LEVEL) {
            search_key = search_key.substr(utf8_char_len(search_key, 0));
        }

        ProbVector combined_probs(A, 0.0);
        bool found_match = false;

        // 2. THE ELASTIC BACKOFF
        while (!search_key.empty()) {
            auto it = ngram_distr.find(search_key);
            
            if (it != ngram_distr.end()) {
                combined_probs = it->second; 
                found_match = true;
                break;
            }
            search_key = search_key.substr(utf8_char_len(search_key, 0));
        }

        // 3. FALLBACK
        if (!found_match) {
            std::uniform_int_distribution<int> random_char(0, A - 1);
            combined_probs[random_char(gen)] = 1.0; 
        }

        // 4. Pick the next character
        std::discrete_distribution<int> dist(combined_probs.begin(), combined_probs.end());
        std::string next_char = index_to_char[dist(gen)];
        
        result += next_char;
        current_context += next_char;

        // 5. Keep the running context from growing larger than CONTEXT_LEN
        auto ctx_chars = utf8_split(current_context);
        if (ctx_chars.size() > CONTEXT_LEN) {
            current_context = current_context.substr(utf8_char_len(current_context, 0));
        }
    }

    return result;
}

// --- Main Program ---
int main(int argc, char const *argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <text_file>" << std::endl;
        return 1;
    }

    std::ifstream file(argv[1]);
    if (!file) {
        std::cerr << "Failed to open file." << std::endl;
        return 1;
    }

    std::string text;
    std::string line;
    
    // Read file and convert to lowercase
    while (std::getline(file, line)) {
        if (line.empty()) continue; 
        for (char &c : line) c = std::tolower(static_cast<unsigned char>(c));
        text += line + " "; 
    }

    // Crush all consecutive spaces/tabs/newlines into a single space
    std::string cleaned_text;
    bool in_space = false;
    for (char c : text) {
        if (std::isspace(static_cast<unsigned char>(c))) {
            if (!in_space) {
                cleaned_text += ' ';
                in_space = true;
            }
        } else {
            cleaned_text += c;
            in_space = false;
        }
    }
    text = cleaned_text;

    // Train and generate
    NgramMap ngram_distr;
    train(ngram_distr, text);

    std::string rtxt = generate(ngram_distr, OUTPUT_LENGTH);
    std::cout << rtxt << std::endl;

    return 0;
}