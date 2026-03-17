#include <iostream>
#include <fstream>
#include <string>
#include <unordered_map>
#include <vector>
#include <random>
#include <numeric>
#include <locale>
#include <codecvt>
#include <cctype>

// Set your N-gram context length here
// For N=6, the context length is 5 (it looks at 5 letters to guess the 6th).
const size_t CONTEXT_LEN = 4; 

const std::vector<std::string> alphabet = {
    "a","b","c","d","e","f","g","h","i","j","k","l","m",
    "n","o","p","q","r","s","t","u","v","w","x","y","z",
    " ",".","!",",","?",";",":",
    "à","è","é","ì","ò","ù","_","\"","\'"
};



std::unordered_map<std::string, int> char_to_index;
std::vector<std::string> index_to_char;

using ProbVector = std::vector<double>;
using NgramMap = std::unordered_map<std::string, ProbVector>;

void init_alphabet() {
    for (size_t i = 0; i < alphabet.size(); ++i) {
        char_to_index[alphabet[i]] = i;
        index_to_char.push_back(alphabet[i]);
    }
}

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

// Train N-gram model
void train(NgramMap& ngram_distr, const std::string& text) {
    auto chars = utf8_split(text);
    
    // --- 1. DYNAMIC ALPHABET BUILDER ---
    for (const auto& c : chars) {
        // If we haven't seen this character before, add it to our maps!
        if (char_to_index.find(c) == char_to_index.end()) {
            char_to_index[c] = index_to_char.size();
            index_to_char.push_back(c);
        }
    }
    size_t A = index_to_char.size();

    // Start from index 1, looking backwards to build the context
    for (size_t i = 1; i < chars.size(); ++i) {
        auto it_next = char_to_index.find(chars[i]);
        if (it_next == char_to_index.end()) continue; // skip invalid characters

        std::string key = "";
        
        // Look backwards to store 1-grams, 2-grams, up to CONTEXT_LEN-grams
        for (int j = i - 1; j >= 0 && (i - j) <= CONTEXT_LEN; --j) {
            key = chars[j] + key; // prepend character to the key
            
            if (ngram_distr.find(key) == ngram_distr.end())
                ngram_distr[key] = ProbVector(A, 0.0);
                
            ngram_distr[key][it_next->second] += 1.0;
        }
    }

    // Normalize probabilities
    for (auto& [key, vec] : ngram_distr) {
        double sum = std::accumulate(vec.begin(), vec.end(), 0.0);
        if (sum > 0)
            for (auto& p : vec) p /= sum;
    }
}

// Generate text from N-gram model
std::string generate(const NgramMap& ngram_distr, size_t length) {
    if (length < 1 || ngram_distr.empty()) return "";

    std::random_device rd;
    std::mt19937 gen(rd());

    // Start with a single random character from the alphabet
    std::string c1 = index_to_char[std::uniform_int_distribution<int>(0, alphabet.size() - 1)(gen)];
    std::string result = c1;
    std::string current_context = c1;

    // YOUR NOISE LEVER: Adjust this to make the text more or less crazy!
    // 0.0 = strict memorization.
    double NOISE_LEVEL = 0.01;
    
    //noise attenuation
    NOISE_LEVEL = NOISE_LEVEL*NOISE_LEVEL;
        
    // 1. Fill with evenly distributed noise based on the slider
    double base_noise = NOISE_LEVEL / alphabet.size();
    ProbVector combined_probs(alphabet.size(), base_noise);

    for (size_t i = 1; i < length; ++i) {
        // 1. ADDITIVE SMOOTHING: Start with a baseline probability for EVERY character
        ProbVector combined_probs(alphabet.size(), NOISE_LEVEL); 
        
        std::string search_key = current_context;

        // 2. THE ELASTIC BACKOFF: Try to find the longest matching context
        while (!search_key.empty()) {
            auto it = ngram_distr.find(search_key);
            
            if (it != ngram_distr.end()) {
                // We found a match! Add the trained probabilities on top of the noise
                for (size_t k = 0; k < alphabet.size(); ++k) {
                    // Give trained data higher weight so it overrides the noise
                    combined_probs[k] += (it->second[k] * (1.0 - std::log(NOISE_LEVEL+1))); 
                }
                break; // Stop shrinking the context, we found our anchor!
            }
            // If unseen, shrink the search key by dropping the oldest UTF-8 character
            search_key = search_key.substr(utf8_char_len(search_key, 0));
        }

        // 3. Pick the next character using our combined noise + trained data
        std::discrete_distribution<int> dist(combined_probs.begin(), combined_probs.end());
        std::string next_char = index_to_char[dist(gen)];
        
        result += next_char;
        current_context += next_char;

        // 4. Keep the running context from growing larger than CONTEXT_LEN
        // (If the number of bytes is larger than expected, we know we have to drop characters)
        auto ctx_chars = utf8_split(current_context);
        if (ctx_chars.size() > CONTEXT_LEN) {
            current_context = current_context.substr(utf8_char_len(current_context, 0));
        }
    }

    return result;
}

int main(int argc, char const *argv[])
{
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <text_file>" << std::endl;
        return 1;
    }
    init_alphabet();

    std::ifstream file(argv[1]);
    if (!file) {
        std::cerr << "Failed to open file." << std::endl;
        return 1;
    }

    std::string text;
    std::string line;
    while (std::getline(file, line)) {
        for (char &c : line) c = std::tolower(static_cast<unsigned char>(c));
        text += line + " "; 
    }

    NgramMap ngram_distr;
    train(ngram_distr, text);

    std::string rtxt = generate(ngram_distr, 2000);
    std::cout << rtxt << std::endl;

    return 0;
}