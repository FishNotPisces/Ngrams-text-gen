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

// Set your N-gram context length here!
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
    size_t A = alphabet.size();

    if (chars.size() <= CONTEXT_LEN) return;

    for (size_t i = CONTEXT_LEN; i < chars.size(); ++i) {
        // Check if the target character is in our alphabet
        auto it_next = char_to_index.find(chars[i]);
        if (it_next == char_to_index.end()) continue;

        // Build the dynamic key (e.g., last 5 characters)
        bool valid = true;
        std::string key = "";
        for (size_t j = i - CONTEXT_LEN; j < i; ++j) {
            if (char_to_index.find(chars[j]) == char_to_index.end()) {
                valid = false;
                break;
            }
            key += chars[j];
        }

        // If the whole sequence is valid, record it!
        if (valid) {
            if (ngram_distr.find(key) == ngram_distr.end())
                ngram_distr[key] = ProbVector(A, 0.0);
            ngram_distr[key][it_next->second] += 1.0;
        }
    }

    for (auto& [key, vec] : ngram_distr) {
        double sum = std::accumulate(vec.begin(), vec.end(), 0.0);
        if (sum > 0)
            for (auto& p : vec) p /= sum;
    }
}

// Generate text from N-gram model
std::string generate(const NgramMap& ngram_distr, size_t length) {
    if (length <= CONTEXT_LEN || ngram_distr.empty()) return "";

    std::random_device rd;
    std::mt19937 gen(rd());

    // --- MATRIX SLICE IDEA (Fully Dynamic) ---
    std::string c1 = index_to_char[std::uniform_int_distribution<int>(0, alphabet.size() - 1)(gen)];
    
    std::vector<std::string> matching_keys;
    for (const auto& [k, v] : ngram_distr) {
        if (k.substr(0, c1.size()) == c1) {
            matching_keys.push_back(k);
        }
    }
    
    std::string key = matching_keys.empty() ? ngram_distr.begin()->first 
                    : matching_keys[std::uniform_int_distribution<size_t>(0, matching_keys.size() - 1)(gen)];
    
    std::string result = key;

    for (size_t i = CONTEXT_LEN; i < length; ++i) {
        auto it = ngram_distr.find(key);
        
        if (it == ngram_distr.end()) {
            auto random_it = std::next(ngram_distr.begin(), std::uniform_int_distribution<size_t>(0, ngram_distr.size() - 1)(gen));
            key = random_it->first;
            it = ngram_distr.find(key); 
        }

        std::discrete_distribution<int> dist(it->second.begin(), it->second.end());
        int next_idx = dist(gen);

        std::string next_char = index_to_char[next_idx];
        result += next_char;

        // Our earlier bug fix handles shifting the N-length key perfectly!
        key = key.substr(utf8_char_len(key, 0)) + next_char;
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