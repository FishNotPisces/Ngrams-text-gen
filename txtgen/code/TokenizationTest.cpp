#include <string>
#include <vector>
#include <algorithm>
#include <fstream>
#include <cctype>
#include <iostream>
#include <unordered_map>
#include <memory>
#include <list>

const std::string word_delimiter_symbol = "</w>";

struct PairHash {
    std::size_t operator()(const std::pair<std::string, std::string>& p) const {
        auto h1 = std::hash<std::string>{}(p.first);
        auto h2 = std::hash<std::string>{}(p.second);
        // A standard C++ trick to combine two hashes safely
        return h1 ^ (h2 << 1); 
    }
};

struct VocabWord {
    std::list<std::string> tokens;
    int frequency;
};


// Standard UTF-8 byte-length detector
size_t get_utf8_char_len(unsigned char c) {
    if ((c & 0b10000000) == 0b00000000) return 1; // Standard ASCII (1 byte)
    if ((c & 0b11100000) == 0b11000000) return 2; // Latin/European (2 bytes, like 'è')
    if ((c & 0b11110000) == 0b11100000) return 3; // Asian/Symbols (3 bytes, like '’')
    if ((c & 0b11111000) == 0b11110000) return 4; // Emojis (4 bytes)
    return 1; // Fallback to prevent infinite loops on corrupted text
}

// The Slicer
std::list<std::string> split_to_utf8_list(const std::string& word) {
    std::list<std::string> tokens;
    size_t i = 0;
    
    while (i < word.length()) {
        // Look at the first byte to see how long the character is
        size_t len = get_utf8_char_len(word[i]);
        
        // Extract that exact number of bytes as a single token
        tokens.push_back(word.substr(i, len));
        
        // Jump forward to the start of the next character
        i += len;
    }
    
    return tokens;
}

// THE TALLY FUNCTION
std::unordered_map<std::pair<std::string, std::string>, int, PairHash> 
get_pair_counts(const std::vector<VocabWord>& dictionary) {
    
    std::unordered_map<std::pair<std::string, std::string>, int, PairHash> pair_counts;

    // Cycle 1: Go through every unique word in the dictionary
    for (const auto& word : dictionary) {
        
        // Safety check: if a word is only 1 token long, there are no pairs to count!
        if (word.tokens.size() < 2) continue; 

        // Cycle 2: Setup two pointers (iterators) for the linked list
        auto it = word.tokens.begin();       // Points to the first token
        auto next_it = std::next(it);        // Points to the second token

        // Traverse the list until the right pointer falls off the end
        while (next_it != word.tokens.end()) {
            
            // Grab the actual strings using the '*' operator
            std::pair<std::string, std::string> current_pair = {*it, *next_it};
            
            // Add this word's global frequency to the scoreboard
            pair_counts[current_pair] += word.frequency;
            
            // Shift both pointers to the right by one step
            it++;
            next_it++;
        }
    }

    return pair_counts;
}

void merge_pair_in_dictionary(
    std::vector<VocabWord>& dictionary,
    const std::pair<std::string, std::string>& target)
{
    for (auto& vw : dictionary) {
        auto it = vw.tokens.begin();

        while (it != vw.tokens.end()) {
            auto next = std::next(it);
            if (next == vw.tokens.end())
                break;

            if (*it == target.first && *next == target.second) {
                // merge into a single token
                *it += *next;

                // remove the second element
                vw.tokens.erase(next);
                // stay on same 'it' to allow cascading merges if needed
            } else {
                ++it;
            }
        }
    }
}

void print_dictionary(const std::vector<VocabWord>& dictionary) {
    for (const auto& vw : dictionary) {
        for (const auto& token : vw.tokens) {
            std::cout << token << " ";
        }
        std::cout << "-> " << vw.frequency << '\n';
    }
}

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <input_file>" << std::endl;
        return 1;
    }

    std::string text = argv[1];

    std::ifstream file(text);
    if (!file) {
        std::cerr << "Error: Could not open " << text << std::endl;
        return 1;
    }

    std::unordered_map<std::string, int> words_reader;
    std::vector<VocabWord> dictionary;
    std::unordered_map<std::pair<std::string, std::string>, int, PairHash> pair_counters;

    std::string word;
    while (file >> word) {
        ++words_reader[word];  // inserts if not present, then increments
    }    

    for (const auto& pair : words_reader) {
        VocabWord vw;
        vw.tokens = split_to_utf8_list(pair.first);
        vw.tokens.push_back(word_delimiter_symbol); // Adding end word delimiter
        vw.frequency = pair.second;                 // Copy the count
        
        dictionary.push_back(vw);
    }

    int num_merges = std::max(10, static_cast<int>(words_reader.size() * 0.30)); // The BPE Merge Budget
    for (int i = 0; i < num_merges; ++i) {
        
        // 1. Tally all the pairs
        auto pair_counters = get_pair_counts(dictionary);

        // --- THE FIX: Stop if there is nothing left to merge! ---
        if (pair_counters.empty()) {
            std::cout << "No more pairs to merge! Stopping early at loop " << i << ".\n";
            break;
        }

        // 2. Find the champion
        auto best = *std::max_element(
            pair_counters.begin(),
            pair_counters.end(),
            [](const auto& a, const auto& b) {
                return a.second < b.second;
            }
        );

        // std::cout << "Merge #" << i + 1 << ": Champion is {" 
        //           << best.first.first << ", " << best.first.second 
        //           << "} with " << best.second << " points.\n";

        // 3. Execute the merge across the dictionary
        merge_pair_in_dictionary(dictionary, best.first);
    }

    print_dictionary(dictionary);

    return 0;
}