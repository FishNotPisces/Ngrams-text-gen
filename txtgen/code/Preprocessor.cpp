#include "Preprocessor.h"
#include <iostream>
#include <cmath>
#include <cctype>

std::unordered_set<char> Preprocessor::find_rigid_symbols(const std::string& text, double entropy_threshold) {
    std::unordered_map<char, std::unordered_map<char, int>> transitions;
    std::unordered_map<char, int> total_occurrences;

    // Count all Bigram transitions
    for (size_t i = 0; i < text.length() - 1; ++i) {
        char current = text[i];
        char next = text[i + 1];
        transitions[current][next]++;
        total_occurrences[current]++;
    }

    std::unordered_set<char> rigid_symbols;

    // Calculate Shannon Entropy for every character
    for (const auto& [current_char, next_chars] : transitions) {
        // NEW: Ignore UTF-8 multi-byte characters entirely (> 127)
        if (static_cast<unsigned char>(current_char) > 127 || 
            std::isalnum(static_cast<unsigned char>(current_char)) || 
            std::isspace(static_cast<unsigned char>(current_char))) {
            continue;
        }

        double entropy = 0.0;
        double total = total_occurrences[current_char];

        for (const auto& [next_char, count] : next_chars) {
            double probability = count / total;
            entropy -= probability * std::log2(probability);
        }

        // If the entropy is below our threshold, it is a rigid structural joint!
        if (entropy < entropy_threshold) {
            rigid_symbols.insert(current_char);
            std::cout << "Detected rigid symbol: [" << current_char << "] (Entropy: " << entropy << ")\n";
        }
    }

    return rigid_symbols;
}

std::string Preprocessor::apply_edge_splitting(const std::string& text, const std::unordered_set<char>& rigid_symbols) {
    std::string result = "";
    std::string current_word = "";

    // A quick helper lambda to process a single word
    auto process_word = [&]() {
        if (current_word.empty()) return;

        std::string left_symbols = "";
        std::string right_symbols = "";

        // FIX: Strip from the RIGHT edge and put the space AFTER the symbol
        while (!current_word.empty() && rigid_symbols.count(current_word.back())) {
            right_symbols = std::string(1, current_word.back()) + " " + right_symbols;
            current_word.pop_back();
        }

        // Strip from the LEFT edge (This one was already correct)
        while (!current_word.empty() && rigid_symbols.count(current_word.front())) {
            left_symbols += std::string(1, current_word.front()) + " ";
            current_word.erase(0, 1);
        }

        // Reassemble
        result += left_symbols;
        if (!current_word.empty()) {
            result += current_word + " ";
        }
        result += right_symbols;
        
        current_word = "";
    };

    // Walk the text
    for (char c : text) {
        if (std::isspace(static_cast<unsigned char>(c))) {
            process_word();
        } else {
            current_word += c;
        }
    }
    process_word(); // Catch the very last word

    return result;
}