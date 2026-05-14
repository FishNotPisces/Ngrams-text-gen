// Preprocessor.h
#pragma once

#include <string>
#include <unordered_map>
#include <unordered_set>

class Preprocessor {
public:
    /**
     * @brief Mathematically deduces which symbols are rigid boundaries
     *        by calculating Shannon Entropy over bigram transitions.
     *
     * @param text               The input text to analyze.
     * @param entropy_threshold  Symbols with entropy below this value are
     *                           considered rigid (default: 1.5).
     * @return An unordered_set of characters identified as rigid boundary symbols.
     */
    static std::unordered_set<char> find_rigid_symbols(
        const std::string& text,
        double entropy_threshold = 1.5
    );

    /**
     * @brief Applies the Edge-Only detachment rule, splitting rigid symbols
     *        away from word edges so the tokenizer sees them as separate tokens.
     *
     * @param text          The input text to process.
     * @param rigid_symbols The set of symbols (from find_rigid_symbols) to detach.
     * @return A new string with rigid edge symbols space-separated from their words.
     */
    static std::string apply_edge_splitting(
        const std::string& text,
        const std::unordered_set<char>& rigid_symbols
    );
};