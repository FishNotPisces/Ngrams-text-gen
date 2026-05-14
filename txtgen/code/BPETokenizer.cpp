#include "BPETokenizer.h"
#include <fstream>
#include <iostream>
#include <algorithm>
#include <sstream> // Put this at the top

BPETokenizer::BPETokenizer(std::string delimiter) : word_delimiter_symbol(delimiter) {}

size_t BPETokenizer::get_utf8_char_len(unsigned char c) const {
    if ((c & 0b10000000) == 0b00000000) return 1; 
    if ((c & 0b11100000) == 0b11000000) return 2; 
    if ((c & 0b11110000) == 0b11100000) return 3; 
    if ((c & 0b11111000) == 0b11110000) return 4; 
    return 1; 
}

std::list<std::string> BPETokenizer::split_to_utf8_list(const std::string& word) const {
    std::list<std::string> tokens;
    size_t i = 0;
    while (i < word.length()) {
        size_t len = get_utf8_char_len(word[i]);
        tokens.push_back(word.substr(i, len));
        i += len;
    }
    return tokens;
}

std::unordered_map<std::pair<std::string, std::string>, int, BPETokenizer::PairHash> 
BPETokenizer::get_pair_counts() const {
    std::unordered_map<std::pair<std::string, std::string>, int, PairHash> pair_counts;

    for (const auto& word : dictionary) {
        if (word.tokens.size() < 2) continue; 

        auto it = word.tokens.begin();       
        auto next_it = std::next(it);        

        while (next_it != word.tokens.end()) {
            std::pair<std::string, std::string> current_pair = {*it, *next_it};
            pair_counts[current_pair] += word.frequency;
            it++;
            next_it++;
        }
    }
    return pair_counts;
}

void BPETokenizer::merge_pair_in_dictionary(const std::pair<std::string, std::string>& target) {
    for (auto& vw : dictionary) {
        auto it = vw.tokens.begin();

        while (it != vw.tokens.end()) {
            auto next = std::next(it);
            if (next == vw.tokens.end()) break;

            if (*it == target.first && *next == target.second) {
                *it += *next;
                vw.tokens.erase(next);
            } else {
                ++it;
            }
        }
    }
}



bool BPETokenizer::train_from_text(const std::string& text_buffer, double budget_ratio) {
    std::unordered_map<std::string, int> words_reader;
    std::istringstream stream(text_buffer);
    std::string word;
    
    while (stream >> word) { ++words_reader[word]; }    

    dictionary.clear();
    for (const auto& pair : words_reader) {
        VocabWord vw;
        vw.tokens = split_to_utf8_list(pair.first);
        vw.tokens.push_back(word_delimiter_symbol); 
        vw.frequency = pair.second;                 
        dictionary.push_back(vw);
    }

    // =========================================================
    // PHASE 2: INITIALIZATION PASS (Count everything once)
    // =========================================================
    true_pair_counts.clear();
    pair_to_words.clear();
    pq = std::priority_queue<QueueItem>(); // Clear the queue

    for (size_t word_id = 0; word_id < dictionary.size(); ++word_id) {
        const auto& tokens = dictionary[word_id].tokens;
        int word_freq = dictionary[word_id].frequency;

        if (tokens.size() < 2) continue;

        auto it = tokens.begin();
        auto next_it = std::next(it);

        while (next_it != tokens.end()) {
            std::pair<std::string, std::string> current_pair = {*it, *next_it};
            
            true_pair_counts[current_pair] += word_freq;
            pair_to_words[current_pair].insert(word_id);
            
            it++;
            next_it++;
        }
    }

    for (const auto& [pair, count] : true_pair_counts) {
        pq.push({count, pair});
    }

    // =========================================================
    // PHASE 3: THE FAST BPE LOOP
    // =========================================================
    int desired_merges = static_cast<int>(words_reader.size() * budget_ratio);
    // You no longer need the 1500 hard cap! You can run all of them now.
    int num_merges = std::max(10, desired_merges); 

    std::cout << "Executing " << num_merges << " optimized BPE merges...\n";

    for (int i = 0; i < num_merges; ++i) {
        std::pair<std::string, std::string> best_pair;
        int best_score = 0;

        // The Lazy Pop
        while (!pq.empty()) {
            QueueItem top = pq.top();
            pq.pop();

            if (top.score == true_pair_counts[top.pair]) {
                best_pair = top.pair;
                best_score = top.score;
                break; // Found the true champion!
            }
        }

        // If no more valid pairs exist, stop early
        if (best_score <= 0) break;

        // Execute the fast merge
        surgical_merge(best_pair); 
    }

    return true;
}

void BPETokenizer::print_dictionary() const {
    for (const auto& vw : dictionary) {
        for (const auto& token : vw.tokens) {
            std::cout << token << " ";
        }
        std::cout << "-> " << vw.frequency << '\n';
    }
}

const std::vector<BPETokenizer::VocabWord>& BPETokenizer::get_dictionary() const {
    return dictionary;
}

void BPETokenizer::surgical_merge(const std::pair<std::string, std::string>& best_pair) {
    // 1. Grab the list of words that actually contain this pair
    auto words_to_update = pair_to_words[best_pair];
    
    // 2. The champion pair is being merged, so destroy its raw counts
    true_pair_counts[best_pair] = 0;
    pair_to_words[best_pair].clear(); 

    std::string merged_token = best_pair.first + best_pair.second;

    // 3. Teleport only to the specific words that need updating
    for (int word_id : words_to_update) {
        auto& tokens = dictionary[word_id].tokens;
        int freq = dictionary[word_id].frequency;

        auto it = tokens.begin();
        while (it != tokens.end()) {
            auto next_it = std::next(it);
            if (next_it == tokens.end()) break;

            if (*it == best_pair.first && *next_it == best_pair.second) {
                // We found the exact pair inside the list!
                auto prev_it = (it != tokens.begin()) ? std::prev(it) : tokens.end();
                auto next_next_it = std::next(next_it);

                // --- DESTROY OLD TOUCHING PAIRS ---
                if (prev_it != tokens.end()) {
                    std::pair<std::string, std::string> left_pair = {*prev_it, *it};
                    true_pair_counts[left_pair] -= freq;
                    pair_to_words[left_pair].erase(word_id);
                }

                if (next_next_it != tokens.end()) {
                    std::pair<std::string, std::string> right_pair = {*next_it, *next_next_it};
                    true_pair_counts[right_pair] -= freq;
                    pair_to_words[right_pair].erase(word_id);
                }

                // --- EXECUTE THE MERGE ---
                *it = merged_token;     // Transform the first token
                tokens.erase(next_it);  // Delete the second token

                // --- CREATE NEW TOUCHING PAIRS ---
                if (prev_it != tokens.end()) {
                    std::pair<std::string, std::string> new_left_pair = {*prev_it, *it};
                    true_pair_counts[new_left_pair] += freq;
                    pair_to_words[new_left_pair].insert(word_id);
                    pq.push({true_pair_counts[new_left_pair], new_left_pair}); // Toss on pile
                }

                next_next_it = std::next(it); // needs to be recalculated because the old next_next_it may be stale after the erase 
                if (next_next_it != tokens.end()) {
                    std::pair<std::string, std::string> new_right_pair = {*it, *next_next_it};
                    true_pair_counts[new_right_pair] += freq;
                    pair_to_words[new_right_pair].insert(word_id);
                    pq.push({true_pair_counts[new_right_pair], new_right_pair}); // Toss on pile
                }
            } 
            it++; // Move to the next token in the word
        }
    }
}