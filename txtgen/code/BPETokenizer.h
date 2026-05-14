#ifndef BPE_TOKENIZER_H
#define BPE_TOKENIZER_H

#include <string>
#include <vector>
#include <list>
#include <unordered_map>
#include <utility>
#include <unordered_set>
#include <queue>

class BPETokenizer {
public:
    // We keep your struct public so the main program can read it later
    struct VocabWord {
        std::list<std::string> tokens;
        int frequency;
    };

    // Constructor
    BPETokenizer(std::string delimiter = "</w>");

    // Core functions
    bool train_from_file(const std::string& filepath, double budget_ratio = 0.30);
    void print_dictionary() const;

    // A simple getter so your N-gram engine can grab the dictionary later
    const std::vector<VocabWord>& get_dictionary() const;

    bool train_from_text(const std::string& text_buffer, double budget_ratio = 0.30);

private:
    std::string word_delimiter_symbol;
    std::vector<VocabWord> dictionary;

    struct PairHash {
        std::size_t operator()(const std::pair<std::string, std::string>& p) const {
            auto h1 = std::hash<std::string>{}(p.first);
            auto h2 = std::hash<std::string>{}(p.second);
            return h1 ^ (h2 * 2654435761ULL);
        }
    };

    struct QueueItem {
        int score;
        std::pair<std::string, std::string> pair;
        
        // This tells the queue how to sort (Highest score at the top)
        bool operator<(const QueueItem& other) const {
            return score < other.score; 
        }
    };

    std::unordered_map<std::pair<std::string, std::string>, int, PairHash> true_pair_counts;
    std::unordered_map<std::pair<std::string, std::string>, std::unordered_set<int>, PairHash> pair_to_words;
    std::priority_queue<QueueItem> pq;
    
    // Private helper functions (hidden from the user)
    size_t get_utf8_char_len(unsigned char c) const;
    std::list<std::string> split_to_utf8_list(const std::string& word) const;
    std::unordered_map<std::pair<std::string, std::string>, int, PairHash> get_pair_counts() const;
    void merge_pair_in_dictionary(const std::pair<std::string, std::string>& target);
    void surgical_merge(const std::pair<std::string, std::string>& best_pair);
};

#endif // BPE_TOKENIZER_H