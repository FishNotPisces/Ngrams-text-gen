#include <iostream>
#include <fstream>
#include <cctype>
#include "NgramEngine.h"

#include <sstream>
#include "BPETokenizer.h"

#include "Preprocessor.h"

std::vector<std::string> build_training_sequence(const std::string& text_buffer, const BPETokenizer& tokenizer) {
    std::unordered_map<std::string, std::list<std::string>> fast_lookup;
    for (const auto& vw : tokenizer.get_dictionary()) {
        std::string original_word = "";
        for (const auto& t : vw.tokens) { original_word += t; }
        fast_lookup[original_word] = vw.tokens;
    }

    std::vector<std::string> chronological_sequence;
    std::istringstream stream(text_buffer);
    std::string word;

    while (stream >> word) {
        std::string search_target = word + "</w>"; 
        auto it = fast_lookup.find(search_target);
        if (it != fast_lookup.end()) {
            for (const auto& piece : it->second) {
                chronological_sequence.push_back(piece);
            }
        }
    }
    return chronological_sequence;
}

void sanitize_text(std::string& text) {
    std::vector<std::pair<std::string, std::string>> replacements = {
        {"“", "\""}, {"”", "\""},  // Smart double quotes to normal
        {"‘", "'"},  {"’", "'"},   // Smart single quotes to normal
        {"—", " - "}               // Em-dash to spaced hyphen
    };

    for (const auto& [bad, good] : replacements) {
        size_t pos = 0;
        while ((pos = text.find(bad, pos)) != std::string::npos) {
            text.replace(pos, bad.length(), good);
            pos += good.length();
        }
    }
}

int main(int argc, char const *argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <input_text_file> <output_model_file.dat>" << std::endl;
        return 1;
    }

    std::string input_file = argv[1];
    std::string output_file = argv[2];

    std::ifstream file(input_file);
    if (!file) {
        std::cerr << "Error: Could not open " << input_file << std::endl;
        return 1;
    }

    // Read and clean the text (your Ultimate Space Crusher logic)
    std::string text;
    std::string line;
    while (std::getline(file, line)) {
        if (line.empty()) continue;
        for (char &c : line) c = std::tolower(static_cast<unsigned char>(c));
        text += line + " ";
    }

    std::string precleaned_text;
    bool in_space = false;
    for (char c : text) {
        if (std::isspace(static_cast<unsigned char>(c))) {
            if (!in_space) {
                precleaned_text += ' ';
                in_space = true;
            }
        } else {
            precleaned_text += c;
            in_space = false;
        }
    }
    

    sanitize_text(precleaned_text);
    auto rigid_symbols = Preprocessor::find_rigid_symbols(precleaned_text, 1.5);
    // Detach the punctuation from the edges
    std::string cleaned_text = Preprocessor::apply_edge_splitting(precleaned_text, rigid_symbols);

    std::cout << "Training tokenizer on " << cleaned_text.size() << " bytes of text..." << std::endl;
    
    // 1. Train BPE Vocabulary
    BPETokenizer tokenizer;
    tokenizer.train_from_text(cleaned_text, 0.50);
    
    // 2. Translate text into BPE Tokens
    std::cout << "Translating text into BPE subwords..." << std::endl;
    std::vector<std::string> bpe_tokens = build_training_sequence(cleaned_text, tokenizer);

    // 3. Train the N-gram Engine
    std::cout << "Training N-gram Engine on " << bpe_tokens.size() << " tokens..." << std::endl;
    NgramEngine engine(4);

    // --- NEW: The Control Panel for Training ---
    TrainingParams t_params;
    t_params.enable_pruning = true;
    t_params.prune_low_freq = 0.00005; // You can adjust your typo filter here!
    t_params.prune_high_freq = 0.01;   // You can adjust your grammar glitch filter here!

    // Pass the config struct into the engine
    engine.train(bpe_tokens, t_params);

    std::cout << "Training complete. Saving to binary file..." << std::endl;
    if (engine.save_model(output_file)) {
        std::cout << "Success! Model saved to " << output_file << std::endl;
    } else {
        std::cerr << "Error: Failed to save model." << std::endl;
        return 1;
    }

    return 0;
}
