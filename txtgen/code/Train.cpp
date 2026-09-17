#include <iostream>
#include <fstream>
#include <cctype>
#include <vector>
#include <sstream>

#include "NgramEngine.h"
#include "BPETokenizer.h"
#include "Preprocessor.h"

// --- HELPER FUNCTIONS (Unchanged) ---
std::vector<std::string> build_training_sequence(
    const std::string& text_buffer, 
    const std::unordered_map<std::string, std::list<std::string>>& fast_lookup,
    int& global_dropped) // Passed by reference
{
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
        } else {
            global_dropped++; // Add to the global tally
        }
    }
    return chronological_sequence;
}

void sanitize_text(std::string& text) {
    std::vector<std::pair<std::string, std::string>> replacements = {
        {"“", "\""}, {"”", "\""},  
        {"‘", "'"},  {"’", "'"},   
        {"—", " - "}               
    };

    for (const auto& [bad, good] : replacements) {
        size_t pos = 0;
        while ((pos = text.find(bad, pos)) != std::string::npos) {
            text.replace(pos, bad.length(), good);
            pos += good.length();
        }
    }
}


// --- MAIN PIPELINE ---
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

    // =========================================================
    // PHASE 1: PARAGRAPH EXTRACTION & CLEANING
    // =========================================================
    std::vector<std::string> precleaned_paragraphs;
    std::string current_paragraph = "";
    std::string line;

    while (std::getline(file, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();

        // Empty line defines a paragraph boundary
        if (line.empty() || line.find_first_not_of(" \t") == std::string::npos) {
            if (!current_paragraph.empty()) {
                precleaned_paragraphs.push_back(current_paragraph);
                current_paragraph = "";
            }
        } else {
            for (char &c : line) c = std::tolower(static_cast<unsigned char>(c));
            current_paragraph += line + " ";
        }
    }
    if (!current_paragraph.empty()) {
        precleaned_paragraphs.push_back(current_paragraph);
    }

    // Apply the Space Crusher and Sanitize to each paragraph individually
    std::string global_text_for_training = "";
    for (auto& p : precleaned_paragraphs) {
        std::string crushed = "";
        bool in_space = false;
        for (char c : p) {
            if (std::isspace(static_cast<unsigned char>(c))) {
                if (!in_space) { crushed += ' '; in_space = true; }
            } else {
                crushed += c; in_space = false;
            }
        }
        sanitize_text(crushed);
        p = crushed; // Save back to the vector
        global_text_for_training += p + " "; // Stitch for the BPE phase
    }

    // =========================================================
    // PHASE 2: GLOBAL TRAINING (Entropy & BPE)
    // =========================================================
    std::cout << "Calculating entropy for rigid symbols..." << std::endl;
    auto rigid_symbols = Preprocessor::find_rigid_symbols(global_text_for_training, 1.5);
    std::string global_cleaned_text = Preprocessor::apply_edge_splitting(global_text_for_training, rigid_symbols);

    std::cout << "Training BPE Tokenizer on " << global_cleaned_text.size() << " bytes..." << std::endl;
    BPETokenizer tokenizer;
    tokenizer.train_from_text(global_cleaned_text, 0.50);

    // =========================================================
    // PHASE 3: STREAMING N-GRAM OBSERVATION
    // =========================================================
    std::cout << "Building fast lookup matrix..." << std::endl;
    std::unordered_map<std::string, std::list<std::string>> fast_lookup;
    for (const auto& vw : tokenizer.get_dictionary()) {
        std::string original_word = "";
        for (const auto& t : vw.tokens) { original_word += t; }
        fast_lookup[original_word] = vw.tokens;
    }

    std::cout << "Streaming paragraphs into N-gram Engine..." << std::endl;
    NgramEngine engine(4);
    int total_dropped_words = 0; // Initialize global counter

    for (const auto& paragraph : precleaned_paragraphs) {
        std::string final_p = Preprocessor::apply_edge_splitting(paragraph, rigid_symbols);
        
        // Pass the counter here!
        std::vector<std::string> bpe_tokens = build_training_sequence(final_p, fast_lookup, total_dropped_words);

        if (!bpe_tokens.empty()) {
            engine.observe_text(bpe_tokens);
        }
    }

    // Print once at the end
    if (total_dropped_words > 0) {
        std::cerr << "Warning: " << total_dropped_words << " words dropped during paragraph streaming.\n";
    }

    // =========================================================
    // PHASE 4: COMPILATION & EXPORT
    // =========================================================
    std::cout << "Compiling probability distributions..." << std::endl;
    engine.compile_probabilities();

    std::cout << "Training complete. Saving to binary file..." << std::endl;
    if (engine.save_model(output_file)) {
        std::cout << "Success! Model saved to " << output_file << std::endl;
    } else {
        std::cerr << "Error: Failed to save model." << std::endl;
        return 1;
    }

    return 0;
}