#include <iostream>
#include <fstream>
#include <cctype>
#include "NgramEngine.h"

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

    std::cout << "Training engine on " << cleaned_text.size() << " bytes of text..." << std::endl;

    // Boot up the engine (Context Length N=8)
    NgramEngine engine(8);

    // --- NEW: The Control Panel for Training ---
    TrainingParams t_params;
    t_params.enable_pruning = true;
    t_params.prune_low_freq = 0.00005; // You can adjust your typo filter here!
    t_params.prune_high_freq = 0.01;   // You can adjust your grammar glitch filter here!

    // Pass the config struct into the engine
    engine.train(cleaned_text, t_params);

    std::cout << "Training complete. Saving to binary file..." << std::endl;
    if (engine.save_model(output_file)) {
        std::cout << "Success! Model saved to " << output_file << std::endl;
    } else {
        std::cerr << "Error: Failed to save model." << std::endl;
        return 1;
    }

    return 0;
}
