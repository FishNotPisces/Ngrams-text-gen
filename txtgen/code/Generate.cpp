#include <iostream>
#include <string>
#include "NgramEngine.h"

void smart_detokenize(std::string& text) {
    // We added the apostrophe in here to make it super clean!
    std::vector<std::string> left_snaps = {
        " .", " ,", " !", " ?", " :", " ;", " %", " '"
    };
    
    for (const auto& mark : left_snaps) {
        size_t pos = 0;
        while ((pos = text.find(mark, pos)) != std::string::npos) {
            // Because the mark is " .", the space is perfectly at 'pos'.
            // Instead of shifting memory with replace, we just delete the 1 space!
            // The punctuation automatically slides to the left.
            text.erase(pos, 1); 
        }
    }
}

void print_decoded(const std::string& raw_output) {
    std::string final_text = "";
    std::string target = "</w>";
    size_t start = 0;
    size_t pos = 0;
    
    // Build the string iteratively (O(N) time) instead of shifting memory!
    while ((pos = raw_output.find(target, start)) != std::string::npos) {
        final_text += raw_output.substr(start, pos - start); // Grab the token
        final_text += " ";                                   // Add the space
        start = pos + target.length();                       // Jump over the </w>
    }
    
    // Catch any remaining text after the very last token
    final_text += raw_output.substr(start);
    
    // Now detokenize and print
    smart_detokenize(final_text);
    std::cout << final_text << std::endl;
}


int main(int argc, char const *argv[]) {
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " <model_file.dat> <length> <noise_level>" << std::endl;
        return 1;
    }

    std::string model_file = argv[1];
    size_t length = std::stoul(argv[2]);
    double noise = std::stod(argv[3]);

    NgramEngine engine; // Context length doesn't matter here, it just needs to load

    std::cout << "Loading model..." << std::endl;
    if (!engine.load_model(model_file)) {
        std::cerr << "Error: Could not load " << model_file << std::endl;
        return 1;
    }

    // --- NEW: The Control Panel for Generation ---
    GenerationParams g_params;
    g_params.length = length;          // From argv[2]
    g_params.noise_level = noise;      // From argv[3]
    g_params.use_attention = true;     // Toggle your new suffix-blending on!
    g_params.attention_threshold = 2.0;
    g_params.temperature = 1.5;       // Tweak this! Higher = more creative, lower = more repetitive
    g_params.K = 5;

    std::cout << "Model loaded! Generating text..." << std::endl;
    std::cout << "-----------------------------------" << std::endl;

    // Pass the config struct instead of raw variables
    std::string output = engine.generate(g_params);
    print_decoded(output); // Decodes the BPE tags beautifully!

    return 0;
}
