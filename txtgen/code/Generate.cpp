#include <iostream>
#include <string>
#include "NgramEngine.h"

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

    std::cout << "Model loaded! Generating text..." << std::endl;
    std::cout << "-----------------------------------" << std::endl;

    std::string output = engine.generate(length, noise);
    std::cout << output << std::endl;

    return 0;
}
