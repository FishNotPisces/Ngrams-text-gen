#include <iostream>
#include "BPETokenizer.h"

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <input_file>" << std::endl;
        return 1;
    }

    std::string text_file = argv[1];

    // 1. Create the Tokenizer
    BPETokenizer tokenizer;

    // 2. Train it on the text file
    if (!tokenizer.train_from_file(text_file, 0.30)) {
        return 1; // Exit if the file fails to load
    }

    // 3. Print the results
    tokenizer.print_dictionary();

    // 4. (Ready for next steps) 
    // auto dict = tokenizer.get_dictionary();

    return 0;
}