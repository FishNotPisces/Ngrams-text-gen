# C++ N-Gram Generator

This is the primary `main` branch of the N-Gram Text Generator.

(Note: The original, purely procedural Markov Chain implementation has been archived to the `main-old` branch.)

This architecture features a highly optimized, Object-Oriented pipeline utilizing dynamic subword tokenization, Shannon Entropy heuristics, and a paragraph-streaming architecture.

## Key Features

* **Byte Pair Encoding (BPE):** Replaces rigid word-level tokens with dynamically merged subwords, allowing the engine to handle vast vocabularies and construct out-of-vocabulary words without memory bloat.
* **Entropy-Based Preprocessor:** Mathematically deduces rigid punctuation joints using Shannon Entropy on bigram transitions, cleanly detaching formatting from word edges without hardcoded rules.
* **Paragraph-Streaming Architecture:** The N-gram engine strictly isolates context windows within paragraph boundaries, preventing cross-paragraph statistical contamination and "ghost link" hallucinations.
* **Elastic Backoff & Cache Boost:** When hitting a dead end, the engine gracefully shrinks its context window to find statistical anchors. A sliding 100-token memory cache simultaneously boosts the probability of recently used words to maintain long-range thematic coherence.
* **Binary Serialization:** Training and generation are now completely separated. You can train a model once, save its "brain" directly to a `.dat` binary file, and load it into memory in milliseconds to generate text on demand.



## (IMPORTANT) AI-Assisted Workflow

This repository was built relying heavily on AI generated code that had been then checked for correctness.

## Compiling the Project

Because the project is decoupled, you need to compile two separate executables: one for training, and one for generating. The `-O3` flag is highly suggested.

```bash
# Compile the training executable
g++ -O3 -std=c++17 Train.cpp NgramEngine.cpp BPETokenizer.cpp Preprocessor.cpp -o train_model

# Compile the generator executable
g++ -O3 -std=c++17 Generate.cpp NgramEngine.cpp -o generate_text

```

## How to use

### Train model

Feed a raw .txt file into the training executable. It will process the probabilities and output a binary .dat file.

```bash
./train_model input_text.txt my_model.dat

```

### Generate text

Load your trained .dat file, specify how many tokens you want to generate, and tune the Temperature (creativity) and Top-K (noise filtering) parameters.

```bash
./generate_text my_model.dat 2000 1.5 5

```
