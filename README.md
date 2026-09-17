# C++ N-Gram Generator

This is the primary `main` branch of the N-Gram Text Generator.

*(Note: The original procedural word-level Markov Chain implementation is archived in the `main-old` branch.)*

This version implements an object-oriented $n$-gram language model pipeline utilizing Byte Pair Encoding (BPE), entropy-based boundary detection, and a segregated paragraph-level training architecture.

## Architecture & Features

* **Byte Pair Encoding (BPE):** Tokenizes text into subword units rather than strict word boundaries. This limits the vocabulary size used by the transition matrix and allows unseen words to be represented as combinations of known subword units.
* **Entropy-Based Preprocessor:** Calculates Shannon Entropy on character bigram transitions to identify structural boundaries (e.g., punctuation). Low-entropy symbols are dynamically detached from adjacent alphanumeric characters prior to tokenization, causing punctuation and other structural markers to be treated as separate tokens.
* **Paragraph-Level Context Isolation:** The training ingestor processes text as discrete paragraph streams. The context window is constrained to paragraph boundaries to prevent the matrix from calculating transition probabilities across distinct text blocks.
* **Elastic Backoff & Cache Interpolation:** During generation, if an $n$-gram context is absent from the distribution matrix, the engine progressively drops the oldest token to search shorter $n-1$ contexts. A sliding window cache interpolates localized probabilities to bias the selection toward recently generated tokens.
* **Binary Serialization:** The probability distribution matrices and BPE vocabulary are serialized to a custom binary `.dat` format, separating the training phase from generation.

## AI-Assisted Workflow

This repository was developed with substantial assistance from AI-generated code, primarily during implementation and refactoring. Generated code was subsequently reviewed, modified, and tested during development.

## Compilation

The pipeline is separated into training and generation executables. Standard `-O3` optimization is recommended.

```bash
# Compile the training executable
g++ -O3 -std=c++17 Train.cpp NgramEngine.cpp BPETokenizer.cpp Preprocessor.cpp -o train_model

# Compile the generator executable
g++ -O3 -std=c++17 Generate.cpp NgramEngine.cpp -o generate_text

```

## Usage

### Training

Processes a raw text corpus, computes subword and $n$-gram frequencies, normalizes the probability distributions, and serializes the state to a binary file.

```bash
./train_model input_text.txt my_model.dat

```

### Generation

Loads the serialized model and generates text based on the computed distributions.

**Arguments:** `<model_file> <length> <temperature> <top-k>`

* `length`: Total number of tokens to generate.
* `temperature`: Scales the probability distribution ($P^{1/T}$). Values $>1.0$ flatten the distribution; values $<1.0$ sharpen it.
* `top-k`: Truncates the candidate list to the $K$ most probable tokens before sampling.

```bash
./generate_text my_model.dat 2000 1.5 5

```

---
