# Architecture & Technical Details

This document outlines the internal mechanics, math, and data structures used in this specific branch of the N-Gram Generator. 

## 1. The Core Pipeline: Decoupling Training and Generation
In standard procedural N-gram scripts, the script reads a file, builds a dictionary in RAM, generates text, and then deletes the RAM when the script closes. 

This engine is built like a machine learning pipeline. The `NgramEngine` class acts as the central brain. 
* `train.cpp` acts as the data ingestion script. It builds the dynamic alphabet and populates the probability matrices.
* `generate.cpp` acts as the inference script. It relies purely on pre-calculated math to generate a piece of text.

## 2. Binary Serialization (The `.dat` file)
To achieve persistence, this engine writes raw CPU memory directly to the hard drive. 
Instead of saving human-readable text (which is slow to parse), the `save_model` function uses `reinterpret_cast` to dump the exact bytes of the `size_t` variables, integers, and `double` arrays. 

When `load_model` is called, it uses the $O(1)$ `std::unordered_map::reserve()` optimization. By reading the total size of the maps from the binary header first, it pre-allocates the exact amount of heap memory required. This prevents the C++ standard library from having to dynamically resize and re-hash the tree.

## 3. The Attention Mechanism (Suffix Indexing)
Standard Markov models are rigid. If they encounter a context they haven't seen, they fail. If you set the context window too high (e.g., $N=20$), they overfit and become simple photocopiers of the training data.

To solve this, this engine implements a primitive form of **Attention**:
1. **The Suffix Index:** During training, every context string is split, and its final two characters are used as a key in a secondary map (`suffix_map`).
2. **Similarity Scoring:** During generation, the engine calculates a right-to-left similarity score between the current context and all other contexts that share the same suffix.
3. **Probability Blending:** If similar contexts are found, their probability vectors are multiplied by an exponential weight (`std::exp(score)`) and blended into a single, unified probability vector. 

This allows the engine to gracefully hallucinate. It can smoothly transition between two entirely different concepts because it found a structural spelling overlap between them.

## 4. Stochastic Truncation
To prevent the engine from getting stuck in infinite loops or perfectly regurgitating long strings of training text, a `NOISE_LEVEL` variable is introduced. 

At each step of generation, the engine rolls a random float between `0.0` and `1.0`. If the roll is below the noise threshold, it intentionally chops the oldest character off its current memory context. This mimics human ADHD—forcing the model to "forget" how the sentence started, which forces it to rely on the Attention mechanism to find a creative, unexpected way to finish the thought.

## 5. Limitations

While this engine successfully mimics the foundational math of modern language models, it operates at the character level. This inherently introduces several hard physical and mathematical limits:

### The Memory Bloat (Dense Arrays)
Currently, the probability distribution is stored as a Dense Array (`std::vector<double>`). If the engine discovers 500 unique characters in a messy dataset, every single context string saves an array of 500 probabilities even if 499 of them are `0.0`. This causes massive memory bloat. Upgrading to a Sparse Data Structure (where only non-zero probabilities are saved) is required to train on larger datasets.

### The "Amnesia vs. Photocopier" Trap
The engine is highly sensitive to the context length ($N$):
* **If $N$ is too low (e.g., $N=8$):** The model suffers from severe amnesia. It can only remember 8 characters into the past, meaning by the time it finishes the word "elephant", it has already forgotten the beginning of the sentence. It can never maintain a single train of thought.
* **If $N$ is too high (e.g., $N=20$):** The model suffers from data sparsity. A 20-character string usually only appears exactly once in a training text. The model stops "guessing" and just becomes a photocopier, rigidly regurgitating the exact text it was trained on.

### Spelling vs. Meaning
Because the engine only looks at letters, it has absolutely no concept of what a "word" is, let alone grammar or semantics. The Attention mechanism operates purely on spelling overlaps (e.g., blending "king" and "ring" because they share "ing"). Modern Large Language Models use Word Embeddings to map the *meaning* of words, which is why this engine peaks at generating surreal, hardly readable portmanteaus rather than logically coherent paragraphs.