# C++ N-Gram Generator (Attention & Serialization Branch)

This is an experimental branch of the N-Gram Text Generator

While the `main` branch contains a pure, procedural Markov Chain, this branch refactors the engine into an Object-Oriented pipeline, implements a custom "Attention" mechanism for creative text blending, and decouples the training and generation phases using raw binary serialization.

## Key Features in this Branch

* **Object-Oriented Design:** The core logic has been encapsulated into the `NgramEngine` class, isolating memory management and making the codebase highly modular.
* **Primitive Attention Mechanism:** Instead of rigid exact-matching, the engine indexes context suffixes. If it gets stuck, it searches for structurally similar contexts and mathematically blends their probability vectors, resulting in fluid, dream-like hallucinations.
* **Stochastic Truncation:** A built-in "noise" variable randomly chops the context window during generation, forcing the model to forget its history and jump to new, adjacent topics.
* **Binary Serialization:** Training and generation are now completely separated. You can train a model once, save its "brain" directly to a `.dat` binary file, and load it into memory in milliseconds to generate text on demand.

## (IMPORTANT) AI-Assisted Workflow
This repository was built relying heavily on AI generated code that had been then checked for correctness.

## Compiling the Project

Because the project is now decoupled, you need to compile two separate executables: one for training, and one for generating. The `-O3` flag is highly suggested.

```bash
# Compile the training executable
g++ -O3 -std=c++17 train.cpp NgramEngine.cpp -o train_model

# Compile the generator executable
g++ -O3 -std=c++17 generate.cpp NgramEngine.cpp -o generate_text
```

## How to use

### Train model

Feed a raw .txt file into the training executable. It will process the probabilities and output a binary .dat file.

```bash
./train_model input_text.txt my_model.dat
```

### Generate text

Load your trained .dat file, specify how many characters you want to generate, and set the noise level (e.g., 0.1 for 10% chance of context truncation).

```
./generate_text my_model.dat 2000 0.1
```