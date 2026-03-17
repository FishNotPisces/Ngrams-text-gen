# Ngrams-text-gen
Character level n-gram text generator based on input text learning

This small project sparked out of pure curiosity. I was wondering: *If I just generate characters one by one based on their probability of appearing in a real text, would the output actually look like natural language?*

Starting with that simple idea, I began incrementally adding complexity. I moved from basic letter probabilities to N-grams (looking at the last N characters to predict the next), tackled UTF-8 parsing to handle multi-byte characters, and eventually implemented concepts like Additive Smoothing and Katz's Backoff to prevent the model from getting stuck in dead ends or just plagiarizing the training text.

## (IMPORTANT) AI-Assisted Workflow
This repository was built relying heavily on AI generated code that had been then checked for correctness.
I drove the logic, identified the mathematical bottlenecks (like overfitting and dead keys), conceptualized the fixes (like "matrix slicing" the initial seeds and injecting distance-based probability noise), directed the iterations, and wrote the initial rudimentary version of the program.
**AI's Role:** I used an AI assistant to translate my logic and rough code into optimized, modern C++.

## Features
* **Dynamic Context Length:** Easily tweak how many characters the model remembers to control the balance between spelling accuracy and creative chaos.
* **Robust UTF-8 Support:** Handles standard ASCII alongside multi-byte characters like accented letters without corrupting the memory strings.
* **Backoff & Smoothing:** Injects distance-based noise to keep the text creative, but dynamically "backs off" to shorter memory sequences to instantly self-correct before the text degrades into pure gibberish.

## Relevant Generated Outputs
Tweaking the memory length and the training data yields wildly different results. Here are a few of my favorites from testing:

**"Fever Dream" (N=5, trained on a mashup of Wikipedia, Moby Dick, and Frankenstein):**
> *"slang in that once and it undred part of sunlighten he glasses only cheek likely as were running, but quickly smile. it was flung on his earliest been skylarking..."*

**"Italian Simlish" (Trigram model trained on Italian text):**
> *"za suovivito il tra ine, unaressiela so partanza fria dintavano di da inciabbame lendo che musciontembusi quel trava seggerva..."*

**"Accidental Plagiarism" (N=10, trained on Pride and Prejudice):**
> *"it was, moreover, including checks, online payments and credit card donations or connections, and in a prudential light it is certainly leave kent..."* (It memorized the modern Project Gutenberg copyright boilerplate and wove it into Jane Austen's prose)

## How to Compile and Run
1. Clone the repository to your local machine.
2. Compile the C++ file using a modern compiler:
   ```bash
   g++ main.cpp -o ngram_gen -O3 -std=c++17
   ```
3. Run the executable, passing your training text file as an argument:
   ```bash
   ./ngram_gen your_training_data.txt
   ```
