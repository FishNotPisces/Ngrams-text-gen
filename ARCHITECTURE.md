# Architecture

This document describes how the current version of the project is structured and how data flows through it. It is derived from the source code and describes only what the code does. For build and usage instructions, see [README.md](README.md).

## 1. Overview

The project is a character-level-seeded, subword-token **n-gram text generator** written in C++17 with no dependencies beyond the standard library. It consists of two command-line programs:

- **`train_model`**: reads a plain-text corpus, preprocesses it, learns a Byte Pair Encoding (BPE) vocabulary, counts token n-gram transitions, normalizes them into probabilities, and writes the result to a binary model file.
- **`generate_text`**: loads a model file and samples a sequence of tokens from it, then converts the tokens back to readable text.

The model is a conventional back-off n-gram model over BPE tokens with a maximum context of 4 tokens, plus a small recency-based "cache" interpolation applied at sampling time. There are no neural components and no learned parameters other than the n-gram probability tables.

## 2. Source files

| File | Role |
|---|---|
| `Train.cpp` | Entry point of `train_model`. Reads and cleans the corpus, orchestrates the preprocessing, BPE and n-gram phases, saves the model. |
| `Preprocessor.h/.cpp` | Detects "rigid" punctuation symbols using bigram entropy and detaches them from word edges. |
| `BPETokenizer.h/.cpp` | Learns a BPE vocabulary from a text buffer and exposes, for every distinct word, its final token sequence. |
| `NgramEngine.h/.cpp` | Stores n-gram statistics, normalizes them, serializes/deserializes the model, and generates token sequences. |
| `Generate.cpp` | Entry point of `generate_text`. Loads a model, calls the generator, decodes tokens to text. |

Build targets (from the README):

```
train_model   = Train.cpp + NgramEngine.cpp + BPETokenizer.cpp + Preprocessor.cpp
generate_text = Generate.cpp + NgramEngine.cpp
```

`generate_text` does not link the tokenizer or the preprocessor: the model file alone is sufficient for generation.

## 3. Pipeline at a glance

```
                              TRAINING (train_model)

 corpus.txt
     │
     ▼
 [1] Paragraph extraction & cleaning        (Train.cpp)
     │   lowercase, whitespace collapse, quote/dash normalization
     ▼
 [2] Rigid-symbol detection                 (Preprocessor::find_rigid_symbols)
     │   bigram Shannon entropy on the whole corpus
     ▼
 [3] Edge splitting                         (Preprocessor::apply_edge_splitting)
     │   detach rigid symbols from word edges
     ▼
 [4] BPE training                           (BPETokenizer::train_from_text)
     │   → dictionary: word → list of tokens
     ▼
 [5] N-gram observation, paragraph by paragraph
     │   each paragraph: edge-split → BPE tokens → NgramEngine::observe_text
     ▼
 [6] Normalization                          (NgramEngine::compile_probabilities)
     ▼
 [7] Serialization                          (NgramEngine::save_model)
     ▼
 model.dat


                            GENERATION (generate_text)

 model.dat ──► NgramEngine::load_model ──► NgramEngine::generate ──► raw token string
                                                                          │
                                                                          ▼
                                                              print_decoded (Generate.cpp)
                                                              "</w>" → space, punctuation fix-up
                                                                          │
                                                                          ▼
                                                                    text on stdout
```

## 4. Training pipeline in detail

### 4.1 Paragraph extraction and cleaning (`Train.cpp`)

The input file is read line by line.

- A trailing `\r` is removed from each line.
- A blank line (empty or only spaces/tabs) ends the current paragraph.
- Non-blank lines are lowercased byte by byte with `std::tolower` and appended to the current paragraph, separated by a space.

Each paragraph is then processed independently:

1. Runs of whitespace are collapsed into a single space.
2. `sanitize_text` replaces a fixed set of typographic characters: `“` and `”` become `"`, `‘` and `’` become `'`, and `—` becomes ` - `.

The cleaned paragraphs are kept in a vector (used again in phase 5) and also concatenated, separated by spaces, into a single string used for the global training phases (2–4).

Note: lowercasing is ASCII-only. Bytes above 127 are left unchanged, so non-ASCII letters are not lowercased.

### 4.2 Rigid-symbol detection (`Preprocessor::find_rigid_symbols`)

Goal: find punctuation-like characters whose following character is highly predictable, so they can be treated as separate tokens instead of being glued to words.

Algorithm:

1. Over the whole text (as bytes), count bigram transitions `current → next` and the total number of occurrences of each `current` character.
2. For every `current` character that is **not** alphanumeric, **not** whitespace and **not** a byte above 127, compute the Shannon entropy of its next-character distribution:

   `H(c) = − Σ p(next | c) · log2 p(next | c)`

3. If `H(c)` is below the threshold, `c` is added to the set of rigid symbols. Each detected symbol and its entropy is printed to stdout.

The threshold is passed as an argument (default 1.5 in the header). `Train.cpp` calls it with `1.5`.

Only the distribution of the character *following* the symbol is considered. Multi-byte UTF-8 characters are never classified as rigid. Which symbols end up rigid depends entirely on the corpus.

### 4.3 Edge splitting (`Preprocessor::apply_edge_splitting`)

The text is split on whitespace. For each word:

1. Rigid symbols are stripped one by one from the **right** edge.
2. Rigid symbols are stripped one by one from the **left** edge.
3. The word is re-emitted as: left symbols, the remaining core, right symbols, each separated by a single space.

Only symbols at the edges of a word are detached. Rigid symbols in the interior of a word are left in place. The output is a space-separated string with a trailing space.

Example, assuming `.` and `,` are rigid:

```
"hello, world."   →   "hello , world . "
"3.14"            →   "3.14 "           (interior symbol, not detached)
```

The function is applied twice in `Train.cpp`: once to the global text (input for BPE training), and once to each paragraph (input for n-gram observation). Because it operates word by word with no context, both passes yield the same words.

### 4.4 BPE training (`BPETokenizer::train_from_text`)

`Train.cpp` calls `train_from_text(global_cleaned_text, 0.50)`.

**Initialization**

- The text is split on whitespace and distinct words are counted.
- Each distinct word becomes a `VocabWord`: a `std::list<std::string>` of tokens plus a frequency. The initial tokens are the word's UTF-8 characters (the splitter is multi-byte aware), followed by the end-of-word marker `</w>` as a separate token.

**Pair statistics**

Three structures are maintained:

- `true_pair_counts`: for each adjacent token pair, its total frequency-weighted count across the vocabulary.
- `pair_to_words`: for each pair, the set of word indices that contain it, so a merge only visits relevant words.
- `pq`: a max-heap of `(score, pair)` entries.

**Merge loop**

- Number of merges: `max(10, floor(distinct_words × budget_ratio))`. With the value used by `Train.cpp` this is half the number of distinct words, with a minimum of 10.
- Each iteration pops entries from the heap until one is found whose score equals the current value in `true_pair_counts` (lazy invalidation: stale entries are simply discarded). If the heap is exhausted without a valid entry, training stops early.
- `surgical_merge` then rewrites only the words listed in `pair_to_words[best_pair]`. For every occurrence of the pair inside a word it decrements the counts of the two neighbouring pairs, replaces the pair with the concatenated token, increments the counts of the new neighbouring pairs, and pushes the updated pairs onto the heap.

**Output**

The result is available through `get_dictionary()`: for each distinct word, its final token list (still ending with the `</w>` marker, which may have been merged into the preceding token, e.g. `ing</w>`). The sequence of merge rules is **not** stored.

### 4.5 N-gram observation (`Train.cpp` + `NgramEngine::observe_text`)

1. A lookup table `fast_lookup` is built from the BPE dictionary: the concatenation of a word's tokens (e.g. `cat</w>`) maps to its token list.
2. For each cleaned paragraph: it is edge-split, then each whitespace-separated word is looked up as `word + "</w>"` and its tokens are appended to a token sequence. Words not found in the table are skipped and counted; a warning with the total is printed at the end. Since the dictionary is built from the same corpus, this count is normally zero.
3. The token sequence of each paragraph is passed to `observe_text`. **Each paragraph is an independent sequence**, so context windows never span a paragraph boundary.

### 4.6 Normalization and export

After all paragraphs have been observed, `compile_probabilities` converts the counts in each context entry into probabilities, and `save_model` writes the model to disk (see section 6).

## 5. The n-gram engine (`NgramEngine`)

### 5.1 Data structures

```cpp
size_t CONTEXT_LEN;                                              // max context, in tokens
std::unordered_map<std::string, int> token_to_id;                // token → id
std::vector<std::string> id_to_token;                            // id → token
std::unordered_map<std::string,
    std::unordered_map<int, double>> ngram_distr;                // context key → {next token id → value}
```

The value stored in `ngram_distr` is a raw count after training and a probability after `compile_probabilities`.

### 5.2 Observation

For every position `i ≥ 1` of a token sequence, and for every context length `L` from 1 to `CONTEXT_LEN` (limited by the available preceding tokens), the key is the **concatenation of the `L` preceding tokens** and the count of token `i` under that key is incremented. A single position therefore updates up to `CONTEXT_LEN` entries (4 in the shipped configuration).

Example with tokens `the</w> cat</w> sat</w> .</w>` at the position of `sat</w>`:

```
key "cat</w>"          → count of "sat</w>" += 1
key "the</w>cat</w>"   → count of "sat</w>" += 1
```

Notes:

- The first token of each sequence is never a prediction target; it only serves as context.
- Keys are built by plain string concatenation, with no separator between tokens (see limitations).
- The vocabulary (`token_to_id`, `id_to_token`) is extended as new tokens are encountered.

### 5.3 Normalization

For each context key, the counts are divided by their sum, so the values sum to 1:

`P(t | context) = count(context, t) / Σ count(context, ·)`

### 5.4 Generation (`NgramEngine::generate`)

Inputs (`GenerationParams`): `length` (number of tokens), `temperature`, `K`.

**Initialization**

- The random generator is a `std::mt19937` seeded from `std::random_device`; there is no option to set a seed, so output is not reproducible.
- The first token is drawn uniformly from the whole vocabulary. It may therefore be a subword that does not start a word.
- The output contains exactly `length` tokens (the first one plus `length − 1` sampled ones). An empty string is returned if `length < 1` or the model is empty.

**For each subsequent token**

1. **Context and back-off.** The context is the last up to `CONTEXT_LEN` generated tokens. The engine looks up their concatenation in `ngram_distr`. If the key is missing, the oldest token is dropped and the lookup is repeated with the shorter context, until a match is found or the context is empty.
2. **Fallback.** If no context matches, a token is chosen uniformly at random from the vocabulary.
3. **Cache interpolation.** The engine keeps a sliding window of the last 100 generated tokens (`CACHE_SIZE = 100`). For each candidate token `t` returned by the n-gram lookup:

   `P'(t) = 0.85 · P_ngram(t) + 0.15 · (count of t in window / window size)`

   (`CACHE_WEIGHT = 0.15`). The cache only re-weights candidates that already exist in the n-gram distribution; it never adds new ones. Both constants are hard-coded in `generate`.
4. **Top-K filtering.** Candidates are sorted by probability in descending order and only the first `K` are kept.
5. **Temperature.** Each surviving weight is transformed to `w = P'(t)^(1/T)`. `std::discrete_distribution` normalizes the weights and samples one token. `T > 1` flattens the distribution, `T < 1` sharpens it.
6. **Update.** The token is appended to the output string and to the context, the cache window is updated, and the context is trimmed to `CONTEXT_LEN` tokens.

The returned value is the plain concatenation of all generated tokens.

### 5.5 Decoding (`Generate.cpp`)

1. `print_decoded` replaces every `</w>` marker with a space.
2. `smart_detokenize` removes the space before each of: `.` `,` `!` `?` `:` `;` `%` `'`.
3. The result is printed to stdout, after a few status lines (`Loading model...`, etc.) that are also written to stdout.

Because the training text is lowercased, the output is lowercase.

## 6. Model file format

The model is a single binary file written with raw `write`/`read` calls. There is no header, magic number, version field or checksum. Integers use the host's native representation and byte order, and `size_t` has the host's width (8 bytes on typical 64-bit platforms).

```
size_t   A                          number of tokens
  A ×  string                       id_to_token, in id order

size_t   M                          number of entries in token_to_id
  M ×  { string key; int id }       token_to_id (redundant with id_to_token)

size_t   N                          number of context keys
  N ×  {
      string key                    concatenated context
      size_t S                      number of candidate tokens
      S × { int token_id; double probability }
  }

string = { size_t byte_length; byte[byte_length] }     (no terminator)
```

Entries are written in `unordered_map` iteration order, so two runs on the same corpus may not produce byte-identical files.

`CONTEXT_LEN` is **not** stored in the file (see limitations).

## 7. Parameters and where they are set

| Parameter | Value | Location | Configurable at runtime |
|---|---|---|---|
| Rigid-symbol entropy threshold | 1.5 | `Train.cpp` (call to `find_rigid_symbols`) | No |
| BPE merge budget ratio | 0.50 of distinct words (min. 10 merges) | `Train.cpp` (call to `train_from_text`) | No |
| End-of-word marker | `</w>` | `BPETokenizer` constructor default; also hard-coded in `Train.cpp` and `Generate.cpp` | No |
| Context length | 4 | `Train.cpp` (`NgramEngine engine(4)`); default of the `NgramEngine` constructor, which `Generate.cpp` relies on | No |
| Cache size / weight | 100 / 0.15 | `NgramEngine::generate` | No |
| Output length | argument 2 | `generate_text` CLI | Yes |
| Temperature | argument 3 | `generate_text` CLI | Yes |
| Top-K | argument 4 | `generate_text` CLI | Yes |

All four `generate_text` arguments are required; the defaults in `GenerationParams` (2000, 1.5, 5) are not used by the CLI.

## 8. Design notes and known limitations

These are properties of the current implementation that are relevant when using or extending it.

**Tokenizer**

- The vocabulary is closed. The BPE merge rules are not saved, and there is no function that encodes new text. The model can only produce tokens that appeared in the training corpus, and unseen words cannot be represented as combinations of subwords at generation time.
- `BPETokenizer::train_from_file` is declared in the header but has no implementation; only `train_from_text` is available.
- `pair_to_words` entries are removed per merge occurrence. If a word contains the same neighbouring pair more than once and only one occurrence is touched by a merge, the word may be removed from that pair's index and some later occurrences can be left unmerged. The tokenization remains lossless (tokens always concatenate back to the original word) but may differ slightly from a textbook BPE.
- Ties between pairs with equal counts are resolved by heap order, which depends on `unordered_map` iteration order. The learned vocabulary may therefore differ between standard-library implementations.

**N-gram engine**

- Context keys are concatenations of tokens without a separator. Different token sequences that concatenate to the same string share the same key, which merges their statistics. In practice the `</w>` markers make this uncommon but not impossible.
- The context length is not saved in the model file. `Generate.cpp` constructs `NgramEngine` with the default (4), so a model trained with a different context length requires the same change in the generator. A shorter generation context than the training one silently discards the longer keys.
- Memory use grows with corpus size: the whole corpus, the BPE structures and the full n-gram table are held in memory, and each observed position can create up to `CONTEXT_LEN` entries.

**Input handling**

- `generate_text` does not validate its arguments. `K` should be ≥ 1 and `temperature` > 0; other values lead to undefined behaviour or invalid weights. Non-numeric arguments cause `std::stoul`/`std::stod`/`std::stoi` to throw an uncaught exception.
- `Preprocessor::find_rigid_symbols` computes `text.length() - 1` on an unsigned type, which underflows for an empty string. A non-empty corpus is assumed.
- `load_model` does not check stream state, so a truncated or incompatible file is not reported as an error.
- The model file is not portable across platforms with different endianness or `size_t` width.

**Text normalization and decoding**

- Only the five typographic characters listed in section 4.1 are normalized. Other non-ASCII characters are kept as is, are never detected as rigid symbols, and are not lowercased.
- The detokenizer applies a fixed list of rules. It handles spacing before closing punctuation, but not spacing after opening brackets or quotation marks, and it removes the space before every apostrophe, including opening single quotes.
- Because the starting token is random and sampling is per token, generated text can begin or end in the middle of a word.
