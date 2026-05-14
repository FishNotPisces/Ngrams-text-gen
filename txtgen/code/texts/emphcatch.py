import os
import re

def remove_emphasis_underscores(text):
    # Replace _word_ with word (keeps the inner content)
    return re.sub(r"\b_(.*?)_\b", r"\1", text)

def process_files_in_place(folder):
    files = [
        f for f in os.listdir(folder)
        if f.startswith("c_") and f.endswith(".txt")
    ]

    for filename in files:
        path = os.path.join(folder, filename)

        with open(path, "r", encoding="utf-8") as f:
            text = f.read()

        cleaned = remove_emphasis_underscores(text)

        # Overwrite the same file
        with open(path, "w", encoding="utf-8") as f:
            f.write(cleaned)

        print(f"Processed {filename}")

if __name__ == "__main__":
    process_files_in_place(".")  # current directory
