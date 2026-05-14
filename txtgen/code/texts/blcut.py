import re

def strip_gutenberg_boilerplate(text):
    start_pattern = re.compile(r"\*\*\* START OF (THIS|THE) PROJECT GUTENBERG EBOOK .* \*\*\*", re.IGNORECASE)
    end_pattern = re.compile(r"\*\*\* END OF (THIS|THE) PROJECT GUTENBERG EBOOK .* \*\*\*", re.IGNORECASE)

    lines = text.splitlines()

    start_idx = 0
    end_idx = len(lines)

    # Find start
    for i, line in enumerate(lines):
        if start_pattern.search(line):
            start_idx = i + 1
            break

    # Find end
    for i, line in enumerate(lines):
        if end_pattern.search(line):
            end_idx = i
            break

    core_lines = lines[start_idx:end_idx]
    return "\n".join(core_lines).strip()


def remove_bracketed_text(text):
    # Remove anything inside [...] including the brackets
    return re.sub(r"\[.*?\]", "", text, flags=re.DOTALL)


def clean_file(input_path, output_path):
    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()

    text = strip_gutenberg_boilerplate(text)
    text = remove_bracketed_text(text)

    # Optional: clean up extra spaces left behind
    text = re.sub(r"\n\s*\n", "\n\n", text)  # collapse multiple blank lines
    text = re.sub(r"[ \t]+", " ", text)      # normalize spaces

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(text.strip())


if __name__ == "__main__":
    input_file = input("filename: ")
    output_file = "c_" + input_file

    clean_file(input_file, output_file)
