import os

def combine_chapter_files(input_folder, output_file):
    print("Working in:", os.getcwd())

    files = [
        f for f in os.listdir(input_folder)
        if f.lower().startswith("c_") and f.lower().endswith(".txt")
    ]

    if not files:
        print("No files matched.")
        return

    files.sort()
    print("Merging:", files)

    with open(output_file, "w", encoding="utf-8") as outfile:
        for filename in files:
            path = os.path.join(input_folder, filename)
            with open(path, "r", encoding="utf-8") as infile:
                outfile.write(infile.read())
                outfile.write("\n\n")

    print("Done.")

if __name__ == "__main__":
    combine_chapter_files(".", "combined.txt")
