
import os
import glob
import json
import numpy as np
import argparse

def compute_stats(input_dir):
    """Read all *_easyocr.json files in input_dir and return lists of line and word counts."""
    line_counts, word_counts = [], []
    pattern = os.path.join(input_dir, "*_easyocr.json")
    for path in glob.glob(pattern):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        rec = data["recognitionResults"][0]
        line_counts.append(len(rec["lines"]))
        word_counts.append(len(rec["words"]))
    return line_counts, word_counts

def summary(arr):
    """Compute summary statistics for a numeric list."""
    return {
        "mean": np.mean(arr),
        "median": np.median(arr),
        "std": np.std(arr),
        "p10": np.percentile(arr, 10),
        "p90": np.percentile(arr, 90),
    }

def bucket_hist(arr, bins, labels):
    """Count how many values fall into each bin."""
    hist = {label: 0 for label in labels}
    for cnt in arr:
        for i in range(len(bins) - 1):
            if bins[i] <= cnt < bins[i+1]:
                hist[labels[i]] += 1
                break
    return hist

def main():
    parser = argparse.ArgumentParser(description="Compute OCR dataset statistics")
    parser.add_argument("input_dir", help="Directory containing *_easyocr.json files")
    args = parser.parse_args()

    line_counts, word_counts = compute_stats(args.input_dir)

    # 1. Summary Statistics
    print("=== SUMMARY STATISTICS ===")
    for name, arr in [("Lines", line_counts), ("Words", word_counts)]:
        stats = summary(arr)
        print(f"\n{name} per document:")
        print(f"  Mean   = {stats['mean']:.1f}")
        print(f"  Median = {stats['median']:.1f}")
        print(f"  Std    = {stats['std']:.1f}")
        print(f"  10th   = {stats['p10']:.0f}")
        print(f"  90th   = {stats['p90']:.0f}")

    # 2. Bucketed Distribution
    bins = [0, 10, 30, 60, 100, 200, 500, 1000, float('inf')]
    labels = ["0-10", "10-30", "30-60", "60-100", "100-200", "200-500", "500-1000", "1000+"]
    line_hist = bucket_hist(line_counts, bins, labels)
    word_hist = bucket_hist(word_counts, bins, labels)

    print("\n=== LINE COUNT DISTRIBUTION ===")
    for lbl in labels:
        print(f" {lbl:8s}: {line_hist[lbl]} documents")

    print("\n=== WORD COUNT DISTRIBUTION ===")
    for lbl in labels:
        print(f" {lbl:8s}: {word_hist[lbl]} documents")

if __name__ == "__main__":
    main()

