import os
import subprocess
import time
import glob

input_dir = "data/input"
output_dir = "data/output"

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

scores = []
times = []

# Get all .txt files in data/input
input_files = glob.glob(os.path.join(input_dir, "*.txt"))

input_files.sort()

total_files = len(input_files)
print(f"Found {total_files} input file(s).")

for idx, input_file in enumerate(input_files, start=1):
    base_name = os.path.basename(input_file)
    output_file = os.path.join(output_dir, base_name)
    print(f"[{idx}/{total_files}] Processing {base_name}...")

    start_time = time.time()
    with open(input_file, "r") as infile:
        result = subprocess.run(
            ["python", "ver1.py"],
            stdin=infile,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
        )
    end_time = time.time()
    elapsed = end_time - start_time
    times.append(elapsed)

    with open(output_file, "w") as f:
        f.write(result.stdout)

    try:
        # Expect stderr like: "score=123.45"
        stderr_line = result.stderr.strip()
        if stderr_line.startswith("score="):
            score_str = stderr_line.split("=")[1]
            score = float(score_str)
            scores.append(score)
            print(f"  {base_name}: Score = {score}, Time = {elapsed:.4f} sec")
        else:
            raise ValueError("Unexpected stderr format")
    except Exception:
        print(
            f"Warning: Could not parse score from {base_name} stderr. Time = {elapsed:.4f} sec"
        )

    print(f"[{idx}/{total_files}] Finished processing {base_name}.\n")

avg_score = sum(scores) / len(scores) if scores else 0.0
avg_time = sum(times) / len(times) if times else 0.0
max_time = max(times) if times else 0.0

print("Final Score Average:", avg_score)
print("Execution Time Average: {:.4f} seconds".format(avg_time))
print("Maximum Execution Time: {:.4f} seconds".format(max_time))
