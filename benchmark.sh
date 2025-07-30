#!/bin/bash

# Scripts to benchmark
scripts=("discrete/main.py" "voronoi/main.py" "voronoi_union/main.py")

# Values for N and seeds
nodes=(10 50 100 200)

# Output file
output_file="benchmark_results.txt"
> "$output_file"  # Clear the file

# Header for the results file
echo "N script avg_runtime" >> "$output_file"

# Loop over N and each script
for N in "${nodes[@]}"; do
  for script in "${scripts[@]}"; do
    total_time=0
    count=0

    # Run for all seeds
    for seed in "${nodes[@]}"; do
      # Capture the runtime (assumes script prints runtime in seconds)
      runtime=$(python3 "$script" "$N" "$seed")

      # Accumulate runtime
      total_time=$(echo "$total_time + $runtime" | bc)
      ((count++))
    done

    # Compute average time for this N and script
    avg_time=$(echo "scale=5; $total_time / $count" | bc)

    # Save results (N, script name, avg runtime)
    echo "$N $script $avg_time" >> "$output_file"
  done
done

echo "Benchmarking complete. Results saved in $output_file"
