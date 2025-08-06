#!/bin/bash

# Scripts to benchmark
scripts=("discrete/main.py" "voronoi/main.py" "voronoi_union/main.py")

# Values for N and seeds
nodes=(10 50 100 200 300 400 500 600 700)

# Output file
output_file="benchmark_results.txt"
> "$output_file"  # Clear previous content

# Header
echo "N script avg_runtime secondary_values" >> "$output_file"

# Loop over N and each script
for N in "${nodes[@]}"; do
  for script in "${scripts[@]}"; do
    total_time=0
    count=0
    secondary_values=()

    echo "Running: $N with $script"

    for seed in "${nodes[@]}"; do
      # Run the script and capture both runtime and secondary metric
      output=$(python "$script" "$N" "$seed")
      
      # Parse two values from output
      runtime=$(echo "$output" | awk '{print $1}')
      secondary=$(echo "$output" | awk '{print $2}')
      
      # Accumulate runtime
      total_time=$(echo "$total_time + $runtime" | bc)
      ((count++))

      # Save secondary value
      secondary_values=("$secondary")
    done

    # Compute average runtime
    avg_time=$(echo "scale=5; $total_time / $count" | bc)

    # Join all secondary values into space-separated string
    secondary_str=$(IFS=','; echo "${secondary_values[*]}")

    # Write line to output
    echo "$N $script $avg_time $secondary_str" >> "$output_file"
  done
done

echo "Benchmarking complete. Results saved in $output_file"
