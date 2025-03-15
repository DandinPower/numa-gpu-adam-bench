#!/bin/bash

ITERATION_PER_BENCH=10

# benchmark_element_items=("1835008" "12845056" "67895296" "544997376")    # This is the possible size for Qwen2.5 7B, excluding sizes that are too small.
benchmark_element_items=("5242880" "20971520" "73400320" "671088640")    # This is the possible size for Mistral Nemo 12B, excluding sizes that are too small.
ngpus_items=("1" "2")
numa_nodes_items=("0" "3")  # Node 3 in my machine is a CXL memory node, while the other is a CPU node.

echo "ngpus,numa_nodes,update_element,avg_h2d_latency(ms),avg_compute_latency(ms),avg_d2h_latency(ms)"

for ngpus in "${ngpus_items[@]}"
do
    for numa_nodes in "${numa_nodes_items[@]}"
    do
        for benchmark_element in "${benchmark_element_items[@]}"
        do
            output=$(numactl --interleave="$numa_nodes" python mp_bench.py \
                --nprocess "$ngpus" \
                --param_size "$benchmark_element" \
                --num_bench "$ITERATION_PER_BENCH" \
            )

            avg_h2d_latency=$(echo "$output" | grep "Average H2D Latency per step:" | awk '{print $6}')
            avg_compute_latency=$(echo "$output" | grep "Average Computation Latency per step:" | awk '{print $6}')
            avg_d2h_latency=$(echo "$output" | grep "Average D2H Latency per step:" | awk '{print $6}')

            echo "$ngpus,\"$numa_nodes\",$benchmark_element,$avg_h2d_latency,$avg_compute_latency,$avg_d2h_latency"
        done
    done
done