#!/bin/bash
LOCAL_NODE=0
CXL_NODE=3  # Node 3 in my machine is a CXL memory node, while the other is a CPU node.
numa_nodes="$LOCAL_NODE,$CXL_NODE"

ITERATION_PER_BENCH=10
# benchmark_element_items=("1835008" "12845056" "67895296" "544997376")    # This is the possible size for Qwen2.5 7B, excluding sizes that are too small.
benchmark_element_items=("5242880" "20971520" "73400320" "671088640")    # This is the possible size for Mistral Nemo 12B, excluding sizes that are too small.
ngpus_items=("1" "2")
numa_weight_items=("1,1" "1,2" "1,3" "1,4") # This specifies weights for memory interleaving: local_node:weight1, cxl_node:weight2

echo "ngpus,numa_nodes,numa_weight,update_element,avg_h2d_latency(ms),avg_compute_latency(ms),avg_d2h_latency(ms)"

for ngpus in "${ngpus_items[@]}"
do
    for numa_weight in "${numa_weight_items[@]}"
    do
        local_node_weight=$(echo "$numa_weight" | awk -F',' '{print $1}')
        cxl_node_weight=$(echo "$numa_weight" | awk -F',' '{print $2}')

        echo "$local_node_weight" > /sys/kernel/mm/mempolicy/weighted_interleave/node$LOCAL_NODE || {
            echo "Error: Failed to set weight for node $LOCAL_NODE"
            exit 1
        }
        echo "$cxl_node_weight" > /sys/kernel/mm/mempolicy/weighted_interleave/node$CXL_NODE || {
            echo "Error: Failed to set weight for node $CXL_NODE"
            exit 1
        }

        for benchmark_element in "${benchmark_element_items[@]}"
        do
            output=$(numactl --weighted-interleave="$numa_nodes" python mp_bench.py \
                --nprocess "$ngpus" \
                --param_size "$benchmark_element" \
                --num_bench "$ITERATION_PER_BENCH" \
            )

            avg_h2d_latency=$(echo "$output" | grep "Average H2D Latency per step:" | awk '{print $6}')
            avg_compute_latency=$(echo "$output" | grep "Average Computation Latency per step:" | awk '{print $6}')
            avg_d2h_latency=$(echo "$output" | grep "Average D2H Latency per step:" | awk '{print $6}')

            echo "$ngpus,\"$numa_nodes\",\"$numa_weight\",$benchmark_element,$avg_h2d_latency,$avg_compute_latency,$avg_d2h_latency"
        done
    done
done