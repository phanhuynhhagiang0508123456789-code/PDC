#!/bin/bash

THREADS=(1 4 8 16 32)
RUNS=5

N=2000
STEPS=200
DT=0.01
FREQ=1

OUTPUT="part2b_results.csv"

echo "implementation,threads,run,time_seconds" > "$OUTPUT"

run_benchmark () {
    NAME=$1
    PROGRAM=$2

    for THREAD in "${THREADS[@]}"
    do
        for RUN in $(seq 1 $RUNS)
        do
            echo "Running $NAME | threads=$THREAD | run=$RUN"

            RESULT=$($PROGRAM "$THREAD" "$N" "$STEPS" "$DT" "$FREQ" g)

            TIME=$(echo "$RESULT" | awk '/Elapsed time/ {print $4}')

            echo "$NAME,$THREAD,$RUN,$TIME" >> "$OUTPUT"
        done
    done
}

run_benchmark "Basic" "./bench_basic"
run_benchmark "Reduced_Default" "./bench_red_default"
run_benchmark "Reduced_Forces_Cyclic" "./bench_red_forces_cyclic"
run_benchmark "Reduced_All_Cyclic" "./bench_red_all_cyclic"

echo "Benchmark complete."
echo "Results saved to $OUTPUT"
