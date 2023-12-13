#!/bin/bash

# Function to run the command with different values of $f
do_job() {
    f="$1"
    mpirun -np 8 mb-mpi "$f" &
}


# Run the jobs and ensure not more than 100 processes at a time
counter=0
for f in "${files[@]}"; do
    do_job "$f"
    ((counter+=8))
    if (( counter % 100 == 0 )); then
        wait  # wait for the current batch of processes to finish before starting the next one
    fi
done

wait  # wait for the last batch of processes to finish
