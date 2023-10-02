#!/bin/bash

# Function to run the command with different values of $f
do_job() {
    f="$1"
    mpirun -np 8 mb-mpi "$f" &
}

# An array of your files
files=(
    "constenlachibchan_combined.mb.nex"
    "dravlex_combined.mb.nex"
    "hattorijaponic_combined.mb.nex"
    "leekoreanic_combined.mb.nex"
    "walworthpolynesian_combined.mb.nex"
    "crossandean_combined.mb.nex"
    "felekesemitic_combined.mb.nex"
    "houchinese_combined.mb.nex"
    "robinsonap_combined.mb.nex"
    "zhivlovobugrian_combined.mb.nex"
)

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
