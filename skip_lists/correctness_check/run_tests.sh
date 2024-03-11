#!/bin/bash

# Compile 1.cpp and 2.cpp
g++ -I../ -o 1 skip_list_check.cpp
g++ -o 2 set_code.cpp

$(rm skip_list_time.txt set_time.txt)

# Number of tests
NUM_TESTS=1000

# Loop over tests
for ((i=1; i<=NUM_TESTS; i++)); do
    # Generate input using g.py
    echo "$i" >> temp.txt;
    input=$(python3 generate_input.py < temp.txt)

    # Measure time taken by 1
    start=$(date +%s.%N)
    echo "$input" | ./1 > /dev/null
    end1=$(date +%s.%N)

    # Measure time taken by 2
    start2=$(date +%s.%N)
    echo "$input" | ./2 > /dev/null
    end2=$(date +%s.%N)

    # Calculate time taken
    runtime1=$(echo "$end1 - $start" | bc)
    runtime2=$(echo "$end2 - $start2" | bc)

    echo "$i $runtime1" >> skip_list_time.txt
    echo "$i $runtime2" >> set_time.txt
    $(rm temp.txt)
done

$(rm 1 2)
