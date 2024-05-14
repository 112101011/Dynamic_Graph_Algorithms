NUM_TESTS=1000
n=10000
q=1000
$(g++ serial_approach.cpp -o serial_approach.out)
$(rm time_consumed.txt)

# Loop over tests
for ((i=1; i<=NUM_TESTS; i++)); do
    # Provide input directly to generate_input.py
    echo "$n $q" | python3 generate_input.py > input.txt

    start=$(date +%s.%N)
    $(./serial_approach.out < input.txt > output.txt)
    end=$(date +%s.%N)

    runtime=$(echo "$end - $start" | bc)
    echo "$n $q $runtime" >> time_consumed.txt
    $(rm input.txt output.txt)
    q=$((q+20)) 
done