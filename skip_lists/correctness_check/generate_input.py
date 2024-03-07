import random as rd

# insertions with proabability 0.8
# deletions with probability 0.2
# num_operations = int(input("Enter number of operations = " ))
num_operations = 100000
low = 1
high = 100
samples = [i for i in range(low, high+1)]
print(num_operations)
for i in range(num_operations):
    op = rd.choice(['+', '-', '?'])
    num = rd.choice(samples)
    print(op, num)