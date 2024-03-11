import random as rd

num_operations = int(input( ))
low = 1
high = 1000
samples = [i for i in range(low, high+1)]
print(num_operations)
for i in range(num_operations):
    op = rd.choice(['+', '?', '-'])
    num = rd.choice(samples)
    print(op, num)
