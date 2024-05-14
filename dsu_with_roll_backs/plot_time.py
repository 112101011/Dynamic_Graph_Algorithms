import matplotlib.pyplot as plt
file_path = "time_consumed.txt"

x_values = []
y_values = []

# Open the file in read mode ('r' is the default mode, so you can omit it)
with open(file_path) as file:
    # Read the entire file content
    for line in file:
        n, q, time = line.split()
        n = int(n)
        q = int(q)
        time = float(time)
        x_values.append(q)
        y_values.append(time)
 
plt.plot(x_values, y_values, label = "Time consumed using serial approach")
plt.xlabel("Number of operations(q) ->")
plt.ylabel("Time consumed")
plt.title("Time plot (n = {})".format(n))
plt.legend()
plt.show()

