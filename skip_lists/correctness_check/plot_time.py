import matplotlib.pyplot as plt
file_path = "skip_list_time.txt"

x_values = []
y1_values = []
y2_values = []
# Open the file in read mode ('r' is the default mode, so you can omit it)
with open(file_path) as file:
    # Read the entire file content
    for line in file:
    	n, time = line.split()
    	n = int(n)
    	time = float(time)
    	x_values.append(n)
    	y1_values.append(time)
    	
file_path = "set_time.txt"

# Open the file in read mode ('r' is the default mode, so you can omit it)
with open(file_path) as file:
    # Read the entire file content
    for line in file:
    	n, time = line.split()
    	n = int(n)
    	time = float(time)
    	y2_values.append(time)
    	
 
plt.plot(x_values, y1_values, label = "Time consumed using skip_list");
plt.plot(x_values, y2_values, label = "Time consumed using set of c++ library");
plt.xlabel("Number of operations ->")
plt.ylabel("Time consumed")
plt.title("Time comparision")
plt.legend()
plt.show();

