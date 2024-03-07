	INCLUDE=-I../
all : 
	python3 generate_input.py > input.txt
	g++ ${INCLUDE} skip_list_check.cpp -o ./skip_list_check.out
	./skip_list_check.out < input.txt > output.txt
	cat output.txt
	# rm skip_list_check.out input.txt
