	INCLUDE=-I../
all : 
	python3 generate_input.py > input.txt
	g++ ${INCLUDE} skip_list_check.cpp -o ./skip_list_check.out
	./skip_list_check.out < input.txt > skip_list_output.txt
	g++ set_code.cpp -o set_code.out
	./set_code.out < input.txt > set_output.txt
	rm set_code.out skip_list_check.out
	diff set_output.txt skip_list_output.txt
