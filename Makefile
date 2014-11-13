FLAGS= `pkg-config --cflags --libs opencv`
CC=g++
all: 
	$(CC) circle_detection.cpp -o a $(FLAGS)
	
clean:
	rm -f a

