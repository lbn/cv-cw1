FLAGS= `pkg-config --cflags --libs opencv`
CC=g++
all: 
	$(CC) sobel.cpp -o a $(FLAGS)
	
clean:
	rm -f a

