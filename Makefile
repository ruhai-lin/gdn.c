CC = gcc

.PHONY: all
all: run

run: run.c
	$(CC) -O3 -o run run.c -lm

.PHONY: fast
fast: run.c
	$(CC) -Ofast -march=native -o run run.c -lm

.PHONY: omp
omp: run.c
	$(CC) -Ofast -fopenmp -march=native -o run run.c -lm

.PHONY: debug
debug: run.c
	$(CC) -g -O0 -o run run.c -lm

.PHONY: clean
clean:
	rm -f run
