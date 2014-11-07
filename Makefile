##########################################################################
# Makefile for sparse pegasos
##########################################################################

HDR = simple_sparse_vec_hash.h WeightVector.h pegasos_optimize.h
SRC = simple_sparse_vec_hash.cc  WeightVector.cc pegasos_optimize.cc cmd_line.cc

CC = g++

CC      = g++
CFLAGS  = -Wall -O3 
#CFLAGS  = -g 
LFLAGS  = -lm

OBJS = $(SRC:.cc=.o)

all: pegasos test_objective

test_objective: $(OBJS) test_objective.o
	$(CC) $(OBJS) test_objective.o $(LFLAGS) -o test_objective

pegasos: $(OBJS) main.o
	$(CC) $(OBJS) main.o $(LFLAGS) -o pegasos

tar: $(SRC) $(HDR) Makefile 
	tar zcvf pegasos.tgz *.cc *.h Makefile license.txt README data.tgz

%.o: %.cc $(HDR)
	$(CC) $(CFLAGS) -c $<

clean:
	rm -f *.o *.od *.oc *~ \#*\# depend pegasos pegasos.exe test_objective.exe
