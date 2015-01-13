##########################################################################
# Makefile for sparse pegasos
##########################################################################

HDR = include/simple_sparse_vec_hash.h include/WeightVector.h include/optimize.h
SRC = simple_sparse_vec_hash.cc  WeightVector.cc optimize.cc cmd_line.cc

CC = g++ 

CFLAGS  = -Wall -O3 
#CFLAGS  = -g 
LFLAGS  = -lm

OBJS = $(SRC:.cc=.o)

all: optimize 
#test_objective

#test_objective: $(OBJS) test_objective.o
#	$(CC) $(OBJS) test_objective.o $(LFLAGS) -o test_objective

optimize: $(OBJS) main.o
	$(CC) -std=c++11 $(OBJS) main.o $(LFLAGS) -o optimize 

tar: $(SRC) $(HDR) Makefile 
	tar zcvf pegasos.tgz *.cc *.h Makefile license.txt README data.tgz

%.o: %.cc $(HDR)
	$(CC) -std=c++11 $(CFLAGS) -c $<

clean:
	rm -f *.o *.od *.oc *~ \#*\# depend optimize
