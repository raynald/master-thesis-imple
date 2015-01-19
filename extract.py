#!/usr/bin/python
import sys
import random

if __name__ == '__main__':
    random.seed(2015)
    sample = []
    for line in sys.stdin:
        line = line.strip()
        sample += [line]
    random.shuffle(sample)
    for i in range(len(sample)/10+1, len(sample)/10+len(sample)/10*5):
        print sample[i]

