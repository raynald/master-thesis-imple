#!/usr/bin/python
import sys
import random

if __name__ == '__main__':
    sample = []
    for line in sys.stdin:
        line = line.strip()
        sample += [line]
    random.shuffle(sample)
    for i in range(len(sample)/10):
        print sample[i]

