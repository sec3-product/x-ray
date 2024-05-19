#!/usr/bin/python3

import sys

if len(sys.argv) < 3:
    print("compare.py [report1] [report2]")
    sys.exit(1)

def read_data(filename):
    with open(filename, 'r') as f:
        data = f.readlines()
    data = [d.strip().split() for d in data]
    return data

report1 = read_data(sys.argv[1])
report2 = read_data(sys.argv[2])

for i in range(len(report1) - 4):
    #print(i, report1[i][0])
    old = report1[i][1]
    new = report2[i][1]
    if old != new:
        print(report1[i][0], old, new)

