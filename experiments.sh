#!/bin/bash
# the 1 is the file name and 2 is the file line
line_idx=$(($2 + 1))





FILE_LINE=$(sed -n "${line_idx}p" ${1})

/home/hzhang747/miniconda3/envs/spf/bin/python3 experiments.py ${FILE_LINE}