#!/bin/bash

## Parameters (No space!!!)
start_index=0
subj_num=2
percentage_value=0.7
refer_subj_num=10
refer_str_num=10
sigma_value=0.9

#CUDA_VISIBLE_DEVICES=${cuda_id} python3 evaluate_main.py --start_index 0 --subj_num ${subj_num} --cluster_num ${cluster_num} --percentage_value ${percentage_value} --refer_subj_num ${refer_subj_num} --refer_str_num ${refer_str_num} --sigma_value ${sigma_value} --cuda_id ${cuda_id} &
python3 evaluate_main.py --start_index 0 --subj_num ${subj_num} --percentage_value ${percentage_value} --refer_subj_num ${refer_subj_num} --refer_str_num ${refer_str_num} --sigma_value ${sigma_value} &
python3 evaluate_main.py --start_index 2 --subj_num ${subj_num} --percentage_value ${percentage_value} --refer_subj_num ${refer_subj_num} --refer_str_num ${refer_str_num} --sigma_value ${sigma_value} &
python3 evaluate_main.py --start_index 4 --subj_num ${subj_num} --percentage_value ${percentage_value} --refer_subj_num ${refer_subj_num} --refer_str_num ${refer_str_num} --sigma_value ${sigma_value} &
python3 evaluate_main.py --start_index 6 --subj_num ${subj_num} --percentage_value ${percentage_value} --refer_subj_num ${refer_subj_num} --refer_str_num ${refer_str_num} --sigma_value ${sigma_value} &
python3 evaluate_main.py --start_index 8 --subj_num ${subj_num} --percentage_value ${percentage_value} --refer_subj_num ${refer_subj_num} --refer_str_num ${refer_str_num} --sigma_value ${sigma_value} &
python3 evaluate_main.py --start_index 10 --subj_num ${subj_num} --percentage_value ${percentage_value} --refer_subj_num ${refer_subj_num} --refer_str_num ${refer_str_num} --sigma_value ${sigma_value} &
python3 evaluate_main.py --start_index 12 --subj_num ${subj_num} --percentage_value ${percentage_value} --refer_subj_num ${refer_subj_num} --refer_str_num ${refer_str_num} --sigma_value ${sigma_value} &
python3 evaluate_main.py --start_index 14 --subj_num ${subj_num} --percentage_value ${percentage_value} --refer_subj_num ${refer_subj_num} --refer_str_num ${refer_str_num} --sigma_value ${sigma_value} &
python3 evaluate_main.py --start_index 16 --subj_num ${subj_num} --percentage_value ${percentage_value} --refer_subj_num ${refer_subj_num} --refer_str_num ${refer_str_num} --sigma_value ${sigma_value} &
python3 evaluate_main.py --start_index 18 --subj_num ${subj_num} --percentage_value ${percentage_value} --refer_subj_num ${refer_subj_num} --refer_str_num ${refer_str_num} --sigma_value ${sigma_value} ;
python3 evaluate_main.py --start_index 20 --subj_num ${subj_num} --percentage_value ${percentage_value} --refer_subj_num ${refer_subj_num} --refer_str_num ${refer_str_num} --sigma_value ${sigma_value} &
python3 evaluate_main.py --start_index 22 --subj_num ${subj_num} --percentage_value ${percentage_value} --refer_subj_num ${refer_subj_num} --refer_str_num ${refer_str_num} --sigma_value ${sigma_value} &
python3 evaluate_main.py --start_index 24 --subj_num ${subj_num} --percentage_value ${percentage_value} --refer_subj_num ${refer_subj_num} --refer_str_num ${refer_str_num} --sigma_value ${sigma_value} &
python3 evaluate_main.py --start_index 26 --subj_num ${subj_num} --percentage_value ${percentage_value} --refer_subj_num ${refer_subj_num} --refer_str_num ${refer_str_num} --sigma_value ${sigma_value} &
python3 evaluate_main.py --start_index 28 --subj_num ${subj_num} --percentage_value ${percentage_value} --refer_subj_num ${refer_subj_num} --refer_str_num ${refer_str_num} --sigma_value ${sigma_value} &
python3 evaluate_main.py --start_index 30 --subj_num ${subj_num} --percentage_value ${percentage_value} --refer_subj_num ${refer_subj_num} --refer_str_num ${refer_str_num} --sigma_value ${sigma_value} &
python3 evaluate_main.py --start_index 32 --subj_num ${subj_num} --percentage_value ${percentage_value} --refer_subj_num ${refer_subj_num} --refer_str_num ${refer_str_num} --sigma_value ${sigma_value} &
python3 evaluate_main.py --start_index 34 --subj_num ${subj_num} --percentage_value ${percentage_value} --refer_subj_num ${refer_subj_num} --refer_str_num ${refer_str_num} --sigma_value ${sigma_value} &
python3 evaluate_main.py --start_index 36 --subj_num ${subj_num} --percentage_value ${percentage_value} --refer_subj_num ${refer_subj_num} --refer_str_num ${refer_str_num} --sigma_value ${sigma_value} &
python3 evaluate_main.py --start_index 38 --subj_num ${subj_num} --percentage_value ${percentage_value} --refer_subj_num ${refer_subj_num} --refer_str_num ${refer_str_num} --sigma_value ${sigma_value} ;
python3 evaluate_main.py --start_index 40 --subj_num ${subj_num} --percentage_value ${percentage_value} --refer_subj_num ${refer_subj_num} --refer_str_num ${refer_str_num} --sigma_value ${sigma_value} &
python3 evaluate_main.py --start_index 42 --subj_num ${subj_num} --percentage_value ${percentage_value} --refer_subj_num ${refer_subj_num} --refer_str_num ${refer_str_num} --sigma_value ${sigma_value} &
python3 evaluate_main.py --start_index 44 --subj_num ${subj_num} --percentage_value ${percentage_value} --refer_subj_num ${refer_subj_num} --refer_str_num ${refer_str_num} --sigma_value ${sigma_value} &
python3 evaluate_main.py --start_index 46 --subj_num ${subj_num} --percentage_value ${percentage_value} --refer_subj_num ${refer_subj_num} --refer_str_num ${refer_str_num} --sigma_value ${sigma_value} &
python3 evaluate_main.py --start_index 48 --subj_num ${subj_num} --percentage_value ${percentage_value} --refer_subj_num ${refer_subj_num} --refer_str_num ${refer_str_num} --sigma_value ${sigma_value} &
python3 evaluate_main.py --start_index 50 --subj_num ${subj_num} --percentage_value ${percentage_value} --refer_subj_num ${refer_subj_num} --refer_str_num ${refer_str_num} --sigma_value ${sigma_value} &
python3 evaluate_main.py --start_index 52 --subj_num ${subj_num} --percentage_value ${percentage_value} --refer_subj_num ${refer_subj_num} --refer_str_num ${refer_str_num} --sigma_value ${sigma_value} &
python3 evaluate_main.py --start_index 54 --subj_num ${subj_num} --percentage_value ${percentage_value} --refer_subj_num ${refer_subj_num} --refer_str_num ${refer_str_num} --sigma_value ${sigma_value} &
python3 evaluate_main.py --start_index 56 --subj_num ${subj_num} --percentage_value ${percentage_value} --refer_subj_num ${refer_subj_num} --refer_str_num ${refer_str_num} --sigma_value ${sigma_value} &
python3 evaluate_main.py --start_index 58 --subj_num ${subj_num} --percentage_value ${percentage_value} --refer_subj_num ${refer_subj_num} --refer_str_num ${refer_str_num} --sigma_value ${sigma_value} ;
python3 evaluate_main.py --start_index 60 --subj_num ${subj_num} --percentage_value ${percentage_value} --refer_subj_num ${refer_subj_num} --refer_str_num ${refer_str_num} --sigma_value ${sigma_value} &
python3 evaluate_main.py --start_index 62 --subj_num ${subj_num} --percentage_value ${percentage_value} --refer_subj_num ${refer_subj_num} --refer_str_num ${refer_str_num} --sigma_value ${sigma_value} &
python3 evaluate_main.py --start_index 64 --subj_num ${subj_num} --percentage_value ${percentage_value} --refer_subj_num ${refer_subj_num} --refer_str_num ${refer_str_num} --sigma_value ${sigma_value} &
python3 evaluate_main.py --start_index 66 --subj_num ${subj_num} --percentage_value ${percentage_value} --refer_subj_num ${refer_subj_num} --refer_str_num ${refer_str_num} --sigma_value ${sigma_value} &
python3 evaluate_main.py --start_index 68 --subj_num ${subj_num} --percentage_value ${percentage_value} --refer_subj_num ${refer_subj_num} --refer_str_num ${refer_str_num} --sigma_value ${sigma_value} &
python3 evaluate_main.py --start_index 70 --subj_num ${subj_num} --percentage_value ${percentage_value} --refer_subj_num ${refer_subj_num} --refer_str_num ${refer_str_num} --sigma_value ${sigma_value} &
python3 evaluate_main.py --start_index 72 --subj_num ${subj_num} --percentage_value ${percentage_value} --refer_subj_num ${refer_subj_num} --refer_str_num ${refer_str_num} --sigma_value ${sigma_value} &
python3 evaluate_main.py --start_index 74 --subj_num ${subj_num} --percentage_value ${percentage_value} --refer_subj_num ${refer_subj_num} --refer_str_num ${refer_str_num} --sigma_value ${sigma_value} &
python3 evaluate_main.py --start_index 76 --subj_num ${subj_num} --percentage_value ${percentage_value} --refer_subj_num ${refer_subj_num} --refer_str_num ${refer_str_num} --sigma_value ${sigma_value} &
python3 evaluate_main.py --start_index 78 --subj_num ${subj_num} --percentage_value ${percentage_value} --refer_subj_num ${refer_subj_num} --refer_str_num ${refer_str_num} --sigma_value ${sigma_value} ;
python3 evaluate_main.py --start_index 80 --subj_num ${subj_num} --percentage_value ${percentage_value} --refer_subj_num ${refer_subj_num} --refer_str_num ${refer_str_num} --sigma_value ${sigma_value} &
python3 evaluate_main.py --start_index 82 --subj_num ${subj_num} --percentage_value ${percentage_value} --refer_subj_num ${refer_subj_num} --refer_str_num ${refer_str_num} --sigma_value ${sigma_value} &
python3 evaluate_main.py --start_index 84 --subj_num ${subj_num} --percentage_value ${percentage_value} --refer_subj_num ${refer_subj_num} --refer_str_num ${refer_str_num} --sigma_value ${sigma_value} &
python3 evaluate_main.py --start_index 86 --subj_num ${subj_num} --percentage_value ${percentage_value} --refer_subj_num ${refer_subj_num} --refer_str_num ${refer_str_num} --sigma_value ${sigma_value} &
python3 evaluate_main.py --start_index 88 --subj_num ${subj_num} --percentage_value ${percentage_value} --refer_subj_num ${refer_subj_num} --refer_str_num ${refer_str_num} --sigma_value ${sigma_value} &
python3 evaluate_main.py --start_index 90 --subj_num ${subj_num} --percentage_value ${percentage_value} --refer_subj_num ${refer_subj_num} --refer_str_num ${refer_str_num} --sigma_value ${sigma_value} &
python3 evaluate_main.py --start_index 92 --subj_num ${subj_num} --percentage_value ${percentage_value} --refer_subj_num ${refer_subj_num} --refer_str_num ${refer_str_num} --sigma_value ${sigma_value} &
python3 evaluate_main.py --start_index 94 --subj_num ${subj_num} --percentage_value ${percentage_value} --refer_subj_num ${refer_subj_num} --refer_str_num ${refer_str_num} --sigma_value ${sigma_value} &
python3 evaluate_main.py --start_index 96 --subj_num ${subj_num} --percentage_value ${percentage_value} --refer_subj_num ${refer_subj_num} --refer_str_num ${refer_str_num} --sigma_value ${sigma_value} &
python3 evaluate_main.py --start_index 98 --subj_num ${subj_num} --percentage_value ${percentage_value} --refer_subj_num ${refer_subj_num} --refer_str_num ${refer_str_num} --sigma_value ${sigma_value} ;
python3 evaluate_main.py --start_index 100 --subj_num ${subj_num} --percentage_value ${percentage_value} --refer_subj_num ${refer_subj_num} --refer_str_num ${refer_str_num} --sigma_value ${sigma_value} &
python3 evaluate_main.py --start_index 102 --subj_num ${subj_num} --percentage_value ${percentage_value} --refer_subj_num ${refer_subj_num} --refer_str_num ${refer_str_num} --sigma_value ${sigma_value} &
python3 evaluate_main.py --start_index 104 --subj_num ${subj_num} --percentage_value ${percentage_value} --refer_subj_num ${refer_subj_num} --refer_str_num ${refer_str_num} --sigma_value ${sigma_value} &
python3 evaluate_main.py --start_index 106 --subj_num ${subj_num} --percentage_value ${percentage_value} --refer_subj_num ${refer_subj_num} --refer_str_num ${refer_str_num} --sigma_value ${sigma_value} &
python3 evaluate_main.py --start_index 108 --subj_num ${subj_num} --percentage_value ${percentage_value} --refer_subj_num ${refer_subj_num} --refer_str_num ${refer_str_num} --sigma_value ${sigma_value} &
python3 evaluate_main.py --start_index 110 --subj_num ${subj_num} --percentage_value ${percentage_value} --refer_subj_num ${refer_subj_num} --refer_str_num ${refer_str_num} --sigma_value ${sigma_value} &
python3 evaluate_main.py --start_index 112 --subj_num ${subj_num} --percentage_value ${percentage_value} --refer_subj_num ${refer_subj_num} --refer_str_num ${refer_str_num} --sigma_value ${sigma_value} &
python3 evaluate_main.py --start_index 114 --subj_num ${subj_num} --percentage_value ${percentage_value} --refer_subj_num ${refer_subj_num} --refer_str_num ${refer_str_num} --sigma_value ${sigma_value} &
python3 evaluate_main.py --start_index 116 --subj_num ${subj_num} --percentage_value ${percentage_value} --refer_subj_num ${refer_subj_num} --refer_str_num ${refer_str_num} --sigma_value ${sigma_value} &
python3 evaluate_main.py --start_index 118 --subj_num ${subj_num} --percentage_value ${percentage_value} --refer_subj_num ${refer_subj_num} --refer_str_num ${refer_str_num} --sigma_value ${sigma_value} ;
python3 evaluate_main.py --start_index 120 --subj_num ${subj_num} --percentage_value ${percentage_value} --refer_subj_num ${refer_subj_num} --refer_str_num ${refer_str_num} --sigma_value ${sigma_value} &
python3 evaluate_main.py --start_index 122 --subj_num ${subj_num} --percentage_value ${percentage_value} --refer_subj_num ${refer_subj_num} --refer_str_num ${refer_str_num} --sigma_value ${sigma_value} &
python3 evaluate_main.py --start_index 124 --subj_num ${subj_num} --percentage_value ${percentage_value} --refer_subj_num ${refer_subj_num} --refer_str_num ${refer_str_num} --sigma_value ${sigma_value} &
python3 evaluate_main.py --start_index 126 --subj_num ${subj_num} --percentage_value ${percentage_value} --refer_subj_num ${refer_subj_num} --refer_str_num ${refer_str_num} --sigma_value ${sigma_value} &
python3 evaluate_main.py --start_index 128 --subj_num ${subj_num} --percentage_value ${percentage_value} --refer_subj_num ${refer_subj_num} --refer_str_num ${refer_str_num} --sigma_value ${sigma_value} &
python3 evaluate_main.py --start_index 130 --subj_num ${subj_num} --percentage_value ${percentage_value} --refer_subj_num ${refer_subj_num} --refer_str_num ${refer_str_num} --sigma_value ${sigma_value} &
python3 evaluate_main.py --start_index 132 --subj_num ${subj_num} --percentage_value ${percentage_value} --refer_subj_num ${refer_subj_num} --refer_str_num ${refer_str_num} --sigma_value ${sigma_value} &
python3 evaluate_main.py --start_index 134 --subj_num ${subj_num} --percentage_value ${percentage_value} --refer_subj_num ${refer_subj_num} --refer_str_num ${refer_str_num} --sigma_value ${sigma_value} &
python3 evaluate_main.py --start_index 136 --subj_num ${subj_num} --percentage_value ${percentage_value} --refer_subj_num ${refer_subj_num} --refer_str_num ${refer_str_num} --sigma_value ${sigma_value} &
python3 evaluate_main.py --start_index 138 --subj_num ${subj_num} --percentage_value ${percentage_value} --refer_subj_num ${refer_subj_num} --refer_str_num ${refer_str_num} --sigma_value ${sigma_value} ;
python3 evaluate_main.py --start_index 140 --subj_num ${subj_num} --percentage_value ${percentage_value} --refer_subj_num ${refer_subj_num} --refer_str_num ${refer_str_num} --sigma_value ${sigma_value} &
python3 evaluate_main.py --start_index 142 --subj_num ${subj_num} --percentage_value ${percentage_value} --refer_subj_num ${refer_subj_num} --refer_str_num ${refer_str_num} --sigma_value ${sigma_value} &
python3 evaluate_main.py --start_index 144 --subj_num ${subj_num} --percentage_value ${percentage_value} --refer_subj_num ${refer_subj_num} --refer_str_num ${refer_str_num} --sigma_value ${sigma_value} &
python3 evaluate_main.py --start_index 146 --subj_num ${subj_num} --percentage_value ${percentage_value} --refer_subj_num ${refer_subj_num} --refer_str_num ${refer_str_num} --sigma_value ${sigma_value} &
python3 evaluate_main.py --start_index 148 --subj_num ${subj_num} --percentage_value ${percentage_value} --refer_subj_num ${refer_subj_num} --refer_str_num ${refer_str_num} --sigma_value ${sigma_value} &
python3 evaluate_main.py --start_index 150 --subj_num ${subj_num} --percentage_value ${percentage_value} --refer_subj_num ${refer_subj_num} --refer_str_num ${refer_str_num} --sigma_value ${sigma_value} &
python3 evaluate_main.py --start_index 152 --subj_num ${subj_num} --percentage_value ${percentage_value} --refer_subj_num ${refer_subj_num} --refer_str_num ${refer_str_num} --sigma_value ${sigma_value} &
python3 evaluate_main.py --start_index 154 --subj_num ${subj_num} --percentage_value ${percentage_value} --refer_subj_num ${refer_subj_num} --refer_str_num ${refer_str_num} --sigma_value ${sigma_value} &
python3 evaluate_main.py --start_index 156 --subj_num ${subj_num} --percentage_value ${percentage_value} --refer_subj_num ${refer_subj_num} --refer_str_num ${refer_str_num} --sigma_value ${sigma_value} &
python3 evaluate_main.py --start_index 158 --subj_num ${subj_num} --percentage_value ${percentage_value} --refer_subj_num ${refer_subj_num} --refer_str_num ${refer_str_num} --sigma_value ${sigma_value} ;
python3 evaluate_main.py --start_index 160 --subj_num ${subj_num} --percentage_value ${percentage_value} --refer_subj_num ${refer_subj_num} --refer_str_num ${refer_str_num} --sigma_value ${sigma_value} &
python3 evaluate_main.py --start_index 162 --subj_num ${subj_num} --percentage_value ${percentage_value} --refer_subj_num ${refer_subj_num} --refer_str_num ${refer_str_num} --sigma_value ${sigma_value} &
python3 evaluate_main.py --start_index 164 --subj_num ${subj_num} --percentage_value ${percentage_value} --refer_subj_num ${refer_subj_num} --refer_str_num ${refer_str_num} --sigma_value ${sigma_value} &
python3 evaluate_main.py --start_index 166 --subj_num ${subj_num} --percentage_value ${percentage_value} --refer_subj_num ${refer_subj_num} --refer_str_num ${refer_str_num} --sigma_value ${sigma_value} &
python3 evaluate_main.py --start_index 168 --subj_num ${subj_num} --percentage_value ${percentage_value} --refer_subj_num ${refer_subj_num} --refer_str_num ${refer_str_num} --sigma_value ${sigma_value} &
python3 evaluate_main.py --start_index 170 --subj_num ${subj_num} --percentage_value ${percentage_value} --refer_subj_num ${refer_subj_num} --refer_str_num ${refer_str_num} --sigma_value ${sigma_value} &
python3 evaluate_main.py --start_index 172 --subj_num ${subj_num} --percentage_value ${percentage_value} --refer_subj_num ${refer_subj_num} --refer_str_num ${refer_str_num} --sigma_value ${sigma_value} &
python3 evaluate_main.py --start_index 174 --subj_num ${subj_num} --percentage_value ${percentage_value} --refer_subj_num ${refer_subj_num} --refer_str_num ${refer_str_num} --sigma_value ${sigma_value} &
python3 evaluate_main.py --start_index 176 --subj_num ${subj_num} --percentage_value ${percentage_value} --refer_subj_num ${refer_subj_num} --refer_str_num ${refer_str_num} --sigma_value ${sigma_value} &
python3 evaluate_main.py --start_index 178 --subj_num ${subj_num} --percentage_value ${percentage_value} --refer_subj_num ${refer_subj_num} --refer_str_num ${refer_str_num} --sigma_value ${sigma_value} ;
python3 evaluate_main.py --start_index 180 --subj_num ${subj_num} --percentage_value ${percentage_value} --refer_subj_num ${refer_subj_num} --refer_str_num ${refer_str_num} --sigma_value ${sigma_value} &
python3 evaluate_main.py --start_index 182 --subj_num ${subj_num} --percentage_value ${percentage_value} --refer_subj_num ${refer_subj_num} --refer_str_num ${refer_str_num} --sigma_value ${sigma_value} &
python3 evaluate_main.py --start_index 184 --subj_num ${subj_num} --percentage_value ${percentage_value} --refer_subj_num ${refer_subj_num} --refer_str_num ${refer_str_num} --sigma_value ${sigma_value} &
python3 evaluate_main.py --start_index 186 --subj_num ${subj_num} --percentage_value ${percentage_value} --refer_subj_num ${refer_subj_num} --refer_str_num ${refer_str_num} --sigma_value ${sigma_value} &
python3 evaluate_main.py --start_index 188 --subj_num ${subj_num} --percentage_value ${percentage_value} --refer_subj_num ${refer_subj_num} --refer_str_num ${refer_str_num} --sigma_value ${sigma_value} &
python3 evaluate_main.py --start_index 190 --subj_num ${subj_num} --percentage_value ${percentage_value} --refer_subj_num ${refer_subj_num} --refer_str_num ${refer_str_num} --sigma_value ${sigma_value} &
python3 evaluate_main.py --start_index 192 --subj_num ${subj_num} --percentage_value ${percentage_value} --refer_subj_num ${refer_subj_num} --refer_str_num ${refer_str_num} --sigma_value ${sigma_value} &
python3 evaluate_main.py --start_index 194 --subj_num ${subj_num} --percentage_value ${percentage_value} --refer_subj_num ${refer_subj_num} --refer_str_num ${refer_str_num} --sigma_value ${sigma_value} &
python3 evaluate_main.py --start_index 196 --subj_num ${subj_num} --percentage_value ${percentage_value} --refer_subj_num ${refer_subj_num} --refer_str_num ${refer_str_num} --sigma_value ${sigma_value} &
python3 evaluate_main.py --start_index 198 --subj_num ${subj_num} --percentage_value ${percentage_value} --refer_subj_num ${refer_subj_num} --refer_str_num ${refer_str_num} --sigma_value ${sigma_value}

wait

