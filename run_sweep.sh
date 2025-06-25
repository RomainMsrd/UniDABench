#!/bin/bash

#python main.py --dataset "HHAR" --da_method "OVANet"
#python main_sweep.py --dataset "HAR" --da_method "UniOT"
#python main_sweep.py --dataset "HAR" --da_method "UDA"
python main_sweep.py --dataset "HAR" --da_method "PPOT"
python main_sweep.py --dataset "HAR" --da_method "DANCE"
python main_sweep.py --dataset "HAR" --da_method "OVANet" --hp_search_strategy "grid"
python main_sweep.py --dataset "HAR" --da_method "UniJDOT"
#shutdown


