#!/bin/bash
set -x

for cat in {0..19}
do
    python supervised_weighting.py --cat "$cat" --fs 10000 --lr 0.01
done
