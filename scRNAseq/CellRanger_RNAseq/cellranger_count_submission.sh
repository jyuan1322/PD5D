#!/bin/bash
for i in $(cat annotations_021721.csv | awk -F "," '(NR>1){print $2}'); do
    bsub -q big "sh cellranger_counts.sh $i"
done