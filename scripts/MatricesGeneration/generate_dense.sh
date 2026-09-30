#!/bin/bash

DATASET_FOLDER=$1
for cells in "10" "100" "1000" "10000"
do
	for features in "10" "100" "1000" "10000" "100000"
	do
		num_elements=$(( cells * features ))

		if [[ $num_elements > 10000000 ]]
		then
			continue
		else
			echo $cells, $features
            mkdir -p $DATASET_FOLDER/Generated/
			python3 generate_matrices.py --cells $cells --genes $features --output $DATASET_FOLDER/Generated/${cells}_cells_${features}_features.csv
		fi
	done	
done
