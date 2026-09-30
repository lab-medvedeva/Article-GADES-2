#!/bin/bash

DATASET_FOLDER=$1

for i in 10 100 1000 10000
do
    for j in 10 100 1000 10000 100000
    do
        output=$(( i * j ))

        if [[ $output -ge 100000 && $output -le 10000000 ]]
        then
            echo $output "OK", $i, $j
        else
            continue
        fi
        echo "Here"
        for sparsity in 0.5 0.75 0.9 0.95 0.99
        do
            echo Generating ${i}_${j} matrix for ${sparsity} rep
            generated_folder=$DATASET_FOLDER/GeneratedSparse/${i}_cells_${j}_features
            mkdir -p ${generated_folder}
            echo ${generated_folder}
            python generate_matrices_sparse.py --cells=$i --genes=$j --sparsity $sparsity --output ${generated_folder}/$sparsity.mtx
        done
    done
done
