#!/bin/bash

for x in $*
do  
    mkdir results/BCenterSurround/
    mkdir results/BCenterSurround/"$x"/
    ./../../bin/testCenterSurroundHistogram -verbose dataset/images/B/$x dataset/labels/B/"$x"_data.txt results/BCenterSurround/"$x"/
done
