#!/bin/bash
# Bash script to run the various parts of the computer vision framework
# Chris Claoue-Long

USAGE="Usage:  make run\n\n\t./run clean\n\t./run [csh csd msc] <imageDirectory>\n\t./run [train test mark] <imageDirectory> <labelDirectory>\n"

echo
# main part of the script
if [ $# -eq "0" ]; then 
    echo -e $USAGE
elif [ $# -eq "1" -a $1 == "clean" ]; then
    # purge results
    echo "Purging results..."
    rm -rf results/
    mkdir results
    echo "Results folder is now empty!"
elif [ $# -eq "2" ]; then
    # create csh/csd/msc images
    echo "Computing $1 renderings for images in $2"
    mkdir results/$1
    ../../bin/getFeatureMaps -x -v $1 $2 results/$1/
    echo "Results saved to results/$1/"
elif [ $# -eq "7" ]; then 
    # imgDir mscDir cshDir csdDir outputDir outputLbls lambda
    # test and mark the saliency detector
    echo "WARNING:" 
    echo "THIS ASSUMES YOU HAVE ALREADY COMPUTED ALL CSH, CSD AND MSC RENDERINGS FOR THE IMAGES IN $2"
    echo "Running the saliency detector..."
    mkdir results/output
    ../../bin/testModel $@
    echo "Please input the ground truth label file's absolute path"
    echo -n "> " # prompt for input
    read groundTruth
    ../../bin/scoreModel groundTruth $6
else
    echo -e $USAGE
fi