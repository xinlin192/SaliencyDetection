for x in $*
do  
    mkdir results/labels/
    mkdir results/labels/"$x"/
    ./../../bin/getLabelledImages -verbose -x dataset/images/B/$x dataset/labels/B/"$x"_data.txt results/labels/"$x"/
done
