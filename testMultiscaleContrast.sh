for x in 0
do  
    mkdir results/BMultiscaleContrast/
    mkdir results/BMultiscaleContrast/"$x"/
    ./../../bin/trainModel -verbose -x dataset/images/B/$x dataset/labels/B/"$x"_data.txt results/BMultiscaleContrast/"$x"/
done
