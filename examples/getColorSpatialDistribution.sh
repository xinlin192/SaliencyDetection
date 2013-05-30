for x in $*
do  
    mkdir results/ColorSpatialDistributionNopydown/
    mkdir results/ColorSpatialDistributionNopydown/"$x"/
    ./../../bin/testColorSpatialDistribution -verbose dataset/images/B/$x dataset/labels/B/"$x"_data.txt results/ColorSpatialDistributionNopydown/"$x"/
done
