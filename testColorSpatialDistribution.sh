echo "Which training data set to use: "
read x
mkdir results/ColorSpatialDistribution/"$x"/
./../../bin/testColorSpatialDistribution -verbose dataset/images/A/"$x" dataset/labels/A/"$x"_data.txt results/ColorSpatialDistribution/"$x"/
