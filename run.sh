echo "Which training data set to use: "
read x
mkdir results/testMixturesGaussian/"$x"/
./../../bin/trainModel -verbose -x  dataset/images/A/"$x" dataset/labels/A/"$x"_data.txt results/testMixturesGaussian/"$x"/
