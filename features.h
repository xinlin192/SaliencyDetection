/*****************************************************************************
** DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
** Distributed under the terms of the BSD license (see the LICENSE file)
** Copyright (c) 2007-2013, Stephen Gould
** All rights reserved.
**
******************************************************************************
** FILENAME:    2550Common.h
** AUTHOR(S):   Stephen Gould <stephen.gould@anu.edu.au>
**              Jimmy Lin <u5223173@uds.anu.edu.anu>
**              Chris Claoue-Long <u5183532@anu.edu.au>
*****************************************************************************/
#include <cmath>
#include <set>
using namespace std;
using namespace Eigen;

// feature extraction algorithms -----------------------------------------------

// Get the contrast of the image
cv::Mat getContrast(cv::Mat img, int windowSize){
    int imageWidth = img.cols;
    int imageHeight = img.rows;
    // temporary variable used in the loop
    int tempx, tempy;
    double contrast;
    double green, red, blue;
    double redNeighbours, greenNeighbours, blueNeighbours;
    Vec3b intensity;
    Vec3b intensityNeighbours;
    // initialise objective matrix
    cv::Mat contrastMap = cv::Mat::zeros(imageHeight, imageWidth, CV_64F);

    for (int y = 0; y < imageHeight; y++){
        for (int x = 0; x < imageWidth; x++){
            std::set< pair<int, int> > neighbours;
            int nNeighbours = 0;
            // add neighbour to set
            for (int offsetx = -1 * windowSize + 1 ; offsetx < windowSize; offsetx ++) {
                tempx = x + offsetx;  
                for (int offsety = -1 * windowSize + 1; offsety < windowSize; offsety ++) {
                    tempy = y + offsety;
                    if (tempx >= 0 && tempx < imageWidth && tempy >= 0 && tempy < imageHeight) {
                        neighbours.insert(std::pair<int,int>(tempy, tempx));
                        nNeighbours ++;
                    } 
                }
            }
            // get intensity of currently objective point.
            intensity = img.at<Vec3b>(y,x);
            red = intensity.val[0];
            green = intensity.val[1];
            blue = intensity.val[2];
            // traverse all neighbour in the set.
            contrast = 0.0;
            for (std::set< pair<int,int> >::iterator it = neighbours.begin(); it != neighbours.end(); ++ it) {
                intensityNeighbours = img.at<Vec3b>(it -> first, it -> second); // pair(y,x)
                redNeighbours = intensityNeighbours.val[0];
                greenNeighbours = intensityNeighbours.val[1];
                blueNeighbours = intensityNeighbours.val[2];
                contrast += abs(redNeighbours - red) + 
                    abs(greenNeighbours - green) + abs(blueNeighbours - blue);
            }
            contrast /= nNeighbours;
            contrastMap.at<double>(y, x) = contrast;
        }
    }
    return contrastMap;
}

// Get the multiscale contrast map of the image (to 6 scales) 
cv::Mat getMultiScaleContrast(cv::Mat img, const int windowSize, const int nPyLevel){
    // constant declaration
    const int imageWidth = img.cols;
    const int imageHeight = img.rows;
    // initialise objective matrix, data type of entry is float.
    cv::Mat msc = cv::Mat(imageHeight, imageWidth, CV_64F);
    // generate the width and height of image in each pyramid level 
    vector<int> heights(nPyLevel, imageHeight);
    vector<int> widths(nPyLevel, imageWidth);
    for (int i = 1; i < nPyLevel; i ++) {
        // no change for i = 0, orginal image
        heights[i] = heights[i-1] / 2;
        widths[i] = widths[i-1] / 2;
    }
    // vector to restore pointer to multiscale image
    vector< cv::Mat > pyramid(nPyLevel, img);
    pyramid[0] = img;
    // work out all scaled image 
    for (int i = 1; i < nPyLevel; i ++) {
        // no change for i = 0, orginal image
        cv::pyrDown( pyramid[i-1], pyramid[i], Size( widths[i] , heights[i] ) );
    }
    // vector to store pointer to multiscale contrast map
    vector< cv::Mat > contrastMaps(nPyLevel, img);
    // work out constrast of all scaled image
    for (int i = 0; i < nPyLevel; i ++ ) {
        contrastMaps[i] = getContrast( pyramid[i], windowSize);
    }

    // calculate the feature map incorporates all level of constrast.
    int scaling = 0;
    int tempx, tempy;
    double tempContrast, multiContrast;
    double maxContrast = -1, minContrast = 1e6;
    Vec3b intensity;
    for (int y = 0; y < imageHeight; y ++ ) {
        for (int x = 0; x < imageWidth; x ++) {
            multiContrast = 0;
            for (int l = 0 ; l < nPyLevel; l ++) {
                scaling = (int) pow(2.0, l);
                tempy = y / scaling;
                tempx = x / scaling;
                tempContrast = contrastMaps[l].at<double>(tempy, tempx);
                multiContrast += tempContrast;
            }
            msc.at<double>(y,x) = multiContrast;
            // participate max and min selection
            if (multiContrast < minContrast) minContrast = multiContrast;
            if (multiContrast > maxContrast) maxContrast = multiContrast;
        }
    }
    cout << minContrast << ", " << maxContrast << endl;
    // normalisation
    double range = maxContrast - minContrast;
    for (int y = 0; y < imageHeight; y ++ ) {
        for (int x = 0; x < imageWidth; x ++) {
            msc.at<double>(y,x) = (msc.at<double>(y,x) - minContrast) / range ;
        }
    }

    return msc;
}

// struct for centre-surround histogram feature
struct CentreSurround {
    double dist;
    vector<int> rect;
};

// Get the value from a center-surround histogram
// rect1 is (xval, yval, widthtoright, heightdown), 255 must divide into bins exactly - eg 15 works, 16 does not.
CentreSurround getCentreSurround(cv::Mat img, vector<int> rect1, int bins){
    CentreSurround csv; // centre-surround distance and appropriate rectangle
    vector<vector<int> > histogramBins1(bins);
    vector<vector<int> > histogramBins2(bins);
    int binDelimiter=255/bins;
    Vec3b pixColour;

    // initialise to 0 for all bins in the histogram
    for(long i = 0; i < bins; i++){
        histogramBins1.at(i).resize(3);
        histogramBins1.at(i)[0] = 0;
        histogramBins1.at(i)[1] = 0;
        histogramBins1.at(i)[2] = 0;
        histogramBins2.at(i).resize(3);
        histogramBins2.at(i)[0] = 0;
        histogramBins2.at(i)[1] = 0;
        histogramBins2.at(i)[2] = 0;
    }

    // TODO create a rectangle around rect1 such that its area - area of rect1 is equal to that of rect1
    // for the moment, a really approximate method is here so we have something working.
    // we can maybe modify this to do away with correct aspect ratio choice by some clever formula too...?

    vector<int> rect2(4);
    rect2.at(0)=rect1.at(0)-(rect1.at(2)/2);
    if(rect2.at(0)<0) { rect2.at(0)=0; } // ensure positive values only
    rect2.at(1)=rect1.at(1)-(rect1.at(3)/2);
    if(rect2.at(1)<0) { rect2.at(1)=0; }
    rect2.at(2)=rect1.at(2)*2;
    rect2.at(3)=rect1.at(3)*2;

    // perform RGB histogram calculation
    for(int y = 0; y < img.rows; y++){
        for(int x = 0; x < img.cols; x++){
            if (y < rect2.at(1) | y > rect2.at(1)+rect2.at(3) | x < rect2.at(0) | x > rect2.at(0)+rect2.at(2) ){ continue; }
            pixColour=img.at<Vec3b>(y,x);

            // really simplistic algorithm to convert the pixel BGR values into their destination bin
            // can be from 0 to bins
            pixColour[0] /= binDelimiter;
            pixColour[1] /= binDelimiter;
            pixColour[2] /= binDelimiter;

            if (y >= rect1.at(1) & y <= rect1.at(1)+rect1.at(3) & x >= rect1.at(0) & x <= rect1.at(2) ) { // pixel in rect1
                histogramBins1.at((int) pixColour[0])[0]++;
                histogramBins1.at((int) pixColour[1])[1]++;
                histogramBins1.at((int) pixColour[2])[2]++;
            } else { // pixel in rect2
                histogramBins2.at((int) pixColour[0])[0]++;
                histogramBins2.at((int) pixColour[1])[1]++;
                histogramBins2.at((int) pixColour[2])[2]++;
            }   
        }

    }

    // calculate the chi-squared value
    float sum = 0.0;
    for(int i = 0; i < bins; i++){
        for(int j = 0; j < 3; j++){
            sum+= (pow((float)(histogramBins1.at(i)[j]-histogramBins2.at(i)[j]), 2))/(histogramBins1.at(i)[j]+histogramBins2.at(i)[j]);
        }
    }
    csv.dist = sum/2; // calculated 1/2*sum_i[(histR1_i-histR2_i)^2/(histR1_i+histR2_i)]
    csv.rect = rect2;
    return csv;
}


// Get the colour spatial distribution as a gaussian mixture model
cv::Mat getSpatialDistribution(cv::Mat img){
    /*{{{*/
    // constant declaration
    const int nComponents = 5;
    const int nDimensions = 3;
    const int imageWidth = img.cols;
    const int imageHeight = img.rows;
    const int nPixels = imageHeight * imageWidth;
    // initialise objective matrix.
    cv::Mat cdi = cv::Mat(imageHeight, imageWidth, CV_64F);
    // temporary variable used in the loop
    int index = -1;
    Vec3b intensity;
    // use scale down to construct a small size of data set for 
    // training mixture of gaussians
    cv::Mat smallImg;
    cv::pyrDown( img, smallImg, Size( imageWidth/2 , imageHeight/2 ) );
    // loop to read data from an scaled image
    vector<vector<double> > features(nPixels, vector<double>(3, 0.0));
    for(int y = 0; y < smallImg.rows; y++){
        for(int x = 0; x < smallImg.cols; x++){
            index = y * imageWidth + x;
            intensity = smallImg.at<Vec3b>(y,x);
            // load RGB value to each row
            features[index][0] = intensity.val[0];
            features[index][1] = intensity.val[1];
            features[index][2] = intensity.val[2];
        }
    }
    // initialise gaussian model
    drwnGaussianMixture gmm(nDimensions, nComponents); 
    // train the mixture model on the features given
    gmm.train(features); 

    // create table of responsibilities p(c|I_x) 
    vector<vector<double> > responsibilities(nComponents, vector<double>(nPixels, 0.0) );
    vector<double> temp(nDimensions, 0.0); // vector storing RGB value
    for (int y = 0; y < imageHeight; y ++) {
        for (int x = 0; x < imageWidth; x ++) {
            index = y * imageWidth + x;
            double normalisation = 0;
            for (int k = 0; k < nComponents; k ++) {
                intensity = img.at<Vec3b>(y,x);
                temp[0] = intensity.val[0];
                temp[1] = intensity.val[1];
                temp[2] = intensity.val[2];
                responsibilities[k][index] = exp( gmm.component(k).evaluateSingle(temp)) * gmm.weight(k);
                normalisation += responsibilities[k][index];
            }
            for (int k = 0; k < nComponents; k ++) {
                responsibilities[k][index] = responsibilities[k][index] / normalisation;
            }
        }
    }
    // compute horizontal mean and vertical mean
    vector<double> hMean(nComponents, 0.0); // horizontal mean
    vector<double> vMean(nComponents, 0.0); // vertical mean
    vector<double> eNumber(nComponents, 0.0); // effective number of assignment
    for (int y = 0; y < imageHeight; y ++) {
        for (int x = 0; x < imageWidth; x ++) {
            index = y * imageWidth + x;
            for (int k = 0; k < nComponents; k ++) {
                hMean[k] += x * responsibilities[k][index];
                vMean[k] += y * responsibilities[k][index];
                eNumber[k] += responsibilities[k][index];
            }
        }
    }
    // average all the values
    for (int k = 0; k < nComponents; k ++) {
        hMean[k] /= eNumber[k];
        vMean[k] /= eNumber[k];
    }
    // compute horizontal covariance and vertical covariance
    vector<double> hCovariance(nComponents, 0.0);
    vector<double> vCovariance(nComponents, 0.0);
    for (int y = 0; y < imageHeight; y ++) {
        for (int x = 0; x < imageWidth; x ++) {
            index = y * imageWidth + x;
            for (int k = 0; k < nComponents; k ++) {
                hCovariance[k] += pow((x-hMean[k]), 2) * responsibilities[k][index];
                vCovariance[k] += pow((y-vMean[k]), 2)* responsibilities[k][index];
            }
        }
    }
    // average all values 
    for (int k = 0; k < nComponents; k ++) {
        hCovariance[k] /= eNumber[k];
        vCovariance[k] /= eNumber[k];
    }
    // sum up to get overall covariance of each component
    vector<double> oCovariance(nComponents, 0.0);
    for (int k = 0; k < nComponents; k ++) {
        oCovariance[k] = hCovariance[k] + vCovariance[k] ;
    }
    // normalisation
    std::vector<double>::iterator max = std::max_element(oCovariance.begin(), oCovariance.end());
    std::vector<double>::iterator min = std::min_element(oCovariance.begin(), oCovariance.end());
    double range = *max - *min;
    for (int k = 0 ; k < nComponents; k ++) {
        oCovariance[k] = (oCovariance[k] - *min) / range;
    }
    // assign color spatial feature to each pixel
    vector<double> unfs(nPixels, 0.0); // unnormalised spatial feature map
    for (int y = 0; y < imageHeight; y ++) {
        for (int x = 0; x < imageWidth; x ++) {
            index = y * imageWidth + x;
            for (int k = 0; k < nComponents; k ++) {
                unfs[index] += responsibilities[k][index] * (1 - oCovariance[k]);
            }
        }
    }
    // normalise the spatial feature
    std::vector<double>::iterator maxfs = std::max_element(unfs.begin(), unfs.end());
    std::vector<double>::iterator minfs = std::min_element(unfs.begin(), unfs.end());
    range = *maxfs - *minfs;
    for (int y = 0; y < imageHeight; y ++) {
        for (int x = 0; x < imageWidth; x ++) {
            index = y * imageWidth + x;
            cdi.at<double>(y,x) = (unfs[index] - *minfs) / range;
        }
    }
    /*}}}*/
    return cdi;
}

