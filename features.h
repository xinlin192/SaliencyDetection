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

// Average intensity of an image (for contrast mapping) TODO NOT QUITE FINISHED.
float avgintensity(cv::Mat img){
    float avgint = 0.0;
}

// Get the contrast of the image
cv::Mat contrast(cv::Mat img){
    cv::Mat contrasted = img; // initialised to the same thing to begin with
    float avgint = avgintensity(img);
    float normaliser = 1/(img.rows * img.cols);
    
    double intpix;
    Vec3b intensity;
    for (int y = 0; y < img.rows; y++){
        for (int x = 0; x < img.cols; x++){
        
            intensity = img.at<Vec3b>(y,x);
            intpix = 0; // reset to 0, new pixel
            for(int i = 0; i < 3; i++) {
                intpix += intensity.val[i];
            }
            intpix = abs(intpix - avgint);
        }
    }

    return contrasted;
}

// Get the multiscale contrast map of the image (to 6 scales) 
// TODO NOT QUITE FINISHED.
cv::Mat getMultiScaleContrast(cv::Mat img){
    cv::Mat msc = img; // the return matrix, initialised to the input by default
    cv::Mat cont;
    cv::Mat tmp;
    cv::Mat dst;
    vector<cv::Mat> pyramid;
    pyramid.resize(6); // 6 images in this gaussian pyramid
    
    cont = contrast(img);
    
    pyramid[0] = cont; // the original contrasted image, base of the gaussian pyramid
    tmp = cont;
    dst = tmp; // initialised
    for(int i = 1; i < 6; i++){
        pyrDown(tmp, dst, Size(tmp.cols/2, tmp.rows/2) );
        pyramid[i] = tmp;
        tmp = dst; // to perform gaussian modelling again
    }
    
    // run the multiscale contrast thingummy to flatten the image, put into msc
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
    const int nComponents = 10;
    const int nDimensions = 3;
    const int imageWidth = img.cols;
    const int imageHeight = img.rows;
    const int nPixels = imageHeight * imageWidth;
    // initialise objective matrix.
    cv::Mat cdi = cv::Mat(imageHeight, imageWidth, CV_64F);
    // local variable 
    vector<vector<double> > features(nPixels, vector<double>(3, 0.0));
    // temporary variable used in the loop
    int index = -1;
    Vec3b intensity;
    // loop to read data from an image
    for(int y = 0; y < imageHeight; y++){
        for(int x = 0; x < imageWidth; x++){
            index = y * imageWidth + x;
            intensity = img.at<Vec3b>(y,x);
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
    for (int k = 0; k < nComponents; k++) {
        gmm.component(k).evaluate(features, responsibilities[k]);
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

