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
loat avgintensity(cv::Mat img){
    float avgint = 0.0;
    
}


// Get the contrast of the image
cv::Mat contrast(cv::Mat img){
    cv::Mat contrasted = img; // initialised to the same thing to begin with
    float avgint = avgintensity(img);
    float normaliser = 1/(img.rows * img.cols) // TODO check this is valid float divide
    
    double intpix;
    Vec3b intensity;
    for (int y = 0; y < img.rows; y++){
        for (int x = 0; x < seg.cols; x++){
        
            intensity = img.at<Vec3b>(y,x);
            intpix = 0; // reset to 0, new pixel
            for(int i = 0; i < 3; i++){
                intpix += intensity.val[i];
            }
            intpix = abs(intpix - avgint);
        }
    }

    return contrasted;
}

// Get the multiscale contrast map of the image (to 6 scales) TODO NOT QUITE FINISHED.
cv::Mat multiScaleContrast(cv::Mat img){
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
    for(int i = 1; i < 6; i++{
        pyrDown(tmp, dst, Size(tmp.cols/2, tmp.rows/2) );
        pyramid[i] = tmp;
        tmp = dst; // to perform gaussian modelling again
    }
    
    // run the multiscale contrast thingummy to flatten the image, put into msc
    return msc;   
}

// Get the center-surround histogram
cv::Mat centerSurround(cv::Mat img){
    cv::Mat csi;
    
    return csi;
}

// Get the colour spatial distribution as a gaussian mixture model
cv::Mat colourDist(cv::Mat img){
    cv::Mat cdi;
    
    return cdi;
}