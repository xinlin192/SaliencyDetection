/*****************************************************************************
** DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
** Distributed under the terms of the BSD license (see the LICENSE file)
** Copyright (c) 2007-2013, Stephen Gould
** All rights reserved.
**
******************************************************************************
** FILENAME:    2550Common.h
** AUTHOR(S):   Jimmy Lin <u5223173@uds.anu.edu.anu>
**              Chris Claoue-Long <u5183532@anu.edu.au>
*****************************************************************************/
#include <cmath>
#include <set>
#include <math.h> 

using namespace std;
using namespace Eigen;

// ----------------------- MultiScale Contrast -----------------------------------------------

// Get the contrast of one single image (one scale only)
cv::Mat getContrast(cv::Mat img, int windowSize){
/*{{{*/
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
/*}}}*/
    return contrastMap;
}

// Get the multiscale contrast map of the image (to 6 scales) 
cv::Mat getMultiScaleContrast(cv::Mat img, const int windowSize, const int nPyLevel){
/*{{{*/
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
    int tempx, tempy;
    double tempContrast, multiContrast;
    double maxContrast = -1, minContrast = 1e6;
    Vec3b intensity;
    for (int y = 0; y < imageHeight; y ++ ) {
        for (int x = 0; x < imageWidth; x ++) {
            multiContrast = 0;
            for (int l = 0 ; l < nPyLevel; l ++) {
                //scaling = (int) pow(2.0, l);
                // decide the horizontal and vertical coordinate 
                // in current scale image.
                tempy = y >> l;
                tempx = x >> l;
                // to avoid exception 
                tempy =(tempy >= contrastMaps[l].rows)?contrastMaps[l].rows-1:tempy; 
                tempx =(tempx >= contrastMaps[l].cols)?contrastMaps[l].cols-1:tempx; 
                // get contrast map of that scaled image
                tempContrast = contrastMaps[l].at<double>(tempy, tempx);
                // add to accumulation variable
                multiContrast += tempContrast;
            }
            // set msc matrix of that entry to derived multi-scale Contrast
            msc.at<double>(y,x) = multiContrast;
            // participate max and min selection for latter normalisation
            if (multiContrast < minContrast) minContrast = multiContrast;
            if (multiContrast > maxContrast) maxContrast = multiContrast;
        }
    }
    // normalisation
    double range = maxContrast - minContrast;
    for (int y = 0; y < imageHeight; y ++ ) {
        for (int x = 0; x < imageWidth; x ++) {
            msc.at<double>(y,x) = (msc.at<double>(y,x) - minContrast) / range ;
        }
    }
/*}}}*/
    return msc;
}

// ----------------------- Center Surround Histogram-----------------------------------------------
typedef struct { 
    // surround rectangle parameter
    int SLeft, STop;
    int SWidth, SHeight;
    // center rectangle parameter
    int CLeft, CTop;
    int CWidth, CHeight;
    // chi square distance
    double chiDistance; 
} CSRectangle;

double getChiDistance(CSRectangle csr, cv::Mat histImage, const int nBinsPerDim) {
/*{{{*/
    // initialise the objective variable
    double chidist = 0.0;
    // initialise parameters
    const int nBins =  nBinsPerDim * nBinsPerDim *nBinsPerDim;
    // initialise histogram for center rectangle 
    vector<double> SHistogram(nBins, 0.0);
    vector<double> CHistogram(nBins, 0.0);
    // compute the histogram for center and surround histogram
    for (int y = csr.STop; y < csr.STop + csr.SHeight; y ++) {
        for (int x = csr.SLeft; x < csr.SLeft + csr.SWidth; x ++) {
            // for center histogram
            if (y < csr.CTop + csr.CHeight && y >= csr.CTop && 
                    x < csr.CLeft + csr.SWidth && x >= csr.CLeft) {
                CHistogram[ histImage.at<short>(y,x) ] += 1;
                continue;
            }
            // for surround histogram
            SHistogram[ histImage.at<short>(y,x) ] += 1;
        }
    }

    int nCenterPixels = csr.CWidth * csr.CHeight;
    // compute chi squre distance chidist
    double tempC, tempS;
    for (int i = 0 ; i < nBins ; i ++) {
        //  surround is the surrounding region of the rectangle, which 
        //  is not a rectangle, but has the same area with center
        tempC = 1.0 * CHistogram[i] / (float)nCenterPixels;
        tempS = 1.0 * SHistogram[i] / (float)nCenterPixels;
        if (tempC != 0 || tempS != 0 ) {
            chidist += (tempC-tempS) *(tempC-tempS) / (tempC + tempS);
        }
    }

    // get histogram for surround rectangle
/*}}}*/
    return chidist;
}


CSRectangle getMostDistinctCSRectangle(const int ordinate, const int abscissa, cv::Mat histImage, const int nBinsPerDim) {
/*{{{*/
    const int imageHeight = histImage.rows;
    const int imageWidth = histImage.cols;
    const int nAspectRatio = 5;
    const int nSizeChoice = 12;
    const int minOfSide = (imageWidth>imageHeight)?imageHeight:imageWidth;
    double aspectRatio [] = {0.5, 0.75, 1.0, 1.5, 2.0};
    double sizeRange [] = {0.18, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75};
    std::list<CSRectangle> CSRs;

    int tempSWidth, tempSHeight, tempSLeft, tempSTop;
    for (int i = 0 ; i < nAspectRatio; i ++) {
        for (int j = 0; j < nSizeChoice; j ++) {
            tempSWidth = (int) (minOfSide * sizeRange[j]);
            tempSHeight = (int) (aspectRatio[i] * tempSWidth);
            tempSTop = ordinate - tempSHeight/2;
            tempSLeft = abscissa - tempSWidth/2;
            // examiner
            if (tempSTop < 0 || ordinate + tempSHeight/2 >= imageHeight 
                    || tempSLeft < 0 || abscissa + tempSWidth/2 >= imageWidth) {
                continue;
            } else {
                // construct center surround rectangle object
                CSRectangle tempRect = {-1, -1, -1, -1, -1, -1, -1, -1, -1};
                tempRect.STop = tempSTop;
                tempRect.SLeft = tempSLeft;
                tempRect.SWidth = tempSWidth;
                tempRect.SHeight = tempSHeight;
                tempRect.CWidth = (int) ( tempRect.SWidth/sqrt(2) );
                tempRect.CHeight = (int) ( tempRect.SHeight/sqrt(2) );
                tempRect.CLeft = (int) (abscissa - tempRect.CWidth / 2);
                tempRect.CTop = (int) (ordinate - tempRect.CHeight / 2);
               // cout << tempRect.STop << "," << tempRect.SLeft << "," << tempRect.SWidth << "," << tempRect.SHeight << 
                 //   " : "<< tempRect.CTop << "," << tempRect.CLeft <<"," << tempRect.CWidth << "," << tempRect.CHeight << endl;
                // add to set and to be traverse
                CSRs.push_back(tempRect);
            }
        }
    }

    // initialise objective - most distinct center surround rectangle
    CSRectangle mostDistinctCSR;
    // set it to have invalid chi distance.
    mostDistinctCSR.chiDistance = -1.0;
    // traverse all possible triangle
    double tempChi;
    for (std::list<CSRectangle>::iterator it = CSRs.begin(); it != CSRs.end() ; ++it) {
        tempChi = getChiDistance(*it, histImage, nBinsPerDim);
        if (tempChi > mostDistinctCSR.chiDistance ) {
            (*it).chiDistance = tempChi;
            mostDistinctCSR = *it;
        }
    }
/*}}}*/
    return mostDistinctCSR;
}

cv::Mat getCenterSurround(const cv::Mat img){
/*{{{*/
    // parameters
    int nBinsPerDim = 4;
    // local variable storage for convenient invocation
    const int imageWidth = img.cols;
    const int imageHeight = img.rows;

    // center-surround histogram
    cv::Mat csv(imageHeight, imageWidth, CV_64F);
    cv::Mat histImage(imageHeight, imageWidth, CV_16S);

    //
    int binWidth = 255 / nBinsPerDim;
    int RED, GREEN, BLUE;
    Vec3b intensity;
    for (int y = 0; y < imageHeight; y ++) {
        for (int x = 0; x < imageWidth; x ++) {
            // get intensity of each pixel
            intensity = img.at<Vec3b>(y, x);
            RED = intensity.val[0];
            GREEN = intensity.val[1];
            BLUE = intensity.val[2];
            // get index of bin
            RED /= binWidth;
            GREEN /= binWidth;
            BLUE /= binWidth;
            // handle exception
            RED=(RED>=nBinsPerDim)?(nBinsPerDim-1):RED;
            GREEN=(GREEN>=nBinsPerDim)?(nBinsPerDim-1):GREEN;
            BLUE=(BLUE>=nBinsPerDim)?(nBinsPerDim-1):BLUE;
            // store value in the histImage
            histImage.at<short>(y,x) = RED + nBinsPerDim*GREEN+ nBinsPerDim*nBinsPerDim*BLUE;
        }
    }
    // 
    CSRectangle tempCSRect;
    double fallOff; // gaussian falloff coefficient
    for (int y = 0; y < imageHeight; y ++) {
        for (int x = 0; x < imageWidth; x ++) {
            // get the most distinct center surround pair centered at current pixel
            tempCSRect = getMostDistinctCSRectangle(y, x, histImage, nBinsPerDim);
            //cout << "(" << x << "," << y << ") " << tempCSRect.chiDistance << endl;
            if (tempCSRect.chiDistance <= 0) {  continue;}
            // assign contribution of this center surround pair to pixels in its scope
            for (int tempy = tempCSRect.CTop ; tempy < tempCSRect.CTop + tempCSRect.CHeight ; tempy ++ ) {
                for (int tempx = tempCSRect.CLeft ; tempx <= tempCSRect.CLeft + tempCSRect.CWidth ; tempx ++ ) {
                    fallOff = exp(-0.5 * pow( tempCSRect.CWidth  / 3.0, -2) * 
                            ( (tempx - x)*( tempx - x) + (tempy - y)*(tempy - y) ) ) ;
                    csv.at<double>(tempy, tempx) += fallOff * tempCSRect.chiDistance;
                    //cout << fallOff << "  ,   " << tempCSRect.chiDistance << endl;
                }
            }
        }
    }
    // find min and max
    double minValue = 1e6, maxValue = -1;
    double tempCSHValue;
    for (int y = 0; y < imageHeight; y ++) {
        for (int x = 0; x < imageWidth; x ++) {
            tempCSHValue = csv.at<double>(y, x);
            if (tempCSHValue < minValue) 
                minValue = tempCSHValue;
            if (tempCSHValue > maxValue)
                maxValue = tempCSHValue;
        }
    }
    // normalisation
    double range = maxValue - minValue;
    for (int y = 0; y < imageHeight; y ++) {
        for (int x = 0; x < imageWidth; x ++) {
            csv.at<double>(y, x) = (csv.at<double>(y,x) - minValue) / range;
        }
    }
/*}}}*/
    return csv;
}


// ----------------------- Color Spatial Distribution -----------------------------------------------
// Get the colour spatial distribution as a gaussian mixture model
cv::Mat getSpatialDistribution(cv::Mat img){
    /*{{{*/
    // constant declaration
    const int nComponents = 5;
    const int nDimensions = 3;
    const int imageWidth = img.cols;
    const int imageHeight = img.rows;
    const int nPixels = imageHeight * imageWidth;
    const bool isPydown = false;
    // initialise objective matrix.
    cv::Mat cdi = cv::Mat(imageHeight, imageWidth, CV_64F);
    // temporary variable used in the loop
    int index = -1;
    Vec3b intensity;
    // use scale down to construct a small size of data set for 
    // training mixture of gaussians
    cv::Mat smallImg;
    if (isPydown) {
        cv::pyrDown( img, smallImg, Size( imageWidth/2 , imageHeight/2 ) );
    } else {
        smallImg = cv::Mat(img);
    }
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
    gmm.train(features, 0.1); 

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

