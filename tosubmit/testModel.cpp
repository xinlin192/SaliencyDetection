/*****************************************************************************
** DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
** Distributed under the terms of the BSD license (see the LICENSE file)
** Copyright (c) 2007-2013, Stephen Gould
** All rights reserved.
**
******************************************************************************
** FILENAME:    testCOMP3130Model.cpp
** AUTHOR(S):   
**     Jimmy Lin (u5223173) - linxin@gmail.com
**     Chris Claoue-Long (u5183532) - u5183532@anu.edu.au
**
*****************************************************************************/

// c++ standard headers
#include <cstdlib>
#include <cstdio>
#include <iostream>
#include <fstream>
#include <iomanip>

// eigen matrix library headers
#include "Eigen/Core"

// opencv library headers
#include "cv.h"
#include "cxcore.h"
#include "highgui.h"

// darwin library headers
#include "drwnBase.h"
#include "drwnIO.h"
#include "drwnML.h"
#include "drwnVision.h"

#include "mexImageCRF.h"

using namespace std;
using namespace Eigen;

// usage ---------------------------------------------------------------------

void usage()
{
    cerr << DRWN_USAGE_HEADER << endl;
    cerr << "USAGE: ./testModel [OPTIONS] <imgDir> <mscDir> <cshDir> <csdDir> <outputDir> <outputLbls> <lambda>\n";
    cerr << "OPTIONS:\n"
         << "  -o <lblDir>       :: output directory for predicted labels\n"
         << "  -x                :: visualize\n"
         << DRWN_STANDARD_OPTIONS_USAGE
	 << endl;
}

// main ----------------------------------------------------------------------

int main (int argc, char * argv[]) {

    // Set default value for optional command line arguments.
    const char *modelFile = NULL;
    bool bVisualize = false;

    DRWN_BEGIN_CMDLINE_PROCESSING(argc, argv)
        DRWN_CMDLINE_STR_OPTION("-o", modelFile)
        DRWN_CMDLINE_BOOL_OPTION("-x", bVisualize)
    DRWN_END_CMDLINE_PROCESSING(usage());

    // Check for the correct number of required arguments
    if (DRWN_CMDLINE_ARGC != 10) {
        usage();
        return -1;
    }

    // Check that the image directory and labels directory exist. All
    // images with a ".jpg" extension will be used for training the
    // model. It is assumed that the labels directory contains files
    // with the same base as the image directory, but with extension
    // ".txt". 
     
    const char *imgDir = DRWN_CMDLINE_ARGV[0]; // directory restores original images
    const char *mscDir = DRWN_CMDLINE_ARGV[1]; // directory restores multiscale contrast feature map
    const char *cshDir = DRWN_CMDLINE_ARGV[2]; // directory restores center surround histogram feature map
    const char *csdDir = DRWN_CMDLINE_ARGV[3]; // directory restores color spatial distribution feature map
    const char *outputDir = DRWN_CMDLINE_ARGV[4]; // directory for resulting images
    const char *outLbls = DRWN_CMDLINE_ARGV[5]; // output labels file
    const double lambda1 = atof(DRWN_CMDLINE_ARGV[6]); // lambda for local feature
    const double lambda2 = atof(DRWN_CMDLINE_ARGV[7]); // lambda for regional feature 
    const double lambda3 = atof(DRWN_CMDLINE_ARGV[8]); // lambda for global feature
    const double lambda0 = atof(DRWN_CMDLINE_ARGV[9]); // lambda for pairwise term
    
    // Check for existence of the directory containing orginal images
    DRWN_ASSERT_MSG(drwnDirExists(imgDir), "image directory " << imgDir << " does not exist");
    DRWN_ASSERT_MSG(drwnDirExists(mscDir), "Multiscale Contrast directory " << mscDir << " does not exist");
    DRWN_ASSERT_MSG(drwnDirExists(cshDir), "Centre-Surround Histogram directory " << cshDir << " does not exist");
    DRWN_ASSERT_MSG(drwnDirExists(csdDir), "Colour Spatial Distribution directory " << csdDir << " does not exist");
    DRWN_ASSERT_MSG(drwnDirExists(outputDir), "Output directory " << outputDir << " does not exist");

    // Get a list of images from the image directory.
    vector<string> baseNames = drwnDirectoryListing(imgDir, ".jpg", false, false);
    DRWN_LOG_MESSAGE("Loading " << baseNames.size() << " images and labels...");
    string dir = string(imgDir);
    //int dirIndex = dir.find_last_of(DRWN_DIRSEP)-1; // get the last occurrence
    ofstream outputLbls;
    outputLbls.open(outLbls, ios::out | ios::trunc);
    if (!outputLbls.is_open()){
        cerr << "ERROR CREATING OUTPUT LABEL FILE";
        return -1;
    }
    outputLbls << baseNames.size() << "\n\n"; // number of labels there will be in this file

    //often-used variables of the loop are here to save on memory!
    cv::Mat img;
    cv::Mat msc;
    cv::Mat csh;
    cv::Mat csd;
    cv::Mat binaryMask;
    cv::Mat bounding;
    vector<vector<cv::Point> > contours;
    unsigned int area;
    int idx;
    cv::Rect box;
    cv::Point pt1, pt2;
    int tempSaliency;
    vector< cv::Mat > unary(2);
    double grayscale;
    
    for (unsigned i = 0; i < baseNames.size(); i++) {
        String processedImage = baseNames[i] + ".jpg";
        DRWN_LOG_STATUS("...processing image " << baseNames[i]);
        // read the image and draw the rectangle of labels of training data
        img = cv::imread(string(imgDir) + DRWN_DIRSEP + processedImage);
        msc = cv::imread(string(mscDir) + DRWN_DIRSEP + processedImage);
        csh = cv::imread(string(cshDir) + DRWN_DIRSEP + processedImage);
        csd = cv::imread(string(csdDir) + DRWN_DIRSEP + processedImage);
        
        if (bVisualize) {
            //drwnDrawRegionBoundaries and drwnShowDebuggingImage use OpenCV 1.0 C API
            IplImage cvimg = (IplImage)img;
            IplImage *canvas = cvCloneImage(&cvimg);
            drwnShowDebuggingImage(canvas, "image", false);
            cvReleaseImage(&canvas);
        }
        
        cv::Mat tempMat(img.rows, img.cols, CV_64F);
        double maxValue = -1e6, minValue = 1e6;
        for (int y = 0; y < img.rows; y ++) {
            for (int x = 0 ; x < img.cols; x ++) {
                grayscale = lambda1 * (msc.at<Vec3b>(y,x).val[0] / 255.0) +  
                    lambda2 * (csh.at<Vec3b>(y,x).val[0] / 255.0 )  + 
                       lambda3 * (csd.at<Vec3b>(y,x).val[0] / 255.0 );
                tempMat.at<double>(y,x) = grayscale;
                maxValue = (maxValue < grayscale)?grayscale:maxValue;
                minValue = (minValue > grayscale)?grayscale:minValue;
            }
        }

        double range = maxValue - minValue;
        // get unary potential and combine them by pre-computed parameters 
        unary[0] = cv::Mat(img.rows, img.cols, CV_64F);
        unary[1] = cv::Mat(img.rows, img.cols, CV_64F);
        for (int y = 0; y < img.rows; y ++) {
            for (int x = 0 ; x < img.cols; x ++) {
                unary[1].at<double>(y,x) = (tempMat.at<double>(y,x) - minValue) / range;
                unary[0].at<double>(y,x) = 1 - unary[1].at<double>(y,x);
            }
        }

        // compute binary mask of each pixel
        binaryMask = mexFunction(img, unary, lambda0);

        // interpret the binary mask as a two-color image
        cv::Mat pres(img.rows, img.cols, CV_8UC3);
        for (int y = 0 ; y < img.rows; y ++) {
            for (int x = 0 ; x < img.cols; x ++) {
                tempSaliency = binaryMask.at<short>(y,x)*255>125?255:0;
                pres.at<cv::Vec3b>(y,x) = cv::Vec3b(tempSaliency, tempSaliency, tempSaliency);
            }
        }
        
        // present the derived binary mask
        IplImage pcvimg = (IplImage) pres;
        IplImage *present = cvCloneImage(&pcvimg);
        cv::imwrite(string(outputDir) + baseNames[i] + "BM.jpg", pres);
        
        if (bVisualize) { 
            // draw the processed feature map and display it on the screen
            drwnShowDebuggingImage(present, "Composed Graph", false);
            cvReleaseImage(&present);
        }
        
        
        // convert pres to a suitable image for working on finding the bounding box using OpenCV functions
        cvtColor(pres, bounding, CV_BGR2GRAY);
        
        // find the contours, returning only extreme external bounds, find the largest contour area.
        // FIXME change to getting a bounding box around all white pixels
        findContours(bounding,contours,CV_RETR_EXTERNAL,CV_CHAIN_APPROX_SIMPLE);
        
        // find the largest contour area
        area = 0;
        for(unsigned int it=0; it<contours.size();it++) {
            if(area < contours[it].size())
                idx = it; 
        }
        box = boundingRect(contours[idx]);
        pt1.x = box.x;
        pt1.y = box.y;
        pt2.x = box.x + box.width;
        pt2.y = box.y + box.height;
        bounding = cv::Mat(img);
        for (int k = 0; k < 5 ; k +=3) {
            cv::rectangle(bounding, cv::Point(pt1.x-k, pt1.y-k), cv::Point(pt2.x+k, pt2.y+k), Scalar(255, 0, 0));
            cv::rectangle(bounding, cv::Point(pt1.x-k-1, pt1.y-k-1), cv::Point(pt2.x+k+1, pt2.y+k+1), Scalar(0, 255, 0));
            cv::rectangle(bounding, cv::Point(pt1.x-k-2, pt1.y-k-2), cv::Point(pt2.x+k+2, pt2.y+k+2), Scalar(0, 0, 255));
        }

        if(bVisualize){
            // Draw the bounding box in the original image and show it
            IplImage bound = (IplImage)bounding;
            IplImage *boundCanvas = cvCloneImage(&bound);
            drwnShowDebuggingImage(boundCanvas, "Rectangle", false);
            cvReleaseImage(&boundCanvas);
        }
        
        cv::imwrite(string(outputDir) + baseNames[i] + "RECT.jpg", bounding);

        // add the rectangle to the list of output labels
        // the \r here is to mimic the windows-style carriage returns that are contained in the truth label files
        outputLbls << baseNames[i].substr(0,1) << "\\" << baseNames[i] << ".jpg\r\n";
        outputLbls << img.cols << " " << img.rows << "\n";
        // we're getting the second labels so put bogus padding values as the first set
        outputLbls << "0 0 0 0; " << pt1.x << " " << pt1.y << " " << pt2.x << " " << pt2.y << ";\n\n";
    }
    
    outputLbls.close();
    // Clean up by freeing memory and printing profile information.
    cvDestroyAllWindows();
    drwnCodeProfiler::print();
    return 0;
}
