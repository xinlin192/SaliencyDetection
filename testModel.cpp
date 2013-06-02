/*****************************************************************************
** DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
** Distributed under the terms of the BSD license (see the LICENSE file)
** Copyright (c) 2007-2013, Stephen Gould
** All rights reserved.
**
******************************************************************************
** FILENAME:    testCOMP3130Model.cpp
** AUTHOR(S):   Chris Claoue-Long (u5183532) - u5183532@anu.edu.au
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
    cerr << "USAGE: ./testModel [OPTIONS] <imgDir> <mscDir> <cshDir> <csdDir> <outputDir> <lambda>\n";
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
    if (DRWN_CMDLINE_ARGC != 6) {
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
    const double lambda = atof(DRWN_CMDLINE_ARGV[5]); // lambda calculation
    
    // Check for existence of the directory containing orginal images
    DRWN_ASSERT_MSG(drwnDirExists(imgDir), "image directory " << imgDir << " does not exist");
    DRWN_ASSERT_MSG(drwnDirExists(mscDir), "Multiscale Contrast directory " << mscDir << " does not exist");
    DRWN_ASSERT_MSG(drwnDirExists(cshDir), "Centre-Surround Histogram directory " << cshDir << " does not exist");
    DRWN_ASSERT_MSG(drwnDirExists(csdDir), "Colour Spatial Distribution directory " << csdDir << " does not exist");
    DRWN_ASSERT_MSG(drwnDirExists(outputDir), "Output directory " << outputDir << " does not exist");

    // Get a list of images from the image directory.
    vector<string> baseNames = drwnDirectoryListing(imgDir, ".jpg", false, false);
    DRWN_LOG_MESSAGE("Loading " << baseNames.size() << " images and labels...");


    //declare all often-used variables of the loop here to save on memory!
    cv::Mat img;
    cv::Mat msc;
    cv::Mat csh;
    cv::Mat csd;
    cv::Mat binaryMask;
    cv::Mat bounding;
    vector<vector<cv::Point> > v;
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
        
        // get unary potential and combine them by pre-computed parameters 
        unary[0] = cv::Mat(img.rows, img.cols, CV_64F);
        unary[1] = cv::Mat(img.rows, img.cols, CV_64F);
        for (int y = 0; y < img.rows; y ++) {
            for (int x = 0 ; x < img.cols; x ++) {
                grayscale = 0.22*msc.at<Vec3b>(y,x).val[0] + 0.54*csh.at<Vec3b>(y,x).val[0] + 0.24*csd.at<Vec3b>(y,x).val[0];
                unary[0].at<double>(y,x) = grayscale / 255.0;
                unary[1].at<double>(y,x) = 1 - unary[0].at<double>(y,x);
            }
        }

        // compute binary mask of each pixel
        binaryMask = mexFunction(img, unary, lambda);

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
        cv::imwrite(string(outputDir) + baseNames[i] + ".jpg", pres);
        if (bVisualize) { // draw the processed feature map and display it on the screen
            drwnShowDebuggingImage(present, "Composed Graph", false);
            cvReleaseImage(&present);
        }
        
        // convert pres to a suitable image for working on finding the bounding box using OpenCV functions
        cvtColor(pres, bounding, CV_BGR2GRAY);
        
        // find the contours, returnign only extreme external bounds
        findContours(bounding,v,CV_RETR_EXTERNAL,CV_CHAIN_APPROX_NONE);
        area = 0;
        for(unsigned int it=0; it<v.size();it++) {
            if(area < v[it].size())
                idx = it; 
        }
        box = boundingRect(v[idx]);
        pt1.x = box.x;
        pt1.y = box.y;
        pt2.x = box.x + box.width;
        pt2.y = box.y + box.height;
        // Draws the bounding box in the original image and show it
        cv::rectangle(bounding, pt1, pt2, Scalar(255,0,0));
        // show the bounding box!
        IplImage bound = (IplImage)bounding;
        IplImage *boundCanvas = cvCloneImage(&bound);
        drwnShowDebuggingImage(boundCanvas, "Rectangle", false);
        cvReleaseImage(&boundCanvas);
        cv::imwrite(string(outputDir) + baseNames[i] + "RECT.jpg", bounding);

    }
    
    
    
    // Clean up by freeing memory and printing profile information.
    cvDestroyAllWindows();
    drwnCodeProfiler::print();
    return 0;
}
