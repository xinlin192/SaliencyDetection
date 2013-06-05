/*****************************************************************************
** DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
** Distributed under the terms of the BSD license (see the LICENSE file)
** Copyright (c) 2007-2013, Stephen Gould
** All rights reserved.
**
******************************************************************************
** MODULE: trainModel.cpp
** VERSION: 1.0
** SINCE 2013-04-24
** AUTHOR(S) Jimmy Lin (u5223173) - u5223173@uds.anu.edu.au
**           Chris Claoue-Long (u5183532) - u5183532@anu.edu.au
**
*****************************************************************************/

// c++ standard headers
#include <cstdlib>
#include <cstdio>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <map>

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
#include "parseLabel.h"
#include "Classifier.h"

using namespace std;
using namespace Eigen;

// usage ---------------------------------------------------------------------

void usage() {
    cerr << DRWN_USAGE_HEADER << endl;
    cerr << "USAGE: ./trainModel [OPTIONS] <imgDir> <mscDir> <cshDir> <csdDir> <lblFile> \n";
    cerr << "OPTIONS:\n"
         << "  -o <model>        :: output model\n"
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
    if (DRWN_CMDLINE_ARGC != 5) {
        usage();
        return -1;
    }

    /* Check that the image directory and labels directory exist. All
     * images with a ".jpg" extension will be used for training the
     * model. It is assumed that the labels directory contains files
     * with the same base as the image directory, but with extension
     * ".txt". 
     */
    const char *imgDir = DRWN_CMDLINE_ARGV[0]; // directory restores original images
    const char *mscDir = DRWN_CMDLINE_ARGV[1]; // directory restores multiscale contrast feature map
    const char *cshDir = DRWN_CMDLINE_ARGV[2]; // directory restores center surround histogram feature map
    const char *csdDir = DRWN_CMDLINE_ARGV[3]; // directory restores color spatial distribution feature map
    const char *lblFile = DRWN_CMDLINE_ARGV[4]; // a single text file with ground truth rectangle

    // check their existence
    DRWN_ASSERT_MSG(drwnDirExists(imgDir), "image directory " << imgDir << " does not exist");
    DRWN_ASSERT_MSG(drwnDirExists(mscDir), "Multiscale Contrast directory " << mscDir << " does not exist");
    DRWN_ASSERT_MSG(drwnDirExists(imgDir), "Centre-Surround histogram directory " << cshDir << " does not exist");
    DRWN_ASSERT_MSG(drwnDirExists(imgDir), "Colour Spatial Distribution directory " << csdDir << " does not exist");
    DRWN_ASSERT_MSG(drwnFileExists(lblFile), "Labels file " << lblFile << " does not exist");

    // Get a list of images from the image directory.
    vector<string> baseNames = drwnDirectoryListing(imgDir, ".jpg", false, false);
    DRWN_LOG_MESSAGE("Loading " << baseNames.size() << " images and labels...");

    map< string, vector<int> > fileLabelPairs = parseLabel(lblFile);
    int left, top, right, bottom;
    vector<int> tempRectangle;
    String processedImage;
    vector<double> lambda(3, 0.0);

    // Build a dataset by loading images and labels. For each image,
    // find the salient area using the labels and then compute the set of features
    // that determine this saliency
    drwnClassifierDataset dataset;

    // initialise one 2-class logistic model, data dimension = 3
    Classifier classifier;
    classifier.initialize(3, 2);

    for (unsigned i = 0; i < baseNames.size(); i++) {
        processedImage = baseNames[i] + ".jpg";
        DRWN_LOG_STATUS("...processing image " << baseNames[i]);
        
        // read the image and draw the rectangle of labels of training data
        cv::Mat img = cv::imread(string(imgDir) + DRWN_DIRSEP + processedImage);
        cv::Mat msc = cv::imread(string(mscDir) + DRWN_DIRSEP + processedImage);
        cv::Mat csh = cv::imread(string(cshDir) + DRWN_DIRSEP + processedImage);
        cv::Mat csd = cv::imread(string(csdDir) + DRWN_DIRSEP + processedImage);

        // basic info of currently processed image
        const int H = img.rows;
        const int W = img.cols;

        // form feature vector for each pixel
        vector< vector<double> > features;
        features.resize(H * W, vector<double>(3, 0.0));
        for (int y = 0 ; y < H ; y ++) {
            for (int x = 0 ; x < W ; x ++) {
                features[ y * W + x ][0] = msc.at<Vec3b>(y,x).val[0] / 255.0;
                features[ y * W + x ][1] = csh.at<Vec3b>(y,x).val[0] / 255.0;
                features[ y * W + x ][2] = csd.at<Vec3b>(y,x).val[0] / 255.0;
            }
        }
        
        // ground truth label
        tempRectangle = fileLabelPairs.find(processedImage)->second ;
        left =  tempRectangle [0];
        top = tempRectangle [1];
        right = tempRectangle [2];
        bottom = tempRectangle[3];

        // form target vector for the current image
        vector<int> targets( H * W, 0);
        for (int y = 0 ; y < H ; y ++) {
            for (int x = 0 ; x < W ; x ++) {
                if ( y >= top && y <= bottom && x >= left && x <= right)
                    targets[ y * W + x ] = 1;
            }
        }
        classifier.train(features, targets);
        cout <<  "train finished.." << endl;
        
        lambda = classifier.getWeights();
        cout << "get weights" << endl;
        cout << lambda[0] << "," << lambda[1] << "," << lambda[2] << endl;

        // show the image and feature maps 
        if (bVisualize) {
            //drwnDrawRegionBoundaries and drwnShowDebuggingImage use OpenCV 1.0 C API
            IplImage cvimg = (IplImage)img;
            IplImage *canvas = cvCloneImage(&cvimg);
            drwnShowDebuggingImage(canvas, "image", false);
            cvReleaseImage(&canvas);
        }
    }
        
    // Clean up by freeing memory and printing profile information.
    cvDestroyAllWindows();
    drwnCodeProfiler::print();
    return 0;
}
