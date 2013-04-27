/*****************************************************************************
** DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
** Distributed under the terms of the BSD license (see the LICENSE file)
** Copyright (c) 2007-2013, Stephen Gould
** All rights reserved.
**
******************************************************************************
** FILENAME:    testCOMP3130Model.cpp
** AUTHOR(S):   Stephen Gould <stephen.gould@anu.edu.au>
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

#include "crfCommon.h"

using namespace std;
using namespace Eigen;

// usage ---------------------------------------------------------------------

void usage()
{
    cerr << DRWN_USAGE_HEADER << endl;
    cerr << "USAGE: ./testModel [OPTIONS] <model> <imgDir>\n";
    cerr << "OPTIONS:\n"
         << "  -o <lblDir>       :: output directory for predicted labels\n"
         << "  -x                :: visualize\n"
         << DRWN_STANDARD_OPTIONS_USAGE
	 << endl;
}

// main ----------------------------------------------------------------------

int main(int argc, char *argv[])
{
    // Set default value for optional command line arguments.
    const char *lblDirOut = NULL;
    bool bVisualize = false;

    // Process command line arguments using Darwin.
    DRWN_BEGIN_CMDLINE_PROCESSING(argc, argv)
        DRWN_CMDLINE_STR_OPTION("-o", lblDirOut)
        DRWN_CMDLINE_BOOL_OPTION("-x", bVisualize)
    DRWN_END_CMDLINE_PROCESSING(usage());

    // Check for the correct number of required arguments, otherwise
    // print usage statement and exit.
    if (DRWN_CMDLINE_ARGC != 2) {
        usage();
        return -1;
    }

    const char *modelFile = DRWN_CMDLINE_ARGV[0];
    const char *imgDir = DRWN_CMDLINE_ARGV[1];

    // Load the classifier.
    drwnDecisionTree classifier;
    classifier.read(modelFile);

    // Create output directory if it does not already exist.
    if ((lblDirOut != NULL) && !drwnDirExists(lblDirOut)) {
        drwnCreateDirectory(lblDirOut);
    }

    // Get a list of images from the image directory.
    vector<string> baseNames = drwnDirectoryListing(imgDir, ".jpg", false, false);
    DRWN_LOG_MESSAGE("Loading " << baseNames.size() << " images and labels...");

    // Configure system for CRF stuff here

    // Iterate over images.
    for (unsigned i = 0; i < baseNames.size(); i++) {
        DRWN_LOG_STATUS("...processing image " << baseNames[i]);

        // Read the image and compute saliency.

        // Show the image and saliency.
        if (bVisualize) {
            // drwnDrawRegionBoundaries and drwnShowDebuggingImage use OpenCV 1.0 C API
            IplImage cvimg = (IplImage)img;
            CvMat cvseg = (CvMat)seg;
            IplImage *canvas = cvCloneImage(&cvimg);
            //drwnDrawRegionBoundaries(canvas, &cvseg, CV_RGB(255, 255, 255), 3);
            //drwnDrawRegionBoundaries(canvas, &cvseg, CV_RGB(255, 0, 0), 1);
            drwnShowDebuggingImage(canvas, "image", false);
            cvReleaseImage(&canvas);
        }

        // Extract saliency features and classify the pixels
       

        // put a "mask"/rectangle around salient areas.
    
        // Write out the image.
        if (lblDirOut != NULL) {
            cv::imwrite(string(lblDirOut) + DRWN_DIRSEP + baseNames[i] + string(".png"), lbls);
        }

        // Show results.
        if (bVisualize) {
            IplImage cvimg = (IplImage)lbls;
            drwnShowDebuggingImage(&cvimg, "results", false);
        }
    }

    // Clean up by freeing memory and printing profile information.
    cvDestroyAllWindows();
    drwnCodeProfiler::print();
    return 0;
}
