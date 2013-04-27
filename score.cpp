/*****************************************************************************
** DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
** Distributed under the terms of the BSD license (see the LICENSE file)
** Copyright (c) 2007-2013, Stephen Gould
** All rights reserved.
**
******************************************************************************
** FILENAME:    scoreCOMP3130Results.cpp
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


using namespace std;
using namespace Eigen;

// usage ---------------------------------------------------------------------

void usage()
{
    cerr << DRWN_USAGE_HEADER << endl;
    cerr << "USAGE: ./score [OPTIONS] <resultsDir> <lblDir>\n";
    cerr << "OPTIONS:\n"
         << "  -x                :: visualize\n"
         << DRWN_STANDARD_OPTIONS_USAGE
	 << endl;
}

// main ----------------------------------------------------------------------

int main(int argc, char *argv[])
{
    // Set default value for optional command line arguments.
    bool bVisualize = false;

    // Process command line arguments using Darwin.
    DRWN_BEGIN_CMDLINE_PROCESSING(argc, argv)
        DRWN_CMDLINE_BOOL_OPTION("-x", bVisualize)
    DRWN_END_CMDLINE_PROCESSING(usage());

    // Check for the correct number of required arguments, otherwise
    // print usage statement and exit.
    if (DRWN_CMDLINE_ARGC != 2) {
        usage();
        return -1;
    }

    const char *resultsDir = DRWN_CMDLINE_ARGV[0];
    const char *labelsDir = DRWN_CMDLINE_ARGV[1];

    // Get a list of images from the results directory.
    vector<string> baseNames = drwnDirectoryListing(resultsDir, ".png", false, false);
    DRWN_LOG_MESSAGE("Analysing " << baseNames.size() << " results...");

    // Configure system for saliency dataset.
    //gMultiSegRegionDefs.initializeForDataset(DRWN_DS_MSRC);

    // How do we want to score this??? left confusion matrix in for the moment

    // Initialize the confusion matrix which counts how many times a
    // of a given class was classified as each of the possible target
    // classes.
    drwnConfusionMatrix confusion(gMultiSegRegionDefs.maxKey() + 1);

    for (unsigned i = 0; i < baseNames.size(); i++) {
        DRWN_LOG_STATUS("...processing image " << baseNames[i]);

        // read the predicted labels
        const string  predictedFilename = string(resultsDir) + DRWN_DIRSEP + baseNames[i] + string(".png");
        MatrixXi predicted = gMultiSegRegionDefs.convertImageToLabels(predictedFilename.c_str());

        // read the actual labels
        const string actualFilename = string(labelsDir) + DRWN_DIRSEP + baseNames[i] + string(".png");
        MatrixXi actual = gMultiSegRegionDefs.convertImageToLabels(actualFilename.c_str());

        // accumulate pixelwise accuracy
        for (int i = 0; i < predicted.rows(); i++) {
            for (int j = 0; j < predicted.cols(); j++) {
                if (actual(i, j) < 0) continue;
                confusion.accumulate(actual(i, j), predicted(i, j));
            }
        }
    }

    // Write out the confusion matrix and print the overall accuracy.
    confusion.write(cout);
    DRWN_LOG_MESSAGE("accuracy: " << confusion.accuracy());

    // Clean up by freeing memory and printing profile information.
    cvDestroyAllWindows();
    drwnCodeProfiler::print();
    return 0;
}
