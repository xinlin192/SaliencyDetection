/*****************************************************************************
* @file trainModel.c 
* @version 1.0
* @since 2013-04-24
* @author Jimmy Lin (u5223173) - u5223173@uds.anu.edu.au
* @author Christopher Claou'e-Long (u5183532) - u5183532@anu.edu.au
******************************************************************************/

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
#include "features.h"
#include "parseLabel.h"

using namespace std;
using namespace Eigen;

// usage ---------------------------------------------------------------------

// copied from Stephen Gould's trainCOMP3130Model.cpp 2013 version
void usage() {
    cerr << DRWN_USAGE_HEADER << endl;
    cerr << "USAGE: ./trainModel [OPTIONS] <imgDir> <lblFile>\n";
    cerr << "OPTIONS:\n"
         << "  -o <model>        :: output model\n"
         << "  -x                :: visualize\n"
         << DRWN_STANDARD_OPTIONS_USAGE
	 << endl;
}

// main ----------------------------------------------------------------------

int main (int argc, char * argv[]) {
    // what way are we building up a default classifier?
    // Set default value for optional command line arguments.
    const char *modelFile = NULL;
    bool bVisualize = false;

    DRWN_BEGIN_CMDLINE_PROCESSING(argc, argv)
        DRWN_CMDLINE_STR_OPTION("-o", modelFile)
        DRWN_CMDLINE_BOOL_OPTION("-x", bVisualize)
    DRWN_END_CMDLINE_PROCESSING(usage());

    // Check for the correct number of required arguments
    if (DRWN_CMDLINE_ARGC != 3) {
        usage();
        return -1;
    }

    /* Check that the image directory and labels directory exist. All
     * images with a ".jpg" extension will be used for training the
     * model. It is assumed that the labels directory contains files
     * with the same base as the image directory, but with extension
     * ".txt". 
     */
    const char *imgDir = DRWN_CMDLINE_ARGV[0];
    const char *lblFile = DRWN_CMDLINE_ARGV[1];
    const char *outputDir = DRWN_CMDLINE_ARGV[2];
    //DRWN_ASSERT_MSG(drwnDirExists(imgDir), "image directory " << imgDir << " does not exist");
    // second argument is not directory any more, 
    // it's a single text file with multiple rectangle
    //DRWN_ASSERT_MSG(drwnDirExists(lblFile), "labels directory " << lblFile << " does not exist");

    // Get a list of images from the image directory.
    vector<string> baseNames = drwnDirectoryListing(imgDir, ".jpg", false, false);
    DRWN_LOG_MESSAGE("Loading " << baseNames.size() << " images and labels...");

    /* Build a dataset by loading images and labels. For each image,
     find the salient area using the labels and then compute the set of features
     that determine this saliency.  Compute the values for the rest of the image
     as well (maybe as superpixels??? unsure.
    */
    drwnClassifierDataset dataset;
    //  MAP FROM FILENAME TO RECTANGLE
    map< string, vector<int> > fileLabelPairs = parseLabel(lblFile);
    int left, top, right, bottom;
    vector<int> tempRectangle;

    for (unsigned i = 0; i < baseNames.size(); i++) {
        String processedImage = baseNames[i] + ".jpg";
        DRWN_LOG_STATUS("...processing image " << baseNames[i]);
        // read the image and draw the rectangle of labels of training data
        cv::Mat img = cv::imread(string(imgDir) + DRWN_DIRSEP + processedImage);
        tempRectangle = fileLabelPairs.find(processedImage)->second ;
        left =  tempRectangle [0];
        top = tempRectangle [1];
        right = tempRectangle [2];
        bottom = tempRectangle[3];

        // show the image and feature maps 
        if (bVisualize) { // draw the current image comparison
            //drwnDrawRegionBoundaries and drwnShowDebuggingImage use OpenCV 1.0 C API
            IplImage cvimg = (IplImage)img;
            IplImage *canvas = cvCloneImage(&cvimg);
            drwnShowDebuggingImage(canvas, "image", false);
            cvReleaseImage(&canvas);

            cv::Mat labeledimg = cv::Mat(img);
            // draw rectangle .. label
            for (int l = 0; l < 8 ; l += 2) {
                cv::rectangle(labeledimg, cv::Point(left-l-1, top-l-1), cv::Point(right+l+1, bottom+l+1), Scalar(255,255,255));
                cv::rectangle(labeledimg, cv::Point(left-l, top-l), cv::Point(right+l, bottom+l), Scalar(0,0,0));
            }
            IplImage pcvimg = (IplImage) labeledimg;
            IplImage *present = cvCloneImage(&pcvimg);
            drwnShowDebuggingImage(present, "Labels", false);
            cvReleaseImage(&present);

            cv::imwrite(string(outputDir) + baseNames[i] + ".jpg", labeledimg);
        }
    }

    // Clean up by freeing memory and printing profile information.
    cvDestroyAllWindows();
    drwnCodeProfiler::print();
    return 0;
}
