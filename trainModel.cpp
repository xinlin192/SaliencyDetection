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
#include <regex>

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

// copied from Stephen Gould's trainCOMP3130Model.cpp 2013 version
void usage(){
    cerr << DRWN_USAGE_HEADER << endl;
    cerr << "USAGE: ./trainModel [OPTIONS] <imgDir> <lblDir>\n";
    cerr << "OPTIONS:\n"
         << "  -o <model>        :: output model\n"
         << "  -x                :: visualize\n"
         << DRWN_STANDARD_OPTIONS_USAGE
	 << endl;
}

// parseLabel ----------------------------------------------------------------

/* Form a hash map between the filenames and integer vectors representing the 
 * salient rectangle boundaries in the corresponding images. 
 */
map< string, vector<int> > parseLabel (string filename) {
    std::map< string, vector<int> > fileLabelPairs;
    std::string line;
    std::ifstream input (filename) ;
    for (; std::getline(input, line); ){
        // need to normalise the lines somehow to ensure that we will always
        // get the right data.  Should we just 
    }
    
    fileLabelPairs.insert(/* stuff */);
    return fileLabelPairs;
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
    if (DRWN_CMDLINE_ARGC != 2) {
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
    const char *lblDir = DRWN_CMDLINE_ARGV[1];
    DRWN_ASSERT_MSG(drwnDirExists(imgDir), "image directory " << imgDir << " does not exist");
    DRWN_ASSERT_MSG(drwnDirExists(lblDir), "labels directory " << lblDir << " does not exist");

    // Get a list of images from the image directory.
    vector<string> baseNames = drwnDirectoryListing(imgDir, ".jpg", false, false);
    DRWN_LOG_MESSAGE("Loading " << baseNames.size() << " images and labels...");

    /* Build a dataset by loading images and labels. For each image,
     find the salient area using the labels and then compute the set of features
     that determine this saliency.  Compute the values for the rest of the image
     as well (maybe as superpixels??? unsure.
    */
    drwnClassifierDataset dataset;

    for (unsigned i = 0; i < baseNames.size(); i++) {
        DRWN_LOG_STATUS("...processing image " << baseNames[i]);
        // read the image and compute superpixels
        cv::Mat img = cv::imread(string(imgDir) + DRWN_DIRSEP + baseNames[i] + string(".jpg"));
        cv::rectangle(img, cv::Point(89, 10), cv::Point(371, 252), Scalar(0,0,255));
        cv::rectangle(img, cv::Point(87, 9), cv::Point(379, 279), Scalar(0,255,0));
        cv::rectangle(img, cv::Point(89, 11), cv::Point(376, 275), Scalar(255,0,0));
        // show the image and superpixels
        if (bVisualize) { // draw the current image comparison
            //drwnDrawRegionBoundaries and drwnShowDebuggingImage use OpenCV 1.0 C API
            IplImage cvimg = (IplImage)img;
            CvMat cvseg = (CvMat)seg;
            IplImage *canvas = cvCloneImage(&cvimg);
            drwnDrawRegionBoundaries(canvas, &cvseg, CV_RGB(255, 255, 255), 3);
            drwnDrawRegionBoundaries(canvas, &cvseg, CV_RGB(255, 0, 0), 1);
            drwnShowDebuggingImage(canvas, "image", false);
            cvReleaseImage(&canvas);
        }
    }

    // Clean up by freeing memory and printing profile information.
    cvDestroyAllWindows();
    drwnCodeProfiler::print();
    return 0;
}