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
map< string, vector<int> > parseLabel (const char * labelFileName) {
    map< string, vector<int> > fileLabelPairs;
    string line;
    ifstream labelFile ;
    labelFile.open(labelFileName);
    while ( !labelFile.eof() ){
        unsigned int pos;
        string imagePackage;
        string imageFilename;
        int widthOfImage, heightOfImage;
        vector<int> posRectangle;
        posRectangle.resize(4);
        // restore every line of the file
        getline(labelFile, line); 
        if (line.find( ".jpg") != std::string::npos) {
            // PARSE for the first line 
            // - which stores the image name and package
            pos = line.find("\\");
            imageFilename = line.substr(pos+1);
            imagePackage = line.substr(0, pos);
            // print for test
            cout << "imagePackage:" << imagePackage << "\t" << "imageFilename:" << imageFilename << endl;

            // PARSE for the SECOND line
            //  width and height of that training image
            getline(labelFile, line);
            pos = line.find(" ");
            widthOfImage = atoi(line.substr(0,pos).c_str());
            heightOfImage = atoi(line.substr(pos+1).c_str());
            // print for test
            cout << "Width:" << widthOfImage << "\t" << "Height:" << heightOfImage << endl;

            // PARSE for the third line
            // three saliency rectangles, each with four parameter
            getline(labelFile, line);
            string temp;
            unsigned int whiteSpacePos;
            for (int i = 0 ; i < 3; i ++) {
                pos = line.find(";");
                temp = line.substr(0, pos);
                sscanf(temp.c_str(), "%d %d %d %d", &posRectangle[0],
                        &posRectangle[1], &posRectangle[2], &posRectangle[3]);
                line = line.substr(pos+1);
                printf("Left: %3d \t Top: %4d \t Right: %4d \t Bottom: %4d \n",
                        posRectangle[0], posRectangle[1], posRectangle[2], posRectangle[3]);
            }
            cout << endl;
        }
    }
    labelFile.close();
    
    //fileLabelPairs.insert( );
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
    // second argument is not directory any more, 
    // it's a single text file with multiple rectangle
    //DRWN_ASSERT_MSG(drwnDirExists(lblDir), "labels directory " << lblDir << " does not exist");

    // Get a list of images from the image directory.
    vector<string> baseNames = drwnDirectoryListing(imgDir, ".jpg", false, false);
    DRWN_LOG_MESSAGE("Loading " << baseNames.size() << " images and labels...");

    /* Build a dataset by loading images and labels. For each image,
     find the salient area using the labels and then compute the set of features
     that determine this saliency.  Compute the values for the rest of the image
     as well (maybe as superpixels??? unsure.
    */
    drwnClassifierDataset dataset;

    parseLabel(lblDir);
    for (unsigned i = 0; i < baseNames.size(); i++) {
        DRWN_LOG_STATUS("...processing image " << baseNames[i]);
        // read the image and draw the rectangle of labels of training data
        cv::Mat img = cv::imread(string(imgDir) + DRWN_DIRSEP + baseNames[i] + string(".jpg"));
        cv::rectangle(img, cv::Point(89, 10), cv::Point(371, 252), Scalar(0,0,255));
        cv::rectangle(img, cv::Point(87, 9), cv::Point(379, 279), Scalar(0,255,0));
        cv::rectangle(img, cv::Point(89, 11), cv::Point(376, 275), Scalar(255,0,0));
        // show the image and superpixels
        if (bVisualize) { // draw the current image comparison
            //drwnDrawRegionBoundaries and drwnShowDebuggingImage use OpenCV 1.0 C API
            IplImage cvimg = (IplImage)img;
            //CvMat cvseg = (CvMat) seg;
            IplImage *canvas = cvCloneImage(&cvimg);
            //drwnDrawRegionBoundaries(canvas, &cvseg, CV_RGB(255, 255, 255), 3);
            //drwnDrawRegionBoundaries(canvas, &cvseg, CV_RGB(255, 0, 0), 1);
            drwnShowDebuggingImage(canvas, "image", false);
            cvReleaseImage(&canvas);
        }
    }

    // Clean up by freeing memory and printing profile information.
    cvDestroyAllWindows();
    drwnCodeProfiler::print();
    return 0;
}
