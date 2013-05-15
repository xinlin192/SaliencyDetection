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

// parseLabel ----------------------------------------------------------------

/* Form a hash map between the filenames and integer vectors representing the 
 * salient rectangle boundaries in the corresponding images. 
 */
map< string, vector<int> > parseLabel (const char * labelFileName) {
    map< string, vector<int> > fileLabelPairs; // MAP FROM FILENAME TO RECTANGLE
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
            imageFilename = line.substr(pos+1); // get filename
            imagePackage = line.substr(0, pos); // get package number
            imageFilename = imageFilename.substr(0, imageFilename.size() - 1); // truncate string
            // test for printing
            //cout << "imageFilename:" << imageFilename  << endl;
            //cout << imageFilename.size() << endl;

            // PARSE for the SECOND line
            //  width and height of that training image
            //  use c lib's string 2 integer
            getline(labelFile, line);
            pos = line.find(" ");
            widthOfImage = atoi(line.substr(0,pos).c_str()); 
            heightOfImage = atoi(line.substr(pos+1).c_str()); 
            // print for test
            //cout << "Width:" << widthOfImage << "\t" << "Height:" << heightOfImage << endl;

            // PARSE for the third line
            // three saliency rectangles, each with four parameter
            getline(labelFile, line);
            string temp;
            for (int i = 0 ; i < 3; i ++) {
                pos = line.find(";");
                temp = line.substr(0, pos);
                sscanf(temp.c_str(), "%d %d %d %d", &posRectangle[0],
                        &posRectangle[1], &posRectangle[2], &posRectangle[3]);
                line = line.substr(pos+1);
                // printing for test
                //printf("Left: %3d \t Top: %4d \t Right: %4d \t Bottom: %4d \n",
                        //posRectangle[0], posRectangle[1], posRectangle[2], posRectangle[3]);
                if (i == 1) { // we choose second data here.
                    fileLabelPairs[imageFilename] = posRectangle;
                }
            }
            //cout << endl;
        }
    }
    labelFile.close();
    
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
            //cv::rectangle(img, cv::Point(left-1, top-1), cv::Point(right+1, bottom+1), Scalar(255,255,255));
            //cv::rectangle(img, cv::Point(left, top), cv::Point(right, bottom), Scalar(0,0,0));
            IplImage cvimg = (IplImage)img;
            //CvMat cvseg = (CvMat) seg;
            IplImage *canvas = cvCloneImage(&cvimg);
            //drwnDrawRegionBoundaries(canvas, &cvseg, CV_RGB(255, 255, 255), 3);
            //drwnDrawRegionBoundaries(canvas, &cvseg, CV_RGB(255, 0, 0), 1);
            drwnShowDebuggingImage(canvas, "image", false);
            cvReleaseImage(&canvas);
            cv::Mat cdi = getSpatialDistribution(img);
            cv::Mat pres (img.rows, img.cols, CV_8UC3);
            double grayscale;
            for (int y = 0 ; y < cdi.rows; y ++) {
                for (int x = 0 ; x < cdi.cols; x ++) {
                    grayscale = cdi.at<double>(y,x);
                    pres.at<Vec3b>(y,x) = Vec3b(grayscale*255, grayscale*255, grayscale*255);
                }
            }
            IplImage pcvimg = (IplImage) pres;
            IplImage *present = cvCloneImage(&pcvimg);
            drwnShowDebuggingImage(present, "Color Spatial Distribution", false);
            cvReleaseImage(&present);
            cvSaveImage((string(outputDir) + baseNames[i] + ".jpg").c_str(), present);
            cv::imwrite(string(outputDir) + baseNames[i] + ".jpg", pres);
        }
    }

    // Clean up by freeing memory and printing profile information.
    cvDestroyAllWindows();
    drwnCodeProfiler::print();
    return 0;
}
