/*****************************************************************************
** DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
** Distributed under the terms of the BSD license (see the LICENSE file)
** Copyright (c) 2007-2013, Stephen Gould
** All rights reserved.
**
******************************************************************************
** FILENAME:    scoreModel.cpp
** AUTHOR(S):   Chris Claoue-Long (u5183532@anu.edu.au)
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

#include "parseLabel.h"


using namespace std;
using namespace Eigen;

// usage ---------------------------------------------------------------------

void usage()
{
    cerr << DRWN_USAGE_HEADER << endl;
    cerr << "USAGE: ./score [OPTIONS] <resultLblFile> <truthLblFile>\n";
    cerr << "OPTIONS:\n"
         << "  -x                :: visualize\n"
         << DRWN_STANDARD_OPTIONS_USAGE
	 << endl;
}

// main ----------------------------------------------------------------------

int main(int argc, char *argv[]){
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
    
    const char *resultLbls = DRWN_CMDLINE_ARGV[0];
    const char *truthLbls = DRWN_CMDLINE_ARGV[1];
    
    DRWN_ASSERT_MSG(drwnFileExists(resultLbls), "Results file " << resultLbls << " does not exist");
    DRWN_ASSERT_MSG(drwnFileExists(truthLbls), "Ground truth file " << truthLbls << " does not exist");
    
    // Process labels and calculate distance between them
    DRWN_LOG_MESSAGE("Comparing resultant labels to ground truth...");

    map< string, vector<int> > resultPairs = parseLabel(resultLbls);
    map< string, vector<int> > truthPairs = parseLabel(truthLbls);
    vector<int> resultRect, truthRect;
    int topDisp, leftDisp, rightDisp, bottomDisp;
    float avgBDE;
    string currFile;

    // get the result and the truth labels for each file, calculate their Boundary-Displacement Error
    for (std::map<string, vector<int> >::iterator it = resultPairs.begin(); it != resultPairs.end(); ++it){
        currFile = it->first;
        truthRect = truthPairs.find(currFile)->second;
        resultRect = it->second;
        leftDisp = abs(truthRect.at(0)-resultRect.at(0));
        topDisp = abs(truthRect.at(1)-resultRect.at(1));
        rightDisp = abs(truthRect.at(2)-resultRect.at(2));
        bottomDisp = abs(truthRect.at(3)-resultRect.at(3));
        cout << it->first << "\n";
        cout << leftDisp << " " << topDisp << " " << rightDisp << " " << bottomDisp << "\n\n";
    }
    
    float avgAcc = 1000.0; // TODO return actual accuracy!
    cout << "Average accuracy: " << avgAcc << "\%\n";
    


    // Clean up by freeing memory and printing profile information.
    cvDestroyAllWindows();
    drwnCodeProfiler::print();
    return 0;
}
