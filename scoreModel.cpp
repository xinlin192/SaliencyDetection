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
    float topDisp, leftDisp, rightDisp, bottomDisp;
    int rleft, rtop, rright, rbottom, tleft, ttop, tright, tbottom;
    float avgBDE = 0;
    float avgDistance, currDistance, minDistance;
    string currFile;

    // get the result and the truth labels for each file, calculate their Boundary-Displacement Error
    for (std::map<string, vector<int> >::iterator it = resultPairs.begin(); it != resultPairs.end(); ++it){
        currFile = it->first;
        if(truthPairs.find(currFile) == truthPairs.end()){
            cerr << "ERROR FINDING MAP FROM " << currFile << " TO BOUNDING RECTANGLE IN TRUTH LABELS\n";
            return -1;
        }
        truthRect = truthPairs.find(currFile)->second;
        resultRect = it->second;
        rleft = resultRect.at(0);
        tleft = truthRect.at(0);
        rtop = resultRect.at(1);
        ttop = truthRect.at(1);
        rright = resultRect.at(2);
        tright = truthRect.at(2);
        rbottom = resultRect.at(3);
        tbottom = truthRect.at(3);
        leftDisp = (rleft-tleft);
        topDisp = (rtop-ttop);
        rightDisp = (rright-tright);
        bottomDisp = (rbottom-tbottom);
        // debugging
        cout << it->first << "\n";
        cout << leftDisp << " " << topDisp << " " << rightDisp << " " << bottomDisp << "\n\n";
        
        
        // calculate the minimum distance between each point x in the results and all the points in the truth label
        // add this result to the avgDistance for each point x
        avgDistance = 0.0;
        minDistance = 10000.0; // ridiculously large number, there will always be a smaller value to take!
        for(int rcol = rleft; rcol < rright; rcol++){
            for(int rrow = rtop; rrow < rbottom; rrow++){
                for(int tcol = tleft; tcol < tright; tcol++){
                    for (int trow = ttop; trow < tbottom; trow++){
                        currDistance = sqrt(pow((float)rcol-tcol,2) + pow((float)rrow-trow,2) );
                        if(currDistance < minDistance) minDistance = currDistance;
                    }
                }
                
                avgDistance += minDistance;
            }
            
        }
        // add this to the overall BDE (it will be 0 if a perfect label, else slightly off)
        avgBDE += avgDistance;

    }
    
    // get the average BDE overal;
    avgBDE /= (float)resultPairs.size();
    cout << "Average Boundary Displacement Error: " << avgBDE << "\n";
    


    // Clean up by freeing memory and printing profile information.
    cvDestroyAllWindows();
    drwnCodeProfiler::print();
    return 0;
}
