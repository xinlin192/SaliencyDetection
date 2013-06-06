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


// BDEdistance ----------------------------------------------------------------

float BDEdistance(int largeLeft, int largeTop, int largeRight, int largeBottom, int smallLeft, int smallTop, int smallRight, int smallBottom){

    float dist = 0.0;
    float currDistance, minDistance;
    
    // calculate the minimum distance between each point x in the results and all the points in the truth label
    // add this result to the avgDistance for each point x
    minDistance = 10000.0; // ridiculously large number, there will always be a smaller value to take!
    
    for(int x = largeLeft; x <= largeRight; x++){
        for(int y = largeTop; y <= largeBottom; y++){
            if(!(x >= smallLeft && x <= smallRight && y >= smallTop && y <= smallBottom)){
                    
                // we have a non-zero displacement
                for(int smallx = smallLeft; smallx <= smallRight; smallx++){
                    for (int smally = smallTop; smally <= smallBottom; smally++){
                        currDistance = sqrt(pow((float)x-smallx,2) + pow((float)y-smally,2) );
                        if(currDistance < minDistance) minDistance = currDistance;
                        //cout << "CURR: " << currDistance << endl ;
                        //cout << "MIN: " << minDistance << endl << endl;
                    }
                }
                // add it to the accumulated distance
                dist += minDistance;
                minDistance = 10000.0; // reset to a ridiculously large number to get the smaller value once more

            }
        }
        cout << ".";
            
    }

    return dist;
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
    // float topDisp, leftDisp, rightDisp, bottomDisp;
    int rleft, rtop, rright, rbottom, tleft, ttop, tright, tbottom;
    float avgDistance = 0.0;
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
//         leftDisp = (rleft-tleft);
//         topDisp = (rtop-ttop);
//         rightDisp = (rright-tright);
//         bottomDisp = (rbottom-tbottom);
        // debugging
        //cout << it->first << "\n";
        //cout << leftDisp << " " << topDisp << " " << rightDisp << " " << bottomDisp << "\n\n";
        
        // get the largest area rectangle, use this to calculate the distance
        if((rright-rleft)*(rbottom-rtop) > (tright-tleft)*(tbottom-ttop))
            avgDistance += (BDEdistance(rleft, rtop, rright, rbottom, tleft, ttop, tright, tbottom)/((rright-rleft)*(rbottom-rtop) ) );
        else avgDistance += (BDEdistance(tleft, ttop, tright, tbottom, rleft, rtop, rright, rbottom)/((tright-tleft)*(tbottom-ttop) ) );

        cout << "Picture: " << currFile << endl;
        cout << "avg Distance so far: " << avgDistance << endl << endl;
    }
    
    // get the average BDE overal;
    avgDistance /= (float)resultPairs.size();
    cout << "Average Boundary Displacement Error: " << avgDistance << "\n";
    
    // Clean up by freeing memory and printing profile information.
    cvDestroyAllWindows();
    drwnCodeProfiler::print();
    return 0;
}