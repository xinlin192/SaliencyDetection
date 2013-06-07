/*****************************************************************************
** DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
** Distributed under the terms of the BSD license (see the LICENSE file)
** Copyright (c) 2007-2013, Stephen Gould
** All rights reserved.
**
******************************************************************************
** FILENAME:    mexImageCRF.cpp
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

// openCV headers
#include "cxcore.h"

// darwin library headers
#include "drwnBase.h"
#include "drwnIO.h"
#include "drwnML.h"
#include "drwnPGM.h"
#include "drwnVision.h"

using namespace std;
using namespace Eigen;

// function prototypes --------------------------------------------------------

void addUnaryTerms(drwnMaxFlow *g, const vector< cv::Mat > unary,
    const cv::Mat labels, int alpha);

void addPairwiseTerms(drwnMaxFlow *g, const drwnPixelNeighbourContrasts contrast,
    double lambda, const cv::Mat labels, int alpha);

cv::Mat alphaExpansionTest( vector< cv::Mat > unary,
    const drwnPixelNeighbourContrasts &contrast, double lambda);

// main -----------------------------------------------------------------------

/*
void usage()
{
    mexPrintf(DRWN_USAGE_HEADER);
    mexPrintf("\n");
    mexPrintf("USAGE: x = mexImageCRF(image, unary, lambda, [options]);\n");
    mexPrintf("  image    :: H-by-W-by-3 image\n");
    mexPrintf("  unary    :: H-by-W-by-L unary potentials\n");
    mexPrintf("  lambda   :: contrast-sensitive pairwise smoothness weight (>= 0)\n");
    mexPrintf("OPTIONS:\n");
    drwnMatlabUtils::printStandardOptions();
    mexPrintf("  test     :: run alphaExpansion test code\n");
    mexPrintf("\n");
}
*/

cv::Mat mexFunction(cv::Mat img, vector< cv::Mat > unary, const double lambda)
{
    // parse image
    IplImage temp  = (IplImage) img;
    IplImage *image = cvCloneImage( & temp );
    const int H = image->height;
    const int W = image->width;

    drwnPixelNeighbourContrasts contrast;
    contrast.initialize(image);

    // parse unary potentials
    const int L = (int) unary.size();
    DRWN_ASSERT_MSG(L > 1, "invalid number of labels");
    DRWN_ASSERT_MSG((unary[0].rows == H) && (unary[0].cols == W),
        "unary potentials must match image size " << H << "-by-" << W);

    // parse pairwise contrast weight
    //const double lambda = mxGetScalar(prhs[2]);
    DRWN_ASSERT_MSG(lambda >= 0.0, "lambda must be non-negative");

    // initialize labeling
    cv::Mat labels = cv::Mat::zeros(H, W, CV_16S);
    for (int x = 0; x < W; x++) {
        for (int y = 0; y < H; y++) {
            double e = unary[0].at<double>(y, x);
            for (int l = 1; l < L; l++) {
                if (unary[l].at<double>(y, x) < e) {
                    e = unary[l].at<double>(y, x);
                    labels.at<short>(y, x) = l;
                }
            }
        }
    }

    // run alpha expansion
    drwnMaxFlow *g = new drwnBKMaxFlow(H * W);
    g->addNodes(H * W);

    bool bChanged = (lambda > 0.0);
    int lastChanged = -1;
    double minEnergy = numeric_limits<double>::max();
    for (int nCycle = 0; bChanged; nCycle += 1) {
        bChanged = false;
        for (int alpha = 0; alpha < L; alpha++) {
            if (alpha == lastChanged)
                break;

            g->reset();

            // add unary terms
            addUnaryTerms(g, unary, labels, alpha);

            // add pairwise terms
            addPairwiseTerms(g, contrast, lambda, labels, alpha);

            // run inference
            const double e = g->solve();

            DRWN_LOG_DEBUG("...cycle " << nCycle << ", iteration " << alpha << " has energy " << e);
            if (e < minEnergy) {
                minEnergy = e;
                lastChanged = alpha;
                bChanged = true;

                int varIndx = 0;
                for (int x = 0; x < W; x++) {
                    for (int y = 0; y < H; y++) {
                        if (g->inSetS(varIndx)) {
                            labels.at<short>(y, x) = alpha;
                        }
                        varIndx += 1;
                    }
                }
            }
        }
    }

    delete g;

    // testing code
    labels = alphaExpansionTest(unary, contrast, lambda);

    // release memory
    cvReleaseImage(&image);

    // print profile information
    drwnCodeProfiler::toc(drwnCodeProfiler::getHandle("mex"));
    drwnCodeProfiler::print();
    return labels;
}

// private functions -------------------------------------------------------


void addUnaryTerms(drwnMaxFlow *g, const vector< cv::Mat > unary,
    cv::Mat labels, int alpha)
{
    const int H = labels.rows;
    const int W = labels.cols;

    int varIndx = 0;
    for (int x = 0; x < W; x++) {
        for (int y = 0; y < H; y++) {
            g->addSourceEdge(varIndx, unary[labels.at<short>(y, x)].at<double>(y, x));
            g->addTargetEdge(varIndx, unary[alpha].at<double>(y, x));
            varIndx += 1;
        }
    }
}

void addPairwiseTerms(drwnMaxFlow *g, const drwnPixelNeighbourContrasts contrast,
    double lambda, cv::Mat labels, int alpha)
{
    const int H = labels.rows;
    const int W = labels.cols;

    // add horizontal pairwise terms
    for (int x = 1; x < W; x++) {
        for (int y = 0; y < H; y++) {
            const int u = H * x + y;
            const int v = H * (x - 1) + y;

            const double w = lambda * contrast.contrastW(x, y);

            const int labelA = labels.at<short>(y, x);
            const int labelB = labels.at<short>(y, x - 1);

            if ((labelA == alpha) && (labelB == alpha)) continue;

            if (labelA == alpha) {
                g->addSourceEdge(v, w);
            } else if (labelB == alpha) {
                g->addSourceEdge(u, w);
            } else if (labelA == labelB) {
                g->addEdge(u, v, w, w);
            } else {
                g->addSourceEdge(u, w);
                g->addEdge(u, v, w, 0.0);
            }
        }
    }

    // add vertical pairwise terms
    for (int x = 0; x < W; x++) {
        for (int y = 1; y < H; y++) {
            const int u = H * x + y;
            const int v = H * x + y - 1;

            const double w = lambda * contrast.contrastN(x, y);

            const int labelA = labels.at<short>(y, x);
            const int labelB = labels.at<short>(y - 1, x);

            if ((labelA == alpha) && (labelB == alpha)) continue;

            if (labelA == alpha) {
                g->addSourceEdge(v, w);
            } else if (labelB == alpha) {
                g->addSourceEdge(u, w);
            } else if (labelA == labelB) {
                g->addEdge(u, v, w, w);
            } else {
                g->addSourceEdge(u, w);
                g->addEdge(u, v, w, 0.0);
            }
        }
    }

    // add diagonal pairwise terms
    for (int x = 1; x < W; x++) {
        for (int y = 1; y < H; y++) {
            const int u = H * x + y;
            const int v = H * (x - 1) + y - 1;

            const double w = lambda * contrast.contrastNW(x, y);

            const int labelA = labels.at<short>(y, x);
            const int labelB = labels.at<short>(y - 1, x - 1);

            if ((labelA == alpha) && (labelB == alpha)) continue;

            if (labelA == alpha) {
                g->addSourceEdge(v, w);
            } else if (labelB == alpha) {
                g->addSourceEdge(u, w);
            } else if (labelA == labelB) {
                g->addEdge(u, v, w, w);
            } else {
                g->addSourceEdge(u, w);
                g->addEdge(u, v, w, 0.0);
            }
        }
    }

    for (int x = 1; x < W; x++) {
        for (int y = 1; y < H; y++) {
            const int u = H * x + y - 1;
            const int v = H * (x - 1) + y;

            const double w = lambda * contrast.contrastSW(x, y - 1);

            const int labelA = labels.at<short>(y - 1, x);
            const int labelB = labels.at<short>(y, x - 1);

            if ((labelA == alpha) && (labelB == alpha)) continue;

            if (labelA == alpha) {
                g->addSourceEdge(v, w);
            } else if (labelB == alpha) {
                g->addSourceEdge(u, w);
            } else if (labelA == labelB) {
                g->addEdge(u, v, w, w);
            } else {
                g->addSourceEdge(u, w);
                g->addEdge(u, v, w, 0.0);
            }
        }
    }
}

cv::Mat alphaExpansionTest(vector< cv::Mat > unary,
    const drwnPixelNeighbourContrasts &contrast, double lambda)
{
    DRWN_FCN_TIC;

    const int L = (int)unary.size();
    const int H = contrast.height();
    const int W = contrast.width();

    // create graph
    drwnVarUniversePtr universe(new drwnVarUniverse(H * W, L));
    drwnFactorGraph graph(universe);

    // add unary terms
    for (int i = 0; i < H * W; i++) {
        drwnTableFactor *phi = new drwnTableFactor(universe);
        phi->addVariable(i);
        for (int xi = 0; xi < L; xi++) {
            (*phi)[xi] = unary[xi].at<double>(i % H, i / H);
        }
        graph.addFactor(phi);
    }

    // add horizontal pairwise terms
    for (int x = 1; x < W; x++) {
        for (int y = 0; y < H; y++) {
            const int u = H * x + y;
            const int v = H * (x - 1) + y;

            const double w = lambda * contrast.contrastW(x, y);

            drwnTableFactor *phi = new drwnTableFactor(universe);
            phi->addVariable(u);
            phi->addVariable(v);

            for (int xi = 0; xi < L; xi++) {
                for (int xj = 0; xj < L; xj++) {
                    (*phi)[xi * L + xj] = (xi == xj) ? 0.0 : w;
                }
            }

            graph.addFactor(phi);
        }
    }

    // add vertical pairwise terms
    for (int x = 0; x < W; x++) {
        for (int y = 1; y < H; y++) {
            const int u = H * x + y;
            const int v = H * x + y - 1;

            const double w = lambda * contrast.contrastN(x, y);

            drwnTableFactor *phi = new drwnTableFactor(universe);
            phi->addVariable(u);
            phi->addVariable(v);

            for (int xi = 0; xi < L; xi++) {
                for (int xj = 0; xj < L; xj++) {
                    (*phi)[xi * L + xj] = (xi == xj) ? 0.0 : w;
                }
            }

            graph.addFactor(phi);
        }
    }

    // add diagonal pairwise terms
    for (int x = 1; x < W; x++) {
        for (int y = 1; y < H; y++) {
            const int u = H * x + y;
            const int v = H * (x - 1) + y - 1;

            const double w = lambda * contrast.contrastNW(x, y);

            drwnTableFactor *phi = new drwnTableFactor(universe);
            phi->addVariable(u);
            phi->addVariable(v);

            for (int xi = 0; xi < L; xi++) {
                for (int xj = 0; xj < L; xj++) {
                    (*phi)[xi * L + xj] = (xi == xj) ? 0.0 : w;
                }
            }

            graph.addFactor(phi);
        }
    }

    for (int x = 1; x < W; x++) {
        for (int y = 1; y < H; y++) {
            const int u = H * x + y - 1;
            const int v = H * (x - 1) + y;

            const double w = lambda * contrast.contrastSW(x, y - 1);

            drwnTableFactor *phi = new drwnTableFactor(universe);
            phi->addVariable(u);
            phi->addVariable(v);

            for (int xi = 0; xi < L; xi++) {
                for (int xj = 0; xj < L; xj++) {
                    (*phi)[xi * L + xj] = (xi == xj) ? 0.0 : w;
                }
            }

            graph.addFactor(phi);
        }
    }

    // run inference
    drwnAlphaExpansionInference inf(graph);
    drwnFullAssignment assignment;
    double e = inf.inference(assignment);

    cv::Mat labels(H, W, CV_16S);
    for (int i = 0; i < H * W; i++) {
        labels.at<short>(i % H, i / H) = assignment[i];
    }

    DRWN_FCN_TOC;
    return labels;
}
