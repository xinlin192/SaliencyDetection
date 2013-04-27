Main project for COMP2550 Sem1 2013 - Jimmy and Chris

This will focus on salient object detection - further details TBA

At the moment, the framework contains only a skeleton implementation (it probably
doesn't even compile yet), copied from the COMP3130warmup project code to be
refactored into something that is useable for salient object detection.

Update in Apr. 28th by Jimmy
1. I just implement the parsing text file for labels of training data. I would achieve the hash map for storing rectangle data as soon as possible.
2. After setting up the acceptance of training data, I would turn to address the CRF and mixtures of Gaussians (which is needed by third feature) implementation in DARWIN. 

Note that 
a. please keep file compilable, if not, please make it as soon as possible.
b. do not use /* */ to comment code within a function. If you do so, the same notation cannot seal up the whole block code.