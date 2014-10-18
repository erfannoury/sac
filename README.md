sac
===

#Code Structure

These codes consist of two parts.
1. Sample Consensus methods and implementations

Currently only consensus methods of `ransac`, `prosac` and `mlesac` are included. Other methods maybe added later.
2. Model estimation implementations

Currently only two model estimations are included as samples, `homography_estimator` and `line_estimator`. You can implement other model estimators based on these samples provided. More documentation on this will be added later.

#Code origin
__These codes are not mine__!
I have copied these implementations from this great project named __[Theia](https://github.com/kip622/Theia)__. Great work has been put into this work, so if you use this code in your publication, please cite this project. Details on how to cite this project is provided [Here](http://cs.ucsb.edu/~cmsweeney/theia/index.html). I warmly thank [Chris Sweeney](http://cs.ucsb.edu/~cmsweeney).

I have changed these codes so that model estimation code can be compiled using `MEX` in MATLAB. See `homography_estimator.cpp` for an implementation of the Homography estimation model and also `line_estimator.cpp`  for an implementation of the Line estimation model. You can use these templates to implement other estimation models.

#Dependencies

Main project is dependent on some external projects. However almost all of these dependencies are removed in the provided code. Though the new code should be double-checked to search for the new bugs I might have introduced.

Nevertheless, `Eigen` matrix library remains as the only dependency. This is a header-only library, so there wouldn't be any serious problems using this library. You can get Eigen from [here](eigen.tuxfamily.org/).

#How to compile using mex
First you should have a compatible compiler installed in your system.

Then run `mex -setup` and setup your compiler of choice to be used by MATLAB.

Finally, to compile the estimation method, run `mex <model.cpp> -I"path\to\Eigen\folder"`.

e.g. to compile the Homography estimator model, run:

`mex homography_estimator.cpp -I"path\to\Eigen\folder"`

#Syntax for using Estimation models:
###Homography Estimator:
```matlab
H = homography_estimator(points1, points2, 'mode')
% H -> 9-by-9 homography matrix as output
% points1 -> n-by-2 matrix, coordinates of the keypoints of the first image
% points2 -> n-by-2 matrix, coordinates of the keypoints of the second image
% for all 1 <= i <= n : points1[i,:] correspondents to points2[i,:]
% 'mode' -> {'ransac', 'prosac', 'mlesac'}
```

#Future Work
If you have requests for implementation, new model estimators or everything else regarding this project, create a new issue.

#Copyright
This work is governed by an MIT license.
Also the original license of the Theia project is provided for notice.
```
// Copyright (C) 2013 The Regents of the University of California (Regents).
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
//
//     * Redistributions in binary form must reproduce the above
//       copyright notice, this list of conditions and the following
//       disclaimer in the documentation and/or other materials provided
//       with the distribution.
//
//     * Neither the name of The Regents or University of California nor the
//       names of its contributors may be used to endorse or promote products
//       derived from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
// Please contact the author of this library if you have any questions.
// Author: Chris Sweeney (cmsweeney@cs.ucsb.edu)
```

###This is a project for the [IPL@Sharif](http://ipl.ce.sharif.edu).
