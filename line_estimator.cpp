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

#include "matrix.h"
#include "mex.h"

#include <math.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <random>

#include "estimator.h"
#include "prosac.h"
#include "ransac.h"
#include "mlesac.h"

using namespace std;
using namespace theia;

struct Point
{
	double x;
	double y;
	Point() {}
	Point(double _x, double _y) : x(_x), y(_y) {}
	Point(const Point& other) : x(other.x), y(other.y) {}
};

// y = mx + b
struct Line
{
	double m;
	double b;
	Line() {}
	Line(double _m, double _b) : m(_m), b(_b) {}
	Line(const Line& other) : m(other.m), b(other.b) {}
};

class LineEstimator : public Estimator < Point, Line >
{
public:
	LineEstimator() {}
	~LineEstimator() {}

	int SampleSize() const { return 2; }

	bool EstimateModel(const std::vector<Point>& data, std::vector<Line>* models) const
	{
		Line model;
		model.m = (data[1].y - data[0].y) / (data[1].x - data[0].x);
		model.b = data[1].y - model.m * data[1].x;
		models->push_back(model);
		return true;
	}

	double Error(const Point& point, const Line& line) const
	{
		double a = -1.0 * line.m;
		double b = 1.0;
		double c = -1.0 * line.b;

		return fabs(a * point.x + b * point.y + c) / sqrt(a * a + b * b);
	}
};

void mexFunction(int nlhs, mxArray *plhs [], int nrhs, const mxArray*prhs [])
{
	if (nlhs != 1)
	{
		mexPrintf("Only one output expected\n");
		return;
	}
	if (nrhs != 2)
	{
		mexPrintf("Only two inputs expected\n");
		return;
	}

	const mwSize *dims;
	char* modeStr;
	double *points;


	dims = mxGetDimensions(prhs[0]);

	int dimy = (int) dims[0]; int dimx = (int) dims[1];

	if (dimx != 2)
	{
		mexPrintf("Error! Points must be 2-D!\n");
		return;
	}

	modeStr = mxArrayToString(prhs[1]);

	if (!modeStr || mxGetDimensions(prhs[1])[0] != 1)
	{
		mexPrintf("Error! Last argument must be an string!\n");
		return;
	}

	mexPrintf("string:\n%s\n", modeStr);
	vector<Point> pnts(dimy);
	points = mxGetPr(prhs[0]);
	for (size_t j = 0; j < dimy; j++)
	{
		/*Correspondence corr;
		corr.p1.x = points1[j];
		corr.p2.x = points2[j];
		corr.p1.y = points1[dim1y + j];
		corr.p2.y = points2[dim1y + j];*/
		pnts[j] = Point(points[j], points[dimy + j]);
		Line lineModel;
		RansacParameters params;
		params.error_thresh = 0.5;
		LineEstimator lineEstimator;
		SampleConsensusEstimator<LineEstimator> *sacEstimator;
		// RANSAC mode
		if (!strcmp("ransac", modeStr))
		{
			sacEstimator = new Ransac<LineEstimator>(params, lineEstimator);
		}
		// MLESAC mode
		else if (!strcmp("mlesac", modeStr))
		{
			sacEstimator = new Mlesac<LineEstimator>(params, lineEstimator);
		}
		// PROSAC mode
		else if (!strcmp("prosac", modeStr))
		{
			sacEstimator = new Prosac<LineEstimator>(params, lineEstimator);
		}
		// Unknown mode
		else
		{
			mexPrintf("Error! unknown estimation mode selected!\n");
			return;
		}


		sacEstimator->Initialize();
		RansacSummary summary;
		sacEstimator->Estimate(pnts, &lineModel, &summary);

		auto outArray = mxCreateDoubleMatrix(1, 2, mxREAL);
		double *outp = mxGetPr(outArray);
		// Copy the estimated homography matrix to the output matrix, note the difference between matrix's column-wise arrays vs. C's row-wise arrays convention
		outp[0] = lineModel.m;
		outp[1] = lineModel.b;

		plhs[0] = outArray;
	}
}


