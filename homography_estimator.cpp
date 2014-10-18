#include "matrix.h"
#include "mex.h"


#include <iostream>
#include <math.h>
#include <chrono>
#include <random>
#include <vector>
#include <Eigen\Dense>
#include <Eigen\SVD>


#include "estimator.h"
#include "prosac.h"
#include "mlesac.h"
#include "ransac.h"

using namespace std;
using namespace theia;
using namespace Eigen;

// Structure for a 2-D point
struct Point 
{
	double x;
	double y;
	Point() {}
	Point(double _x, double _y) : x(_x), y(_y) {}
	Point(const Point& p)
	{
		x = p.x;
		y = p.y;
	}
};

// A pair of points that correspond to each other. These two points are two keypoints in to separate images that have been matched to each other
struct Correspondence
{
	Point p1;
	Point p2;
	Correspondence() {}
	Correspondence(double x1, double y1, double x2, double y2)
	{
		p1.x = x1;
		p1.y = y1;
		p2.x = x2;
		p2.y = y2;
	}
	Correspondence(Point& _p1, Point& _p2) : p1(_p1), p2(_p2) {}
	Correspondence(const Correspondence& other) : p1(other.p1), p2(other.p2) { }
};

// Homography matrix is a 3-by-3 matrix which is the model we are trying to estimate
struct Homography
{
	double Matrix[9];
	Homography() {}
	Homography(const Homography& other)
	{
		for (size_t i = 0; i < 9; i++)
		{
			Matrix[i] = other.Matrix[i];
		}
	}
};

// This is the estimator class for estimating a homography matrix between two images. A model estimation method and error calculation method are implemented
class HomographyEstimator : public Estimator < Correspondence, Homography >
{
public:
	HomographyEstimator() {}
	~HomographyEstimator() {}



	int SampleSize() const { return 4; }
	bool EstimateModel(const vector<Correspondence>& data, vector<Homography>* models) const
	{
		Homography model;

		//calculate homography -- see gc_lecture_notes pp. 17-19
		MatrixXd Q = MatrixXd::Zero(2 * data.size(), 9);
		for (size_t i = 0; i < data.size(); i++)
		{
			Q(i, 0) = data[i].p2.x;
			Q(i, 1) = data[i].p2.y;
			Q(i, 2) = 1.0;
			Q(i, 6) = -1 * data[i].p2.x * data[i].p1.x;
			Q(i, 7) = -1 * data[i].p2.y * data[i].p1.x;
			Q(i, 8) = -1 * data[i].p1.x;

			Q(i + data.size(), 3) = data[i].p2.x;
			Q(i + data.size(), 4) = data[i].p2.y;
			Q(i + data.size(), 5) = 1.0;
			Q(i + data.size(), 6) = -1.0 * data[i].p2.x * data[i].p1.y;
			Q(i + data.size(), 7) = -1.0 * data[i].p2.y * data[i].p1.y;
			Q(i + data.size(), 8) = -1.0 * data[i].p1.y;
		}

		MatrixXd qtq = Q.transpose() * Q;
		JacobiSVD<MatrixXd> svd(qtq, ComputeThinU | ComputeThinV);
		svd.computeV();
		auto minEigenVector = svd.matrixV().col(8);
		for (size_t i = 0; i < 9; i++)
		{
			model.Matrix[i] = minEigenVector(i);
		}
		models->push_back(model);
		return true;
	}

	double Error(const Correspondence& correspondence, const Homography& model) const
	{
		// calculate reporojection error
		double projectionPointZ = model.Matrix[6] * correspondence.p1.x + model.Matrix[7] * correspondence.p1.y + model.Matrix[8];
		double projectionPointX = (model.Matrix[0] * correspondence.p1.x + model.Matrix[1] * correspondence.p1.y + model.Matrix[2]) / projectionPointZ;
		double projectionPointY = (model.Matrix[3] * correspondence.p1.x + model.Matrix[4] * correspondence.p1.y + model.Matrix[5]) / projectionPointZ;

		return (double) (fabs(projectionPointX - correspondence.p2.x) + fabs(projectionPointY - correspondence.p2.y));
	}
};

// This is the entry point from MATLAB to this routine
// plhs and prhs represent a pointer to the left hand side (output) and a pointer to the right hand side (input) respectively
// matlab function signature: H = homography_estimator(points1, points2, 'mode')
// H -> 9-by-9 homography matrix as output
// points1 -> n-by-2 matrix, coordinates of the keypoints of the first image
// points2 -> n-by-2 matrix, coordinates of the keypoints of the second image
// 'mode' -> {'ransac', 'prosac', 'mlesac'} 
void mexFunction(int nlhs, mxArray *plhs [], int nrhs, const mxArray*prhs [])
{
	if (nlhs != 1)
	{
		mexPrintf("Only one output expected\n");
		return;
	}
	if (nrhs != 3)
	{
		mexPrintf("Only three inputs expected\n");
		return;
	}

	const mwSize *dims1, *dims2;
	char* modeStr;
	double *points1, *points2;


	dims1 = mxGetDimensions(prhs[0]);
	dims2 = mxGetDimensions(prhs[1]);

	int dim1y = (int) dims1[0]; int dim1x = (int) dims1[1];
	int dim2y = (int) dims2[0]; int dim2x = (int) dims2[1];

	if (dim1x != dim2x || dim1y != dim2y)
	{
		mexPrintf("Error! two input points arrays must be of the same size\n");
		return;
	}

	if (dim1x != 2 || dim2x != 2)
	{
		mexPrintf("Error! Points must be 2-D!\n");
		return;
	}

	modeStr = mxArrayToString(prhs[2]);

	if (!modeStr || mxGetDimensions(prhs[2])[0] != 1)
	{
		mexPrintf("Error! Last argument must be an string!\n");
		return;
	}

	mexPrintf("string:\n%s\n", modeStr);
	vector<Correspondence> correspondences(dim1y);
	points1 = mxGetPr(prhs[0]);
	points2 = mxGetPr(prhs[1]);
	for (size_t j = 0; j < dim1y; j++)
	{
		/*Correspondence corr;
		corr.p1.x = points1[j];
		corr.p2.x = points2[j];
		corr.p1.y = points1[dim1y + j];
		corr.p2.y = points2[dim1y + j];*/
		correspondences[j] = Correspondence(points1[j], points1[dim1y + j], points2[j], points2[dim1y + j]);
		Homography homographyModel;
		RansacParameters params;
		params.error_thresh = 0.5;
		HomographyEstimator homographyEstimator;
		SampleConsensusEstimator<HomographyEstimator> *sacEstimator;
		// RANSAC mode
		if (!strcmp("ransac", modeStr))
		{
			sacEstimator = new Ransac<HomographyEstimator>(params, homographyEstimator);
		}
		// MLESAC mode
		else if (!strcmp("mlesac", modeStr))
		{
			sacEstimator = new Mlesac<HomographyEstimator>(params, homographyEstimator);
		}
		// PROSAC mode
		else if (!strcmp("prosac", modeStr))
		{
			sacEstimator = new Prosac<HomographyEstimator>(params, homographyEstimator);
		}
		// Unknown mode
		else
		{
			mexPrintf("Error! unknown estimation mode selected!\n");
			return;
		}


		sacEstimator->Initialize();
		RansacSummary summary;
		sacEstimator->Estimate(correspondences, &homographyModel, &summary);

		auto outArray = mxCreateDoubleMatrix(3, 3, mxREAL);
		double *outp = mxGetPr(outArray);
		// Copy the estimated homography matrix to the output matrix, note the difference between matrix's column-wise arrays vs. C's row-wise arrays convention
		for (size_t i = 0; i < 3; i++)
		{
			for (size_t j = 0; j < 3; j++)
			{
				outp[i * 3 + j] = homographyModel.Matrix[j * 3 + i];
			}
		}

		plhs[0] = outArray;
	}
}