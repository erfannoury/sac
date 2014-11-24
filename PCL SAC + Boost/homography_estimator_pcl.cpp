#include "matrix.h"
#include "mex.h"


#include <iostream>
#include <math.h>
#include <chrono>
#include <random>
#include <vector>
#include <Eigen\Dense>
#include <Eigen\SVD>

#include "SampleConsensusProblem.hpp"
#include "SampleConsensus.hpp"
#include "Ransac.hpp"
#include "Msac.hpp"
#include "Prosac.hpp"
#include "Lmeds.hpp"

using namespace std;
using namespace Eigen;
using namespace aslam;
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

class HomographyEstimatorProblem : public aslam::SampleConsensusProblem <Homography>
{
public:
	HomographyEstimatorProblem() { }
	virtual ~HomographyEstimatorProblem() { }

	size_t numElements() const{ return points_.size(); }

	virtual int getSampleSize() const { return 4; }

	virtual bool computeModelCoefficients(const std::vector<int> & indices, Homography & model) const
	{
		if (indices.size() != getSampleSize())
		{
			mexPrintf("Invalid set of sample points given (%d)!\n", indices.size());
			return false;
		}

		//calculate homography -- see gc_lecture_notes pp. 17-19
		MatrixXd Q = MatrixXd::Zero(2 * indices.size(), 9);
		for (size_t i = 0; i < indices.size(); i++)
		{
			Q(i, 0) = points_[indices[i]].p2.x;
			Q(i, 1) = points_[indices[i]].p2.y;
			Q(i, 2) = 1.0;
			Q(i, 6) = -1 * points_[indices[i]].p2.x * points_[indices[i]].p1.x;
			Q(i, 7) = -1 * points_[indices[i]].p2.y * points_[indices[i]].p1.x;
			Q(i, 8) = -1 * points_[indices[i]].p1.x;

			Q(i + indices.size(), 3) = points_[indices[i]].p2.x;
			Q(i + indices.size(), 4) = points_[indices[i]].p2.y;
			Q(i + indices.size(), 5) = 1.0;
			Q(i + indices.size(), 6) = -1.0 * points_[indices[i]].p2.x * points_[indices[i]].p1.y;
			Q(i + indices.size(), 7) = -1.0 * points_[indices[i]].p2.y * points_[indices[i]].p1.y;
			Q(i + indices.size(), 8) = -1.0 * points_[indices[i]].p1.y;
			// mexPrintf("p1.x = %lf, p1.y = %ld; p2.x = %lf, p2.y = %lf\n", points_[indices[i]].p1.x, points_[indices[i]].p1.y, points_[indices[i]].p2.x, points_[indices[i]].p2.y);
		}

		MatrixXd qtq = Q.transpose() * Q;
		JacobiSVD<MatrixXd> svd(qtq, ComputeThinV);
		svd.computeV();
		MatrixXd minEigenVector = svd.matrixV().col(8);
		for (size_t i = 0; i < 9; i++)
		{
			model.Matrix[i] = minEigenVector(i) /*/minEigenVector(8)*/;
			// mexPrintf("minEigenVector(%d) = %lf\n", i, minEigenVector(i));
		}
		// end calculate homography
		return true;
	}

	virtual void optimizeModelCoefficients(const vector<int> & inliers, const Homography& model, Homography& optimized_model)
	{
		// optimized_model = Homography(model);
		for (int i = 0; i < 9; i++)
		{
			optimized_model.Matrix[i] = model.Matrix[i];
		}
	}

	void calculateModelUsingAllInliers(const vector<int>& inliers, Homography& finalModel)
	{
		MatrixXd Q = MatrixXd::Zero(2 * inliers.size(), 9);
		for (size_t i = 0; i < inliers.size(); i++)
		{
			Q(i, 0) = points_[inliers[i]].p2.x;
			Q(i, 1) = points_[inliers[i]].p2.y;
			Q(i, 2) = 1.0;
			Q(i, 6) = -1 * points_[inliers[i]].p2.x * points_[inliers[i]].p1.x;
			Q(i, 7) = -1 * points_[inliers[i]].p2.y * points_[inliers[i]].p1.x;
			Q(i, 8) = -1 * points_[inliers[i]].p1.x;

			Q(i + inliers.size(), 3) = points_[inliers[i]].p2.x;
			Q(i + inliers.size(), 4) = points_[inliers[i]].p2.y;
			Q(i + inliers.size(), 5) = 1.0;
			Q(i + inliers.size(), 6) = -1.0 * points_[inliers[i]].p2.x * points_[inliers[i]].p1.y;
			Q(i + inliers.size(), 7) = -1.0 * points_[inliers[i]].p2.y * points_[inliers[i]].p1.y;
			Q(i + inliers.size(), 8) = -1.0 * points_[inliers[i]].p1.y;
				// mexPrintf("p1.x = %lf, p1.y = %ld; p2.x = %lf, p2.y = %lf\n", points_[indices[i]].p1.x, points_[indices[i]].p1.y, points_[indices[i]].p2.x, points_[indices[i]].p2.y);
		}

		MatrixXd qtq = Q.transpose() * Q;
		JacobiSVD<MatrixXd> svd(qtq, ComputeThinV);
		svd.computeV();
		MatrixXd minEigenVector = svd.matrixV().col(8);
		for (size_t i = 0; i < 9; i++)
		{
			finalModel.Matrix[i] = minEigenVector(i) /*/minEigenVector(8)*/;
			// mexPrintf("minEigenVector(%d) = %lf\n", i, minEigenVector(i));
		}
	}

	/// evaluate the score for the elements at indices based on this model.
	/// low scores mean a good fit.
	virtual void getSelectedDistancesToModel(const Homography & model, const vector<int> & indices, vector<double> & scores) const
	{
		scores.resize(indices.size());
		// Iterate through correspondences and calculate the projection error
		for (size_t i = 0; i < indices.size(); ++i)
		{
			// calculate the reproject error
			auto correspondence = points_[indices[i]];

			double projectionPointZ = model.Matrix[6] * correspondence.p1.x + model.Matrix[7] * correspondence.p1.y + model.Matrix[8];
			double projectionPointX = (model.Matrix[0] * correspondence.p1.x + model.Matrix[1] * correspondence.p1.y + model.Matrix[2]) / projectionPointZ;
			double projectionPointY = (model.Matrix[3] * correspondence.p1.x + model.Matrix[4] * correspondence.p1.y + model.Matrix[5]) / projectionPointZ;

			// scores[i] = (double) (fabs(projectionPointX - correspondence.p2.x) + fabs(projectionPointY - correspondence.p2.y));
			scores[i] = (projectionPointX - correspondence.p2.x)*(projectionPointX - correspondence.p2.x) + (projectionPointY - correspondence.p2.y)*(projectionPointY - correspondence.p2.y);
		}
	}

	vector<Correspondence> points_;
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

	if (dim1y < 4)
	{
		mexPrintf("Not enough correspondences provided\n");
		return;
	}

	modeStr = mxArrayToString(prhs[2]);

	if (!modeStr || mxGetDimensions(prhs[2])[0] != 1)
	{
		mexPrintf("Error! Last argument must be an string!\n");
		return;
	}

	mexPrintf("string:\n%s\n", modeStr);
	mexPrintf("dimx = %d, dimy = %d\n", dim1x, dim1y);
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
		//  mexPrintf("correspondence[%d] = (%lf, %lf; %lf, %lf)\n", j, points1[j], points1[dim1y + j], points2[j], points2[dim1y + j]);
		// mexPrintf("correspondence[%d] = (%lf, %lf; %lf, %lf)\n", j, correspondences[j].p1.x, correspondences[j].p1.y, correspondences[j].p2.x, correspondences[j].p2.y);
	}


	// HomographyEstimatorProblem *homoproblem = new HomographyEstimatorProblem();
	boost::shared_ptr<HomographyEstimatorProblem> homoproblem_ptr(new HomographyEstimatorProblem);
	HomographyEstimatorProblem& homoproblem = *homoproblem_ptr;
	homoproblem.points_.resize(dim1y);
	homoproblem.setUniformIndices(dim1y);
	homoproblem.points_ = correspondences;

	SampleConsensus<HomographyEstimatorProblem> *sac;
	if(!strcmp("ransac", modeStr))
	{
		sac = new Ransac<HomographyEstimatorProblem>(1000, 0.1, 0.9);
	}
	else if(!strcmp("lmeds", modeStr))
	{
		sac = new Lmeds<HomographyEstimatorProblem>(1000, 0.1, 0.9);
	}
	else if(!strcmp("msac", modeStr))
	{
		sac = new Msac<HomographyEstimatorProblem>(1000, 0.1, 0.9);
	}
	else if(!strcmp("prosac", modeStr))
	{
		sac = new Prosac<HomographyEstimatorProblem>(1000, 0.1, 0.9);
	}
	else
	{
		mexPrintf("Error! unknown estimation mode selected!\n");
		return;
	}
	// if(!strcmp("rmsac", modeStr))
	// {
	// 	sac = new Rmsac<HomographyEstimatorProblem>(1000, 0.1, 0.9);
	// }

	// Msac<HomographyEstimatorProblem> ransac(1000, 0.1, 0.9);

	sac->sac_model_ = homoproblem_ptr;
	sac->computeModel(4);
	Homography homographyModel = Homography(sac->model_coefficients_);
	mexPrintf("Number of iterations: %d\n", sac->iterations_);
	mexPrintf("Inliers size: %d (out of %d input correspondences)\n", sac->inliers_.size(), homoproblem.points_.size());

	homoproblem.calculateModelUsingAllInliers(sac->inliers_, homographyModel);

	auto outArray = mxCreateDoubleMatrix(3, 3, mxREAL);
	double *outp = mxGetPr(outArray);

	// for (size_t i = 0; i < 3; i++)
	// {
	// 	for (size_t j = 0; j < 3; j++)
	// 	{
	// 		mexPrintf("homography matrix(%d, %d) = %lf\n", i, j, ransac.model_coefficients_.Matrix[j*3+i]);
	// 	}
	// }

	// Copy the estimated homography matrix to the output matrix, note the difference between matrix's column-wise arrays vs. C's row-wise arrays convention
	for (size_t i = 0; i < 3; i++)
	{
		for (size_t j = 0; j < 3; j++)
		{
			outp[i * 3 + j] = homographyModel.Matrix[j * 3 + i];
		}
	}

	plhs[0] = outArray;
	mexUnlock();
}
