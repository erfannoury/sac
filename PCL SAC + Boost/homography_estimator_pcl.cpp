#include "matrix.h"
#include "mex.h"


#include <iostream>
#include <math.h>
#include <chrono>
#include <random>
#include <vector>

#include <Eigen\Eigen>
#include <Eigen\LU>
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


class HomographyEstimatorProblem : public aslam::SampleConsensusProblem <MatrixXd>
{
public:
	HomographyEstimatorProblem() { }
	virtual ~HomographyEstimatorProblem() { }

	size_t numElements() const{ return points1_.rows(); }

	virtual int getSampleSize() const { return 4; }

	// Take a subset of size 4 of points and estimate homography matrix using them
	virtual bool computeModelCoefficients(const std::vector<int> & indices, MatrixXd& model) const
	{
		if (indices.size() != getSampleSize())
		{
			mexPrintf("Invalid set of sample points given (%d)!\n", indices.size());
			return false;
		}

		MatrixXd p1(indices.size(), 3);
		MatrixXd p2(indices.size(), 3);
		for (int i = 0; i < indices.size(); i++)
		{
			p1.row(i) = points1_.row(indices[i]);
			p2.row(i) = points2_.row(indices[i]);
		}
		auto H = calculateHomography(p1, p2);

		model = H;
		// end calculate homography
		return true;
	}

	virtual void optimizeModelCoefficients(const vector<int> & inliers, const MatrixXd& model, MatrixXd& optimized_model)
	{
		optimized_model = model;
	}

	// It will return a matrix H, the homography matrix, such that x1=H*x2
	// Note: points are assumed to be normalized and in homogeneous coordinate
	// Sizes of two sets of points must be equal. for performace purposes, this condition is not checked in this function
	// Note: since normalise2Dpoints works in-place on the points matrix, normalizing points inside this function will 
	// harm our collection of points. But because of two observations, we are allowed to normalise points inside this function
	// and it will cause no harm:
	//	1) Points from each step of ransac will be passed to this function to calculate the homography matrix. To do so, we will 
	//	   copy those points from the main point matrix to temporary matrices. So normalisation won't affect the original matrix.
	//  2) After exhausting the SAC operation, the final Homography matrix will be calculated using all of the inliers. After this
	//	   calculation, the program will return. So affecting the data points, won't do harm.
	MatrixXd calculateHomography(MatrixXd& x1, MatrixXd& x2) const
	{
		MatrixXd T1, T2;
		normalize2Dpoints(x1, T1);
		normalize2Dpoints(x2, T2);
		MatrixXd A = MatrixXd::Zero(3 * x1.rows(), 9);
		for (int i = 0; i < x1.rows(); i++)
		{
			A(3 * i, 3) = -x1(i, 0);
			A(3 * i, 4) = -x1(i, 1);
			A(3 * i, 5) = -x1(i,2);
			A(3 * i, 6) = x2(i, 1) * x1(i, 0);
			A(3 * i, 7) = x2(i, 1) * x1(i, 1);
			A(3 * i, 8) = x2(i, 1) * x1(i, 2);

			A(3 * i + 1, 0) = x1(i, 0);
			A(3 * i + 1, 1) = x1(i, 1);
			A(3 * i + 1, 2) = x1(i, 2);
			A(3 * i + 1, 6) = -x2(i, 0) * x1(i, 0);
			A(3 * i + 1, 7) = -x2(i, 0) * x1(i, 1);
			A(3 * i + 1, 8) = -x2(i, 0) * x1(i, 2);

			A(3 * i + 2, 0) = x2(i, 1) * x1(i, 0);
			A(3 * i + 2, 1) = x2(i, 1) * x1(i, 1);
			A(3 * i + 2, 2) = x2(i, 1) * x1(i, 2);
			A(3 * i + 2, 3) = x2(i, 0) * x1(i, 0);
			A(3 * i + 2, 4) = x2(i, 0) * x1(i, 1);
			A(3 * i + 2, 5) = x2(i, 0) * x1(i, 2);
		}
		JacobiSVD<MatrixXd> svd(A, ComputeThinV);
		svd.computeV();
		MatrixXd minEigenVector = MatrixXd(svd.matrixV().col(8));
		minEigenVector.resize(3, 3);
		auto H = T2.inverse() * minEigenVector * T1;
		return H;
	}

	// As the final step, estimate the homography matrix using all of the inlier correspondences
	void calculateModelUsingAllInliers(const vector<int>& inliers, MatrixXd& finalModel)
	{
		MatrixXd p1(inliers.size(), 3);
		MatrixXd p2(inliers.size(), 3);
		for (int i = 0; i < inliers.size(); i++)
		{
			p1.row(i) = points1_.row(inliers[i]);
			p2.row(i) = points2_.row(inliers[i]);
		}
		auto H = calculateHomography(p1, p2);

		finalModel = H;
	}

	/// evaluate the score for the elements at indices based on this model.
	/// low scores mean a good fit.
	virtual void getSelectedDistancesToModel(const MatrixXd& model, const vector<int> & indices, vector<double> & scores) const
	{
		scores.resize(indices.size());
		// Iterate through correspondences and calculate the projection error
		
		auto H = model;
		for (size_t i = 0; i < indices.size(); ++i)
		{
			auto x1 = points1_.row(indices[i]);
			auto x2 = points2_.row(indices[i]);
			auto Hx1 = H * x1;
			auto invHx2 = H.inverse() * x2;
			x1 /= x1(2);
			x2 /= x2(2);
			Hx1 /= Hx1(2);
			invHx2 /= invHx2(2);
			scores[i] = (x1 - invHx2).unaryExpr([](double a) {return a*a; }).sum() + (x2 - Hx1).unaryExpr([](double a) {return a*a; }).sum();
		}
	}

	// normalize 2d points so that they would have a zero mean and their mean distance from origin would be sqrt(2)
	void normalize2Dpoints(MatrixXd& points, MatrixXd& normalizationMat) const
	{
		double meanx = points.col(0).mean();
		double meany = points.col(1).mean();
		/*auto colx = points.col(0).unaryExpr([meanx](double x) { return x - meanx; });
		auto coly = points.col(1).unaryExpr([meany](double y) { return y - meany; });
		auto dist = (colx.unaryExpr([](double x) { return x*x; }) + coly.unaryExpr([](double y) {return y*y; })).unaryExpr([](double d) {return sqrt(d); });*/
		double meandist = (points.col(0).unaryExpr([meanx](double x) { return (x - meanx)*(x - meanx); }) + points.col(1).unaryExpr([meany](double y) { return (y - meany)*(y - meany); }))
			.unaryExpr([](double d){return sqrt(d); }).mean();
		double scale = sqrt(2) / meandist;

		MatrixXd T(3, 3);
		T << scale, 0, -scale*meanx
			, 0, scale, -scale*meany
			, 0, 0, 1;
		points = (T*points.transpose()).transpose();
		normalizationMat = T;
	}

	MatrixXd points1_;
	MatrixXd norm1_;
	MatrixXd points2_;
	MatrixXd norm2_;
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
	//vector<Correspondence> correspondences(dim1y);
	points1 = mxGetPr(prhs[0]);
	points2 = mxGetPr(prhs[1]);

	boost::shared_ptr<HomographyEstimatorProblem> homoproblem_ptr(new HomographyEstimatorProblem);
	HomographyEstimatorProblem& homoproblem = *homoproblem_ptr;
	homoproblem.points1_.resize(dim1y, 3);
	homoproblem.points1_.col(2).setOnes();
	homoproblem.points2_.resize(dim1y, 3);
	homoproblem.points2_.col(2).setOnes();
	homoproblem.setUniformIndices(dim1y);



	for (size_t j = 0; j < dim1y; j++)
	{
		/*Correspondence corr;
		corr.p1.x = points1[j];
		corr.p2.x = points2[j];
		corr.p1.y = points1[dim1y + j];
		corr.p2.y = points2[dim1y + j];*/
		//correspondences[j] = Correspondence(points1[j], points1[dim1y + j], points2[j], points2[dim1y + j]);
		//  mexPrintf("correspondence[%d] = (%lf, %lf; %lf, %lf)\n", j, points1[j], points1[dim1y + j], points2[j], points2[dim1y + j]);
		// mexPrintf("correspondence[%d] = (%lf, %lf; %lf, %lf)\n", j, correspondences[j].p1.x, correspondences[j].p1.y, correspondences[j].p2.x, correspondences[j].p2.y);
		homoproblem.points1_(j, 0) = points1[j];
		homoproblem.points1_(j, 1) = points1[dim1y + j];
		homoproblem.points2_(j, 0) = points2[j];
		homoproblem.points2_(j, 1) = points2[dim1y + j];
	}


	homoproblem.normalize2Dpoints(homoproblem.points1_, homoproblem.norm1_);
	homoproblem.normalize2Dpoints(homoproblem.points2_, homoproblem.norm2_);


	// HomographyEstimatorProblem *homoproblem = new HomographyEstimatorProblem();

	SampleConsensus<HomographyEstimatorProblem> *sac;
	if (!strcmp("ransac", modeStr))
	{
		sac = new Ransac<HomographyEstimatorProblem>(1000, 0.1, 0.9);
	}
	else if (!strcmp("lmeds", modeStr))
	{
		sac = new Lmeds<HomographyEstimatorProblem>(1000, 0.1, 0.9);
	}
	else if (!strcmp("msac", modeStr))
	{
		sac = new Msac<HomographyEstimatorProblem>(1000, 0.1, 0.9);
	}
	else if (!strcmp("prosac", modeStr))
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
	auto homographyModel = sac->model_coefficients_;
	mexPrintf("Number of iterations: %d\n", sac->iterations_);
	mexPrintf("Inliers size: %d (out of %d input correspondences)\n", sac->inliers_.size(), homoproblem.points1_.rows());

	homoproblem.calculateModelUsingAllInliers(sac->inliers_, homographyModel);
	auto H = homoproblem.norm2_.inverse() * homographyModel * homoproblem.norm1_;

	auto outArray = mxCreateDoubleMatrix(3, 3, mxREAL);
	double *outp = mxGetPr(outArray);

	// for (size_t i = 0; i < 3; i++)
	// {
	// 	for (size_t j = 0; j < 3; j++)
	// 	{
	// 		mexPrintf("homography matrix(%d, %d) = %lf\n", i, j, ransac.model_coefficients_.Matrix[j*3+i]);
	// 	}
	// }

	// Copy the estimated homography matrix to the output matrix, note the difference between MATLAB's column-wise arrays vs. C's row-wise arrays convention
	for (size_t i = 0; i < 3; i++)
	{
		for (size_t j = 0; i < 3; j++)
		{
			outp[i + 3 * j] = H(i, j);
		}
	}

	plhs[0] = outArray;
}
