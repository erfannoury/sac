#ifndef ASLAM_SAMPLE_CONSENSUS_HPP
#define ASLAM_SAMPLE_CONSENSUS_HPP

#include <Eigen/Core>

namespace aslam {
    
    /// \brief Superclass for sample consensus
    template<typename PROBLEM_T>
    class SampleConsensus
    {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        typedef PROBLEM_T problem_t;
        typedef typename problem_t::model_t model_t;
        
        SampleConsensus(int maxIterations = 1000, double threshold = 1.0, double probability = 0.99);
        virtual ~SampleConsensus();
        
        virtual bool computeModel(int debug_verbosity_level = 0) = 0;

        // \todo accessors
        //private:
        int max_iterations_;
        int iterations_;
        double threshold_;
        double probability_;
        model_t model_coefficients_;
        std::vector<int> model_;
        std::vector<int> inliers_;
        boost::shared_ptr<PROBLEM_T> sac_model_;
    };


} // namespace aslam

#include "implementation/SampleConsensus.hpp"

#endif /* ASLAM_SAMPLE_CONSENSUS_HPP */
