#ifndef ASLAM_LMEDS_HPP
#define ASLAM_LMEDS_HPP

#include <boost/shared_ptr.hpp>
#include <vector>
#include "SampleConsensus.hpp"

namespace aslam {

    /// \brief plain old lmeds
    template<typename PROBLEM_T>
    class Lmeds : public SampleConsensus<PROBLEM_T>
    {
    public:
        typedef PROBLEM_T problem_t;
        typedef typename problem_t::model_t model_t;
        using SampleConsensus<problem_t>::max_iterations_;
        using SampleConsensus<problem_t>::threshold_;
        using SampleConsensus<problem_t>::iterations_;
        using SampleConsensus<problem_t>::sac_model_;
        using SampleConsensus<problem_t>::model_;
        using SampleConsensus<problem_t>::model_coefficients_;
        using SampleConsensus<problem_t>::inliers_;
        using SampleConsensus<problem_t>::probability_;

        Lmeds(int maxIterations = 1000, double threshold = 1.0, double probability = 0.99);
        virtual ~Lmeds();

        bool computeModel(int debug_verbosity_level = 0);

    };

} // namespace aslam

#include "implementation/Lmeds.hpp"

#endif /* ASLAM_LMEDS_HPP */
