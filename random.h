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

#ifndef THEIA_UTIL_RANDOM_H_
#define THEIA_UTIL_RANDOM_H_

namespace theia 
{
	namespace
	{
		std::default_random_engine util_generator;
	}  // namespace

	// Initializes the random generator to be based on the current time. Does not
	// have to be called before calling RandDouble, but it works best if it is.
	void InitRandomGenerator()
	{
		unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
		util_generator.seed(seed);
	}

	// Get a random double between lower and upper (inclusive).
	double RandDouble(double lower, double upper)
	{
		std::uniform_real_distribution<double> distribution(lower, upper);
		return distribution(util_generator);
	}

	// Get a random int between lower and upper (inclusive).
	int RandInt(int lower, int upper)
	{
		std::uniform_int_distribution<int> distribution(lower, upper);
		return distribution(util_generator);
	}

	// Gaussian Distribution with the corresponding mean and std dev.
	double RandGaussian(double mean, double std_dev)
	{
		std::normal_distribution<double> distribution(mean, std_dev);
		return distribution(util_generator);
	}


}  // namespace theia

#endif  // THEIA_UTIL_RANDOM_H_
