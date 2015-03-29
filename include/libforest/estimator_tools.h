#ifndef LIBF_ESTIMATOR_TOOLS_H
#define LIBF_ESTIMATOR_TOOLS_H

#include "estimator.h"

namespace libf {
    

    /*
     * Used to assess the quality of a density estimation given the true
     * Gaussian mixture density.
     */
    class GaussianKullbackLeiblerTool {
    public:
        
        /**
         * Measures the density accuracy using the Kulback-Leibler divergence on a discretegrid.
         */
        float measure(std::shared_ptr<EstimatorInterface> estimator, std::vector<Gaussian> & gaussians,
                const std::vector<float> & weights, int N);
        
        /**
         * Print the Kulback-Leibler divergence.
         */
        void print(float kl);
        
        /**
         * Measure and print the accuracy in terms of the Kulback-Leibler divergence.
         */
        void measureAndPrint(std::shared_ptr<EstimatorInterface> estimator, std::vector<Gaussian> & gaussians,
                const std::vector<float> & weights, int N);
        
    };
    
    class GaussianSquaredErrorTool {
    public:
        
        /**
         * Measures the density accuracy in terms for squared error.
         */
        float measure(std::shared_ptr<EstimatorInterface> estimator, std::vector<Gaussian> & gaussians,
                const std::vector<float> & weights, int N);
        
        /**
         * Print the squared error.
         */
        void print(float se);
        
        /**
         * Measure and print the accuracy in terms of the squared error
         */
        void measureAndPrint(std::shared_ptr<EstimatorInterface> estimator, std::vector<Gaussian> & gaussians,
                const std::vector<float> & weights, int N);
        
    };
    
}

#endif