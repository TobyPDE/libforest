#ifndef LIBF_TOOLS_H
#define LIBF_TOOLS_H

#include <vector>
#include <boost/filesystem.hpp>
#include <memory>

#include "data.h"
#include "classifiers.h"

/**
 * This file contains some function that can evaluate the performance of a
 * learned classifier. 
 */

namespace libf {
    class RandomForestLearner;
    class Estimator;
    class Gaussian;
    
    /**
     * Computes the accuracy on the data set.
     */
    class AccuracyTool {
    public:
        /**
         * Returns the accuracy
         */
        float measure(const AbstractClassifier::ptr classifier, AbstractDataStorage::ptr storage) const;
        
        /**
         * Prints the accuracy
         */
        void print(float accuracy) const;
        
        /**
         * Prints and measures the accuracy. 
         */
        void measureAndPrint(const AbstractClassifier::ptr classifier, AbstractDataStorage::ptr storage) const;
    };
    
    /**
     * Computes the confusion matrix on the data set.
     */
    class ConfusionMatrixTool {
    public:
        /**
         * Returns the accuracy
         */
        void measure(AbstractClassifier::ptr classifier, AbstractDataStorage::ptr storage, std::vector< std::vector<float> > & result) const;
        
        /**
         * Prints the accuracy
         */
        void print(const std::vector< std::vector<float> > & result) const;
        
        /**
         * Prints and measures the accuracy. 
         */
        void measureAndPrint(AbstractClassifier::ptr classifier, AbstractDataStorage::ptr storage) const;
    };
    
    /**
     * Measures the correlation between the the trees of an ensemble by using
     * the hamming distance on their results. 
     */
    class CorrelationTool {
    public:
        /**
         * Returns the correlation
         */
        void measure(const RandomForest::ptr classifier, AbstractDataStorage::ptr storage, std::vector< std::vector<float> > & result) const;
        
        /**
         * Prints the correlation
         */
        void print(const std::vector< std::vector<float> > & result) const;
        
        /**
         * Prints and measures the correlation. 
         */
        void measureAndPrint(const RandomForest::ptr classifier, AbstractDataStorage::ptr storage) const;
    };
    
    /**
     * Reports the variable importance computed during training.
     */
    class VariableImportanceTool {
    public:
        /**
         * Returns the variable importance (simple wrapper around 
         * getImportance).
         */
        virtual void measure(RandomForestLearner* learner, std::vector<float> & result) const;
        
        /**
         * Prints the variable importance
         */
        virtual void print(const std::vector<float> & result) const;
        
        /**
         * Retrieves (measures) and prints the variable importance
         */
        virtual void measureAndPrint(RandomForestLearner* learner) const;
    };
    
    /**
     * Backprojects the variable importance onto a square image with the given
     * given width/height.
     */
    class PixelImportanceTool : public VariableImportanceTool {
    public:
        
        /**
         * Retrieves variable importance and stores an image visualizing variable
         * importance where the image has size rows x rows.
         */
        void measureAndSave(RandomForestLearner* learner, boost::filesystem::path file, int rows) const;
    };
    
    
    /**
     * Computes the relative frequency of the individual classes in a data storage.
     */
    class ClassStatisticsTool {
    public:
        /**
         * Measures the relative class frequencies. The last entry of result
         * contains the number of data points without a label. 
         * 
         * @param storage The data storage to examine
         * @param result An array of relative frequencies
         */
        void measure(AbstractDataStorage::ptr storage, std::vector<float> & result) const;
        
        /**
         * Prints the accuracy
         * 
         * @param result An array of relative frequencies
         */
        void print(const std::vector<float> & result) const;
        
        /**
         * Prints and measures the accuracy. 
         * 
         * @param storage The data storage to examine
         */
        void measureAndPrint(AbstractDataStorage::ptr storage) const;
    };
    
    /*
     * Used to assess the quality of a density estimation given the true
     * Gaussian mixture density.
     */
    class GaussianKullbackLeiblerTool {
    public:
        
        /**
         * Measures the density accuracy using the Kulback-Leibler divergence on a discretegrid.
         */
        float measure(std::shared_ptr<Estimator> estimator, std::vector<Gaussian> & gaussians,
                const std::vector<float> & weights, int N);
        
        /**
         * Print the Kulback-Leibler divergence.
         */
        void print(float kl);
        
        /**
         * Measure and print the accuracy in terms of the Kulback-Leibler divergence.
         */
        void measureAndPrint(std::shared_ptr<Estimator> estimator, std::vector<Gaussian> & gaussians,
                const std::vector<float> & weights, int N);
        
    };
    
    class GaussianSquaredErrorTool {
    public:
        
        /**
         * Measures the density accuracy in terms for squared error.
         */
        float measure(std::shared_ptr<Estimator> estimator, std::vector<Gaussian> & gaussians,
                const std::vector<float> & weights, int N);
        
        /**
         * Print the squared error.
         */
        void print(float se);
        
        /**
         * Measure and print the accuracy in terms of the squared error
         */
        void measureAndPrint(std::shared_ptr<Estimator> estimator, std::vector<Gaussian> & gaussians,
                const std::vector<float> & weights, int N);
        
    };
}

#endif