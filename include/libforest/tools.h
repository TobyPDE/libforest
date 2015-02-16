#ifndef LIBF_TOOLS_H
#define LIBF_TOOLS_H

#include <vector>

/**
 * This file contains some function that can evaluate the performance of a
 * learned classifier. 
 */

namespace libf {
    class DataStorage;
    class Classifier;
    
    /**
     * Computes the accuracy on the data set.
     */
    class AccuracyTool {
    public:
        /**
         * Returns the accuracy
         */
        float measure(const Classifier* classifier, const DataStorage* storage) const;
        
        /**
         * Prints the accuracy
         */
        void print(float accuracy) const;
        
        /**
         * Prints and measures the accuracy. 
         */
        void measureAndPrint(const Classifier* classifier, const DataStorage* storage) const;
    };
    
    /**
     * Computes the confusion matrix on the data set.
     */
    class ConfusionMatrixTool {
    public:
        /**
         * Returns the accuracy
         */
        void measure(const Classifier* classifier, const DataStorage* storage, std::vector< std::vector<float> > & result) const;
        
        /**
         * Prints the accuracy
         */
        void print(const std::vector< std::vector<float> > & result) const;
        
        /**
         * Prints and measures the accuracy. 
         */
        void measureAndPrint(const Classifier* classifier, const DataStorage* storage) const;
    };
    
    /**
     * Measures the correlation between the the trees of an ensemble by using
     * the hamming distance on their results. 
     */
#if 0
    class CorrelationTool {
    public:
        /**
         * Returns the correlation
         */
        void measure(const Classifier* classifier, const DataStorage* storage, std::vector< std::vector<float> > & result) const;
        
        /**
         * Prints the correlation
         */
        void print(const std::vector< std::vector<float> > & result) const;
        
        /**
         * Prints and measures the correlation. 
         */
        float measureAndPrint(const Classifier* classifier, const DataStorage* storage) const
        {
            std::vector< std::vector<float> > result;
            measure(classifier, storage, result);
            print(result);
        }
    };
#endif
}

#endif