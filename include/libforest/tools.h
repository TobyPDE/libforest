#ifndef LIBF_TOOLS_H
#define LIBF_TOOLS_H

#include <vector>
#include <boost/filesystem.hpp>

/**
 * This file contains some function that can evaluate the performance of a
 * learned classifier. 
 */

namespace libf {
    class DataStorage;
    class Classifier;
    class RandomForest;
    class RandomForestLearner;
    
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
    class CorrelationTool {
    public:
        /**
         * Returns the correlation
         */
        void measure(const RandomForest* classifier, const DataStorage* storage, std::vector< std::vector<float> > & result) const;
        
        /**
         * Prints the correlation
         */
        void print(const std::vector< std::vector<float> > & result) const;
        
        /**
         * Prints and measures the correlation. 
         */
        void measureAndPrint(const RandomForest* classifier, const DataStorage* storage) const;
    };
    
    /**
     * Reports the variable importance computed during training.
     */
    class VariableImportanceTool {
    public:
        /**
         * Returns the variable importance (simple wrapper around 
         * getMDIImportance).
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
}

#endif