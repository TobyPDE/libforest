#ifndef MCMCF_CLASSIFIERS_H
#define MCMCF_CLASSIFIERS_H

/**
 * This file contains the data structures for the classifiers. There are 
 * basically two kinds ot classifiers:
 * 1. Decision trees
 * 2. Random forests
 */

#include <iostream>
#include <vector>

namespace libf {
    /**
     * Forward declarations to speed up compiling
     */
    class DataStorage;
    class DataPoint;
    
    /**
     * The base class for all classifiers. This allows use to use the evaluation
     * tools for both trees and forests. 
     */
    class Classifier {
    public:
        /**
         * Assigns an integer class label to some data point
         */
        virtual int classify(DataPoint* x) const = 0;
        
        /**
         * Classifies an entire data set and uses the integer values. 
         */
        virtual void classify(DataStorage* storage, std::vector<int> & results) const;
        
        /**
         * Assigns a class label to some data point
         */
        //virtual std::string classify(DataPoint* x) const;
        
        /**
         * Classifies an entire data set. 
         */
        //virtual void classify(DataStorage* storage, std::vector<std::string> & results) const;
        
        /**
         * Outputs the class probabilities for a given data point.
         */
        //virtual void classProbabilities(DataPoint* x, std::vector<float> & probabilities) const = 0;
        
        /**
         * Reads the classifier from a stream
         */
        virtual void read(std::istream & stream) = 0;
        
        /**
         * Writes the classifier to a stream
         */
        virtual void write(std::ostream & stream) const = 0;
        
        /**
         * Sets the class label map
         */
        void setClassLabelMap(const std::vector<std::string> & _classLabelMap) 
        {
            classLabelMap = _classLabelMap;
        }
        
        /**
         * Returns the class label map
         */
        const std::vector<std::string> & getClassLabelMap() const
        {
            return classLabelMap;
        }
        
    protected:
        /**
         * The class label map: int -> string
         */
        std::vector<std::string> classLabelMap;
    };
    
    /**
     * This class represents a decision tree.
     */
    class DecisionTree : public Classifier {
    public:
        /**
         * Creates a new decision tree
         */
        DecisionTree();
        
        /**
         * Sets the split feature for a node
         */
        void setSplitFeature(int node, int feature)
        {
            splitFeatures[node] = feature;
        }
        
        /**
         * Sets the threshold for a node
         */
        void setThreshold(int node, float threshold)
        {
            thresholds[node] = threshold;
        }
        
        /**
         * Sets the class label for a node
         */
        void setClassLabel(int node, int label)
        {
            classLabels[node] = label;
        }
        
        /**
         * Splits a child node and returns the index of the left child. 
         */
        int splitNode(int node);
        
        /**
         * Classifies a data point
         */
        virtual int classify(DataPoint* x) const;
        
        /**
         * Classifies an entire data set. 
         */
        virtual void classify(DataStorage* storage, std::vector<int> & results) const
        {
            Classifier::classify(storage, results);
        }
        
        /**
         * Reads the tree from a stream
         */
        virtual void read(std::istream & stream);
        
        /**
         * Writes the tree to a stream
         */
        virtual void write(std::ostream & stream) const;
        
    private:
        /**
         * Adds a plain new node.
         */
        void addNode();
        
        /**
         * The split feature at each node. 
         */
        std::vector<int> splitFeatures;
        /**
         * The threshold at each node
         */
        std::vector<float> thresholds;
        /**
         * The left child node of each node. If the left child not is 0, then 
         * this is a leaf node. The right child node is left + 1. 
         */
        std::vector<int> leftChild;
        /**
         * The class label for each node
         */
        std::vector<int> classLabels;
        /**
         * The histograms for each node. We only store actual histograms at 
         * leaf nodes. 
         */
        std::vector< std::vector<int> > histograms;
    };
    
    /**
     * A random forests that classifies a data point using a simple majority 
     * voting schema. 
     */
    class RandomForest : public Classifier {
    public:
        /**
         * Classifies a data point
         */
        virtual int classify(DataPoint* x) const;
        
        /**
         * Classifies an entire data set. 
         */
        virtual void classify(DataStorage* storage, std::vector<int> & results) const
        {
            Classifier::classify(storage, results);
        }
        
        /**
         * Reads the tree from a stream
         */
        virtual void read(std::istream & stream);
        
        /**
         * Writes the tree to a stream
         */
        virtual void write(std::ostream & stream) const;
        
        /**
         * Adds a tree to the ensemble
         */
        void addTree(DecisionTree* tree)
        {
            trees.push_back(tree);
        }
        
        /**
         * Returns the number of trees
         */
        int getSize() const
        {
            return static_cast<int>(trees.size());
        }
        
        /**
         * Returns the i-th tree
         */
        DecisionTree* getTree(int i)
        {
            return trees[i];
        }
        
        /**
         * Returns the i-th tree
         */
        const DecisionTree* getTree(int i) const
        {
            return trees[i];
        }
        
        /**
         * Removes the i-th tree
         */
        void removeTree(int i)
        {
            // Delete the tree
            delete trees[i];
            // Remove it from the array
            trees.erase(trees.begin() + i);
        }
        
    private:
        /**
         * The individual decision trees. 
         */
        std::vector<DecisionTree*> trees;
        /**
         * The smoothing parameter for classification
         */
        float smoothing;
    };
}
#endif