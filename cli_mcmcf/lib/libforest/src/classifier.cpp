#include "libforest/classifiers.h"
#include "libforest/data.h"
#include <ios>
#include <iostream>
#include <string>
#include <cmath>

using namespace libf;


////////////////////////////////////////////////////////////////////////////////
/// Helper functions
////////////////////////////////////////////////////////////////////////////////

/**
 * Writes a binary value to a stream
 */
template<typename T>
void writeBinary(std::ostream& stream, const T& value)
{
    stream.write(reinterpret_cast<const char*>(&value), sizeof(T));
}

/**
 * Reads a binary value from a stream
 */
template<typename T>
void readBinary(std::istream& stream, T& value)
{
    stream.read(reinterpret_cast<char*>(&value), sizeof(T));
}

/**
 * Writes a binary string to a stream
 */
template <>
void writeBinary(std::ostream & stream, const std::string & value)
{
    
}

/**
 * Reads a binary string from a stream
 */
template <>
void readBinary(std::istream & stream, std::string & value)
{
    
}

/**
 * Writes a vector to a stream
 */
template <class T>
static void writeVector(std::ostream & stream, const std::vector<T> & v)
{
    writeBinary(stream, static_cast<int>(v.size()));
    for (size_t i = 0; i < v.size(); i++)
    {
        writeBinary(stream, v[i]);
    }
}

/**
 * Reads a vector of N elements from a stream. 
 */
template <class T>
static void readVector(std::istream & stream, std::vector<T> & v)
{
    int N;
    readBinary(stream, N);
    v.resize(N);
    for (int i = 0; i < N; i++)
    {
        //readBinary(stream, v[i]);
        stream >> v[i];
    }
}

////////////////////////////////////////////////////////////////////////////////
/// Classifier
////////////////////////////////////////////////////////////////////////////////

void Classifier::classify(DataStorage* storage, std::vector<int> & results) const
{
    // Clean the result set
    results.erase(results.begin(), results.end());
    results.resize(storage->getSize());
    
    // Classify each individual data point
    for (int i = 0; i < storage->getSize(); i++)
    {
        results[i] = classify(storage->getDataPoint(i));
    }
}

////////////////////////////////////////////////////////////////////////////////
/// DecisionTree
////////////////////////////////////////////////////////////////////////////////

DecisionTree::DecisionTree()
{
    splitFeatures.reserve(5000);
    thresholds.reserve(5000);
    classLabels.reserve(5000);
    leftChild.reserve(5000);
    histograms.reserve(5000);
    // Add at least the root node with index 0
    addNode();
}

void DecisionTree::addNode()
{
    splitFeatures.push_back(0);
    thresholds.push_back(0);
    classLabels.push_back(0);
    leftChild.push_back(0);
    histograms.push_back(std::vector<int>());
}

int DecisionTree::splitNode(int node)
{
    // Make sure this is a valid node ID
    assert(0 <= node && node < static_cast<int>(splitFeatures.size()));
    // Make sure this is a former child node
    assert(leftChild[node] == 0);
    
    // Determine the index of the new left child
    const int leftNode = static_cast<int>(splitFeatures.size());
    
    // Add the child nodes
    addNode();
    addNode();
    
    // Set the child relation
    leftChild[node] = leftNode;
    
    return leftNode;
}

int DecisionTree::findLeafNode(DataPoint* x) const
{
    // Select the root node as current node
    int node = 0;
    // Follow the tree until we hit a leaf node
    while (leftChild[node] != 0)
    {
        // Check the threshold
        if (x->at(splitFeatures[node]) < thresholds[node])
        {
            // Go to the left
            node = leftChild[node];
        }
        else
        {
            // Go to the right
            node = leftChild[node] + 1;
        }
    }
    return node;
}

int DecisionTree::classify(DataPoint* x) const
{
    return classLabels[findLeafNode(x)];
}

void DecisionTree::read(std::istream& stream)
{
    // Write the attributes to the file
    readVector(stream, splitFeatures);
    readVector(stream, thresholds);
    readVector(stream, leftChild);
    readVector(stream, classLabels);
    
}

void DecisionTree::write(std::ostream& stream) const
{
    // Write the attributes to the file
    writeVector(stream, splitFeatures);
    writeVector(stream, thresholds);
    writeVector(stream, leftChild);
    writeVector(stream, classLabels);
}

////////////////////////////////////////////////////////////////////////////////
/// DecisionTree
////////////////////////////////////////////////////////////////////////////////

int RandomForest::classify(DataPoint* x) const
{
    // Initialize the voting system
    std::vector<float> votes;
    
    // Let the crowd decide
    for (size_t i = 0; i < trees.size(); i++)
    {
        const int node = trees[i]->findLeafNode(x);
        const std::vector<int> & hist = trees[i]->getHistogram(node);
        
        const int C = static_cast<int>(hist.size());
        
        // Initialize the vote histogram
        if (votes.size() == 0)
        {
            votes.resize(C);
            for (int c = 0; c < C; c++)
            {
                votes[c] = 0;
            }
        }
        
        // Accumulate the votes
        int sum = 0;
        for (int c = 0; c < C; c++)
        {
            sum += hist[c];
        }
        for (int  c = 0; c < C; c++)
        {
            votes[c] += std::log((hist[c] + smoothing)/(sum + C*smoothing));
        }
    }
    
    // Predict the final label
    float maxVotes = votes[0];
    int finalLabel = 0;
    for (size_t c = 0; c < votes.size(); c++)
    {
        if (maxVotes < votes[c])
        {
            finalLabel = static_cast<int>(c);
            maxVotes = votes[c];
        }
    }
    
    return finalLabel;
}

void RandomForest::write(std::ostream& stream) const
{
    // Write the number of trees in this ensemble
    writeBinary(stream, getSize());
    
    // Write the individual trees
    for (int i = 0; i < getSize(); i++)
    {
        getTree(i)->write(stream);
    }
}

void RandomForest::read(std::istream& stream)
{
    // Read the number of trees in this ensemble
    int size;
    readBinary(stream, size);
    
    // Read the trees
    for (int i = 0; i < size; i++)
    {
        std::cout << i << "\n";
        DecisionTree* tree = new DecisionTree();
        tree->read(stream);
        addTree(tree);
    }
}