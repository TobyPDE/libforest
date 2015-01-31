#include "lib_mcmcf/classifiers.h"
#include "lib_mcmcf/data.h"

using namespace mcmcf;


////////////////////////////////////////////////////////////////////////////////
/// Helper functions
////////////////////////////////////////////////////////////////////////////////

/**
 * Writes a vector to a stream
 */
template <class T>
static void writeVector(std::ostream & stream, const std::vector<T> & v)
{
    for (size_t i = 0; i < v.size(); i++)
    {
        stream << v[i] << ' ';
    }
}

/**
 * Reads a vector of N elements from a stream. 
 */
template <class T>
static void readVector(std::istream & stream, std::vector<T> & v, int N)
{
    v.resize(N);

    for (int i = 0; i < N; i++)
    {
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
    // Add at least the root node with index 0
    addNode();
}

void DecisionTree::addNode()
{
    splitFeatures.push_back(0);
    thresholds.push_back(0);
    classLabels.push_back(0);
    leftChild.push_back(0);
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

int DecisionTree::classify(DataPoint* x) const
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
    
    return classLabels[node];
}

void DecisionTree::read(std::istream& stream)
{
    // Read the number of nodes from the stream
    int N;
    stream >> N;
    
    // Write the attributes to the file
    readVector(stream, splitFeatures, N);
    readVector(stream, thresholds, N);
    readVector(stream, leftChild, N);
    readVector(stream, classLabels, N);
    
}

void DecisionTree::write(std::ostream& stream) const
{
    // Write the total number of nodes
    stream << splitFeatures.size() << ' ';
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
    std::vector<int> votes;
    int finalLabel = 0;
    
    // Let the crowd decide
    for (size_t i = 0; i < trees.size(); i++)
    {
        const int label = trees[i]->classify(x);
        // Did we already encounter this label?
        // Nope, add the intermediate bins
        const int isize = static_cast<int>(votes.size());
        for (int k = label; k >= isize; k--)
        {
            votes.push_back(0);
        }
        
        votes[label]++;
        
        if (votes[label] > votes[finalLabel])
        {
            finalLabel = label;
        }
    }
    
    return finalLabel;
}

void RandomForest::write(std::ostream& stream) const
{
    // Write the number of trees in this ensemble
    stream << getSize() << ' ';
    // Write the individual trees
    for (int i = 0; i < getSize(); i++)
    {
        getTree(i)->write(stream);
        stream << ' ';
    }
}

void RandomForest::read(std::istream& stream)
{
    // Read the number of trees in this ensemble
    int size;
    stream >> size;
    // Read the trees
    for (int i = 0; i < size; i++)
    {
        DecisionTree* tree = new DecisionTree();
        tree->read(stream);
        addTree(tree);
    }
}