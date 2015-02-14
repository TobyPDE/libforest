#include "libforest/data.h"
#include "util.h"
#include <fstream>
#include <iterator>
#include <boost/tokenizer.hpp>
#include <boost/tokenizer.hpp>
#include <cstdlib>
#include <random>
#include <iostream>


using namespace libf;

////////////////////////////////////////////////////////////////////////////////
/// DataPoint
////////////////////////////////////////////////////////////////////////////////

DataPoint::DataPoint(int newD, float x)
{
    D = newD;
    data = new float[D];
    // Initialize the point
    for (int i = 0; i < D; i++)
    {
        data[i] = x;
    }
}

DataPoint::DataPoint(const DataPoint & other)
{
    D = other.D;
    data = new float[D];
    for (int i = 0; i < D; i++)
    {
        data[i] = other.data[i];
    }
}

void DataPoint::resize(int newD)
{
    freeData();
    D = newD;
    data = new float[D];
}

DataPoint & DataPoint::operator=(const DataPoint & other)
{
    // Do nothing at self assignments. 
    if (this != &other)
    {
        // Resize this vector
        resize(other.D);
        // Copy the data
        for (int i = 0; i < D; i++)
        {
            data[i] = other.data[i];
        }
    }
    return *this;
}

void DataPoint::freeData()
{
    if (data != 0)
    {
        delete[] data;
        data = 0;
    }
}

////////////////////////////////////////////////////////////////////////////////
/// DataStorage
////////////////////////////////////////////////////////////////////////////////

DataStorage::DataStorage(const DataStorage & other)
{
    // Copy all arrays
    dataPoints = other.dataPoints;
    classLabels = other.classLabels;
    freeFlags = other.freeFlags;
    
    // Set all free flags to false as this is not the owner of the data points
    for (size_t i = 0; i < freeFlags.size(); i++)
    {
        freeFlags[i] = false;
    }
}

DataStorage::~DataStorage()
{
    for (size_t i = 0; i < dataPoints.size(); i++)
    {
        if (freeFlags[i])
        {
            delete dataPoints[i];
            dataPoints[i] = 0;
        }
    }
}

void DataStorage::computeIntClassLabels()
{
    // Reset the current map and class label list
    classLabelMap.erase(classLabelMap.begin(), classLabelMap.end());
    intClassLabels.erase(intClassLabels.begin(), intClassLabels.end());

    // Prepare the integer class labels
    intClassLabels.resize(dataPoints.size());

    // The current class label counter
    int classLabelCounter = 0;

    // Compute the labels
    for (size_t i = 0; i < classLabels.size(); i++)
    {
        // Did we already observe this label?
        if (classLabelMap.find(classLabels[i]) != classLabelMap.end())
        {
            // Yes, we already observed it
            // FIXME: intClassLabels[i] = classLabelMap[classLabels[i]];
            intClassLabels[i] = atoi(classLabels[i].c_str());
        }
        else
        {
            // Nope, we did not observe it
            // Add the class label to the label map
            classLabelMap[classLabels[i]] = classLabelCounter;
            // FIXME: intClassLabels[i] = classLabelCounter;
            intClassLabels[i] = atoi(classLabels[i].c_str());

            classLabelCounter++;
        }
    }
}

void DataStorage::computeIntClassLabels(const DataStorage* dataStorage)
{
    classLabelMap = dataStorage->classLabelMap;

    // Prepare the integer class labels
    intClassLabels.resize(dataPoints.size());

    // Compute the labels
    for (size_t i = 0; i < classLabels.size(); i++)
    {
        // Did we already observe this label?
        if (classLabelMap.find(classLabels[i]) != classLabelMap.end())
        {
            // Yes, we already observed it
            // FIXME: intClassLabels[i] = classLabelMap[classLabels[i]];
            intClassLabels[i] = atoi(classLabels[i].c_str());
        }
        else
        {
            // Nope, we did not observe it
            // This means the two data sets have different class labels
            throw Exception("Data storages are not compatible.");
        }
    }
}
    
int DataStorage::getDimensionality() const
{
    if (getSize() == 0)
    {
        // There are no data points
        return 0;
    }
    else
    {
        // Take the dimensionality of the first data point
        return getDataPoint(0)->getDimensionality();
    }
}

void DataStorage::permute(const std::vector<int>& permutation)
{
    // Copy the arrays because we cannot permute in-place
    std::vector< DataPoint* > dataPointsCopy(dataPoints);
    std::vector<std::string> classLabelsCopy(classLabels);
    std::vector<bool> freeFlagsCopy(freeFlags);
    
    // Permute the original arrays
    Util::permute(permutation, dataPointsCopy, dataPoints);
    Util::permute(permutation, classLabelsCopy, classLabels);
    Util::permute(permutation, freeFlagsCopy, freeFlags);
    
    computeIntClassLabels();
}

void DataStorage::randPermute()
{
    // Set up a random permutation
    std::vector<int> permutation(getSize());
    for (int i = 0; i < getSize(); i++)
    {
        permutation[i] = i;
    }
    
    // TODO: Add random permutation
    
    permute(permutation);
}

void DataStorage::split(float ratio, DataStorage* other)
{
    // Calculate the number of points that go to the other set
    const int numPoints = getSize() * ratio;
    
    for (int i = getSize() - 1; i > numPoints; i--)
    {
        other->addDataPoint(dataPoints[i], classLabels[i], freeFlags[i]);
        dataPoints.pop_back();
        classLabels.pop_back();
        freeFlags.pop_back();
        intClassLabels.pop_back();
    }
}

void DataStorage::bootstrap(int N, DataStorage* dataStorage) const
{
    std::vector<bool> sampled;
    bootstrap(N, dataStorage, sampled);
}

void DataStorage::bootstrap(int N, DataStorage* dataStorage, std::vector<bool> & sampled) const
{
    // Set up a probability distribution
    std::random_device rd;
    std::mt19937 g(rd());
    
    std::uniform_int_distribution<int> distribution(0, getSize() - 1);
    
    // Initialize the flag array
    sampled.resize(getSize());
    for (int n = 0; n < getSize(); n++)
    {
        sampled[n] = false;
    }
    
    // Add the points
    for (int i = 0; i < N; i++)
    {
        // Select some point
        const int point = distribution(g);
        dataStorage->addDataPoint(getDataPoint(point), getClassLabel(point), false);
        sampled[point] = true;
    }
}

////////////////////////////////////////////////////////////////////////////////
/// CSVDataProvider
////////////////////////////////////////////////////////////////////////////////

void CSVDataProvider::read(const std::string & source, DataStorage* dataStorage)
{
    // Open the file
    std::ifstream in(source);
    if (!in.is_open())
    {
        throw Exception("Could not open data set file.");
    }

    // Tokenize the stream
    typedef boost::tokenizer< boost::escaped_list_separator<char> > Tokenizer;

    std::vector< std::string > row;
    std::string line;

    while (std::getline(in,line))
    {
        // Tokenize the line
        Tokenizer tok(line);
        row.assign(tok.begin(),tok.end());

        // Do not consider blank line
        if (row.size() == 0) continue;
        
        // Load the data point
        DataPoint* dataPoint = new DataPoint(static_cast<int>(row.size() - 1));
        std::string label;
        const int isize = static_cast<int>(row.size());
        for (int  i = 0; i < isize; i++)
        {
            if (i == classColumnIndex)
            {
                label = row[i];
            }
            else if (i < classColumnIndex)
            {
                dataPoint->at(i) = atof(row[i].c_str());
            }
            else
            {
                dataPoint->at(i - 1) = atof(row[i].c_str());
            }
        }
        
        dataStorage->addDataPoint(dataPoint, label);
    }
}
