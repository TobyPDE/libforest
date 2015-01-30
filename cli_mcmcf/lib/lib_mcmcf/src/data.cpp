#include "lib_mcmcf/data.h"
#include "util.h"
#include <fstream>
#include <iterator>
#include <boost/tokenizer.hpp>
#include <boost/tokenizer.hpp>
#include <cstdlib>


using namespace mcmcf;

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

DataStorage::~DataStorage()
{
    for (size_t i = 0; i < dataPoints.size(); i++)
    {
        delete dataPoints[i];
        dataPoints[i] = 0;
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
            intClassLabels[i] = classLabelMap[classLabels[i]];
        }
        else
        {
            // Nope, we did not observe it
            // Add the class label to the label map
            classLabelMap[classLabels[i]] = classLabelCounter;
            intClassLabels[i] = classLabelCounter;

            classLabelCounter++;
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

////////////////////////////////////////////////////////////////////////////////
/// DataPoint
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
