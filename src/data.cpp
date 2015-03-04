#include "libforest/data.h"
#include "libforest/io.h"
#include "libforest/util.h"
#include <fstream>
#include <iterator>
#include <boost/tokenizer.hpp>
#include <cstdlib>
#include <random>
#include <iostream>
#include <iomanip>

using namespace libf;

std::random_device rd;

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

void DataPoint::read(std::istream& stream)
{
    // Read the dimensionality
    readBinary(stream, D);
    
    // Create the vector
    freeData();
    data = new float[D];
    
    for (int d = 0; d < D; d++)
    {
        readBinary(stream, data[d]);
    }
}

void DataPoint::write(std::ostream& stream) const
{
    // Write the dimensionality
    writeBinary(stream, D);
    
    // Write the vector
    for (int d = 0; d < D; d++)
    {
        writeBinary(stream, data[d]);
    }
}

////////////////////////////////////////////////////////////////////////////////
/// ClassLabelMap
////////////////////////////////////////////////////////////////////////////////

void ClassLabelMap::computeIntClassLabels(std::vector<int> & intLabelMap)
{
    // First: Get all string class labels
    std::vector<std::string> stringClassLabels(inverseLabelMap);
    
    // Because all class labels are distinct, the lexicographical ordering
    // is unique. This has the effect that the integer class labels are
    // always the same regardless of the order in which the data points
    // were loaded.
    std::sort(stringClassLabels.begin(), stringClassLabels.end());
    
    // Update the maps
    for (size_t i = 0; i < stringClassLabels.size(); i++)
    {
        // Update the map
        labelMap[stringClassLabels[i]] = static_cast<int>(i);
    }
    intLabelMap.resize(inverseLabelMap.size());
    
    // Compute the intLabelMap
    for (size_t i = 0; i < intLabelMap.size(); i++)
    {
        intLabelMap[i] = labelMap[inverseLabelMap[i]];
    }
    inverseLabelMap = stringClassLabels;
}

void ClassLabelMap::write(std::ostream& stream) const
{
    // Write the number of correspondences
    writeBinary(stream, static_cast<int>(inverseLabelMap.size()));
    
    // Write the pairs of strings and integers
    for (size_t i = 0; i < inverseLabelMap.size(); i++)
    {
        writeBinary(stream, inverseLabelMap[i]);
    }
}

void ClassLabelMap::read(std::istream& stream)
{
    // Get the number of labels
    int numLabels;
    readBinary(stream, numLabels);
    
    // Load the label maps
    for (int i = 0; i < numLabels; i++)
    {
        std::string label;
        readBinary(stream, label);
        inverseLabelMap[i] = label;
        labelMap[label] = i;
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
    classLabelMap = other.classLabelMap;
    classcount = other.classcount;
    
    // Set all free flags to false as this is not the owner of the data points
    for (size_t i = 0; i < freeFlags.size(); i++)
    {
        freeFlags[i] = false;
    }
    
}

DataStorage DataStorage::excerpt(int begin, int end)
{
    assert(begin >= 0 && begin <= end);
    assert(end >= begin && end < dataPoints.size());
    
    DataStorage excerpt;
    
    excerpt.dataPoints = std::vector<DataPoint*>(end - begin + 1);
    excerpt.freeFlags = std::vector<bool>(end - begin + 1);
    excerpt.classLabels = std::vector<int>(end - begin + 1);
    
    int m = 0;
    for (int n = begin; n <= end; n++, m++)
    {
        // The points do not belong to the excerpt.
        excerpt.freeFlags[m] = false;
        excerpt.dataPoints[m] = dataPoints[n];
        excerpt.classLabels[m] = classLabels[n];
    }
    
    excerpt.classLabelMap = classLabelMap;
    excerpt.classcount = classcount;
}

DataStorage & DataStorage::operator=(const DataStorage & other)
{
    if (this != &other)
    {
        free();
        dataPoints = other.dataPoints;
        classLabels = other.classLabels;
        freeFlags = other.freeFlags;
        classLabelMap = other.classLabelMap;
        classcount = other.classcount;

        // Set all free flags to false as this is not the owner of the data points
        for (size_t i = 0; i < freeFlags.size(); i++)
        {
            freeFlags[i] = false;
        }
    }
    return *this;
}

void DataStorage::free()
{
    for (size_t i = 0; i < dataPoints.size(); i++)
    {
        if (freeFlags[i])
        {
            delete dataPoints[i];
            dataPoints[i] = 0;
        }
    }
    
    dataPoints.empty();
    classLabels.empty();
    freeFlags.empty();
}

DataStorage::~DataStorage()
{
    free();
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
    std::vector<int> classLabelsCopy(classLabels);
    std::vector<bool> freeFlagsCopy(freeFlags);
    
    // Permute the original arrays
    Util::permute(permutation, dataPointsCopy, dataPoints);
    Util::permute(permutation, classLabelsCopy, classLabels);
    Util::permute(permutation, freeFlagsCopy, freeFlags);
}

void DataStorage::randPermute()
{
    // Set up a random permutation
    std::vector<int> permutation(getSize());
    for (int i = 0; i < getSize(); i++)
    {
        permutation[i] = i;
    }
    
    std::shuffle(permutation.begin(), permutation.end(), std::default_random_engine(rd()));
    
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
    }
    other->classLabelMap = classLabelMap;
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
    dataStorage->classLabelMap = classLabelMap;
}

void DataStorage::dumpInformation(std::ostream & stream)
{
    std::vector<int> intLabelMap;
    getClassLabelMap().computeIntClassLabels(intLabelMap);
    
    stream << std::setw(30) << "Size" << ": " << getSize() << "\n";
    stream << std::setw(30) << "Dimensionality" << ": " << getDimensionality() << "\n";
    stream << std::setw(30) << "Classes" << ": " << getClasscount() << "\n";
}

////////////////////////////////////////////////////////////////////////////////
/// DataProvider
////////////////////////////////////////////////////////////////////////////////
void DataProvider::read(const std::string& filename, DataStorage* dataStorage)
{
    // Open the file
    std::ifstream stream(filename, std::ios::binary);
    if (!stream.is_open())
    {
        throw Exception("Could not open file.");
    }
    read(stream, dataStorage);
    stream.close();
}


////////////////////////////////////////////////////////////////////////////////
/// DataWriter
////////////////////////////////////////////////////////////////////////////////
void DataWriter::write(const std::string& filename, DataStorage* dataStorage)
{
    // Open the file
    std::ofstream stream(filename, std::ios::binary);
    if (!stream.is_open())
    {
        throw Exception("Could not open file.");
    }
    write(stream, dataStorage);
    stream.close();
}


////////////////////////////////////////////////////////////////////////////////
/// CSVDataProvider
////////////////////////////////////////////////////////////////////////////////

void CSVDataProvider::read(std::istream & stream, DataStorage* dataStorage)
{
    // Tokenize the stream
    typedef boost::tokenizer< boost::escaped_list_separator<char> > Tokenizer;

    // @see http://www.boost.org/doc/libs/1_36_0/libs/tokenizer/escaped_list_separator.htm
    std::string escape("\\");
    std::string separator(columnSeparator);
    std::string quote("\"");

    boost::escaped_list_separator<char> els(escape, separator, quote);
    
    std::vector< std::string > row;
    std::string line;

    while (std::getline(stream,line))
    {
        // Tokenize the line
        Tokenizer tok(line, els);
        row.assign(tok.begin(), tok.end());

        // Do not consider blank line
        if (row.size() == 0) continue;
        
        // TODO: more elegant solution
        int isize = static_cast<int>(row.size());
        if (row[row.size() - 1].empty()) 
        {
            // We have an empty trailing column ...
            isize--;
        }
        
        // Load the data point
        DataPoint* dataPoint = new DataPoint(isize - 1);
        int  label = 0;
        for (int  i = 0; i < isize; i++)
        {
            if (i == classColumnIndex)
            {
                label = dataStorage->getClassLabelMap().addClassLabel(row[i]);
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
        
        dataStorage->addDataPoint(dataPoint, label, true);
    }
    
    // Compute the integer class label
    std::vector<int> intLabelMap;
    dataStorage->getClassLabelMap().computeIntClassLabels(intLabelMap);
    
    // Update the class labels
    for (int i = 0; i < dataStorage->getSize(); i++)
    {
        dataStorage->getClassLabel(i) = intLabelMap[dataStorage->getClassLabel(i)];
    }
}

////////////////////////////////////////////////////////////////////////////////
/// LibforestDataProvider
////////////////////////////////////////////////////////////////////////////////

void LibforestDataProvider::read(std::istream& stream, DataStorage* dataStorage)
{
    // Read the number of data points
    int N;
    readBinary(stream, N);
    
    // Read the data set
    for (int n = 0; n < N; n++)
    {
        // Read the class label
        int label;
        readBinary(stream, label);
        // Set up the data point
        DataPoint* v = new DataPoint();
        v->read(stream);
        dataStorage->addDataPoint(v, label, true);
    }
}

////////////////////////////////////////////////////////////////////////////////
/// LibforestDataWriter
////////////////////////////////////////////////////////////////////////////////

void LibforestDataWriter::write(std::ostream& stream, DataStorage* dataStorage)
{
    // Write the number of data points
    writeBinary(stream, dataStorage->getSize());
    // Write the content
    for (int n = 0; n < dataStorage->getSize(); n++)
    {
        writeBinary(stream, dataStorage->getClassLabel(n));
        dataStorage->getDataPoint(n)->write(stream);
    }
}