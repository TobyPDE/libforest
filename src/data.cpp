#include "libforest/data.h"
#include "libforest/io.h"
#include "libforest/util.h"
#include "libforest/tools.h"
#include <fstream>
#include <iterator>
#include <boost/tokenizer.hpp>
#include <boost/spirit/include/qi.hpp>
#include <boost/fusion/adapted/std_pair.hpp>
#include <cstdlib>
#include <random>
#include <iostream>
#include <iomanip>

using namespace libf;

static std::random_device rd;

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
/// AbstractDataStorage
////////////////////////////////////////////////////////////////////////////////

bool AbstractDataStorage::containsUnlabeledPoints() const
{
    // Go through the storage and check for unlabeled points
    for (int n = 0; n < getSize(); n++)
    {
        if (getClassLabel(n) == NO_LABEL)
        {
            return true;
        }
    }
    return false;
}

        
int AbstractDataStorage::getDimensionality() const
{
    if (getSize() == 0)
    {
        // There are no data points
        return 0;
    }
    else
    {
        // Take the dimensionality of the first data point
        // By the constraints, this is the dimensionality of all data 
        // points
        return getDataPoint(0).rows();
    }
}

AbstractDataStorage::ptr AbstractDataStorage::excerpt(int begin, int end) const
{
    BOOST_ASSERT_MSG(begin >= 0 && begin <= end, "Invalid indices.");
    BOOST_ASSERT_MSG(end < getSize(), "Invalid indices.");
    
    ReferenceDataStorage::ptr storage = std::make_shared<ReferenceDataStorage>(shared_from_this());
    
    // Add the data points
    for (int n = begin; n <= end; n++)
    {
        storage->addDataPoint(n);
    }
    
    return storage;
}

AbstractDataStorage::ptr AbstractDataStorage::bootstrap(int N, std::vector<bool> & sampled) const
{
    BOOST_ASSERT_MSG(N >= 0, "The number of bootstrap examples must be non-negative.");
    
    ReferenceDataStorage::ptr storage = std::make_shared<ReferenceDataStorage>(shared_from_this());
    
    // Set up a probability distribution
    std::mt19937 g(rd());
    
    std::uniform_int_distribution<int> distribution(0, getSize() - 1);
    
    // Initialize the flag array
    sampled.resize(getSize(), false);
    
    // Add the points
    for (int i = 0; i < N; i++)
    {
        // Select some point
        const int n = distribution(g);
        sampled[n] = true;
        storage->addDataPoint(n);
    }
    
    return storage;
}

void AbstractDataStorage::randPermute()
{
    // Set up a random permutation
    std::vector<int> permutation;
    Util::generateRandomPermutation(getSize(), permutation);    
    
    // Permute the storage.
    permute(permutation);
}

void AbstractDataStorage::dumpInformation(std::ostream & stream)
{
    stream << std::setw(30) << "Size" << ": " << getSize() << "\n";
    stream << std::setw(30) << "Dimensionality" << ": " << getDimensionality() << "\n";
    // Dump the class statistics of the storage
    stream << std::setw(30) << "Class statistics" << "\n";
    
    ClassStatisticsTool csTool;
    csTool.measureAndPrint(shared_from_this());
}

////////////////////////////////////////////////////////////////////////////////
/// DataStorage
////////////////////////////////////////////////////////////////////////////////

void DataStorage::permute(const std::vector<int> & permutation)
{
    // We need to copy the class labels because the permutation cannot 
    // be done in place
    std::vector<int> classLabelsCopy(classLabels);
    Util::permute(permutation, classLabelsCopy, classLabels);
    std::vector<DataPoint> dataPointsCopy(dataPoints);
    Util::permute(permutation, dataPointsCopy, dataPoints);
}

////////////////////////////////////////////////////////////////////////////////
/// ReferenceDataStorage
////////////////////////////////////////////////////////////////////////////////

void ReferenceDataStorage::permute(const std::vector<int> & permutation)
{
    // We need to copy the class labels because the permutation cannot 
    // be done in place
    std::vector<int> dataPointIndicesCopy(dataPointIndices);
    Util::permute(permutation, dataPointIndicesCopy, dataPointIndices);
}

////////////////////////////////////////////////////////////////////////////////
/// DataReader
////////////////////////////////////////////////////////////////////////////////

void AbstractDataReader::read(const std::string& filename, DataStorage::ptr dataStorage) throw(IOException)
{
    // Try to open the file
    std::ifstream stream(filename, std::ios::binary);
    
    // Check if we could open the file
    if (!stream.is_open())
    {
        throw IOException("Could not open file.");
    }
    
    // Read the actual data
    read(stream, dataStorage);
    
    // Close the stream
    stream.close();
}

////////////////////////////////////////////////////////////////////////////////
/// CSVDataReader
////////////////////////////////////////////////////////////////////////////////

void CSVDataReader::read(std::istream & stream, DataStorage::ptr dataStorage, ClassLabelMap & classLabelMap)
{
    // Tokenize the stream
    typedef boost::tokenizer< boost::escaped_list_separator<char> > Tokenizer;

    // @see http://www.boost.org/doc/libs/1_36_0/libs/tokenizer/escaped_list_separator.htm
    const std::string escape("\\");
    const std::string separator(columnSeparator);
    const std::string quote("\"");

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
        
        int isize = static_cast<int>(row.size());
        if (row[row.size() - 1].empty()) 
        {
            // We have an empty trailing column ...
            isize--;
        }
        
        // If we read class labels, the number of columns is one more than the
        // number of dimensions
        int dimensionality = isize;
        if (readClassLabels)
        {
            dimensionality--;
        }
        
        // Load the data point
        DataPoint dataPoint(dimensionality);
        int  label = 0;
        
        for (int  i = 0; i < isize; i++)
        {
            // Do we read class labels?
            if (readClassLabels)
            {
                // We do
                // Is this the label column?
                if (i == classLabelColumnIndex)
                {
                    // It is
                    // Get a preliminary class label
                    label = classLabelMap.addClassLabel(row[i]);
                }
                else if (i < classLabelColumnIndex)
                {
                    dataPoint(i) = atof(row[i].c_str());
                }
                else
                {
                    dataPoint(i - 1) = atof(row[i].c_str());
                }
            }
            else
            {
                dataPoint(i) = atof(row[i].c_str());
            }
        }
        
        if (readClassLabels)
        {
            dataStorage->addDataPoint(dataPoint, label);
        }
        else
        {
            dataStorage->addDataPoint(dataPoint);
        }
    }
    
    // Compute the integer class label
    std::vector<int> intLabelMap;
    classLabelMap.computeIntClassLabels(intLabelMap);
    
    // Update the class labels
    for (int i = 0; i < dataStorage->getSize(); i++)
    {
        dataStorage->getClassLabel(i) = intLabelMap[dataStorage->getClassLabel(i)];
    }
}


////////////////////////////////////////////////////////////////////////////////
/// LIBSVMDataReader
////////////////////////////////////////////////////////////////////////////////

void LIBSVMDataReader::read(std::istream& stream, DataStorage::ptr dataStorage)
{
    std::string line;
    
    // First: Parse the entire thing
    std::vector< std::pair<int, std::vector< std::pair<int, float> > > > lines;
    
    // Iterate over the data points and read them
    while (std::getline(stream, line))
    {
        // Skip this line if it's empty or starts with a #
        if (line.size() == 0 || line[0] == '#')
        {
            continue;
        }
        lines.push_back(std::pair<int, std::vector< std::pair<int, float> > >());
        
        parseLine(line, lines.back());
    }
    
    // Determine the dimensionality
    int dimensionality = 0;
    for (size_t i = 0; i < lines.size(); i++)
    {
        for (size_t d = 0; d < lines[i].second.size(); d++)
        {
            dimensionality = std::max(dimensionality, static_cast<int>(lines[i].second[d].first));
        }
    }
    
    if (dimensionality == 0 && lines.size() > 0)
    {
        throw IOException("Invalid LIBSVM data set. No dimensions.");
    }
    
    // Create the data points
    // We do it like this to make the transformation in-place
    while (lines.size() > 0)
    {
        DataPoint x = DataPoint::Zero(dimensionality);
        
        for (size_t d = 0; d < lines.back().second.size(); d++)
        {
            const int id = static_cast<int>(lines.back().second[d].first);
            x(id) = lines.back().second[d].second;
        }
        
        dataStorage->addDataPoint(x, lines.back().first);
        
        lines.pop_back();
    }
}

void LIBSVMDataReader::parseLine(const std::string & line, std::pair<int, std::vector< std::pair<int, float> > > & result) const
{
    // Set up the iterators
    std::string::const_iterator first = line.begin();
    std::string::const_iterator last = line.end();
    
    using boost::spirit::qi::float_;
    using boost::spirit::qi::int_;
    using boost::spirit::qi::phrase_parse;
    using boost::spirit::ascii::space;
std::pair<int, std::vector< std::pair<int, float> > > result2;
    // Parse the line using boost spirit
    bool r = phrase_parse(
        first, 
        last, 
        int_ >> *(int_ >> ':' >> float_),
        space,
        result2
    );
    result = result2;
    
    // There was a mismatch. This is not a valid LIBSVM file
    if (first != last || !r)
    {
        throw IOException("Invalid LIBSVM line.");
    }
}

////////////////////////////////////////////////////////////////////////////////
/// LibforestDataProvider
////////////////////////////////////////////////////////////////////////////////

void LibforestDataReader::read(std::istream& stream, DataStorage::ptr dataStorage)
{
    // Read the number of data points
    int N;
    readBinary(stream, N);
    
    // Read the data set
    for (int n = 0; n < N; n++)
    {
        if (readClassLabels)
        {
            // Read the class label
            int label;
            readBinary(stream, label);
            // Set up the data point
            DataPoint v;
            readDataPoint(stream, v);
            dataStorage->addDataPoint(v, label);
        }
        else
        {
            // Set up the data point
            DataPoint v;
            readDataPoint(stream, v);
            dataStorage->addDataPoint(v);
        }
    }
}

void LibforestDataReader::readDataPoint(std::istream& stream, DataPoint& v)
{
    // Read the dimensionality
    int D;
    readBinary(stream, D);
    
    // Resize the data point
    v.resize(D);
    
    // Load the content
    for (int d = 0; d < D; d++)
    {
        readBinary(stream, v(d));
    }
}

////////////////////////////////////////////////////////////////////////////////
/// AbstractDataWriter
////////////////////////////////////////////////////////////////////////////////

void AbstractDataWriter::write(const std::string & filename, DataStorage::ptr dataStorage) throw(IOException)
{
    // Open the file
    std::ofstream stream(filename, std::ios::binary);
    if (!stream.is_open())
    {
        throw IOException("Could not open data file.");
    }
    write(stream, dataStorage);
    stream.close();
}

////////////////////////////////////////////////////////////////////////////////
/// CSVDataWriter
////////////////////////////////////////////////////////////////////////////////

void CSVDataWriter::write(std::ostream& stream, DataStorage::ptr dataStorage)
{
    for (size_t n = 0; n < dataStorage->getSize(); n++)
    {
        const DataPoint & v = dataStorage->getDataPoint(n);
        
        // Do we write class labels?
        if (writeClassLabels)
        {
            // Yes we do, thus we have to take care of them
            for (int d = 0; d < v.rows()+1; d++)
            {
                if (d < classLabelColumnIndex)
                {
                    stream << v(d);
                }
                else if (d == classLabelColumnIndex)
                {
                    stream << dataStorage->getClassLabel(d);
                }
                else
                {
                    stream << v(d-1);
                }
                
                // Write the column separator?
                if (d < v.rows())
                {
                    stream << columnSeparator;
                }
            }
        }
        else
        {
            // Do not write class labels
            for (int d = 0; d < v.rows(); d++)
            {
                stream << v(d);
                
                // Write the column separator?
                if (d < v.rows())
                {
                    stream << columnSeparator;
                }
            }
        }
        stream << std::endl;
    }
}

////////////////////////////////////////////////////////////////////////////////
/// LibforestDataWriter
////////////////////////////////////////////////////////////////////////////////

void LibforestDataWriter::write(std::ostream& stream, DataStorage::ptr dataStorage)
{
    // Write the number of data points
    writeBinary(stream, dataStorage->getSize());
    // Write the content
    for (int n = 0; n < dataStorage->getSize(); n++)
    {
        if (writeClassLabels)
        {
            writeBinary(stream, dataStorage->getClassLabel(n));
        }
        writeDataPoint(stream, dataStorage->getDataPoint(n));
    }
}

void LibforestDataWriter::writeDataPoint(std::ostream& stream, DataPoint& v)
{
    writeBinary(stream, v.rows());
    for (int d = 0; d < v.rows(); d++)
    {
        writeBinary(stream, v(d));
    }
}

////////////////////////////////////////////////////////////////////////////////
/// LIBSVMDataWriter
////////////////////////////////////////////////////////////////////////////////

void LIBSVMDataWriter::write(std::ostream& stream, DataStorage::ptr dataStorage)
{
    stream << "# " << dataStorage->getSize() << " data points" << std::endl;
    
    for (int n = 0; n < dataStorage->getSize(); n++)
    {
        std::cout << dataStorage->getClassLabel(n) << " ";
        writeDataPoint(stream, dataStorage->getDataPoint(n));
        std::cout << std::endl;
    }
}

void LIBSVMDataWriter::writeDataPoint(std::ostream& stream, DataPoint& v)
{
    for (int d = 0; d < v.rows(); d++)
    {
        if (std::abs(v(d)) > 1e-15)
        {
            std::cout << (d+1) << ':' << v(d) << ' ';
        }
    }
}