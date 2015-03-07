#ifndef LIBF_DATA_H
#define LIBF_DATA_H

/**
 * This file contains everything that is related to reading data sets. We use 
 * an internal data representation that makes learning decision trees fast. You
 * can load any type of data set by implementing a so called data provider. A
 * data provider reads the data source and outputs a list of pairs (x_i, y_i)
 * where x_i is a vector and y_i is the class label. You can use either integer
 * class labels or string class labels. 
 * 
 * The library comes pre equiped with data providers for LIBSVM data files as 
 * well as CSV data files. 
 */

#include <vector>
#include <utility>
#include <map>
#include <cassert>
#include <string>
#include <iostream>

namespace libf {
    /**
     * This class represents in individual data point. 
     */
    class DataPoint {
    public:
        /**
         * Creates a new data point of size 0
         */
        DataPoint() : D(0), data(0) {}
        
        /**
         * Creates a new data point of size D that is not initialized
         */
        DataPoint(int D) : D(D), data(new float[D]) {}
        
        /**
         * Creates a new data point of size D that is initialized with x
         */
        DataPoint(int D, float x);
        
        /**
         * Copy constructor
         */
        DataPoint(const DataPoint & other);
        
        /**
         * Destructor
         */
        virtual ~DataPoint()
        {
            freeData();
        }
        
        /**
         * Resizes the data point
         */
        void resize(int newD);
        
        /**
         * Assignment operator
         */
        DataPoint & operator=(const DataPoint & other);
        
        /**
         * Returns the i-th entry of the vector
         */
        const float & at(int i) const
        {
            assert(0 <= i && i < D);
            return data[i];
        }
        
        /**
         * Returns the i-th entry of the vector
         */
        float & at(int i)
        {
            assert(0 <= i && i < D);
            return data[i];
        }
        
        /**
         * Returns the dimensionality
         */
        int getDimensionality() const
        {
            return D;
        }
        
        /**
         * Writes the class label map to a stream. 
         */
        void write(std::ostream & stream) const;
        
        /**
         * Reads the class label map from a file
         */
        void read(std::istream & stream);
        
    private:
        /**
         * Frees the data array
         */
        void freeData();
        
        /**
         * The dimensionality of the feature vector
         */
        int D;
        /**
         * The actual data
         */
        float* data;
    };
    
    /**
     * This is a class label map. The internal data storage works using integer
     * class labels. When loading a data set from a file, the class labels are
     * transformed from strings to integers. This map stores the relation 
     * between them.
     * 
     * Please note: All labels have to be in the map before the integer class
     * labels can be computed. 
     */
    class ClassLabelMap {
    public:
        /**
         * Returns the integer class label for a given string class label
         */
        int getClassLabel(const std::string & label) const
        {
            return labelMap.find(label)->second;
        }
        
        /**
         * Returns the string class label for a given integer class label.
         */
        const std::string & getClassLabel(int label) const
        {
            return inverseLabelMap[label];
        }
        
        /**
         * Adds a string class label and returns a primarily integer class label. 
         */
        int addClassLabel(const std::string & label)
        {
            if (labelMap.find(label) == labelMap.end())
            {
                inverseLabelMap.push_back(label);
                labelMap[label] = static_cast<int>(inverseLabelMap.size() - 1);
            }
            return labelMap[label];
        }
        
        /**
         * Returns the number of classes.
         */
        int getClassCount() const
        {
            return static_cast<int>(inverseLabelMap.size());
        }
        
        /**
         * Computes the integer class labels and return a mapping from the 
         * primarily class labels to the final class labels. 
         */
        void computeIntClassLabels(std::vector<int> & intLabelMap);
        
        /**
         * Writes the class label map to a stream. 
         */
        void write(std::ostream & stream) const;
        
        /**
         * Reads the class label map from a file
         */
        void read(std::istream & stream);
        
    private:
        /**
         * The actual map
         */
        std::map<std::string, int> labelMap;
        /**
         * The inverse map for fast class label prediction
         */
        std::vector<std::string> inverseLabelMap;
    };
    
    /**
     * Abstract data storage.
     */
    class AbstractDataStorage {
    public:
        AbstractDataStorage() {};
        
        /**
         * Copy constructor
         */
        AbstractDataStorage(const AbstractDataStorage & other);
        
        /**
         * Destructor
         */
        virtual ~AbstractDataStorage()
        {
            free();
        };
        
        /**
         * Assignment operator
         */
        virtual AbstractDataStorage & operator=(const AbstractDataStorage & other);
        
        /**
         * Permutes the data points according to some permutation. 
         */
        virtual void permute(const std::vector<int> & permutation);
        
        /**
         * Permutes the data points randomly. 
         */
        void randPermute();
        
        /**
         * Returns the number of data points. 
         */
        int getSize() const
        {
            return dataPoints.size();
        }
        
        /**
         * Returns a reference to the i-th data point
         */
        DataPoint* getDataPoint(int i) const
        {
            return dataPoints[i];
        }
        
        /**
         * Returns the dimensionality of the data storage. 
         */
        int getDimensionality() const
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
        
        /**
         * Dumps information about the data storage
         */
        virtual void dumpInformation(std::ostream & stream = std::cout);
        
    protected:
        /**
         * Frees the data points and resets the array structures. 
         */
        virtual void free();
        /**
         * This is a list of data points. 
         */
        std::vector< DataPoint* > dataPoints;
        /**
         * This keeps track on which data points have to be freed and which 
         * don't. If we for example bootstrap the data set, then the points
         * don't have to be freed. 
         */
        std::vector<bool> freeFlags;
    };
    
    /**
     * Storage for unlabeled data. 
     */
    class UnlabeledDataStorage : public AbstractDataStorage {
    public:
        
    };
    
    /**
     * Basic labeled data storage.
     */
    class DataStorage : public AbstractDataStorage {
    public:
        DataStorage() : classcount(0) {}
        
        /**
         * Copy constructor
         */
        DataStorage(const DataStorage & other);
        
        /**
         * Assignment operator
         */
        DataStorage & operator=(const DataStorage & other);
        
        /**
         * Create a DataStorage for an excerpt of thisDataStorage.
         */
        DataStorage excerpt(int begin, int end);
        
        /**
         * Returns the i-th class label. 
         */
        int getClassLabel(int i) const
        {
            return classLabels[i];
        }
        
        /**
         * Returns the i-th class label. 
         */
        int & getClassLabel(int i) 
        {
            return classLabels[i];
        }
        
        /**
         * Adds a data point to the storage. 
         */
        void addDataPoint(DataPoint* point, int label, bool free = true)
        {
            dataPoints.push_back(point);
            classLabels.push_back(label);
            freeFlags.push_back(free);
            if (label >= classcount)
            {
                classcount = label+1;
            }
        }
        
        /**
         * Returns the number of classes. You have to call 
         * getCLassLabelMap.computeIntClassLabels() first. 
         */
        int getClasscount() const
        {
            return classcount;
        }
        
        /**
         * Permutes the data points according to some permutation. 
         */
        void permute(const std::vector<int> & permutation);
        
        /**
         * Bootstraps the training set and creates a set of N examples. The 
         * sampled array contains a flag for each data point. If it is true, then
         * the point was used in the bootstrap process.
         */
        void bootstrap(int N, DataStorage* dataStorage, std::vector<bool> & sampled) const;
        
        /**
         * Returns the class label map that maps the string class labels to 
         * integers.
         */
        const ClassLabelMap & getClassLabelMap() const
        {
            return classLabelMap;
        }
        
        /**
         * Returns the class label map that maps the string class labels to 
         * integers.
         */
        ClassLabelMap & getClassLabelMap()
        {
            return classLabelMap;
        }
        
        /**
         * Sets the number of class label manually. Do not use this unless you
         * are sure what you are doing.
         */
        void setClasscount(int _classcount)
        {
            classcount = _classcount;
        }
        
        /**
         * Dumps information about the data storage
         */
        virtual void dumpInformation(std::ostream & stream = std::cout);
        
    protected:
        /**
         * Frees the data points and resets the array structures. 
         */
        void free();
        /**
         * The total number of classes
         */
        int classcount;
        /**
         * These are the corresponding class labels to the data points
         */
        std::vector<int> classLabels;
        /**
         * The class label map
         */
        ClassLabelMap classLabelMap;
    };
    
    /**
     * This is the interface that has to be implemented if you wish to implement
     * a custom data provider. 
     */
    class DataProvider {
    public:
        virtual ~DataProvider() {}
        
        /**
         * Reads a labeled dataset from a stream.
         */
        virtual void read(std::istream & stream, DataStorage* dataStorage) = 0;
        
        /**
         * Reads an unlabeled dataset from a stream.
         */
        virtual void read(std::istream & stream, UnlabeledDataStorage* dataStorage) = 0;
        
        /**
         * Reads a labeled dataset from a file.
         */
        virtual void read(const std::string & filename, DataStorage* dataStorage);
        
        /**
         * Reads an unlabeled dataset from a file.. 
         */
        virtual void read(const std::string & filename, UnlabeledDataStorage* dataStorage);
    };
    
    /**
     * This data provider reads data from a local CSV file. 
     */
    class CSVDataProvider : public DataProvider{
    public:
        using DataProvider::read;    

        /**
         * Constructor: read csv with the given columns as label column.
         */
        CSVDataProvider(int classColumnIndex) : classColumnIndex(classColumnIndex), columnSeparator(",") {}
        
        /**
         * Read CSV with the given columns as label column, separated by the
         * given separator.
         */
        CSVDataProvider(int classColumnIndex, std::string columnSeperator) : classColumnIndex(classColumnIndex), columnSeparator(columnSeperator) {}
        
        /**
         * Destructor.
         */
        virtual ~CSVDataProvider() {}
        
        /**
         * Reads a labeled dataset from a stream.
         */
        virtual void read(std::istream & stream, DataStorage* dataStorage);
        
        /**
         * Reads an unlabeled dataset from a stream.
         */
        virtual void read(std::istream & stream, UnlabeledDataStorage* dataStorage);
        
    private:
        /**
         * The index of the column that contains the class label
         */
        int classColumnIndex;
        /**
         * Separator used between columns; default usually is ','
         */
        std::string columnSeparator;
    };
    
    /**
     * This data provider reads data from a local LIBSVM file. 
     */
    class LIBSVMDataProvider : public DataProvider {
    public:
        using DataProvider::read;
        
        /**
         * Reads a labeled dataset from a stream.
         */
        virtual void read(std::istream & stream, DataStorage* dataStorage);
    };
    
    /**
     * Reads the data set from a binary libforest format. This is the fastest
     * way to load a data set. 
     */
    class LibforestDataProvider : public DataProvider {
    public:
        using DataProvider::read;
        
        /**
         * Reads a labeled dataset from a stream.
         */
        virtual void read(std::istream & stream, DataStorage* dataStorage);
        
        /**
         * Reads an unlabeled dataset from a stream.
         */
        virtual void read(std::istream & stream, UnlabeledDataStorage* dataStorage);        
    };
    
    /**
     * This is the basic class for a data writer.
     */
    class DataWriter {
    public:
        /**
         * Writes the data to a stream. 
         */
        virtual void write(std::ostream & stream, DataStorage* dataStorage) = 0;
        
        /**
         * Writes the data to a file and add them to the data storage. 
         */
        virtual void write(const std::string & filename, DataStorage* dataStorage);
    };
    
    
    /**
     * This data provider reads data from a local CSV file. 
     */
    class CSVDataWriter : public DataWriter {
    public:
        using DataWriter::write;
        
        /**
         * Writes the data to a stream. 
         */
        virtual void write(std::ostream & stream, DataStorage* dataStorage) = 0;
    };
    
    /**
     * This data provider write data to a local LIBSVM file. 
     */
    class LIBSVMDataWriter : public DataWriter {
    public:
        using DataWriter::write;
        
        /**
         * Writes the data to a stream. 
         */
        virtual void write(std::ostream & stream, DataStorage* dataStorage) = 0;
    };
    
    /**
     * Writes the data set to a binary libforest format. This is the fastest
     * way to save a data set. 
     */
    class LibforestDataWriter : public DataWriter {
    public:
        using DataWriter::write;
        
        /**
         * Writes the data to a stream. 
         */
        virtual void write(std::ostream & stream, DataStorage* dataStorage);
    };
}

#endif