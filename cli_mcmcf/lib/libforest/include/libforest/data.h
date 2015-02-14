#ifndef MCMCF_DATA_H
#define MCMCF_DATA_H

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

namespace libf {
    /**
     * This is the data point interface. You can implement this interface if
     * you wish to use special feature vector such as pixel pairs, 
     * categorical data and so on. 
     */
    class _DataPoint {
    public:
        /**
         * Returns the i-th entry of the feature vector
         */
        virtual const float & at(int i) const = 0;
        
        /**
         * Returns the i-th entry of the feature vector
         */
        virtual float & at(int i) = 0;
        
        /**
         * Returns the dimensionality
         */
        virtual int getDimensionality() const = 0;
    };
    
    /**
     * This class represents in individual data point. 
     * TODO: Make the data point class a derived class.
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
     */
    class ClassLabelMap {
    public:
        
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
     * This class provides some baseline data storage properties. It is used 
     * to load data sets. It can be used directly in order to learn and evaluate
     * the performance of the classifiers. 
     * 
     * Important: All data points are freed by the storage object!
     * TODO: Remove string labels in favor of a class map.
     * TODO: Use abstract data point class.
     */
    class DataStorage {
    public:
        explicit DataStorage() {}
        
        /**
         * Copy constructor
         */
        DataStorage(const DataStorage & other);
        
        /**
         * Destructor
         */
        virtual ~DataStorage();
        
        /**
         * Returns the number of data points. 
         */
        int getSize() const
        {
            return static_cast<int>(dataPoints.size());
        }
        
        /**
         * Returns the i-th data point together with the i-th class label. 
         */
        std::pair< DataPoint*, std::string> operator[](int i) const
        {
            return std::make_pair(dataPoints[i], classLabels[i]);
        }
        
        /**
         * Returns a reference to the i-th data point
         */
        DataPoint* getDataPoint(int i) const
        {
            return dataPoints[i];
        }
        
        /**
         * Returns the i-th class label. 
         */
        const std::string & getClassLabel(int i ) const
        {
            return classLabels[i];
        }
        
        /**
         * Returns the integer class label for a certain data point. In order
         * for this to work, the integer class labels have to be computed first.
         */
        int getIntClassLabel(int i) const
        {
            return intClassLabels[i];
        }
        
        /**
         * Computes the integer class labels. 
         */
        void computeIntClassLabels();
        
        /**
         * Computes the integer class labels using the class label map from the
         * provided storage.
         */
        void computeIntClassLabels(const DataStorage* dataStorage);
        
        /**
         * Adds a data point to the storage. 
         */
        void addDataPoint(DataPoint* point, const std::string & label, bool free = true)
        {
            dataPoints.push_back(point);
            classLabels.push_back(label);
            freeFlags.push_back(free);
        }
        
        /**
         * Returns the dimensionality of the data storage. 
         */
        int getDimensionality() const;
        
        /**
         * Returns the number of classes. You have to call computeIntClassLabels()
         * first. 
         */
        int getClasscount() const
        {
            return static_cast<int>(classLabelMap.size());
        }
        
        /**
         * Permutes the data points according to some permutation. 
         */
        void permute(const std::vector<int> & permutation);
        
        /**
         * Permutes the data points randomly. 
         */
        void randPermute();
        
        /**
         * Splits the data set into two sets. The ratio determines how many 
         * points stay in this set and how many points will go to the other
         * set. 
         */
        void split(float ratio, DataStorage* other);
        
        /**
         * Bootstraps the training set and creates a set of N examples.  
         */
        void bootstrap(int N, DataStorage* dataStorage) const;
        
        /**
         * Bootstraps the training set and creates a set of N examples. The 
         * sampled array contains a flag for each data point. If it is true, then
         * the point was used in the bootstrap process.
         */
        void bootstrap(int N, DataStorage* dataStorage, std::vector<bool> & sampled) const;
        
    private:
        /**
         * This is a list of data points. 
         */
        std::vector< DataPoint* > dataPoints;
        /**
         * These are the corresponding class labels to the data points
         */
        std::vector<std::string> classLabels;
        /**
         * This keeps track on which data points have to be freed and which 
         * don't. If we for example bootstrap the data set, then the points
         * don't have to be freed. 
         */
        std::vector<bool> freeFlags;
        /**
         * The integer class labels. 
         */
        std::vector<int> intClassLabels;
        /**
         * The class label map
         */
        std::map<std::string, int> classLabelMap;
    };
    
    /**
     * This is the interface that has to be implemented if you wish to implement
     * a custom data provider. 
     * 
     * TODO: Give option to load string labels/int labels
     */
    class DataProvider {
    public:
        /**
         * Reads the data from a source and add them to the data storage. 
         */
        virtual void read(const std::string & source, DataStorage* dataStorage) = 0;
    };
    
    /**
     * This data provider reads data from a local CSV file. 
     */
    class CSVDataProvider : public DataProvider{
    public:
        CSVDataProvider(int classColumnIndex) : classColumnIndex(classColumnIndex) {}
        
        /**
         * Reads the data from a source and add them to the data storage. 
         */
        virtual void read(const std::string & source, DataStorage* dataStorage);
        
    private:
        /**
         * The index of the column that contains the class label
         */
        int classColumnIndex;
    };
    
    /**
     * This data provider reads data from a local LIBSVM file. 
     */
    class LIBSVMDataProvider : public DataProvider {
    public:
        /**
         * Reads the data from a source and add them to the data storage. 
         */
        virtual void read(const std::string & source, DataStorage* dataStorage);
    };
    
    /**
     * This is the basic class for a data writer.
     */
    class DataWriter {
    public:
        /**
         * Writes the data
         */
        virtual void write(const std::string & dest, DataStorage* dataStorage) = 0;
    };
}

#endif