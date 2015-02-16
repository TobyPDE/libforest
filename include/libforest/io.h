#ifndef LIBF_IO_H
#define LIBF_IO_H

#include "util.h"

#include <iostream>
#include <fstream>

namespace libf {
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
    inline void writeBinary(std::ostream & stream, const std::string & value)
    {
        // Write the length of the string
        writeBinary(stream, static_cast<int>(value.size()));
        // Write the individual characters
        for (size_t i = 0; i < value.size(); i++)
        {
            writeBinary(stream, value[i]);
        }
    }

    /**
     * Reads a binary string from a stream
     */
    template <>
    inline void readBinary(std::istream & stream, std::string & value)
    {
        // Read the length
        int length;
        readBinary(stream, length);
        value.resize(length);
        
        // Read the characters
        for (int i = 0; i < length; i++)
        {
            readBinary(stream, value[i]);
        }
    }
    
    /**
     * Writes a vector to a stream
     */
    template <class T>
    void writeBinary(std::ostream & stream, const std::vector<T> & v)
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
    void readBinary(std::istream & stream, std::vector<T> & v)
    {
        int N;
        readBinary(stream, N);
        v.resize(N);
        for (int i = 0; i < N; i++)
        {
            readBinary(stream, v[i]);
        }
    }
    
    /**
     * Reads anything from a binary file
     */
    template <class T>
    void read(const std::string & filename, T* o)
    {
        // Open the file
        std::ifstream stream(filename, std::ios::binary);
        if (!stream.is_open())
        {
            throw Exception("Could not open file.");
        }
        o->read(stream);
        stream.close();
    }
    
    /**
     * Reads anything from a binary file
     */
    template <class T>
    void write(const std::string & filename, T* o)
    {
        // Open the file
        std::ofstream stream(filename, std::ios::binary);
        if (!stream.is_open())
        {
            throw Exception("Could not open file.");
        }
        o->write(stream);
        stream.close();
    }
}

#endif
