#ifndef LIBF_UTIL_H
#define LIBF_UTIL_H

#include <iostream>

/**
 * This makro helps you creating custom exception classes which accept error messages as constructor arguments. 
 * You can define a new exception class by: DEFINE_EXCEPTION(classname)
 * You can throw a new exception by: throw classname("Error message");
 */
#define LIBF_DEFINE_EXCEPTION(classname)		\
	class classname : public std::exception {	\
	public:		\
		classname() { this->ptrMessage = 0; };	\
		classname(const char* _ptrMessage) : ptrMessage(_ptrMessage) {};	\
		classname(std::string str) {_msg = str; ptrMessage = _msg.c_str(); };	\
		classname(const char* _ptrMessage, int l) : ptrMessage(_ptrMessage) { };	\
        virtual ~classname() throw() {}; \
		virtual const char* what() const throw() { return (this->ptrMessage != 0 ? this->ptrMessage : "No Message"); }	\
	private: \
        std::string _msg; \
		const char* ptrMessage;		\
	};

/**
 * This is the buffer size for the arrays in the graph structures
 */
#define LIBF_GRAPH_BUFFER_SIZE 5000

namespace libf {
    LIBF_DEFINE_EXCEPTION(Exception)

    class Util {
    public:
        /**
         * Applies a permutation to the vector of elements. This function does
         * not work in-place.
         */
        template <class T>
        static void permute(const std::vector<int> & permutation, const std::vector<T> & in, std::vector<T> & out)
        {
            assert(permutation.size() == in.size());

            // Make the output array of the correct sice. 
            out.resize(in.size());

            // Copy the elements
            for (size_t i = 0; i < permutation.size(); i++)
            {
                out[permutation[i]] = in[i];
            }
        }
        
        /**
         * Computes the hamming distance between two vectors
         */
        template <class T>
        static int hammingDist(const std::vector<T> & v1, const std::vector<T> & v2)
        {
            assert(v1.size() == v2.size());
            int result = 0;
            for (size_t i = 0; i < v1.size(); i++)
            {
                if (v1[i] != v2[i])
                {
                    result++;
                }
            }
            return result;
        }
    };
    
    template <class T>
    void dumpVector(const std::vector<T> & v)
    {
        for (size_t i = 0; i < v.size(); i++)
        {
            std::cout << i << ": " << v[i] << std::endl;
        }
        std::cout.flush();
    }
}
#endif
 