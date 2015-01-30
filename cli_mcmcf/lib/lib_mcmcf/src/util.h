#ifndef MCMCF_UTIL_H
#define MCMCF_UTIL_H

/**
 * This makro helps you creating custom exception classes which accept error messages as constructor arguments. 
 * You can define a new exception class by: DEFINE_EXCEPTION(classname)
 * You can throw a new exception by: throw classname("Error message");
 */
#define DEFINE_EXCEPTION(classname)		\
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

namespace mcmcf {
    DEFINE_EXCEPTION(Exception)
}
#endif
 