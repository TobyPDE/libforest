#ifndef LIBF_ONLINE_LEARNING_H
#define	LIBF_ONLINE_LEARNING_H

#include <cassert>
#include <functional>
#include <iostream>
#include <chrono>
#include <cmath>
#include <vector>
#include "classifiers.h"
#include "learning.h"

namespace libf {
    /**
     * Forward declarations.
     */
    class DataStorage;
    class DecisionTree;
    class RandomForest;
    
    /**
     * This is the base class for all online learners
     */
    template<class T, class S>
    class OnlineLearner : public AbstractLearner<T, S> {
    public:
        /**
         * Learns/updates a classifier online
         */
        virtual T* learn(const DataStorage* storage, T* model = NULL) const = 0;
    };
}

#endif	/* LIBF_ONLINE_LEARNING_H */

