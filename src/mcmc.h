#ifndef MCMCF_MCMC_H
#define MCMCF_MCMC_H

#include <vector>
#include <random>
#include <cassert>

namespace libf {
    /**
     * This is the cooling schedule interface. 
     */
    class SACoolingSchedule {
    public:
        /**
         * Calculates the next temperature based on the iteration and the 
         * current temperature. 
         */
        virtual float calcTemperature(int iteration, float temperature) const = 0;
        
        /**
         * Returns the start temperature
         */
        virtual float getStartTemperature() const = 0;
        
        /**
         * Returns the end temperature
         */
        virtual float getEndTemperature() const = 0;
    };
    
    /**
     * This is the interface one has to implement for a callback function for
     * SA.
     */
    template <class T>
    class SACallback {
    public:
        /**
         * The function that is called
         */
        virtual int callback(const T & state, float energy, const T & bestState, float bestEnergy, int iteration, float temperature) = 0;
    };
    
    /**
     * This is the interface one has to implement for a SA move.
     */
    template <class T>
    class SAMove {
    public:
        /**
         * Computes the move
         */
        virtual void move(const T & state, T & newState) = 0;
    };
    
    /**
     * This is the interface one has to implement for a SA energy function.
     */
    template <class T>
    class SAEnergy {
    public:
        /**
         * Computes the energy
         */
        virtual float energy(const T & state) = 0;
    };
    
    /**
     * This class implements general simulated annealing. T is the type of 
     * the state variable. We assume that the proposal distribution is symmetric. 
     * i.e.
     *   q(i|j) = q(j|i)
     */
    template <class T>
    class SimulatedAnnealing {
    public:
        SimulatedAnnealing() : coolingSchedule(0), energyFunction(0), numInnerLoops(500) {};
        
        /**
         * Adds a move
         */
        void addMove(SAMove<T>* move, float prob)
        {
            moves.push_back(move);
            moveProbabilities.push_back(prob);
        }
        
        /**
         * Sets the cooling schedule
         */
        void setCoolingSchedule(SACoolingSchedule* schedule)
        {
            coolingSchedule = schedule;
        }
        
        /**
         * Returns the cooling schedule
         */
        const SACoolingSchedule* getCoolingSchedule() const
        {
            return coolingSchedule;
        }
        
        /**
         * Returns the cooling schedule
         */
        SACoolingSchedule* getCoolingSchedule()
        {
            return coolingSchedule;
        }
        
        /**
         * Adds a callback function
         */
        void addCallback(SACallback<T>* callback)
        {
            callbacks.push_back(callback);
        }
        
        /**
         * Sets the number of inner loops
         */
        void setNumInnerLoops(int _numInnerLoops)
        {
            numInnerLoops = _numInnerLoops;
        }
        
        /**
         * Returns the number of inner loops
         */
        int getNumInnerLoops() const
        {
            return numInnerLoops;
        }
        
        /**
         * Sets the energy function
         */
        void setEnergyFunction(SAEnergy<T>* function)
        {
            energyFunction = function;
        }
        
        /**
         * Optimizes the error function for a given initialization. 
         */
        void optimize(T & state)
        {
            assert(coolingSchedule != 0);
            assert(energyFunction != 0);
            assert(moves.size() > 0);
            std::random_device rd;
            std::mt19937 g(rd());
            // Set up a uniform distribution. We need this in order to accept 
            // hill climbing steps and choose the moves
            std::uniform_real_distribution<float> uniformDist(0,1);
            
            // Get the temperature
            float temperature = coolingSchedule->getStartTemperature();
            
            // Get the current error
            float currentEnergy = energyFunction->energy(state);
            
            // Keep track on the optimum
            float bestEnergy = currentEnergy;
            T bestState = state;
            
            // Start the optimization
            int iteration = 0;
            while (temperature > coolingSchedule->getEndTemperature())
            {
                iteration++;
                temperature = coolingSchedule->calcTemperature(iteration, temperature);
                
                for (int inner = 0; inner < numInnerLoops; inner++)
                {
                    // Choose a move at random
                    const float u = uniformDist(g);
                    int randomMove = -1;
                    float probSum = 0;
                    for (size_t i = 0; i < moveProbabilities.size(); i++)
                    {
                        if (probSum <= u && u < probSum+moveProbabilities[i])
                        {
                            randomMove = static_cast<int>(i);
                            break;
                        }
                        else
                        {
                            probSum += moveProbabilities[i];
                        }
                    }
                    // Did we find a move?
                    if (randomMove < 0)
                    {
                        // Nope, then we sampled the last move
                        randomMove = static_cast<int>(moveProbabilities.size() - 1);
                    }

                    // Get the result of the move
                    T newState;
                    moves[randomMove]->move(state, newState);

                    // Calculate the new error and the improvement
                    const float newError = energyFunction->energy(newState);
                    const float improvement = newError - currentEnergy;

                    // What to do?
                    if (improvement <= 0)
                    {
                        // We improved the energy, accept this step
                        currentEnergy = newError;
                        state = newState;
                    }
                    else
                    {
                        // We did not improve. Accept the step with a certain 
                        // probability
                        const float u = uniformDist(g);
                        if (std::log(u) <= -improvement/temperature)
                        {
                            currentEnergy = newError;
                            state = newState;
                        }
                    }
                    
                    if (currentEnergy < bestEnergy)
                    {
                        bestEnergy = currentEnergy;
                        bestState = state;
                    }
                }
                // Call the callback functions
                for (size_t i = 0; i < callbacks.size(); i++)
                {
                    callbacks[i]->callback(state, currentEnergy, bestState, bestEnergy, iteration, temperature);
                }
            }
            state = bestState;
        }
        
    private:
        /**
         * These are the registered moves
         */
        std::vector<SAMove<T>*> moves;
        /**
         * The probability for choosing this move
         */
        std::vector<float> moveProbabilities;
        /**
         * The cooling schedule that determines the temperature
         */
        SACoolingSchedule* coolingSchedule;
        /**
         * The callback functions
         */
        std::vector<SACallback<T>*> callbacks;
        /**
         * The number of inner loops
         */
        int numInnerLoops;
        /**
         * The error function
         */
        SAEnergy<T>* energyFunction;
    };
    
    /**
     * This is a geometric cooling schedule: t_k+1 = t_k * alpha. 
     */
    class GeometricCoolingSchedule : public SACoolingSchedule {
    public:
        GeometricCoolingSchedule() : startTemperature(100), endTemperature(1), alpha(0.8f) {}
        
        /**
         * Sets the start temperature
         */
        void setStartTemperature(float temp)
        {
            startTemperature = temp;
        }
        
        /**
         * Returns the start temperature
         */
        virtual float getStartTemperature() const
        {
            return startTemperature;
        }
        
        /**
         * Sets the end temperature
         */
        void setEndTemperature(float temp)
        {
            endTemperature = temp;
        }
        
        /**
         * Returns the end temperature
         */
        virtual float getEndTemperature() const
        {
            return endTemperature;
        }
        
        /**
         * Calculates the next temperature based on the iteration and the 
         * current temperature. 
         */
        virtual float calcTemperature(int, float temperature) const
        {
            return temperature*alpha;
        }
        
        /**
         * Sets alpha
         */
        void setAlpha(float _alpha)
        {
            alpha = _alpha;
        }
        
        /**
         * Returns alpha
         */
        float getAlpha() const
        {
            return alpha;
        }
        
    private:
        /**
         * The start temperature
         */
        float startTemperature;
        /**
         * The end temperature
         */
        float endTemperature;
        /**
         * Cooling schedule alpha
         */
        float alpha;
    };
}
#endif