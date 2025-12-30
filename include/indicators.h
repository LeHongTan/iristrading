#ifndef INDICATORS_H
#define INDICATORS_H

#include <vector>

class Indicators {
    public:
        double calculateSMA(const std::vector<double>& data, int period);
};

#endif