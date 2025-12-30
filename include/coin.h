#ifndef COIN_H
#define COIN_H

#include <string>

class Coin {
    private:
        std::string _symbol;
        std::string _interval;

    public:
        Coin(const std::string& symbol, const std::string& interval);

        std::string getSymbol() const;
        std::string getInterval() const;
};

#endif