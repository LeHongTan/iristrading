#include "coin.h"

Coin::Coin(const std::string& symbol, const std::string& interval)
    : _symbol(symbol), _interval(interval) {}

std::string Coin::getSymbol() const {
    return _symbol;
}

std::string Coin::getInterval() const {
    return _interval;
}