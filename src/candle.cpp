#include "candle.h"

Candle::Candle(long long timeOpen, long long timeClose, double open, double high, double low, double close, double volume, double turnover) 
{
    _timeOpen = timeOpen;
    _timeClose = timeClose;
    _open = open;
    _high = high;
    _low = low;
    _close = close;
    _volume = volume;
    _turnover = turnover;
}

long long Candle::getTimeOpen() const {
    return _timeOpen;
}

long long Candle::getTimeClose() const {
    return _timeClose;
}   

double Candle::getOpen() const {
    return _open;
}

double Candle::getHigh() const {
    return _high;
}

double Candle::getLow() const {
    return _low;
}

double Candle::getClose() const {
    return _close;
}   

double Candle::getVolume() const {
    return _volume;
}

double Candle::getTurnover() const {
    return _turnover;
}

