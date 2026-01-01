#ifndef CANDLE_H
#define CANDLE_H

class Candle {
    private:
        long long _timeOpen;
        long long _timeClose;

        double _open;
        double _high;
        double _low;
        double _close;
        double _volume;
        double _turnover;

    public:
        Candle(long long timeOpen, long long timeClose, double open, double high, double low, double close, double volume, double turnover);

        long long getTimeOpen() const;
        long long getTimeClose() const;

        double getOpen() const;
        double getHigh() const;
        double getLow() const;
        double getClose() const;
        double getVolume() const;
        double getTurnover() const;
};

#endif