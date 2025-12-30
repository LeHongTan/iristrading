#ifndef UTILS_H
#define UTILS_H

#include <string>

extern long long STARTTIME;

class Utils {
    public:
        std::string sendRequest(const std::string& url);
        void saveDataToFile(const std::string& data, const std::string& filename);

        void downloadData(std::string symbol, std::string interval, long long startTime);
};

#endif  