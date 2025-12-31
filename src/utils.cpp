#include "utils.h"
#include <curl/curl.h>
#include <fstream>

#include <thread>
#include <chrono>
#include <ctime>
#include <iomanip>
#include <sstream>

#include <iostream>

#include <nlohmann/json.hpp>
using json = nlohmann::json;

long long STARTTIME = 1514739600000; // Example start time: January 1, 2021

size_t WriteCallback(void *contents, size_t size, size_t nmemb, void *userp)
{
    ((std::string *)userp)->append((char *)contents, size * nmemb);
    return size * nmemb;
}

std::string formatTimestamp(long long timestamp)
{
    std::time_t time = timestamp / 1000; // Convert milliseconds to seconds
    std::tm *tm_ptr = std::gmtime(&time);
    char buffer[20];
    std::strftime(buffer, sizeof(buffer), "%Y-%m-%d %H:%M:%S", tm_ptr);
    return std::string(buffer);
}

std::string Utils::sendRequest(const std::string &url)
{
    CURL *curl;
    CURLcode res;

    std::string responseString;

    curl = curl_easy_init();
    if (curl)
    {
        curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, &responseString);

        curl_easy_setopt(curl, CURLOPT_TIMEOUT, 10L);

        res = curl_easy_perform(curl);
        if (res != CURLE_OK)
        {
            std::string timeLog = std::to_string(std::time(nullptr));
            saveDataToFile("[ERROR] " + formatTimestamp(std::stoll(timeLog)) + ": " + std::string(curl_easy_strerror(res)), "report/error_log.txt");
        }
        curl_easy_cleanup(curl);
    }
    else
    {
        std::string timeLog = std::to_string(std::time(nullptr));
        saveDataToFile("[ERROR] " + formatTimestamp(std::stoll(timeLog)) + ": Failed to initialize CURL", "report/error_log.txt");
    }

    return responseString;
}

void Utils::saveDataToFile(const std::string &data, const std::string &filename)
{
    std::ofstream outFile(filename, std::ios::app); // Append mode for error logs
    if (outFile.is_open())
    {
        outFile << data << std::endl;
        outFile.close();
    }
    else
    {
        std::cerr << "Failed to open file: " << filename << std::endl;
    }
}

void Utils::downloadData(std::string symbol, std::string interval, long long startTimestamp)
{
    long long startTime = startTimestamp;

    std::string filename = "data/" + symbol + "_" + interval + ".csv";
    std::ofstream outFile(filename, std::ios::app);

    if (!outFile.is_open())
    {
        std::ofstream errorLog("report/error_log.txt", std::ios::app);
        if (errorLog.is_open())
        {
            std::string timeLog = std::to_string(std::time(nullptr));
            errorLog << "[ERROR] " << formatTimestamp(std::stoll(timeLog)) << ": Failed to open file: " << filename << std::endl;
            errorLog.close();
        }
        else
        {
            std::cerr << "Failed to open error log file." << std::endl;
        }

        return;
    }

    std::ifstream checkFile(filename);
    if (checkFile.peek() == std::ifstream::traits_type::eof()) {
        outFile << "Time,Open,High,Low,Close,Volume,Turnover\n";
    }
    checkFile.close();

    while (true)
    {
        std::string url = "https://api.bybit.com/v5/market/kline?category=linear"
                          "&symbol=" + symbol +
                          "&interval=" + interval +
                          "&start=" + std::to_string(startTime) +
                          "&limit=1000";

        std::cout << "Fetching data from: " << startTime << std::endl;

        std::string response = sendRequest(url);
        if (response.empty() || response == "[]")
        {
            std::cout << "No more data to fetch." << std::endl;
            break;
        }

        auto j = json::parse(response);
        if (!j.contains("result") || j["result"]["list"].empty())
        {
            std::cout << "Downloaded successfully." << std::endl;
            break;
        }
        auto dataList = j["result"]["list"];

        for (const auto &item : dataList)
        {
            long long openTime = std::stoll(item[0].get<std::string>());
            double open = std::stod(item[1].get<std::string>());
            double high = std::stod(item[2].get<std::string>());
            double low = std::stod(item[3].get<std::string>());
            double close = std::stod(item[4].get<std::string>());
            double volume = std::stod(item[5].get<std::string>());
            double turnover = std::stod(item[6].get<std::string>());

            outFile << formatTimestamp(openTime) << "," 
            << std::fixed << std::setprecision(3) << open << "," 
            << std::fixed << std::setprecision(3) << high << "," 
            << std::fixed << std::setprecision(3) << low << "," 
            << std::fixed << std::setprecision(3) << close << "," 
            << std::fixed << std::setprecision(5) << volume << "," 
            << std::fixed << std::setprecision(2) << turnover << "\n"; 
        }

        long long latestTsInBatch = std::stoll(dataList[0][0].get<std::string>());
        int intervalMin = (interval == "D") ? 1440 : std::stoi(interval);
        startTime = latestTsInBatch + intervalMin * 60 * 1000;

        // Throttle requests
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }

    outFile.close();

    std::ofstream successLog("report/success_log.txt", std::ios::app);
    if (successLog.is_open())
    {
        std::string timeLog = std::to_string(std::time(nullptr));
        successLog << "[SUCCESS] " << formatTimestamp(std::stoll(timeLog)) << ": Data downloaded for " << symbol << " at interval " << interval << std::endl;
        successLog.close();
        std::cout << "Saved data to " << filename << std::endl;
    }
}