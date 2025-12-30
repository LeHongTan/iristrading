#include "utils.h"
#include <curl/curl.h>
#include <fstream>

#include <thread>
#include <chrono>
#include <ctime>

#include <iostream>

long long STARTTIME = 1514739600000; // Example start time: January 1, 2021

size_t WriteCallback(void *contents, size_t size, size_t nmemb, void *userp)
{
    ((std::string *)userp)->append((char *)contents, size * nmemb);
    return size * nmemb;
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
            saveDataToFile("[ERROR] " + timeLog + ": " + std::string(curl_easy_strerror(res)), "report/error_log.txt");
        }
        curl_easy_cleanup(curl);
    }

    return responseString;
}

void Utils::saveDataToFile(const std::string &data, const std::string &filename)
{
    std::ofstream outFile(filename);
    if (outFile.is_open())
    {
        outFile << data;
        outFile.close();
    }
}

void Utils::downloadData(std::string symbol, std::string interval, long long startTimestamp) 
{
    long long startTime = startTimestamp;

    while (true)
    {
        std::string url = "https://api.bybit.com/v5/market/kline?category=linear"
                          "&symbol=" + symbol +
                          "&interval=" + interval +
                          "&start=" + std::to_string(startTime) +
                          "&limit=1000";

        std::cout << "Fetching data from: " << startTime << std::endl;

        std::string response = sendRequest(url);
        if (response.empty() || response.find("\"list\":[]") != std::string::npos) 
        {
            std::cout << "No more data to fetch." << std::endl;
            break;

        }

        std::string filename = "data/" + symbol + "_" + interval + "_" + std::to_string(startTime) + ".json";
        saveDataToFile(response, filename);

        // Update startTime for next batch
        startTime += 1000 * 60 * std::stoi(interval) * 1000; // interval in minutes

        // Throttle requests
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }
}