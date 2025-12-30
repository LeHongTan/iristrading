#include <iostream>
#include <vector>

#include "utils.h"
#include "coin.h"

int main() {
    std::cout << "IRIS TRADING\n";

    Utils utils;

    std::vector<Coin> coins = {
        Coin("BTCUSDT", "5"), Coin("BTCUSDT", "15"), Coin("BTCUSDT", "60"), Coin("BTCUSDT", "240"), Coin("BTCUSDT", "D"),
        Coin("ETHUSDT", "5"), Coin("ETHUSDT", "15"), Coin("ETHUSDT", "60"), Coin("ETHUSDT", "240"), Coin("ETHUSDT", "D"),
        Coin("SOLUSDT", "5"), Coin("SOLUSDT", "15"), Coin("SOLUSDT", "60"), Coin("SOLUSDT", "240"), Coin("SOLUSDT", "D"),
    };

    for (const auto& coin : coins) {
        std::cout << "Downloading data for " << coin.getSymbol() << " at interval " << coin.getInterval() << "\n";
        utils.downloadData(coin.getSymbol(), coin.getInterval(), STARTTIME);
    }

    return 0;
}