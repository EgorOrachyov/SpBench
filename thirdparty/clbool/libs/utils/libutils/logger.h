#pragma once
#include <string>
#include <iostream>
#include <fstream>
#include <utility>


struct Logger {
private:
    std::fstream fstream;
public:
    std::string path;
    std::ostream &stream = std::cout;

    Logger() = default;

    explicit Logger(std::string path)
            : path(std::move(path))
            , stream(fstream)
    {
        fstream.open(this->path, std::fstream::in | std::fstream::out | std::fstream::trunc);
        if (!fstream.is_open()) {
            std::cerr << "cannot open file\n";
        }
    }

    const Logger& operator*() const {
        stream << "\n[LOGGING] ";
        return *this;
    }

    const Logger& operator*(int) const {
        stream << "\n";
        return *this;
    }

    template <class T>
    const Logger& operator<<(const T &str) const {
        stream << str;
        return *this;
    }
};