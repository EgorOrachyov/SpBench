////////////////////////////////////////////////////////////////////////////////////
// MIT License                                                                    //
//                                                                                //
// Copyright (c) 2021 Egor Orachyov                                               //
//                                                                                //
// Permission is hereby granted, free of charge, to any person obtaining a copy   //
// of this software and associated documentation files (the "Software"), to deal  //
// in the Software without restriction, including without limitation the rights   //
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell      //
// copies of the Software, and to permit persons to whom the Software is          //
// furnished to do so, subject to the following conditions:                       //
//                                                                                //
// The above copyright notice and this permission notice shall be included in all //
// copies or substantial portions of the Software.                                //
//                                                                                //
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR     //
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,       //
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE    //
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER         //
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,  //
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE  //
// SOFTWARE.                                                                      //
////////////////////////////////////////////////////////////////////////////////////

#ifndef SPBENCH_ARGS_PROCESSOR_HPP
#define SPBENCH_ARGS_PROCESSOR_HPP

#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <sstream>
#include <cassert>

namespace benchmark {

    // Parse benchmark input args
    class ArgsProcessor {
    public:

        struct Entry {
            std::string name;
            bool isUndirected;
            size_t iterations;
        };

        void parse(int argc, const char** argv) {
            assert(argc >= 2);
            mArgc = argc;
            mArgv = argv;

            if (std::string(argv[1]) == "-E") {
                assert(argc == 5);

                std::string name = argv[2];
                int isUndirected = 0;
                size_t iterations = 0;

                std::stringstream lineParser(argv[3]);
                lineParser >> isUndirected;

                lineParser = std::stringstream(argv[4]);
                lineParser >> iterations;

                Entry entry{ std::move(name), isUndirected != 0, iterations };
                mEntries.push_back(std::move(entry));
            } else {
                // Suppose, the second one is the name of the file with the config of input data
                const char* configName = argv[1];
                std::fstream configFile;
                configFile.open(configName, std::ios_base::in);

                if (!configFile.is_open()) {
                    std::cerr << "Failed to open config file: " << configName << std::endl;
                    return;
                }

                while (!configFile.eof()) {
                    std::string line;
                    std::getline(configFile, line);

                    if (line.empty() || line[0] == '%')
                        continue;

                    std::string name;
                    int isUndirected = 0;
                    size_t iterations = 0;

                    std::stringstream lineParser(line);
                    lineParser >> name >> isUndirected >> iterations;

                    Entry entry{ std::move(name), isUndirected != 0, iterations };
                    mEntries.push_back(std::move(entry));
                }
            }

            mIsParsed = true;
        }

        bool isParsed() const {
            return mIsParsed;
        }

        const std::vector<Entry> &getEntries() const {
            return mEntries;
        }

        size_t getExperimentsCount() const {
            return mEntries.size();
        }

    private:
        int mArgc = 0;
        const char** mArgv = nullptr;
        bool mIsParsed = false;
        std::vector<Entry> mEntries;
    };

}

#endif //SPBENCH_ARGS_PROCESSOR_HPP
