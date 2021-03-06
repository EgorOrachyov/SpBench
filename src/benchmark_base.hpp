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

#ifndef CUBOOL_BENCHMARK_BASE_HPP
#define CUBOOL_BENCHMARK_BASE_HPP

#include <chrono>
#include <string>
#include <functional>
#include <iostream>
#include <vector>
#include <cassert>
#include <fstream>
#include <cmath>
#include <iomanip>

namespace benchmark {

    struct Timer {
    public:
        void start() {
            mStart = mEnd = clock::now();
        }

        void end() {
            mEnd = clock::now();
        }

        double getElapsedTimeMs() const {
            using namespace std::chrono;
            return (double) duration_cast<nanoseconds>(mEnd - mStart).count() / 1.0e6;
        }

    private:
        using clock = std::chrono::high_resolution_clock;
        using timepoint = clock::time_point;
        timepoint mStart;
        timepoint mEnd;
    };

    struct TimeQuery {
    public:
        void addTimeSample(double ms) {
            mSamplesCount += 1;
            mTimeSumMS += ms;
        }

        double getAverageTimeMs() const {
            return mTimeSumMS / (double) mSamplesCount;
        }

        double getTotalTimeMS() const {
            return mTimeSumMS;
        }

        int getSamplesCount() const {
            return mSamplesCount;
        }

    private:
        double mTimeSumMS = 0.0f;
        int mSamplesCount = 0;
    };

    class BenchmarkBase {
    protected:

        //////////////////////////////////////////////////
        // Benchmark config


        /** Name of the benchmark */
        std::string benchmarkName;
        /** Total number of experiments to run */
        size_t experimentsCount = 0;

        //////////////////////////////////////////////////
        // Benchmark results

        struct PerExperiment {
            std::string userFriendlyName;
            size_t iterationsCount = 0;
            double totalTime = 0.0;
            double averageTime = 0.0;
            double averageTimeDropFirst = 0.0;
            double minIterationTime = 0.0;
            double maxIterationTime = 0.0;
            double standardDeviationMs = 0.0f;
            std::vector<double> samplesMs;
        };

        std::vector<PerExperiment> results;

        //////////////////////////////////////////////////
        // Override functions below for your benchmark

        virtual void setupBenchmark() = 0;
        virtual void tearDownBenchmark() = 0;

        virtual void setupExperiment(size_t experimentIdx, size_t& iterationsCount, std::string& name) = 0;
        virtual void tearDownExperiment(size_t experimentIdx) = 0;

        virtual void setupIteration(size_t experimentIdx, size_t iterationIdx) = 0;
        virtual void execIteration(size_t experimentIdx, size_t iterationIdx) = 0;
        virtual void tearDownIteration(size_t experimentIdx, size_t iterationIdx) = 0;

    public:

        //////////////////////////////////////////////////
        // Exec this to run benchmark

        std::fstream log;

        void runBenchmark() {
            assert(experimentsCount > 0);
            assert(results.empty());

            auto logName = "Log-" + benchmarkName + ".txt";

            log.open(logName, std::ios_base::out | std::ios_base::app);
            if (!log.is_open()) {
                std::cerr << "Failed to open log file" << std::endl;
                return;
            }

            log << "=-=-=-=-=-= RUN: " << benchmarkName << " =-=-=-=-=-=" << std::endl << std::endl;

            setupBenchmark();

            for (auto experimentIdx = 0; experimentIdx < experimentsCount; experimentIdx++) {
                size_t iterationsCount;
                std::string name;


                setupExperiment(experimentIdx, iterationsCount, name);

                log << "> Begin experiment: " << experimentIdx << " name: "<< name << std::endl;

                PerExperiment perExperiment{};
                perExperiment.userFriendlyName = std::move(name);
                perExperiment.iterationsCount = iterationsCount;
                perExperiment.minIterationTime = std::numeric_limits<double>::max();
                perExperiment.samplesMs.reserve(iterationsCount);

                TimeQuery timeQuery;
                double firstIteration = 0.0;

                for (auto iterationIdx = 0; iterationIdx < iterationsCount; iterationIdx++) {
                    setupIteration(experimentIdx, iterationIdx);

                    Timer timer; {
                        timer.start();
                        execIteration(experimentIdx, iterationIdx);
                        timer.end();
                    }

                    tearDownIteration(experimentIdx, iterationIdx);

                    double elapsedTimeMs = timer.getElapsedTimeMs();

                    timeQuery.addTimeSample(elapsedTimeMs);
                    perExperiment.maxIterationTime = std::max(perExperiment.maxIterationTime, elapsedTimeMs);
                    perExperiment.minIterationTime = std::min(perExperiment.minIterationTime, elapsedTimeMs);
                    perExperiment.samplesMs.push_back(elapsedTimeMs);

                    log << "[" << iterationIdx << "] time: " << elapsedTimeMs << " ms" << std::endl;

                    if (iterationIdx == 0) {
                        firstIteration = timer.getElapsedTimeMs();
                    }
                }

                perExperiment.totalTime = timeQuery.getTotalTimeMS();
                perExperiment.averageTime = timeQuery.getAverageTimeMs();
                perExperiment.averageTimeDropFirst = iterationsCount > 1? (timeQuery.getTotalTimeMS() - firstIteration) / (double)(timeQuery.getSamplesCount() - 1): 0.0;

                if (iterationsCount > 1) {
                    double sd = 0.0f;
                    for (auto sample: perExperiment.samplesMs) {
                        auto diff = (sample - perExperiment.averageTime);
                        sd += diff * diff;
                    }

                    sd = sd / (double) (iterationsCount - 1);
                    perExperiment.standardDeviationMs = std::sqrt(sd);
                }

                tearDownExperiment(experimentIdx);

                log << "> End experiment: " << experimentIdx << std::endl
                    << "> Stats: " << std::endl
                    << ">  iterations   = " << perExperiment.iterationsCount << std::endl
                    << ">  total        = " << perExperiment.totalTime << " ms" << std::endl
                    << ">  average      = " << perExperiment.averageTime << " ms" << std::endl
                    << ">  average (-1) = " << perExperiment.averageTimeDropFirst << " ms" << std::endl
                    << ">  sd           = " << perExperiment.standardDeviationMs << " ms" << std::endl
                    << ">  min          = " << perExperiment.minIterationTime << " ms" << std::endl
                    << ">  max          = " << perExperiment.maxIterationTime << " ms" << std::endl;

                log << ">  samples: " << std::endl;
                auto id = 0;
                for (auto sample: perExperiment.samplesMs) {
                    log << ">   " << id << ": " << sample << " ms" << std::endl;
                    id += 1;
                }

                log << std::endl;
                results.push_back(std::move(perExperiment));
            }

            tearDownBenchmark();

            // Print final summary stuff here
            {
                std::string summaryName = "Summary-" + benchmarkName + ".txt";
                std::fstream summaryFile;

                const int alignSamples = 15;
                const int alignExpect = 15;
                const int alignSd = 15;
                const int maxNameLength = 50;

                summaryFile.open(summaryName, std::ios_base::in); {
                    // File does no exists
                    if (!summaryFile.is_open()) {
                        summaryFile.open(summaryName, std::ios_base::out);
                        assert(summaryFile.is_open());
                        summaryFile << "Benchmarking: " << benchmarkName << std::endl << std::endl;

                        summaryFile << std::setw(maxNameLength) << "Friendly name" << "| "
                                    << std::setw(alignSamples) << "iterations" << "| "
                                    << std::setw(alignExpect) << "expectation ms" << "| "
                                    << std::setw(alignSd) << "sd ms" << "| " << std::endl;
                    }
                    else {
                        summaryFile.close();
                        summaryFile.open(summaryName, std::ios_base::app);
                    }
                }

                if (summaryFile.is_open()) {
                    for (auto& r: results) {
                        summaryFile << std::setw(maxNameLength) << r.userFriendlyName << ": "
                                    << std::setw(alignSamples) << r.iterationsCount << "  "
                                    << std::setw(alignExpect) << r.averageTimeDropFirst << "  "
                                    << std::setw(alignSd) << r.standardDeviationMs << std::endl;
                    }
                }
            }

            log << "=-=-=-=-=-= FINISH: " << benchmarkName << " =-=-=-=-=-=" << std::endl;
        }
    };

}

#endif //CUBOOL_BENCHMARK_BASE_HPP