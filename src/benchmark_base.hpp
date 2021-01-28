//
// Created by Egor.Orachev on 26.01.2021.
//

#ifndef CUBOOL_BENCHMARK_BASE_HPP
#define CUBOOL_BENCHMARK_BASE_HPP

#include <chrono>
#include <string>
#include <functional>
#include <iostream>
#include <vector>
#include <cassert>
#include <fstream>

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
            size_t iterationsCount = 0;
            double totalTime = 0.0;
            double averageTime = 0.0;
            double minIterationTime = 0.0;
            double maxIterationTime = 0.0;
            std::vector<double> samplesMs;
        };

        std::vector<PerExperiment> results;

        //////////////////////////////////////////////////
        // Override functions below for your benchmark

        virtual void setupBenchmark() = 0;
        virtual void tearDownBenchmark() = 0;

        virtual void setupExperiment(size_t experimentIdx, size_t& iterationsCount) = 0;
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

            log.open(benchmarkName + ".txt", std::ios_base::out);
            if (!log.is_open()) {
                std::cerr << "Failed to open log file" << std::endl;
                return;
            }

            log << "=-=-=-=-=-= RUN: " << benchmarkName << " =-=-=-=-=-=" << std::endl << std::endl;

            setupBenchmark();

            for (auto experimentIdx = 0; experimentIdx < experimentsCount; experimentIdx++) {
                size_t iterationsCount;

                log << "> Begin experiment: " << experimentIdx << std::endl;

                setupExperiment(experimentIdx, iterationsCount);

                PerExperiment perExperiment;
                perExperiment.iterationsCount = iterationsCount;
                perExperiment.minIterationTime = std::numeric_limits<double>::max();
                perExperiment.samplesMs.reserve(iterationsCount);

                TimeQuery timeQuery;

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
                }

                perExperiment.totalTime = timeQuery.getTotalTimeMS();
                perExperiment.averageTime = timeQuery.getAverageTimeMs();

                tearDownExperiment(experimentIdx);

                log << "> End experiment: " << experimentIdx << std::endl
                          << "> Stats: " << std::endl
                          << ">  iterations = " << perExperiment.iterationsCount << std::endl
                          << ">  total      = " << perExperiment.totalTime << " ms" << std::endl
                          << ">  average    = " << perExperiment.averageTime << " ms" << std::endl
                          << ">  min        = " << perExperiment.minIterationTime << " ms" << std::endl
                          << ">  max        = " << perExperiment.maxIterationTime << " ms" << std::endl;

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

            log << "=-=-=-=-=-= FINISH: " << benchmarkName << " =-=-=-=-=-=" << std::endl;
        }

    };

}

#endif //CUBOOL_BENCHMARK_BASE_HPP