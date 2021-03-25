//
// Created by Egor.Orachev on 25.03.2021.
//

#ifndef SPBENCH_PROFILE_MEM_HPP
#define SPBENCH_PROFILE_MEM_HPP

#include <unistd.h>
#include <ios>
#include <iostream>
#include <fstream>
#include <string>

// Original: https://stackoverflow.com/questions/669438/how-to-get-memory-usage-at-runtime-using-c

// On Linux, I've never found an ioctl() solution. For our applications,
// we coded a general utility routine based on reading files in /proc/pid.
// There are a number of these files which give differing results.
// Here's the one we settled on (the question was tagged C++,
// and we handled I/O using C++ constructs, but it should be easily
// adaptable to C i/o routines if you need to):

//////////////////////////////////////////////////////////////////////////////
//
// process_mem_usage(double &, double &) - takes two doubles by reference,
// attempts to read the system-dependent data for a process' virtual memory
// size and resident set size, and return the results in KB.
//
// On failure, returns 0.0, 0.0

void process_mem_usage(double& vm_usage, double& resident_set)
{
    using std::ios_base;
    using std::ifstream;
    using std::string;

    vm_usage     = 0.0;
    resident_set = 0.0;

    // 'file' stat seems to give the most reliable results
    //
    ifstream stat_stream("/proc/self/stat",ios_base::in);

    // dummy vars for leading entries in stat that we don't care about
    //
    string pid, comm, state, ppid, pgrp, session, tty_nr;
    string tpgid, flags, minflt, cminflt, majflt, cmajflt;
    string utime, stime, cutime, cstime, priority, nice;
    string O, itrealvalue, starttime;

    // the two fields we want
    //
    unsigned long vsize;
    long rss;

    stat_stream >> pid >> comm >> state >> ppid >> pgrp >> session >> tty_nr
                >> tpgid >> flags >> minflt >> cminflt >> majflt >> cmajflt
                >> utime >> stime >> cutime >> cstime >> priority >> nice
                >> O >> itrealvalue >> starttime >> vsize >> rss; // don't care about the rest

    stat_stream.close();

    long page_size_kb = sysconf(_SC_PAGE_SIZE) / 1024; // in case x86-64 is configured to use 2MB pages
    vm_usage     = (double) vsize / 1024.0;
    resident_set = (double) rss * (double) page_size_kb;
}

void output_mem_profile(const std::string& filename, const std::string& label) {
    std::ofstream s;
    s.open(filename, std::ios::app);

    if (!s.is_open()) {
        std::cerr << "Failed to open file " << filename;
        return;
    }

    double vm_usage, resident_set;
    process_mem_usage(vm_usage, resident_set);

    s << "Label: " << label << std::endl
      << " - vm_usage: " << vm_usage << " KBs ( " <<  vm_usage / 1000.0 << " MBs)" << std::endl
      << " - resident_set: " << resident_set << " KBs ( " << resident_set / 1000.0 << " MBs)" << std::endl;

    s.close();
}

#endif //SPBENCH_PROFILE_MEM_HPP
