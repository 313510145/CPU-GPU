#include "CPU_manager.h"

#include <stdexcept>
#include <chrono>
#include <omp.h>

void CPU_manager::set_time_stamp(const std::string& key) {
    time_stamp[key] = omp_get_wtime();
}

double CPU_manager::get_time_duration(const std::string& key_start, const std::string& key_end) {
    if (time_stamp.find(key_start) == time_stamp.end() || time_stamp.find(key_end) == time_stamp.end()) {
        throw std::runtime_error("Time stamp not found for keys: " + key_start + " or " + key_end);
    }
    return time_stamp[key_end] - time_stamp[key_start];
}

CPU_manager::CPU_manager() {}

CPU_manager::~CPU_manager() {}
