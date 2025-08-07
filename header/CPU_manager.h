#ifndef CPU_MANAGER_H
#define CPU_MANAGER_H

#include "manager_base.h"

#include <unordered_map>

class CPU_manager: public manager_base {
    public:
        void set_time_stamp(const std::string& key) override;
        void set_time_stamp_openmp(const std::string& key);
        double get_time_duration(const std::string& key_start, const std::string& key_end) override;
        CPU_manager();
        ~CPU_manager() override;
    private:
        std::unordered_map<std::string, double> time_stamp;
};

#endif  // CPU_MANAGER_H
