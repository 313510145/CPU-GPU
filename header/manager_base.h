#ifndef MANAGER_BASE_H
#define MANAGER_BASE_H

#include <string>

class manager_base {
    public:
        virtual void set_time_stamp(const std::string& key) = 0;
        virtual double get_time_duration(const std::string& key_start, const std::string& key_end) = 0;
        manager_base();
        virtual ~manager_base();
};

#endif  // MANAGER_BASE_H
