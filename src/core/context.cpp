#include <vector>
#include <unordered_map>
#include <string>
#include <iostream>
#include "context.h"
namespace ETL{

    void Context::set_mode_from_subscript(const std::string &subscript)
    {
        auto it = subscript_to_mode.find(subscript);
        if (it == subscript_to_mode.end())
        {
            subscript_to_mode[subscript] = next_mode;
            mode_to_subscript[next_mode] = subscript;
            next_mode++;
        }
    }

    void Context::set_extent_from_mode(int mode, int64_t extent)
    {
        auto it = mode_to_extent.find(mode);
        if (it != mode_to_extent.end() && it->second != extent)
        {
            throw std::runtime_error("Mode " + std::to_string(mode) + " already mapped to a different extent");
        }
        mode_to_extent[mode] = extent;
    }

    void Context::set_extent_from_subscript(const std::string &subscript, int64_t extent)
    {
        int mode = subscript2mode(subscript); // throws if not found
        set_extent_from_mode(mode, extent);
    }

    int64_t Context::mode2extent(int mode) const
    {
        return mode_to_extent.at(mode); // throws std::out_of_range if not found
    }

    std::vector<int64_t> Context::mode2extent_v(const std::vector<int> &modes) const
    {
        std::vector<int64_t> result;
        result.reserve(modes.size());
        for (int mode : modes)
        {
            result.push_back(mode2extent(mode));
        }
        return result;
    }

    std::string Context::mode2subscript(const int mode) const
    {
        return mode_to_subscript.at(mode);
    }
    SubscriptsType Context::mode2subscript_v(const Modes modes) const
    {
        SubscriptsType ret;
        for (const int &mode : modes)
        {
            ret.push_back(mode2subscript(mode));
        }
        return ret;
    }

    int Context::subscript2mode(const std::string &subscript) const
    {
        auto it = subscript_to_mode.find(subscript);
        if (it == subscript_to_mode.end())
        {
            throw std::runtime_error("Subscript '" + subscript + "' not found");
        }
        return it->second;
    }

    std::vector<int> Context::subscript2mode_v(const std::vector<std::string> &subscripts) const
    {
        std::vector<int> result;
        result.reserve(subscripts.size());
        for (const auto &sub : subscripts)
        {
            result.push_back(subscript2mode(sub));
        }
        return result;
    }

    int64_t Context::subscript2extent(const std::string &subscript) const
    {
        return mode2extent(subscript2mode(subscript));
    }
    std::vector<int64_t> Context::subscript2extent_v(const std::vector<std::string> &subscripts) const
    {
        return mode2extent_v(subscript2mode_v(subscripts));
    }
};
