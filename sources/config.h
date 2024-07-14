#pragma once

#include "program_options.h"

namespace po = ProgramOptions;

class AppCfgs
{
public:
    AppCfgs();
    virtual ~AppCfgs(){};

private:
    int m_data_parsing_flag;
    std::string m_regx;
    std::string m_regy;

public:
    int get_parsing_flag() { return m_data_parsing_flag; }
    std::string get_x_axis() { return m_regx; }
    std::string get_y_axis() { return m_regy; }

public:
    bool parseCfg(int argc, char *argv[]);
    std::vector<int> parse_string2int(const std::string &str);
};