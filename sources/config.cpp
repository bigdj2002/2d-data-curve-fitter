#include "config.h"

AppCfgs::AppCfgs()
{
}

bool AppCfgs::parseCfg(int argc, char *argv[])
{
  bool do_help = false;
  std::string outputColourSpaceConvert;
  int warnUnknowParameter = 0;
  po::Options opts;

  // clang-format off
  opts.addOptions()
  ("help", do_help, false, "this help text\n")
  ("-data_parsing", m_data_parsing_flag, 0, "\n")
  ("-reg_x", m_regx, std::string(""), "\n")
  ("-reg_y", m_regy, std::string(""), "\n");

  po::setDefaults(opts);
  po::ErrorReporter err;
  const std::list<const char *> &argv_unhandled = po::scanArgv(opts, argc, (const char **)argv, err);

  for (std::list<const char *>::const_iterator it = argv_unhandled.begin(); it != argv_unhandled.end(); it++)
  {
    msg(ERROR, "Unhandled argument ignored: `%s'\n", *it);
  }

  if (argc == 1 || do_help)
  {
    po::doHelp(std::cout, opts);
    return false;
  }

  if (err.is_errored)
  {
    if (!warnUnknowParameter)
    {
      return false;
    }
  }

  return true;
}

std::vector<int> AppCfgs::parse_string2int(const std::string& str) 
{
    std::vector<int> result;
    std::istringstream stream(str);
    std::string item;

    try 
    {
      while (std::getline(stream, item, ',')) 
      {
        result.push_back(std::stoi(item));
      }
    } 
    catch (const std::invalid_argument& e) 
    {
      throw std::invalid_argument("List items must be integers");
    } 
    catch (const std::out_of_range& e) 
    {
      throw std::out_of_range("List item is out of integer range");
    }

    return result;
}