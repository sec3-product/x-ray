#ifndef CONFLIB_CPP_CONFLIB_H
#define CONFLIB_CPP_CONFLIB_H

#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <libgen.h>
#include <set>
#include <sys/stat.h>

#include <jsoncons/json.hpp>
#include <jsoncons_ext/jsonpath/json_query.hpp>

namespace conflib {

extern std::vector<jsoncons::json> confs_;
extern std::set<std::string> merge_keys_;
extern void MergeJson_(jsoncons::json &jsink, jsoncons::json &jnew);

std::vector<std::string>
Initialize(const std::map<std::string, std::string> &short_args_map,
           bool use_env_cmdline_opts, int argc = 0, char *argv[] = nullptr);

void Get(const std::string &key, std::vector<std::string> &rs);

template <typename T> T Get(const std::string &key, T defaultValue) {
  std::string first_stage;
  auto loc = key.find('.');
  if (loc == std::string::npos) {
    first_stage = key;
  } else {
    first_stage = key.substr(0, loc);
  }

  if (merge_keys_.find(first_stage) == merge_keys_.end()) {
    std::string jpath = "$." + key;

    for (auto itr = confs_.crbegin(); itr != confs_.crend(); ++itr) {
      jsoncons::json r = jsoncons::jsonpath::json_query(*itr, jpath);
      if (!r.empty()) {
        return r[0].as<T>();
      }
    }
  } else {
    // this a merge key.
    int i = confs_.size() - 1;
    for (; i >= 0; --i) {
      if (confs_[i].contains(first_stage)) {
        break;
      }
    }
    if (i < 0) {
      return defaultValue;
    }

    std::string jpath = "$." + first_stage;
    jsoncons::json j = jsoncons::jsonpath::json_query(confs_[i], jpath)[0];
    i++;
    for (; i < confs_.size(); i++) {
      jpath = "$.@" + first_stage;
      auto r = jsoncons::jsonpath::json_query(confs_[i], jpath);
      if (!r.empty())
        MergeJson_(j, r[0]);
    }

    jsoncons::json j1;
    j1.insert_or_assign(first_stage, j);

    jpath = "$." + key;
    jsoncons::json r = jsoncons::jsonpath::json_query(j1, jpath);
    if (!r.empty()) {
      return r[0].as<T>();
    }
  }

  return defaultValue;
}

} // namespace conflib

#endif // CONFLIB_CPP_CONFLIB_H
