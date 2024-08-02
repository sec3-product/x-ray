//
// Created by peiming on 5/18/20.
//

#include "conflib/conflib.h"

#include <unistd.h>
#include <vector>
#include <set>

namespace conflib {

static const std::string VERSION_ = "0.0.1";


#ifdef _WIN32
static const std::string OS_PATHSEP_("\\");
#else
static const std::string OS_PATHSEP_("/");
#endif


static const std::string DEFAULT_CONFIGURATION_NAME_ = "smalltalk.json";
static const std::string HOME_CONFIGURATION_PATH_ = "~/.smalltalk.json";
static const std::string ENV_INSTALLATION_DIRECTORY_ = "SMALLTALK_HOME";

//
//struct ConfLibException : public std::exception {
//public:
//    static const int UNKNOWN = 1;
//    static const int INVALID_PARAMETER = 2;
//    static const int INVALID_DATA_TYPE = 3;
//
//public:
//    ConfLibException(int code) : code_(code) {}
//    ~ConfLibException() = default;
//
//    int code() {
//        return code_;
//    }
//
//private:
//    int code_;
//};


std::vector<jsoncons::json> confs_;
std::set<std::string> merge_keys_;


/**
 * Merges jnew into jsink
 */
void MergeJson_(jsoncons::json& jsink, jsoncons::json& jnew) {
    for (const auto& member : jnew.object_range()) {
        jsink.insert_or_assign(member.key(), member.value());
    }
}


/**
 * @param s is a key-value string like "-abc=def".
 * @return the value
 */
static inline std::string GetConfValue_(const char* s) {
    std::string str(s);
    size_t pos = str.find_first_of('=');
    if (pos == std::string::npos) {
        printf("conflib fatal error: invalid parameter\n"); \
        std::abort();
    }

    return str.substr(pos+1);
}


static inline bool FileExists_(const std::string& path) {
    struct stat st;
    return stat(path.c_str(), &st) == 0;
}


static inline bool IsInteger_(const std::string& s) {
    char *p = nullptr;
    strtol(s.c_str(), &p, 10);
    return (*p == 0);
}


static inline bool IsDouble_(const std::string& s) {
    char *p = nullptr;
    strtof(s.c_str(), &p);
    return (*p == 0);
}


static inline void Split_(std::vector<std::string>& stages, std::string s) {
    size_t pos_of_first_char = 0;
    size_t pos = s.find_first_of('.');
    while (pos != std::string::npos) {
        stages.push_back(s.substr(pos_of_first_char, pos-pos_of_first_char));
        pos_of_first_char = pos + 1;
        pos = s.find_first_of('.', pos_of_first_char);
    }

    stages.push_back(s.substr(pos_of_first_char));
}

static inline std::string BuildPath_(const std::vector<std::string>& stages, int last_stage) {
    std::string s;
    for (int i = 0; i <= last_stage; i++) {
        if (!s.empty()) {
            s += '.' + stages[i];
        }
        else {
            s += stages[i];
        }
    }

    return "$." + s;
}


static inline void AssignValue_(jsoncons::json& j, const std::string& key, const std::string& value) {
    if (value == "true") {
        j[key] = true;
    }
    else if (value == "false") {
        j[key] = false;
    }
    else if (IsDouble_(value)) {
        j[key] = atof(value.c_str());
    }
    else if (IsInteger_(value)) {
        j[key] = atol(value.c_str());
    }
    else {
        j[key] = value;
    }
}


static inline void AssignValues_(jsoncons::json& j, const std::string& key, const std::vector<std::string>& value) {
    j[key] = value;
}


static inline void ReplaceValue_(jsoncons::json& j, const std::string& jpath, const std::string& value) {
    if (value == "true") {
        jsoncons::jsonpath::json_replace(j, jpath, true);
    }
    else if (value == "false") {
        jsoncons::jsonpath::json_replace(j, jpath, false);
    }
    else if (IsDouble_(value)) {
        jsoncons::jsonpath::json_replace(j, jpath, atof(value.c_str()));
    }
    else if (IsInteger_(value)) {
        jsoncons::jsonpath::json_replace(j, jpath, atol(value.c_str()));
    }
    else {
        jsoncons::jsonpath::json_replace(j, jpath, value);
    }
}

/**
 * 's' has the format like '-abc.def=ghi'. This method needs to
 * insert the value of 'ghi' into the json object 'j' at the location
 *
 * {
 *   "abc": {
 *     "def": "ghi"
 *   }
 * }
 *
 */
static inline void InsertConfiguration_(jsoncons::json& j, const std::string& s) {
    size_t pos = s.find_first_of('=');
    bool is_array = (s[0] == '@');
    std::string key = s.substr(1, pos-1);
    std::string value = s.substr(pos+1);
    std::vector<std::string> values;
    if (is_array) {
        int b = 0;
        for (int e = 0; e < value.length(); e++) {
            char ch = value[e];
            if (ch == ',') {
                std::string subs = value.substr(b, e-b);
                values.push_back(subs);
                b = e + 1;
            }
        }

        values.push_back(value.substr(b));
    }

    std::vector<std::string> stages;
    Split_(stages, key);

    jsoncons::json jtmp;
    for (int i = stages.size()-1; i>=0; i--) {
        std::string jpath = BuildPath_(stages, i);
        auto r = jsoncons::jsonpath::json_query(j, jpath);
        if (!r.empty()) {
            if (jtmp.empty()) {
                if (!is_array)
                    jsoncons::jsonpath::json_replace(j, jpath, value);
                else
                    jsoncons::jsonpath::json_replace(j, jpath, values);
            }
            else {
                if (r[0].is_object()) {
                    jtmp.merge(std::move(r[0]));
                }
                jsoncons::jsonpath::json_replace(j, jpath, jtmp);
            }

            break;
        }
        else { // r.empty()
            if (stages.size() == 1) {
                if (!is_array)
                    AssignValue_(j, stages[i], value);
                else
                    AssignValues_(j, stages[i], values);

                break;
            }

            if (jtmp.empty()) {
                if (!is_array)
                    AssignValue_(jtmp, stages[i], value);
                else
                    AssignValues_(jtmp, stages[i], values);
            }
            else {
                if (i == 0) {
                    j[stages[i]] = jtmp;
                }
                else {
                    jsoncons::json jtmp2;
                    jtmp2[stages[i]] = jtmp;
                    jtmp = jtmp2;
                }
            }
        }
    }
}



static inline void LoadConfFile_(const std::string &conf_path, bool searchForInclude) {
    jsoncons::json j{};

    if (!conf_path.empty() && FileExists_(conf_path)) {
        std::ifstream ins(conf_path);
        j = jsoncons::json::parse(ins);
    } else {
        j = jsoncons::json::parse("{}");
    }
    if (j.size() == 0) {
        return;
    }
    confs_.emplace_back(j);

    // check if there are top-level '@' keys in j, and add them into
    // merge_keys_ if there are.
    for (const auto& member : j.object_range()) {
        auto key = member.key();
        if (key[0] == '@') {
            merge_keys_.insert(key.substr(1));
        }
    }

    if (searchForInclude) {
        // resolve "include"
        std::string jpath = "$.include";

        jsoncons::json r = jsoncons::jsonpath::json_query(j, jpath);
        if (!r.empty()) {
            // get the path prefix
            char fullpath[PATH_MAX];
            realpath(conf_path.c_str(), fullpath);
            char *dirp = dirname(fullpath);

            auto includes = r[0].as<std::vector<std::string>>();
            for (auto const &str : includes) {
                auto includeFilePath = std::string(dirp) + "/" + str;
                LoadConfFile_(includeFilePath, false);
            }
        }
    }
}


static std::vector<std::string> ParseCmdlineArgs_(int argc,
                                                  char* argv[],
                                                  const std::map<std::string, std::string>& shortArgsMap) {
    std::vector<std::string> remaining;
    jsoncons::json j;

    // convert all cmdline parameters into a json
    for (int i = 0; i < argc; i++) {
        if (argv[i][0] != '-' && argv[i][0] != '@') {
            // we already consumed all arguments. return the remaining
            // to the caller so that it deals with them by itself
            for (int j = i; j < argc; j++) {
                remaining.emplace_back(argv[j]);
            }
            break;
        }

        if (strstr(argv[i], "-conf=") == argv[i] || argv[i][0] == '=')
            continue;

        std::string arg(argv[i]);

        // the arg is single-char such as "-o". we need
        // to check if it's a short alias of a formal
        // argument such as "-report.outputDir" by looking
        // up "shortArgsMap"
        if (arg.size() == 2) {
            std::string tmp = arg.substr(1);
            if (shortArgsMap.find(tmp) != shortArgsMap.end()) {
                auto longArgs = shortArgsMap.find(tmp)->second;
                if (i == argc-1 || argv[i+1][0] == '-') {
                    arg = "-" + longArgs + "=true";
                }
                else {
                    arg = "-" + longArgs + "=" + std::string(argv[i+1]);
                    i++;
                }
            }
        }

        size_t pos = arg.find_first_of('=');
        if (pos == std::string::npos) {
            // sth like -racedetector.enableFunction
            arg += "=true";
        }

        InsertConfiguration_(j, arg);
    }
    confs_.emplace_back(j);

    return remaining;
}



/**
 * Assumes 'cwd' is the project directory.
 *
 * cmdline < custom < project < home < default
 *
 * Any trailing arguments that can't be consumed by conflib will be
 * returned as a vector of string. For example,
 *
 * command -report.color=blue -report.enableCompress abc.c a.out
 *
 * "abc.c" and "a.out" can't be consumed and will be returned.
 */
std::vector<std::string> Initialize(const std::map<std::string, std::string>& short_args_map,
                                    bool use_env_cmdline_opts,
                                    int argc,
                                    char *argv[]) {
    std::string conf_path;

    // load the default configuration file
    char* home_dir = std::getenv(ENV_INSTALLATION_DIRECTORY_.c_str());
    if (home_dir != nullptr) {
        conf_path = std::string(home_dir) + OS_PATHSEP_ + "conf" + OS_PATHSEP_ + DEFAULT_CONFIGURATION_NAME_;
    }
    else {
        // we assume the base dir of the application is the installation dir
        // todo
    }
    LoadConfFile_(conf_path, true);

    // load the home configuration file
    LoadConfFile_(HOME_CONFIGURATION_PATH_, true);

    // load the project configuration file
    LoadConfFile_("./smalltalk.json", true);

    // load the custom configuration file
    conf_path = "";
    for ( int i = 1; i < argc; i++) {
        if (strstr(argv[i], "-conf=") == argv[i]) {
            conf_path = GetConfValue_(argv[i]);
            break;
        }
    }
    if (conf_path.length() > 0) {
        LoadConfFile_(conf_path, true);
    }

    // prepare cmdline opts for ParseCmdlineArgs_
    char** effective_argv = nullptr;
    int effective_argc = 0;

    if (use_env_cmdline_opts) {
        char* tmp = std::getenv("SMALLTALK_CMDLINE_OPTS");
        if (tmp != nullptr) {
            jsoncons::json j = jsoncons::json::parse(tmp);
            const jsoncons::json &jopts = j["Opts"];
            effective_argc = jopts.size();
            if (effective_argc > 0) {
                effective_argv = new char*[effective_argc];
                int i = 0;
                for (const auto& item : jopts.array_range()) {
                    size_t len = item.as<std::string>().length();
                    effective_argv[i] = new char[len + 1];
                    memcpy(effective_argv[i], item.as<std::string>().c_str(), len);
                    effective_argv[i][len] = 0;

                    if (strstr(effective_argv[i], "-conf=") == effective_argv[i]) {
                        conf_path = GetConfValue_(effective_argv[i]);
                        LoadConfFile_(conf_path, true);
                    }

                    i++;
                }
            }
        }
    }
    else {
        effective_argc = argc - 1;
        if (effective_argc > 0) {
            effective_argv = new char*[effective_argc];
            for (int i = 0; i < effective_argc; i++) {
                effective_argv[i] = argv[i+1];
            }
        }
    }
    return ParseCmdlineArgs_(effective_argc, effective_argv, short_args_map);
}


static inline std::vector<std::string> ParseArray_(jsoncons::json& j) {
    std::vector<std::string> r;
    for (const auto& item : j.array_range())
    {
        r.push_back(item.as<std::string>());
    }

    return r;
}


// inline avoid multiple definition errors when conflib.h is included in multiple files
void Get(const std::string& key, std::vector<std::string>& rs) {
    std::string jpath = "$." + key;

    for (auto itr = confs_.crbegin(); itr != confs_.crend(); ++itr) {
        jsoncons::json r = jsoncons::jsonpath::json_query(*itr, jpath);
        if (!r.empty()) {
            for (const auto& item : r[0].array_range()) {
                rs.push_back(item.as<std::string>());
            }
            break;
        }
    }
}


} // namespace conflib
