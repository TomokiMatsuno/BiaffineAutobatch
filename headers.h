//
// Created by 松野智紀 on 2017/10/17.
//

#ifndef BIAFFINE_HEADERS_H_H
#define BIAFFINE_HEADERS_H_H

#endif //BIAFFINE_HEADERS_H_H

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string.h>
#include <algorithm>
#include <regex>

#include "dynet/nodes.h"
#include "dynet/dynet.h"
#include "dynet/training.h"
#include "dynet/timing.h"
#include "dynet/rnn.h"
#include "dynet/gru.h"
#include "dynet/lstm.h"
#include "dynet/dict.h"
#include "dynet/expr.h"
#include "dynet/io.h"
//#include "getpid.h"

using namespace std;
using namespace dynet;


//#include "../BiAffine/read-file/read_ud_data.h"
//#include "../BiAffine/read-file/read_ud_data.cpp"

#include "file-paths.h"
#include "config.h"
#include "parameters.h"
#include "network.h"
#include "lib/utils.h"
#include "lib/utils.cpp"
#include "data.cpp"
#include "network.cpp"
