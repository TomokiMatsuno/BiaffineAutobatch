//
// Created by 松野智紀 on 2017/10/18.
//

#ifndef BIAFFINE_2ND_UTILS_H
#define BIAFFINE_2ND_UTILS_H
//#include "../headers.h"

vector<unsigned> bucket_sizes(unsigned n_bkts, map<unsigned, unsigned> len_counter){

    vector<unsigned> vec;
    for(auto it = len_counter.begin(); it != len_counter.end(); it++){
        vector<unsigned> tmp(it->second, it->first);
        vec.insert(vec.end(), tmp.begin(), tmp.end());
    }

    vector<unsigned> ret(0);
    if(n_bkts <= 1){auto it = len_counter.end(); it--; ret.push_back(it->first); return ret;}

    unsigned bucket_size = ((vec.size()/(n_bkts-1)));
    for(int i = 0; i < vec.size() - bucket_size + 1; i += bucket_size){
//        vector<unsigned> tmp(bucket_size, vec[i + bucket_size - 1]);
//        ret.insert(ret.end(), tmp.begin(), tmp.end());
        ret.push_back(vec[i + bucket_size - 1]);
    }
//    vector<unsigned> tmp(vec.size()-ret.size(), vec[vec.size() - 1]);
//    ret.insert(ret.end(), tmp.begin(), tmp.end());
    ret.push_back(vec[vec.size() - 1]);

    return ret;
}

vector<vector<vector<unsigned>>> padding(vector<vector<vector<unsigned>>>& vec, unsigned pad_len, unsigned pad = 0){
    vector<vector<vector<unsigned>>> pad_vec(pad_len - vec.size());
    vec.insert(vec.end(), pad_vec.begin(), pad_vec.end());
    return vec;
}


#endif //BIAFFINE_2ND_UTILS_H

