//
// Created by 松野智紀 on 2017/10/18.
//
#include <stdexcept>
#include <string>
#include <boost/format.hpp>
#include <iostream>
#include <map>
#include <unordered_map>

class KMeans{
    unsigned _k;
    std::map<unsigned, unsigned> _len_cntr;
    std::map<unsigned, unsigned> _lengths = _len_cntr;

    KMeans(unsigned k, unsigned len_cntr){
        if(len_cntr < k){
            boost::format fmt = boost::format("Trying to sort %d data into %d buckets") % (len_cntr, k);
            throw std::invalid_argument(fmt.str());
        }
    }


};