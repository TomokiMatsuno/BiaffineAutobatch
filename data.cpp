//
// Created by 松野智紀 on 2017/10/28.
//
//#include "headers.h"
//#include "lib/utils.h"
#include <sstream>
#include <string.h>
using namespace std;
bool pret_embs;

class UtilDict{

//protected:
public:
    unordered_map<unsigned, string> i2x;
    unordered_map<string, unsigned> x2i;
    unsigned dict_idx;

public:

    UtilDict(unsigned dict_idx){
        vector<string> initial_entries = {"ROOT", "PAD", "UNK"};
        for(int i = 0; i < initial_entries.size(); i++) {
            this->x2i[initial_entries[i]] = i;
            this->i2x[x2i.size()] = initial_entries[i];
        }
        this->dict_idx = dict_idx;
    }


    bool has_entry(string str){
        return this->x2i.find(str) != this->x2i.end();
    }

    virtual unsigned get_idx(string){}
    virtual vector<unsigned> get_idx(vector<string>){}

    string get_entry(unsigned idx){
        return this->i2x.at(idx);
    }

    vector<string> get_entry(vector<unsigned> indices){
        vector<string> ret;
        for(unsigned idx : indices){
            ret.push_back(this->i2x.at(idx));
        }
        return ret;
    }

    unsigned idx(){
        return this->dict_idx;
    }

//    void url2token(vector<string> patterns, vector<string> pidx2token){
//        for(int i = 0; i < patterns.size(); i++) {
//            regex re(pattern);
//            if(regex_match(ent, re)){
//                this->x2i[pidx2token[i]] = this->x2i.size();
//                this->i2x[this->x2i[ent]] = ent;
//                return;
//            }
//        }
//    }

    void add_entry(string ent) {
        if (has_entry(ent)) {
            return;
        } else {
            this->x2i[ent] = this->x2i.size();
//            cout << x2i[ent] << "\t" << ent << "\t" << x2i.size() << endl;
            this->i2x[this->x2i[ent]] = ent;
            return;
        }
    }

    unsigned vocab_size() {
        return x2i.size();
    }

    void showElemsInDict(){
        for(auto it = this->x2i.begin(); it != this->x2i.end(); it++){
            cout << it->first << "\t" << it->second << endl;
        }
    }


//    virtual ~UtilDict(){};
};

class ordinalUtilDict : public UtilDict{
public:

    unsigned get_idx(string str){
        return this->x2i.at(str);
    }

    vector<unsigned> get_idx(vector<string> strs){
        vector<unsigned> ret;
        for(auto str : strs){
            ret.push_back(this->x2i.at(str));
        }
        return ret;
    }

    ordinalUtilDict(unsigned dict_idx) : UtilDict(dict_idx){}
//    virtual ~ordinalUtilDict();

};

class stoiUtilDict : public UtilDict{
public:
    unsigned get_idx(const string& str){
        return stoi(str);
    }

    vector<unsigned> get_idx(const vector<string>& strs){
        vector<unsigned> ret;
        for(auto& s : strs) ret.push_back(stoi(s));
        return ret;
    }

    stoiUtilDict(unsigned dict_idx) : UtilDict(dict_idx){}
//    virtual ~stoiUtilDict();

};

class tupleMaker{
    vector<UtilDict> dicts;

public:
    tupleMaker(vector<UtilDict>& dicts){
        this->dicts = dicts;
    }

    vector<unsigned> make_tuple(vector<string>& tokens){
        vector<unsigned> ret;
        vector<unsigned> dict_indices;
        for(auto& d : dicts) dict_indices.push_back(d.idx()); //indices of elements to be picked up in input lines
        unsigned j = 0; //index in vector dicts
        for(unsigned i : dict_indices) {
            ret.push_back(this->dicts[j].x2i[tokens[i]]);
            j++;
        }
        return ret;
    }
};

class Vocab{
    const unsigned PAD = 0;
    const unsigned ROOT = 1;
    const unsigned UNK = 2;

    unsigned _words_in_train;
    vector<UtilDict> dicts;
    char delimiter;

public:
    UtilDict wd = 1;
    UtilDict td = 3;
    UtilDict rd = 7;

    void add_entry(vector<string> tokens){
//        wd.add_entry(tokens[wd.idx()]);
//        td.add_entry(tokens[td.idx()]);
//        rd.add_entry(tokens[rd.idx()]);
        for(auto& d : this->dicts){
            d.add_entry(tokens[d.idx()]);
        }
    }

    vector<UtilDict>& get_dicts(){
        return this->dicts;
    }

    char get_delimiter(){
        return this->delimiter;
    }


public:
    Vocab(string file_name, char delimiter, vector<UtilDict> dicts){
        this->dicts = dicts;
        this->delimiter = delimiter;
        ifstream ifs;
        ifs.open(file_name);
        string line_buffer;
        if(!ifs){
            cout << "failed reading file" << endl;
        }
        else{
            //add_entry(root_line);
            while(!ifs.eof()){
                //read lines one by one
                getline(ifs, line_buffer);
                istringstream line_separator(line_buffer);
                string token_buffer;
                vector<string> tokens;
                //split lines into tokens
                while(getline(line_separator, token_buffer, delimiter)){
                    tokens.push_back(token_buffer);
                };
                //skip illegal sentences
                if(tokens.empty() || tokens.size() != 10) continue;

                //add elements into dictionaries
                add_entry(tokens);
            }
        }
        return;
    }

    Expression get_word_embs(ComputationGraph& cg, unsigned word_dims){
        if(pret_embs) return random_normal(cg, {wd.vocab_size(), word_dims});
        return dynet::zeroes(cg, {wd.vocab_size(), word_dims});
    }

    Expression get_tag_embs(ComputationGraph& cg, unsigned tag_dims){
        return random_normal(cg, {td.vocab_size(), tag_dims});
    }

    unsigned words_in_train(){
        return this->_words_in_train;
    }
};

class DataLoader{
    char delimiter;
    vector<UtilDict> dicts;
    unsigned n_bkts;

protected:
    vector<unsigned> _bucket_sizes;
    vector<vector<vector<vector<unsigned>>>> _buckets;
    vector<vector<vector<unsigned>>> _sents;
    vector<pair<unsigned, unsigned>> _record;

public:
    DataLoader(string input_file, unsigned n_bkts, Vocab& vocab){
        tupleMaker tm(vocab.get_dicts());
        this->delimiter = vocab.get_delimiter();
        this->n_bkts = n_bkts;
        vector<vector<vector<vector<unsigned>>>> buckets(n_bkts);
        this->_buckets = buckets;

        vector<vector<vector<unsigned>>> sents;
        vector<vector<unsigned>> sent;
        vocab.add_entry(root_line);
        vector<unsigned> word = tm.make_tuple(root_line);   //begin the first sentence with a root token

        sent.push_back(word);

        ifstream ifs(input_file);
        string line_buffer;
        while(getline(ifs, line_buffer)){                   //read lines one by one
            string separator_buffer;
            vector<string> tokens;
            stringstream ss(line_buffer);
            while(getline(ss, separator_buffer, delimiter)){//separate lines into tokens (e.g. FORM, POS, HEAD, REL)
                tokens.push_back(separator_buffer);
            }
            if(line_buffer.empty()){
                sents.push_back(sent);                      //if it reads a empty line, add complete sentence to sents
                word = tm.make_tuple(root_line);            //and begin reading new sentence with root token
                sent.clear();
                sent.push_back(word);
                continue;
            }
            if(tokens.size() != 10) continue;               //skip illegal lines
            vocab.add_entry(tokens);                      //update dictionaries
            word.clear();
            word = tm.make_tuple(tokens);

            sent.push_back(word);                           //add a word to a incompete sentence
        }

        map<unsigned, unsigned> len_counter;
        for(auto& sent : sents) len_counter[sent.size()] += 1;    //count lengths of sentences
        this->_bucket_sizes = bucket_sizes(n_bkts, len_counter);

        map<unsigned, unsigned> len2bkt;                    //return the idx of the corresponding bucket

        int prev_size = -1;
        for(int i = 0; i < this->_bucket_sizes.size(); i++){
            for(int j = prev_size+1; j < _bucket_sizes[i]+1; j++){
                len2bkt[j] = i;
            }
            prev_size = _bucket_sizes[i];
        }

        //Todo: bucketing is not working well. needs to be fixed.

        for(auto& sent : sents){
            unsigned bkt_idx = len2bkt[sent.size()];
            this->_buckets[bkt_idx].push_back(sent);
            unsigned idx = this->_buckets[bkt_idx].size() - 1;
            pair<unsigned, unsigned> p(bkt_idx, idx);
            this->_record.push_back(p);
        }

        this->_sents = sents;

    }

    vector<vector<vector<unsigned>>> get_sents(){
        return this->_sents;
    }

    vector<unsigned> idx_sequence(){
        map<pair<unsigned, unsigned>, unsigned> m;
        vector<unsigned> ret;
        for(int i = 0; i < this->_record.size(); i++){
            m[this->_record[i]] = i;
        }
        for(auto itr = m.begin(); itr != m.end(); ++itr){
            ret.push_back((*itr).second);
        }
        return ret;
    }

    struct range_func{
        static vector<unsigned> shuffled(unsigned size){
            vector<unsigned> vec;
            for(unsigned i = 0; i < size; i++){
                vec.push_back(i);
            }
            random_shuffle(vec.begin(), vec.end());
            return vec;
        }

        static vector<unsigned> ascending(unsigned size){
            vector<unsigned> vec;
            for(unsigned i = 0; i < size; i++){
                vec.push_back(i);
            }
            return vec;
        }

    };

    vector<vector<unsigned>> array_split(vector<unsigned> vec, unsigned denom){
        unsigned split_len = vec.size() / denom;
        vector<vector<unsigned>> ret;
        vector<unsigned> tmp;
        for(unsigned i = 0; i < vec.size(); i++){
            tmp.push_back(vec[i]);
            if(i % split_len == 0 && i != 0) ret.push_back(tmp); tmp.clear();
        }
        return ret;
    }

    vector<vector<vector<vector<unsigned>>>> get_batches(bool shuffle = true){
        //ret[batch][sent][token][feature]

        vector<vector<unsigned>> batches;
        //batches[idx of bucket][idx of sent]

        for(int i = 0; i < this->_buckets.size(); i++){
            unsigned bucket_len = this->_buckets[i].size();
//            unsigned n_tokens = bucket_len * this->_bucket_sizes[i];
//            unsigned one = 1;
////            unsigned n_splits = std::max(n_tokens / batch_size, one);
//            unsigned n_splits = std::max(bucket_len / batch_size, one);
            auto func = &range_func::shuffled;
            if(shuffle){
                auto func = &range_func::shuffled;
            }
            else{
                auto func = &range_func::ascending;
            }

            vector<unsigned> order = func(bucket_len);
//            vector<vector<unsigned>> split_order = array_split(order, n_splits);
//            for(auto so : split_order){
//                batches.push_back(so);
//            }

            batches.push_back(order);
            //batches.push_back(batch);
        }


        vector<vector<vector<vector<unsigned>>>> ret(0);

        for(unsigned i = 0; i < batches.size(); i++){
            vector<vector<vector<unsigned>>> tmp;
            for(unsigned j = 0; j < batches[i].size(); j++){
                tmp.push_back(this->_buckets[i][batches[i][j]]);   //batch[i][j] th sentence in i th bucket
            }
            ret.push_back(tmp);                                 //all sentences in i th batch
        }

        if(shuffle)
            random_shuffle(ret.begin(), ret.end());

        return ret;                                             //batched sentences
    }

    vector<unsigned> get_bucket_sizes(){
        return this->_bucket_sizes;
    }


};