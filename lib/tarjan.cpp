//
// Created by 松野智紀 on 2017/11/01.
//

#include "../headers.h"

class Tarjan{
private:
    map<unsigned, vector<unsigned>> _edges;
    set<unsigned> _vertices{0};
    vector<unsigned> _indices;
    vector<unsigned> _lowlinks;
    vector<bool> _onstack;
    vector<vector<unsigned>> _SCCs;
    unsigned index = 0;
    vector<unsigned> stack;

    void strongconnect(const unsigned v, unsigned index, vector<unsigned>& stack){
        this->_indices[v] = index;
        this->_lowlinks[v] = index;
        index += 1;
        stack.push_back(v);
        this->_onstack[v] = true;
        for(auto w : this->_edges[v]){
            if(find(this->_indices.begin(), this->_indices.end(), w) == this->_indices.end()){
                strongconnect(w, index, stack);
                this->_lowlinks[v] = min(this->_lowlinks[v], this->_lowlinks[w]);
            }
            else if(this->_onstack[w]){
                this->_lowlinks[v] = min(this->_lowlinks[v], this->_indices[w]);
            }
        }

        if(this->_lowlinks[v] == this->_indices[v]){
            vector<unsigned> tmp;
            this->_SCCs.push_back(tmp);
            while(stack[stack.size()-1] != v){
                unsigned w = stack.back(); stack.pop_back();
                this->_onstack[w] = false;
                this->_SCCs[_SCCs.size()-1].push_back(w);
            }
            unsigned w = stack.back(); stack.pop_back();
            this->_onstack[w] = false;
            this->_SCCs[_SCCs.size() - 1].push_back(w);
        }
        return;

    }

    Tarjan(vector<unsigned> prediction, vector<unsigned> tokens){
        for(unsigned i = 0; i < prediction.size(); i++){
            this->_vertices.insert(i + 1);
            this->_edges[prediction[i]].push_back(i + 1);
        }
        for(auto v : this->_vertices){
            if(_vertices.count(v))
                strongconnect(v, this->index, this->stack);
        }


    }

public:
    map<unsigned, vector<unsigned>> edges(){
        return this->_edges;
    }

    set<unsigned> vertices(){
        return this->_vertices;
    }

    vector<unsigned> indices(){
        return this->_indices;
    }

    vector<vector<unsigned>> SCCs(){
        return this->_SCCs;
    }
};