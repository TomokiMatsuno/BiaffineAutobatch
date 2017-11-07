//
// Created by 松野智紀 on 2017/10/28.
//

#ifndef BIAFFINE_2ND_CONFIG_H
#define BIAFFINE_2ND_CONFIG_H

#endif //BIAFFINE_2ND_CONFIG_H

vector<string> train_files = {"just-one-sentence.txt", "en-ud-train.txt", "ten-sentences.txt"};
string train_file = "/Users/tomoki/CLionProjects/BiAffine/file-data/" + train_files[1];
vector<string> root_line  = {"0", "ROOT", "ROOT", "ROOT", "ROOT", "ROOT", "0", "ROOT", "_", "_"};

bool shuffled = false;
unsigned batch_size = 20;
unsigned epoc = 100;