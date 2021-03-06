#include "headers.h"

int main(int argc, char** argv) {
    dynet::initialize(argc, argv);
    std::cout << "Hello, World!" << std::endl;

    ordinalUtilDict wd = 1;            //Dictionary which deals with i th token in each input line.
    ordinalUtilDict td = 3;
    stoiUtilDict hd = 6;
    ordinalUtilDict rd = 7;

    hd.i2x.clear(); hd.x2i.clear();
    for(int i = 0; i < 10000; i++) {
        hd.add_entry(to_string(i));
    }

    Vocab vocab(train_file, '\t', {wd, td, hd, rd});
    DataLoader dl(train_file, 1, vocab);

    vector<vector<vector<vector<unsigned>>>> batches = dl.get_batches(false);

    vector<vector<vector<vector<unsigned>>>> new_batches(batches[0].size() / batch_size);
    for(int i = 0; i * batch_size < batches[0].size() - batch_size + 1; i++){
        for(int j = i; j < batch_size + i; j++){
            new_batches[i].push_back(batches[0][j]);
        }
    }
    batches.clear();
    batches.insert(batches.end(), new_batches.begin(), new_batches.end());

//    showSents(vocab, batches);

    ParameterCollection m;
    SimpleSGDTrainer sgd(m);
    Parser<VanillaLSTMBuilder> parser(m, vocab);

    for(unsigned e = 0; e < epoc; e++) {
        cerr << "epoc: " << e << endl;
        for (unsigned i = 0; i < batches.size(); i++) {
            cerr << "sentence: " << i << endl;
            ComputationGraph cg;
            parser.l2rbuilder.new_graph(cg);  // reset RNN builder for new graph
            parser.r2lbuilder.new_graph(cg);  // reset RNN builder for new graph

            Expression s;
            double loss = 0;
            vector<Expression> errs(batches[i].size());

            for (unsigned j = 0; j < batches[i].size(); j++) {
                cout << i << " " << j << endl;
                vector<unsigned> seq_word, seq_pos, seq_head, seq_rel;
                for (unsigned k = 0; k < batches[i][j].size(); k++) {
                    seq_word.push_back(batches[i][j][k][0]);
                    seq_pos.push_back(batches[i][j][k][1]);
                    seq_head.push_back(batches[i][j][k][2]);
                    seq_rel.push_back(batches[i][j][k][3]);
//                    cout << vocab.get_dicts()[2].i2x[seq_head[seq_head.size() - 1]] << " ";
//                    cout << vocab.get_dicts()[0].i2x[seq_word[seq_word.size() - 1]] << " ";
                }
//                cout << endl;
                parser.l2rbuilder.start_new_sequence();
                parser.r2lbuilder.start_new_sequence();
                errs[j] = parser.BuildParser(cg, seq_word, seq_pos, seq_head, seq_rel);
            }

//            Expression pred_arc = concatenate(errs);
            Expression sum_errs = sum(errs);
            loss += as_scalar(cg.forward(sum_errs));
            cg.backward(sum_errs);
            sgd.update();
            //cg.invalidate();

            cout << "loss " << loss << endl;
        }
    }


    return 0;
}

void showSents(Vocab& vocab, vector<vector<vector<vector<unsigned>>>>& batches){
    for(int i = 0; i < batches.size(); i++){
        cout << "batch_" << endl;
        for(int j = 0; j < batches[i].size(); j++){
            for(int k = 0; k < batches[i][j].size(); k++){
                cout << vocab.get_dicts()[0].i2x[batches[i][j][k][0]] << " ";
            }
            cout << endl;
            cout << endl;
        }
    }
}
