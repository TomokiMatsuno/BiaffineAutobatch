//
// Created by 松野智紀 on 2017/10/18.
//

//#include "lib/utils.h"
//#include "headers.h"

void showElemsInBatches(vector<vector<vector<vector<unsigned>>>> batches){
    for(int b = 0; b < batches.size(); b++){
        for(int s = 0; s < batches[b].size(); s++){
            for(int t = 0; t < batches[b][s].size(); t++){
                cout << batches[b][s][t][0] << " ";
            }
            cout << endl;
        }
    }

    return;
}

Expression& ones(ComputationGraph& cg,  unsigned vec_len){
    vector<float> ones_vec(vec_len, 1);
    Expression ones_vec_expr = dynet::input(cg, {vec_len}, &ones_vec);
    return ones_vec_expr;
}

Expression& bilinear(ComputationGraph& cg, Expression& x, Expression& W, Expression& y, unsigned input_size, unsigned seq_len, unsigned num_outputs = 1, bool bias_x=false, bool bias_y=false){
    unsigned nx = x.dim().cols();
    unsigned ny = y.dim().cols();

    //Todo: somehow, after concatenating a Expression with one initialized with float, get length_error vector. Do not use bilinear bias term for now.
    if(bias_x){
        x = concatenate({x, ones(cg, nx)}, 0);
    }
    if(bias_y){
        y = concatenate({y, ones(cg, ny)}, 0);
    }
    nx += bias_x;
    ny += bias_y;
    Expression lin = W * x;
    if(num_outputs > 1){
        lin = reshape(lin, {ny, num_outputs * seq_len});
    }
    //cg.forward(lin);
    Expression blin = transpose(y) * lin;
    if(num_outputs > 1){
        blin = reshape(blin, {seq_len, num_outputs, seq_len});
    }
    //cg.forward(blin);
    return blin;
}
