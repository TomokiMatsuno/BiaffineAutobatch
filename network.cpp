//#include "headers.h"

const unsigned EMBD_SIZE = 100;
const unsigned LSTM_SIZE = 400;
const unsigned ArcMLP_SIZE = 500;
const unsigned LabelMLP_SIZE = 100;
const unsigned LAYERS = 3;
const float pdrop = 0.33;

unsigned wrds_size;
unsigned poss_size;
unsigned arcs_size;

template <class Builder>
struct Parser {
    LookupParameter lp_w;
    LookupParameter lp_p;
    Parameter p_ArcMLP_head;
    Parameter p_ArcMLP_dep;
    Parameter p_ArcMLP_head_bias;
    Parameter p_ArcMLP_dep_bias;
    Parameter p_LabelMLP_head;
    Parameter p_LabelMLP_dep;
    Parameter p_LabelMLP_head_bias;
    Parameter p_LabelMLP_dep_bias;

    Parameter p_U_arc;
    Parameter p_U_label;

    Builder l2rbuilder;
    Builder r2lbuilder;

    explicit Parser(ParameterCollection& model, Vocab& vocab):
            l2rbuilder(LAYERS, EMBD_SIZE * 2, LSTM_SIZE, model),
            r2lbuilder(LAYERS, EMBD_SIZE * 2, LSTM_SIZE, model){
        wrds_size = vocab.get_dicts()[0].vocab_size();
        poss_size = vocab.get_dicts()[1].vocab_size();
        arcs_size = vocab.get_dicts()[3].vocab_size();

        lp_w = model.add_lookup_parameters(wrds_size, {EMBD_SIZE});
        lp_p = model.add_lookup_parameters(poss_size, {EMBD_SIZE});

        p_ArcMLP_head = model.add_parameters({ArcMLP_SIZE, LSTM_SIZE * 2});
        p_ArcMLP_head_bias = model.add_parameters({ArcMLP_SIZE});
        p_ArcMLP_dep = model.add_parameters({ArcMLP_SIZE, LSTM_SIZE * 2});
        p_ArcMLP_dep_bias = model.add_parameters({ArcMLP_SIZE});

        p_LabelMLP_head = model.add_parameters({LabelMLP_SIZE * arcs_size, LSTM_SIZE * 2});
        p_LabelMLP_head_bias = model.add_parameters({LabelMLP_SIZE});
        p_LabelMLP_dep = model.add_parameters({LabelMLP_SIZE * arcs_size, LSTM_SIZE * 2});
        p_LabelMLP_dep_bias = model.add_parameters({LabelMLP_SIZE});

//        p_U_label = model.add_parameters({LabelMLP_SIZE, LabelMLP_SIZE + 1});
//        p_U_arc = model.add_parameters({ArcMLP_SIZE, ArcMLP_SIZE + 1});
        p_U_label = model.add_parameters({LabelMLP_SIZE, LabelMLP_SIZE});
        p_U_arc = model.add_parameters({ArcMLP_SIZE, ArcMLP_SIZE});

    }

    Expression& BuildParser(ComputationGraph& cg, vector<unsigned> seq_word, vector<unsigned> seq_pos, vector<unsigned> seq_head, vector<unsigned> seq_rel) {
        const unsigned slen = seq_word.size();

//        l2rbuilder.new_graph(cg);  // reset RNN builder for new graph
//        l2rbuilder.start_new_sequence();
//        r2lbuilder.new_graph(cg);  // reset RNN builder for new graph
//        r2lbuilder.start_new_sequence();

        vector<Expression> embds_word(slen);
        vector<Expression> embds_pos(slen);

        vector<Expression> fwds(slen);
        vector<Expression> bwds(slen);

        vector<Expression> bilstm_outputs(slen);

        Expression R_ArcMLP_head; //stacked MLP outputs
        Expression R_ArcMLP_dep;
        Expression R_LabelMLP_head;
        Expression R_LabelMLP_dep;

        Expression ArcMLP_head = parameter(cg, p_ArcMLP_head);
        Expression ArcMLP_head_bias = parameter(cg, p_ArcMLP_head_bias);
        Expression ArcMLP_dep = parameter(cg, p_ArcMLP_dep);
        Expression ArcMLP_dep_bias = parameter(cg, p_ArcMLP_dep_bias);

        Expression LabelMLP_head = parameter(cg, p_LabelMLP_head);
        Expression LabelMLP_head_bias = parameter(cg, p_LabelMLP_head_bias);
        Expression LabelMLP_dep = parameter(cg, p_LabelMLP_dep);
        Expression LabelMLP_dep_bias = parameter(cg, p_LabelMLP_dep_bias);

        Expression U_label = parameter(cg, p_U_label);
        Expression U_arc = parameter(cg, p_U_arc);

        Expression S_arc;

//        Expression bLin;

//        l2rbuilder.new_graph(cg);  // reset RNN builder for new graph
//        r2lbuilder.new_graph(cg);  // reset RNN builder for new graph


//        l2rbuilder.start_new_sequence();

        for(unsigned t = 0; t < slen; ++t){
            embds_word[t] = lookup(cg, lp_w, seq_word[t]);
            embds_pos[t] = lookup(cg, lp_p, seq_pos[t]);
            fwds[t] = l2rbuilder.add_input(concatenate({embds_word[t], embds_pos[t]}));
        }
        //fwds[t]: LSTM_SIZE

//        r2lbuilder.start_new_sequence();
        for(unsigned t = 0; t < slen; ++t){
            bwds[slen - t - 1] = r2lbuilder.add_input(concatenate({embds_word[slen - t - 1], embds_pos[slen - t - 1]}));
        }
        //bwds[t]: LSTM_SIZE

        for(unsigned t = 0; t < slen; ++t){
            bilstm_outputs[t] = concatenate({fwds[t], bwds[t]});
        }
        //bilstm_outputs[t]: (LSTM_SIZE x 2)

        R_ArcMLP_head = affine_transform({ArcMLP_head_bias, ArcMLP_head, concatenate(bilstm_outputs, 1)});
        //R_ArcMLP_head: ArcMLP_SIZE * slen

        R_ArcMLP_dep = affine_transform({ArcMLP_dep_bias, ArcMLP_dep, concatenate(bilstm_outputs, 1)});
        //R_ArcMLP_dep: ArcMLP_SIZE * slen
//        cg.forward(R_ArcMLP_dep);
        S_arc = bilinear(cg, R_ArcMLP_dep, U_arc, R_ArcMLP_head, EMBD_SIZE, slen, 1, false, false);
        //S_arc: slen * slen
//        cg.forward(S_arc);
///
//        bool bias_x = true;
//        bool bias_y = false;
//        unsigned num_outputs = 1;
//        unsigned nx = R_ArcMLP_dep.dim().cols();
//        unsigned ny = R_ArcMLP_head.dim().cols();
//        unsigned seq_len = slen;
//        if(bias_x){
//            R_ArcMLP_dep = concatenate({R_ArcMLP_dep, ones(cg, nx)}, 0);
//        }
//        if(bias_y){
//            R_ArcMLP_head = concatenate({R_ArcMLP_head, ones(cg, ny)}, 0);
//        }
//        nx += bias_x;
//        ny += bias_y;
//        Expression lin = U_arc * R_ArcMLP_dep;
//        if(num_outputs > 1){
//            lin = reshape(lin, {ny, num_outputs * seq_len});
//        }
//        cg.forward(lin);
//        Expression blin = transpose(R_ArcMLP_head) * lin;
//        if(num_outputs > 1){
//            blin = reshape(blin, {seq_len, num_outputs, seq_len});
//        }
//        cg.forward(blin);
//
//
//        S_arc = blin;
///
        Dim dim_pick({slen}, slen);
//        cg.forward(S_arc);
//        Dim dim_({slen * 2, slen / 2});
//        cg.forward(reshape(S_arc, dim_));
        S_arc = reshape(S_arc, dim_pick);
        //(slen), slen

//        cg.forward(S_arc);
//        S_arc = pick(S_arc, seq_head);
//        cg.forward(S_arc);
        Expression err_arc = pickneglogsoftmax(S_arc, seq_head);
//        cg.forward(err_arc);

//        return err_arc;
//        Expression sum_err_arc = sum_elems(err_arc);
        Expression sum_err_arc = sum_batches(err_arc);
        return sum_err_arc;

    }

};
