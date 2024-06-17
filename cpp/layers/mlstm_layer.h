#ifndef MLSTM_LAYER_H
#define MLSTM_LAYER_H

#include <vector>
#include "../cuda/kernels/mlstm_kernels.h"

template <typename T>
class MLSTMLayer {
public:
    MLSTMLayer(int input_size, int hidden_size);
    ~MLSTMLayer();

    const T* get_weights() const { return weights_; }
    const T* get_biases() const { return biases_; }
    const T* get_grad_weights() const { return grad_weights_; }
    const T* get_grad_biases() const { return grad_biases_; }

    void forward(const T* input, const T* h_prev, const T* C_prev, const T* n_prev,
                 T* h, T* C, T* n);
    void backward(const T* grad_h, const T* C, const T* n, const T* input,
                  T* grad_input, T* grad_C_prev, T* grad_n_prev);

private:
    int input_size_;
    int hidden_size_;

    T* weights_;
    T* biases_;
    T* grad_weights_;
    T* grad_biases_;

    T* w_k_;
    T* w_v_;
    T* w_q_;
    T* w_i_;
    T* w_f_;
    T* w_o_;
    T* b_k_;
    T* b_v_;
    T* b_q_;
    T* b_i_;
    T* b_f_;
    T* b_o_;

    T* grad_w_k_;
    T* grad_w_v_;
    T* grad_w_q_;
    T* grad_w_i_;
    T* grad_w_f_;
    T* grad_w_o_;
    T* grad_b_k_;
    T* grad_b_v_;
    T* grad_b_q_;
    T* grad_b_i_;
    T* grad_b_f_;
    T* grad_b_o_;

    void allocate_memory();
    void free_memory();
};

#endif // MLSTM_LAYER_H