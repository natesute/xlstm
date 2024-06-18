#ifndef BLOCK_KERNELS_H
#define BLOCK_KERNELS_H

#include "slstm_kernels.h"
#include "mlstm_kernels.h"

template <typename T>
void launch_xlstm_block_forward(const T *x,
                                const T *h_prev,
                                const T *c_prev,
                                const T *C_prev,
                                const T *n_prev,
                                T *h,
                                T *c,
                                T *C,
                                T *n,
                                const T *w_proj,
                                const T *w_gate,
                                const T *b_proj,
                                const T *b_gate,
                                const T *w_slstm,
                                const T *w_mlstm,
                                const T *b_slstm,
                                const T *b_mlstm,
                                int batch_size,
                                int input_size,
                                int hidden_size,
                                int proj_size,
                                bool use_mlstm);

template <typename T>
void launch_xlstm_block_backward(const T *grad_h,
                                 const T *h,
                                 const T *c,
                                 const T *C,
                                 const T *n,
                                 const T *x,
                                 const T *w_proj,
                                 const T *w_gate,
                                 const T *w_slstm,
                                 const T *w_mlstm,
                                 T *grad_x,
                                 T *grad_h_prev,
                                 T *grad_c_prev,
                                 T *grad_C_prev,
                                 T *grad_n_prev,
                                 T *grad_w_proj,
                                 T *grad_w_gate,
                                 T *grad_b_proj,
                                 T *grad_b_gate,
                                 const T *grad_w_slstm,
                                 const T *grad_w_mlstm,
                                 const T *grad_b_slstm,
                                 const T *grad_b_mlstm,
                                 int batch_size,
                                 int input_size,
                                 int hidden_size,
                                 int proj_size,
                                 bool use_mlstm);


#ifdef __CUDACC__
// xLSTM block forward pass kernel
template <typename T>
__global__ void xlstm_block_forward_kernel(const T *__restrict__ x,
                                const T *__restrict__ h_prev,
                                const T *__restrict__ c_prev,
                                const T *__restrict__ C_prev,
                                const T *__restrict__ n_prev,
                                T *__restrict__ h,
                                T *__restrict__ c,
                                T *__restrict__ C,
                                T *__restrict__ n,
                                const T *__restrict__ w_proj,
                                const T *__restrict__ w_gate,
                                const T *__restrict__ b_proj,
                                const T *__restrict__ b_gate,
                                const T *__restrict__ w_slstm,
                                const T *__restrict__ w_mlstm,
                                const T *__restrict__ b_slstm,
                                const T *__restrict__ b_mlstm,
                                int batch_size,
                                int input_size,
                                int hidden_size,
                                int proj_size,
                                bool use_mlstm);

// xLSTM block backward pass kernel
template <typename T>
__global__ void xlstm_block_backward_kernel(const T *__restrict__ grad_h,
                                    const T *__restrict__ h,
                                    const T *__restrict__ c,
                                    const T *__restrict__ C,
                                    const T *__restrict__ n,
                                    const T *__restrict__ x,
                                    const T *__restrict__ w_proj,
                                    const T *__restrict__ w_gate,
                                    const T *__restrict__ b_gate,
                                    const T *__restrict__ w_slstm,
                                    const T *__restrict__ w_mlstm,
                                    const T *__restrict__ grad_x,
                                    const T *__restrict__ grad_h_prev,
                                    const T *__restrict__ grad_c_prev,
                                    const T *__restrict__ grad_C_prev,
                                    const T *__restrict__ grad_n_prev,
                                    const T *__restrict__ grad_w_proj,
                                    const T *__restrict__ grad_w_gate,
                                    const T *__restrict__ grad_b_proj,
                                    const T *__restrict__ grad_b_gate,
                                    const T *__restrict__ grad_w_slstm,
                                    const T *__restrict__ grad_w_mlstm,
                                    const T *__restrict__ grad_b_slstm,
                                    const T *__restrict__ grad_b_mlstm,
                                    int batch_size,
                                    int input_size,
                                    int hidden_size,
                                    int proj_size,
                                    bool use_mlstm);

#endif // __CUDACC__
#endif // BLOCK_KERNELS_H
