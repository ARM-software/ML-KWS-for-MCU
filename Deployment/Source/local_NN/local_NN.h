#include "arm_nnsupportfunctions.h"
#include "arm_nn_tables.h"

#define USE_INTRINSIC

#ifdef __cplusplus
extern    "C"
{
#endif

arm_status arm_convolve_HWC_q7_basic_nonsquare(const q7_t * Im_in,
        const uint16_t dim_im_in_X,
        const uint16_t dim_im_in_Y,
        const uint16_t ch_im_in,
        const q7_t * wt,
        const uint16_t ch_im_out,
        const uint16_t dim_kernel_X,
        const uint16_t dim_kernel_Y,
        const uint16_t padding_X,
        const uint16_t padding_Y,
        const uint16_t stride_X,
        const uint16_t stride_Y,
        const q7_t * bias,
        const uint16_t bias_shift,
        const uint16_t out_shift,
        q7_t * Im_out, 
        const uint16_t dim_im_out_X, 
        const uint16_t dim_im_out_Y, 
        q15_t * bufferA, 
        q7_t * bufferB);

void arm_avepool_q7_HWC_nonsquare (
        const q7_t * Im_in,         
        const uint16_t dim_im_in_x,   
        const uint16_t dim_im_in_y,   
        const uint16_t ch_im_in,    
        const uint16_t dim_kernel_x,  
        const uint16_t dim_kernel_y,  
        const uint16_t padding_x,     
        const uint16_t padding_y,     
        const uint16_t stride_x,      
        const uint16_t stride_y,      
        const uint16_t dim_im_out_x,  
        const uint16_t dim_im_out_y,  
        q7_t * bufferA,             
        q7_t * Im_out,
        const uint16_t out_lshift);


#ifdef __cplusplus
}
#endif


