#include "arm_nnsupportfunctions.h"
#include "arm_nn_tables.h"

#define USE_INTRINSIC

#ifdef __cplusplus
extern    "C"
{
#endif

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


