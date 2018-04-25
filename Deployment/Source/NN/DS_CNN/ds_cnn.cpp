/*
 * Copyright (C) 2018 Arm Limited or its affiliates. All rights reserved.
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the License); you may
 * not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an AS IS BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "ds_cnn.h"

const q7_t DS_CNN::conv1_wt[CONV1_OUT_CH*CONV1_KX*CONV1_KY]=CONV1_WT;
const q7_t DS_CNN::conv1_bias[CONV1_OUT_CH]=CONV1_BIAS;
const q7_t DS_CNN::conv2_ds_wt[CONV1_OUT_CH*CONV2_DS_KX*CONV2_DS_KY]=CONV2_DS_WT;
const q7_t DS_CNN::conv2_ds_bias[CONV1_OUT_CH]=CONV2_DS_BIAS;
const q7_t DS_CNN::conv2_pw_wt[CONV2_OUT_CH*CONV1_OUT_CH]=CONV2_PW_WT;
const q7_t DS_CNN::conv2_pw_bias[CONV2_OUT_CH]=CONV2_PW_BIAS;
const q7_t DS_CNN::conv3_ds_wt[CONV2_OUT_CH*CONV3_DS_KX*CONV3_DS_KY]=CONV3_DS_WT;
const q7_t DS_CNN::conv3_ds_bias[CONV2_OUT_CH]=CONV3_DS_BIAS;
const q7_t DS_CNN::conv3_pw_wt[CONV3_OUT_CH*CONV2_OUT_CH]=CONV3_PW_WT;
const q7_t DS_CNN::conv3_pw_bias[CONV3_OUT_CH]=CONV3_PW_BIAS;
const q7_t DS_CNN::conv4_ds_wt[CONV3_OUT_CH*CONV4_DS_KX*CONV4_DS_KY]=CONV4_DS_WT;
const q7_t DS_CNN::conv4_ds_bias[CONV3_OUT_CH]=CONV4_DS_BIAS;
const q7_t DS_CNN::conv4_pw_wt[CONV4_OUT_CH*CONV3_OUT_CH]=CONV4_PW_WT;
const q7_t DS_CNN::conv4_pw_bias[CONV4_OUT_CH]=CONV4_PW_BIAS;
const q7_t DS_CNN::conv5_ds_wt[CONV4_OUT_CH*CONV5_DS_KX*CONV5_DS_KY]=CONV5_DS_WT;
const q7_t DS_CNN::conv5_ds_bias[CONV4_OUT_CH]=CONV5_DS_BIAS;
const q7_t DS_CNN::conv5_pw_wt[CONV5_OUT_CH*CONV4_OUT_CH]=CONV5_PW_WT;
const q7_t DS_CNN::conv5_pw_bias[CONV5_OUT_CH]=CONV5_PW_BIAS;
const q7_t DS_CNN::final_fc_wt[CONV5_OUT_CH*OUT_DIM]=FINAL_FC_WT;
const q7_t DS_CNN::final_fc_bias[OUT_DIM]=FINAL_FC_BIAS;

DS_CNN::DS_CNN()
{
  scratch_pad = new q7_t[SCRATCH_BUFFER_SIZE];
  buffer1 = scratch_pad;
  buffer2 = buffer1 + (CONV1_OUT_CH*CONV1_OUT_X*CONV1_OUT_Y);
  col_buffer = buffer2 + (CONV2_OUT_CH*CONV2_OUT_X*CONV2_OUT_Y);
  frame_len = FRAME_LEN;
  frame_shift = FRAME_SHIFT;
  num_mfcc_features = NUM_MFCC_COEFFS;
  num_frames = NUM_FRAMES;
  num_out_classes = OUT_DIM;
  in_dec_bits = MFCC_DEC_BITS;
}

DS_CNN::~DS_CNN()
{
  delete scratch_pad;
}

void DS_CNN::run_nn(q7_t* in_data, q7_t* out_data)
{
  //CONV1 : regular convolution
  arm_convolve_HWC_q7_basic_nonsquare(in_data, CONV1_IN_X, CONV1_IN_Y, 1, conv1_wt, CONV1_OUT_CH, CONV1_KX, CONV1_KY, CONV1_PX, CONV1_PY, CONV1_SX, CONV1_SY, conv1_bias, CONV1_BIAS_LSHIFT, CONV1_OUT_RSHIFT, buffer1, CONV1_OUT_X, CONV1_OUT_Y, (q15_t*)col_buffer, NULL);
  arm_relu_q7(buffer1,CONV1_OUT_X*CONV1_OUT_Y*CONV1_OUT_CH);

  //CONV2 : DS + PW conv
  //Depthwise separable conv (batch norm params folded into conv wts/bias)
  arm_depthwise_separable_conv_HWC_q7_nonsquare(buffer1,CONV2_IN_X,CONV2_IN_Y,CONV1_OUT_CH,conv2_ds_wt,CONV1_OUT_CH,CONV2_DS_KX,CONV2_DS_KY,CONV2_DS_PX,CONV2_DS_PY,CONV2_DS_SX,CONV2_DS_SY,conv2_ds_bias,CONV2_DS_BIAS_LSHIFT,CONV2_DS_OUT_RSHIFT,buffer2,CONV2_OUT_X,CONV2_OUT_Y,(q15_t*)col_buffer, NULL);
  arm_relu_q7(buffer2,CONV2_OUT_X*CONV2_OUT_Y*CONV2_OUT_CH);

  //Pointwise conv
  arm_convolve_1x1_HWC_q7_fast_nonsquare(buffer2, CONV2_OUT_X, CONV2_OUT_Y, CONV1_OUT_CH, conv2_pw_wt, CONV2_OUT_CH, 1, 1, 0, 0, 1, 1, conv2_pw_bias, CONV2_PW_BIAS_LSHIFT, CONV2_PW_OUT_RSHIFT, buffer1, CONV2_OUT_X, CONV2_OUT_Y, (q15_t*)col_buffer, NULL);
  arm_relu_q7(buffer1,CONV2_OUT_X*CONV2_OUT_Y*CONV2_OUT_CH);

  //CONV3 : DS + PW conv
  //Depthwise separable conv (batch norm params folded into conv wts/bias)
  arm_depthwise_separable_conv_HWC_q7_nonsquare(buffer1,CONV3_IN_X,CONV3_IN_Y,CONV2_OUT_CH,conv3_ds_wt,CONV2_OUT_CH,CONV3_DS_KX,CONV3_DS_KY,CONV3_DS_PX,CONV3_DS_PY,CONV3_DS_SX,CONV3_DS_SY,conv3_ds_bias,CONV3_DS_BIAS_LSHIFT,CONV3_DS_OUT_RSHIFT,buffer2,CONV3_OUT_X,CONV3_OUT_Y,(q15_t*)col_buffer, NULL);
  arm_relu_q7(buffer2,CONV3_OUT_X*CONV3_OUT_Y*CONV3_OUT_CH);
  //Pointwise conv
  arm_convolve_1x1_HWC_q7_fast_nonsquare(buffer2, CONV3_OUT_X, CONV3_OUT_Y, CONV2_OUT_CH, conv3_pw_wt, CONV3_OUT_CH, 1, 1, 0, 0, 1, 1, conv3_pw_bias, CONV3_PW_BIAS_LSHIFT, CONV3_PW_OUT_RSHIFT, buffer1, CONV3_OUT_X, CONV3_OUT_Y, (q15_t*)col_buffer, NULL);
  arm_relu_q7(buffer1,CONV3_OUT_X*CONV3_OUT_Y*CONV3_OUT_CH);

  //CONV4 : DS + PW conv
  //Depthwise separable conv (batch norm params folded into conv wts/bias)
  arm_depthwise_separable_conv_HWC_q7_nonsquare(buffer1,CONV4_IN_X,CONV4_IN_Y,CONV3_OUT_CH,conv4_ds_wt,CONV3_OUT_CH,CONV4_DS_KX,CONV4_DS_KY,CONV4_DS_PX,CONV4_DS_PY,CONV4_DS_SX,CONV4_DS_SY,conv4_ds_bias,CONV4_DS_BIAS_LSHIFT,CONV4_DS_OUT_RSHIFT,buffer2,CONV4_OUT_X,CONV4_OUT_Y,(q15_t*)col_buffer, NULL);
  arm_relu_q7(buffer2,CONV4_OUT_X*CONV4_OUT_Y*CONV4_OUT_CH);
  //Pointwise conv
  arm_convolve_1x1_HWC_q7_fast_nonsquare(buffer2, CONV4_OUT_X, CONV4_OUT_Y, CONV3_OUT_CH, conv4_pw_wt, CONV4_OUT_CH, 1, 1, 0, 0, 1, 1, conv4_pw_bias, CONV4_PW_BIAS_LSHIFT, CONV4_PW_OUT_RSHIFT, buffer1, CONV4_OUT_X, CONV4_OUT_Y, (q15_t*)col_buffer, NULL);
  arm_relu_q7(buffer1,CONV4_OUT_X*CONV4_OUT_Y*CONV4_OUT_CH);

  //CONV5 : DS + PW conv
  //Depthwise separable conv (batch norm params folded into conv wts/bias)
  arm_depthwise_separable_conv_HWC_q7_nonsquare(buffer1,CONV5_IN_X,CONV5_IN_Y,CONV4_OUT_CH,conv5_ds_wt,CONV4_OUT_CH,CONV5_DS_KX,CONV5_DS_KY,CONV5_DS_PX,CONV5_DS_PY,CONV5_DS_SX,CONV5_DS_SY,conv5_ds_bias,CONV5_DS_BIAS_LSHIFT,CONV5_DS_OUT_RSHIFT,buffer2,CONV5_OUT_X,CONV5_OUT_Y,(q15_t*)col_buffer, NULL);
  arm_relu_q7(buffer2,CONV5_OUT_X*CONV5_OUT_Y*CONV5_OUT_CH);
  //Pointwise conv
  arm_convolve_1x1_HWC_q7_fast_nonsquare(buffer2, CONV5_OUT_X, CONV5_OUT_Y, CONV4_OUT_CH, conv5_pw_wt, CONV5_OUT_CH, 1, 1, 0, 0, 1, 1, conv5_pw_bias, CONV5_PW_BIAS_LSHIFT, CONV5_PW_OUT_RSHIFT, buffer1, CONV5_OUT_X, CONV5_OUT_Y, (q15_t*)col_buffer, NULL);
  arm_relu_q7(buffer1,CONV5_OUT_X*CONV5_OUT_Y*CONV5_OUT_CH);

  //Average pool
  arm_avepool_q7_HWC_nonsquare (buffer1,CONV5_OUT_X,CONV5_OUT_Y,CONV5_OUT_CH,CONV5_OUT_X,CONV5_OUT_Y,0,0,1,1,1,1,NULL,buffer2, 2);

  arm_fully_connected_q7(buffer2, final_fc_wt, CONV5_OUT_CH, OUT_DIM, FINAL_FC_BIAS_LSHIFT, FINAL_FC_OUT_RSHIFT, final_fc_bias, out_data, (q15_t*)col_buffer);

}


