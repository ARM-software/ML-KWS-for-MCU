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

/*
 * Description: Keyword spotting example code using MFCC feature extraction
 * and neural network. 
 */

#include "kws.h"

KWS::KWS()
{
}

KWS::~KWS()
{
  delete nn;
  delete mfcc;
  delete mfcc_buffer;
  delete output;
  delete predictions;
  delete averaged_output;
}

void KWS::init_kws()
{
  num_mfcc_features = nn->get_num_mfcc_features();
  num_frames = nn->get_num_frames();
  frame_len = nn->get_frame_len();
  frame_shift = nn->get_frame_shift();
  int mfcc_dec_bits = nn->get_in_dec_bits();
  num_out_classes = nn->get_num_out_classes();
  mfcc = new MFCC(num_mfcc_features, frame_len, mfcc_dec_bits);
  mfcc_buffer = new q7_t[num_frames*num_mfcc_features];
  output = new q7_t[num_out_classes];
  averaged_output = new q7_t[num_out_classes];
  predictions = new q7_t[sliding_window_len*num_out_classes];
  audio_block_size = recording_win*frame_shift;
  audio_buffer_size = audio_block_size + frame_len - frame_shift;
}

void KWS::extract_features() 
{
  if(num_frames>recording_win) {
    //move old features left 
    memmove(mfcc_buffer,mfcc_buffer+(recording_win*num_mfcc_features),(num_frames-recording_win)*num_mfcc_features);
  }
  //compute features only for the newly recorded audio
  int32_t mfcc_buffer_head = (num_frames-recording_win)*num_mfcc_features; 
  for (uint16_t f = 0; f < recording_win; f++) {
    mfcc->mfcc_compute(audio_buffer+(f*frame_shift),&mfcc_buffer[mfcc_buffer_head]);
    mfcc_buffer_head += num_mfcc_features;
  }
}

void KWS::classify()
{
  nn->run_nn(mfcc_buffer, output);
  // Softmax
  arm_softmax_q7(output,num_out_classes,output);
}

int KWS::get_top_class(q7_t* prediction)
{
  int max_ind=0;
  int max_val=-128;
  for(int i=0;i<num_out_classes;i++) {
    if(max_val<prediction[i]) {
      max_val = prediction[i];
      max_ind = i;
    }    
  }
  return max_ind;
}

void KWS::average_predictions()
{
  //shift right old predictions 
  arm_copy_q7((q7_t *)predictions, (q7_t *)(predictions+num_out_classes), (sliding_window_len-1)*num_out_classes);
  //add new predictions
  arm_copy_q7((q7_t *)output, (q7_t *)predictions, num_out_classes);
  //compute averages
  int sum;
  for(int j=0;j<num_out_classes;j++) {
    sum=0;
    for(int i=0;i<sliding_window_len;i++) 
      sum += predictions[i*num_out_classes+j];
    averaged_output[j] = (q7_t)(sum/sliding_window_len);
  }   
}
  
