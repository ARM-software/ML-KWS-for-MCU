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
 * and DNN model. 
 */

#include "kws.h"
#include "string.h"

KWS::KWS(int16_t* audio_buffer, q7_t* scratch_buffer)
: audio_buffer(audio_buffer)
{
  mfcc = new MFCC;
  nn = new DNN(scratch_buffer);
}

KWS::~KWS()
{
  delete mfcc;
  delete nn;
}

void KWS::extract_features()
{
  int32_t mfcc_buffer_head=0;
  for (uint16_t f = 0; f < NUM_FRAMES; f++) {
    mfcc->mfcc_compute(audio_buffer+(f*FRAME_SHIFT),2,&mfcc_buffer[mfcc_buffer_head]);
    mfcc_buffer_head += NUM_MFCC_COEFFS;
  }
}

/* This overloaded function is used in streaming audio case */
void KWS::extract_features(uint16_t num_frames) 
{
  //move old features left 
  memmove(mfcc_buffer,mfcc_buffer+(num_frames*NUM_MFCC_COEFFS),(NUM_FRAMES-num_frames)*NUM_MFCC_COEFFS);
  //compute features only for the newly recorded audio
  int32_t mfcc_buffer_head = (NUM_FRAMES-num_frames)*NUM_MFCC_COEFFS; 
  for (uint16_t f = 0; f < num_frames; f++) {
    mfcc->mfcc_compute(audio_buffer+(f*FRAME_SHIFT),2,&mfcc_buffer[mfcc_buffer_head]);
    mfcc_buffer_head += NUM_MFCC_COEFFS;
  }
}

void KWS::classify()
{
  nn->run_nn(mfcc_buffer, output);

  // Softmax
  arm_softmax_q7(output,OUT_DIM,output);

  //do any post processing here
}

int KWS::get_top_detection(q7_t* prediction)
{
  int max_ind=0;
  int max_val=-128;
  for(int i=0;i<OUT_DIM;i++) {
    if(max_val<prediction[i]) {
      max_val = prediction[i];
      max_ind = i;
    }    
  }
  return max_ind;
}

void KWS::average_predictions(int window_len)
{
  //shift right old predictions 
  for(int i=window_len-1;i>0;i--) {
    for(int j=0;j<OUT_DIM;j++)
      predictions[i][j]=predictions[i-1][j];
  }
  //add new predictions
  for(int j=0;j<OUT_DIM;j++)
    predictions[0][j]=output[j];
  //compute averages
  int sum;
  for(int j=0;j<OUT_DIM;j++) {
    sum=0;
    for(int i=0;i<window_len;i++) 
      sum += predictions[i][j];
    averaged_output[j] = (q7_t)(sum/window_len);
  }   
}
