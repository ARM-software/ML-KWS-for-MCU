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

#ifndef __KWS_H__
#define __KWS_H__

#include "arm_math.h"
#include "mbed.h"
#include "dnn.h"
#include "mfcc.h"

#define MAX_SLIDING_WINDOW 10

class KWS{

public:
  KWS(int16_t* audio_buffer, q7_t* scratch_buffer);
  ~KWS();
  
  void extract_features();
  //overloaded function for 
  void extract_features(uint16_t num_frames);
  void classify();
  void average_predictions(int window_len);
  int get_top_detection(q7_t* prediction);
  int16_t* audio_buffer;
  q7_t mfcc_buffer[MFCC_BUFFER_SIZE];
  q7_t output[OUT_DIM];
  q7_t predictions[MAX_SLIDING_WINDOW][OUT_DIM];
  q7_t averaged_output[OUT_DIM];
  
private:
  MFCC *mfcc;
  DNN *nn;
  
};

#endif
