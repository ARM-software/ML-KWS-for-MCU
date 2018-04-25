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

#ifndef __KWS_F746NG_H__
#define __KWS_F746NG_H__

#include "AUDIO_DISCO_F746NG.h"
#include "kws_ds_cnn.h"
//#include "kws_dnn.h"

// Change the parent class to KWS_DNN to switch to DNN model
//class KWS_F746NG : public KWS_DNN {
class KWS_F746NG : public KWS_DS_CNN {
public:
  KWS_F746NG(int recording_win, int sliding_window_len);
  ~KWS_F746NG();
  void start_kws();
  void set_volume(int vol);
  int16_t* audio_buffer_in;
  //for debugging: microphone to headphone loopback
  int16_t* audio_buffer_out; 

private:
  AUDIO_DISCO_F746NG audio;

};

#endif
