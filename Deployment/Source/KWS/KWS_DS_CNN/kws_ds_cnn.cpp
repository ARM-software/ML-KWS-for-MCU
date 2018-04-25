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


#include "kws_ds_cnn.h"

KWS_DS_CNN::KWS_DS_CNN(int record_win, int sliding_win_len)
{
  nn = new DS_CNN();
  recording_win = record_win;
  sliding_window_len = sliding_win_len;
  init_kws();
}

KWS_DS_CNN::KWS_DS_CNN(int16_t* audio_data_buffer)
{
  nn = new DS_CNN();
  audio_buffer = audio_data_buffer;
  recording_win = nn->get_num_frames();
  sliding_window_len = 1;
  init_kws();
}

KWS_DS_CNN::~KWS_DS_CNN()
{
  delete nn;
}

