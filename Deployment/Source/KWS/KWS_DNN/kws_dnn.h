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

#ifndef __KWS_DNN_H__
#define __KWS_DNN_H__

#include "kws.h"
#include "dnn.h"

class KWS_DNN : public KWS {
public:
  KWS_DNN(int recording_win, int sliding_window_len);
  KWS_DNN(int16_t* audio_buffer);
  ~KWS_DNN();
};

#endif
