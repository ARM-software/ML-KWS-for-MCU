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

#include "nn.h"

NN::~NN() {
}

int NN::get_frame_len() {
  return frame_len;
}

int NN::get_frame_shift() {
  return frame_shift;
}

int NN::get_num_mfcc_features() {
  return num_mfcc_features;
}

int NN::get_num_frames() {
  return num_frames;
}

int NN::get_num_out_classes() {
  return num_out_classes;
}

int NN::get_in_dec_bits() {
  return in_dec_bits;
}

