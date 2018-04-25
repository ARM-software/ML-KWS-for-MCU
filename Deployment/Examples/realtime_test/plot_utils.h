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

#include "kws_f746ng.h"
#include "LCD_DISCO_F746NG.h"

#define LCD_COLOR_ARM_BLUE ((uint32_t) 0xFF00C1DE)
#define LCD_COLOR_ARM_DARK ((uint32_t) 0xFF333E48)

extern LCD_DISCO_F746NG lcd;
extern KWS_F746NG *kws;

void init_plot();
void plot_mfcc();
void plot_waveform();
uint32_t calculate_rgb(int min, int max, int value);


