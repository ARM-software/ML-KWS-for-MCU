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

#include "plot_utils.h"

uint32_t *mfcc_plot_buffer;
int *audio_plot_buffer;
int mfcc_update_counter=0;
int screen_size_x, screen_size_y;

void init_plot()
{
  mfcc_plot_buffer = new uint32_t[kws->num_mfcc_features*kws->num_frames*10];
   
  screen_size_x = lcd.GetXSize();
  screen_size_y = lcd.GetYSize();

  audio_plot_buffer = new int[screen_size_x];

  lcd.Clear(LCD_COLOR_ARM_BLUE);
  lcd.SetBackColor(LCD_COLOR_ARM_BLUE);
  lcd.SetTextColor(LCD_COLOR_WHITE);
  mfcc_update_counter=0;
}

void plot_mfcc()
{
  memcpy(mfcc_plot_buffer, mfcc_plot_buffer+2*kws->num_mfcc_features, 4*kws->num_mfcc_features*(10*kws->num_frames-2));

  int x_step = 1;
  int y_step = 6;

  uint32_t* pBuffer = mfcc_plot_buffer + kws->num_mfcc_features*(10*kws->num_frames-2);
  int sum = 0;

  for (int i=0;i<2;i++) {
    for (int j=0;j<kws->num_mfcc_features;j++) {
      int value = kws->mfcc_buffer[(kws->num_mfcc_features*(kws->num_frames-2))+i*kws->num_mfcc_features+j];
      uint32_t RGB  = calculate_rgb(-128, 127, value*4);
      sum += std::abs(value);
      pBuffer[i*kws->num_mfcc_features+j] = RGB;
    }
  }
  int x_start = (screen_size_x - (kws->num_frames*10))/2;
  x_start = (x_start>0) ? x_start:0;
  mfcc_update_counter++;
  if(mfcc_update_counter==10) {
    lcd.FillRect(0, screen_size_y/3, screen_size_x, screen_size_y/3);
    for (int i=0;i<10*kws->num_frames;i++) {
      for (int j=0;j<kws->num_mfcc_features;j++) {
        for (int x=0;x<x_step;x++) {
          for (int y=0;y<y_step;y++) {
            lcd.DrawPixel(x_start+i*x_step+x,100+j*y_step+y, mfcc_plot_buffer[i*kws->num_mfcc_features+j]);
          }
        }
      }
    }
  mfcc_update_counter=0;
  }
}

uint32_t calculate_rgb(int min, int max, int value) {
  uint32_t ret = 0xFF000000;
  int mid_point = (min + max) / 2;
  int range = (max - min);
  if (value >= mid_point) {
    uint32_t delta = (value - mid_point)*512 / range;
    if (delta > 255) {  delta = 255;  }
    ret = ret | (delta << 16);
    ret = ret | ( (255-delta) << 8 );  
  } else {
    int delta = value*512 / range;
    if (delta > 255) {  delta = 255;  }
    ret = ret | (delta << 8);
    ret = ret | (255 - delta);
  }
  return ret;
}

void plot_waveform()
{

  int stride = (kws->audio_block_size/screen_size_x);
  int y_center = screen_size_y/6;
  int audio_magnitude = y_center;
  lcd.FillRect(0,0,screen_size_x,screen_size_y/3);
  for(int i=0;i<screen_size_x;i++)
  {
    audio_magnitude = y_center + (int)(kws->audio_buffer[(kws->frame_len-kws->frame_shift)+i*stride]/8);
    if (audio_magnitude < 0)
      audio_magnitude = 0;
    if (audio_magnitude > 2*y_center)
      audio_magnitude = 2*y_center - 1;
    audio_plot_buffer[i] = audio_magnitude;
  }
  lcd.SetTextColor(LCD_COLOR_ARM_DARK);
  for(int i=0;i<screen_size_x-1;i++)
    lcd.DrawLine(i,audio_plot_buffer[i],i+1,audio_plot_buffer[i+1]);
  lcd.SetTextColor(LCD_COLOR_WHITE);
    
}

