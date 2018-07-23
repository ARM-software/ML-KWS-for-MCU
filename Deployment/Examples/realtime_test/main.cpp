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
 * Description: End-to-end example code for running keyword spotting on 
 * STM32F746NG development kit (DISCO_F746NG in mbed-cli). The example is 
 * derived from https://os.mbed.com/teams/ST/code/DISCO-F746NG_AUDIO_demo
 */

#include "kws_f746ng.h"
#include "plot_utils.h"
#include "LCD_DISCO_F746NG.h"

LCD_DISCO_F746NG lcd;
Serial pc(USBTX, USBRX);
KWS_F746NG *kws;
Timer T;

char lcd_output_string[64];
char output_class[12][8] = {"Silence", "Unknown","yes","no","up","down",
                            "left","right","on","off","stop","go"};
// Tune the following three parameters to improve the detection accuracy
//  and reduce false positives
// Longer averaging window and higher threshold reduce false positives
//  but increase detection latency and reduce true positive detections.

// (recording_win*frame_shift) is the actual recording window size
int recording_win = 3; 
// Averaging window for smoothing out the output predictions
int averaging_window_len = 3;  
int detection_threshold = 90;  //in percent

void run_kws();

int main()
{
  pc.baud(9600);
  kws = new KWS_F746NG(recording_win,averaging_window_len);
  init_plot();
  kws->start_kws();

  T.start();

  while (1) {
  /* A dummy loop to wait for the interrupts. Feature extraction and
     neural network inference are done in the interrupt service routine. */
    __WFI();
  }
}


/*
 * The audio recording works with two ping-pong buffers.
 * The data for each window will be tranfered by the DMA, which sends
 * sends an interrupt after the transfer is completed.
 */

// Manages the DMA Transfer complete interrupt.
void BSP_AUDIO_IN_TransferComplete_CallBack(void)
{
  arm_copy_q7((q7_t *)kws->audio_buffer_in + kws->audio_block_size*4, (q7_t *)kws->audio_buffer_out + kws->audio_block_size*4, kws->audio_block_size*4);
  if(kws->frame_len != kws->frame_shift) {
    //copy the last (frame_len - frame_shift) audio data to the start
    arm_copy_q7((q7_t *)(kws->audio_buffer)+2*(kws->audio_buffer_size-(kws->frame_len-kws->frame_shift)), (q7_t *)kws->audio_buffer, 2*(kws->frame_len-kws->frame_shift));
  }
  // copy the new recording data 
  for (int i=0;i<kws->audio_block_size;i++) {
    kws->audio_buffer[kws->frame_len-kws->frame_shift+i] = kws->audio_buffer_in[2*kws->audio_block_size+i*2];
  }
  run_kws();
  return;
}

// Manages the DMA Half Transfer complete interrupt.
void BSP_AUDIO_IN_HalfTransfer_CallBack(void)
{
  arm_copy_q7((q7_t *)kws->audio_buffer_in, (q7_t *)kws->audio_buffer_out, kws->audio_block_size*4);
  if(kws->frame_len!=kws->frame_shift) {
    //copy the last (frame_len - frame_shift) audio data to the start
    arm_copy_q7((q7_t *)(kws->audio_buffer)+2*(kws->audio_buffer_size-(kws->frame_len-kws->frame_shift)), (q7_t *)kws->audio_buffer, 2*(kws->frame_len-kws->frame_shift));
  }
  // copy the new recording data 
  for (int i=0;i<kws->audio_block_size;i++) {
    kws->audio_buffer[kws->frame_len-kws->frame_shift+i] = kws->audio_buffer_in[i*2];
  }
  run_kws();
  return;
}

void run_kws()
{
  kws->extract_features();    //extract mfcc features
  kws->classify();	      //classify using dnn
  kws->average_predictions();
  plot_mfcc();
  plot_waveform();
  int max_ind = kws->get_top_class(kws->averaged_output);
  if(kws->averaged_output[max_ind]>detection_threshold*128/100)
    sprintf(lcd_output_string,"%d%% %s",((int)kws->averaged_output[max_ind]*100/128),output_class[max_ind]);
  lcd.ClearStringLine(8);
  lcd.DisplayStringAt(0, LINE(8), (uint8_t *) lcd_output_string, CENTER_MODE);
}

