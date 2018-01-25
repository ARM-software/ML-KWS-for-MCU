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

#include "kws.h"
#include "AUDIO_DISCO_F746NG.h"
#include "LCD_DISCO_F746NG.h"

#define LCD_COLOR_ARM_BLUE ((uint32_t) 0xFF00C1DE)
#define LCD_COLOR_ARM_DARK ((uint32_t) 0xFF333E48)

AUDIO_DISCO_F746NG audio;
LCD_DISCO_F746NG lcd;
Serial pc(USBTX, USBRX);
Timer T;

char lcd_output_string[256];
char output_class[12][8] = {"Silence", "Unknown","yes","no","up","down","left","right","on","off","stop","go"};
int current_pixel_location=0;
uint32_t mfcc_plot_buffer[NUM_MFCC_COEFFS*NUM_FRAMES*10];
int mfcc_update_counter=0;


/*
 * The audio recording works with two windows, each of size 80 ms. 
 * The data for each window will be tranfered by the DMA, which sends
 * sends an interrupt after the transfer is completed.
 */

/* AUDIO_BLOCK_SIZE is the number of audio samples in each recording window */
#define AUDIO_BLOCK_SIZE   (2*FRAME_LEN)

int16_t audio_io_buffer[AUDIO_BLOCK_SIZE*8]; //2 (L/R) channels x 2 input/output x 2 for ping-pong buffer
int16_t audio_buffer[AUDIO_BLOCK_SIZE];

int16_t* AUDIO_BUFFER_IN = audio_io_buffer;
int16_t* AUDIO_BUFFER_OUT = (AUDIO_BUFFER_IN + (AUDIO_BLOCK_SIZE * 4));

q7_t scratch_buffer[SCRATCH_BUFFER_SIZE];
KWS *kws;

static uint8_t SetSysClock_PLL_HSE_200MHz();
void run_kws();
void plot_mfcc(q7_t* mfcc_buffer);
void plot_waveform();
uint32_t calculate_rgb(int min, int max, int value);

int main()
{
  SetSysClock_PLL_HSE_200MHz();
  pc.baud(9600);

  kws = new KWS(audio_buffer,scratch_buffer);

  lcd.Clear(LCD_COLOR_ARM_BLUE);
  lcd.SetBackColor(LCD_COLOR_ARM_BLUE);
  lcd.DisplayStringAt(0, LINE(1), (uint8_t *)"Keyword Spotting Example", CENTER_MODE);
  wait(1);
  lcd.Clear(LCD_COLOR_ARM_BLUE);
  lcd.SetBackColor(LCD_COLOR_ARM_BLUE);
  lcd.SetTextColor(LCD_COLOR_WHITE);

  int size_x, size_y;
  size_x = lcd.GetXSize();
  size_y = lcd.GetYSize();
  lcd.FillRect(0, 0, size_x, size_y/3);

  /* Initialize buffers */
  memset(AUDIO_BUFFER_IN, 0, AUDIO_BLOCK_SIZE*8);
  memset(AUDIO_BUFFER_OUT, 0, AUDIO_BLOCK_SIZE*8);

  /* May need to adjust volume to get better accuracy/user-experience */
  audio.IN_SetVolume(80);

  /* Start Recording */
  audio.IN_Record((uint16_t*)AUDIO_BUFFER_IN, AUDIO_BLOCK_SIZE * 4);

  /* Start Playback for listening to what is being classified */
  audio.OUT_SetAudioFrameSlot(CODEC_AUDIOFRAME_SLOT_02);
  audio.OUT_Play((uint16_t*)AUDIO_BUFFER_OUT, AUDIO_BLOCK_SIZE * 8);

  T.start();

  while (1) {
  /* A dummy loop to wait for the interrupts. Feature extraction and
     neural network inference are done in the interrupt service routine. */
  }
}

/*
 * Manages the DMA Transfer complete interrupt.
 */
void BSP_AUDIO_IN_TransferComplete_CallBack(void)
{
  arm_copy_q7((q7_t *)AUDIO_BUFFER_IN + AUDIO_BLOCK_SIZE*4, (q7_t *)AUDIO_BUFFER_OUT + AUDIO_BLOCK_SIZE*4, AUDIO_BLOCK_SIZE*4);
  // copy the new recording data 
  for (int i=0;i<AUDIO_BLOCK_SIZE;i++) {
    audio_buffer[i] = AUDIO_BUFFER_IN[2*AUDIO_BLOCK_SIZE+i*2];
  }
  run_kws();
  return;
}

/*
 * Manages the DMA Half Transfer complete interrupt.
 */
void BSP_AUDIO_IN_HalfTransfer_CallBack(void)
{
  arm_copy_q7((q7_t *)AUDIO_BUFFER_IN, (q7_t *)AUDIO_BUFFER_OUT, AUDIO_BLOCK_SIZE*4);
  // copy the new recording data 
  for (int i=0;i<AUDIO_BLOCK_SIZE;i++) {
    audio_buffer[i] = AUDIO_BUFFER_IN[i*2];
  }
  run_kws();
  return;
}

void run_kws()
{
 
  //Averaging window for smoothing out the output predictions
  int averaging_window_len = 3;  //i.e. average over 3 inferences or 240ms
  int detection_threshold = 70;  //in percent

  int start = T.read_us();
  kws->extract_features(2); //extract mfcc features
  kws->classify();	    //classify using dnn
  kws->average_predictions(averaging_window_len);

  plot_waveform();
  plot_mfcc(kws->mfcc_buffer);
  int end = T.read_us();
  int max_ind = kws->get_top_detection(kws->averaged_output);
  if(kws->averaged_output[max_ind]>detection_threshold*128/100)
    sprintf(lcd_output_string,"%d%% %s",((int)kws->averaged_output[max_ind]*100/128),output_class[max_ind]);
  lcd.ClearStringLine(8);
  lcd.DisplayStringAt(0, LINE(8), (uint8_t *) lcd_output_string, CENTER_MODE);

}

void plot_mfcc(q7_t* mfcc_buffer)
{
  memcpy(mfcc_plot_buffer, mfcc_plot_buffer+2*NUM_MFCC_COEFFS, 4*NUM_MFCC_COEFFS*(10*NUM_FRAMES-2));

  int size_x, size_y;
  size_x = lcd.GetXSize();
  size_y = lcd.GetYSize();

  int x_step = 1;
  int y_step = 6;

  uint32_t* pBuffer = mfcc_plot_buffer + NUM_MFCC_COEFFS*(10*NUM_FRAMES-2);
  int sum = 0;;

  for (int i=0;i<2;i++) {
    for (int j=0;j<NUM_MFCC_COEFFS;j++) {
      int value = mfcc_buffer[(NUM_MFCC_COEFFS*(NUM_FRAMES-2))+i*NUM_MFCC_COEFFS+j];
      uint32_t RGB  = calculate_rgb(-128, 127, value*4);
      sum += std::abs(value);
      pBuffer[i*NUM_MFCC_COEFFS+j] = RGB;
    }
  }
  mfcc_update_counter++;
  if(mfcc_update_counter==10) {
    lcd.FillRect(0, size_y/3, size_x, size_y/3);
    for (int i=0;i<10*NUM_FRAMES;i++) {
      for (int j=0;j<NUM_MFCC_COEFFS;j++) {
        for (int x=0;x<x_step;x++) {
          for (int y=0;y<y_step;y++) {
            lcd.DrawPixel(120+i*x_step+x,90+j*y_step+y, mfcc_plot_buffer[i*NUM_MFCC_COEFFS+j]);
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
  
  int size_x, size_y;
  size_x = lcd.GetXSize();
  size_y = lcd.GetYSize();

  int x_width = 128*3/2;

  int x_start = size_x/2 - x_width*current_pixel_location;
  lcd.FillRect(x_start, 0, x_width, 1*size_y/3);
  //lcd.FillRect(0, 0, size_x, 1*size_y/3);
  current_pixel_location = 1 - current_pixel_location;
  int y_center = size_y/6;

  int stride = 2 * (AUDIO_BLOCK_SIZE / x_width / 2);

  for (int i=0;i<x_width;i++) {
    int audio_magnitude = y_center+(int)(audio_buffer[i*stride+1]/8);
    if (audio_magnitude < 0)  {  audio_magnitude = 0;  }
    if (audio_magnitude > 2*y_center) { audio_magnitude = 2*y_center - 1;  }

    lcd.DrawPixel(x_start+i,audio_magnitude, LCD_COLOR_ARM_DARK); 
  }
}

static uint8_t SetSysClock_PLL_HSE_200MHz()
{
  RCC_ClkInitTypeDef RCC_ClkInitStruct;
  RCC_OscInitTypeDef RCC_OscInitStruct;

  // Enable power clock  
  __PWR_CLK_ENABLE();
  
  // Enable HSE oscillator and activate PLL with HSE as source
  RCC_OscInitStruct.OscillatorType      = RCC_OSCILLATORTYPE_HSE;
  RCC_OscInitStruct.HSEState            = RCC_HSE_ON; /* External xtal on OSC_IN/OSC_OUT */

  // Warning: this configuration is for a 25 MHz xtal clock only
  RCC_OscInitStruct.PLL.PLLState        = RCC_PLL_ON;
  RCC_OscInitStruct.PLL.PLLSource       = RCC_PLLSOURCE_HSE;
  RCC_OscInitStruct.PLL.PLLM            = 25;            // VCO input clock = 1 MHz (25 MHz / 25)
  RCC_OscInitStruct.PLL.PLLN            = 400;           // VCO output clock = 400 MHz (1 MHz * 400)
  RCC_OscInitStruct.PLL.PLLP            = RCC_PLLP_DIV2; // PLLCLK = 200 MHz (400 MHz / 2)
  RCC_OscInitStruct.PLL.PLLQ            = 8;             // USB clock = 50 MHz (400 MHz / 8)
  
  if (HAL_RCC_OscConfig(&RCC_OscInitStruct) != HAL_OK)
  {
    return 0; // FAIL
  }

  // Activate the OverDrive to reach the 216 MHz Frequency
  if (HAL_PWREx_EnableOverDrive() != HAL_OK)
  {
    return 0; // FAIL
  }
  
  // Select PLL as system clock source and configure the HCLK, PCLK1 and PCLK2 clocks dividers
  RCC_ClkInitStruct.ClockType      = (RCC_CLOCKTYPE_SYSCLK | RCC_CLOCKTYPE_HCLK | RCC_CLOCKTYPE_PCLK1 | RCC_CLOCKTYPE_PCLK2);
  RCC_ClkInitStruct.SYSCLKSource   = RCC_SYSCLKSOURCE_PLLCLK; // 200 MHz
  RCC_ClkInitStruct.AHBCLKDivider  = RCC_SYSCLK_DIV1;         // 200 MHz
  RCC_ClkInitStruct.APB1CLKDivider = RCC_HCLK_DIV4;           //  50 MHz
  RCC_ClkInitStruct.APB2CLKDivider = RCC_HCLK_DIV2;           // 100 MHz
  
  if (HAL_RCC_ClockConfig(&RCC_ClkInitStruct, FLASH_LATENCY_7) != HAL_OK)
  {
    return 0; // FAIL
  }
  HAL_RCC_MCOConfig(RCC_MCO1, RCC_MCO1SOURCE_HSE, RCC_MCODIV_4);
  return 1; // OK
}

