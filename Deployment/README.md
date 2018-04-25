# Keyword Spotting on Arm Cortex-M boards.
The first step in deploying the trained keyword spotting models on microcontrollers is quantization, which is described [here](Quant_guide.md). This directory consists of example codes and steps for running a quantized DNN model on any Cortex-M board using [mbed-cli](https://github.com/ARMmbed/mbed-cli) and [CMSIS-NN](https://github.com/ARM-software/CMSIS_5) library. It also consists of an example of integration of the KWS model onto a Cortex-M development board with an on-board microphone to demonstrate keyword spotting on live audio data. 

## Get the CMSIS-NN library and install mbed-cli
Clone [CMSIS-5](https://github.com/ARM-software/CMSIS_5) library, which consists of the optimized neural network kernels for Cortex-M.
```bash
cd Deployment
git clone https://github.com/ARM-software/CMSIS_5.git
```
Install [mbed-cli](https://github.com/ARMmbed/mbed-cli) and its python dependencies.
```bash
pip install mbed-cli
```
## Build and run a simple KWS inference 
In this example, the KWS inference is run on the audio data provided through a .h file.
First create a new project and install any python dependencies prompted when project is created for the first time after the installation of mbed-cli.
```bash
mbed new kws_simple_test --mbedlib 
```
Fetch the required mbed libraries for compilation.
```bash
cd kws_simple_test
mbed deploy
```
Compile the code for the mbed board (for example NUCLEO\_F411RE).
```bash
mbed compile -m NUCLEO_F411RE -t GCC_ARM --source . \
  --source ../Source/KWS --source ../Source/NN --source ../Source/MFCC \
  --source ../Source/local_NN --source ../Examples/simple_test \
  --source ../CMSIS_5/CMSIS/NN/Include --source ../CMSIS_5/CMSIS/NN/Source \
  --source ../CMSIS_5/CMSIS/DSP/Include --source ../CMSIS_5/CMSIS/DSP/Source \
  --source ../CMSIS_5/CMSIS/Core/Include \
  --profile ../release_O3.json -j 8 
```
Copy the binary (.bin) to the board (Make sure the board is detected and mounted). Open a serial terminal (e.g. putty or minicom) and see the final classification output on screen. 
```bash
cp ./BUILD/NUCLEO_F411RE/GCC_ARM/kws_simple_test.bin /media/<user>/NODE_F411RE/
sudo minicom
```
## Run KWS inference on live audio on [STM32F746NG development kit](http://www.st.com/en/evaluation-tools/32f746gdiscovery.html)
This example runs keyword spotting inference on live audio captured using the on-board microphones on the STM32F746NG discovery kit. When performing keyword spotting on live audio data with multiple noise sources, outputs are typically averaged over a specified window to generate smooth predictions. The averaging window length and the detection threshold (which may also be different for each keyword) are two key parameters in determining the overall keyword spotting accuracy and user experience.
```bash
mbed new kws_realtime_test --create-only
cd kws_realtime_test
cp ../Examples/realtime_test/mbed_libs/*.lib .
mbed deploy
mbed compile -m DISCO_F746NG -t GCC_ARM \
  --source . --source ../Source --source ../Examples/realtime_test \
  --source ../CMSIS_5/CMSIS/NN/Include --source ../CMSIS_5/CMSIS/NN/Source \
  --source ../CMSIS_5/CMSIS/DSP/Include --source ../CMSIS_5/CMSIS/DSP/Source \
  --source ../CMSIS_5/CMSIS/Core/Include \
  --profile ../release_O3.json -j 8
cp ./BUILD/DISCO_F746NG/GCC_ARM/kws_realtime_test.bin /media/<user>/DIS_F746NG/
```
**Note:** The examples provided use floating point operations for MFCC feature extraction, but it should be possible to convert them to fixed-point operations for deploying on microcontrollers that do not have dedicated floating point units.
