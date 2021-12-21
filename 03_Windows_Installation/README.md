### This video is the part of the Tensorflow Lite C++ series. In this video, I cover TensorFlow Lite C++ Installation in Windows 10.

<!-- row 6 -->

- ### Follow the steps given in the above video to install the TensorFlow lite library, or I already build it for windows get it from the root directory tflite-dist.
- ### To run the above code 
  ```
  git clone https://github.com/karthickai/tflite.git
  cd tflite/03_Windows_Installation
  mkdir build && cd build
  cmake ..
  #open TFLiteCheck.sln in Visual Studio 2019.
  #Build Release x64
  cd Release
  .\TFLiteCheck.exe ..\..\..\models\classification\mobilenet_v1_1.0_224_quant.tflite
  ```
