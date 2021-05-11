[![Alt text](../images/TFLite-01-thumbnail.png "TensorFlow Lite C++")](https://youtu.be/tdSaPdDqyRc)
### This video is the part of the Tensorflow Lite C++ series. In this video, I cover TensorFlow Lite C++ Installation in Mac Os.

<!-- row 6 -->

- ### Follow the steps given in the above video to install the TensorFlow lite library, or I already build it for Mac OS get it from the root directory tflite-dist.
- ### To run the above code 
  ```
  brew install cmake
  git clone https://github.com/karthickai/tflite.git
  cd tflite/03_Mac_Installation
  mkdir build && cd build
  cmake ..
  make
  ./TFLiteCheck YOUR_MODEL.tflite
  ```
