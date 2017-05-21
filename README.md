# Parallel-JPEG

## What is?

Parallel JPEG is a parallel implementation of baseline JPEG encoder and decoder. It exploits the SIMD nature of the methods in encoder and decoder.
The software uses CUDA platform to implement the functions on GPU. Current implementation is only upto encoder. The project is still in development.

## How to build?

### What you need:
* Visual studio community/express 2013/2015.
* OpenCV (ver > 3.0)
* CUDA toolkit (any version >= 7.0)

After installing everying:

* Add path in the PATH variable to include OpenCV bin directory (eg. C:\OpenCV-3.1.0\opencv\build\x64\vc12\bin ).
* Set the project properties to add both OpenCV and CUDA libraries and include directories. 

Build the project :)

