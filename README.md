# Salt-and-Pepper
Salt-and-pepper noise is a form of noise sometimes seen on images. It is also known as impulse noise. This noise can be caused by sharp and sudden disturbances in the image signal. It presents itself as sparsely occurring white and black pixels.

An effective noise reduction method for this type of noise is a median filter or a morphological filter. For reducing either salt noise or pepper noise, but not both, a contraharmonic mean filter can be effective.

## Prerequisites
1. Microsoft visual studio 19
2. Nvidia GPU (cuda SUPPORT)
3. CUDA Toolkit 11
4. EasyBMP
## Build and Run
1. Make new CUDA-project.
2. Include in the project "saltAndPepper.cu".
## System configuration
| Name  | Values  |
|-------|---------|
| CPU  | Intel® Pentium® G3430 (2x3.30 GHz) |
| RAM  | 4 GB DDR3 |
| GPU  | GeForce GTX 750 Ti 2GB |
| OS   | Windows 10 64-bit  |
## Results
<img src="https://github.com/ultimofuego/Salt-and-Pepper/blob/master/catNoise.bmp" /> |
------------ |
Input 500x375 + noise 5%

<img src="https://github.com/ultimofuego/Salt-and-Pepper/blob/master/CatGPUout.bmp" /> | <img src="https://github.com/ultimofuego/Salt-and-Pepper/blob/master/CatCPUout.bmp" />
------------ | ------------- 
Output GPU 500x375 | Output CPU 500x375

Average results after 100 times of runs.

|    Input size  |          CPU        |         GPU       | Acceleration |
|-------------|--------------------|-------------------|--------------|
| 250x188   |40 ms               | 0.44 ms            |    90.90      |
| 500x375   |152 ms               | 1.71 ms            |    88.89      |
| 1000x750   |937 ms              | 5.92 ms             |    158.28      |
| 2000x1500   |2414 ms              | 25.67 ms             |    94.04      |
