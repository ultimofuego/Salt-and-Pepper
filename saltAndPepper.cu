#include <cuda_runtime.h>
#include <iostream>
#include <ctime>

#include "EBMP/EasyBMP.h"
#include <algorithm> 

// 2D float texture
texture<float, cudaTextureType2D, cudaReadModeElementType> texRefR;
texture<float, cudaTextureType2D, cudaReadModeElementType> texRefG;
texture<float, cudaTextureType2D, cudaReadModeElementType> texRefB;

using namespace std;

#define checkCudaErrors(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true)
{
	if (code != cudaSuccess)
	{
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

void saveImage(float* imageR, float* imageG, float* imageB, int height, int width, bool method) {
	BMP Output;
	Output.SetSize(width, height);

	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			RGBApixel pixel;
			pixel.Red = imageR[i * width + j];
			pixel.Green = imageG[i * width + j];
			pixel.Blue = imageB[i * width + j];
			Output.SetPixel(j, i, pixel);
		}
	}
	if (method)
		Output.WriteToFile("CatGPUout.bmp");
	else
		Output.WriteToFile("CatCPUout.bmp");

}

void noiseImg(float* imageR,float* imageG, float* imageB, int height, int width, int per) {
	BMP Output;
	Output.SetSize(width, height);

	int countOfPixels = int(height * width / 100 * per);

	while (countOfPixels > 0) {
		int i = rand() % height;
		int j = rand() % width;
		int r = rand() % 2;
		int g = rand() % 2;
		int b = rand() % 2;

		if (r == 1)
			imageR[i * width + j] = 255;
		else
			imageR[i * width + j] = 0;

		if (g == 1)
			imageG[i * width + j] = 255;
		else
			imageG[i * width + j] = 0;

		if (b == 1)
			imageB[i * width + j] = 255;
		else
			imageB[i * width + j] = 0;

		countOfPixels--;
	}


	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			RGBApixel pixel;
			pixel.Red = imageR[i * width + j];
			pixel.Green = imageG[i * width + j];
			pixel.Blue = imageB[i * width + j];
			Output.SetPixel(j, i, pixel);
		}
	}
	Output.WriteToFile("catNoise.bmp");

}

void bubble(float* window,int n,int m) {
	bool swapped = true;
	int t = 0;
	int tmp;

	while (swapped) {
		swapped = false;
		t++;
		for (int i = 0; i < m * n - t; i++) {
			if (window[i] > window[i + 1]) {
				tmp = window[i];
				window[i] = window[i + 1];
				window[i + 1] = tmp;
				swapped = true;
			}
		}
	}
}

__device__
void bubbleG(float* window, int n, int m) {
	bool swapped = true;
	int t = 0;
	int tmp;

	while (swapped) {
		swapped = false;
		t++;
		for (int i = 0; i < m * n - t; i++) {
			if (window[i] > window[i + 1]) {
				tmp = window[i];
				window[i] = window[i + 1];
				window[i + 1] = tmp;
				swapped = true;
			}
		}
	}
}

void medianFilterCPU(float* imageR, float* imageG, float* imageB, float* resaultR, float* resaultG, float* resaultB, int height, int width)
{
	//mask3x3
	int m = 3;
	int n = 3;
	int mean = m * n / 2;
	int pad = m / 2;

	float* expandImageArrayR = (float*)calloc((height + 2 * pad) * (width + 2 * pad), sizeof(float));
	float* expandImageArrayG = (float*)calloc((height + 2 * pad) * (width + 2 * pad), sizeof(float));
	float* expandImageArrayB = (float*)calloc((height + 2 * pad) * (width + 2 * pad), sizeof(float));

	for (int j = 0; j < height; j++) {
		for (int i = 0; i < width; i++) {
			expandImageArrayR[(j + pad) * (width + 2 * pad) + i + pad] = imageR[j * width + i];
			expandImageArrayG[(j + pad) * (width + 2 * pad) + i + pad] = imageG[j * width + i];
			expandImageArrayB[(j + pad) * (width + 2 * pad) + i + pad] = imageB[j * width + i];
		}
	}

	for (int j = 0; j < height; j++) {
		for (int i = 0; i < width; i++) {
			float* windowR = (float*)calloc(m * n, sizeof(float));
			float* windowG = (float*)calloc(m * n, sizeof(float));
			float* windowB = (float*)calloc(m * n, sizeof(float));

			for (int k = 0; k < m; k++) {
				for (int t = 0; t < n; t++) {
					windowR[k * n + t] = expandImageArrayR[j * (width + 2 * pad) + i + k * (width + 2 * pad) + t];
					windowG[k * n + t] = expandImageArrayG[j * (width + 2 * pad) + i + k * (width + 2 * pad) + t];
					windowB[k * n + t] = expandImageArrayB[j * (width + 2 * pad) + i + k * (width + 2 * pad) + t];
				}
			}

			//sort
			bubble(windowR, n, m);
			bubble(windowG, n, m);
			bubble(windowB, n, m);


			resaultR[j * width + i] = windowR[mean];
			resaultG[j * width + i] = windowG[mean];
			resaultB[j * width + i] = windowB[mean];
		}
	}

}

__global__ void myFilter(float* outputR, float* outputG, float* outputB, int imageWidth, int imageHeight) {

	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;

	// mask 3x3
	float windowR[9];
	float windowG[9];
	float windowB[9];
	int m = 3;
	int n = 3;
	int mean = m * n / 2;
	int pad = m / 2;

	for (int i = -pad; i <= pad; i++) {
		for (int j = -pad; j <= pad; j++) {
			windowR[(i + pad) * n + j + pad] = tex2D(texRefR, col + j, row + i);
			windowG[(i + pad) * n + j + pad] = tex2D(texRefG, col + j, row + i);
			windowB[(i + pad) * n + j + pad] = tex2D(texRefB, col + j, row + i);
		}
	}

	//sort
	bubbleG(windowR, n, m);
	bubbleG(windowG, n, m);
	bubbleG(windowB, n, m);

	outputR[row * imageWidth + col] = windowR[mean];
	outputG[row * imageWidth + col] = windowG[mean];
	outputB[row * imageWidth + col] = windowB[mean];
}



int main(void)
{
	int nIter = 100;
	BMP Image;
	Image.ReadFromFile("cat500x375.bmp");
	int height = Image.TellHeight();
	int width = Image.TellWidth();

	float* imageArrayR = (float*)calloc(height * width, sizeof(float));
	float* imageArrayG = (float*)calloc(height * width, sizeof(float));
	float* imageArrayB = (float*)calloc(height * width, sizeof(float));
	float* outputCPUr = (float*)calloc(height * width, sizeof(float));
	float* outputCPUg = (float*)calloc(height * width, sizeof(float));
	float* outputCPUb = (float*)calloc(height * width, sizeof(float));

	float* outputGPUr = (float*)calloc(height * width, sizeof(float));
	float* outputDeviceR;
	float* outputGPUg = (float*)calloc(height * width, sizeof(float));
	float* outputDeviceG;
	float* outputGPUb = (float*)calloc(height * width, sizeof(float));
	float* outputDeviceB;


	for (int j = 0; j < Image.TellHeight(); j++) {
		for (int i = 0; i < Image.TellWidth(); i++) {
			imageArrayR[j * width + i] = Image(i, j)->Red;
			imageArrayG[j * width + i] = Image(i, j)->Green;
			imageArrayB[j * width + i] = Image(i, j)->Blue;
		}
	}

	noiseImg(imageArrayR,imageArrayG,imageArrayB, height, width, 5);

	unsigned int start_time = clock();

	for (int j = 0; j < nIter; j++) {
		medianFilterCPU(imageArrayR,imageArrayG,imageArrayB, outputCPUr, outputCPUg, outputCPUb, height, width);
	}

	unsigned int elapsedTime = clock() - start_time;
	float msecPerMatrixMulCpu = elapsedTime / nIter;

	cout << "CPU time: " << msecPerMatrixMulCpu << endl;

	int device_count = 0;
	cudaGetDeviceCount(&device_count);

	// Allocate CUDA array in device memory

			//Returns a channel descriptor with format f and number of bits of each component x, y, z, and w
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
	cudaArray* cu_arr;

	checkCudaErrors(cudaMallocArray(&cu_arr, &channelDesc, width, height));
	checkCudaErrors(cudaMemcpyToArray(cu_arr, 0, 0, imageArrayR, height * width * sizeof(float), cudaMemcpyHostToDevice));	// set texture parameters
	texRefR.addressMode[0] = cudaAddressModeClamp;
	texRefR.addressMode[1] = cudaAddressModeClamp;
	texRefR.filterMode = cudaFilterModePoint;


	// Bind the array to the texture
	cudaBindTextureToArray(texRefR, cu_arr, channelDesc);

	cudaArray* cu_arrG;

	checkCudaErrors(cudaMallocArray(&cu_arrG, &channelDesc, width, height));
	checkCudaErrors(cudaMemcpyToArray(cu_arrG, 0, 0, imageArrayG, height * width * sizeof(float), cudaMemcpyHostToDevice));	// set texture parameters
	texRefG.addressMode[0] = cudaAddressModeClamp;
	texRefG.addressMode[1] = cudaAddressModeClamp;
	texRefG.filterMode = cudaFilterModePoint;


	// Bind the array to the texture
	cudaBindTextureToArray(texRefG, cu_arrG, channelDesc);

	cudaArray* cu_arrB;

	checkCudaErrors(cudaMallocArray(&cu_arrB, &channelDesc, width, height));
	checkCudaErrors(cudaMemcpyToArray(cu_arrB, 0, 0, imageArrayB, height * width * sizeof(float), cudaMemcpyHostToDevice));	// set texture parameters
	texRefB.addressMode[0] = cudaAddressModeClamp;
	texRefB.addressMode[1] = cudaAddressModeClamp;
	texRefB.filterMode = cudaFilterModePoint;


	// Bind the array to the texture
	cudaBindTextureToArray(texRefB, cu_arrB, channelDesc);

	checkCudaErrors(cudaMalloc(&outputDeviceR, height * width * sizeof(float)));
	checkCudaErrors(cudaMalloc(&outputDeviceG, height * width * sizeof(float)));
	checkCudaErrors(cudaMalloc(&outputDeviceB, height * width * sizeof(float)));

		dim3 threadsPerBlock(32, 32);
		dim3 blocksPerGrid((width + threadsPerBlock.x - 1) / threadsPerBlock.x,
			(height + threadsPerBlock.y - 1) / threadsPerBlock.y);

		cudaEvent_t start;
		cudaEvent_t stop;
		checkCudaErrors(cudaEventCreate(&start));
		checkCudaErrors(cudaEventCreate(&stop));

		// start record
		checkCudaErrors(cudaEventRecord(start, 0));

		for (int j = 0; j < nIter; j++) {
			myFilter << <blocksPerGrid, threadsPerBlock >> > (outputDeviceR, outputDeviceG, outputDeviceB, width, height);
		}

		// stop record
		checkCudaErrors(cudaEventRecord(stop, 0));

		// wait end of event
		checkCudaErrors(cudaEventSynchronize(stop));

		float msecTotal = 0.0f;
		checkCudaErrors(cudaEventElapsedTime(&msecTotal, start, stop));

		float msecPerMatrixMul = msecTotal / nIter;

		cout << "GPU time: " << msecPerMatrixMul << endl;

		cudaDeviceSynchronize();

		checkCudaErrors(cudaMemcpy(outputGPUr, outputDeviceR, height * width * sizeof(float), cudaMemcpyDeviceToHost));
		checkCudaErrors(cudaMemcpy(outputGPUg, outputDeviceG, height* width * sizeof(float), cudaMemcpyDeviceToHost));
		checkCudaErrors(cudaMemcpy(outputGPUb, outputDeviceB, height* width * sizeof(float), cudaMemcpyDeviceToHost));
		
		cudaDeviceSynchronize();

		saveImage(outputGPUr, outputGPUg, outputGPUb, height, width, true);
		saveImage(outputCPUr, outputCPUg, outputCPUb, height, width, false);

		checkCudaErrors(cudaFreeArray(cu_arr));
		checkCudaErrors(cudaFree(outputDeviceR));
		checkCudaErrors(cudaFreeArray(cu_arrG));
		checkCudaErrors(cudaFree(outputDeviceG));
		checkCudaErrors(cudaFreeArray(cu_arrB));
		checkCudaErrors(cudaFree(outputDeviceB));
	
	return 0;
}

