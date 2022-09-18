#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "sdf.cuh"
#include "helper_math.h"
#include <stdio.h>
#include <iostream>
#include <cmath>

#include "lodepng.h"

struct cam {
	float3 position;
	float3 direction;
	float3 up;
	float3 right;
	float fov;
};

void savePNG(std::string path, int width, int height, const float rgba[]);

__global__ void render(int width, int height, float* result, cam camera);

float3 __device__ ray_march(float3 origin, float3 direction);

__device__ float3 get_normal(float3 origin);
