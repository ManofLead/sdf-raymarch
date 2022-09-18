#pragma once
#include "cuda_runtime.h"

__host__ __device__ float3 operator*(const float3 &a, const float1 &b);
__host__ __device__ float3 operator+(const float3 &a, const float3 &b);


