#include "sdf.cuh"


float __device__ sphere(float3 pos, float1 r) {
	return length(pos) - r.x;
}

float __device__ torus(float3 pos, float2 torus) {
	float2 q = make_float2(length(make_float2(pos.x, pos.z)) - torus.x, pos.y);
	return length(q) - torus.y;
}

float __device__ getSdf(float3 pos) {
	//return sphere(pos, make_float1(10));
	return torus(pos, make_float2(20, 5));
}