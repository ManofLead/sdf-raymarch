#include "vec_utils.cuh"

__host__ __device__ float3 operator*(const float3& a, const float1& b)
{
	return make_float3(a.x * b.x, a.y * b.x, a.z * b.x);
}

__host__ __device__ float3 operator+(const float3& a, const float3& b)
{
	return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}