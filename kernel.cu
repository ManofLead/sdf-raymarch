
#include "kernel.cuh"



int main()
{
	dim3 threads(8, 8);
	dim3 blocks(1920 / 8 + 1, 1080 / 8 + 1);
	cam camera;
	camera.position = make_float3(30, 30, 30);
	camera.direction = normalize(make_float3(-30, -30, -30));
	camera.right = normalize(cross(camera.direction, make_float3(0, 1, 0)));
	camera.up = normalize(cross(camera.right, camera.direction));
	camera.fov = 1.0f / std::tan(((180.0-70.0)/360.0) * 3.1415926);
	float* deviceImage;
	cudaMalloc(&deviceImage, 4 * 1920 * 1080 * sizeof(float));

	//render
	render<<<blocks, threads>>> (1920, 1080, deviceImage, camera);

	//move image to host, write to file as png
	float* hostImage = new float[4 * 1920 * 1080];
	cudaMemcpy(hostImage, deviceImage, 4 * 1920 * 1080 * sizeof(float), cudaMemcpyDeviceToHost);
	savePNG("G:/projects/sdf-raymarch/render/render.png", 1920, 1080, hostImage);

}

//save array of rgba floats to rgba8888 png
void savePNG(std::string path, int width, int height, const float rgba[]) {
	std::vector<unsigned char> out;
	out.resize(4 * width * height);
	for (int i = 0; i < width * height; i++) {
		out[i * 4 + 0] = (unsigned char)std::fmax(std::fmin(rgba[i * 4 + 0] * 255, 255), 0);
		out[i * 4 + 1] = (unsigned char)std::fmax(std::fmin(rgba[i * 4 + 1] * 255, 255), 0);
		out[i * 4 + 2] = (unsigned char)std::fmax(std::fmin(rgba[i * 4 + 2] * 255, 255), 0);
		out[i * 4 + 3] = (unsigned char)std::fmax(std::fmin(rgba[i * 4 + 3] * 255, 255), 0);
	}
	unsigned error = lodepng::encode(path, out, width, height);
	if (error) std::cout << lodepng_error_text(error) << std::endl;
}

__global__ void render(int width, int height, float* result, cam camera) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x >= width || y >= height) return;

	float2 cam_offset = make_float2((x / float(width) - 0.5) * 2.0, ((y / float(height) - 0.5) * -2.0) * float(height)/float(width));
	float3 raydir = normalize(camera.right * cam_offset.x + camera.up * cam_offset.y + camera.direction * camera.fov);
	float3 colour = ray_march(camera.position, raydir);

	result[x * 4 + y * 4 * width + 0] = colour.x;
	result[x * 4 + y * 4 * width + 1] = colour.y;
	result[x * 4 + y * 4 * width + 2] = colour.z;
	result[x * 4 + y * 4 * width + 3] = 255;
}

__device__ float3 ray_march(float3 origin, float3 direction) {
	const int raymarch_max_steps = 32;
	const float raymarch_min_distance = 0.002;
	const float raymarch_max_distance = 10000.0;

	float total_distance = getSdf(origin);
	for (int i = 0; i < raymarch_max_steps; ++i) {
		float3 current_position = origin + (direction * total_distance);
		float step_distance = getSdf(current_position);
		if (step_distance < raymarch_min_distance) {
			return get_normal(current_position);
		}
		total_distance += step_distance;

		//cut off when suitably far away
		if (total_distance > raymarch_max_distance) {
			break;
		}
	}
	return make_float3(0.5, 0.5, 0.5);
}

__device__ float3 get_normal(float3 origin) {
	const float step = 0.001;
	float gradient_x = getSdf(origin + make_float3(step, 0, 0)) - getSdf(origin - make_float3(step, 0, 0));
	float gradient_y = getSdf(origin + make_float3(0, step, 0)) - getSdf(origin - make_float3(0, step, 0));
	float gradient_z = getSdf(origin + make_float3(0, 0, step)) - getSdf(origin - make_float3(0, 0, step));
	return normalize(make_float3(gradient_x, gradient_y, gradient_z));
}