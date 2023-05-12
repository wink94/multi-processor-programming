
#include <iostream>
#include <vector>
#include <fstream>
#include <chrono>
#include "lodepng.h"
#include <CL/cl.hpp>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <chrono>
#include <algorithm>

using namespace std;

#define MATRIX_SIZE 100

/// STEP 1:
/// 


unsigned char rgbaToGrayscale(unsigned char r, unsigned char g, unsigned char b) {
	return 0.2126 * r + 0.7152 * g + 0.0722 * b;
}

void convertToGrayscaleNormalized(const unsigned char* rgba_image, unsigned char* grayscale_image, unsigned width, unsigned height) {
	for (unsigned i = 0; i < width * height; ++i) {
		unsigned index = i * 4;
		unsigned char r = rgba_image[index];
		unsigned char g = rgba_image[index + 1];
		unsigned char b = rgba_image[index + 2];

		grayscale_image[i] = rgbaToGrayscale(r, g, b);
	}
}

void resizeImage(const unsigned char* original_image, unsigned char* resized_image, unsigned original_width, unsigned original_height) {
	unsigned resized_width = original_width / 4;
	unsigned resized_height = original_height / 4;

	for (unsigned y = 0; y < resized_height; ++y) {
		for (unsigned x = 0; x < resized_width; ++x) {
			unsigned original_index = (y * 4 * original_width + x * 4) * 4;
			unsigned resized_index = (y * resized_width + x) * 4;

			for (int k = 0; k < 4; ++k) {
				resized_image[resized_index + k] = original_image[original_index + k];
			}
		}
	}
}


/// STEP 2
/// 

void Add_Matrix(int** matrix1, int** matrix2, int** result, int size) {
	for (int i = 0; i < size; ++i) {
		for (int j = 0; j < size; ++j) {
			result[i][j] = matrix1[i][j] + matrix2[i][j];
		}
	}
}

int addMatrixMain() {
	int** matrix1 = (int**)malloc(MATRIX_SIZE * sizeof(int*));
	int** matrix2 = (int**)malloc(MATRIX_SIZE * sizeof(int*));
	int** result = (int**)malloc(MATRIX_SIZE * sizeof(int*));

	for (int i = 0; i < MATRIX_SIZE; ++i) {
		matrix1[i] = (int*)malloc(MATRIX_SIZE * sizeof(int));
		matrix2[i] = (int*)malloc(MATRIX_SIZE * sizeof(int));
		result[i] = (int*)malloc(MATRIX_SIZE * sizeof(int));
	}

	for (int i = 0; i < MATRIX_SIZE; ++i) {
		for (int j = 0; j < MATRIX_SIZE; ++j) {
			matrix1[i][j] = i * j;
			matrix2[i][j] = i + j;
		}
	}

	clock_t start = clock();
	Add_Matrix(matrix1, matrix2, result, MATRIX_SIZE);
	clock_t end = clock();

	double execution_time = (double)(end - start) / CLOCKS_PER_SEC;
	printf("Execution time: %f seconds\n", execution_time);

	for (int i = 0; i < MATRIX_SIZE; ++i) {
		free(matrix1[i]);
		free(matrix2[i]);
		free(result[i]);
	}

	free(matrix1);
	free(matrix2);
	free(result);

	return 0;
}



const char* kernel_source = "__kernel void add_matrix(__global const float *A, __global const float *B, __global float *C, int N) {"
"  int gid_x = get_global_id(0);"
"  int gid_y = get_global_id(1);"
"  int index = gid_y * N + gid_x;"
"  C[index] = A[index] + B[index];"
"}";

void checkError(cl_int err, const char* operation) {
	if (err != CL_SUCCESS) {
		fprintf(stderr, "Error during %s: %d\n", operation, err);
		exit(1);
	}
}


void step2Opencl() {

	int size = 1024;
	int matrix_size = size * size;
	size_t matrix_bytes = matrix_size * sizeof(int);

	int* matrix1 = (int*)malloc(matrix_bytes);
	int* matrix2 = (int*)malloc(matrix_bytes);
	int* result = (int*)malloc(matrix_bytes);

	// Initialize the matrices
	for (int i = 0; i < size; i++) {
		for (int j = 0; j < size; j++) {
			int idx = i * size + j;
			matrix1[idx] = i * j;
			matrix2[idx] = i + j;
		}
	}


	// OpenCL setup
	cl_platform_id platform_id;
	cl_device_id device_id;
	cl_context context;
	cl_command_queue command_queue;
	cl_program program;
	cl_kernel kernel;
	cl_mem buffer_matrix1, buffer_matrix2, buffer_result;

	clGetPlatformIDs(1, &platform_id, NULL);
	clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_DEFAULT, 1, &device_id, NULL);
	context = clCreateContext(NULL, 1, &device_id, NULL, NULL, NULL);
	command_queue = clCreateCommandQueueWithProperties(context, device_id, 0, NULL);

	program = clCreateProgramWithSource(context, 1, &kernel_source, NULL, NULL);
	clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);
	kernel = clCreateKernel(program, "add_matrix", NULL);

	// Create buffers
	buffer_matrix1 = clCreateBuffer(context, CL_MEM_READ_ONLY, matrix_bytes, NULL, NULL);
	buffer_matrix2 = clCreateBuffer(context, CL_MEM_READ_ONLY, matrix_bytes, NULL, NULL);
	buffer_result = clCreateBuffer(context, CL_MEM_WRITE_ONLY, matrix_bytes, NULL, NULL);

	// Copy data to the buffers
	clEnqueueWriteBuffer(command_queue, buffer_matrix1, CL_TRUE, 0, matrix_bytes, matrix1, 0, NULL, NULL);
	clEnqueueWriteBuffer(command_queue, buffer_matrix2, CL_TRUE, 0, matrix_bytes, matrix2, 0, NULL, NULL);

	// Set the kernel arguments
	clSetKernelArg(kernel, 0, sizeof(cl_mem), &buffer_matrix1);
	clSetKernelArg(kernel, 1, sizeof(cl_mem), &buffer_matrix2);
	clSetKernelArg(kernel, 2, sizeof(cl_mem), &buffer_result);
	clSetKernelArg(kernel, 3, sizeof(int), &size);

	size_t global_size[] = { size, size };
	size_t local_size[] = { 16, 16 };

	// Execute the kernel
	cl_event kernel_event;
	clEnqueueNDRangeKernel(command_queue, kernel, 2, NULL, global_size, local_size, 0, NULL, &kernel_event);

	clEnqueueReadBuffer(command_queue, buffer_result, CL_TRUE, 0, matrix_bytes, result, 0, NULL, NULL);

	// Get the execution time
	cl_ulong start, end;
	clGetEventProfilingInfo(kernel_event, CL_PROFILING_COMMAND_START, sizeof(start), &start, NULL);
	clGetEventProfilingInfo(kernel_event, CL_PROFILING_COMMAND_END, sizeof(end), &end, NULL);
	double execution_time = (double)(end - start) * 1e-9;

	// Display the execution time
	printf("Execution time: %f seconds\n", execution_time);

	// Display the device information
	char device_name[256];
	clGetDeviceInfo(device_id, CL_DEVICE_NAME, sizeof(device_name), device_name, NULL);
	printf("Device used: %s\n", device_name);

	// Cleanup
	clReleaseMemObject(buffer_matrix1);
	clReleaseMemObject(buffer_matrix2);
	clReleaseMemObject(buffer_result);
	clReleaseKernel(kernel);
	clReleaseProgram(program);
	clReleaseCommandQueue(command_queue);
	clReleaseContext(context);

	free(matrix1);
	free(matrix2);
	free(result);


}

/// STEP 3
/// 
/// 

int getDeviceInfo() {
	cl_uint platform_count;
	clGetPlatformIDs(0, NULL, &platform_count);

	cl_platform_id* platforms = new cl_platform_id[platform_count];
	clGetPlatformIDs(platform_count, platforms, NULL);

	for (cl_uint i = 0; i < platform_count; i++) {
		cl_uint device_count;
		clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, 0, NULL, &device_count);

		cl_device_id* devices = new cl_device_id[device_count];
		clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, device_count, devices, NULL);

		for (cl_uint j = 0; j < device_count; j++) {
			char device_name[1024];
			char hardware_version[1024];
			char driver_version[1024];
			char opencl_version[1024];

			clGetDeviceInfo(devices[j], CL_DEVICE_NAME, 1024, device_name, NULL);
			clGetDeviceInfo(devices[j], CL_DEVICE_VERSION, 1024, hardware_version, NULL);
			clGetDeviceInfo(devices[j], CL_DRIVER_VERSION, 1024, driver_version, NULL);
			clGetDeviceInfo(devices[j], CL_DEVICE_OPENCL_C_VERSION, 1024, opencl_version, NULL);

			cl_uint compute_units;
			clGetDeviceInfo(devices[j], CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(compute_units), &compute_units, NULL);

			size_t max_work_item_dims;
			clGetDeviceInfo(devices[j], CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, sizeof(max_work_item_dims), &max_work_item_dims, NULL);

			std::cout << "Platform Count: " << platform_count << std::endl;
			std::cout << "Device Count on Platform: " << device_count << std::endl;
			std::cout << "Device Name: " << device_name << std::endl;
			std::cout << "Hardware Version: " << hardware_version << std::endl;
			std::cout << "Driver Version: " << driver_version << std::endl;
			std::cout << "OpenCL Version: " << opencl_version << std::endl;
			std::cout << "Parallel Compute Units: " << compute_units << std::endl;
			std::cout << "Max Work Item Dimensions: " << max_work_item_dims << std::endl;
		}

		delete[] devices;
	}

	delete[] platforms;
	return 0;
}

string get_file_string() {
	std::ifstream ifs("movingFilter.cl");
	return string((std::istreambuf_iterator<char>(ifs)),
		(std::istreambuf_iterator<char>()));
}

const char* kernel_source_grey_scale = R"(
__kernel void rgb_to_grayscale(__global uchar4* input, __global uchar* output)
{
    int gid_x = get_global_id(0);
    int gid_y = get_global_id(1);

    uchar4 rgba = input[gid_y * get_global_size(0) + gid_x];
    uchar gray = (uchar)(0.299f * rgba.x + 0.587f * rgba.y + 0.114f * rgba.z);
    output[gid_y * get_global_size(0) + gid_x] = gray;
}
)";

void step3Function() {
	const char* filename = "image3.png";
	const char* outFileGrey = "image_out_grey.png";
	const char* outFile = "image_out.png";
	unsigned error;
	unsigned char* image = 0;
	unsigned int width, height;
	unsigned char* newImage = 0;

	auto start_load_image = std::chrono::high_resolution_clock::now();
	error = lodepng_decode32_file(&image, &width, &height, filename);
	auto end_load_image = std::chrono::high_resolution_clock::now();

	if (error) {
		printf("error %u: %s\n", error, lodepng_error_text(error));
	}

	auto start_gray_scale = std::chrono::high_resolution_clock::now();
	// Set up OpenCL
	cl_uint num_platforms;
	clGetPlatformIDs(0, NULL, &num_platforms);
	std::vector<cl_platform_id> platforms(num_platforms);
	clGetPlatformIDs(num_platforms, platforms.data(), NULL);

	cl_uint num_devices;
	clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_GPU, 0, NULL, &num_devices);
	std::vector<cl_device_id> devices(num_devices);
	clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_GPU, num_devices, devices.data(), NULL);

	cl_context context = clCreateContext(NULL, num_devices, devices.data(), NULL, NULL, NULL);
	cl_command_queue queue = clCreateCommandQueueWithProperties(context, devices[0], 0, NULL);

	// Set up OpenCL memory buffers
	cl_mem input_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY, width * height * 4, NULL, NULL);
	cl_mem output_buffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY, width * height, NULL, NULL);

	// Copy the input image to the input buffer
	clEnqueueWriteBuffer(queue, input_buffer, CL_TRUE, 0, width * height * 4, image, 0, NULL, NULL);

	// Set up the OpenCL kernel
	cl_program program = clCreateProgramWithSource(context, 1, &kernel_source_grey_scale, NULL, NULL);
	clBuildProgram(program, num_devices, devices.data(), NULL, NULL, NULL);
	cl_kernel kernel = clCreateKernel(program, "rgb_to_grayscale", NULL);

	// Set the kernel arguments
	clSetKernelArg(kernel, 0, sizeof(cl_mem), &input_buffer);
	clSetKernelArg(kernel, 1, sizeof(cl_mem), &output_buffer);

	// Execute the kernel
	size_t global_size[2] = { width, height };
	clEnqueueNDRangeKernel(queue, kernel, 2, NULL, global_size, NULL, 0, NULL, NULL);

	// Read the output buffer
	unsigned char* grayscale_image = new unsigned char[width * height];
	clEnqueueReadBuffer(queue, output_buffer, CL_TRUE, 0, width * height, grayscale_image, 0, NULL, NULL);

	// Clean up OpenCL resources
	clReleaseMemObject(input_buffer);
	clReleaseMemObject(output_buffer);
	clReleaseKernel(kernel);
	clReleaseProgram(program);
	//clReleaseCommandQueue(queue);
	//clReleaseContext(context);

	error = lodepng_encode_file(outFileGrey, grayscale_image, width, height, LodePNGColorType::LCT_GREY, 8);
	auto end_gray_scale = std::chrono::high_resolution_clock::now();
	if (error) {
		printf("error %u: %s\n", error, lodepng_error_text(error));
	}

	auto start_moving_filter = std::chrono::high_resolution_clock::now();
	// Create filter
	const int filter_size = 5;
	unsigned char* filter = new unsigned char[filter_size * filter_size];
	for (int i = 0; i < filter_size * filter_size; i++) {
		filter[i] = 1;
	}
	unsigned char* grayscale_image_copy = new unsigned char[width * height];
	std::copy(grayscale_image, grayscale_image + width * height, grayscale_image_copy);

	// Create input and output buffers for the moving filter operation
	cl_mem gray_image_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, width * height * 4, NULL, NULL);
	cl_mem output_filtered_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, width * height * 4, NULL, NULL);
	cl_mem filter_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY, filter_size * filter_size * 4, NULL, NULL);

	// Copy grayscale image to input buffer and filter to filter buffer
	clEnqueueWriteBuffer(queue, gray_image_buffer, CL_TRUE, 0, width * height * 8, grayscale_image_copy, 0, NULL, NULL);
	clEnqueueWriteBuffer(queue, filter_buffer, CL_TRUE, 0, filter_size * filter_size * 8, filter, 0, NULL, NULL);

	// Load moving filter kernel from file
	std::string moving_filter_kernel_code = get_file_string();
	const char* moving_filter_kernel_src = moving_filter_kernel_code.c_str();

	// Create program and kernel for the moving filter operation
	cl_program moving_filter_program = clCreateProgramWithSource(context, 1, &moving_filter_kernel_src, NULL, NULL);
	clBuildProgram(moving_filter_program, num_devices, devices.data(), NULL, NULL, NULL);
	cl_kernel moving_filter_kernel = clCreateKernel(moving_filter_program, "moving_filter", NULL);

	// Set kernel arguments
	clSetKernelArg(moving_filter_kernel, 0, sizeof(cl_mem), &gray_image_buffer);
	clSetKernelArg(moving_filter_kernel, 1, sizeof(cl_mem), &output_filtered_buffer);
	clSetKernelArg(moving_filter_kernel, 2, sizeof(cl_mem), &filter_buffer);
	clSetKernelArg(moving_filter_kernel, 3, sizeof(cl_int), &width);
	clSetKernelArg(moving_filter_kernel, 4, sizeof(cl_int), &height);
	clSetKernelArg(moving_filter_kernel, 5, sizeof(cl_int), &filter_size);

	// Execute the moving filter kernel

	clEnqueueNDRangeKernel(queue, moving_filter_kernel, 2, NULL, global_size, NULL, 0, NULL, NULL);
	clFinish(queue);

	// Read output_filtered_buffer back into filtered_image

	unsigned char* filtered_image = new unsigned char[width * height];
	clEnqueueReadBuffer(queue, output_filtered_buffer, CL_TRUE, 0, width * height * 8, filtered_image, 0, NULL, NULL);
	auto end_moving_filter = std::chrono::high_resolution_clock::now();

	//// Save the filtered grayscale image
	auto start_save_image = std::chrono::high_resolution_clock::now();
	error = lodepng_encode_file("output_filtered.png", filtered_image, width, height, LodePNGColorType::LCT_GREY, 8);
	auto end_save_image = std::chrono::high_resolution_clock::now();
	if (error) {
		printf("error %u: %s\n", error, lodepng_error_text(error));
	}

	// Clean up OpenCL resources
	clReleaseMemObject(gray_image_buffer);
	clReleaseMemObject(output_filtered_buffer);
	clReleaseMemObject(filter_buffer);
	clReleaseKernel(moving_filter_kernel);


	// Calculate durations
	auto load_image_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_load_image - start_load_image).count();
	auto gray_scale_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_gray_scale - start_gray_scale).count();
	auto moving_filter_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_moving_filter - start_moving_filter).count();
	auto save_image_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_save_image - start_save_image).count();

	// Print durations
	cout << "Load and read image time: " << load_image_duration << " ms" << std::endl;
	cout << "Convert to grayscale time: " << gray_scale_duration << " ms" << std::endl;
	cout << "Moving filter operation time: " << moving_filter_duration << " ms" << std::endl;
	cout << "Save image time: " << save_image_duration << " ms" << std::endl;


	// Print device details
	getDeviceInfo();

}


int main()
{

	const char* filename = "image3.png";
	const char* outFilename = "imageo.png";
	unsigned error;
	unsigned char* image = 0;
	unsigned char* newImage = 0;
	unsigned int width, height;

	error = lodepng_decode32_file(&image, &width, &height, filename);
	if (error) {
		printf("error %u: %s\n", error, lodepng_error_text(error));
	}

	unsigned char* grayscale_image = new unsigned char[width * height];
	convertToGrayscaleNormalized(image, grayscale_image, width, height);

	unsigned resized_width = width / 4;
	unsigned resized_height = height / 4;
	unsigned char* resized_image = new unsigned char[resized_width * resized_height * 4];
	resizeImage(image, resized_image, width, height);

	error = lodepng_encode_file(outFilename, grayscale_image, resized_width, resized_height, LodePNGColorType::LCT_GREY,8);
	if (error) {
		printf("error %u: %s\n", error, lodepng_error_text(error));
	}

	step3Function();
}