

#include<CL/cl.hpp>
#include<iostream>
#include <fstream>

int main()
{

	std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);

    auto platform = platforms.front();
    std::vector<cl::Device> devices;
    platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);

    auto device = devices.front();

    std::ifstream helloWorldFile("hello.cl");
    std::string src(std::istreambuf_iterator<char>(helloWorldFile), (std::istreambuf_iterator<char>()));

    cl::Program::Sources sources(1, std::make_pair(src.c_str(), src.length() + 1));

    cl::Context context(device);
    cl::Program program(context, sources);

    auto err = program.build("-cl-std=CL1.2");

    char buf[16];
    cl::Buffer memBuf(context, CL_MEM_WRITE_ONLY | CL_MEM_HOST_READ_ONLY, sizeof(buf));
    cl::Kernel kernel(program, "Hello", &err);
    kernel.setArg(0, memBuf);

    cl::CommandQueue queue(context, device);
    //queue.enqueueNDRangeKernel(kernel, cl::NullRange cl::NDRange(vector_elements), cl::NullRange);
    queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(12), cl::NullRange);
    queue.enqueueReadBuffer(memBuf, CL_TRUE, 0, sizeof(buf), buf);

    std::cout << buf;
    std::cin.get();

}
