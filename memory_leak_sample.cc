#include <iostream>
#include <vector>

#include <cuda_runtime.h>
#include <nccl.h>

/*
Sample should be run on host with 2 GPUs.

Usage sample:
Compile:
clang++ -I ${NCCL_HOME}/include/ -L ${NCCL_HOME}/lib -I ${CUDA_PATH}/include/  -L ${CUDA_PATH}/lib  -lcudart -lcuda memory_leak_sample.cc -lnccl_static -o memory_leak_sample


Run:
NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=ALLOC ./memory_leak_sample
*/
int main(int argc, char *argv[]) {
  /*Variables*/
  int num_local_devices = 2;
  std::cout << "num_local_devices: " << num_local_devices << std::endl;

  /*------Storing the data on devices------*/
  const int data_size = 8;
  std::vector<double> host_data(data_size);
  for (int i = 0; i < data_size; i++) host_data[i] = 1;

  /*Array of device memories per GPU*/
  std::vector<double *> devices_data(num_local_devices);

  /*Allocate memory on device and store host data there.*/
  for (int device = 0; device < num_local_devices; device++) {
    cudaSetDevice(device);
    cudaMalloc(&devices_data[device], data_size * sizeof(double));
    cudaMemcpy(devices_data[device], host_data.data(),
               data_size * sizeof(double), cudaMemcpyHostToDevice);
  }

  /*-----Initialize NCCL communicators-----*/
  std::vector<ncclComm_t> communicators(num_local_devices);
  ncclCommInitAll(communicators.data(), num_local_devices, /*devlist=*/nullptr);

  /*-----Initialize CUDA Streams-----*/
  std::vector<cudaStream_t> streams(num_local_devices);
  for (int device = 0; device < num_local_devices; device++) {
    cudaSetDevice(device);
    cudaStreamCreate(&streams[device]);
  }

  const int root_gpu = 0;

  int a = 0;
  while (true) {
    a++;
    if (a%1000 == 0) {
        std::cout << "Iteration: " << a << std::endl;
    } 
    ncclGroupStart();
    for (int device = 0; device < num_local_devices; device++) {
      cudaSetDevice(device);

      ncclReduce(
          devices_data[device],
          devices_data[device], data_size, ncclDouble, ncclSum,
          root_gpu,
          communicators[device], streams[device]);
    }

    ncclGroupEnd();

    /*Synchronizing CUDA Streams*/
    for (int device = 0; device < num_local_devices; device++) {
      cudaSetDevice(device);
      cudaStreamSynchronize(streams[device]);
    }
  }

  /*-----Destroy CUDA Streams-----*/
  for (int device = 0; device < num_local_devices; device++) {
    cudaSetDevice(device);
    cudaStreamDestroy(streams[device]);
  }

  /*-----Finalizing NCCL-----*/
  for (int device = 0; device < num_local_devices; device++) {
    ncclCommDestroy(communicators[device]);
  }

  /*-----Clean device buffers-----*/
  for (int device = 0; device < num_local_devices; device++) {
    cudaFree(devices_data[device]);
  }

  return 0;
}
