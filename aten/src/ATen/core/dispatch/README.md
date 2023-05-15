This folder contains the c10 dispatcher. This dispatcher is a single point
through which we are planning to route all kernel calls.
Existing dispatch mechanisms from legacy PyTorch or caffe2 are planned to
be replaced.  
通过dispatcher，路由到所有的kernel调用

This folder contains the following files:
- Dispatcher.h: Main facade interface. Code using the dispatcher should only use this.  
  主要接口，使用dispatcher的代码，应该只使用该文件
- DispatchTable.h: Implementation of the actual dispatch mechanism. Hash table with kernels, lookup, ...  
  实际调度机制到的实现。带有内核的哈希表...
- KernelFunction.h: The core interface (i.e. function pointer) for calling a kernel  
  调用内核的核心接口（即函数指针）
  