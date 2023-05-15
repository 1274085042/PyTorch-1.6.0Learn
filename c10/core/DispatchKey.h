#pragma once

#include <iostream>
#include <string>
#include <c10/macros/Macros.h>

namespace c10 {

// Semantically, a dispatch key identifies a possible "level" in our
// dispatch, for which a handler may be registered.  Traditional
// backends like CPU and CUDA get dispatch keys; however, so do
// "wrapping" layers like Variable (for autograd handling).
//
// 从语义上讲，调度键标识了调度的级别，可以为其注册处理程序
// In implementation terms, the dispatch key identifies a specific "bit" in a
// DispatchKeySet.  Higher bit indexes get handled by dispatching first (because
// we "count leading zeros" when we extract the highest priority dispatch
// key.)
// 在实现层面，dispatch key就是DispatchKeySet中的一个bit
enum class DispatchKey : uint8_t {

  // ~~~~~~~~~~~~~~~~~~~~~~~~~~ UNDEFINED ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
  // This is not a "real" tensor id, but it exists to give us a "nullopt"
  // element we can return for cases when a DispatchKeySet contains no elements.
  // You can think a more semantically accurate definition of DispatchKey is:
  //
  //    using DispatchKey = optional<RealDispatchKey>
  //
  // and Undefined == nullopt.  We didn't actually represent
  // it this way because optional<RealDispatchKey> would take two
  // words, when DispatchKey fits in eight bits.

  Undefined = 0,        // 0

  // Define an alias for Undefined to represent CatchAll (long term
  // this will get eliminated, but for now it's convenient)
  CatchAll = Undefined, // 0

  // ~~~~~~~~~~~~~~~~~~~~~~~~~~ BACKENDS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
  // A "backend" is colloquially used to refer to handlers for dispatch
  // which actually implement the numerics of an operation in question.
  //
  // Due to the nature of the enum, these backends are specified in
  // an ordered way, but for most backends this order is not semantically
  // meaningful (e.g., it's valid to reorder these backends without changing
  // semantics).  The only situation when backend ordering is meaningful
  // is when the backend participates in multiple dispatch with another
  // backend; e.g., CPU and SparseCPU (sparse must have
  // higher priority).

  // Here are backends which you think of as traditionally specifying
  // how to implement operations on some device.
  CPU, // registered at build/aten/src/ATen/CPUType.cpp   // 1
  
  // 2 
  CUDA, // registered at build/aten/src/ATen/CUDAType.cpp

  // 3
  HIP, // NB: I think this is not actually used, due to Note [Masquerading as
       // CUDA]

  // 4
  FPGA, // Xilinx support lives out of tree at https://gitlab.com/pytorch-complex/vitis_kernels

  // 5
  MSNPU, // unused externally, but tested at
         // test/cpp_extensions/msnpu_extension.cpp
  
  // 6
  XLA, // lives out of tree at https://github.com/pytorch/xla

  // 7
  Vulkan,

  // These are Caffe2 device types which we grandfathered into
  // DispatchKey.
  // TODO: Caffe2-only DispatchKeys actually should be removed from this enum
  // and just simply be undispatchable.
  // 8
  MKLDNN, // (MKLDNN is treated as another "device" in Caffe2)

  // 9
  OpenGL,

  // 10
  OpenCL,

  // 11
  IDEEP,

  // Here are backends which specify more specialized operators
  // based on the dtype of the tensor.
  // 12
  QuantizedCPU, // registered at build/aten/src/ATen/QuantizedCPUType.cpp

  // 13
  QuantizedCUDA, // registered at build/aten/src/ATen/QuantizedCUDAType.cpp

  // 14
  ComplexCPU, // lives out of tree at
              // https://gitlab.com/pytorch-complex/pytorch-cpu-strided-complex
  
  // 15
  ComplexCUDA, // and
               // https://gitlab.com/pytorch-complex/pytorch-cuda-strided-complex
  // tested at test/cpp_extensions/complex_registration_extension.cpp
  // TODO: Remove Complex dispatch keys when Complex is moved in tree

  // This backend is to support custom RNGs; it lets you go
  // to a different kernel if you pass in a generator that is not a
  // traditional CPUGeneratorImpl/CUDAGeneratorImpl.  To make use of this
  // key:
  //  1) set it as a second parameter of at::Generator constructor call in
  //     the user-defined PRNG class.
  //  2) use it as a dispatch key while registering custom kernels
  //     (templatized kernels specialized for user-defined PRNG class)
  // intended for out of tree use; tested by aten/src/ATen/test/rng_test.cpp
  // 16
  CustomRNGKeyId,

  // Here are backends which specify more specialized operators
  // based on the layout of the tensor.  Note that the sparse backends
  // are one case where ordering matters: sparse multi-dispatches with
  // the corresponding dense tensors, and must be handled before them.
  // 17
  MkldnnCPU, // registered at build/aten/src/ATen/MkldnnCPUType.cpp
  // NB: not to be confused with MKLDNN, which is Caffe2 only
  // 18
  SparseCPU, // registered at build/aten/src/ATen/SparseCPUType.cpp
  // 19
  SparseCUDA, // registered at build/aten/src/ATen/SparseCUDAType.cpp
  // 20
  SparseHIP, // TODO: I think this is not actually used, due to Note
             // [Masquerading as CUDA]

  // Here are reserved backends for user-defined backends, see Note [Private use
  // DispatchKey]
  // To see some example about how to use this, check out MSNPU
  // 21
  PrivateUse1,
  // 22
  PrivateUse2,
  // 23
  PrivateUse3,

  // The meta function characterizes how an operation affects the metadata of a
  // tensor (shape, dtype) without doing any of the actual computation.  A
  // meta tensor can be used to dry run operators without actually doing
  // any computation, e.g., add on two meta tensors would give you another
  // meta tensor with the output shape and dtype, but wouldn't actually
  // add anything.  A meta implementation typically would look something like:
  //
  //  Tensor meta::add(const Tensor& self, const Tensor& other) {
  //    TORCH_CHECK(self.size().equals(other.size()));
  //    return at::empty_like(self, self.size());
  //  }
  //
  // The meta function would get invoked if you ran an operator passing
  // in meta tensors.  The call stack in such a case would look something like
  // this:
  //
  //  at::add(x: Meta, y: Meta) {
  //    return [dispatch] meta::add(x: Meta, y: Meta) {
  //      output_shape = ...
  //      [dispatch] meta::empty(output_shape) {
  //        return ... meta tensor with output_shape but no data allocated ...
  //      }
  //    }
  //  }
  //
  // Meta functions have an important secondary function, which is they can
  // be used as tensor "allocators".  A typical backend implementation should
  // be implemented in this way:
  //
  //  Tensor cpu::add(const Tensor& self, const Tensor& other) {
  //    Tensor result = meta::add(self, other);
  //    // ... do the actual computation into result ...
  //    return result;
  //  }
  //
  // In this case, the internal at::empty_like invocation would dispatch to the
  // CPU factory function, not the meta factory function.  The call stack in
  // this case looks like:
  //
  //  at::add(x: CPU, y: CPU) {
  //    return [dispatch] cpu::add(x: CPU, y: CPU) {
  //      output = [direct] meta::add(x: CPU, y: CPU) {
  //        output_shape = ...
  //        [dispatch] cpu::empty(output_shape)
  //      }
  //      ... compute on output ...
  //      return output;
  //    }
  //  }
  //
  // 24
  Meta,

  // In some situations, it is not immediately obvious what the correct
  // backend for function is, because the function in question doesn't
  // have any "tensor" arguments.  In this case, a BackendSelect function
  // can be registered to implement the custom determination of the
  // correct backend.
  // 25
  BackendSelect,

  // The named dispatch key is set for any tensors with named dimensions.
  // Although we have a dispatch key for named tensors, for historical reasons,
  // this dispatch key doesn't do any of the substantive functionality for named
  // tensor (though, hypothetically, it could!)  At the moment, it's just
  // responsible for letting us give good error messages when operations
  // don't support named tensors.
  //
  // NB: If you ever consider moving named tensor functionality into
  // this dispatch key, note that it might be necessary add another dispatch
  // key that triggers before composite operators, in case a composite operator
  // has named dimension propagation that doesn't match that of its
  // constituent parts.
  // 26
  Named,

  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~ AUTOGRAD ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
  // All backends are oblivious to autograd; autograd is handled as a
  // layer which happens on top of all backends.  It inspects the autograd
  // metadata of all inputs, determines what autograd metadata should be
  // constructed by the output, and otherwise defers to the backend to
  // actually do the numeric computation.  Autograd contains
  // the bulk of this logic.
  // 27
  Autograd,

  // 28
  Profiler,

  // 29
  Tracer,

  // Pre-autograd dispatch keys allow backends to override the autograd behavior
  // (aka Autograd) for operators which have a Variable kernel
  // already registered.  For example, XLA wants to define autograd for
  // einsum directly.  Registering a custom autograd implementation at the
  // XLA key won't work because we process Autograd
  // before XLA.  This key has higher priority and gets processed
  // first.  You generally should NOT redispatch after handling autograd
  // here (since that would result in execution of the Autograd
  // operator, which you're trying to skip).  In PreAutograd implementations,
  // you are responsible for handling autograd yourself, or deferring to other
  // operators which support autograd.
  // 30
  XLAPreAutograd,

  // Autocasting precedes VariableTypeId, to ensure casts are autograd-exposed
  // and inputs are saved for backward in the post-autocast type.
  // 31
  Autocast,

  // Here are some reserved pre-autograd keys for user-defined backends, see
  // Note [Private use DispatchKey]
  // 32
  PrivateUse1_PreAutograd,
  // 33
  PrivateUse2_PreAutograd,
  // 34
  PrivateUse3_PreAutograd,

  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~ WRAPPERS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
  // There are a number of alternative modes which may want to handle before
  // autograd; for example, error checking, tracing, profiling or vmap.  They
  // go here.

  // This is the dispatch key for BatchedTensorImpl, which is used to implement
  // batching rules for vmap.
  // 35
  Batched,

  // TESTING: This is intended to be a generic testing tensor type id.
  // Don't use it for anything real; its only acceptable use is within a single
  // process test.  Use it by creating a TensorImpl with this DispatchKey, and
  // then registering operators to operate on this type id.  See
  // aten/src/ATen/core/dispatch/backend_fallback_test.cpp for a usage example.
  // 36
  TESTING_ONLY_GenericWrapper,

  // TESTING: This is intended to be a generic testing tensor type id.
  // Don't use it for anything real; its only acceptable use is within a ingle
  // process test.  Use it by toggling the mode on and off via
  // TESTING_ONLY_tls_generic_mode_set_enabled and then registering operators
  // to operate on this type id.  See
  // aten/src/ATen/core/dispatch/backend_fallback_test.cpp
  // for a usage example
  // 37
  TESTING_ONLY_GenericMode,

  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ FIN ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
  // 38
  NumDispatchKeys, // Sentinel

  // ~~~~~~~~~~~~~~~~~~~~~~~~~ BC ALIASES ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
  // The aliases exist for backwards compatibility reasons, they shouldn't
  // be used
  // 39
  CPUTensorId = CPU,
  // 40
  CUDATensorId = CUDA,
};

// Note [Private use DispatchKey]
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~
// Private use tensor IDs are preallocated tensor type IDs for use in user
// applications.  Similar to private use fields in HTTP, they can be used
// by end users for experimental or private applications, without needing
// to "standardize" the tensor ID (which would be done by submitting a PR
// to PyTorch to add your type ID).
//
// Private use tensor IDs are appropriate to use if you want to experiment
// with adding a new tensor type (without having to patch PyTorch first) or
// have a private, non-distributed application that needs to make use of a
// new tensor type.  Private use tensor IDs are NOT appropriate to use for
// libraries intended to be distributed to further users: please contact
// the PyTorch developers to get a type ID registered in this case.
//
// We provide two classes of private user tensor id: regular DispatchKeys
// and PreAutograd DispatchKeys.  DispatchKeys serve the role of ordinary "backend"
// DispatchKeys; if you were adding support for a new type of accelerator, you
// would use a DispatchKey, and reuse autograd definitions already defined in
// PyTorch for operators you define.  PreAutograd DispatchKeys serve as "wrapper"
// DispatchKeys: they are most appropriate for tensors that compose multiple
// internal tensors, and for cases when the built-in autograd formulas for
// operators are not appropriate.

static_assert(
  static_cast<uint8_t>(DispatchKey::NumDispatchKeys) < 64,
  "DispatchKey is used as index into 64-bit bitmask; you must have less than 64 entries");

C10_API const char* toString(DispatchKey);
C10_API std::ostream& operator<<(std::ostream&, DispatchKey);

// These are some convenience identifiers for dispatch keys which are
// shorter to type than their long counterparts.  Note that some of these
// dispatch keys directly correspond to DeviceType; and most APIs that
// accept DispatchKey also accept DeviceType; e.g.,
// torch::dispatch(torch::kCPU, ...) is also valid.
constexpr DispatchKey kAutograd = DispatchKey::Autograd;

} // namespace c10

namespace torch {
  // Expose the constant, but not the TYPE (DispatchKey is an implementation
  // detail!)
  using c10::kAutograd;
}

// NB: You really shouldn't use this instance; this enum is guaranteed
// to be pretty small so a regular array should be acceptable.
namespace std {
template <>
struct hash<c10::DispatchKey> {
  typedef size_t result_type;
  typedef c10::DispatchKey argument_type;

  size_t operator()(c10::DispatchKey x) const {
    return static_cast<size_t>(x);
  }
};
}
