#pragma once

#include <c10/core/DispatchKey.h>
#include <c10/util/llvmMathExtras.h>
#include <c10/util/Exception.h>
#include <ostream>

namespace c10 {

// A representation of a set of DispatchKeys.  A tensor may have multiple
// tensor type ids, e.g., a Variable tensor can also be a CPU tensor; the
// DispatchKeySet specifies what type ids apply.  The internal representation is
// as a 64-bit bit set (this means only 64 tensor type ids are supported).
// DispatchKey集合表示，一个tensor可以有多个tensor类型id，例如：Variable tensor也可以是
// 一个CPU tensor，DispatchKey是一个64位的bit set
//
// Note that DispatchKeys are ordered; thus, we can ask questions like "what is
// the highest priority DispatchKey in the set"?  (The set itself is not
// ordered; two sets with the same ids will always have the ids ordered in the
// same way.)
//
// At the moment, there are no nontrivial uses of this set; tensors are always
// singletons.  In the near future, this set will represent variable? + tensor
// type id.  In the far future, it will be requires grad? + profiling? +
// tracing? + lazy? + tensor type id.
//tensors are always singletons. 张量总是单例

// (The difference between variable and requires grad, is that
// there are currently three states a tensor can be:
//  1. Not a variable
//  2. Variable with requires_grad=False
//  3. Variable with requires_grad=True
// Eventually, we want to kill state (1), and only dispatch to autograd
// handling code if one of the inputs requires grad.)
// variable和requires grad之间的区别是张量可以有三种状态：
// 1. 不是variable
// 2. requires_grad=False的variable
// 3. requires_grad=True的variable
// 最终，我们想要终止状态(1)，如果输入中的一个需要grad分派到autograd处理代码
//
// An undefined tensor is one with an empty tensor type set.
class DispatchKeySet final {
public:
  enum Full { FULL };
  enum FullAfter { FULL_AFTER };
  enum Raw { RAW };

  // NB: default constructor representation as zero is MANDATORY as
  // use of DispatchKeySet in TLS requires this.
  DispatchKeySet()
    : repr_(0) {}
  DispatchKeySet(Full)
    : repr_(std::numeric_limits<decltype(repr_)>::max()) {}
  DispatchKeySet(FullAfter, DispatchKey t)
    // LSB after t are OK, but not t itself.
    : repr_((1ULL << (static_cast<uint8_t>(t) - 1)) - 1) {}
  // Public version of DispatchKeySet(uint64_t) API; external users
  // must be explicit when they do this!
  DispatchKeySet(Raw, uint64_t x)
    : repr_(x) {}
  explicit DispatchKeySet(DispatchKey t)
    : repr_(t == DispatchKey::Undefined
              ? 0
              : 1ULL << (static_cast<uint8_t>(t) - 1)) {}
  explicit DispatchKeySet(std::initializer_list<DispatchKey> ks)
    : DispatchKeySet() {
    for (auto k : ks) {
      repr_ |= DispatchKeySet(k).repr_;
    }
  }
  // Test if a DispatchKey is in the set
  bool has(DispatchKey t) const {
    TORCH_INTERNAL_ASSERT(t != DispatchKey::Undefined);
    return static_cast<bool>(repr_ & DispatchKeySet(t).repr_);
  }
  // Perform set union
  DispatchKeySet operator|(DispatchKeySet other) const {
    return DispatchKeySet(repr_ | other.repr_);
  }
  // Perform set intersection
  DispatchKeySet operator&(DispatchKeySet other) const {
    return DispatchKeySet(repr_ & other.repr_);
  }
  // Compute the set difference self - other
  DispatchKeySet operator-(DispatchKeySet other) const {
    return DispatchKeySet(repr_ & ~other.repr_);
  }
  // Perform set equality
  bool operator==(DispatchKeySet other) const {
    return repr_ == other.repr_;
  }
  // Add a DispatchKey to the DispatchKey set.  Does NOT mutate,
  // returns the extended DispatchKeySet!
  C10_NODISCARD DispatchKeySet add(DispatchKey t) const {
    return *this | DispatchKeySet(t);
  }
  // Remove a DispatchKey from the DispatchKey set.  This is
  // generally not an operation you should be doing (it's
  // used to implement operator<<)
  C10_NODISCARD DispatchKeySet remove(DispatchKey t) const {
    return DispatchKeySet(repr_ & ~DispatchKeySet(t).repr_);
  }
  // Is the set empty?  (AKA undefined tensor)
  bool empty() const {
    return repr_ == 0;
  }
  uint64_t raw_repr() { return repr_; }
  // Return the type id in this set with the highest priority (i.e.,
  // is the largest in the DispatchKey enum).  Intuitively, this
  // type id is the one that should handle dispatch (assuming there
  // aren't any further exclusions or inclusions).
  // 返回集合中具有最高优先级的类型ID（即DispatchKey枚举中最大的）
  DispatchKey highestPriorityTypeId() const {
    // TODO: If I put Undefined as entry 64 and then adjust the
    // singleton constructor to shift from the right, we can get rid of the
    // subtraction here.  It's modestly more complicated to get right so I
    // didn't do it for now.
    return static_cast<DispatchKey>(64 - llvm::countLeadingZeros(repr_));
  }
private:
  DispatchKeySet(uint64_t repr) : repr_(repr) {}
  uint64_t repr_ = 0;
};

C10_API std::string toString(DispatchKeySet);
C10_API std::ostream& operator<<(std::ostream&, DispatchKeySet);

// Historically, every tensor only had a single DispatchKey, and it was always
// something like CPU, and there wasn't any of this business where TLS
// could cause the DispatchKey of a tensor to change.  But we still have some
// legacy code that is still using DispatchKey for things like instanceof
// checks; if at all possible, refactor the code to stop using DispatchKey in
// those cases.
static inline DispatchKey legacyExtractDispatchKey(DispatchKeySet s) {
  // NB: If you add any extra keys that can be stored in TensorImpl on
  // top of existing "normal" keys like CPU/CUDA, you need to add it
  // here.  At the moment, RequiresGrad (replacement for Variable)
  // is the most likely key that will need this treatment; note that
  // Autograd does NOT need this as it is applied universally
  // (and doesn't show up in TensorImpl)
  return s.highestPriorityTypeId();
}

// For backwards compatibility with XLA repository
// (I don't want to fix this in XLA right now because there might be
// more renaming coming in the future.)
static inline DispatchKeySet XLA() {
  return DispatchKeySet{DispatchKey::XLA, DispatchKey::XLAPreAutograd};
}

}
