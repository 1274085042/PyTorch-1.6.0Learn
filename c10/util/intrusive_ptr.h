#pragma once

#include <c10/util/C++17.h>
#include <c10/util/Exception.h>
#include <atomic>
#include <stdexcept>

namespace c10 {
class intrusive_ptr_target;
namespace raw {
  namespace weak_intrusive_ptr {
    inline void incref(intrusive_ptr_target* self);
  }
  namespace intrusive_ptr {
    inline void incref(intrusive_ptr_target * self);
  }
}
/**
 * intrusive_ptr<T> is an alternative to shared_ptr<T> that has better
 * performance because it does the refcounting intrusively
 * (i.e. in a member of the object itself).
 * Your class T needs to inherit from intrusive_ptr_target to allow it to be
 * used in an intrusive_ptr<T>.
 * 
 * intrusive_ptr<T>是shared_ptr<T>的替代品，它具有更好的性能，因为它用侵入式的
 * 引用计数（即在对象本身的成员中）
 * 你的类T需要继承intrusive_ptr_target，以允许使用intrusive_ptr<T>
 */

// Note [Stack allocated intrusive_ptr_target safety]
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// A well known problem with std::enable_shared_from_this is that it
// allows you to create a std::shared_ptr from a stack allocated object,
// which is totally bogus because the object will die once you return
// from the stack.  In intrusive_ptr, we can detect that this has occurred,
// because we set the refcount/weakcount of objects which inherit from
// intrusive_ptr_target to zero, *unless* we can prove that the object
// was dynamically allocated (e.g., via make_intrusive).
//
// std::enable_shared_from_this一个众所周知的问题是允许从一个栈分配的对象创建
// std::shared_ptr，这完全是伪造的，因为一旦从栈中返回该对象就会死亡。在intrusive_ptr
// 中，我们可以检测到这种情况的发生，因为我们可以设置继承自intrusive_ptr_target的对象的
// refcount/weakcount为0,除非我们可以证明对象是动态分配的（例如，通过make_intrusive）
//
// Thus, whenever you transmute a T* into a intrusive_ptr<T>, we check
// and make sure that the refcount isn't zero (or, a more subtle
// test for weak_intrusive_ptr<T>, for which the refcount may validly
// be zero, but the weak refcount better not be zero), because that
// tells us if the object was allocated by us.  If it wasn't, no
// intrusive_ptr for you!
// 因此当你装换T*到intrusive_ptr<T>时，我们会检测并确保引用计数不为0（或者，对 weak_intrusive_ptr<T>
// 进行更微妙的测试，其引用计数可以有效的为0,但是weak refcount最好不要为0），因为这会告诉我们
// 对象是否由我们分配，如果不是这没有intrusive_ptr给你
//

class C10_API intrusive_ptr_target {
  // Note [Weak references for intrusive refcounting]
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // Here's the scheme:
  //
  //  - refcount == number of strong references to the object
  //                 对象的强引用数
  //    weakcount == number of weak references to the object,
  //                 对象的弱引用数
  //      plus one more if refcount > 0
  //    An invariant: refcount > 0  =>  weakcount > 0
  //
  //  - THStorage stays live as long as there are any strong
  //    or weak pointers to it (weakcount > 0, since strong
  //    references count as a +1 to weakcount)
  //    只要有强指针或弱指针指向THStorage，它就会保持活跃
  //
  //  - finalizers are called and data_ptr is deallocated when refcount == 0
  //    当refcount==0,调用终结器释放data_ptr
  //
  //  - Once refcount == 0, it can never again be > 0 (the transition
  //    from > 0 to == 0 is monotonic)
  //    一旦refcount==0,它就再也不会>0（从 > 0 到 == 0 是单调的）
  //
  //  - When you access THStorage via a weak pointer, you must
  //    atomically increment the use count, if it is greater than 0.
  //    If it is not, you must report that the storage is dead.
  //    当你用弱指针访问THStorage，如果引用计数大于0,你必须原子性的增加引用计数。
  //    如果不是，你必须报告storage已经死了

  mutable std::atomic<size_t> refcount_;
  mutable std::atomic<size_t> weakcount_;

  template <typename T, typename NullType>
  friend class intrusive_ptr;
  friend inline void raw::intrusive_ptr::incref(intrusive_ptr_target* self);

  template <typename T, typename NullType>
  friend class weak_intrusive_ptr;
  friend inline void raw::weak_intrusive_ptr::incref(intrusive_ptr_target* self);

 protected:
  // protected destructor. We never want to destruct intrusive_ptr_target*
  // directly.
  // 受保护的析够，我们永远不想直接析够intrusive_ptr_target*
  virtual ~intrusive_ptr_target() {
// Disable -Wterminate and -Wexceptions so we're allowed to use assertions
// (i.e. throw exceptions) in a destructor.
// We also have to disable -Wunknown-warning-option and -Wpragmas, because
// some other compilers don't know about -Wterminate or -Wexceptions and
// will show a warning about unknown warning options otherwise.
// 禁用-Wterminate和-Wexceptions，这样就可以在析够函数中使用断言（即抛出异常）
// 我们还必须禁用-Wunknown-warning-option和-Wpragmas，因为一些其它编译器
// 不知道-Wterminate或-Wexceptions，并会显示未知警告选项的警告

#if defined(_MSC_VER) && !defined(__clang__)
#  pragma warning(push)
#  pragma warning(disable: 4297) // function assumed not to throw an exception but does
#else
#  pragma GCC diagnostic push
#  pragma GCC diagnostic ignored "-Wpragmas"
#  pragma GCC diagnostic ignored "-Wunknown-warning-option"
#  pragma GCC diagnostic ignored "-Wterminate"
#  pragma GCC diagnostic ignored "-Wexceptions"
#endif
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
        refcount_.load() == 0,
        "Tried to destruct an intrusive_ptr_target that still has intrusive_ptr to it");
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
        weakcount_.load() == 0,
        "Tried to destruct an intrusive_ptr_target that still has weak_intrusive_ptr to it");
#if defined(_MSC_VER) && !defined(__clang__)
#  pragma warning(pop)
#else
#  pragma GCC diagnostic pop
#endif
  }

  constexpr intrusive_ptr_target() noexcept : refcount_(0), weakcount_(0) {}

  // intrusive_ptr_target supports copy and move: but refcount and weakcount don't
  // participate (since they are intrinsic properties of the memory location)
  // intrusive_ptr_target支持拷贝和移动，但是refcount和weakcount不参与（因为它们是内存位置的固有属性）
  intrusive_ptr_target(intrusive_ptr_target&& other) noexcept : intrusive_ptr_target() {}
  intrusive_ptr_target& operator=(intrusive_ptr_target&& other) noexcept { return *this; }
  intrusive_ptr_target(const intrusive_ptr_target& other) noexcept : intrusive_ptr_target() {}
  intrusive_ptr_target& operator=(const intrusive_ptr_target& other) noexcept { return *this; }

 private:
  /**
   * This is called when refcount reaches zero.
   * You can override this to release expensive resources.
   * There might still be weak references, so your object might not get
   * destructed yet, but you can assume the object isn't used anymore,
   * i.e. no more calls to methods or accesses to members (we just can't
   * destruct it yet because we need the weakcount accessible).
   * 当refcount为0时，该函数被调用。你可以override该函数来释放昂贵的资源。
   * 可能仍然有弱引用，因此你的对象可能没有被析够，但是你可以假设对象不再使用，
   * 即不再调用方法或者访问成员（我们只是不析够它，因为我们需要weakcount可访问）
   * 
   * Even if there are no weak references (i.e. your class is about to be
   * destructed), this function is guaranteed to be called first.
   * However, if you use your class for an object on the stack that is
   * destructed by the scope (i.e. without intrusive_ptr), this function will
   * not be called.
   * 即使没有弱引用（即你的类被析够），该函数保证首先被调用。如果你的类用于在栈上被作用域
   * 析够的对象（即没有intrusive_ptr），则不会调用该函数
   */
  virtual void release_resources() {}
};

namespace detail {
template <class TTarget>
struct intrusive_target_default_null_type final {
  static constexpr TTarget* singleton() noexcept {
    return nullptr;
  }
};

template<class TTarget, class ToNullType, class FromNullType>
TTarget* assign_ptr_(TTarget* rhs) {
  if (FromNullType::singleton() == rhs) {
    return ToNullType::singleton();
  } else {
    return rhs;
  }
}
} // namespace detail

template <class TTarget, class NullType>
class weak_intrusive_ptr;

template <
    class TTarget,
    class NullType = detail::intrusive_target_default_null_type<TTarget>>
class intrusive_ptr final {
 private:
//  the following static assert would be nice to have but it requires
//  the target class T to be fully defined when intrusive_ptr<T> is instantiated
//  this is a problem for classes that contain pointers to themselves
//  下面的静态断言会更好，它要求当intrusive_ptr<T>实例化时，完全定义目标类T，这对于
//  包含指向自己的指针的类来说是个问题
//  static_assert(
//      std::is_base_of<intrusive_ptr_target, TTarget>::value,
//      "intrusive_ptr can only be used for classes that inherit from intrusive_ptr_target.");
#ifndef _WIN32
  // This static_assert triggers on MSVC
  //  error C2131: expression did not evaluate to a constant
  static_assert(
      NullType::singleton() == NullType::singleton(),
      "NullType must have a constexpr singleton() method");
#endif
  static_assert(
      std::is_same<TTarget*, decltype(NullType::singleton())>::value,
      "NullType::singleton() must return a element_type* pointer");

  TTarget* target_;

  template <class TTarget2, class NullType2>
  friend class intrusive_ptr;
  friend class weak_intrusive_ptr<TTarget, NullType>;

  void retain_() {
    if (target_ != NullType::singleton()) {
      size_t new_refcount = ++target_->refcount_;
      TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
          new_refcount != 1,
          "intrusive_ptr: Cannot increase refcount after it reached zero.");
    }
  }

  void reset_() noexcept {
    if (target_ != NullType::singleton() && --target_->refcount_ == 0) {
      // justification for const_cast: release_resources is basically a destructor
      // and a destructor always mutates the object, even for const objects.
      // const_cast理由：release_resources基本上是一个析构函数，并且析构函数总是修改对象
      // 即使是const对象
      const_cast<std::remove_const_t<TTarget>*>(target_)->release_resources();

      // See comment above about weakcount. As long as refcount>0,
      // weakcount is one larger than the actual number of weak references.
      // So we need to decrement it here.
      // 查看上面的weakcount注释，只要refcount>0，weakcount就比实际的弱引用数大1
      // 所以在这里减掉1
      if (--target_->weakcount_ == 0) {
        delete target_;
      }
    }
    target_ = NullType::singleton();
  }

  // This constructor will not increase the ref counter for you.
  // This is not public because we shouldn't make intrusive_ptr out of raw
  // pointers except from inside the make_intrusive() and
  // weak_intrusive_ptr::lock() implementations
  // 该构造函数不会增加引用计数，它不是public，因为我们不应该通过原始指针创建
  // intrusive_ptr，除了make_intrusive()和weak_intrusive_ptr::lock()的实现
  explicit intrusive_ptr(TTarget* target) noexcept : target_(target) {}

 public:
  using element_type = TTarget;

  intrusive_ptr() noexcept : intrusive_ptr(NullType::singleton()) {}

  intrusive_ptr(intrusive_ptr&& rhs) noexcept : target_(rhs.target_) {
    rhs.target_ = NullType::singleton();
  }

  template <class From, class FromNullType>
  /* implicit */ intrusive_ptr(intrusive_ptr<From, FromNullType>&& rhs) noexcept
      : target_(detail::assign_ptr_<TTarget, NullType, FromNullType>(rhs.target_)) {
    static_assert(
        std::is_convertible<From*, TTarget*>::value,
        "Type mismatch. intrusive_ptr move constructor got pointer of wrong type.");
    rhs.target_ = FromNullType::singleton();
  }

  intrusive_ptr(const intrusive_ptr& rhs) : target_(rhs.target_) {
    retain_();
  }

  template <class From, class FromNullType>
  /* implicit */ intrusive_ptr(
      const intrusive_ptr<From, FromNullType>& rhs)
      : target_(detail::assign_ptr_<TTarget, NullType, FromNullType>(rhs.target_)) {
    static_assert(
        std::is_convertible<From*, TTarget*>::value,
        "Type mismatch. intrusive_ptr copy constructor got pointer of wrong type.");
    retain_();
  }

  ~intrusive_ptr() noexcept {
    reset_();
  }

  intrusive_ptr& operator=(intrusive_ptr&& rhs) & noexcept {
    return operator=<TTarget, NullType>(std::move(rhs));
  }

  template <class From, class FromNullType>
      intrusive_ptr& operator=(intrusive_ptr<From, FromNullType>&& rhs) &
      noexcept {
    static_assert(
        std::is_convertible<From*, TTarget*>::value,
        "Type mismatch. intrusive_ptr move assignment got pointer of wrong type.");
    intrusive_ptr tmp = std::move(rhs);
    swap(tmp);
    return *this;
  }

  intrusive_ptr& operator=(const intrusive_ptr& rhs) & noexcept {
    return operator=<TTarget, NullType>(rhs);
  }

  template <class From, class FromNullType>
      intrusive_ptr& operator=(const intrusive_ptr<From, NullType>& rhs) & {
    static_assert(
        std::is_convertible<From*, TTarget*>::value,
        "Type mismatch. intrusive_ptr copy assignment got pointer of wrong type.");
    intrusive_ptr tmp = rhs;
    swap(tmp);
    return *this;
  }

  TTarget* get() const noexcept {
    return target_;
  }

  TTarget& operator*() const noexcept {
    return *target_;
  }

  TTarget* operator->() const noexcept {
    return target_;
  }

  operator bool() const noexcept {
    return target_ != NullType::singleton();
  }

  void reset() noexcept {
    reset_();
  }

  void swap(intrusive_ptr& rhs) noexcept {
    TTarget* tmp = target_;
    target_ = rhs.target_;
    rhs.target_ = tmp;
  }

  // We do a lot of null-pointer checks in our code, good to have this be cheap.
  bool defined() const noexcept {
    return target_ != NullType::singleton();
  }

  size_t use_count() const noexcept {
    if (target_ == NullType::singleton()) {
      return 0;
    }
    return target_->refcount_.load();
  }

  size_t weak_use_count() const noexcept {
    if (target_ == NullType::singleton()) {
      return 0;
    }
    return target_->weakcount_.load();
  }

  bool unique() const noexcept {
    return use_count() == 1;
  }

  /**
   * Returns an owning (!) pointer to the underlying object and makes the
   * intrusive_ptr instance invalid. That means the refcount is not decreased.
   * You *must* put the returned pointer back into a intrusive_ptr using
   * intrusive_ptr::reclaim(ptr) to properly destruct it.
   * This is helpful for C APIs.
   * 返回一个指向底层对象的owning (!)指针，并且使intrusive_ptr实例无效。这意味着引用计数
   * 不会减1。你必须用intrusive_ptr::reclaim(ptr)把返回的指针放入intrusive_ptr来正确
   * 的析够它。
   */
  TTarget* release() noexcept {
    TTarget* result = target_;
    target_ = NullType::singleton();
    return result;
  }

  /**
   * Takes an owning pointer to TTarget* and creates an intrusive_ptr that takes
   * over ownership. That means the refcount is not increased.
   * This is the counter-part to intrusive_ptr::release() and the pointer
   * passed in *must* have been created using intrusive_ptr::release().
   * 接受一个TTarget*指针，并创建一个接管所有权的intrusive_ptr，这意味着refcount不会增加
   * 该函数与intrusive_ptr::release()对应，传递的指针必须是用intrusive_ptr::release()
   * 创建的。
   */
  static intrusive_ptr reclaim(TTarget* owning_ptr) {
    return intrusive_ptr(owning_ptr);
  }

  template <class... Args>
  static intrusive_ptr make(Args&&... args) {
    auto result = intrusive_ptr(new TTarget(std::forward<Args>(args)...));
    // We can't use retain_(), because we also have to increase weakcount
    // and because we allow raising these values from 0, which retain_()
    // has an assertion against.
    // 我们不能使用retain_()，因为我们还必须增加weakcount，并且我们允许weakcount
    // 从0开始增加，retain_()有一个反对断言。
    ++result.target_->refcount_;
    ++result.target_->weakcount_;

    return result;
  }

  /**
   * Turn a **non-owning raw pointer** to an intrusive_ptr.
   * 将一个non-owning raw pointer变为intrusive_ptr
   *
   * This method is potentially dangerous (as it can mess up refcount).
   * 该方法存在潜在风险，因为它会弄乱引用计数
   */
  static intrusive_ptr unsafe_reclaim_from_nonowning(TTarget* raw_ptr) {
    // See Note [Stack allocated intrusive_ptr_target safety]
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
        raw_ptr == NullType::singleton() || raw_ptr->refcount_.load() > 0,
        "intrusive_ptr: Can only reclaim pointers that are owned by someone");
    auto ptr = reclaim(raw_ptr); // doesn't increase refcount
    ptr.retain_();
    return ptr;
  }
};

template <
    class TTarget,
    class NullType = detail::intrusive_target_default_null_type<TTarget>,
    class... Args>
inline intrusive_ptr<TTarget, NullType> make_intrusive(Args&&... args) {
  return intrusive_ptr<TTarget, NullType>::make(std::forward<Args>(args)...);
}

template <class TTarget, class NullType>
inline void swap(
    intrusive_ptr<TTarget, NullType>& lhs,
    intrusive_ptr<TTarget, NullType>& rhs) noexcept {
  lhs.swap(rhs);
}

// To allow intrusive_ptr inside std::map or std::set, we need operator<
template <class TTarget1, class NullType1, class TTarget2, class NullType2>
inline bool operator<(
    const intrusive_ptr<TTarget1, NullType1>& lhs,
    const intrusive_ptr<TTarget2, NullType2>& rhs) noexcept {
  return lhs.get() < rhs.get();
}

template <class TTarget1, class NullType1, class TTarget2, class NullType2>
inline bool operator==(
    const intrusive_ptr<TTarget1, NullType1>& lhs,
    const intrusive_ptr<TTarget2, NullType2>& rhs) noexcept {
  return lhs.get() == rhs.get();
}

template <class TTarget1, class NullType1, class TTarget2, class NullType2>
inline bool operator!=(
    const intrusive_ptr<TTarget1, NullType1>& lhs,
    const intrusive_ptr<TTarget2, NullType2>& rhs) noexcept {
  return !operator==(lhs, rhs);
}

template <
    typename TTarget,
    class NullType = detail::intrusive_target_default_null_type<TTarget>>
class weak_intrusive_ptr final {
 private:
  static_assert(
      std::is_base_of<intrusive_ptr_target, TTarget>::value,
      "intrusive_ptr can only be used for classes that inherit from intrusive_ptr_target.");
#ifndef _WIN32
  // This static_assert triggers on MSVC
  //  error C2131: expression did not evaluate to a constant
  static_assert(
      NullType::singleton() == NullType::singleton(),
      "NullType must have a constexpr singleton() method");
#endif
  static_assert(
      std::is_same<TTarget*, decltype(NullType::singleton())>::value,
      "NullType::singleton() must return a element_type* pointer");

  TTarget* target_;

  template <class TTarget2, class NullType2>
  friend class weak_intrusive_ptr;

  void retain_() {
    if (target_ != NullType::singleton()) {
      size_t new_weakcount = ++target_->weakcount_;
      TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
          new_weakcount != 1,
          "weak_intrusive_ptr: Cannot increase weakcount after it reached zero.");
    }
  }

  void reset_() noexcept {
    if (target_ != NullType::singleton() && --target_->weakcount_ == 0) {
      delete target_;
    }
    target_ = NullType::singleton();
  }

  constexpr explicit weak_intrusive_ptr(TTarget* target) : target_(target) {}

 public:
  using element_type = TTarget;

  explicit weak_intrusive_ptr(const intrusive_ptr<TTarget, NullType>& ptr)
      : weak_intrusive_ptr(ptr.get()) {
    retain_();
  }

  weak_intrusive_ptr(weak_intrusive_ptr&& rhs) noexcept : target_(rhs.target_) {
    rhs.target_ = NullType::singleton();
  }

  template <class From, class FromNullType>
  /* implicit */ weak_intrusive_ptr(
      weak_intrusive_ptr<From, FromNullType>&& rhs) noexcept
      : target_(detail::assign_ptr_<TTarget, NullType, FromNullType>(rhs.target_)) {
    static_assert(
        std::is_convertible<From*, TTarget*>::value,
        "Type mismatch. weak_intrusive_ptr move constructor got pointer of wrong type.");
    rhs.target_ = FromNullType::singleton();
  }

  weak_intrusive_ptr(const weak_intrusive_ptr& rhs)
      : target_(rhs.target_) {
    retain_();
  }

  template <class From, class FromNullType>
  /* implicit */ weak_intrusive_ptr(
      const weak_intrusive_ptr<From, FromNullType>& rhs)
      : target_(detail::assign_ptr_<TTarget, NullType, FromNullType>(rhs.target_)) {
    static_assert(
        std::is_convertible<From*, TTarget*>::value,
        "Type mismatch. weak_intrusive_ptr copy constructor got pointer of wrong type.");
    retain_();
  }

  ~weak_intrusive_ptr() noexcept {
    reset_();
  }

  weak_intrusive_ptr& operator=(weak_intrusive_ptr&& rhs) & noexcept {
    return operator=<TTarget, NullType>(std::move(rhs));
  }

  template <class From, class FromNullType>
      weak_intrusive_ptr& operator=(
          weak_intrusive_ptr<From, FromNullType>&& rhs) &
      noexcept {
    static_assert(
        std::is_convertible<From*, TTarget*>::value,
        "Type mismatch. weak_intrusive_ptr move assignment got pointer of wrong type.");
    weak_intrusive_ptr tmp = std::move(rhs);
    swap(tmp);
    return *this;
  }

  weak_intrusive_ptr& operator=(const weak_intrusive_ptr& rhs) & noexcept {
    return operator=<TTarget, NullType>(rhs);
  }

  template <class From, class FromNullType>
      weak_intrusive_ptr& operator=(
          const weak_intrusive_ptr<From, NullType>& rhs) & {
    static_assert(
        std::is_convertible<From*, TTarget*>::value,
        "Type mismatch. weak_intrusive_ptr copy assignment got pointer of wrong type.");
    weak_intrusive_ptr tmp = rhs;
    swap(tmp);
    return *this;
  }

  void reset() noexcept {
    reset_();
  }

  void swap(weak_intrusive_ptr& rhs) noexcept {
    TTarget* tmp = target_;
    target_ = rhs.target_;
    rhs.target_ = tmp;
  }

  // NB: This should ONLY be used by the std::hash implementation
  // for weak_intrusive_ptr.  Another way you could do this is
  // friend std::hash<weak_intrusive_ptr>, but this triggers two
  // bugs:
  // 注意：这应该只被weak_intrusive_ptr的hash实现使用，你可以用另一种方法
  // friend std::hash<weak_intrusive_ptr>，但这会触发两个bug:
  //
  //  (1) It triggers an nvcc bug, where std::hash in a friend class
  //      declaration gets preprocessed into hash, which then cannot
  //      actually be found.  The error in this case looks like:
  //
  //        error: no template named 'hash'; did you mean 'std::hash'?
  //
  //  (2) On OS X, std::hash is declared as a struct, not a class.
  //      This twings:
  //
  //        error: class 'hash' was previously declared as a struct
  //        [-Werror,-Wmismatched-tags]
  //
  // Both of these are work-aroundable, but on the whole, I decided
  // it would be simpler and easier to make work if we just expose
  // an unsafe getter for target_
  //
  TTarget* _unsafe_get_target() const noexcept {
    return target_;
  }

  size_t use_count() const noexcept {
    if (target_ == NullType::singleton()) {
      return 0;
    }
    return target_->refcount_.load(); // refcount, not weakcount!
  }

  size_t weak_use_count() const noexcept {
    if (target_ == NullType::singleton()) {
      return 0;
    }
    return target_->weakcount_.load();
  }

  bool expired() const noexcept {
    return use_count() == 0;
  }

  intrusive_ptr<TTarget, NullType> lock() const noexcept {
    auto refcount = target_->refcount_.load();
    do {
      if (refcount == 0) {
        // Object already destructed, no strong references left anymore.
        // Return nullptr.
        return intrusive_ptr<TTarget, NullType>(NullType::singleton());
      }
    } while (!target_->refcount_.compare_exchange_weak(refcount, refcount + 1));
    return intrusive_ptr<TTarget, NullType>(target_);
  }

  /**
   * Returns an owning (but still only weakly referenced) pointer to the
   * underlying object and makes the weak_intrusive_ptr instance invalid.
   * That means the weakcount is not decreased.
   * You *must* put the returned pointer back into a weak_intrusive_ptr using
   * weak_intrusive_ptr::reclaim(ptr) to properly destruct it.
   * This is helpful for C APIs.
   */
  TTarget* release() noexcept {
    TTarget* result = target_;
    target_ = NullType::singleton();
    return result;
  }

  /**
   * Takes an owning (but must be weakly referenced) pointer to TTarget* and
   * creates a weak_intrusive_ptr that takes over ownership.
   * Thas means the weakcount is not increased.
   * This is the counter-part to weak_intrusive_ptr::release() and the pointer
   * passed in *must* have been created using weak_intrusive_ptr::release().
   */
  static weak_intrusive_ptr reclaim(TTarget* owning_weak_ptr) {
    // See Note [Stack allocated intrusive_ptr_target safety]
    // if refcount > 0, weakcount must be >1 for weak references to exist.
    // see weak counting explanation at top of this file.
    // if refcount == 0, weakcount only must be >0.
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
        owning_weak_ptr == NullType::singleton() ||
        owning_weak_ptr->weakcount_.load() > 1 ||
            (owning_weak_ptr->refcount_.load() == 0 &&
             owning_weak_ptr->weakcount_.load() > 0),
        "weak_intrusive_ptr: Can only weak_intrusive_ptr::reclaim() owning pointers that were created using weak_intrusive_ptr::release().");
    return weak_intrusive_ptr(owning_weak_ptr);
  }

  template <class TTarget1, class NullType1, class TTarget2, class NullType2>
  friend bool operator<(
      const weak_intrusive_ptr<TTarget1, NullType1>& lhs,
      const weak_intrusive_ptr<TTarget2, NullType2>& rhs) noexcept;
  template <class TTarget1, class NullType1, class TTarget2, class NullType2>
  friend bool operator==(
      const weak_intrusive_ptr<TTarget1, NullType1>& lhs,
      const weak_intrusive_ptr<TTarget2, NullType2>& rhs) noexcept;
};

template <class TTarget, class NullType>
inline void swap(
    weak_intrusive_ptr<TTarget, NullType>& lhs,
    weak_intrusive_ptr<TTarget, NullType>& rhs) noexcept {
  lhs.swap(rhs);
}

// To allow weak_intrusive_ptr inside std::map or std::set, we need operator<
template <class TTarget1, class NullType1, class TTarget2, class NullType2>
inline bool operator<(
    const weak_intrusive_ptr<TTarget1, NullType1>& lhs,
    const weak_intrusive_ptr<TTarget2, NullType2>& rhs) noexcept {
  return lhs.target_ < rhs.target_;
}

template <class TTarget1, class NullType1, class TTarget2, class NullType2>
inline bool operator==(
    const weak_intrusive_ptr<TTarget1, NullType1>& lhs,
    const weak_intrusive_ptr<TTarget2, NullType2>& rhs) noexcept {
  return lhs.target_ == rhs.target_;
}

template <class TTarget1, class NullType1, class TTarget2, class NullType2>
inline bool operator!=(
    const weak_intrusive_ptr<TTarget1, NullType1>& lhs,
    const weak_intrusive_ptr<TTarget2, NullType2>& rhs) noexcept {
  return !operator==(lhs, rhs);
}

// Alias for documentary purposes, to more easily distinguish
// weak raw intrusive pointers from intrusive pointers.
using weak_intrusive_ptr_target = intrusive_ptr_target;

// This namespace provides some methods for working with
// raw pointers that subclass intrusive_ptr_target.  They are not provided
// as methods on intrusive_ptr_target, because ideally you would not need these
// methods at all (use smart pointers), but if you are dealing with legacy code
// that still needs to pass around raw pointers, you may find these quite
// useful.
// 这个命名空间提供了一些方法来处理raw pointer，这些row pointer是intrusive_ptr_target的 
// 子类，这些方法没有在intrusive_ptr_target中提供，因为你用智能指针基本上不需要这些方法，，
// 但是要处理传递rwa pointer的遗留代码，这些方法还是相当管用的
//
// An important usage note: some functions are only valid if you have a
// strong raw pointer to the object, while others are only valid if you
// have a weak raw pointer to the object.  ONLY call intrusive_ptr namespace
// functions on strong pointers, and weak_intrusive_ptr namespace functions
// on weak pointers.  If you mix it up, you may get an assert failure.
// 当有一个strong raw pointer指向对象时，一些函数才是有效的
// 当有一个weak raw pointer指向对象时，另一些函数是有效的
// 在strong pointers上调用intrusive_ptr命名空间的函数
// 在weak pointers上调用weak_intrusive_ptr命名空间的函数

namespace raw {

namespace intrusive_ptr {

  // WARNING: Unlike the reclaim() API, it is NOT valid to pass
  // NullType::singleton to this function
  inline void incref(intrusive_ptr_target* self) {
    if (self) {
      ++self->refcount_;
    }
  }

  // WARNING: Unlike the reclaim() API, it is NOT valid to pass
  // NullType::singleton to this function
  inline void decref(intrusive_ptr_target* self) {
    // Let it die
    c10::intrusive_ptr<intrusive_ptr_target>::reclaim(self);
    // NB: Caller still has 'self' pointer, but it's now invalid.
    // If you want more safety, used the actual c10::intrusive_ptr class
  }

  template <typename T>
  inline T* make_weak(T* self) {
    // NB: 'this' is a strong pointer, but we return a weak pointer
    auto ptr = c10::intrusive_ptr<T>::reclaim(self);
    c10::weak_intrusive_ptr<T> wptr(ptr);
    ptr.release();
    return wptr.release();
  }

  inline uint32_t use_count(intrusive_ptr_target* self) {
    auto ptr = c10::intrusive_ptr<intrusive_ptr_target>::reclaim(self);
    auto r = ptr.use_count();
    ptr.release();
    return r;
  }

} // namespace intrusive_ptr_target

namespace weak_intrusive_ptr {

  inline void incref(weak_intrusive_ptr_target* self) {
    ++self->weakcount_;
  }

  inline void decref(weak_intrusive_ptr_target* self) {
    // Let it die
    c10::weak_intrusive_ptr<intrusive_ptr_target>::reclaim(self);
    // NB: You still "have" the 'self' pointer, but it's now invalid.
    // If you want more safety, used the actual c10::weak_intrusive_ptr class
  }

  template <typename T>
  inline T* lock(T* self) {
    auto wptr = c10::weak_intrusive_ptr<T>::reclaim(self);
    auto ptr = wptr.lock();
    wptr.release();
    return ptr.release();
  }

  // This gives the STRONG refcount of a WEAK pointer
  inline uint32_t use_count(weak_intrusive_ptr_target* self) {
    auto wptr = c10::weak_intrusive_ptr<intrusive_ptr_target>::reclaim(self);
    auto r = wptr.use_count();
    wptr.release();
    return r;
  }

} // namespace weak_intrusive_ptr_target

} // namespace raw

} // namespace c10

namespace std {
// To allow intrusive_ptr and weak_intrusive_ptr inside std::unordered_map or
// std::unordered_set, we need std::hash
template <class TTarget, class NullType>
struct hash<c10::intrusive_ptr<TTarget, NullType>> {
  size_t operator()(const c10::intrusive_ptr<TTarget, NullType>& x) const {
    return std::hash<TTarget*>()(x.get());
  }
};
template <class TTarget, class NullType>
struct hash<c10::weak_intrusive_ptr<TTarget, NullType>> {
  size_t operator()(const c10::weak_intrusive_ptr<TTarget, NullType>& x) const {
    return std::hash<TTarget*>()(x._unsafe_get_target());
  }
};
} // namespace std
