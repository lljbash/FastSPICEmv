#ifndef PREALLOC_H
#define PREALLOC_H

#include <iterator>

namespace prealloc {

// Minimum support
template <typename T>
class Vector {
public:
    Vector() = default;
    Vector(size_t size) {
        arr_ = new T[size];
        size_ = size;
    }
    ~Vector() {
        delete[] arr_;
    }
    Vector(const Vector&) = delete;
    Vector& operator=(const Vector&) = delete;

    void alloc(size_t size_) {
        delete[] arr_;
        arr_ = new T[size_];
    }

    void push_back(const T& element) { arr_[size_++] = element; }
    void push_back(T&& element) { arr_[size_++] = std::move(element); }

    template <class... Args>
    void emplace_back(Args&&... args) {
        arr_[size_++] = T(std::forward<Args>(args)...);
    }

    size_t size() const { return size_; }
    void clear() { size_ = 0; }

    // access
          T* data()       { return arr_; }
    const T* data() const { return arr_; }
          T& operator[] (size_t index)       { return arr_[index]; }
    const T& operator[] (size_t index) const { return arr_[index]; }
          T& front()       { return arr_[0]; }
    const T& front() const { return arr_[0]; }
          T& back()       { return arr_[size_ - 1]; }
    const T& back() const { return arr_[size_ - 1]; }

    // range-based for support
          T* begin()        { return arr_; }
          T* end()          { return arr_ + size_; }
    const T* cbegin() const { return arr_; }
    const T* cend() const   { return arr_ + size_; }
    const T* begin() const  { return arr_; }
    const T* end() const    { return arr_ + size_; }

private:
    T* arr_ = nullptr;
    size_t size_ = 0;
};

template <typename T>
using vector = Vector<T>;

}

#endif /* ifndef PREALLOC_H_ */
