#ifndef PADDING_H
#define PADDING_H

#include <new>
#include <iterator>
#include <stdint.h>
#include <immintrin.h>

//#define CPP17_ALIGNED_NEW

#define CACHE_LINE 64

namespace padded {

template <typename T, size_t align>
struct alignas(align) Object {
    static_assert(align >= sizeof(T));
    T obj;
    uint8_t padding[align - sizeof(T)];
};

// Minimum support
template <typename T, size_t align>
class Iterator {
    Object<T,align>* it;
public:
    using value_type        = T;
    using reference         = T&;
    using pointer           = T*;
    using difference_type   = size_t;
    using iterator_category = std::random_access_iterator_tag;
    Iterator(Object<T,align>* ptr) : it(ptr) {}
    Iterator& operator++() { ++ it; return *this; }
    Iterator operator++(int) { Iterator t(it); ++ it; return t; }
    bool operator==(const Iterator& another) const { return it == another.it; }
    bool operator!=(const Iterator& another) const { return it != another.it; }
    T& operator*() const { return it->obj; }
};

// Minimum support
template <typename T, size_t align>
class Vector {
    using Obj = Object<T,align>;
    Obj* obj;
    size_t s;
public:
    using iterator = Iterator<T, align>;
#ifdef CPP17_ALIGNED_NEW // `Object` is marked as `alignas(align)`, so C++17 should automatically call the aligned `new` operator
    Obj* get_ptr() { return obj; }
    Vector(size_t size) : s(size) {
        obj = new Obj[size]; 
    }
    ~Vector() {
        delete[] obj;
    }
#else
    Obj* get_ptr() { return std::launder(obj); }
    Vector(size_t size) : s(size) {
        obj = (Obj*) _mm_malloc(sizeof(Obj) * size, align);
        new (obj) Obj[size];
    }
    ~Vector() {
        Obj* o = get_ptr();
        for(size_t i = 0; i < s; ++i) {
            o[i].~Obj();
        }
        _mm_free(obj);
    }
#endif
          T& operator[] (ptrdiff_t index)       { return get_ptr()[index].obj; }
    const T& operator[] (ptrdiff_t index) const { return get_ptr()[index].obj; }
    iterator begin() { return iterator(get_ptr()); }
    iterator end()   { return iterator(get_ptr() + s); }
};

template <typename T>
using object = Object<T, CACHE_LINE>;
template <typename T>
using vector = Vector<T, CACHE_LINE>;

}

#endif /* ifndef SPMV_H_ */