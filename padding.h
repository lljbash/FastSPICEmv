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
template <typename T, typename Ptr, size_t align>
class Iterator {
    Ptr it;
public:
    using value_type        = T;
    using reference         = T&;
    using pointer           = T*;
    using difference_type   = size_t;
    using iterator_category = std::random_access_iterator_tag;
    Iterator(Ptr ptr) : it(ptr) {}
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
#ifdef CPP17_ALIGNED_NEW // `Object` is marked as `alignas(align)`, so C++17 should automatically call the aligned `new` operator
    Obj*       data()       { return obj; }
    const Obj* data() const { return obj; }
    Vector(size_t size) : s(size) {
        obj = new Obj[size]; 
    }
    ~Vector() {
        delete[] obj;
    }
#else
    Obj*       data()       { return std::launder(obj); }
    const Obj* data() const { return std::launder(obj); }
    Vector(size_t size) : s(size) {
        obj = (Obj*) _mm_malloc(sizeof(Obj) * size, align);
        new (obj) Obj[size];
    }
    ~Vector() {
        Obj* o = data();
        for(size_t i = 0; i < s; ++i) {
            o[i].~Obj();
        }
        _mm_free(obj);
    }
#endif
    using iterator       = Iterator<      T,       Object<T, align>*, align>;
    using const_iterator = Iterator<const T, const Object<T, align>*, align>;
          T& operator[] (ptrdiff_t index)       { return data()[index].obj; }
    const T& operator[] (ptrdiff_t index) const { return data()[index].obj; }
    iterator       begin()        { return iterator(data()); }
    iterator       end()          { return iterator(data() + s); }
    const_iterator begin()  const { return const_iterator(data()); }
    const_iterator end()    const { return const_iterator(data() + s); }
    const_iterator cbegin() const { return const_iterator(data()); }
    const_iterator cend()   const { return const_iterator(data() + s); }
};

template <typename T>
using object = Object<T, CACHE_LINE>;
template <typename T>
using vector = Vector<T, CACHE_LINE>;

}

#endif /* ifndef SPMV_H_ */
