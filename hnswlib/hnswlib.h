#pragma once
#ifndef NO_MANUAL_VECTORIZATION
#ifdef __SSE__
#define USE_SSE
#ifdef __AVX__
#define USE_AVX
#ifdef __AVX512F__
#define USE_AVX512
#endif
#endif
#endif
#endif

#if defined(USE_AVX) || defined(USE_SSE)
#ifdef _MSC_VER
#include <intrin.h>
#include <stdexcept>
#include "cpu_x86.h"
void cpu_x86::cpuid(int32_t out[4], int32_t eax, int32_t ecx) {
    __cpuidex(out, eax, ecx);
}
__int64 xgetbv(unsigned int x) {
    return _xgetbv(x);
}
#else
#include <x86intrin.h>
#include <cpuid.h>
#include <stdint.h>
#include "prio_queue.h"

void cpuid(int32_t cpuInfo[4], int32_t eax, int32_t ecx) {
    __cpuid_count(eax, ecx, cpuInfo[0], cpuInfo[1], cpuInfo[2], cpuInfo[3]);
}
uint64_t xgetbv(unsigned int index) {
    uint32_t eax, edx;
    __asm__ __volatile__("xgetbv" : "=a"(eax), "=d"(edx) : "c"(index));
    return ((uint64_t)edx << 32) | eax;
}
#endif

#if defined(USE_AVX512)
#include <immintrin.h>
#endif

#if defined(__GNUC__)
#define PORTABLE_ALIGN32 __attribute__((aligned(32)))
#define PORTABLE_ALIGN64 __attribute__((aligned(64)))
#else
#define PORTABLE_ALIGN32 __declspec(align(32))
#define PORTABLE_ALIGN64 __declspec(align(64))
#endif

// Adapted from https://github.com/Mysticial/FeatureDetector
#define _XCR_XFEATURE_ENABLED_MASK  0

bool AVXCapable() {
    int cpuInfo[4];

    // CPU support
    cpuid(cpuInfo, 0, 0);
    int nIds = cpuInfo[0];

    bool HW_AVX = false;
    if (nIds >= 0x00000001) {
        cpuid(cpuInfo, 0x00000001, 0);
        HW_AVX = (cpuInfo[2] & ((int)1 << 28)) != 0;
    }

    // OS support
    cpuid(cpuInfo, 1, 0);

    bool osUsesXSAVE_XRSTORE = (cpuInfo[2] & (1 << 27)) != 0;
    bool cpuAVXSuport = (cpuInfo[2] & (1 << 28)) != 0;

    bool avxSupported = false;
    if (osUsesXSAVE_XRSTORE && cpuAVXSuport) {
        uint64_t xcrFeatureMask = xgetbv(_XCR_XFEATURE_ENABLED_MASK);
        avxSupported = (xcrFeatureMask & 0x6) == 0x6;
    }
    return HW_AVX && avxSupported;
}

bool AVX512Capable() {
    if (!AVXCapable()) return false;

    int cpuInfo[4];

    // CPU support
    cpuid(cpuInfo, 0, 0);
    int nIds = cpuInfo[0];

    bool HW_AVX512F = false;
    if (nIds >= 0x00000007) { //  AVX512 Foundation
        cpuid(cpuInfo, 0x00000007, 0);
        HW_AVX512F = (cpuInfo[1] & ((int)1 << 16)) != 0;
    }

    // OS support
    cpuid(cpuInfo, 1, 0);

    bool osUsesXSAVE_XRSTORE = (cpuInfo[2] & (1 << 27)) != 0;
    bool cpuAVXSuport = (cpuInfo[2] & (1 << 28)) != 0;

    bool avx512Supported = false;
    if (osUsesXSAVE_XRSTORE && cpuAVXSuport) {
        uint64_t xcrFeatureMask = xgetbv(_XCR_XFEATURE_ENABLED_MASK);
        avx512Supported = (xcrFeatureMask & 0xe6) == 0xe6;
    }
    return HW_AVX512F && avx512Supported;
}
#endif

#include <queue>
#include <vector>
#include <iostream>
#include <string.h>

namespace hnswlib {
    typedef size_t labeltype;

    template <typename T>
    class pairGreater {
    public:
        bool operator()(const T& p1, const T& p2) {
            return p1.first > p2.first;
        }
    };

    template<typename T>
    static void writeBinaryPOD(std::ostream &out, const T &podRef) {
        out.write((char *) &podRef, sizeof(T));
    }

    template<typename T>
    static void readBinaryPOD(std::istream &in, T &podRef) {
        in.read((char *) &podRef, sizeof(T));
    }

    template<typename MTYPE>
    using DISTFUNC = MTYPE(*)(const void *, const void *, const void *);


    template<typename MTYPE>
    class SpaceInterface {
    public:
        //virtual void search(void *);
        virtual size_t get_data_size() = 0;

        virtual DISTFUNC<MTYPE> get_dist_func() = 0;

        virtual void *get_dist_func_param() = 0;

        virtual bool need_norm(){return false;};

        virtual DISTFUNC<MTYPE> get_fast_l2(){return DISTFUNC<MTYPE>();};

        virtual ~SpaceInterface() {}
    };

    struct CompareByFirst {
        constexpr bool operator()(std::pair<float, labeltype> const &a,
                                  std::pair<float, labeltype> const &b) const noexcept {
            return a.first < b.first;
        }
    };

    template<typename dist_t>
    class AlgorithmInterface {
    public:
        virtual void addPoint(const void *datapoint, labeltype label)=0;
        virtual void batchAddPoints(const void *data_point, labeltype* label,size_t nq, int level);
        virtual  std::vector<rollbear::prio_queue<32,dist_t, unsigned int,std::greater<dist_t>>>
        searchKnn(const void *query_data, size_t k, size_t nq, size_t num_thread=1) const = 0;

        // Return k nearest neighbor in the order of closer fist
        virtual std::vector<std::vector<std::pair<dist_t, labeltype>>>
            searchKnnCloserFirst(const void* query_data, size_t k,size_t nq, size_t num_thread=1) const;

        virtual void saveIndex(const std::string &location)=0;
        virtual ~AlgorithmInterface(){
        }
    };

    template<typename dist_t>
    std::vector<std::vector<std::pair<dist_t, labeltype>>>
    AlgorithmInterface<dist_t>::searchKnnCloserFirst(const void* query_data, size_t k,size_t nq, size_t num_thread) const {
        std::vector<std::vector<std::pair<dist_t, labeltype>>> result;
        result.resize(nq);
        // here searchKnn returns the result in the order of further first
        auto ret = searchKnn(query_data, k, nq,num_thread);
        {
            for(int i=0;i<nq;i++){
                size_t sz = ret[i].size();
                int j=sz;
                result[i].resize(sz);
                while (!ret[i].empty()) {
                    result[i][--j] = ret[i].top();
                    ret[i].pop();
                }
            }
        }
        return result;
    }

    template<typename dist_t>
    void AlgorithmInterface<dist_t>::batchAddPoints(const void *data_point, labeltype *label, size_t nq, int level) {

    }
}

#include "space_l2.h"
#include "space_ip.h"
#include "bruteforce.h"
#include "hnswalg.h"
