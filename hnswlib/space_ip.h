#pragma once
#include "hnswlib.h"

namespace hnswlib {

    static float
    InnerProduct(const void *pVect1, const void *pVect2, const void *qty_ptr) {
        size_t qty = *((size_t *) qty_ptr);
        float res = 0;
        for (unsigned i = 0; i < qty; i++) {
            res += ((float *) pVect1)[i] * ((float *) pVect2)[i];
        }
        return res;

    }

    static float
    InnerProductDistance(const void *pVect1, const void *pVect2, const void *qty_ptr) {
        return 1.0f - InnerProduct(pVect1, pVect2, qty_ptr);
    }

#if defined(USE_AVX)

// Favor using AVX if available.
    static float
    InnerProductSIMD4ExtAVX(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
        float PORTABLE_ALIGN32 TmpRes[8];
        float *pVect1 = (float *) pVect1v;
        float *pVect2 = (float *) pVect2v;
        size_t qty = *((size_t *) qty_ptr);

        size_t qty16 = qty / 16;
        size_t qty4 = qty / 4;

        const float *pEnd1 = pVect1 + 16 * qty16;
        const float *pEnd2 = pVect1 + 4 * qty4;

        __m256 sum256 = _mm256_set1_ps(0);

        while (pVect1 < pEnd1) {
            //_mm_prefetch((char*)(pVect2 + 16), _MM_HINT_T0);

            __m256 v1 = _mm256_loadu_ps(pVect1);
            pVect1 += 8;
            __m256 v2 = _mm256_loadu_ps(pVect2);
            pVect2 += 8;
            sum256 = _mm256_add_ps(sum256, _mm256_mul_ps(v1, v2));

            v1 = _mm256_loadu_ps(pVect1);
            pVect1 += 8;
            v2 = _mm256_loadu_ps(pVect2);
            pVect2 += 8;
            sum256 = _mm256_add_ps(sum256, _mm256_mul_ps(v1, v2));
        }

        __m128 v1, v2;
        __m128 sum_prod = _mm_add_ps(_mm256_extractf128_ps(sum256, 0), _mm256_extractf128_ps(sum256, 1));

        while (pVect1 < pEnd2) {
            v1 = _mm_loadu_ps(pVect1);
            pVect1 += 4;
            v2 = _mm_loadu_ps(pVect2);
            pVect2 += 4;
            sum_prod = _mm_add_ps(sum_prod, _mm_mul_ps(v1, v2));
        }

        _mm_store_ps(TmpRes, sum_prod);
        float sum = TmpRes[0] + TmpRes[1] + TmpRes[2] + TmpRes[3];;
        return sum;
    }
    
    static float
    InnerProductDistanceSIMD4ExtAVX(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
        return 1.0f - InnerProductSIMD4ExtAVX(pVect1v, pVect2v, qty_ptr);
    }

#endif

#if defined(USE_SSE)

    static float
    InnerProductSIMD4ExtSSE(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
        float PORTABLE_ALIGN32 TmpRes[8];
        float *pVect1 = (float *) pVect1v;
        float *pVect2 = (float *) pVect2v;
        size_t qty = *((size_t *) qty_ptr);

        size_t qty16 = qty / 16;
        size_t qty4 = qty / 4;

        const float *pEnd1 = pVect1 + 16 * qty16;
        const float *pEnd2 = pVect1 + 4 * qty4;

        __m128 v1, v2;
        __m128 sum_prod = _mm_set1_ps(0);

        while (pVect1 < pEnd1) {
            v1 = _mm_loadu_ps(pVect1);
            pVect1 += 4;
            v2 = _mm_loadu_ps(pVect2);
            pVect2 += 4;
            sum_prod = _mm_add_ps(sum_prod, _mm_mul_ps(v1, v2));

            v1 = _mm_loadu_ps(pVect1);
            pVect1 += 4;
            v2 = _mm_loadu_ps(pVect2);
            pVect2 += 4;
            sum_prod = _mm_add_ps(sum_prod, _mm_mul_ps(v1, v2));

            v1 = _mm_loadu_ps(pVect1);
            pVect1 += 4;
            v2 = _mm_loadu_ps(pVect2);
            pVect2 += 4;
            sum_prod = _mm_add_ps(sum_prod, _mm_mul_ps(v1, v2));

            v1 = _mm_loadu_ps(pVect1);
            pVect1 += 4;
            v2 = _mm_loadu_ps(pVect2);
            pVect2 += 4;
            sum_prod = _mm_add_ps(sum_prod, _mm_mul_ps(v1, v2));
        }

        while (pVect1 < pEnd2) {
            v1 = _mm_loadu_ps(pVect1);
            pVect1 += 4;
            v2 = _mm_loadu_ps(pVect2);
            pVect2 += 4;
            sum_prod = _mm_add_ps(sum_prod, _mm_mul_ps(v1, v2));
        }

        _mm_store_ps(TmpRes, sum_prod);
        float sum = TmpRes[0] + TmpRes[1] + TmpRes[2] + TmpRes[3];

        return sum;
    }

    static float
    InnerProductDistanceSIMD4ExtSSE(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
        return 1.0f - InnerProductSIMD4ExtSSE(pVect1v, pVect2v, qty_ptr);
    }

#endif



    static float
    InnerProductSIMD16ExtAVX512(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
        float PORTABLE_ALIGN64 TmpRes[16];
        float *pVect1 = (float *) pVect1v;
        float *pVect2 = (float *) pVect2v;
        size_t qty = *((size_t *) qty_ptr);

        size_t qty16 = qty / 16;


        const float *pEnd1 = pVect1 + 16 * qty16;

        __m512 sum512 = _mm512_set1_ps(0);

        while (pVect1 < pEnd1) {
            //_mm_prefetch((char*)(pVect2 + 16), _MM_HINT_T0);

            __m512 v1 = _mm512_loadu_ps(pVect1);
            pVect1 += 16;
            __m512 v2 = _mm512_loadu_ps(pVect2);
            pVect2 += 16;
            sum512 = _mm512_add_ps(sum512, _mm512_mul_ps(v1, v2));
        }

        _mm512_store_ps(TmpRes, sum512);
        float sum = TmpRes[0] + TmpRes[1] + TmpRes[2] + TmpRes[3] + TmpRes[4] + TmpRes[5] + TmpRes[6] + TmpRes[7] + TmpRes[8] + TmpRes[9] + TmpRes[10] + TmpRes[11] + TmpRes[12] + TmpRes[13] + TmpRes[14] + TmpRes[15];

        return sum;
    }

    static float
    InnerProductDistanceSIMD16ExtAVX512(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
        return 1.0f - InnerProductSIMD16ExtAVX512(pVect1v, pVect2v, qty_ptr);
    }


#if defined(USE_AVX)

    static float
    InnerProductSIMD16ExtAVX(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
        float PORTABLE_ALIGN32 TmpRes[8];
        float *pVect1 = (float *) pVect1v;
        float *pVect2 = (float *) pVect2v;
        size_t qty = *((size_t *) qty_ptr);

        size_t qty16 = qty / 16;


        const float *pEnd1 = pVect1 + 16 * qty16;

        __m256 sum256 = _mm256_set1_ps(0);

        while (pVect1 < pEnd1) {
            //_mm_prefetch((char*)(pVect2 + 16), _MM_HINT_T0);

            __m256 v1 = _mm256_loadu_ps(pVect1);
            pVect1 += 8;
            __m256 v2 = _mm256_loadu_ps(pVect2);
            pVect2 += 8;
            sum256 = _mm256_add_ps(sum256, _mm256_mul_ps(v1, v2));

            v1 = _mm256_loadu_ps(pVect1);
            pVect1 += 8;
            v2 = _mm256_loadu_ps(pVect2);
            pVect2 += 8;
            sum256 = _mm256_add_ps(sum256, _mm256_mul_ps(v1, v2));
        }

        _mm256_store_ps(TmpRes, sum256);
        float sum = TmpRes[0] + TmpRes[1] + TmpRes[2] + TmpRes[3] + TmpRes[4] + TmpRes[5] + TmpRes[6] + TmpRes[7];

        return sum;
    }

#define ALIGNED(x) __attribute__((aligned(x)))

    static inline __m128 masked_read(int d, const float* x) {
        ALIGNED(16) float buf[4] = {0, 0, 0, 0};
        switch (d) {
            case 3:
                buf[2] = x[2];
            case 2:
                buf[1] = x[1];
            case 1:
                buf[0] = x[0];
        }
        return _mm_load_ps(buf);
        // cannot use AVX2 _mm_mask_set1_epi32
    }

    static inline __m256 masked_read_8(int d, const float* x) {
        if (d < 4) {
            __m256 res = _mm256_setzero_ps();
            res = _mm256_insertf128_ps(res, masked_read(d, x), 0);
            return res;
        } else {
            __m256 res = _mm256_setzero_ps();
            res = _mm256_insertf128_ps(res, _mm_loadu_ps(x), 0);
            res = _mm256_insertf128_ps(res, masked_read(d - 4, x + 4), 1);
            return res;
        }
    }

    static float fvec_inner_product_faiss(const float* x, const float* y, size_t d) {
        __m256 msum1 = _mm256_setzero_ps();

        while (d >= 8) {
            __m256 mx = _mm256_loadu_ps(x);
            x += 8;
            __m256 my = _mm256_loadu_ps(y);
            y += 8;
            msum1 = _mm256_add_ps(msum1, _mm256_mul_ps(mx, my));
            d -= 8;
        }

        __m128 msum2 = _mm256_extractf128_ps(msum1, 1);
        msum2 = _mm_add_ps(msum2, _mm256_extractf128_ps(msum1, 0));

        if (d >= 4) {
            __m128 mx = _mm_loadu_ps(x);
            x += 4;
            __m128 my = _mm_loadu_ps(y);
            y += 4;
            msum2 = _mm_add_ps(msum2, _mm_mul_ps(mx, my));
            d -= 4;
        }

        if (d > 0) {
            __m128 mx = masked_read(d, x);
            __m128 my = masked_read(d, y);
            msum2 = _mm_add_ps(msum2, _mm_mul_ps(mx, my));
        }

        msum2 = _mm_hadd_ps(msum2, msum2);
        msum2 = _mm_hadd_ps(msum2, msum2);
        return _mm_cvtss_f32(msum2);
    }

    static float fvec_inner_product_faiss_prefetch(const float* x, const float* y, size_t d) {
        __m256 msum1 = _mm256_setzero_ps();
        while (d >= 8) {
            __m256 mx = _mm256_loadu_ps(x);
            x += 8;
            __m256 my = _mm256_loadu_ps(y);
            y += 8;
            msum1 = _mm256_add_ps(msum1, _mm256_mul_ps(mx, my));
            d -= 8;
        }

        __m128 msum2 = _mm256_extractf128_ps(msum1, 1);
        msum2 = _mm_add_ps(msum2, _mm256_extractf128_ps(msum1, 0));

        if (d >= 4) {
            __m128 mx = _mm_loadu_ps(x);
            x += 4;
            __m128 my = _mm_loadu_ps(y);
            y += 4;
            msum2 = _mm_add_ps(msum2, _mm_mul_ps(mx, my));
            d -= 4;
        }

        if (d > 0) {
            __m128 mx = masked_read(d, x);
            __m128 my = masked_read(d, y);
            msum2 = _mm_add_ps(msum2, _mm_mul_ps(mx, my));
        }

        msum2 = _mm_hadd_ps(msum2, msum2);
        msum2 = _mm_hadd_ps(msum2, msum2);
        return _mm_cvtss_f32(msum2);
    }

    static float
    InnerProductDistanceSIMD16ExtAVX(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
        return 1.0f - InnerProductSIMD16ExtAVX(pVect1v, pVect2v, qty_ptr);
    }

#endif

#if defined(USE_SSE)

    static float
    InnerProductSIMD16ExtSSE(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
        float PORTABLE_ALIGN32 TmpRes[8];
        float *pVect1 = (float *) pVect1v;
        float *pVect2 = (float *) pVect2v;
        size_t qty = *((size_t *) qty_ptr);

        size_t qty16 = qty / 16;

        const float *pEnd1 = pVect1 + 16 * qty16;

        __m128 v1, v2;
        __m128 sum_prod = _mm_set1_ps(0);

        while (pVect1 < pEnd1) {
            v1 = _mm_loadu_ps(pVect1);
            pVect1 += 4;
            v2 = _mm_loadu_ps(pVect2);
            pVect2 += 4;
            sum_prod = _mm_add_ps(sum_prod, _mm_mul_ps(v1, v2));

            v1 = _mm_loadu_ps(pVect1);
            pVect1 += 4;
            v2 = _mm_loadu_ps(pVect2);
            pVect2 += 4;
            sum_prod = _mm_add_ps(sum_prod, _mm_mul_ps(v1, v2));

            v1 = _mm_loadu_ps(pVect1);
            pVect1 += 4;
            v2 = _mm_loadu_ps(pVect2);
            pVect2 += 4;
            sum_prod = _mm_add_ps(sum_prod, _mm_mul_ps(v1, v2));

            v1 = _mm_loadu_ps(pVect1);
            pVect1 += 4;
            v2 = _mm_loadu_ps(pVect2);
            pVect2 += 4;
            sum_prod = _mm_add_ps(sum_prod, _mm_mul_ps(v1, v2));
        }
        _mm_store_ps(TmpRes, sum_prod);
        float sum = TmpRes[0] + TmpRes[1] + TmpRes[2] + TmpRes[3];

        return sum;
    }

    static float
    InnerProductDistanceSIMD16ExtSSE(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
        return 1.0f - InnerProductSIMD16ExtSSE(pVect1v, pVect2v, qty_ptr);
    }

#endif

#if defined(USE_SSE) || defined(USE_AVX) || defined(USE_AVX512)
    DISTFUNC<float> InnerProductSIMD16Ext = InnerProductSIMD16ExtSSE;
    DISTFUNC<float> InnerProductSIMD4Ext = InnerProductSIMD4ExtSSE;
    DISTFUNC<float> InnerProductDistanceSIMD16Ext = InnerProductDistanceSIMD16ExtSSE;
    DISTFUNC<float> InnerProductDistanceSIMD4Ext = InnerProductDistanceSIMD4ExtSSE;

    static float
    InnerProductDistanceSIMD16ExtResiduals(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
        size_t qty = *((size_t *) qty_ptr);
        size_t qty16 = qty >> 4 << 4;
        float res = InnerProductSIMD16Ext(pVect1v, pVect2v, &qty16);
        float *pVect1 = (float *) pVect1v + qty16;
        float *pVect2 = (float *) pVect2v + qty16;

        size_t qty_left = qty - qty16;
        float res_tail = InnerProduct(pVect1, pVect2, &qty_left);
        return 1.0f - (res + res_tail);
    }

    static float
    InnerProductDistanceSIMD4ExtResiduals(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
        size_t qty = *((size_t *) qty_ptr);
        size_t qty4 = qty >> 2 << 2;

        float res = InnerProductSIMD4Ext(pVect1v, pVect2v, &qty4);
        size_t qty_left = qty - qty4;

        float *pVect1 = (float *) pVect1v + qty4;
        float *pVect2 = (float *) pVect2v + qty4;
        float res_tail = InnerProduct(pVect1, pVect2, &qty_left);

        return 1.0f - (res + res_tail);
    }
#endif

    class InnerProductSpace : public SpaceInterface<float> {

        DISTFUNC<float> fstdistfunc_;
        size_t data_size_;
        size_t dim_;
    public:
        InnerProductSpace(size_t dim) {
            fstdistfunc_ = InnerProductDistance;
    #if defined(USE_AVX) || defined(USE_SSE) || defined(USE_AVX512)
        #if defined(USE_AVX512)
            if (AVX512Capable()) {
                InnerProductSIMD16Ext = InnerProductSIMD16ExtAVX512;
                InnerProductDistanceSIMD16Ext = InnerProductDistanceSIMD16ExtAVX512;
            } else if (AVXCapable()) {
                InnerProductSIMD16Ext = InnerProductSIMD16ExtAVX;
                InnerProductDistanceSIMD16Ext = InnerProductDistanceSIMD16ExtAVX;
            }
        #elif defined(USE_AVX)
            if (AVXCapable()) {
                InnerProductSIMD16Ext = InnerProductSIMD16ExtAVX;
                InnerProductDistanceSIMD16Ext = InnerProductDistanceSIMD16ExtAVX;
            }
        #endif
        #if defined(USE_AVX)
            if (AVXCapable()) {
                InnerProductSIMD4Ext = InnerProductSIMD4ExtAVX;
                InnerProductDistanceSIMD4Ext = InnerProductDistanceSIMD4ExtAVX;
            }
        #endif

            if (dim % 16 == 0)
                fstdistfunc_ = InnerProductDistanceSIMD16Ext;
            else if (dim % 4 == 0)
                fstdistfunc_ = InnerProductDistanceSIMD4Ext;
            else if (dim > 16)
                fstdistfunc_ = InnerProductDistanceSIMD16ExtResiduals;
            else if (dim > 4)
                fstdistfunc_ = InnerProductDistanceSIMD4ExtResiduals;
    #endif
            dim_ = dim;
            data_size_ = dim * sizeof(float);
        }

        size_t get_data_size() {
            return data_size_;
        }

        DISTFUNC<float> get_dist_func() {
            return fstdistfunc_;
        }

        void *get_dist_func_param() {
            return &dim_;
        }

    ~InnerProductSpace() {}
    };

}
