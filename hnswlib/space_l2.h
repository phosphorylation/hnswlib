#pragma once
#include "hnswlib.h"
#include "space_ip.h"

namespace hnswlib {
    static float
    InnerProduct_noNormal(const void *pVect1, const void *pVect2, const void *qty_ptr) {
        size_t qty = *((size_t *) qty_ptr);
        float res = 0;
        for (unsigned i = 0; i < qty; i++) {
            res += ((float *) pVect1)[i] * ((float *) pVect2)[i];
        }
        return res;

    }

#if defined(USE_AVX)

// Favor using AVX if available.
    static float
    InnerProductSIMD4Ext_noNormal(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
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

#elif defined(USE_SSE)

    static float
    InnerProductSIMD4Ext_noNormal(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
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

#endif

#if defined(USE_AVX)

    static float
    InnerProductSIMD16Ext_noNormal(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
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

#elif defined(USE_SSE)

    static float
      InnerProductSIMD16Ext_noNormal(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
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

#endif

#if defined(USE_SSE) || defined(USE_AVX)
    static float
    InnerProductSIMD16ExtResiduals_noNormal(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
        size_t qty = *((size_t *) qty_ptr);
        size_t qty16 = qty >> 4 << 4;
        float res = InnerProductSIMD16Ext_noNormal(pVect1v, pVect2v, &qty16);
        float *pVect1 = (float *) pVect1v + qty16;
        float *pVect2 = (float *) pVect2v + qty16;

        size_t qty_left = qty - qty16;
        float res_tail = InnerProduct_noNormal(pVect1, pVect2, &qty_left);
        return res + res_tail;
    }

    static float
    InnerProductSIMD4ExtResiduals_noNormal(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
        size_t qty = *((size_t *) qty_ptr);
        size_t qty4 = qty >> 2 << 2;

        float res = InnerProductSIMD4Ext_noNormal(pVect1v, pVect2v, &qty4);
        size_t qty_left = qty - qty4;

        float *pVect1 = (float *) pVect1v + qty4;
        float *pVect2 = (float *) pVect2v + qty4;
        float res_tail = InnerProduct_noNormal(pVect1, pVect2, &qty_left);

        return res + res_tail;
    }
#endif


    static float
    L2Sqr(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
        float *pVect1 = (float *) pVect1v;
        float *pVect2 = (float *) pVect2v;
        size_t qty = *((size_t *) qty_ptr);

        float res = 0;
        for (size_t i = 0; i < qty; i++) {
            float t = *pVect1 - *pVect2;
            pVect1++;
            pVect2++;
            res += t * t;
        }
        return (res);
    }

#if defined(USE_AVX512)

    // Favor using AVX512 if available.
    static float
    L2SqrSIMD16ExtAVX512(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
        float *pVect1 = (float *) pVect1v;
        float *pVect2 = (float *) pVect2v;
        size_t qty = *((size_t *) qty_ptr);
        float PORTABLE_ALIGN64 TmpRes[16];
        size_t qty16 = qty >> 4;

        const float *pEnd1 = pVect1 + (qty16 << 4);

        __m512 diff, v1, v2;
        __m512 sum = _mm512_set1_ps(0);

        while (pVect1 < pEnd1) {
            v1 = _mm512_loadu_ps(pVect1);
            pVect1 += 16;
            v2 = _mm512_loadu_ps(pVect2);
            pVect2 += 16;
            diff = _mm512_sub_ps(v1, v2);
            // sum = _mm512_fmadd_ps(diff, diff, sum);
            sum = _mm512_add_ps(sum, _mm512_mul_ps(diff, diff));
        }

        _mm512_store_ps(TmpRes, sum);
        float res = TmpRes[0] + TmpRes[1] + TmpRes[2] + TmpRes[3] + TmpRes[4] + TmpRes[5] + TmpRes[6] +
                TmpRes[7] + TmpRes[8] + TmpRes[9] + TmpRes[10] + TmpRes[11] + TmpRes[12] +
                TmpRes[13] + TmpRes[14] + TmpRes[15];

        return (res);
}
#endif

#if defined(USE_AVX)

    // Favor using AVX if available.
    static float
    L2SqrSIMD16ExtAVX(const  void * __restrict__ pVect1v, const void * __restrict__ pVect2v, const void * __restrict__ qty_ptr) {
        float *pVect1 = (float *) pVect1v;
        float *pVect2 = (float *) pVect2v;
        size_t qty = *((size_t *) qty_ptr);
        float PORTABLE_ALIGN32 TmpRes[8];
        size_t qty16 = qty >> 4;

        const float *pEnd1 = pVect1 + (qty16 << 4);

        __m256 diff, v1, v2;
        __m256 sum = _mm256_set1_ps(0);

        while (pVect1 < pEnd1) {
            v1 = _mm256_loadu_ps(pVect1);
            pVect1 += 8;
            v2 = _mm256_loadu_ps(pVect2);
            pVect2 += 8;
            diff = _mm256_sub_ps(v1, v2);
            sum = _mm256_add_ps(sum, _mm256_mul_ps(diff, diff));

            v1 = _mm256_loadu_ps(pVect1);
            pVect1 += 8;
            v2 = _mm256_loadu_ps(pVect2);
            pVect2 += 8;
            diff = _mm256_sub_ps(v1, v2);
            sum = _mm256_add_ps(sum, _mm256_mul_ps(diff, diff));
        }

        _mm256_store_ps(TmpRes, sum);
        return TmpRes[0] + TmpRes[1] + TmpRes[2] + TmpRes[3] + TmpRes[4] + TmpRes[5] + TmpRes[6] + TmpRes[7];
    }

    static float L2SqrAVX(const void* x_v, const void* y_v, const void* qty_ptr) {
        float *x = (float *) x_v;
        float *y = (float *) y_v;
        __m256 sum1 = _mm256_setzero_ps(), sum2 = _mm256_setzero_ps();
        size_t d = *((size_t *) qty_ptr);
        const float* end = x + d;
        while (x < end) {
            auto xx1 = _mm256_loadu_ps(x);
            x += 8;
            auto yy1 = _mm256_loadu_ps(y);
            y += 8;
            auto t1 = _mm256_sub_ps(xx1, yy1);
            sum1 = _mm256_fmadd_ps(t1, t1, sum1);
            auto xx2 = _mm256_loadu_ps(x);
            x += 8;
            auto yy2 = _mm256_loadu_ps(y);
            y += 8;
            auto t2 = _mm256_sub_ps(xx2, yy2);
            sum2 = _mm256_fmadd_ps(t2,t2, sum2);
        }
        sum1 = _mm256_add_ps(sum1, sum2);
        auto sumh =
                _mm_add_ps(_mm256_castps256_ps128(sum1), _mm256_extractf128_ps(sum1, 1));
        auto tmp1 = _mm_add_ps(sumh, _mm_movehl_ps(sumh, sumh));
        auto tmp2 = _mm_add_ps(tmp1, _mm_movehdup_ps(tmp1));
        return _mm_cvtss_f32(tmp2);
    }


    static float fvec_L2sqr_faiss_prefetch(const void* __restrict__ x_v, const void* __restrict__ y_v,const void * __restrict__ qty_ptr) {
        float *x = (float *) x_v;
        float *y = (float *) y_v;
        __m256 msum1 = _mm256_setzero_ps();
        size_t d = *((size_t *) qty_ptr);
        __m256 mx;
        __m256 my;
        while (d >= 16) {
            mx = _mm256_loadu_ps(x);
            x += 8;
            my = _mm256_loadu_ps(y);
            y += 8;
            const __m256 a_m_b1 = _mm256_sub_ps(mx, my);
            msum1 = _mm256_add_ps(msum1, _mm256_mul_ps(a_m_b1, a_m_b1));
            mx = _mm256_loadu_ps(x);
            x += 8;
            my = _mm256_loadu_ps(y);
            y += 8;
            const __m256 a_m_b2 = _mm256_sub_ps(mx, my);
            msum1 = _mm256_add_ps(msum1, _mm256_mul_ps(a_m_b2, a_m_b2));
            d -= 16;
        }

        __m128 msum2 = _mm256_extractf128_ps(msum1, 1);
        msum2 = _mm_add_ps(msum2, _mm256_extractf128_ps(msum1, 0));

        if (d >= 4) {
            __m128 mx = _mm_loadu_ps(x);
            x += 4;
            __m128 my = _mm_loadu_ps(y);
            y += 4;
            const __m128 a_m_b1 = _mm_sub_ps(mx, my);
            msum2 = _mm_add_ps(msum2, _mm_mul_ps(a_m_b1, a_m_b1));
            d -= 4;
        }

        if (d > 0) {
            __m128 mx = masked_read(d, x);
            __m128 my = masked_read(d, y);
            __m128 a_m_b1 = _mm_sub_ps(mx, my);
            msum2 = _mm_add_ps(msum2, _mm_mul_ps(a_m_b1, a_m_b1));
        }

        msum2 = _mm_hadd_ps(msum2, msum2);
        msum2 = _mm_hadd_ps(msum2, msum2);
        return _mm_cvtss_f32(msum2);
    }

#endif

#if defined(USE_SSE)

    static float
    L2SqrSIMD16ExtSSE(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
        float *pVect1 = (float *) pVect1v;
        float *pVect2 = (float *) pVect2v;
        size_t qty = *((size_t *) qty_ptr);
        float PORTABLE_ALIGN32 TmpRes[8];
        size_t qty16 = qty >> 4;

        const float *pEnd1 = pVect1 + (qty16 << 4);

        __m128 diff, v1, v2;
        __m128 sum = _mm_set1_ps(0);

        while (pVect1 < pEnd1) {
            //_mm_prefetch((char*)(pVect2 + 16), _MM_HINT_T0);
            v1 = _mm_loadu_ps(pVect1);
            pVect1 += 4;
            v2 = _mm_loadu_ps(pVect2);
            pVect2 += 4;
            diff = _mm_sub_ps(v1, v2);
            sum = _mm_add_ps(sum, _mm_mul_ps(diff, diff));

            v1 = _mm_loadu_ps(pVect1);
            pVect1 += 4;
            v2 = _mm_loadu_ps(pVect2);
            pVect2 += 4;
            diff = _mm_sub_ps(v1, v2);
            sum = _mm_add_ps(sum, _mm_mul_ps(diff, diff));

            v1 = _mm_loadu_ps(pVect1);
            pVect1 += 4;
            v2 = _mm_loadu_ps(pVect2);
            pVect2 += 4;
            diff = _mm_sub_ps(v1, v2);
            sum = _mm_add_ps(sum, _mm_mul_ps(diff, diff));

            v1 = _mm_loadu_ps(pVect1);
            pVect1 += 4;
            v2 = _mm_loadu_ps(pVect2);
            pVect2 += 4;
            diff = _mm_sub_ps(v1, v2);
            sum = _mm_add_ps(sum, _mm_mul_ps(diff, diff));
        }

        _mm_store_ps(TmpRes, sum);
        return TmpRes[0] + TmpRes[1] + TmpRes[2] + TmpRes[3];
    }
#endif

#if defined(USE_SSE) || defined(USE_AVX) || defined(USE_AVX512)
    DISTFUNC<float> L2SqrSIMD16Ext = L2SqrSIMD16ExtSSE;

    static float
    L2SqrSIMD16ExtResiduals(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
        size_t qty = *((size_t *) qty_ptr);
        size_t qty16 = qty >> 4 << 4;
        float res = L2SqrSIMD16Ext(pVect1v, pVect2v, &qty16);
        float *pVect1 = (float *) pVect1v + qty16;
        float *pVect2 = (float *) pVect2v + qty16;

        size_t qty_left = qty - qty16;
        float res_tail = L2Sqr(pVect1, pVect2, &qty_left);
        return (res + res_tail);
    }
#endif


#if defined(USE_SSE)
    static float
    L2SqrSIMD4Ext(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
        float PORTABLE_ALIGN32 TmpRes[8];
        float *pVect1 = (float *) pVect1v;
        float *pVect2 = (float *) pVect2v;
        size_t qty = *((size_t *) qty_ptr);


        size_t qty4 = qty >> 2;

        const float *pEnd1 = pVect1 + (qty4 << 2);

        __m128 diff, v1, v2;
        __m128 sum = _mm_set1_ps(0);

        while (pVect1 < pEnd1) {
            v1 = _mm_loadu_ps(pVect1);
            pVect1 += 4;
            v2 = _mm_loadu_ps(pVect2);
            pVect2 += 4;
            diff = _mm_sub_ps(v1, v2);
            sum = _mm_add_ps(sum, _mm_mul_ps(diff, diff));
        }
        _mm_store_ps(TmpRes, sum);
        return TmpRes[0] + TmpRes[1] + TmpRes[2] + TmpRes[3];
    }

    static float
    L2SqrSIMD4ExtResiduals(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
        size_t qty = *((size_t *) qty_ptr);
        size_t qty4 = qty >> 2 << 2;

        float res = L2SqrSIMD4Ext(pVect1v, pVect2v, &qty4);
        size_t qty_left = qty - qty4;

        float *pVect1 = (float *) pVect1v + qty4;
        float *pVect2 = (float *) pVect2v + qty4;
        float res_tail = L2Sqr(pVect1, pVect2, &qty_left);

        return (res + res_tail);
    }
#endif

    class L2Space : public SpaceInterface<float> {

        DISTFUNC<float> fstdistfunc_;
        size_t data_size_;
        size_t dim_;
    public:
        L2Space(size_t dim) {
            fstdistfunc_ = L2Sqr;
    #if defined(USE_SSE) || defined(USE_AVX) || defined(USE_AVX512)
        #if defined(USE_AVX512)
            if (AVX512Capable())
                L2SqrSIMD16Ext = L2SqrSIMD16ExtAVX512;
            else if (AVXCapable())
                L2SqrSIMD16Ext = L2SqrSIMD16ExtAVX;
        #elif defined(USE_AVX)
            if (AVXCapable())
                L2SqrSIMD16Ext = L2SqrSIMD16ExtAVX;
        #endif

            if (dim % 16 == 0)
                fstdistfunc_ = L2SqrSIMD16Ext;
            else if (dim % 4 == 0)
                fstdistfunc_ = L2SqrSIMD4Ext;
            else if (dim > 16)
                fstdistfunc_ = L2SqrSIMD16ExtResiduals;
            else if (dim > 4)
                fstdistfunc_ = L2SqrSIMD4ExtResiduals;
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

        ~L2Space() {}
    };

    static int
    L2SqrI4x(const void *__restrict pVect1, const void *__restrict pVect2, const void *__restrict qty_ptr) {

        size_t qty = *((size_t *) qty_ptr);
        int res = 0;
        unsigned char *a = (unsigned char *) pVect1;
        unsigned char *b = (unsigned char *) pVect2;

        qty = qty >> 2;
        for (size_t i = 0; i < qty; i++) {

            res += ((*a) - (*b)) * ((*a) - (*b));
            a++;
            b++;
            res += ((*a) - (*b)) * ((*a) - (*b));
            a++;
            b++;
            res += ((*a) - (*b)) * ((*a) - (*b));
            a++;
            b++;
            res += ((*a) - (*b)) * ((*a) - (*b));
            a++;
            b++;
        }
        return (res);
    }

    static int L2SqrI(const void* __restrict pVect1, const void* __restrict pVect2, const void* __restrict qty_ptr) {
        size_t qty = *((size_t*)qty_ptr);
        int res = 0;
        unsigned char* a = (unsigned char*)pVect1;
        unsigned char* b = (unsigned char*)pVect2;

        for(size_t i = 0; i < qty; i++)
        {
            res += ((*a) - (*b)) * ((*a) - (*b));
            a++;
            b++;
        }
        return (res);
    }

    class L2SpaceI : public SpaceInterface<int> {

        DISTFUNC<int> fstdistfunc_;
        size_t data_size_;
        size_t dim_;
    public:
        L2SpaceI(size_t dim) {
            if(dim % 4 == 0) {
                fstdistfunc_ = L2SqrI4x;
            }
            else {
                fstdistfunc_ = L2SqrI;
            }
            dim_ = dim;
            data_size_ = dim * sizeof(unsigned char);
        }

        size_t get_data_size() {
            return data_size_;
        }

        DISTFUNC<int> get_dist_func() {
            return fstdistfunc_;
        }

        void *get_dist_func_param() {
            return &dim_;
        }

        ~L2SpaceI() {}
    };


    class L2Spacefast : public SpaceInterface<float> {

        DISTFUNC<float> fstdistfunc_;
        DISTFUNC<float> fstdistfunc_2;
        size_t data_size_;
        size_t dim_;
    public:
        L2Spacefast(size_t dim) {

            fstdistfunc_ = L2Sqr;

#if defined(USE_SSE) || defined(USE_AVX) || defined(USE_AVX512)
#if defined(USE_AVX512)
            if (AVX512Capable())
                L2SqrSIMD16Ext = L2SqrSIMD16ExtAVX512;
            else if (AVXCapable())
                L2SqrSIMD16Ext = L2SqrSIMD16ExtAVX;
#elif defined(USE_AVX)
            if (AVXCapable())
                L2SqrSIMD16Ext = L2SqrAVX;
#endif

            if (dim % 16 == 0)
                fstdistfunc_ = L2SqrSIMD16Ext;
            else if (dim % 4 == 0)
                fstdistfunc_ = L2SqrSIMD4Ext;
            else if (dim > 16)
                fstdistfunc_ = L2SqrSIMD16ExtResiduals;
            else if (dim > 4)
                fstdistfunc_ = L2SqrSIMD4ExtResiduals;
#endif
            fstdistfunc_2 = InnerProduct_noNormal;
#if defined(USE_AVX) || defined(USE_SSE)
            if (dim % 16 == 0)
                fstdistfunc_2 = InnerProductSIMD16Ext_noNormal;
            else if (dim % 4 == 0)
                fstdistfunc_2 = InnerProductSIMD4Ext_noNormal;
            else if (dim > 16)
                fstdistfunc_2 = InnerProductSIMD16ExtResiduals_noNormal;
            else if (dim > 4)
                fstdistfunc_2 = InnerProductSIMD4ExtResiduals_noNormal;
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

        DISTFUNC<float> get_fast_l2() {
            return fstdistfunc_2;
        }

        bool need_norm() {
            return true;
        }

        ~L2Spacefast() {}
    };



}
