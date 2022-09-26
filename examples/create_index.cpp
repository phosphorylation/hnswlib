//
// Created by 陈卓 on 9/24/22.
//

#include "../hnswlib/hnswlib.h"

#include <assert.h>

#include <vector>
#include <iostream>
#include <sys/time.h>
//#include <omp.h>
namespace {

    using idx_t = hnswlib::labeltype;

    double elapsed() {
        struct timeval tv;
        gettimeofday(&tv, NULL);
        return tv.tv_sec + tv.tv_usec * 1e-6;
    }

    void test() {
        //omp_set_num_threads(16);
        int d = 128;
        idx_t n = 10000;
        idx_t nq = 10000;
        size_t k = 50;

        std::vector<float> data(n * d);
        std::vector<float> query(nq * d);

        std::mt19937 rng;
        rng.seed(47);
        std::uniform_real_distribution<> distrib;

        for (idx_t i = 0; i < n * d; ++i) {
            data[i] = distrib(rng);
        }
        for (idx_t i = 0; i < nq * d; ++i) {
            query[i] = distrib(rng);
        }


        hnswlib::L2Spacefast spacefast(d);
        hnswlib::HierarchicalNSW<float> *alg_l2_fast = new hnswlib::HierarchicalNSW<float>(&spacefast, n, 32);
#pragma omp parallel for
        for (size_t i = 0; i < n; ++i) {
            alg_l2_fast->addPoint(data.data() + d * i, i);
        }
        alg_l2_fast->saveIndex("index.h");
    }

}
int main() {
    std::cout << "Testing ..." << std::endl;
    test();
    std::cout << "Test ok" << std::endl;

    return 0;
}