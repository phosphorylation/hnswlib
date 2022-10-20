// This is a test file for testing the interface
//  >>> virtual std::vector<std::pair<dist_t, labeltype>>
//  >>>    searchKnnCloserFirst(const void* query_data, size_t k) const;
// of class AlgorithmInterface

#include "../hnswlib/hnswlib.h"

#include <assert.h>

#include <vector>
#include <iostream>
#include <sys/time.h>
#include <omp.h>
namespace
{

using idx_t = hnswlib::labeltype;

double elapsed() {
        struct timeval tv;
        gettimeofday(&tv, NULL);
        return tv.tv_sec + tv.tv_usec * 1e-6;
    }

void test() {
    omp_set_num_threads(16);
    int d = 128;
    idx_t n = 1000000;
    idx_t nq = 10000;
    size_t k = 50;

    std::vector<float> query(nq * d);

    std::mt19937 rng;
    rng.seed(47);
    std::uniform_real_distribution<> distrib;

    for (idx_t i = 0; i < nq * d; ++i) {
        query[i] = distrib(rng);
    }


    hnswlib::L2Space spacefast(d);
    hnswlib::HierarchicalNSW<float> *alg_l2_fast = new hnswlib::HierarchicalNSW<float>(&spacefast, n);
    alg_l2_fast->ef_=100;
    alg_l2_fast->loadIndex("index.h",&spacefast,n);
    {
        auto time0 = elapsed();
        auto res = alg_l2_fast->searchKnnCloserFirst(query.data(), k,nq,16);
        auto time1 = elapsed();
        std::cout<<"time taken for l2fast:"<<(time1-time0);
    }

    delete alg_l2_fast;
}

} // namespace

int main() {
    std::cout << "Testing ..." << std::endl;
    test();
    std::cout << "Test ok" << std::endl;

    return 0;
}
