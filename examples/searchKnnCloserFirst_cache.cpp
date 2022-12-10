// This is a test file for testing the interface
//  >>> virtual std::vector<std::pair<dist_t, labeltype>>
//  >>>    searchKnnCloserFirst(const void* query_data, size_t k) const;
// of class AlgorithmInterface

#include "../hnswlib/hnswlib.h"

#include <assert.h>

#include <vector>
#include <iostream>
#include <sys/time.h>
//#include <omp.h>
namespace
{

using idx_t = hnswlib::labeltype;

double elapsed() {
        struct timeval tv;
        gettimeofday(&tv, NULL);
        return tv.tv_sec + tv.tv_usec * 1e-6;
    }

void inline calculate_acc(idx_t* truth,idx_t* results,int k,float nq){
    float found_1 = 0;
    float found_10 = 0;
    float found_50 = 0;

    for(int i=0;i<nq;i++){
        for(int j=0;j<k;j++){
            if(results[i*k+j]==truth[i])
            {
                if(j<1)
                    found_1++;
                if(j<10)
                    found_10++;
                if(j<50)
                    found_50++;
            }
        }
    }
    std::cout<<"accuracy: Acc@1 is: "<<found_1/nq<<"\n";
    std::cout<<"accuracy: Acc@10 is: "<<found_10/nq<<"\n";
    std::cout<<"accuracy: Acc@50 is: "<<found_50/nq<<"\n";
}

void test() {
    int num_threads=4;
    int ef_s = 20;
    omp_set_num_threads(num_threads);
    int d = 128;
    idx_t n = 1000000;
    idx_t nq = 1000000;
    size_t k = 10;

    std::vector<float> query(nq * d);

    std::mt19937 rng;
    rng.seed(47);
    std::uniform_real_distribution<> distrib;

    std::vector<idx_t> truth(nq);
    for(int i=0;i<nq;i++){
        truth[i]=i;
    }
    for (idx_t i = 0; i < nq * d; ++i) {
        query[i] = distrib(rng);
    }

    hnswlib::L2Space spacefast(d);
    hnswlib::HierarchicalNSW<float> *alg_l2_fast = new hnswlib::HierarchicalNSW<float>(&spacefast, n);
    alg_l2_fast->loadIndex("index_1M_32.h",&spacefast,n);
    alg_l2_fast->ef_=ef_s;
    for(int z = 0;z<10;z++)
    {
        auto time0 = elapsed();
        auto res = alg_l2_fast->searchKnnCloserFirst(query.data(), k, nq, num_threads);
        auto time1 = elapsed();
        std::cout << "time taken for l2fast :" << (time1 - time0) << "at ef_s " << ef_s << "\n";

        std::vector<idx_t> results;
        for (auto r:res) {
            for (auto p:r) {
                results.emplace_back(p.second);
            }
        }
        calculate_acc(truth.data(), results.data(), k, nq);
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
