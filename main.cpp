#include <iostream>
#include <cstdlib>
#include <vector>
#include <cmath>
#include <numeric>
#include <algorithm>
#include "ticktock.h"
#include "pod.h"
#include <tbb/parallel_for.h>
#include <tbb/parallel_reduce.h>
#include <tbb/parallel_scan.h>
#include <tbb/blocked_range.h>
#include <tbb/partitioner.h>

// TODO: 并行化所有这些 for 循环

template <class T, class Func>
std::vector<T> fill(std::vector<T> &arr, Func const &func) {
    TICK(fill);
    int n = arr.size();
    // 使用 parallel_for 并行
    tbb::parallel_for(tbb::blocked_range<size_t>(0, n), 
    [&] (tbb::blocked_range<size_t> r) {
        for (size_t i = r.begin(); i < r.end(); i++) {
            arr[i] = func(i);
        }
    });
    TOCK(fill);
    return arr;
}

template <class T>
void saxpy(T a, std::vector<T> &x, std::vector<T> const &y) {
    TICK(saxpy);
    tbb::parallel_for(tbb::blocked_range<size_t>(0, x.size()), 
    [&] (tbb::blocked_range<size_t> r) {
        std::vector<T> local_x;
        int begin = r.begin();
        local_x.reserve(r.size());

        for (size_t i = begin; i < r.end(); i++) {
            local_x.push_back(a * x[i] + y[i]);
        }

        std::copy(local_x.begin(), local_x.end(), x.begin() + begin);
    });
    TOCK(saxpy);
}

template <class T>
T sqrtdot(std::vector<T> const &x, std::vector<T> const &y) {
    TICK(sqrtdot);

    T ret = tbb::parallel_reduce(tbb::blocked_range<size_t>(0, std::min(x.size(), y.size())), 0.0f,
    [&] (tbb::blocked_range<size_t> r, T local_sum) -> float {
        for (size_t i = r.begin(); i < r.end(); i++){
            local_sum += x[i] * y[i];
        }

        return local_sum;
    }, [&] (T x, T y) {
        return x + y;
    });

    ret = std::sqrt(ret);
    TOCK(sqrtdot);
    return ret;
}

template <class T>
T minvalue(std::vector<T> const &x) {
    TICK(minvalue);
    T ret = tbb::parallel_reduce(tbb::blocked_range<size_t>(1, x.size()), x[0], 
    [&] (tbb::blocked_range<size_t> r, T minval) {
        T local_minima = minval;
        for (size_t i = r.begin(); i < r.end(); i++) {
            local_minima = std::min(local_minima, x[i]);
        }

        return local_minima;
    }, [&] (T x, T y) {
        return std::min(x, y);
    });
    TOCK(minvalue);
    return ret;
}

template <class T>
std::vector<T> magicfilter(std::vector<T> const &x, std::vector<T> const &y) {
    TICK(magicfilter);
    std::vector<T> res;
    res.reserve(std::min(x.size(), y.size()) * 2);

    std::mutex mtx;
    int n = std::min(x.size(), y.size());
    tbb::parallel_for(tbb::blocked_range<size_t>(0, n, n / 32), 
    [&] (tbb::blocked_range<size_t> r) {
        std::vector<T> local_res;
        local_res.reserve(r.size() * 2);

        for (size_t i = r.begin(); i < r.end(); i++) {
            if (x[i] > y[i]) {
                local_res.push_back(x[i]);
            } else if (y[i] > 0.5f && y[i] > x[i]) {
                local_res.push_back(y[i]);
                local_res.push_back(x[i] * y[i]);
            }
        }

        std::lock_guard grd(mtx);
        std::copy(local_res.begin(), local_res.end(), std::back_inserter(res));
    }, tbb::simple_partitioner{});

    TOCK(magicfilter);
    return res;
}

template <class T>
T scanner(std::vector<T> &x) {
    TICK(scanner);

    T ret = tbb::parallel_scan(tbb::blocked_range<size_t>(0, x.size()), (T) 0,
    [&] (tbb::blocked_range<size_t> r, T local_sum, auto is_final) {
        for (size_t i = r.begin(); i < r.end(); i++) {
            local_sum += x[i];
            if (is_final) {
                x[i] = local_sum;
            }
        }
        return local_sum;
    }, [] (T x, T y) {
        return x + y;
    });

    TOCK(scanner);
    return ret;
}

int main() {
    size_t n = 1<<26;
    std::vector<float> x(n);
    std::vector<float> y(n);

    fill(x, [&] (size_t i) { return std::sin(i); });
    fill(y, [&] (size_t i) { return std::cos(i); });

    saxpy(0.5f, x, y);

    std::cout << sqrtdot(x, y) << std::endl;
    std::cout << minvalue(x) << std::endl;

    auto arr = magicfilter(x, y);
    std::cout << arr.size() << std::endl;

    scanner(x);
    std::cout << std::reduce(x.begin(), x.end()) << std::endl;

    return 0;
}
