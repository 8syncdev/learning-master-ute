#include <algorithm>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <vector>

using namespace std;
using Clock = chrono::steady_clock;

struct SearchResult { int index; uint64_t comparisons; };

#if defined(__GNUC__) || defined(__clang__)
__attribute__((noinline))
#endif
SearchResult sequentialSearch(const vector<int>& data, int target) {
    uint64_t comparisons = 0;
    for (size_t i = 0; i < data.size(); ++i) {
        ++comparisons;
        if (data[i] == target) return {static_cast<int>(i), comparisons};
    }
    return {-1, comparisons};
}

int binarySearchRecursiveImpl(const vector<int>& data, int target,
                              int low, int high, uint64_t& comparisons) {
    if (low > high) return -1;
    const int mid = low + (high - low) / 2;
    ++comparisons;
    if (data[mid] == target) return mid;
    if (target < data[mid])
        return binarySearchRecursiveImpl(data, target, low, mid - 1, comparisons);
    return binarySearchRecursiveImpl(data, target, mid + 1, high, comparisons);
}

SearchResult binarySearchRecursive(const vector<int>& data, int target) {
    uint64_t comparisons = 0;
    const int index = binarySearchRecursiveImpl(
        data, target, 0, static_cast<int>(data.size()) - 1, comparisons
    );
    return {index, comparisons};
}

template <typename Fn>
double benchmarkNsPerSearch(Fn fn, int repetitions) {
    volatile int sink = 0;
    const auto start = Clock::now();
    for (int r = 0; r < repetitions; ++r) sink = sink ^ fn();
    const auto stop = Clock::now();
    (void)sink;
    return chrono::duration<double, nano>(stop - start).count() / repetitions;
}

int sequentialRepetitions(int n) {
    if (n <= 10'000) return 1000;
    if (n <= 100'000) return 200;
    if (n <= 1'000'000) return 20;
    return 4;
}

int main() {
    const vector<int> sizes = {10'000, 100'000, 1'000'000, 10'000'000};
    ofstream csv("results.csv");
    csv << "n,target,sequential_ns,binary_ns,sequential_comparisons,binary_comparisons\n";

    cout << "Nguyen Phuong Anh Tu - MSSV 2611328\n";
    cout << "Sequential Search vs Binary Search (Divide & Conquer)\n\n";

    for (int n : sizes) {
        vector<int> data(static_cast<size_t>(n));
        for (int i = 0; i < n; ++i) data[static_cast<size_t>(i)] = i * 2;
        const int target = data.back();

        const SearchResult seqOnce = sequentialSearch(data, target);
        const SearchResult binOnce = binarySearchRecursive(data, target);
        if (seqOnce.index != n - 1 || binOnce.index != n - 1) return 1;

        const double seqNs = benchmarkNsPerSearch(
            [&]() { return sequentialSearch(data, target).index; },
            sequentialRepetitions(n)
        );
        const double binNs = benchmarkNsPerSearch(
            [&]() { return binarySearchRecursive(data, target).index; },
            500'000
        );

        csv << n << ',' << target << ','
            << fixed << setprecision(3) << seqNs << ',' << binNs << ','
            << seqOnce.comparisons << ',' << binOnce.comparisons << "\n";

        cout << "n=" << n
             << " | Sequential=" << seqNs / 1000.0 << " us"
             << " | Binary=" << binNs / 1000.0 << " us"
             << " | comparisons=" << seqOnce.comparisons
             << " vs " << binOnce.comparisons << "\n";
    }

    cout << "\nSequential: O(n)\n";
    cout << "Binary recursive: T(n)=T(n/2)+O(1) => O(log n)\n";
}
