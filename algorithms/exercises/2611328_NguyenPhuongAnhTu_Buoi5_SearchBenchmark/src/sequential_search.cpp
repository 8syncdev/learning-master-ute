#include <iostream>
#include <vector>
using namespace std;

int sequentialSearch(const vector<int>& a, int target, long long& comparisons) {
    comparisons = 0;
    for (int i = 0; i < static_cast<int>(a.size()); ++i) {
        ++comparisons;
        if (a[i] == target) return i;
    }
    return -1;
}

int main() {
    const int n = 1'000'000;
    vector<int> a(n);
    for (int i = 0; i < n; ++i) a[i] = 2 * i;

    const int target = a.back();
    long long comparisons = 0;
    const int index = sequentialSearch(a, target, comparisons);

    cout << "index = " << index << "\n";
    cout << "comparisons = " << comparisons << "\n";
}
