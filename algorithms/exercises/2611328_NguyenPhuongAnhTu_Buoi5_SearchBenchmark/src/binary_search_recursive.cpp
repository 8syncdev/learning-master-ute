#include <iostream>
#include <vector>
using namespace std;

int binarySearchRecursive(const vector<int>& a, int target,
                          int low, int high, long long& comparisons) {
    if (low > high) return -1;

    const int mid = low + (high - low) / 2;
    ++comparisons;

    if (a[mid] == target) return mid;
    if (target < a[mid])
        return binarySearchRecursive(a, target, low, mid - 1, comparisons);
    return binarySearchRecursive(a, target, mid + 1, high, comparisons);
}

int main() {
    const int n = 1'000'000;
    vector<int> a(n);
    for (int i = 0; i < n; ++i) a[i] = 2 * i;

    const int target = a.back();
    long long comparisons = 0;
    const int index = binarySearchRecursive(
        a, target, 0, static_cast<int>(a.size()) - 1, comparisons
    );

    cout << "index = " << index << "\n";
    cout << "comparisons = " << comparisons << "\n";
}
