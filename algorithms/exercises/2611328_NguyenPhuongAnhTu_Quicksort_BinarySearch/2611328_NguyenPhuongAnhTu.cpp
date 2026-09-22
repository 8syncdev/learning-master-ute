#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <vector>

using namespace std;

void printArray(const vector<int>& a) {
    cout << "[";
    for (size_t i = 0; i < a.size(); ++i) {
        if (i) cout << ", ";
        cout << a[i];
    }
    cout << "]";
}

int partitionLomuto(vector<int>& a, int low, int high, int& step) {
    const int pivot = a[high];
    int i = low - 1;

    for (int j = low; j < high; ++j) {
        if (a[j] <= pivot) {
            ++i;
            swap(a[i], a[j]);
        }
    }
    swap(a[i + 1], a[high]);

    cout << "Partition " << ++step
         << " | doan [" << low << ", " << high << "]"
         << " | pivot = " << pivot
         << " | pivot ve index " << (i + 1) << "\n";
    cout << "  Mang: ";
    printArray(a);
    cout << "\n\n";

    return i + 1;
}

void quickSort(vector<int>& a, int low, int high, int& step) {
    if (low >= high) return;
    const int p = partitionLomuto(a, low, high, step);
    quickSort(a, low, p - 1, step);
    quickSort(a, p + 1, high, step);
}

int binarySearchRecursive(const vector<int>& a, int target, int low, int high,
                          int& comparisons, bool verbose) {
    if (low > high) return -1;

    const int mid = low + (high - low) / 2;
    ++comparisons;

    if (verbose) {
        cout << "  Lan " << comparisons
             << ": low=" << low
             << ", high=" << high
             << ", mid=" << mid
             << ", a[mid]=" << a[mid] << "\n";
    }

    if (a[mid] == target) return mid;
    if (target < a[mid])
        return binarySearchRecursive(a, target, low, mid - 1, comparisons, verbose);
    return binarySearchRecursive(a, target, mid + 1, high, comparisons, verbose);
}

int main() {
    vector<int> data = {10, 33, 25, 45, 50, 100, 72, 95, 90, 85, 61};
    const int target = 72;

    cout << "NGUYEN PHUONG ANH TU - MSSV 2611328\n";
    cout << "BAI TAP QUICKSORT + TIM KIEM NHI PHAN DE QUY\n\n";

    cout << "1) QUICKSORT TANG DAN (Lomuto, chon phan tu cuoi lam pivot)\n";
    cout << "Mang ban dau: ";
    printArray(data);
    cout << "\n\n";

    int partitionStep = 0;
    quickSort(data, 0, static_cast<int>(data.size()) - 1, partitionStep);

    cout << "Mang sau khi sap xep: ";
    printArray(data);
    cout << "\n\n";

    cout << "2) TIM KIEM NHI PHAN DE QUY TIM 72\n";
    int comparisons = 0;
    const int index = binarySearchRecursive(
        data, target, 0, static_cast<int>(data.size()) - 1, comparisons, true
    );

    if (index != -1) {
        cout << "\nTim thay " << target
             << " tai index 0-based = " << index
             << ", vi tri 1-based = " << (index + 1) << ".\n";
    } else {
        cout << "\nKhong tim thay " << target << ".\n";
    }
    cout << "So lan so sanh trong vi du: " << comparisons << "\n";

    constexpr int REPEAT = 500000;
    auto start = chrono::steady_clock::now();
    int sink = 0;
    for (int k = 0; k < REPEAT; ++k) {
        int c = 0;
        sink += binarySearchRecursive(
            data, target, 0, static_cast<int>(data.size()) - 1, c, false
        );
    }
    auto stop = chrono::steady_clock::now();

    const double totalNs = chrono::duration<double, nano>(stop - start).count();
    const double avgNs = totalNs / REPEAT;

    cout << fixed << setprecision(2);
    cout << "Runtime trung binh (" << REPEAT << " lan): " << avgNs << " ns/lan\n";
    cout << "Luu y: runtime thuc te phu thuoc CPU/compiler; Big O moi la danh gia tong quat.\n";
    cout << "T(n) = T(n/2) + O(1) => O(log n).\n";

    if (sink == -1) cout << sink << '\n';
    return 0;
}
