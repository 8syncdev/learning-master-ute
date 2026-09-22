#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <unordered_set>
#include <vector>

using namespace std;

struct Point {
    double x, y;
    string name;
};

struct Answer {
    double d;
    Point a, b;
};

double dist(const Point& a, const Point& b) {
    const double dx = a.x - b.x;
    const double dy = a.y - b.y;
    return sqrt(dx * dx + dy * dy);
}

Answer better(const Answer& a, const Answer& b) {
    return a.d < b.d ? a : b;
}

Answer bruteForce(const vector<Point>& points) {
    Answer ans{numeric_limits<double>::infinity(), {}, {}};

    for (size_t i = 0; i < points.size(); ++i) {
        for (size_t j = i + 1; j < points.size(); ++j) {
            const double d = dist(points[i], points[j]);
            if (d < ans.d) ans = {d, points[i], points[j]};
        }
    }
    return ans;
}

Answer closestRec(const vector<Point>& px, const vector<Point>& py) {
    const int n = static_cast<int>(px.size());

    if (n <= 3) {
        return bruteForce(px);
    }

    const int mid = n / 2;
    const double midX = (px[mid - 1].x + px[mid].x) / 2.0;

    vector<Point> leftX(px.begin(), px.begin() + mid);
    vector<Point> rightX(px.begin() + mid, px.end());

    unordered_set<string> leftIds;
    leftIds.reserve(leftX.size() * 2);
    for (const auto& p : leftX) leftIds.insert(p.name);

    vector<Point> leftY, rightY;
    leftY.reserve(leftX.size());
    rightY.reserve(rightX.size());

    // Py đã sort theo y. Chỉ partition, KHÔNG sort lại.
    for (const auto& p : py) {
        if (leftIds.count(p.name)) leftY.push_back(p);
        else rightY.push_back(p);
    }

    Answer leftAns = closestRec(leftX, leftY);
    Answer rightAns = closestRec(rightX, rightY);
    Answer ans = better(leftAns, rightAns);
    double d = ans.d;

    // Lọc từ Py nên strip tự động giữ thứ tự y.
    vector<Point> strip;
    strip.reserve(n);
    for (const auto& p : py) {
        if (fabs(p.x - midX) < d) strip.push_back(p);
    }

    // Định lý hình học: chỉ cần kiểm tra tối đa 7 điểm tiếp theo.
    for (int i = 0; i < static_cast<int>(strip.size()); ++i) {
        const int last = min(static_cast<int>(strip.size()), i + 8);
        for (int j = i + 1; j < last; ++j) {
            if (strip[j].y - strip[i].y >= d) break;

            const double current = dist(strip[i], strip[j]);
            if (current < ans.d) {
                ans = {current, strip[i], strip[j]};
                d = current;
            }
        }
    }

    return ans;
}

Answer closestPair(vector<Point> points) {
    vector<Point> px = points;
    vector<Point> py = points;

    sort(px.begin(), px.end(), [](const Point& a, const Point& b) {
        return a.x != b.x ? a.x < b.x : a.y < b.y;
    });

    sort(py.begin(), py.end(), [](const Point& a, const Point& b) {
        return a.y != b.y ? a.y < b.y : a.x < b.x;
    });

    return closestRec(px, py);
}

int main() {
    vector<Point> points = {
        {2, 3, "A"},
        {5, 18, "B"},
        {8, 7, "C"},
        {9, 8, "D"},
        {13, 17, "E"},
        {20, 3, "F"}
    };

    const Answer ans = closestPair(points);

    cout << fixed << setprecision(6);
    cout << "Closest pair: "
         << ans.a.name << "(" << ans.a.x << "," << ans.a.y << ") and "
         << ans.b.name << "(" << ans.b.x << "," << ans.b.y << ")\n";
    cout << "Distance: " << ans.d << "\n";
    cout << "Expected: C-D, sqrt(2) ~= 1.414214\n";

    return 0;
}
