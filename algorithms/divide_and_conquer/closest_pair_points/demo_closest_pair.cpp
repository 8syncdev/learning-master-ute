#include <bits/stdc++.h>
using namespace std;

struct Point {
    double x, y;
    string name;
};

double dist(const Point& a, const Point& b) {
    double dx = a.x - b.x;
    double dy = a.y - b.y;
    return sqrt(dx * dx + dy * dy);
}

struct Answer {
    double d;
    Point a, b;
};

Answer better(const Answer& x, const Answer& y) {
    return x.d < y.d ? x : y;
}

Answer bruteForce(const vector<Point>& p, int l, int r) {
    Answer ans{1e18, {}, {}};
    for (int i = l; i <= r; ++i) {
        for (int j = i + 1; j <= r; ++j) {
            double d = dist(p[i], p[j]);
            if (d < ans.d) ans = {d, p[i], p[j]};
        }
    }
    return ans;
}

Answer closestRec(vector<Point>& px, int l, int r) {
    int n = r - l + 1;

    // Base case: ít điểm thì so sánh trực tiếp.
    if (n <= 3) {
        return bruteForce(px, l, r);
    }

    int mid = (l + r) / 2;
    double midX = px[mid].x;

    Answer leftAns = closestRec(px, l, mid);
    Answer rightAns = closestRec(px, mid + 1, r);
    Answer ans = better(leftAns, rightAns);

    double d = ans.d;

    // Tạo strip: chỉ lấy các điểm nằm gần đường chia hơn d.
    vector<Point> strip;
    for (int i = l; i <= r; ++i) {
        if (fabs(px[i].x - midX) < d) {
            strip.push_back(px[i]);
        }
    }

    sort(strip.begin(), strip.end(), [](const Point& a, const Point& b) {
        return a.y < b.y;
    });

    // Với mỗi điểm trong strip, chỉ cần so sánh với các điểm có chênh lệch y < d.
    for (int i = 0; i < (int)strip.size(); ++i) {
        for (int j = i + 1; j < (int)strip.size() && strip[j].y - strip[i].y < d; ++j) {
            double cur = dist(strip[i], strip[j]);
            if (cur < ans.d) {
                ans = {cur, strip[i], strip[j]};
                d = cur;
            }
        }
    }

    return ans;
}

Answer closestPair(vector<Point> points) {
    sort(points.begin(), points.end(), [](const Point& a, const Point& b) {
        if (a.x != b.x) return a.x < b.x;
        return a.y < b.y;
    });
    return closestRec(points, 0, (int)points.size() - 1);
}

int main() {
    vector<Point> points = {
        {2, 3, "A"},
        {12, 30, "B"},
        {40, 50, "C"},
        {5, 1, "D"},
        {12, 10, "E"},
        {3, 4, "F"}
    };

    Answer ans = closestPair(points);

    cout << fixed << setprecision(6);
    cout << "Closest pair: " << ans.a.name << "(" << ans.a.x << "," << ans.a.y << ") and "
         << ans.b.name << "(" << ans.b.x << "," << ans.b.y << ")\n";
    cout << "Distance: " << ans.d << "\n";

    return 0;
}
