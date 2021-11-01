#pragma once

#include <stddef.h>
#include <stdint.h>
#include <algorithm>
#include <cassert>
#include <cmath>
#include <exception>
#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
#include <random>
#include <sstream>
#include <vector>

namespace frk {

static constexpr double DBL_MAX = std::numeric_limits<double>::max();

// Approximate LogSumExp implementation
inline double logsumexp(double x, double y) {
    static const double MINUS_LOG_EPSILON = 50.0;
    const double vmin = std::min(x, y);
    const double vmax = std::max(x, y);
    if (vmax > vmin + MINUS_LOG_EPSILON) {
        return vmax;
    } else {
        return vmax + std::log(std::exp(vmin - vmax) + 1.0);
    }
}

// Row-major Matrix
template <class T>
class Matrix {
  public:
    Matrix() = default;

    Matrix(size_t _nrows, size_t _ncols) : m_nrows(_nrows), m_ncols(_ncols), m_mat(_nrows * _ncols) {}

    void resize(size_t _nrows, size_t _ncols) {
        m_nrows = _nrows;
        m_ncols = _ncols;
        m_mat.resize(_nrows * _ncols);
    }

    const T& operator()(size_t i, size_t j) const {
        return m_mat[i * m_ncols + j];
    }
    T& operator()(size_t i, size_t j) {
        return m_mat[i * m_ncols + j];
    }

    const T& get(size_t i, size_t j) const {
        return m_mat[i * m_ncols + j];
    }
    void set(size_t i, size_t j, const T& v) {
        m_mat[i * m_ncols + j] = v;
    }

    size_t nrows() const {
        return m_nrows;
    }
    size_t ncols() const {
        return m_ncols;
    }

  private:
    size_t m_nrows = 0;
    size_t m_ncols = 0;
    std::vector<T> m_mat;
};

using MatrixF64 = Matrix<double>;

struct Point {
    double x;
    double y;
};
using Traj = std::vector<Point>;

inline double euclidean(const Point& a, const Point& b) {
    const double diff_x = a.x - b.x;
    const double diff_y = a.y - b.y;
    return std::sqrt(diff_x * diff_x + diff_y * diff_y);
}
inline double edistance(const Point& a, const Point& b, const double gamma = 1.0) {
    return std::exp(-euclidean(a, b) / gamma);
}

inline bool compare_f64(double a, double b) {
    constexpr double e = std::numeric_limits<double>::epsilon();
    return std::abs(a - b) <= e;
}

inline double discrete_frechet_distance(const MatrixF64& dmat) {
    const size_t nrows = dmat.nrows();
    const size_t ncols = dmat.ncols();
    std::vector<double> prev_row(nrows), curr_row(nrows);

    curr_row[0] = dmat(0, 0);
    for (size_t i = 1; i < nrows; ++i) {
        curr_row[i] = std::max(curr_row[i - 1], dmat(i, 0));
    }

    for (size_t j = 1; j < ncols; ++j) {
        std::swap(prev_row, curr_row);
        curr_row[0] = std::max(prev_row[0], dmat(0, j));
        for (size_t i = 1; i < nrows; ++i) {
            const double min_dist = std::min(curr_row[i - 1], std::min(prev_row[i - 1], prev_row[i]));
            curr_row[i] = std::max(min_dist, dmat(i, j));
        }
    }
    return curr_row[nrows - 1];
}

inline Traj load_traj_from_csv(const std::string& path) {
    std::ifstream ifs(path);
    if (!ifs) {
        tfm::errorfln("cannot open %s", path);
        abort();
    }

    Traj t;
    for (std::string line; std::getline(ifs, line);) {
        std::replace(line.begin(), line.end(), ',', ' ');
        std::istringstream iss(line);

        Point p;
        iss >> p.x >> p.y;
        t.push_back(p);
    }

    return t;
}

inline std::string get_ext(const std::string& fn) {
    return fn.substr(fn.find_last_of(".") + 1);
}

}  // namespace frk