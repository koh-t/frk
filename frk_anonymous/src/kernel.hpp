#pragma once

#include "topk.hpp"

namespace frk {

class Kernel {
  public:
    // Walk Probability
    struct wpr_tuple {
        float to_up;
        float to_left;
        float to_upleft;
    };

    using point_type = TopkFrechet::point_type;

    // O(maxsize^2)
    Kernel(size_t maxsize) : m_maxsize(maxsize), m_wprs(maxsize, maxsize) {
        // 1) Count all possible alignments
        MatrixF64 cnt_mat(maxsize, maxsize);

        for (size_t j = 0; j < maxsize; j++) {
            cnt_mat(0, j) = 0.0;
        }
        for (size_t i = 1; i < maxsize; i++) {
            cnt_mat(i, i - 1) = cnt_mat(i - 1, i);
            for (size_t j = i; j < maxsize; j++) {
                cnt_mat(i, j) = logsumexp(logsumexp(cnt_mat(i - 1, j - 1), cnt_mat(i, j - 1)), cnt_mat(i - 1, j));
            }
        }

        // 2) Compute random walk probability
        m_wprs(0, 0) = wpr_tuple{0.0, 0.0, 0.0};
        for (size_t j = 1; j < m_maxsize; j++) {
            m_wprs(0, j) = wpr_tuple{0.0, 1.0, 0.0};
        }
        for (size_t i = 1; i < m_maxsize; i++) {
            for (size_t j = 0; j < i; j++) {
                wpr_tuple t = m_wprs(j, i);
                std::swap(t.to_up, t.to_left);
                m_wprs(i, j) = t;
            }
            for (size_t j = i; j < m_maxsize; j++) {
                const float pr_up = static_cast<float>(std::exp(cnt_mat(i - 1, j) - cnt_mat(i, j)));
                const float pr_left = static_cast<float>(std::exp(cnt_mat(i, j - 1) - cnt_mat(i, j)));
                m_wprs(i, j) = wpr_tuple{pr_up, pr_left, 1.0f - (pr_up + pr_left)};
            }
        }

        // wpr_tuple t = m_wprs(m_maxsize - 1, m_maxsize - 1);
        // tfm::printfln("maxsize=%d: (%f,%f,%f)\t", maxsize, t.to_left, t.to_upleft, t.to_up);
    }

    double compute(const MatrixF64& emat, size_t nsamples, double beta, double diag_wgt, size_t seed) const {
        const size_t nrows = emat.nrows();
        const size_t ncols = emat.ncols();
        if ((nrows > m_maxsize) or (ncols > m_maxsize)) {
            throw std::runtime_error(tfm::format("error: nrows and ncols must be no more than %d", m_maxsize));
        }

        std::vector<double> sampled_e;
        sampled_e.reserve(nrows + ncols);

        // For random walk
        std::mt19937 gen(seed);
        const float adj_wgt = float(diag_wgt) - 1.0;

        // will be k4 * nsamples
        double sum = 0.0;

        for (size_t k = 0; k < nsamples; k++) {
            sampled_e.clear();

            /**
             *  (2.1) Sample an alignment by randome walk backward
             */
            size_t i = nrows - 1;
            size_t j = ncols - 1;

            while ((i != 0) and (j != 0)) {
                // tfm::printf("(%d,%d) -> ", i, j);
                sampled_e.push_back(emat.get(i, j));

                const wpr_tuple& wpr = m_wprs(i, j);
                std::uniform_real_distribution<float> dist(0.0, 1.0 + wpr.to_upleft * adj_wgt);

                const float rnd = dist(gen);

                if (rnd <= wpr.to_up) {
                    i = i - 1;  // move to upper
                } else if (rnd <= wpr.to_up + wpr.to_left) {
                    j = j - 1;  // move to left
                } else {
                    i = i - 1;  // move to upper
                    j = j - 1;  // move to left
                }
            }
            while (i != 0) {
                // tfm::printf("(%d,%d) -> ", i, j);
                sampled_e.push_back(emat.get(i, j));
                i = i - 1;  // move to upper
            }
            while (j != 0) {
                // tfm::printf("(%d,%d) -> ", i, j);
                sampled_e.push_back(emat.get(i, j));
                j = j - 1;  // move to left
            }
            assert(i == 0 and j == 0);

            // tfm::printfln("(%d,%d)", i, j);
            sampled_e.push_back(emat.get(i, j));

            /**
             *  (2.2) Compute k2
             */
            double x = 0.0;
            double z = 0.0;
            for (double e : sampled_e) {
                x += e * std::exp(-beta * e);
                z += std::exp(-beta * e);
            }
            sum += x / z;
        }

        return sum / nsamples;
    }

    // The distances between pairs are computed incrementally
    double compute_inc(const Traj& traj_x, const Traj& traj_y, double gamma,  //
                       size_t nsamples, double beta, float diag_wgt, size_t seed) const {
        const size_t nrows = traj_x.size();
        const size_t ncols = traj_y.size();
        if (nrows > m_maxsize or ncols > m_maxsize) {
            throw std::overflow_error(tfm::format("error: nrows and ncols must be no more than %d", m_maxsize));
        }

        std::vector<double> sampled_e;
        sampled_e.reserve(nrows + ncols);

        // For random walk
        std::mt19937 gen(seed);
        const float adj_wgt = diag_wgt - 1.0;

        // will be k4 * nsamples
        double sum = 0.0;

        for (size_t k = 0; k < nsamples; k++) {
            sampled_e.clear();

            /**
             *  (2.1) Sample an alignment by randome walk backward
             */
            size_t i = nrows - 1;
            size_t j = ncols - 1;

            while (i != 0 and j != 0) {
                // tfm::printf("(%d,%d) -> ", i, j);
                sampled_e.push_back(edistance(traj_x[i], traj_y[j], gamma));

                const wpr_tuple& wpr = m_wprs(i, j);
                std::uniform_real_distribution<float> dist(0.0, 1.0 + wpr.to_upleft * adj_wgt);

                const float rnd = dist(gen);

                if (rnd <= wpr.to_up) {
                    i = i - 1;  // move to upper
                } else if (rnd <= wpr.to_up + wpr.to_left) {
                    j = j - 1;  // move to left
                } else {
                    i = i - 1;  // move to upper
                    j = j - 1;  // move to left
                }
            }
            while (i != 0) {
                // tfm::printf("(%d,%d) -> ", i, j);
                sampled_e.push_back(edistance(traj_x[i], traj_y[j], gamma));
                i = i - 1;  // move to upper
            }
            while (j != 0) {
                // tfm::printf("(%d,%d) -> ", i, j);
                sampled_e.push_back(edistance(traj_x[i], traj_y[j], gamma));
                j = j - 1;  // move to left
            }
            assert(i == 0 and j == 0);

            // tfm::printfln("(%d,%d)", i, j);
            sampled_e.push_back(edistance(traj_x[i], traj_y[j], gamma));

            /**
             *  (2.2) Compute k2
             */
            double x = 0.0;
            double z = 0.0;
            for (double e : sampled_e) {
                x += e * std::exp(-beta * e);
                z += std::exp(-beta * e);
            }
            sum += x / z;
        }

        return sum / nsamples;
    }


  private:
    size_t m_maxsize = 0;
    Matrix<wpr_tuple> m_wprs;
};

}  // namespace frk
