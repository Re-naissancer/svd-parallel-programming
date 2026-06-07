#include "gkh.h"

#include "givens.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <limits>
#include <stdexcept>
#include <vector>

#include <mpi.h>

#if defined(SVD_DISABLE_NEON) || !defined(__ARM_NEON) || !defined(__aarch64__)
#error "This SIMD+MPI lab build requires AArch64 NEON. Compile it on the ARM server with mpicxx."
#endif

#include <arm_neon.h>

namespace
{
    struct Block
    {
        int l;
        int r;
    };

    struct MpiProfile
    {
        long long tasks = 0;
        long long steps = 0;
        long long bytes_sent = 0;
        long long bytes_recv = 0;
        long long single_block_events = 0;
        long long max_pool_size = 0;
        double wait_time = 0.0;
        double merge_time = 0.0;
        double worker_compute_time = 0.0;
    };

    enum Tags
    {
        TAG_READY = 100,
        TAG_TASK = 101,
        TAG_STOP = 102,
        TAG_RESULT = 103,
        TAG_DATA_U = 104,
        TAG_DATA_V = 105,
        TAG_DATA_B = 106,
        TAG_PROFILE = 107
    };

    static long long block_work(const Block &blk)
    {
        const long long len = static_cast<long long>(blk.r - blk.l + 1);
        return len * len;
    }

    static int read_positive_env(const char *name, int fallback)
    {
        const char *env = std::getenv(name);
        if (env == nullptr)
        {
            return fallback;
        }
        const int value = std::atoi(env);
        return value > 0 ? value : fallback;
    }

    static inline void rotate_right_adjacent_cols_neon(double *row, int c0, const float64x2_t vc, const float64x2_t vsign)
    {
        const float64x2_t ab = vld1q_f64(row + c0);
        const float64x2_t ba = vextq_f64(ab, ab, 1);
        const float64x2_t out = vaddq_f64(vmulq_f64(ab, vc), vmulq_f64(ba, vsign));
        vst1q_f64(row + c0, out);
    }

    static inline void rotate_right_pair_scalar(double *row, int c0, int c1, double c, double s)
    {
        const double a = row[c0];
        const double b = row[c1];
        row[c0] = a * c - b * s;
        row[c1] = a * s + b * c;
    }

    static void apply_left_rows_range(Matrix &M, int r0, int r1, int c_begin, int c_end, double c, double s)
    {
        c_begin = std::max(c_begin, 0);
        c_end = std::min(c_end, M.cols() - 1);
        if (c_begin > c_end)
        {
            return;
        }

        double *row0 = M.row_ptr(r0);
        double *row1 = M.row_ptr(r1);
        int j = c_begin;
        const float64x2_t vc = vdupq_n_f64(c);
        const float64x2_t vs = vdupq_n_f64(s);
        for (; j + 1 <= c_end; j += 2)
        {
            const float64x2_t a = vld1q_f64(row0 + j);
            const float64x2_t b = vld1q_f64(row1 + j);
            const float64x2_t new0 = vaddq_f64(vmulq_f64(vc, a), vmulq_f64(vs, b));
            const float64x2_t new1 = vsubq_f64(vmulq_f64(vc, b), vmulq_f64(vs, a));
            vst1q_f64(row0 + j, new0);
            vst1q_f64(row1 + j, new1);
        }
        for (; j <= c_end; ++j)
        {
            const double a = row0[j];
            const double b = row1[j];
            row0[j] = c * a + s * b;
            row1[j] = -s * a + c * b;
        }
    }

    static void apply_right_cols(Matrix &M, int c0, int c1, double c, double s)
    {
        int i = 0;
        const int rows = M.rows();

        if (c1 == c0 + 1)
        {
            const float64x2_t vc = vdupq_n_f64(c);
            const double signs[2] = {-s, s};
            const float64x2_t vsign = vld1q_f64(signs);

            for (; i + 1 < rows; i += 2)
            {
                rotate_right_adjacent_cols_neon(M.row_ptr(i), c0, vc, vsign);
                rotate_right_adjacent_cols_neon(M.row_ptr(i + 1), c0, vc, vsign);
            }
            for (; i < rows; ++i)
            {
                rotate_right_adjacent_cols_neon(M.row_ptr(i), c0, vc, vsign);
            }
            return;
        }

        for (; i < rows; ++i)
        {
            rotate_right_pair_scalar(M.row_ptr(i), c0, c1, c, s);
        }
    }

    static void apply_right_cols_range(Matrix &M, int c0, int c1, int r_begin, int r_end, double c, double s)
    {
        r_begin = std::max(r_begin, 0);
        r_end = std::min(r_end, M.rows() - 1);
        if (r_begin > r_end)
        {
            return;
        }

        int i = r_begin;
        if (c1 == c0 + 1)
        {
            const float64x2_t vc = vdupq_n_f64(c);
            const double signs[2] = {-s, s};
            const float64x2_t vsign = vld1q_f64(signs);

            for (; i + 1 <= r_end; i += 2)
            {
                rotate_right_adjacent_cols_neon(M.row_ptr(i), c0, vc, vsign);
                rotate_right_adjacent_cols_neon(M.row_ptr(i + 1), c0, vc, vsign);
            }
            for (; i <= r_end; ++i)
            {
                rotate_right_adjacent_cols_neon(M.row_ptr(i), c0, vc, vsign);
            }
            return;
        }

        for (; i <= r_end; ++i)
        {
            rotate_right_pair_scalar(M.row_ptr(i), c0, c1, c, s);
        }
    }

    static void accumulate_left_into_U(Matrix &U, int r0, int r1, double c, double s)
    {
        apply_right_cols(U, r0, r1, c, -s);
    }

    static double block_wilkinson_shift(const Matrix &B, int l, int r)
    {
        if (r == l)
        {
            return B.at(l, l) * B.at(l, l);
        }

        const double d1 = B.at(r - 1, r - 1);
        const double e1 = B.at(r - 1, r);
        const double d2 = B.at(r, r);
        const double e0 = (r - 1 > l) ? B.at(r - 2, r - 1) : 0.0;

        const double a = d1 * d1 + e0 * e0;
        const double b = d1 * e1;
        const double d = d2 * d2 + e1 * e1;

        const double tr = a + d;
        const double det = a * d - b * b;
        double disc = 0.25 * tr * tr - det;
        if (disc < 0.0)
        {
            disc = 0.0;
        }

        const double root = std::sqrt(disc);
        const double lam1 = 0.5 * tr + root;
        const double lam2 = 0.5 * tr - root;
        return (std::fabs(lam1 - d) <= std::fabs(lam2 - d)) ? lam1 : lam2;
    }

    static void cleanup_bidiagonal(Matrix &B, double tol)
    {
        const int m = B.rows();
        const int n = B.cols();

        for (int i = 0; i < m; ++i)
        {
            const int sub_col = i - 1;
            if (sub_col >= 0 && sub_col < n && std::fabs(B.at(i, sub_col)) <= tol)
            {
                B.at(i, sub_col) = 0.0;
            }

            const int super2_col = i + 2;
            if (super2_col < n && std::fabs(B.at(i, super2_col)) <= tol)
            {
                B.at(i, super2_col) = 0.0;
            }
        }
    }

    static void one_block_step(Matrix &U, Matrix &B, Matrix &V, int l, int r)
    {
        if (r <= l)
        {
            return;
        }

        const double mu = block_wilkinson_shift(B, l, r);

        double c = 1.0;
        double s = 0.0;
        double rr = 0.0;

        const double x = B.at(l, l) * B.at(l, l) - mu;
        const double z = B.at(l, l) * B.at(l, l + 1);
        givens_rotation(x, z, c, s, rr, false);
        apply_right_cols_range(B, l, l + 1, l, std::min(l + 1, r), c, s);
        apply_right_cols(V, l, l + 1, c, s);

        givens_rotation(B.at(l, l), B.at(l + 1, l), c, s, rr, true);
        apply_left_rows_range(B, l, l + 1, l, std::min(l + 2, r), c, s);
        accumulate_left_into_U(U, l, l + 1, c, s);

        for (int k = l + 1; k <= r - 1; ++k)
        {
            givens_rotation(B.at(k - 1, k), B.at(k - 1, k + 1), c, s, rr, false);
            apply_right_cols_range(B, k, k + 1, std::max(k - 1, l), std::min(k + 1, r), c, s);
            apply_right_cols(V, k, k + 1, c, s);

            givens_rotation(B.at(k, k), B.at(k + 1, k), c, s, rr, true);
            apply_left_rows_range(B, k, k + 1, std::max(k - 1, l), std::min(k + 2, r), c, s);
            accumulate_left_into_U(U, k, k + 1, c, s);
        }
    }

    static bool chase_zero_diagonal(Matrix &U, Matrix &B, Matrix &V, int k, double tol)
    {
        const int m = B.rows();
        const int n = B.cols();
        if (k < 0 || k >= n - 1)
        {
            return false;
        }
        if (std::fabs(B.at(k, k + 1)) <= tol)
        {
            return false;
        }

        bool changed = false;
        for (int i = k; i <= n - 2; ++i)
        {
            double c = 1.0;
            double s = 0.0;
            double rr = 0.0;

            givens_rotation(B.at(i, i), B.at(i, i + 1), c, s, rr, false);
            apply_right_cols_range(B, i, i + 1, i - 1, i + 1, c, s);
            apply_right_cols(V, i, i + 1, c, s);

            if (i + 1 < m)
            {
                givens_rotation(B.at(i, i), B.at(i + 1, i), c, s, rr, true);
                apply_left_rows_range(B, i, i + 1, i - 1, i + 2, c, s);
                accumulate_left_into_U(U, i, i + 1, c, s);
            }

            changed = true;
        }

        cleanup_bidiagonal(B, tol);
        return changed;
    }

    static bool handle_diagonal_zeros(Matrix &U, Matrix &B, Matrix &V, double tol)
    {
        const int n = B.cols();
        bool changed = false;

        const double eps = std::numeric_limits<double>::epsilon();
        const double diag_tol = tol;
        const double super_tol = tol * (1.0 + 10.0 * eps);

        for (int k = 0; k < n - 1; ++k)
        {
            if (std::fabs(B.at(k, k)) <= diag_tol && std::fabs(B.at(k, k + 1)) > super_tol)
            {
                changed = chase_zero_diagonal(U, B, V, k, tol) || changed;
            }
        }

        return changed;
    }

    static std::vector<Block> split_active_blocks(Matrix &B, int n, double tol)
    {
        for (int k = 0; k < n - 1; ++k)
        {
            const double a = std::fabs(B.at(k, k));
            const double d = std::fabs(B.at(k + 1, k + 1));
            const double crit = tol * (a + d + 1.0);
            if (std::fabs(B.at(k, k + 1)) <= crit)
            {
                B.at(k, k + 1) = 0.0;
            }
        }

        std::vector<Block> blocks;
        int l = 0;
        while (l < n)
        {
            int r = l;
            while (r < n - 1 && std::fabs(B.at(r, r + 1)) > 0.0)
            {
                ++r;
            }
            blocks.push_back({l, r});
            l = r + 1;
        }
        return blocks;
    }

    static void make_nonnegative_and_sort(Matrix &U, Matrix &B, Matrix &V)
    {
        const int m = B.rows();
        const int n = B.cols();

        for (int i = 0; i < n; ++i)
        {
            if (B.at(i, i) < 0.0)
            {
                B.at(i, i) = -B.at(i, i);
                for (int r = 0; r < m; ++r)
                {
                    U.at(r, i) = -U.at(r, i);
                }
            }
        }

        std::vector<int> idx(n);
        for (int i = 0; i < n; ++i)
        {
            idx[i] = i;
        }
        std::sort(idx.begin(), idx.end(), [&](int a, int b)
                  { return B.at(a, a) > B.at(b, b); });

        Matrix U2 = U;
        Matrix V2 = V;
        Matrix D(B.rows(), B.cols(), 0.0);

        for (int new_i = 0; new_i < n; ++new_i)
        {
            const int old_i = idx[new_i];
            D.at(new_i, new_i) = B.at(old_i, old_i);

            for (int r = 0; r < U.rows(); ++r)
            {
                U2.at(r, new_i) = U.at(r, old_i);
            }
            for (int r = 0; r < V.rows(); ++r)
            {
                V2.at(r, new_i) = V.at(r, old_i);
            }
        }

        U = U2;
        V = V2;
        B = D;
    }

    static void push_non_singletons(std::deque<Block> &pool, Matrix &B, int base_l, int base_r, double tol)
    {
        const int len = base_r - base_l + 1;
        Matrix local(len, len, 0.0);
        for (int i = 0; i < len; ++i)
        {
            for (int j = 0; j < len; ++j)
            {
                local.at(i, j) = B.at(base_l + i, base_l + j);
            }
        }

        const std::vector<Block> blocks = split_active_blocks(local, len, tol);
        for (const Block &blk : blocks)
        {
            if (blk.r > blk.l)
            {
                pool.push_back({base_l + blk.l, base_l + blk.r});
            }
        }
    }

    static bool has_split_or_converged(Matrix &B, int len, double tol)
    {
        const std::vector<Block> blocks = split_active_blocks(B, len, tol);
        if (blocks.empty())
        {
            return true;
        }
        if (blocks.size() > 1)
        {
            return true;
        }
        return blocks[0].r == blocks[0].l;
    }

    static int process_local_block(Matrix &Ucols, Matrix &Bblk, Matrix &Vcols, int budget, double tol)
    {
        const int len = Bblk.cols();
        int steps = 0;

        cleanup_bidiagonal(Bblk, tol);
        handle_diagonal_zeros(Ucols, Bblk, Vcols, tol);

        while (steps < budget && !has_split_or_converged(Bblk, len, tol))
        {
            one_block_step(Ucols, Bblk, Vcols, 0, len - 1);
            cleanup_bidiagonal(Bblk, tol);
            handle_diagonal_zeros(Ucols, Bblk, Vcols, tol);
            ++steps;
        }

        return steps;
    }

    static void pack_columns(const Matrix &M, int col0, int len, std::vector<double> &buf)
    {
        buf.resize(static_cast<size_t>(M.rows()) * len);
        for (int r = 0; r < M.rows(); ++r)
        {
            for (int c = 0; c < len; ++c)
            {
                buf[static_cast<size_t>(r) * len + c] = M.at(r, col0 + c);
            }
        }
    }

    static void unpack_columns(Matrix &M, int col0, int len, const std::vector<double> &buf)
    {
        for (int r = 0; r < M.rows(); ++r)
        {
            for (int c = 0; c < len; ++c)
            {
                M.at(r, col0 + c) = buf[static_cast<size_t>(r) * len + c];
            }
        }
    }

    static void pack_block(const Matrix &B, int l, int r, std::vector<double> &buf)
    {
        const int len = r - l + 1;
        buf.resize(static_cast<size_t>(len) * len);
        for (int i = 0; i < len; ++i)
        {
            for (int j = 0; j < len; ++j)
            {
                buf[static_cast<size_t>(i) * len + j] = B.at(l + i, l + j);
            }
        }
    }

    static void unpack_block(Matrix &B, int l, int r, const std::vector<double> &buf)
    {
        const int len = r - l + 1;
        for (int i = 0; i < len; ++i)
        {
            for (int j = 0; j < len; ++j)
            {
                B.at(l + i, l + j) = buf[static_cast<size_t>(i) * len + j];
            }
        }
    }

    static Matrix columns_to_matrix(const std::vector<double> &buf, int rows, int len)
    {
        Matrix M(rows, len, 0.0);
        for (int r = 0; r < rows; ++r)
        {
            for (int c = 0; c < len; ++c)
            {
                M.at(r, c) = buf[static_cast<size_t>(r) * len + c];
            }
        }
        return M;
    }

    static Matrix block_to_matrix(const std::vector<double> &buf, int len)
    {
        Matrix M(len, len, 0.0);
        for (int i = 0; i < len; ++i)
        {
            for (int j = 0; j < len; ++j)
            {
                M.at(i, j) = buf[static_cast<size_t>(i) * len + j];
            }
        }
        return M;
    }

    static void matrix_to_columns(const Matrix &M, std::vector<double> &buf)
    {
        buf.resize(static_cast<size_t>(M.rows()) * M.cols());
        for (int r = 0; r < M.rows(); ++r)
        {
            for (int c = 0; c < M.cols(); ++c)
            {
                buf[static_cast<size_t>(r) * M.cols() + c] = M.at(r, c);
            }
        }
    }

    static void matrix_to_block(const Matrix &M, std::vector<double> &buf)
    {
        buf.resize(static_cast<size_t>(M.rows()) * M.cols());
        for (int i = 0; i < M.rows(); ++i)
        {
            for (int j = 0; j < M.cols(); ++j)
            {
                buf[static_cast<size_t>(i) * M.cols() + j] = M.at(i, j);
            }
        }
    }

    static void send_task(int worker, const Block &blk, int budget, Matrix &U, Matrix &B, Matrix &V, MpiProfile &prof)
    {
        const int len = blk.r - blk.l + 1;
        int header[4] = {blk.l, blk.r, budget, len};

        std::vector<double> ubuf;
        std::vector<double> vbuf;
        std::vector<double> bbuf;
        pack_columns(U, blk.l, len, ubuf);
        pack_columns(V, blk.l, len, vbuf);
        pack_block(B, blk.l, blk.r, bbuf);

        MPI_Send(header, 4, MPI_INT, worker, TAG_TASK, MPI_COMM_WORLD);
        MPI_Send(ubuf.data(), static_cast<int>(ubuf.size()), MPI_DOUBLE, worker, TAG_DATA_U, MPI_COMM_WORLD);
        MPI_Send(vbuf.data(), static_cast<int>(vbuf.size()), MPI_DOUBLE, worker, TAG_DATA_V, MPI_COMM_WORLD);
        MPI_Send(bbuf.data(), static_cast<int>(bbuf.size()), MPI_DOUBLE, worker, TAG_DATA_B, MPI_COMM_WORLD);

        prof.bytes_sent += static_cast<long long>(4 * sizeof(int));
        prof.bytes_sent += static_cast<long long>(ubuf.size() + vbuf.size() + bbuf.size()) * static_cast<long long>(sizeof(double));
        ++prof.tasks;
    }

    static void merge_result(int worker, Matrix &U, Matrix &B, Matrix &V, std::deque<Block> &pool, double tol, MpiProfile &prof)
    {
        int header[4] = {0, 0, 0, 0};
        double times[1] = {0.0};
        MPI_Recv(header, 4, MPI_INT, worker, TAG_RESULT, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(times, 1, MPI_DOUBLE, worker, TAG_PROFILE, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        const int l = header[0];
        const int r = header[1];
        const int steps = header[2];
        const int len = header[3];

        std::vector<double> ubuf(static_cast<size_t>(U.rows()) * len);
        std::vector<double> vbuf(static_cast<size_t>(V.rows()) * len);
        std::vector<double> bbuf(static_cast<size_t>(len) * len);

        MPI_Recv(ubuf.data(), static_cast<int>(ubuf.size()), MPI_DOUBLE, worker, TAG_DATA_U, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(vbuf.data(), static_cast<int>(vbuf.size()), MPI_DOUBLE, worker, TAG_DATA_V, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(bbuf.data(), static_cast<int>(bbuf.size()), MPI_DOUBLE, worker, TAG_DATA_B, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        const double t0 = MPI_Wtime();
        unpack_columns(U, l, len, ubuf);
        unpack_columns(V, l, len, vbuf);
        unpack_block(B, l, r, bbuf);
        cleanup_bidiagonal(B, tol);
        push_non_singletons(pool, B, l, r, tol);
        prof.merge_time += MPI_Wtime() - t0;

        prof.steps += steps;
        prof.worker_compute_time += times[0];
        prof.bytes_recv += static_cast<long long>(4 * sizeof(int) + sizeof(double));
        prof.bytes_recv += static_cast<long long>(ubuf.size() + vbuf.size() + bbuf.size()) * static_cast<long long>(sizeof(double));
    }

    static void worker_loop(Matrix &U, Matrix &B, Matrix &V, double tol)
    {
        const int m = U.rows();
        const int n = V.rows();

        while (true)
        {
            MPI_Send(nullptr, 0, MPI_BYTE, 0, TAG_READY, MPI_COMM_WORLD);

            MPI_Status status;
            int header[4] = {0, 0, 0, 0};
            MPI_Recv(header, 4, MPI_INT, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
            if (status.MPI_TAG == TAG_STOP)
            {
                break;
            }

            const int l = header[0];
            const int r = header[1];
            const int budget = header[2];
            const int len = header[3];

            std::vector<double> ubuf(static_cast<size_t>(m) * len);
            std::vector<double> vbuf(static_cast<size_t>(n) * len);
            std::vector<double> bbuf(static_cast<size_t>(len) * len);
            MPI_Recv(ubuf.data(), static_cast<int>(ubuf.size()), MPI_DOUBLE, 0, TAG_DATA_U, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(vbuf.data(), static_cast<int>(vbuf.size()), MPI_DOUBLE, 0, TAG_DATA_V, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(bbuf.data(), static_cast<int>(bbuf.size()), MPI_DOUBLE, 0, TAG_DATA_B, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            Matrix Ucols = columns_to_matrix(ubuf, m, len);
            Matrix Vcols = columns_to_matrix(vbuf, n, len);
            Matrix Bblk = block_to_matrix(bbuf, len);

            const double t0 = MPI_Wtime();
            const int steps = process_local_block(Ucols, Bblk, Vcols, budget, tol);
            const double compute_time = MPI_Wtime() - t0;

            matrix_to_columns(Ucols, ubuf);
            matrix_to_columns(Vcols, vbuf);
            matrix_to_block(Bblk, bbuf);

            int result_header[4] = {l, r, steps, len};
            double times[1] = {compute_time};
            MPI_Send(result_header, 4, MPI_INT, 0, TAG_RESULT, MPI_COMM_WORLD);
            MPI_Send(times, 1, MPI_DOUBLE, 0, TAG_PROFILE, MPI_COMM_WORLD);
            MPI_Send(ubuf.data(), static_cast<int>(ubuf.size()), MPI_DOUBLE, 0, TAG_DATA_U, MPI_COMM_WORLD);
            MPI_Send(vbuf.data(), static_cast<int>(vbuf.size()), MPI_DOUBLE, 0, TAG_DATA_V, MPI_COMM_WORLD);
            MPI_Send(bbuf.data(), static_cast<int>(bbuf.size()), MPI_DOUBLE, 0, TAG_DATA_B, MPI_COMM_WORLD);
        }
    }

    static bool serial_task_pool(Matrix &U, Matrix &B, Matrix &V, int max_iter, double tol, int local_budget, MpiProfile &prof)
    {
        const int n = B.cols();
        std::deque<Block> pool;

        cleanup_bidiagonal(B, tol);
        handle_diagonal_zeros(U, B, V, tol);
        const std::vector<Block> initial = split_active_blocks(B, n, tol);
        for (const Block &blk : initial)
        {
            if (blk.r > blk.l)
            {
                pool.push_back(blk);
            }
        }

        while (!pool.empty() && prof.steps < max_iter)
        {
            prof.max_pool_size = std::max(prof.max_pool_size, static_cast<long long>(pool.size()));
            Block blk = pool.front();
            pool.pop_front();
            if (pool.empty())
            {
                ++prof.single_block_events;
            }

            const int len = blk.r - blk.l + 1;
            std::vector<double> ubuf;
            std::vector<double> vbuf;
            std::vector<double> bbuf;
            pack_columns(U, blk.l, len, ubuf);
            pack_columns(V, blk.l, len, vbuf);
            pack_block(B, blk.l, blk.r, bbuf);

            Matrix Ucols = columns_to_matrix(ubuf, U.rows(), len);
            Matrix Vcols = columns_to_matrix(vbuf, V.rows(), len);
            Matrix Bblk = block_to_matrix(bbuf, len);

            const int budget = std::min(local_budget, max_iter - static_cast<int>(prof.steps));
            const double t0 = MPI_Wtime();
            const int steps = process_local_block(Ucols, Bblk, Vcols, budget, tol);
            prof.worker_compute_time += MPI_Wtime() - t0;
            prof.steps += steps;
            ++prof.tasks;

            matrix_to_columns(Ucols, ubuf);
            matrix_to_columns(Vcols, vbuf);
            matrix_to_block(Bblk, bbuf);
            unpack_columns(U, blk.l, len, ubuf);
            unpack_columns(V, blk.l, len, vbuf);
            unpack_block(B, blk.l, blk.r, bbuf);
            cleanup_bidiagonal(B, tol);
            push_non_singletons(pool, B, blk.l, blk.r, tol);
        }

        return pool.empty();
    }

    static void process_master_local_task(Matrix &U, Matrix &B, Matrix &V, std::deque<Block> &pool,
                                          int max_iter, double tol, int local_budget, MpiProfile &prof)
    {
        Block blk = pool.front();
        pool.pop_front();
        ++prof.single_block_events;

        const int len = blk.r - blk.l + 1;
        std::vector<double> ubuf;
        std::vector<double> vbuf;
        std::vector<double> bbuf;
        pack_columns(U, blk.l, len, ubuf);
        pack_columns(V, blk.l, len, vbuf);
        pack_block(B, blk.l, blk.r, bbuf);

        Matrix Ucols = columns_to_matrix(ubuf, U.rows(), len);
        Matrix Vcols = columns_to_matrix(vbuf, V.rows(), len);
        Matrix Bblk = block_to_matrix(bbuf, len);

        const int remaining = std::max(1, max_iter - static_cast<int>(prof.steps));
        const int budget = std::min(local_budget, remaining);
        const double t0 = MPI_Wtime();
        const int steps = process_local_block(Ucols, Bblk, Vcols, budget, tol);
        prof.worker_compute_time += MPI_Wtime() - t0;
        prof.steps += steps;
        ++prof.tasks;

        const double merge0 = MPI_Wtime();
        matrix_to_columns(Ucols, ubuf);
        matrix_to_columns(Vcols, vbuf);
        matrix_to_block(Bblk, bbuf);
        unpack_columns(U, blk.l, len, ubuf);
        unpack_columns(V, blk.l, len, vbuf);
        unpack_block(B, blk.l, blk.r, bbuf);
        cleanup_bidiagonal(B, tol);
        push_non_singletons(pool, B, blk.l, blk.r, tol);
        prof.merge_time += MPI_Wtime() - merge0;
    }

    static bool mpi_master_task_pool(Matrix &U, Matrix &B, Matrix &V, int max_iter, double tol, int local_budget, int world_size, MpiProfile &prof)
    {
        const int n = B.cols();
        std::deque<Block> pool;
        std::vector<int> idle_workers;
        std::vector<char> stopped(world_size, 0);
        int active_tasks = 0;
        int stopped_count = 0;
        bool stopping = false;

        cleanup_bidiagonal(B, tol);
        handle_diagonal_zeros(U, B, V, tol);
        const std::vector<Block> initial = split_active_blocks(B, n, tol);
        for (const Block &blk : initial)
        {
            if (blk.r > blk.l)
            {
                pool.push_back(blk);
            }
        }

        auto dispatch_ready_workers = [&]() {
            while (active_tasks == 0 && pool.size() == 1 && prof.steps < max_iter)
            {
                prof.max_pool_size = std::max(prof.max_pool_size, static_cast<long long>(pool.size()));
                process_master_local_task(U, B, V, pool, max_iter, tol, local_budget, prof);
            }

            while (!idle_workers.empty() && !pool.empty() && prof.steps < max_iter)
            {
                prof.max_pool_size = std::max(prof.max_pool_size, static_cast<long long>(pool.size()));
                const int worker = idle_workers.back();
                idle_workers.pop_back();
                Block blk = pool.front();
                pool.pop_front();
                const int budget = std::max(1, std::min(local_budget, max_iter - static_cast<int>(prof.steps)));
                send_task(worker, blk, budget, U, B, V, prof);
                ++active_tasks;
            }
        };

        while (!stopping)
        {
            dispatch_ready_workers();

            if ((pool.empty() || prof.steps >= max_iter) && active_tasks == 0)
            {
                for (int worker : idle_workers)
                {
                    int stop_header[4] = {0, 0, 0, 0};
                    MPI_Send(stop_header, 4, MPI_INT, worker, TAG_STOP, MPI_COMM_WORLD);
                    if (!stopped[worker])
                    {
                        stopped[worker] = 1;
                        ++stopped_count;
                    }
                }
                idle_workers.clear();
                stopping = true;
                break;
            }

            MPI_Status status;
            const double wait0 = MPI_Wtime();
            MPI_Probe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
            prof.wait_time += MPI_Wtime() - wait0;
            const int src = status.MPI_SOURCE;

            if (status.MPI_TAG == TAG_READY)
            {
                MPI_Recv(nullptr, 0, MPI_BYTE, src, TAG_READY, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                idle_workers.push_back(src);
            }
            else if (status.MPI_TAG == TAG_RESULT)
            {
                merge_result(src, U, B, V, pool, tol, prof);
                --active_tasks;
            }
        }

        while (stopped_count < world_size - 1)
        {
            MPI_Status status;
            MPI_Probe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
            const int src = status.MPI_SOURCE;
            if (status.MPI_TAG == TAG_READY)
            {
                MPI_Recv(nullptr, 0, MPI_BYTE, src, TAG_READY, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                int stop_header[4] = {0, 0, 0, 0};
                MPI_Send(stop_header, 4, MPI_INT, src, TAG_STOP, MPI_COMM_WORLD);
                if (!stopped[src])
                {
                    stopped[src] = 1;
                    ++stopped_count;
                }
            }
            else if (status.MPI_TAG == TAG_RESULT)
            {
                merge_result(src, U, B, V, pool, tol, prof);
            }
        }

        return pool.empty();
    }

    static void broadcast_final_state(Matrix &U, Matrix &B, Matrix &V, bool &converged)
    {
        int flag = converged ? 1 : 0;
        MPI_Bcast(&flag, 1, MPI_INT, 0, MPI_COMM_WORLD);
        converged = (flag != 0);
        MPI_Bcast(U.data(), U.rows() * U.cols(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Bcast(B.data(), B.rows() * B.cols(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Bcast(V.data(), V.rows() * V.cols(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }
}

bool gkh_svd_from_bidiagonal(Matrix &U, Matrix &B, Matrix &V, int max_iter, double tol)
{
    const int m = B.rows();
    const int n = B.cols();

    if (m < n)
    {
        throw std::invalid_argument("gkh_svd_from_bidiagonal_mpi: requires m >= n");
    }
    if (U.rows() != m || U.cols() != m)
    {
        throw std::invalid_argument("gkh_svd_from_bidiagonal_mpi: U must be m x m");
    }
    if (V.rows() != n || V.cols() != n)
    {
        throw std::invalid_argument("gkh_svd_from_bidiagonal_mpi: V must be n x n");
    }

    int initialized = 0;
    MPI_Initialized(&initialized);
    if (!initialized)
    {
        int argc = 0;
        char **argv = nullptr;
        int provided = 0;
        MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    }

    int rank = 0;
    int world_size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    const int local_budget = read_positive_env("SVD_MPI_LOCAL_STEPS", 64);
    bool converged = false;
    MpiProfile prof;

    if (world_size == 1)
    {
        converged = serial_task_pool(U, B, V, max_iter, tol, local_budget, prof);
    }
    else if (rank == 0)
    {
        converged = mpi_master_task_pool(U, B, V, max_iter, tol, local_budget, world_size, prof);
    }
    else
    {
        worker_loop(U, B, V, tol);
    }

    if (rank == 0)
    {
        cleanup_bidiagonal(B, tol);
        for (int i = 0; i < n - 1; ++i)
        {
            B.at(i, i + 1) = 0.0;
        }
        make_nonnegative_and_sort(U, B, V);
    }

    if (world_size > 1)
    {
        broadcast_final_state(U, B, V, converged);
    }

    if (rank == 0)
    {
        std::printf("[mpi pool] ranks=%d local_steps=%d tasks=%lld bulge_steps=%lld max_pool=%lld single_pool_events=%lld\n",
                    world_size, local_budget, prof.tasks, prof.steps, prof.max_pool_size, prof.single_block_events);
        std::printf("[mpi pool] bytes_sent=%lld bytes_recv=%lld wait_s=%.6f merge_s=%.6f worker_compute_s=%.6f\n",
                    prof.bytes_sent, prof.bytes_recv, prof.wait_time, prof.merge_time, prof.worker_compute_time);
    }

    return converged;
}
