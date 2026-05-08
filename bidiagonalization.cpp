// // bidiagonalization.cpp
// // 将 m×n 矩阵（本框架保证m ≥ n）通过 Householder 变换化为上双对角形
// //
// // 算法说明（你需要结合代码看）：
// // 对上双对角化，需要交替从左侧和右侧应用 Householder 变换：
// // 第 k 步（k = 0, 1, ..., n-1）：
// //    - 从左侧作用 H_k，消去第 k 列中位置 (k+1,k), (k+2,k), ..., (m-1,k) 的元素
// //    - 如果 k < n-2，从右侧作用 V_k，消去第 k 行中位置 (k,k+2), (k,k+3), ..., (k,n-1) 的元素
// //
// // 例如，对一个 4x4 矩阵 A，第一步 k=0：
// //   - 从左侧作用 H_0，消去 A(1,0), A(2,0), A(3,0)，得到 B_0，同时更新 U = U * H_0
// //   - 从右侧作用 V_0，消去 B_0(0,2)，B_0(0,3)，得到 B_1，同时更新 V = V * V_0
// //
// // 最终得到上双对角矩阵 B，只有主对角线和上次对角线有非零元素
// //
// // 本组件输出：A = U * B * V^T
// // 其中 U（m×m）和 V（n×n）均为正交矩阵，B（m×n）为上双对角矩阵

// #include "matrix.h"
// #include <cmath>
// #include <stdexcept>
// #include <vector>

// // 辅助函数，计算向量的范数（平方和开根）
// static double vector_norm(const std::vector<double> &v)
// {
//     double sum = 0.0;
//     for (double x : v)
//         sum += x * x;
//     return std::sqrt(sum);
// }

// // 将 m×n 矩阵 A（m ≥ n）化为上双对角形，返回 B，同时输出 U（m×m）和 V（n×n）
// Matrix to_bidiagonal(const Matrix &A, Matrix &U, Matrix &V)
// {
//     if (A.rows() < A.cols())
//     {
//         throw std::invalid_argument("to_bidiagonal: requires m >= n");
//     }

//     const int m = A.rows();
//     const int n = A.cols();
//     Matrix B = A;

//     // U = I_m，V = I_n
//     U = Matrix(m, m, 0.0);
//     for (int i = 0; i < m; ++i)
//         U.at(i, i) = 1.0;
//     V = Matrix(n, n, 0.0);
//     for (int i = 0; i < n; ++i)
//         V.at(i, i) = 1.0;

//     for (int k = 0; k < n; ++k)
//     {
//         // ================================================================
//         // 步骤 1: 从左侧作用 Householder 变换，消去第 k 列中对角线以下的元素
//         // ================================================================

//         // 提取第 k 列从第 k 行往下的子向量
//         // 例如：k=0 时提取 A(0:m-1, 0)，长度为 m-k+1 ; k=1 时提取 A(1:m-1, 1)
//         std::vector<double> x(m - k);
//         for (int i = 0; i < m - k; ++i)
//         {
//             x[i] = B.at(k + i, k);
//         }

//         double norm_x = vector_norm(x);

//         if (norm_x > 1e-14 && k < m - 1)
//         {
//             // sign(x[0])：此处规定 x[0]==0 时取 +1
//             double sigma = (x[0] >= 0.0 ? 1.0 : -1.0) * norm_x;

//             // 实际上这里是+或者-都可以，手册里 Householder 一节是 -αe_1
//             // 但我们这里 sigma 取了 sign(x[0]) * norm_x，所以是 +sigma * e_1 的形式
//             std::vector<double> v(x);
//             v[0] += sigma; // v = x + sigma * e_1

//             // 计算 v^T v
//             double vTv = 0.0;
//             for (double vi : v)
//                 vTv += vi * vi;

//             // TODO(SIMD编程)：此处的Householder变换可以通过 SIMD 指令加速，你可以尝试实现
//             if (vTv > 1e-28)
//             {
//                 const double beta = 2.0 / vTv;

//                 // 手册里的 Householder 矩阵定义为 H = I - beta * v * v^T，其中 beta = 2 / (v^T v)
//                 // 从左侧作用 H：B_new = H * B_old = B_old - beta * v * (v^T * B_old)
//                 std::vector<double> w(n - k, 0.0);
//                 for (int j = 0; j < n - k; ++j)
//                     for (int i = 0; i < m - k; ++i)
//                         w[j] += v[i] * B.at(k + i, k + j);
//                 for (int i = 0; i < m - k; ++i)
//                     for (int j = 0; j < n - k; ++j)
//                         B.at(k + i, k + j) -= beta * v[i] * w[j];

//                 // 累积 U：U_new = U_old * H_k
//                 // U[:, k:m] -= beta * (U[:, k:m] * v) * v^T
//                 std::vector<double> wU(m, 0.0);
//                 for (int i = 0; i < m; ++i)
//                     for (int j = 0; j < m - k; ++j)
//                         wU[i] += U.at(i, k + j) * v[j];
//                 for (int i = 0; i < m; ++i)
//                     for (int j = 0; j < m - k; ++j)
//                         U.at(i, k + j) -= beta * wU[i] * v[j];
//             }
//         }

//         // 清除第 k 列中对角线以下的元素
//         // 理论上应为 0，但不能完全保证全是 0，这里强制置零
//         for (int i = k + 1; i < m; ++i)
//         {
//             B.at(i, k) = 0.0;
//         }

//         // ================================================================
//         // 步骤 2: 从右侧作用 Householder 变换，消去第 k 行中 (k,k+2) 及右边的元素
//         //        （只在 k < n-2 时需要）
//         // ================================================================

//         if (k < n - 2)
//         {
//             // 提取第 k 行从第 k+1 列往右的子向量（长度 n-k-1）
//             std::vector<double> y(n - k - 1);
//             for (int j = 0; j < n - k - 1; ++j)
//             {
//                 y[j] = B.at(k, k + 1 + j);
//             }

//             // 与之前类似，计算模长
//             double norm_y = vector_norm(y);

//             if (norm_y > 1e-14)
//             {
//                 double sigma = (y[0] >= 0.0 ? 1.0 : -1.0) * norm_y;

//                 // 构造 Householder 向量 v = y + sigma * e_1
//                 std::vector<double> v(y);
//                 v[0] += sigma;

//                 double vTv = 0.0;
//                 for (double vi : v)
//                     vTv += vi * vi;

//                 // TODO(SIMD编程)：此处的Householder变换可以通过 SIMD 指令加速，你可以尝试实现
//                 if (vTv > 1e-28)
//                 {
//                     const double beta = 2.0 / vTv;

//                     // 注意：这里是从右侧作用 V_k
//                     // B_new = B_old * V_k = B_old - beta * (B_old * v) * v^T
//                     std::vector<double> w(m - k, 0.0);
//                     for (int i = 0; i < m - k; ++i)
//                         for (int j = 0; j < n - k - 1; ++j)
//                             w[i] += B.at(k + i, k + 1 + j) * v[j];
//                     for (int i = 0; i < m - k; ++i)
//                         for (int j = 0; j < n - k - 1; ++j)
//                             B.at(k + i, k + 1 + j) -= beta * w[i] * v[j];

//                     // 累积 V：V_new = V_old * V_k
//                     // V[:, k+1:n] -= beta * (V[:, k+1:n] * v) * v^T
//                     std::vector<double> wV(n, 0.0);
//                     for (int i = 0; i < n; ++i)
//                         for (int j = 0; j < n - k - 1; ++j)
//                             wV[i] += V.at(i, k + 1 + j) * v[j];
//                     for (int i = 0; i < n; ++i)
//                         for (int j = 0; j < n - k - 1; ++j)
//                             V.at(i, k + 1 + j) -= beta * wV[i] * v[j];
//                 }
//             }

//             // 强制置零
//             for (int j = k + 2; j < n; ++j)
//             {
//                 B.at(k, j) = 0.0;
//             }
//         }
//     }

//     return B;
// }


// bidiagonalization.cpp
// 将 m×n 矩阵（本框架保证m ≥ n）通过 Householder 变换化为上双对角形
//
// 算法说明（你需要结合代码看）：
// 对上双对角化，需要交替从左侧和右侧应用 Householder 变换：
// 第 k 步（k = 0, 1, ..., n-1）：
//    - 从左侧作用 H_k，消去第 k 列中位置 (k+1,k), (k+2,k), ..., (m-1,k) 的元素
//    - 如果 k < n-2，从右侧作用 V_k，消去第 k 行中位置 (k,k+2), (k,k+3), ..., (k,n-1) 的元素
//
// 例如，对一个 4x4 矩阵 A，第一步 k=0：
//   - 从左侧作用 H_0，消去 A(1,0), A(2,0), A(3,0)，得到 B_0，同时更新 U = U * H_0
//   - 从右侧作用 V_0，消去 B_0(0,2)，B_0(0,3)，得到 B_1，同时更新 V = V * V_0
//
// 最终得到上双对角矩阵 B，只有主对角线和上次对角线有非零元素
//
// 本组件输出：A = U * B * V^T
// 其中 U（m×m）和 V（n×n）均为正交矩阵，B（m×n）为上双对角矩阵

#include "matrix.h"
#include <cmath>
#include <stdexcept>
#include <vector>

#if defined(__ARM_NEON) && defined(__aarch64__) && !defined(SVD_DISABLE_NEON)
#include <arm_neon.h>
#define SVD_USE_NEON 1
#else
#define SVD_USE_NEON 0
#endif

// 连续内存点积。ARMv8/AArch64 上使用 NEON 的 128-bit float64x2_t，
// 非 ARM 平台保留标量版本，方便本地调试。
static double dot_contiguous(const double *a, const double *b, int len)
{
    double sum = 0.0;
    int i = 0;
#if SVD_USE_NEON
    float64x2_t acc = vdupq_n_f64(0.0);
    for (; i + 1 < len; i += 2)
    {
        acc = vaddq_f64(acc, vmulq_f64(vld1q_f64(a + i), vld1q_f64(b + i)));
    }
    double lanes[2];
    vst1q_f64(lanes, acc);
    sum = lanes[0] + lanes[1];
#endif
    for (; i < len; ++i)
    {
        sum += a[i] * b[i];
    }
    return sum;
}

static void add_scaled_row(double *dst, const double *src, int len, double scale)
{
    int j = 0;
#if SVD_USE_NEON
    const float64x2_t s = vdupq_n_f64(scale);
    for (; j + 1 < len; j += 2)
    {
        float64x2_t d = vld1q_f64(dst + j);
        const float64x2_t x = vld1q_f64(src + j);
        d = vaddq_f64(d, vmulq_f64(s, x));
        vst1q_f64(dst + j, d);
    }
#endif
    for (; j < len; ++j)
    {
        dst[j] += scale * src[j];
    }
}

static void sub_scaled_row(double *dst, const double *src, int len, double scale)
{
    int j = 0;
#if SVD_USE_NEON
    const float64x2_t s = vdupq_n_f64(scale);
    for (; j + 1 < len; j += 2)
    {
        float64x2_t d = vld1q_f64(dst + j);
        const float64x2_t x = vld1q_f64(src + j);
        d = vsubq_f64(d, vmulq_f64(s, x));
        vst1q_f64(dst + j, d);
    }
#endif
    for (; j < len; ++j)
    {
        dst[j] -= scale * src[j];
    }
}

// 辅助函数，计算向量的范数（平方和开根）
static double vector_norm(const std::vector<double> &v)
{
    return std::sqrt(dot_contiguous(v.data(), v.data(), static_cast<int>(v.size())));
}

// 将 m×n 矩阵 A（m ≥ n）化为上双对角形，返回 B，同时输出 U（m×m）和 V（n×n）
Matrix to_bidiagonal(const Matrix &A, Matrix &U, Matrix &V)
{
    if (A.rows() < A.cols())
    {
        throw std::invalid_argument("to_bidiagonal: requires m >= n");
    }

    const int m = A.rows();
    const int n = A.cols();
    Matrix B = A;

    // U = I_m，V = I_n
    U = Matrix(m, m, 0.0);
    for (int i = 0; i < m; ++i)
        U.at(i, i) = 1.0;
    V = Matrix(n, n, 0.0);
    for (int i = 0; i < n; ++i)
        V.at(i, i) = 1.0;

    for (int k = 0; k < n; ++k)
    {
        // ================================================================
        // 步骤 1: 从左侧作用 Householder 变换，消去第 k 列中对角线以下的元素
        // ================================================================

        // 提取第 k 列从第 k 行往下的子向量
        // 例如：k=0 时提取 A(0:m-1, 0)，长度为 m-k+1 ; k=1 时提取 A(1:m-1, 1)
        std::vector<double> x(m - k);
        for (int i = 0; i < m - k; ++i)
        {
            x[i] = B.at(k + i, k);
        }

        double norm_x = vector_norm(x);

        if (norm_x > 1e-14 && k < m - 1)
        {
            // sign(x[0])：此处规定 x[0]==0 时取 +1
            double sigma = (x[0] >= 0.0 ? 1.0 : -1.0) * norm_x;

            // 实际上这里是+或者-都可以，手册里 Householder 一节是 -αe_1
            // 但我们这里 sigma 取了 sign(x[0]) * norm_x，所以是 +sigma * e_1 的形式
            std::vector<double> v(x);
            v[0] += sigma; // v = x + sigma * e_1

            // 计算 v^T v
            double vTv = dot_contiguous(v.data(), v.data(), static_cast<int>(v.size()));

            // SIMD实现：vTv、w/wU 点积与 rank-1 更新在 AArch64 上走 NEON。
            if (vTv > 1e-28)
            {
                const double beta = 2.0 / vTv;

                // 手册里的 Householder 矩阵定义为 H = I - beta * v * v^T，其中 beta = 2 / (v^T v)
                // 从左侧作用 H：B_new = H * B_old = B_old - beta * v * (v^T * B_old)
                std::vector<double> w(n - k, 0.0);
                for (int i = 0; i < m - k; ++i)
                    add_scaled_row(w.data(), B.row_ptr(k + i) + k, n - k, v[i]);
                for (int i = 0; i < m - k; ++i)
                    sub_scaled_row(B.row_ptr(k + i) + k, w.data(), n - k, beta * v[i]);

                // 累积 U：U_new = U_old * H_k
                // U[:, k:m] -= beta * (U[:, k:m] * v) * v^T
                std::vector<double> wU(m, 0.0);
                for (int i = 0; i < m; ++i)
                    wU[i] = dot_contiguous(U.row_ptr(i) + k, v.data(), m - k);
                for (int i = 0; i < m; ++i)
                    sub_scaled_row(U.row_ptr(i) + k, v.data(), m - k, beta * wU[i]);
            }
        }

        // 清除第 k 列中对角线以下的元素
        // 理论上应为 0，但不能完全保证全是 0，这里强制置零
        for (int i = k + 1; i < m; ++i)
        {
            B.at(i, k) = 0.0;
        }

        // ================================================================
        // 步骤 2: 从右侧作用 Householder 变换，消去第 k 行中 (k,k+2) 及右边的元素
        //        （只在 k < n-2 时需要）
        // ================================================================

        if (k < n - 2)
        {
            // 提取第 k 行从第 k+1 列往右的子向量（长度 n-k-1）
            std::vector<double> y(n - k - 1);
            for (int j = 0; j < n - k - 1; ++j)
            {
                y[j] = B.at(k, k + 1 + j);
            }

            // 与之前类似，计算模长
            double norm_y = vector_norm(y);

            if (norm_y > 1e-14)
            {
                double sigma = (y[0] >= 0.0 ? 1.0 : -1.0) * norm_y;

                // 构造 Householder 向量 v = y + sigma * e_1
                std::vector<double> v(y);
                v[0] += sigma;

                double vTv = dot_contiguous(v.data(), v.data(), static_cast<int>(v.size()));

                // SIMD实现：行连续点积与行向量更新在 AArch64 上走 NEON。
                if (vTv > 1e-28)
                {
                    const double beta = 2.0 / vTv;

                    // 注意：这里是从右侧作用 V_k
                    // B_new = B_old * V_k = B_old - beta * (B_old * v) * v^T
                    std::vector<double> w(m - k, 0.0);
                    for (int i = 0; i < m - k; ++i)
                        w[i] = dot_contiguous(B.row_ptr(k + i) + k + 1, v.data(), n - k - 1);
                    for (int i = 0; i < m - k; ++i)
                        sub_scaled_row(B.row_ptr(k + i) + k + 1, v.data(), n - k - 1, beta * w[i]);

                    // 累积 V：V_new = V_old * V_k
                    // V[:, k+1:n] -= beta * (V[:, k+1:n] * v) * v^T
                    std::vector<double> wV(n, 0.0);
                    for (int i = 0; i < n; ++i)
                        wV[i] = dot_contiguous(V.row_ptr(i) + k + 1, v.data(), n - k - 1);
                    for (int i = 0; i < n; ++i)
                        sub_scaled_row(V.row_ptr(i) + k + 1, v.data(), n - k - 1, beta * wV[i]);
                }
            }

            // 强制置零
            for (int j = k + 2; j < n; ++j)
            {
                B.at(k, j) = 0.0;
            }
        }
    }

    return B;
}
