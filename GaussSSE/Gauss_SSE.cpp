#include<iostream>
#include <time.h>
#include "xmmintrin.h"
#include "pmmintrin.h"

using namespace std;

const int maxN = 1026;

void swap(float& a, float& b) {
    float temp = a;
    a = b;
    b = temp;
}
/**
 * 串行消元，这里加上了"等于号"后面的值这一列，便于求值，所以a的行是n，列是n+1
 * @param n 矩阵规模
 * @param a 将要消的矩阵
 */
void LU(int n, float a[][maxN]) {
    //依据上一行的数值进行消元
    for (int i = 0; i < n - 1; ++i) {
        //遍历一下所有行，将前i个都置为0
        for (int j = i + 1; j < n; ++j) {
            //求出相差倍数
            float temp = a[j][i] / a[i][i];

            //遍历这一行的所有值，将i后面的数值依次减去相对应的值乘以倍数
            for (int k = i + 1; k <= n; ++k) {
                a[j][k] -= a[i][k] * temp;
            }
            //第i个为0
            a[j][i] = 0.00;
        }
    }
}
/**
 * 回代函数，求出x的值
 * @param n 未知数个数
 * @param a 方程矩阵
 * @param x 未知数的值数组
 */
void generation(int n, float a[][maxN], float x[]) {
    //从最后一个未知数开始求，依次向上求解
    for (int i = n - 1; i >= 0; --i) {
        // 未知数等于"等于号"后面的值除以系数
        x[i] = a[i][n] / a[i][i];
        for (int j = i - 1; j >= 0; --j) {
            // 求出x[i]后，依次代入上面的每一个方程，更新"等于号"后面的值
            a[j][n] -= x[i] * a[j][i];
        }
    }
}

/**
 * SSE算法消元设计
 * @param n 矩阵规模
 * @param a 方程矩阵，规模是（n，n+1）
 */
void SSE_LU(int n, float a[][maxN]) {
    float temp;
    __m128 div, t1, t2, sub;
    for (int i = 0; i < n - 1; ++i) {
        for (int j = i + 1; j < n; ++j) {
            // 用temp暂存相差的倍数
            temp = a[j][i] / a[i][i];
            // div全部用于存储temp，方便后面计算
            div = _mm_set1_ps(temp);

            //每四个一组进行计算，思想和串行类似
            int k = n - 3;
            for (; k >= i + 1; k -= 4) {
                t1 = _mm_loadu_ps(a[i] + k);
                t2 = _mm_loadu_ps(a[j] + k);
                sub = _mm_sub_ps(t2, _mm_mul_ps(t1, div));
                _mm_store_ss(a[j] + k, sub);
            }
            //处理剩余部分
            for (k += 3; k >= i + 1; --k) {
                a[j][k] -= a[i][k] * temp;
            }
            a[j][i] = 0.00;
        }
    }
}
/**
 * SSE实现回代过程向量化
 * @param n 未知数个数
 * @param a 方程矩阵
 * @param b 未知数的值数组
 */
void SSE_generation(int n, float a[][maxN], float b[]) {
    __m128 temp, t1, t2, sub;
    for (int i = n - 1; i >= 0; --i) {
        b[i] = a[i][n] / a[i][i];
        temp = _mm_set1_ps(b[i]);
        // 和串行算法思路类似，这里先将矩阵转置，方便计算
        for (int k = 0; k < i; ++k) {
            swap(a[k][n], a[n][k]);
            swap(a[k][i], a[i][k]);
        }
        //每四个一组进行计算
        int j = i - 4;
        for (; j >= 0; j -= 4) {
            t1 = _mm_loadu_ps(a[i] + j);
            t2 = _mm_loadu_ps(a[n] + j);
            sub = _mm_sub_ps(t2, _mm_mul_ps(t1, temp));
            _mm_store_ss(a[n] + j, sub);
        }
        //处理剩余部分
        for (j += 3; j >= 0; --j) {
            a[n][j] -= a[i][j] * b[i];
        }
        //转置回来
        for (int k = 0; k < i; ++k) {
            swap(a[k][n], a[n][k]);
            swap(a[k][i], a[i][k]);
        }
    }
}


int GapSize = 128;//设置间隙大小，每一个矩阵规模自增GapSize
int SizeCounts = 10;//设置区间个数，可以自由调整
int Counts = 50;//设置每次循环的次数
float a[maxN][maxN];
float b[maxN];
float tempA[maxN][maxN];//用于暂时存储a数组的值，控制变量唯一

//用于矩阵改变数值,为防止数据溢出,随机数的区间为100以内的浮点数
void change(int n, float a[][maxN]) {
    srand((unsigned)time(NULL));
    for (int i = 0; i < n; i++) {
        for (int j = 0; j <= n; j++) {
            a[i][j] = (float)(rand() % 10000) / 100.00;
        }
    }
}

//用于暂时存储a数组，控制变量
void store(int n, float a[][maxN], float b[][maxN]) {
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j <= n; ++j) {
            b[i][j] = a[i][j];
        }
    }
}

int main(int arg, char* argv[]) {
    //设置初始时间和结束时间
    struct timespec startTime, stopTime;
    for (int nowSize = GapSize, counts = 1; counts <= SizeCounts; nowSize += GapSize, counts++) {
        cout << "size: " << nowSize << endl;
        //设置每一个矩阵规模的总时间,每一个循环都加入到改变量中
        double eli_time = 0, solve_time = 0, sse_eli_time = 0, sse_solve_time = 0;
        //循环Counts次
        for (int i = 0; i < Counts; ++i) {
            change(nowSize, a);
            store(nowSize, a, tempA);//暂时将a数组存储在tempA中
            //计算串行消元时间
            struct timespec sts, ets;
            timespec_get(&sts, TIME_UTC);
            // to measure 
            LU(nowSize, a);
            timespec_get(&ets, TIME_UTC);
            time_t dsec = ets.tv_sec - sts.tv_sec;
            long dnsec = ets.tv_nsec - sts.tv_nsec;
            if (dnsec < 0) {
                dsec--;
                dnsec += 1000000000ll;
            }
            //printf(" % lld. % 09llds\n", dsec, dnsec);
            eli_time += (dsec) * 1000 + (double)(dnsec) * 0.000001;

            struct timespec sts1, ets1;
            timespec_get(&sts1, TIME_UTC);
            // to measure 
            generation(nowSize, a, b);
            timespec_get(&ets1, TIME_UTC);
            time_t dsec1 = ets1.tv_sec - sts1.tv_sec;
            long dnsec1 = ets1.tv_nsec - sts1.tv_nsec;
            if (dnsec1 < 0) {
                dsec1--;
                dnsec1 += 1000000000ll;
            }
            // printf(" % lld. % 09llds\n", dsec1, dnsec1);
            solve_time += (dsec1) * 1000 + (double)(dnsec1) * 0.000001;

            struct timespec sts2, ets2;
            timespec_get(&sts2, TIME_UTC);
            // to measure 
            SSE_LU(nowSize, tempA);
            timespec_get(&ets2, TIME_UTC);
            time_t dsec2 = ets2.tv_sec - sts2.tv_sec;
            long dnsec2 = ets2.tv_nsec - sts2.tv_nsec;
            if (dnsec2 < 0) {
                dsec2--;
                dnsec2 += 1000000000ll;
            }
            // printf(" % lld. % 09llds\n", dsec2, dnsec2);
            sse_eli_time += (dsec2) * 1000 + (double)(dnsec2) * 0.000001;


            struct timespec sts3, ets3;
            timespec_get(&sts3, TIME_UTC);
            // to measure 
            SSE_LU(nowSize, tempA);
            timespec_get(&ets3, TIME_UTC);
            time_t dsec3 = ets3.tv_sec - sts3.tv_sec;
            long dnsec3 = ets3.tv_nsec - sts3.tv_nsec;
            if (dnsec3 < 0) {
                dsec3--;
                dnsec3 += 1000000000ll;
            }
            // printf(" % lld. % 09llds\n", dsec3, dnsec3);
            sse_solve_time += (dsec3) * 1000 + (double)(dnsec3) * 0.000001;
        }

         cout << "串行消元时间：" << eli_time / Counts << "ms" << endl;
         cout << "串行回代时间：" << solve_time / Counts << "ms" << endl;
         cout << "并行消元时间：" << sse_eli_time / Counts << "ms" << endl;
         cout << "并行回代时间：" << sse_solve_time / Counts << "ms" << endl;
         cout << endl;
    }
}
