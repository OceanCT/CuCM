#include <cstdint>
#include <cub/cub.cuh>
#include <fstream>
#include <iostream>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>

typedef uint8_t U8;

#define CHECK_CUDA(call)                                                \
  do {                                                                  \
    cudaError_t err = call;                                             \
    if (err != cudaSuccess) {                                           \
      fprintf(stderr, "CUDA error at %s:%d\n  call: %s\n  error: %s\n", \
              __FILE__, __LINE__, #call, cudaGetErrorString(err));      \
      exit(EXIT_FAILURE);                                               \
    }                                                                   \
  } while (0)
#define CHECK_KERNEL(kernel_call)                                             \
  do {                                                                        \
    kernel_call;                                                              \
    cudaError_t err = cudaGetLastError();                                     \
    if (err != cudaSuccess) {                                                 \
      fprintf(stderr,                                                         \
              "CUDA kernel launch error at %s:%d\n  call: %s\n  error: %s\n", \
              __FILE__, __LINE__, #kernel_call, cudaGetErrorString(err));     \
      exit(1);                                                                \
    }                                                                         \
  } while (0)
uint8_t state_table[256][2] = {
    {1, 2}, {3, 5}, {4, 6}, {7, 10}, {8, 12}, {9, 13}, {11, 14}, // 0
    {15, 19},
    {16, 23},
    {17, 24},
    {18, 25},
    {20, 27},
    {21, 28},
    {22, 29}, // 7
    {26, 30},
    {31, 33},
    {32, 35},
    {32, 35},
    {32, 35},
    {32, 35},
    {34, 37}, // 14
    {34, 37},
    {34, 37},
    {34, 37},
    {34, 37},
    {34, 37},
    {36, 39},
    {36, 39}, // 21
    {36, 39},
    {36, 39},
    {38, 40},
    {41, 43},
    {42, 45},
    {42, 45},
    {44, 47}, // 28
    {44, 47},
    {46, 49},
    {46, 49},
    {48, 51},
    {48, 51},
    {50, 52},
    {53, 43}, // 35
    {54, 57},
    {54, 57},
    {56, 59},
    {56, 59},
    {58, 61},
    {58, 61},
    {60, 63}, // 42
    {60, 63},
    {62, 65},
    {62, 65},
    {50, 66},
    {67, 55},
    {68, 57},
    {68, 57}, // 49
    {70, 73},
    {70, 73},
    {72, 75},
    {72, 75},
    {74, 77},
    {74, 77},
    {76, 79}, // 56
    {76, 79},
    {62, 81},
    {62, 81},
    {64, 82},
    {83, 69},
    {84, 71},
    {84, 71}, // 63
    {86, 73},
    {86, 73},
    {44, 59},
    {44, 59},
    {58, 61},
    {58, 61},
    {60, 49}, // 70
    {60, 49},
    {76, 89},
    {76, 89},
    {78, 91},
    {78, 91},
    {80, 92},
    {93, 69}, // 77
    {94, 87},
    {94, 87},
    {96, 45},
    {96, 45},
    {48, 99},
    {48, 99},
    {88, 101}, // 84
    {88, 101},
    {80, 102},
    {103, 69},
    {104, 87},
    {104, 87},
    {106, 57},
    {106, 57}, // 91
    {62, 109},
    {62, 109},
    {88, 111},
    {88, 111},
    {80, 112},
    {113, 85},
    {114, 87}, // 98
    {114, 87},
    {116, 57},
    {116, 57},
    {62, 119},
    {62, 119},
    {88, 121},
    {88, 121}, // 105
    {90, 122},
    {123, 85},
    {124, 97},
    {124, 97},
    {126, 57},
    {126, 57},
    {62, 129}, // 112
    {62, 129},
    {98, 131},
    {98, 131},
    {90, 132},
    {133, 85},
    {134, 97},
    {134, 97}, // 119
    {136, 57},
    {136, 57},
    {62, 139},
    {62, 139},
    {98, 141},
    {98, 141},
    {90, 142}, // 126
    {143, 95},
    {144, 97},
    {144, 97},
    {68, 57},
    {68, 57},
    {62, 81},
    {62, 81}, // 133
    {98, 147},
    {98, 147},
    {100, 148},
    {149, 95},
    {150, 107},
    {150, 107},
    {108, 151}, // 140
    {108, 151},
    {100, 152},
    {153, 95},
    {154, 107},
    {108, 155},
    {100, 156},
    {157, 95}, // 147
    {158, 107},
    {108, 159},
    {100, 160},
    {161, 105},
    {162, 107},
    {108, 163},
    {110, 164}, // 154
    {165, 105},
    {166, 117},
    {118, 167},
    {110, 168},
    {169, 105},
    {170, 117},
    {118, 171}, // 161
    {110, 172},
    {173, 105},
    {174, 117},
    {118, 175},
    {110, 176},
    {177, 105},
    {178, 117}, // 168
    {118, 179},
    {110, 180},
    {181, 115},
    {182, 117},
    {118, 183},
    {120, 184},
    {185, 115}, // 175
    {186, 127},
    {128, 187},
    {120, 188},
    {189, 115},
    {190, 127},
    {128, 191},
    {120, 192}, // 182
    {193, 115},
    {194, 127},
    {128, 195},
    {120, 196},
    {197, 115},
    {198, 127},
    {128, 199}, // 189
    {120, 200},
    {201, 115},
    {202, 127},
    {128, 203},
    {120, 204},
    {205, 115},
    {206, 127}, // 196
    {128, 207},
    {120, 208},
    {209, 125},
    {210, 127},
    {128, 211},
    {130, 212},
    {213, 125}, // 203
    {214, 137},
    {138, 215},
    {130, 216},
    {217, 125},
    {218, 137},
    {138, 219},
    {130, 220}, // 210
    {221, 125},
    {222, 137},
    {138, 223},
    {130, 224},
    {225, 125},
    {226, 137},
    {138, 227}, // 217
    {130, 228},
    {229, 125},
    {230, 137},
    {138, 231},
    {130, 232},
    {233, 125},
    {234, 137}, // 224
    {138, 235},
    {130, 236},
    {237, 125},
    {238, 137},
    {138, 239},
    {130, 240},
    {241, 125}, // 231
    {242, 137},
    {138, 243},
    {130, 244},
    {245, 135},
    {246, 137},
    {138, 247},
    {140, 248}, // 238
    {249, 135},
    {250, 69},
    {80, 251},
    {140, 252},
    {249, 135},
    {250, 69},
    {80, 251}, // 245
    {140, 252},
    {0, 0},
    {0, 0},
    {0, 0}}; // 252

__device__ __constant__ uint8_t d_State_table[256][2] = {
    {1, 2}, {3, 5}, {4, 6}, {7, 10}, {8, 12}, {9, 13}, {11, 14}, // 0
    {15, 19},
    {16, 23},
    {17, 24},
    {18, 25},
    {20, 27},
    {21, 28},
    {22, 29}, // 7
    {26, 30},
    {31, 33},
    {32, 35},
    {32, 35},
    {32, 35},
    {32, 35},
    {34, 37}, // 14
    {34, 37},
    {34, 37},
    {34, 37},
    {34, 37},
    {34, 37},
    {36, 39},
    {36, 39}, // 21
    {36, 39},
    {36, 39},
    {38, 40},
    {41, 43},
    {42, 45},
    {42, 45},
    {44, 47}, // 28
    {44, 47},
    {46, 49},
    {46, 49},
    {48, 51},
    {48, 51},
    {50, 52},
    {53, 43}, // 35
    {54, 57},
    {54, 57},
    {56, 59},
    {56, 59},
    {58, 61},
    {58, 61},
    {60, 63}, // 42
    {60, 63},
    {62, 65},
    {62, 65},
    {50, 66},
    {67, 55},
    {68, 57},
    {68, 57}, // 49
    {70, 73},
    {70, 73},
    {72, 75},
    {72, 75},
    {74, 77},
    {74, 77},
    {76, 79}, // 56
    {76, 79},
    {62, 81},
    {62, 81},
    {64, 82},
    {83, 69},
    {84, 71},
    {84, 71}, // 63
    {86, 73},
    {86, 73},
    {44, 59},
    {44, 59},
    {58, 61},
    {58, 61},
    {60, 49}, // 70
    {60, 49},
    {76, 89},
    {76, 89},
    {78, 91},
    {78, 91},
    {80, 92},
    {93, 69}, // 77
    {94, 87},
    {94, 87},
    {96, 45},
    {96, 45},
    {48, 99},
    {48, 99},
    {88, 101}, // 84
    {88, 101},
    {80, 102},
    {103, 69},
    {104, 87},
    {104, 87},
    {106, 57},
    {106, 57}, // 91
    {62, 109},
    {62, 109},
    {88, 111},
    {88, 111},
    {80, 112},
    {113, 85},
    {114, 87}, // 98
    {114, 87},
    {116, 57},
    {116, 57},
    {62, 119},
    {62, 119},
    {88, 121},
    {88, 121}, // 105
    {90, 122},
    {123, 85},
    {124, 97},
    {124, 97},
    {126, 57},
    {126, 57},
    {62, 129}, // 112
    {62, 129},
    {98, 131},
    {98, 131},
    {90, 132},
    {133, 85},
    {134, 97},
    {134, 97}, // 119
    {136, 57},
    {136, 57},
    {62, 139},
    {62, 139},
    {98, 141},
    {98, 141},
    {90, 142}, // 126
    {143, 95},
    {144, 97},
    {144, 97},
    {68, 57},
    {68, 57},
    {62, 81},
    {62, 81}, // 133
    {98, 147},
    {98, 147},
    {100, 148},
    {149, 95},
    {150, 107},
    {150, 107},
    {108, 151}, // 140
    {108, 151},
    {100, 152},
    {153, 95},
    {154, 107},
    {108, 155},
    {100, 156},
    {157, 95}, // 147
    {158, 107},
    {108, 159},
    {100, 160},
    {161, 105},
    {162, 107},
    {108, 163},
    {110, 164}, // 154
    {165, 105},
    {166, 117},
    {118, 167},
    {110, 168},
    {169, 105},
    {170, 117},
    {118, 171}, // 161
    {110, 172},
    {173, 105},
    {174, 117},
    {118, 175},
    {110, 176},
    {177, 105},
    {178, 117}, // 168
    {118, 179},
    {110, 180},
    {181, 115},
    {182, 117},
    {118, 183},
    {120, 184},
    {185, 115}, // 175
    {186, 127},
    {128, 187},
    {120, 188},
    {189, 115},
    {190, 127},
    {128, 191},
    {120, 192}, // 182
    {193, 115},
    {194, 127},
    {128, 195},
    {120, 196},
    {197, 115},
    {198, 127},
    {128, 199}, // 189
    {120, 200},
    {201, 115},
    {202, 127},
    {128, 203},
    {120, 204},
    {205, 115},
    {206, 127}, // 196
    {128, 207},
    {120, 208},
    {209, 125},
    {210, 127},
    {128, 211},
    {130, 212},
    {213, 125}, // 203
    {214, 137},
    {138, 215},
    {130, 216},
    {217, 125},
    {218, 137},
    {138, 219},
    {130, 220}, // 210
    {221, 125},
    {222, 137},
    {138, 223},
    {130, 224},
    {225, 125},
    {226, 137},
    {138, 227}, // 217
    {130, 228},
    {229, 125},
    {230, 137},
    {138, 231},
    {130, 232},
    {233, 125},
    {234, 137}, // 224
    {138, 235},
    {130, 236},
    {237, 125},
    {238, 137},
    {138, 239},
    {130, 240},
    {241, 125}, // 231
    {242, 137},
    {138, 243},
    {130, 244},
    {245, 135},
    {246, 137},
    {138, 247},
    {140, 248}, // 238
    {249, 135},
    {250, 69},
    {80, 251},
    {140, 252},
    {249, 135},
    {250, 69},
    {80, 251}, // 245
    {140, 252},
    {0, 0},
    {0, 0},
    {0, 0}}; // 252

template <typename T>
__host__ __device__ inline T d_clamp(T v, T lo, T hi) {
  return (v < lo) ? lo : (v > hi ? hi : v);
}

__device__ int d_stretch_t[4096];
__device__ int d_squash_t[4096];
__device__ int d_dt[1024];
int stretch_t[4096];
int squash_t[4096];
int dt[1024];

#define abk 8
const int ordernum = 8;
const int statemap_size = 1u << 25;
const int mixer_size = 80;

// #define printf(...) ((void)0)

struct StateMap {
  uint8_t states[statemap_size];
  uint32_t statep[256];
  __host__ StateMap() {
    for (int i = 0; i < statemap_size; i++)
      states[i] = 0;
    for (int i = 0; i < 256; i++)
      statep[i] = 1u << 31;
  }
  __host__ void initHost() {
    for (int i = 0; i < statemap_size; i++)
      states[i] = 0;
    for (int i = 0; i < 256; i++)
      statep[i] = 1u << 31;
  }
  __device__ void init() {
    for (int i = 0; i < statemap_size; i++)
      states[i] = 0;
    for (int i = 0; i < 256; i++)
      statep[i] = 1u << 31;
  }
  __host__ __device__ int predict(int hash) {
    return statep[states[hash]] >> 20;
  }
  __host__ void updateHost(int hash, uint64_t real) {
    int state = states[hash];
    int cnt = statep[state] & 1023;
    int error = (real << 22) - (statep[state] >> 10);
    int adjust_v = ((error >> 3) * dt[cnt]) & 0xfffffc00;
    statep[state] += adjust_v;
    if (cnt < 1023)
      statep[state]++;
    states[hash] = state_table[states[hash]][real];
  }
  __device__ void update(int hash, uint64_t real) {
    int state = states[hash];
    int cnt = statep[state] & 1023;
    int error = (real << 22) - (statep[state] >> 10);
    int adjust_v = ((error >> 3) * d_dt[cnt]) & 0xfffffc00;
    statep[state] += adjust_v;
    if (cnt < 1023)
      statep[state]++;
    states[hash] = d_State_table[states[hash]][real];
  }
  __device__ void updateWithState(int state, uint64_t real) {
    int cnt = statep[state] & 1023;
    int error = (real << 22) - (statep[state] >> 10);
    int adjust_v = ((error >> 3) * d_dt[cnt]) & 0xfffffc00;
    statep[state] += adjust_v;
    if (cnt < 1023)
      statep[state]++;
  }
  __host__ __device__ void printHashUse() {
    int sum = 0;
    for (int i = 0; i < statemap_size; i++) {
      if (states[i] != 0)
        sum++;
    }
    printf("used hash: %d, total hash: %d, ratio: %f\n", sum, statemap_size,
           sum / (float)statemap_size);
  }
};

struct Mixer {
  int wx[mixer_size * ordernum];
  __host__ __device__ Mixer() {
    for (int i = 0; i < mixer_size * ordernum; i++)
      wx[i] = 0;
  }
  __host__ int pHost(int cxt, int *smp) {
    int64_t sum = 0;
    for (int i = 0; i < ordernum; i++) {
      sum += (int64_t)wx[cxt * ordernum + i] * smp[i];
    }
    sum = (sum >> 16);
    int idx = sum < -2047 ? -2047 : (sum > 2048 ? 2048 : sum);
    return squash_t[idx + 2047];
  }
  __device__ int p(int cxt, int *smp) {
    int64_t sum = 0;
    for (int i = 0; i < ordernum; i++) {
      sum += (int64_t)wx[cxt * ordernum + i] * smp[i];
    }
    sum = (sum >> 16);
    int idx = sum < -2047 ? -2047 : (sum > 2048 ? 2048 : sum);
    return d_squash_t[idx + 2047];
  }
  __host__ void updateHost(int y, int pr, int cxt, int *smp,
                           int printlog = false) {
    // printlog = true;
    int err = ((y << 12) - pr) * 7;
    for (int i = 0; i < ordernum; i++) {
      wx[cxt * ordernum + i] += ((int64_t)err * smp[i]) >> 16;
    }
    if (printlog) {
      printf("y: %d, pr: %d, cxt: %d, err: %d, smp: %d, %d, %d, %d, %d, %d, "
             "wx: %d, %d, %d, %d, %d, %d\n",
             y, pr, cxt, err, smp[0], smp[1], smp[2], smp[3], smp[4], smp[5],
             wx[cxt * ordernum + 0], wx[cxt * ordernum + 1],
             wx[cxt * ordernum + 2], wx[cxt * ordernum + 3],
             wx[cxt * ordernum + 4], wx[cxt * ordernum + 5]);
    }
  }
  // noticeably if atomicAdd is used, the update stage can be split into
  // multiple
  __device__ void update(int y, int pr, int cxt, int *smp,
                         int printlog = false) {
    // printlog = true;
    int err = ((y << 12) - pr) * 7;
    for (int i = 0; i < ordernum; i++) {
      atomicAdd(&wx[cxt * ordernum + i], ((int64_t)err * smp[i]) >> 16);
    }
    if (printlog) {
      printf("y: %d, pr: %d, cxt: %d, err: %d, smp: %d, %d, %d, %d, %d, %d, "
             "wx: %d, %d, %d, %d, %d, %d\n",
             y, pr, cxt, err, smp[0], smp[1], smp[2], smp[3], smp[4], smp[5],
             wx[cxt * ordernum + 0], wx[cxt * ordernum + 1],
             wx[cxt * ordernum + 2], wx[cxt * ordernum + 3],
             wx[cxt * ordernum + 4], wx[cxt * ordernum + 5]);
    }
  }
};

void init() {
  int t[33] = {1, 2, 3, 6, 10, 16, 27, 45, 73,
               120, 194, 310, 488, 747, 1101, 1546, 2047, 2549,
               2994, 3348, 3607, 3785, 3901, 3975, 4022, 4050, 4068,
               4079, 4085, 4089, 4092, 4093, 4094};
  for (int k = 0; k <= 4094; k++) {
    int d = k - 2047;
    int w = d & 127;
    d = (d >> 7) + 16;
    squash_t[k] = (t[d] * (128 - w) + t[d + 1] * w + 64) >> 7;
  }
  squash_t[4095] = 4095;
  int pi = 0;
  for (int x = -2047; x <= 2047; x++) {
    int i = squash_t[x + 2047];
    for (int j = pi; j <= i; j++) {
      stretch_t[j] = x;
    }
    pi = i + 1;
  }
  stretch_t[4095] = 2047;
  for (int i = 0; i < 1024; i++) {
    dt[i] = 16384 / (i + i + 3);
  }
}

// block num should be chunknum * ordernum * (unitlength/32), with each order of
// each chunk running in different gpu sms blockid = chunkid * ordernum *
// (unitlength/32) + orderid * (unitlength/32) + unitinsidepartid each block
// should contain exact 1 warp (32 threads)
__global__ void cudapredictUnit(const uint8_t *inputs,
                                const uint64_t input_size, const int chunknum,
                                StateMap *sms, uint64_t *poss, uint16_t *pred,
                                uint64_t *smhashes, const uint64_t chunksize,
                                const int unitlength) {
  int chunkid = blockIdx.x / (ordernum * unitlength / 32);
  int orderid = (blockIdx.x / (unitlength / 32)) % ordernum;
  int unitinsidepartid = blockIdx.x % (unitlength / 32);
  int bitid = threadIdx.x;
  int realbitid = unitinsidepartid * 32 + bitid;
  if (chunkid >= chunknum)
    return;
  uint64_t pos = poss[chunkid] + realbitid + 8 * chunksize * chunkid;
  if (pos >= input_size * 8)
    return;
  // Use fixed-size array in registers for CUDA friendliness (max ordernum = 8)
  uint64_t h[9] = {0}; // h[0] = current, h[1..8] = history
  // fill previous bytes
  for (int i = 1; i <= orderid; i++) {
    if (poss[chunkid] + realbitid >= 8 * i) {
      h[i] = inputs[pos / 8 - i];
    }
  }
  // current byte
  h[0] = inputs[pos / 8] >> (8 - pos % 8);
  h[0] = (1 << (pos % 8)) | h[0];
  // compute hash using unrolled loop
  uint64_t hash = 2166136261;
  for (int i = 0; i <= orderid; i++) {
    hash ^= h[i];
    hash *= 16777619;
  }
  hash &= 0x1ffffff;
  StateMap &sm = sms[chunkid * ordernum + orderid];
  int p = sm.predict(hash);
  pred[chunkid * ordernum * unitlength + orderid * unitlength + realbitid] = p;
  smhashes[chunkid * ordernum * unitlength + orderid * unitlength + realbitid] = hash;
  // if(chunkid == 1)
  // printf("chunkid: %d, orderid: %d, realposs: %lu, h: %lu, %lu, %lu, %lu, %lu, %lu, %lu, %lu, hash: %lu, pr: %d, state: %d\n", chunkid, orderid, pos, h[0], h[1], h[2], h[3], h[4], h[5], h[6], h[7], hash, p, sm.states[hash]);
}

// each chunk should be divided into different blocks
// so each chunk should contain unitlength / 32 blocks(with each block
// containing 32 threads) blockid = chunkid * (unitlength/32) + unitinsidepartid
__global__ void cudamixUnit(const uint8_t *inputs, const uint64_t input_size,
                            Mixer *mixers, uint64_t *poss, uint16_t *pred,
                            uint16_t *prm, uint8_t *cxts,
                            const uint64_t chunksize, const int chunknum,
                            const int unitlength) {
  int chunkid = blockIdx.x / (unitlength / 32);
  int unitinsidepartid = blockIdx.x % (unitlength / 32);
  int bitid = threadIdx.x;
  int realbitid = unitinsidepartid * 32 + bitid;
  if (chunkid >= chunknum)
    return;
  uint64_t pos = poss[chunkid] + realbitid + 8 * chunksize * chunkid;
  if (pos >= input_size * 8 || poss[chunkid] + realbitid >= 8 * chunksize)
    return;
  Mixer &mx = mixers[chunkid];
  int smp[8];
#pragma unroll
  for (int i = 0; i < ordernum && i < 8; i++) {
    smp[i] = d_stretch_t[pred[chunkid * ordernum * unitlength + i * unitlength +
                              realbitid]];
  }
  int cxt = 0;
  if (poss[chunkid] + realbitid >= 8)
    cxt = inputs[pos / 8 - 1];
  int olen = 0;
#pragma unroll
  for (int i = 0; i < ordernum && i < 8; i++) {
    if (smp[i] != 1)
      olen++;
  }
  cxt = olen + (cxt >> 5) * 10;
  int pr = mx.p(cxt, smp);
  if (pr < 2048)
    pr++;
  int y = (inputs[pos / 8] >> (7 - pos % 8)) & 1;
  prm[chunkid * unitlength + realbitid] = pr << 1 | y;
  cxts[chunkid * unitlength + realbitid] = cxt;
}

__global__ void cudaupdateMixer(uint8_t *cxts, Mixer *mixers, uint64_t *poss,
                                uint16_t *pred, uint16_t *prm,
                                const uint64_t chunksize, const int chunknum,
                                const uint64_t input_size,
                                const int unitlength) {
  int chunkid = blockIdx.x / (unitlength / 32);
  int unitinsidepartid = blockIdx.x % (unitlength / 32);
  int bitid = threadIdx.x;
  int realbitid = unitinsidepartid * 32 + bitid;
  if (chunkid >= chunknum)
    return;
  uint64_t pos = poss[chunkid] + realbitid + 8 * chunksize * chunkid;
  if (pos >= input_size * 8)
    return;
  Mixer &mx = mixers[chunkid];
  int smp[8];
  int y = prm[chunkid * unitlength + realbitid] & 1;
#pragma unroll
  for (int i = 0; i < ordernum && i < 8; i++) {
    smp[i] = d_stretch_t[pred[chunkid * ordernum * unitlength + i * unitlength +
                              realbitid]];
  }
  int cxt = cxts[chunkid * unitlength + realbitid];
  int pr = prm[chunkid * unitlength + realbitid] >> 1;
  mx.update(y, pr, cxt, smp);
}

__global__ void cudaStateMapUpdate(uint64_t *smhashes, uint16_t *prm,
                                   StateMap *sms, uint64_t *poss,
                                   uint8_t *sequpdate, const int chunknum,
                                   const uint64_t input_size,
                                   const uint64_t chunksize, const int unitlength) {
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id >= chunknum * ordernum)
    return;
  int chunkid = id / ordernum;
  int orderid = id % ordernum;
  int base = chunkid * ordernum * unitlength + orderid * unitlength;
  int pos = poss[chunkid] + chunksize * chunkid;
  int length =
      unitlength < (input_size * 8 - pos) ? unitlength : (input_size * 8 - pos);
  for (int i = base; i < base + length; i++) {
    // int target_state = sms[chunkid * ordernum + orderid].states[smhashes[i]];
    sms[chunkid * ordernum + orderid].update(smhashes[i], prm[chunkid * unitlength + i - base] & 1);
  }
}

__global__ void cudaencodeUnit(uint16_t *prs, uint64_t *x1s, uint64_t *x2s,
                               const int chunknum, uint8_t *output,
                               uint64_t *offsets, uint64_t *poss,
                               const uint64_t chunksize, const int unitlength) {
  int id = threadIdx.x + blockDim.x * blockIdx.x;
  if (id >= chunknum)
    return;
  uint32_t x1 = x1s[id], x2 = x2s[id];
  uint64_t offset = offsets[id];
  for (int i = 0; i < unitlength; i++) {
    if (poss[id] + i >= chunksize * 8)
      break;
    int pr = prs[id * unitlength + i] >> 1;
    int y = prs[id * unitlength + i] & 1;
    uint32_t xmid =
        x1 + ((x2 - x1) >> 12) * pr + ((x2 - x1 & 0xfff) * pr >> 12);
    y ? (x2 = xmid) : (x1 = xmid + 1);
    while (((x1 ^ x2) & 0xff000000) == 0) {
      output[offset++] = x2 >> 24;
      x1 <<= 8;
      x2 = (x2 << 8) + 255;
    }
  }
  offsets[id] = offset;
  x1s[id] = x1;
  x2s[id] = x2;
  poss[id] += unitlength;
}

__global__ void cudaCalEncodeSize(uint64_t *offsets0, uint64_t *offsets1,
                                  const int chunknum,
                                  const uint64_t chunksize) {
  int id = threadIdx.x + blockDim.x * blockIdx.x;
  if (id >= chunknum) return;
  offsets1[id] = offsets0[id] - id * chunksize * abk + 1;
  // printf("Calculate encode size: Chunk[%d]: size: %lu\n", id,
        //  offsets1[id]);
}

__global__ void cudaencodefinal(uint64_t *x1s, const int chunknum,
                                const uint64_t chunksize, uint8_t *chunkoutput,
                                uint64_t *chunkOffsets, uint8_t *output,
                                uint64_t *offsets, int headlen,
                                const int input_size) {
  int id = threadIdx.x + blockDim.x * blockIdx.x;
  if (id >= chunknum) return;
  if (id == 0) {
    for (int i = 0; i < chunknum; i++) {
      uint64_t startchunkoffset = chunksize * abk * i;
      uint64_t compressed_length = chunkOffsets[i] - startchunkoffset;
      if (i != chunknum - 1)
        printf("chunk[%d]: compressed_length: %lu, ratio: %f\n", i,
               compressed_length, chunksize / (float)compressed_length);
      else
        printf("chunk[%d]: compressed_length: %lu, ratio: %f\n", i,
               compressed_length,
               (input_size - chunksize * (chunknum - 1)) /
                   (float)compressed_length);
    }
  }
  uint64_t chunkidsize = chunkOffsets[id] - abk * chunksize * id;
  chunkoutput[chunkOffsets[id]] = x1s[id] >> 24;
  offsets[id] += headlen + chunknum * 4 - 1;
  output[headlen + id * 4] = (offsets[id] >> 24) & 0xff;
  output[headlen + id * 4 + 1] = (offsets[id] >> 16) & 0xff;
  output[headlen + id * 4 + 2] = (offsets[id] >> 8) & 0xff;
  output[headlen + id * 4 + 3] = offsets[id] & 0xff;
  uint64_t oset = offsets[id];
  uint64_t pset = chunkOffsets[id];
  for (uint64_t i = 0; i <= chunkidsize; i++) {
    output[oset] = chunkoutput[pset];
    oset--, pset--;
  }
}

__global__ void cudaPostRunPrint(StateMap *sms, const int chunknum) {
  int id = threadIdx.x + blockDim.x * blockIdx.x;
  if (id >= chunknum)
    return;
  StateMap &sm = sms[id];
  sm.printHashUse();
}

void prelearn(StateMap sm[8], Mixer *mixer, uint8_t *input, uint64_t input_size,
              uint64_t learn_size, uint8_t *output, uint64_t *output_size) {
  if (learn_size > input_size)
    learn_size = input_size;
  printf("prelearning %lu bytes\n", learn_size);
  uint32_t x1 = 0, x2 = 0xffffffff;
  uint64_t offset = 0;
  for (int pos = 0; pos < learn_size * 8; pos++) {
    uint64_t h[9] = {0};
    for (int i = 1; i < ordernum; i++) {
      if (pos >= 8 * i)
        h[i] = input[pos / 8 - i];
    }
    h[0] = input[pos / 8] >> (8 - pos % 8);
    h[0] = (1 << (pos % 8)) | h[0];
    int smp[8];
    int smhash[8];
    for (int i = 0; i < ordernum; i++) {
      uint64_t hash = 2166136261;
      for (int j = 0; j <= i && j < 8; j++) {
        hash ^= h[j];
        hash *= 16777619;
      }
      hash &= 0x1ffffff;
      smp[i] = stretch_t[sm[i].predict(hash)];
      smhash[i] = hash;
    }
    // context
    int cxt = 0;
    if (pos >= 8) cxt = input[pos / 8 - 1];
    int olen = 0;
    for (int i = 0; i < ordernum && i < 8; i++) {
      if (smp[i] != 1)
        olen++;
    }
    cxt = olen + (cxt >> 5) * 10;
    int pr = mixer->pHost(cxt, smp);
    if (pr < 2048) pr++;
    int y = (input[pos / 8] >> (7 - pos % 8)) & 1;

    // update sm and mixer
    mixer->updateHost(y, pr, cxt, smp);
    for (int i = 0; i < ordernum; i++) {
      sm[i].updateHost(smhash[i], y);
    }
    // encode with pr
    uint64_t xmid = x1 + ((x2 - x1) >> 12) * pr + ((x2 - x1 & 0xfff) * pr >> 12);
    y ? (x2 = xmid) : (x1 = xmid + 1);
    while (((x1 ^ x2) & 0xff000000) == 0) {
      output[offset++] = x2 >> 24;
      x1 <<= 8;
      x2 = (x2 << 8) + 255;
    }
  }
  // write the last unequal byte
  output[offset++] = x1 >> 24;
  *output_size = offset;
}

double g_time_mem = 0.0;     // cudaMalloc/cudaMemcpy/cudaMemset
double g_time_predict = 0.0; // predict 阶段
double g_time_mix = 0.0;     // mix 阶段
double g_time_encode = 0.0;  // encode+update 阶段
double g_time_total = 0.0;   // compress_batch 总耗时
double g_time_finalize = 0.0; // final encode 阶段

void compress_batch(const uint8_t *input, uint8_t **output, const uint64_t input_size,
                    uint64_t *output_size, const uint64_t chunksize,
                    uint32_t unitlength, StateMap *sm_init = NULL, Mixer *mixer_init = NULL) {
  using namespace std::chrono;

  auto t_start_total = high_resolution_clock::now();

  std::cout << "compress_batch: " << input_size
            << " bytes, chunksize: " << chunksize
            << ", unitlength: " << (int)unitlength << std::endl;

  if (sm_init == NULL || mixer_init == NULL) {
    sm_init = (StateMap *)malloc(sizeof(StateMap) * ordernum);
    for (int i = 0; i < ordernum; i++) {
      sm_init[i].initHost();
    }
    mixer_init = new Mixer();
  }

  auto t_start_mem = high_resolution_clock::now();

  uint8_t *d_input, *d_output, *d_final_output, *d_cxts;
  uint16_t *d_pred, *d_prs;
  uint64_t compressed_size = 0, *smhashes;
  uint64_t *d_offsets, *d_x1s, *d_x2s, *d_offsets1, *d_offsets2;
  StateMap *d_Statemaps;
  uint8_t *d_serUpdateFlag;
  Mixer *d_mxs;

  int headlen = 0;
  uint64_t chunknum = (input_size + chunksize - 1) / chunksize;
  uint64_t blocknum = chunknum * ordernum * (unitlength / 32);

  // assign GPU memory
  cudaMalloc((void **)&d_input, sizeof(uint8_t) * input_size);
  cudaMemcpy(d_input, input, input_size, cudaMemcpyHostToDevice);
  cudaMalloc((void **)&d_output, chunksize * abk * chunknum);
  cudaMemset(d_output, 0, chunksize * abk * chunknum);
  cudaMalloc((void **)&d_final_output, chunksize * abk * chunknum);
  cudaMemset(d_final_output, 0, chunksize * abk * chunknum);
  cudaMalloc((void **)&d_offsets, sizeof(uint64_t) * chunknum);
  cudaMalloc((void **)&d_offsets1, sizeof(uint64_t) * chunknum);
  cudaMalloc((void **)&d_offsets2, sizeof(uint64_t) * chunknum);
  cudaMalloc((void **)&d_Statemaps, sizeof(StateMap) * chunknum * ordernum);
  cudaMalloc((void **)&smhashes, sizeof(uint64_t) * chunknum * ordernum * unitlength);
  cudaMalloc((void **)&d_pred, sizeof(uint16_t) * chunknum * ordernum * unitlength);
  cudaMemset(d_pred, 0, sizeof(uint16_t) * chunknum * ordernum * unitlength);
  cudaMalloc((void **)&d_mxs, sizeof(Mixer) * chunknum);
  cudaMemset(d_mxs, 0, sizeof(Mixer) * chunknum);
  cudaMalloc((void **)&d_prs, sizeof(uint16_t) * chunknum * unitlength);
  cudaMemset(d_prs, 0, sizeof(uint16_t) * chunknum * unitlength);
  cudaMalloc((void **)&d_x1s, sizeof(uint64_t) * chunknum);
  cudaMemset(d_x1s, 0, sizeof(uint64_t) * chunknum);
  cudaMalloc((void **)&d_x2s, sizeof(uint64_t) * chunknum);
  cudaMemset(d_x2s, 0xff, sizeof(uint64_t) * chunknum);
  uint64_t *d_pos;
  cudaMalloc((void **)&d_pos, sizeof(uint64_t) * chunknum);
  cudaMemset(d_pos, 0, sizeof(uint64_t) * chunknum);
  cudaMalloc((void **)&d_cxts, sizeof(uint8_t) * chunknum * unitlength);
  cudaMemset(d_cxts, 0, sizeof(uint8_t) * chunknum * unitlength);
  cudaMalloc((void **)&d_serUpdateFlag, sizeof(uint8_t) * chunknum * ordernum * unitlength);
  cudaMemset(d_serUpdateFlag, 0, sizeof(uint8_t) * chunknum * ordernum * unitlength);

  // copy prelearned model
  for (int chunki = 0; chunki < chunknum; chunki++) {
    cudaMemcpy(d_Statemaps + chunki * ordernum, sm_init, sizeof(StateMap) * ordernum, cudaMemcpyHostToDevice);
    cudaMemcpy(d_mxs + chunki, mixer_init, sizeof(Mixer), cudaMemcpyHostToDevice);
  }

  uint64_t *offsets = (uint64_t *)malloc(sizeof(uint64_t) * chunknum);
  for (int i = 0; i < chunknum; i++) offsets[i] = i * chunksize * abk;
  cudaMemcpy(d_offsets, offsets, sizeof(uint64_t) * chunknum, cudaMemcpyHostToDevice);

  cudaDeviceSynchronize();
  auto t_end_mem = high_resolution_clock::now();
  g_time_mem += duration<double, std::milli>(t_end_mem - t_start_mem).count();

  for (uint64_t i = 0; i < chunksize * 8 / unitlength + 1; i++) {
    // predict
    auto t_start_predict = high_resolution_clock::now();
    cudapredictUnit<<<blocknum, 32>>>(d_input, input_size, chunknum, d_Statemaps, d_pos, d_pred, smhashes, chunksize, unitlength);
    cudaDeviceSynchronize();
    auto t_end_predict = high_resolution_clock::now();
    g_time_predict += duration<double, std::milli>(t_end_predict - t_start_predict).count();

    // mix
    auto t_start_mix = high_resolution_clock::now();
    cudamixUnit<<<blocknum / ordernum, 32>>>(d_input, input_size, d_mxs, d_pos, d_pred, d_prs, d_cxts, chunksize, chunknum, unitlength);
    cudaDeviceSynchronize();
    auto t_end_mix = high_resolution_clock::now();
    g_time_mix += duration<double, std::milli>(t_end_mix - t_start_mix).count();

    // encode + update
    auto t_start_encode = high_resolution_clock::now();
    cudaupdateMixer<<<blocknum / ordernum, 32>>>(d_cxts, d_mxs, d_pos, d_pred, d_prs, chunksize, chunknum, input_size, unitlength);
    cudaStateMapUpdate<<<chunknum * ordernum / 32 + 1, 32>>>(smhashes, d_prs, d_Statemaps, d_pos, d_serUpdateFlag, chunknum, input_size, chunksize, unitlength);
    cudaencodeUnit<<<chunknum / 32 + 1, 32>>>(d_prs, d_x1s, d_x2s, chunknum, d_output, d_offsets, d_pos, chunksize, unitlength);
    cudaDeviceSynchronize();
    auto t_end_encode = high_resolution_clock::now();
    g_time_encode += duration<double, std::milli>(t_end_encode - t_start_encode).count();
  }
  auto t_start_finalize = high_resolution_clock::now();
  uint64_t *d_tmp_store = NULL;
  size_t tmp_store_bytes = 0;
  cudaCalEncodeSize<<<chunknum / 32 + 1, 32>>>(d_offsets, d_offsets1,
                                               chunknum, chunksize);
  cub::DeviceScan::InclusiveSum(nullptr, tmp_store_bytes, d_offsets1,
                                d_offsets2, chunknum);
  CHECK_CUDA(cudaMalloc(&d_tmp_store, tmp_store_bytes));
  cub::DeviceScan::InclusiveSum(d_tmp_store, tmp_store_bytes, d_offsets1,
                                d_offsets2, chunknum);
  cudaDeviceSynchronize();
  cudaencodefinal<<<chunknum / 32 + 1, 32>>>(
      d_x1s, chunknum, chunksize, d_output, d_offsets, d_final_output,
      d_offsets2, headlen, input_size);
  cudaDeviceSynchronize();
  auto t_end_finalize = high_resolution_clock::now();
  g_time_finalize += duration<double, std::milli>(t_end_finalize - t_start_finalize).count();
  t_start_mem = high_resolution_clock::now();
  cudaMemcpy(&compressed_size, d_offsets2 + chunknum - 1, sizeof(uint64_t),
             cudaMemcpyDeviceToHost);
  compressed_size++;
  *output_size = compressed_size;
  *output = (uint8_t *)malloc(sizeof(uint8_t) * (*output_size));
  cudaMemcpy(*output, d_final_output, *output_size,
             cudaMemcpyDeviceToHost);
  // printf("batch compressed size: %lu, ratio: %f\n", *output_size,
        //  input_size / (float)(*output_size));
  cudaDeviceSynchronize();
  t_end_mem = high_resolution_clock::now();
  g_time_mem += duration<double, std::milli>(t_end_mem - t_start_mem).count();
  // free GPU memory
  cudaFree(d_input);
  cudaFree(d_output);
  cudaFree(d_final_output);
  cudaFree(d_offsets);
  cudaFree(d_offsets1);
  cudaFree(d_offsets2);
  cudaFree(d_Statemaps);
  cudaFree(smhashes);
  cudaFree(d_pred);
  cudaFree(d_mxs);
  cudaFree(d_prs);
  cudaFree(d_x1s);
  cudaFree(d_x2s);
  cudaFree(d_pos);
  cudaFree(d_cxts);
  cudaFree(d_serUpdateFlag);
  cudaFree(d_tmp_store);
  auto t_end_total = high_resolution_clock::now();
  g_time_total += duration<double, std::milli>(t_end_total - t_start_total).count();

    // 输出结果
  std::cout << "Timing (ms): mem=" << g_time_mem
              << ", predict=" << g_time_predict
              << ", mix=" << g_time_mix
              << ", encode+update=" << g_time_encode
              << ", finalize=" << g_time_finalize
              << ", total=" << g_time_total << std::endl;
}

void compress_file(std::ifstream &infile, std::ofstream &outfile, const uint64_t i_input_size,
                   uint64_t *output_size,
                   const uint64_t chunksize, uint32_t unitlength,
                   uint64_t prelearn_size = 0, int batch_num = 100) {
  init();
  // read prelearn to memory
  uint8_t *input = (uint8_t *)malloc(sizeof(uint8_t) * prelearn_size);
  infile.read((char *)input, prelearn_size);
  // copy d_stretch_t and d_squash_t and dt to GPU
  cudaMemcpyToSymbol(d_stretch_t, stretch_t, sizeof(int) * 4096);
  cudaMemcpyToSymbol(d_squash_t, squash_t, sizeof(int) * 4096);
  cudaMemcpyToSymbol(d_dt, dt, sizeof(int) * 1024);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("After[[TableInit]], CUDA error: %s\n", cudaGetErrorString(err));
    exit(1);
  }
  int chunknum = (i_input_size - prelearn_size + chunksize - 1) / chunksize;
  printf("compressing chunksize: %lu, input_size: %lu, chunknum: %d, ordernum: %d, unitlength: %d\n",
         chunksize, i_input_size, chunknum, ordernum, unitlength);
  // prelearn
  uint8_t *prelearn_output =
      (uint8_t *)malloc(sizeof(uint8_t) * prelearn_size * abk);
  uint64_t prelearn_compressed_size = 0;
  StateMap *sm_init = (StateMap *)malloc(sizeof(StateMap) * ordernum);
  for (int i = 0; i < ordernum; i++) {
    sm_init[i].initHost();
  }
  Mixer mixer_init = Mixer();
  if (prelearn_size > 0) {
    prelearn(sm_init, &mixer_init, (uint8_t *)input, i_input_size,
             prelearn_size, prelearn_output, &prelearn_compressed_size);
    printf("prelearn compressed size: %lu, ratio: %f\n",
           prelearn_compressed_size,
           prelearn_size / (float)prelearn_compressed_size);
  }
  // output buffer
  uint64_t prelearn_header_size = 21 + (prelearn_compressed_size < prelearn_size ? prelearn_compressed_size : prelearn_size);
  uint8_t *output = (uint8_t *)malloc(sizeof(uint8_t) * prelearn_header_size);
  // original size, batch num, unitlength, chunksize, prelearn compressed size
  // prelearn original size, prelearn compressed size
  // prelearn compressed flag
  // prelearn data
  // compressed batch data 1
  // compressed batch data 2
  // ...
  output[0] = (i_input_size >> 24) & 0xff;
  output[1] = (i_input_size >> 16) & 0xff;
  output[2] = (i_input_size >> 8) & 0xff;
  output[3] = (i_input_size)&0xff;
  output[4] = (batch_num >> 8) & 0xff;
  output[5] = (batch_num)&0xff;
  output[6] = (unitlength >> 8) & 0xff;
  output[7] = (unitlength)&0xff;
  output[8] = (chunksize >> 24) & 0xff;
  output[9] = (chunksize >> 16) & 0xff;
  output[10] = (chunksize >> 8) & 0xff;
  output[11] = (chunksize)&0xff;
  output[12] = (prelearn_compressed_size >> 24) & 0xff;
  output[13] = (prelearn_compressed_size >> 16) & 0xff;
  output[14] = (prelearn_compressed_size >> 8) & 0xff;
  output[15] = (prelearn_compressed_size)&0xff;
  output[16] = (prelearn_size >> 24) & 0xff;
  output[17] = (prelearn_size >> 16) & 0xff;
  output[18] = (prelearn_size >> 8) & 0xff;
  output[19] = (prelearn_size)&0xff;
  if (prelearn_size <= prelearn_compressed_size) {
    output[20] = 0;
    memcpy(output + 21, input, prelearn_size);
  } else {
    output[20] = 1;
    memcpy(output + 21, prelearn_output, prelearn_compressed_size);
  }
  outfile.write((char *)output, prelearn_header_size);
  free(prelearn_output);
  free(output);
  // compress batch by batch
  uint64_t batch_size = chunksize * batch_num;
  int batchnum = (chunknum + batch_num - 1) / batch_num;
  input = (uint8_t *)realloc(input, sizeof(uint8_t) * batch_size);
  *output_size = prelearn_header_size;
  for (int i = 0; i < batchnum; i++) {
    uint64_t current_batch_size = (i == batchnum - 1) ? (i_input_size - batch_size * i - prelearn_size) : batch_size;
    // read input
    // printf("i: %d, i_input_size: %lu, batch_size: %lu, prelearn_size: %lu, batchnum: %d\n", i, i_input_size, batch_size, prelearn_size, batchnum);
    // printf("reading size: %lu\n", current_batch_size);
    infile.read((char *)input, current_batch_size);
    uint8_t *tmp_output = NULL;
    uint64_t tmp_output_size = 0;
    compress_batch(input, &tmp_output, current_batch_size, &tmp_output_size, chunksize, unitlength, sm_init, &mixer_init);
    // write to file, assuming tmp_output_size < 4GB
    uint32_t tmp_output_size_32 = (uint32_t)tmp_output_size;
    outfile.write((char *)&tmp_output_size_32, sizeof(uint32_t));
    outfile.write((char *)tmp_output, tmp_output_size);
    printf("batch %d/%d compressed size: %lu, compression ratio: %.2f\n", i + 1, batchnum, tmp_output_size, current_batch_size / (float)tmp_output_size);
    *output_size += tmp_output_size;
    free(tmp_output);
  }
  cudaDeviceReset();
}

uint64_t parse_size(const std::string &s) {
  size_t i = 0;
  while (i < s.size() && std::isspace(s[i]))
    i++;
  if (i == s.size())
    return 0;

  size_t j = i;
  while (j < s.size() && std::isdigit(s[j]))
    j++;
  if (j == i)
    return 0;

  uint64_t val = std::stoul(s.substr(i, j - i));

  while (j < s.size() && std::isspace(s[j]))
    j++;
  if (j < s.size()) {
    char unit = std::toupper(s[j]);
    switch (unit) {
    case 'K':
      val *= 1024;
      break;
    case 'M':
      val *= 1024 * 1024;
      break;
    case 'G':
      val *= 1024 * 1024 * 1024;
      break;
    default:
      break;
    }
  }
  return val;
}

void print_help(const char *prog) {
  std::cout << "Usage: " << prog << " [options]\n";
  std::cout << "Options:\n";
  std::cout << "  -c, --chunknum <N>    Number of chunks (default: file_size/chunksize + 1)\n";
  std::cout << "  -s, --chunksize <SZ>  Size of each chunk, supports K/M/G suffix (default: 1M)\n";
  std::cout << "  -u, --unit <SZ>       Unit size, supports K/M/G suffix (default: 128)\n";
  std::cout << "  -f, --file <path>     Input file to compress (default: ./enwik8)\n";
  std::cout << "  -o, --output <path>   Output compressed file (default: ./compressed.bin)\n";
  std::cout << "  -p, --prelearn <SZ>   Prelearn size, supports K/M/G suffix (default: 0 = disabled)\n";
  std::cout << "  -b, --batch <NUM>     Max parallel chunk number, depends on GPU memory size (default: 100)\n";
  std::cout << "  -h, --help            Show this help message\n";
}

int main(int argc, char *argv[]) {
  if (argc == 1) {
    print_help(argv[0]);
    return 0;
  }
  int unit_size = 128;
  int chunknum = -1;
  uint64_t chunk_size = 1024 * 1024;
  std::string filename = "./enwik8";
  std::string output_filename = "./compressed.bin"; // 默认输出文件
  uint64_t prelearn_size = 0;
  int batch = 100;

  for (int i = 1; i < argc; i++) {
    if ((strcmp(argv[i], "-c") == 0 || strcmp(argv[i], "--chunknum") == 0) &&
        i + 1 < argc) {
      chunknum = std::atoi(argv[++i]);
      if (chunknum <= 0) {
        std::cerr << "Invalid chunknum: " << chunknum << std::endl;
        return 1;
      }
    } else if ((strcmp(argv[i], "-s") == 0 ||
                strcmp(argv[i], "--chunksize") == 0) &&
               i + 1 < argc) {
      chunk_size = parse_size(argv[++i]);
      if (chunk_size == 0) {
        std::cerr << "Invalid chunk size: " << argv[i] << std::endl;
        return 1;
      }
    } else if ((strcmp(argv[i], "-f") == 0 || strcmp(argv[i], "--file") == 0) &&
               i + 1 < argc) {
      filename = argv[++i];
    } else if ((strcmp(argv[i], "-o") == 0 || strcmp(argv[i], "--output") == 0) &&
               i + 1 < argc) {
      output_filename = argv[++i];
    } else if ((strcmp(argv[i], "-p") == 0 ||
                strcmp(argv[i], "--prelearn") == 0) &&
               i + 1 < argc) {
      prelearn_size = parse_size(argv[++i]);
      if (prelearn_size == 0) {
        std::cerr << "Warning: prelearn disabled (size=0)" << std::endl;
      }
    } else if ((strcmp(argv[i], "-u") == 0 || strcmp(argv[i], "--unit") == 0) &&
               i + 1 < argc) {
      unit_size = std::atoi(argv[++i]);
      if (unit_size <= 0) {
        std::cerr << "Invalid unit size: " << unit_size << std::endl;
        return 1;
      }
    } else if ((strcmp(argv[i], "-b") == 0 ||
                strcmp(argv[i], "--batch") == 0) &&
               i + 1 < argc) {
      batch = std::atoi(argv[++i]);
      if (batch <= 0) {
        std::cerr << "Invalid batch number: " << batch << std::endl;
        return 1;
      }
    } else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
      print_help(argv[0]);
      return 0;
    } else {
      std::cerr << "Unknown option: " << argv[i] << std::endl;
      print_help(argv[0]);
      return 1;
    }
  }

  std::ifstream infile(filename, std::ios::in | std::ios::binary | std::ios::ate);
  if (!infile) {
    std::cerr << "Failed to open input file: " << filename << std::endl;
    return 1;
  }
  std::streamsize file_size = infile.tellg();
  infile.seekg(0, std::ios::beg);

  if (chunknum == -1) {
    chunknum = static_cast<int>((file_size + chunk_size - 1) / chunk_size);
  }

  std::cout << "Compressing file: " << filename
            << ", filesize: " << file_size
            << ", chunksize: " << chunk_size
            << ", chunknum: " << chunknum
            << ", unit: " << unit_size
            << ", prelearn: " << prelearn_size
            << ", batch: " << batch
            << ", output: " << output_filename
            << std::endl;

  std::ofstream outfile(output_filename, std::ios::out | std::ios::binary);
  if (!outfile) {
    std::cerr << "Failed to open output file: " << output_filename << std::endl;
    return 1;
  }

  uint64_t total_compressed_size = 0;

  auto start_time = std::chrono::high_resolution_clock::now();
  if (chunknum != -1) {
    file_size = chunk_size * chunknum < file_size ? chunk_size * chunknum : file_size;
  }
  compress_file(infile, outfile, file_size, &total_compressed_size,
                chunk_size, unit_size, prelearn_size, batch);

  infile.close();
  outfile.close();

  auto end_time = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed = end_time - start_time;

  std::cout << "Original size  : " << file_size << " bytes\n";
  std::cout << "Compressed size: " << total_compressed_size << " bytes\n";
  std::cout << "Compression ratio: "
            << (double)file_size / total_compressed_size << "\n";
  std::cout << "Total time: " << elapsed.count() << " sec" << std::endl;

  return 0;
}

/*
Final Output Format
[4 bytes: original file size]
[2 bytes: batchnum]
[2 bytes: unitlength]
[4 bytes: chunksize]
[4 bytes: prelearn compressed size]
[4 bytes: prelearn original size]
[1 byte: prelearn compressed flag]
[... prelearn data ...]
--batch 1--
[4 byte: batch compressed size]
[4 byte: compressed chunk 0 end offset]
[4 byte: compressed chunk 1 end offset]
[4 byte: compressed chunk 2 end offset]
...
[compressed data chunk 0]
[compressed data chunk 1]
[compressed data chunk 2]
...
--batch 2--
--batch 3--
--batch 4--
...
*/