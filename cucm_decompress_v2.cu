#include <chrono>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int stretch_t[4096];
int squash_t[4096];
int dt[1024];
__device__ int d_stretch_t[4096];
__device__ int d_squash_t[4096];
__device__ int d_dt[1024];
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

#define abk 1.1


// #define printf(...) ((void)0)


const int ordernum = 8;
const int statemap_size = 1u << 25;
const int mixer_size = 80;

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
    if (cnt < 1023) statep[state]++;
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
  __host__ void initHost() {
    for (int i = 0; i < mixer_size * ordernum; i++)
      wx[i] = 0;
  }
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
  // copy to device
  cudaMemcpyToSymbol(d_stretch_t, stretch_t, sizeof(stretch_t));
  cudaMemcpyToSymbol(d_squash_t, squash_t, sizeof(squash_t));
  cudaMemcpyToSymbol(d_dt, dt, sizeof(dt));
}

// total thread num should be: chunknum * ordernum * (2^(maxassumed_length + 1) - 1)
// from the predid we can get the preassumed context
// 0 -> no preassumed context
// 1 2 -> assumed context length: 1
// 3 4 5 6 -> assumed context length: 2
// 7 8 9 10 11 12 13 14 -> assumed context length: 3
// ...
// so the preassumed context length = floor(log2(predid + 1))
// and the preassumed context should be predid - (2^preassumed_cxt_length - 1)
// so either
// id = chunkid * ordernum * (2^(maxassumed_length + 1) - 1) + orderid * (2^(maxassumed_length + 1) - 1) + predid; or
// id = chunkid * ordernum *(2^(maxassumed_length + 1) - 1) + predid * ordernum + orderid
// the latter is better for memory access of the mixer kernel
__global__ void predict(const uint8_t *d_output, StateMap *d_sms, uint64_t *poss, int *pred, int *smhashes, const uint64_t chunksize, const int chunknum, const uint16_t unitlength, int maxassumed_length, uint64_t output_size) {
  uint64_t id = blockIdx.x * blockDim.x + threadIdx.x;
  uint64_t predtotalnum = ((uint64_t)1 << (maxassumed_length + 1)) - 1;
  int chunkid = id / (ordernum * predtotalnum);
  int orderid = (id / predtotalnum) % ordernum;
  uint64_t predid = id % predtotalnum;
  if (chunkid >= chunknum) return;
  // from the predid we can get the preassumed context
  // if predid == 0, means no preassumed context and we should use the real context
  int preassumed_length = 31 - __clz(predid + 1);
  int preassumed_cxt = predid - ((1 << preassumed_length) - 1);
  int pos_in_chunk = poss[chunkid] % unitlength;
  uint64_t realposs = poss[chunkid] + chunkid * chunksize * 8 + preassumed_length;
  if (pos_in_chunk + preassumed_length >= unitlength || poss[chunkid] + preassumed_length >= chunksize * 8 || realposs >= output_size * 8) {
    return;
  }
  uint64_t h[ordernum + 1] = {0};
  for (int i = 1; i <= orderid; i++) {
    if (realposs >= 8 * i) {
      h[i] = d_output[realposs / 8 - i];
    }
  }
  // printf h array
  // printf("|PP|chunkid: %d, orderid: %d, predid: %lu, preassumed_length: %d, preassumed_cxt: %d, realposs: %lu, h before: %lu, %lu, %lu, %lu, %lu, %lu, %lu, %lu\n", 
    // chunkid, orderid, predid, preassumed_length, preassumed_cxt, realposs,
    // h[0], h[1], h[2], h[3], h[4], h[5], h[6], h[7]
  // );
  // add the preassumed context to h array accordingly
  int current_byte_length = realposs % 8;
  if (current_byte_length == 0) {
    h[0] = 0;
  } else if (preassumed_length >= current_byte_length) {
    // then h[0] should just be withdraw from preassumed_cxt
    h[0] = preassumed_cxt & ((1 << current_byte_length) - 1);
    preassumed_length -= current_byte_length;
    preassumed_cxt >>= current_byte_length;
  } else {
    // so h[0] should be read from the output and the rest from preassumed_cxt
    h[0] = d_output[realposs / 8] >> (8 - current_byte_length);
    h[0] += preassumed_cxt;
    preassumed_length = 0;
  }
  h[0] = (1 << current_byte_length) | h[0];
  // else {
  //   h[0] = d_output[realposs / 8] >> (8 - current_byte_length);
  //   h[0] += preassumed_cxt & ((1 << current_byte_length) - 1);
  //   preassumed_length -= current_byte_length;
  //   h[0] = (1 << current_byte_length) | h[0];
  // }
  // change the rest of h
  int target = 1;
  while (preassumed_length > 0) {
    h[target++] += preassumed_cxt & 0xff;
    preassumed_cxt >>= 8;
    preassumed_length -= 8;
  }
  uint64_t hash = 2166136261;
  for (int i = 0; i <= orderid; i++) {
    hash ^= h[i];
    hash *= 16777619;
  }
  hash &= 0x1ffffff;
  StateMap &sm = d_sms[chunkid * ordernum + orderid];
  pred[chunkid * ordernum * predtotalnum + predid * ordernum + orderid] = sm.predict(hash);
  smhashes[chunkid * ordernum * predtotalnum + predid * ordernum + orderid] = hash;
  // if(chunkid == 1)
  // printf("chunkid: %d, orderid: %d, byte: %x, realposs: %lu,  h: %lu, %lu, %lu, %lu, %lu, %lu, %lu, %lu, hash: %lu, pr: %d, state: %d\n", 
  //   chunkid, orderid, d_output[realposs / 8 - 1],realposs, 
  //   h[0], h[1], h[2], h[3], h[4], h[5], h[6], h[7], hash, 
  //   pred[chunkid * ordernum * predtotalnum + predid * ordernum + orderid], 
  //   sm.states[hash]
  // );
}

// mix should be almost the same as pred, but total thread num should be chunknum * (2^(maxassumed_length + 1) - 1)
// and the id = chunkid * predtotalnum + predid
__global__ void mix(const uint8_t *d_output, const uint64_t output_size, Mixer *d_mxs, uint64_t *poss, int *pred, int *prs, int *cxt, const int chunknum, const int maxassumed_length, const uint64_t chunksize, const uint16_t unitlength) {
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  uint64_t predtotalnum = ((uint64_t)1 << (maxassumed_length + 1)) - 1;
  int chunkid = id / predtotalnum;
  if (chunkid >= chunknum) return;
  uint32_t predid = id % predtotalnum;
  int smp[ordernum];  
  int preassumed_length = 31 - __clz(predid + 1);
  int preassumed_cxt = predid - ((1 << preassumed_length) - 1);
  uint64_t realbitid = poss[chunkid] + preassumed_length;
  uint64_t realposs = chunkid * chunksize * 8 + realbitid;
  // if the realposs exceed the chunk boundary or the unit boundary, just skip and return
  if (poss[chunkid] % unitlength + preassumed_length >= unitlength || realbitid >= chunksize * 8 || realposs >= output_size * 8) {
    return;
  }
  for (int i = 0; i < ordernum; i++) {
    smp[i] = d_stretch_t[pred[chunkid * ordernum * predtotalnum + predid * ordernum + i]];
  }
  cxt[id] = 0;
  if (realbitid >= 8) {
    int rightshift = realposs % 8;
    preassumed_cxt >>= rightshift;
    preassumed_length -= rightshift;
    cxt[id] = ((preassumed_cxt & 0xff) + d_output[realposs / 8 - 1]) >> 5;
  }
  cxt[id] = cxt[id] * 10;
  for (int i = 0; i < ordernum; i++) {
    if (smp[i] != 1) cxt[id]++;
  }
  int pr = d_mxs[chunkid].p(cxt[id], smp);
  if (pr < 2048) pr++;
  prs[chunkid * predtotalnum + predid] = pr;
  // if(chunkid == 1)
  // printf("|Mix|chunkid: %d, realposs: %lu, cxt: %d, pr: %d, smp: %d, %d, %d, %d, %d, %d, %d, %d, weight: %d, %d, %d, %d, %d, %d, %d, %d\n", chunkid, realposs, cxt[id], pr, smp[0], smp[1], smp[2], smp[3], smp[4], smp[5], smp[6], smp[7], d_mxs[chunkid].wx[cxt[id] * ordernum + 0], d_mxs[chunkid].wx[cxt[id] * ordernum + 1], d_mxs[chunkid].wx[cxt[id] * ordernum + 2], d_mxs[chunkid].wx[cxt[id] * ordernum + 3], d_mxs[chunkid].wx[cxt[id] * ordernum + 4], d_mxs[chunkid].wx[cxt[id] * ordernum + 5], d_mxs[chunkid].wx[cxt[id] * ordernum + 6], d_mxs[chunkid].wx[cxt[id] * ordernum + 7]);
  
}

// total thread num should just be chunknum
// decode the next assumed length bits according to the prs
__global__ void decode(const uint8_t *d_input, uint64_t input_size, const int *prs, uint32_t *x1s, uint32_t *x2s, uint32_t *xs, const int chunknum, uint8_t *d_output, uint64_t *input_offsets, uint64_t *i_offsets_bondary, uint64_t *poss, const uint64_t chunksize, const uint64_t lastchunksize, const uint16_t unitlength, const int maxassumed_length, int *realpred, int *realprs, int *realsmhashes, int *realcxt, int *pred, int *smhashes, int *cxt) {
  int chunkid = blockIdx.x * blockDim.x + threadIdx.x;
  if (chunkid >= chunknum) return;
  uint32_t x1 = x1s[chunkid], x2 = x2s[chunkid], x = xs[chunkid];
  uint64_t ioffset = input_offsets[chunkid];
  int preassumed_cxt = 0;
  uint64_t predtotalnum = ((uint64_t)1 << (maxassumed_length + 1)) - 1;
  for (int i = 0; i <= maxassumed_length; i++) {
    int unitid = (poss[chunkid] + i) % unitlength;
    // break if its already out of the chunk or unit boundary
    // printf("|TryDecode| poss[chunkid]: %lu, i: %d, chunksize * 8: %lu, unitlength: %d\n", poss[chunkid], i, chunksize * 8, unitlength);
    if (poss[chunkid] + i >= chunksize * 8 || poss[chunkid] % unitlength + i >= unitlength) break;
    if (chunkid == chunknum - 1 && poss[chunkid] + i >= lastchunksize * 8) break;
    // get the pr
    int predid = 0;
    if (i) predid = (1 << i) - 1 + preassumed_cxt;
    int p = prs[chunkid * predtotalnum + predid];
    // printf("|DecodeBefore| x1: %x, x2: %x, x: %x, pr: %d", x1, x2, x, p);
    // decode
    uint32_t xmid = x1 + ((x2 - x1) >> 12) * p + ((x2 - x1 & 0xfff) * p >> 12);
    int y = x <= xmid;
    y ? (x2 = xmid) : (x1 = xmid + 1);
    while (((x1 ^ x2) & 0xff000000) == 0) {
      x1 <<= 8;
      x2 = (x2 << 8) + 255;
      if (ioffset < input_size && ioffset < i_offsets_bondary[chunkid]) {
        // x = (x << 8) + (input[ioffset++] & 255);
        x = (x << 8) + (d_input[ioffset++] & 255);
      } else {
        x = (x << 8) + 255;
      }
    }
    // if(chunkid == 1)
    // printf("|Decode|pos: %lu, y: %d, pr: %d, x: %x, x1: %x, x2: %x, xmid: %x, offset: %lu\n", poss[chunkid] + i, y, p, x, x1, x2, xmid, ioffset);
    // if(chunkid == 1)
    // printf("|DecodeMove|pos: %lu, unitid: %d, Update sm: %d, %d, %d, %d, %d, %d, %d, %d, smhash: %d, %d, %d, %d, %d, %d, %d, %d and Update mx: cxt: %d, pr: %d, y: %d\n", poss[chunkid] + i, unitid, pred[chunkid * ordernum * predtotalnum + predid * ordernum + 0], pred[chunkid * ordernum * predtotalnum + predid * ordernum + 1], pred[chunkid * ordernum * predtotalnum + predid * ordernum + 2], pred[chunkid * ordernum * predtotalnum + predid * ordernum + 3], pred[chunkid * ordernum * predtotalnum + predid * ordernum + 4], pred[chunkid * ordernum * predtotalnum + predid * ordernum + 5], pred[chunkid * ordernum * predtotalnum + predid * ordernum + 6], pred[chunkid * ordernum * predtotalnum + predid * ordernum + 7], smhashes[chunkid * ordernum * predtotalnum + predid * ordernum + 0], smhashes[chunkid * ordernum * predtotalnum + predid * ordernum + 1], smhashes[chunkid * ordernum * predtotalnum + predid * ordernum + 2], smhashes[chunkid * ordernum * predtotalnum + predid * ordernum + 3], smhashes[chunkid * ordernum * predtotalnum + predid * ordernum + 4], smhashes[chunkid * ordernum * predtotalnum + predid * ordernum + 5], smhashes[chunkid * ordernum * predtotalnum + predid * ordernum + 6], smhashes[chunkid * ordernum * predtotalnum + predid * ordernum + 7], cxt[chunkid * predtotalnum + predid], p, y); 

    // get ready for the update stage
    for (int orderid = 0; orderid < ordernum; orderid++) {
      realpred[chunkid * ordernum * unitlength + orderid * unitlength + unitid] = pred[chunkid * ordernum * predtotalnum + predid * ordernum + orderid];
      realsmhashes[chunkid * ordernum * unitlength + orderid * unitlength + unitid] = smhashes[chunkid * ordernum * predtotalnum + predid * ordernum + orderid];
    }
    realprs[chunkid * unitlength + unitid] = prs[chunkid * predtotalnum + predid] << 1 | y;
    realcxt[chunkid * unitlength + unitid] = cxt[chunkid * predtotalnum + predid];
    // update preassumed_cxt
    preassumed_cxt = (preassumed_cxt << 1) | y;
    // write the newly y to the output
    d_output[chunkid * chunksize + (poss[chunkid] + i) / 8] |= (y << (7 - (poss[chunkid] + i) % 8));
    
    // printf("|DecodeAfter| chunkid: %d, poss: %lu, i: %d, d_output[%lu]: %x\n", chunkid, poss[chunkid], i, chunkid * chunksize + (poss[chunkid] + i) / 8, d_output[chunkid * chunksize + (poss[chunkid] + i) / 8]);
  }
  x1s[chunkid] = x1;
  x2s[chunkid] = x2;
  xs[chunkid] = x;
  input_offsets[chunkid] = ioffset;
}

__global__ void updateStateMap(StateMap *d_sms, int *smhashes, int *prs, const int input_size, const int chunknum, const uint16_t unitlength, const uint64_t chunksize) {
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  int chunkid = id / ordernum;
  int orderid = id % ordernum;
  if (chunkid >= chunknum) return;
  StateMap &sm = d_sms[chunkid * ordernum + orderid];
  for (int i = 0; i < unitlength; i++) {
    int target_state = sm.states[smhashes[chunkid * ordernum * unitlength + orderid * unitlength + i]];
    uint32_t originalstatep = sm.statep[target_state];
    sm.update(smhashes[chunkid * ordernum * unitlength + orderid * unitlength + i], prs[chunkid * unitlength + i] & 1);
    // if(chunkid == 1)
    // printf("|SMUpdate|chunkid: %d, orderid: %d, i: %d, hash: %d, y: %d, updatestate: %d, originalstatep: %u, updatedstatep: %d\n", chunkid, orderid, i, smhashes[orderid * unitlength + i], prs[i] & 1, target_state, originalstatep, sm.statep[target_state]);
  }
}

__global__ void updateMixer(Mixer *d_mxs, int *prs, int *pred, int *cxt, int chunknum, uint16_t unitlength) {
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id >= chunknum) return;
  int smp[ordernum];
  for(int unitid = 0; unitid < unitlength; unitid++) {
    for(int i = 0; i < ordernum; i++) {
      smp[i] = d_stretch_t[pred[id * ordernum * unitlength + i * unitlength + unitid]];
    }
    d_mxs[id].update(prs[id * unitlength + unitid] & 1, prs[id * unitlength + unitid] >> 1, cxt[id * unitlength + unitid], smp);
    // printf("|MXUpdate|chunkid: %d, unitid: %d, y: %d, pr: %d, cxt: %d, smp: %d, %d, %d, %d, %d, %d, %d, %d, weight: %d, %d, %d, %d, %d, %d, %d, %d\n", id, unitid, prs[id * unitlength + unitid] & 1, prs[id * unitlength + unitid] >> 1, cxt[id * unitlength + unitid], smp[0], smp[1], smp[2], smp[3], smp[4], smp[5], smp[6], smp[7], d_mxs[id].wx[cxt[id * unitlength + unitid] * ordernum + 0], d_mxs[id].wx[cxt[id * unitlength + unitid] * ordernum + 1], d_mxs[id].wx[cxt[id * unitlength + unitid] * ordernum + 2], d_mxs[id].wx[cxt[id * unitlength + unitid] * ordernum + 3], d_mxs[id].wx[cxt[id * unitlength + unitid] * ordernum + 4], d_mxs[id].wx[cxt[id * unitlength + unitid] * ordernum + 5], d_mxs[id].wx[cxt[id * unitlength + unitid] * ordernum + 6], d_mxs[id].wx[cxt[id * unitlength + unitid] * ordernum + 7]);
  }
}

void decompressbatch(const uint8_t *input, const uint64_t input_size,
                     uint8_t *output, uint64_t output_size,
                     uint64_t *chunkstarts,
                     const uint64_t chunksize, const uint16_t unitlength,
                     StateMap *sm_init = nullptr,
                     Mixer *mixer_init = nullptr, const int maxassumed_length = 3) {
  printf("decompress input size: %lu, output size: %lu, chunksize: %lu, "
         "unitlength: %d, maxassumed_length: %d\n",
         input_size, output_size, chunksize, unitlength, maxassumed_length);
  // // print the whole batch bytes:
  // for(uint64_t i = 0; i < input_size; i++) {
  //   printf("%02x ", input[i]);
  // }
  uint64_t chunknum = (output_size + chunksize - 1) / chunksize;
  // // print chunk starts
  // for (int i = 0; i <= chunknum; i++) {
  //   printf("chunk %d starts at %lu\n", i, chunkstarts[i]);
  // }
  if (!sm_init) {
    sm_init = new StateMap[ordernum];
    mixer_init = new Mixer[1];
  }

  // allocate device memory
  uint8_t *d_input, *d_output;
  StateMap *d_sms;
  Mixer *d_mxs;
  int *d_pred, *d_smhashes, *d_cxt;
  int *d_prs;
  uint64_t *d_poss;
  uint32_t *d_x1s, *d_x2s, *d_xs;
  uint64_t *d_input_offsets;
  uint64_t *d_offsets_bondary;
  cudaError_t err;
  // for update
  int *d_realpred, *d_realsmhashes, *d_realcxt, *d_realprs;

  cudaMalloc(&d_input, input_size);
  cudaMalloc(&d_output, output_size);
  cudaMalloc(&d_sms, sizeof(StateMap) * ordernum * chunknum);
  cudaMalloc(&d_mxs, sizeof(Mixer) * chunknum);
  cudaMalloc(&d_pred, sizeof(int) * ordernum * chunknum * ((1 << (maxassumed_length + 1)) - 1));
  cudaMalloc(&d_smhashes, sizeof(int) * ordernum * chunknum * ((1 << (maxassumed_length + 1)) - 1));
  cudaMalloc(&d_cxt, sizeof(int) * chunknum * ((1 << (maxassumed_length + 1)) - 1));
  cudaMalloc(&d_prs, sizeof(int) * chunknum * ((1 << (maxassumed_length + 1)) - 1));
  cudaMalloc(&d_poss, sizeof(uint64_t) * chunknum);
  cudaMalloc(&d_x1s, sizeof(uint32_t) * chunknum);
  cudaMalloc(&d_x2s, sizeof(uint32_t) * chunknum);
  cudaMalloc(&d_xs, sizeof(uint32_t) * chunknum);
  cudaMalloc(&d_offsets_bondary, sizeof(uint64_t) * chunknum);
  cudaMalloc(&d_input_offsets, sizeof(uint64_t) * chunknum);
  cudaMalloc(&d_realpred, sizeof(int) * ordernum * chunknum * unitlength);
  cudaMalloc(&d_realsmhashes, sizeof(int) * ordernum * chunknum * unitlength);
  cudaMalloc(&d_realcxt, sizeof(int) * chunknum * unitlength);
  cudaMalloc(&d_realprs, sizeof(int) * chunknum * unitlength);
  // copy data to device
  cudaMemcpy(d_input, input, input_size, cudaMemcpyHostToDevice);
  cudaMemset(d_output, 0, output_size);
  for (int chunki = 0; chunki < chunknum; chunki++) {
    cudaMemcpy(d_sms + chunki * ordernum, sm_init,
               sizeof(StateMap) * ordernum, cudaMemcpyHostToDevice);
    cudaMemcpy(d_mxs + chunki, mixer_init, sizeof(Mixer),
               cudaMemcpyHostToDevice);
  }

  cudaMemset(d_poss, 0, sizeof(uint64_t) * chunknum);
  cudaMemset(d_x1s, 0, sizeof(uint32_t) * chunknum);
  cudaMemset(d_x2s, 0xff, sizeof(uint32_t) * chunknum);
  cudaMemcpy(d_offsets_bondary, chunkstarts + 1, sizeof(uint64_t) * chunknum, cudaMemcpyHostToDevice);
  for (int chunki = 0; chunki < chunknum; chunki++) {
    uint32_t x = 0;
    for (int i = 0; i < 4; ++i) x = (x << 8) + (input[chunkstarts[chunki]++] & 255);
    // printf("initial x for chunk %d: %x, %u\n", chunki, x, x);
    cudaMemcpy(d_xs + chunki, &x, sizeof(uint32_t), cudaMemcpyHostToDevice);
  }
  // input offsets should be chunkstarts and output offsets should be according to chunksize
  cudaMemcpy(d_input_offsets, chunkstarts, sizeof(uint64_t) * chunknum, cudaMemcpyHostToDevice);


  for (int i = 0; i < chunksize * 8 / unitlength + 1; i++) {
    // for each unit in the chunk
    // always pred, mix and decode one by one
    // should last (unit + maxassumed_length - 1) / maxassumed_length times
    int total_pred_num = chunknum * ordernum * ((1 << (maxassumed_length + 1)) - 1);
    for (int j = 0; j < (unitlength + maxassumed_length + 1 - 1) / (maxassumed_length + 1); j++) {
      predict<<<(total_pred_num + 31) / 32, 32>>>(d_output, d_sms, d_poss, d_pred, d_smhashes, chunksize, chunknum, unitlength, maxassumed_length, output_size);
      cudaDeviceSynchronize();
      err = cudaGetLastError();
      if (err != cudaSuccess) {
        printf("CUDA Error in predict: %s\n", cudaGetErrorString(err));
        exit(-1);
      }
      // mix
      int total_mix_num = chunknum * ((1 << (maxassumed_length + 1)) - 1);
      mix<<<(total_mix_num + 31) / 32, 32>>>(d_output, output_size, d_mxs, d_poss, d_pred, d_prs, d_cxt, chunknum, maxassumed_length, chunksize, unitlength);
      cudaDeviceSynchronize();
      err = cudaGetLastError();
      if (err != cudaSuccess) {
        printf("CUDA Error in mix: %s\n", cudaGetErrorString(err));
        exit(-1);
      }
      // decode
      decode<<<(chunknum + 31) / 32, 32>>>(d_input, input_size, d_prs, d_x1s, d_x2s, d_xs, chunknum, d_output, d_input_offsets, d_offsets_bondary, d_poss, chunksize, (output_size % chunksize) ? (output_size % chunksize) : chunksize, unitlength, maxassumed_length, d_realpred, d_realprs, d_realsmhashes, d_realcxt, d_pred, d_smhashes, d_cxt);
      cudaDeviceSynchronize();
      err = cudaGetLastError();
      if (err != cudaSuccess) {
        printf("CUDA Error in decode: %s\n", cudaGetErrorString(err));
        exit(-1);
      }
      // update poss
      uint64_t pos = 0;
      for (int chunki = 0; chunki < chunknum; chunki++) {
        cudaMemcpy(&pos, d_poss + chunki, sizeof(uint64_t), cudaMemcpyDeviceToHost);
        pos += unitlength - (pos % unitlength) < maxassumed_length + 1 ? (unitlength - (pos % unitlength)) : (maxassumed_length + 1);
        cudaMemcpy(d_poss + chunki, &pos, sizeof(uint64_t), cudaMemcpyHostToDevice);
      }
      cudaDeviceSynchronize();
      err = cudaGetLastError();
      if (err != cudaSuccess) {
        printf("CUDA Error in posupdate: %s\n", cudaGetErrorString(err));
        exit(-1);
      }
    }
    // delay the update when the whole unit is decoded
    // update StateMap
    updateStateMap<<<(chunknum * ordernum + 31) / 32, 32>>>(d_sms, d_realsmhashes, d_realprs, input_size, chunknum, unitlength, chunksize);
    // update Mixer
    updateMixer<<<(chunknum + 31) / 32, 32>>>(d_mxs, d_realprs, d_realpred, d_realcxt, chunknum, unitlength);
    cudaDeviceSynchronize();
    err = cudaGetLastError();
    if (err != cudaSuccess) {
      printf("CUDA Error in update: %s\n", cudaGetErrorString(err));
      exit(-1);
    }
  }
  // after all chunks are done, copy the output back to host and reset gpu
  cudaMemcpy(output, d_output, output_size, cudaMemcpyDeviceToHost);
  // cudaDeviceReset();
  // free device memory
  cudaFree(d_input);
  cudaFree(d_output);
  cudaFree(d_sms);
  cudaFree(d_mxs);
  cudaFree(d_pred);
  cudaFree(d_smhashes);
  cudaFree(d_cxt);
  cudaFree(d_prs);
  cudaFree(d_poss);
  cudaFree(d_x1s);
  cudaFree(d_x2s);
  cudaFree(d_xs);
  cudaFree(d_input_offsets);
  cudaFree(d_realpred);
  cudaFree(d_realsmhashes);
  cudaFree(d_realcxt);
  cudaFree(d_realprs);
}

void decompressprelearn(StateMap sms[8], Mixer *mixer, const uint8_t *input,
                        const uint64_t input_size, uint8_t *output,
                        uint64_t output_size) {
  printf("decompress with prelearn, input size: %lu, output size: %lu\n",
         input_size, output_size);
  // decompress the prelearn data
  uint32_t x1 = 0, x2 = 0xffffffff, x = 0;
  // x should be the first 4 bytes
  int ioffset = 0;
  for (int i = 0; i < 4; ++i) x = (x << 8) + (input[ioffset++] & 255);
  printf("initial x: %x, %u\n", x, x);
  for (int pos = 0; pos < output_size * 8; pos++) {
    uint64_t h[9] = {0};
    for (int i = 0; i < ordernum; i++) {
      if (pos >= 8 * i) {
        h[i] = output[pos / 8 - i];
      }
    }
    h[0] = output[pos / 8] >> (8 - pos % 8);
    h[0] = (1 << (pos % 8)) | h[0];
    int smp[ordernum] = {0};
    int smhash[ordernum];
    for (int i = 0; i < ordernum; i++) {
      uint64_t hash = 2166136261;
      for (int j = 0; j <= i && j < 8; j++) {
        hash ^= h[j];
        hash *= 16777619;
      }
      hash &= 0x1ffffff;
      StateMap &sm = sms[i];
      smp[i] = stretch_t[sm.predict(hash)];
      smhash[i] = hash;
    }
    Mixer &mx = mixer[0];
    int cxt = 0;
    if (pos >= 8) {
      cxt = output[pos / 8 - 1];
    }
    int olen = 0;
    for (int i = 0; i < ordernum && i < 8; i++) {
      if (smp[i] != 1)
        olen++;
    }
    cxt = olen + (cxt >> 5) * 10;
    int pr = mx.pHost(cxt, smp);
    if (pr < 2048) pr++;
    int xmid = x1 + ((x2 - x1) >> 12) * pr + ((x2 - x1 & 0xfff) * pr >> 12);
    int y = x <= xmid;
    // printf("pr: %d, smp: %d, %d, %d, %d, %d, %d, %d, %d, y: %d, x1: %u, x2: %u, x: %u, xmid: %u\n", pr, smp[0], smp[1], smp[2], smp[3], smp[4], smp[5], smp[6], smp[7], y, x1, x2, x, xmid);
    y ? (x2 = xmid) : (x1 = xmid + 1);
    // update mixer and sm
    mx.updateHost(y, pr, cxt, smp);
    for (int i = 0; i < ordernum; i++) {
      // printf("update pos: %d, orderid: %d\n", pos, i);
      // printf("sm[%d] state before update: %d, statep before update: %u\n", i, sms[i].states[smhash[i]], sms[i].statep[sms[i].states[smhash[i]]]);
      sms[i].updateHost(smhash[i], y);
    }
    // printf("pos: %d, pr: %d, y: %d, x1: %u, x2: %u, x: %u\n", pos, pr, y, x1,
    // x2, x);
    while (((x1 ^ x2) & 0xff000000) == 0) {
      x1 <<= 8;
      x2 = (x2 << 8) + 255;
      if (ioffset < input_size) {
        if (ioffset < input_size) {
          x = (x << 8) + input[ioffset++];
          // printf("read byte: %x, x: %u\n", input[ioffset - 1], x);
        } else
          x = (x << 8) + 255;
      } else {
        x = (x << 8) + 255;
      }
    }
    output[pos / 8] |= (y << (7 - pos % 8));
  }
  // printf("output[0..4]: %x, %x, %x, %x, %x,", output[0], output[1],
  //  output[2], output[3], output[4]);
}

void prelearn(StateMap sm[8], Mixer *mixer, uint8_t *input, uint64_t learn_size) {
  uint32_t x1 = 0, x2 = 0xffffffff;
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
    uint64_t xmid =
        x1 + ((x2 - x1) >> 12) * pr + ((x2 - x1 & 0xfff) * pr >> 12);
    y ? (x2 = xmid) : (x1 = xmid + 1);
    while (((x1 ^ x2) & 0xff000000) == 0) {
      x1 <<= 8;
      x2 = (x2 << 8) + 255;
    }
  }
}

// the first 2 byte should be the batch size
void decompress(std::ifstream &fin, uint64_t input_size, std::ofstream &fout, uint64_t output_size, int maxassumed_length = 3) {
  init();
  uint16_t batch_size = 0;
  uint64_t chunksize = 0;
  for (int i = 0; i < 2; ++i) batch_size = (batch_size << 8) + (fin.get() & 255);
  uint16_t unitlength = 0;
  for (int i = 0; i < 2; ++i) unitlength = (unitlength << 8) + (fin.get() & 255);
  for (int i = 0; i < 4; ++i) chunksize = (chunksize << 8) + (fin.get() & 255);
  printf("batch size: %d, unit length: %d, chunksize: %lu, output_size: %lu\n", batch_size, unitlength, chunksize, output_size);
  StateMap *sm_init = new StateMap[ordernum];
  Mixer *mixer_init = new Mixer[1];
  for (int i = 0; i < ordernum; i++) {
    sm_init[i].initHost();
  }
  mixer_init[0].initHost();
  uint64_t prelearn_size = 0, prelearn_compressed_size = 0;
  for (int i = 0; i < 4; ++i) prelearn_compressed_size = (prelearn_compressed_size << 8) + (fin.get() & 255);
  for (int i = 4; i < 8; ++i) prelearn_size = (prelearn_size << 8) + (fin.get() & 255);
  int prelearn_compressed = fin.get() & 255;
  printf("prelearn size: %lu, prelearn compressed size: %lu, prelearn compressed: %d\n",
         prelearn_size, prelearn_compressed_size, prelearn_compressed);
  if (!prelearn_compressed) prelearn_compressed_size = prelearn_size;
  uint8_t *prelearn_input = (uint8_t *)malloc(prelearn_compressed_size);
  fin.read((char *)prelearn_input, prelearn_compressed_size);
  if (!prelearn_compressed) {
    fout.write((char *)prelearn_input, prelearn_size);
    // initialize sm and mixer with compress
    prelearn(sm_init, mixer_init, prelearn_input, prelearn_size);
  } else {
    uint8_t *prelearn_output = (uint8_t *)malloc(prelearn_size);
    memset(prelearn_output, 0, prelearn_size);
    decompressprelearn(sm_init, mixer_init, prelearn_input, prelearn_compressed_size, prelearn_output, prelearn_size);
    fout.write((char *)prelearn_output, prelearn_size);
    free(prelearn_output);
  }
  free(prelearn_input);
  int total_chunk_num = (output_size - prelearn_size + chunksize - 1) / chunksize;
  printf("total chunk num: %d\n", total_chunk_num);
  int batchnum = (total_chunk_num + batch_size - 1) / batch_size;
  for (int batchid = 0; batchid < batchnum; batchid++) {
    int this_batch_chunk_num = (batchid == batchnum - 1) ? (total_chunk_num - batchid * batch_size) : batch_size;
    uint64_t batch_output_size = (batchid == batchnum - 1) ? (output_size - prelearn_size - batchid * batch_size * chunksize) : (this_batch_chunk_num * chunksize);
    printf("Decompressing batch %d/%d, chunk num: %d, output size: %lu\n", batchid + 1, batchnum, this_batch_chunk_num, batch_output_size);
    // read the batch_compressed_size and startchunkoffsets
    uint32_t batch_compressed_size = 0;
    fin.read((char *)&batch_compressed_size, sizeof(uint32_t));
    uint64_t *chunkstarts = (uint64_t *)malloc(sizeof(uint64_t) * (this_batch_chunk_num + 1));
    chunkstarts[0] = 0;
    for (int i = 1; i <= this_batch_chunk_num; i++) {
      uint64_t offset = 0;
      for (int j = 0; j < 4; j++) offset = (offset << 8) + (fin.get() & 255);
      chunkstarts[i] = offset - 4 * this_batch_chunk_num + 1;
    }
    uint8_t *input = (uint8_t *)malloc(batch_compressed_size - 4 * this_batch_chunk_num);
    fin.read((char *)input, batch_compressed_size - 4 * this_batch_chunk_num);
    uint8_t *batch_output = (uint8_t *)malloc(output_size);
    decompressbatch(input, batch_compressed_size - 4 * this_batch_chunk_num, batch_output, batch_output_size, chunkstarts, chunksize, unitlength, sm_init, mixer_init, maxassumed_length);
    fout.write((char *)batch_output, batch_output_size);
    free(input);
    free(batch_output);
    free(chunkstarts);
    std::cout << "Decompressed batch " << (batchid + 1) << "/" << batchnum << "\n";
  }
}

void print_help(const char *prog) {
  std::cout << "Usage: " << prog
            << " -i <input_file> -o <output_file> [-a <maxassumed_length>]\n";
  std::cout << "  -i <input_file>        Input compressed file\n";
  std::cout << "  -o <output_file>       Output decompressed file\n";
  std::cout << "  -a <N>                 Set maxassumed_length (default: 3)\n";
}

int main(int argc, char *argv[]) {
  const char *input_file = nullptr;
  const char *output_file = nullptr;
  int maxassumed_length = 3;

  for (int i = 1; i < argc; i++) {
    if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) {
      input_file = argv[++i];
    } else if (strcmp(argv[i], "-o") == 0 && i + 1 < argc) {
      output_file = argv[++i];
    } else if (strcmp(argv[i], "-a") == 0 && i + 1 < argc) {
      maxassumed_length = std::atoi(argv[++i]);
    } else {
      print_help(argv[0]);
      return 1;
    }
  }

  if (!input_file || !output_file) {
    print_help(argv[0]);
    return 1;
  }

  // 打开输入文件
  std::ifstream fin(input_file, std::ios::binary | std::ios::ate);
  if (!fin) {
    std::cerr << "Error: cannot open input file " << input_file << "\n";
    return 1;
  }

  size_t input_size = fin.tellg();
  fin.seekg(0, std::ios::beg);

  std::ofstream fout(output_file, std::ios::binary);
  if (!fout) {
    std::cerr << "Error: cannot open output file " << output_file << "\n";
    return 1;
  }

  // read the first 8 bytes to get the output size
  uint64_t output_size = 0;
  for (int i = 0; i < 4; ++i) {
    output_size = (output_size << 8) + (fin.get() & 255);
  }
  auto start_time = std::chrono::high_resolution_clock::now();
  decompress(fin, input_size, fout, output_size, maxassumed_length);

  fin.close();
  fout.close();

  auto end_time = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed = end_time - start_time;
  std::cout << "Decompression finished: " << input_file << " -> " << output_file
            << " (" << output_size << " bytes, maxassumed_length=" << maxassumed_length << ")\n";
  std::cout << "Total time: " << elapsed.count() << " sec" << std::endl;
  return 0;
}