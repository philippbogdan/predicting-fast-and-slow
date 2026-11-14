// Per-decision cost of distrust gates on top of Heimdall's admission path.
//
// Input (binary, little endian), produced by scripts/pipeline/bench_cost.py:
//   int64 n, then n records of: int64 x[12] (model inputs as io_replayer.c
//   builds them), int64 now_ns, int64 oldest_submit_ns (oldest outstanding
//   I/O), int64 out_bytes (bytes outstanding), int64 latency_us, int64
//   last_done_ns (the last completion before the submission, -1 if none);
//   then the fixed-point network (flashnet_algo.c layout) and gate params.
// For each gate the benchmark runs the full decision (inference + gate) over
// all records, repeated, on one pinned core, and reports ns per decision;
// "added" = gate variant minus inference-only.
#define _GNU_SOURCE
#include <math.h>
#include <sched.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define L0 12
#define L1 128
#define L2 16

typedef struct { int64_t x[L0]; int64_t now_ns, oldest_submit_ns, out_bytes, latency_us, last_done_ns; } rec_t;

static int64_t w0[L0], b0[L0], W1[L1 * L0], B1[L1], W2[L2 * L1], B2[L2], W3[L2], B3;
// gate params
static float g_mu[L0], g_pinv[L0 * L0], g_thr_e, g_thr_m, g_xmin[L0], g_xrng[L0];
static int64_t g_t_slow_ns, g_t_fast_ns;
static int n_cal;
static float *cal_z, *cal_s;  // conformal calibration set
static float tau = 1.0f;

static inline int64_t infer(const int64_t *x) {
    int64_t mid[L0], h1[L1], h2[L2];
    for (int j = 0; j < L0; j++) mid[j] = (x[j] - w0[j]) * b0[j];
    for (int j = 0; j < L1; j++) {
        int64_t s = 0;
        for (int i = 0; i < L0; i++) s += mid[i] * W1[j * L0 + i] >> 30;
        s += B1[j];
        h1[j] = s > 0 ? s : 0;
    }
    for (int j = 0; j < L2; j++) {
        int64_t s = 0;
        for (int i = 0; i < L1; i++) s += h1[i] * W2[j * L1 + i];
        s += B2[j];
        h2[j] = s > 0 ? s : 0;
    }
    int64_t s = 0;
    for (int i = 0; i < L2; i++) s += h2[i] * W3[i];
    return s + B3;
}

static inline void norm(const int64_t *x, float *z) {
    for (int i = 0; i < L0; i++) z[i] = ((float)x[i] - g_xmin[i]) / g_xrng[i];
}

// decisions: 0 admit, 1 reroute, 2 hedge
static inline int d_model(const rec_t *r) { return infer(r->x) >= 0; }

static inline int d_euclid(const rec_t *r) {
    int d = infer(r->x) >= 0;
    float z[L0], acc = 0;
    norm(r->x, z);
    for (int i = 0; i < L0; i++) { float t = z[i] - g_mu[i]; acc += t * t; }
    return acc > g_thr_e * g_thr_e ? 2 : d;
}

static inline int d_mahal(const rec_t *r) {
    int d = infer(r->x) >= 0;
    float z[L0], dz[L0], acc = 0;
    norm(r->x, z);
    for (int i = 0; i < L0; i++) dz[i] = z[i] - g_mu[i];
    for (int i = 0; i < L0; i++) {
        float t = 0;
        for (int k = 0; k < L0; k++) t += g_pinv[i * L0 + k] * dz[k];
        acc += dz[i] * t;
    }
    return acc > g_thr_m * g_thr_m ? 2 : d;
}

static inline int d_conformal(const rec_t *r) {
    int64_t f = infer(r->x);
    int d = f >= 0;
    float p1 = 1.0f / (1.0f + expf(-(float)f * 1e-9f));
    float s_alt = d ? p1 : 1 - p1;
    float z[L0];
    norm(r->x, z);
    float num = 1, den = 1;
    for (int c = 0; c < n_cal; c++) {
        const float *zc = cal_z + c * L0;
        float acc = 0;
        for (int i = 0; i < L0; i++) { float t = zc[i] - z[i]; acc += t * t; }
        float w = expf(-sqrtf(acc) / tau);
        den += w;
        num += (cal_s[c] >= s_alt) ? w : 0;
    }
    return (num / den > 0.1f) ? 2 : d;  // alternative label plausible -> uncertain
}

// an earlier outcome rule, timed for comparison (sizering_* in results/cost/cost.json): per size class the slow bits of
// the last 8 completed reads with a running count, and the oldest outstanding
// read's submit time (the client already stamps every submission, so now_ns
// is free).  Decision = class lookup + compare + age check; each read also
// costs one O(1) ring update when it completes (included here).
#define NCLS 32
static unsigned char ring_[NCLS][8], pos_[NCLS], cnt_[NCLS], nslow_[NCLS];
static inline int size_class(int64_t size) {
    uint64_t s = size < 512 ? 512 : (uint64_t)size;
    int c = 63 - __builtin_clzll(s);
    return c < NCLS ? c : NCLS - 1;
}
static inline void rule_update(int64_t size, int64_t latency_us) {
    int k = size_class(size);
    unsigned char sl = latency_us > g_t_slow_ns / 1000;
    if (cnt_[k] == 8) nslow_[k] -= ring_[k][pos_[k]]; else cnt_[k]++;
    ring_[k][pos_[k]] = sl;
    nslow_[k] += sl;
    pos_[k] = (pos_[k] + 1) & 7;
}
static inline int rule_decide(int d, const rec_t *r) {
    int k = size_class(r->x[1]);
    int ev = (cnt_[k] && 2 * nslow_[k] >= cnt_[k]) || (r->now_ns - r->oldest_submit_ns > g_t_slow_ns);
    if (!d && ev) return 2;      // distrusted "fast": both devices
    if (d && !ev) return 0;      // distrusted "slow": stay
    return d;
}
static inline int d_rule(const rec_t *r) {
    int d = infer(r->x) >= 0;
    int out = rule_decide(d, r);
    rule_update(r->x[1], r->latency_us);
    return out;
}

// the table (admission/tracker.py): one running slow-rate per
// (size class, load class, age class, quiet bit) cell.  Decision = cell index (two clz,
// four compares) + one load + compare; each read also costs one
// multiply-add on its cell when it completes (included here).  The cell is
// kept with the request between submission and completion.
#define NSZ 32
#define NLD 16
#define NAG 4
#define NQ 2
#define QUIET 16
static float trk_[NSZ * NLD * NAG * NQ];
static int64_t g_t_read_ns;
static inline int trk_cell(int64_t size, int64_t out_bytes, int64_t age_ns, int64_t now_ns, int64_t last_done_ns) {
    int c = size > 1 ? 63 - __builtin_clzll((uint64_t)size) : 0;
    c = c < NSZ ? c : NSZ - 1;
    uint64_t x = (uint64_t)out_bytes >> 12;
    int b = x ? 64 - __builtin_clzll(x) : 0;
    b = b < NLD ? b : NLD - 1;
    int a = (age_ns >= g_t_read_ns) + (age_ns >= 2 * g_t_read_ns) + (age_ns >= 3 * g_t_read_ns);
    int z = (last_done_ns >= 0) & (now_ns - last_done_ns > QUIET * g_t_read_ns);
    return ((c * NLD + b) * NAG + a) * NQ + z;
}
static inline int trk_decide(int d, const rec_t *r, int *cell) {
    int k = trk_cell(r->x[1], r->out_bytes, r->now_ns - r->oldest_submit_ns, r->now_ns, r->last_done_ns);
    *cell = k;
    int ev = trk_[k] >= 0.5f;
    // branch-free: distrusted "fast" (d=0, ev=1) -> 2 both devices; distrusted "slow" (d=1, ev=0) -> 0 stay
    return d ? ev : (ev << 1);
}
static inline void trk_update(int cell, int64_t latency_us) {
    float sl = (float)(latency_us * 1000 > g_t_read_ns);
    trk_[cell] += (sl - trk_[cell]) * (1.0f / 64.0f);
}
// The timeout: a read kept on the primary is watched until H after its submission.
// Submissions arrive in time order, so the deadlines form a FIFO; at each decision the overdue entries
// are taken off and duplicated unless they completed meanwhile.  (In the bench a read's completion is
// marked when it is decided if it finishes within H (by its deadline it is done), so the duplicates
// counted are the reads that would get one.)
#define WMASK ((1 << 16) - 1)
static int64_t w_dead[WMASK + 1];
static int32_t w_id[WMASK + 1];
static uint64_t w_head, w_tail;
static uint8_t *w_done;
static int64_t g_hedge_ns, g_dups, w_last_now;
static inline int trk_watch(int out, int64_t now_ns, int64_t id) {
    int dup = 0;
    if (now_ns < w_last_now) w_head = w_tail;   // the bench replayed the stream from its start again
    w_last_now = now_ns;
    while (w_head < w_tail && w_dead[w_head & WMASK] <= now_ns) { dup += !w_done[w_id[w_head & WMASK]]; w_head++; }
    if (out == 0) { w_dead[w_tail & WMASK] = now_ns + g_hedge_ns; w_id[w_tail & WMASK] = (int32_t)id; w_tail++; }
    return dup;
}
static inline void trk_complete(int cell, int64_t id, int64_t latency_us) {
    trk_update(cell, latency_us);
    w_done[id] = latency_us * 1000 < g_hedge_ns;
}
static void trk_reset(void) {
    for (int k = 0; k < NSZ * NLD * NAG * NQ; k++) trk_[k] = 0.5f;
    w_head = w_tail = 0;
    w_last_now = 0;
    g_dups = 0;
}
static const rec_t *g_R0;   // base of the record array (request ids are record indexes)
static inline int d_tracker(const rec_t *r) {
    int d = infer(r->x) >= 0, cell;
    int out = trk_decide(d, r, &cell);
    g_dups += trk_watch(out, r->now_ns, r - g_R0);
    trk_complete(cell, r - g_R0, r->latency_us);
    return out;
}

typedef int (*dec_fn)(const rec_t *);

// gate logic alone, the model decision already made (isolates the gate's own cost)
static int *pre_dec;
static double bench_tracker_only(const rec_t *R, int64_t n, int reps, int64_t *sink) {
    double best = 1e30;
    for (int rep = 0; rep < reps; rep++) {
        trk_reset();
        int64_t acc = 0;
        struct timespec a, b;
        clock_gettime(CLOCK_MONOTONIC, &a);
        for (int64_t i = 0; i < n; i++) {
            int cell;
            int out = trk_decide(pre_dec[i], &R[i], &cell);
            acc += out + trk_watch(out, R[i].now_ns, i);
            trk_complete(cell, i, R[i].latency_us);
        }
        clock_gettime(CLOCK_MONOTONIC, &b);
        *sink += acc;
        double per = ((b.tv_sec - a.tv_sec) + (b.tv_nsec - a.tv_nsec) * 1e-9) / n * 1e9;
        if (per < best) best = per;
    }
    return best;
}

static double bench_rule_only(const rec_t *R, int64_t n, int reps, int64_t *sink) {
    double best = 1e30;
    for (int rep = 0; rep < reps; rep++) {
        int64_t acc = 0;
        struct timespec a, b;
        clock_gettime(CLOCK_MONOTONIC, &a);
        for (int64_t i = 0; i < n; i++) {
            acc += rule_decide(pre_dec[i], &R[i]);
            rule_update(R[i].x[1], R[i].latency_us);
        }
        clock_gettime(CLOCK_MONOTONIC, &b);
        *sink += acc;
        double per = ((b.tv_sec - a.tv_sec) + (b.tv_nsec - a.tv_nsec) * 1e-9) / n * 1e9;
        if (per < best) best = per;
    }
    return best;
}

// gate logic alone (decision precomputed), for every gate: isolates each gate's own cost
static int g_kind;
static inline int gate_only(const rec_t *r, int d) {
    if (g_kind == 1) {            // Euclidean distance to the training centroid
        float z[L0], acc = 0;
        norm(r->x, z);
        for (int i = 0; i < L0; i++) { float t = z[i] - g_mu[i]; acc += t * t; }
        return acc > g_thr_e * g_thr_e ? 2 : d;
    } else if (g_kind == 2) {     // Mahalanobis distance
        float z[L0], dz[L0], acc = 0;
        norm(r->x, z);
        for (int i = 0; i < L0; i++) dz[i] = z[i] - g_mu[i];
        for (int i = 0; i < L0; i++) {
            float t = 0;
            for (int k = 0; k < L0; k++) t += g_pinv[i * L0 + k] * dz[k];
            acc += dz[i] * t;
        }
        return acc > g_thr_m * g_thr_m ? 2 : d;
    } else {                      // distance-weighted conformal, n_cal samples
        float p1 = d ? 0.9f : 0.1f, s_alt = d ? p1 : 1 - p1, z[L0];
        norm(r->x, z);
        float num = 1, den = 1;
        for (int c = 0; c < n_cal; c++) {
            const float *zc = cal_z + c * L0;
            float acc = 0;
            for (int i = 0; i < L0; i++) { float t = zc[i] - z[i]; acc += t * t; }
            float w = expf(-sqrtf(acc) / tau);
            den += w;
            num += (cal_s[c] >= s_alt) ? w : 0;
        }
        return (num / den > 0.1f) ? 2 : d;
    }
}
static double bench_gate_only(int kind, const rec_t *R, int64_t n, int reps, int64_t *sink);

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static double bench_gate_only(int kind, const rec_t *R, int64_t n, int reps, int64_t *sink) {
    g_kind = kind;
    double best = 1e30;
    for (int rep = 0; rep < reps; rep++) {
        int64_t acc = 0;
        double t0 = now_s();
        for (int64_t i = 0; i < n; i++) acc += gate_only(&R[i], pre_dec[i]);
        double t1 = now_s();
        *sink += acc;
        double per = (t1 - t0) / n * 1e9;
        if (per < best) best = per;
    }
    return best;
}

// In situ: the full decision path (model inference, then the tracker), with only the tracker's
// instructions (decision + completion update) bracketed by serialising time-stamp-counter reads, and
// an empty bracketed region measured in the same loop to subtract the bracket's own cost.  The
// fences forbid overlap with the surrounding inference, so this is an upper bound on what the
// tracker adds to the path; the model's working set is in cache exactly as in deployment.
#if defined(__x86_64__)
#include <x86intrin.h>
static inline uint64_t tsc_begin(void) { _mm_lfence(); uint64_t t = __rdtsc(); _mm_lfence(); return t; }
static inline uint64_t tsc_end(void) { unsigned aux; uint64_t t = __rdtscp(&aux); _mm_lfence(); return t; }
static double tsc_per_ns(void) {
    struct timespec a, b;
    uint64_t t0 = tsc_begin();
    clock_gettime(CLOCK_MONOTONIC, &a);
    do { clock_gettime(CLOCK_MONOTONIC, &b); } while ((b.tv_sec - a.tv_sec) * 1e9 + (b.tv_nsec - a.tv_nsec) < 2e8);
    uint64_t t1 = tsc_end();
    return (double)(t1 - t0) / ((b.tv_sec - a.tv_sec) * 1e9 + (b.tv_nsec - a.tv_nsec));
}
static double insitu_tracker_ns(const rec_t *R, int64_t n, int reps, int64_t *sink, double *bracket_ns,
                                double *decide_ns) {
    double f = tsc_per_ns(), best = 1e30, best0 = 0, bestd = 1e30;   // decision bracket includes the timeout watch
    for (int rep = 0; rep < reps; rep++) {
        trk_reset();
        uint64_t sum = 0, sum0 = 0, sumd = 0;
        int64_t acc = 0;
        for (int64_t i = 0; i < n; i++) {
            int d = infer(R[i].x) >= 0, cell;
            uint64_t a = tsc_begin();
            int out = trk_decide(d, &R[i], &cell);
            out += trk_watch(out, R[i].now_ns, i) << 2;
            uint64_t m = tsc_end();
            uint64_t a2 = tsc_begin();
            trk_complete(cell, i, R[i].latency_us);
            uint64_t b = tsc_end();
            uint64_t c0 = tsc_begin();
            uint64_t c1 = tsc_end();
            sumd += m - a;
            sum += (m - a) + (b - a2);
            sum0 += c1 - c0;
            acc += out;
        }
        *sink += acc;
        double per = ((double)sum - 2.0 * (double)sum0) / n / f;     // two brackets: decision, update
        double perd = ((double)sumd - (double)sum0) / n / f;
        if (per < best) { best = per; best0 = (double)sum0 / n / f; }
        if (perd < bestd) bestd = perd;
    }
    *bracket_ns = best0;
    *decide_ns = bestd;
    return best;
}
#else
static double insitu_tracker_ns(const rec_t *R, int64_t n, int reps, int64_t *sink, double *bracket_ns,
                                double *decide_ns) {
    (void)R; (void)n; (void)reps; (void)sink; *bracket_ns = -1; *decide_ns = -1; return -1;   // x86 TSC only
}
#endif

// Paired blocks: the full decision path with and without the tracker, timed on the same block of
// records back to back (order alternating between repetitions so drift cancels); the added cost is
// the median over repetitions of the per-decision difference.  Robust to the +-300 ns run-to-run
// variation that makes "best full path with" minus "best full path without" meaningless at about 1.8 us.
static int cmp_dbl(const void *a, const void *b) {
    double x = *(const double *)a, y = *(const double *)b;
    return x < y ? -1 : x > y;
}
static double paired_added(dec_fn with, dec_fn without, const rec_t *R, int64_t n, int reps, int64_t *sink,
                           double *q10, double *q90) {
    double *d = malloc(sizeof(double) * reps);
    for (int rep = 0; rep < reps; rep++) {
        double tw = 0, to = 0;
        for (int half = 0; half < 2; half++) {
            int first_with = (rep + half) % 2 == 0;
            dec_fn f = first_with ? with : without;
            int64_t acc = 0;
            double t0 = now_s();
            for (int64_t i = 0; i < n; i++) acc += f(&R[i]);
            double dt = now_s() - t0;
            *sink += acc;
            if (first_with) tw += dt; else to += dt;
        }
        d[rep] = (tw - to) / n * 1e9;
    }
    qsort(d, reps, sizeof(double), cmp_dbl);
    double med = d[reps / 2];
    *q10 = d[reps / 10];
    *q90 = d[(reps * 9) / 10];
    free(d);
    return med;
}

static double bench(dec_fn f, const rec_t *R, int64_t n, int reps, int64_t *sink) {
    double best = 1e30;
    for (int rep = 0; rep < reps; rep++) {
        int64_t acc = 0;
        double t0 = now_s();
        for (int64_t i = 0; i < n; i++) acc += f(&R[i]);
        double t1 = now_s();
        *sink += acc;
        double per = (t1 - t0) / n * 1e9;
        if (per < best) best = per;
    }
    return best;
}

static int cmp_i64(const void *a, const void *b) {
    int64_t x = *(const int64_t *)a, y = *(const int64_t *)b;
    return x < y ? -1 : x > y;
}

static void rd(void *p, size_t sz, size_t n, FILE *fp) {
    if (fread(p, sz, n, fp) != n) { fprintf(stderr, "short read\n"); exit(1); }
}

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: gate_cost input.bin [reps] [cpu]\n"); return 1; }
    int reps = argc > 2 ? atoi(argv[2]) : 15;
    int cpu = argc > 3 ? atoi(argv[3]) : 0;
#ifdef __linux__
    cpu_set_t set; CPU_ZERO(&set); CPU_SET(cpu, &set);
    if (sched_setaffinity(0, sizeof(set), &set) != 0) perror("sched_setaffinity");
#endif
    FILE *fp = fopen(argv[1], "rb");
    if (!fp) { perror("open"); return 1; }
    int64_t n;
    rd(&n, 8, 1, fp);
    rec_t *R = malloc(sizeof(rec_t) * n);
    rd(R, sizeof(rec_t), n, fp);
    rd(w0, 8, L0, fp); rd(b0, 8, L0, fp); rd(W1, 8, L1 * L0, fp); rd(B1, 8, L1, fp);
    rd(W2, 8, L2 * L1, fp); rd(B2, 8, L2, fp); rd(W3, 8, L2, fp); rd(&B3, 8, 1, fp);
    rd(g_xmin, 4, L0, fp); rd(g_xrng, 4, L0, fp); rd(g_mu, 4, L0, fp); rd(g_pinv, 4, L0 * L0, fp);
    rd(&g_thr_e, 4, 1, fp); rd(&g_thr_m, 4, 1, fp); rd(&g_t_slow_ns, 8, 1, fp); rd(&g_t_fast_ns, 8, 1, fp);
    g_t_read_ns = g_t_slow_ns;
    g_R0 = R;
    w_done = calloc(n, 1);
    {   // H: 98th percentile of read latency over the first third of the timed reads (a stand-in for the training window)
        int64_t m = n / 3 > 0 ? n / 3 : n;
        int64_t *lat = malloc(sizeof(int64_t) * m);
        for (int64_t i = 0; i < m; i++) lat[i] = R[i].latency_us;
        qsort(lat, m, sizeof(int64_t), cmp_i64);
        g_hedge_ns = lat[(int64_t)(0.98 * (m - 1))] * 1000;
        free(lat);
    }
    int32_t nc;
    rd(&nc, 4, 1, fp);
    float *cz = malloc(sizeof(float) * L0 * nc), *cs = malloc(sizeof(float) * nc);
    rd(cz, 4, (size_t)L0 * nc, fp); rd(cs, 4, nc, fp);
    fclose(fp);
    int64_t sink = 0;
    pre_dec = malloc(sizeof(int) * n);
    for (int64_t i = 0; i < n; i++) pre_dec[i] = infer(R[i].x) >= 0;
    double t_tracker_only = bench_tracker_only(R, n, reps, &sink);
    double t_rule_only = bench_rule_only(R, n, reps, &sink);
    double t_model = bench(d_model, R, n, reps, &sink);
    trk_reset();
    double t_tracker = bench(d_tracker, R, n, reps, &sink);
    double dup_share = (double)g_dups / ((double)n * reps);
    double t_rule = bench(d_rule, R, n, reps, &sink);
    double t_e = bench(d_euclid, R, n, reps, &sink);
    double t_m = bench(d_mahal, R, n, reps, &sink);
    cal_z = cz; cal_s = cs;
    n_cal = nc < 100 ? nc : 100;
    int64_t nsub = n < 200000 ? n : 200000;
    double t_c100 = bench(d_conformal, R, nsub, reps > 3 ? 3 : reps, &sink);
    n_cal = nc;
    int64_t nsub2 = n < 20000 ? n : 20000;
    double t_c1000 = bench(d_conformal, R, nsub2, 3, &sink);
    for (int k = 0; k < NSZ * NLD * NAG * NQ; k++) trk_[k] = 0.5f;
    int64_t nblk = n < 50000 ? n : 50000;
    double p10, p90, p10e, p90e;
    double bracket_ns, insitu_decide;
    double insitu = insitu_tracker_ns(R, n < 1000000 ? n : 1000000, 5, &sink, &bracket_ns, &insitu_decide);
    // the tracker alone on a cache-resident block of requests (the deciding request's fields are hot in a
    // controller; streaming millions of 136-byte records from memory would time the benchmark, not the rule)
    int64_t nres = n < 2048 ? n : 2048;
    double t_tracker_hot = 1e30;
    for (int rep = 0; rep < 2000; rep++) {
        int64_t acc = 0;
        trk_reset();
        double t0 = now_s();
        for (int64_t i = 0; i < nres; i++) {
            int cell;
            int out = trk_decide(pre_dec[i], &R[i], &cell);
            acc += out + trk_watch(out, R[i].now_ns, i);
            trk_complete(cell, i, R[i].latency_us);
        }
        double per = (now_s() - t0) / nres * 1e9;
        sink += acc;
        if (per < t_tracker_hot) t_tracker_hot = per;
    }
    double paired = paired_added(d_tracker, d_model, R, nblk, 61, &sink, &p10, &p90);
    double paired_e = paired_added(d_euclid, d_model, R, nblk, 61, &sink, &p10e, &p90e);
    double e_only = bench_gate_only(1, R, n, reps, &sink);
    double m_only = bench_gate_only(2, R, n, reps, &sink);
    n_cal = nc < 100 ? nc : 100;
    double c100_only = bench_gate_only(3, R, nsub, reps > 3 ? 3 : reps, &sink);
    n_cal = nc;
    double c1000_only = bench_gate_only(3, R, nsub2, 3, &sink);
    printf("{\"euclid_only_ns\": %.3f, \"mahal_only_ns\": %.3f, \"conformal100_only_ns\": %.2f, "
           "\"conformal_full_only_ns\": %.2f, ", e_only, m_only, c100_only, c1000_only);
    printf("\"tracker_only_ns\": %.3f, \"tracker_ns\": %.2f, \"added_tracker_ns\": %.2f, ",
           t_tracker_only, t_tracker, t_tracker - t_model);
    printf("\"hedge_us\": %lld, \"duplicated_share\": %.4f, ", (long long)(g_hedge_ns / 1000), dup_share);
    printf("\"insitu_tracker_ns\": %.3f, \"insitu_decide_ns\": %.3f, \"insitu_bracket_ns\": %.3f, "
           "\"tracker_hot_ns\": %.3f, ", insitu, insitu_decide, bracket_ns, t_tracker_hot);
    printf("\"paired_added_tracker_ns\": %.2f, \"paired_added_tracker_q10_ns\": %.2f, "
           "\"paired_added_tracker_q90_ns\": %.2f, \"paired_added_euclid_ns\": %.2f, "
           "\"paired_added_euclid_q10_ns\": %.2f, \"paired_added_euclid_q90_ns\": %.2f, \"paired_block\": %lld, ",
           paired, p10, p90, paired_e, p10e, p90e, (long long)nblk);
    printf("\"n\": %lld, \"model_ns\": %.2f, \"sizering_ns\": %.2f, \"euclid_ns\": %.2f, \"mahal_ns\": %.2f, "
           "\"conformal100_ns\": %.2f, \"conformal%d_ns\": %.2f, \"added_sizering_ns\": %.2f, "
           "\"added_euclid_ns\": %.2f, \"added_mahal_ns\": %.2f, \"sizering_only_ns\": %.3f, \"sink\": %lld}\n",
           (long long)n, t_model, t_rule, t_e, t_m, t_c100, nc, t_c1000, t_rule - t_model,
           t_e - t_model, t_m - t_model, t_rule_only, (long long)sink);
    return 0;
}
