// Closed-loop replay of a two-client, two-SSD pair on real drives, with Heimdall and the gates of the paper.
//
// The device counterpart of the repository's simulator (admission/sim.py).  Two clients, each replaying its own
// trace with 8 worker threads that take I/Os in trace order and sleep until each I/O's timestamp (Heimdall's
// io_replayer.c).  Writes go to the client's own drive; a policy routes each read to its own drive, the other
// drive, or both.  Every decision is made from what the client knows at that instant, as the simulator does:
//   * Heimdall's 12 inputs as io_replayer.c builds them (read flag, size, the client's own outstanding I/Os on
//     its drive including this one, and queue length / latency / throughput of its last 3 completions there),
//     evaluated by the trained network in double precision ("slow" when the logit is >= 0, as the simulator);
//     the inputs of every 64th read are kept, so that the fixed-point network of Heimdall's flashnet_algo.c can
//     be compared with it afterwards (admission/heimdall.py FixedPointModel);
//   * the study's distance gates: Euclidean / Mahalanobis distance of the normalised input to the training
//     centroid above the 90th percentile of the training distances;
//   * the table of admission/tracker.py: one table for the client's own drive, learned from the
//     completions of its own reads there; theta 1/2;
//   * the mistake oracle: distrusts exactly the decisions that disagree with the read's label (its latency in
//     the baseline replay the network was trained on, above the knee of that replay's reads);
//   * actions on a distrusted read, as the simulator's: 0 hedge, 1 flip, 2 mixed (the table's gate), 3 hedge
//     whenever the final call is "slow", 4 keep the model's reroutes and hedge a distrusted "fast";
//   * the timeout (policy 4, or --timeout with any policy): a read sent to one drive only and unanswered H after
//     its submission gets a copy on the other drive (its own drive if it was moved); the worker still waits for
//     its own copy, the first answer is the read's latency; the copy counts as outstanding nowhere.
// I/O: O_DIRECT on one pre-filled file per drive, Linux AIO with one context per worker, so a worker whose read
// is hedged returns at the first answer.  A losing copy cannot be cancelled on a drive: it runs to completion and
// teaches the client nothing (the simulator removes it).  Every I/O whose trace time is within the duration
// ((int64)(ts_ms * 1e6) <= duration in ns) is issued, however late, so every policy is measured on the same
// requests (as the simulator does).
//
// With --cut 1 (Heimdall's baseline replayer, io_replayer.c), no I/O is issued once the replay has run for the
// duration in wall time; the I/Os not issued by then are left out (the held-out baseline replays).
//
// Our own I/O is counted per drive in a shared-memory block (--counters) that the quiet-machine recorder reads
// once a second beside /proc/diskstats.
//
// --map0 PATH --map1 PATH keep each drive's I/O inside the chunks of one member of a RAID0 array, for two drives
// that are the two members of one array (the identical enterprise drives).  Each map lists, in order, the file
// offsets (int64) of the file's 512 KiB chunks that lie wholly on that member (from FIEMAP).  An offset is wrapped
// to the member's chunks and aligned to 4 KiB; an I/O of at most one chunk is kept inside one chunk (moved back
// within it if it would cross the chunk's end); a larger I/O starts at a chunk and is split into one piece per
// chunk, all submitted at once, and is answered when its last piece completes.  Without maps an offset is wrapped
// to the file.
//
// Usage: pair_replayer --policy P [--gate G] [--action A] [--timeout 0|1] [--cut 0|1] --duration-ns N --counters PATH
//            --c0 PARAMS TRACE FILE OUT [LABELS] --c1 PARAMS TRACE FILE OUT [LABELS]
//            [--map0 MAP --map1 MAP]
//   policy: 0 no admission control, 2 Heimdall, 3 Heimdall + gate, 4 the timeout alone (no model)
//   gate:   1 Euclidean, 2 Mahalanobis, 4 mistake oracle (needs LABELS), 7 tracker
// Trace (little endian): int64 n; double ts_ms[n]; int64 offset[n]; int32 size[n]; int8 type[n] (1 read)
// Params: char magic[8] "PRPARAM1"; double xmin[12] xrng[12] W1[12*128] b1[128] W2[128*16] b2[16] W3[16] b3
//         mu[12] pinv[144] thr_e thr_m; int64 T_ns H_ns
// Labels: int8 per trace I/O (1 slow, 0 fast, -1 none)
// Out (little endian): char magic[8] "PRLOG001"; int64 n; then per I/O, arrays: int64 submit_ns[n],
//   int64 lat_ns[n] (first answer; -1 not issued), int64 own_lat_ns[n] (-1 none), int8 route[n] (0 own, 1 other,
//   2 both), int8 ans_other[n], int8 tcopy[n], int8 pred[n], int8 flag[n], float q[n] (tracker entry or
//   distance), float logit[n], int32 qlen[n], then int64 ns = (n + 63) / 64 and int64 x[ns][12]: the model's
//   inputs of I/O 64k (all 0 where it is a write or no model ran), then int64 dseq[n] and cseq[n]: the order of
//   the client's state changes (its decision on each I/O; its own copy leaving the books), -1 where none
#define _GNU_SOURCE
#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <linux/aio_abi.h>
#include <math.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>

#define NW 8          // workers per client (io_replayer.c)
#define NSLOT 256     // copies in flight per worker (its own read's copies plus losers not yet reaped)
#define L0 12
#define L1 128
#define L2 16
#define MAXOUT 256
#define CHUNK (512 * 1024)  // the RAID0's chunk
#define MAXPC 8             // pieces of one I/O (the largest, 3 MiB, is 6 chunks)

// the table (admission/tracker.py)
#define NSZ 32
#define NLD 16
#define NAG 4
#define NQ 2
#define NCELLS (NSZ * NLD * NAG * NQ)
#define QUIET 16
static const double ALPHA = 1.0 / 64.0;
static const double THETA = 0.5;

// ---------------------------------------------------------------- shared counters (read by quietrec)
typedef struct {
    char magic[8];                 // "PRCOUNT1"
    int64_t start_mono_ns;         // CLOCK_MONOTONIC at the replay's start (0 until it starts)
    int64_t end_mono_ns;           // ... at its end (0 until it ends)
    int64_t done[2][2][2];         // [drive][0 write, 1 read][0 count, 1 bytes] completed on that drive
    int64_t sub[2][2][2];          // the same, submitted
    int64_t issued[2];             // I/Os issued per client
} counters_t;
static counters_t *K;

static int io_setup_(unsigned nr, aio_context_t *c) { return syscall(__NR_io_setup, nr, c); }
static int io_submit_(aio_context_t c, long n, struct iocb **p) { return syscall(__NR_io_submit, c, n, p); }
static int io_getevents_(aio_context_t c, long mn, long mx, struct io_event *e, struct timespec *t) {
    return syscall(__NR_io_getevents, c, mn, mx, e, t);
}

static int64_t mono_ns(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (int64_t)t.tv_sec * 1000000000LL + t.tv_nsec;
}

static int g_policy = -1, g_gate = 0, g_action = 0, g_timeout = 0, g_cut = 0;
static int64_t g_dur_ns, g_start;
static int g_fd[2];
static int64_t g_fsize[2];
static int64_t *g_map[2], g_nmap[2];   // per drive: file offsets of the member's whole chunks (NULL: no map)

typedef struct {
    double xmin[L0], xrng[L0], W1[L0 * L1], b1[L1], W2[L1 * L2], b2[L2], W3[L2], b3;
    double mu[L0], pinv[L0 * L0], thr_e, thr_m;
    int64_t T_ns, H_ns;
} params_t;

typedef struct client {
    int id;
    params_t P;
    int64_t n, max_size;
    double *ts_ms;
    int64_t *off;
    int32_t *size;
    int8_t *type;
    int8_t *label;
    int64_t *sub_ns, *lat_ns, *own_lat_ns;
    int32_t *qlen;
    int8_t *route, *ans_other, *tcopy, *pred, *flag;
    float *q, *logit;
    int64_t *xs;                        // the inputs of every 64th I/O
    int64_t *dseq, *cseq;               // the order of the client's state changes: each decision, each own copy
    int64_t seq;                        // leaving the books (completed or given up), under the client's lock
    void *buf;
    pthread_mutex_t mu;                 // the client's state below
    int64_t adm;                        // own outstanding I/Os on its drive (model input)
    int64_t hq[3], hl[3], ht[3];
    int nh;
    int64_t out_t[MAXOUT], out_b[MAXOUT];
    int out_on[MAXOUT];
    int64_t last_done;                  // last observed completion of an own admitted I/O on its drive (-1 none)
    double tab[NCELLS];
    _Atomic int64_t next;
    _Atomic int64_t late;
} client_t;

static client_t C[2];

// ---------------------------------------------------------------- the network, as the simulator evaluates it
static double logit_of(const params_t *P, const double *x) {
    double z[L0], h1[L1], h2[L2];
    for (int i = 0; i < L0; i++) z[i] = (x[i] - P->xmin[i]) / P->xrng[i];
    for (int j = 0; j < L1; j++) {
        double s = P->b1[j];
        for (int i = 0; i < L0; i++) s += z[i] * P->W1[i * L1 + j];
        h1[j] = s > 0 ? s : 0.0;
    }
    for (int j = 0; j < L2; j++) {
        double s = P->b2[j];
        for (int i = 0; i < L1; i++) s += h1[i] * P->W2[i * L2 + j];
        h2[j] = s > 0 ? s : 0.0;
    }
    double s = P->b3;
    for (int i = 0; i < L2; i++) s += h2[i] * P->W3[i];
    return s;
}

static double distance(const params_t *P, const double *x, int maha) {
    double z[L0], acc = 0;
    for (int i = 0; i < L0; i++) z[i] = (x[i] - P->xmin[i]) / P->xrng[i] - P->mu[i];
    if (!maha) {
        for (int i = 0; i < L0; i++) acc += z[i] * z[i];
    } else {
        for (int i = 0; i < L0; i++)
            for (int k = 0; k < L0; k++) acc += z[i] * P->pinv[i * L0 + k] * z[k];
    }
    return sqrt(acc > 0 ? acc : 0);
}

// the cell of a read, exactly as admission/tracker.py cell_of
static int cell_of(int64_t size, int64_t out_bytes, int64_t oldest_ns, int64_t since_ns, int64_t T_ns) {
    int c = 0;
    int64_t s = size;
    while (s > 1 && c < NSZ - 1) { s >>= 1; c++; }
    int64_t x = out_bytes >> 12;
    int b = 0;
    while (x > 0 && b < NLD - 1) { x >>= 1; b++; }
    int64_t a = T_ns > 0 ? oldest_ns / T_ns : 0;
    if (a > NAG - 1) a = NAG - 1;
    if (a < 0) a = 0;
    int z = (since_ns >= 0 && since_ns > QUIET * T_ns) ? 1 : 0;
    return ((c * NLD + b) * NAG + (int)a) * NQ + z;
}

// ---------------------------------------------------------------- per-worker copies
typedef struct {
    struct iocb cb[MAXPC];
    int np, left;                       // pieces of this copy, and those not yet completed
    int busy, kind, dev, cell, rw;      // kind: 0 own, 1 moved, 2 hedge own, 3 hedge other, 4 timeout copy
    int64_t idx, req, bytes;
    int out_slot, cancelled;
} copy_t;

typedef struct {
    client_t *c;
    int w;
    aio_context_t ctx;
    copy_t slot[NSLOT];
    int64_t req;
    int answered, primary_done;
    copy_t *own, *oth;
} worker_t;

static int out_add(client_t *c, int64_t t, int64_t b) {
    for (int k = 0; k < MAXOUT; k++)
        if (!c->out_on[k]) { c->out_on[k] = 1; c->out_t[k] = t; c->out_b[k] = b; return k; }
    fprintf(stderr, "outstanding table full\n");
    exit(1);
}

static void out_state(client_t *c, int64_t now, int64_t *bytes, int64_t *oldest) {
    int64_t b = 0, o = 0;
    for (int k = 0; k < MAXOUT; k++)
        if (c->out_on[k]) { b += c->out_b[k]; if (now - c->out_t[k] > o) o = now - c->out_t[k]; }
    *bytes = b;
    *oldest = o;
}

// an own admitted copy completes (observed), or leaves the books because the other copy answered first (not
// observed, as the simulator's cancelled copy)
static void own_complete(client_t *c, copy_t *cp, int64_t t_done, int observed) {
    int64_t i = cp->idx;
    pthread_mutex_lock(&c->mu);
    c->cseq[i] = c->seq++;
    c->adm--;
    c->out_on[cp->out_slot] = 0;
    if (observed) {
        int64_t lat_us = (t_done - c->sub_ns[i]) / 1000;
        for (int h = 2; h > 0; h--) { c->hq[h] = c->hq[h - 1]; c->hl[h] = c->hl[h - 1]; c->ht[h] = c->ht[h - 1]; }
        c->hq[0] = c->qlen[i];
        c->hl[0] = lat_us;
        c->ht[0] = c->size[i] / (lat_us + 1);
        if (c->nh < 3) c->nh++;
        c->last_done = t_done;
        if (c->type[i] == 1 && cp->cell >= 0) {
            double sl = (t_done - c->sub_ns[i]) > c->P.T_ns ? 1.0 : 0.0;
            c->tab[cp->cell] += ALPHA * (sl - c->tab[cp->cell]);
        }
    }
    pthread_mutex_unlock(&c->mu);
}

static void reap(worker_t *W, struct io_event *ev, int64_t t_done) {
    client_t *c = W->c;
    copy_t *cp = (copy_t *)(uintptr_t)ev->data;
    const struct iocb *piece = (const struct iocb *)(uintptr_t)ev->obj;
    if ((int64_t)ev->res < 0 || (uint64_t)ev->res != piece->aio_nbytes)
        fprintf(stderr, "client %d idx %" PRId64 ": I/O returned %lld\n", c->id, cp->idx, (long long)ev->res);
    if (--cp->left > 0) return;                            // an I/O is answered when its last piece is
    __atomic_fetch_add(&K->done[cp->dev][cp->rw][0], 1, __ATOMIC_RELAXED);
    __atomic_fetch_add(&K->done[cp->dev][cp->rw][1], cp->bytes, __ATOMIC_RELAXED);
    cp->busy = 0;
    if (cp->req != W->req) return;                         // a loser of an earlier read: teaches nothing
    int64_t i = cp->idx;
    int first = !W->answered;
    if (first) {
        W->answered = 1;
        c->lat_ns[i] = t_done - c->sub_ns[i];
        c->ans_other[i] = cp->dev != c->id;
    }
    if (cp->kind == 0 || cp->kind == 2) {
        c->own_lat_ns[i] = t_done - c->sub_ns[i];
        if (!cp->cancelled) own_complete(c, cp, t_done, 1);
        if (cp->kind == 2 && first && W->oth) W->oth->cancelled = 1;
        W->own = NULL;
        W->primary_done = 1;
    } else if (cp->kind == 1) {
        W->oth = NULL;
        W->primary_done = 1;                                // a moved read's only copy
    } else if (cp->kind == 3) {
        if (first && W->own && !W->own->cancelled) {
            W->own->cancelled = 1;                          // the own copy leaves the books, as if cancelled
            own_complete(c, W->own, t_done, 0);
        }
        W->oth = NULL;
    }
}

static copy_t *submit(worker_t *W, int dev, int64_t i, int kind, int out_slot, int cell) {
    client_t *c = W->c;
    copy_t *cp = NULL;
    for (;;) {
        for (int k = 0; k < NSLOT; k++)
            if (!W->slot[k].busy) { cp = &W->slot[k]; break; }
        if (cp) break;
        struct io_event ev;                                 // every slot holds a loser in flight: reap one
        if (io_getevents_(W->ctx, 1, 1, &ev, NULL) == 1) reap(W, &ev, mono_ns() - g_start);
    }
    cp->busy = 1;
    cp->kind = kind;
    cp->dev = dev;
    cp->idx = i;
    cp->req = W->req;
    cp->out_slot = out_slot;
    cp->cell = cell;
    cp->cancelled = 0;
    cp->rw = c->type[i] == 1 ? 1 : 0;
    int64_t sz = c->size[i];
    int64_t poff[MAXPC], plen[MAXPC];
    int np = 1;
    if (!g_map[dev]) {
        int64_t off = (c->off[i] % g_fsize[dev]) / 4096 * 4096;
        if (off + sz > g_fsize[dev]) off = (g_fsize[dev] - sz) / 4096 * 4096;
        poff[0] = off;
        plen[0] = sz;
    } else {
        int64_t M = g_nmap[dev];
        int64_t o = (c->off[i] % (M * CHUNK)) / 4096 * 4096;
        int64_t k = o / CHUNK, w = o % CHUNK;
        if (sz <= CHUNK) {
            if (w + sz > CHUNK) w = (CHUNK - sz) / 4096 * 4096;
            poff[0] = g_map[dev][k] + w;
            plen[0] = sz;
        } else {
            np = (int)((sz + CHUNK - 1) / CHUNK);
            if (np > MAXPC) { fprintf(stderr, "I/O of %" PRId64 " bytes: more than %d chunks\n", sz, MAXPC); exit(1); }
            if (k + np > M) k = M - np;
            for (int j = 0; j < np; j++) {
                poff[j] = g_map[dev][k + j];
                plen[j] = (j < np - 1) ? CHUNK : sz - (int64_t)(np - 1) * CHUNK;
            }
        }
    }
    cp->bytes = sz;
    cp->np = cp->left = np;
    struct iocb *pp[MAXPC];
    for (int j = 0; j < np; j++) {
        memset(&cp->cb[j], 0, sizeof(cp->cb[j]));
        cp->cb[j].aio_fildes = g_fd[dev];
        cp->cb[j].aio_lio_opcode = c->type[i] == 1 ? IOCB_CMD_PREAD : IOCB_CMD_PWRITE;
        cp->cb[j].aio_buf = (uint64_t)(uintptr_t)((char *)c->buf + (j * (int64_t)CHUNK)); // one buffer per client
        cp->cb[j].aio_nbytes = (uint64_t)plen[j];
        cp->cb[j].aio_offset = poff[j];
        cp->cb[j].aio_data = (uint64_t)(uintptr_t)cp;
        pp[j] = &cp->cb[j];
    }
    int r, done_sub = 0;
    while (done_sub < np) {
        r = io_submit_(W->ctx, np - done_sub, pp + done_sub);
        if (r > 0) { done_sub += r; continue; }
        if (r < 0 && errno == EAGAIN) { usleep(10); continue; }
        fprintf(stderr, "io_submit failed: %d errno %d\n", r, errno);
        exit(1);
    }
    __atomic_fetch_add(&K->sub[dev][cp->rw][0], 1, __ATOMIC_RELAXED);
    __atomic_fetch_add(&K->sub[dev][cp->rw][1], sz, __ATOMIC_RELAXED);
    return cp;
}

static void *worker(void *arg) {
    worker_t *W = (worker_t *)arg;
    client_t *c = W->c;
    const int me = c->id, other = 1 - c->id;
    if (io_setup_(NSLOT * (g_map[0] || g_map[1] ? MAXPC : 2), &W->ctx) < 0) { perror("io_setup"); exit(1); }
    for (;;) {
        int64_t i = atomic_fetch_add(&c->next, 1);
        if (i >= c->n) break;
        int64_t due = (int64_t)(c->ts_ms[i] * 1e6);
        if (due > g_dur_ns) break;
        int64_t el = mono_ns() - g_start;
        if (el < due) {
            struct timespec ts = {(due - el) / 1000000000LL, (due - el) % 1000000000LL};
            while (nanosleep(&ts, &ts) != 0 && errno == EINTR) {}
        } else {
            atomic_fetch_add(&c->late, 1);
        }
        if (g_cut && mono_ns() - g_start > g_dur_ns) break;   // io_replayer.c: the run ends at the duration
        W->req++;
        W->answered = W->primary_done = 0;
        W->own = W->oth = NULL;
        int64_t t_sub = mono_ns() - g_start;
        c->sub_ns[i] = t_sub;
        __atomic_fetch_add(&K->issued[me], 1, __ATOMIC_RELAXED);
        int is_read = c->type[i] == 1;
        int target = 0, pred = 0, flag = 0, cell = -1, out_slot = -1;
        float q = -1.0f, lg_out = 0.0f;
        pthread_mutex_lock(&c->mu);
        c->dseq[i] = c->seq++;
        if (is_read && (g_policy == 2 || g_policy == 3)) {
            double x[L0];
            int64_t xi[L0];
            xi[0] = 1;
            xi[1] = c->size[i];
            xi[2] = c->adm + 1;
            for (int h = 0; h < 3; h++) {
                xi[3 + h] = h < c->nh ? c->hq[h] : 0;
                xi[6 + h] = h < c->nh ? c->hl[h] : 0;
                xi[9 + h] = h < c->nh ? c->ht[h] : 0;
            }
            for (int k = 0; k < L0; k++) x[k] = (double)xi[k];
            double lg = logit_of(&c->P, x);
            lg_out = (float)lg;
            pred = lg >= 0;
            if ((i & 63) == 0) memcpy(&c->xs[(i >> 6) * L0], xi, sizeof(xi));
            target = pred ? 1 : 0;
            if (g_policy == 3) {
                int unc = 0;
                if (g_gate == 1 || g_gate == 2) {
                    double d = distance(&c->P, x, g_gate == 2);
                    q = (float)d;
                    unc = d > (g_gate == 1 ? c->P.thr_e : c->P.thr_m);
                } else if (g_gate == 4) {
                    if (c->label && c->label[i] >= 0) unc = pred != (c->label[i] == 1);
                } else if (g_gate == 7) {
                    int64_t byt, oldest;
                    out_state(c, t_sub, &byt, &oldest);
                    int64_t quiet = c->last_done >= 0 ? t_sub - c->last_done : -1;
                    cell = cell_of(c->size[i], byt, oldest, quiet, c->P.T_ns);
                    q = (float)c->tab[cell];
                    unc = pred ? (c->tab[cell] < THETA) : (c->tab[cell] >= THETA);
                }
                flag = unc;
                if (g_action == 3) {
                    if (g_gate == 1 || g_gate == 2) target = (unc || pred) ? 2 : 0;
                    else target = (pred != unc) ? 2 : 0;
                } else if (g_action == 4) {
                    target = pred ? 1 : (unc ? 2 : 0);
                } else if (unc) {
                    if (g_action == 0) target = 2;
                    else if (g_action == 1) target = pred ? 0 : 1;
                    else target = pred ? 0 : 2;
                }
            }
        }
        c->qlen[i] = (int32_t)(c->adm + 1);
        if (target == 0 || target == 2) {
            c->adm++;
            out_slot = out_add(c, t_sub, c->size[i]);
        }
        pthread_mutex_unlock(&c->mu);
        c->route[i] = (int8_t)target;
        c->pred[i] = (int8_t)pred;
        c->flag[i] = (int8_t)flag;
        c->q[i] = q;
        c->logit[i] = lg_out;
        if (target == 0 || target == 2) W->own = submit(W, me, i, target == 0 ? 0 : 2, out_slot, is_read ? cell : -1);
        if (target == 1 || target == 2) W->oth = submit(W, other, i, target == 1 ? 1 : 3, -1, -1);
        int tmo = is_read && (g_policy == 4 || g_timeout) && target != 2;
        int64_t deadline = t_sub + c->P.H_ns;
        int dup = 0;
        while (!(target == 2 ? W->answered : W->primary_done)) {
            struct io_event ev[NSLOT];
            struct timespec ts, *tp = NULL;
            if (tmo && !dup && !W->answered) {
                int64_t left = deadline - (mono_ns() - g_start);
                if (left <= 0) {
                    submit(W, target == 1 ? me : other, i, 4, -1, -1);
                    dup = 1;
                    c->tcopy[i] = 1;
                    continue;
                }
                ts.tv_sec = left / 1000000000LL;
                ts.tv_nsec = left % 1000000000LL;
                tp = &ts;
            }
            int r = io_getevents_(W->ctx, 1, NSLOT, ev, tp);
            int64_t t_done = mono_ns() - g_start;
            if (r < 0) {
                if (errno == EINTR) continue;
                perror("io_getevents");
                exit(1);
            }
            for (int k = 0; k < r; k++) reap(W, &ev[k], t_done);
        }
    }
    W->req = -1;                                            // drain the copies still in flight
    for (;;) {
        int busy = 0;
        for (int k = 0; k < NSLOT; k++) busy += W->slot[k].busy;
        if (!busy) break;
        struct io_event ev;
        if (io_getevents_(W->ctx, 1, 1, &ev, NULL) == 1) reap(W, &ev, mono_ns() - g_start);
    }
    return NULL;
}

static void *xread(FILE *f, size_t sz, size_t n) {
    void *p = malloc(sz * (n ? n : 1));
    if (!p || fread(p, sz, n, f) != n) { fprintf(stderr, "short read\n"); exit(1); }
    return p;
}

static void load_trace(client_t *c, const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) { perror(path); exit(1); }
    if (fread(&c->n, 8, 1, f) != 1) exit(1);
    c->ts_ms = xread(f, 8, c->n);
    c->off = xread(f, 8, c->n);
    c->size = xread(f, 4, c->n);
    c->type = xread(f, 1, c->n);
    fclose(f);
    c->max_size = 4096;
    for (int64_t i = 0; i < c->n; i++) if (c->size[i] > c->max_size) c->max_size = c->size[i];
    c->max_size = (c->max_size + 4095) / 4096 * 4096;
    if (posix_memalign(&c->buf, 4096, (size_t)c->max_size)) { perror("memalign"); exit(1); }
    memset(c->buf, 0x5a, (size_t)c->max_size);
    int64_t n = c->n;
    c->sub_ns = calloc(n, 8); c->lat_ns = malloc(n * 8); c->own_lat_ns = malloc(n * 8); c->qlen = calloc(n, 4);
    for (int64_t i = 0; i < n; i++) { c->lat_ns[i] = -1; c->own_lat_ns[i] = -1; }
    c->route = calloc(n, 1); c->ans_other = calloc(n, 1); c->tcopy = calloc(n, 1); c->pred = calloc(n, 1);
    c->flag = calloc(n, 1); c->q = calloc(n, 4); c->logit = calloc(n, 4);
    c->xs = calloc((n + 63) / 64 * L0, 8);
    c->dseq = malloc((n ? n : 1) * 8); c->cseq = malloc((n ? n : 1) * 8);
    for (int64_t i = 0; i < n; i++) { c->dseq[i] = -1; c->cseq[i] = -1; }
    // touch every array now, so that no page fault of ours lands inside the measurement
    memset(c->sub_ns, 0, n * 8); memset(c->qlen, 0, n * 4); memset(c->route, 0, n); memset(c->ans_other, 0, n);
    memset(c->tcopy, 0, n); memset(c->pred, 0, n); memset(c->flag, 0, n); memset(c->q, 0, n * 4);
    memset(c->logit, 0, n * 4); memset(c->xs, 0, (n + 63) / 64 * L0 * 8);
}

static void load_params(client_t *c, const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) { perror(path); exit(1); }
    char magic[8];
    if (fread(magic, 1, 8, f) != 8 || memcmp(magic, "PRPARAM1", 8)) { fprintf(stderr, "%s: bad magic\n", path); exit(1); }
    params_t *P = &c->P;
    size_t ok = 0, want = 0;
#define RD(ptr, sz, cnt) do { ok += fread(ptr, sz, cnt, f); want += (cnt); } while (0)
    RD(P->xmin, 8, L0); RD(P->xrng, 8, L0); RD(P->W1, 8, L0 * L1); RD(P->b1, 8, L1); RD(P->W2, 8, L1 * L2);
    RD(P->b2, 8, L2); RD(P->W3, 8, L2); RD(&P->b3, 8, 1);
    RD(P->mu, 8, L0); RD(P->pinv, 8, L0 * L0); RD(&P->thr_e, 8, 1); RD(&P->thr_m, 8, 1);
    RD(&P->T_ns, 8, 1); RD(&P->H_ns, 8, 1);
#undef RD
    if (ok != want || fgetc(f) != EOF) { fprintf(stderr, "%s: %zu of %zu values, or trailing bytes\n", path, ok, want); exit(1); }
    fclose(f);
}

static void write_out(client_t *c, const char *path) {
    FILE *f = fopen(path, "wb");
    if (!f) { perror(path); exit(1); }
    fwrite("PRLOG001", 1, 8, f);
    fwrite(&c->n, 8, 1, f);
    int64_t n = c->n;
    for (int64_t i = 0; i < n; i++) if (c->lat_ns[i] < 0) c->sub_ns[i] = -1;
    fwrite(c->sub_ns, 8, n, f); fwrite(c->lat_ns, 8, n, f); fwrite(c->own_lat_ns, 8, n, f);
    fwrite(c->route, 1, n, f); fwrite(c->ans_other, 1, n, f); fwrite(c->tcopy, 1, n, f); fwrite(c->pred, 1, n, f);
    fwrite(c->flag, 1, n, f); fwrite(c->q, 4, n, f); fwrite(c->logit, 4, n, f); fwrite(c->qlen, 4, n, f);
    int64_t ns = (n + 63) / 64;
    fwrite(&ns, 8, 1, f);
    fwrite(c->xs, 8, ns * L0, f);
    fwrite(c->dseq, 8, n, f); fwrite(c->cseq, 8, n, f);
    if (fclose(f) != 0) { perror(path); exit(1); }
}

int main(int argc, char **argv) {
    const char *cpath = NULL, *spec[2][5] = {{0}};
    double dur_s = -1;
    int64_t dur_ns_arg = -1;      // --duration-ns: exact, so that the caller can tell which I/Os are issued
    for (int a = 1; a < argc; a++) {
        if (!strcmp(argv[a], "--policy") && a + 1 < argc) g_policy = atoi(argv[++a]);
        else if (!strcmp(argv[a], "--gate") && a + 1 < argc) g_gate = atoi(argv[++a]);
        else if (!strcmp(argv[a], "--action") && a + 1 < argc) g_action = atoi(argv[++a]);
        else if (!strcmp(argv[a], "--timeout") && a + 1 < argc) g_timeout = atoi(argv[++a]);
        else if (!strcmp(argv[a], "--cut") && a + 1 < argc) g_cut = atoi(argv[++a]);
        else if (!strcmp(argv[a], "--duration") && a + 1 < argc) dur_s = atof(argv[++a]);
        else if (!strcmp(argv[a], "--duration-ns") && a + 1 < argc) { dur_ns_arg = atoll(argv[++a]); dur_s = dur_ns_arg / 1e9; }
        else if (!strcmp(argv[a], "--counters") && a + 1 < argc) cpath = argv[++a];
        else if ((!strcmp(argv[a], "--map0") || !strcmp(argv[a], "--map1")) && a + 1 < argc) {
            int k = argv[a][5] - '0';
            const char *mp = argv[++a];
            FILE *f = fopen(mp, "rb");
            if (!f) { perror(mp); return 1; }
            fseek(f, 0, SEEK_END);
            g_nmap[k] = ftell(f) / 8;
            fseek(f, 0, SEEK_SET);
            g_map[k] = xread(f, 8, g_nmap[k]);
            fclose(f);
            if (g_nmap[k] < MAXPC) { fprintf(stderr, "%s: too few chunks\n", mp); return 1; }
        }
        else if ((!strcmp(argv[a], "--c0") || !strcmp(argv[a], "--c1")) && a + 4 < argc) {
            int k = argv[a][3] - '0';
            for (int j = 0; j < 4; j++) spec[k][j] = argv[a + 1 + j];
            a += 4;
            if (a + 1 < argc && strncmp(argv[a + 1], "--", 2)) spec[k][4] = argv[++a];
        } else { fprintf(stderr, "bad argument %s\n", argv[a]); return 2; }
    }
    if (g_policy < 0 || dur_s <= 0 || !cpath || !spec[0][0] || !spec[1][0]) {
        fprintf(stderr, "usage: pair_replayer --policy P [--gate G] [--action A] [--timeout 0|1] --duration S "
                        "--counters PATH --c0 PARAMS TRACE FILE OUT [LABELS] --c1 PARAMS TRACE FILE OUT [LABELS]\n");
        return 2;
    }
    if (g_policy == 3 && g_gate == 4 && (!spec[0][4] || !spec[1][4])) { fprintf(stderr, "the oracle needs labels\n"); return 2; }
    g_dur_ns = dur_ns_arg >= 0 ? dur_ns_arg : (int64_t)(dur_s * 1e9);
    int cfd = open(cpath, O_RDWR | O_CREAT | O_TRUNC, 0644);
    if (cfd < 0 || ftruncate(cfd, sizeof(counters_t)) != 0) { perror(cpath); return 1; }
    K = mmap(NULL, sizeof(counters_t), PROT_READ | PROT_WRITE, MAP_SHARED, cfd, 0);
    if (K == MAP_FAILED) { perror("mmap"); return 1; }
    memset(K, 0, sizeof(counters_t));
    memcpy(K->magic, "PRCOUNT1", 8);
    for (int k = 0; k < 2; k++) {
        client_t *c = &C[k];
        memset(c, 0, sizeof(*c));
        c->id = k;
        load_params(c, spec[k][0]);
        load_trace(c, spec[k][1]);
        c->label = NULL;
        if (spec[k][4]) {
            FILE *f = fopen(spec[k][4], "rb");
            if (!f) { perror(spec[k][4]); return 1; }
            c->label = xread(f, 1, c->n);
            fclose(f);
        }
        g_fd[k] = open(spec[k][2], O_RDWR | O_DIRECT | O_NOATIME);
        if (g_fd[k] < 0) g_fd[k] = open(spec[k][2], O_RDWR | O_DIRECT);
        if (g_fd[k] < 0) { perror(spec[k][2]); return 1; }
        struct stat st;
        fstat(g_fd[k], &st);
        g_fsize[k] = st.st_size;
        pthread_mutex_init(&c->mu, NULL);
        c->last_done = -1;
        for (int j = 0; j < NCELLS; j++) c->tab[j] = 0.5;
    }
    static worker_t W[2][NW];
    pthread_t tid[2][NW];
    g_start = mono_ns();
    K->start_mono_ns = g_start;
    for (int k = 0; k < 2; k++)
        for (int w = 0; w < NW; w++) {
            W[k][w].c = &C[k];
            W[k][w].w = w;
            pthread_create(&tid[k][w], NULL, worker, &W[k][w]);
        }
    for (int k = 0; k < 2; k++)
        for (int w = 0; w < NW; w++) pthread_join(tid[k][w], NULL);
    int64_t t_end = mono_ns();
    K->end_mono_ns = t_end;
    double run_s = (t_end - g_start) / 1e9;
    for (int k = 0; k < 2; k++) {
        write_out(&C[k], spec[k][3]);
        close(g_fd[k]);
    }
    struct rusage ru;
    getrusage(RUSAGE_SELF, &ru);
    fprintf(stderr, "{\"run_s\": %.3f, \"start_mono_ns\": %" PRId64 ", \"end_mono_ns\": %" PRId64
                    ", \"late\": [%" PRId64 ", %" PRId64 "], \"issued\": [%" PRId64 ", %" PRId64 "]"
                    ", \"cpu_user_s\": %.3f, \"cpu_sys_s\": %.3f}\n",
            run_s, g_start, t_end, (int64_t)C[0].late, (int64_t)C[1].late, K->issued[0], K->issued[1],
            ru.ru_utime.tv_sec + ru.ru_utime.tv_usec / 1e6, ru.ru_stime.tv_sec + ru.ru_stime.tv_usec / 1e6);
    return 0;
}
