// Once-a-second record of everything else that uses a machine's CPU and drives while a measurement runs.
//
// Each tick (every --interval seconds, default 1, on CLOCK_MONOTONIC) it writes one JSON line to --out with what
// changed since the previous tick:
//   cpu      the machine's CPU time by state (USER_HZ ticks summed over every CPU: user nice system idle iowait
//            irq softirq steal), and the mean clock of its cores (MHz, /proc/cpuinfo)
//   cls      CPU ticks of processes by who they are: "ours" (the process tree under --root, the measurement
//            itself), "mine" (other processes of our user, e.g. the job runner), "users" (other people's),
//            "system" (system users' daemons) and "kernel" (kernel threads)
//   top      the five processes outside "ours" that used the most CPU in the tick: [pid, uid, comm, ticks]
//   disks    per whole drive (nvme*n*, sd*): reads, sectors read, writes, sectors written, in flight, ms busy
//   our_io   our own completed I/O per test drive from the replayer's shared counters (--counters):
//            [reads, read bytes, writes, write bytes]; with the drives' own counts this gives the I/O of
//            everything else, tick by tick
//   temp     drive temperatures (milli-degrees C, hwmon), load average, user sessions (utmp) and seated sessions
// Lines are written to --out (put it on a memory file system, e.g. /dev/shm, so that the record itself touches
// no drive), flushed each tick.  It stops on SIGTERM or SIGINT, after a last tick.
// Usage: quietrec --out PATH --root PID [--counters PATH] [--interval S]
#define _GNU_SOURCE
#include <ctype.h>
#include <dirent.h>
#include <fcntl.h>
#include <inttypes.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>
#include <utmpx.h>

typedef struct {
    char magic[8];
    int64_t start_mono_ns, end_mono_ns;
    int64_t done[2][2][2];
    int64_t sub[2][2][2];
    int64_t issued[2];
} counters_t;

#define MAXP 65536
typedef struct { int pid; uint64_t start; uint64_t ticks; } pent_t;
static pent_t prev[MAXP], cur[MAXP];
static int nprev, ncur;
static volatile sig_atomic_t stop_flag;
static void on_sig(int s) { (void)s; stop_flag = 1; }

static int64_t mono_ns(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (int64_t)t.tv_sec * 1000000000LL + t.tv_nsec;
}

static int read_file(const char *path, char *buf, size_t cap) {
    int fd = open(path, O_RDONLY);
    if (fd < 0) return -1;
    ssize_t n = read(fd, buf, cap - 1);
    close(fd);
    if (n < 0) return -1;
    buf[n] = 0;
    return (int)n;
}

typedef struct { int pid, ppid, uid, kthread; uint64_t start, ticks; char comm[32]; } proc_t;
static proc_t P[MAXP];
static int np;

static void scan_procs(void) {
    np = 0;
    DIR *d = opendir("/proc");
    if (!d) return;
    struct dirent *e;
    char path[300], buf[1024];
    while ((e = readdir(d)) && np < MAXP) {
        if (!isdigit((unsigned char)e->d_name[0])) continue;
        snprintf(path, sizeof path, "/proc/%s/stat", e->d_name);
        if (read_file(path, buf, sizeof buf) <= 0) continue;
        char *lp = strchr(buf, '('), *rp = strrchr(buf, ')');
        if (!lp || !rp) continue;
        proc_t *p = &P[np];
        p->pid = atoi(buf);
        size_t cl = (size_t)(rp - lp - 1);
        if (cl > sizeof p->comm - 1) cl = sizeof p->comm - 1;
        memcpy(p->comm, lp + 1, cl);
        p->comm[cl] = 0;
        for (char *q = p->comm; *q; q++) if (*q == '"' || *q == '\\' || (unsigned char)*q < 32) *q = '_';
        // fields after ')': state ppid pgrp session tty tpgid flags minflt cminflt majflt cmajflt utime stime
        // cutime cstime priority nice threads itrealvalue starttime
        char st;
        int ppid;
        unsigned long flags, utime, stime;
        unsigned long long starttime;
        int got = sscanf(rp + 2, "%c %d %*d %*d %*d %*d %lu %*u %*u %*u %*u %lu %lu %*d %*d %*d %*d %*d %*d %llu",
                         &st, &ppid, &flags, &utime, &stime, &starttime);
        if (got != 6) continue;
        p->ppid = ppid;
        p->kthread = (flags & 0x00200000) != 0 || ppid == 2 || p->pid == 2;
        p->ticks = utime + stime;
        p->start = starttime;
        struct stat sb;
        snprintf(path, sizeof path, "/proc/%s", e->d_name);
        p->uid = stat(path, &sb) == 0 ? (int)sb.st_uid : -1;
        np++;
    }
    closedir(d);
}

static uint64_t prev_ticks(int pid, uint64_t start, int *found) {
    // prev[] is kept sorted by pid
    int lo = 0, hi = nprev - 1;
    while (lo <= hi) {
        int m = (lo + hi) / 2;
        if (prev[m].pid == pid) {
            if (prev[m].start == start) { *found = 1; return prev[m].ticks; }
            break;
        }
        if (prev[m].pid < pid) lo = m + 1; else hi = m - 1;
    }
    *found = 0;
    return 0;
}

static int cmp_pent(const void *a, const void *b) { return ((const pent_t *)a)->pid - ((const pent_t *)b)->pid; }

static int cmp_proc(const void *a, const void *b) { return ((const proc_t *)a)->pid - ((const proc_t *)b)->pid; }

static int find_proc(int pid) {   // P[] sorted by pid
    int lo = 0, hi = np - 1;
    while (lo <= hi) {
        int m = (lo + hi) / 2;
        if (P[m].pid == pid) return m;
        if (P[m].pid < pid) lo = m + 1; else hi = m - 1;
    }
    return -1;
}

static int is_ours(int idx, int root) {
    int pid = P[idx].pid;
    for (int step = 0; step < 64 && pid > 1; step++) {   // the parent chain, up to the root
        if (pid == root) return 1;
        int k = find_proc(pid);
        if (k < 0) return 0;
        pid = P[k].ppid;
    }
    return 0;
}

int main(int argc, char **argv) {
    const char *out = NULL, *cpath = NULL;
    int root = -1;
    double interval = 1.0;
    for (int a = 1; a < argc; a++) {
        if (!strcmp(argv[a], "--out") && a + 1 < argc) out = argv[++a];
        else if (!strcmp(argv[a], "--root") && a + 1 < argc) root = atoi(argv[++a]);
        else if (!strcmp(argv[a], "--counters") && a + 1 < argc) cpath = argv[++a];
        else if (!strcmp(argv[a], "--interval") && a + 1 < argc) interval = atof(argv[++a]);
        else { fprintf(stderr, "bad argument %s\n", argv[a]); return 2; }
    }
    if (!out || root <= 0) { fprintf(stderr, "usage: quietrec --out PATH --root PID [--counters PATH]\n"); return 2; }
    FILE *fo = fopen(out, "w");
    if (!fo) { perror(out); return 1; }
    signal(SIGTERM, on_sig);
    signal(SIGINT, on_sig);
    uid_t me = getuid();
    counters_t *K = NULL;
    static char buf[1 << 20];
    unsigned long long pcpu[8] = {0}, ccpu[8];
    long long pdisk[64][6], cdisk[64][6];
    char dname[64][32];
    int nd = 0;
    int64_t pio[2][4] = {{0}}, cio[2][4];
    int64_t t0 = mono_ns(), step = (int64_t)(interval * 1e9);
    for (long tick = 0;; tick++) {
        // ---- sample
        int64_t t = mono_ns();
        if (read_file("/proc/stat", buf, sizeof buf) > 0)
            sscanf(buf, "cpu %llu %llu %llu %llu %llu %llu %llu %llu", &ccpu[0], &ccpu[1], &ccpu[2], &ccpu[3],
                   &ccpu[4], &ccpu[5], &ccpu[6], &ccpu[7]);
        int ndc = 0;
        if (read_file("/proc/diskstats", buf, sizeof buf) > 0) {
            char *line = strtok(buf, "\n");
            while (line && ndc < 64) {
                char name[32];
                long long v[11];
                if (sscanf(line, "%*d %*d %31s %lld %*d %lld %*d %lld %*d %lld %*d %lld %lld", name, &v[0], &v[1],
                           &v[2], &v[3], &v[4], &v[5]) == 7) {
                    int whole = (!strncmp(name, "nvme", 4) && strchr(name, 'n') && !strchr(name + 4, 'p') &&
                                 strchr(name + 4, 'n')) || (!strncmp(name, "sd", 2) && !isdigit((unsigned char)name[strlen(name) - 1]));
                    if (whole) {
                        int k = -1;
                        for (int j = 0; j < nd; j++) if (!strcmp(dname[j], name)) { k = j; break; }
                        if (k < 0 && nd < 64) { k = nd++; strcpy(dname[k], name); memset(pdisk[k], 0, sizeof pdisk[k]); }
                        if (k >= 0) for (int j = 0; j < 6; j++) cdisk[k][j] = v[j];
                        ndc++;
                    }
                }
                line = strtok(NULL, "\n");
            }
        }
        if (cpath && !K) {
            int cfd = open(cpath, O_RDONLY);
            if (cfd >= 0) {
                struct stat sb;
                if (fstat(cfd, &sb) == 0 && sb.st_size >= (off_t)sizeof(counters_t)) {
                    void *m = mmap(NULL, sizeof(counters_t), PROT_READ, MAP_SHARED, cfd, 0);
                    if (m != MAP_FAILED) K = m;
                }
                close(cfd);
            }
        }
        for (int dv = 0; dv < 2; dv++) {
            cio[dv][0] = K ? __atomic_load_n(&K->done[dv][1][0], __ATOMIC_RELAXED) : 0;
            cio[dv][1] = K ? __atomic_load_n(&K->done[dv][1][1], __ATOMIC_RELAXED) : 0;
            cio[dv][2] = K ? __atomic_load_n(&K->done[dv][0][0], __ATOMIC_RELAXED) : 0;
            cio[dv][3] = K ? __atomic_load_n(&K->done[dv][0][1], __ATOMIC_RELAXED) : 0;
        }
        scan_procs();
        qsort(P, np, sizeof(proc_t), cmp_proc);
        double mhz = 0;
        int nmhz = 0;
        if (read_file("/proc/cpuinfo", buf, sizeof buf) > 0) {
            for (char *p = strstr(buf, "cpu MHz"); p; p = strstr(p + 7, "cpu MHz")) {
                char *c = strchr(p, ':');
                if (c) { mhz += atof(c + 1); nmhz++; }
            }
        }
        // ---- what changed
        uint64_t cls[5] = {0};   // ours, mine, users, system, kernel
        int topi[5] = {-1, -1, -1, -1, -1};
        uint64_t topt[5] = {0};
        ncur = 0;
        for (int k = 0; k < np; k++) {
            int found;
            uint64_t was = prev_ticks(P[k].pid, P[k].start, &found);
            uint64_t d = tick == 0 ? 0 : (found ? P[k].ticks - was : P[k].ticks);
            cur[ncur].pid = P[k].pid; cur[ncur].start = P[k].start; cur[ncur].ticks = P[k].ticks; ncur++;
            if (d == 0) continue;
            int c;
            if (P[k].kthread) c = 4;
            else if (is_ours(k, root)) c = 0;
            else if (P[k].uid == (int)me) c = 1;
            else if (P[k].uid >= 1000 && P[k].uid < 60000) c = 2;
            else c = 3;
            cls[c] += d;
            if (c != 0) {
                for (int s = 0; s < 5; s++) {
                    if (topi[s] < 0 || d > topt[s]) {
                        for (int r = 4; r > s; r--) { topi[r] = topi[r - 1]; topt[r] = topt[r - 1]; }
                        topi[s] = k; topt[s] = d;
                        break;
                    }
                }
            }
        }
        qsort(cur, ncur, sizeof(pent_t), cmp_pent);
        memcpy(prev, cur, ncur * sizeof(pent_t));
        nprev = ncur;
        int sessions = 0;
        setutxent();
        struct utmpx *u;
        while ((u = getutxent())) if (u->ut_type == USER_PROCESS) sessions++;
        endutxent();
        int seated = 0;
        DIR *sd = opendir("/run/systemd/sessions");
        if (sd) {
            struct dirent *e;
            while ((e = readdir(sd))) {
                if (e->d_name[0] == '.' || strchr(e->d_name, '.')) continue;
                char path[300];
                snprintf(path, sizeof path, "/run/systemd/sessions/%s", e->d_name);
                if (read_file(path, buf, sizeof buf) > 0 && strstr(buf, "\nSEAT=") && strstr(buf, "CLASS=user") &&
                    !strstr(buf, "STATE=closing")) seated++;
            }
            closedir(sd);
        }
        char lavg[64] = "";
        if (read_file("/proc/loadavg", buf, sizeof buf) > 0) { strncpy(lavg, buf, 63); char *n = strchr(lavg, ' '); if (n) { n = strchr(n + 1, ' '); if (n) { n = strchr(n + 1, ' '); if (n) *n = 0; } } }
        // ---- the line
        fprintf(fo, "{\"tick\": %ld, \"t_ns\": %" PRId64 ", \"cpu\": [", tick, t);
        for (int j = 0; j < 8; j++) fprintf(fo, "%s%llu", j ? ", " : "", tick ? ccpu[j] - pcpu[j] : 0ULL);
        fprintf(fo, "], \"mhz\": %.0f, \"cls\": [%" PRIu64 ", %" PRIu64 ", %" PRIu64 ", %" PRIu64 ", %" PRIu64 "], \"top\": [",
                nmhz ? mhz / nmhz : 0.0, cls[0], cls[1], cls[2], cls[3], cls[4]);
        int first = 1;
        for (int s = 0; s < 5; s++) if (topi[s] >= 0) {
            proc_t *p = &P[topi[s]];
            fprintf(fo, "%s[%d, %d, \"%s\", %" PRIu64 "]", first ? "" : ", ", p->pid, p->uid, p->comm, topt[s]);
            first = 0;
        }
        fprintf(fo, "], \"disks\": {");
        for (int k = 0; k < nd; k++) {
            fprintf(fo, "%s\"%s\": [", k ? ", " : "", dname[k]);
            for (int j = 0; j < 6; j++) fprintf(fo, "%s%lld", j ? ", " : "", tick ? (j == 4 ? cdisk[k][j] : cdisk[k][j] - pdisk[k][j]) : 0LL);
            fprintf(fo, "]");
        }
        fprintf(fo, "}, \"our_io\": [");
        for (int dv = 0; dv < 2; dv++) {
            fprintf(fo, "%s[", dv ? ", " : "");
            for (int j = 0; j < 4; j++) fprintf(fo, "%s%" PRId64, j ? ", " : "", tick ? cio[dv][j] - pio[dv][j] : 0);
            fprintf(fo, "]");
        }
        fprintf(fo, "], \"replay\": [%" PRId64 ", %" PRId64 "]", K ? K->start_mono_ns : 0, K ? K->end_mono_ns : 0);
        fprintf(fo, ", \"temp\": {");
        int ft = 1;
        for (int h = 0; h < 32; h++) {
            char p1[128], nm[64] = "", tv[32] = "";
            snprintf(p1, sizeof p1, "/sys/class/hwmon/hwmon%d/name", h);
            if (read_file(p1, nm, sizeof nm) <= 0) continue;
            nm[strcspn(nm, "\n")] = 0;
            if (strcmp(nm, "nvme")) continue;
            snprintf(p1, sizeof p1, "/sys/class/hwmon/hwmon%d/temp1_input", h);
            if (read_file(p1, tv, sizeof tv) <= 0) continue;
            char dev[256] = "";
            snprintf(p1, sizeof p1, "/sys/class/hwmon/hwmon%d/device", h);
            ssize_t rl = readlink(p1, dev, sizeof dev - 1);
            if (rl > 0) dev[rl] = 0;
            char *b = strrchr(dev, '/');
            fprintf(fo, "%s\"%s\": %ld", ft ? "" : ", ", b ? b + 1 : dev, atol(tv));
            ft = 0;
        }
        fprintf(fo, "}, \"load\": \"%s\", \"sessions\": %d, \"seated\": %d}\n", lavg, sessions, seated);
        fflush(fo);
        memcpy(pcpu, ccpu, sizeof pcpu);
        memcpy(pdisk, cdisk, sizeof pdisk);
        memcpy(pio, cio, sizeof pio);
        if (stop_flag) break;
        // ---- sleep to the next tick boundary
        int64_t next = t0 + (tick + 1) * step, now = mono_ns();
        while (now < next && !stop_flag) {
            int64_t left = next - now;
            struct timespec ts = {left / 1000000000LL, left % 1000000000LL};
            nanosleep(&ts, NULL);
            now = mono_ns();
        }
    }
    fclose(fo);
    return 0;
}
