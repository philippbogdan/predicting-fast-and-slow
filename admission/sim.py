"""Trace-driven simulator of Heimdall's two-client, two-SSD client-level replay.

Clients (io_replayer.c): each client replays its trace with 8 worker threads
that take I/Os in trace order, sleep until the I/O's timestamp, issue it
synchronously and wait for completion; I/Os not issued by the end of the
run (``duration`` seconds) are dropped.  Writes always go to the client's own
(primary) device; a policy routes each read: own device, other device, or
both (hedge).

Devices: each SSD is a processor-sharing server with P parallel lanes: with
n outstanding I/Os each progresses at rate min(1, P/n).  An I/O's service
"work" is taken from the baseline replay: for the device's own client, the
same I/O's baseline latency integrated against the baseline concurrency
(so the baseline is reproduced exactly); for a read arriving from the other
client, the work of the device's own read nearest in the own client's
progress (GC is driven by the own client's writes), rescaled for size.

Client-side model features (queue length, history of the last three
completions) are maintained exactly as io_replayer.c does, from the I/Os the
client admitted to its own device.

Policies (``policy`` codes):
  0 baseline, 1 random (49% of reads to the other device, as random_algo()),
  2 Heimdall network, 3 Heimdall network + gate (gated reads hedged),
  4 artefact hedging (duplicate to the other device after a timeout; the
  worker still waits for its own I/O; logs the first finisher).
Gate codes (policy 3): 1 Euclidean distance, 2 Mahalanobis distance,
  3 an earlier decision-time evidence rule (admission.evidence), 4 oracle (distrusts exactly
  the decisions that disagree with the ground-truth label), 5 outcome-feedback
  rule (recent same-size-class reads on the primary, or an outstanding read
  already older than T; threshold g_theta), 6 combined evidence, 7 situation
  tracker (admission.tracker: per client, a running slow-rate per (size class,
  outstanding bytes, oldest-age, quiet) cell of its primary device, learned only from
  the client's reads that completed there (a cancelled hedge copy teaches
  nothing; trk_censored: see below); distrust "fast" if q >= g_theta,
  "slow" if q < g_theta).
Gate action: 0 hedge the distrusted read to both devices (first response
  wins, the other copy is cancelled), 1 flip the model's decision, 2 mixed:
  a distrusted "slow" stays home, a distrusted "fast" is hedged, 3 hedge
  every read whose final call is "slow" (the model's call, flipped where
  distrusted; distance gates: uncertain or model-slow) and keep the rest home,
  4 keep the model's reroutes and hedge a distrusted "fast".
"""
from __future__ import annotations

import numba
import numpy as np

from .tracker import ALPHA as TRK_ALPHA, N_CELLS as TRK_CELLS, cell_of as trk_cell_of

N_THREADS = 8


@numba.njit(cache=True)
def work_from_baseline(ts_us, lat, P):
    """Service work of each I/O under P-lane processor sharing given the
    baseline's own concurrency.  ts_us/lat in submission order."""
    n = ts_us.shape[0]
    fin = ts_us + lat
    # event sweep: concurrency is piecewise constant between submits/completions
    ev_t = np.empty(2 * n)
    ev_d = np.empty(2 * n, np.int64)
    for i in range(n):
        ev_t[2 * i] = ts_us[i]
        ev_d[2 * i] = 1
        ev_t[2 * i + 1] = fin[i]
        ev_d[2 * i + 1] = -1
    o = np.argsort(ev_t, kind="mergesort")
    # cumulative "rate integral" R(t) = int min(1, P/n(t)) dt at event times
    R = np.empty(2 * n)
    cur = 0
    acc = 0.0
    last = ev_t[o[0]]
    for e in range(2 * n):
        t = ev_t[o[e]]
        if cur > 0:
            r = 1.0 if cur <= P else P / cur
            acc += r * (t - last)
        else:
            acc += (t - last)
        R[e] = acc
        last = t
        cur += ev_d[o[e]]
    # R at arbitrary t: interpolate through sorted event times
    ts_sorted = ev_t[o]
    work = np.empty(n)
    for i in range(n):
        a = np.searchsorted(ts_sorted, ts_us[i], side="right") - 1
        b = np.searchsorted(ts_sorted, fin[i], side="right") - 1
        work[i] = R[b] - R[a]
        if work[i] <= 0:
            work[i] = lat[i]
    return work


@numba.njit(cache=True)
def find_stalls(ts_us, lat, G):
    """Device stalls in a baseline replay: intervals with I/Os outstanding but
    no completion for longer than G us.  Returns (start, end) arrays; a stall
    ends one median completion gap before the completion that ends it (the
    service of that I/O after the device resumes)."""
    n = ts_us.shape[0]
    fin = np.sort(ts_us + lat)
    sub = np.sort(ts_us)
    gaps = np.empty(n - 1)
    busy = np.zeros(n - 1, np.bool_)
    for i in range(n - 1):
        gaps[i] = fin[i + 1] - fin[i]
        busy[i] = (np.searchsorted(sub, fin[i], side="right") - (i + 1)) > 0
    med = np.median(gaps[busy]) if busy.sum() > 0 else 0.0
    m = 0
    for i in range(n - 1):
        if busy[i] and gaps[i] > G:
            m += 1
    a = np.empty(m)
    b = np.empty(m)
    m = 0
    for i in range(n - 1):
        if busy[i] and gaps[i] > G:
            a[m] = fin[i]
            b[m] = fin[i + 1] - med
            m += 1
    return a, b


@numba.njit(cache=True)
def work_from_baseline_stalls(ts_us, lat, P, st_a, st_b):
    """As work_from_baseline, but the device does no work during stalls."""
    n = ts_us.shape[0]
    fin = ts_us + lat
    ns = st_a.shape[0]
    ev_t = np.empty(2 * n + 2 * ns)
    ev_d = np.empty(2 * n + 2 * ns, np.int64)
    for i in range(n):
        ev_t[2 * i] = ts_us[i]
        ev_d[2 * i] = 1
        ev_t[2 * i + 1] = fin[i]
        ev_d[2 * i + 1] = -1
    for j in range(ns):
        ev_t[2 * n + 2 * j] = st_a[j]
        ev_d[2 * n + 2 * j] = 100        # stall begins
        ev_t[2 * n + 2 * j + 1] = st_b[j]
        ev_d[2 * n + 2 * j + 1] = -100   # stall ends
    o = np.argsort(ev_t, kind="mergesort")
    m = ev_t.shape[0]
    R = np.empty(m)
    cur = 0
    stalled = 0
    acc = 0.0
    last = ev_t[o[0]]
    for e in range(m):
        t = ev_t[o[e]]
        if stalled == 0:
            if cur > 0:
                r = 1.0 if cur <= P else P / cur
                acc += r * (t - last)
            else:
                acc += (t - last)
        R[e] = acc
        last = t
        d = ev_d[o[e]]
        if d == 100:
            stalled += 1
        elif d == -100:
            stalled -= 1
        else:
            cur += d
    ts_sorted = ev_t[o]
    work = np.empty(n)
    for i in range(n):
        a = np.searchsorted(ts_sorted, ts_us[i], side="right") - 1
        b = np.searchsorted(ts_sorted, fin[i], side="right") - 1
        work[i] = R[b] - R[a]
        if work[i] <= 0:
            work[i] = min(lat[i], 1.0)
    return work


@numba.njit(cache=True, inline="always")
def _foreign_work(template, own, beta):
    if beta <= 0.0:
        return template
    if beta >= 1.0:
        return own
    return np.exp((1.0 - beta) * np.log(max(template, 1e-3)) + beta * np.log(max(own, 1e-3)))


@numba.njit(cache=True)
def _mlp_logit(x, xmin, xrng, W1, b1, W2, b2, W3, b3, z_out):
    for j in range(12):
        z_out[j] = (x[j] - xmin[j]) / xrng[j]
    h1 = np.empty(W1.shape[1])
    for j in range(W1.shape[1]):
        s = b1[j]
        for i in range(12):
            s += z_out[i] * W1[i, j]
        h1[j] = s if s > 0 else 0.0
    h2 = np.empty(W2.shape[1])
    for j in range(W2.shape[1]):
        s = b2[j]
        for i in range(W1.shape[1]):
            s += h1[i] * W2[i, j]
        h2[j] = s if s > 0 else 0.0
    s = b3[0]
    for i in range(W2.shape[1]):
        s += h2[i] * W3[i, 0]
    return s


@numba.njit(cache=True)
def simulate(arr0, size0, io0, work0, rd_idx0, rd_work0, rd_size0,
             arr1, size1, io1, work1, rd_idx1, rd_work1, rd_size1,
             P0, P1, alpha_bytes, duration_us, policy, seed,
             m_xmin, m_xrng, m_W1, m_b1, m_W2, m_b2, m_W3, m_b3,
             gate, g_thr, g_mu, g_Pinv, g_t_slow, g_t_fast,
             hedge_t0, hedge_t1, ylab0, ylab1, gate_action, g_theta,
             st_k0, st_d0, st_k1, st_d1, foreign_beta, tmpl_W, fmult0=1.0, fmult1=1.0, timeout_hedge=False,
             timeout_awake=False, trk_censored=False):
    """Run one policy.  Per-client inputs: arrival times (us), sizes, io types,
    own-device work per I/O, and the own-device read templates (index, work,
    size) used to price reads arriving from the other client.  Model weights
    are stacked [client] arrays.  Returns per-I/O issue time, latency, route
    (0 own, 1 other, 2 hedged; -1 not issued) and the gate flag per client.
    st_k*/st_d*: device stalls of each client's own device (empty: none):
    stall j begins when the client issues its own I/O st_k[j] and lasts
    st_d[j] us, during which that device serves nothing (own or foreign).
    foreign_beta: a read served by the other device costs
    template^(1-beta) * own_work^beta (0: the receiving device's state alone,
    1: the read's own cost on its home device).
    tmpl_W: the template is drawn uniformly from the receiving device's last
    tmpl_W own reads before its client's current position (1: the nearest one,
    which over-samples reads its client is stuck waiting on).
    fmult0/fmult1: multiplier on the cost of client 0's / client 1's reads
    served by the other device (fitted on the training window of the artefact's random-routing replay,
    scripts/sim_validate.py --calibrate).
    timeout_hedge: arm the artefact's hedging timer under any policy (policy 4
    always does): a read sent to one device only and still not answered after
    hedge_t gets a duplicate on the device not serving it; a read already sent
    to both gets none.
    timeout_awake: optional, off in every reported result: the timer's duplicate is sent only if the
    device it would go to has completed some I/O within the last hedge_t.
    trk_censored: optional, off in every reported result: the tracker (gate 7) also counts as slow a
    read whose own copy is cancelled because the other device answered first, if it was already older
    than T (the rule's line 3 updates on completions only)."""
    arrs = (arr0, arr1)
    sizes = (size0, size1)
    ios = (io0, io1)
    works = (work0, work1)
    rd_idx = (rd_idx0, rd_idx1)
    rd_work = (rd_work0, rd_work1)
    rd_size = (rd_size0, rd_size1)
    Ps = (P0, P1)
    hedge_t = (hedge_t0, hedge_t1)
    ylab = (ylab0, ylab1)
    nio = (arr0.shape[0], arr1.shape[0])
    issue = (np.full(nio[0], -1.0), np.full(nio[1], -1.0))
    latency = (np.full(nio[0], -1.0), np.full(nio[1], -1.0))
    route = (np.full(nio[0], -1, np.int8), np.full(nio[1], -1, np.int8))
    flag = (np.zeros(nio[0], np.int8), np.zeros(nio[1], np.int8))
    np.random.seed(seed)

    MAXJ = 4096
    # request table (ring of slots)
    j_dev = np.zeros(MAXJ, np.int64)
    j_rem = np.zeros(MAXJ)
    j_cli = np.zeros(MAXJ, np.int64)
    j_idx = np.zeros(MAXJ, np.int64)
    j_kind = np.zeros(MAXJ, np.int64)   # 0 primary-only, 1 rerouted, 2 hedge-own, 3 hedge-other, 4 art-hedge dup
    j_partner = np.full(MAXJ, -1, np.int64)
    j_live = np.zeros(MAXJ, np.bool_)
    free_slots = np.arange(MAXJ - 1, -1, -1)
    n_free = MAXJ
    act = np.zeros((2, MAXJ), np.int64)
    n_act = np.zeros(2, np.int64)

    # client state
    next_k = np.zeros(2, np.int64)
    # scheduled issues: FIFO of (time, index) per client, at most N_THREADS outstanding
    q_t = np.zeros((2, N_THREADS + 1))
    q_k = np.zeros((2, N_THREADS + 1), np.int64)
    q_head = np.zeros(2, np.int64)
    q_len = np.zeros(2, np.int64)
    # Heimdall client-side features: admitted in-flight and completion history
    adm_n = np.zeros(2, np.int64)
    hist_q = np.zeros((2, 3), np.int64)
    hist_l = np.zeros((2, 3), np.int64)
    hist_t = np.zeros((2, 3), np.int64)
    hist_n = np.zeros(2, np.int64)
    last_done = np.full(2, -1.0)
    dev_last = np.full(2, -1.0)   # last completion of any request on each device (timeout_awake)
    qlen_at = (np.zeros(nio[0], np.int64), np.zeros(nio[1], np.int64))
    # artefact hedging: pending duplicate timers per client (time, index)
    ht_t = np.zeros((2, 1 << 16))
    ht_k = np.zeros((2, 1 << 16), np.int64)
    ht_head = np.zeros(2, np.int64)
    ht_tail = np.zeros(2, np.int64)
    done_flag = (np.zeros(nio[0], np.bool_), np.zeros(nio[1], np.bool_))

    NCL, KCL = 32, 8
    cls_ring = np.full((2, NCL, KCL), -1.0)
    trk_tab = np.full((2, TRK_CELLS), 0.5)
    trk_at = (np.zeros(nio[0], np.int64), np.zeros(nio[1], np.int64))
    nb_iss = np.full((2, 8), -1.0)
    nb_lat = np.full((2, 8), -1.0)
    nb_id = np.full((2, 8), -1, np.int64)
    nb_pos = np.zeros(2, np.int64)
    x = np.zeros(12)
    z = np.zeros(12)
    now = 0.0
    st_k = (st_k0, st_k1)
    st_d = (st_d0, st_d1)
    st_ptr = np.zeros(2, np.int64)
    stall_end = np.zeros(2)

    # threads take the first I/Os at t=0
    for c in range(2):
        for _ in range(N_THREADS):
            k = next_k[c]
            if k < nio[c]:
                p = (q_head[c] + q_len[c]) % (N_THREADS + 1)
                q_t[c, p] = arrs[c][k]
                q_k[c, p] = k
                q_len[c] += 1
                next_k[c] += 1

    while True:
        # next issue event
        t_iss = 1e300
        c_iss = -1
        for c in range(2):
            if q_len[c] > 0:
                tt = q_t[c, q_head[c]]
                if tt < t_iss:
                    t_iss = tt
                    c_iss = c
        # next duplicate-timer event (artefact hedging)
        t_ht = 1e300
        c_ht = -1
        for c in range(2):
            if ht_head[c] < ht_tail[c]:
                tt = ht_t[c, ht_head[c] & 0xFFFF]
                if tt < t_ht:
                    t_ht = tt
                    c_ht = c
        # next completion
        t_cmp = 1e300
        j_cmp = -1
        for d in range(2):
            na = n_act[d]
            if na == 0:
                continue
            rate = 1.0 if na <= Ps[d] else Ps[d] / na
            t_go = stall_end[d] if stall_end[d] > now else now
            for a in range(na):
                j = act[d, a]
                tt = t_go + j_rem[j] / rate
                if tt < t_cmp:
                    t_cmp = tt
                    j_cmp = j
        t_next = min(t_iss, t_ht, t_cmp)
        if t_next >= 1e299:
            break
        # advance processor sharing
        dt = t_next - now
        if dt > 0:
            for d in range(2):
                na = n_act[d]
                if na == 0:
                    continue
                rate = 1.0 if na <= Ps[d] else Ps[d] / na
                dd = t_next - (stall_end[d] if stall_end[d] > now else now)
                if dd <= 0:
                    continue
                for a in range(na):
                    j = act[d, a]
                    j_rem[j] -= dd * rate
        now = t_next

        if t_cmp <= t_iss and t_cmp <= t_ht:
            # ---- completion of request j_cmp
            j = j_cmp
            d = j_dev[j]
            dev_last[d] = now
            c = j_cli[j]
            k = j_idx[j]
            kind = j_kind[j]
            # remove from device
            for a in range(n_act[d]):
                if act[d, a] == j:
                    n_act[d] -= 1
                    act[d, a] = act[d, n_act[d]]
                    break
            j_live[j] = False
            free_slots[n_free] = j
            n_free += 1
            freed_thread = False
            if kind == 4:
                # artefact hedge duplicate finished: log if the original is still pending
                if not done_flag[c][k]:
                    latency[c][k] = now - issue[c][k]
                    done_flag[c][k] = True
                    route[c][k] = 2
            else:
                if kind == 2 or kind == 3:
                    pj = j_partner[j]
                    if pj >= 0 and j_live[pj]:
                        # cancel the loser
                        dd = j_dev[pj]
                        for a in range(n_act[dd]):
                            if act[dd, a] == pj:
                                n_act[dd] -= 1
                                act[dd, a] = act[dd, n_act[dd]]
                                break
                        j_live[pj] = False
                        free_slots[n_free] = pj
                        n_free += 1
                        if kind == 3:
                            # own copy cancelled: it leaves the admitted set without history
                            adm_n[c] -= 1
                            if gate == 7 and trk_censored and ios[c][k] == 1 and now - issue[c][k] > g_t_slow[c]:
                                # censored, but already known to be slow
                                trk_tab[c, trk_at[c][k]] += TRK_ALPHA * (1.0 - trk_tab[c, trk_at[c][k]])
                if not done_flag[c][k]:
                    latency[c][k] = now - issue[c][k]
                    done_flag[c][k] = True
                if kind == 0 or kind == 2:
                    # admitted completion -> client history (io_replayer.c)
                    adm_n[c] -= 1
                    lt = int(now - issue[c][k])
                    for h in range(2, 0, -1):
                        hist_q[c, h] = hist_q[c, h - 1]
                        hist_l[c, h] = hist_l[c, h - 1]
                        hist_t[c, h] = hist_t[c, h - 1]
                    if ios[c][k] == 1:
                        if gate == 7:
                            sl = 1.0 if now - issue[c][k] > g_t_slow[c] else 0.0
                            trk_tab[c, trk_at[c][k]] += TRK_ALPHA * (sl - trk_tab[c, trk_at[c][k]])
                        for h in range(8):
                            if nb_id[c, h] == k:
                                nb_lat[c, h] = now - issue[c][k]
                        cl = min(NCL - 1, max(0, int(np.log2(max(sizes[c][k], 512.0)))))
                        for h in range(KCL - 1, 0, -1):
                            cls_ring[c, cl, h] = cls_ring[c, cl, h - 1]
                        cls_ring[c, cl, 0] = now - issue[c][k]
                    hist_q[c, 0] = qlen_at[c][k]
                    hist_l[c, 0] = lt
                    hist_t[c, 0] = sizes[c][k] // (lt + 1)
                    if hist_n[c] < 3:
                        hist_n[c] += 1
                    last_done[c] = now
                freed_thread = True
            if freed_thread:
                kk = next_k[c]
                if kk < nio[c] and arrs[c][kk] <= duration_us:
                    p = (q_head[c] + q_len[c]) % (N_THREADS + 1)
                    q_t[c, p] = max(now, arrs[c][kk])
                    q_k[c, p] = kk
                    q_len[c] += 1
                    next_k[c] += 1
            continue

        if t_ht < t_iss:
            # ---- artefact hedging timer fires
            c = c_ht
            k = ht_k[c, ht_head[c] & 0xFFFF]
            ht_head[c] += 1
            o = c if route[c][k] == 1 else 1 - c
            if timeout_awake and (dev_last[o] < 0 or now - dev_last[o] > hedge_t[c]):
                continue
            if not done_flag[c][k] and route[c][k] != 2:
                # the duplicate goes to the device not serving the read (its own device if a gate moved it)
                j = free_slots[n_free - 1]
                n_free -= 1
                j_partner[j] = -1
                j_live[j] = True
                j_dev[j] = o
                j_cli[j] = c
                j_idx[j] = k
                j_kind[j] = 4
                if o == c:
                    j_rem[j] = works[c][k]
                else:
                    ti = rd_idx[o].shape[0]
                    pos = np.searchsorted(rd_idx[o], next_k[o]) - 1 - (np.random.randint(0, tmpl_W) if tmpl_W > 1 else 0) - (np.random.randint(0, tmpl_W) if tmpl_W > 1 else 0)
                    if pos < 0:
                        pos = 0
                    if pos >= ti:
                        pos = ti - 1
                    j_rem[j] = _foreign_work(rd_work[o][pos] * (alpha_bytes + sizes[c][k]) / (alpha_bytes + rd_size[o][pos]),
                                             works[c][k], foreign_beta) * (fmult0 if c == 0 else fmult1)
                act[o, n_act[o]] = j
                n_act[o] += 1
            continue

        # ---- issue event
        c = c_iss
        k = q_k[c, q_head[c]]
        q_head[c] = (q_head[c] + 1) % (N_THREADS + 1)
        q_len[c] -= 1
        # no cut-off at the horizon: every I/O that arrived before it is issued and completes (the replay
        # drains), so every policy is measured on the same requests however far its workers lag
        issue[c][k] = now
        while st_ptr[c] < st_k[c].shape[0] and st_k[c][st_ptr[c]] <= k:
            # the own device stalls here, as it did at this point of the own workload in the baseline
            stall_end[c] = (stall_end[c] if stall_end[c] > now else now) + st_d[c][st_ptr[c]]
            st_ptr[c] += 1
        o = 1 - c
        target = 0  # 0 own, 1 other, 2 both
        if ios[c][k] == 1:
            if policy == 1:
                if np.random.randint(0, 100) > 50:
                    target = 1
            elif policy == 2 or policy == 3:
                x[0] = 1.0
                x[1] = sizes[c][k]
                x[2] = adm_n[c] + 1
                for h in range(3):
                    if h < hist_n[c]:
                        x[3 + h] = hist_q[c, h]
                        x[6 + h] = hist_l[c, h]
                        x[9 + h] = hist_t[c, h]
                    else:
                        x[3 + h] = 0.0
                        x[6 + h] = 0.0
                        x[9 + h] = 0.0
                lg = _mlp_logit(x, m_xmin[c], m_xrng[c], m_W1[c], m_b1[c], m_W2[c], m_b2[c], m_W3[c], m_b3[c], z)
                pred_slow = lg >= 0
                target = 1 if pred_slow else 0
                if policy == 3:
                    unc = False
                    if gate == 1 or gate == 2:
                        acc = 0.0
                        if gate == 1:
                            for i in range(12):
                                acc += (z[i] - g_mu[c, i]) ** 2
                        else:
                            for i in range(12):
                                for i2 in range(12):
                                    acc += (z[i] - g_mu[c, i]) * g_Pinv[c, i, i2] * (z[i2] - g_mu[c, i2])
                        unc = np.sqrt(max(acc, 0.0)) > g_thr[c]
                    elif gate == 3:
                        oldest = 0.0
                        # oldest age over the client's admitted in-flight I/Os
                        for a in range(n_act[c]):
                            jj = act[c, a]
                            if j_cli[jj] == c and (j_kind[jj] == 0 or j_kind[jj] == 2):
                                age = now - issue[c][j_idx[jj]]
                                if age > oldest:
                                    oldest = age
                        since = now - last_done[c] if last_done[c] >= 0 else 1e9
                        if not pred_slow:
                            unc = oldest > g_t_slow[c]
                        else:
                            unc = (oldest < g_t_fast[c]) and (since < g_t_fast[c])
                    elif gate == 5:
                        # evidence: recent same-size-class reads on the primary slower than T,
                        # or an outstanding read on the primary already older than T
                        cl = min(NCL - 1, max(0, int(np.log2(max(sizes[c][k], 512.0)))))
                        cnt = 0
                        sl = 0
                        for h in range(KCL):
                            if cls_ring[c, cl, h] >= 0:
                                cnt += 1
                                if cls_ring[c, cl, h] > g_t_slow[c]:
                                    sl += 1
                        q = sl / cnt if cnt > 0 else 0.5
                        for a in range(n_act[c]):
                            jj = act[c, a]
                            if j_cli[jj] == c and (j_kind[jj] == 0 or j_kind[jj] == 2) and ios[c][j_idx[jj]] == 1:
                                if now - issue[c][j_idx[jj]] > g_t_slow[c]:
                                    q = 1.0
                        if pred_slow:
                            unc = q <= 1.0 - g_theta
                        else:
                            unc = q >= g_theta
                    elif gate == 6:
                        # combined: size-class outcome share and neighbourhood share, averaged;
                        # an outstanding read already older than T forces "slow"
                        cl = min(NCL - 1, max(0, int(np.log2(max(sizes[c][k], 512.0)))))
                        cnt = 0
                        sl = 0
                        for h in range(KCL):
                            if cls_ring[c, cl, h] >= 0:
                                cnt += 1
                                if cls_ring[c, cl, h] > g_t_slow[c]:
                                    sl += 1
                        qc = sl / cnt if cnt > 0 else 0.5
                        cnb = 0
                        snb = 0
                        for h in range(8):
                            if nb_iss[c, h] >= 0:
                                cnb += 1
                                if nb_lat[c, h] >= 0:
                                    if nb_lat[c, h] > g_t_slow[c]:
                                        snb += 1
                                elif now - nb_iss[c, h] > g_t_slow[c]:
                                    snb += 1
                        qn = snb / cnb if cnb > 0 else qc
                        q = 0.5 * (qc + qn)
                        for a in range(n_act[c]):
                            jj = act[c, a]
                            if j_cli[jj] == c and (j_kind[jj] == 0 or j_kind[jj] == 2) and ios[c][j_idx[jj]] == 1:
                                if now - issue[c][j_idx[jj]] > g_t_slow[c]:
                                    q = 1.0
                        if pred_slow:
                            unc = q <= 1.0 - g_theta
                        else:
                            unc = q >= g_theta
                    elif gate == 7:
                        # situation tracker: the cell of this read on the primary device, from the
                        # client's own outstanding I/Os there (bytes, oldest age) and the time since
                        # its last completion there (quiet bit)
                        byt = 0.0
                        oldest = 0.0
                        for a in range(n_act[c]):
                            jj = act[c, a]
                            if j_cli[jj] == c and (j_kind[jj] == 0 or j_kind[jj] == 2):
                                byt += sizes[c][j_idx[jj]]
                                age = now - issue[c][j_idx[jj]]
                                if age > oldest:
                                    oldest = age
                        quiet = now - last_done[c] if last_done[c] >= 0 else -1.0
                        cc = trk_cell_of(sizes[c][k], byt, oldest, quiet, g_t_slow[c])
                        trk_at[c][k] = cc
                        q = trk_tab[c, cc]
                        if pred_slow:
                            unc = q < g_theta
                        else:
                            unc = q >= g_theta
                    elif gate == 4:
                        # oracle: distrust exactly the decisions that disagree with the label
                        if ylab[c][k] >= 0:
                            unc = pred_slow != (ylab[c][k] == 1)
                    if unc:
                        flag[c][k] = 1
                    if gate_action == 3:
                        # hedge whenever the final call is "slow": a direction-aware gate's distrust
                        # flips the model's call; a distance gate's "uncertain" is hedged
                        if gate == 1 or gate == 2:
                            target = 2 if (unc or pred_slow) else 0
                        else:
                            target = 2 if (pred_slow != unc) else 0
                    elif gate_action == 4:
                        # keep the model's reroutes; hedge a distrusted "fast" (distance: uncertain)
                        if pred_slow:
                            target = 1
                        else:
                            target = 2 if unc else 0
                    elif unc:
                        if gate_action == 0:
                            target = 2                       # hedge to both devices
                        elif gate_action == 1:
                            target = 0 if pred_slow else 1   # follow the evidence: flip
                        else:
                            target = 0 if pred_slow else 2   # mixed: keep home / hedge
            elif policy == 6 or policy == 7:
                # reference routing without the model (bounds what routing alone achieves): send the read
                # to the other device when its outstanding load is lower by more than g_theta
                # (6: outstanding bytes, host-visible; 7: an oracle's expected wait: remaining stall plus
                # remaining service work over the lanes, in us).  (4 is the artefact's timeout hedging.)
                l_own = 0.0
                l_oth = 0.0
                for a in range(n_act[c]):
                    jj = act[c, a]
                    l_own += sizes[j_cli[jj]][j_idx[jj]] if policy == 6 else j_rem[jj] / Ps[c]
                for a in range(n_act[o]):
                    jj = act[o, a]
                    l_oth += sizes[j_cli[jj]][j_idx[jj]] if policy == 6 else j_rem[jj] / Ps[o]
                if policy == 7:
                    l_own += max(stall_end[c] - now, 0.0)
                    l_oth += max(stall_end[o] - now, 0.0)
                target = 1 if l_own - l_oth > g_theta else 0
        route[c][k] = target
        own_j = -1
        if (target == 0 or target == 2) and ios[c][k] == 1:
            nb_iss[c, nb_pos[c]] = now
            nb_lat[c, nb_pos[c]] = -1.0
            nb_id[c, nb_pos[c]] = k
            nb_pos[c] = (nb_pos[c] + 1) % 8
        if target == 0 or target == 2:
            qlen_at[c][k] = adm_n[c] + 1
            adm_n[c] += 1
            j = free_slots[n_free - 1]
            n_free -= 1
            j_partner[j] = -1
            j_live[j] = True
            j_dev[j] = c
            j_cli[j] = c
            j_idx[j] = k
            j_kind[j] = 0 if target == 0 else 2
            j_rem[j] = works[c][k]
            act[c, n_act[c]] = j
            n_act[c] += 1
            own_j = j
        if target == 1 or target == 2:
            j = free_slots[n_free - 1]
            n_free -= 1
            j_partner[j] = -1
            j_live[j] = True
            j_dev[j] = o
            j_cli[j] = c
            j_idx[j] = k
            j_kind[j] = 1 if target == 1 else 3
            ti = rd_idx[o].shape[0]
            pos = np.searchsorted(rd_idx[o], next_k[o]) - 1 - (np.random.randint(0, tmpl_W) if tmpl_W > 1 else 0)
            if pos < 0:
                pos = 0
            if pos >= ti:
                pos = ti - 1
            j_rem[j] = _foreign_work(rd_work[o][pos] * (alpha_bytes + sizes[c][k]) / (alpha_bytes + rd_size[o][pos]),
                                     works[c][k], foreign_beta) * (fmult0 if c == 0 else fmult1)
            act[o, n_act[o]] = j
            n_act[o] += 1
            if target == 2:
                j_partner[j] = own_j
                j_partner[own_j] = j
        if (policy == 4 or timeout_hedge) and ios[c][k] == 1 and route[c][k] != 2:
            p = ht_tail[c] & 0xFFFF
            ht_t[c, p] = now + hedge_t[c]
            ht_k[c, p] = k
            ht_tail[c] += 1
    return issue[0], latency[0], route[0], flag[0], issue[1], latency[1], route[1], flag[1]
