import math, heapq, random, csv, os
from dataclasses import dataclass, field

try:
    import numpy as np
except Exception:
    np = None

EV_ELECT_START = "ELECT"
EV_VR_ARR = "VR_ARR"
EV_VOTE_DEC = "VOTE_DEC"
EV_LEADER_ELECTED = "LEAD"
EV_APP_ARR = "APP_ARR"
EV_HB_SEND = "HB_SEND"
EV_HB_ARR = "HB_ARR"
EV_FAIL = "FAIL"
EV_GEN = "GEN"  # still supported if --ae-on-gen is used

@dataclass
class Node:
    nid: int
    buffer_count: int = 0
    vote_due: float | None = None
    seen: dict = field(default_factory=dict)  # cand_id -> ET
    min_occ: int = 10**9
    max_occ: int = 0
    sum_occ: int = 0
    n_samples: int = 0

    def on_flush(self):
        self.buffer_count = 0

    def record(self):
        # record current buffer_count as a sampling point (we sample just-before-flush)
        self.min_occ = min(self.min_occ, self.buffer_count)
        self.max_occ = max(self.max_occ, self.buffer_count)
        self.sum_occ += self.buffer_count
        self.n_samples += 1

    def avg_occ(self):
        return self.sum_occ / self.n_samples if self.n_samples else 0.0

class CartelSimV8a:
    def __init__(self, N=60, T_phi=0.02, delta_min=0.001, delta_max=0.008,
                 ET_min=0.15, ET_max=0.30, pending=0.02, hb=0.05,
                 seed=None, duration=30.0, failures=1, fail_start=10.0, fail_end=20.0,
                 ae_on_gen=False, outdir="out_v8a", ts=False):
        assert 0.0 <= delta_min <= delta_max
        assert 2*delta_max <= pending < ET_min - 2*delta_max, "Need 2*δ_max <= pending < ET_min - 2*δ_max"
        assert 0.0 <= fail_start < fail_end <= duration
        self.N=N; self.T_phi=T_phi; self.delta_min=delta_min; self.delta_max=delta_max
        self.ET_min=ET_min; self.ET_max=ET_max; self.pending=pending; self.hb=hb
        self.duration=duration; self.failures=failures; self.fail_start=fail_start; self.fail_end=fail_end
        self.ae_on_gen=ae_on_gen; self.outdir=outdir; self.ts=ts; self.seed=seed

        self.rng = random.Random() if seed is None else random.Random(seed)

        # nodes & state
        self.nodes=[Node(i) for i in range(N)]
        self.time=0.0; self.term=0; self.leader=None
        self.votes={}; self.et_map={}
        self.events=[]; self.seq=0
        self.leader_set=set()
        os.makedirs(outdir, exist_ok=True)

        # positions -> pairwise delays delta_ij in [delta_min, delta_max]
        # vectorized for speed if numpy is available
        if np is not None:
            pos = self.rng.random
            xs = np.array([pos() for _ in range(N)])
            ys = np.array([pos() for _ in range(N)])
            dx = xs.reshape(-1,1) - xs.reshape(1,-1)
            dy = ys.reshape(-1,1) - ys.reshape(1,-1)
            d = np.hypot(dx, dy)
            maxd = float(np.max(d))
            if maxd < 1e-12: maxd = 1.0
            dn = d / maxd
            self.delta = (delta_min + dn * (delta_max - delta_min)).astype(float)
            np.fill_diagonal(self.delta, 0.0)
        else:
            # fallback (pure Python)
            pos = [(self.rng.random(), self.rng.random()) for _ in range(N)]
            maxd = 0.0
            for i in range(N):
                for j in range(i+1, N):
                    dx = pos[i][0]-pos[j][0]; dy = pos[i][1]-pos[j][1]
                    d = math.hypot(dx, dy)
                    if d > maxd: maxd = d
            maxd = max(maxd, 1e-9)
            self.delta = [[0.0]*N for _ in range(N)]
            for i in range(N):
                for j in range(N):
                    if i==j: self.delta[i][j]=0.0
                    else:
                        dx = pos[i][0]-pos[j][0]; dy = pos[i][1]-pos[j][1]
                        d = math.hypot(dx, dy)/maxd  # normalized 0..1
                        self.delta[i][j] = delta_min + d*(delta_max - delta_min)

        # per-sender periodic generation: phase phi_j
        if np is not None:
            self.phi = np.array([ self.rng.random()*T_phi for _ in range(N) ], dtype=float)
        else:
            self.phi = [ self.rng.random()*T_phi for _ in range(N) ]

        # last_seen_ij: number of releases from j that i has already "counted"
        if np is not None:
            self.last_seen = np.zeros((N,N), dtype=np.int64)
        else:
            self.last_seen = [ [0]*N for _ in range(N) ]

        # per-node timeseries for excel (optional; disabled by default to keep it light)
        self.ts_rows_nodes=[]

        # schedule generation markers only if AE-on-gen is enabled
        if self.ae_on_gen:
            for j in range(N):
                self.schedule(self._phi(j), EV_GEN, {"node": j, "k": 1})

        # randomized failure times
        self.fail_times = sorted(self.rng.uniform(self.fail_start, self.fail_end) for _ in range(self.failures))
        for ft in self.fail_times: self.schedule(ft, EV_FAIL, {})

        # start election
        self.start_election_window()

    # utilities
    def schedule(self, t, evtype, payload):
        if t > self.duration: return
        self.seq += 1
        heapq.heappush(self.events, (t, self.seq, evtype, payload))

    def _phi(self, j:int) -> float:
        return float(self.phi[j]) if np is not None else self.phi[j]

    def gen_count(self, j:int, t:float) -> int:
        """# of releases generated by sender j up to (and including) time t."""
        pj = self._phi(j)
        if t < pj - 1e-12:
            return 0
        return int(math.floor((t - pj) / self.T_phi)) + 1

    def absorb_from_all_senders(self, i:int, t:float):
        """At time t, node i absorbs all arrivals whose generation time <= t - delta_ij, for all j != i.
           Vectorized when numpy is available.
        """
        ni = self.nodes[i]
        if np is not None:
            # limit per sender
            limit = t - self.delta[i, :]
            # mask where a sender's first release has occurred wrt limit
            mask = limit >= self.phi
            cnts = np.where(mask, np.floor((limit - self.phi)/self.T_phi).astype(np.int64) + 1, 0)
            cnts[i] = 0  # ignore self
            # compute deltas vs last_seen
            diff = cnts - self.last_seen[i, :]
            diff[diff < 0] = 0
            gained = int(diff.sum())
            if gained > 0:
                ni.buffer_count += gained
                self.last_seen[i, :] = cnts
        else:
            gained = 0
            for j in range(self.N):
                if j == i: continue
                limit = t - (self.delta[i][j] if isinstance(self.delta[0], list) else self.delta[i][j])
                cnt = self.gen_count(j, limit)
                add = cnt - self.last_seen[i][j]
                if add > 0:
                    gained += add
                    self.last_seen[i][j] = cnt
            if gained > 0:
                ni.buffer_count += gained

    def flush_node(self, i:int, t:float):
        """Flush node i and synchronize last_seen to now (so arrivals aren't recounted)."""
        self.nodes[i].on_flush()
        if np is not None:
            limit = t - self.delta[i, :]
            mask = limit >= self.phi
            cnts = np.where(mask, np.floor((limit - self.phi)/self.T_phi).astype(np.int64) + 1, 0)
            cnts[i] = 0
            self.last_seen[i, :] = cnts
        else:
            for j in range(self.N):
                if j == i: continue
                self.last_seen[i][j] = self.gen_count(j, t - (self.delta[i][j] if isinstance(self.delta[0], list) else self.delta[i][j]))

    def delay_to_from(self, recv:int, sender:int) -> float:
        return float(self.delta[recv, sender]) if np is not None else self.delta[recv][sender]

    def start_election_window(self):
        self.term += 1
        self.leader = None
        self.votes.clear(); self.et_map.clear()
        t0 = max(self.time, 0.0)
        for n in range(self.N):
            ET = self.rng.uniform(self.ET_min, self.ET_max)
            self.et_map[n] = ET
            self.schedule(t0 + ET, EV_ELECT_START, {"term": self.term, "cand": n, "ET": ET})
        for n in self.nodes:
            n.seen.clear(); n.vote_due = None

    def start_heartbeats(self):
        self.schedule(self.time, EV_HB_SEND, {"term": self.term, "leader": self.leader})

    def maybe_record_ts(self):
        if not self.ts: return
        self.ts_rows_nodes.append([self.time] + [n.buffer_count for n in self.nodes])

    def broadcast_ae_from(self, leader_id:int):
        base = self.time
        for f in range(self.N):
            if f == leader_id: continue
            arr = base + self.delay_to_from(f, leader_id)
            self.schedule(arr, EV_APP_ARR, {"term": self.term, "follower": f, "leader": leader_id})

    def run(self):
        while self.events:
            t, _, ev, payload = heapq.heappop(self.events)
            if t > self.duration: break
            self.time = t

            if ev == EV_GEN:
                # Only relevant when --ae-on-gen is ON
                j = payload["node"]
                if self.leader is not None and self.ae_on_gen:
                    self.broadcast_ae_from(self.leader)
                self.schedule(self.time + self.T_phi, EV_GEN, {"node": j, "k": (payload.get("k",0)+1)})

            elif ev == EV_ELECT_START:
                if payload["term"] != self.term: continue
                cand = payload["cand"]; ET = payload["ET"]
                for f in range(self.N):
                    if f == cand: continue
                    arr = self.time + self.delay_to_from(f, cand)
                    self.schedule(arr, EV_VR_ARR, {"term": self.term, "cand": cand, "ET": ET, "follower": f})

            elif ev == EV_VR_ARR:
                if payload["term"] != self.term: continue
                f = payload["follower"]; c = payload["cand"]; ET = payload["ET"]
                # NO FLUSH on VoteRequest arrival (v8a change)
                nf = self.nodes[f]
                nf.seen[c] = ET
                if nf.vote_due is None:
                    nf.vote_due = self.time + self.pending
                    self.schedule(nf.vote_due, EV_VOTE_DEC, {"term": self.term, "follower": f})

            elif ev == EV_VOTE_DEC:
                if payload["term"] != self.term: continue
                f = payload["follower"]; nf = self.nodes[f]
                if not nf.seen:
                    nf.vote_due=None; continue
                best_c, best_ET = None, None
                for c,ET in nf.seen.items():
                    if best_ET is None or ET < best_ET or (abs(ET-best_ET)<1e-12 and c<best_c):
                        best_c, best_ET = c, ET
                self.votes[best_c] = self.votes.get(best_c, 0) + 1
                if self.votes[best_c] > self.N//2 and self.leader is None:
                    self.leader = best_c
                    self.leader_set.add(best_c)
                    # leadership announcement time offset ~ random fraction of min incoming delay
                    if np is not None:
                        mind = float(np.min(self.delta[:, best_c] + 1e-12))
                    else:
                        mind = min(self.delta[s][best_c] for s in range(self.N) if s!=best_c)
                    lead_time = self.time + self.rng.random()*mind
                    self.schedule(lead_time, EV_LEADER_ELECTED, {"term": self.term, "leader": best_c})
                nf.vote_due = None

            elif ev == EV_LEADER_ELECTED:
                if payload["term"] != self.term: continue
                self.leader = payload["leader"]
                # announce (AE) -> followers will ABSORB then FLUSH on arrival
                self.broadcast_ae_from(self.leader)
                # start heartbeats
                self.start_heartbeats()

            elif ev == EV_APP_ARR:
                if payload["term"] != self.term: continue
                f = payload["follower"]
                # ABSORB first, record the peak, then FLUSH
                self.absorb_from_all_senders(f, self.time)
                self.nodes[f].record()
                self.maybe_record_ts()
                self.flush_node(f, self.time)

            elif ev == EV_HB_SEND:
                if payload["term"] != self.term or self.leader is None: continue
                base = self.time
                for f in range(self.N):
                    if f == self.leader: continue
                    arr = base + self.delay_to_from(f, self.leader)
                    self.schedule(arr, EV_HB_ARR, {"term": self.term, "follower": f})
                self.schedule(self.time + self.hb, EV_HB_SEND, {"term": self.term, "leader": self.leader})

            elif ev == EV_HB_ARR:
                if payload["term"] != self.term: continue
                f = payload["follower"]
                # ABSORB first, record the peak, then FLUSH
                self.absorb_from_all_senders(f, self.time)
                self.nodes[f].record()
                self.maybe_record_ts()
                self.flush_node(f, self.time)

            elif ev == EV_FAIL:
                if self.leader is not None:
                    # Conceptual flush for leader, then drop leader and re-elect
                    self.flush_node(self.leader, self.time)
                    self.leader = None
                    self.start_election_window()

        # --- Outputs ---
        # 1) Per-node summary
        with open(f"{self.outdir}/buffer_summary.csv","w",newline="") as f:
            w=csv.writer(f); w.writerow(["node","is_leader_ever","min_occ","avg_occ","max_occ","samples"])
            for n in self.nodes:
                avg = (n.sum_occ/n.n_samples) if n.n_samples else 0.0
                w.writerow([n.nid, (n.nid in self.leader_set), n.min_occ if n.n_samples else 0, avg, n.max_occ, n.n_samples])

        # 2) Cluster-level followers-only aggregate (like the table we discussed)
        followers = [n for n in self.nodes if n.nid not in self.leader_set]
        if followers:
            mean_of_avg = sum((n.sum_occ/n.n_samples if n.n_samples else 0.0) for n in followers) / len(followers)
            max_of_max = max(n.max_occ for n in followers)
            mean_samples = sum(n.n_samples for n in followers)/len(followers)
        else:
            mean_of_avg = 0.0; max_of_max = 0.0; mean_samples = 0.0
        expected_peak = (self.N - 1) * (self.hb / self.T_phi) if self.T_phi > 0 else 0.0
        ratio = (mean_of_avg / expected_peak) if expected_peak > 0 else 0.0

        with open(f"{self.outdir}/cluster_summary.csv","w",newline="") as f:
            w=csv.writer(f); w.writerow([
                "N","followers_count","followers_mean_avg_occ","followers_max_of_max_occ","followers_mean_samples",
                "hb","T_phi","expected_peak","ratio_measured_to_expected","ae_on_gen",
                "ET_min","ET_max","pending","delta_min","delta_max","duration","failures","fail_start","fail_end","seed"
            ])
            w.writerow([
                self.N, len(followers), round(mean_of_avg,6), max_of_max, round(mean_samples,6),
                self.hb, self.T_phi, round(expected_peak,6), round(ratio,6), bool(self.ae_on_gen),
                self.ET_min, self.ET_max, self.pending, self.delta_min, self.delta_max, self.duration, self.failures, self.fail_start, self.fail_end, self.seed
            ])

        # 3) Optional Excel export
        if self.ts:
            try:
                import pandas as _pd
                xlsx_path = os.path.join(self.outdir, "buffer_per_node_v8a.xlsx")
                cols = ["time"] + [f"node_{i}" for i in range(self.N)]
                df_ts = _pd.DataFrame(self.ts_rows_nodes, columns=cols)
                df_sum = _pd.DataFrame([{
                    "node": n.nid, "is_leader_ever": (n.nid in self.leader_set),
                    "min_occ": (n.min_occ if n.n_samples else 0),
                    "avg_occ": (n.sum_occ/n.n_samples if n.n_samples else 0.0),
                    "max_occ": n.max_occ, "samples": n.n_samples
                } for n in self.nodes])
                with _pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
                    df_ts.to_excel(writer, sheet_name="pre_flush_timeseries", index=False)
                    df_sum.sort_values(["avg_occ","max_occ"], ascending=[False, False]).to_excel(writer, sheet_name="per_node_summary", index=False)
            except Exception as e:
                print("Excel export skipped (install pandas + openpyxl):", e)

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--nodes", type=int, default=60)
    ap.add_argument("--tphi", type=float, default=0.02)
    ap.add_argument("--delta-min", type=float, default=0.001)
    ap.add_argument("--delta-max", type=float, default=0.008)
    ap.add_argument("--et-min", type=float, default=0.15)
    ap.add_argument("--et-max", type=float, default=0.30)
    ap.add_argument("--pending", type=float, default=0.02)
    ap.add_argument("--hb", type=float, default=0.05)
    ap.add_argument("--seed", type=lambda s: None if s=="None" else int(s), default=None)
    ap.add_argument("--duration", type=float, default=30.0)
    ap.add_argument("--failures", type=int, default=1)
    ap.add_argument("--fail-start", type=float, default=10.0)
    ap.add_argument("--fail-end", type=float, default=20.0)
    ap.add_argument("--ae-on-gen", action="store_true", default=False)
    ap.add_argument("--outdir", type=str, default="out_v8a")
    ap.add_argument("--ts", action="store_true", default=False, help="Export timeseries to Excel")
    args = ap.parse_args()

    sim = CartelSimV8a(N=args.nodes, T_phi=args.tphi, delta_min=args.delta_min, delta_max=args.delta_max,
                      ET_min=args.et_min, ET_max=args.et_max, pending=args.pending, hb=args.hb,
                      seed=args.seed, duration=args.duration, failures=args.failures,
                      fail_start=args.fail_start, fail_end=args.fail_end,
                      ae_on_gen=args.ae_on_gen, outdir=args.outdir, ts=args.ts)
    sim.run()
    print(f"Wrote: {args.outdir}/buffer_summary.csv")
    print(f"Wrote: {args.outdir}/cluster_summary.csv")
    if args.ts:
        print(f"Excel: {os.path.join(args.outdir, 'buffer_per_node_v8a.xlsx')}")

if __name__ == "__main__":
    main()
