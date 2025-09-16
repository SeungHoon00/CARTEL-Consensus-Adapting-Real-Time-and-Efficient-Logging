import math, random, csv, os, statistics
from dataclasses import dataclass
from typing import Optional
try:
    import numpy as np
except Exception:
    np = None

@dataclass
class Params:
    N: int = 200
    ET_min: float = 0.15
    ET_max: float = 0.30
    pending: float = 0.02
    delta_min: float = 0.001
    delta_max: float = 0.008
    trials: int = 200
    resample_positions: bool = False
    seed: Optional[int] = 42
    outdir: str = "out_election"
    export_ts: bool = False  # reserved

class CETSim:
    def __init__(self, p: Params):
        assert 0.0 <= p.delta_min <= p.delta_max
        assert 2*p.delta_max <= p.pending < p.ET_min - 2*p.delta_max, \
            "Need 2*δ_max <= pending < ET_min - 2*δ_max"
        self.p = p
        self.rng = random.Random(p.seed) if p.seed is not None else random.Random()
        os.makedirs(p.outdir, exist_ok=True)
        self.delta = self._make_delta()  # fixed by default

    def _make_delta(self):
        N, dmin, dmax = self.p.N, self.p.delta_min, self.p.delta_max
        if np is not None:
            # random positions on unit square; normalized Euclidean distance -> [dmin, dmax]
            xs = np.array([self.rng.random() for _ in range(N)])
            ys = np.array([self.rng.random() for _ in range(N)])
            dx = xs.reshape(-1,1) - xs.reshape(1,-1)
            dy = ys.reshape(-1,1) - ys.reshape(1,-1)
            d = np.hypot(dx, dy)
            maxd = float(np.max(d))
            if maxd < 1e-12: maxd = 1.0
            dn = d / maxd
            delta = (dmin + dn * (dmax - dmin)).astype(float)
            np.fill_diagonal(delta, 0.0)
            return delta
        else:
            # pure Python fallback
            pos = [(self.rng.random(), self.rng.random()) for _ in range(N)]
            maxd = 0.0
            for i in range(N):
                for j in range(i+1,N):
                    dx = pos[i][0]-pos[j][0]; dy = pos[i][1]-pos[j][1]
                    dij = math.hypot(dx,dy); maxd = max(maxd, dij)
            maxd = max(maxd, 1e-9)
            delta = [[0.0]*N for _ in range(N)]
            for r in range(N):
                for c in range(N):
                    if r==c: delta[r][c]=0.0
                    else:
                        dx = pos[r][0]-pos[c][0]; dy = pos[r][1]-pos[c][1]
                        dn = math.hypot(dx,dy)/maxd
                        delta[r][c] = dmin + dn*(dmax-dmin)
            return delta

    def _sample_ET(self):
        if np is not None:
            return self.p.ET_min + (self.p.ET_max - self.p.ET_min)*np.random.random(self.p.N)
        else:
            return [self.rng.uniform(self.p.ET_min, self.p.ET_max) for _ in range(self.p.N)]

    def run(self):
        p = self.p
        rows = []
        for trial in range(1, p.trials+1):
            if p.resample_positions:
                self.delta = self._make_delta()

            ET = self._sample_ET()  # shape (N,)
            if np is not None:
                ET = np.array(ET, dtype=float)

            # Identify leader-by-ET (smallest ET; break ties by smallest id)
            if np is not None:
                min_et = float(ET.min())
                candidates = np.where(ET == min_et)[0]
                leader_id = int(candidates.min())
            else:
                min_et = min(ET)
                leader_id = min(i for i,e in enumerate(ET) if e==min_et)

            # Matrix of VR arrival times at followers: A[f, c] = ET[c] + δ_{f,c}
            if np is not None:
                A = ET.reshape(1,-1) + self.delta  # shape (N,N); row=follower, col=candidate
                # self-VR arrival is irrelevant; keep it but it won't be used (we don't include self in visible set)
                # First VR arrival per follower
                t_first = A.min(axis=1)  # shape (N,)
                vote_due = t_first + p.pending
                # Visible set mask
                visible = A <= vote_due.reshape(-1,1)
                # Mask ET by visibility to pick candidate with minimal ET among those seen
                ET_row = np.broadcast_to(ET, (p.N, p.N))
                masked_ET = np.where(visible, ET_row, np.inf)
                chosen = masked_ET.argmin(axis=1)  # one candidate per follower
                # Vote decision time at follower
                t_vote_local = vote_due  # shape (N,)
                # Determine which followers vote for leader_id
                voters_mask = (chosen == leader_id)
                voter_indices = np.where(voters_mask)[0]
                # Self-vote time (at candidate)
                t_self_vote = float(ET[leader_id])
                # Vote reply arrival times at leader: t_vote_local + δ_{leader, follower}
                if voter_indices.size > 0:
                    reply_delays = self.delta[leader_id, voter_indices]  # shape (M,)
                    t_reply_arr = t_vote_local[voter_indices] + reply_delays
                    # Collect all arrival times at leader (including self vote)
                    arrivals = np.concatenate(([t_self_vote], t_reply_arr))
                else:
                    arrivals = np.array([t_self_vote])
                # Sort and find time when >N/2 votes reached
                arrivals.sort()
                K = (p.N // 2) + 1
                if arrivals.size >= K:
                    t_leader_knows = float(arrivals[K-1])
                else:
                    # Shouldn't happen under constraint; fall back to last arrival
                    t_leader_knows = float(arrivals[-1])
                # Also compute local-majority (ignoring reply delays): kth vote due among voters
                if voter_indices.size >= (K-1):  # excluding self vote, need K-1 followers
                    t_local_majority = float(np.sort(t_vote_local[voter_indices])[K-1-1]) if K-1>0 else t_self_vote
                else:
                    t_local_majority = float('nan')

                # AE/heartbeat announcement timings
                # first follower to receive AE
                delta_to_followers = self.delta[:, leader_id]  # vector (N,): recv=follower, sender=leader
                delta_to_followers = np.delete(delta_to_followers, leader_id)  # exclude leader itself
                t_first_ae = t_leader_knows + float(delta_to_followers.min()) if delta_to_followers.size else t_leader_knows
                t_all_ae = t_leader_knows + float(delta_to_followers.max()) if delta_to_followers.size else t_leader_knows
                # Sanity flags
                # split means some follower did not see leader's VR within pending (should not under constraint)
                split = bool((voters_mask.sum() <= p.N//2))

            else:
                N = p.N
                # Pure Python branch
                # Build arrival matrix
                A = [[ET[c] + (self.delta[r][c]) for c in range(N)] for r in range(N)]
                t_first = [min(row) for row in A]
                vote_due = [tf + p.pending for tf in t_first]
                chosen = []
                for f in range(N):
                    visible_cands = [c for c in range(N) if A[f][c] <= vote_due[f] and c != f]
                    if not visible_cands:
                        # should not happen: at least first VR should be visible
                        visible_cands = [min(range(N), key=lambda c: A[f][c])]
                    # pick minimal ET among visible; tie by id
                    best_c = min(visible_cands, key=lambda c: (ET[c], c))
                    chosen.append(best_c)
                t_vote_local = vote_due[:]
                voters = [f for f in range(N) if chosen[f]==leader_id]
                t_self_vote = ET[leader_id]
                arrivals = [t_self_vote] + [t_vote_local[f] + self.delta[leader_id][f] for f in voters]
                arrivals.sort()
                K = (N//2)+1
                t_leader_knows = arrivals[K-1] if len(arrivals)>=K else arrivals[-1]
                t_local_majority = sorted([t_vote_local[f] for f in voters])[K-1-1] if (len(voters) >= (K-1) and K-1>0) else float('nan')
                deltas_to_followers = [self.delta[f][leader_id] for f in range(N) if f!=leader_id]
                t_first_ae = t_leader_knows + (min(deltas_to_followers) if deltas_to_followers else 0.0)
                t_all_ae = t_leader_knows + (max(deltas_to_followers) if deltas_to_followers else 0.0)
                split = (len(voters) <= N//2)

            rows.append({
                "trial": trial,
                "leader_id": leader_id,
                "ET_leader": float(min_et),
                "t_leader_knows": t_leader_knows,
                "t_first_AE": t_first_ae,
                "t_all_AE": t_all_ae,
                "t_local_majority_lowerbound": t_local_majority,
                "split_vote_detected": split
            })

        # write samples
        samples_path = os.path.join(p.outdir, "election_samples.csv")
        with open(samples_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)

        # aggregates
        def stats(vec):
            vec_sorted = sorted(vec)
            n = len(vec_sorted)
            def quantile(q):
                if n==0: return float('nan')
                pos = q*(n-1); lo = int(math.floor(pos)); hi=int(math.ceil(pos))
                if lo==hi: return vec_sorted[lo]
                return vec_sorted[lo] + (vec_sorted[hi]-vec_sorted[lo])*(pos-lo)
            mean = sum(vec_sorted)/n if n else float('nan')
            return {
                "mean": mean,
                "p50": quantile(0.5),
                "p90": quantile(0.9),
                "p95": quantile(0.95),
                "p99": quantile(0.99),
                "max": vec_sorted[-1] if n else float('nan'),
                "min": vec_sorted[0] if n else float('nan'),
            }

        t_leader_knows_vals = [r["t_leader_knows"] for r in rows]
        t_first_AE_vals = [r["t_first_AE"] for r in rows]
        t_all_AE_vals = [r["t_all_AE"] for r in rows]
        t_local_majority_vals = [r["t_local_majority_lowerbound"] for r in rows if not math.isnan(r["t_local_majority_lowerbound"])]
        splits = sum(1 for r in rows if r["split_vote_detected"])

        agg = {
            "N": p.N,
            "trials": p.trials,
            "delta_min": p.delta_min,
            "delta_max": p.delta_max,
            "ET_min": p.ET_min,
            "ET_max": p.ET_max,
            "pending": p.pending,
            "resample_positions": p.resample_positions,
            "seed": p.seed,
            "split_votes": splits,
            "t_leader_knows_mean": stats(t_leader_knows_vals)["mean"],
            "t_leader_knows_p95": stats(t_leader_knows_vals)["p95"],
            "t_first_AE_mean": stats(t_first_AE_vals)["mean"],
            "t_first_AE_p95": stats(t_first_AE_vals)["p95"],
            "t_all_AE_mean": stats(t_all_AE_vals)["mean"],
            "t_all_AE_p95": stats(t_all_AE_vals)["p95"],
            "t_local_majority_mean": stats(t_local_majority_vals)["mean"] if t_local_majority_vals else float('nan'),
            "t_local_majority_p95": stats(t_local_majority_vals)["p95"] if t_local_majority_vals else float('nan'),
        }

        # write cluster summary
        summary_path = os.path.join(p.outdir, "cluster_summary.csv")
        with open(summary_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(list(agg.keys()))
            w.writerow([agg[k] for k in agg.keys()])

        print(f"Wrote: {samples_path}")
        print(f"Wrote: {summary_path}")


def main():
    import argparse
    ap = argparse.ArgumentParser(description="CARTEL Election-Time Simulator (CETSim v1)")
    ap.add_argument("--nodes", type=int, default=200)
    ap.add_argument("--et-min", type=float, default=0.15)
    ap.add_argument("--et-max", type=float, default=0.30)
    ap.add_argument("--pending", type=float, default=0.02)
    ap.add_argument("--delta-min", type=float, default=0.001)
    ap.add_argument("--delta-max", type=float, default=0.008)
    ap.add_argument("--trials", type=int, default=200)
    ap.add_argument("--resample-positions", action="store_true", default=False)
    ap.add_argument("--seed", type=lambda s: None if s=="None" else int(s), default=42)
    ap.add_argument("--outdir", type=str, default="out_election")
    args = ap.parse_args()

    params = Params(N=args.nodes, ET_min=args.et_min, ET_max=args.et_max, pending=args.pending,
                    delta_min=args.delta_min, delta_max=args.delta_max, trials=args.trials,
                    resample_positions=args.resample_positions, seed=args.seed, outdir=args.outdir)
    sim = CETSim(params)
    sim.run()

if __name__ == "__main__":
    main()
