#!/usr/bin/env python3
"""
X5 Tech Smart 3D Pallet Packing Solver
Adapted from Kaggle 'Packing Santa's Sleigh' maxrects solution.

Usage:
    python solve_x5.py <input.json> <output.json>
"""
from __future__ import annotations

import json
import sys
import time
from typing import Any, Dict, List

from validator import evaluate_solution

# ---------------------------------------------------------------------------
# Maxrects 2D packer (from Kaggle notebook)
# ---------------------------------------------------------------------------

def _ri(a, b):
    return not (a[0]+a[2]<=b[0] or b[0]+b[2]<=a[0] or a[1]+a[3]<=b[1] or b[1]+b[3]<=a[1])

def _co(o, i):
    return o[0]<=i[0] and o[1]<=i[1] and o[0]+o[2]>=i[0]+i[2] and o[1]+o[3]>=i[1]+i[3]

def _pr(rects):
    rects = [r for r in rects if r[2]>0 and r[3]>0]
    return [ri for i,ri in enumerate(rects) if not any(i!=j and _co(rj,ri) for j,rj in enumerate(rects))]

def _split_free(free, placed):
    px,py,pw,pd = placed
    new_free = []
    for fx,fy,fw,fh in free:
        if not _ri((fx,fy,fw,fh), placed):
            new_free.append((fx,fy,fw,fh)); continue
        th=py-fy
        if th>0: new_free.append((fx,fy,fw,th))
        by2=py+pd; bh=(fy+fh)-by2
        if bh>0: new_free.append((fx,by2,fw,bh))
        ot=max(fy,py); ob=min(fy+fh,py+pd); oh=ob-ot
        lw=px-fx
        if lw>0 and oh>0: new_free.append((fx,ot,lw,oh))
        rx2=px+pw; rw2=(fx+fw)-rx2
        if rw2>0 and oh>0: new_free.append((rx2,ot,rw2,oh))
    return _pr(new_free)

def _find_best(free, bw, bd):
    best = None
    for rx,ry,rw,rh in free:
        if bw<=rw and bd<=rh:
            s = (rw*rh-bw*bd, min(rw-bw,rh-bd))
            c = (s,rx,ry,bw,bd)
            if best is None or c<best: best=c
        if bw!=bd and bd<=rw and bw<=rh:
            s = (rw*rh-bw*bd, min(rw-bd,rh-bw))
            c = (s,rx,ry,bd,bw)
            if best is None or c<best: best=c
    return best

def pack_maxrects(bin_w, bin_d, items):
    """items: [(idx,w,d)]. Returns [(idx,x,y,pw,pd)] or None."""
    free = [(0,0,bin_w,bin_d)]
    results = []
    for idx,bw,bd in items:
        best = _find_best(free, bw, bd)
        if best is None: return None
        _,px,py,pw,pd = best
        results.append((idx,px,py,pw,pd))
        free = _split_free(free, (px,py,pw,pd))
    return results

def pack_partial(bin_w, bin_d, items, exclude=None):
    """Pack as many as possible, skip those that don't fit."""
    free = [(0,0,bin_w,bin_d)]
    if exclude:
        for er in exclude:
            free = _split_free(free, er)
    results = []
    for idx,bw,bd in items:
        best = _find_best(free, bw, bd)
        if best is None: continue
        _,px,py,pw,pd = best
        results.append((idx,px,py,pw,pd))
        free = _split_free(free, (px,py,pw,pd))
    return results

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_orients(L, W, H, upright):
    if upright:
        return list({(L,W,H),(W,L,H)})
    return list({(L,W,H),(L,H,W),(W,L,H),(W,H,L),(H,L,W),(H,W,L)})

def rot_code(oL, oW, oH, px, py, pz):
    dims, labels = [oL,oW,oH], ["L","W","H"]
    used=[False]*3; code=[]
    for p in [px,py,pz]:
        for i in range(3):
            if not used[i] and dims[i]==p: code.append(labels[i]); used[i]=True; break
        else:
            for i in range(3):
                if not used[i]: code.append(labels[i]); used[i]=True; break
    return "".join(code)

def _sup(px, py, pw, pd, zl, fpm):
    if zl==0: return 1.0
    a = pw*pd
    if a==0: return 0.0
    rects = fpm.get(zl, [])
    s = 0.0
    for bx,by,bw,bd in rects:
        s += max(0, min(px+pw,bx+bw)-max(px,bx)) * max(0, min(py+pd,by+bd)-max(py,by))
    return s / a

# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------

def solve_task(task: Dict[str, Any]) -> Dict[str, Any]:
    t0 = time.time()
    pal = task["pallet"]
    PL, PW, PH = pal["length_mm"], pal["width_mm"], pal["max_height_mm"]
    MW = pal["max_weight_kg"]

    sd = {}
    for bs in task["boxes"]:
        sd[bs["sku_id"]] = (bs["length_mm"], bs["width_mm"], bs["height_mm"])

    # Expand and sort by weight desc
    items = []
    for bs in task["boxes"]:
        sid = bs["sku_id"]
        for inst in range(bs["quantity"]):
            items.append({
                "s": sid, "i": inst,
                "L": bs["length_mm"], "W": bs["width_mm"], "H": bs["height_mm"],
                "w": bs["weight_kg"], "u": bs["strict_upright"], "f": bs["fragile"],
            })
    items.sort(key=lambda it: (-it["w"], -(it["L"]*it["W"]*it["H"])))

    out = []
    fpm: Dict[int, List] = {}
    cz = 0; cw = 0.0; layer_num = 0
    rem = list(items)

    for _ in range(500):
        if not rem or cz >= PH: break
        wb = MW - cw

        # Height candidates
        hc = {}
        for it in rem[:500]:
            for bx,by,bz in get_orients(it["L"],it["W"],it["H"],it["u"]):
                if bz+cz<=PH and ((bx<=PL and by<=PW) or (by<=PL and bx<=PW)):
                    hc[bz] = hc.get(bz,0)+1
        if not hc: break

        sh = sorted(hc.keys(), key=lambda h: -hc[h])[:10]
        bi=[]; bu=set(); bh=0; bf=[]

        for th in sh:
            if cz+th>PH: continue
            cands = []
            for ri,it in enumerate(rem):
                best_base = None
                for bx,by,bz in get_orients(it["L"],it["W"],it["H"],it["u"]):
                    if bz==th:
                        if bx<=PL and by<=PW:
                            a = bx*by
                            if best_base is None or a > best_base[2]:
                                best_base = (bx,by,a)
                        if bx!=by and by<=PL and bx<=PW:
                            a = bx*by
                            if best_base is None or a > best_base[2]:
                                best_base = (by,bx,a)
                if best_base:
                    cands.append((ri,it,best_base[0],best_base[1]))
            if not cands: continue

            # Try multiple sort strategies
            for sk in [lambda c:-(c[2]*c[3]), lambda c:-max(c[2],c[3])]:
                sc = sorted(cands, key=sk)[:800]
                pi = [(i,c[2],c[3]) for i,c in enumerate(sc)]

                def tp(n, _c=sc, _w=wb, _p=pi):
                    ww=sum(_c[i][1]["w"] for i in range(n))
                    if ww>_w+1e-6: return None
                    return pack_maxrects(PL,PW,_p[:n])

                bc,br = 0,None
                lo,hi = 1,len(pi)
                while lo<=hi:
                    m=(lo+hi)//2; r=tp(m)
                    if r is not None: bc,br=m,r; lo=m+1
                    else: hi=m-1
                for n in range(bc+1, min(len(pi)+1, bc+15)):
                    r=tp(n)
                    if r is None: break
                    bc,br = n,r
                if br is None: continue

                v=[]; vu=set(); vf=[]
                for ci,px,py,pw,pd in br:
                    ri2,it2,_,_ = sc[ci]
                    if _sup(px,py,pw,pd,cz,fpm) >= 0.6-1e-9:
                        v.append((ri2,it2,px,py,pw,pd,th)); vu.add(ri2); vf.append((px,py,pw,pd))
                # Score: combined vol + count, normalized per unit height
                pv = PL*PW*PH
                ti = len(items) if items else 1
                sc2 = (0.8*sum(pw2*pd2*th2 for _,_,_,_,pw2,pd2,th2 in v)/pv + 0.1*len(v)/ti) / th
                bsc = (0.8*sum(pw2*pd2*th2 for _,_,_,_,pw2,pd2,th2 in bi)/pv + 0.1*len(bi)/ti) / bh if bi else 0
                if sc2 > bsc:
                    bi=v; bu=vu; bh=th; bf=vf

        if not bi: break

        # Gap fill with shorter items
        gr2=[it for ri,it in enumerate(rem) if ri not in bu]
        gp=[]; gu=set()
        gw=wb-sum(it["w"] for _,it,_,_,_,_,_ in bi)
        gc=[]
        for gi,it in enumerate(gr2):
            if it["w"]>gw+1e-6: continue
            for bx,by,bz in get_orients(it["L"],it["W"],it["H"],it["u"]):
                if bz<bh:
                    if bx<=PL and by<=PW: gc.append((gi,it,bx,by,bz)); break
                    if by<=PL and bx<=PW: gc.append((gi,it,by,bx,bz)); break
        if gc:
            gc.sort(key=lambda c:-(c[2]*c[3]))
            gc=gc[:300]
            gpi=[(i,c[2],c[3]) for i,c in enumerate(gc)]
            gres=pack_partial(PL,PW,gpi,bf)
            for ci,px,py,pw,pd in gres:
                gi2,it2,_,_,ih2 = gc[ci]
                if _sup(px,py,pw,pd,cz,fpm)>=0.6-1e-9 and gw>=it2["w"]-1e-6:
                    gp.append((it2,px,py,pw,pd,ih2)); gu.add(gi2); gw-=it2["w"]

        # Commit
        layer_num += 1
        zt = cz+bh
        for _,it,px,py,pw,pd,th in bi:
            oL,oW,oH = sd[it["s"]]
            out.append({"sku_id":it["s"],"instance_index":it["i"],
                "position":{"x_mm":px,"y_mm":py,"z_mm":cz},
                "dimensions_placed":{"length_mm":pw,"width_mm":pd,"height_mm":th},
                "rotation_code":rot_code(oL,oW,oH,pw,pd,th),"layer":layer_num})
            cw+=it["w"]
        for it,px,py,pw,pd,ih in gp:
            oL,oW,oH = sd[it["s"]]
            out.append({"sku_id":it["s"],"instance_index":it["i"],
                "position":{"x_mm":px,"y_mm":py,"z_mm":cz},
                "dimensions_placed":{"length_mm":pw,"width_mm":pd,"height_mm":ih},
                "rotation_code":rot_code(oL,oW,oH,pw,pd,ih),"layer":layer_num})
            cw+=it["w"]

        fpm[zt]=bf
        r2=[it for ri,it in enumerate(rem) if ri not in bu]
        rem=[it for gi,it in enumerate(r2) if gi not in gu]
        cz=zt

    # Final pass: try to pack remaining items using partial packing (greedy)
    for _fp in range(50):
        if not rem or cz >= PH or cw >= MW - 1e-6: break
        wb2 = MW - cw
        # Collect all possible orientations for remaining items that fit
        fc = []
        for ri,it in enumerate(rem):
            if it["w"] > wb2 + 1e-6: continue
            best_o = None
            for bx,by,bz in get_orients(it["L"],it["W"],it["H"],it["u"]):
                if cz+bz > PH: continue
                if bx<=PL and by<=PW:
                    if best_o is None or bx*by > best_o[3]:
                        best_o = (ri, bx, by, bx*by, bz)
                if bx!=by and by<=PL and bx<=PW:
                    if best_o is None or bx*by > best_o[3]:
                        best_o = (ri, by, bx, bx*by, bz)
            if best_o:
                fc.append((best_o[0], it, best_o[1], best_o[2], best_o[4]))
        if not fc: break
        fc.sort(key=lambda c: -(c[2]*c[3]))
        fc = fc[:300]
        pi2 = [(i, c[2], c[3]) for i,c in enumerate(fc)]
        # Use partial packing at current z
        pr = pack_partial(PL, PW, pi2)
        if not pr: break
        layer_num += 1
        placed_any = False
        placed_ri = set()
        max_h = 0
        layer_fp = []
        for ci,px,py,pw,pd in pr:
            ri2,it2,_,_,ih2 = fc[ci]
            if cw + it2["w"] > MW + 1e-6: continue
            if _sup(px,py,pw,pd,cz,fpm) < 0.6-1e-9: continue
            oL,oW,oH = sd[it2["s"]]
            out.append({"sku_id":it2["s"],"instance_index":it2["i"],
                "position":{"x_mm":px,"y_mm":py,"z_mm":cz},
                "dimensions_placed":{"length_mm":pw,"width_mm":pd,"height_mm":ih2},
                "rotation_code":rot_code(oL,oW,oH,pw,pd,ih2),"layer":layer_num})
            cw+=it2["w"]
            if ih2 > max_h: max_h = ih2
            if ih2 == max(c[4] for c in fc):  # only full-height items provide support
                layer_fp.append((px,py,pw,pd))
            placed_ri.add(ri2)
            placed_any = True
        if not placed_any: break
        if max_h > 0:
            zt2 = cz + max_h
            if layer_fp:
                fpm[zt2] = fpm.get(zt2, []) + layer_fp
            cz = zt2
        rem = [it for ri,it in enumerate(rem) if ri not in placed_ri]

    ms=int((time.time()-t0)*1000)
    pk={(p["sku_id"],p["instance_index"]) for p in out}
    ub={}
    for it in items:
        if (it["s"],it["i"]) not in pk: ub[it["s"]]=ub.get(it["s"],0)+1
    return {
        "task_id":task["task_id"],"solver_version":"1.0.0","solve_time_ms":ms,
        "placements":out,
        "unplaced":[{"sku_id":s,"quantity_unplaced":q,"reason":"no_space"} for s,q in ub.items()],
    }


def main():
    if len(sys.argv)<3:
        print(f"Usage: {sys.argv[0]} <input.json> <output.json>",file=sys.stderr); sys.exit(1)
    with open(sys.argv[1],"r",encoding="utf-8") as f: tasks=json.load(f)

    resps=[]; ts=0.0; vc=0
    for i,task in enumerate(tasks):
        resp=solve_task(task); resps.append(resp)
        r=evaluate_solution(task,resp); tid=task["task_id"]
        if r["valid"]:
            s=r["final_score"]; ts+=s; vc+=1; m=r["metrics"]
            print(f"[{i+1:4d}/{len(tasks)}] {tid}: score={s:.4f} "
                f"vol={m['volume_utilization']:.4f} cov={m['item_coverage']:.4f} "
                f"frag={m['fragility_score']:.4f} time={m['time_score']:.4f} "
                f"({len(resp['placements'])} placed, {resp['solve_time_ms']}ms)")
        else:
            print(f"[{i+1:4d}/{len(tasks)}] {tid}: INVALID - {r['error']}")

    avg=ts/len(tasks) if tasks else 0.0
    print(f"\n{'='*60}\nTasks: {len(tasks)} | Valid: {vc} | Avg score: {avg:.4f}")
    with open(sys.argv[2],"w",encoding="utf-8") as f: json.dump(resps,f,ensure_ascii=False,indent=2)
    print(f"Saved to: {sys.argv[2]}")

if __name__=="__main__": main()
