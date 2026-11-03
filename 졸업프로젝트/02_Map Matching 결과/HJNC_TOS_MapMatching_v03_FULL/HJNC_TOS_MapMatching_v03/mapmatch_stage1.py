import os, re, glob, math, json, time, csv, gzip, shutil
import multiprocessing as mp
from collections import Counter, defaultdict
import numpy as np
import pandas as pd
import orjson
import tables
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT='/mnt/data'
SRC=os.path.join(ROOT,'TOS_REPLAY_EXTRACTED')
NET=os.path.join(ROOT,'HJNC_Node_Link_Review_v03')
OUT=os.path.join(ROOT,'HJNC_TOS_MapMatching_v03')
PART=os.path.join(OUT,'parts')
DX=358.0; DY=469.75
ACCEPT=10.0
os.makedirs(PART, exist_ok=True)

# ---------- network metadata ----------
lane_df=pd.read_csv(os.path.join(NET,'lane_master.csv'))
link_df=pd.read_csv(os.path.join(NET,'link_master.csv'))
line_re=re.compile(r'LINESTRING \(([\d.+-]+) ([\d.+-]+), ([\d.+-]+) ([\d.+-]+)\)')
def parse_line(w):
    m=line_re.fullmatch(str(w).strip())
    if not m: raise ValueError(w)
    return tuple(map(float,m.groups()))
lane_geom=np.array([parse_line(w) for w in lane_df.geometry_wkt], dtype=np.float64)
lane_axis=lane_df.axis.astype(str).to_numpy()
lane_dir=lane_df.direction.astype(int).to_numpy(np.int8)
lane_ids=lane_df.lane_id.astype(str).tolist()
lane_to_idx={x:i for i,x in enumerate(lane_ids)}

link_geom=np.array([parse_line(w) for w in link_df.geometry_wkt], dtype=np.float64)
link_ids=link_df.link_id.astype(str).tolist()
link_lane_idx=np.array([lane_to_idx[x] for x in link_df.lane_id.astype(str)], dtype=np.int16)
# intervals per lane; each tuple (mins,maxs,link_indices), sorted by min coordinate
lane_intervals={}
for li in range(len(lane_df)):
    idx=np.where(link_lane_idx==li)[0]
    ax=lane_axis[li]
    if ax=='H':
        mn=np.minimum(link_geom[idx,0],link_geom[idx,2]); mx=np.maximum(link_geom[idx,0],link_geom[idx,2])
    else:
        mn=np.minimum(link_geom[idx,1],link_geom[idx,3]); mx=np.maximum(link_geom[idx,1],link_geom[idx,3])
    order=np.argsort(mn)
    lane_intervals[li]=(mn[order],mx[order],idx[order])

# save lookups
lane_df2=lane_df.copy(); lane_df2.insert(0,'lane_idx',np.arange(len(lane_df2),dtype=int))
lane_df2.to_csv(os.path.join(OUT,'lane_lookup.csv'),index=False)
link_df2=link_df.copy(); link_df2.insert(0,'link_idx',np.arange(len(link_df2),dtype=int))
link_df2.to_csv(os.path.join(OUT,'link_lookup.csv'),index=False)

# ---------- source file ordering ----------
def first_outer_ts(path):
    with open(path,'rb') as f:
        for line in f:
            try:
                o=orjson.loads(line); t=o.get('updateTime')
            except Exception: continue
            if isinstance(t,int): return t
    return 0
files=glob.glob(os.path.join(SRC,'**','*tosMessage*.log'),recursive=True)
file_meta=sorted([(first_outer_ts(p),p) for p in files])

# ---------- geometry helpers ----------
def lane_distance_arrays(x,y):
    n=len(x)
    best_geo=np.full(n,np.inf,dtype=np.float32)
    best_geo_idx=np.full(n,-1,dtype=np.int16)
    # return no matrix, just best geometric
    for i,(x1,y1,x2,y2) in enumerate(lane_geom):
        if lane_axis[i]=='H':
            cx=np.clip(x,min(x1,x2),max(x1,x2)); d=np.hypot(x-cx,y-y1)
        else:
            cy=np.clip(y,min(y1,y2),max(y1,y2)); d=np.hypot(x-x1,y-cy)
        d=d.astype(np.float32,copy=False)
        m=d<best_geo
        best_geo[m]=d[m]; best_geo_idx[m]=i
    return best_geo,best_geo_idx

def map_lanes(x,y,motion_axis,heading_sign,valid):
    n=len(x)
    best_score=np.full(n,np.inf,dtype=np.float32)
    best_idx=np.full(n,-1,dtype=np.int16)
    best_dist=np.full(n,np.inf,dtype=np.float32)
    geo_dist=np.full(n,np.inf,dtype=np.float32)
    geo_idx=np.full(n,-1,dtype=np.int16)
    for i,(x1,y1,x2,y2) in enumerate(lane_geom):
        if lane_axis[i]=='H':
            cx=np.clip(x,min(x1,x2),max(x1,x2)); d=np.hypot(x-cx,y-y1)
            axis_code=1
        else:
            cy=np.clip(y,min(y1,y2),max(y1,y2)); d=np.hypot(x-x1,y-cy)
            axis_code=2
        d=d.astype(np.float32,copy=False)
        mg=d<geo_dist
        geo_dist[mg]=d[mg];geo_idx[mg]=i
        score=d.copy()
        known=motion_axis!=0
        score += ((known)&(motion_axis!=axis_code))*4.0
        same=(known)&(motion_axis==axis_code)&(heading_sign!=0)
        score += (same & (heading_sign!=lane_dir[i]))*2.5
        score[~valid]=np.inf
        m=score<best_score
        best_score[m]=score[m];best_idx[m]=i;best_dist[m]=d[m]
    return best_idx,best_dist,geo_idx,geo_dist

def links_from_lanes(lidx,x,y):
    out=np.full(len(x),-1,dtype=np.int16)
    for li in np.unique(lidx[lidx>=0]):
        ii=np.where(lidx==li)[0]
        coord=x[ii] if lane_axis[li]=='H' else y[ii]
        mn,mx,lks=lane_intervals[int(li)]
        # nearest interval: first max >= coordinate; if gap, compare adjacent interval distances
        j=np.searchsorted(mx,coord,side='left')
        j=np.clip(j,0,len(lks)-1)
        # fix points that fall before candidate min and previous is closer
        jj=j.copy()
        mask=(coord<mn[j])&(j>0)
        if np.any(mask):
            jp=j[mask]-1
            dprev=np.maximum(mn[jp]-coord[mask],coord[mask]-mx[jp])
            dcur=np.maximum(mn[j[mask]]-coord[mask],coord[mask]-mx[j[mask]])
            choose=dprev<dcur
            jj_sub=jj[mask]; jj_sub[choose]=jp[choose]; jj[mask]=jj_sub
        out[ii]=lks[jj].astype(np.int16)
    return out

# ---------- worker ----------
def process_file(args):
    order, first_ts, path=args
    st=time.time()
    obs=[]; pt=[]; yt=[]; xs=[]; ys=[]
    missing=0; msgs=0
    with open(path,'rb',buffering=4*1024*1024) as f:
        for line in f:
            if b'"id":"TOS-11-2"' not in line or b'"equNo":"YT' not in line: continue
            o=orjson.loads(line); ts=int(o.get('updateTime') or 0); msgs+=1
            for d in o.get('data',[]):
                e=d.get('equNo','')
                if not isinstance(e,str) or not e.startswith('YT'): continue
                try: xv=float(d.get('x')); yv=float(d.get('y'))
                except Exception: missing+=1; continue
                try: yn=int(e[2:])
                except: continue
                u=str(d.get('updateTime') or '')
                try: pu=int(u[:14]) if len(u)>=14 and u[:14].isdigit() else 0
                except: pu=0
                obs.append(ts);pt.append(pu);yt.append(yn);xs.append(xv);ys.append(yv)
    obs=np.asarray(obs,dtype=np.int64);pt=np.asarray(pt,dtype=np.int64);yt=np.asarray(yt,dtype=np.int16)
    x=np.asarray(xs,dtype=np.float32);y=np.asarray(ys,dtype=np.float32)
    n=len(x)
    valid=np.isfinite(x)&np.isfinite(y)&(np.abs(x)<=2000)&(np.abs(y)<=2000)
    gross=(~valid).astype(np.uint8)
    cadx=(x+DX).astype(np.float32);cady=(y+DY).astype(np.float32)

    # motion hints by YT, with short stationary forward-fill
    motion=np.zeros(n,dtype=np.int8); sign=np.zeros(n,dtype=np.int8); speed=np.zeros(n,dtype=np.float32); jump=np.zeros(n,dtype=np.uint8)
    for yn in np.unique(yt):
        ii=np.flatnonzero(yt==yn)
        if len(ii)<2: continue
        xx=x[ii]; yy=y[ii]; tt=obs[ii]
        dx=np.diff(xx,prepend=xx[0]); dy=np.diff(yy,prepend=yy[0]); dt=np.diff(tt,prepend=tt[0]).astype(np.float64)/1000.0
        ds=np.hypot(dx,dy); sp=np.divide(ds,np.maximum(dt,1e-3)); speed[ii]=sp.astype(np.float32)
        jmp=(sp>60)&(dt>0)&(dt<=5)&(ds>30); jump[ii]=jmp.astype(np.uint8)
        mh=np.zeros(len(ii),dtype=np.int8); sg=np.zeros(len(ii),dtype=np.int8)
        reliable=(ds>=0.35)&(sp<=35)&(dt>0)&(dt<=10)
        h=reliable&(np.abs(dx)>=1.5*np.abs(dy)); v=reliable&(np.abs(dy)>=1.5*np.abs(dx))
        mh[h]=1;sg[h]=np.sign(dx[h]).astype(np.int8)
        mh[v]=2;sg[v]=np.sign(dy[v]).astype(np.int8)
        # carry heading over stationary / tiny motion for max 10 seconds
        last_a=0;last_s=0;last_t=-10**18
        for k in range(len(ii)):
            if mh[k]!=0:
                last_a=int(mh[k]);last_s=int(sg[k]);last_t=int(tt[k])
            elif int(tt[k])-last_t<=10000:
                mh[k]=last_a;sg[k]=last_s
        motion[ii]=mh;sign[ii]=sg

    lidx,dist,gidx,gdist=map_lanes(cadx,cady,motion,sign,valid)
    linkidx=links_from_lanes(lidx,cadx,cady)
    status=np.full(n,5,dtype=np.uint8)
    status[(dist<=20)&valid]=4
    status[(dist<=10)&valid]=3
    status[(dist<=5)&valid]=2
    status[(dist<=3)&valid]=1
    status[~valid]=0
    accepted=((dist<=ACCEPT)&valid).astype(np.uint8)
    # source day from path
    m=re.search(r'2026010[678]',path); source_day=int(m.group(0)) if m else 0
    day=np.full(n,source_day,dtype=np.int32)

    outpath=os.path.join(PART,f'part_{order:02d}_{first_ts}.npz')
    np.savez(outpath,obs=obs,point_ts=pt,yt=yt,x=x,y=y,cadx=cadx,cady=cady,
             lane=lidx,link=linkidx,dist=dist,geo_lane=gidx,geo_dist=gdist,
             motion=motion,sign=sign,speed=speed,jump=jump,status=status,accepted=accepted,source_day=day)
    meta=dict(order=order,source_file=os.path.relpath(path,SRC),first_ts=int(obs.min()) if n else first_ts,last_ts=int(obs.max()) if n else first_ts,
              rows=int(n),missing=int(missing),gross_outlier=int((~valid).sum()),jump_flag=int(jump.sum()),
              within3=int((status==1).sum()),within5=int(((status>=1)&(status<=2)).sum()),within10=int((accepted==1).sum()),within20=int(((status>=1)&(status<=4)).sum()),
              geo_within10=int(((gdist<=10)&valid).sum()),direction_adjusted=int(((lidx!=gidx)&valid).sum()),seconds=round(time.time()-st,3))
    with open(outpath+'.json','w',encoding='utf-8') as f: json.dump(meta,f,ensure_ascii=False,indent=2)
    return meta

if __name__=='__main__':
    tasks=[(i,ts,p) for i,(ts,p) in enumerate(file_meta)]
    with mp.Pool(min(8,len(tasks))) as pool:
        metas=pool.map(process_file,tasks)
    metas=sorted(metas,key=lambda x:x['order'])
    pd.DataFrame(metas).to_csv(os.path.join(OUT,'source_partition_summary.csv'),index=False)
    with open(os.path.join(OUT,'_stage1_done.json'),'w') as f:json.dump(metas,f,indent=2)
    print(json.dumps(metas,indent=2))
