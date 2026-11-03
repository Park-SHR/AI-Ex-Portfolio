import os,glob,h5py,numpy as np,pandas as pd,json
from collections import Counter,defaultdict
OUT='/mnt/data/HJNC_TOS_MapMatching_v03'; PART=OUT+'/parts'
parts=sorted(glob.glob(PART+'/part_*.npz'))
fields=[('obs','obs_time_ms','i8'),('point_ts','point_time_num','i8'),('source_day','source_day','i4'),('yt','yt_num','i2'),('x','tos_x','f4'),('y','tos_y','f4'),('cadx','cad_x','f4'),('cady','cad_y','f4'),('lane','lane_idx','i2'),('link','link_idx','i2'),('dist','match_dist','f4'),('geo_lane','geo_lane_idx','i2'),('geo_dist','geo_dist','f4'),('motion','motion_axis','i1'),('sign','heading_sign','i1'),('speed','speed_est','f4'),('jump','jump_flag','u1'),('status','match_status','u1'),('accepted','accepted','u1')]
ns=[]
for p in parts:
 z=np.load(p);ns.append(len(z['yt']));z.close()
N=sum(ns)
p=OUT+'/yt_mapmatched_points.h5'
if os.path.exists(p):os.remove(p)
with h5py.File(p,'w') as h:
 g=h.create_group('points'); pos=0
 dsets={}
 for src,dst,dt in fields:
  dsets[src]=g.create_dataset(dst,shape=(N,),dtype=dt,chunks=(262144,),compression='lzf',shuffle=True)
 for pp,n in zip(parts,ns):
  z=np.load(pp);sl=slice(pos,pos+n)
  for src,dst,dt in fields:dsets[src][sl]=z[src]
  pos+=n;z.close()
 h.attrs['rows']=N;h.attrs['schema']='See HDF5_SCHEMA.md';h.attrs['map_transform']='cad_x=tos_x+358.0; cad_y=tos_y+469.75';h.attrs['accept_threshold_local']=10.0
print('wrote',N,p,os.path.getsize(p))
