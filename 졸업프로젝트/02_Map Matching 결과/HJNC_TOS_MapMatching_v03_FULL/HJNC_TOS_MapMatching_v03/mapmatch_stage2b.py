import os,re,json,math,gzip,shutil,zipfile
from collections import Counter
import h5py,numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
OUT='/mnt/data/HJNC_TOS_MapMatching_v03';NET='/mnt/data/HJNC_Node_Link_Review_v03'
lanes=pd.read_csv(OUT+'/lane_lookup.csv');links=pd.read_csv(OUT+'/link_lookup.csv')
lane_ids=lanes.lane_id.astype(str).tolist();link_ids=links.link_id.astype(str).tolist()
with h5py.File(OUT+'/yt_mapmatched_points.h5','r') as h:
 g=h['points']
 Y=g['yt_num'][:];T=g['obs_time_ms'][:];LK=g['link_idx'][:];LA=g['lane_idx'][:];D=g['match_dist'][:];GD=g['geo_dist'][:];A=g['accepted'][:];J=g['jump_flag'][:];X=g['cad_x'][:];YY=g['cad_y'][:];DAY=g['source_day'][:];ST=g['match_status'][:]; TX=g['tos_x'][:]; TY=g['tos_y'][:]
N=len(Y); validcoord=ST!=0
# global stats
stats={
'input_archive':'TosIFB_Replay_all_20260106-20260108.zip','network_version':'HJNC_Node_Link_Review_v03',
'map_transform':{'type':'translation_only_candidate','cad_x':'tos_x + 358.0','cad_y':'tos_y + 469.75','operationally_verified':False},
'raw_coordinate_rows_including_missing':int(N+7275),'coordinate_rows_with_numeric_xy':int(N),'missing_xy_skipped':7275,'gross_coordinate_outliers':int((ST==0).sum()),'unique_yt':int(len(np.unique(Y))),'match_threshold_local':10.0,
}
for th in [3,5,10,20]:
 m=(D<=th)&validcoord;stats[f'within_{th}_count']=int(m.sum());stats[f'within_{th}_pct']=float(m.sum()/N*100)
stats.update({'median_match_distance_local':float(np.median(D[validcoord])),'p90_match_distance_local':float(np.percentile(D[validcoord],90)),'p95_match_distance_local':float(np.percentile(D[validcoord],95)),'speed_jump_flags_not_silently_deleted':int(J.sum())})
# coverage summaries
def cov_table(group,namefmt):
 rows=[]
 for v in np.unique(group):
  m=group==v;p=int(m.sum());r={'key':namefmt(int(v)),'points':p,'gross_outliers':int((ST[m]==0).sum()),'jump_flags':int(J[m].sum())}
  for th in [5,10,20]:
   c=int(((D[m]<=th)&(ST[m]!=0)).sum());r[f'within{th}']=c;r[f'within{th}_pct']=c/p*100
  rows.append(r)
 return pd.DataFrame(rows)
daydf=cov_table(DAY,lambda v:str(v)).rename(columns={'key':'source_day'});daydf.to_csv(OUT+'/coverage_by_source_day.csv',index=False)
ytdf=cov_table(Y,lambda v:f'YT{v}').rename(columns={'key':'yt_no'});ytdf.to_csv(OUT+'/coverage_by_yt.csv',index=False)
# conservative sequences
valid=(A==1)&(J==0)&(LK>=0)
idx=np.flatnonzero(valid); order=np.lexsort((T[idx],Y[idx])); idx=idx[order]
y=Y[idx];t=T[idx];lk=LK[idx];la=LA[idx];dd=D[idx];xx=X[idx];yy=YY[idx];day=DAY[idx]
b=np.ones(len(idx),dtype=bool)
if len(idx)>1:b[1:]=(y[1:]!=y[:-1])|(lk[1:]!=lk[:-1])|((t[1:]-t[:-1])>10000)
st=np.flatnonzero(b);en=np.r_[st[1:]-1,len(idx)-1];cnt=en-st+1
sumd=np.add.reduceat(dd,st);maxd=np.maximum.reduceat(dd,st)
runs=pd.DataFrame({'source_day':day[st],'yt_no':['YT%d'%v for v in y[st]],'lane_id':[lane_ids[int(v)] for v in la[st]],'link_id':[link_ids[int(v)] for v in lk[st]],'start_time_ms':t[st],'end_time_ms':t[en],'duration_s':(t[en]-t[st])/1000.,'point_count':cnt,'mean_match_dist':sumd/cnt,'max_match_dist':maxd,'start_cad_x':xx[st],'start_cad_y':yy[st],'end_cad_x':xx[en],'end_cad_y':yy[en]})
runs.to_csv(OUT+'/yt_link_sequence.csv.gz',index=False,compression='gzip')
stats['conservative_sequence_points']=int(valid.sum());stats['link_sequence_runs']=int(len(runs))
# usage
ai=np.flatnonzero(valid);link_count=np.bincount(LK[ai].astype(int),minlength=len(links));lane_count=np.bincount(LA[ai].astype(int),minlength=len(lanes))
lu=links[['link_idx','link_id','lane_id','zone','length_local','road_band_id']].copy();lu['matched_point_count']=link_count
uy=runs[['link_id','yt_no']].drop_duplicates().groupby('link_id').size();lu['unique_yt_count']=lu.link_id.map(uy).fillna(0).astype(int);lu['used']=lu.matched_point_count>0;lu.sort_values('matched_point_count',ascending=False).to_csv(OUT+'/link_usage_summary.csv',index=False)
laa=lanes[['lane_idx','lane_id','axis','zone','road_band_id','direction']].copy();laa['matched_point_count']=lane_count
uy2=runs[['lane_id','yt_no']].drop_duplicates().groupby('lane_id').size();laa['unique_yt_count']=laa.lane_id.map(uy2).fillna(0).astype(int);laa['used']=laa.matched_point_count>0;laa.sort_values('matched_point_count',ascending=False).to_csv(OUT+'/lane_usage_summary.csv',index=False)
stats['used_links']=int((link_count>0).sum());stats['total_links']=int(len(links));stats['used_lanes']=int((lane_count>0).sum());stats['total_lanes']=int(len(lanes))
# transition summary
perm_df=pd.read_csv(NET+'/turn_permissions.csv');perm={}
for r in perm_df.itertuples(index=False):
 key=(str(r.from_link),str(r.to_link))
 if int(r.analysis_routing_enabled)==1:perm[key]='analysis_allowed'
 elif str(r.permission).startswith('prohibited'):perm.setdefault(key,'prohibited_by_symbol')
 else:perm.setdefault(key,'review_required')
ry=runs.yt_no.to_numpy();rl=runs.link_id.to_numpy();rs=runs.start_time_ms.to_numpy(np.int64);re_=runs.end_time_ms.to_numpy(np.int64)
mask=(ry[1:]==ry[:-1])&((rs[1:]-re_[:-1])<=30000)
a=rl[:-1][mask];bb=rl[1:][mask];gaps=(rs[1:][mask]-re_[:-1][mask])/1000.
tdf=pd.DataFrame({'from_link':a,'to_link':bb,'gap_s':gaps});tdf['transition_status']=[perm.get((x,y),'not_directly_encoded') for x,y in zip(a,bb)]
ts=tdf.groupby(['from_link','to_link','transition_status'],as_index=False).agg(count=('gap_s','size'),mean_gap_s=('gap_s','mean')).sort_values('count',ascending=False);ts.to_csv(OUT+'/transition_summary.csv',index=False)
# unmatched hotspots 10-unit
um=(GD>20)&validcoord; bx=(np.floor(X[um]/10)*10).astype(np.int32);by=(np.floor(YY[um]/10)*10).astype(np.int32);keys=np.stack([bx,by],axis=1);u,c=np.unique(keys,axis=0,return_counts=True);ordr=np.argsort(-c)
pd.DataFrame({'cad_x_bin':u[ordr,0],'cad_y_bin':u[ordr,1],'count':c[ordr]}).to_csv(OUT+'/unmatched_hotspots_10unit.csv',index=False)
# sample every stride -> 100k max
step=max(1,N//100000);si=np.arange(0,N,step)[:100000]
sdf=pd.DataFrame({'obs_time_ms':T[si],'yt_no':['YT%d'%v for v in Y[si]],'tos_x':TX[si],'tos_y':TY[si],'cad_x':X[si],'cad_y':YY[si],'lane_id':[lane_ids[int(v)] if v>=0 else '' for v in LA[si]],'link_id':[link_ids[int(v)] if v>=0 else '' for v in LK[si]],'match_dist':D[si],'match_status':ST[si],'accepted':A[si],'jump_flag':J[si]});sdf.to_csv(OUT+'/yt_mapmatched_sample.csv.gz',index=False,compression='gzip')
# validation json
stats['notes']=['Spatial coverage against current v03 main-yard graph; not a ground-truth lane accuracy rate.','Gross coordinate outliers retained as status=0/accepted=0.','Speed-jump diagnostics retained point-level and excluded only from conservative link sequence.','Gate/North/Apron and some turn connectors remain incomplete, so unmatched residuals include graph incompleteness.']
with open(OUT+'/map_matching_validation.json','w',encoding='utf-8') as f:json.dump(stats,f,ensure_ascii=False,indent=2)
# schema
open(OUT+'/HDF5_SCHEMA.md','w',encoding='utf-8').write('''# yt_mapmatched_points.h5\nGroup `/points` contains one fixed-length dataset per field.\n\n- obs_time_ms: TOS wrapper updateTime epoch ms\n- point_time_num: equipment updateTime YYYYMMDDhhmmss integer when present\n- source_day: archive folder date\n- yt_num: numeric YT id\n- tos_x/tos_y: raw Terminal coordinates\n- cad_x/cad_y: candidate-transformed coordinates\n- lane_idx/link_idx: indices into lane_lookup.csv / link_lookup.csv\n- match_dist: direction-aware selected lane distance\n- geo_lane_idx/geo_dist: pure geometric nearest lane and distance\n- motion_axis: 0 unknown, 1 horizontal, 2 vertical\n- heading_sign: -1/0/+1\n- speed_est: diagnostic local-unit/s\n- jump_flag: high-speed jump diagnostic\n- match_status: 0 gross outlier, 1 <=3, 2 <=5, 3 <=10, 4 <=20, 5 >20\n- accepted: 1 if match_dist <= 10 and coordinate valid\n''')
# visuals helpers
line_re=re.compile(r'[-+]?\d*\.\d+')
fig,ax=plt.subplots(figsize=(18,8),dpi=160)
for r in lanes.itertuples(index=False):
 q=list(map(float,line_re.findall(str(r.geometry_wkt))));ax.plot([q[0],q[2]],[q[1],q[3]],lw=.7,alpha=.45,c='black')
md=sdf.match_dist
for m,col,label in [(md<=5,'#2ca02c','<=5'),((md>5)&(md<=10),'#1f77b4','5–10'),((md>10)&(md<=20),'#ff7f0e','10–20'),(md>20,'#d62728','>20 / unresolved')]:ax.scatter(sdf.loc[m,'cad_x'],sdf.loc[m,'cad_y'],s=.8,alpha=.13,c=col,label=label)
ax.set_title('HJNC TOS YT Map Matching vs v03 Lane Graph');ax.set_xlabel('CAD local X');ax.set_ylabel('CAD local Y');ax.set_aspect('equal',adjustable='box');ax.grid(alpha=.12);ax.legend(markerscale=6);fig.tight_layout();fig.savefig(OUT+'/mapmatch_overview.png',bbox_inches='tight');plt.close(fig)
segments=[];vals=[]
for i,r in links.iterrows():
 q=list(map(float,line_re.findall(str(r.geometry_wkt))));segments.append([(q[0],q[1]),(q[2],q[3])]);vals.append(math.log1p(link_count[i]))
fig,ax=plt.subplots(figsize=(18,8),dpi=160);lc=LineCollection(segments,array=np.array(vals),cmap='viridis',linewidths=1.4);ax.add_collection(lc);ax.autoscale();ax.set_aspect('equal',adjustable='box');ax.set_title('Matched YT Link Usage (log1p point count)');ax.set_xlabel('CAD local X');ax.set_ylabel('CAD local Y');fig.colorbar(lc,ax=ax,label='log1p(matched points)');fig.tight_layout();fig.savefig(OUT+'/link_usage_map.png',bbox_inches='tight');plt.close(fig)
vals=np.sort(GD[validcoord][::max(1,int(validcoord.sum())//500000)]);cdf=np.arange(1,len(vals)+1)/len(vals)*100
fig,ax=plt.subplots(figsize=(10,6),dpi=160);ax.plot(vals,cdf,lw=2)
for z in [5,10,20]:ax.axvline(z,ls='--',lw=1)
ax.set_xlim(0,min(120,float(np.percentile(vals,99))));ax.set_ylim(0,100);ax.set_xlabel('Nearest lane distance (local unit)');ax.set_ylabel('Cumulative coverage (%)');ax.set_title('Map Matching Spatial Coverage CDF');ax.grid(alpha=.25);fig.tight_layout();fig.savefig(OUT+'/distance_cdf.png',bbox_inches='tight');plt.close(fig)
# readme
s=stats
readme=f'''# HJNC TOS 3-day Map Matching v03\n\n## 수행 대상\n- TOS archive: `TosIFB_Replay_all_20260106-20260108.zip`\n- Spatial graph: `HJNC_Node_Link_Review_v03`\n- Candidate transform: `CAD X = TOS X + 358.0`, `CAD Y = TOS Y + 469.75`\n- Primary acceptance threshold: 10 local units\n\n## 처리 결과\n- Raw YT coordinate rows including missing XY: {s['raw_coordinate_rows_including_missing']:,}\n- Numeric XY rows mapped: {s['coordinate_rows_with_numeric_xy']:,}\n- Missing XY skipped: {s['missing_xy_skipped']:,}\n- Gross coordinate outliers: {s['gross_coordinate_outliers']:,}\n- Unique YT: {s['unique_yt']}\n\n## 공간 커버리지\n- <=3: {s['within_3_count']:,} ({s['within_3_pct']:.2f}%)\n- <=5: {s['within_5_count']:,} ({s['within_5_pct']:.2f}%)\n- <=10: {s['within_10_count']:,} ({s['within_10_pct']:.2f}%) — accepted threshold\n- <=20: {s['within_20_count']:,} ({s['within_20_pct']:.2f}%)\n- Median match distance: {s['median_match_distance_local']:.3f}\n- P95 match distance: {s['p95_match_distance_local']:.3f}\n\n> 이 수치는 현재 v03 도로망에 대한 **공간 커버리지**이며 정답 라벨 기반 lane accuracy가 아닙니다. Gate/North/Apron 및 일부 회전 연결이 미완성이라 해당 영역의 잔차가 포함됩니다.\n\n## 핵심 산출물\n- `yt_mapmatched_points.h5` — 전체 {s['coordinate_rows_with_numeric_xy']:,}건 point-level 결과\n- `yt_link_sequence.csv.gz` — <=10 + jump 제외 보수적 Link sequence\n- `coverage_by_yt.csv`, `coverage_by_source_day.csv`\n- `link_usage_summary.csv`, `lane_usage_summary.csv`\n- `transition_summary.csv`\n- `unmatched_hotspots_10unit.csv`\n- `yt_mapmatched_sample.csv.gz`\n- `map_matching_validation.json`\n- `mapmatch_overview.png`, `link_usage_map.png`, `distance_cdf.png`\n\n## 데이터 품질 처리\n- XY 결측 {s['missing_xy_skipped']:,}건은 좌표 매칭 대상에서 제외했습니다.\n- 비정상 대좌표 {s['gross_coordinate_outliers']:,}건은 삭제하지 않고 HDF5에 status=0으로 보존했습니다.\n- 속도점프 진단 {s['speed_jump_flags_not_silently_deleted']:,}건도 삭제하지 않았고, 보수적 Link sequence 생성 시에만 제외했습니다.\n\n## 해석\n10-unit 기준 {s['within_10_pct']:.2f}%가 현 v03 Lane Graph에 직접 붙습니다. 20-unit 기준은 {s['within_20_pct']:.2f}%입니다. 나머지는 좌표 오류로 단정하지 않고 Gate/North/Apron·미복원 도로·접속 구조 및 좌표정합 오차를 함께 포함하는 **보완 우선영역**으로 분리했습니다.\n'''
open(OUT+'/README_KO.md','w',encoding='utf-8').write(readme)
print(json.dumps(stats,ensure_ascii=False,indent=2));print('runs',len(runs),'transitions',len(ts),'h5MB',os.path.getsize(OUT+'/yt_mapmatched_points.h5')/1e6)
