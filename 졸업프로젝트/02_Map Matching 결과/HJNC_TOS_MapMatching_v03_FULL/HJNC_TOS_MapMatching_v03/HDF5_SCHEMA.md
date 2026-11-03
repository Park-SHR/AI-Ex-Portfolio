# yt_mapmatched_points.h5
Group `/points` contains one fixed-length dataset per field.

- obs_time_ms: TOS wrapper updateTime epoch ms
- point_time_num: equipment updateTime YYYYMMDDhhmmss integer when present
- source_day: archive folder date
- yt_num: numeric YT id
- tos_x/tos_y: raw Terminal coordinates
- cad_x/cad_y: candidate-transformed coordinates
- lane_idx/link_idx: indices into lane_lookup.csv / link_lookup.csv
- match_dist: direction-aware selected lane distance
- geo_lane_idx/geo_dist: pure geometric nearest lane and distance
- motion_axis: 0 unknown, 1 horizontal, 2 vertical
- heading_sign: -1/0/+1
- speed_est: diagnostic local-unit/s
- jump_flag: high-speed jump diagnostic
- match_status: 0 gross outlier, 1 <=3, 2 <=5, 3 <=10, 4 <=20, 5 >20
- accepted: 1 if match_dist <= 10 and coordinate valid
