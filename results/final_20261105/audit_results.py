"""保存CSVから、計算本体を使わずに全運用・集計・対応比較を検算する。"""
from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd

OUT=Path(__file__).resolve().parent
STAGE=OUT.parents[1]
TOL=1e-9
r={name:pd.read_csv(OUT/f'{name}.csv',float_precision='round_trip',keep_default_na=True)
   for name in ['ship_data','plans','schedules','run_metrics','summary','paired_comparison','plan_diagnostics']}
c=json.loads((OUT/'config.json').read_text())
data=r['ship_data'].set_index(['demand_multiplier','trial','ship'])
schedules=r['schedules']
runs=r['run_metrics']
metrics=['total_waiting_time','average_waiting_time','max_waiting_time','total_turnaround_time',
         'average_turnaround_time','completion_time','target_waiting_time','other_average_waiting_time',
         'other_waiting_ships','other_max_waiting_time','berth_changes']
keys=['experiment_group','demand_multiplier','trial','target_position','shift_hours','scenario','policy']
cache={}
count=0
for key,table in schedules.groupby(keys,sort=False):
 group,demand,trial,position,width,scenario,policy=key
 original=data.loc[(demand,trial)]
 assert set(table.ship)==set(original.index)
 assert table.ship.is_unique
 np.testing.assert_allclose(table.actual_arrival_time,table.ship.map(original.actual_arrival_time),rtol=0,atol=TOL)
 expected_handling=np.array([original.loc[row.ship,f'handling_time_{row.assigned_berth}'] for row in table.itertuples()])
 np.testing.assert_allclose(table.handling_time,expected_handling,rtol=0,atol=TOL)
 np.testing.assert_allclose(table.waiting_time,table.start_time-table.actual_arrival_time,rtol=0,atol=TOL)
 np.testing.assert_allclose(table.turnaround_time,table.completion_time-table.actual_arrival_time,rtol=0,atol=TOL)
 np.testing.assert_allclose(table.completion_time,table.start_time+table.handling_time,rtol=0,atol=TOL)
 assert (table.start_time>=table.actual_arrival_time-TOL).all()
 assert (table.start_time>=table.earliest_start_time-TOL).all()
 if policy!='unrestricted':
  limit=0 if policy=='fixed' or group=='sensitivity' else 2
  assert (table.start_time>=table.planned_start_time-limit-TOL).all()
 if policy in ['fixed','same_berth']:
  assert (table.assigned_berth==table.planned_berth).all()
 for _,berth in table.groupby('assigned_berth'):
  berth=berth.sort_values('start_time')
  assert (berth.start_time.to_numpy()[1:]>=berth.completion_time.to_numpy()[:-1]-TOL).all()
 if scenario=='normal':
  assert table.waiting_time.abs().max()<=TOL
  assert (table.assigned_berth==table.planned_berth).all()
  assert (table.reported_arrival_time==table.actual_arrival_time).all()
 else:
  target=table.target_ship.iloc[0]
  shift=-width if scenario=='early' else width
  expected=np.where(table.ship==target,np.maximum(0,table.actual_arrival_time+shift),table.actual_arrival_time)
  np.testing.assert_allclose(table.reported_arrival_time,expected,rtol=0,atol=TOL)
  plan=r['plans']
  plan=plan[(plan.demand_multiplier==demand)&(plan.trial==trial)&(plan.target_position==position)
            &(plan.shift_hours==width)&(plan.scenario==scenario)&(plan.experiment_group==group)].set_index('ship')
  np.testing.assert_array_equal(table.planned_berth,table.ship.map(plan.assigned_berth))
  np.testing.assert_allclose(table.planned_start_time,table.ship.map(plan.planned_start_time),rtol=0,atol=TOL)
 cache[key]=table
 count+=1

for row in runs.itertuples():
 key=(row.experiment_group,row.demand_multiplier,row.trial,
      'baseline' if row.scenario=='normal' else row.target_position,row.shift_hours,row.scenario,row.policy)
 table=cache[key]
 others=table[table.ship!=row.target_ship]
 target=table[table.ship==row.target_ship]
 expected=[table.waiting_time.sum(),table.waiting_time.mean(),table.waiting_time.max(),
           table.turnaround_time.sum(),table.turnaround_time.mean(),table.completion_time.max(),
           target.waiting_time.iloc[0],others.waiting_time.mean(),int((others.waiting_time>TOL).sum()),
           others.waiting_time.max(),int((table.assigned_berth!=table.planned_berth).sum())]
 np.testing.assert_allclose([getattr(row,m) for m in metrics],expected,rtol=0,atol=TOL)

join_keys=['experiment_group','demand_multiplier','trial','target_position','policy']
normal=runs[runs.scenario=='normal'].set_index(join_keys)
assert normal.index.is_unique
attacks=runs[runs.scenario!='normal'].join(normal[metrics],on=join_keys,rsuffix='_normal')
cells=['experiment_group','demand_multiplier','target_position','shift_hours','scenario','policy']
summary=r['summary'].set_index(cells+['metric'])
comparison=r['paired_comparison'].set_index(cells+['metric'])
for key,rows in attacks.groupby(cells,sort=False):
 for m in metrics:
  s=summary.loc[key+(m,)];p=comparison.loc[key+(m,)]
  differences=rows[m]-rows[f'{m}_normal']
  np.testing.assert_allclose([s['mean'],s['std'],p.mean_difference,p.std_difference,p.min_difference,p.max_difference],
                            [rows[m].mean(),rows[m].std(ddof=1),differences.mean(),differences.std(ddof=1),differences.min(),differences.max()],rtol=0,atol=TOL)
  assert p.trials==10
  assert p.increased_trials==int((differences>TOL).sum())
  assert p.decreased_trials==int((differences < -TOL).sum())
  assert p.unchanged_trials==int((differences.abs()<=TOL).sum())
  if key[0]=='main':
   fixed_key=key[:-1]+('fixed',m)
   np.testing.assert_allclose(p.mean_mitigation_vs_fixed,comparison.loc[fixed_key].mean_difference-p.mean_difference,rtol=0,atol=TOL)
  else:
   assert pd.isna(p.mean_mitigation_vs_fixed)

assert len(r['plan_diagnostics'])==30
assert len(attacks.query("experiment_group=='main'"))==2160
assert len(attacks.query("experiment_group=='sensitivity'"))==60
for name,digest in c['code_sha256'].items():
 assert hashlib.sha256((STAGE/name).read_bytes()).hexdigest()==digest
reference=json.loads((OUT/'kobe_reference.json').read_text())
for source in reference['sources']:
 assert hashlib.sha256(Path(source['file']).read_bytes()).hexdigest()==source['sha256']
forecast=pd.read_csv(OUT/'kobe_daily_forecasts.csv')
assert forecast.scheduled_ship_days.sum()==reference['scheduled_ship_days']==103
assert len(forecast)==13
np.testing.assert_allclose(c['base_mean_candidate_interval_hours'],24*13/103)
for n in range(1,6):
 for ext in ['png','pdf']:
  files=list(OUT.glob(f'{n:02d}_*.{ext}'))
  assert len(files)==1 and files[0].stat().st_size>5000

main=runs.query("experiment_group=='main' and scenario!='normal'")
index=['demand_multiplier','trial','target_position','shift_hours','scenario']
wide=main.pivot(index=index,columns='policy',values='other_average_waiting_time')
comparisons={}
for a,b in [('fixed','same_berth'),('same_berth','reassign'),('fixed','reassign'),('reassign','unrestricted')]:
 difference=wide[b]-wide[a]
 comparisons[f'{a}_to_{b}']={'improved':int((difference < -TOL).sum()),'worsened':int((difference>TOL).sum()),
                            'unchanged':int((difference.abs()<=TOL).sum()),'mean_difference_hours':difference.mean(),
                            'worst_worsening_hours':difference.max()}
worst=wide.assign(delta=wide.reassign-wide.same_berth).sort_values('delta',ascending=False).head(3)
example=main.query("demand_multiplier==1 and target_position=='middle' and shift_hours==4 and scenario=='early' and policy=='fixed'")
example=example.sort_values(['other_average_waiting_time','trial'],ascending=[False,True]).iloc[0]
analysis={'validated_schedule_executions':count,'validated_schedule_rows':len(schedules),'normal_metrics':len(runs[runs.scenario=='normal']),
          'normal_max_waiting_hours':runs.query("scenario=='normal'").max_waiting_time.max(),
          'overall_equal_cell_means':main.groupby(['policy','scenario'])[['other_average_waiting_time','other_waiting_ships','berth_changes']].mean().reset_index().to_dict('records'),
          'policy_pair_counts_over_540_cells':comparisons,
          'reassignment_worsening_examples':worst.reset_index().to_dict('records'),
          'baseline_4h_middle':main.query("demand_multiplier==1 and target_position=='middle' and shift_hours==4").groupby(['policy','scenario'])[['other_average_waiting_time','target_waiting_time','berth_changes']].mean().reset_index().to_dict('records'),
          'gantt_example':example[['seed','trial','target_ship','other_average_waiting_time']].to_dict(),
          'normal_diagnostics':r['plan_diagnostics'].groupby('demand_multiplier').agg(ships_mean=('num_ships','mean'),ships_min=('num_ships','min'),ships_max=('num_ships','max'),
               adjustment_mean=('mean_arrival_adjustment','mean'),adjustment_max=('max_arrival_adjustment','max'),
               latest_arrival=('last_actual_arrival','max'),arrivals_after_period=('arrivals_after_period','sum'),gap_mean=('mean_reserved_gap','mean')).reset_index().to_dict('records'),
          'all_unrestricted_attack_waiting_zero':bool((main.query("policy=='unrestricted'").total_waiting_time.abs()<=TOL).all()),
          'checks':['all ships retained','physical constraints','handling time lookup','only target report changed',
                    'initial plans shared','all eleven metrics reconstructed','summaries and paired differences reconstructed',
                    'normal zero','source hashes','forecast 103/13','five PNG/PDF pairs']}
# Supplementary comparisons are reconstructed on every audit, not manually maintained.
mid = main[(main.target_position=='middle') & (main.shift_hours==4)]
join_key=['demand_multiplier','trial','scenario']
sens=mid[mid.policy=='reassign'].set_index(join_key)[['other_average_waiting_time']].rename(columns={'other_average_waiting_time':'limit2'})
zero=runs[(runs.experiment_group=='sensitivity') & (runs.scenario!='normal')]
sens=sens.join(zero.set_index(join_key).other_average_waiting_time.rename('limit0'))
sens=sens.join(mid[mid.policy=='unrestricted'].set_index(join_key).other_average_waiting_time.rename('unlimited'))
difference=sens.limit2-sens.limit0
analysis['sensitivity_means']=sens.groupby(['demand_multiplier','scenario']).mean().reset_index().to_dict('records')
analysis['sensitivity_2_vs_0']={'improved':int((difference < -TOL).sum()),'worsened':int((difference>TOL).sum()),'unchanged':int((difference.abs()<=TOL).sum())}
case=schedules[(schedules.demand_multiplier==1.5)&(schedules.trial==8)&(schedules.target_position=='early')
               &(schedules.shift_hours==4)&(schedules.scenario=='late')&(schedules.experiment_group=='main')]
b=case[case.policy=='same_berth'].set_index('ship');c_case=case[case.policy=='reassign'].set_index('ship')
joined_case=b[['actual_arrival_time','planned_start_time','planned_berth','assigned_berth','start_time','completion_time','handling_time','waiting_time']].join(c_case[['assigned_berth','start_time','completion_time','handling_time','waiting_time']],rsuffix='_C')
joined_case['extra_wait_C']=joined_case.waiting_time_C-joined_case.waiting_time
joined_case['handling_change_C']=joined_case.handling_time_C-joined_case.handling_time
analysis['worsening_example_details']=joined_case.loc[['V18','V19','V20','V34']].reset_index().to_dict('records')
analysis['worsening_example_target']=case.target_ship.iloc[0]
analysis['worsening_example_ships']=len(b)
attack_target_rows=schedules[(schedules.experiment_group=='main')&(schedules.scenario!='normal')
                             &(schedules.policy=='fixed')&(schedules.is_target==True)]
analysis['clamped_main_cases']=int((attack_target_rows.arrival_shift.abs() < attack_target_rows.shift_hours-TOL).sum())
analysis['largest_other_wait_hours']=main.other_max_waiting_time.max()
analysis['no_effect_main_cases_by_policy']=main.groupby('policy').other_average_waiting_time.apply(lambda v:int((v.abs()<=TOL).sum())).to_dict()

(OUT/'validation.json').write_text(json.dumps(analysis,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(analysis,ensure_ascii=False,indent=2))
