#!/usr/bin/env python3
"""Compare native PX4 RTL climb, return and landing references and responses.

Consumes progress_analysis.json from compare_flight_replay.py. No simulator
settings are modified. Heights are relative to each logged home; upward speed
is positive. LAND's Navigator altitude is not a position-control target.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
from tools.plotting.analyze_replay_dynamics import interpolate
from tools.plotting.flight_replay_progress import changes
from tools.scenario.px4_mission import read_log, parameter_at

PARAMETERS = (
    'RTL_TYPE','RTL_RETURN_ALT','RTL_DESCEND_ALT','RTL_LAND_DELAY','RTL_MIN_DIST','RTL_CONE_ANG',
    'MPC_Z_V_AUTO_UP','MPC_Z_V_AUTO_DN','MPC_Z_VEL_MAX_UP','MPC_Z_VEL_MAX_DN','MPC_Z_P',
    'MPC_Z_VEL_P_ACC','MPC_Z_VEL_I_ACC','MPC_Z_VEL_D_ACC','MPC_ACC_UP_MAX','MPC_ACC_DOWN_MAX',
    'MPC_JERK_AUTO','MPC_LAND_ALT1','MPC_LAND_ALT2','MPC_LAND_ALT3','MPC_LAND_SPEED',
    'MPC_LAND_CRWL','MPC_THR_HOVER','MPC_THR_MAX','MPC_USE_HTE','HTE_THR_RANGE',
    'MC_BAT_SCALE_EN','BAT1_N_CELLS','BAT1_V_EMPTY','BAT1_V_CHARGED','BAT1_CAPACITY',
    'SIM_BAT_ENABLE','SIM_BAT_DRAIN','SIM_BAT_MIN_PCT','COM_DISARM_LAND',
    'COM_RC_IN_MODE','COM_RC_LOSS_T','COM_FAIL_ACT_T','NAV_RCL_ACT','COM_RCL_EXCEPT',
    'COM_LOW_BAT_ACT','COM_FLTT_LOW_ACT','LNDMC_TRIG_TIME')
EVENTS = ('rtl','climb','return','land','ground_contact','landed','disarmed')
STAGES = {0:'Before RTL',1:'Climb',2:'Return',3:'LAND',4:'Landed',5:'Disarmed'}


def home_height(local_z, ref_alt, home_alt):
    """Translate local NED z into height above logged home in the AMSL frame."""
    return np.asarray(ref_alt) - np.asarray(home_alt) - np.asarray(local_z)


def first_event(data, field, value, after, before):
    mask = (np.asarray(data['timestamp']) >= after) & (np.asarray(data['timestamp']) <= before)
    indices = np.flatnonzero(mask & (np.asarray(data[field]) == value))
    if not len(indices):
        raise ValueError(f'No {field}={value} observation in requested RTL window')
    return int(data['timestamp'][indices[0]])


def rtl_events(status, navigator, land, start, end):
    rtl = first_event(status,'nav_state',5,start,end)
    climb = first_event(navigator,'nav_cmd',31,rtl,end)
    returning = first_event(navigator,'nav_cmd',16,climb,end)
    landing = first_event(navigator,'nav_cmd',21,returning,end)
    contact = first_event(land,'ground_contact',1,landing,end)
    landed = first_event(land,'landed',1,contact,end)
    disarmed = first_event(status,'arming_state',1,landed,end)
    return dict(zip(EVENTS,[(stamp-start)/1e6 for stamp in (rtl,climb,returning,landing,contact,landed,disarmed)]))


def stage_series(t, events):
    result=np.zeros(len(t),int)
    for index,event in enumerate(('climb','return','land','landed','disarmed'),1):
        result[np.asarray(t)>=events[event]]=index
    return result


def summary_stats(trace, lower, upper):
    mask=(trace['time_s']>=lower)&(trace['time_s']<upper)
    result={'window_s':[lower,upper],'samples':int(mask.sum()),'mean':{},'min':{},'max':{},'rms':{},'finite_fraction':{}}
    for key,values in trace.items():
        if key in ('time_s','rtl_time_s','land_time_s'):
            continue
        v=values[mask];finite=v[np.isfinite(v)]
        result['finite_fraction'][key]=float(len(finite)/len(v)) if len(v) else 0.
        for op in ('mean','min','max'):
            result[op][key]=float(getattr(np,op)(finite)) if len(finite) else None
    for key in ('trajectory_altitude_error_m','controller_altitude_error_m','velocity_tracking_error_m_s'):
        v=trace[key][mask];v=v[np.isfinite(v)]
        result['rms'][key]=float(np.sqrt(np.mean(v*v))) if len(v) else None
    for key,condition in [('thrust_saturated_fraction',trace['thrust_command']>=.999),
                          ('allocation_not_achieved_fraction',trace['thrust_achieved']<.5)]:
        source='thrust_command' if 'saturated' in key else 'thrust_achieved'
        valid=mask & np.isfinite(trace[source]);result[key]=float(condition[valid].mean()) if valid.any() else None
    return result


def extract(source, begin):
    log,digest=read_log(source['path'])
    if digest!=source['sha256']:
        raise ValueError('Native log hash mismatch')
    bounds=source['bounds'];start,end=bounds['start_time_us'],bounds['end_time_us']
    datasets={d.name:d.data for d in log.data_list if d.multi_id==0}
    for name in ('vehicle_status','navigator_mission_item','vehicle_land_detected','vehicle_local_position','home_position'):
        if name not in datasets:
            raise ValueError(f'Missing required topic: {name}')
    events=rtl_events(datasets['vehicle_status'],datasets['navigator_mission_item'],datasets['vehicle_land_detected'],start,end)
    t=np.arange(begin,(end-start)/1e6,.05)
    gaps={}

    def sample(topic,fields,hold=False,max_gap=.5):
        data=datasets.get(topic)
        if data is None or not all(f in data for f in fields):
            return np.full((len(t),len(fields)),np.nan)
        time=(data['timestamp'].astype(float)-start)/1e6
        v=np.column_stack([data[f] for f in fields]).astype(float)
        if topic=='vehicle_local_position':
            for f in ('z_valid','v_z_valid'):
                if f in data:v[~data[f].astype(bool)]=np.nan
        inside=time[(time>=begin)&(time<=(end-start)/1e6)]
        gaps[topic]=float(np.max(np.diff(inside))) if len(inside)>1 else None
        return interpolate(time,v,t,hold=hold,max_gap=max_gap)

    h=sample('home_position',['alt','z'],hold=True,max_gap=np.inf)
    lp=sample('vehicle_local_position',['z','vz','az','ref_alt','dist_bottom','dist_bottom_valid'])
    gp=sample('vehicle_global_position',['alt'])[:,0]
    ts=sample('trajectory_setpoint',['position[2]','velocity[2]','acceleration[2]'])
    cs=sample('vehicle_local_position_setpoint',['z','vz','acceleration[2]'])
    nav=sample('position_setpoint_triplet',['current.alt','current.type','current.valid'],hold=True,max_gap=np.inf)
    altitude=home_height(lp[:,0],lp[:,3],h[:,0]);trajectory=home_height(ts[:,0],lp[:,3],h[:,0]);controller=home_height(cs[:,0],lp[:,3],h[:,0])
    navigator=nav[:,0]-h[:,0];navigator[(nav[:,2]<.5)|(nav[:,1]==4)]=np.nan
    q=sample('vehicle_attitude',[f'q[{i}]' for i in range(4)])
    tilt=np.rad2deg(np.arccos(np.clip(1-2*np.sum(q[:,1:3]**2,axis=1)/np.sum(q**2,axis=1),-1,1)))
    motors=sample('actuator_motors',[f'control[{i}]' for i in range(4)])
    status=sample('vehicle_status',['nav_state','arming_state'],hold=True,max_gap=5.)
    battery=sample('battery_status',['voltage_v','current_a','remaining','warning'])
    allocator=sample('control_allocator_status',['thrust_setpoint_achieved','unallocated_thrust[2]'],hold=True)
    limits=sample('vehicle_constraints',['speed_up','speed_down'],hold=True,max_gap=3.)
    trace=dict(tilt_deg=tilt,time_s=t,rtl_time_s=t-events['rtl'],land_time_s=t-events['land'],
        actual_altitude_m=altitude,global_altitude_m=gp-h[:,0],trajectory_altitude_m=trajectory,
        controller_altitude_m=controller,navigator_altitude_m=navigator,
        raw_navigator_altitude_m=nav[:,0]-h[:,0],navigator_type=nav[:,1],
        home_z_height_m=h[:,1]-lp[:,0],dist_bottom_m=lp[:,4],dist_bottom_valid=lp[:,5],
        actual_up_m_s=-lp[:,1],trajectory_up_m_s=-ts[:,1],controller_up_m_s=-cs[:,1],
        actual_up_acceleration_m_s2=-lp[:,2],controller_up_acceleration_m_s2=-cs[:,2],
        trajectory_altitude_error_m=altitude-trajectory,controller_altitude_error_m=altitude-controller,
        velocity_tracking_error_m_s=-lp[:,1]+cs[:,1],
        thrust_command=-sample('vehicle_attitude_setpoint',['thrust_body[2]'])[:,0],
        hover_thrust=sample('hover_thrust_estimate',['hover_thrust'])[:,0],motor_max=np.max(motors,axis=1),
        thrust_achieved=allocator[:,0],unallocated_thrust_up=-allocator[:,1],
        voltage_v=battery[:,0],current_a=np.where(battery[:,1]>=0,battery[:,1],np.nan),
        battery_remaining=battery[:,2],battery_warning=sample('battery_status',['warning'],hold=True)[:,0],nav_state=status[:,0],arming_state=status[:,1],
        constraint_speed_up=limits[:,0],constraint_speed_down=limits[:,1],stage=stage_series(t,events))
    durations={name:events[b]-events[a] for name,a,b in [
        ('rtl_to_return','rtl','return'),('climb','climb','return'),('return','return','land'),
        ('land_to_contact','land','ground_contact'),('land_to_landed','land','landed'),
        ('landed_to_disarm','landed','disarmed'),('rtl_to_disarm','rtl','disarmed')]}
    metrics={name:summary_stats(trace,events[a],events[b]) for name,a,b in
             [('climb','climb','return'),('return','return','land'),('landing','land','landed')]}
    metrics['window_70_75s']=summary_stats(trace,70,75)
    metrics['window_81_85s']=summary_stats(trace,81,85)
    metrics['land_first_2_8s']=summary_stats(trace,events['land']+2,events['land']+8)
    metrics['pre_rtl_60_63s']=summary_stats(trace,60,63)
    parameters={}
    for key in PARAMETERS:
        try:parameters[key]=parameter_at(log,key,start+events['rtl']*1e6)
        except ValueError:parameters[key]=None
    native_events={}
    for topic,fields in [('vehicle_status',['nav_state','arming_state']),('navigator_mission_item',['nav_cmd','altitude','altitude_is_relative']),('vehicle_land_detected',['ground_contact','maybe_landed','landed']),('battery_status',['warning']),('failsafe_flags',['manual_control_signal_lost','battery_warning','battery_low_remaining_time']),('home_position',['alt','z'])]:
        if topic in datasets:native_events[topic]=changes(datasets[topic],fields,bounds)
    messages=[{'time_s':(m.timestamp-start)/1e6,'message':m.message} for m in log.logged_messages if start+begin*1e6<=m.timestamp<=end]
    return trace,dict(source=source,events=events,durations_s=durations,metrics=metrics,parameters_at_rtl=parameters,
                      native_events=native_events,logged_messages=messages,
                      quality={'dropouts':len(log.dropouts),'max_topic_gap_s':gaps})


def paired_metrics(traces,info):
    result={}
    real,sitl=traces['real'],traces['sitl']
    for label,align in [('mission_clock',False),('land_entry_aligned',True)]:
        rt=real['land_time_s'] if align else real['time_s'];st=sitl['land_time_s'] if align else sitl['time_s']
        lo=0 if align else max(info[n]['events']['rtl'] for n in info)
        hi=min(info[n]['events']['landed']-info[n]['events']['land'] for n in info) if align else min(rt[-1],st[-1])
        times=np.arange(lo,hi,.05);entry={'window_s':[float(times[0]),float(times[-1])],'samples':len(times)}
        for key in ('actual_altitude_m','trajectory_altitude_m','actual_up_m_s','trajectory_up_m_s','controller_up_m_s'):
            a=interpolate(rt,real[key],times)[:,0];b=interpolate(st,sitl[key],times)[:,0];delta=b-a;finite=delta[np.isfinite(delta)]
            entry[key+'_rmse']=float(np.sqrt(np.mean(finite**2))) if len(finite) else None
            entry[key+'_max_abs']=float(np.max(np.abs(finite))) if len(finite) else None
            entry[key+'_finite_samples']=len(finite)
        result[label]=entry
    return result


def plot_results(traces, info, output, sitl_label='SITL'):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 10, 'axes.grid': True, 'grid.alpha': .22})
    colors = {'real': '#1469aa', 'sitl': '#cf691a'}
    names = {'real': 'Real', 'sitl': sitl_label}
    lo = min(x['time_s'][0] for x in traces.values())
    hi = max(x['events']['disarmed'] for x in info.values()) + .5

    def decorate(ax, name, title, ylabel, markers=True):
        ax.set(title=title, ylabel=ylabel, xlim=(lo, hi))
        if markers:
            for event, color in [('rtl','#555555'), ('return','#915d9f'), ('land','#b32e3b'), ('landed','#308b55')]:
                ax.axvline(info[name]['events'][event], color=color, lw=.85, ls=':', alpha=.8)
        handles, _ = ax.get_legend_handles_labels()
        if handles:
            ax.legend(loc='best', fontsize=8)

    def save(fig, stem, title):
        fig.suptitle(title, fontsize=14)
        fig.tight_layout(rect=(0, 0, 1, .965))
        for suffix in ('png', 'svg'):
            fig.savefig(output / f'{stem}.{suffix}', dpi=170)
        plt.close(fig)

    fig, axes = plt.subplots(3, 2, figsize=(15, 11), sharex=True, sharey='row')
    for col, (name, trace) in enumerate(traces.items()):
        t = trace['time_s']
        for key, label, color, ls in [
            ('navigator_altitude_m','Navigator destination','#777777',':'),
            ('trajectory_altitude_m','Trajectory reference','#e19c24','--'),
            ('controller_altitude_m','Controller position reference','#499858','-.'),
            ('actual_altitude_m','Actual (local position)',colors[name],'-')]:
            axes[0,col].plot(t, trace[key], label=label, color=color, ls=ls, lw=1.5)
        decorate(axes[0,col], name, names[name], 'Height above logged home [m]')
        axes[0,col].legend(loc='lower left', fontsize=8)
        for event, label in [('rtl','RTL'),('return','Return'),('land','LAND'),('landed','Landed')]:
            axes[0,col].text(info[name]['events'][event]+.15, 33, label, rotation=90, va='top', fontsize=8)
        axes[0,col].set_ylim(-1.5, 35)
        for key,label,color,ls in [
            ('trajectory_up_m_s','Trajectory velocity','#e19c24','--'),
            ('controller_up_m_s','Controller velocity demand','#499858','-.'),
            ('actual_up_m_s','Actual (logged vz)',colors[name],'-')]:
            axes[1,col].plot(t,trace[key],label=label,color=color,ls=ls,lw=1.5)
        axes[1,col].axhline(0,color='#666666',lw=.5)
        decorate(axes[1,col],name,'Vertical velocity: up positive','Vertical velocity [m/s]')
        axes[2,col].plot(t,trace['trajectory_altitude_error_m'],label='Actual - trajectory',color=colors[name])
        axes[2,col].plot(t,trace['controller_altitude_error_m'],label='Actual - controller position',color='#499858',ls='--')
        axes[2,col].axhline(0,color='#666666',lw=.5)
        decorate(axes[2,col],name,'Position tracking error','Height error [m]')
        axes[2,col].set_xlabel('Time since AUTO_MISSION entry [s]')
    save(fig,'rtl_vertical_comparison','RTL vertical references and response | same mission clock; no time rescaling')

    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
    for name, trace in traces.items():
        mode_events=info[name]['native_events']['vehicle_status']
        axes[0].step([e['time_s'] for e in mode_events],[e['nav_state'] for e in mode_events],where='post',color=colors[name],label=names[name],lw=1.8,ls='-' if name=='real' else '--')
        stage_times=[lo]+[info[name]['events'][e] for e in ('climb','return','land','landed','disarmed')]
        axes[1].step(stage_times,list(STAGES),where='post',color=colors[name],label=names[name],lw=1.8,ls='-' if name=='real' else '--')
    axes[0].set(yticks=[3,4,5],yticklabels=['AUTO_MISSION (3)','AUTO_LOITER / HOLD (4)','AUTO_RTL (5)'],title='Logged vehicle_status.nav_state (post-disarm state also shown)')
    axes[1].set(yticks=list(STAGES),yticklabels=list(STAGES.values()),title='RTL work items from navigator_mission_item + land detector',xlabel='Time since AUTO_MISSION entry [s]',xlim=(lo,hi))
    for ax in axes:ax.legend(loc='best')
    save(fig,'rtl_timeline','Mode transitions and internal RTL stages | LAND remains AUTO_RTL')

    fig, axes = plt.subplots(4,2,figsize=(15,12),sharex=True,sharey='row')
    for col,(name,trace) in enumerate(traces.items()):
        t=trace['time_s']
        axes[0,col].plot(t,trace['voltage_v'],color=colors[name],label='battery_status.voltage_v')
        decorate(axes[0,col],name,names[name],'Logged battery voltage [V]')
        for key,label,color,ls in [('thrust_command','Collective thrust demand',colors[name],'-'),('motor_max','Maximum motor command','#777777',':'),('hover_thrust','Estimated hover thrust','#499858','--')]:
            axes[1,col].plot(t,trace[key],label=label,color=color,ls=ls)
        axes[1,col].axhline(1,color='#b32e3b',lw=.6)
        decorate(axes[1,col],name,'Normalized commands (not measured force)','Normalized command')
        axes[2,col].step(t,trace['thrust_achieved'],where='post',color=colors[name],label='thrust_setpoint_achieved')
        axes[2,col].set(yticks=[0,1],yticklabels=['No','Yes'])
        decorate(axes[2,col],name,'Allocator command feasibility (not force feedback)','Thrust demand achieved')
        axes[3,col].step(t,trace['battery_warning'],where='post',color=colors[name])
        axes[3,col].set(yticks=[0,1,2,3],yticklabels=['None','Low','Critical','Emergency'])
        decorate(axes[3,col],name,'Logged battery warning','battery_status.warning')
        axes[3,col].set_xlabel('Time since AUTO_MISSION entry [s]')
    save(fig,'rtl_power_comparison','RTL power demand and battery telemetry | simulated voltage does not measure physical motor power')

    fig, axes = plt.subplots(3,1,figsize=(12,11),sharex=True)
    for name,trace in traces.items():
        end=info[name]['durations_s']['land_to_landed']
        mask=(trace['land_time_s']>=0)&(trace['land_time_s']<=end)
        t=trace['land_time_s'][mask];color=colors[name]
        axes[0].plot(t,trace['actual_altitude_m'][mask],color=color,label=names[name]+' actual')
        axes[0].plot(t,trace['trajectory_altitude_m'][mask],color=color,ls='--',alpha=.7,label=names[name]+' trajectory')
        axes[1].plot(t,trace['actual_up_m_s'][mask],color=color,label=names[name]+' actual')
        axes[1].plot(t,trace['trajectory_up_m_s'][mask],color=color,ls='--',alpha=.7,label=names[name]+' trajectory')
        axes[2].plot(t,trace['velocity_tracking_error_m_s'][mask],color=color,label=names[name])
        for ax in axes:ax.axvline(end,color=color,ls=':',lw=.8)
    for ax,ylabel in zip(axes,['Height above logged home [m]','Vertical velocity [m/s], up positive','Actual - controller demand [m/s]']):
        ax.set_ylabel(ylabel);ax.legend(loc='best',fontsize=8)
    axes[-1].set_xlabel('Time since each logged Navigator LAND entry [s]')
    save(fig,'landing_aligned','LAND-entry alignment (diagnostic only) | removes the earlier climb delay; no duration scaling')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--progress',type=Path,required=True,help='progress_analysis.json from compare_flight_replay.py')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--phase',default='flight_000')
    parser.add_argument('--sitl-label',default='SITL')
    parser.add_argument('--start-s',type=float,default=60.,help='Plot/statistical extraction start, seconds since AUTO_MISSION entry')
    args=parser.parse_args()
    progress=json.loads(args.progress.read_text())
    phase=next((p for p in progress['phases'] if p['id']==args.phase),None)
    if phase is None:parser.error(f'Unknown phase: {args.phase}')
    traces,info={},{}
    for name in ('real','sitl'):
        traces[name],info[name]=extract(phase['sources'][name],args.start_s)
        if args.start_s>info[name]['events']['rtl']:
            parser.error('--start-s must be no later than the RTL entry')
    args.output.mkdir(parents=True,exist_ok=True)
    comparison=paired_metrics(traces,info)
    param_diff={key:{name:info[name]['parameters_at_rtl'][key] for name in info}
                for key in PARAMETERS if info['real']['parameters_at_rtl'][key]!=info['sitl']['parameters_at_rtl'][key]}
    result=dict(phase=args.phase,source_progress=str(args.progress.resolve()),
        convention={'time':'seconds since respective AUTO_MISSION entry; unchanged phase bounds',
                    'height':'local ref_alt - home_position.alt - local z; home-relative, not absolute AMSL',
                    'velocity':'up positive = -logged NED vz; no position differentiation',
                    'trajectory':'trajectory_setpoint.position[2] and velocity[2]',
                    'controller':'vehicle_local_position_setpoint.z and vz (includes position correction)',
                    'rtl_stage':'Navigator nav_cmd 31 -> 16 -> 21, then land detector / disarm',
                    'resampling':'20 Hz; linear continuous fields, previous-sample discrete fields, no extrapolation',
                    'thrust':'-vehicle_attitude_setpoint.thrust_body[2], normalized command, not measured force',
                    'allocator':'thrust_setpoint_achieved describes command allocation, not measured thrust',
                    'landing_alignment':'diagnostic alignment to LAND entry, never scales elapsed time'},
        sources=info,comparison=comparison,parameter_differences=param_diff)
    (args.output/'rtl_summary.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    for name,trace in traces.items():
        with (args.output/f'{name}_rtl_trace.csv').open('w',newline='') as f:
            writer=csv.writer(f);writer.writerow(trace);writer.writerows(zip(*trace.values()))
    with (args.output/'transition_times.csv').open('w',newline='') as f:
        writer=csv.writer(f);writer.writerow(['event','real_s','sitl_s','sitl_minus_real_s'])
        for event in EVENTS:
            r,s=(info[n]['events'][event] for n in ('real','sitl'))
            writer.writerow([event,r,s,s-r])
    plot_results(traces,info,args.output,args.sitl_label)
    print(json.dumps({'output':str(args.output.resolve()),'events':{n:info[n]['events'] for n in info},
                      'durations_s':{n:info[n]['durations_s'] for n in info},'comparison':comparison},indent=2))


if __name__=='__main__':
    main()
