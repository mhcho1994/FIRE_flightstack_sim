#!/usr/bin/env python3
"""Compare PX4 replay velocity, attitude and propulsion response from native ULogs.

Inputs are progress_analysis.json files emitted by compare_flight_replay.py.
See --help. Motor speed feedback in this repository's Gazebo bridge is a command
loopback in rad/s, despite the esc_rpm field name: never treat it as measured RPM.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from tools.scenario.px4_mission import read_log, parameter_at

G = 9.80665


def interpolate(t, values, target, *, hold=False, max_gap=.5):
    """No extrapolation or interpolation across missing/invalid source samples."""
    t, values, target = np.asarray(t), np.asarray(values, float), np.asarray(target)
    if values.ndim == 1:
        values = values[:, None]
    if not len(t):
        return np.full((len(target), values.shape[1]), np.nan)
    order = np.argsort(t, kind='stable')
    t, values = t[order], values[order]
    # Keep the latest publication at a timestamp; this is diagnostic resampling,
    # not the strict state audit used by compare_flight_replay.
    keep = np.r_[np.diff(t) > 0, True]
    t, values = t[keep], values[keep]
    right = np.searchsorted(t, target, side='right')
    left = np.clip(right - 1, 0, len(t) - 1)
    right = np.clip(right, 0, len(t) - 1)
    inside = (target >= t[0]) & (target <= t[-1])
    if hold:
        result = values[left].copy()
        valid = inside & ((target-t[left]) <= max_gap)
    else:
        frac = np.divide(target-t[left], t[right]-t[left],
                         out=np.zeros(len(target)), where=t[right] > t[left])
        result = values[left] + frac[:, None] * (values[right]-values[left])
        exact = target == t[left]
        result[exact] = values[left[exact]]
        valid = inside & (((t[right]-t[left]) <= max_gap) | (target == t[left]))
    result[~valid] = np.nan
    return result


def quaternion_matrix(q):
    """Hamilton wxyz, FRD body to NED local frame."""
    q = q / np.linalg.norm(q, axis=1)[:, None]
    w, x, y, z = q.T
    return np.stack([1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y),
                     2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x),
                     2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)], axis=1).reshape(-1, 3, 3)


def attitude_degrees(q):
    r = quaternion_matrix(q)
    return np.rad2deg(np.column_stack([np.arctan2(r[:, 2, 1], r[:, 2, 2]),
            np.arcsin(np.clip(-r[:, 2, 0], -1, 1)), np.arctan2(r[:, 1, 0], r[:, 0, 0])]))


def model_constants(path):
    root = ET.parse(path).getroot()
    rotors = [p for p in root.iter('plugin') if p.get('name', '').endswith('::MulticopterMotorModel')]
    rotors.sort(key=lambda p: int(p.findtext('actuator_number')))
    if len(rotors) != 4:
        raise ValueError('Force reconstruction requires a four-rotor model')
    body = [p for p in root.iter('plugin') if p.get('filename') == 'FireBodyDragSystem']
    return dict(path=str(Path(path).resolve()), sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest(),
        mass_kg=sum(float(p.text) for p in root.findall('.//link/inertial/mass')),
        motor_constant=[float(p.findtext('motorConstant')) for p in rotors],
        max_rot_velocity=[float(p.findtext('maxRotVelocity')) for p in rotors],
        rotor_drag_coefficient=[float(p.findtext('rotorDragCoefficient')) for p in rotors],
        body_drag_factor=np.sum([.5*float(p.findtext('air_density'))*
            np.fromstring(p.findtext('drag_coefficient'), sep=' ')*
            np.fromstring(p.findtext('reference_area'), sep=' ') for p in body], axis=0).tolist())


def extract(source, window, steady, model=None):
    log, digest = read_log(source['path'])
    if digest != source['sha256']:
        raise ValueError(f"Log hash mismatch: {source['path']}")
    start = source['bounds']['start_time_us']
    duration = (source['bounds']['end_time_us']-start)/1e6
    t = np.arange(0, min(duration, max(window[1], steady[1])+2), .02)
    gaps = {}

    def raw(topic):
        try:
            return log.get_dataset(topic).data
        except (KeyError, IndexError):
            return None

    def sample(topic, fields, *, quat=False, hold=False, max_gap=.5):
        d = raw(topic)
        if d is None or not all(f in d for f in fields):
            return np.full((len(t), len(fields)), np.nan)
        times = (d['timestamp'].astype(float)-start)/1e6
        values = np.column_stack([d[f] for f in fields]).astype(float)
        if topic == 'vehicle_local_position':
            for valid in ('xy_valid', 'z_valid', 'v_xy_valid', 'v_z_valid'):
                if valid in d:
                    values[~d[valid].astype(bool)] = np.nan
        if quat:
            values /= np.linalg.norm(values, axis=1)[:, None]
            signs = np.ones(len(values))
            signs[1:] = np.cumprod(np.where(np.sum(values[:-1]*values[1:], axis=1) < 0, -1., 1.))
            values *= signs[:, None]
        inside = times[(times >= window[0]) & (times <= window[1])]
        gaps[topic] = float(np.max(np.diff(inside))) if len(inside)>1 else None
        result = interpolate(times, values, t, hold=hold, max_gap=max_gap)
        if quat:
            result /= np.linalg.norm(result, axis=1)[:, None]
        return result

    fields = lambda p, n=3: [f'{p}[{i}]' for i in range(n)]
    q = sample('vehicle_attitude', fields('q', 4), quat=True)
    qsp = sample('vehicle_attitude_setpoint', fields('q_d', 4), quat=True)
    r = quaternion_matrix(q)
    v = sample('vehicle_local_position', ['vx', 'vy', 'vz'])
    vsp = sample('vehicle_local_position_setpoint', ['vx', 'vy', 'vz'])
    ff = sample('trajectory_setpoint', fields('velocity'))
    asp = sample('vehicle_local_position_setpoint', fields('acceleration'))
    a = sample('vehicle_local_position', ['ax', 'ay', 'az'])
    fbody = sample('vehicle_acceleration', fields('xyz'))
    fworld = np.einsum('nij,nj->ni', r, fbody)
    # A body-z specific force proxy, not a load-cell/physical thrust measurement.
    thrust_proxy = -fbody[:, 2]
    thrust_proxy_world = -r[:, :, 2]*thrust_proxy[:, None]
    imu_a = fworld + [0, 0, G]
    direction_samples = ff[(t>=steady[0]) & (t<steady[1]), :2]
    direction = np.nanmean(direction_samples, axis=0)
    if not np.all(np.isfinite(direction)) or np.linalg.norm(direction) < .1:
        raise ValueError('Steady window needs a finite nonzero horizontal trajectory velocity')
    direction /= np.linalg.norm(direction)
    along = lambda x: x[:, :2] @ direction
    angles, angles_sp = attitude_degrees(q), attitude_degrees(qsp)
    tilt = np.rad2deg(np.arccos(np.clip(r[:, 2, 2], -1, 1)))
    tiltsp = np.rad2deg(np.arccos(np.clip(quaternion_matrix(qsp)[:, 2, 2], -1, 1)))
    att_error = np.rad2deg(2*np.arccos(np.clip(abs(np.sum(q*qsp, axis=1)), 0, 1)))
    rate = sample('vehicle_angular_velocity', fields('xyz'))
    rate_sp = sample('vehicle_rates_setpoint', ['roll','pitch','yaw'])
    thrust_cmd = -sample('vehicle_attitude_setpoint', ['thrust_body[2]'])[:, 0]
    motor = sample('actuator_motors', fields('control', 4))
    hover = sample('hover_thrust_estimate', ['hover_thrust'])[:, 0]
    allocator = sample('control_allocator_status', ['thrust_setpoint_achieved','torque_setpoint_achieved'], hold=True)
    mode = sample('vehicle_status', ['nav_state'], hold=True, max_gap=10.)[:, 0]
    item = sample('mission_result', ['seq_current'], hold=True, max_gap=10.)[:, 0]
    ff_speed = np.linalg.norm(ff[:, :2], axis=1)
    onset_indices = np.flatnonzero((ff_speed>=.5) & (mode==3) & (t>5))
    if not len(onset_indices):
        raise ValueError('No horizontal acceleration onset found')
    onset = float(t[onset_indices[0]])
    data = dict(time_s=t, event_time_s=t-onset, speed=np.linalg.norm(v[:, :2],axis=1),
        speed_sp=np.linalg.norm(vsp[:, :2],axis=1), speed_ff=ff_speed,
        velocity_error=np.linalg.norm(v[:, :2]-vsp[:, :2],axis=1),
        acceleration_along=along(a), acceleration_sp_along=along(asp),
        imu_acceleration_along=along(imu_a), tilt=tilt, tilt_sp=tiltsp, attitude_error=att_error,
        roll=angles[:,0], pitch=angles[:,1], yaw=angles[:,2],
        roll_sp=angles_sp[:,0], pitch_sp=angles_sp[:,1], yaw_sp=angles_sp[:,2],
        rate_error=np.rad2deg(np.linalg.norm(rate-rate_sp,axis=1)),
        thrust_command=thrust_cmd, hover_thrust=hover, body_z_specific_force=thrust_proxy,
        thrust_proxy_horizontal=along(thrust_proxy_world),
        opposing_specific_force=along(thrust_proxy_world-fworld),
        motor_min=np.min(motor,axis=1), motor_max=np.max(motor,axis=1),
        thrust_achieved=allocator[:,0], torque_achieved=allocator[:,1], mode=mode,item=item)
    if model:
        omega = np.minimum(np.abs(sample('esc_status', [f'esc[{i}].esc_rpm' for i in range(4)])), model['max_rot_velocity'])
        # The local bridge echoes rad/s commands, not measured rotor speed.
        # Quasi-static estimates omit motor filtering and angular point velocity.
        rg = sample('vehicle_attitude_groundtruth', fields('q',4), quat=True)
        vg = sample('vehicle_local_position_groundtruth', ['vx','vy','vz'])
        if np.isfinite(rg).any() and np.isfinite(vg).any():
            rm, vm = quaternion_matrix(rg), vg
            data['estimate_groundtruth_speed_error'] = np.linalg.norm(v-vg,axis=1)
            data['estimate_groundtruth_attitude_error'] = np.rad2deg(2*np.arccos(np.clip(abs(np.sum(q*rg,axis=1)),0,1)))
            angle_error = (attitude_degrees(q)-attitude_degrees(rg)+180)%360-180
            data['estimate_groundtruth_yaw_error'] = angle_error[:,2]
            data['estimate_groundtruth_roll_error'] = angle_error[:,0]
            data['estimate_groundtruth_pitch_error'] = angle_error[:,1]
        else:
            rm, vm = r, v
        body_v = np.einsum('nji,nj->ni',rm,vm)
        body_force = -np.array(model['body_drag_factor'])*np.abs(body_v)*body_v
        body_force_world = np.einsum('nij,nj->ni',rm,body_force)
        perpendicular = vm - rm[:,:,2]*np.sum(vm*rm[:,:,2],axis=1)[:,None]
        rotor_force = -np.sum(omega*np.array(model['rotor_drag_coefficient']),axis=1)[:,None]*perpendicular
        total_thrust = np.sum(omega**2*np.array(model['motor_constant']),axis=1)
        data.update(model_body_drag_along_N=-along(body_force_world),
            model_body_drag_up_N=-body_force_world[:,2], model_rotor_drag_up_N=-rotor_force[:,2],
            model_thrust_up_N=rm[:,2,2]*total_thrust,
            model_rotor_drag_along_N=-along(rotor_force), model_thrust_N=total_thrust,
            model_thrust_acceleration=total_thrust/model['mass_kg'],
            model_net_acceleration_along=along(-rm[:,:,2]*total_thrust[:,None]+body_force_world+rotor_force)/model['mass_kg'])
    stats = {'pre_acceleration_heading_deg': float(np.nanmean(angles[(t>=onset-2)&(t<onset-1),2]))}
    for label, limits in [('acceleration_event', (onset,onset+3)), ('steady',steady), ('window',window)]:
        mask = (t>=limits[0]) & (t<limits[1])
        def mean(k):
            arr=data[k][mask];return float(np.nanmean(arr)) if np.isfinite(arr).any() else None
        def rms(k):
            arr=data[k][mask];return float(np.sqrt(np.nanmean(arr**2))) if np.isfinite(arr).any() else None
        def maximum(k):
            arr=data[k][mask];return float(np.nanmax(arr)) if np.isfinite(arr).any() else None
        stats[label]=dict(window_s=list(limits),samples=int(mask.sum()),
            observed_modes=np.unique(mode[mask & np.isfinite(mode)]).tolist(),
            observed_items=np.unique(item[mask & np.isfinite(item)]).tolist(),
            mean={k:mean(k) for k in data if k not in ('time_s','event_time_s')},
            rms={k:rms(k) for k in ('velocity_error','attitude_error','rate_error')},
            maximum={k:maximum(k) for k in ('tilt_sp','thrust_command','motor_max')},
            finite_fraction={k:float(np.isfinite(data[k][mask]).mean()) for k in data})
    for threshold in (4.,4.5):
        hit=np.flatnonzero((t>=onset)&(t<onset+12)&(data['speed']>=threshold))
        stats[f'time_to_{threshold:g}m_s_after_onset']=float(t[hit[0]]-onset) if len(hit) else None
    warnings=[dict(time_s=(m.timestamp-start)/1e6,message=m.message) for m in log.logged_messages
              if any(w in m.message.lower() for w in ('timeout','fail:'))]
    parameters={key:parameter_at(log,key,start) for key in
        ('MPC_TILTMAX_AIR','MPC_THR_HOVER','MPC_USE_HTE','THR_MDL_FAC','MPC_THR_MAX','MPC_ACC_DECOUPLE')}
    return data,dict(source=source,model=model,acceleration_onset_s=onset,
        onset_definition='First AUTO_MISSION horizontal trajectory velocity >= 0.5 m/s after t=5 s',
        along_track_ned_xy=direction.tolist(),metrics=stats,parameters=parameters,
        log_quality=dict(dropouts=len(log.dropouts),warnings=warnings,max_topic_gap_s=gaps))


def plot(out, traces, window):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    labels=list(traces)
    for selected,filename,event in [(labels[:3],'attitude_thrust_comparison',False),
                                    (labels[:3],'acceleration_event_comparison',True)]:
        fig,axes=plt.subplots(7,len(selected),figsize=(5.3*len(selected),19),layout='constrained',squeeze=False)
        rows=[(['speed_ff','speed_sp','speed'],['Trajectory','Controller','Actual'],'Horizontal speed [m/s]'),
              (['roll_sp','roll'],['Target','Actual'],'Roll [deg]'),
              (['pitch_sp','pitch'],['Target','Actual'],'Pitch [deg]'),
              (['yaw_sp','yaw'],['Target','Actual'],'Yaw [deg]'),
              (['thrust_command','hover_thrust','motor_min','motor_max'],['Thrust target','Hover estimate','Motor min','Motor max'],'Normalized command'),
              (['body_z_specific_force','model_thrust_acceleration'],['Body-z specific force proxy','Model from motor commands'],'Specific force [m/s²]'),
              (['acceleration_sp_along','acceleration_along','opposing_specific_force'],['Controller accel target','Actual logged accel','Opposing force / mass'],'Along-track [m/s²]')]
        for col,label in enumerate(selected):
            d=traces[label];x=d['event_time_s'] if event else d['time_s']
            for row,(keys,names,ylabel) in enumerate(rows):
                ax=axes[row,col]
                for key,name in zip(keys,names):
                    if key in d:ax.plot(x,d[key],label=name,lw=1.2,ls='--' if key.endswith('_sp') else '-')
                ax.set(xlim=(-1,10) if event else window,ylabel=ylabel)
                if row==0:ax.set_title(label)
                if row==6:ax.set_xlabel('Time from acceleration reference onset [s]' if event else 'Time from AUTO_MISSION entry [s]')
                ax.grid(alpha=.25);ax.legend(fontsize=7)
        # Shared row limits permit direct visual comparison.
        for row in range(len(rows)):
            vals=[]
            for label in selected:
                d=traces[label];x=d['event_time_s'] if event else d['time_s'];lo,hi=(-1,10) if event else window
                for key in rows[row][0]:
                    if key in d:vals.extend(d[key][(x>=lo)&(x<=hi)].tolist())
            vals=np.asarray(vals);vals=vals[np.isfinite(vals)]
            if len(vals):
                low,high=np.min(vals),np.max(vals);pad=max((high-low)*.08,.05)
                for ax in axes[row]:ax.set_ylim(low-pad,high+pad)
        fig.suptitle('Flight replay dynamics: native log states; body-z specific force is NOT a thrust sensor')
        for ext in ('png','svg'):fig.savefig(out/f'{filename}.{ext}',dpi=130)
        plt.close(fig)
    fig,axes=plt.subplots(3,2,figsize=(15,12),layout='constrained')
    for label,d in traces.items():
        x=d['event_time_s']
        for ax,key,title,units in zip(axes.flat,
            ['speed','velocity_error','tilt','attitude_error','thrust_command','opposing_specific_force'],
            ['Actual horizontal speed','Controller velocity tracking error','Actual tilt','Attitude tracking error','Normalized thrust target','Opposing specific force along track'],
            ['m/s','m/s','deg','deg','normalized','m/s²']):
            ax.plot(x,d[key],label=label,lw=1.3);ax.set(xlim=(-1,10),title=title,ylabel=units,xlabel='Time from acceleration reference onset [s]');ax.grid(alpha=.25)
    for ax in axes.flat:ax.legend(fontsize=8)
    for ext in ('png','svg'):fig.savefig(out/f'physics_sensitivity.{ext}',dpi=150)
    plt.close(fig)


def plot_force_balance(out, traces, window):
    import matplotlib.pyplot as plt
    selected=[label for label,d in traces.items() if 'model_rotor_drag_along_N' in d]
    if not selected:
        return
    fig,axes=plt.subplots(2,len(selected),figsize=(4.5*len(selected),8),layout='constrained',squeeze=False)
    for col,label in enumerate(selected):
        d=traces[label]
        for row,keys,names in [(0,['model_rotor_drag_along_N','model_body_drag_along_N'],['Rotor drag','Body drag']),
                              (1,['model_thrust_up_N','model_rotor_drag_up_N','model_body_drag_up_N'],['Thrust vertical','Rotor drag vertical','Body drag vertical'])]:
            ax=axes[row,col]
            for key,name in zip(keys,names):ax.plot(d['time_s'],d[key],label=name)
            ax.set(xlim=window,title=label,ylabel='Along-track resistance [N]' if row==0 else 'Upward force [N]',xlabel='Time from AUTO_MISSION entry [s]')
            ax.grid(alpha=.25);ax.legend(fontsize=8)
    fig.suptitle('Approximate plant forces from SDF + motor commands + ground truth motion; not force measurements')
    for ext in ('png','svg'):fig.savefig(out/f'force_balance.{ext}',dpi=140)
    plt.close(fig)


def mappings(entries):
    result={}
    for entry in entries:
        name,path=entry.split('=',1)
        if name in result:raise ValueError(f'Duplicate label: {name}')
        result[name]=Path(path)
    return result


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--report',action='append',required=True,metavar='LABEL=PROGRESS_JSON')
    p.add_argument('--model',action='append',default=[],metavar='LABEL=SDF',help='Optional SDF snapshot for approximate SITL force reconstruction')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--window',nargs=2,type=float,default=[12,28])
    p.add_argument('--steady',nargs=2,type=float,default=[20,25])
    p.add_argument('--phase',default='flight_000')
    args=p.parse_args(argv)
    if args.window[0]<0 or args.window[1]<=args.window[0] or args.steady[1]<=args.steady[0]:p.error('Invalid time window')
    sources={};models=mappings(args.model)
    for label,path in mappings(args.report).items():
        if label == 'Real':
            p.error('Real is reserved for the shared flight log')
        report=json.loads(path.read_text());phase=next(q for q in report['phases'] if q['id']==args.phase)
        real=phase['sources']['real']
        if 'Real' in sources and sources['Real'] != real:raise ValueError('Reports use different real log sources or phase bounds')
        sources.setdefault('Real',real);sources[label]=phase['sources']['sitl']
    if set(models)-set(sources) or 'Real' in models:
        p.error('--model labels must identify a SITL --report label')
    out=args.output;out.mkdir(parents=True,exist_ok=True)
    traces={};summary=dict(window_s=args.window,steady_window_s=args.steady,sources={},notes=[
        'Actual speed: vehicle_local_position.vx/vy; no position-derived speed.',
        'Attitude: vehicle_attitude.q vs vehicle_attitude_setpoint.q_d (Hamilton wxyz FRD->NED).',
        'Velocity and acceleration controller targets: vehicle_local_position_setpoint; trajectory velocity is feedforward.',
        'Thrust target is normalized -vehicle_attitude_setpoint.thrust_body[2], not Newtons.',
        'Body-z specific force = -vehicle_acceleration.xyz[2], a net specific-force proxy, not a physical thrust measurement.',
        'Opposing specific force projects negative non-body-z accelerometer force along the trajectory direction; includes aerodynamic/other forces and sensor errors.',
        'SITL esc_rpm is a rad/s motor COMMAND echo in the local GZMixingInterfaceESC, not measured RPM. Model forces are quasi-static approximations, not ground truth forces; omit filter dynamics and angular point velocity.',
        'Native timestamps are publication clocks; linear/normalized quaternion interpolation at 50 Hz with no >0.5 s gap bridging. Discrete states use hold.',
        'Event plots align reference acceleration onset, without time scaling. Global comparisons retain AUTO_MISSION time.',
        'No force sensors or real motor RPM are available. One flight per variant does not identify true physical coefficients.'])
    for index,(label,source) in enumerate(sources.items()):
        d,s=extract(source,args.window,args.steady,model_constants(models[label]) if label in models else None)
        traces[label]=d;summary['sources'][label]=s
        keys=list(d);filename=f'trace_{index:02d}.csv';s['csv']=filename
        np.savetxt(out/filename,np.column_stack([d[k] for k in keys]),delimiter=',',comments='',header=','.join(keys))
    (out/'dynamics_summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    plot(out,traces,args.window)
    plot_force_balance(out,traces,args.window)
    print(json.dumps({label:{k:s['metrics'][k] for k in ('time_to_4m_s_after_onset','time_to_4.5m_s_after_onset')} for label,s in summary['sources'].items()},indent=2))
    print(out/'dynamics_summary.json')


if __name__=='__main__':
    main()
