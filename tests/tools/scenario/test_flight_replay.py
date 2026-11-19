import copy
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
import hashlib

import numpy as np
import pytest

from tools.scenario.capture_flight_parameters import select_snapshot, snapshot_time_us, read_flight_parameters
from tests.tools.scenario.test_capture_flight_parameters_file import write_bin, write_ulog, values
from tools.scenario.mission_log_validation import match_executions, compare_plan
from tools.commander import mission_session_runner as runner_module
from tools.commander.pymavlink_ardupilot_commander import ArduPilotMissionRunner
from tools.plotting.compare_flight_replay import compare_phase


def test_time_snapshot_retains_last_observation_and_never_backfills():
    h = dict(initial={'A':1, 'B':5}, final={'A':3, 'B':5}, first_seen_us={'A':100, 'B':500},
             changes=[dict(parameter='A', timestamp_us=200, before=1, after=2),
                      dict(parameter='A', timestamp_us=400, before=2, after=3)], end_time_us=600)
    assert select_snapshot(h, 'time:0.000199') == {'A':1}
    assert select_snapshot(h, 'time:0.000200') == {'A':2}
    assert select_snapshot(h, 'time:0.000500') == {'A':3,'B':5}
    with pytest.raises(ValueError, match='beyond'):
        select_snapshot(h, 'time:1')
    for invalid in ('time:nan', 'time:-1', 'time:0.0000001', '123', 'time:inf'):
        with pytest.raises(ValueError): snapshot_time_us(invalid)


@pytest.mark.parametrize('ap,writer,suffix', [('ardupilot',write_bin,'.BIN'),('px4',write_ulog,'.ulg')])
def test_time_snapshot_decodes_real_formats(tmp_path, ap, writer, suffix):
    baseline=values(ap); name=next(iter(baseline))
    path=writer(tmp_path/f'log{suffix}',baseline,[(3000,name,2.),(5000,name,3.)])
    profile, report=read_flight_parameters(path,snapshot='time:0.004')
    assert profile[ap]['baseline'][name] == 2
    assert report['snapshot']=='time:0.004'


def fixture_plan(tmp_path):
    items=[dict(command=c,frame=3,params=[0,0,0,None,40.,-86.,h]) for c,h in [(22,2),(16,10),(21,0)]]
    path=tmp_path/'2.plan';path.write_text(json.dumps(dict(mission=dict(items=items))))
    return path,items


def flight_records(items, offset=0):
    def row(kind,t,**kw):return dict(mavpackettype=kind,TimeUS=offset+t,**kw)
    rows=[row('ORGN',1,Type=1,Lat=40.,Lng=-86.,Alt=150.),row('ARM',10,ArmState=1)]
    commands=[]
    for seq,item in enumerate([dict(command=16,frame=0,params=[0,0,0,0,40.,-86.,150])]+items):
        p=item['params'];commands.append(row('CMD',11+seq,CTot=len(items)+1,CNum=seq,CId=item['command'],
            Frame=item['frame'],Prm1=p[0],Prm2=p[1],Prm3=p[2],Prm4=p[3] or 0,Lat=p[4],Lng=p[5],Alt=p[6]))
    rows+=commands
    rows.append(row('MODE',20,ModeNum=3))
    for seq in range(1,len(items)+1):
        r=dict(commands[seq],mavpackettype='MISE',TimeUS=offset+20+seq)
        rows.append(r)
        if seq==1: rows.append(dict(r,TimeUS=offset+22))
    rows += [row('EV',90,Id=18),row('ARM',100,ArmState=0),dict(commands[1],mavpackettype='MISE',TimeUS=offset+101)]
    return rows


def test_match_repeated_flights_not_duplicate_mise_or_post_disarm_reset(tmp_path):
    path,items=fixture_plan(tmp_path)
    phases,_=match_executions([path],flight_records(items)+flight_records(items,1000))
    assert len(phases)==2
    assert [p['start_time_us'] for p in phases]==[21,1021]
    assert all(p['executed_sequence']==[1,2,3] for p in phases)


@pytest.mark.parametrize('mutation', ['no_land','interrupted','missing_disarm','different_plan','ambiguous'])
def test_incomplete_mismatched_or_ambiguous_replay_refused(tmp_path, mutation):
    path,items=fixture_plan(tmp_path);records=flight_records(items);plans=[path]
    if mutation=='no_land': records=[r for r in records if r['mavpackettype']!='EV']
    elif mutation=='interrupted': records.insert(-3,dict(mavpackettype='MODE',TimeUS=50,ModeNum=5))
    elif mutation=='missing_disarm':records=[r for r in records if r['mavpackettype']!='ARM' or r['ArmState']]
    elif mutation=='different_plan':
        items[1]['params'][4]+=0.01;path.write_text(json.dumps(dict(mission=dict(items=items))))
    else:
        other=tmp_path/'duplicate.plan';other.write_bytes(path.read_bytes());plans.append(other)
    with pytest.raises(ValueError):match_executions(plans,records)


def test_comparison_never_extrapolates_or_crosses_phases():
    real=np.array([[0,0,0,0],[1,1,0,0],[2,2,0,0],[3,3,0,0]],float)
    sim=np.array([[1,2,0,0],[2,3,0,0]],float)
    metrics,paired=compare_phase(real,sim)
    assert metrics['position_rmse_m']==1
    assert list(paired[:,0])==[1,2]
    with pytest.raises(ValueError):compare_phase(real,sim+np.array([10,0,0,0]))


def test_unexpected_disarm_not_success():
    assert runner_module.phase_disarm_ok(True,True,True)
    for state in [(False,True,True),(True,False,True),(True,True,False)]:
        assert not runner_module.phase_disarm_ok(*state)



def test_native_bin_comparison_uses_mise_and_disarm_not_file_start(tmp_path):
    from tools.plotting.compare_flight_replay import bin_trajectories
    records=[dict(mavpackettype='MISE',TimeUS=1_010_000,CNum=1,CId=22),
             dict(mavpackettype='POS',TimeUS=1_100_000,Lat=40.,Lng=-86.,RelHomeAlt=1.),
             dict(mavpackettype='POS',TimeUS=1_900_000,Lat=40.,Lng=-86.,RelHomeAlt=0.),
             dict(mavpackettype='ARM',TimeUS=2_010_000,ArmState=0),
             dict(mavpackettype='MISE',TimeUS=2_020_000,CNum=1,CId=22)]
    runtime=dict(phases=[dict(id='flight_000',sitl_start_time_us=1_000_000,sitl_end_time_us=2_000_000)])
    session=dict(target_origin_lla=[40.,-86.,150.])
    with patch('tools.launcher.attempt_logs.flight_logs',return_value=[tmp_path/'log.BIN']), \
         patch('tools.scenario.mission_log_validation.read_log',return_value=(records,'hash')):
        trajectories, source=bin_trajectories(tmp_path,session,runtime)
        assert np.allclose(trajectories['flight_000'][:,0],[.09,.89])
        assert source['markers']['flight_000']==dict(start_time_us=1_010_000,end_time_us=2_010_000)
        runtime['phases'][0]['sitl_start_time_us']=3_000_000
        with pytest.raises(ValueError,match='start marker'):
            bin_trajectories(tmp_path,session,runtime)


