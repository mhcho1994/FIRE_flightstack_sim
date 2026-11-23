"""Timeline replay uses native fixtures and audits incomplete evidence."""
import copy
import json
from pathlib import Path

import pytest
import yaml

from tests.tools.scenario.test_px4_mission import fixture, save, native_log, FLIGHT_PARAMETERS
from tools.scenario.flight_log import read_flight_log
from tools.scenario.flight_replay import timeline_flights
from tools.scenario.import_real_flight_mission import main
from tools.commander.mission_session import normalize_session


def interrupted():
    plan, data = fixture()
    data['vehicle_status'][2:2] = [dict(timestamp=t, armed_time=1_100_000, arming_state=2, nav_state=mode)
        for t,mode in [(4_500_000,4),(4_800_000,3),(5_500_000,4),(6_000_000,5)]]
    data['navigator_mission_item'] = [r for r in data['navigator_mission_item'] if r['timestamp']<5_500_000]
    resumed = dict(data['navigator_mission_item'][-1],timestamp=4_810_000)
    data['navigator_mission_item'].append(resumed)
    data['position_setpoint_triplet'].append(dict(data['position_setpoint_triplet'][2],timestamp=4_810_010))
    data['mission_result'] = [r for r in data['mission_result'] if r['timestamp']<5_500_000]
    data['mission_result'].append(dict(data['mission_result'][-1],timestamp=4_810_020,seq_reached=-1))
    return plan,data


def test_partial_match_hold_resume_and_rtl_are_not_a_completed_plan(tmp_path):
    p,path=save(tmp_path,*interrupted())
    log=read_flight_log(path)
    flights,excluded,audit=timeline_flights(log,[p])
    f=flights[0]
    assert f['source_mission_outcome']=='interrupted'
    assert f['source_flight_outcome']=='landed_disarmed'
    assert [e['action'] for e in f['events']]==['hold','resume','hold','rtl']
    assert [e['time_s'] for e in f['events']]==[2.5,2.8,3.5,4.]
    assert f['events'][1]['expected_source_sequence']==1
    assert f['match']['revision_coverage']==[0,1]
    assert not excluded
    assert audit['unused_plans']==[]


def test_generation_copies_events_parameters_and_reference_without_full_completion(tmp_path, capsys):
    p,path=save(tmp_path,*interrupted(),changes=[(5_000_000,'MC_ROLL_P',8.)])
    out=tmp_path/'run'
    assert main(['--plan',str(p),'--flight-log',str(path),'--outdir',str(out)])==0
    s=normalize_session(yaml.safe_load((out/'scenario.yaml').read_text()),'px4')
    assert s['version']==2
    events=s['phases'][0]['events']
    assert next(e for e in events if e['action']=='parameters')['values']=={'MC_ROLL_P':8.}
    assert (out/'replay/reference_trajectory.csv').exists()
    report = json.loads((out/'replay_report.json').read_text())
    assert report['warnings'][0]['code']=='mission_interrupted'
    assert report['warnings']==s['warnings']
    assert report['warnings']==json.loads((out/'import_report.json').read_text())['warnings']
    assert 'WARNING [mission_interrupted]' in capsys.readouterr().err


def test_complete_flight_still_compiles_without_external_events(tmp_path):
    p,path=save(tmp_path,*fixture())
    flights,_,_=timeline_flights(read_flight_log(path),[p])
    assert flights[0]['events']==[]
    assert flights[0]['source_mission_outcome']=='completed'


def test_unused_plan_is_reported_but_shared_prefix_is_ambiguous(tmp_path):
    plan,data=interrupted();p,path=save(tmp_path,plan,data)
    other=tmp_path/'other.plan'
    changed=copy.deepcopy(plan);changed['mission']['items'][1]['params'][4]+=.01
    other.write_text(json.dumps(changed))
    log=read_flight_log(path)
    _,_,audit=timeline_flights(log,[p,other])
    assert audit['unused_plans']==[str(other)]
    changed=copy.deepcopy(plan);changed['mission']['items'][-1]['params'][4]+=.01
    other.write_text(json.dumps(changed))
    with pytest.raises(ValueError,match='ambiguous'):
        timeline_flights(log,[p,other])


@pytest.mark.parametrize('change,reason',[('no_disarm','disarm'),('bad_target','unmatched'),('no_land','landing'),('manual','manual'),('no_start','TAKEOFF')])
def test_unsupported_outcomes_remain_in_audit_and_cannot_generate(tmp_path,change,reason):
    plan,data=interrupted()
    if change=='no_disarm':data['vehicle_status'].pop()
    elif change=='bad_target':plan['mission']['items'][1]['params'][4]+=.01
    elif change=='no_land':data['vehicle_land_detected'].pop()
    elif change=='manual':data['vehicle_status'][2]['nav_state']=2
    else:data['navigator_mission_item'].pop(0)
    p,path=save(tmp_path,plan,data);log=read_flight_log(path)
    assert log.analyze([p])['flights']
    with pytest.raises(ValueError,match=reason):timeline_flights(log,[p])


def test_repeated_flights_keep_independent_clocks_and_can_be_selected(tmp_path):
    plan,data=interrupted()
    for rows in data.values():
        extra=copy.deepcopy(rows)
        for r in extra:
            r['timestamp']+=12_000_000
            if r.get('armed_time'):r['armed_time']+=12_000_000
        rows.extend(extra)
    p,path=save(tmp_path,plan,data);log=read_flight_log(path)
    fs,_,_=timeline_flights(log,[p]);assert len(fs)==2
    assert fs[0]['events'][0]['time_s']==fs[1]['events'][0]['time_s']
    fs,_,_=timeline_flights(log,[p],['flight_001']);assert [f['id'] for f in fs]==['flight_001']
    with pytest.raises(ValueError,match='Unknown'):timeline_flights(log,[p],['flight_999'])


def test_changed_mission_during_one_flight_is_separate_revision(tmp_path):
    plan,data=fixture();second=copy.deepcopy(plan)
    second['mission']['items'][1]['params'][4]+=.0002
    data['mission_result'].append(dict(data['mission_result'][1],timestamp=5_000_000,mission_id=999,seq_reached=-1))
    for r in data['mission_result']:
        if r['timestamp']>5_000_000:r['mission_id']=999
    item=dict(data['navigator_mission_item'][2],timestamp=5_010_000,
              latitude=second['mission']['items'][1]['params'][4])
    data['navigator_mission_item'].append(item)
    sp=copy.deepcopy(data['position_setpoint_triplet'][2]);sp['timestamp']=5_010_010
    sp['current']['lat']=second['mission']['items'][1]['params'][4]
    data['position_setpoint_triplet'].append(sp)
    p,path=save(tmp_path,plan,data);other=tmp_path/'second.plan';other.write_text(json.dumps(second))
    fs,_,_=timeline_flights(read_flight_log(path),[p,other])
    assert len(fs)==1
    assert fs[0]['events'][0]['action']=='replace_mission'
    assert fs[0]['events'][0]['plan']==str(other)
    assert [m['outcome'] for m in fs[0]['mission_executions']]==['interrupted','completed']
    assert fs[0]['source_mission_outcome']=='completed'
    assert fs[0]['warnings'][0]['plan']==str(p)
    assert fs[0]['warnings'][0]['mission_id']=='mission_000'


def test_analysis_only_does_not_publish_output(tmp_path,capsys):
    p,path=save(tmp_path,*interrupted());out=tmp_path/'run'
    main(['--plan',str(p),'--flight-log',str(path),'--analyze-only','--outdir',str(out)])
    assert json.loads(capsys.readouterr().out)['flights']
    assert not out.exists()


def test_ardupilot_snapshot_before_arm_and_home_refresh_are_supported(tmp_path):
    from tests.tools.scenario.test_flight_replay import fixture_plan,flight_records
    from tools.scenario import ardupilot_mission
    from tools.scenario.mission_timeline import build_timeline
    from tools.scenario.mission_matching import match_timeline
    p,items=fixture_plan(tmp_path)
    records=flight_records(items)
    records.extend([dict(mavpackettype='EV',TimeUS=2,Id=18),
        dict(mavpackettype='ORGN',TimeUS=19,Type=1,Lat=40.,Lng=-86.,Alt=151.),
        dict(mavpackettype='POS',TimeUS=22,Lat=40.,Lng=-86.,RelHomeAlt=2.)])
    # A complete snapshot is evidence even when it was logged before arming.
    for r in records:
        if r['mavpackettype']=='CMD':r['TimeUS']-=10
    timeline=match_timeline(build_timeline(**ardupilot_mission.decode_events(sorted(records,key=lambda r:r['TimeUS']))),[p],'ardupilot')
    f=timeline['flights'][0]
    assert f['outcome']=='landed_disarmed'
    assert f['initial_revision']['snapshot']
    seg=next(s for s in f['segments'] if s['observations'])
    assert seg['match']['status']=='matched_observed'
    assert seg['mission_outcome']=='completed'


def test_analysis_keeps_ground_only_and_truncated_armed_intervals(tmp_path):
    plan,data=fixture()
    data['vehicle_status']=[dict(timestamp=1_100_000,armed_time=1_100_000,arming_state=2,nav_state=2)]
    data['vehicle_land_detected']=data['vehicle_land_detected'][:1]
    p,path=save(tmp_path,plan,data)
    f=read_flight_log(path).analyze([p])['flights'][0]
    assert f['outcome']=='unknown'
    assert 'disarm not observed' in f['issues']
    assert f['mission_segments']==[]


def test_parameter_event_bounds_are_validated_before_publication(tmp_path):
    p,path=save(tmp_path,*interrupted(),changes=[(5_000_000,'MPC_XY_CRUISE',-2.)])
    out=tmp_path/'run'
    with pytest.raises(SystemExit):
        main(['--plan',str(p),'--flight-log',str(path),'--outdir',str(out)])
    assert not out.exists()


def test_nonfinite_native_target_never_matches_a_partial_plan():
    from tools.scenario.mission_matching import item_matches
    plan,_=fixture();item=plan['mission']['items'][1]
    observation=dict(command=16,params=list(item['params']),target_confirmed=True,autocontinue=True,frame=3)
    observation['params'][4]=float('nan')
    assert not item_matches(observation,item,'ardupilot')


def test_complete_and_interrupted_imports_share_parameter_capture_and_startup(tmp_path):
    from tests.tools.scenario.test_capture_flight_parameters_file import values
    sessions = []
    for label, source in [('complete',fixture),('interrupted',interrupted)]:
        root=tmp_path/label;root.mkdir()
        plan,data=source();p,path=save(root,plan,data)
        native_log(path,data,changes=[(1_500_000,'MC_ROLL_P',6.5),(5_000_000,'NAV_ACC_RAD',3.)],
            parameters={**values('px4'),**FLIGHT_PARAMETERS})
        out=root/'run'
        main(['--plan',str(p),'--flight-log',str(path),'--outdir',str(out)])
        sessions.append(normalize_session(yaml.safe_load((out/'scenario.yaml').read_text()),'px4'))
    a,b=[s['phases'][0] for s in sessions]
    assert all(s['version']==2 for s in sessions)
    assert a['replay_policy']==b['replay_policy']=='recorded_time'
    assert a['gain_capture']['snapshot']==b['gain_capture']['snapshot']=='time:2.000000'
    assert a['controller_gains']['values']==b['controller_gains']['values']
    assert a['controller_gains']['values']['MC_ROLL_P']==6.5
    assert a['flight_parameters']==b['flight_parameters']
    assert a['flight_parameters']['values']==pytest.approx(FLIGHT_PARAMETERS)
    for phase in (a,b):
        event=next(e for e in phase['events'] if e['action']=='parameters')
        assert event['values']=={'NAV_ACC_RAD':3.}
        assert event['time_s']==3.


def test_unused_plan_warning_and_dry_run_json_do_not_require_all_candidates(tmp_path,capsys):
    plan,data=interrupted();p,path=save(tmp_path,plan,data)
    other=tmp_path/'unused.plan';plan['mission']['items'][1]['params'][4]+=.01
    other.write_text(json.dumps(plan))
    out=tmp_path/'run'
    main(['--plan',str(p),str(other),'--flight-log',str(path),'--outdir',str(out),'--dry-run'])
    printed=capsys.readouterr();s=json.loads(printed.out)
    assert {w['code'] for w in s['warnings']}=={'mission_interrupted','unused_plan'}
    assert 'WARNING [unused_plan]' in printed.err
    assert not out.exists()


def test_recorded_speed_and_gain_changes_are_always_replayed(tmp_path):
    p,path=save(tmp_path,*fixture(),changes=[(1_500_000,'MPC_XY_CRUISE',6.),
        (5_000_000,'MPC_XY_CRUISE',8.),(5_100_000,'MC_ROLL_P',7.)])
    out=tmp_path/'run'
    main(['--plan',str(p),'--flight-log',str(path),'--outdir',str(out)])
    phase=yaml.safe_load((out/'scenario.yaml').read_text())['mission_session']['phases'][0]
    assert phase['flight_parameters']['values']['MPC_XY_CRUISE']==6.
    changes={k:v for e in phase['events'] if e['action']=='parameters' for k,v in e['values'].items()}
    assert changes['MC_ROLL_P']==7.
    assert changes['MPC_XY_CRUISE']==8.


def test_default_import_can_select_one_repeated_flight(tmp_path,capsys):
    plan,data=interrupted()
    for rows in data.values():
        extra=copy.deepcopy(rows)
        for r in extra:
            r['timestamp']+=12_000_000
            if r.get('armed_time'):r['armed_time']+=12_000_000
        rows.extend(extra)
    p,path=save(tmp_path,plan,data)
    main(['--plan',str(p),'--flight-log',str(path),'--outdir',str(tmp_path/'run'),
          '--flight-id','flight_001','--dry-run'])
    s=json.loads(capsys.readouterr().out)
    assert [f['id'] for f in s['phases']]==['flight_001']
    assert s['phases'][0]['gain_capture']['snapshot']=='time:14.000000'
    assert s['excluded_armed_intervals']==[{'id':'flight_000','reason':'not selected'}]


def test_sequence_rewind_preserves_earlier_interrupted_attempt(tmp_path):
    plan,data=fixture()
    data['navigator_mission_item'].append(dict(data['navigator_mission_item'][0],timestamp=4_500_000))
    data['position_setpoint_triplet'].append(dict(data['position_setpoint_triplet'][0],timestamp=4_500_010))
    p,path=save(tmp_path,plan,data)
    flights,_,_=timeline_flights(read_flight_log(path),[p]);f=flights[0]
    assert f['events'][0]['action']=='restart_mission'
    assert f['events'][0]['start_source_sequence']==0
    assert [e['outcome'] for e in f['mission_executions']]==['interrupted','completed']
    assert f['source_mission_outcome']=='completed'
    assert [w['mission_id'] for w in f['warnings']]==['mission_000']


def test_truncated_mission_is_unknown_in_analysis(tmp_path):
    plan,data=fixture()
    for name,rows in data.items():
        data[name]=[r for r in rows if r['timestamp']<5_000_000]
    p,path=save(tmp_path,plan,data)
    f=read_flight_log(path).analyze([p])['flights'][0]
    assert f['outcome']=='unknown'
    assert f['mission_executions'][0]['outcome']=='unknown'


@pytest.mark.parametrize('valid', [True,False])
def test_native_failure_is_an_outcome_but_invalid_mission_is_an_evidence_error(tmp_path,valid,capsys):
    plan,data=interrupted()
    data['mission_result'][-1].update(failure=True,valid=valid)
    p,path=save(tmp_path,plan,data);out=tmp_path/'run'
    args=['--plan',str(p),'--flight-log',str(path),'--outdir',str(out)]
    if not valid:
        with pytest.raises(SystemExit):main(args)
        assert 'invalid mission_result' in capsys.readouterr().err
        assert not out.exists()
    else:
        assert main(args)==0
        s=json.loads((out/'replay_report.json').read_text())
        assert s['warnings'][0]['code']=='mission_failure_recorded'
        assert s['phases'][0]['source_mission_outcome']=='interrupted'
        assert [e['action'] for e in s['phases'][0]['events']]==['hold','resume','hold','rtl']
        assert s['phases'][0]['mission_executions'][0]['mission_failure_observed']
        assert 'WARNING [mission_failure_recorded]' in capsys.readouterr().err
