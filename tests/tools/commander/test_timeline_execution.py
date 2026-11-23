from types import SimpleNamespace
from unittest.mock import MagicMock, patch
import pytest

from tools.commander.mission_session_runner import TimelineExecution
from tests.tools.commander.test_mission_session import msg


def engine(events):
    r=SimpleNamespace(session_mode_name=lambda m:m.mode,session_set_mode=MagicMock())
    connection=MagicMock()
    phase=dict(events=events,upload_items=[dict(source_sequence=i) for i in range(3)])
    entry={};t=TimelineExecution(r,connection,phase,entry)
    t.update(msg('HEARTBEAT',base_mode=128,mode='MISSION'),1000)
    return t,r,entry


def test_event_timing_mode_ack_and_no_forced_resume_jump():
    events=[dict(id='hold',time_s=2.,action='hold'),dict(id='resume',time_s=4.,action='resume',expected_source_sequence=1)]
    t,r,entry=engine(events)
    t.update(msg('MISSION_CURRENT',seq=2),1500)
    t.update(msg('GLOBAL_POSITION_INT'),2999);r.session_set_mode.assert_not_called()
    t.update(msg('GLOBAL_POSITION_INT'),3000);r.session_set_mode.assert_called_once_with(t.connection,'HOLD')
    assert entry['events'][0]['status']=='sent'
    t.update(msg('HEARTBEAT',base_mode=128,mode='HOLD'),3100)
    assert entry['events'][0]['status']=='confirmed'
    t.update(msg('GLOBAL_POSITION_INT'),5000)
    t.update(msg('HEARTBEAT',base_mode=128,mode='MISSION'),5100)
    assert len(entry['divergences'])==1
    t.connection.mav.mission_set_current_send.assert_not_called()


def test_rtl_can_end_flight_without_reaching_last_plan_item():
    t,r,entry=engine([dict(id='rtl',time_s=2.,action='rtl')])
    assert not t.landing_allowed(False)
    t.update(msg('GLOBAL_POSITION_INT'),3000)
    assert not t.landing_allowed(False)
    t.update(msg('HEARTBEAT',base_mode=128,mode='RTL'),3100)
    assert t.landing_allowed(False)


def test_unexpected_mode_and_unacknowledged_event_fail():
    t,_,_=engine([])
    with pytest.raises(RuntimeError,match='Unexpected'):
        t.update(msg('HEARTBEAT',base_mode=128,mode='RTL'),2000)
    t,_,_=engine([dict(id='hold',time_s=1.,action='hold')])
    with patch('tools.commander.mission_session_runner.time.monotonic',return_value=10.):
        t.update(msg('GLOBAL_POSITION_INT'),2000)
    with patch('tools.commander.mission_session_runner.time.monotonic',return_value=19.):
        with pytest.raises(TimeoutError,match='not confirmed'):
            t.update(msg('GLOBAL_POSITION_INT'),2500)


def test_request_clock_anchor_is_preserved_after_delayed_heartbeat():
    r=SimpleNamespace(session_mode_name=lambda m:m.mode,session_set_mode=MagicMock())
    entry={};phase=dict(events=[],upload_items=[])
    t=TimelineExecution(r,MagicMock(),phase,entry,start_ms=1000)
    t.update(msg('HEARTBEAT',base_mode=128,mode='MISSION'),1900)
    assert t.anchor_ms==1000
    assert entry['initial_mode_confirmation_delay_s']==.9


def test_parameter_change_reads_back_without_disarm_or_gain_reset():
    t,r,entry=engine([dict(id='gain',time_s=1.,action='parameters',values={'MC_ROLL_P':6.},effective_values={'MC_ROLL_P':6.})])
    client=MagicMock();client.get_param.return_value=6.
    t.phase['_parameter_client']=client
    t.update(msg('GLOBAL_POSITION_INT'),2000)
    client.set_param.assert_called_once_with('MC_ROLL_P',6.)
    client.require_disarmed.assert_not_called()
    assert entry['events'][0]['actual']=={'MC_ROLL_P':6.}


def test_mission_replacement_waits_for_hold_and_current_item_confirmation():
    event=dict(id='replace',time_s=1.,action='replace_mission',plan='B',start_source_sequence=1)
    t,r,entry=engine([event]);r.autopilot='px4';r._upload_mission_items=MagicMock(return_value=True)
    t.phase['mission_definitions']={'B':[dict(source_sequence=0,p4=None),dict(source_sequence=1,p4=None)]}
    t.phase['_origin']=[40.,-86.,0.]
    t.update(msg('GLOBAL_POSITION_INT'),2000)
    r._upload_mission_items.assert_not_called()
    t.update(msg('HEARTBEAT',base_mode=128,mode='HOLD'),2100)
    r._upload_mission_items.assert_called_once()
    t.update(msg('HEARTBEAT',base_mode=128,mode='MISSION'),2200)
    assert entry['events'][0]['status']=='sent'
    t.update(msg('MISSION_CURRENT',seq=1),2300)
    t.update(msg('HEARTBEAT',base_mode=128,mode='MISSION'),2400)
    assert entry['events'][0]['status']=='confirmed'


def test_restart_after_replacement_validates_against_the_active_mission():
    import copy
    from tools.commander.mission_session import normalize_session
    from tests.tools.commander.test_mission_session import scenario
    data=scenario('px4');session=data['mission_session'];session['version']=2
    phase=session['phases'][0]
    for i,item in enumerate(phase['mission_items']):item['source_sequence']=i
    longer=copy.deepcopy(phase['mission_items'])
    longer.insert(2,copy.deepcopy(longer[1]))
    for i,item in enumerate(longer):item['source_sequence']=i
    phase.update(replay_policy='recorded_time',completion='recorded_landing',mission_definitions={'B':longer},
        events=[dict(id='replace',time_s=1.,action='replace_mission',plan='B',start_source_sequence=1),
                dict(id='restart',time_s=2.,action='restart_mission',start_source_sequence=3)])
    assert normalize_session(data,'px4')['phases'][0]['events'][1]['start_source_sequence']==3
