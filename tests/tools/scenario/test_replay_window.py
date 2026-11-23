"""Native landing evidence and replay windows inside longer armed intervals."""
import csv
import json

import pytest
import yaml

from tests.tools.scenario.test_flight_replay import fixture_plan, flight_records, bin_timeline
from tests.tools.scenario.test_px4_mission import fixture, save
from tools.scenario.ardupilot_mission import decode_events
from tools.scenario.import_real_flight_mission import main
from tools.scenario.mission_matching import match_timeline
from tools.scenario.mission_timeline import build_timeline, POST_DISARM_EVENT_WINDOW_US


def row(kind, t, **values):
    return dict(mavpackettype=kind, TimeUS=t, **values)


def analyze(records, plan):
    decoded = decode_events(sorted(records, key=lambda r: r['TimeUS']))
    return match_timeline(build_timeline(**decoded), [plan], 'ardupilot')


def prefixed_records(items):
    return flight_records(items) + [
        row('MODE', 11, ModeNum=4), row('EV', 12, Id=28),
        row('POS', 16, Lat=40., Lng=-86., RelHomeAlt=2.), row('EV', 18, Id=18),
        # The first landing can also retain a positive height above home.
        row('POS', 19, Lat=40., Lng=-86., RelHomeAlt=1.4),
    ]


def test_native_landing_is_not_undone_by_positive_height(tmp_path):
    plan, items = fixture_plan(tmp_path)
    records = flight_records(items) + [row('POS', 91, Lat=40., Lng=-86., RelHomeAlt=1.393)]
    phases, _, audit = bin_timeline([plan], records)
    f = audit['flights'][0]
    assert f['airborne_intervals'] == [dict(start_time_us=22, end_time_us=90)]
    assert f['outcome'] == 'landed_disarmed'
    assert phases[0]['source_mission_outcome'] == 'completed'
    assert [e['evidence'] for e in f['events'] if e['kind'] == 'land'] == [
        'EV.NOT_LANDED', 'EV.LAND_COMPLETE']


def test_pos_alone_never_supplies_takeoff_evidence(tmp_path):
    plan, items = fixture_plan(tmp_path)
    records = [r for r in flight_records(items) if not (r['mavpackettype'] == 'EV' and r['Id'] == 28)]
    f = analyze(records, plan)['flights'][0]
    assert f['airborne_intervals'] == []
    with pytest.raises(ValueError, match='takeoff'):
        bin_timeline([plan], records)


def test_prior_flight_is_excluded_only_after_confirmed_ground(tmp_path):
    plan, items = fixture_plan(tmp_path)
    phases, _, audit = bin_timeline([plan], prefixed_records(items))
    phase, f = phases[0], audit['flights'][0]
    assert phase['arm_time_us'] == 10
    assert phase['start_time_us'] == 20
    assert phase['end_time_us'] == 100
    assert phase['source_landing_time_us'] == 90
    assert phase['source_mission_outcome'] == 'completed'
    assert phase['events'] == []
    assert len(f['airborne_intervals']) == 2
    assert [s['mode'] for s in phase['timeline_segments']] == ['MISSION']
    assert any(s['mode'] == 'GUIDED' for s in f['segments'])
    window = phase['replay_window']
    assert window['initial_landed'] is True
    assert window['ground_evidence']['timestamp_us'] == 18
    assert window['airborne_intervals'] == [dict(start_time_us=22, end_time_us=90)]
    excluded = window['excluded_intervals'][0]
    assert (excluded['start_time_us'], excluded['end_time_us']) == (10, 20)
    assert excluded['airborne_intervals'] == [dict(start_time_us=12, end_time_us=18)]
    assert phase['warnings'][0]['code'] == 'pre_mission_flight_excluded'


@pytest.mark.parametrize('change, reason', [
    ('no_ground', 'ground state'), ('mid_mission_manual', 'manual/offboard'),
    ('second_takeoff', 'repeated takeoffs'), ('earlier_mission', 'repeated takeoffs'),
    ('unobserved_earlier_mission', 'cannot discard a possible mission attempt'),
])
def test_crop_never_hides_unsupported_mission_state(tmp_path, change, reason):
    plan, items = fixture_plan(tmp_path)
    records = prefixed_records(items)
    if change == 'no_ground':
        records = [r for r in records if not (r['mavpackettype'] == 'EV' and r['TimeUS'] == 18)]
    elif change == 'mid_mission_manual':
        records += [row('MODE', 50, ModeNum=4), row('MODE', 60, ModeNum=3)]
    elif change == 'second_takeoff':
        records += [row('EV', 50, Id=18), row('EV', 60, Id=28)]
    else:
        records = [r for r in records if not (r['mavpackettype'] == 'MODE' and r['TimeUS'] == 11)]
        takeoff = next(r for r in records if r['mavpackettype'] == 'MISE')
        records += [row('MODE', 11, ModeNum=3), row('MODE', 19, ModeNum=4)]
        if change == 'earlier_mission':
            records.append(dict(takeoff, TimeUS=13))
    with pytest.raises(ValueError, match=reason):
        bin_timeline([plan], records)


def test_landing_matches_its_mission_not_the_last_flight_in_the_arm_interval(tmp_path):
    plan, items = fixture_plan(tmp_path)
    records = [r for r in flight_records(items) if r['TimeUS'] < 100]
    records += [row('MODE', 95, ModeNum=4), row('EV', 120, Id=28),
                row('EV', 180, Id=18), row('ARM', 200, ArmState=0)]
    f = analyze(records, plan)['flights'][0]
    mission = next(s for s in f['segments'] if s['observations'])
    assert len(f['airborne_intervals']) == 2
    assert mission['landing_time_us'] == 90
    assert mission['mission_outcome'] == 'completed'
    with pytest.raises(ValueError, match='repeated takeoffs'):
        bin_timeline([plan], records)


def test_disarm_reset_is_audited_but_does_not_prove_touchdown(tmp_path):
    plan, items = fixture_plan(tmp_path)
    records = [r for r in flight_records(items) if not (r['mavpackettype'] == 'EV' and r['TimeUS'] == 90)]
    records += [row('EV', 109, Id=18)]
    f = analyze(records, plan)['flights'][0]
    assert f['disarm_time_us'] == f['end_time_us'] == 100
    assert f['post_disarm_land_events'][0]['timestamp_us'] == 109
    assert f['airborne_intervals'][-1]['end_time_us'] is None
    assert f['outcome'] == 'disarmed_landing_unconfirmed'
    assert f['mission_executions'][0]['outcome'] == 'unknown'
    with pytest.raises(ValueError, match='landing before disarm'):
        bin_timeline([plan], records)
    # The native state still supplies the next arm interval's initial state.
    records += [row('ARM', 200, ArmState=1), row('ARM', 300, ArmState=0)]
    assert analyze(records, plan)['flights'][1]['initial_landed'] is True


def test_post_disarm_association_is_bounded_and_cannot_cross_rearm(tmp_path):
    plan, items = fixture_plan(tmp_path)
    records = flight_records(items)
    late = row('EV', 100 + POST_DISARM_EVENT_WINDOW_US + 1, Id=18)
    assert 'post_disarm_land_events' not in analyze(records + [late], plan)['flights'][0]
    events = [row('ARM', 105, ArmState=1), row('EV', 109, Id=18), row('ARM', 120, ArmState=0)]
    audit = analyze(records + events, plan)
    assert 'post_disarm_land_events' not in audit['flights'][0]
    assert any(e['kind'] == 'land' and e['timestamp_us'] == 109 for e in audit['flights'][1]['events'])


def test_cropped_native_ulog_keeps_snapshot_events_and_reference_on_one_clock(tmp_path, capsys):
    plan, data = fixture()
    data['vehicle_land_detected'] += [dict(timestamp=1_300_000, landed=False),
                                     dict(timestamp=1_800_000, landed=True)]
    data['vehicle_global_position'].append(dict(data['vehicle_global_position'][0], timestamp=1_500_000))
    p, log = save(tmp_path, plan, data, changes=[(1_500_000, 'MC_ROLL_P', 6.5),
        (1_600_000, 'MPC_XY_CRUISE', 6.), (5_000_000, 'MC_ROLL_P', 7.)])
    out = tmp_path / 'output'
    assert main(['--plan', str(p), '--flight-log', str(log), '--outdir', str(out)]) == 0
    session = yaml.safe_load((out / 'scenario.yaml').read_text())['mission_session']
    phase = session['phases'][0]
    assert phase['gain_capture']['snapshot'] == 'time:2.000000'
    assert phase['controller_gains']['values']['MC_ROLL_P'] == 6.5
    assert phase['flight_parameters']['values']['MPC_XY_CRUISE'] == 6.
    assert phase['events'] == [dict(id=phase['events'][0]['id'], action='parameters',
        time_s=3., source_time_us=5_000_000, values={'MC_ROLL_P': 7.})]
    assert len(session['analysis']['flights'][0]['airborne_intervals']) == 2
    with (out / 'replay/reference_trajectory.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) >= 2
    assert all(phase['start_time_us'] <= int(r['boot_time_us']) <= phase['end_time_us'] for r in rows)
    assert all(float(r['time_s']) == pytest.approx((int(r['boot_time_us']) - 2_000_000) / 1e6) for r in rows)
    report = json.loads((out / 'replay_report.json').read_text())
    assert report['phases'][0]['replay_window'] == phase['replay_window']
    assert 'WARNING [pre_mission_flight_excluded]' in capsys.readouterr().err
    main(['--plan', str(p), '--flight-log', str(log), '--outdir', str(out), '--analyze-only'])
    audit = json.loads(capsys.readouterr().out)
    assert audit['flights'][0]['replay_window'] == phase['replay_window']


def test_post_disarm_not_landed_cancels_landing_association(tmp_path):
    plan, items = fixture_plan(tmp_path)
    records = flight_records(items) + [row('EV', 105, Id=28), row('EV', 109, Id=18)]
    assert 'post_disarm_land_events' not in analyze(records, plan)['flights'][0]
