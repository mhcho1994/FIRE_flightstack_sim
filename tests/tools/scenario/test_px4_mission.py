"""Native ULog fixtures cover execution evidence and published replay artifacts."""
import copy
import csv
import json
from pathlib import Path
import struct

import pytest
import yaml

from tools.scenario import px4_mission as px4
from tools.scenario.flight_log import read_flight_log
from tools.scenario.import_real_flight_mission import main
from tools.commander.mission_session import normalize_session
from tools.launcher.startup_parameters import collect_boot_parameters
from tests.tools.scenario.test_capture_flight_parameters_file import values, write_ulog


# These are valid ULog message definitions, including the nested current setpoint.
FORMATS = {
    'vehicle_status': 'uint64_t timestamp;uint64_t armed_time;uint8_t arming_state;uint8_t nav_state;',
    'vehicle_land_detected': 'uint64_t timestamp;bool landed;',
    'home_position': 'uint64_t timestamp;double lat;double lon;float alt;bool valid_hpos;bool valid_alt;',
    'mission_result': ('uint64_t timestamp;uint32_t mission_id;int32_t seq_reached;uint16_t seq_current;'
                       'uint16_t seq_total;bool valid;bool finished;bool failure;'),
    'navigator_mission_item': ('uint64_t timestamp;uint16_t sequence_current;uint16_t nav_cmd;float latitude;'
                               'float longitude;float altitude;float time_inside;float acceptance_radius;'
                               'bool altitude_is_relative;bool autocontinue;uint8_t frame;'),
    'test_setpoint': 'double lat;double lon;float alt;bool valid;',
    'position_setpoint_triplet': 'uint64_t timestamp;test_setpoint current;',
    'vehicle_global_position': ('uint64_t timestamp;double lat;double lon;float alt;bool lat_lon_valid;'
                                'bool alt_valid;uint8_t alt_reset_counter;'),
    'vehicle_command': ('uint64_t timestamp;uint32_t command;float param1;float param2;float param3;float param4;'
                        'double param5;double param6;float param7;bool from_external;'),
}
PACK = {'uint64_t': 'Q', 'uint32_t': 'I', 'int32_t': 'i', 'uint16_t': 'H',
        'uint8_t': 'B', 'double': 'd', 'float': 'f', 'bool': '?'}


def fixture():
    lat, lon, alt = 40.44237474, -86.88899243, 160.
    def item(cmd, x, y, z):
        return dict(type='SimpleItem', command=cmd, frame=3, autoContinue=True,
                    params=[0, 0, 0, None, x, y, z])
    plan = dict(fileType='Plan', version=1, mission=dict(version=2, firmwareType=12, vehicleType=2,
        plannedHomePosition=[lat, lon, alt], hoverSpeed=1., items=[item(22, lat, lon, 3.),
        item(16, lat+.0001, lon+.0001, 4.), item(21, lat, lon, 0.)]))
    data = {name: [] for name in FORMATS if name != 'test_setpoint'}
    def add(name, seconds, **row):
        data[name].append(dict(timestamp=round(seconds*1e6), **row))
    add('vehicle_status', 1.1, armed_time=1_100_000, arming_state=2, nav_state=2)
    add('vehicle_status', 2., armed_time=1_100_000, arming_state=2, nav_state=3)
    add('vehicle_status', 10., armed_time=0, arming_state=1, nav_state=3)
    for t, landed in [(1.11, True), (2.5, False), (9., True)]:
        add('vehicle_land_detected', t, landed=landed)
    add('home_position', 1.1, lat=lat, lon=lon, alt=alt, valid_hpos=True, valid_alt=True)
    for t, current, reached, done in [(1.2, 0, -1, False), (4., 1, 0, False),
                                      (6., 2, 1, False), (9.1, 3, 2, True)]:
        add('mission_result', t, mission_id=1234, seq_current=current, seq_reached=reached,
            seq_total=3, valid=True, finished=done, failure=False)
    for t, seq, cmd, height, relative in [(2.01, 0, 22, 163., False), (3.01, 0, 16, 3., True),
                                        (4.01, 1, 16, 4., True), (6.01, 2, 16, 164., False),
                                        (7.01, 2, 21, 0., True)]:
        p = plan['mission']['items'][seq]['params']
        add('navigator_mission_item', t, sequence_current=seq, nav_cmd=cmd,
            latitude=p[4], longitude=p[5], altitude=height, altitude_is_relative=relative,
            autocontinue=True, time_inside=0., acceptance_radius=0., frame=0)
        add('position_setpoint_triplet', t+.00001,
            current=dict(lat=p[4], lon=p[5], alt=height+alt if relative else height, valid=True))
    for t in (2.1, 4.1, 6.1, 9.9):
        add('vehicle_global_position', t, lat=lat, lon=lon, alt=alt+2., lat_lon_valid=True,
            alt_valid=True, alt_reset_counter=0)
    return plan, data


def native_log(path, data, changes=(), *, parameters=None):
    """Serialize actual ULog bytes; no mocks in importer integration checks."""
    parameters = dict(values('px4'), MPC_XY_CRUISE=5.) if parameters is None else parameters
    write_ulog(path, parameters, firmware=dict(sys_name='PX4', ver_sw_release=0x011100ff,
                                             ver_sw='d6f12ad1c4f70ad3230afd7d86e971421e02fef4'))
    original = path.read_bytes()
    def msg(kind, payload):
        return struct.pack('<HB', len(payload), ord(kind)) + payload
    def encode(name, row):
        out = bytearray()
        for field in FORMATS[name].strip(';').split(';'):
            typ, key = field.split()
            out += struct.pack('<'+PACK[typ], row[key]) if typ in PACK else encode(typ, row[key])
        return bytes(out)
    # Definitions must precede the first subscription in a ULog.
    split = 16
    while original[split+2] != ord('A'):
        split += 3 + struct.unpack_from('<H', original, split)[0]
    content = bytearray(original[:split])
    for name, definition in FORMATS.items():
        content += msg('F', (name+':'+definition).encode())
    content += original[split:]
    events = []
    for mid, (name, rows) in enumerate(data.items(), 1):
        content += msg('A', struct.pack('<BH', 0, mid)+name.encode())
        for row in rows:
            events.append((row['timestamp'], msg('D', struct.pack('<H', mid)+encode(name, row))))
    for t, key, value in changes:
        # Ensure pyulog has a data timestamp immediately before the parameter.
        payload = ('float '+key).encode()
        events.append((t, msg('D', struct.pack('<HQf', 0, t, 0.)) +
                       msg('P', struct.pack('<B', len(payload))+payload+struct.pack('<f', value))))
    for _, payload in sorted(events, key=lambda event: event[0]):
        content += payload
    path.write_bytes(content)
    return path


def save(tmp_path, plan, data, changes=()):
    p = tmp_path/'mission.plan'
    p.write_text(json.dumps(plan))
    log = native_log(tmp_path/'flight.ulg', data, changes)
    return p, log


def test_completed_native_ulog_matches_internal_work_items_and_frame_quirk(tmp_path):
    plan, data = fixture()
    p, path = save(tmp_path, plan, data)
    log = read_flight_log(path)
    flights, excluded = log.match_executions([p])
    assert excluded == []
    assert log.autopilot == 'px4'
    assert flights[0]['executed_sequence'] == [0, 1, 2]
    assert flights[0]['start_time_us'] == 2_010_000
    assert flights[0]['end_time_us'] == 10_000_000
    assert log.positions(flights[0])[0][-1] == 2.
    renamed = tmp_path/'misleading.BIN'
    renamed.write_bytes(path.read_bytes())
    assert read_flight_log(renamed).autopilot == 'px4'


@pytest.mark.parametrize('mutation,reason', [
    ('rtl', 'interrupted AUTO_MISSION'), ('hold_resume', 'Hold'),
    ('no_disarm', 'disarm not observed'), ('no_landing', 'landing'),
    ('not_finished', 'mission completion'), ('mission_changed', 'mission_id changed'),
    ('no_start', 'mission start/TAKEOFF'), ('bad_target', 'target/altitude'),
    ('wrong_count', 'count differs'), ('home_changed', 'home changed'),
    ('missing_topic', 'missing required topic'), ('skipped_item', 'incomplete/skipped'),
    ('result_gap', 'does not confirm every navigation item'),
    ('stale_finished', 'mission completion'),
])
def test_incomplete_or_unverifiable_flights_do_not_generate(tmp_path, mutation, reason):
    plan, data = fixture()
    if mutation == 'rtl':
        data['vehicle_status'].insert(2, dict(timestamp=6_500_000, armed_time=1_100_000, arming_state=2, nav_state=5))
    elif mutation == 'hold_resume':
        data['vehicle_status'][2:2] = [dict(timestamp=t, armed_time=1_100_000, arming_state=2, nav_state=mode)
                                     for t, mode in [(4_500_000, 4), (5_000_000, 3)]]
    elif mutation == 'no_disarm': data['vehicle_status'].pop()
    elif mutation == 'no_landing': data['vehicle_land_detected'].pop()
    elif mutation == 'not_finished': data['mission_result'][-1]['finished'] = False
    elif mutation == 'mission_changed': data['mission_result'][-1]['mission_id'] += 1
    elif mutation == 'no_start': data['navigator_mission_item'].pop(0)
    elif mutation == 'bad_target': plan['mission']['items'][1]['params'][4] += .001
    elif mutation == 'wrong_count': data['mission_result'][-1]['seq_total'] = 4
    elif mutation == 'home_changed': data['home_position'].append(dict(data['home_position'][0], timestamp=5_000_000, alt=161.))
    elif mutation == 'result_gap': data['mission_result'] = [r for r in data['mission_result'] if r['seq_current'] != 1]
    elif mutation == 'stale_finished': data['mission_result'][-1]['timestamp'] = 1_500_000
    elif mutation == 'missing_topic': del data['mission_result']
    elif mutation == 'skipped_item': data['navigator_mission_item'] = [r for r in data['navigator_mission_item'] if r['sequence_current'] != 1]
    p, log = save(tmp_path, plan, data)
    out = tmp_path/'output'
    with pytest.raises(ValueError, match=reason):
        read_flight_log(log).match_executions([p])
    with pytest.raises(SystemExit):
        main(['--input', str(p), '--flight-log', str(log), '--outdir', str(out)])
    assert not out.exists()


def test_ambiguous_matching_plans_are_rejected(tmp_path):
    p, log = save(tmp_path, *fixture())
    other = tmp_path/'duplicate.plan'; other.write_bytes(p.read_bytes())
    with pytest.raises(ValueError, match='ambiguous'):
        read_flight_log(log).match_executions([p, other])


def test_generation_preserves_native_gains_speed_indices_and_flat_output(tmp_path):
    p, log = save(tmp_path, *fixture(), changes=[(1_500_000, 'MC_ROLL_P', 6.5)])
    out = tmp_path/'output'
    assert main(['--input', str(p), '--flight-log', str(log), '--outdir', str(out)]) == 0
    assert not (out/'run_000').exists()
    scenario = yaml.safe_load((out/'scenario.yaml').read_text())
    assert list(scenario['autopilots']) == ['px4']
    session = normalize_session(scenario, 'px4')
    phase = session['phases'][0]
    assert phase['gain_capture']['snapshot'] == 'time:2.010000'
    assert phase['controller_gains']['values']['MC_ROLL_P'] == 6.5
    assert phase['flight_parameters']['values'] == {'MPC_XY_CRUISE': 5.}
    assert [it['source_sequence'] for it in phase['mission_items']] == [0, 1, 2]
    content, boot, _ = collect_boot_parameters(scenario, 'px4', out, phase=phase)
    assert content and boot['MPC_XY_CRUISE'] == 5. and boot['MC_ROLL_P'] == 6.5
    with (out/'replay/reference_trajectory.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    assert float(rows[0]['time_s']) == pytest.approx(.09)
    assert float(rows[0]['up_m']) == 2.


@pytest.mark.parametrize('options,expected', [(['--speed-m-s', '3'], {'MPC_XY_CRUISE': 3.}),
                                             (['--use-firmware-speed'], None)])
def test_speed_options(tmp_path, options, expected):
    p, log = save(tmp_path, *fixture())
    out = tmp_path/'output'
    main(['--input', str(p), '--flight-log', str(log), '--outdir', str(out), *options])
    phase = yaml.safe_load((out/'scenario.yaml').read_text())['mission_session']['phases'][0]
    assert (phase['flight_parameters']['values'] if phase['flight_parameters'] else None) == expected


@pytest.mark.parametrize('name,reason', [('MC_ROLL_P', 'gain changed'), ('MPC_XY_CRUISE', 'speed changed')])
def test_inflight_parameter_changes_are_rejected(tmp_path, name, reason, capsys):
    p, log = save(tmp_path, *fixture(), changes=[(5_000_000, name, 8.)])
    with pytest.raises(SystemExit):
        main(['--input', str(p), '--flight-log', str(log), '--outdir', str(tmp_path/'output')])
    assert reason in capsys.readouterr().err
    assert not (tmp_path/'output').exists()


def test_repeated_flights_have_independent_markers_and_ground_wait(tmp_path):
    plan, data = fixture()
    for rows in data.values():
        extra = copy.deepcopy(rows)
        for row in extra:
            row['timestamp'] += 12_000_000
            if row.get('armed_time'): row['armed_time'] += 12_000_000
        rows.extend(extra)
    p, log = save(tmp_path, plan, data)
    out = tmp_path/'output'
    main(['--input', str(p), '--flight-log', str(log), '--outdir', str(out)])
    phases = yaml.safe_load((out/'scenario.yaml').read_text())['mission_session']['phases']
    assert len(phases) == 2
    assert phases[1]['ground_wait_s'] == pytest.approx(3.1)
    assert phases[1]['gain_capture']['snapshot'] == 'time:14.010000'


def test_header_guard_rejects_ulog_before_bin_reader(tmp_path):
    from tools.scenario.ardupilot_mission import read_log
    _, log = save(tmp_path, *fixture())
    with pytest.raises(ValueError, match='requires a BIN log'):
        read_log(log)


def test_explicit_speed_command_requires_matching_issued_parameters(tmp_path):
    plan, data = fixture()
    params = [1., 3., -1., 0., 0., 0., 0.]
    plan['mission']['items'].insert(2, dict(type='SimpleItem', command=178, frame=2,
                                          autoContinue=True, params=params))
    for r in data['mission_result']:
        r['seq_total'] = 4
        if r['seq_current'] >= 2: r['seq_current'] += 1
        if r['seq_reached'] >= 1: r['seq_reached'] += 1
    for r in data['navigator_mission_item']:
        if r['sequence_current'] == 2: r['sequence_current'] = 3
    data['navigator_mission_item'].append(dict(data['navigator_mission_item'][0],
        timestamp=4_500_010, sequence_current=2, nav_cmd=178))
    issued = dict(timestamp=4_500_000, command=178, from_external=False,
                  **{f'param{i+1}': v for i,v in enumerate(params)})
    data['vehicle_command'].append(issued)
    p, path = save(tmp_path, plan, data)
    selected, _ = read_flight_log(path).match_executions([p])
    assert selected[0]['executed_sequence'] == [0, 1, 2, 3]
    issued['param2'] = 8.
    native_log(path, data)
    with pytest.raises(ValueError, match='CHANGE_SPEED parameters not confirmed'):
        read_flight_log(path).match_executions([p])


@pytest.mark.parametrize('mode', ['plan', 'option'])
def test_log_plan_and_requested_autopilot_must_agree(tmp_path, mode, capsys):
    plan, data = fixture()
    options = []
    if mode == 'plan': plan['mission']['firmwareType'] = 3
    else: options = ['--autopilots', 'ardupilot']
    p, log = save(tmp_path, plan, data)
    with pytest.raises(SystemExit):
        main(['--input', str(p), '--flight-log', str(log), '--outdir', str(tmp_path/'output'), *options])
    assert 'firmware' in capsys.readouterr().err
    assert not (tmp_path/'output').exists()


def test_log_speed_does_not_depend_on_plan_hover_speed_metadata(tmp_path):
    plan, data = fixture()
    del plan['mission']['hoverSpeed']
    p, log = save(tmp_path, plan, data)
    out = tmp_path/'output'
    main(['--input', str(p), '--flight-log', str(log), '--outdir', str(out)])
    phase = yaml.safe_load((out/'scenario.yaml').read_text())['mission_session']['phases'][0]
    assert phase['flight_parameters']['values'] == {'MPC_XY_CRUISE': 5.}


def test_dropouts_and_position_resets_cannot_certify_a_complete_flight(tmp_path):
    plan, data = fixture()
    p, path = save(tmp_path, plan, data)
    with path.open('ab') as stream:
        stream.write(struct.pack('<HBH', 2, ord('O'), 100))
    with pytest.raises(ValueError, match='dropouts'):
        read_flight_log(path).match_executions([p])
    data['vehicle_global_position'][-1]['alt_reset_counter'] = 1
    native_log(path, data)
    log = read_flight_log(path)
    selected, _ = log.match_executions([p])
    with pytest.raises(ValueError, match='alt_reset_counter'):
        log.positions(selected[0])


def test_dry_run_and_existing_destination_are_not_written(tmp_path, capsys):
    p, log = save(tmp_path, *fixture())
    out = tmp_path/'output'
    args = ['--input', str(p), '--flight-log', str(log), '--outdir', str(out)]
    main([*args, '--dry-run'])
    assert not out.exists()
    report = json.loads(capsys.readouterr().out)
    assert report['source_log_format'] == 'ulog'
    out.mkdir()
    marker = out/'keep.txt'; marker.write_text('existing output')
    with pytest.raises(SystemExit): main(args)
    assert marker.read_text() == 'existing output'
    assert list(out.iterdir()) == [marker]
