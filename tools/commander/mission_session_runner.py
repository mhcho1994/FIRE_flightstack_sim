"""One mission lifecycle for ArduPilot/PX4, single flights and multi-flight sessions."""
from __future__ import annotations

import copy
import csv
import hashlib
import json
import time
from pathlib import Path

import yaml
from pymavlink import mavutil

from tools.commander.mission_session import EXECUTION_VERSION, normalize_session, home_item
from tools.commander.mission_wind_adapter import WindPolicy, WindMissionMonitor, add_wind_checkpoints
from tools.commander.mavlink_parameters import ParameterClient, values_match, verify_controller_parameters, verify_flight_parameters
from tools.commander.firmware_identity import verify_firmware_identity
from tools.launcher.parameter_bridge import bridge_gains, bridge_flight


def phase_disarm_ok(airborne, final_land_started, on_ground):
    return airborne and final_land_started and on_ground


def monitor_phase(runner, m, phase, start_ms, origin, writer, stream, entry, wind_monitor=None):
    from tools.scenario.import_real_flight_mission import lla_to_ned
    started = time.monotonic()
    last_clock_progress = last_position_at = landed_at = started
    last_ms = start_ms
    airborne = final_land = on_ground = False
    altitude = vertical_speed = None
    stable_since = None
    sequence = []
    disarm_seen = False
    disarm_at = None
    while time.monotonic() - started < phase['wall_timeout_s']:
        runner._check_stop()
        msg = m.recv_match(blocking=True, timeout=.1)
        now = time.monotonic()
        if now - last_clock_progress > 15:
            raise TimeoutError('Mission position clock stopped or telemetry was lost')
        if msg is not None and (msg.get_srcSystem(), msg.get_srcComponent()) != (m.target_system, m.target_component):
            msg = None
        if wind_monitor is not None:
            wind_monitor.update(msg)
        if disarm_seen and now - disarm_at > 2:
            raise RuntimeError(f'{phase["id"]}: disarm without fresh landing confirmation')
        if msg is None:
            continue
        kind = msg.get_type()
        if kind == 'GLOBAL_POSITION_INT':
            if msg.time_boot_ms < last_ms:
                continue
            if msg.time_boot_ms > last_ms:
                last_clock_progress = now
            last_ms, last_position_at = msg.time_boot_ms, now
            if (last_ms - start_ms) / 1000. > phase['timeout_s']:
                raise TimeoutError(f'{phase["id"]}: simulated flight duration exceeded')
            altitude, vertical_speed = msg.relative_alt / 1000., msg.vz / 100.
            airborne = airborne or altitude > 1.
            if altitude > 1.:
                on_ground = False
            low = abs(altitude) < .3 and abs(vertical_speed) < .5
            stable_since = (last_ms if stable_since is None else stable_since) if low else None
            n, e, _ = lla_to_ned([msg.lat/1e7, msg.lon/1e7, altitude], origin)
            writer.writerow([phase['id'], last_ms*1000, (last_ms-start_ms)/1000., n, e, altitude])
        elif kind == 'EXTENDED_SYS_STATE':
            on_ground = msg.landed_state == mavutil.mavlink.MAV_LANDED_STATE_ON_GROUND
            landed_at = now
            airborne = airborne or msg.landed_state == mavutil.mavlink.MAV_LANDED_STATE_IN_AIR
        elif kind == 'MISSION_CURRENT':
            if not sequence or sequence[-1] != msg.seq:
                sequence.append(msg.seq)
            final_land = final_land or msg.seq == len(phase['upload_items']) - 1
        elif kind == 'STATUSTEXT':
            runner.session_observe(msg)
        elif kind == 'HEARTBEAT':
            armed = bool(msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED)
            if not armed:
                if not airborne or not final_land:
                    raise RuntimeError(f'{phase["id"]}: unexpected disarm before final LAND')
                disarm_seen, disarm_at = True, disarm_at or now
            elif disarm_seen:
                raise RuntimeError('Vehicle rearmed before phase completion')
            elif now > started + 3 and not runner.session_mode_ok(msg, final_land):
                raise RuntimeError(f'{phase["id"]}: unexpected mode change to {msg.custom_mode}')
        fresh_position = altitude is not None and now - last_position_at < 2
        fallback_ground = fresh_position and stable_since is not None and last_ms - stable_since >= 1000
        # A stale landed flag cannot override current airborne altitude.
        ground = (on_ground and now - landed_at < 2 and (not fresh_position or altitude <= 1.)) or fallback_ground
        if disarm_seen and phase_disarm_ok(airborne, final_land, ground):
            if wind_monitor is not None and not wind_monitor.done:
                raise RuntimeError('Disarmed before completing wind checkpoints')
            entry.update(sitl_end_time_us=last_ms*1000, mission_sequence=sequence,
                         completion_reason='landed_disarmed')
            stream.flush()
            return
    raise TimeoutError(f'{phase["id"]}: wall-clock mission limit exceeded')


def wait_position(runner, connection):
    """Read the target's recent boot clock/position on the existing connection."""
    until = time.monotonic() + 5
    while time.monotonic() < until:
        runner._check_stop()
        msg = connection.recv_match(type='GLOBAL_POSITION_INT', blocking=True, timeout=.2)
        if msg is not None and (msg.get_srcSystem(), msg.get_srcComponent()) == (connection.target_system, connection.target_component):
            return msg
    raise TimeoutError('No position/boot clock before mission start')


class MissionSessionRunner:
    """The adapter owns firmware commands; this class owns session completion."""

    def __init__(self, adapter):
        self.runner = adapter

    def run(self):
        runner = self.runner
        state = type(runner.status.state)
        scenario_bytes = runner.scenario_path.read_bytes()
        digest = hashlib.sha256(scenario_bytes).hexdigest()
        scenario = yaml.safe_load(scenario_bytes)
        session = normalize_session(scenario, runner.autopilot)
        policy = WindPolicy.parse(scenario.get('common', {}).get('scenario', {}).get('wind_policy'))
        if policy.mode == 'mission_only' and runner.wind is None:
            raise ValueError('mission_only requires the Gazebo launcher and wind controller')
        boot = runner.boot_parameters
        if boot is not None and boot['scenario_sha256'] != digest:
            raise ValueError('Scenario changed after boot preparation')
        target = boot['target_firmware'] if boot is not None else runner.target_firmware
        if target is None:
            from tools.launcher.startup_parameters import identify_scenario_firmware
            target = identify_scenario_firmware(scenario, runner.autopilot)
        effective = []
        for phase in session['phases']:
            gains, gc = bridge_gains(phase['controller_gains'], runner.autopilot, target)
            flight, fc = bridge_flight(phase['flight_parameters'], runner.autopilot, target)
            values = {} if gains is None else dict(gains['values'])
            if flight:
                if values.keys() & flight['values'].keys():
                    raise ValueError('Controller and flight parameters overlap')
                values.update(flight['values'])
            effective.append((gains, flight, values, gc + fc))
        logs = runner.scenario_path.parent / ('ardu_logs' if runner.autopilot == 'ardupilot' else 'px4_logs')
        logs.mkdir(parents=True, exist_ok=True)
        path = logs / 'session_result.json'
        result = dict(version=1, execution_version=EXECUTION_VERSION, autopilot=runner.autopilot,
                      status='running', scenario_sha256=digest, target_firmware=target, phases=[])
        if session.get('source_log_sha256'):
            result['source_log_sha256'] = session['source_log_sha256']

        def save():
            temporary = path.with_suffix('.json.tmp')
            temporary.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
            temporary.replace(path)

        save()
        m = None
        try:
            runner._set_status(state.CONNECTING, 'connecting mission session')
            m = mavutil.mavlink_connection(runner.session_connect_url(scenario), source_system=255, source_component=0)
            runner._m = m
            heartbeat = m.wait_heartbeat(timeout=60)
            if heartbeat is None:
                raise TimeoutError('Session heartbeat timeout')
            m.target_system, m.target_component = heartbeat.get_srcSystem(), heartbeat.get_srcComponent()
            verify_firmware_identity(m, target, log_path=logs/'firmware_identity.json',
                                     check_stop=runner._check_stop, on_message=runner.session_observe)
            client = ParameterClient(m, check_stop=runner._check_stop, on_message=runner.session_observe)
            client.require_disarmed(runner.autopilot)
            # First phase must already match startup files; do not conceal a bad boot.
            first_gains, first_flight, _, _ = effective[0]
            for config, verifier, name in [(first_gains, verify_controller_parameters, 'controller_gains'),
                                            (first_flight, verify_flight_parameters, 'flight_parameters')]:
                if config is not None:
                    verifier(m, config, runner.autopilot, logs/f'{name}.json',
                             check_stop=runner._check_stop, on_message=runner.session_observe)
            # Omitted values restore session-start values, never inherit an earlier phase accidentally.
            baseline = {name: client.get_param(name) for name in sorted(set().union(*(set(e[2]) for e in effective)))}
            with (logs/'session_trajectory.csv').open('w', newline='') as stream:
                writer = csv.writer(stream)
                writer.writerow(['phase', 'boot_time_us', 'time_s', 'north_m', 'east_m', 'up_m'])
                for index, (phase, (gains, flight, values, conversions)) in enumerate(zip(session['phases'], effective)):
                    entry = dict(id=phase['id'], status='preparing', conversions=conversions)
                    for key in ('plan', 'gain_capture'):
                        if key in phase:
                            entry[key] = phase[key]
                    for key in ('start_time_us', 'end_time_us'):
                        if key in phase:
                            entry['source_' + key] = phase[key]
                    result['phases'].append(entry); save()
                    runner._set_status(state.RUNNING, f'{phase["id"]}: preparing ({index+1}/{len(effective)})')
                    deadline = time.monotonic() + phase['ground_wait_s']
                    while time.monotonic() < deadline:
                        runner._check_stop()
                        msg = m.recv_match(blocking=True, timeout=.1)
                        if msg is not None:
                            runner.session_observe(msg)
                            if msg.get_type() == 'HEARTBEAT' and msg.get_srcSystem() == m.target_system and msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED:
                                raise RuntimeError('Unexpected arm during inter-flight ground wait')
                    client.require_disarmed(runner.autopilot)
                    entry['parameters'] = {}
                    for name, value in (baseline | values).items():
                        before = client.get_param(name)
                        if not values_match(before, value):
                            client.set_param(name, value)
                        actual = client.get_param(name)
                        if not values_match(actual, value):
                            raise RuntimeError(f'{name}: phase readback mismatch')
                        entry['parameters'][name] = dict(before=before, requested=value, actual=actual)
                    save()
                    runner.session_ready(m)
                    logical = copy.deepcopy(phase['mission_items'])
                    if runner.wind is not None:
                        logical, _ = add_wind_checkpoints(logical)
                        if runner.autopilot == 'ardupilot':
                            ground_position = wait_position(runner, m)
                            for item in logical:
                                if item.get('wind_checkpoint') == 'post_takeoff':
                                    item.update(p5=ground_position.lat/1e7, p6=ground_position.lon/1e7)
                    items = ([home_item(session['target_origin_lla'])] if runner.autopilot == 'ardupilot' else []) + logical
                    items = [{k: (float('nan') if k == 'p4' and v is None else v) for k, v in item.items()} for item in items]
                    phase['upload_items'] = items
                    entry['mission_item_map'] = [dict(upload_sequence=i, command=it['command'],
                        scenario_index=it.get('scenario_index'), source_sequence=it.get('source_sequence'),
                        wind_checkpoint=it.get('wind_checkpoint')) for i, it in enumerate(items)]
                    if not runner._clear_mission(m) or not runner._upload_mission_items(m, items):
                        raise RuntimeError('Mission clear/upload failed')
                    runner.session_prepare(m, phase)
                    for message_id, hz in ((33, 10), (245, 5), (42, 5), (30, 10)):
                        m.mav.command_long_send(m.target_system, m.target_component,
                            mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL, 0, message_id, 1e6/hz, 0, 0, 0, 0, 0)
                    position = wait_position(runner, m)
                    entry.update(status='running', sitl_start_time_us=position.time_boot_ms*1000); save()
                    wind_monitor = None
                    if runner.wind is not None:
                        wind_monitor = WindMissionMonitor(runner, m, items, runner.wind, phase['wall_timeout_s'], phase_id=phase['id'])
                    try:
                        runner.session_start(m, phase)
                        runner._set_status(state.RUNNING, f'{phase["id"]}: flying ({index+1}/{len(effective)})')
                        monitor_phase(runner, m, phase, position.time_boot_ms, session['target_origin_lla'],
                                      writer, stream, entry, wind_monitor)
                    finally:
                        if wind_monitor is not None:
                            wind_monitor.close()
                    entry['status'] = 'completed'; save()
            result['status'] = 'completed'; save()
            runner._set_status(state.COMPLETED, f'all {len(effective)} missions completed', done=True, success=True)
        except Exception as exc:
            result.update(status='interrupted' if runner._stop_requested else 'failed', error=str(exc))
            if result['phases'] and result['phases'][-1]['status'] != 'completed':
                result['phases'][-1].update(status=result['status'], error=str(exc))
            save()
            raise
        finally:
            if m is not None:
                m.close()
