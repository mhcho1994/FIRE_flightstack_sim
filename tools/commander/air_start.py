"""Prepare a PX4 in-air mission start without restoring hidden controller state."""
from __future__ import annotations
import math
import time
from pymavlink import mavutil


def state_errors(actual, position, velocity, yaw):
    return dict(position_m=math.dist(actual['position_ned_m'], position),
                velocity_m_s=math.dist(actual['velocity_ned_m_s'], velocity),
                yaw_rad=abs(math.atan2(math.sin(actual['yaw_rad']-yaw), math.cos(actual['yaw_rad']-yaw))))


def within_tolerance(errors, setup):
    return all(errors[key] <= setup[limit] for key, limit in (
        ('position_m', 'position_tolerance_m'), ('velocity_m_s', 'velocity_tolerance_m_s'),
        ('yaw_rad', 'yaw_tolerance_rad')))


class AirStartPreparation:
    """Settle at a lead-in point, then approach the target with logged velocity."""

    def __init__(self, state):
        self.state, self.setup = state, state['setup']
        self.goal, self.velocity, self.yaw = state['position_ned_m'], state['velocity_ned_m_s'], state['yaw_rad']
        self.lead_s = max(3., math.dist(self.velocity, [0., 0., 0.]))
        if self.velocity[2] < 0:
            self.lead_s = min(self.lead_s, (-self.goal[2]-.5)/-self.velocity[2])
        if self.lead_s <= 0:
            raise ValueError('No above-ground lead-in for the requested initial velocity')
        self.lead = [p-v*self.lead_s for p, v in zip(self.goal, self.velocity)]
        self.stable_since_ms = self.moving_since_ms = None

    def target(self, boot_ms):
        if self.moving_since_ms is None:
            return self.lead, [0., 0., 0.]
        elapsed = (boot_ms-self.moving_since_ms)/1000.
        if elapsed > self.lead_s + 5:
            raise TimeoutError('Setup passed the initial state without matching position/velocity/yaw')
        position = [p+v*elapsed for p, v in zip(self.lead, self.velocity)]
        if position[2] > -.3:
            raise RuntimeError('Setup trajectory would reach ground before matching the initial state')
        return position, self.velocity

    def observe(self, actual):
        boot_ms = actual['boot_time_us']/1000.
        if self.moving_since_ms is None:
            errors = state_errors(actual, self.lead, [0., 0., 0.], self.yaw)
            if within_tolerance(errors, self.setup):
                if self.stable_since_ms is None:
                    self.stable_since_ms = boot_ms
                if boot_ms-self.stable_since_ms >= self.setup['settle_time_s']*1000:
                    self.moving_since_ms = boot_ms
            else:
                self.stable_since_ms = None
            return None
        errors = state_errors(actual, self.goal, self.velocity, self.yaw)
        if within_tolerance(errors, self.setup):
            return dict(policy=self.setup['policy'], matched=True, actual=actual,
                        errors=errors, tolerances=self.setup, internal_state_restored=False)
        return None


def prepare_air_start(runner, connection, phase):
    from tools.scenario.scenario_generator import ned_to_lla
    from tools.scenario.import_real_flight_mission import lla_to_ned
    preparation = AirStartPreparation(phase['initial_state'])
    origin, sequence = phase['_origin'], phase['start_source_sequence']
    targets = [i for i, item in enumerate(phase['upload_items']) if item.get('source_sequence') == sequence]
    if len(targets) != 1:
        raise ValueError('Initial mission sequence has no unique upload mapping')
    for message_id, hz in ((33, 20), (30, 20), (42, 5), (245, 5)):
        connection.mav.command_long_send(connection.target_system, connection.target_component,
            mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL, 0, message_id, 1e6/hz, 0, 0, 0, 0, 0)
    connection.mav.mission_set_current_send(connection.target_system, connection.target_component, targets[0])
    started = time.monotonic()
    last_send = last_command = started - 2
    boot_ms = 0
    attitude = None
    armed = offboard = False
    current = None
    last_telemetry = started
    try:
        while time.monotonic()-started < preparation.setup['timeout_s']:
            runner._check_stop()
            now = time.monotonic()
            if now-last_telemetry > 5:
                raise TimeoutError('Setup position telemetry stopped')
            if now-last_send >= .05:
                position, velocity = preparation.target(boot_ms)
                lat, lon, height = ned_to_lla(position, origin)
                mask = (mavutil.mavlink.POSITION_TARGET_TYPEMASK_AX_IGNORE |
                        mavutil.mavlink.POSITION_TARGET_TYPEMASK_AY_IGNORE |
                        mavutil.mavlink.POSITION_TARGET_TYPEMASK_AZ_IGNORE |
                        mavutil.mavlink.POSITION_TARGET_TYPEMASK_YAW_RATE_IGNORE)
                connection.mav.set_position_target_global_int_send(int(boot_ms),
                    connection.target_system, connection.target_component,
                    mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT_INT, mask,
                    round(lat*1e7), round(lon*1e7), height, *velocity, 0., 0., 0., preparation.yaw, 0.)
                last_send = now
            if now-started >= 1.2 and now-last_command >= 1:
                if current != targets[0]:
                    connection.mav.mission_set_current_send(connection.target_system, connection.target_component, targets[0])
                if not offboard:
                    connection.mav.set_mode_send(connection.target_system,
                        mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED, 6 << 16)
                elif not armed:
                    connection.mav.command_long_send(connection.target_system, connection.target_component,
                        mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM, 0, 1, 0, 0, 0, 0, 0, 0)
                last_command = now
            message = connection.recv_match(blocking=True, timeout=.02)
            if message is None or (message.get_srcSystem(), message.get_srcComponent()) != (connection.target_system, connection.target_component):
                continue
            kind = message.get_type()
            if kind == 'HEARTBEAT':
                was_armed = armed
                armed = bool(message.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED)
                offboard = ((message.custom_mode >> 16) & 0xff) == 6
                if was_armed and (not armed or not offboard):
                    raise RuntimeError('Unexpected disarm or mode change during in-air setup')
            elif kind == 'MISSION_CURRENT':
                current = message.seq
            elif kind == 'ATTITUDE':
                attitude = message
            elif kind == 'STATUSTEXT':
                runner.session_observe(message)
            elif kind == 'GLOBAL_POSITION_INT':
                if message.time_boot_ms < boot_ms:
                    continue
                boot_ms = message.time_boot_ms
                last_telemetry = now
                if not armed or not offboard or current != targets[0] or attitude is None:
                    continue
                if abs(attitude.time_boot_ms - boot_ms) > 200:
                    continue
                height = message.relative_alt/1000.
                actual = dict(boot_time_us=boot_ms*1000,
                    position_ned_m=list(lla_to_ned([message.lat/1e7, message.lon/1e7, height], origin)),
                    velocity_ned_m_s=[message.vx/100., message.vy/100., message.vz/100.],
                    yaw_rad=attitude.yaw, attitude_rpy_rad=[attitude.roll, attitude.pitch, attitude.yaw],
                    angular_velocity_body_rad_s=[attitude.rollspeed, attitude.pitchspeed, attitude.yawspeed])
                if not all(math.isfinite(v) for value in actual.values()
                           for v in (value if isinstance(value, list) else [value])):
                    raise ValueError('Nonfinite setup telemetry')
                matched = preparation.observe(actual)
                if matched is not None:
                    matched['start_source_sequence'] = sequence
                    return matched
        raise TimeoutError('Initial position/velocity/yaw did not reach setup tolerances')
    except BaseException:
        if armed:
            runner.session_set_mode(connection, 'LAND')
        raise
