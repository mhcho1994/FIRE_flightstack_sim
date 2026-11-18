"""Translate frozen source-native values once, immediately before SITL boot."""

from tools.commander.fc_parameter_profiles import parse_controller_parameters
from tools.commander.fc_parameter_registry import LEGACY_SCHEMAS, parameter_names, schema_groups
from tools.commander.flight_parameters import parse_flight_parameters

BRIDGE_VERSION = 'native-parameters-v1'


def bridge_gains(config, autopilot, target):
    if config is None:
        return None, []
    source = parse_controller_parameters(config, autopilot)
    before = source.get('parameter_schema', LEGACY_SCHEMAS[autopilot])
    after = target['parameter_schema']
    schema_groups(autopilot, after)
    changes, baseline, values = [], {}, {}
    for group in source['groups']:
        old_names = parameter_names(autopilot, [group], before)
        new_names = parameter_names(autopilot, [group], after)
        if len(old_names) != len(new_names):
            raise ValueError(f'No complete {group} mapping from {before} to {after}')
        for old, new in zip(old_names, new_names):
            factor = 1.0
            if before != after:
                if {before, after} != {'arducopter-4.6', 'arducopter-4.7'}:
                    raise ValueError(f'Unsupported gain bridge: {before} -> {after}')
                if group == 'acceleration':
                    factor = .1 if after == 'arducopter-4.7' else 10.0
            if new in values:
                raise ValueError(f'Duplicate target parameter: {new}')
            baseline[new] = source['baseline'][old] * factor
            values[new] = source['values'][old] * factor
            if old != new or factor != 1:
                changes.append(dict(source_parameter=old, target_parameter=new, multiplier=factor,
                                    source_value=source['values'][old], target_value=values[new]))
    effective = parse_controller_parameters(dict(groups=source['groups'], baseline=baseline,
        values=values, parameter_schema=after, firmware=target), autopilot)
    return effective, changes


def bridge_flight(config, autopilot, target):
    source = parse_flight_parameters(config, autopilot)
    if source is None:
        return None, []
    old, value = next(iter(source['values'].items()))
    after = target['parameter_schema']
    schema_groups(autopilot, after)
    new = ('WP_SPD' if after == 'arducopter-4.7' else 'WPNAV_SPEED') if autopilot == 'ardupilot' else 'MPC_XY_CRUISE'
    factor = 1.0 if old == new else (.01 if new == 'WP_SPD' else 100.0)
    effective = parse_flight_parameters(dict(values={new: value * factor}, parameter_schema=after,
                                             source=source.get('source', 'scenario')), autopilot)
    changes = [] if old == new else [dict(source_parameter=old, target_parameter=new, multiplier=factor,
                                         source_value=value, target_value=effective['values'][new])]
    return effective, changes
