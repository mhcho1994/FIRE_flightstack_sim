"""Identify the executable, independently of a potentially newer source checkout."""

import hashlib
import json
import os
from pathlib import Path
import re
import secrets
import shlex
import signal
import socket
import subprocess
import tempfile
import time

from tools.commander.fc_parameter_registry import schema_for_firmware

PROBE_VERSION = 1


def binary_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def binary_path(root, autopilot, vehicle='ArduCopter'):
    root = Path(root).resolve()
    if autopilot == 'px4':
        return root / 'build/px4_sitl_default/bin/px4'
    if vehicle not in ('ArduCopter', 'copter'):
        raise ValueError('Native controller gain bridging supports ArduCopter only')
    return root / 'build/sitl/bin/arducopter'


def _stop(process):
    if process.poll() is None:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        try:
            process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=3)


def _ardu_probe(binary, work, timeout):
    from pymavlink import mavutil
    from tools.commander.firmware_identity import read_firmware_identity
    sockets = []
    for _ in range(50):
        base = 20000 + secrets.randbelow(30000)
        try:
            for offset in (0, 2, 3):
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sockets.append(sock)
                sock.bind(('127.0.0.1', base + offset))
            break
        except OSError:
            for sock in sockets:
                sock.close()
            sockets = []
    if not sockets:
        raise RuntimeError('No free ports for firmware probe')
    for sock in sockets:
        sock.close()
    command = [str(binary), '--model', 'x', '--base-port', str(base),
               '--home', '0,0,0,0', '-w', '--speedup', '1']
    connection = None
    with (work / 'probe.log').open('w+') as output:
        process = subprocess.Popen(command, cwd=work, stdout=output, stderr=subprocess.STDOUT,
                                   start_new_session=True)
        deadline = time.monotonic() + timeout
        def check():
            if time.monotonic() >= deadline or process.poll() is not None:
                output.flush()
                raise RuntimeError('ArduPilot firmware probe failed/timed out: ' +
                                   (work / 'probe.log').read_text()[-2000:])
        try:
            while connection is None:
                check()
                try:
                    connection = mavutil.mavlink_connection(f'tcp:127.0.0.1:{base}',
                        source_system=245, retries=1, autoreconnect=False)
                except OSError:
                    time.sleep(.1)
            while True:
                check()
                hb = connection.recv_match(type='HEARTBEAT', blocking=True, timeout=.2)
                if hb is not None and hb.get_srcComponent() == 1:
                    connection.target_system = hb.get_srcSystem()
                    connection.target_component = hb.get_srcComponent()
                    break
            return read_firmware_identity(connection, 'ardupilot', check_stop=check)
        finally:
            if connection is not None:
                connection.close()
            _stop(process)


def _px4_probe(binary, work, timeout):
    # Unique instance, private rootfs, and only ver/shutdown; no simulator or mission.
    instance = 100000 + secrets.randbelow(1000000)
    for name in ('ver', 'shutdown'):
        (work / f'px4-{name}').symlink_to(binary)
    script = work / 'probe.sh'
    script.write_text(f'{shlex.quote(str(work / "px4-ver"))} --instance "$1" all\n'
                      f'{shlex.quote(str(work / "px4-shutdown"))} --instance "$1"\n')
    # PX4 passes -s through a shell; the relative filename contains no metacharacters.
    command = [str(binary), '-d', '-i', str(instance), '-w', str(work), '-s', 'probe.sh']
    with (work / 'probe.log').open('w+') as output:
        process = subprocess.Popen(command, cwd=work, stdout=output, stderr=subprocess.STDOUT,
                                   start_new_session=True)
        try:
            process.wait(timeout=timeout)
        finally:
            _stop(process)
        output.flush()
        text = (work / 'probe.log').read_text()
    version = re.search(r'PX4 version: (?:Release )?(\d+\.\d+\.\d+).*?\((\d+)\)', text)
    commit = re.search(r'PX4 git-hash: ([a-fA-F0-9]{7,40})', text)
    if process.returncode != 0 or version is None or commit is None:
        raise RuntimeError('PX4 firmware probe failed: ' + text[-2000:])
    return dict(autopilot='px4', version=version[1], version_type=int(version[2]) & 255,
                git_hash=commit[1].lower(), identity_source='ver all')


def add_version_hint(identity, root):
    """Use an exact release tag at the binary's commit, never the checkout branch."""
    identity = dict(identity)
    if identity.get('version') not in (None, '0.0.0') or not identity.get('git_hash'):
        return identity
    result = subprocess.run(['git', '-C', str(root), 'describe', '--tags', '--exact-match',
                             '--match', 'v[0-9]*', identity['git_hash']],
                            capture_output=True, text=True, timeout=10)
    if result.returncode == 0:
        match = re.fullmatch(r'v(1\.(?:15|16|17)\.\d+)', result.stdout.strip())
        if match:
            identity.update(schema_version_hint=match[1], schema_hint_source='binary commit release tag')
    return identity


def identify_firmware(root, autopilot, vehicle='ArduCopter', *, parameter_schema=None,
                      cache_dir=None, timeout=25):
    binary = binary_path(root, autopilot, vehicle)
    digest = binary_sha256(binary)
    cache = Path(cache_dir) if cache_dir is not None else Path(tempfile.gettempdir()) / f'fire-firmware-{os.getuid()}'
    cache.mkdir(parents=True, exist_ok=True, mode=0o700)
    cached = cache / f'{autopilot}-{digest}-v{PROBE_VERSION}.json'
    try:
        identity = json.loads(cached.read_text())
        if identity.get('binary_sha256') != digest or identity.get('autopilot') != autopilot:
            raise ValueError('Invalid firmware cache')
    except (OSError, ValueError, AttributeError):
        with tempfile.TemporaryDirectory(prefix='fire-firmware-probe-') as folder:
            identity = (_ardu_probe if autopilot == 'ardupilot' else _px4_probe)(binary, Path(folder), timeout)
        if binary_sha256(binary) != digest:
            raise ValueError('Firmware binary changed during probe')
        identity['binary_sha256'] = digest
        with tempfile.NamedTemporaryFile(mode='w', dir=cache, delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(json.dumps(identity, sort_keys=True) + '\n')
        temporary.replace(cached)
    identity = add_version_hint(identity, root)
    identity['binary'] = str(binary)
    identity['parameter_schema'] = schema_for_firmware(identity, parameter_schema)
    if parameter_schema is not None:
        identity['schema_selection'] = 'explicit'
    else:
        identity['schema_selection'] = 'firmware'
    return identity
