import time
import threading
from pymavlink import mavutil

connection = mavutil.mavlink_connection('udp:127.0.0.1:14540')
connection.wait_heartbeat()
print(f"드론 연결 성공 (System ID: {connection.target_system})")

target_x, target_y, target_z = 0.0, 0.0, -5.0

def setpoint_streamer():
    while True:
        connection.mav.set_position_target_local_ned_send(
            0, connection.target_system, connection.target_component,
            mavutil.mavlink.MAV_FRAME_LOCAL_NED,
            0b0000111111111000, # 위치 제어 마스크
            target_x, target_y, target_z,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        )
        time.sleep(0.1)

stream_thread = threading.Thread(target=setpoint_streamer, daemon=True)
stream_thread.start()

time.sleep(2.0)

print("시동(Arm)")
connection.mav.command_long_send(
    connection.target_system, connection.target_component,
    mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM, 0, 
    1.0, 21196.0, 0.0, 0.0, 0.0, 0.0, 0.0
)
time.sleep(1.0) 

print("Offboard 모드 드가자")
# mode_id = connection.mode_mapping()['OFFBOARD']

# if isinstance(mode_id, tuple):
#     mode_id_float = float(mode_id[0])
# else:
#     mode_id_float = float(mode_id)

# print(f"OFFBOARD 모드 ID: {mode_id_float}")
connection.mav.command_long_send(
    connection.target_system, connection.target_component,
    mavutil.mavlink.MAV_CMD_DO_SET_MODE, 0,
    float(mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED), 6.0, 0.0, 0.0, 0.0, 0.0, 0.0
)

waypoints = [(5.0, 0.0, -5.0), (5.0, 5.0, -5.0), (0.0, 5.0, -5.0), (0.0, 0.0, -5.0)]

for wp in waypoints:
    print(f"목표 좌표 업데이트: {wp}")
    target_x, target_y, target_z = wp  # 스레드가 읽어가는 변수 업데이트
    time.sleep(10.0) # 10초간 해당 웨이포인트로 이동 및 유지 대기

print("사각형 비행 미션 완료!")