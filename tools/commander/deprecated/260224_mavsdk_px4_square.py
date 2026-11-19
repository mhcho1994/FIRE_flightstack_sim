#!/usr/bin/env python3
import asyncio
import math
import logging

from mavsdk import System
from mavsdk.mission import MissionItem, MissionPlan

logging.basicConfig(level=logging.INFO)

EARTH_RADIUS_M = 6378137.0  # WGS84 
NED_SQUARE_M = 5.0          # 5m square
ALT_M = 5.0                 # takeoff relative altitude 5m
SPEED_M_S = 2.0             # waypoint speed


def add_north_east_m_to_gps(lat_deg: float, lon_deg: float, north_m: float, east_m: float) -> tuple[float, float]:
    """
    compute (lat,lon) of the points that are away from HOME by (north_m, east_m).
    approximation valid for 5m level (equirectangular).
    """
    lat_rad = math.radians(lat_deg)
    dlat = north_m / EARTH_RADIUS_M
    dlon = east_m / (EARTH_RADIUS_M * math.cos(lat_rad))
    return (lat_deg + math.degrees(dlat), lon_deg + math.degrees(dlon))


async def wait_connected(drone: System) -> None:
    print("Waiting for drone to connect...")
    async for state in drone.core.connection_state():
        if state.is_connected:
            print("-- Connected")
            return


async def wait_global_ok(drone: System) -> None:
    print("Waiting for global position + home position...")
    async for health in drone.telemetry.health():
        if health.is_global_position_ok and health.is_home_position_ok:
            print("-- Global position & home position OK")
            return


async def get_home_latlon(drone: System) -> tuple[float, float]:
    # home() -> Position(lat/lon/alt)
    async for home in drone.telemetry.home():
        if not math.isnan(home.latitude_deg) and not math.isnan(home.longitude_deg):
            print(f"-- Home lat/lon: {home.latitude_deg:.8f}, {home.longitude_deg:.8f}")
            return home.latitude_deg, home.longitude_deg


async def print_mission_progress(drone: System) -> None:
    async for mp in drone.mission.mission_progress():
        print(f"Mission progress: {mp.current}/{mp.total}")


async def observe_is_in_air(drone: System, running_tasks: list[asyncio.Task]) -> None:
    """
    task clear after landing
    """
    was_in_air = False
    async for in_air in drone.telemetry.in_air():
        if in_air:
            was_in_air = True

        if was_in_air and not in_air:
            print("-- Landed. Stopping...")
            for t in running_tasks:
                t.cancel()
                try:
                    await t
                except asyncio.CancelledError:
                    pass
            return


async def run():
    drone = System()

    # PX4 SITL default QGC port 14540.
    await drone.connect(system_address="udpin://0.0.0.0:14540")

    await wait_connected(drone)
    await wait_global_ok(drone)

    home_lat, home_lon = await get_home_latlon(drone)

    # home N/E 5m square (clockwise/anti-clockwise)
    # (0,0) -> (5,0) -> (5,5) -> (0,5) -> (0,0)
    corners_ne = [
        (0.0, 0.0),
        (NED_SQUARE_M, 0.0),
        (NED_SQUARE_M, NED_SQUARE_M),
        (0.0, NED_SQUARE_M),
        (0.0, 0.0),
    ]

    mission_items = []

    # 1) TAKEOFF (vehicle_action=TAKEOFF)
    #    home lat/lon, relative_altitude_m=ALT_M
    mission_items.append(
        MissionItem(
            home_lat,
            home_lon,
            ALT_M,
            SPEED_M_S,
            False,  # takeoff -> no fly-through for safety
            float("nan"),
            float("nan"),
            MissionItem.CameraAction.NONE,
            float("nan"),
            float("nan"),
            1.0,          # acceptance_radius_m
            float("nan"), # yaw_deg
            float("nan"),
            MissionItem.VehicleAction.TAKEOFF,
        )
    )

    # 2) square WAYPOINTs
    for north_m, east_m in corners_ne[1:]:  # first (0,0) included in takeoff
        lat, lon = add_north_east_m_to_gps(home_lat, home_lon, north_m, east_m)
        mission_items.append(
            MissionItem(
                lat,
                lon,
                ALT_M,
                SPEED_M_S,
                True,  # fly-through (False -> stop at verticies)
                float("nan"),
                float("nan"),
                MissionItem.CameraAction.NONE,
                float("nan"),
                float("nan"),
                1.0,          # acceptance radius (m)
                float("nan"),
                float("nan"),
                MissionItem.VehicleAction.NONE,
            )
        )

    # 3) (optional) LAND
    # mission_items.append(
    #     MissionItem(
    #         float("nan"),
    #         float("nan"),
    #         ALT_M,
    #         SPEED_M_S,
    #         False,
    #         float("nan"),
    #         float("nan"),
    #         MissionItem.CameraAction.NONE,
    #         float("nan"),
    #         float("nan"),
    #         1.0,
    #         float("nan"),
    #         float("nan"),
    #         MissionItem.VehicleAction.LAND,
    #     )
    # )

    mission_plan = MissionPlan(mission_items)

    # QGC RTL -> True
    await drone.mission.set_return_to_launch_after_mission(True)

    print("-- Uploading mission")
    await drone.mission.upload_mission(mission_plan)

    progress_task = asyncio.create_task(print_mission_progress(drone))
    termination_task = asyncio.create_task(observe_is_in_air(drone, [progress_task]))

    print("-- Arming")
    await drone.action.arm()

    print("-- Starting mission (AUTO.MISSION)")
    await drone.mission.start_mission()

    await termination_task


if __name__ == "__main__":
    asyncio.run(run())