import asyncio
from mavsdk import System

async def print_statustext(drone):
    async for st in drone.telemetry.status_text():
        # PX4가 "Arming denied: ..."
        print(f"[STATUSTEXT] {st.type}: {st.text}")

async def wait_armable(drone):
    async for h in drone.telemetry.health():
        print("health:", h)
        if h.is_armable:
            print("-- is_armable = True")
            return

async def main():
    drone = System()
    await drone.connect(system_address="udpin://0.0.0.0:14540")

    # Wait for the drone to connect
    async for cs in drone.core.connection_state():
        if cs.is_connected:
            break

    asyncio.create_task(print_statustext(drone))

    # Check the error message while waiting for armable
    await wait_armable(drone)

    print("arming...")
    await drone.action.arm()
    print("armed!")

asyncio.run(main())