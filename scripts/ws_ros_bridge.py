#!/usr/bin/env python3
import sys, time
import roslaunch

def main():
    port = sys.argv[1] if len(sys.argv) > 1 else "9090"

    uuid = roslaunch.rlutil.get_or_generate_uuid(None, False)
    roslaunch.configure_logging(uuid)

    launch_file = roslaunch.rlutil.resolve_launch_arguments(
        ["rosbridge_server", "rosbridge_websocket.launch"]
    )[0]

    launch = roslaunch.parent.ROSLaunchParent(uuid, [(launch_file, [f"port:={port}", "address:=0.0.0.0"])])
    launch.start()
    print(f"[ws_ros_bridge] rosbridge_websocket up on ws://0.0.0.0:{port}")

    try:
        # Keep the process alive until Ctrl+C
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        pass
    finally:
        launch.shutdown()

if __name__ == "__main__":
    main()
