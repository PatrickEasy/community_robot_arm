#!/usr/bin/env python3
"""
Servo control for Raspberry Pi 4B with SparkFun Pi Servo pHAT.

Calibrated for a ~225-degree servo. The SparkFun library only accepts
inputs up to about 140 before silently ignoring them, but the servo's
real travel is 0-225 degrees. This script lets you pass the actual
physical angle you want and handles the scaling internally.

Usage:
    python3 servo_control.py --angle 110              # halfway-ish
    python3 servo_control.py --angle 5                # near one end
    python3 servo_control.py --angle 220              # near the other end
    python3 servo_control.py --angle 90 --channel 2   # different channel
    python3 servo_control.py --angle 90 --hold 3      # hold 3 seconds
    python3 servo_control.py --demo                   # sweep + presets
    python3 servo_control.py --status                 # report current position

Setup (one-time):
    sudo raspi-config           # enable I2C
    sudo apt install -y i2c-tools
    pip3 install sparkfun-pi-servo-hat
"""

import argparse
import json
import os
import signal
import sys
import time
from datetime import datetime

import pi_servo_hat


# ---- Calibration --------------------------------------------------------
# The library's `move_servo_position` accepts input values in [0, LIBRARY_MAX]
# and maps that to the servo's full physical travel of [0, PHYSICAL_MAX] degrees.
# Above LIBRARY_MAX the library silently ignores the call.
LIBRARY_MAX = 140
PHYSICAL_MAX = 225

# Permitted physical angle range. Margin keeps the servo off the mechanical stops.
MIN_ANGLE = 5
MAX_ANGLE = 220

# Where to remember each channel's commanded angle between invocations.
STATE_FILE = os.path.expanduser("~/.cache/servo_control_state.json")
# -------------------------------------------------------------------------


# ---- Position cache (persists across script runs) -----------------------

def _load_state() -> dict:
    try:
        with open(STATE_FILE) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _save_state(state: dict) -> None:
    os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


def _record_position(channel: int, angle: float) -> None:
    """Remember the angle we last commanded on a channel."""
    state = _load_state()
    state[str(channel)] = {
        "angle_deg": angle,
        "timestamp": time.time(),
    }
    _save_state(state)
# -------------------------------------------------------------------------


def angle_to_library(angle: float) -> float:
    """Convert a physical angle in degrees to the library's input scale."""
    return angle * LIBRARY_MAX / PHYSICAL_MAX


def library_to_angle(library_value: float) -> float:
    """Convert a library input value back to physical degrees."""
    return library_value * PHYSICAL_MAX / LIBRARY_MAX


def move_to(hat, channel: int, angle: float) -> None:
    """Move the servo on `channel` to `angle` physical degrees."""
    angle = max(MIN_ANGLE, min(MAX_ANGLE, angle))
    hat.move_servo_position(channel, angle_to_library(angle))
    _record_position(channel, angle)


def get_position(hat, channel: int) -> dict:
    """
    Report what was last commanded on this channel and whether the chip
    is currently actively driving the servo. Returns a dict:

        {
            "angle_deg":      float | None,   # last commanded angle, if known
            "commanded_at":   float | None,   # unix timestamp of that command
            "is_driving":     bool,           # is the chip currently sending pulses?
            "pulse_ticks":    int  | None,    # raw OFF-register tick count
        }

    Caveats:
      - Hobby servos are open-loop; this is what we *commanded*, not measured.
      - The cache is per-machine and only updated when this script does the move.
        If something else moved the servo, the cache will be stale.
      - If the servo was relaxed (--hold finished), `is_driving` will be False
        but `angle_deg` will still report the last commanded angle.
    """
    cached = _load_state().get(str(channel), {})

    try:
        ticks = hat.PCA9685.get_channel_word(channel, 1)
    except Exception:
        ticks = None

    return {
        "angle_deg": cached.get("angle_deg"),
        "commanded_at": cached.get("timestamp"),
        "is_driving": bool(ticks) if ticks is not None else False,
        "pulse_ticks": ticks,
    }


def print_position(hat, channel: int) -> None:
    """Pretty-print the current position to stdout."""
    pos = get_position(hat, channel)

    if pos["angle_deg"] is None:
        print(f"Channel {channel}: no commanded position recorded yet "
              f"(this script hasn't moved it on this machine).")
    else:
        when = ""
        if pos["commanded_at"]:
            when = f" at {datetime.fromtimestamp(pos['commanded_at']).strftime('%H:%M:%S')}"
        print(f"Channel {channel}: last commanded {pos['angle_deg']:.1f} deg{when}.")

    if pos["is_driving"]:
        print(f"  Chip is currently driving the servo "
              f"({pos['pulse_ticks']} ticks).")
    else:
        print(f"  Chip is NOT currently driving the servo "
              f"(relaxed -- the servo will hold its last position by inertia "
              f"or drift if unloaded).")


def relax(hat, channel: int) -> None:
    """Stop driving the channel so the servo isn't holding torque."""
    try:
        hat.PCA9685.set_channel_word(channel, 1, 0)
    except (AttributeError, Exception):
        pass


def install_signal_handlers(hat, channel: int) -> None:
    """Make sure Ctrl-C / SIGTERM leaves the servo in a relaxed state."""
    def _handler(signum, _frame):
        print(f"\nCaught signal {signum}, releasing servo and exiting.")
        relax(hat, channel)
        sys.exit(0)
    signal.signal(signal.SIGINT, _handler)
    signal.signal(signal.SIGTERM, _handler)


def sweep(hat, channel: int, start: float, end: float, step: float = 4,
          delay: float = 0.02) -> None:
    """Smoothly sweep from `start` to `end` (in physical degrees)."""
    direction = step if end >= start else -step
    angle = start
    while (direction > 0 and angle <= end) or (direction < 0 and angle >= end):
        move_to(hat, channel, angle)
        time.sleep(delay)
        angle += direction
    move_to(hat, channel, end)


def demo(hat, channel: int) -> None:
    """Walk the servo through useful positions."""
    mid = (MIN_ANGLE + MAX_ANGLE) / 2

    print(f"Centering ({mid:.0f} deg)")
    move_to(hat, channel, mid)
    time.sleep(0.6)

    print(f"Sweeping {MIN_ANGLE} -> {MAX_ANGLE}")
    sweep(hat, channel, MIN_ANGLE, MAX_ANGLE)
    time.sleep(0.4)

    print(f"Sweeping {MAX_ANGLE} -> {MIN_ANGLE}")
    sweep(hat, channel, MAX_ANGLE, MIN_ANGLE)
    time.sleep(0.4)

    print("Hitting presets")
    for angle in (MIN_ANGLE, 60, 110, 160, MAX_ANGLE, mid):
        print(f"  -> {angle} deg")
        move_to(hat, channel, angle)
        time.sleep(0.6)

    print("Done.")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--channel", type=int, default=0,
                   help="Servo channel 0-15 (default: 0)")

    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--angle", type=float,
                     help=f"Physical angle in degrees ({MIN_ANGLE}-{MAX_ANGLE})")
    mode.add_argument("--demo", action="store_true",
                     help="Run a sweep + preset demo")
    mode.add_argument("--status", action="store_true",
                     help="Report the servo's current position and exit")

    p.add_argument("--hold", type=float, default=0.5,
                   help="Seconds to hold position before releasing (default: 0.5)")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    if not 0 <= args.channel <= 15:
        print("Channel must be between 0 and 15", file=sys.stderr)
        return 2

    if args.angle is not None and not MIN_ANGLE <= args.angle <= MAX_ANGLE:
        print(f"Angle must be between {MIN_ANGLE} and {MAX_ANGLE}",
              file=sys.stderr)
        return 2

    hat = pi_servo_hat.PiServoHat()
    hat.restart()
    install_signal_handlers(hat, args.channel)

    try:
        if args.status:
            # Don't relax the servo afterwards -- the user is just checking
            print_position(hat, args.channel)
            return 0
        elif args.demo:
            demo(hat, args.channel)
        elif args.angle is not None:
            print(f"Channel {args.channel} -> {args.angle} deg "
                  f"(library input: {angle_to_library(args.angle):.1f})")
            move_to(hat, args.channel, args.angle)
            time.sleep(args.hold)
            # Confirm by reading back from the chip
            print_position(hat, args.channel)
        else:
            print("Pass --angle <N>, --demo, or --status. Use --help for details.")
            return 1
    finally:
        # --status leaves the channel alone; everything else relaxes it
        if not args.status:
            relax(hat, args.channel)

    return 0


if __name__ == "__main__":
    sys.exit(main())