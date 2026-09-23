#!/usr/bin/env python3
"""
A library of higher-level servo movements for the SparkFun Pi Servo pHAT.

Uses the calibration and position-cache logic from servo_control.py, so:
  - Angles are in physical degrees (0-225 mechanically; clamped 5-220 in use)
  - Final position of every movement is recorded to the same state file
    that `servo_control.py --status` reads
  - relax() uses the same channel-write the rest of the toolkit uses

Both files must live in the same directory.

Run directly to see a demo that walks through every movement on channel 0:
    python3 servo_movements.py
    python3 servo_movements.py --channel 2
    python3 servo_movements.py --status        # report current position

Import in your own scripts:
    from servo_movements import ease_to, oscillate, scan, lifelike_idle
"""

import argparse
import math
import random
import signal
import sys
import time

import pi_servo_hat

# Re-use the calibration, persistence, and helpers from servo_control.
# Both files are expected to live in the same directory.
from servo_control import (
    MIN_ANGLE,
    MAX_ANGLE,
    angle_to_library,
    _record_position,
    print_position,
    relax,
    install_signal_handlers,
)


# ---- Tunables -----------------------------------------------------------
PWM_TICK = 0.02   # 50 Hz update rate. Don't push much faster than this.
# A natural "centre" is the middle of the usable range.
CENTER = (MIN_ANGLE + MAX_ANGLE) / 2     # ~112 degrees
SPAN = MAX_ANGLE - MIN_ANGLE             # 215 degrees of usable travel
# -------------------------------------------------------------------------


# ---- Easing curves ------------------------------------------------------
# Each takes t in [0, 1] and returns a "warped" t in [0, 1].

def linear(t: float) -> float:
    return t

def ease_in_out_sine(t: float) -> float:
    return 0.5 * (1 - math.cos(math.pi * t))

def ease_in_out_cubic(t: float) -> float:
    if t < 0.5:
        return 4 * t * t * t
    p = 2 * t - 2
    return 1 + 0.5 * p * p * p

def ease_out_back(t: float) -> float:
    """Overshoots slightly then settles -- feels mechanical/snappy."""
    c1 = 1.70158
    c3 = c1 + 1
    return 1 + c3 * (t - 1) ** 3 + c1 * (t - 1) ** 2
# -------------------------------------------------------------------------


# ---- Internal helpers ---------------------------------------------------

def _clamp(angle: float) -> float:
    return max(MIN_ANGLE, min(MAX_ANGLE, angle))


def _write(hat, channel: int, angle: float) -> None:
    """Send a single position to the servo, without touching the cache.
    Used inside tight motion loops so we don't do 50 disk writes/second."""
    hat.move_servo_position(channel, angle_to_library(_clamp(angle)))


def _commit(channel: int, angle: float) -> None:
    """Record the final angle once a movement completes."""
    _record_position(channel, _clamp(angle))


# ---- Movements ----------------------------------------------------------

def ease_to(hat, channel: int, start: float, end: float,
            duration: float = 1.0, ease=ease_in_out_sine) -> None:
    """Move smoothly from `start` to `end` over `duration` seconds."""
    steps = max(2, int(duration / PWM_TICK))
    for i in range(steps + 1):
        t = i / steps
        angle = start + (end - start) * ease(t)
        _write(hat, channel, angle)
        time.sleep(PWM_TICK)
    _commit(channel, end)


def oscillate(hat, channel: int, center: float = CENTER, amplitude: float = 40,
              period: float = 2.0, cycles: float = 3) -> None:
    """
    Sinusoidal motion around `center` +/- `amplitude` degrees.
    `period` = seconds per full cycle. Good for breathing/idle effects.
    """
    total = period * cycles
    start = time.monotonic()
    while True:
        elapsed = time.monotonic() - start
        if elapsed >= total:
            break
        phase = 2 * math.pi * elapsed / period
        _write(hat, channel, center + amplitude * math.sin(phase))
        time.sleep(PWM_TICK)
    _write(hat, channel, center)
    _commit(channel, center)


def scan(hat, channel: int, low: float = 30, high: float = 190,
         dwell: float = 0.4, sweep_time: float = 1.2, passes: int = 2) -> None:
    """
    Security-camera-style scan: sweep low->high with a dwell at each end.
    """
    for _ in range(passes):
        ease_to(hat, channel, low, high, duration=sweep_time)
        time.sleep(dwell)
        ease_to(hat, channel, high, low, duration=sweep_time)
        time.sleep(dwell)
    _commit(channel, low)


def lifelike_idle(hat, channel: int, center: float = CENTER,
                  jitter: float = 15, duration: float = 8.0,
                  min_pause: float = 0.4, max_pause: float = 2.5) -> None:
    """
    Random small moves around `center` for `duration` seconds.
    Mimics a creature that is awake but not focused on anything.
    """
    end = time.monotonic() + duration
    current = center
    while time.monotonic() < end:
        target = center + random.uniform(-jitter, jitter)
        ease_to(hat, channel, current, target,
                duration=random.uniform(0.25, 0.6),
                ease=ease_in_out_sine)
        current = target
        time.sleep(random.uniform(min_pause, max_pause))
    ease_to(hat, channel, current, center, duration=0.4)


def twitch(hat, channel: int, base: float = CENTER,
           offset: float = 45, hold: float = 0.15) -> None:
    """Snap to base+offset, hold briefly, snap back. Reaction-style."""
    ease_to(hat, channel, base, base + offset, duration=0.12, ease=ease_out_back)
    time.sleep(hold)
    ease_to(hat, channel, base + offset, base, duration=0.18, ease=ease_in_out_cubic)


def keyframe_sequence(hat, channel: int, frames) -> None:
    """
    Play a list of keyframes. Each frame is (angle, duration, ease_fn).
    Starts at the first frame's angle (snapped instantly).
    Example (angles in physical degrees):
        keyframe_sequence(hat, 0, [
            (60,  0.0, linear),               # snap to start
            (165, 1.5, ease_in_out_sine),
            (110, 0.8, ease_out_back),
            (110, 0.5, linear),               # hold
        ])
    """
    if not frames:
        return
    current = frames[0][0]
    _write(hat, channel, current)
    for angle, duration, ease in frames[1:]:
        if duration <= 0:
            _write(hat, channel, angle)
        else:
            ease_to(hat, channel, current, angle, duration=duration, ease=ease)
        current = angle
    _commit(channel, current)


def wave(hat, channels, center: float = CENTER, amplitude: float = 40,
         period: float = 1.5, cycles: float = 3, phase_step: float = 0.25) -> None:
    """
    Run a sine wave across multiple channels with a phase offset between
    each, so motion ripples along the row. `phase_step` is in cycles
    (0.25 = quarter-wave between adjacent servos).
    """
    total = period * cycles
    start = time.monotonic()
    while True:
        elapsed = time.monotonic() - start
        if elapsed >= total:
            break
        for i, ch in enumerate(channels):
            phase = 2 * math.pi * (elapsed / period - i * phase_step)
            _write(hat, ch, center + amplitude * math.sin(phase))
        time.sleep(PWM_TICK)
    for ch in channels:
        _write(hat, ch, center)
        _commit(ch, center)


# ---- Demo --------------------------------------------------------------

def demo(hat, channel: int) -> None:
    print("1) Eased moves between presets")
    ease_to(hat, channel, CENTER, MIN_ANGLE + 30, duration=0.8)
    ease_to(hat, channel, MIN_ANGLE + 30, MAX_ANGLE - 30, duration=1.4,
            ease=ease_in_out_cubic)
    ease_to(hat, channel, MAX_ANGLE - 30, CENTER, duration=0.8, ease=ease_out_back)
    time.sleep(0.5)
    print_position(hat, channel)

    print("\n2) Oscillate (breathing)")
    oscillate(hat, channel, center=CENTER, amplitude=25, period=2.0, cycles=3)
    time.sleep(0.5)
    print_position(hat, channel)

    print("\n3) Scan with dwell")
    scan(hat, channel, low=30, high=190, dwell=0.4, sweep_time=1.2, passes=2)
    time.sleep(0.5)
    print_position(hat, channel)

    print("\n4) Twitch (reaction)")
    twitch(hat, channel, base=CENTER, offset=45)
    time.sleep(0.3)
    twitch(hat, channel, base=CENTER, offset=-45)
    time.sleep(0.5)
    print_position(hat, channel)

    print("\n5) Lifelike idle (5 seconds)")
    lifelike_idle(hat, channel, center=CENTER, jitter=18, duration=5.0)
    time.sleep(0.5)
    print_position(hat, channel)

    print("\n6) Keyframe sequence")
    keyframe_sequence(hat, channel, [
        (60,  0.0, linear),
        (165, 1.4, ease_in_out_sine),
        (110, 0.6, ease_out_back),
        (110, 0.4, linear),
        (200, 0.6, ease_in_out_cubic),
        (15,  1.2, ease_in_out_cubic),
        (110, 0.9, ease_in_out_sine),
    ])
    print_position(hat, channel)

    print("\nDone.")


# ---- CLI ----------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Demo of complex servo movements.")
    p.add_argument("--channel", type=int, default=0, help="Servo channel 0-15")
    p.add_argument("--status", action="store_true",
                   help="Report the servo's current position and exit")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    if not 0 <= args.channel <= 15:
        print("Channel must be between 0 and 15", file=sys.stderr)
        return 2

    hat = pi_servo_hat.PiServoHat()
    hat.restart()

    if args.status:
        # Read-only mode: don't drive the servo, don't relax it on exit.
        print_position(hat, args.channel)
        return 0

    install_signal_handlers(hat, args.channel)
    try:
        #demo(hat, args.channel)
        # One final position report after everything settles.
        for i in range(3):
            wave(hat, [args.channel], center=CENTER, amplitude=200, period=1.5, cycles=4)
            print_position(hat, args.channel)
    finally:
        relax(hat, args.channel)
    return 0


if __name__ == "__main__":
    sys.exit(main())