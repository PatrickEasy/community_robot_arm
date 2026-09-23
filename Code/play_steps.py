#!/usr/bin/env python3
"""
Play a v6 step CSV on the real arm.  Runs on the Raspberry Pi.

v6.py works out the angles and writes them to a CSV; this reads that file and
drives the servos through it.  The split means the Pi needs no numpy and no
matplotlib, and every row it is handed has already been checked against the
arm's limits by the simulation - including main_arm's ceiling that tracks
control_arm_a, which the servo layer has no way of knowing about.

    python3 play_steps.py v6_steps_arc.csv --check        # validate, move nothing
    python3 play_steps.py v6_steps_arc.csv --dry-run      # full run, no hardware
    python3 play_steps.py v6_steps_arc.csv                # for real
    python3 play_steps.py v6_steps_arc.csv --speed 0.5    # half pace
    python3 play_steps.py --goto base_rotation=0,control_arm_a=90,main_arm=180

Why this doesn't just call servo_control.py per step:

  * servo_control.py is one-shot.  Each run calls hat.restart() and then
    relaxes the channel on the way out, so a path played that way would drop
    the arm limp between every step and fight the restart each time.
  * Its move_to() clamps silently to the servo's end stops.  For a single
    test that is friendly; for an arm it means a bad mapping quietly parks a
    joint in the wrong place instead of stopping.
  * Its move_to() also reads and rewrites a JSON cache on every call.  Three
    servos over a few hundred steps is thousands of SD card writes.

So this holds one hat open for the whole run, drives the channels directly,
range-checks every row before it moves anything, and writes the position
cache once at the end.  It still uses servo_control for the hat itself and
the calibration constants, so there is one definition of those.
"""

import argparse
import signal
import sys
import time

from servo_map import ServoMap, OutOfTravel, read_steps


class FakeHat:
    """Stands in for the hat so a run can be rehearsed without hardware."""

    def __init__(self):
        self.moves = 0
        self.last = {}

    def restart(self):
        pass

    def move_servo_position(self, channel, value):
        self.moves += 1
        self.last[channel] = value


def open_hat(dry_run):
    if dry_run:
        return FakeHat(), True
    try:
        import pi_servo_hat
    except ImportError:
        print("pi_servo_hat isn't installed - this needs to run on the Pi.\n"
              "Use --dry-run to rehearse anywhere else.", file=sys.stderr)
        raise SystemExit(2)
    hat = pi_servo_hat.PiServoHat()
    hat.restart()
    return hat, False


def relax_all(hat, channels):
    for channel in channels:
        try:
            hat.PCA9685.set_channel_word(channel, 1, 0)
        except Exception:
            pass


def send(hat, servos, angles):
    """Put the arm in one pose.  Every channel is worked out before any of
    them moves, so a bad angle stops the whole pose rather than leaving the
    arm half way to somewhere it shouldn't be."""
    commanded = servos.to_servo(angles)
    for channel, servo_angle in commanded.items():
        hat.move_servo_position(channel, servos.library_value(servo_angle))
    return commanded


def validate(servos, steps):
    """Check every row maps onto the servos before anything is powered."""
    problems = []
    for i, angles in enumerate(steps):
        for joint, angle in angles.items():
            if joint not in servos.joints:
                continue
            try:
                servos.servo_angle(joint, angle)
            except OutOfTravel as error:
                problems.append(f"row {i}: {error}")
    return problems


def dwell_for(previous, angles, servos, speed):
    """How long to wait for the servos to actually arrive.

    Paced off the joint that has furthest to go, because they all start
    together and the pose isn't reached until the slowest one lands.
    """
    if previous is None:
        return None
    travel = max((abs(angles[j] - previous[j]) * abs(servos.joints[j]["degrees_per_unit"])
                  for j in angles if j in servos.joints), default=0.0)
    return max(0.02, travel / max(servos.degrees_per_second * speed, 1e-6))


def play(hat, servos, steps, speed=1.0, approach=1.5, settle=0.0, quiet=False):
    previous = None
    for i, angles in enumerate(steps):
        send(hat, servos, angles)
        if previous is None:
            time.sleep(approach)          # the first move can be a long one
        else:
            time.sleep(dwell_for(previous, angles, servos, speed))
        previous = angles
        if not quiet and (i % 25 == 0 or i == len(steps) - 1):
            shown = "  ".join(f"{j} {a:7.2f}" for j, a in angles.items())
            print(f"  step {i:4d}/{len(steps) - 1}   {shown}")
    if settle:
        time.sleep(settle)
    return previous


def parse_pose(text):
    out = {}
    for part in text.split(","):
        joint, _, value = part.partition("=")
        out[joint.strip()] = float(value)
    return out


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("csv", nargs="?", help="a step CSV written by v6.py")
    parser.add_argument("--config", default="arm_config_v6.json")
    parser.add_argument("--goto", help="one pose, e.g. control_arm_a=90,main_arm=180")
    parser.add_argument("--check", action="store_true",
                        help="validate the file and the mapping, move nothing")
    parser.add_argument("--dry-run", action="store_true",
                        help="run the whole thing against a stand-in hat")
    parser.add_argument("--speed", type=float, default=1.0,
                        help="pace multiplier; below 1 waits longer per step")
    parser.add_argument("--approach", type=float, default=1.5,
                        help="seconds allowed for the first move (default 1.5)")
    parser.add_argument("--settle", type=float, default=0.5,
                        help="seconds to hold at the end before relaxing")
    parser.add_argument("--hold", action="store_true",
                        help="keep driving the servos after the run instead of relaxing")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    servos = ServoMap.from_file(args.config)
    problems = servos.check(verbose=not args.quiet)
    if problems:
        print("\nThe mapping itself is unsafe - fix the config first.", file=sys.stderr)
        return 2

    if args.goto:
        steps = [parse_pose(args.goto)]
    elif args.csv:
        steps = read_steps(args.csv)
    else:
        parser.error("give a step CSV, or --goto")

    print(f"\n{len(steps)} step(s), joints: {', '.join(steps[0])}")
    problems = validate(servos, steps)
    if problems:
        for problem in problems[:10]:
            print(f"  {problem}", file=sys.stderr)
        if len(problems) > 10:
            print(f"  ... and {len(problems) - 10} more", file=sys.stderr)
        print("\nNothing has been moved.", file=sys.stderr)
        return 2
    print("Every row fits the servo travel.")

    if args.check:
        return 0

    hat, fake = open_hat(args.dry_run)
    channels = sorted(servos.channels.values())

    def stop(signum, _frame):
        print(f"\nSignal {signum} - relaxing and stopping.")
        relax_all(hat, channels)
        sys.exit(0)
    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)

    print(f"{'Rehearsing' if fake else 'Running'} on channels "
          f"{', '.join(str(c) for c in channels)}...")
    started = time.time()
    try:
        final = play(hat, servos, steps, args.speed, args.approach, args.settle,
                     args.quiet)
    finally:
        if args.hold:
            print("Holding position - the servos are still driven.")
        else:
            relax_all(hat, channels)
            print("Relaxed.")

    print(f"Done in {time.time() - started:.1f}s"
          + (f", {hat.moves} channel writes" if fake else ""))

    # record where we left it, once, rather than on every step
    if final and not fake:
        try:
            import servo_control
            for joint, angle in final.items():
                if joint in servos.joints:
                    servo_control._record_position(servos.channel(joint),
                                                   servos.servo_angle(joint, angle))
        except Exception as error:
            print(f"(couldn't update the position cache: {error})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
