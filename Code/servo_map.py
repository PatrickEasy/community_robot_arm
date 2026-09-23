"""
Joint angles -> servo angles.

The bridge between v6.py, which thinks in joint angles, and servo_control.py,
which thinks in the servo's own 0-225 degree travel.  Nothing here touches
hardware or imports anything outside the standard library, so it runs and can
be tested on any machine - which is the point: get the mapping right on the
desk before the arm can hurt itself.

    from servo_map import ServoMap
    servos = ServoMap.from_file("arm_config_v6.json")
    servos.check()                                  # every joint limit fits?
    servos.to_servo({"base_rotation": -30.0, ...})  # -> {channel: angle}

The mapping is one line per joint:

    servo_angle = servo_at_zero + degrees_per_unit * joint_angle

and the whole job is picking those two numbers per joint, then proving that
every angle the joint is allowed to take lands inside the servo's safe travel.
"""

import json


class OutOfTravel(ValueError):
    """A joint angle maps to a servo angle the servo can't safely reach."""


class ServoMap:

    def __init__(self, config):
        servos = config["servos"]
        self.servo_min = float(servos.get("servo_min", 5.0))
        self.servo_max = float(servos.get("servo_max", 220.0))
        self.library_max = float(servos.get("library_max", 140.0))
        self.physical_max = float(servos.get("physical_max", 225.0))
        self.degrees_per_second = float(servos.get("degrees_per_second", 180.0))
        self.joints = {name: dict(spec) for name, spec in servos["joints"].items()}

        # the joint limits come from the arm's own config, so the two can't drift
        self.limits = {}
        spin = config.get("base", {}).get("rotation", {})
        self.limits[spin.get("name", "base_rotation")] = (
            float(spin.get("min_angle", -90.0)), float(spin.get("max_angle", 90.0)))
        for spec in config.get("chain", []):
            if spec.get("role") != "control":
                continue
            self.limits[spec["name"]] = (spec.get("min_angle"), spec.get("max_angle"))
        self.config = config

    @classmethod
    def from_file(cls, path="arm_config_v6.json"):
        try:
            handle = open(path)
        except FileNotFoundError:
            here = __file__.replace("\\", "/")
            beside = (here.rsplit("/", 1)[0] if "/" in here else ".") + "/" + path
            handle = open(beside)
        with handle as f:
            return cls(json.load(f))

    # -- the mapping --------------------------------------------------------

    def servo_angle(self, joint, angle):
        """One joint angle as a servo angle.  Raises if it's off the travel."""
        spec = self._spec(joint)
        value = spec["servo_at_zero"] + spec["degrees_per_unit"] * float(angle)
        if not self.servo_min - 1e-9 <= value <= self.servo_max + 1e-9:
            raise OutOfTravel(
                f"{joint} at {angle:.2f} maps to servo {value:.2f}, outside the "
                f"safe travel {self.servo_min:g} to {self.servo_max:g}. Recalibrate "
                f"servo_at_zero, or the arm will drive into its stop.")
        return value

    def joint_angle(self, joint, servo_angle):
        """The other way round - what joint angle a servo reading means."""
        spec = self._spec(joint)
        return (float(servo_angle) - spec["servo_at_zero"]) / spec["degrees_per_unit"]

    def to_servo(self, angles):
        """A whole pose: {joint: angle} -> {channel: servo angle}.

        Joints with no servo listed are ignored, so a config describing only
        some of the arm still works.
        """
        return {self._spec(joint)["channel"]: self.servo_angle(joint, angle)
                for joint, angle in angles.items() if joint in self.joints}

    def library_value(self, servo_angle):
        """The number servo_control.move_servo_position actually wants.
        Same arithmetic as servo_control.angle_to_library, repeated here so
        this module stays importable without the hardware library."""
        return float(servo_angle) * self.library_max / self.physical_max

    def channel(self, joint):
        return self._spec(joint)["channel"]

    @property
    def channels(self):
        return {name: spec["channel"] for name, spec in self.joints.items()}

    def _spec(self, joint):
        try:
            return self.joints[joint]
        except KeyError:
            raise KeyError(f"No servo listed for {joint!r}. Have: "
                           f"{', '.join(self.joints)}") from None

    # -- proving it before you power anything -------------------------------

    def joint_travel(self, joint):
        """(min, max) the joint can be driven to.

        A limit that tracks another arm - main_arm's ceiling follows
        control_arm_a - is resolved to its widest, because the servo has to
        cover every pose the arm is allowed to take, not just today's.
        """
        low, high = self.limits.get(joint, (None, None))
        return self._widest(low, "min"), self._widest(high, "max")

    def _widest(self, limit, end):
        if limit is None:
            return None
        if isinstance(limit, dict):
            other_low, other_high = self.joint_travel(limit["relative_to"])
            reference = other_low if end == "min" else other_high
            if reference is None:
                return None
            return reference + float(limit.get("offset", 0.0))
        return float(limit)

    def check(self, verbose=True):
        """Does every angle each joint is allowed to take fit on its servo?

        Run this once after calibrating and before the linkage is attached.
        Returns a list of problems - empty means the mapping is safe.
        """
        problems = []
        lines = []
        for joint in self.joints:
            low, high = self.joint_travel(joint)
            if low is None or high is None:
                problems.append(f"{joint}: no joint limits in the config, so its "
                                f"servo travel can't be checked")
                continue
            ends = []
            for name, angle in (("min", low), ("max", high)):
                try:
                    ends.append(self.servo_angle(joint, angle))
                except OutOfTravel as error:
                    problems.append(str(error))
                    ends.append(None)
            if None in ends:
                continue
            span = abs(ends[1] - ends[0])
            head = min(ends) - self.servo_min
            tail = self.servo_max - max(ends)
            lines.append(f"  {joint:15s} joint {low:7.1f} to {high:7.1f}  ->  "
                         f"servo {min(ends):6.1f} to {max(ends):6.1f}  "
                         f"({span:5.1f} deg used, {head:.1f} spare below, "
                         f"{tail:.1f} above)")
        if verbose:
            print(f"Servo travel {self.servo_min:g} to {self.servo_max:g} degrees")
            print("\n".join(lines))
            for problem in problems:
                print(f"  PROBLEM  {problem}")
        return problems

    def solve_offset(self, joint, joint_angle, measured_servo_angle):
        """Work out servo_at_zero from one known pose.

        Put the arm somewhere you can measure - a hard stop, a printed
        protractor, the arm dead level - note the joint angle that pose means
        in v6, and the servo angle it took to get there.  This gives the
        number to paste into the config.
        """
        spec = self._spec(joint)
        return float(measured_servo_angle) - spec["degrees_per_unit"] * float(joint_angle)

    def solve_scale(self, joint, pose_a, pose_b):
        """Work out degrees_per_unit from two poses, each (joint_angle,
        measured_servo_angle).  Tells you the direction and any gearing at
        the same time - and if it doesn't come out near +/-1 or your gear
        ratio, something is wrong with the assumed geometry."""
        (joint_a, servo_a), (joint_b, servo_b) = pose_a, pose_b
        if abs(joint_b - joint_a) < 1e-9:
            raise ValueError("Use two different joint angles")
        return (float(servo_b) - float(servo_a)) / (float(joint_b) - float(joint_a))

    def describe(self):
        return "\n".join(
            f"{joint:15s} channel {spec['channel']}  "
            f"servo = {spec['servo_at_zero']:+.2f} "
            f"{spec['degrees_per_unit']:+.4f} * joint"
            for joint, spec in self.joints.items())


# -- reading a step CSV, without needing numpy ------------------------------

def read_steps(path):
    """Read a v6 step CSV into a list of {joint: angle} dicts.

    Only the joint columns are kept - the turn_ and position columns are
    along for the ride.  Rows come back in file order, which is the order
    they should be played.
    """
    with open(path) as handle:
        rows = handle.read().strip().split("\n")
    header = rows[0].split(",")
    # save_movement writes: step, one column per joint, then turn_<joint> for
    # each, then the node position and maybe leg.  So the joints are exactly
    # the columns between step and the first turn_ - which beats guessing by
    # name, and survives a joint that happens to be called something_z.
    first_turn = next((i for i, name in enumerate(header)
                       if name.startswith("turn_")), len(header))
    joints = [name for name in header[:first_turn] if name != "step"]
    index = {name: header.index(name) for name in joints}
    out = []
    for row in rows[1:]:
        if not row.strip():
            continue
        cells = row.split(",")
        out.append({name: float(cells[i]) for name, i in index.items()})
    return out


if __name__ == "__main__":
    servos = ServoMap.from_file("arm_config_v6.json")
    print(servos.describe())
    print()
    problems = servos.check()
    print()
    print("Mapping is safe." if not problems
          else f"{len(problems)} problem(s) - fix these before powering the arm.")
