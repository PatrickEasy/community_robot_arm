"""
Community robot arm - v6

A cut-down arm with a third degree of freedom.  Two changes from v5:

  * The linkage is an open chain, not a closed loop.  control_arm_a is driven
    directly, and main_arm is driven too - its angle arrives through the
    parallel linkage from control_arm_b, so from the simulation's point of
    view it is simply another input.  Nothing has to be solved backwards, so
    there are no circle intersections and no elbow branches.

  * The base pivots.  The arms stay in one vertical plane; that plane turns
    about the base axis.

That second point is what keeps this simple.  The linkage is solved in its
own plane using (r, z) - r measured outward from the base axis, z upward -
and the base angle only rotates the finished result into (x, y, z).  So the
kinematics stay two-dimensional and exact, and the volume the arm can reach
is just its flat cross-section revolved about the axis.

Plane angles run 0 degrees along +r and 90 degrees along +z, which makes the
tool arm's fixed -90 degrees mean "straight down, parallel with the base
axis", exactly as the real one is held.

The base rotation is treated as another control arm, so robot.angles, the
step CSVs, planning and animation all carry it without special cases.

    source .venv/bin/activate
    python v6.py
"""

import json
import math

import numpy as np
import matplotlib.pyplot as plt


TOL = 1e-9
EPS = 1e-7


# ---------------------------------------------------------------- results --

class Result:
    """What came back from a calculation.  Truthy when it worked, carrying the
    reason when it didn't, so callers branch on it instead of catching."""

    def __init__(self, ok, reason="", **data):
        self.ok = bool(ok)
        self.reason = reason
        self.data = data

    def __bool__(self):
        return self.ok

    def __getitem__(self, key):
        return self.data[key]

    def get(self, key, default=None):
        return self.data.get(key, default)

    def __repr__(self):
        return f"<Result ok>" if self.ok else f"<Result failed: {self.reason}>"


# ------------------------------------------------------------------ nodes --

class Node:
    """A point on the arm.

    It is stored where it is actually worked out - in the arm's own plane, as
    (r, z) - and reports its world position by turning that through the base
    angle.  So there is one source of truth, and no pair of coordinates that
    can drift apart.
    """

    def __init__(self, name, robot, r=0.0, z=0.0):
        self.name = name
        self.robot = robot
        self.r = float(r)
        self.z = float(z)

    @property
    def planar(self):
        """(r, z) in the arm's own plane."""
        return (self.r, self.z)

    @planar.setter
    def planar(self, rz):
        self.r, self.z = float(rz[0]), float(rz[1])

    @property
    def x(self):
        return self.r * math.cos(math.radians(self.robot.base_angle))

    @property
    def y(self):
        return self.r * math.sin(math.radians(self.robot.base_angle))

    @property
    def position(self):
        """(x, y, z) in the world."""
        return (self.x, self.y, self.z)

    def __repr__(self):
        x, y, z = self.position
        return (f"<Node {self.name} world ({x:.2f}, {y:.2f}, {z:.2f}) "
                f"plane (r {self.r:.2f}, z {self.z:.2f})>")


# ---------------------------------------------------------------- vectors --

class Vector:
    """A rigid bar between two nodes, lying in the arm's plane.

    role 'control' - you drive the angle
    role 'fixed'   - the angle never changes; the tool arm is one of these,
                     pinned at -90 so it hangs parallel with the base axis

    An angle is measured in the plane unless angle_relative_to names another
    bar, in which case it is measured from that bar.  A parallelogram linkage
    gives you the first kind, which is why main_arm is left absolute.
    """

    def __init__(self, name, start, end, length, angle=0.0, role="control",
                 min_angle=None, max_angle=None, angle_relative_to=None, robot=None):
        self.name = name
        self.start = start
        self.end = end
        self.length = float(length)
        self.role = role
        self.robot = robot
        self._angle = float(angle)
        self.min_angle = min_angle
        self.max_angle = max_angle
        self.angle_relative_to = angle_relative_to

    @property
    def is_control(self):
        return self.role == "control"

    @property
    def angle(self):
        """The driven value - what you set, and what the limits apply to."""
        return self._angle

    def set_angle(self, value):
        if self.role == "fixed":
            raise TypeError(f"{self.name} is fixed - its angle can't be driven")
        self._angle = float(value)

    @property
    def plane_angle(self):
        """The bar's actual angle in the plane, once any relative reference
        has been added in."""
        if self.angle_relative_to is None:
            return self._angle
        return self._angle + self.robot.vector(self.angle_relative_to).plane_angle

    # -- limits -------------------------------------------------------------

    def _resolve(self, limit):
        if limit is None:
            return None
        if isinstance(limit, dict):
            other = self.robot.vector(limit["relative_to"])
            return other.angle + float(limit.get("offset", 0.0))
        return float(limit)

    @property
    def limits(self):
        return self._resolve(self.min_angle), self._resolve(self.max_angle)

    def in_limits(self, angle=None):
        angle = self.angle if angle is None else angle
        lo, hi = self.limits
        if lo is not None and angle < lo - TOL:
            return False
        if hi is not None and angle > hi + TOL:
            return False
        return True

    def clamp(self, angle):
        lo, hi = self.limits
        if lo is not None:
            angle = max(angle, lo)
        if hi is not None:
            angle = min(angle, hi)
        return angle

    def limit_text(self):
        lo, hi = self.limits
        lo = "-inf" if lo is None else f"{lo:.1f}"
        hi = "+inf" if hi is None else f"{hi:.1f}"
        return f"{lo} to {hi}"

    # -- geometry -----------------------------------------------------------

    def body(self, samples=12):
        """Points spread along the bar in the plane - used for swept area."""
        r0, z0 = self.start.planar
        r1, z1 = self.end.planar
        return [(r0 + (r1 - r0) * i / samples, z0 + (z1 - z0) * i / samples)
                for i in range(samples + 1)]

    def __repr__(self):
        return (f"<Vector {self.name} ({self.role}) {self.start.name}->{self.end.name} "
                f"len {self.length:g} angle {self.angle:.1f}>")


class BaseRotation:
    """The pivot at the bottom.

    Deliberately shaped like a Vector - same angle, limits, clamp and name -
    so the robot can hold it in the same list of controls as the arms and
    everything downstream works without knowing it is different.
    """

    def __init__(self, name="base_rotation", angle=0.0, min_angle=-90.0,
                 max_angle=90.0, robot=None):
        self.name = name
        self.role = "control"
        self.robot = robot
        self._angle = float(angle)
        self.min_angle = min_angle
        self.max_angle = max_angle

    is_control = True

    @property
    def angle(self):
        return self._angle

    def set_angle(self, value):
        self._angle = float(value)

    @property
    def limits(self):
        lo = None if self.min_angle is None else float(self.min_angle)
        hi = None if self.max_angle is None else float(self.max_angle)
        return lo, hi

    in_limits = Vector.in_limits
    clamp = Vector.clamp
    limit_text = Vector.limit_text

    def _resolve(self, limit):
        return None if limit is None else float(limit)

    def __repr__(self):
        return f"<BaseRotation {self.angle:.1f} limits {self.limit_text()}>"


# ------------------------------------------------------------------ areas --

class Region:
    """An area in the arm's plane, held as an occupancy grid.

    Because the base rotation only turns the plane, the cross-section IS the
    whole story: the volume the arm can reach is this shape revolved about
    the base axis.  revolve() turns it into surfaces for the 3D view.
    """

    def __init__(self, points, cell=0.2, close=0.6, name="region", color="tab:blue"):
        pts = np.asarray(list(points), dtype=float)
        if pts.size == 0:
            raise ValueError("No points to build a region from")

        self.cell = float(cell)
        self.name = name
        self.color = color
        self._polys = None

        pad = max(3, int(round(close / cell)) + 2)
        self.r0 = pts[:, 0].min() - pad * cell
        self.z0 = pts[:, 1].min() - pad * cell
        nr = int(math.ceil((pts[:, 0].max() - self.r0) / cell)) + pad + 1
        nz = int(math.ceil((pts[:, 1].max() - self.z0) / cell)) + pad + 1

        grid = np.zeros((nz, nr), dtype=bool)
        grid[np.rint((pts[:, 1] - self.z0) / cell).astype(int),
             np.rint((pts[:, 0] - self.r0) / cell).astype(int)] = True

        r = int(round(close / cell))
        if r > 0:
            grid = _erode(_dilate(grid, r), r)   # the array grows by r a side
            self.r0 -= r * cell                  # so the origin moves with it
            self.z0 -= r * cell
        self.grid = grid

    @classmethod
    def _from_grid(cls, grid, r0, z0, cell, name, color, pad=3):
        obj = cls.__new__(cls)
        obj.grid = np.pad(grid, pad, constant_values=False)
        obj.r0 = r0 - pad * cell
        obj.z0 = z0 - pad * cell
        obj.cell = cell
        obj.name = name
        obj.color = color
        obj._polys = None
        return obj

    def grow(self, clearance, name=None, color=None):
        """A copy padded outwards by clearance - the safe perimeter."""
        r = int(round(clearance / self.cell))
        return Region._from_grid(
            _dilate(self.grid, r), self.r0 - r * self.cell, self.z0 - r * self.cell,
            self.cell, name if name else f"{self.name} + {clearance:g} clearance",
            color if color else self.color)

    @property
    def area(self):
        """Cross-section area, in square units."""
        return float(self.grid.sum()) * self.cell ** 2

    @property
    def bounds(self):
        zs, rs = np.nonzero(self.grid)
        return (float(self.r0 + rs.min() * self.cell), float(self.z0 + zs.min() * self.cell),
                float(self.r0 + rs.max() * self.cell), float(self.z0 + zs.max() * self.cell))

    def contains(self, r, z):
        ir = int(round((r - self.r0) / self.cell))
        iz = int(round((z - self.z0) / self.cell))
        if not (0 <= ir < self.grid.shape[1] and 0 <= iz < self.grid.shape[0]):
            return False
        return bool(self.grid[iz, ir])

    def mask_at(self, radii, z):
        """Which of these r values are inside the region at height z.
        The array version of contains(), for slicing the volume quickly."""
        radii = np.asarray(radii, dtype=float)
        iz = int(round((z - self.z0) / self.cell))
        out = np.zeros(radii.shape, dtype=bool)
        if not (0 <= iz < self.grid.shape[0]):
            return out
        ir = np.rint((radii - self.r0) / self.cell).astype(int)
        on_grid = (ir >= 0) & (ir < self.grid.shape[1])
        out[on_grid] = self.grid[iz, ir[on_grid]]
        return out

    def inner_box(self, margin=0.0):
        """The largest rectangle lying wholly inside the region, as
        (min_r, min_z, max_r, max_z).  bounds gives the box around a blob,
        including corners that are out of reach; this gives the opposite."""
        grid = self.grid
        heights = np.zeros(grid.shape[1], dtype=int)
        best = (0, 0, 0, 0, 0)
        for row in range(grid.shape[0]):
            heights = np.where(grid[row], heights + 1, 0)
            stack = []
            for col in range(len(heights) + 1):
                height = int(heights[col]) if col < len(heights) else 0
                start = col
                while stack and stack[-1][1] >= height:
                    start, tall = stack.pop()
                    area = tall * (col - start)
                    if area > best[0]:
                        best = (area, row, start, col - 1, tall)
                stack.append((start, height))
        if best[0] == 0:
            return self.bounds
        _, row, left, right, tall = best
        return (self.r0 + left * self.cell + margin,
                self.z0 + (row - tall + 1) * self.cell + margin,
                self.r0 + right * self.cell - margin,
                self.z0 + row * self.cell - margin)

    def polygons(self):
        """The cross-section outline as closed (r, z) loops."""
        if self._polys is None:
            rs = self.r0 + np.arange(self.grid.shape[1]) * self.cell
            zs = self.z0 + np.arange(self.grid.shape[0]) * self.cell
            fig = plt.figure()
            try:
                cs = fig.add_subplot(111).contour(rs, zs, self.grid.astype(float),
                                                  levels=[0.5])
                self._polys = [np.asarray(s) for s in cs.allsegs[0] if len(s) > 2]
            finally:
                plt.close(fig)
        return self._polys

    def volume(self, sweep_degrees):
        """How much space the revolved shape occupies.

        Pappus's theorem: the volume of a solid of revolution is the area
        times the distance its centroid travels.  Cells at negative r are
        counted at |r|, since they sweep out real space too.
        """
        zs, rs = np.nonzero(self.grid)
        if not len(rs):
            return 0.0
        radii = np.abs(self.r0 + rs * self.cell)
        return float(radii.sum()) * self.cell ** 2 * math.radians(abs(sweep_degrees))

    def revolve(self, beta_min, beta_max, steps=25, stride=1):
        """Surfaces for the 3D view: the outline swept about the base axis.
        Returns a list of (X, Y, Z) meshes, one per outline loop."""
        betas = np.radians(np.linspace(beta_min, beta_max, steps))
        out = []
        for poly in self.polygons():
            loop = poly[::stride]
            R, B = np.meshgrid(loop[:, 0], betas)
            Z = np.tile(loop[:, 1], (len(betas), 1))
            out.append((R * np.cos(B), R * np.sin(B), Z))
        return out

    def save_perimeter(self, path):
        """Write the cross-section outline to CSV (loop, r, z)."""
        with open(path, "w") as handle:
            handle.write("loop,r,z\n")
            for i, poly in enumerate(self.polygons()):
                for (r, z) in poly:
                    handle.write(f"{i},{r:.4f},{z:.4f}\n")
        return path

    def plot(self, ax=None, fill=True, alpha=0.18, linewidth=1.6, label=None, zorder=0):
        """Draw the cross-section on 2D axes."""
        ax = ax if ax else plt.gca()
        label = self.name if label is None else label
        rs = self.r0 + np.arange(self.grid.shape[1]) * self.cell
        zs = self.z0 + np.arange(self.grid.shape[0]) * self.cell
        if fill:
            ax.contourf(rs, zs, self.grid.astype(float), levels=[0.5, 1.5],
                        colors=[self.color], alpha=alpha, zorder=zorder)
        first = True
        for poly in self.polygons():
            ax.plot(poly[:, 0], poly[:, 1], color=self.color, linewidth=linewidth,
                    zorder=zorder + 0.1, label=label if first else None)
            first = False
        return ax

    def __repr__(self):
        b = self.bounds
        return (f"<Region {self.name!r} cross-section area={self.area:.1f} "
                f"r {b[0]:.1f}..{b[2]:.1f}  z {b[1]:.1f}..{b[3]:.1f}>")


def _disc_offsets(r_cells):
    return [(dz, dr)
            for dz in range(-r_cells, r_cells + 1)
            for dr in range(-r_cells, r_cells + 1)
            if dr * dr + dz * dz <= r_cells * r_cells]


def _spread(grid, r_cells, fill):
    padded = np.pad(grid, r_cells, constant_values=fill)
    out = np.zeros_like(padded)
    for dz, dr in _disc_offsets(r_cells):
        out |= np.roll(np.roll(padded, dz, axis=0), dr, axis=1)
    return out


def _dilate(grid, r_cells):
    return grid.copy() if r_cells <= 0 else _spread(grid, r_cells, False)


def _erode(grid, r_cells):
    if r_cells <= 0:
        return grid.copy()
    return ~_spread(~grid, r_cells, True)[r_cells:-r_cells, r_cells:-r_cells]


# ------------------------------------------------------------------ robot --

class Robot:
    """The arm as a whole.

    Solving is a walk along an open chain, so there is nothing to fail except
    a limit - but the Result convention from v5 is kept so calling code reads
    the same way.

        robot.move("control_arm_a", 120)         drive one arm
        robot.move("base_rotation", -30)         swing the whole plane round
        robot.move_to("tool_tip", 8, -4, 0)      put a node on a point in space
        robot.position("tool_tip")               (x, y, z)
        robot.positions()                        every node at once
        robot.reachable_area()                   the cross-section it can reach
        robot.reachable_volume()                 that, revolved
    """

    def __init__(self, config):
        self.config = config
        self.name = config.get("name", "robot")
        self.view = config.get("view", {})
        self.tool = config.get("tool", {})
        self.area_settings = config.get("areas", {})
        self.last_result = Result(True)

        base = config.get("base", {})
        self.base_position = tuple(base.get("position", [0.0, 0.0, 0.0]))
        spin = base.get("rotation", {})
        self.base_rotation = BaseRotation(
            name=spin.get("name", "base_rotation"),
            angle=spin.get("angle", 0.0),
            min_angle=spin.get("min_angle", -90.0),
            max_angle=spin.get("max_angle", 90.0),
            robot=self)

        # nodes are named by the chain, so there is no separate list to keep
        # in step with it
        self.nodes = {}
        self.vectors = {}
        self.chain = []
        first = config["chain"][0]["start"]
        self.base_node = self._node(first)
        self.base_node.planar = (math.hypot(self.base_position[0], self.base_position[1]),
                                 self.base_position[2])

        for spec in config["chain"]:
            vector = Vector(
                spec["name"], self._node(spec["start"]), self._node(spec["end"]),
                length=spec["length"], angle=spec.get("angle", 0.0),
                role=spec.get("role", "control"),
                min_angle=spec.get("min_angle"), max_angle=spec.get("max_angle"),
                angle_relative_to=spec.get("angle_relative_to"), robot=self)
            self.vectors[vector.name] = vector
            self.chain.append(vector)

        # the base pivot sits alongside the arms, so everything downstream -
        # angles, planning, the CSVs, animation - carries it for free
        self.controls = [self.base_rotation] + [v for v in self.chain if v.is_control]
        self.planar_controls = [v for v in self.chain if v.is_control]
        if not self.planar_controls:
            raise ValueError("No control arms in the chain - nothing to drive")

        self._fixed_by_end = {v.end.name: v for v in self.chain if v.role == "fixed"}
        self._driven_nodes = [v.end.name for v in self.planar_controls]
        self.solve()

    def _node(self, name):
        if name not in self.nodes:
            self.nodes[name] = Node(name, self)
        return self.nodes[name]

    @classmethod
    def from_file(cls, path):
        """Load a config, falling back to looking beside this script."""
        try:
            handle = open(path)
        except FileNotFoundError:
            here = __file__.replace("\\", "/")
            beside = (here.rsplit("/", 1)[0] if "/" in here else ".") + "/" + path
            try:
                handle = open(beside)
            except FileNotFoundError:
                raise FileNotFoundError(
                    f"Couldn't find {path!r} here or beside the script") from None
        with handle as f:
            return cls(json.load(f))

    # -- handles ------------------------------------------------------------

    def node(self, name):
        try:
            return self.nodes[name]
        except KeyError:
            raise KeyError(f"No node called {name!r}. Have: {', '.join(self.nodes)}")

    def vector(self, name):
        if name == self.base_rotation.name:
            return self.base_rotation
        try:
            return self.vectors[name]
        except KeyError:
            raise KeyError(f"No arm called {name!r}. Have: "
                           f"{self.base_rotation.name}, {', '.join(self.vectors)}")

    @property
    def base_angle(self):
        """Where the plane of the arm is pointing, in degrees."""
        return self.base_rotation.angle

    def position(self, node_name):
        """(x, y, z) of one node."""
        return self.node(node_name).position

    def planar(self, node_name):
        """(r, z) of one node, in the arm's own plane."""
        return self.node(node_name).planar

    def positions(self):
        """Every node's world position, in chain order."""
        return {name: node.position for name, node in self.nodes.items()}

    @property
    def control_names(self):
        return [c.name for c in self.controls]

    @property
    def angles(self):
        """Every driven angle - the arm's whole state, base pivot included."""
        return {c.name: c.angle for c in self.controls}

    def _apply(self, angles):
        for control in self.controls:
            if control.name in angles:
                control.set_angle(angles[control.name])

    # -- limits -------------------------------------------------------------

    def breaches(self, angles=None):
        keep = self.angles
        if angles is not None:
            self._apply(angles)
        out = [f"{c.name} at {c.angle:.1f} (allowed {c.limit_text()})"
               for c in self.controls if not c.in_limits()]
        if angles is not None:
            self._apply(keep)
        return out

    def within_limits(self, angles=None):
        return not self.breaches(angles)

    def clamp(self, angles):
        keep = self.angles
        out = {}
        for control in self.controls:
            control.set_angle(control.clamp(angles.get(control.name, control.angle)))
            out[control.name] = control.angle
        self._apply(keep)
        return out

    # -- the solve ----------------------------------------------------------

    def solve(self, angles=None):
        """Walk the chain and put every node where it belongs.

        An open chain, so each bar simply hangs off the one before it.  The
        base angle isn't used here at all - it only turns the finished plane
        into the world, which the nodes do themselves.
        """
        if angles is not None:
            self._apply(angles)

        self.base_node.planar = (math.hypot(self.base_position[0], self.base_position[1]),
                                 self.base_position[2])
        for vector in self.chain:
            r0, z0 = vector.start.planar
            theta = math.radians(vector.plane_angle)
            vector.end.planar = (r0 + vector.length * math.cos(theta),
                                 z0 + vector.length * math.sin(theta))
        self.last_result = Result(True)
        return self.last_result

    def _fail(self, reason):
        self.last_result = Result(False, reason)
        return self.last_result

    # -- driving it ---------------------------------------------------------

    def move(self, control_name, angle, enforce_limits=True):
        """Drive one arm - or the base pivot - to an angle and re-solve."""
        return self.move_all({control_name: angle}, enforce_limits)

    def move_all(self, angles, enforce_limits=True):
        """Drive several at once.  Nothing changes if the move is refused."""
        keep = self.angles
        wanted = dict(keep)
        for name, angle in angles.items():
            arm = self.vector(name)    # raises on a typo, which is what you want
            if not arm.is_control:
                raise TypeError(f"{name} is fixed - it can't be driven. "
                                f"Drivable: {', '.join(self.control_names)}")
            wanted[name] = float(angle)

        if enforce_limits and not self.within_limits(wanted):
            reason = "outside the limits: " + "; ".join(self.breaches(wanted))
            self.solve(keep)
            return self._fail(reason)
        return self.solve(wanted)

    def nudge(self, control_name, degrees):
        return self.move(control_name, self.vector(control_name).angle + degrees)

    # -- inverse kinematics -------------------------------------------------

    def move_to(self, node_name, x, y, z, apply=True):
        """Put a node on a point in space.

        Closed form, in two halves, which is the payoff for keeping the base
        rotation separate from the linkage:

          1. the base angle follows straight from the target's bearing, since
             turning the base is the only thing that moves a node off the
             plane;
          2. what is left is a flat two-link reach in (r, z), which is the
             textbook elbow-up / elbow-down pair.

        Every combination that comes out is checked against the limits, and
        the one nearest the current pose wins, so a path stays continuous.
        """
        self.node(node_name)
        keep = self.angles
        candidates = []

        for base_angle, radius in self._base_options(x, y, keep[self.base_rotation.name]):
            driven_node, target = self._back_to_driven(node_name, radius, z)
            if driven_node is None:
                continue
            for arm_angles in self._planar_solutions(driven_node, target, keep):
                angles = dict(keep)
                angles[self.base_rotation.name] = base_angle
                angles.update(arm_angles)
                angles = self._normalise(angles, keep)
                if self.within_limits(angles):
                    candidates.append(angles)

        if not candidates:
            self.solve(keep)
            return self._fail(
                f"can't put {node_name} on ({x:.2f}, {y:.2f}, {z:.2f}) - out of "
                f"reach, or only reachable by breaking a limit")

        best = min(candidates,
                   key=lambda a: sum(abs(a[n] - keep[n]) for n in keep))
        self.solve(best if apply else keep)
        self.last_result = Result(True, angles=best)
        return self.last_result

    def _base_options(self, x, y, current):
        """How the base could be turned to bring (x, y) into the arm's plane.

        Two ways: face the point, or face the opposite way and reach backwards
        over the base with a negative radius.  Both are real, and the second
        is what lets the arm cover ground the +/-90 limit would otherwise cut
        off.  Anything outside the base limits is dropped here.
        """
        distance = math.hypot(x, y)
        if distance < EPS:
            return [(current, 0.0)]          # on the axis - any bearing will do

        bearing = math.degrees(math.atan2(y, x))
        out = []
        for angle, radius in ((bearing, distance), (bearing - 180.0, -distance),
                              (bearing + 180.0, -distance)):
            angle = _nearest_equivalent(angle, current, self.base_rotation)
            if self.base_rotation.in_limits(angle) and \
                    not any(abs(angle - a) < TOL for a, _ in out):
                out.append((angle, radius))
        return out

    def _back_to_driven(self, node_name, r, z):
        """Step back from a node through any fixed bars, to the last node the
        control arms actually place.  Returns (node name, (r, z))."""
        node = node_name
        seen = set()
        while node in self._fixed_by_end:
            vector = self._fixed_by_end[node]
            if vector.angle_relative_to is not None:
                return None, None      # can't be undone without knowing the pose
            if node in seen:
                return None, None
            seen.add(node)
            theta = math.radians(vector.plane_angle)
            r -= vector.length * math.cos(theta)
            z -= vector.length * math.sin(theta)
            node = vector.start.name
        return node, (r, z)

    def _planar_solutions(self, node_name, target, prefer):
        """Every set of arm angles putting that node on that (r, z) point."""
        if node_name not in self._driven_nodes:
            return []
        depth = self._driven_nodes.index(node_name) + 1
        arms = self.planar_controls[:depth]
        base_r, base_z = self.base_node.planar
        target_r, target_z = target
        reach = math.hypot(target_r - base_r, target_z - base_z)

        if depth == 1:                                  # one bar: a bearing
            arm = arms[0]
            if abs(reach - arm.length) > 1e-6 * max(1.0, arm.length):
                return []
            angle = math.degrees(math.atan2(target_z - base_z, target_r - base_r))
            return [{arm.name: self._driven_value(arm, angle, prefer, {arm.name: angle})}]

        if depth != 2:
            return []                                   # v6 is a two-bar arm

        first, second = arms
        l1, l2 = first.length, second.length
        slack = EPS * max(1.0, l1 + l2)
        if reach > l1 + l2 + slack or reach < abs(l1 - l2) - slack:
            return []

        if reach < EPS:
            # Folded right back: the far node has landed on the base itself.
            # Not "no solutions" but infinitely many - the arm can point
            # anywhere as long as the second bar doubles back along the first.
            # Offer the ones nearest the current pose and let the scoring in
            # move_to choose; the limits throw out the rest.
            out = []
            for a in (prefer.get(first.name, 0.0),
                      prefer.get(second.name, 0.0) - 180.0,
                      prefer.get(second.name, 0.0) + 180.0):
                plane = {first.name: a, second.name: a + 180.0}
                out.append({first.name: self._driven_value(first, a, prefer, plane),
                            second.name: self._driven_value(second, a + 180.0,
                                                            prefer, plane)})
            return out

        bearing = math.atan2(target_z - base_z, target_r - base_r)
        spread = math.acos(max(-1.0, min(1.0,
                               (l1 ** 2 + reach ** 2 - l2 ** 2) / (2 * l1 * reach))))

        out = []
        for side in (1.0, -1.0):                        # elbow up, elbow down
            a = math.degrees(bearing + side * spread)
            elbow_r = base_r + l1 * math.cos(math.radians(a))
            elbow_z = base_z + l1 * math.sin(math.radians(a))
            m = math.degrees(math.atan2(target_z - elbow_z, target_r - elbow_r))
            # pass the new plane angles along, so a relative arm is measured
            # against where its reference is going, not where it is now
            plane = {first.name: a, second.name: m}
            out.append({first.name: self._driven_value(first, a, prefer, plane),
                        second.name: self._driven_value(second, m, prefer, plane)})
        return out

    def _driven_value(self, vector, plane_angle, prefer, plane=None):
        """Turn an angle in the plane back into the value you'd drive, undoing
        any relative reference.  Whole-turn shifts are left to _normalise."""
        reference = vector.angle_relative_to
        if reference is not None:
            plane = plane or {}
            plane_angle -= plane[reference] if reference in plane \
                else self.vector(reference).plane_angle
        return plane_angle

    def _normalise(self, angles, prefer):
        """Pick each arm's whole-turn representation: the one that sits inside
        its limits and nearest the current pose.

        Adding 360 to an angle doesn't move the arm, it only renames where it
        is - so this can never change the geometry, and it is what lets a
        limit range like 175 to 265 accept an angle atan2 reported as -95.

        Done in control order and applied as it goes, so a limit that tracks
        another arm is read with that arm already in its new place.  Checking
        against the current pose instead would wrongly throw out perfectly
        good answers whenever the reference arm is also moving.
        """
        keep = self.angles
        out = {}
        for control in self.controls:
            value = angles.get(control.name, control.angle)
            control.set_angle(value)
            wanted = prefer.get(control.name, value)
            for option in sorted((value + 360.0 * turns for turns in (0, -1, 1, -2, 2)),
                                 key=lambda v: abs(v - wanted)):
                if control.in_limits(option):
                    value = option
                    break
            control.set_angle(value)
            out[control.name] = value
        self._apply(keep)
        return out

    # -- planning -----------------------------------------------------------

    def plan_to(self, node_name, x, y, z, steps=30, straight=True, verbose=False):
        """A movement taking a node from where it is to a point in space."""
        return self.plan_path(node_name, [(x, y, z)], steps, straight, verbose)

    def plan_path(self, node_name, points, steps=30, straight=True, verbose=False):
        """A movement taking a node through a series of (x, y, z) points.

        straight=True walks the node along the straight line between points in
        space, solving each step.  straight=False interpolates the angles
        instead - easier on the motors, but the node's path bows.

        A movement is a list of angle dicts, one per step.  Nothing is
        printed; robot.last_result records anything skipped.
        """
        keep = self.angles
        frames = []
        skipped, missed = [], []
        here = np.array(self.position(node_name), dtype=float)

        for point in points:
            target = np.array(point, dtype=float)
            if straight:
                landed = None
                for i in range(1, steps + 1):
                    step_to = here + (target - here) * (i / steps)
                    result = self.move_to(node_name, *step_to)
                    if not result:
                        skipped.append(tuple(round(float(v), 6) for v in step_to))
                        if verbose:
                            print(f"  skipped {tuple(round(float(v), 2) for v in step_to)}"
                                  f" - {result.reason}")
                        continue
                    frames.append(result["angles"])
                    landed = step_to
                if landed is None:
                    missed.append(tuple(float(v) for v in target))
                else:
                    here = landed
            else:
                result = self.move_to(node_name, *target)
                if not result:
                    missed.append(tuple(float(v) for v in target))
                    continue
                to_angles = result["angles"]
                from_angles = frames[-1] if frames else keep
                for i in range(1, steps + 1):
                    t = i / steps
                    frames.append({n: from_angles[n] + (to_angles[n] - from_angles[n]) * t
                                   for n in to_angles})
                here = target

        self.solve(keep)
        if frames:
            self.last_result = Result(True, skipped=skipped, missed=missed)
        else:
            self.last_result = Result(False, f"no part of the path to {points} is "
                                             f"reachable", skipped=skipped, missed=missed)
        return frames

    # -- movements out to file ----------------------------------------------

    def steps_to(self, node_name, x, y, z, path="steps.csv", steps=30, straight=True):
        """The angle of every arm at every step of a move to one point.

        Returns the movement and writes the same thing to CSV.  Nothing is
        printed; check robot.last_result if a move came back short.
        """
        movement = self.plan_path(node_name, [(x, y, z)], steps, straight)
        self.save_movement(movement, path, node_name)
        return movement

    def steps_through(self, node_name, points, path="steps.csv", steps=30, straight=True):
        """The same, for a node travelling through several points in order."""
        movement = self.plan_path(node_name, points, steps, straight)
        self.save_movement(movement, path, node_name, targets=points)
        return movement

    def save_movement(self, movement, path="steps.csv", node_name=None, targets=None):
        """Write a movement to CSV: one row per step, a column per arm.

        The step number, each arm's angle (base pivot included), how far that
        arm turns to get there from the step before, and where the node ends
        up in x, y, z.
        """
        node_name = node_name or self.tool.get("node")
        names = self.control_names
        keep = self.angles
        verdict = self.last_result     # writing a file shouldn't erase why a plan fell short

        header = ["step"] + names + [f"turn_{n}" for n in names]
        if node_name:
            header += [f"{node_name}_{axis}" for axis in "xyz"]
        if targets:
            header += ["leg"]
        legs = self._leg_numbers(movement, node_name, targets) if targets else None

        rows = []
        previous = None
        for i, frame in enumerate(movement):
            angles = [float(frame[n]) for n in names]
            turns = [0.0] * len(names) if previous is None else \
                [a - p for a, p in zip(angles, previous)]
            row = [i] + angles + turns
            if node_name:
                self.solve(frame)
                row += list(self.position(node_name))
            if legs is not None:
                row += [legs[i]]
            rows.append(row)
            previous = angles
        self.solve(keep)

        with open(path, "w") as handle:
            handle.write(",".join(header) + "\n")
            for row in rows:
                handle.write(",".join(f"{v:.6f}" if isinstance(v, float) else str(v)
                                      for v in row) + "\n")
        self.last_result = verdict
        return path

    def _leg_numbers(self, movement, node_name, targets):
        keep = self.angles
        legs, leg = [], 0
        for frame in movement:
            self.solve(frame)
            if leg < len(targets):
                here = self.position(node_name)
                if math.dist(here, targets[leg]) < 1e-6:
                    legs.append(leg)
                    leg += 1
                    continue
            legs.append(leg)
        self.solve(keep)
        return legs

    # -- areas and volumes --------------------------------------------------

    def poses(self, step=None):
        """Every legal combination of the ARM angles, on a grid.

        The base pivot is left out on purpose: turning it doesn't change the
        cross-section one bit, it only decides which way the cross-section
        faces.  Sweeping it would be the same work over and over.
        """
        step = step if step is not None else self.area_settings.get("control_step", 1.0)
        steps = step if isinstance(step, dict) else \
            {c.name: step for c in self.planar_controls}
        keep = self.angles
        yield from self._walk(0, steps)
        self._apply(keep)

    def _walk(self, index, steps):
        if index == len(self.planar_controls):
            yield self.angles
            return
        arm = self.planar_controls[index]
        lo, hi = arm.limits
        lo = -180.0 if lo is None else lo
        hi = 180.0 if hi is None else hi
        increment = steps.get(arm.name, 1.0)
        angle = lo
        while angle <= hi + TOL:
            arm.set_angle(angle)
            yield from self._walk(index + 1, steps)
            angle += increment
        if angle - increment < hi - TOL:
            arm.set_angle(hi)
            yield from self._walk(index + 1, steps)

    def sweep(self, collect, step=None, verbose=False):
        """Walk every legal arm pose and pool the (r, z) points collect gives
        back.  One engine; swap the collector to change the question."""
        keep = self.angles
        points = []
        poses = 0
        for angles in self.poses(step):
            self.solve(angles)
            poses += 1
            points.extend(collect(self))
        self.solve(keep)
        if verbose:
            print(f"  swept {poses} poses -> {len(points)} points")
        return points

    @staticmethod
    def at_node(node_name):
        """Collector: where one node is, in the plane."""
        return lambda robot: [robot.planar(node_name)]

    @staticmethod
    def bodies(names=None, samples=12):
        """Collector: the bars themselves - the arm's cross-section footprint."""
        def collect(robot):
            chosen = robot.chain if names is None else [robot.vector(n) for n in names]
            points = []
            for vector in chosen:
                points.extend(vector.body(samples))
            return points
        return collect

    def reachable_area(self, node_name=None, step=None, cell=None, close=0.8,
                       name=None, color="tab:green", verbose=False):
        """The cross-section a node can reach.  Revolve it about the base axis
        and you have the volume."""
        node_name = node_name or self.tool.get("node") or self.chain[-1].end.name
        return Region(self.sweep(self.at_node(node_name), step, verbose),
                      cell=cell or self.area_settings.get("cell", 0.2), close=close,
                      name=name or f"reachable by {node_name}", color=color)

    def swept_area(self, step=None, cell=None, close=0.6, samples=None,
                   name=None, color="tab:orange", verbose=False):
        """The cross-section any part of the arm can occupy."""
        collector = self.bodies(samples=samples or
                                self.area_settings.get("body_samples", 12))
        return Region(self.sweep(collector, step, verbose),
                      cell=cell or self.area_settings.get("cell", 0.2), close=close,
                      name=name or "swept by the arm", color=color)

    def exclusion_zone(self, clearance=None, color="tab:red", verbose=False, **kwargs):
        """The swept cross-section padded out by a clearance."""
        clearance = self.area_settings.get("clearance", 2.0) if clearance is None else clearance
        return self.swept_area(verbose=verbose, **kwargs).grow(
            clearance, name=f"exclusion zone (+{clearance:g})", color=color)

    def area_map(self, node_name=None, clearance=None, verbose=True):
        """All three cross-sections at once, outermost first, ready to draw."""
        if verbose:
            print("Sweeping the arm...")
        clearance = self.area_settings.get("clearance", 2.0) if clearance is None else clearance
        swept = self.swept_area(verbose=verbose)
        return [swept.grow(clearance, name=f"exclusion zone (+{clearance:g})",
                           color="tab:red"),
                swept,
                self.reachable_area(node_name, verbose=verbose)]

    @property
    def base_sweep(self):
        """(min, max) the base pivot can turn through, in degrees."""
        lo, hi = self.base_rotation.limits
        return (-180.0 if lo is None else lo, 180.0 if hi is None else hi)

    def reachable_volume(self, node_name=None, region=None, verbose=False):
        """How much space a node can be put in, in cubic units, and the
        cross-section it came from."""
        region = region if region is not None else \
            self.reachable_area(node_name, verbose=verbose)
        lo, hi = self.base_sweep
        return region.volume(hi - lo), region

    # -- the reachable volume, as a regular grid -----------------------------

    def reachable_grid(self, region=None, step=0.5, z_step=1.0, node_name=None,
                       verify=True, verbose=False):
        """Which points in space the node can be put, on a regular grid.

        Returns (xs, ys, zs, occupied), occupied indexed [layer, y, x], with zs
        running from the top down.

        A point is reachable when some legal base angle brings it into the
        arm's plane and the cross-section covers it there.  That is true two
        ways: face the point, or face the other way and reach back over the
        base with a negative radius - which matters here, because this arm
        reaches along -r, so the second case is the usual one.

        verify then puts every surviving point through the real solver and
        drops the ones it can't actually hit.  Worth leaving on: the
        cross-section is a grid with the sampling speckle closed up, and
        closing is extensive, so its edge sits a shade outside the true
        envelope.  That error points the wrong way - it would hand you points
        the arm can't reach - and the check is closed form, so it costs well
        under a second.
        """
        node_name = node_name or self.tool.get("node") or self.chain[-1].end.name
        region = region if region is not None else self.reachable_area(node_name)
        low, high = self.base_sweep
        r_min, z_min, r_max, z_max = region.bounds
        reach = max(abs(r_min), abs(r_max))

        span = math.ceil(reach / step) * step
        xs = np.arange(-span, span + step / 2, step)
        ys = np.arange(-span, span + step / 2, step)
        top = math.floor(z_max / z_step) * z_step
        bottom = math.ceil(z_min / z_step) * z_step
        zs = np.arange(top, bottom - z_step / 2, -z_step)       # top down

        grid_x, grid_y = np.meshgrid(xs, ys)
        distance = np.hypot(grid_x, grid_y)
        bearing = np.degrees(np.arctan2(grid_y, grid_x))
        on_axis = distance < EPS

        occupied = np.zeros((len(zs), len(ys), len(xs)), dtype=bool)
        for layer, z in enumerate(zs):
            mask = np.zeros(distance.shape, dtype=bool)
            for shift, sign in ((0.0, 1.0), (180.0, -1.0), (-180.0, -1.0)):
                turned = bearing + shift
                legal = (turned >= low - TOL) & (turned <= high + TOL)
                mask |= legal & region.mask_at(sign * distance, z)
            # a point on the axis is reachable from any bearing at all
            if region.contains(0.0, z):
                mask |= on_axis
            occupied[layer] = mask

        if verify:
            keep, verdict = self.angles, self.last_result
            for layer, z in enumerate(zs):
                rows, cols = np.nonzero(occupied[layer])
                for j, i in zip(rows, cols):
                    if not self.move_to(node_name, float(xs[i]), float(ys[j]),
                                        float(z), apply=False):
                        occupied[layer, j, i] = False
            self.solve(keep)
            self.last_result = verdict

        if verbose:
            print(f"  {occupied.sum()} reachable points on a {step:g} x {step:g} x "
                  f"{z_step:g} grid, {len(zs)} layers from z {zs[0]:.0f} to {zs[-1]:.0f}")
        return xs, ys, zs, occupied

    def save_reachable_xyz(self, path="reachable_xyz.csv", step=1.0, z_step=1.0,
                           region=None, node_name=None, verify=True, verbose=False):
        """Write every point the node can reach as x, y, z.

        One row per grid point, walking top layer first.  step sets the x and y
        spacing, z_step the height between layers.  Every point written has
        been through the solver, so the file holds only places the arm can
        genuinely put the node.
        """
        xs, ys, zs, occupied = self.reachable_grid(region, step, z_step, node_name,
                                                   verify, verbose)
        written = 0
        with open(path, "w") as handle:
            handle.write("x,y,z\n")
            for layer, z in enumerate(zs):
                rows, cols = np.nonzero(occupied[layer])
                for j, i in zip(rows, cols):
                    handle.write(f"{xs[i]:.4f},{ys[j]:.4f},{z:.4f}\n")
                    written += 1
        return Result(True, path=path, points=written, layers=len(zs))

    # -- drawing support ----------------------------------------------------

    def planar_segments(self):
        """The bars in the arm's own plane: ((r0, z0), (r1, z1), name, role)."""
        return [(v.start.planar, v.end.planar, v.name, v.role) for v in self.chain]

    def world_segments(self):
        """The bars in the world: ((x0, y0, z0), (x1, y1, z1), name, role)."""
        return [(v.start.position, v.end.position, v.name, v.role) for v in self.chain]

    def report(self):
        lines = [f"{self.name}",
                 f"  base plane facing {self.base_angle:.1f} degrees "
                 f"(allowed {self.base_rotation.limit_text()})"]
        for vector in self.chain:
            lines.append(f"  {vector.name:16s} {vector.angle:8.2f}  "
                         f"{vector.role:8s} limits {vector.limit_text()}")
        for name, node in self.nodes.items():
            x, y, z = node.position
            lines.append(f"  {name:16s} world ({x:7.2f}, {y:7.2f}, {z:7.2f})   "
                         f"plane (r {node.r:7.2f}, z {node.z:7.2f})")
        return "\n".join(lines)

    def __repr__(self):
        return (f"<Robot {self.name!r} {len(self.nodes)} nodes, "
                f"{len(self.chain)} bars, {len(self.controls)} controls>")


# ------------------------------------------------------------- small maths --

def _nearest_equivalent(angle, reference, arm=None):
    """The same angle, shifted by whole turns to sit as close to `reference`
    as possible - so an arm never spins the long way round between frames.
    The shift is dropped if it would step outside the limits."""
    if reference is None:
        return angle
    shifted = angle + 360.0 * round((reference - angle) / 360.0)
    if arm is not None and not arm.in_limits(shifted):
        return angle
    return shifted


# -------------------------------------------------------------- movements --

# A movement is any iterable of angle dicts - {arm: angle}, base pivot
# included - one per frame.  Robot.plan_to / plan_path build the ones that
# need solving; these build the ones that don't.

def hold(angles, frames=10):
    return [dict(angles) for _ in range(frames)]


def angle_move(start, end, steps=45):
    """Drive the arms from one set of angles to another."""
    return [{name: start[name] + (end.get(name, start[name]) - start[name]) * (i / steps)
             for name in start} for i in range(steps + 1)]


def chain_movements(*movements):
    frames = []
    for movement in movements:
        frames.extend(movement)
    return frames


def clamped(robot, movement):
    return [robot.clamp(frame) for frame in movement]


# ----------------------------------------------------------------- output --

# draw() and animate() take a robot, so they don't care what the chain is.
# Two views: the arm's own plane, where the linkage actually lives, and the
# world, where the base rotation puts it.

def _view_limits(robot, view=None):
    view = view if view is not None else robot.view
    rlim = view.get("rlim", [-15.0, 25.0])
    zlim = view.get("zlim", [-12.0, 22.0])
    return rlim, zlim


def _setup_side(ax, robot, view=None):
    rlim, zlim = _view_limits(robot, view)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.5)
    ax.set_xlim(*rlim)
    ax.set_ylim(*zlim)
    ax.set_xlabel("r  (outward from the base axis)")
    ax.set_ylabel("z")
    ax.axvline(0.0, color="0.6", linestyle="--", linewidth=1)


def _setup_world(ax, robot, view=None):
    view = view if view is not None else robot.view
    rlim, zlim = _view_limits(robot, view)
    reach = max(abs(rlim[0]), abs(rlim[1]))
    ax.set_xlim(-reach, reach)
    ax.set_ylim(-reach, reach)
    ax.set_zlim(*zlim)
    try:
        ax.set_box_aspect((2 * reach, 2 * reach, zlim[1] - zlim[0]))
    except Exception:
        pass
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.view_init(elev=view.get("elev", 20) if view else 20,
                 azim=view.get("azim", -58) if view else -58)


def _colour(role):
    return "tab:blue" if role == "control" else "0.25"


def draw_side(robot, ax=None, regions=None, targets=None, labels=True, view=None):
    """The arm in its own plane - where the linkage actually lives."""
    ax = ax if ax else plt.gca()
    for i, region in enumerate(regions or []):
        region.plot(ax=ax, zorder=i)

    if targets:                       # world points, shown at their own radius
        ax.plot([math.hypot(p[0], p[1]) for p in targets], [p[2] for p in targets],
                "x", color="0.3", markersize=8, zorder=5, label="targets")

    for (r0, z0), (r1, z1), name, role in robot.planar_segments():
        ax.plot([r0, r1], [z0, z1], marker="o", color=_colour(role),
                linewidth=2.5 if role == "control" else 1.8, zorder=6)
        if labels:
            ax.text((r0 + r1) / 2, (z0 + z1) / 2 + 0.4, name, ha="center",
                    fontsize=8, zorder=7)
    base = robot.base_node
    ax.plot([base.r], [base.z], "s", color="black", markersize=8, zorder=8)
    _setup_side(ax, robot, view)
    return ax


def draw_world(robot, ax=None, volume=None, targets=None, trail=None, labels=True,
               view=None):
    """The arm in the world, with the base turned to where it is."""
    ax = ax if ax else plt.gcf().add_subplot(projection="3d")

    for region, colour, alpha in (volume or []):
        lo, hi = robot.base_sweep
        stride = max(1, len(region.polygons()[0]) // 140)
        for X, Y, Z in region.revolve(lo, hi,
                                      steps=robot.area_settings.get("revolve_steps", 25),
                                      stride=stride):
            ax.plot_surface(X, Y, Z, color=colour, alpha=alpha, linewidth=0,
                            shade=False, antialiased=False)
        # Ribs: the same cross-section drawn at a handful of bearings, with the
        # two ends of the sweep picked out.  A translucent surface on its own
        # reads as a smear; the ribs are what make it read as a solid of
        # revolution, and they show which way the plane is allowed to turn.
        for beta in np.linspace(lo, hi, 7):
            end = abs(beta - lo) < TOL or abs(beta - hi) < TOL
            cos_b, sin_b = math.cos(math.radians(beta)), math.sin(math.radians(beta))
            for poly in region.polygons():
                ax.plot(poly[:, 0] * cos_b, poly[:, 0] * sin_b, poly[:, 1],
                        color=colour, linewidth=1.6 if end else 0.7,
                        alpha=0.95 if end else 0.4)

    # the base axis, and the arc the plane can swing through
    rlim, zlim = _view_limits(robot, view)
    ax.plot([0, 0], [0, 0], zlim, color="0.6", linestyle="--", linewidth=1)
    lo, hi = robot.base_sweep
    arc = np.radians(np.linspace(lo, hi, 60))
    span = max(abs(rlim[0]), abs(rlim[1])) * 0.9
    ax.plot(span * np.cos(arc), span * np.sin(arc), np.full_like(arc, zlim[0]),
            color="0.6", linewidth=1)

    if targets:
        ax.plot([p[0] for p in targets], [p[1] for p in targets],
                [p[2] for p in targets], "x", color="0.3", markersize=7)
    if trail and len(trail) > 1:
        ax.plot([p[0] for p in trail], [p[1] for p in trail], [p[2] for p in trail],
                "-", color="0.55", linewidth=1.2)

    for (x0, y0, z0), (x1, y1, z1), name, role in robot.world_segments():
        ax.plot([x0, x1], [y0, y1], [z0, z1], marker="o", color=_colour(role),
                linewidth=3 if role == "control" else 2)
        if labels:
            ax.text((x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2 + 0.6, name,
                    fontsize=7)
    x, y, z = robot.base_node.position
    ax.plot([x], [y], [z], "s", color="black", markersize=7)
    _setup_world(ax, robot, view)
    return ax


def _title(robot):
    drives = "   ".join(f"{c.name} {c.angle:.1f}" for c in robot.controls)
    tip = robot.tool.get("node")
    if not tip:
        return drives
    x, y, z = robot.position(tip)
    return f"{drives}\n{tip} at ({x:.1f}, {y:.1f}, {z:.1f})"


def draw(robot, view="both", regions=None, volume=None, targets=None, trail=None,
         labels=True, title=None, show=True, figure=None, limits=None):
    """One picture of the arm as it stands.

    view "side" - the arm's own r-z plane
         "world" - 3D, with the base turned to where it is
         "both" - side by side (the default)
    """
    figure = figure if figure is not None else plt.gcf()
    if view == "side":
        draw_side(robot, figure.add_subplot(111), regions, targets, labels, limits)
    elif view == "world":
        draw_world(robot, figure.add_subplot(111, projection="3d"), volume, targets,
                   trail, labels, limits)
    else:
        draw_side(robot, figure.add_subplot(1, 2, 1), regions, targets, labels, limits)
        draw_world(robot, figure.add_subplot(1, 2, 2, projection="3d"), volume,
                   targets, trail, labels, limits)
    figure.suptitle(title if title else _title(robot), fontsize=10)
    if show:
        plt.show()
    return figure


def animate(robot, movement, pause=0.05, view="both", regions=None, volume=None,
            targets=None, trace=None, on_limit="skip", labels=False, keep_open=True,
            limits=None):
    """Play a movement - any iterable of angle dicts.

    trace : name of a node to leave a trail behind, or None for the tool
    on_limit : "skip" (default), "clamp", or "ignore"
    """
    trace = trace if trace is not None else robot.tool.get("node")
    trail = []
    figure = plt.gcf()

    for frame in movement:
        if on_limit == "clamp":
            frame = robot.clamp(frame)
        result = robot.move_all(frame, enforce_limits=(on_limit != "ignore"))
        if not result:
            print(f"  frame skipped - {result.reason}")
            continue
        if trace:
            trail.append(robot.position(trace))

        figure.clf()
        draw(robot, view=view, regions=regions, volume=volume, targets=targets,
             trail=trail, labels=labels, show=False, figure=figure, limits=limits)
        plt.pause(pause)

    if keep_open:
        plt.show()


def save_reachable_slices(robot, folder="reachable_slices", step=0.25, z_step=1.0,
                          region=None, node_name=None, prefix="slice", dpi=110,
                          verify=True, verbose=False):
    """One PNG per height, looking straight down, from the top layer to the bottom.

    Every slice is drawn on the same axes and at the same scale, so flicking
    through them shows how the footprint really changes with height rather
    than each one being rescaled to fit.  Returns a list of
    (path, z, area) and also writes a summary CSV of area against height.

    step is the x-y resolution of each picture, z_step the gap between
    layers.  As with the CSV, every point drawn has been through the solver.
    """
    import os

    region = region if region is not None else robot.reachable_area(node_name)
    xs, ys, zs, occupied = robot.reachable_grid(region, step, z_step, node_name,
                                                verify, verbose)

    os.makedirs(folder, exist_ok=True)
    reach = max(abs(xs[0]), abs(xs[-1]))
    low, high = robot.base_sweep
    arc = np.radians(np.linspace(low + 180.0, high + 180.0, 120))   # where the arm reaches
    cell_area = step * step
    out = []

    for layer, z in enumerate(zs):
        mask = occupied[layer]
        figure, ax = plt.subplots(figsize=(5.6, 5.6))
        ax.pcolormesh(xs, ys, mask, cmap="Greens", vmin=0, vmax=1.6, shading="nearest")
        if mask.any():
            ax.contour(xs, ys, mask.astype(float), levels=[0.5],
                       colors=["tab:green"], linewidths=1.4)
        ax.plot(reach * 0.97 * np.cos(arc), reach * 0.97 * np.sin(arc),
                color="0.7", linewidth=1)
        ax.plot([0], [0], "s", color="black", markersize=7)          # the base
        ax.set_xlim(-reach, reach)
        ax.set_ylim(-reach, reach)
        ax.set_aspect("equal")
        ax.grid(True, linewidth=0.4, alpha=0.5)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        area = float(mask.sum()) * cell_area
        ax.set_title(f"z = {z:+.1f}      reachable footprint {area:.1f} sq units",
                     fontsize=10)
        path = f"{folder}/{prefix}_{layer:02d}_z{z:+06.1f}.png"
        figure.tight_layout()
        figure.savefig(path, dpi=dpi)
        plt.close(figure)
        out.append((path, float(z), area))

    with open(f"{folder}/{prefix}_areas.csv", "w") as handle:
        handle.write("layer,z,area,file\n")
        for layer, (path, z, area) in enumerate(out):
            handle.write(f"{layer},{z:.4f},{area:.4f},{path.rsplit('/', 1)[-1]}\n")
    return out


def show_areas(robot, regions=None, targets=None, node_name=None, clearance=None,
               title=None, verbose=True):
    """The cross-sections the arm moves through, and the volume they revolve
    into, side by side."""
    regions = regions if regions is not None else \
        robot.area_map(node_name, clearance, verbose=verbose)
    reachable = regions[-1]
    volume, _ = robot.reachable_volume(region=reachable)
    if verbose:
        for region in regions:
            print(f"  {region.name:28s} cross-section {region.area:8.1f} sq units")
        lo, hi = robot.base_sweep
        print(f"  revolved through {hi - lo:.0f} degrees -> {volume:.0f} cubic units")

    figure = plt.gcf()
    draw_side(robot, figure.add_subplot(1, 2, 1), regions, targets, labels=False)
    draw_world(robot, figure.add_subplot(1, 2, 2, projection="3d"),
               volume=[(reachable, "tab:green", 0.13)], targets=targets, labels=False)
    figure.suptitle(title or f"{robot.name}\nreachable cross-section, and the "
                             f"volume it revolves into", fontsize=10)
    plt.show()
    return regions


# -------------------------------------------------------------------- demo --

if __name__ == "__main__":

    robot = Robot.from_file("arm_config_v6.json")
    print(robot.report())

    # --- where everything is ------------------------------------------------
    print("\nNode positions (x, y, z):")
    for name, (x, y, z) in robot.positions().items():
        print(f"  {name:10s} ({x:7.2f}, {y:7.2f}, {z:7.2f})")

    # --- driving it ---------------------------------------------------------
    print("\nDriving the arms:")
    for name, angle in [("base_rotation", -40), ("control_arm_a", 110),
                        ("main_arm", 250), ("main_arm", 300), ("base_rotation", 120)]:
        result = robot.move(name, angle)
        x, y, z = robot.position("tool_tip")
        print(f"  {name:14s} -> {angle:6}   " +
              (f"ok, tool_tip ({x:6.2f}, {y:6.2f}, {z:6.2f})" if result
               else f"refused: {result.reason}"))

    # --- inverse kinematics -------------------------------------------------
    print("\nPutting the tool tip on a point in space:")
    for point in [(-14.0, -5.0, 3.0), (-6.0, 9.0, -2.0), (-20.0, 0.0, 6.0), (40.0, 0.0, 0.0)]:
        result = robot.move_to("tool_tip", *point)
        if result:
            angles = "  ".join(f"{n} {a:7.2f}" for n, a in result["angles"].items())
            print(f"  {str(point):>20s} -> {angles}")
        else:
            print(f"  {str(point):>20s} -> refused: {result.reason}")

    # --- what it can reach --------------------------------------------------
    robot.move_all({"base_rotation": 0.0, "control_arm_a": 90.0, "main_arm": 180.0})
    print("\nAreas of movement:")
    zones = robot.area_map(verbose=False)
    for region in zones:
        print(f"  {region.name:26s} cross-section {region.area:8.1f} sq units")
    volume, reachable = robot.reachable_volume(region=zones[-1])
    low, high = robot.base_sweep
    print(f"  revolved through {high - low:.0f} degrees -> {volume:.0f} cubic units")
    zones[0].save_perimeter("v6_exclusion_cross_section.csv")
    print("  cross-section perimeter written to v6_exclusion_cross_section.csv")

    # --- the reachable volume, as data and as pictures -----------------------
    '''written = robot.save_reachable_xyz("v6_reachable_xyz.csv", step=1.0, z_step=1.0,
                                       region=reachable)
    print(f"  {written['points']} reachable points written to v6_reachable_xyz.csv "
          f"({written['layers']} layers, 1 unit steps)")

    slices = save_reachable_slices(robot, "reachable_slices", step=0.25, z_step=1.0,
                                   region=reachable)
    print(f"  {len(slices)} top-down slices written to reachable_slices/ "
          f"(z {slices[0][1]:+.0f} down to {slices[-1][1]:+.0f}), "
          f"widest {max(a for _, _, a in slices):.0f} sq units at "
          f"z {max(slices, key=lambda s: s[2])[1]:+.0f}")'''

    # a horizontal arc at constant height - the base pivot doing the work
    # A horizontal arc at constant height, swung by the base pivot.  The arm
    # reaches out along -r, so a base bearing of b puts the tool on bearing
    # b + 180 - which is why these points are on the -x side.
    targets = [(-15.0 * math.cos(math.radians(d)), -15.0 * math.sin(math.radians(d)), 3.0)
               for d in range(-70, 71, 20)]

    show_areas(robot, regions=zones, targets=targets, verbose=False)

    # --- the step-by-step angles, as a file ---------------------------------
    robot.move_all({"base_rotation": 0.0, "control_arm_a": 90.0, "main_arm": 180.0})
    movement = robot.steps_through("tool_tip", targets, "v6_steps_arc.csv", steps=10)
    print(f"\n{len(movement)} steps around the arc written to v6_steps_arc.csv")
    if robot.last_result.get("skipped"):
        print(f"  ({len(robot.last_result['skipped'])} steps were out of reach)")

    # --- run it -------------------------------------------------------------
    print("\nRunning the path...")
    animate(robot, movement, regions=zones, volume=[(reachable, "tab:green", 0.10)],
            targets=targets, pause=0.04, view="world")

    """[(-10.0000,3.0000,9.0000),
                (-10.0000,-10.0000,9.0000),
                (-11.0000,6.0000,6.0000),
                (-22.0000,-3.0000,5.0000),
                (-4.0000,-19.0000,2.0000),
                (-18.0000,-11.0000,0.0000),
                (-9.0000,0.0000,-3.0000),
                (-14.0000,-6.0000,-5.0000)]"""

    # other things the same pieces will do:
    #
    #   robot.move_to("node_end", x, y, z)          any node, not just the tip
    #   robot.reachable_area("node_end")            any node's envelope
    #   robot.reachable_area().inner_box(0.4)       a safe (r, z) rectangle
    #   robot.reachable_grid(step=0.5, z_step=0.5)  the volume as a boolean grid
    #   robot.save_reachable_xyz(..., node_name="node_end")     any node
    #   save_reachable_slices(robot, verify=False)  quicker, edge a shade generous
    #   robot.plan_path(..., straight=False)        ease the motors
    #   draw(robot, view="world")                   3D only
    #   animate(robot, angle_move(robot.angles, {"base_rotation": 90}, 40))
