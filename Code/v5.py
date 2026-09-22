"""
Community robot arm - v5

A rebuild rather than another layer on v4.  Three ideas hold it together:

    Node    a point in the linkage.  Pinned in place, free to be solved, or
            attached partway along a vector.  Nodes are shared, so a joint
            two arms meet at is one object, not two copies kept in step.

    Vector  a rigid bar between two nodes, with optional nodes attached
            partway along it.  A control vector's angle is an input you drive;
            a dependent vector's position falls out of the solve.  Movement
            limits live here.

    Robot   owns the nodes and vectors, does the solving, reports failures,
            and answers the questions you actually have: move this arm to that
            angle, put that node on this point, show me where it can reach and
            what floor it sweeps.

Everything is described in a JSON config file, so a different linkage is a
different config rather than different code.  Nothing here reaches for the
command line, and the only imports are the standard library plus numpy and
matplotlib.

    source .venv/bin/activate
    python v5.py
"""

import json
import math

import numpy as np
import matplotlib.pyplot as plt


TOL = 1e-9          # slack when comparing angles against a limit
EPS = 1e-7          # slack when comparing positions


# ---------------------------------------------------------------- results --

class Result:
    """What came back from a calculation.

    Truthy when it worked, and carries the reason when it didn't, so callers
    can branch on it without catching exceptions:

        if not robot.move("control_arm_a", 200):
            print(robot.last_result.reason)
    """

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
        if self.ok:
            return f"<Result ok {self.data if self.data else ''}>".replace("  ", " ")
        return f"<Result failed: {self.reason}>"


# ------------------------------------------------------------------ nodes --

class Node:
    """A point in the linkage.

    fixed    - pinned at a known x, y.  The base of the arm is one of these.
    free     - worked out by the solver from whatever it is connected to.
    attached - a free node that sits partway along a vector (see Vector).
               It is still solved, just by a different rule.
    """

    def __init__(self, name, fixed=None, branch=1):
        self.name = name
        self.fixed_at = tuple(fixed) if fixed is not None else None
        self.branch = 1 if branch >= 0 else -1   # which elbow-up/down solution
        self.x = self.fixed_at[0] if self.fixed_at else 0.0
        self.y = self.fixed_at[1] if self.fixed_at else 0.0
        self.solved = self.fixed_at is not None
        # filled in by Robot while it wires everything together
        self.vectors = []        # vectors that start, end or pass through here
        self.attached_to = None  # (vector_name, distance_from_start) if attached

    @property
    def is_fixed(self):
        return self.fixed_at is not None

    @property
    def is_attached(self):
        return self.attached_to is not None

    @property
    def position(self):
        return (self.x, self.y)

    @position.setter
    def position(self, xy):
        self.x, self.y = float(xy[0]), float(xy[1])
        self.solved = True

    def reset(self):
        """Forget the solved position, unless it is pinned."""
        if self.is_fixed:
            self.x, self.y = self.fixed_at
            self.solved = True
        else:
            self.solved = False

    def distance_to(self, other):
        return math.hypot(other.x - self.x, other.y - self.y)

    def __repr__(self):
        kind = "fixed" if self.is_fixed else ("attached" if self.is_attached else "free")
        return f"<Node {self.name} ({kind}) at ({self.x:.2f}, {self.y:.2f})>"


# ---------------------------------------------------------------- vectors --

class Vector:
    """A rigid bar between two nodes.

    A vector may be described by its two nodes, by a length, by an angle, or
    by a combination - whatever the config gives.  If both ends are fixed
    nodes the length is measured from them; otherwise the length is given.

    Nodes attached partway along the bar are listed in `attachments` as
    {node_name: distance_from_start}.  Everything on one vector is collinear
    and rigid, which is what lets the solver fix the whole bar from any two
    known points on it.

    Movement limits live here.  A limit is a number, or {"relative_to": other
    vector, "offset": degrees} so one arm's limit can track another's angle.
    """

    def __init__(self, name, start, end, length=None, angle=0.0, role="dependent",
                 min_angle=None, max_angle=None, attachments=None, robot=None):
        self.name = name
        self.start = start                  # Node
        self.end = end                      # Node
        self.role = role                    # "control" or "dependent"
        self.robot = robot                  # Robot that owns this vector
        self._angle = float(angle)          # only meaningful for a control vector
        self.min_angle = min_angle
        self.max_angle = max_angle
        self.attachments = dict(attachments) if attachments else {}

        if length is not None:
            self.length = float(length)
        elif start.is_fixed and end.is_fixed:
            self.length = start.distance_to(end)      # measured from the points
        else:
            raise ValueError(
                f"{name}: give a length, or pin both {start.name} and {end.name}"
            )

    # -- what it is ---------------------------------------------------------

    @property
    def is_control(self):
        "Control vectors are the ones you drive; dependent vectors are worked out by the solver."
        return self.role == "control"

    @property
    def angle(self):
        """Degrees.  A control vector reports the angle you set; a dependent
        one reports the angle it has ended up at."""
        if self.is_control:
            return self._angle
        if not (self.start.solved and self.end.solved):
            return float("nan")
        return math.degrees(math.atan2(self.end.y - self.start.y,
                                       self.end.x - self.start.x))

    def set_angle(self, value):
        """Set a control vector's angle without checking limits.
        Robot.move() is the checked way in."""
        if not self.is_control:
            raise TypeError(f"{self.name} is a dependent vector - it can't be driven")
        self._angle = float(value)

    # -- limits -------------------------------------------------------------

    def _resolve(self, limit):
        #Helper Function
        """Turn a limit into a number, or None for no limit.  A limit that tracks another vector is worked out fresh every time."""
        if limit is None:
            return None
        if isinstance(limit, dict):
            other = self.robot.vector(limit["relative_to"])
            return other.angle + float(limit.get("offset", 0.0))
        return float(limit)

    @property
    def limits(self):
        """(min, max) right now.  Either may be None for no limit.  A limit
        that tracks another vector is worked out fresh every time."""
        return self._resolve(self.min_angle), self._resolve(self.max_angle)

    def in_limits(self, angle=None):
        """Is the given angle (or the current one) within the limits?"""
        angle = self.angle if angle is None else angle
        lo, hi = self.limits
        if lo is not None and angle < lo - TOL:
            return False
        if hi is not None and angle > hi + TOL:
            return False
        return True

    def clamp(self, angle):
        """The nearest legal angle to the given one.  A limit that tracks another vector is worked out fresh every time."""
        lo, hi = self.limits
        if lo is not None:
            angle = max(angle, lo)
        if hi is not None:
            angle = min(angle, hi)
        return angle

    def limit_text(self):
        """A human-readable description of the limits, for error messages."""
        lo, hi = self.limits
        lo = "-inf" if lo is None else f"{lo:.1f}"
        hi = "+inf" if hi is None else f"{hi:.1f}"
        return f"{lo}° to {hi}°"

    # -- geometry -----------------------------------------------------------

    @property
    def points(self):
        """((x0, y0), (x1, y1)) - the bar as drawn."""
        return (self.start.x, self.start.y), (self.end.x, self.end.y)

    def point_at(self, distance):
        """The point `distance` along the bar from its start node."""
        t = distance / self.length
        return (self.start.x + (self.end.x - self.start.x) * t,
                self.start.y + (self.end.y - self.start.y) * t)

    def body(self, samples=12):
        """Points spread along the bar - used when working out swept area."""
        (x0, y0), (x1, y1) = self.points
        return [(x0 + (x1 - x0) * i / samples, y0 + (y1 - y0) * i / samples)
                for i in range(samples + 1)]

    @property
    def stations(self):
        """Everything sitting on this bar, as (node, distance from start),
        ordered along it.  Two known stations fix the whole bar."""
        out = [(self.start, 0.0), (self.end, self.length)]
        out += [(self.robot.node(n), float(d)) for n, d in self.attachments.items()]
        return sorted(out, key=lambda s: s[1])

    def __repr__(self):
        return (f"<Vector {self.name} ({self.role}) {self.start.name}->{self.end.name} "
                f"len {self.length:g} angle {self.angle:.1f}°>")


# ----------------------------------------------------------------- areas ---

class Region:
    """An area, held as an occupancy grid.

    Robot hands these back.  A grid rather than a polygon because the shapes
    involved are awkward - concave, sometimes with a hole in the middle - and
    because growing one outwards by a clearance distance is then just a
    dilation, which can't accidentally cut inside the real shape.
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
        self.x0 = pts[:, 0].min() - pad * cell
        self.y0 = pts[:, 1].min() - pad * cell
        nx = int(math.ceil((pts[:, 0].max() - self.x0) / cell)) + pad + 1
        ny = int(math.ceil((pts[:, 1].max() - self.y0) / cell)) + pad + 1

        grid = np.zeros((ny, nx), dtype=bool)
        grid[np.rint((pts[:, 1] - self.y0) / cell).astype(int),
             np.rint((pts[:, 0] - self.x0) / cell).astype(int)] = True

        # close the speckle left by sampling at discrete angles, without
        # filling holes that are genuinely there
        r = int(round(close / cell))
        if r > 0:
            grid = _erode(_dilate(grid, r), r)   # the array grows by r a side
            self.x0 -= r * cell                  # so the origin moves with it
            self.y0 -= r * cell
        self.grid = grid

    @classmethod
    def _from_grid(cls, grid, x0, y0, cell, name, color, pad=3):
        obj = cls.__new__(cls)
        obj.grid = np.pad(grid, pad, constant_values=False)
        obj.x0 = x0 - pad * cell
        obj.y0 = y0 - pad * cell
        obj.cell = cell
        obj.name = name
        obj.color = color
        obj._polys = None
        return obj

    def grow(self, clearance, name=None, color=None):
        """A copy padded outwards by `clearance` - the safe perimeter."""
        r = int(round(clearance / self.cell))
        return Region._from_grid(
            _dilate(self.grid, r),
            self.x0 - r * self.cell, self.y0 - r * self.cell, self.cell,
            name if name else f"{self.name} + {clearance:g} clearance",
            color if color else self.color,
        )

    @property
    def area(self):
        return float(self.grid.sum()) * self.cell ** 2

    @property
    def bounds(self):
        ys, xs = np.nonzero(self.grid)
        return (float(self.x0 + xs.min() * self.cell), float(self.y0 + ys.min() * self.cell),
                float(self.x0 + xs.max() * self.cell), float(self.y0 + ys.max() * self.cell))

    def inner_box(self, margin=0.0):
        """The largest axis-aligned rectangle that fits entirely inside the
        region, as (min_x, min_y, max_x, max_y).

        bounds gives the box AROUND the region, which for a blob-shaped
        reachable area includes corners the arm can't get to.  This gives the
        opposite: a rectangle every point of which is inside.  Somewhere safe
        to lay text or a drawing out in.
        """
        grid = self.grid
        heights = np.zeros(grid.shape[1], dtype=int)
        best = (0, 0, 0, 0, 0)          # area, bottom row, left col, right col, height

        for row in range(grid.shape[0]):
            heights = np.where(grid[row], heights + 1, 0)
            stack = []                  # largest rectangle in this histogram
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
        return (self.x0 + left * self.cell + margin,
                self.y0 + (row - tall + 1) * self.cell + margin,
                self.x0 + right * self.cell - margin,
                self.y0 + row * self.cell - margin)

    def contains(self, x, y):
        ix = int(round((x - self.x0) / self.cell))
        iy = int(round((y - self.y0) / self.cell))
        if not (0 <= ix < self.grid.shape[1] and 0 <= iy < self.grid.shape[0]):
            return False
        return bool(self.grid[iy, ix])

    def polygons(self):
        """The outline as closed (x, y) loops - outer edges and any holes."""
        if self._polys is None:
            xs = self.x0 + np.arange(self.grid.shape[1]) * self.cell
            ys = self.y0 + np.arange(self.grid.shape[0]) * self.cell
            fig = plt.figure()
            try:
                cs = fig.add_subplot(111).contour(xs, ys, self.grid.astype(float),
                                                  levels=[0.5])
                self._polys = [np.asarray(s) for s in cs.allsegs[0] if len(s) > 2]
            finally:
                plt.close(fig)
        return self._polys

    def save_perimeter(self, path):
        """Write the outline to CSV (loop, x, y) - for CAD, or for marking the
        zone out on the bench."""
        with open(path, "w") as f:
            f.write("loop,x,y\n")
            for i, poly in enumerate(self.polygons()):
                for (x, y) in poly:
                    f.write(f"{i},{x:.4f},{y:.4f}\n")
        return path

    def plot(self, ax=None, fill=True, alpha=0.18, linewidth=1.6, label=None, zorder=0):
        ax = ax if ax else plt.gca()
        label = self.name if label is None else label
        xs = self.x0 + np.arange(self.grid.shape[1]) * self.cell
        ys = self.y0 + np.arange(self.grid.shape[0]) * self.cell
        if fill:
            ax.contourf(xs, ys, self.grid.astype(float), levels=[0.5, 1.5],
                        colors=[self.color], alpha=alpha, zorder=zorder)
        first = True
        for poly in self.polygons():
            ax.plot(poly[:, 0], poly[:, 1], color=self.color, linewidth=linewidth,
                    zorder=zorder + 0.1, label=label if first else None)
            first = False
        return ax

    def __repr__(self):
        b = self.bounds
        return (f"<Region {self.name!r} area={self.area:.1f} "
                f"bounds=({b[0]:.1f}, {b[1]:.1f})..({b[2]:.1f}, {b[3]:.1f})>")


def _disc_offsets(r_cells):
    #Helper Function
    """All the (dy, dx) offsets that fit in a disc of radius r_cells."""
    return [(dy, dx)
            for dy in range(-r_cells, r_cells + 1)
            for dx in range(-r_cells, r_cells + 1)
            if dx * dx + dy * dy <= r_cells * r_cells]


def _spread(grid, r_cells, fill):
    #Healper Function
    """Grid OR-ed with itself shifted over a disc.  Padded by r_cells a side."""
    padded = np.pad(grid, r_cells, constant_values=fill)
    out = np.zeros_like(padded)
    for dy, dx in _disc_offsets(r_cells):
        out |= np.roll(np.roll(padded, dy, axis=0), dx, axis=1)
    return out


def _dilate(grid, r_cells):
    #Helper Function
    """Grid OR-ed with itself shifted over a disc.  Padded by r_cells a side."""
    return grid.copy() if r_cells <= 0 else _spread(grid, r_cells, False)


def _erode(grid, r_cells):
    #Helper Function
    """Grid AND-ed with itself shifted over a disc.  Padded by r_cells a side."""
    if r_cells <= 0:
        return grid.copy()
    return ~_spread(~grid, r_cells, True)[r_cells:-r_cells, r_cells:-r_cells]


# ------------------------------------------------------------------ robot --

class Robot:
    """The linkage as a whole.

    Built from a config, it owns the nodes and vectors, solves the geometry,
    and is the way you interact with the arm:

        robot.move("control_arm_a", 120)        drive a control arm
        robot.move_to("node_end", -5, 5)        put a node on a point
        robot.position("node_end")              where something is
        robot.reachable_area("node_end")        where it can go
        robot.swept_area()                      what floor the arm covers
        robot.exclusion_zone(2.0)               that, plus a safety margin

    Every call that can fail returns a Result rather than raising, and the
    reason is kept on robot.last_result.
    """

    # -- building -----------------------------------------------------------

    def __init__(self, config):
        self.config = config
        self.name = config.get("name", "robot")
        self.view = config.get("view", {})
        self.tool = config.get("tool", {})
        self.area_settings = config.get("areas", {})
        self.last_result = Result(True)

        self.nodes = {}
        for node_name, spec in config["nodes"].items():
            fixed = None
            if spec.get("type") == "fixed" or ("x" in spec and "y" in spec):
                fixed = (float(spec.get("x", 0.0)), float(spec.get("y", 0.0)))
            self.nodes[node_name] = Node(node_name, fixed=fixed,
                                         branch=spec.get("branch", 1))

        self.vectors = {}
        for vec_name, spec in config["vectors"].items():
            for end in ("start", "end"):
                if spec[end] not in self.nodes:
                    raise ValueError(f"{vec_name}: no node called {spec[end]!r}")
            vector = Vector(
                vec_name,
                start=self.nodes[spec["start"]],
                end=self.nodes[spec["end"]],
                length=spec.get("length"),
                angle=spec.get("angle", 0.0),
                role=spec.get("role", "dependent"),
                min_angle=spec.get("min_angle"),
                max_angle=spec.get("max_angle"),
                attachments=spec.get("attachments"),
                robot=self,
            )
            self.vectors[vec_name] = vector

        self._wire_up()
        self.solve()

    @classmethod
    def from_file(cls, path):
        """Load a config. Falls back to looking beside this script, so it
        doesn't matter which folder you run from."""
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

    def _wire_up(self):
        #Helper Function
        """Work out, once, what is connected to what.

        Every vector is a rigid line carrying its start node, its end node and
        anything attached partway along.  That gives two things the solver
        needs: the list of stations on each line, and the fixed distance
        between every pair of nodes that share a line.
        """
        self._lines = []        # [(vector, [(node, distance), ...]), ...]
        self._rigid = {}        # node name -> [(other node, distance), ...]

        for vector in self.vectors.values():
            for node_name, distance in vector.attachments.items():
                if node_name not in self.nodes:
                    raise ValueError(f"{vector.name}: no node called {node_name!r}")
                node = self.nodes[node_name]
                if node.is_fixed:
                    raise ValueError(f"{node_name} is pinned, so it can't also be "
                                     f"attached along {vector.name}")
                node.attached_to = (vector.name, float(distance))
                if not 0.0 <= distance <= vector.length:
                    raise ValueError(f"{vector.name}: {node_name} is attached at "
                                     f"{distance}, off the end of a {vector.length} bar")

            stations = vector.stations
            self._lines.append((vector, stations))
            for node, _ in stations:
                if vector not in node.vectors:
                    node.vectors.append(vector)
            # every pair on the line is a fixed distance apart
            for i, (node_a, d_a) in enumerate(stations):
                for (node_b, d_b) in stations[i + 1:]:
                    gap = abs(d_b - d_a)
                    self._rigid.setdefault(node_a.name, []).append((node_b.name, gap))
                    self._rigid.setdefault(node_b.name, []).append((node_a.name, gap))

        self.controls = [v for v in self.vectors.values() if v.is_control]
        if not self.controls:
            raise ValueError("No control vectors - nothing to drive")

    # -- handles ------------------------------------------------------------

    def node(self, name):
        try:
            return self.nodes[name]
        except KeyError:
            raise KeyError(f"No node called {name!r}. Have: {', '.join(self.nodes)}")

    def vector(self, name):
        try:
            return self.vectors[name]
        except KeyError:
            raise KeyError(f"No vector called {name!r}. Have: {', '.join(self.vectors)}")

    def position(self, node_name):
        return self.node(node_name).position

    @property
    def control_names(self):
        return [v.name for v in self.controls]

    @property
    def angles(self):
        """The control angles, as a plain dict - this is the arm's whole state."""
        return {v.name: v.angle for v in self.controls}

    def _apply(self, angles):
        #Helper Function
        """Set control angles in config order, so a limit that tracks another
        arm is always read after that arm has moved."""
        for vector in self.controls:
            if vector.name in angles:
                vector.set_angle(angles[vector.name])

    # -- limits -------------------------------------------------------------

    def breaches(self, angles=None):
        """Which control arms would be outside their limits, described."""
        keep = self.angles
        if angles is not None:
            self._apply(angles)
        out = [f"{v.name} at {v.angle:.1f}° (allowed {v.limit_text()})"
               for v in self.controls if not v.in_limits()]
        if angles is not None:
            self._apply(keep)
        return out

    def within_limits(self, angles=None):
        return not self.breaches(angles)

    def clamp(self, angles):
        """The nearest legal set of angles.  Applied in config order, so a
        dependent limit is measured against the already-clamped arm."""
        keep = self.angles
        out = {}
        for vector in self.controls:
            wanted = angles.get(vector.name, vector.angle)
            vector.set_angle(vector.clamp(wanted))
            out[vector.name] = vector.angle
        self._apply(keep)
        return out

    # -- the solve ----------------------------------------------------------

    def solve(self, angles=None):
        """Work out where every node is, for the current (or given) control
        angles.  Returns a Result; the arm is left in the solved pose.

        Three rules, always tried in this order, one placement at a time:

            driven    a control vector with a placed start puts its end
                      on a known point
            line      two placed stations on a bar fix every other station
                      on that bar
            pair      a node with two placed rigid neighbours sits where
                      two circles cross

        Order matters.  A node that is collinear with its neighbours has two
        circles that meet at a single tangent point, which is numerically
        horrible - so the line rule always gets first refusal, and the pass
        restarts from the top whenever anything is placed.
        """
        if angles is not None:
            self._apply(angles)

        for node in self.nodes.values():
            node.reset()

        while True:
            pending = [n for n in self.nodes.values() if not n.solved]
            if not pending:
                break

            outcome = None
            for rule in (self._place_driven, self._place_on_lines, self._place_from_pair):
                outcome = rule()
                if outcome is not None:
                    break

            if outcome is None:
                return self._fail("under-constrained, couldn't place: "
                                  + ", ".join(n.name for n in pending))
            if outcome is not True:
                return outcome              # a rule reported a real failure

        self.last_result = Result(True)
        return self.last_result

    def _place_driven(self):
        #Helper Function
        for vector in self.controls:
            if vector.start.solved and not vector.end.solved:
                radians = math.radians(vector.angle)
                vector.end.position = (
                    vector.start.x + vector.length * math.cos(radians),
                    vector.start.y + vector.length * math.sin(radians))
                return True
        return None

    def _place_on_lines(self):
        #Helper Function
        for vector, stations in self._lines:
            known = [(n, d) for n, d in stations if n.solved]
            if len(known) < 2 or len(known) == len(stations):
                continue
            # the two furthest apart, for the steadiest extrapolation
            (node_a, d_a), (node_b, d_b) = known[0], known[-1]
            span = d_b - d_a
            measured = node_a.distance_to(node_b)
            if abs(measured - abs(span)) > 1e-6 * max(1.0, vector.length):
                return self._fail(
                    f"{vector.name} doesn't fit: {node_a.name} and {node_b.name} "
                    f"should be {abs(span):.3f} apart but are {measured:.3f}")
            for node, distance in stations:
                if node.solved:
                    continue
                t = (distance - d_a) / span
                node.position = (node_a.x + (node_b.x - node_a.x) * t,
                                 node_a.y + (node_b.y - node_a.y) * t)
            return True
        return None

    def _place_from_pair(self):
        #Helper Function
        for node in self.nodes.values():
            if node.solved:
                continue
            known = [(self.nodes[other], gap)
                     for other, gap in self._rigid.get(node.name, [])
                     if self.nodes[other].solved]
            if len(known) < 2:
                continue
            (node_a, r_a), (node_b, r_b) = known[0], known[1]
            meeting = _meet(node_a.position, r_a, node_b.position, r_b, node.branch)
            if meeting is None:
                return self._fail(
                    f"{node_a.name} and {node_b.name} can't reach each other, so "
                    f"{node.name} has nowhere to go "
                    f"(they are {node_a.distance_to(node_b):.2f} apart, "
                    f"arms are {r_a:g} and {r_b:g})")
            node.position = meeting
            return True
        return None

    def _fail(self, reason):
        #Helper Function
        self.last_result = Result(False, reason)
        return self.last_result

    # -- driving it ---------------------------------------------------------

    def move(self, control_name, angle, enforce_limits=True):
        """Drive one control arm to an angle and re-solve."""
        return self.move_all({control_name: angle}, enforce_limits)

    def move_all(self, angles, enforce_limits=True):
        """Drive several control arms at once and re-solve.  The arm is left
        untouched if the move is refused or the linkage won't close."""
        keep = self.angles
        wanted = dict(keep)
        for name, angle in angles.items():
            self.vector(name)       # raises on a typo, which is what you want
            wanted[name] = float(angle)

        if enforce_limits and not self.within_limits(wanted):
            reason = "outside the limits: " + "; ".join(self.breaches(wanted))
            self.solve(keep)
            return self._fail(reason)

        result = self.solve(wanted)
        if not result:
            reason = result.reason
            self.solve(keep)                 # put it back the way it was
            return self._fail(reason)
        return result

    def nudge(self, control_name, degrees):
        return self.move(control_name, self.vector(control_name).angle + degrees)

    # -- inverse kinematics -------------------------------------------------

    def move_to(self, node_name, x, y, tol=1e-9, max_iter=80, seeds=8, apply=True):
        """Put a node on a point by working out what the control arms have to do.

        Numerical, so it doesn't care how many control arms there are or which
        node you ask for.  It starts from the current pose, which keeps a path
        continuous, and falls back to a spread of starting guesses if that
        doesn't converge.  Returns a Result carrying the angles it found.
        """
        self.node(node_name)
        target = np.array([float(x), float(y)])
        keep = self.angles

        attempts = [keep]
        if seeds:
            attempts += self._seed_angles(seeds)

        best = None
        for start in attempts:
            found = self._newton(node_name, target, start, tol, max_iter)
            if found is None:
                continue
            cost = sum(abs(found[n] - keep[n]) for n in found)
            if best is None or cost < best[0]:
                best = (cost, found)
            if start is keep:
                break        # already the closest possible answer

        if best is None:
            self.solve(keep)
            return self._fail(
                f"can't put {node_name} on ({x:.2f}, {y:.2f}) - out of reach, or "
                f"only reachable by breaking a limit")

        angles = best[1]
        if apply:
            self.solve(angles)
            self.last_result = Result(True, angles=angles)
        else:
            self.solve(keep)
            self.last_result = Result(True, angles=angles)
        return self.last_result

    def _newton(self, node_name, target, start, tol, max_iter):
        #Helper Function
        """Damped Gauss-Newton on the control angles.  None if it doesn't land."""
        keep = self.angles
        angles = self.clamp(dict(start))
        names = self.control_names
        damping = 1e-3
        step = 1e-5          # degrees, for the finite-difference Jacobian

        try:
            if not self.solve(angles):
                return None
            error = target - np.array(self.position(node_name))

            for _ in range(max_iter):
                if np.hypot(*error) < tol:
                    return dict(angles) if self.within_limits(angles) else None

                jacobian = np.zeros((2, len(names)))
                for i, name in enumerate(names):
                    probe = dict(angles)
                    probe[name] += step
                    if not self.solve(probe):
                        probe[name] -= 2 * step
                        if not self.solve(probe):
                            return None
                        moved = -step
                    else:
                        moved = step
                    jacobian[:, i] = (np.array(self.position(node_name))
                                      - (target - error)) / moved

                normal = jacobian.T @ jacobian + damping * np.eye(len(names))
                try:
                    delta = np.linalg.solve(normal, jacobian.T @ error)
                except np.linalg.LinAlgError:
                    return None

                trial = self.clamp({n: angles[n] + delta[i] for i, n in enumerate(names)})
                if not self.solve(trial):
                    damping *= 10          # smaller, safer step next time round
                    if damping > 1e6:
                        return None
                    continue

                trial_error = target - np.array(self.position(node_name))
                if np.hypot(*trial_error) < np.hypot(*error):
                    angles, error = trial, trial_error
                    damping = max(damping * 0.3, 1e-9)
                else:
                    damping *= 10
                    if damping > 1e6:
                        return None
            return None
        finally:
            self.solve(keep)

    def _seed_angles(self, per_control):
        #Helper Function
        """A spread of starting poses, so the solver can find the other elbow
        configurations as well as the nearest one."""
        ranges = []
        for vector in self.controls:
            lo, hi = vector.limits
            lo = -180.0 if lo is None else lo
            hi = 180.0 if hi is None else hi
            span = hi - lo
            ranges.append([lo + span * (i + 0.5) / per_control for i in range(per_control)])
        seeds = []
        for combo in _combinations(ranges):
            seeds.append(dict(zip(self.control_names, combo)))
        return seeds

    # -- planning -----------------------------------------------------------

    def plan_to(self, node_name, x, y, steps=30, straight=True, verbose=False):
        """A movement taking a node from where it is to a point.
        A movement is a list of angle dicts, one per frame."""
        return self.plan_path(node_name, [(x, y)], steps, straight, verbose)

    def plan_path(self, node_name, points, steps=30, straight=True, verbose=False):
        """A movement taking a node through a series of points in order.

        straight=True walks the node along the straight line between points,
        solving every step.  straight=False interpolates the control angles
        instead: easier on the motors, but the node's path bows.
        """
        keep = self.angles
        frames = []
        skipped, missed = [], []          # points passed over, targets never reached
        here = np.array(self.position(node_name))

        for (tx, ty) in points:
            if straight:
                landed = None
                for i in range(1, steps + 1):
                    t = i / steps
                    step_to = here + (np.array([tx, ty]) - here) * t
                    result = self.move_to(node_name, step_to[0], step_to[1])
                    if not result:
                        skipped.append((float(step_to[0]), float(step_to[1])))
                        if verbose:
                            print(f"  skipped ({step_to[0]:.2f}, {step_to[1]:.2f}) "
                                  f"on the way to ({tx:.2f}, {ty:.2f}) - {result.reason}")
                        continue
                    frames.append(result["angles"])
                    landed = step_to
                if landed is None:
                    missed.append((float(tx), float(ty)))
                    if verbose:
                        print(f"  couldn't move towards ({tx:.2f}, {ty:.2f}) at all")
                else:
                    here = landed
            else:
                result = self.move_to(node_name, tx, ty)
                if not result:
                    missed.append((float(tx), float(ty)))
                    if verbose:
                        print(f"  ({tx:.2f}, {ty:.2f}) skipped - {result.reason}")
                    continue
                target_angles = result["angles"]
                from_angles = frames[-1] if frames else keep
                for i in range(1, steps + 1):
                    t = i / steps
                    frames.append({n: from_angles[n] + (target_angles[n] - from_angles[n]) * t
                                   for n in target_angles})
                here = np.array([tx, ty])

        self.solve(keep)
        if frames:
            self.last_result = Result(True, skipped=skipped, missed=missed)
        else:
            self.last_result = Result(False, f"no part of the path to {points} is "
                                             f"reachable", skipped=skipped, missed=missed)
        return frames

    # -- movements out to file ----------------------------------------------

    def steps_to(self, node_name, x, y, path="steps.csv", steps=30, straight=True):
        """The control-arm angles at every step of a move to one point.

        Returns the movement - a list of {control arm: angle} dicts, one per
        step - and writes the same thing to CSV.  Nothing is printed; check
        robot.last_result if a move came back short.
        """
        movement = self.plan_path(node_name, [(x, y)], steps, straight)
        self.save_movement(movement, path, node_name)
        return movement

    def steps_through(self, node_name, points, path="steps.csv", steps=30, straight=True):
        """The same, for a node travelling through several points in order."""
        movement = self.plan_path(node_name, points, steps, straight)
        self.save_movement(movement, path, node_name, targets=points)
        return movement

    def save_movement(self, movement, path="steps.csv", node_name=None, targets=None):
        """Write a movement to CSV: one row per step, a column per control arm.

        Columns are the step number, each control arm's angle, how far that arm
        turns to get there from the step before (handy when driving steppers),
        and where the node ends up.
        """
        node_name = node_name or self.tool.get("node")
        names = self.control_names
        keep = self.angles
        verdict = self.last_result      # writing a file shouldn't erase why a plan fell short

        header = ["step"] + names + [f"turn_{n}" for n in names]
        if node_name:
            header += [f"{node_name}_x", f"{node_name}_y"]
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
                row += list(self.position(node_name)) if self.solve(frame) else ["", ""]
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
        #Helper Function
        """Which leg of a multi-point path each step belongs to, so the CSV can
        be split up per target."""
        keep = self.angles
        legs, leg = [], 0
        for frame in movement:
            if self.solve(frame) and leg < len(targets):
                x, y = self.position(node_name)
                if math.hypot(x - targets[leg][0], y - targets[leg][1]) < 1e-6:
                    legs.append(leg)
                    leg += 1
                    continue
            legs.append(leg)
        self.solve(keep)
        return legs

    # -- areas --------------------------------------------------------------

    def poses(self, step=None):
        """Every legal combination of control angles, on a grid.

        Walked one control at a time, applying each as it goes, so a limit
        that depends on another arm is always read with that arm in place.
        """
        step = step if step is not None else self.area_settings.get("control_step", 1.0)
        steps = step if isinstance(step, dict) else {n: step for n in self.control_names}
        keep = self.angles
        yield from self._walk(0, steps)
        self._apply(keep)

    def _walk(self, index, steps):
        #Helper Function
        if index == len(self.controls):
            yield self.angles
            return
        vector = self.controls[index]
        lo, hi = vector.limits
        lo = -180.0 if lo is None else lo
        hi = 180.0 if hi is None else hi
        increment = steps.get(vector.name, 1.0)
        angle = lo
        while angle <= hi + TOL:
            vector.set_angle(angle)
            yield from self._walk(index + 1, steps)
            angle += increment
        if angle - increment < hi - TOL:        # always include the top of the range
            vector.set_angle(hi)
            yield from self._walk(index + 1, steps)

    def sweep(self, collect, step=None, verbose=False):
        """Walk every legal pose and pool the points `collect` hands back.

        collect(robot) -> iterable of (x, y), called with the arm solved and
        sitting in that pose.  This is the one place poses are enumerated;
        every area below is the same sweep with a different collector.
        """
        keep = self.angles
        points = []
        solved = skipped = 0
        for angles in self.poses(step):
            if not self.solve(angles):
                skipped += 1
                continue
            solved += 1
            points.extend(collect(self))
        self.solve(keep)
        if verbose:
            note = f" ({skipped} wouldn't close)" if skipped else ""
            print(f"  swept {solved} poses -> {len(points)} points{note}")
        return points

    # collectors ------------------------------------------------------------

    @staticmethod
    def at_node(node_name):
        """Collector: just where one node is."""
        return lambda robot: [robot.position(node_name)]

    @staticmethod
    def bodies(names=None, samples=12, tool_radius=0.0, tool_node=None):
        """Collector: the bars themselves - the arm's physical footprint."""
        def collect(robot):
            chosen = robot.vectors.values() if names is None else \
                [robot.vector(n) for n in names]
            points = []
            for vector in chosen:
                points.extend(vector.body(samples))
            if tool_radius > 0 and tool_node:
                points.extend(_disc(*robot.position(tool_node), tool_radius))
            return points
        return collect

    # ready-made questions --------------------------------------------------

    def reachable_area(self, node_name=None, step=None, cell=None, close=0.8,
                       name=None, color="tab:green", verbose=False):
        """Where a node can be put - the working envelope."""
        node_name = node_name or self.tool.get("node") or list(self.nodes)[-1]
        return Region(self.sweep(self.at_node(node_name), step, verbose),
                      cell=cell or self.area_settings.get("cell", 0.2), close=close,
                      name=name or f"reachable by {node_name}", color=color)

    def swept_area(self, step=None, cell=None, close=0.6, samples=None,
                   tool_radius=None, name=None, color="tab:orange", verbose=False):
        """Every point any part of the arm can occupy - the footprint."""
        collector = self.bodies(
            samples=samples or self.area_settings.get("body_samples", 12),
            tool_radius=self.tool.get("radius", 0.0) if tool_radius is None else tool_radius,
            tool_node=self.tool.get("node"),
        )
        return Region(self.sweep(collector, step, verbose),
                      cell=cell or self.area_settings.get("cell", 0.2), close=close,
                      name=name or "swept by the arm", color=color)

    def exclusion_zone(self, clearance=None, color="tab:red", verbose=False, **kwargs):
        """The swept footprint padded out by a clearance - keep this clear."""
        clearance = self.area_settings.get("clearance", 2.0) if clearance is None else clearance
        swept = self.swept_area(verbose=verbose, **kwargs)
        return swept.grow(clearance, name=f"exclusion zone (+{clearance:g})", color=color)

    def area_map(self, node_name=None, clearance=None, verbose=True):
        """All three areas at once, outermost first, ready to draw."""
        if verbose:
            print("Sweeping the arm...")
        clearance = self.area_settings.get("clearance", 2.0) if clearance is None else clearance
        swept = self.swept_area(verbose=verbose)
        return [swept.grow(clearance, name=f"exclusion zone (+{clearance:g})",
                           color="tab:red"),
                swept,
                self.reachable_area(node_name, verbose=verbose)]

    # -- odds and ends ------------------------------------------------------

    def segments(self):
        """The bars as ((x0, y0), (x1, y1), name, role) - what draw() needs."""
        return [(v.points[0], v.points[1], v.name, v.role) for v in self.vectors.values()]

    def report(self):
        lines = [f"{self.name}"]
        for vector in self.controls:
            lines.append(f"  {vector.name:16s} {vector.angle:8.2f}°   "
                         f"limits {vector.limit_text()}")
        for node in self.nodes.values():
            kind = "fixed" if node.is_fixed else ("attached" if node.is_attached else "free")
            lines.append(f"  {node.name:16s} ({node.x:7.2f}, {node.y:7.2f})   {kind}")
        return "\n".join(lines)

    def __repr__(self):
        return f"<Robot {self.name!r} {len(self.nodes)} nodes, {len(self.vectors)} vectors>"


# ------------------------------------------------------------- small maths --

def _meet(centre_a, radius_a, centre_b, radius_b, branch=1):
    #Helper Function
    """Where two circles cross.  `branch` picks which of the two crossings -
    this is the elbow-up / elbow-down choice.  None if they never meet."""
    (x0, y0), (x1, y1) = centre_a, centre_b
    dx, dy = x1 - x0, y1 - y0
    span = math.hypot(dx, dy)
    slack = EPS * max(1.0, radius_a + radius_b)   # full stretch is legal; float noise isn't
    if span == 0 or span > radius_a + radius_b + slack or span < abs(radius_a - radius_b) - slack:
        return None
    along = (radius_a ** 2 - radius_b ** 2 + span ** 2) / (2 * span)
    across = math.sqrt(max(radius_a ** 2 - along ** 2, 0.0))
    mx, my = x0 + along * dx / span, y0 + along * dy / span
    return (mx + branch * across * (-dy) / span, my + branch * across * dx / span)


def _disc(cx, cy, radius, steps=16):
    #Helper Function
    return [(cx + radius * math.cos(2 * math.pi * i / steps),
             cy + radius * math.sin(2 * math.pi * i / steps)) for i in range(steps)]


def _combinations(lists):
    #Helper Function
    """Every combination, one item from each list.  Saves importing itertools."""
    if not lists:
        yield ()
        return
    for item in lists[0]:
        for rest in _combinations(lists[1:]):
            yield (item,) + rest


# -------------------------------------------------------------- movements --

# A movement is any iterable of angle dicts - {control name: angle} - one per
# frame.  animate() just plays it.  Robot.plan_to / plan_path build the ones
# that need solving; these build the ones that don't.

def hold(angles, frames=10):
    """Stay put for a few frames."""
    return [dict(angles) for _ in range(frames)]


def angle_move(start, end, steps=45):
    """Drive the control arms from one set of angles to another."""
    frames = []
    for i in range(steps + 1):
        t = i / steps
        frames.append({name: start[name] + (end.get(name, start[name]) - start[name]) * t
                       for name in start})
    return frames


def chain(*movements):
    """Run several movements back to back as one."""
    frames = []
    for movement in movements:
        frames.extend(movement)
    return frames


def clamped(robot, movement):
    """Pull every frame of a movement inside the control-arm limits."""
    return [robot.clamp(frame) for frame in movement]


# ----------------------------------------------------------------- output --

# draw() and animate() take a robot and a movement, not loose vectors, so they
# work unchanged whatever linkage the config describes.

def _apply_view(ax, robot, view=None):
    #Helper Function
    view = view if view is not None else robot.view
    ax.set_aspect("equal")
    ax.grid(True)
    if view.get("xlim"):
        ax.set_xlim(*view["xlim"])
    if view.get("ylim"):
        ax.set_ylim(*view["ylim"])


def draw(robot, ax=None, regions=None, targets=None, labels=True, title=None,
         view=None, show=True):
    """One still picture of the arm as it stands."""
    ax = ax if ax else plt.gca()

    for i, region in enumerate(regions or []):
        region.plot(ax=ax, zorder=i)

    if targets:
        ax.plot([p[0] for p in targets], [p[1] for p in targets], "x",
                color="0.3", markersize=9, linestyle=":", linewidth=1,
                label="targets", zorder=5)

    # bars are coloured by role, and deliberately kept off the colours the
    # area regions use, so the arm stays readable on top of them
    for (x0, y0), (x1, y1), name, role in robot.segments():
        ax.plot([x0, x1], [y0, y1], marker="o", zorder=6,
                color="tab:blue" if role == "control" else "0.25",
                linewidth=2.5 if role == "control" else 1.8)
        if labels:
            ax.text((x0 + x1) / 2, (y0 + y1) / 2 + 0.3, name, ha="center", zorder=7)

    for node in robot.nodes.values():
        ax.plot([node.x], [node.y], "s" if node.is_fixed else "o",
                color="black" if node.is_fixed else "0.35",
                markersize=7 if node.is_fixed else 4, zorder=8)
        if labels:
            ax.text(node.x, node.y - 0.9, f"{node.name}\n({node.x:.1f}, {node.y:.1f})",
                    ha="center", fontsize=7, color="0.3", zorder=8)

    _apply_view(ax, robot, view)
    ax.set_title(title if title else _pose_title(robot))
    if regions or targets:
        ax.legend(loc="best", fontsize=8, framealpha=0.85)
    if show:
        plt.show()
    return ax


def _pose_title(robot):
    #Helper Function
    drives = "    ".join(f"{v.name}: {v.angle:.1f}° [{v.limit_text()}]"
                         for v in robot.controls)
    tip = robot.tool.get("node")
    if tip:
        x, y = robot.position(tip)
        return f"{drives}\n{tip}: ({x:.1f}, {y:.1f})"
    return drives


def animate(robot, movement, pause=0.05, regions=None, targets=None, trace=None,
            on_limit="skip", view=None, labels=True, keep_open=True):
    """Play a movement - any iterable of angle dicts.

    regions  : Region objects (exclusion zone, swept area, ...) drawn underneath
    trace    : name of a node to leave a trail behind, or None
    on_limit : "skip" (default, reports and moves on), "clamp" (pull the frame
               to the nearest legal angles) or "ignore" (drive it anyway)
    """
    trace = trace if trace is not None else robot.tool.get("node")
    trail_x, trail_y = [], []

    for frame in movement:
        if on_limit == "clamp":
            frame = robot.clamp(frame)
        result = robot.move_all(frame, enforce_limits=(on_limit != "ignore"))
        if not result:
            print(f"  frame skipped - {result.reason}")
            continue

        if trace:
            x, y = robot.position(trace)
            trail_x.append(x)
            trail_y.append(y)

        plt.clf()
        ax = plt.gca()
        if trail_x and len(trail_x) > 1:
            ax.plot(trail_x, trail_y, "-", color="0.7", linewidth=1, zorder=4)
        draw(robot, ax=ax, regions=regions, targets=targets, labels=labels,
             view=view, show=False)
        plt.pause(pause)

    if keep_open:
        plt.show()


def show_areas(robot, regions=None, targets=None, node_name=None, clearance=None,
               title=None, verbose=True):
    """The arm's areas of movement in one picture, with the arm on top."""
    regions = regions if regions is not None else \
        robot.area_map(node_name, clearance, verbose=verbose)
    if verbose:
        for region in regions:
            print(f"  {region.name:28s} {region.area:8.1f} square units")
    draw(robot, regions=regions, targets=targets, labels=False,
         title=title or f"{robot.name} - areas of movement",
         view={}, show=True)
    return regions


# -------------------------------------------------------------------- demo --

if __name__ == "__main__":

    robot = Robot.from_file("arm_config.json")
    print(robot.report())

    # --- driving it ---------------------------------------------------------
    print("\nDriving the control arms:")
    for name, angle in [("control_arm_a", 120), ("control_arm_b", 40),
                        ("control_arm_a", 200), ("control_arm_b", 150)]:
        result = robot.move(name, angle)
        if result:
            print(f"  {name} -> {angle}°   ok, "
                  f"node_end now ({robot.position('node_end')[0]:.2f}, "
                  f"{robot.position('node_end')[1]:.2f})")
        else:
            print(f"  {name} -> {angle}°   refused: {result.reason}")

    # --- inverse kinematics -------------------------------------------------
    print("\nPutting a node on a point:")
    for point in [(-14.0, 3.0), (-6.0, -3.0), (5.0, 5.0)]:
        result = robot.move_to("node_end", *point)
        if result:
            angles = ", ".join(f"{n} {a:.2f}°" for n, a in result["angles"].items())
            print(f"  node_end -> {str(point):>14s}   {angles}")
        else:
            print(f"  node_end -> {str(point):>14s}   refused: {result.reason}")

    # any node, not just the end one
    result = robot.move_to("joint", -3.0, 6.0)
    print(f"  joint    -> ( -3.00, 6.00)   "
          + ("ok" if result else f"refused: {result.reason}"))

    # --- areas of movement --------------------------------------------------
    robot.move_all({"control_arm_a": 90.0, "control_arm_b": 0.0})
    targets = [(-14.0, 3.0), (-6.0, 3.0), (-6.0, -3.0), (-14.0, -3.0), (-14.0, 3.0)]

    print("\nAreas of movement:")
    zones = robot.area_map()
    for region in zones:
        print(f"  {region.name:28s} {region.area:8.1f} square units")
    zones[0].save_perimeter("exclusion_zone.csv")
    print("  perimeter written to exclusion_zone.csv")

    reachable = zones[-1]
    for point in targets[:-1]:
        print(f"  target {str(point):>14s}  reachable: {reachable.contains(*point)}")

    show_areas(robot, regions=zones, targets=targets, verbose=False)

    # --- the step-by-step angles, as a file rather than a wall of numbers ----
    robot.move_all({"control_arm_a": 90.0, "control_arm_b": 0.0})
    single = robot.steps_to("node_end", -14.0, 3.0, "steps_to_target.csv", steps=50)
    print(f"\n{len(single)} steps to (-14, 3) written to steps_to_target.csv")

    robot.move_all({"control_arm_a": 90.0, "control_arm_b": 0.0})
    movement = robot.steps_through("node_end", targets, "steps_full_path.csv", steps=50)
    print(f"{len(movement)} steps around the full path written to steps_full_path.csv")
    if robot.last_result.get("skipped"):
        print(f"  ({len(robot.last_result['skipped'])} steps were out of reach)")

    # --- run it -------------------------------------------------------------
    print("\nRunning the path...")
    animate(robot, movement, targets=targets, pause=0.03, keep_open=False)

    # other things the same pieces will do:
    #
    #   robot.plan_path("node_end", targets, straight=False)   ease the motors
    #   robot.swept_area(tool_radius=1.5)                      allow for a tool
    #   robot.reachable_area("joint")                          any node
    #   robot.sweep(Robot.bodies(["control_arm_a"]))           one bar only
    #   animate(robot, angle_move(robot.angles, {"control_arm_a": 160},
    #                             steps=30), on_limit="clamp")
    #   zones[0].contains(bench_x, bench_y)                    is it in the way?

# source .venv/bin/activate
# python v5.py