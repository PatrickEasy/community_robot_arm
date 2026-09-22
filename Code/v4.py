"""
Five-bar arm simulation - v4

Same geometry as mainv3.py, with three additions:

1.  Angle limits on the control arms.  Each control arm has a min and a max,
    and a limit can be a fixed number or a callable, so one arm's limit can
    depend on the rest of the arm.  Nothing that breaks a limit is ever
    driven, drawn or returned by the solver.

2.  Inverse kinematics.  Give it a list of (x, y) points and it works out the
    A_base / B_base angles needed to put the far end of E_end on each point.

3.  animate() now takes a "movement" instead of four hard-coded angles.
    A movement is simply any iterable of (a_angle, b_angle) pairs - one pair
    per frame.  Several builders are provided (angle_move, point_move,
    path_through_points, hold, chain) and you can write your own.

Only A_base and B_base are ever driven.  Everything else is solved.

    source .venv/bin/activate
    python mainv4.py
"""

import math
import numpy as np
import matplotlib.pyplot as plt

TOL = 1e-9   # slack when comparing against a limit, so 45.0 counts as >= 45


# ---------------------------------------------------------------- geometry --

class Vector:
    def __init__(self, x=0, y=0, length=10, angle=0, parent=None, controllable=True,
                 name="Vector", min_angle=None, max_angle=None):
        self._x = x          # only used if there is no parent
        self._y = y
        self.length = length
        self._angle = angle  # degrees
        self.parent = parent
        self.controllable = controllable
        self.name = name  # for debugging
        # limits: a number, None for "no limit", or a callable taking no
        # arguments and returning a number (worked out fresh each time)
        self.min_angle = min_angle
        self.max_angle = max_angle

    @property
    def angle(self):
        return self._angle

    @angle.setter
    def angle(self, value):
        if not self.controllable:
            raise Exception("This vector is constrained, its angle can't be set directly")
        self._angle = value

    # ------------------------------------------------------------ limits --

    @property
    def limits(self):
        """The (min, max) this arm is allowed to sit at right now.
        Either can be None, meaning unlimited."""
        lo = self.min_angle() if callable(self.min_angle) else self.min_angle
        hi = self.max_angle() if callable(self.max_angle) else self.max_angle
        return lo, hi

    def in_limits(self, angle=None):
        angle = self.angle if angle is None else angle
        lo, hi = self.limits
        if lo is not None and angle < lo - TOL:
            return False
        if hi is not None and angle > hi + TOL:
            return False
        return True

    def clamp(self, angle):
        """The nearest angle this arm is actually allowed to sit at."""
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
        return f"{lo}° to {hi}°"

    # ------------------------------------------------------------ points --

    @property
    def x(self):
        return self.parent.end_x if self.parent else self._x

    @x.setter
    def x(self, value):
        self._x = value

    @property
    def y(self):
        return self.parent.end_y if self.parent else self._y

    @y.setter
    def y(self, value):
        self._y = value

    @property
    def end_x(self):
        return self.x + self.length * math.cos(math.radians(self.angle))

    @property
    def end_y(self):
        return self.y + self.length * math.sin(math.radians(self.angle))


def solve(C, D, E):
    """Set the angles of C and D so their ends meet at the same point,
    then set E from C. Returns False if C and D can't reach each other."""
    x1, y1 = C.x, C.y      # start of C (tip of its parent)
    x2, y2 = D.x, D.y      # start of D (tip of its parent)
    r1, r2 = C.length, D.length

    dx = x2 - x1
    dy = y2 - y1
    d = math.sqrt(dx**2 + dy**2)

    if d > r1 + r2 or d < abs(r1 - r2) or d == 0:
        return False

    a = (r1**2 - r2**2 + d**2) / (2 * d)
    h = math.sqrt(max(r1**2 - a**2, 0.0))

    # midpoint along the line between the two starts
    mx = x1 + a * dx / d
    my = y1 + a * dy / d

    # meeting point (use "-" instead of "+" for the other elbow direction)
    px = mx + h * (-dy) / d
    py = my + h * dx / d

    C._angle = math.degrees(math.atan2(py - y1, px - x1))
    D._angle = math.degrees(math.atan2(py - y2, px - x2))
    E._angle = C._angle - 180
    return True


# --------------------------------------------------------------- limits ----

def limit_breaches(*vectors):
    """Which of these arms are currently outside their limits, described."""
    out = []
    for v in vectors:
        if not v.in_limits():
            out.append(f"{v.name} at {v.angle:.1f}° (allowed {v.limit_text()})")
    return out


def within_limits(A, B, a_angle, b_angle):
    """Would this pair of control angles be legal?
    A is checked first, because B's limits may depend on where A is."""
    a_keep, b_keep = A.angle, B.angle
    A.angle, B.angle = a_angle, b_angle
    ok = A.in_limits() and B.in_limits()
    A.angle, B.angle = a_keep, b_keep
    return ok


def clamp_pair(A, B, a_angle, b_angle):
    """The nearest legal pair. A is clamped first so that B's limits,
    which may depend on A, are worked out from the clamped A."""
    a_keep, b_keep = A.angle, B.angle
    A.angle = A.clamp(a_angle)
    b_clamped = B.clamp(b_angle)
    result = (A.angle, b_clamped)
    A.angle, B.angle = a_keep, b_keep
    return result


# --------------------------------------------------------- forward helpers --

def forward(A, B, C, D, E, a_angle, b_angle, enforce_limits=True):
    """Drive the two control arms and solve the rest.
    Returns (end_x, end_y) of E_end, or None if the angles break a limit
    or the linkage can't close."""
    if enforce_limits and not within_limits(A, B, a_angle, b_angle):
        return None
    A.angle = a_angle
    B.angle = b_angle
    if not solve(C, D, E):
        return None
    return (E.end_x, E.end_y)


def end_point(A, B, C, D, E):
    """Where the far end of E_end currently is, for the angles already set."""
    return forward(A, B, C, D, E, A.angle, B.angle)


# --------------------------------------------------------------------- IK ---

def _circle_intersections(x0, y0, r0, x1, y1, r1):
    """The 0, 1 or 2 points where two circles cross."""
    dx, dy = x1 - x0, y1 - y0
    d = math.hypot(dx, dy)
    if d == 0 or d > r0 + r1 or d < abs(r0 - r1):
        return []

    a = (r0**2 - r1**2 + d**2) / (2 * d)
    h = math.sqrt(max(r0**2 - a**2, 0.0))
    mx, my = x0 + a * dx / d, y0 + a * dy / d

    if h < 1e-12:
        return [(mx, my)]
    return [
        (mx + h * (-dy) / d, my + h * dx / d),
        (mx - h * (-dy) / d, my - h * dx / d),
    ]


def _nearest_equivalent(angle, reference, vector=None):
    """Same angle, shifted by whole turns to sit as close to `reference` as
    possible - this keeps the control arms from spinning the long way round
    between frames. The shift is dropped if it would leave the limits."""
    if reference is None:
        return angle
    shifted = angle + 360.0 * round((reference - angle) / 360.0)
    if vector is not None and not vector.in_limits(shifted):
        return angle
    return shifted


def inverse(A, B, C, D, E, target_x, target_y, prefer=None, tol=1e-6, enforce_limits=True):
    """Work out the (a_angle, b_angle) that put the far end of E_end on
    (target_x, target_y).  Returns None if the point is out of reach or can
    only be reached by breaking a control-arm limit.

    prefer: an (a_angle, b_angle) pair - when more than one solution exists the
    closest one to this is returned, which keeps a path continuous.

    How it works.  E_end starts at A_base's tip and has a fixed length, so
    A_base's tip must sit exactly E.length away from the target: intersect the
    circle A_base sweeps with a circle of radius E.length around the target.
    E_end points opposite to C_elbow, so the C/D meeting point M follows
    directly from A_base's tip.  B_base's tip must then sit D.length away from
    M: a second circle intersection.  Up to four combinations come out, so each
    is fed back through solve() and only the ones that genuinely land on the
    target - and stay inside the limits - are kept.
    """
    candidates = []

    # 1. where A_base's tip has to be
    for ax, ay in _circle_intersections(A.x, A.y, A.length, target_x, target_y, E.length):
        # 2. the C/D meeting point implied by that tip
        #    E_end runs from the tip away from M, so M = tip + C.length * u
        #    where u is the unit vector from the target towards the tip
        ux = (ax - target_x) / E.length
        uy = (ay - target_y) / E.length
        mx = ax + C.length * ux
        my = ay + C.length * uy

        a_angle = math.degrees(math.atan2(ay - A.y, ax - A.x))

        # 3. where B_base's tip has to be so D_elbow reaches M
        for bx, by in _circle_intersections(B.x, B.y, B.length, mx, my, D.length):
            b_angle = math.degrees(math.atan2(by - B.y, bx - B.x))
            candidates.append((a_angle, b_angle))

    if not candidates:
        return None

    # 4. keep only the candidates that survive the forward solve (this is what
    #    picks the elbow branch solve() actually uses) and obey the limits
    a_keep, b_keep = A.angle, B.angle
    valid = []
    for a_angle, b_angle in candidates:
        got = forward(A, B, C, D, E, a_angle, b_angle, enforce_limits=enforce_limits)
        if got is None:
            continue
        if math.hypot(got[0] - target_x, got[1] - target_y) < max(tol, 1e-6 * E.length):
            valid.append((a_angle, b_angle))
    A.angle, B.angle = a_keep, b_keep
    solve(C, D, E)

    if not valid:
        return None

    # 5. pick the one closest to where we already are
    ref_a, ref_b = prefer if prefer is not None else (a_keep, b_keep)
    scored = []
    for a_angle, b_angle in valid:
        a_eq = _nearest_equivalent(a_angle, ref_a, A if enforce_limits else None)
        b_eq = _nearest_equivalent(b_angle, ref_b, B if enforce_limits else None)
        cost = abs(a_eq - ref_a) + abs(b_eq - ref_b)
        scored.append((cost, a_eq, b_eq))
    scored.sort(key=lambda s: s[0])
    return scored[0][1], scored[0][2]


def solve_points(A, B, C, D, E, points, prefer=None, verbose=True):
    """Run inverse() over a list of (x, y) points.
    Returns a list of (point, angles) where angles is None if unreachable."""
    results = []
    ref = prefer if prefer is not None else (A.angle, B.angle)
    for (px, py) in points:
        angles = inverse(A, B, C, D, E, px, py, prefer=ref)
        if angles is None:
            if verbose:
                print(f"({px:6.2f}, {py:6.2f})  ->  out of reach (or outside the control-arm limits)")
        else:
            ref = angles
            if verbose:
                print(f"({px:6.2f}, {py:6.2f})  ->  A {angles[0]:7.2f}°   B {angles[1]:7.2f}°")
        results.append(((px, py), angles))
    return results


def workspace_grid(A, B, C, D, E, a_step=2.0, b_step=2.0):
    """Every end point the arm can reach without breaking a limit.
    Handy for choosing target points, and for drawing the reachable area."""
    a_keep, b_keep = A.angle, B.angle
    points = []
    a_lo, a_hi = A.limits
    a_lo = -180.0 if a_lo is None else a_lo
    a_hi = 180.0 if a_hi is None else a_hi

    a = a_lo
    while a <= a_hi + 1e-9:
        A.angle = a
        b_lo, b_hi = B.limits          # B's limits can depend on A
        b_lo = -180.0 if b_lo is None else b_lo
        b_hi = 180.0 if b_hi is None else b_hi
        b = b_lo
        while b <= b_hi + 1e-9:
            p = forward(A, B, C, D, E, a, b)
            if p:
                points.append(p)
            b += b_step
        a += a_step

    A.angle, B.angle = a_keep, b_keep
    solve(C, D, E)
    return points


# -------------------------------------------------------------- movements --

# A "movement" is any iterable of (a_angle, b_angle) pairs, one pair per frame.
# animate() just walks it.  These builders cover the common cases; writing your
# own is a matter of yielding angle pairs.

def hold(a_angle, b_angle, frames=10):
    """Stay put for a few frames."""
    for _ in range(frames):
        yield (a_angle, b_angle)


def angle_move(a_start, a_end, b_start, b_end, steps=45):
    """The old behaviour: drive both control arms from one angle to another.
    Limits aren't applied here - animate() handles that, or wrap this in
    clamped(A, B, ...) to pull it inside the limits first."""
    for i in range(steps + 1):
        t = i / steps
        yield (a_start + (a_end - a_start) * t, b_start + (b_end - b_start) * t)


def clamped(A, B, movement):
    """Pull every frame of a movement inside the control-arm limits."""
    for (a_angle, b_angle) in movement:
        yield clamp_pair(A, B, a_angle, b_angle)


def point_move(A, B, C, D, E, target, start_angles=None, steps=30, straight=True):
    """Move the end of E_end from where it is now to a single (x, y) point.

    straight=True  - the end travels in a straight line (each frame is solved)
    straight=False - the control-arm angles are interpolated instead, which is
                     smoother for the motors but bows the end's path
    """
    yield from path_through_points(
        A, B, C, D, E, [target],
        start_angles=start_angles, steps_per_segment=steps, straight=straight,
    )


def path_through_points(A, B, C, D, E, points, start_angles=None,
                        steps_per_segment=30, straight=True, verbose=True):
    """Move the end of E_end through a series of (x, y) points in order.

    Starts from wherever the arm currently is (or from start_angles).
    Yields (a_angle, b_angle) pairs - every one already inside the limits.
    Anything unreachable is skipped with a note rather than stopping the path.
    """
    a_keep, b_keep = A.angle, B.angle
    if start_angles is not None:
        A.angle, B.angle = start_angles
        solve(C, D, E)

    breaches = limit_breaches(A, B)
    if breaches:
        A.angle, B.angle = a_keep, b_keep
        solve(C, D, E)
        raise ValueError("A path can't start outside the limits: " + "; ".join(breaches))

    current = end_point(A, B, C, D, E)
    if current is None:
        A.angle, B.angle = a_keep, b_keep
        solve(C, D, E)
        raise ValueError("The arm isn't in a solvable pose, so a path can't start from here")

    frames = []
    ref = (A.angle, B.angle)
    cx, cy = current

    for (tx, ty) in points:
        if straight:
            # walk along the straight line to the target, solving each step
            reached = None
            for i in range(1, steps_per_segment + 1):
                t = i / steps_per_segment
                ix = cx + (tx - cx) * t
                iy = cy + (ty - cy) * t
                angles = inverse(A, B, C, D, E, ix, iy, prefer=ref)
                if angles is None:
                    if verbose:
                        print(f"Skipping ({ix:.2f}, {iy:.2f}) on the way to ({tx:.2f}, {ty:.2f}) - "
                              f"out of reach or outside the limits")
                    continue
                ref = angles
                reached = (ix, iy)
                frames.append(angles)
            if reached is None:
                if verbose:
                    print(f"Couldn't move towards ({tx:.2f}, {ty:.2f}) at all - staying put")
            else:
                cx, cy = reached
        else:
            angles = inverse(A, B, C, D, E, tx, ty, prefer=ref)
            if angles is None:
                if verbose:
                    print(f"({tx:.2f}, {ty:.2f}) is out of reach or outside the limits - skipping")
                continue
            a0, b0 = ref
            a1, b1 = angles
            for i in range(1, steps_per_segment + 1):
                t = i / steps_per_segment
                frames.append((a0 + (a1 - a0) * t, b0 + (b1 - b0) * t))
            ref = angles
            cx, cy = tx, ty

    A.angle, B.angle = a_keep, b_keep
    solve(C, D, E)
    yield from frames


def chain(*movements):
    """Run several movements back to back as one."""
    for movement in movements:
        yield from movement


# ------------------------------------------------------------------ areas --
#
# One sweep engine, used for two different jobs:
#
#   * the reachable area  - where the far end of E_end can get to
#   * the swept area      - every bit of floor any part of the arm passes over
#
# sweep() walks every legal control-arm combination and pools whatever points
# a "collector" hands back.  Swap the collector and you change the question:
#
#   sweep(..., end_only)            -> the working envelope
#   sweep(..., whole_arm())         -> the physical footprint, i.e. the zone
#                                      that has to be kept clear
#   sweep(..., links("A_base"))     -> just one link, e.g. to check one motor
#
# The pooled points become a Region: an occupancy grid with an outline, an
# area, a contains() test and grow(), which pads it out by a clearance
# distance to give the safe perimeter.


def sample_configurations(A, B, a_step=2.0, b_step=2.0):
    """Every legal (a_angle, b_angle) pair on a grid, limits included.
    B's range is worked out fresh for each A, so limits that depend on the
    other arm (like B's ceiling tracking A) are handled correctly."""
    a_lo, a_hi = A.limits
    a_lo = -180.0 if a_lo is None else a_lo
    a_hi = 180.0 if a_hi is None else a_hi

    a_keep, b_keep = A.angle, B.angle
    a = a_lo
    while a <= a_hi + TOL:
        A.angle = a
        b_lo, b_hi = B.limits
        b_lo = -180.0 if b_lo is None else b_lo
        b_hi = 180.0 if b_hi is None else b_hi
        b = b_lo
        while b <= b_hi + TOL:
            yield (a, b)
            b += b_step
        # make sure the very top of the range is included
        if b - b_step < b_hi - TOL:
            yield (a, b_hi)
        a += a_step
    if a - a_step < a_hi - TOL:
        A.angle = a_hi
        b_lo, b_hi = B.limits
        b_lo = -180.0 if b_lo is None else b_lo
        b_hi = 180.0 if b_hi is None else b_hi
        b = b_lo
        while b <= b_hi + TOL:
            yield (a_hi, b)
            b += b_step
    A.angle, B.angle = a_keep, b_keep


# --- collectors: given a solved pose, hand back the points that matter ------

def end_only(A, B, C, D, E):
    """Just the working point - the far end of E_end."""
    return [(E.end_x, E.end_y)]


def links(*names, samples=8):
    """The bodies of the named links only, e.g. links("A_base", "E_end")."""
    wanted = set(names)

    def collector(A, B, C, D, E):
        points = []
        for v in (A, B, C, D, E):
            if v.name in wanted:
                points.extend(_along(v, samples))
        return points
    return collector


def whole_arm(samples=8, include_end_circle=0.0):
    """Every link body - this is the arm's physical footprint.

    include_end_circle: if the tool on the end of E_end sticks out, give its
    radius and a disc of that size is added at the end point."""
    def collector(A, B, C, D, E):
        points = []
        for v in (A, B, C, D, E):
            points.extend(_along(v, samples))
        if include_end_circle > 0:
            points.extend(_disc(E.end_x, E.end_y, include_end_circle))
        return points
    return collector


def _along(v, samples=8):
    """Points spread along a link, base to tip."""
    return [
        (v.x + (v.end_x - v.x) * i / samples,
         v.y + (v.end_y - v.y) * i / samples)
        for i in range(samples + 1)
    ]


def _disc(cx, cy, radius, steps=16):
    return [(cx + radius * math.cos(2 * math.pi * i / steps),
             cy + radius * math.sin(2 * math.pi * i / steps)) for i in range(steps)]


def sweep(A, B, C, D, E, collect=end_only, a_step=2.0, b_step=2.0, verbose=False):
    """Walk every legal pose and pool the points `collect` returns.

    collect(A, B, C, D, E) -> iterable of (x, y), called with the arm already
    solved and sitting in that pose.  This is the one place the poses are
    enumerated - everything else is a collector."""
    a_keep, b_keep = A.angle, B.angle
    points = []
    poses = 0
    for (a_angle, b_angle) in sample_configurations(A, B, a_step, b_step):
        A.angle, B.angle = a_angle, b_angle
        if not solve(C, D, E):
            continue          # legal angles, but the linkage won't close
        poses += 1
        points.extend(collect(A, B, C, D, E))
    A.angle, B.angle = a_keep, b_keep
    solve(C, D, E)
    if verbose:
        print(f"swept {poses} poses -> {len(points)} points")
    return points


# --- turning a point cloud into an area ------------------------------------

def _disc_offsets(r_cells):
    return [(dy, dx)
            for dy in range(-r_cells, r_cells + 1)
            for dx in range(-r_cells, r_cells + 1)
            if dx * dx + dy * dy <= r_cells * r_cells]


def _spread(grid, r_cells, fill):
    """Grid OR-ed with itself shifted over a disc of r_cells.
    The result is padded by r_cells on every side."""
    padded = np.pad(grid, r_cells, constant_values=fill)
    out = np.zeros_like(padded)
    for dy, dx in _disc_offsets(r_cells):
        out |= np.roll(np.roll(padded, dy, axis=0), dx, axis=1)
    return out


def _dilate(grid, r_cells):
    if r_cells <= 0:
        return grid.copy()
    return _spread(grid, r_cells, False)              # grows by r_cells a side


def _erode(grid, r_cells):
    if r_cells <= 0:
        return grid.copy()
    return ~_spread(~grid, r_cells, True)[r_cells:-r_cells, r_cells:-r_cells]


class Region:
    """An area built from a cloud of points.

    Internally it's an occupancy grid, which is what makes the awkward bits
    easy: concave shapes, holes in the middle, and growing the whole thing
    outwards by a clearance distance for a safety perimeter."""

    def __init__(self, points, cell=0.2, close=0.6, name="region", color="tab:blue"):
        pts = np.asarray(list(points), dtype=float)
        if pts.size == 0:
            raise ValueError("No points to build a region from")

        self.cell = cell
        self.name = name
        self.color = color
        self._polys = None

        pad = max(3, int(round(close / cell)) + 2)
        self.x0 = pts[:, 0].min() - pad * cell
        self.y0 = pts[:, 1].min() - pad * cell
        nx = int(math.ceil((pts[:, 0].max() - self.x0) / cell)) + pad + 1
        ny = int(math.ceil((pts[:, 1].max() - self.y0) / cell)) + pad + 1

        grid = np.zeros((ny, nx), dtype=bool)
        ix = np.rint((pts[:, 0] - self.x0) / cell).astype(int)
        iy = np.rint((pts[:, 1] - self.y0) / cell).astype(int)
        grid[iy, ix] = True

        # close up the speckle left by sampling at discrete angles, without
        # filling in holes that are genuinely there
        r = int(round(close / cell))
        if r > 0:
            grid = _erode(_dilate(grid, r), r)   # array grows by r a side...
            self.x0 -= r * cell                  # ...so the origin moves with it
            self.y0 -= r * cell
        self.grid = grid

    # -- construction -------------------------------------------------------

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
        """A copy padded outwards by `clearance` - the safe perimeter.
        Everything within `clearance` of the original area is included."""
        r = int(round(clearance / self.cell))
        grown = _dilate(self.grid, r)
        return Region._from_grid(
            grown, self.x0 - r * self.cell, self.y0 - r * self.cell, self.cell,
            name if name else f"{self.name} + {clearance:g} clearance",
            color if color else self.color,
        )

    # -- what you can ask it ------------------------------------------------

    @property
    def area(self):
        """Square units covered."""
        return float(self.grid.sum()) * self.cell ** 2

    @property
    def bounds(self):
        """(min_x, min_y, max_x, max_y) of the occupied cells."""
        ys, xs = np.nonzero(self.grid)
        return (self.x0 + xs.min() * self.cell, self.y0 + ys.min() * self.cell,
                self.x0 + xs.max() * self.cell, self.y0 + ys.max() * self.cell)

    def contains(self, x, y):
        """Is this point inside the area? Use it to check a target, or to
        check whether something in the workshop is in the way."""
        ix = int(round((x - self.x0) / self.cell))
        iy = int(round((y - self.y0) / self.cell))
        if not (0 <= ix < self.grid.shape[1] and 0 <= iy < self.grid.shape[0]):
            return False
        return bool(self.grid[iy, ix])

    def polygons(self):
        """The outline as a list of closed (x, y) loops - outer edges and any
        holes. This is the thing to export if you want to mark it out."""
        if self._polys is None:
            xs = self.x0 + np.arange(self.grid.shape[1]) * self.cell
            ys = self.y0 + np.arange(self.grid.shape[0]) * self.cell
            fig = plt.figure()
            try:
                cs = fig.add_subplot(111).contour(
                    xs, ys, self.grid.astype(float), levels=[0.5])
                self._polys = [np.asarray(seg) for seg in cs.allsegs[0] if len(seg) > 2]
            finally:
                plt.close(fig)
        return self._polys

    def save_perimeter(self, path):
        """Write the outline to CSV (loop, x, y) - handy for CAD or for
        marking the zone out on the bench."""
        with open(path, "w") as f:
            f.write("loop,x,y\n")
            for i, poly in enumerate(self.polygons()):
                for (x, y) in poly:
                    f.write(f"{i},{x:.4f},{y:.4f}\n")
        return path

    # -- drawing ------------------------------------------------------------

    def plot(self, ax=None, fill=True, alpha=0.18, edge=True, linewidth=1.6,
             linestyle="-", label=None, zorder=0):
        ax = ax if ax else plt.gca()
        label = self.name if label is None else label
        if fill:
            xs = self.x0 + np.arange(self.grid.shape[1]) * self.cell
            ys = self.y0 + np.arange(self.grid.shape[0]) * self.cell
            ax.contourf(xs, ys, self.grid.astype(float), levels=[0.5, 1.5],
                        colors=[self.color], alpha=alpha, zorder=zorder)
        first = True
        for poly in self.polygons():
            ax.plot(poly[:, 0], poly[:, 1],
                    color=self.color, linewidth=linewidth if edge else 0,
                    linestyle=linestyle, zorder=zorder + 0.1,
                    label=label if first else None)
            first = False
        return ax

    def __repr__(self):
        b = self.bounds
        return (f"<Region {self.name!r} area={self.area:.1f} "
                f"bounds=({b[0]:.1f}, {b[1]:.1f})..({b[2]:.1f}, {b[3]:.1f})>")


# --- the two ready-made questions ------------------------------------------

def reachable_region(A, B, C, D, E, a_step=1.0, b_step=1.0, cell=0.2, close=0.8,
                     name="reachable by E_end", color="tab:green", verbose=False):
    """Where the far end of E_end can be put."""
    return Region(sweep(A, B, C, D, E, end_only, a_step, b_step, verbose),
                  cell=cell, close=close, name=name, color=color)


def swept_region(A, B, C, D, E, a_step=1.0, b_step=1.0, cell=0.2, close=0.6,
                 samples=10, tool_radius=0.0,
                 name="swept by the arm", color="tab:orange", verbose=False):
    """Every point any part of the arm can occupy - the physical footprint."""
    collector = whole_arm(samples=samples, include_end_circle=tool_radius)
    return Region(sweep(A, B, C, D, E, collector, a_step, b_step, verbose),
                  cell=cell, close=close, name=name, color=color)


def exclusion_zone(A, B, C, D, E, clearance=2.0, name=None, color="tab:red", **kwargs):
    """The swept area padded out by a clearance - keep this clear of people,
    fixtures and cables."""
    swept = swept_region(A, B, C, D, E, **kwargs)
    return swept.grow(clearance,
                      name=name if name else f"exclusion zone (+{clearance:g})",
                      color=color)


def show_areas(A, B, C, D, E, regions=None, clearance=2.0, draw_arm=True,
               targets=None, title="Arm working areas", verbose=True):
    """One picture of the lot: exclusion zone, swept footprint, reachable area,
    and the arm in its current pose."""
    if regions is None:
        if verbose:
            print("Sweeping the arm...")
        swept = swept_region(A, B, C, D, E, verbose=verbose)
        regions = [
            swept.grow(clearance, name=f"exclusion zone (+{clearance:g})", color="tab:red"),
            swept,
            reachable_region(A, B, C, D, E, verbose=verbose),
        ]

    ax = plt.gca()
    for i, region in enumerate(regions):
        region.plot(ax=ax, zorder=i)
        if verbose:
            print(f"  {region.name}: {region.area:.1f} square units")

    if targets:
        ax.plot([p[0] for p in targets], [p[1] for p in targets], "x",
                color="0.3", markersize=9, label="targets", zorder=5)

    if draw_arm:
        for v in (A, B, C, D, E):
            ax.plot([v.x, v.end_x], [v.y, v.end_y], marker="o", color="0.15",
                    linewidth=2, zorder=6)
        ax.plot([A.x], [A.y], "s", color="black", markersize=8, zorder=7, label="base")

    ax.set_aspect("equal")
    ax.grid(True)
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title(title)
    plt.show()
    return regions


# ------------------------------------------------------------------ output --

def draw(A, B, C, D, E):
    for v in [A, B, C, D, E]:
        plt.plot([v.x, v.end_x], [v.y, v.end_y], marker="o")
        plt.text((v.end_x + v.x) / 2, (v.end_y + v.y) / 2 + 0.3, v.name, ha="center")
    # every distinct point: the base point, the elbows, the meeting point and E's end
    points = [
        (A.x, A.y, A.name),
        (B.x, B.y, B.name),
        (A.end_x, A.end_y, f"{A.name}_end"),
        (B.end_x, B.end_y, f"{B.name}_end"),
        (C.end_x, C.end_y, f"{C.name}_end"),
        (E.end_x, E.end_y, f"{E.name}_end"),
    ]

    for px, py, label in points:
        plt.text(px, py + 0.3, f"({px:.1f}, {py:.1f})", ha="center")

    warn = limit_breaches(A, B)
    plt.title(f"A angle: {A.angle:.1f}°    B angle: {B.angle:.1f}°"
              + ("\nLIMIT: " + "; ".join(warn) if warn else ""))
    plt.axis("equal")
    plt.grid(True)
    plt.show()


def animate(A, B, C, D, E, movement, pause=0.05, reach=25, trace=True, targets=None,
            keep_open=True, on_limit="skip", regions=None):
    """Play any movement - an iterable of (a_angle, b_angle) pairs.

    movement : e.g. angle_move(90, 160, 0, 45)
                    path_through_points(A, B, C, D, E, [(5, 12), (-4, 10)])
                    chain(m1, m2, ...)
    targets  : optional list of (x, y) points to mark on the plot
    trace    : draw the path the end of E_end has taken
    on_limit : what to do with a frame that breaks a control-arm limit -
               "skip" (default, reports and moves on), "clamp" (pull it to the
               nearest legal angles) or "ignore" (drive it anyway)
    regions  : optional list of Region objects (exclusion zone, swept area,
               reachable area) drawn underneath the arm on every frame
    """
    trail_x, trail_y = [], []

    for (a_angle, b_angle) in movement:
        if on_limit == "clamp":
            a_angle, b_angle = clamp_pair(A, B, a_angle, b_angle)
        elif on_limit == "skip" and not within_limits(A, B, a_angle, b_angle):
            A.angle, B.angle = a_angle, b_angle
            print("Outside the limits, skipping: " + "; ".join(limit_breaches(A, B)))
            continue

        A.angle = a_angle
        B.angle = b_angle

        if not solve(C, D, E):
            print(f"Can't reach at A={a_angle:.1f}, B={b_angle:.1f}, skipping")
            continue

        if trace:
            trail_x.append(E.end_x)
            trail_y.append(E.end_y)

        plt.clf()

        if regions:
            for i, region in enumerate(regions):
                region.plot(ax=plt.gca(), zorder=i)

        if targets:
            plt.plot([p[0] for p in targets], [p[1] for p in targets],
                     "x", color="0.4", markersize=9, linestyle=":", linewidth=1)

        if trace and len(trail_x) > 1:
            plt.plot(trail_x, trail_y, "-", color="0.7", linewidth=1)

        for v in [A, B, C, D, E]:
            plt.plot([v.x, v.end_x], [v.y, v.end_y], marker="o")
            plt.text(v.x, v.y + 0.3, f"({v.x:.1f}, {v.y:.1f})", ha="center")
            plt.text(v.end_x, v.end_y + 0.3, f"({v.end_x:.1f}, {v.end_y:.1f})", ha="center")
            plt.text((v.end_x + v.x) / 2, (v.end_y + v.y) / 2 + 0.3, v.name, ha="center")

        plt.title(f"A angle: {a_angle:.1f}° [{A.limit_text()}]    "
                  f"B angle: {b_angle:.1f}° [{B.limit_text()}]\n"
                  f"end: ({E.end_x:.1f}, {E.end_y:.1f})")
        plt.gca().set_aspect("equal")
        plt.grid(True)
        plt.pause(pause)

    if keep_open:
        plt.show()    # keeps the window open at the end


# -------------------------------------------------------------------- demo --

if __name__ == "__main__":
    # the vectors are defined once and shared by draw and animate
    A = Vector(0, 0, length=10, angle=90, name="A_base",
               min_angle=45, max_angle=165)
    B = Vector(0, 0, length=5, angle=0, name="B_base",
               min_angle=-5)

    # B_base's ceiling follows A_base: always at least 5° below wherever A is
    B.max_angle = lambda: A.angle - 5

    C = Vector(length=5, parent=A, controllable=False, name="C_elbow")
    D = Vector(length=10, parent=B, controllable=False, name="D_elbow")
    E = Vector(length=10, parent=A, controllable=False, name="E_end")

    print(f"{A.name} limits: {A.limit_text()}")
    print(f"{B.name} limits: {B.limit_text()}  (max tracks A_base, currently {A.angle:.1f}°)")

    # static picture as before
    if solve(C, D, E):
        print(f"Start pose: end of {E.name} at ({E.end_x:.2f}, {E.end_y:.2f})")
        #draw(A, B, C, D, E)

    # --- the points we want the end of E_end to visit -----------------------
    # all inside the limited workspace - workspace_grid() shows what's legal
    targets = [
        (-14.0, 3.0),
        (-6.0, 3.0),
        (-6.0, -3.0),
        (-14.0, -3.0),
        (-14.0, 3.0),
    ]

    print("\nSolving the control-arm angles for each point:")
    solve_points(A, B, C, D, E, targets)

    # --- the areas the arm occupies -----------------------------------------
    # One sweep engine behind all of these - only the collector changes.
    print("\nWorking out the areas...")
    reach = reachable_region(A, B, C, D, E)                  # where E_end can go
    swept = swept_region(A, B, C, D, E)                      # where the metal goes
    keep_clear = swept.grow(2.0, name="exclusion zone (+2.0)", color="tab:red")

    for region in (reach, swept, keep_clear):
        print(f"  {region.name:28s} {region.area:8.1f} square units")

    # a perimeter you can mark out on the bench
    keep_clear.save_perimeter("exclusion_zone.csv")
    print("  perimeter written to exclusion_zone.csv")

    # the zones are queryable, not just drawable
    for t in targets[:-1]:
        print(f"  target {str(t):>14s}  reachable: {reach.contains(*t)}")

    # picture of the lot
    show_areas(A, B, C, D, E, regions=[keep_clear, swept, reach], targets=targets,
               verbose=False)

    # --- play the path, with the zones underneath ---------------------------
    # straight=True makes the end travel in straight lines between the points
    print("\nRunning the path...")
    animate(
        A, B, C, D, E,
        path_through_points(A, B, C, D, E, targets, steps_per_segment=25, straight=True),
        targets=targets,
        regions=[keep_clear, swept, reach],
        pause=0.03,
    )

    # other things the same sweep can answer:
    #
    #   just one link, e.g. to size a motor guard
    #     Region(sweep(A, B, C, D, E, links("A_base")), name="A_base sweep")
    #
    #   allow for a tool sticking out past the end of E_end
    #     swept_region(A, B, C, D, E, tool_radius=1.5)
    #
    #   is anything in the way?
    #     keep_clear.contains(bench_x, bench_y)
    #
    # the old style still works, it's just one movement among many.  animate()
    # skips anything illegal by default; on_limit="clamp" keeps it moving:
    # animate(A, B, C, D, E, angle_move(90, 200, 45, 10), on_limit="clamp")
    #
    # and movements can be joined:
    # animate(A, B, C, D, E, chain(
    #     angle_move(90, 120, 45, 60, steps=20),
    #     path_through_points(A, B, C, D, E, targets, start_angles=(120, 60)),
    # ), targets=targets)
