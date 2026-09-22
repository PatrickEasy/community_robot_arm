import math
import matplotlib.pyplot as plt

class Vector:
    def __init__(self, x=0, y=0, length=10, angle=0, parent=None, controllable=True, name="Vector"):
        self._x = x          # only used if there is no parent
        self._y = y
        self.length = length
        self._angle = angle  # degrees
        self.parent = parent
        self.controllable = controllable
        self.name = name  # for debugging

    @property
    def angle(self):
        return self._angle

    @angle.setter
    def angle(self, value):
        if not self.controllable:
            raise Exception("This vector is constrained, its angle can't be set directly")
        self._angle = value

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
    h = math.sqrt(r1**2 - a**2)

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

    plt.title(f"A angle: {A.angle:.1f}°    B angle: {B.angle:.1f}°")
    plt.axis("equal")
    plt.grid(True)
    plt.show()

def animate(A, B, C, D, E, a_start, a_end, b_start, b_end, steps=45):
    """Move A and B together from their start angles to their end angles,
    solving and redrawing the arm at every step."""
    # fixed axis limits so the view doesn't jump around (based on the vector lengths)
    reach = 25

    for i in range(steps + 1):
        t = i / steps    # goes from 0 to 1
        A.angle = a_start + (a_end - a_start) * t
        B.angle = b_start + (b_end - b_start) * t

        if not solve(C, D, E):
            print(f"Can't reach at A={A.angle:.1f}, B={B.angle:.1f}, skipping")
            continue

        plt.clf()
        for v in [A, B, C, D, E]:
            plt.plot([v.x, v.end_x], [v.y, v.end_y], marker="o")
            plt.text(v.x, v.y + 0.3, f"({v.x:.1f}, {v.y:.1f})", ha="center")
            plt.text(v.end_x, v.end_y + 0.3, f"({v.end_x:.1f}, {v.end_y:.1f})", ha="center")
            plt.text((v.end_x + v.x) / 2, (v.end_y + v.y) / 2 + 0.3, v.name, ha="center")

        plt.title(f"A angle: {A.angle:.1f}°    B angle: {B.angle:.1f}°")
        plt.gca().set_aspect("equal")
        plt.grid(True)
        plt.pause(0.1)

    plt.show()    # keeps the window open at the end


# the vectors are defined once and shared by draw and animate
A = Vector(0, 0, length=10, angle=90, name="A_base")
B = Vector(0, 0, length=5, angle=0, name="B_base")

C = Vector(length=5, parent=A, controllable=False, name="C_elbow")
D = Vector(length=10, parent=B, controllable=False, name="D_elbow")
E = Vector(length=10, parent=A, controllable=False, name="E_end")   # length 10 is the actual length

# static picture as before
if solve(C, D, E):
    draw(A, B, C, D, E)

# movement: A goes 90 -> 120 while B goes 0 -> 45
animate(A, B, C, D, E, 90, 160, 0, 45)

# source .venv/bin/activate
# python mainv3.py