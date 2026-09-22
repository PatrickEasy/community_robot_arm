"""
Stroke font - target coordinates for A-Z, 0-9 and a little punctuation.

Every glyph is drawn the way a pen draws it: as a few continuous strokes,
never as a filled outline.  That matches what the arm can actually do - move
the end through a list of points - so the output drops straight into the
simulator:

    strokes = text_strokes("HELLO", height=4.0, origin=(-16.0, -1.0))
    for stroke in strokes:
        robot.steps_through("node_end", stroke, f"stroke_{i}.csv")

Coordinates are built in a unit box first: x from 0 to the glyph's own width,
y from 0 on the baseline to 1 at cap height.  Everything is then scaled and
shifted into whatever space you want, so the same glyphs suit a 3 cm signature
or a 3 m one.

Only the standard library is used, and nothing is printed.  preview() pulls in
matplotlib when you call it, and only then.

    source .venv/bin/activate
    python font.py
"""

import math


# ---------------------------------------------------------------- strokes --
#
# A glyph is (width, [stroke, stroke, ...]).
# A stroke is a list of segments, drawn without lifting the pen.
# A segment is one of:
#
#   ("L", [(x, y), ...])                 straight lines through these points
#   ("A", cx, cy, rx, ry, a0, a1)        elliptical arc, degrees, a0 -> a1
#   ("B", p0, c0, c1, p1)                cubic bezier
#
# Arcs and beziers are sampled into points when the glyph is built, so what
# comes out the far end is always plain coordinates.

GLYPHS = {
    " ": (0.40, []),

    "A": (0.60, [[("L", [(0.00, 0.00), (0.30, 1.00), (0.60, 0.00)])],
                 [("L", [(0.12, 0.40), (0.48, 0.40)])]]),

    "B": (0.58, [[("L", [(0.00, 0.00), (0.00, 1.00)])],
                 [("A", 0.00, 0.755, 0.30, 0.245, 90, -90)],
                 [("A", 0.00, 0.255, 0.34, 0.255, 90, -90)]]),

    "C": (0.60, [[("A", 0.30, 0.50, 0.30, 0.50, 50, 310)]]),

    "D": (0.58, [[("L", [(0.00, 0.00), (0.00, 1.00)])],
                 [("A", 0.00, 0.50, 0.58, 0.50, 90, -90)]]),

    "E": (0.55, [[("L", [(0.55, 1.00), (0.00, 1.00), (0.00, 0.00), (0.55, 0.00)])],
                 [("L", [(0.00, 0.50), (0.45, 0.50)])]]),

    "F": (0.55, [[("L", [(0.55, 1.00), (0.00, 1.00), (0.00, 0.00)])],
                 [("L", [(0.00, 0.50), (0.45, 0.50)])]]),

    "G": (0.62, [[("A", 0.31, 0.50, 0.31, 0.50, 50, 340),
                  ("L", [(0.601, 0.329), (0.601, 0.46), (0.32, 0.46)])]]),

    "H": (0.58, [[("L", [(0.00, 0.00), (0.00, 1.00)])],
                 [("L", [(0.58, 0.00), (0.58, 1.00)])],
                 [("L", [(0.00, 0.50), (0.58, 0.50)])]]),

    "I": (0.40, [[("L", [(0.00, 1.00), (0.40, 1.00)])],
                 [("L", [(0.20, 1.00), (0.20, 0.00)])],
                 [("L", [(0.00, 0.00), (0.40, 0.00)])]]),

    "J": (0.50, [[("L", [(0.50, 1.00), (0.50, 0.26)]),
                  ("A", 0.25, 0.26, 0.25, 0.26, 0, -180)]]),

    "K": (0.55, [[("L", [(0.00, 0.00), (0.00, 1.00)])],
                 [("L", [(0.55, 1.00), (0.00, 0.38), (0.55, 0.00)])]]),

    "L": (0.50, [[("L", [(0.00, 1.00), (0.00, 0.00), (0.50, 0.00)])]]),

    "M": (0.70, [[("L", [(0.00, 0.00), (0.00, 1.00), (0.35, 0.32),
                         (0.70, 1.00), (0.70, 0.00)])]]),

    "N": (0.62, [[("L", [(0.00, 0.00), (0.00, 1.00), (0.62, 0.00), (0.62, 1.00)])]]),

    "O": (0.62, [[("A", 0.31, 0.50, 0.31, 0.50, 0, 360)]]),

    "P": (0.54, [[("L", [(0.00, 0.00), (0.00, 1.00)])],
                 [("A", 0.00, 0.735, 0.36, 0.265, 90, -90)]]),

    "Q": (0.62, [[("A", 0.31, 0.50, 0.31, 0.50, 0, 360)],
                 [("L", [(0.38, 0.22), (0.64, -0.06)])]]),

    "R": (0.56, [[("L", [(0.00, 0.00), (0.00, 1.00)])],
                 [("A", 0.00, 0.735, 0.34, 0.265, 90, -90)],
                 [("L", [(0.06, 0.47), (0.56, 0.00)])]]),

    "S": (0.56, [[("B", (0.53, 0.85), (0.54, 1.06), (0.06, 1.05), (0.05, 0.71)),
                  ("B", (0.05, 0.71), (0.04, 0.50), (0.53, 0.55), (0.52, 0.32)),
                  ("B", (0.52, 0.32), (0.51, 0.00), (0.05, 0.00), (0.03, 0.20))]]),

    "T": (0.55, [[("L", [(0.00, 1.00), (0.55, 1.00)])],
                 [("L", [(0.275, 1.00), (0.275, 0.00)])]]),

    "U": (0.58, [[("L", [(0.00, 1.00), (0.00, 0.28)]),
                  ("A", 0.29, 0.28, 0.29, 0.28, 180, 360),
                  ("L", [(0.58, 0.28), (0.58, 1.00)])]]),

    "V": (0.58, [[("L", [(0.00, 1.00), (0.29, 0.00), (0.58, 1.00)])]]),

    "W": (0.78, [[("L", [(0.00, 1.00), (0.195, 0.00), (0.39, 0.62),
                         (0.585, 0.00), (0.78, 1.00)])]]),

    "X": (0.58, [[("L", [(0.00, 1.00), (0.58, 0.00)])],
                 [("L", [(0.00, 0.00), (0.58, 1.00)])]]),

    "Y": (0.58, [[("L", [(0.00, 1.00), (0.29, 0.52), (0.58, 1.00)])],
                 [("L", [(0.29, 0.52), (0.29, 0.00)])]]),

    "Z": (0.56, [[("L", [(0.00, 1.00), (0.56, 1.00), (0.00, 0.00), (0.56, 0.00)])]]),

    "0": (0.56, [[("A", 0.28, 0.50, 0.28, 0.50, 0, 360)]]),

    "1": (0.46, [[("L", [(0.06, 0.80), (0.26, 1.00), (0.26, 0.00)])],
                 [("L", [(0.04, 0.00), (0.46, 0.00)])]]),

    "2": (0.54, [[("B", (0.03, 0.78), (0.03, 1.06), (0.52, 1.06), (0.50, 0.71)),
                  ("L", [(0.50, 0.71), (0.02, 0.00), (0.54, 0.00)])]]),

    "3": (0.54, [[("B", (0.04, 0.86), (0.18, 1.06), (0.54, 0.98), (0.51, 0.73)),
                  ("B", (0.51, 0.73), (0.49, 0.60), (0.40, 0.52), (0.26, 0.52)),
                  ("B", (0.26, 0.52), (0.42, 0.52), (0.55, 0.45), (0.53, 0.26)),
                  ("B", (0.53, 0.26), (0.50, 0.01), (0.15, -0.04), (0.03, 0.12))]]),

    "4": (0.56, [[("L", [(0.42, 0.00), (0.42, 1.00), (0.02, 0.30), (0.56, 0.30)])]]),

    "5": (0.54, [[("L", [(0.50, 1.00), (0.09, 1.00), (0.06, 0.60)]),
                  ("B", (0.06, 0.60), (0.30, 0.73), (0.56, 0.60), (0.53, 0.33)),
                  ("B", (0.53, 0.33), (0.51, 0.04), (0.15, -0.04), (0.02, 0.11))]]),

    "6": (0.54, [[("B", (0.47, 0.93), (0.26, 1.07), (0.03, 0.86), (0.03, 0.26)),
                  ("A", 0.27, 0.26, 0.24, 0.26, 180, 540)]]),

    "7": (0.54, [[("L", [(0.02, 1.00), (0.54, 1.00), (0.19, 0.00)])]]),

    "8": (0.55, [[("A", 0.275, 0.745, 0.235, 0.255, -90, 270)],
                 [("A", 0.275, 0.245, 0.275, 0.245, 90, 450)]]),

    "9": (0.54, [[("B", (0.07, 0.07), (0.28, -0.07), (0.51, 0.14), (0.51, 0.74)),
                  ("A", 0.27, 0.74, 0.24, 0.26, 0, 360)]]),

    ".": (0.24, [[("L", [(0.09, 0.00), (0.15, 0.00)])]]),
    ",": (0.24, [[("L", [(0.15, 0.06), (0.06, -0.12)])]]),
    "-": (0.40, [[("L", [(0.04, 0.46), (0.36, 0.46)])]]),
    "+": (0.46, [[("L", [(0.04, 0.46), (0.42, 0.46)])],
                 [("L", [(0.23, 0.27), (0.23, 0.65)])]]),
    "=": (0.46, [[("L", [(0.04, 0.58), (0.42, 0.58)])],
                 [("L", [(0.04, 0.34), (0.42, 0.34)])]]),
    "/": (0.42, [[("L", [(0.00, 0.00), (0.42, 1.00)])]]),
    ":": (0.24, [[("L", [(0.09, 0.62), (0.15, 0.62)])],
                 [("L", [(0.09, 0.00), (0.15, 0.00)])]]),
    "!": (0.26, [[("L", [(0.13, 1.00), (0.13, 0.30)])],
                 [("L", [(0.10, 0.00), (0.16, 0.00)])]]),
    "?": (0.52, [[("B", (0.05, 0.75), (0.03, 1.07), (0.50, 1.08), (0.48, 0.76)),
                  ("B", (0.48, 0.76), (0.46, 0.57), (0.26, 0.54), (0.26, 0.32))],
                 [("L", [(0.23, 0.00), (0.29, 0.00)])]]),
}

# how finely curves are broken into points, in unit-box distance
CURVE_STEP = 0.035


# ----------------------------------------------------------- the sampling --

def _arc_points(cx, cy, rx, ry, a0, a1, step=CURVE_STEP):
    """An elliptical arc as points.  Angles in degrees; a1 may be below a0 to
    sweep the other way, and may pass 360 to go more than once round."""
    sweep = math.radians(abs(a1 - a0))
    size = max(rx, ry)
    count = max(2, int(math.ceil(sweep * size / step)))
    return [(cx + rx * math.cos(math.radians(a0 + (a1 - a0) * i / count)),
             cy + ry * math.sin(math.radians(a0 + (a1 - a0) * i / count)))
            for i in range(count + 1)]


def _bezier_points(p0, c0, c1, p1, step=CURVE_STEP):
    """A cubic bezier as points, roughly evenly spaced."""
    rough = (math.dist(p0, c0) + math.dist(c0, c1) + math.dist(c1, p1))
    count = max(2, int(math.ceil(rough / step)))
    out = []
    for i in range(count + 1):
        t = i / count
        u = 1 - t
        out.append((u**3 * p0[0] + 3 * u*u*t * c0[0] + 3 * u*t*t * c1[0] + t**3 * p1[0],
                    u**3 * p0[1] + 3 * u*u*t * c0[1] + 3 * u*t*t * c1[1] + t**3 * p1[1]))
    return out


def _segment_points(segment, step=CURVE_STEP):
    kind = segment[0]
    if kind == "L":
        return list(segment[1])
    if kind == "A":
        return _arc_points(*segment[1:], step=step)
    if kind == "B":
        return _bezier_points(*segment[1:], step=step)
    raise ValueError(f"Unknown segment type {kind!r}")


def glyph(character, step=CURVE_STEP):
    """One character as strokes in the unit box: [[(x, y), ...], ...].
    Unknown characters come back as an empty list."""
    entry = GLYPHS.get(character.upper())
    if entry is None:
        return []
    _, strokes = entry
    out = []
    for stroke in strokes:
        points = []
        for segment in stroke:
            piece = _segment_points(segment, step)
            # a stroke is drawn without lifting, so don't repeat the join point
            if points and math.dist(points[-1], piece[0]) < 1e-9:
                piece = piece[1:]
            points.extend(piece)
        if len(points) > 1:
            out.append(points)
    return out


def width(character):
    """How wide a character is, in units of cap height."""
    entry = GLYPHS.get(character.upper())
    return entry[0] if entry else 0.0


def supported():
    """Every character this font knows, as a sorted string."""
    return "".join(sorted(GLYPHS))


# ------------------------------------------------------------------ text ---

def text_strokes(text, height=1.0, origin=(0.0, 0.0), tracking=0.14,
                 line_gap=0.45, step=CURVE_STEP):
    """Lay out a string and return its strokes in world coordinates.

    height   cap height, in whatever units the robot works in
    origin   where the baseline of the first line starts
    tracking gap between characters, as a fraction of cap height
    line_gap gap between baselines, on top of the cap height

    Newlines start a new line.  The result is a list of strokes; each stroke
    is a list of (x, y) the arm can trace without lifting.
    """
    out = []
    x0, y0 = origin
    pen_y = y0
    for line in str(text).split("\n"):
        pen_x = x0
        for character in line:
            for stroke in glyph(character, step):
                out.append([(pen_x + px * height, pen_y + py * height)
                            for px, py in stroke])
            pen_x += (width(character) + tracking) * height
        pen_y -= (1.0 + line_gap) * height
    return out


def text_width(text, height=1.0, tracking=0.14):
    """How wide a single line will be, before you commit to drawing it."""
    line = max(str(text).split("\n"), key=lambda s: sum(width(c) + tracking for c in s))
    total = sum((width(c) + tracking) for c in line)
    return max(0.0, total - tracking) * height


# --------------------------------------------------------------- shaping ---

def bounds(strokes):
    """(min_x, min_y, max_x, max_y) around a set of strokes."""
    points = [p for stroke in strokes for p in stroke]
    if not points:
        return (0.0, 0.0, 0.0, 0.0)
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return (min(xs), min(ys), max(xs), max(ys))


def transform(strokes, scale=1.0, move=(0.0, 0.0), rotate=0.0, about=(0.0, 0.0)):
    """Scale, turn and shift a set of strokes.  Rotation is in degrees, about
    the given point - useful when the arm's comfortable area isn't square on."""
    radians = math.radians(rotate)
    cos, sin = math.cos(radians), math.sin(radians)
    ax, ay = about
    out = []
    for stroke in strokes:
        moved = []
        for x, y in stroke:
            x, y = (x - ax) * scale, (y - ay) * scale
            moved.append((ax + x * cos - y * sin + move[0],
                          ay + x * sin + y * cos + move[1]))
        out.append(moved)
    return out


def fit(strokes, box, margin=0.0, keep_shape=True):
    """Scale and centre strokes to sit inside box = (min_x, min_y, max_x, max_y).

    Point it at the arm's reachable area and the writing lands where the arm
    can actually get to, whatever size the text was built at.
    """
    bx0, by0, bx1, by1 = box
    bx0, by0, bx1, by1 = bx0 + margin, by0 + margin, bx1 - margin, by1 - margin
    x0, y0, x1, y1 = bounds(strokes)
    span_x, span_y = max(x1 - x0, 1e-9), max(y1 - y0, 1e-9)

    scale_x, scale_y = (bx1 - bx0) / span_x, (by1 - by0) / span_y
    scale = min(scale_x, scale_y) if keep_shape else scale_x

    shifted = transform(strokes, scale=scale, about=(x0, y0))
    sx0, sy0, sx1, sy1 = bounds(shifted)
    return transform(shifted, move=((bx0 + bx1) / 2 - (sx0 + sx1) / 2,
                                    (by0 + by1) / 2 - (sy0 + sy1) / 2))


def resample(strokes, max_step):
    """Break long straight runs into hops of at most max_step.

    Worth doing before handing a stroke to the arm: the planner walks in a
    straight line between targets anyway, and evenly spaced targets give
    evenly spaced motor steps.
    """
    out = []
    for stroke in strokes:
        if len(stroke) < 2:
            out.append(list(stroke))
            continue
        points = [stroke[0]]
        for (x0, y0), (x1, y1) in zip(stroke, stroke[1:]):
            span = math.dist((x0, y0), (x1, y1))
            hops = max(1, int(math.ceil(span / max_step)))
            for i in range(1, hops + 1):
                points.append((x0 + (x1 - x0) * i / hops, y0 + (y1 - y0) * i / hops))
        out.append(points)
    return out


def as_path(strokes):
    """Flatten to one list of (x, y, pen).

    pen is 0 on the first point of each stroke - the move to where that stroke
    begins - and 1 for every point drawn after it.  Same convention as the CSV.
    If there is no pen to lift, the pen 0 rows are the moves to hurry through.
    """
    return [(x, y, 0 if i == 0 else 1)
            for stroke in strokes
            for i, (x, y) in enumerate(stroke)]


# --------------------------------------------------------------- saving ----

def save_csv(strokes, path="targets.csv", label=""):
    """Write the target coordinates: one row per point.

    stroke  which continuous stroke the point belongs to
    point   its place within that stroke
    pen     1 while drawing, 0 on the first point of each stroke (the move to
            the start), so the file can be traced straight through
    """
    with open(path, "w") as handle:
        handle.write("label,stroke,point,x,y,pen\n")
        for s, stroke in enumerate(strokes):
            for i, (x, y) in enumerate(stroke):
                handle.write(f"{label},{s},{i},{x:.5f},{y:.5f},{0 if i == 0 else 1}\n")
    return path


def save_alphabet_csv(path="alphabet_targets.csv", height=1.0, step=CURVE_STEP):
    """Every character the font knows, each in its own unit box, in one file.
    The character is the label column, so it's easy to pull one back out.
    """
    with open(path, "w") as handle:
        handle.write("label,stroke,point,x,y,pen\n")
        for character in sorted(GLYPHS):
            for s, stroke in enumerate(glyph(character, step)):
                for i, (x, y) in enumerate(stroke):
                    handle.write(f"{character},{s},{i},{x * height:.5f},"
                                 f"{y * height:.5f},{0 if i == 0 else 1}\n")
    return path


def load_csv(path):
    """Read a targets file back as {label: [stroke, ...]}."""
    out = {}
    with open(path) as handle:
        rows = handle.read().strip().split("\n")[1:]
    for row in rows:
        label, s, _, x, y, _ = row.rsplit(",", 5)   # label may contain commas
        strokes = out.setdefault(label, {})
        strokes.setdefault(int(s), []).append((float(x), float(y)))
    return {label: [strokes[k] for k in sorted(strokes)] for label, strokes in out.items()}


# --------------------------------------------------------------- preview ---

def preview(strokes, path=None, title="", show_travel=True, box=None):
    """Draw the strokes so you can see what the arm would trace.

    matplotlib is only imported here, so generating coordinates needs nothing
    but the standard library.
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(11, 4))
    for i, stroke in enumerate(strokes):
        ax.plot([p[0] for p in stroke], [p[1] for p in stroke],
                "-", color="tab:blue", linewidth=1.8, solid_capstyle="round")
        if show_travel and i:
            previous = strokes[i - 1][-1]
            ax.plot([previous[0], stroke[0][0]], [previous[1], stroke[0][1]],
                    ":", color="0.75", linewidth=0.8)
    if box:
        x0, y0, x1, y1 = box
        ax.plot([x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0], "-", color="tab:red",
                linewidth=1.0)
    ax.set_aspect("equal")
    ax.grid(True, linewidth=0.4, alpha=0.5)
    ax.set_title(title)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=130)
        plt.close(fig)
        return path
    plt.show()
    return None


# ------------------------------------------------------------------ demo ---

if __name__ == "__main__":

    save_alphabet_csv("alphabet_targets.csv")

    sample = text_strokes("ABCDEFGHIJKLM\nNOPQRSTUVWXYZ\n0123456789 .,-+=/:!?",
                          height=1.0, origin=(0.0, 0.0))
    save_csv(sample, "font_sample_targets.csv", label="sample")
    preview(sample, "font_sample.png", title="stroke font - every glyph")

    # Sized for the arm.  This box is the biggest rectangle that fits wholly
    # inside the reachable area for arm_config.json - see the note below for
    # where it comes from.  Don't use the area's bounding box: for a blob-
    # shaped reachable area all four of its corners are out of reach.
    ARM_BOX = (-15.9, -4.6, -3.5, 7.8)

    writing = text_strokes("HELLO 123", height=1.0)
    writing = fit(writing, box=ARM_BOX, margin=0.4)
    writing = resample(writing, max_step=0.35)
    save_csv(writing, "hello_targets.csv", label="HELLO 123")
    preview(writing, "hello_preview.png", title="HELLO 123, fitted to the arm",
            box=ARM_BOX)

    # to work the box out for your own linkage, and trace it with the arm:
    #
    #   import font, v5
    #   robot = v5.Robot.from_file("arm_config.json")
    #   box = robot.reachable_area("node_end").inner_box(margin=0.4)
    #   strokes = font.resample(font.fit(font.text_strokes("HI"), box), 0.35)
    #   for i, stroke in enumerate(strokes):
    #       robot.steps_through("node_end", stroke, f"stroke_{i}.csv", steps=1)
