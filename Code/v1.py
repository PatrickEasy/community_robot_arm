import math


class Vector:
    """A vector defined by its origin, length and angle, with the end point computed.

    A vector can be attached to a parent: its start then follows the parent's
    end, and any change to the parent propagates down the chain.
    """

    def __init__(self, x0: float = 0.0, y0: float = 0.0, length: float = 1.0,
                 angle: float = 0.0, label: str = "",
                 parent: "Vector | None" = None, relative_angle: bool = False):
        self._parent: "Vector | None" = None
        self._children: list["Vector"] = []

        self.x0 = x0
        self.y0 = y0
        self.length = length
        self.angle = angle
        self.label = label
        self.relative_angle = relative_angle
        self.x1 = 0.0
        self.y1 = 0.0

        if parent is not None:
            self.attach_to(parent)
        else:
            self.compute_end()

    # --- geometry -------------------------------------------------------

    @property
    def absolute_angle(self) -> float:
        """Angle in world space. If relative_angle, self.angle is measured from the parent."""
        if self._parent is not None and self.relative_angle:
            return self._parent.absolute_angle + self.angle
        return self.angle

    def compute_end(self):
        """Recompute the start (if attached), the end point, and all dependents."""
        if self._parent is not None:
            self.x0, self.y0 = self._parent.end

        radians = math.radians(self.absolute_angle)
        self.x1 = self.x0 + self.length * math.cos(radians)
        self.y1 = self.y0 + self.length * math.sin(radians)

        for child in self._children:
            child.compute_end()

    @property
    def start(self):
        return (self.x0, self.y0)

    @property
    def end(self):
        return (self.x1, self.y1)

    # --- linking --------------------------------------------------------

    @property
    def parent(self):
        return self._parent

    @property
    def children(self):
        return tuple(self._children)

    def attach_to(self, parent: "Vector | None"):
        """Make this vector start at `parent`'s end. Pass None to detach."""
        if parent is not None and self._is_ancestor_of(parent):
            raise ValueError(f"attaching {self.label!r} to {parent.label!r} would create a cycle")

        if self._parent is not None:
            self._parent._children.remove(self)

        self._parent = parent
        if parent is not None:
            parent._children.append(self)

        self.compute_end()
        return self

    def detach(self):
        """Break the link, freezing this vector where it currently sits."""
        return self.attach_to(None)

    def then(self, length: float, angle: float, label: str = "", relative_angle: bool = False):
        """Convenience: create and return a new vector starting at this one's end."""
        return Vector(length=length, angle=angle, label=label,
                      parent=self, relative_angle=relative_angle)

    def _is_ancestor_of(self, node: "Vector | None") -> bool:
        while node is not None:
            if node is self:
                return True
            node = node._parent
        return False

    # --- mutation -------------------------------------------------------

    def update(self, x0: float = None, y0: float = None,
               length: float = None, angle: float = None):
        """Update properties and recompute this vector and everything downstream.

        x0/y0 are ignored for an attached vector — its start comes from its parent.
        """
        if x0 is not None:
            self.x0 = x0
        if y0 is not None:
            self.y0 = y0
        if length is not None:
            self.length = length
        if angle is not None:
            self.angle = angle
        self.compute_end()
        return self

    def __repr__(self):
        return (f"Vector({self.label!r}, start=({self.x0:.2f}, {self.y0:.2f}), "
                f"end=({self.x1:.2f}, {self.y1:.2f}))")

ControlArmPrimary = Vector(0, 0, 10, 90, "ControlArmPrimary")
ControlArmSecondary = Vector(0, 0, 5, 0, "ControlArmSecondary")

DependentArmPrimary = Vector(0, 0, 5, 0, "DependentArmPrimary", parent=ControlArmPrimary)
DependentArmSecondary = Vector(0, 0, 10, 90, "DependentArmSecondary", parent=ControlArmSecondary)

print(ControlArmPrimary)
print(ControlArmSecondary)
print(DependentArmPrimary)
print(DependentArmSecondary)

ControlArmPrimary.update(angle=45)

# source .venv/bin/activate
# python main.py