import math

class Node:
    # Class-level list to store instances
    _instances = []

    def __init__(self, x: float = 0.0, y: float = 0.0, name: str = ""):
        self.x = x
        self.y = y
        self.name = name
        self.__class__._instances.append(self)  # Add instance to the class-level list

    def set_name(self, name):
        self.name = name

    def __repr__(self):
        return f"Node(name={self.name}, x={self.x}, y={self.y})"

    @classmethod
    def all_instances(cls):
        """Return a list of all instances of the Node class."""
        return cls._instances

def vector_ep_x(x0: float, y0: float, length: float, angle: float) -> float:
    """Compute the x-coordinate of the endpoint of a vector."""
    radians = math.radians(angle)
    return x0 + length * math.cos(radians)

def vector_ep_y(x0: float, y0: float, length: float, angle: float) -> float:
    """Compute the y-coordinate of the endpoint of a vector."""
    radians = math.radians(angle)
    return y0 + length * math.sin(radians)

def vector_endpoint(x0: float, y0: float, length: float, angle: float) -> tuple[float, float]:
    """Compute the endpoint of a vector."""
    x1 = vector_ep_x(x0, y0, length, angle)
    y1 = vector_ep_y(x0, y0, length, angle)
    return (x1, y1)

Node1 = Node(0.0, 0.0, "Node1")

Arm1 = {
    "X0": Node1.x,
    "Y0": Node1.y,
    "Length": 10.0,
    "Angle": 90.0,
}

Node2 = Node(vector_ep_x(Node1.x, Node1.y, 10.0, 90.0), vector_ep_y(Node1.x, Node1.y, 10.0, 90.0), "Node2")

Arm2 = {
    "X0": Node2.x,
    "Y0": Node2.y,
    "Length": 5.0,
    "Angle": 0.0,
}

Arm1_Endpoint = vector_endpoint(Arm1["X0"], Arm1["Y0"], Arm1["Length"], Arm1["Angle"])
Arm2_Endpoint = vector_endpoint(Arm2["X0"], Arm2["Y0"], Arm2["Length"], Arm2["Angle"])

print(f"Arm1 Endpoints: {Arm1_Endpoint}")
print(f"Arm2 Endpoints: {Arm2_Endpoint}")

print(f"All Nodes: {Node.all_instances()}")
# source .venv/bin/activate
# python main.py