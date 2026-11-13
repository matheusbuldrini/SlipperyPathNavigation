class Node:
    def __init__(self, x, y, parent=None, cost=1):
        self.x = x
        self.y = y
        self.parent = parent
        self.cost = cost

    def __eq__(self, other):
        return isinstance(other, Node) and (self.x, self.y) == (other.x, other.y)

    def __hash__(self):
        return hash((self.x, self.y))

    def __str__(self):
        return f"({self.x},{self.y})"
    
    def __repr__(self):
        return str(self)