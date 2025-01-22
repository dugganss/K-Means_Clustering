from Coordinate import Coordinate

class Centroid(Coordinate):
    def __init__(self, id, x, y, z= None):
        super().__init__(x, y, z)
        self.id = id

    def recalculateCentroid(self):
        print("needs implementing")