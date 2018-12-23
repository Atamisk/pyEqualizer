from numpy import array, add, trace

class stress_tensor(object):
    def __init__(self, sx, sy, sz, txy, tyz, tzx):
        self._tensor = array([[sx, txy, tzx],[txy,sy,tyz],[tzx,tyz,sz]])
    
    @property
    def tensor(self):
        return self._tensor
    @property
    def von_mises(self):
        sdev = self.tensor - 1/3*trace(self.tensor)
        J2 = 0.5 * trace(sdev @ sdev)
        return (3 * J2) ** 0.5

    def __add__(self, other):
        return other + self.tensor
    __radd__ = __add__


