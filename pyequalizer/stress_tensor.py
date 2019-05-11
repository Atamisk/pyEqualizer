from numpy import array, add, trace, eye, size, shape, tensordot

class stress_tensor(object):
    def __init__(self, sx, sy, sz, txy, tyz, tzx):
        SX = float(sx)
        SY = float(sy)
        SZ = float(sz)
        TXY = float(txy)
        TYZ = float(tyz)
        TZX = float(tzx)
        self._tensor = array([[SX, TXY, TZX],[TXY,SY,TYZ],[TZX,TYZ,SZ]])

    @classmethod
    def _from_array(self, a):
        return stress_tensor(a[0,0],a[1,1],a[2,2],a[0,1],a[1,2],a[0,2])


    @property
    def tensor(self):
        return self._tensor
    @property
    def von_mises(self):
        sdev = self.deviator
        return (3/2 * tensordot(sdev, sdev))**0.5
    @property
    def deviator(self):
        t = self.tensor
        return t - eye(*shape(t)) * trace(t) / 3.0

    def __add__(self, other):
        try:
            return stress_tensor._from_array(other.tensor + self.tensor)
        except:
            return NotImplemented
    __radd__ = __add__

    def __mul__(self, other):
        try:
            return stress_tensor._from_array(other * self.tensor)
        except:
            return NotImplemented
    __rmul__ = __mul__


