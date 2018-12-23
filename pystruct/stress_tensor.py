from numpy import array, add, trace

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
        sdev = self.tensor - 1/3*trace(self.tensor)
        J2 = 0.5 * trace(sdev @ sdev)
        return (3 * J2) ** 0.5

    def __add__(self, other):
        try:
            return stress_tensor._from_array(other.tensor + self.tensor)
        except:
            return NotImplemented
    __radd__ = __add__


