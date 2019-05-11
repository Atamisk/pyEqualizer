from pyequalizer.fileops import to_nas_real

def to_nas_force(sid,g,cid,f,n1,n2,n3):
    return ['FORCE',str(int(sid)),str(int(g)),str(int(cid)),to_nas_real(f),to_nas_real(n1),
            to_nas_real(n2),to_nas_real(n3)]
