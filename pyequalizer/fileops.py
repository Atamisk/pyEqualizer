# *********************
# *     PyStruct      * 
# *********************
# Module: FileOps
# Purpose: Functions relating to operating on files and filesystems. 
# Author: Aaron Moore
# Date:   2018-07-12

_valid_entries = ["PBAR", "PSHELL"]
import sys
from copy import deepcopy
from subprocess import Popen,call
from pystruct.stress_tensor import stress_tensor
from multiprocessing.pool import ThreadPool
from time import sleep
from math import *
import os

class cd:
    """Context manager for changing the current working directory"""
    def __init__(self, newPath):
        self.newPath = os.path.expanduser(newPath)

    def __enter__(self):
        self.savedPath = os.getcwd()
        os.chdir(self.newPath)

    def __exit__(self, etype, value, traceback):
        os.chdir(self.savedPath)

def to_nas_real(number):
    try:
        exponent = floor(log(abs(number),10))
    except ValueError as e: 
        #We probably passed zero into the function. Assume exponent is 1. 
        exponent = 1
    a = "{:6f}".format(number/(10**exponent))[0:6]
    return "{}{:+}".format(a,exponent)

def from_nas_real(string):
    lst = string.split('+')
    if len(lst) > 1:
        return float(lst[0]) * 10 ** int(lst[1])
    else:
        return float(lst[0])

def split_bulk(instr):
    line = []
    for x in [0,8,16,24,32,40,48,56,64,72]:
        if (len(instr)  > x + 8):
            line.append(instr[x:x+8].replace(" ", ""))
    return line

def strip_props(lines_in):
    return strip_card(lines_in, is_prop_header)

def strip_force(lines_in):
    return strip_card(lines_in, is_force)

def load_from_file(fname):
    with open(fname) as f:
        return [ i for i in f]

def strip_card(lines_in,filter_func):
    dead_lines = []
    lines = deepcopy(lines_in)
    for j in range(len(lines)):
        line = split_bulk(lines[j])
        if filter_func(line):
            dead_lines.append(j)
            k = j
            while needs_cont(lines[k]):
                k += 1
                dead_lines.append(k)
    for x in reversed(sorted(dead_lines)):
        del lines[x]
    return lines
    
def print_lines(lines,tgt):
    for x in lines:
        print(x, end="", file=tgt)

def inject_cards(cards, lines):
    my_lines = deepcopy(lines)
    for x in cards:
        newline = "{:<8s}".format(x[0])
        limit = min(9,len(x))
        for y in range(1, limit):
            newline += "{:>8s}".format(x[y])
        if len(x) > limit:
            newline += "+\n+       "
            for z in range(limit,len(x)):
                newline += "{:>8s}".format(x[z])
        newline += '\n'
        my_lines.insert(-1, newline)
    return my_lines

def is_prop_header(line):
    try:
        if any(line[0] == x for x in _valid_entries):
            return True
        else:
            return False
    except IndexError as e:
        return False
    
def is_force(line):
    try:
        if line[0] == "FORCE":
            return True
        else:
            return False
    except IndexError as e:
        return False
    
def needs_cont(instr):
    try:
        if instr[72] == '+':
            return True
        else:
            return False
    except:
        return False


def read_properties(lines_in):
    return read_cards(lines_in, is_prop_header)

def read_force(lines_in):
    return read_cards(lines_in, is_force)

def read_cards(lines_in, test_func):
    """read_properties(lines)
       Get the element properties from a nastran file. 
       Inputs: 
         lines_in: array of strings representing an input file. 
         test_func: Function to filter the cards requested. 
       Outputs: 
          property_names: List of Property classes describing all found PSHELL entries.
       """
    lines = deepcopy(lines_in)
    armed=False
    property_names = []
    j = 0;
    for i in lines_in:
        j += 1
        line = split_bulk(i)
        if test_func(line):
            property_names.append(line)
            armed = needs_cont(i)
        elif armed == True and len(property_names) > 0 and i[0:8].split() == ['+']:
            for x in line[1:]:
                property_names[len(property_names)-1].append(x) 
            armed = needs_cont(i)
        else:
            armed = False
    return property_names

def multi_file_out(prop_sets, lines, prefix):
    fnames = []
    for i in range(len(prop_sets)):
        fname = prefix + "-sub-" + str(i) + ".dat"
        fnames.append(fname)
        with open(fname, 'w') as f:
            print_lines(inject_cards(prop_sets[i], lines), f)
    return fnames

def run_nastran(nastr_bin, files):
    #print(nastr_bin)
    def call_nas(eff):
        split_path = os.path.split(eff)
        if split_path[0] == '':
            p = call([nastr_bin, split_path[1]])
        else:
            with cd(split_path[0]):
                if os.path.isfile(split_path[1]):
                    p = call([nastr_bin, split_path[1]])
                else:
                    raise IOError("File Not Found: {}".format(split_path[1]))
        tmp = p.communicate()
        
    processes = ThreadPool(8)
    res = []
    for f in files:
        res.append(processes.apply_async(call_nas, (f,)))
    processes.close()
    processes.join()
    processes.terminate()
        #sleep(3)

def skipline(f,n):
    for x in range(n):
        f.readline()

def to_von_mises(s1,s2):
    try:
        sa = float(s1)
        sb = float(s2)
    except:
        print(s1)
        print(s2)
        raise
    return (((sa - sb)**2 + sa**2 + sb**2)/2)**0.5

def to_pred(func, arg):
    a = func(arg)
    if a >= 0:
        return 1
    if a < 0:
        return 0

def act_on_stress_lines(f06_fname, proc_func):
    retval = 0
    for i in range(5):
        try:
            with open(f06_fname) as f:
                for i in f:
                    if i[18:83] == 'S T R E S S E S   I N   G E N E R A L   Q U A D R I L A T E R A L':
                        skipline(f,4)
                        l = f.readline()
                        while not to_pred(l.find,'PAGE') and to_pred(l.find,'E'):
                            loc = l[1:9]
                            retval = proc_func(l, loc, retval)
                            l = f.readline()
                            retval = proc_func(l, loc, retval)
                            l = f.readline()
            return retval
        except Exception as e:
            print("ERROR: {}".format(e))
    return 1.0*10**10 # Return an absurdly high stress is the file isn't found or has failed.

def max_cquad_stress(f06_fname):
    def get_max_stress(l, loc, last_val):
        max_stress = last_val
        max_stress = max(to_von_mises(l[87:100],l[103:116]),max_stress)
        return max_stress
    return act_on_stress_lines(f06_fname, get_max_stress)

def max_cquad_stress_loc(f06_fname):
    def get_max_stress(l, loc, last_val):
        try:
            old_stress = last_val[0]
        except TypeError as e:
            old_stress = 0
        new_stress = max(to_von_mises(l[87:100],l[103:116]),old_stress)
        if old_stress != new_stress:
            return [new_stress, loc]
        else:
            return last_val
    return act_on_stress_lines(f06_fname, get_max_stress)

def stress_at_point(f06_fname, point):
    def get_point_stress(l, loc, last_val):
        if (l[1:9] != '        ' and int(l[1:9]) == point):
            return [l[30:43],l[45:58],l[60:73]]
        else:
            return last_val
    return act_on_stress_lines(f06_fname, get_point_stress)

def stress_all_point(f06_fname):
    def get_point_stress(l, loc, last_val):
        try:
            dummy = last_val[0]
        except:
            last_val = []
        a = last_val + [[l[30:43],l[45:58],l[60:73]]]
        if (l[1:9] != '        '):
            return a
        else:
            return last_val
    return act_on_stress_lines(f06_fname, get_point_stress)


def mass(f06_name):
    """
    mass(f06_name): return nastran-calculated mass
    """
    for i in range(5):
        try:
            with open(f06_name) as f:
                for i in f:
                    if i[47:51] == 'MASS':
                        l = f.readline()
                        valstr = l[41:56]
                        return float(valstr.replace('D', 'E'))
        except Exception as e:
            print("ERROR: {}".format(e))
    return 1.0*10**10 # Return an absurdly high stress is the file isn't found or has failed.

if __name__ == "__main__":
    fname = "/home/atamisk/current_semester/models/test.dat"
    props = []
    props.append(read_properties(fname))
    for x in range(11):
        tmp = deepcopy(props[0])

    print_lines(inject_props(props, strip_props(fname)),sys.stdout)
    print(props)
