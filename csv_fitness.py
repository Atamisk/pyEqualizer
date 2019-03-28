from pystruct.optim import *
from pystruct import cost_stress
from copy import deepcopy
import argparse as arg
import csv

def parseargs():
    parser = arg.ArgumentParser()
    parser.add_argument('fname',help="Filename for CSV file.")
    return parser.parse_args()

def get_props_array(fname, base_props):
    with open(fname) as fdesc:
        pr_arr = []
        cs_dict = csv.DictReader(fdesc, delimiter=',')
        for row in cs_dict:
            prop = deepcopy(base_props)
            prop[0][3] = to_nas_real(float(row['Top Flange Width']))
            prop[1][3] = to_nas_real(float(row['Bottom Flange Width']))
            prop[2][3] = to_nas_real(float(row['Web Thickness']))
            prop[3][3] = to_nas_real(float(row['Doubler Thickness at Hoist Pin']))
            prop[4][3] = to_nas_real(float(row['Doubler Thickness at Load Pin']))
            pr_arr.append(prop)
        return(pr_arr)



args = parseargs()
fname = args.fname
sys = system(0,'/tmp/test_open.dat',10,10,[cost_stress],[])
props_array = get_props_array(fname, sys.base_props)
fit = sys.run_generation(deepcopy,props_array)
for x in fit:
    print(x.fitness[0])
