from pystruct.optim import *
from pystruct import test_open_force_pack
from pystruct import cost_stress
from copy import deepcopy
import argparse as arg
import csv

def parseargs():
    parser = arg.ArgumentParser()
    parser.add_argument('fname',help="Filename for CSV file.")
    parser.add_argument('--beta', help="Get Beta instead of peak stress", action='store_true')
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
if (args.beta == True):
    x_force = test_open_force_pack(1,1000,0,0)
    y_force = test_open_force_pack(1,0,1000,0)
    sto_force_x = nr_var(0,5000)
    sto_force_y = nr_var(150000,19500)
    sys = system_unit(1,'/tmp/test_open.dat', 1,3, x_force, y_force, sto_force_x, sto_force_y)
    fit_index = 1
    fit_sign = -1
else:
    sys = system(0,'/tmp/test_open.dat',10,10,[cost_stress],[])
    fit_index = 0
    fit_sign = 1
print(sys.base_props)
props_array = get_props_array(fname, sys.base_props)
fit = sys.run_generation(deepcopy,props_array)
for x in fit:
    print(fit_sign * x.fitness[fit_index])
