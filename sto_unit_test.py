from pystruct.optim import tensor_ind, system_unit
from pystruct.fileops import *

def print_list(l):
    for x in l:
        print(x)

fname = '/tmp/test_open.dat'
file_lines = load_from_file(fname)
starting_force = read_force(file_lines)
file_lines = load_from_file(fname)
starting_force = read_force(file_lines)

x_force = [['FORCE', '2', '918', '0', '1.', '1.+3', '0', '0.']]
y_force = [['FORCE', '2', '918', '0', '1.', '0.', '1.+3', '0.']]

test = system_unit(1, '/tmp/test_open.dat', 1, 50, [], [], x_force, y_force, force = starting_force)
tens_inds = test.first_generation()
tens_inds[0].apply_stochastic_force(0,150000,130,500)
test_tensors = tens_inds[0].apply_force(0,150000)
test_vms = [x.von_mises for x in test_tensors]
print("***")
print_list(test_vms)
print(test)
