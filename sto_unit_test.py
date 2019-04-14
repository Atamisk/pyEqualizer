from pystruct.optim import tensor_ind, system_unit
from pystruct.fileops import *

def print_list(l):
    for x in l:
        print(x)

def max_stoch_str(objs):
    lst = list([x[0] + x[1] for x in objs])
    return objs[lst.index(max(lst))]

def test_stress(s_x, s_y, mu_x, mu_y, sigma_x, sigma_y):
   print(s_x.tensor)
   print(s_y.tensor)
   print(mu_x)
   print(mu_y)
   print(sigma_x)
   print(sigma_y)
   sd_x = s_x.deviator
   sd_y = s_y.deviator
   alpha = ((sd_x@sd_x)*mu_x**2 + ((sd_x @ sd_y) + (sd_y @ sd_x)) * mu_x * mu_y
           + (sd_y @ sd_y) * mu_y**2).trace()
   fd_alpha_px = ((sd_x@sd_x)*2*mu_x + ((sd_x @ sd_y) + (sd_y @ sd_x)) * mu_y).trace()
   sd_alpha_px = ((sd_x@sd_x)*2).trace()
   fd_alpha_py = (((sd_x @ sd_y) + (sd_y @ sd_x)) * mu_x + (sd_y @ sd_y) * 2 * mu_y).trace()
   sd_alpha_py = ((sd_y @ sd_y) * 2).trace()

   s_vm = (3/2*alpha)**0.5
   fd_svm_px = (3/4) * (3/2*alpha)**-0.5 * fd_alpha_px
   sd_svm_px = ((3/4) * sd_alpha_px * (3/2 * alpha)**-0.5) + ((-9/16) * (3/2*alpha)**(-3/2) * fd_alpha_px**2)
   fd_svm_py = (3/4) * (3/2*alpha)**-0.5 * fd_alpha_py
   sd_svm_py = ((3/4) * sd_alpha_py * (3/2 * alpha)**-0.5) + ((-9/16) * (3/2*alpha)**(-3/2) * fd_alpha_py**2)
   print("***")
   print(fd_svm_px)
   print(sd_svm_px)
   print(fd_svm_py)
   print(sd_svm_py)
   print("***")

   E_svm = s_vm + (1.2) * (sd_svm_px * sigma_x**2 + sd_svm_py * sigma_y**2)
   sigma_svm = (((fd_svm_px * sigma_x)**2) + ((fd_svm_py * sigma_y)**2) + ((1/4) * ((sd_svm_px * sigma_x**2)**2 + 
               (sd_svm_py * sigma_y**2)**2)))**0.5
   return [E_svm, sigma_svm]

s_x = stress_tensor(2500,750,0,320,0,0)
s_y = stress_tensor(250,7500,0,210,0,0)
s = test_stress(s_x, s_y, 25, 150, 3.25, 19.5)
print(s)
print(s[0] / 1099645.66)
print(s[1] / 143863.39)

print(s_x.tensor)
