'''
Monte Carlo Simulation Benchmark
Aaron Moore 
2019-04-14

Purpose: Validate the taylor series approximation of the mean and variance of 
         the Von Mises Stress of a given piece of material. This is done 
         using Mone Carlo Simulation to generate benchmark values.
'''
from pystruct.nr_var import nr_var
from pystruct.stress_tensor import stress_tensor
from numpy.random import normal
from numpy import mean, std
from scipy.stats import norm
# Establish global system variables: 
n = int(1.5e6) #Number of monte carlo Samples. 

U_x_px = 2500
U_y_px = 750
U_xy_px = 320

U_x_py = 250
U_y_py = 7500
U_xy_py = 210

def mc_vm_stress(px, py):
    s_x = U_x_px * px + U_x_py * py
    s_y = U_y_px * px + U_y_py * py
    s_xy = U_xy_px * px + U_xy_py * py
    return (s_x**2 - s_x*s_y + s_y**2 + 3 * s_xy**2)**0.5


# Establish the Random Variables: 
mean_px = 150
std_px = 19.5
px = nr_var(mean_px, std_px)

mean_py = 25
std_py = 3.25
py = nr_var(mean_py, std_py)

# Get the sample vectors to operate on: 
mc_vec_px = normal(px.mu, px.sigma, n)
mc_vec_py = normal(px.mu, px.sigma, n)
mc_vec_s_vm = [mc_vm_stress(mc_vec_px[i], mc_vec_py[i]) for i in range(n)]

#Run the sim
mc_mean = mean(mc_vec_s_vm)
mc_std = std(mc_vec_s_vm)
print("Mean according to MC Simulation:          {}".format(mc_mean))
print("St. Deviation according to MC Simulation: {}".format(mc_std))
           
def get_taylor_vm(px, py):
    mu_x, sigma_x = px.list
    mu_y, sigma_y = py.list
    s_x = stress_tensor(U_x_px, U_y_px, 0, U_xy_px,0,0)
    s_y = stress_tensor(U_x_py, U_y_py, 0, U_xy_py,0,0)
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

    E_svm = s_vm + (1.2) * (sd_svm_px * sigma_x**2 + sd_svm_py * sigma_y**2)
    sigma_svm = (((fd_svm_px * sigma_x)**2) + ((fd_svm_py * sigma_y)**2) + ((1/4) * ((sd_svm_px * sigma_x**2)**2 + 
                       (sd_svm_py * sigma_y**2)**2)))**0.5
    return nr_var(E_svm, sigma_svm)

taylor_stats = get_taylor_vm(px,py)
print("Mean according to Taylor Approximation:          {}".format(taylor_stats.mu))
print("St. Deviation according to Taylor Approximation: {}".format(taylor_stats.sigma))
print(mc_mean/taylor_stats.mu)
