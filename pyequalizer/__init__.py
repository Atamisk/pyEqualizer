from pyequalizer.fileops import *
from pyequalizer.optim import *
from pyequalizer.regression import *
from pyequalizer.nr_var import *
from pyequalizer.nas_utils import *
from matplotlib.pyplot import ioff, savefig, subplots
from multiprocessing.pool import Pool
import sys, getopt
import random 
from time import time
import argparse
import msslhs
import pickle

def test_open_force_pack(f, n1,n2,n3):
    """
    Make a force pack compliant with the master's project force input. 
    inputs: 
        f: scale factor for the force vector. 
        n1,n2,n3: Components of the force vector (x,y,z)
    outputs: 
        pack: Force pack suitable for using with the force adding methods already generated. 
    """
    pack = [to_nas_force(2,918,0,f,n1,n2,n3)] 
    return pack

def cost_stress(files):
    """
    cost_stress(self, files): Categorize the organisms in a generation by cost based on max stress

    Arguments: 
    files: List of nastran input decks. 
    """
    return [max_cquad_stress(f + ".out") for f in files]

def cost_mass(files):
    """
    cost_mass(files): Categorize organisms by mass. 
    """
    return [mass(f+".out") for f in files]

def const_beta(files):
    """
    Categorize organisms by reliability index. 
    """
    def beta(stress):
        mu_strength = 250
        sigma_strength = 32.5
        return (mu_strength - stress) / ((sigma_strength) ** 2 + (0)**2)**0.5
    stresses = cost_stress(files)
    betas = [min(beta(st)-4, 0)*-10**4 for st in stresses]
    return betas

def const_mass(files):
    masses = cost_mass(files)
    ms_out = [min(1000 - x, 0) * -10**4 for x in masses]
    return(ms_out)

def converge_check_pareto_percentage(latest_front, last_front, latest_vec, i):
    threshold = 0.5
    percentage = len(latest_front)/len(latest_vec)
    print("CONVERGENCE PROGRESS: Threshold: {} Current: {}".format(threshold, percentage))
    converged = True if percentage >= threshold else False
    return [converged, i]

def converge_check_change_percentage(latest_front, last_front, latest_vec, ctr):
    threshold = 0.9
    counter_threshold = 10
    def intersection(lst1, lst2): 
        # Use of hybrid method 
        temp = set(lst2) 
        lst3 = [value for value in lst1 if value in temp] 
        return lst3 
    similarity = len(intersection(latest_front, last_front))
    sim_ratio = similarity/(len(latest_front)/2 + len(last_front) /2)
    print("CONVERGENCE PROGRESS: Threshold: {} Current: {}".format(threshold, sim_ratio))
    if sim_ratio >= threshold:
        if ctr >= counter_threshold:
            return [True, 0]
        else: 
            print("                      Over Threshold for {} generations.".format(ctr + 1))
            return [False, ctr + 1]
    else: 
        return [False, 0]


def uniform_random_force(base_forces, n):
    pre = msslhs.sample(len(base_forces)*2, n, 1)[0].transpose().tolist()
    lhs_exp = make_linear_map(0,104000)
    rand_vals = []
    for i in range(len(pre)):
        p = []
        for j in range(len(pre[i])):
            p.append(lhs_exp(pre[i][j]))
        rand_vals.append(p)
    return random_force_base(base_forces, rand_vals, n)

def normal_random_force(base_forces, n, mu_force, sigma_force, mu_angle, sigma_angle):
    rand_vals_norm = msslhs.sample(len(base_forces)*2, n, 1)[1].transpose().tolist()
    conv_force = make_normal_map(mu_force, sigma_force)
    conv_angle = make_normal_map(mu_angle, sigma_angle)
    rand_vals_out = [ [], [] ]
    for x in range(len(rand_vals_norm[0])):
        if conv_force(rand_vals_norm[0][x]) > 190514:
            vals = rnd_to_actual(conv_force(rand_vals_norm[0][x]), conv_angle(rand_vals_norm[1][x]))
            rand_vals_out[0].append(vals[0])
            rand_vals_out[1].append(vals[1])
    print("Of {} generated systems, {} meet the minimum force criterion.".format
            (n, len(rand_vals_out[0])))
    return random_force_base(base_forces, rand_vals_out, n)

def rnd_to_actual(force, angle):
    force_vert = force * math.cos(angle)
    force_horiz = force * math.sin(angle)
    return [force_horiz, -force_vert]

def random_force_base(base_forces, rand_vals, n):
    """
    random_force(base_forces, n): make an array of random force properties. 

    Parameters: 
    base_forces: The base forces as read from the original file. 
    n: number of random forces to generate. 
    """
    out_vec = []
    for i in range(len(rand_vals[0])):
        out_vec.append([])
        for j in range(len(base_forces)):
            out_vec[i].append(deepcopy(base_forces[j]))
            for k in range(5,7):
                val = rand_vals[2*j + k-5][i]
                out_vec[i][j][k] = to_nas_real(val)
    return out_vec


def plot_inds(ax, inds, lab):
    """
    Plot sets of individuals in a standard way.
    """
    [pt_x, pt_y] = get_plot_pts(inds)
    ax.scatter(pt_x, pt_y, label=lab)
    ax.set_xlabel('Weight(kg)');
    ax.set_ylabel('Beta(unitless)');

def plot_with_front(gen, front, title, fname):
    """
    plot with front: Print the generation gen and front, 
    highlighting front as the pareto front on the graph. 

    Parameters: 
    gen: The generation to plot. 
    front: The pareto front extracted from generation gen
    title: Plot Title
    fname: path to output file for plot image. 
    """
    fig, ax = subplots()
    plot_inds(ax,gen,'Non-Dominant')
    plot_inds(ax,front,'Dominant')
    ax.set_title(title)
    ax.legend()
    fig.savefig(fname)
    return [fig, ax]

def no_validate(inds, val_force, fname, max_wt, max_stress):
    """
    Do not perform FEM validation. Used on solutions with more traditional constraints. 
    """
    return inds

def validate_inds(inds, val_force, fname, max_wt, max_stress):
    val_sys = system(99,fname,1,0,[cost_mass, cost_stress], [const_beta], force = val_force)
    val_inds = val_sys.dummy_generation(inds)
    valid_designs = []
    for x in range(len(val_inds)):
        if val_inds[x].fitness[0] < max_wt and val_inds[x].fitness[1] < max_stress:
            valid_designs.append(inds[x])
    return valid_designs

def linspace(lower, upper, length):
    return [lower + x*(upper-lower)/(length-1) for x in range(length)]

def print_ind(ind):
    """
    Compactly print an individual.
    """
    print("  Parent Load Case:\t\t\t{}".format(ind.sys_num))
    print("  Design Spec:")
    print("    Top Flange Width:\t\t\t{:4.2e} mm".format(float(ind.props[0][3])))
    print("    Bottom Flange Width:\t\t{:4.2e} mm".format(float(ind.props[1][3])))
    print("    Web Thickness:\t\t\t{:4.2e} mm".format(float(ind.props[2][3])))
    print("    Doubler Thickness at Hoist Pin:\t{:4.2e} mm".format(float(ind.props[3][3])))
    print("    Doubler Thickness at Load Pin:\t{:4.2e} mm".format(float(ind.props[4][3])))
    print("  Fitness:")
    print("    Max Stress:\t\t\t\t{:4.2e} MPa".format(float(ind.fitness_unconst[1])))
    print("    Weight:\t\t\t\t{:4.2e} kg".format(float(ind.fitness_unconst[0])))

def print_ind_csv(ind):
    """
    Compactly print an individual.
    """
    def prcomma(inStr):
        print(inStr, end=',')
    prcomma((ind.sys_num))
    prcomma((float(ind.props[0][3])))
    prcomma((float(ind.props[1][3])))
    prcomma((float(ind.props[2][3])))
    prcomma((float(ind.props[3][3])))
    prcomma((float(ind.props[4][3])))
    prcomma((float(ind.fitness_unconst[1])))
    print((float(ind.fitness_unconst[0])))


def print_load_case(sys,case_num):
    lc = sys.base_force
    print("Load Case {}".format(case_num))
    print("  X Force:\t{:4.2e}".format(from_nas_real(lc[0][5])))
    print("  Y Force:\t{:4.2e}".format(from_nas_real(lc[0][6])))


def parseargs():
    """
    Helper function to handle argument parsing
    """
    try:
        parser = argparse.ArgumentParser(description='Differentially optimize a better spreader beam')
        parser.add_argument('--n_gen', '-g', type=int, default=5, 
                help='Number of Simulation Generations')
        parser.add_argument('--n_ind', '-i', type=int, default=15, 
                help='Number of Simulation Individuals') 
        parser.add_argument('--n_sys', '-s', type=int, default=5, 
                help='Number of Simulation Systems')
        parser.add_argument('--max_wt', '-w', type=int, default=1000, 
                help='Maximum Weight Desired' )
        parser.add_argument('--max_stress', '-t', type=int, default=95.3, 
                help='Maximum Stress Desired')
        parser.add_argument('--special' ,'-S' ,help='Perform special case NUM', type=int)
        parser.add_argument('--convergence', '-C', help='Add convergence check NUM to the algorithm. Supported values: 1-pareto percentage convergence', type=int)
        parser.add_argument('--csv',default=False, action='store_true', 
                help='Output final systems as a CSV file.')
        parser.add_argument('fname') 
        args = parser.parse_args()
        return args
    except:
        raise()

def gen_case(args, force_func, val_func):
    N_GEN = args.n_gen           # Number of generations per system. 
    N_IND = args.n_ind           # Number of individuals per system. 
    N_SYS = args.n_sys           # Number of systems. 
    MAX_WT = args.max_wt         # Max Weight
    MAX_STRESS = args.max_stress # Max Stress
    fname = args.fname

    # Pull force parameters to randomize
    file_lines = load_from_file(fname)
    starting_force = read_force(file_lines)
    #Generate random forces
    force_packs = force_func(starting_force, N_SYS)
    
    systems = [system(x,fname, 1,N_IND, [cost_mass, cost_stress], 
        [const_beta, const_mass], force = force_packs[x]) for x in range(len(force_packs))]

    all_front = optimize_systems(systems, N_GEN, converged_func = args.conv_func)
    val_func_closed = lambda x: val_func(x, starting_force, fname, MAX_WT, MAX_STRESS)
    if (args.csv == False):
        prepare_report_pretty(all_front, val_func_closed, systems)
    else:
        prepare_report_csv(all_front, val_func_closed, systems)

def optimize_systems(systems, N_GEN, compact=False, converged_func= lambda a,b,c,d: False):
    """
    Main optimization loop for the program. 
    Inputs:
      systems   -- List of pyequalizer.optim.system objects that make up the load cases to be analyzed.
      N_GEN     -- Number of generations to run each optimization for.
    Output: 
      all_front -- A sorted list of pareto fronts from each system, presented as an
                   array of arrays of pyequalizer.optim.Ind objects. 
    """
    def gen_loop(x,i, last_vec):
        print("Generation {} in system {} starting at T+ {:.3f}".format(i, x, time()-start_time))
        latest_vec = main_sys.trial_generation(last_vec)
        latest_props = [a.props for a in latest_vec]
        latest_cost = [a.fitness[0] for a in latest_vec]
        min_cost = min(latest_cost)

        #If any cost values come out to be zero, complain. 
        if min(latest_cost) == 0:
            s = "One or more cost values are zero.\n"
            min_index = latest_cost.index(min_cost)
            min_prop = latest_props[min_index]
            s += "At index {}\n".format(min_index)
            for x in min_prop:
                s += "{}\n".format(str(x))
            raise ValueError(s)
        print("Generation {} in system {} complete at T+ {:.3f}\n".format(i, x, time()-start_time))
        return latest_vec
    print("Analysis Started.")
    start_time = time()
    all_front = []
    #Main analysis loop.
    for x in range(len(systems)):
        main_sys = systems[x]
        latest_vec = main_sys.first_generation()
        last_front = []
        latest_front = isolate_pareto(latest_vec)
        ctr = 0
        for i in range(N_GEN):
            last_front = latest_front
            latest_vec = gen_loop(x,i,latest_vec)
            latest_front = isolate_pareto(latest_vec)
            converged, ctr = converged_func(latest_front, last_front, latest_vec, ctr)
            if converged:
                print("Convergence Achieved")
                break
        #Plot results of this system
        front = latest_front
        fig , ax = plot_with_front(latest_vec, front, 'System {}'.format(str(x)) 
                ,'/tmp/output_sys_' + str(x) + '.png')
        with open('/tmp/output_sys_' + str(x) + '.pickle', 'wb') as f:
            pickle.dump(fig,f)
            pickle.dump(ax,f)
        if (compact):
            for x in front:
                x.strip_tensors()
        all_front.append(front)
    return all_front

def prepare_report_pretty(all_front, val_func, systems):
    prepare_report(all_front, val_func, systems, print_pretty)
def prepare_report_csv(all_front, val_func, systems):
    prepare_report(all_front, val_func, systems, print_csv)


def prepare_report(all_front, val_func, systems, print_func):
    #Gather all fronts combined. 
    all_front_mixed = []
    for x in all_front:
        for y in x:
            all_front_mixed.append(y)

    # Generate plot object
    fig, ax = subplots()

    #Validate all optimal designs against the maximum load 
    valid_designs = val_func(all_front_mixed)

    #Generate final selected designs from all paretos. 
    final_front = isolate_pareto(valid_designs)
    
    #Plot each pareto front from each system individually. 
    for x in all_front:
        x_x, x_y = get_plot_pts(x)
        ax.scatter(x_x, x_y, label='System {}'.format(x[0].sys_num))
    #Highlight valid designs
    vd_x, vd_y = get_plot_pts(valid_designs)
    ax.scatter(vd_x, vd_y, label='Valid Designs')
    #Plot final selected designs.
    ff_x, ff_y = get_plot_pts(final_front)
    ax.scatter(ff_x, ff_y, label='Pareto Front')

    #Plot axis labels
    ax.legend()
    ax.set_title('All Fronts')
    ax.set_xlabel('Weight(kg)');
    ax.set_ylabel('Beta(unitless)');
    #Save Plot
    fig.savefig('/tmp/all_fronts.png')

    #Summary!
    print("Program Complete.")
    print("\nLoad Case Summary:")
    print(  "----------------------")
    for x in range(len(systems)):
        print_load_case(systems[x],x)
    print("\nSummary of valid designs:")
    print(  "--------------------------")
    print_func(final_front)

def print_pretty(final_front):
    for x in range(len(final_front)):
        print("System {}".format(x))
        print_ind(final_front[x])
def print_csv(final_front):
    print("Parent Load Case,Top Flange Width,Bottom Flange Width,Web Thickness," + 
            "Doubler Thickness at Hoist Pin,Doubler Thickness at Load Pin,Fitness 1, Fitness 2")
    for x in range(len(final_front)):
        print_ind_csv(final_front[x])

def det_run(args):
    """
    Perform a "Baseline" optimization for the system under study. 
    Parameters:
    args: Argument object generated by argparse. See the parseargs
          function for details. 
    """
    print("STARTING DETERMINISTIC RUN")
    if (args.n_gen < 100):
        print("NOTE: Minimum generations for det run is 100.")
        args.n_gen = 100
    if (args.n_ind < 50):
        print("NOTE: Minimum individuals for det run is 50.")
        args.n_ind = 50
    def dummy_force(sf, _):
        """
        dummy function to return the given force object back to the caller
        """
        dum_vals = [[0],[-150000]]
        return random_force_base(sf, dum_vals, 1)
    gen_case(args, dummy_force, no_validate)
    # This is the deterministic run. For this one, we are optimizing 
    # using the base force that comes with the nastran file, as i have
    # ensure tht this value is the "test case" for the problem. 

def dwu_run(args):
    print("DWU Run selected.")
    args_out = deepcopy(args)
    args_out.n_sys = 1000
    nrf_closed = lambda x,y: normal_random_force(x, y, 150000, 20670, 0, 0.087)
    gen_case(args_out, nrf_closed, no_validate)


def loc_run(args):
    print("LOCATION RUN SELECTED")
    N_GEN = args.n_gen           # Number of generations per system. 
    N_IND = args.n_ind           # Number of individuals per system. 
    N_SYS = args.n_sys           # Number of systems. 
    MAX_WT = args.max_wt         # Max Weight
    MAX_STRESS = args.max_stress # Max Stress
    x_force = test_open_force_pack(1,1000,0,0)
    y_force = test_open_force_pack(1,0,1000,0)
    sto_force_x = nr_var(0,5000)
    sto_force_y = nr_var(150000,19500)
    fname = args.fname

    # Pull force parameters to randomize
    file_lines = load_from_file(fname)
    starting_force = read_force(file_lines)
    
    systems = [system_unit(1,fname, 1,N_IND,  
               x_force, y_force, sto_force_x, sto_force_y)]
    all_front = optimize_systems(systems, N_GEN, converged_func=args.conv_func)
    val_closed = lambda x: no_validate(x,[],[],[],[])
    if (args.csv == False):
        prepare_report_pretty(all_front, val_closed, systems)
    else:
        prepare_report_csv(all_front, val_closed, systems)
def main():
    args = parseargs()
    if args.convergence:
        conv_funcs = {
                1: converge_check_pareto_percentage,
                2: converge_check_change_percentage
                }
        args.conv_func  = conv_funcs[args.convergence]
    else:
        args.conv_func = lambda a,b,c,d:False
    if args.special:
        cases = {
                1: det_run,
                2: dwu_run,
                3: loc_run
                }
        cases[args.special](args)
    else:
        gen_case(args, uniform_random_force, validate_inds)



