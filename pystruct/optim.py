# *********************
# *     PyStruct      * 
# *********************
"""
 Module: Optim
  Purpose: Functions and Classes relating to running an optimization. 
  Author: Aaron Moore
  Date:   2018-07-12
"""
from pystruct.fileops import *
from pystruct.nr_var import *
from pystruct.math_utils import *
from copy import deepcopy
from numpy import array,trace
from multiprocessing.pool import Pool
import random
import math
import msslhs

class Ind(object):
    def __init__(self, props, sys_num):
        self.props = props
        self.sys_num = sys_num
        self._fitness = -1000

    def to_array(self):
        return [self.props, self.fitness]

    @classmethod
    def from_array(cls, array, sys_num):
        obj = cls(array[0], sys_num)
        obj._fitness = array[1]
        obj.fitness_unconst = array[2]
        return obj
    @property
    def fitness(self):
        return self._fitness

    def __str__(self):
        out = "***SYSTEM DEFINITION***"
        out = out + "\nProperties:\n{}".format(self.props)
        out = out + "\nSystem Number:\n{}".format(self.sys_num)
        out += "Cost:\n{}".format(self.fitness)
        out = out + "\n\n"
        return str(out)

def make_linear_map(low_limit, high_limit):
    """
    Build a function to convert a (0,1) range into an arbitrary space
    """
    return lambda x: low_limit + x * (high_limit - low_limit)

def make_normal_map(mu, sigma):
    """
    Build a function to convert a Z value into a normal
    distribution with the spefified characteristics.
    """
    return lambda x: mu + (sigma * x)


def get_plot_pts(vec):
    cost= [a.fitness_unconst for a in vec]
    return [array(a) for a in zip(*cost)]

def compare_all(trial, chall, comp):
    """
    dominates(trial, chall): Return true if the challenger dominates the trial organism
    """
    for t,c in zip(trial[1], chall[1]):
        if comp(t,c):
            return False
    return True

def dominates(trial, chall):
    return compare_all(trial, chall, lambda x,y: x < y)

def antidominates(trial, chall):
    return compare_all(trial, chall, lambda x,y: x > y)

def isolate_front(vec_ind, dom_func):
    """
    isolate_front(vec): Isolate and return a domination front from a 
                         given vector of organisms. truth controls which front you get. 
    """
    vec = [a.to_array() for a in vec_ind]
    def dominates_all(index):
        ind = list(range(len(vec)))
        ind.pop(index)
        for i in ind:
            if dom_func(vec[index], vec[i]) == True:
                return False
        return True
    front = []
    for i in range(len(vec)):
        if dominates_all(i):
            front.append(vec_ind[i])
    return front

def isolate_pareto(vec):
    """
    isolate_pareto(vec): Isolate and return the pareto front from a 
                         given vector of organisms.
    """
    return isolate_front(vec, dominates)

def isolate_antipareto(vec):
    return isolate_front(vec, antidominates)

def fold_in_force(props, force):
    """
    fold_in_force(props, vec): Fold the force set for a system into the properties for a generation. 

    Parameters: 
    props: A vector representing a generation of individuals. In effect, an array of arrays of property cards. 
    force: A set of FORCE card entries. 
    """
    props_cpy = deepcopy(props)
    for x in props_cpy:
        for y in force:
            x.append(y)
    return props_cpy


class system (object):
    """
    Class 'System'
    
    Contains the core data for a given optimization system. 
    Includes the base file, as well as information on the current generation being processed. 

    Properties:
    base_lines: The base input deck coded as an array of individual lines. 
    base_props: Array of information used to construct baseline property cards. 
    n_gen: Maximum number of generations to optimize for. 
    n_org: The number of organisms per generations. 
    fitness_funcs: a list of fitness functions that take a list of output file names as 
                   input and returns a list of fitness values. See examples in
                   __main__
    prefix: The file name prefix to use for making the input and output decks. 
    binary: The binary for "nastran" or your favorite compatible solver.
    """

    F = 0.1
    CR = 0.45

    def __init__(self, sys_num, fname, n_gen, n_org, 
            fitness_funcs, const_funcs, prefix = "/tmp/nastran/optim", binary = "/usr/bin/nastran", force = []):
        """ 
        Initialize the system class.
        
        Properties:
        fname: Base file name. 
        base_lines: The base input deck coded as an array of individual lines. 
        base_props: Array of information used to construct baseline property cards. 
        n_gen: Maximum number of generations to optimize for. 
        n_org: The number of organisms per generations. 
        fitness_funcs: a list of fitness functions that take a list of output file names as 
                       input and returns a list of fitness values. See examples in
                       __main__
        prefix: The file name prefix to use for making the input and output decks. 
        binary: The binary for "nastran" or your favorite compatible solver.
        """
        self.sys_num = sys_num
        self.__lines = load_from_file(fname)
        self.__base_lines = strip_force(strip_props(self.__lines))
        self.__base_props = read_properties(self.__lines)
        self.n_gen = n_gen
        self.__n_org = n_org
        self.__prefix = prefix
        self.__binary = binary
        self.fitness_funcs = fitness_funcs
        self.const_funcs = const_funcs

        if force == []:
            self.__base_force = read_force(self.__lines)
        else:
            self.__base_force = force
        

    @property
    def n_org(self):
        return deepcopy(self.__n_org)
    @n_org.setter
    def n_org(self,val):
        pass
    
    @property
    def base_props(self):
        return deepcopy(self.__base_props)
    @base_props.setter
    def base_props(self,val):
        pass

    @property
    def base_lines(self):
        return deepcopy(self.__base_lines)
    @base_lines.setter
    def base_lines(self,val):
        pass

    @property
    def prefix(self):
        return deepcopy(self.__prefix)
    @prefix.setter
    def prefix(self,val):
        pass
    
    @property
    def binary(self):
        return deepcopy(self.__binary)
    @binary.setter
    def binary(self,val):
        pass

    @property
    def base_force(self):
        return deepcopy(self.__base_force)
    @base_force.setter
    def base_force(self, val):
        pass

    def gen_generation(self, _):
        """
        gen_generation(): Randomly generate an array of property sets that
                          represent a generation of the current system. 

        """
        props = []
        lhs_exp = make_linear_map(0,250)
        lhs_vals = msslhs.sample(len(self.base_props),self.n_org,1)[0].transpose().tolist()
        for x in range(self.n_org):
            org = self.base_props
            for i in range(len(org)):
                org[i][3] = "{:.3f}".format(lhs_exp(lhs_vals[i][x]))
            props.append(org)
        return props

    def crossover(self, props):
        """
        crossover(props):   Cross over a series of cquad propery handles as a
                          part of differential evolution. 

        Arguments: 
        props: list of property arrays describing a series of 
               individual organisms. 

        Returns: 
        A list of properties that have been mutated and crossed over. 
        """
        props_out = []
        def pull_thicknesses(prop):
            #print(prop)
            return [float(n[3]) for n in prop]
        def push_thicknesses(prop,thk):
            p = deepcopy(prop)
            out = []
            for n in range(len(prop)):
                p[n][3] = "{:.3f}".format(thk[n])
            return p

        for i in range(len(props)): 
            #Select random indices to use in mutation
            indices = list(range(len(props)))
            _ = indices.pop(i)
            j = indices.pop(random.randrange(len(indices)))
            k = indices.pop(random.randrange(len(indices)))

            t_i  = pull_thicknesses(props[i])
            r1_i = pull_thicknesses(props[j])
            r2_i = pull_thicknesses(props[k])
            x_i = t_i
            vi_rhs = [self.F * (r1_i[n] - r2_i[n]) for n in range(len(x_i))]
            
            #create mutated vector -- abs taken because thickness cannot be negative in nastran. 
            v_i = [abs(y + z) for y, z in zip(x_i, vi_rhs)]
            
            u_i = [ x_i[n] if random.random() > self.CR else v_i[n] for n in range(len(x_i))]
            props_out.append(push_thicknesses(props[i],u_i))
        return props_out

    def selection(self, left, right):
        """
        selection(left,right): Given 2 property vectors, provide DE-style selection.

        Parameters: 
        left: Vector of properties and fitnesses. 
        right: ditto. 
        """
        outvec = []
        for l_i,r_i in zip(left,right):
            # test the fitness values (located at array index 1)
            # against one another to determine best choice for next
            # Generation
            l = l_i.to_array()
            r = r_i.to_array()
            if dominates(l,r):
                outvec.append(deepcopy(r_i))
            else:
                outvec.append(deepcopy(l_i))
        return outvec

    def get_fitness_vector(self, props, files):
        """
        Generate the standard fitness vector from a series of properties.
        """
        fitness_unconst = [[] for a in props]
        fitness = deepcopy(fitness_unconst)
        fitness_mults = [1 for a in props]
        for fn in self.fitness_funcs:
            res = fn(files)
            for x in range(len(files)):
                fitness_unconst[x].append(res[x])
        for fn in self.const_funcs:
            res = fn(files)
            for x in range(len(fitness)):
                fitness_mults[x] += res[x]
        for x in range(len(fitness)):
            fitness[x] = [ a * fitness_mults[x] for a in fitness_unconst[x] ]
        return [fitness, fitness_unconst]


    def run_generation(self, prop_func, last_props):
        """
        first_generation(): Run a single generation of the optimiser, stopping at the fitness function generation. 

        Arguments: 
        prop_func: Function that determines the population of the current generation. 
        last_props: props from the last generation. 
        """
        props = prop_func(last_props)
        files = multi_file_out(fold_in_force(props, self.__base_force), self.base_lines, self.prefix)
        run_nastran(self.binary, files)
        fitness, fitness_unconst = self.get_fitness_vector(props, files)
        out = [Ind.from_array(a, self.sys_num) for a in list(zip(props, fitness, fitness_unconst))]
        return out

    def dummy_generation(self, last_vec, ind_cls = Ind):
        """
        Only analyze the supplied vector for fitness. 

        Commonly used when validating an individual against another load case.
        """
        last_props = [deepcopy(a.props) for a in last_vec]
        out = self.run_generation(deepcopy,last_props)
        return out

    def trial_generation(self, last_vec):
        #isolate the property vector from last generation
        last_props = [deepcopy(a.props) for a in last_vec]
        #run the crossover function to obtain trial vector with fitnesses. 
        trial_vec = self.run_generation(self.crossover, last_props)
        return self.selection(last_vec, trial_vec)

    def first_generation(self, ind_cls = Ind):
        """first_generation(): Returns result of initial generation"""
        out = self.run_generation(self.gen_generation, [])
        return out


class tensor_ind(Ind):
    def __init__(self, props, sys_num, x_force, y_force, x_tensor, y_tensor, mass):
        super().__init__(props, sys_num)
        self._x_tensors = x_tensor
        self._y_tensors = y_tensor
        self.x_force = from_nas_real(x_force[0][5])  # Force used in making the tensors
        self.y_force = from_nas_real(y_force[0][6])  # Force used in making the tensors.
        self._mass = mass
        self.min_beta = -1
        self.target_elements = [100,106,219,220,221,222,277,301,575,711,712,713,744,745,824]

    def apply_stochastic_force(self, sto_force_x, sto_force_y):
        """
        Apply a stochasticallay defined force. 
        Inputs: 
            mu_x = mean of the x applied force
            mu_y = mean of the y applied force
            sigma_x = std.dev of the x applied force
            sigma_y = std.dev of the y applied force
        Outputs:
            out: State of stress at each tensor. 
        """
        mu_x, sigma_x = sto_force_x.list
        mu_y, sigma_y = sto_force_y.list
        out = []
        def get_stress_from_index(i):
           s_x = self.x_tensors[i] * (self.x_force**-1)
           s_y = self.y_tensors[i] * (self.y_force**-1)
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
           
        for i in self.target_elements:
            out.append(get_stress_from_index(i))
            
        return out

    def apply_force(self, x_appforce, y_appforce):
        """
        Make a combined stress tensor showing the efects of an applied force.
        """
        target_elements = self.target_elements
        out = []
        all_tensors = [[self.x_tensors[i], self.y_tensors[i]] for i in target_elements]
        for i in range(len(target_elements)):
            out.append( all_tensors[i][0]  * (x_appforce / self.x_force) + all_tensors[i][1] * (y_appforce / self.y_force))
        return out
    def strip_tensors(self):
        self._x_tensors = "STRIPPED"
        self._y_tensors = "STRIPPED"
    @property
    def x_tensors(self):
        return deepcopy(self._x_tensors)
    @property
    def y_tensors(self):
        return deepcopy(self._y_tensors)
    @property
    def mass(self):
        return deepcopy(self._mass)
    @property
    def fitness(self):
        return [self.mass, -1*self.min_beta]
    #  For right now, this is here for compatibility. Future 
    # releases will have these two values differ. 
    fitness_unconst = fitness
class system_unit(system):

    def __init__(self, sys_num, fname, n_gen, n_org, 
              x_force, y_force, sto_force_x, sto_force_y, prefix = "/tmp/nastran/optim", binary = "/usr/bin/nastran"):
        """
        Initializes the class with the passed in parameters. 
        
        Parameters:
        sys_num: Arbitrary system number for reporting purposes. 
        fname:   File name of the base NASTRAN input deck the analysis is based on. 
        n_gen:   Number of generations. 
        n_org:   number of organisms in each generation. 
        x_force: X force to apply when making individual unit tensors. 
        y_force: Y Force to apply when making individual unit tensors.
        sto_force_x: Stochasticaly defined x force. Provided as a nr_var object.
        sto_force_y: Stochasticaly defined y force. Provided as a nr_var object.
        prefix:  Prefix for nastran derived input decks. AKA scratch directory. 
        binary:  Location of the nastran binary
        force:   Actual applied force to the object under test. Presented as a NASTRAN input card. 
        """
        super().__init__(sys_num, fname, n_gen, n_org, 
            [], [], prefix, binary)
        self._x_force = x_force
        self._y_force = y_force
        self._sto_force_x = sto_force_x
        self._sto_force_y = sto_force_y

    @property
    def x_force(self):
        return self._x_force
    @property
    def y_force(self):
        return self._y_force
    
    @property
    def sto_force_x(self):
        return self._sto_force_x
    @property
    def sto_force_y(self):
        return self._sto_force_y

    @property
    def x_applied_force(self):
        return from_nas_real(self.base_force[0][5])

    @property
    def y_applied_force(self):
        return from_nas_real(self.base_force[0][6])

    #def get_fitness_vector(self, inds, files):
    #    pass
    def run_generation(self, prop_func, last_props):
        """
        run_generation(): Run a single generation of the optimiser, stopping at the fitness function generation. 

        Arguments: 
        prop_func: Function that determines the population of the current generation. 
        last_props: props from the last generation. 
        """
        props = prop_func(last_props)
        out = self.get_tensors_from_props(props)
        all_str = self.apply_forces(out)
        strength = nr_var(36000, 36000*0.13)
        for x in range(len(out)):
            out[x].min_beta = min([calc_beta(y, strength) for y in all_str[x]])
        return out

    def call_apply(self, inst, x, y):
        """
        Helper function to call a subordinate objects method
        "apply_force". 
        This is here because multiprocessing in Python is bizarre. 
        Yes, it's a hack. sue me. 
        """
        return inst.apply_stochastic_force(x,y)

    def apply_forces(self, inds):
        """
        Uses multithreading to apply loads to the unit-stress tensors
        in the individuals provided. 

        inputs: 
        inds: tensor_ind objects representing a series of designs. 

        outputs: 
        app: A nxm array of stress tensors, where n is the number of individuals in inds, 
             and m is the number of elements that data was requested from in each individual's 
             apply_force method. 
        """
        pool = Pool(8)
        args_to_pool = [ [x, self.sto_force_x, self.sto_force_y] for x in inds]
        app = pool.starmap(self.call_apply, args_to_pool)
        return app
        

    def get_tensors(self, inds):
        """
        Make the x and y tensors for a given list of individuals based on this system. 
        This function also sets the individual's mass. 
        inputs: 
            inds: List of input individuals
        outputs: 
            (return object): individuals with tensors stored in-class. 
        """
        props = [deepcopy(a.props) for a in inds]
        return self.get_tensors_from_props(props)

    def get_tensors_from_props(self, props):
        def run_tensor(force):
            files = multi_file_out(fold_in_force(props, force), self.base_lines, self.prefix)
            f06_names = [ a + ".out" for a in files ]
            run_nastran(self.binary, files)
        
            tensors = []
            for f in f06_names:
                stresses = stress_all_point(f)
                file_tensors = []
                for x in stresses:
                    file_tensors.append(stress_tensor(float(x[0]),float(x[1]), 0, float(x[2]), 0, 0))
                tensors.append(file_tensors)
            return [tensors, f06_names]
        # Get system mass.
        [x_tensors, _] = run_tensor(self.x_force)
        [y_tensors, f06_names] = run_tensor(self.y_force)
        masses = [mass(f) for f in f06_names]
        inds_with_tensors = []
        for i in range(len(props)):
            inds_with_tensors.append(tensor_ind(props[i], self.sys_num, self.x_force, self.y_force, x_tensors[i], y_tensors[i], masses[i]))
        return inds_with_tensors  

    def split_force_pack(self):
        force_1 = self.x_force
        print(force_1)
        return 1

    def test_and_print(self):
        """
        This subroutine is literally to be used as a testbed for the location based routines.
        """
        forces = self.split_force_pack()



            

