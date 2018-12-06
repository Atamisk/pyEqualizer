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
from copy import deepcopy
from numpy import array
import random
import lhsmdu
import math

# TODO: Safely Implement Location Based sampling. 
# Solutiuon outline: 
#1. Identify points of interest (Probably done manually by the designer. 
#2. Analyze the systems using nastran and a known (x only) load. 
#3. Analyze the systems using nastran and a known (y only) load. 
#4. Use the known load and the stress response to generate coefficients for x and y directions. 
#5. Using these coefficients as input to Dr. Hou's algorithm, find the variance and mean of the stress response given the mean and variance of the applied load. 


class Ind(object):
    def __init__(self, props, sys_num):
        self.props = props
        self.sys_num = sys_num
        self.fitness = -1000

    def to_array(self):
        return [self.props, self.fitness]

    @classmethod
    def from_array(cls, array, sys_num):
        obj = cls(array[0], sys_num)
        obj.fitness = array[1]
        obj.fitness_unconst = array[2]
        return obj

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
    fold_in_force(props, vec): Fold the force set for a system into the properyies for a generation. 

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

        TODO: Implement latin hypercube sampling
        """
        props = []
        lhs_exp = make_linear_map(0,250)
        lhs_vals = lhsmdu.sample(len(self.base_props),self.n_org,randomSeed=random.randint(0,2**32-1)).tolist()
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
        out = list(zip(props, fitness, fitness_unconst))
        return out

    def dummy_generation(self, last_vec):
        """
        Only analyze the supplied vector for fitness. 

        Commonly used when validating an individual against another load case.
        """
        last_props = [deepcopy(a.props) for a in last_vec]
        out = self.run_generation(deepcopy,last_props)
        return [Ind.from_array(a, self.sys_num) for a in out]

    def trial_generation(self, last_vec):
        #isolate the property vector from last generation
        last_props = [deepcopy(a.props) for a in last_vec]
        #run the crossover function to obtain trial vector with fitnesses. 
        trial_vec_basic = self.run_generation(self.crossover, last_props)
        trial_vec = [Ind.from_array(a, self.sys_num) for a in trial_vec_basic]
        return self.selection(last_vec, trial_vec)

    def first_generation(self):
        """first_generation(): Returns result of initial generation"""
        out = self.run_generation(self.gen_generation, [])
        return [Ind.from_array(a, self.sys_num) for a in out]




            

