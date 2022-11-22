import sys

'''
WRITE YOUR CODE BELOW.
'''
import random
from numpy import zeros, float32
#  pgmpy
import pgmpy
from pgmpy.models import BayesianModel
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
#You are not allowed to use following set of modules from 'pgmpy' Library.
#
# pgmpy.sampling.*
# pgmpy.factors.*
# pgmpy.estimators.*


def make_power_plant_net():
    """Create a Bayes Net representation of the above power plant problem. 
    Use the following as the name attribute: "alarm","faulty alarm", "gauge","faulty gauge", "temperature". (for the tests to work.)
    """

    # TODO: finish this function
    bayes_net = BayesianModel()

    nodes = ['temperature','faulty gauge','faulty alarm','gauge','alarm']
    edges = [('temperature','faulty gauge'),('temperature','faulty alarm'),('faulty gauge','gauge'),('gauge','alarm'),('faulty alarm','alarm')]

    bayes_net.add_nodes_from(nodes)
    bayes_net.add_edges_from(edges)

    #raise NotImplementedError
    return bayes_net


def set_probability(bayes_net):
    """Set probability distribution for each node in the power plant system.
    Use the following as the name attribute: "alarm","faulty alarm", "gauge","faulty gauge", "temperature". (for the tests to work.)
    """
    # TODO: set the probability distribution for each node


    cpd_t = TabularCPD('temperature', 2,  values=[[.8],[.2]])

    cpd_fggt =TabularCPD('faulty gauge', 2, values=[[.95, .20],[.05, .8]], evidence=['temperature'], evidence_card=[2])

    cpd_fagt =TabularCPD('faulty alarm', 2, values=[[.15, .85],[.85, .15]],  evidence=['temperature'], evidence_card=[2])

    cpd_ggfg = TabularCPD('gauge', 2, values=[[.05, .8],[.95, .2]], evidence=['faulty gauge'], evidence_card=[2])

    cpd_aggt = TabularCPD('alarm', 2, values=[[0.9, 0.8, 0.4, 0.85],[0.1, 0.2, 0.6, 0.15]], evidence=['gauge', 'faulty alarm'], evidence_card=[2,2])

    bayes_net.add_cpds(cpd_t, cpd_fagt, cpd_fggt, cpd_ggfg, cpd_aggt)

    #raise NotImplementedError
    return bayes_net


def get_alarm_prob(bayes_net):
    """Calculate the marginal 
    probability of the alarm 
    ringing in the 
    power plant system."""
    # TODO: finish this function

    solver = VariableElimination(bayes_net)

    marginal_prob = solver.query(variables=['alarm'], joint=False)

    alarm_prob = marginal_prob['alarm'].values

    #raise NotImplementedError
    return alarm_prob


def get_gauge_prob(bayes_net):
    """Calculate the marginal
    probability of the gauge 
    showing hot in the 
    power plant system."""
    # TODO: finish this function

    solver = VariableElimination(bayes_net)

    marginal_prob = solver.query(variables=['gauge'], joint=False)

    gauge_prob = marginal_prob['gauge'].values

    #raise NotImplementedError
    return gauge_prob


def get_temperature_prob(bayes_net):
    """Calculate the conditional probability 
    of the temperature being hot in the
    power plant system, given that the
    alarm sounds and neither the gauge
    nor alarm is faulty."""
    # TODO: finish this function

    solver = VariableElimination(bayes_net)

    conditional_prob = solver.query(variables=['temperature'], evidence={'gauge':0, 'faulty alarm':0, 'alarm':1}, joint=False)

    temp_prob = conditional_prob['temperature'].values

    #raise NotImplementedError
    return temp_prob


def get_game_network():
    """Create a Bayes Net representation of the game problem.
    Name the nodes as "A","B","C","AvB","BvC" and "CvA".  """

    # TODO: fill this out

    BayesNet = BayesianModel()

    nodes = ['A','B','C','AvB','BvC', 'CvA']
    edges = [('A','AvB'),('B','AvB'),('A','CvA'),('C','CvA'),('B','BvC'), ('C','BvC')]

    BayesNet.add_nodes_from(nodes)
    BayesNet.add_edges_from(edges)

    #raise NotImplementedError
    return BayesNet


def calculate_posterior(bayes_net):
    """Calculate the posterior distribution of the BvC match given that A won against B and tied C. 
    Return a list of probabilities corresponding to win, loss and tie likelihood."""
    posterior = [0,0,0]
    # TODO: finish this function

    A_cpd = TabularCPD('A', 4,  values=[[.15],[.45],[.3],[.1]])
    B_cpd = TabularCPD('B', 4,  values=[[.15],[.45],[.3],[.1]])
    C_cpd = TabularCPD('C', 4,  values=[[.15],[.45],[.3],[.1]])

    AvB_cpd = TabularCPD('AvB', 3, values=[[0.9, 0.7, 0.55, 0.85],[0.1, 0.3, 0.45, 0.15]], evidence=['A', 'B'], evidence_card=[2,2])
    AvC_cpd = TabularCPD('AvC', 3, values=[[0.9, 0.7, 0.55, 0.85],[0.1, 0.3, 0.45, 0.15]], evidence=['A', 'C'], evidence_card=[2,2])
    BvC_cpd = TabularCPD('BvC', 3, values=[[0.9, 0.7, 0.55, 0.85],[0.1, 0.3, 0.45, 0.15]], evidence=['B', 'C'], evidence_card=[2,2])

    bayes_net.add_cpds(A_cpd, B_cpd, C_cpd, AvB_cpd, AvC_cpd, BvC_cpd)

    solver = VariableElimination(bayes_net)

    posterior = solver.query(variables=['BvC'], evidence={'AvB':0, 'AvC':0}, joint=False)

    posterior = posterior['BvC'].values

    #raise NotImplementedError
    return posterior # list 


def Gibbs_sampler(bayes_net, initial_state):
    """Complete a single iteration of the Gibbs sampling algorithm 
    given a Bayesian network and an initial state value. 
    
    initial_state is a list of length 6 where: 
    index 0-2: represent skills of teams A,B,C (values lie in [0,3] inclusive)
    index 3-5: represent results of matches AvB, BvC, CvA (values lie in [0,2] inclusive)
    
    Returns the new state sampled from the probability distribution as a tuple of length 6.
    Return the sample as a tuple.    
    """
    #sample = tuple(initial_state)
    # TODO: finish this function

    v = ['A','B','C','AvB','AvC','CvB']

    s = {v_: s_ for v_, s_ in zip(v, sample)}

    num_of_samples = 6

    solver = VariableElimination(bayes_net)

    sol = list()

    for _ in range(num_of_samples):

        v__ = random.choice(v)

        c__ = {i: j for i, j in s.items() if i != v__}

        prob = solver.query(variables=[v__], evidence=c__, joint=False)

        prob = prob[v__].values

        sol.append(prob)

    sample = tuple(sol)

    #raise NotImplementedError
    return sample


def MH_sampler(bayes_net, initial_state):
    """Complete a single iteration of the MH sampling algorithm given a Bayesian network and an initial state value. 
    initial_state is a list of length 6 where: 
    index 0-2: represent skills of teams A,B,C (values lie in [0,3] inclusive)
    index 3-5: represent results of matches AvB, BvC, CvA (values lie in [0,2] inclusive)    
    Returns the new state sampled from the probability distribution as a tuple of length 6. 
    """

    #sample = tuple(initial_state)
    # TODO: finish this function
    sample = list()
    sig = 0.3
    r = 0
    for i in initial_state:
        i_p = np.random.normal(loc=i, scale=sig)
        a = np.exp(i_p**2) * (2+np.sin(i_p*5) + np.sin(i_p*2))
        b = np.exp(sig**2) * (2+np.sin(sig*5) + np.sin(sig*2))
        acc_p = a/b
        u = np.random.randn()
        if u <= acc_p:
            c = i_p
        else:
            c = i
            r += 1

        sample.append(c)
    sample = tuple(sample)
    #raise NotImplementedError
    return sample


def compare_sampling(bayes_net, initial_state):
    """Compare Gibbs and Metropolis-Hastings sampling by calculating how long it takes for each method to converge."""

    # TODO: finish this function

    t1 = time.now()
    g_ = Gibbs_sampler(bayes_net, initial_state)
    t2 = time.now()

    Gibbs_count = t2-t1

    t3 = time.now()
    mh_ = MH_sampler(bayes_net, initial_state)
    t4 = time.now()

    MH_count = t4-t3

    r = 0
    for i in mh_:
        if i not in initial_state:
            r += 1

    MH_rejection_count = r

    g = g_[-3:]
    Gibbs_convergence = g # posterior distribution of the BvC match as produced by Gibbs

    mh = mh_[-3:]
    MH_convergence = mh # posterior distribution of the BvC match as produced by MH

    #raise NotImplementedError
    return Gibbs_convergence, MH_convergence, Gibbs_count, MH_count, MH_rejection_count


def sampling_question():
    """Question about sampling performance."""
    # TODO: assign value to choice and factor

    options = ['Gibbs','Metropolis-Hastings']

    _, __, g, h,____ = compare_sampling(bayes_net, initial_state)

    if g > h:
        choice = 1
        factor = g/h
    else:
        choice = 0
        factor = h/g

    #raise NotImplementedError
    return options[choice], factor


def return_your_name():
    """Return your name from this function"""
    # TODO: finish this function


    #raise NotImplementedError
    return 'John Doe'
