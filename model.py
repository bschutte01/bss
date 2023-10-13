import gurobipy as gp
from gurobipy import GRB
import numpy as np
import pandas as pd
import scipy.sparse as sp
from pathlib import Path
import os

file_loc = os.path.dirname(os.path.realpath(__file__))
file_name = 'nodal_input.xls'
path_to_file = Path(file_loc)/file_name

df = pd.read_excel(path_to_file)
#what to do about blank values?

df['hour'] = df['date_time'].dt.hour

try:
    m = gp.Model('main')

    ##########################
    ####### Parameters #######
    ##########################

    # Time params #
    t_delta = 5 #length of a timeframe in minutes
    #days = 1 #num of days to consider
    t_horizon = max(df.index)

    #model_t_to_day = {[t for t in range(t_horizon)][i]:np.array([day for day in range(days)]).repeat(24 * 60/t_delta)[i] for i in range(t_horizon)}
    model_t_to_day = {[t for t in list(df.index)][i]:df['hour'][i] for i in list(df.index)}
    #gives the corresponding day when given a model period t as the parameter
    #ex: model_t_to_day[7] to get the corresponding day of the 7th model period


    # Battery params #
    soc_cap = .95    #max battery charge
    soc_min = .2    #min batter charge
    rt_eff = .85    #round trip efficiency, todo: break this down into components based on Nathan's input
    duration = 360    #duration of battery in minutes, inputs will be in 1,2,4,6,8 hours

    #states the battery can be in
    ### in order they are idle, charge, discharge, spinning reserve,
    ### supplemental reserve, regulation up, regulation down
    states = ['i','c','d','spinr','supr','regu','regd']

    # Charging params #
    init_SOC = .25      #state of charge prior to running model
    cd = {states[i]:[0,1*rt_eff,-1,-1,-1,-1,1*rt_eff][i] for i in range(len(states))}    #control if state is charge or discharge
    TP_eff = {states[i]:[0,1,1,0.1,0,0.2,0.2][i] for i in range(len(states))}    #TP efficiency for each state
    J = {s:TP_eff[s]*cd[s]*(t_delta/duration) for s in states} #amount of energy added/removed for each state

    # Prices #
    #just generate some data randomly, need to get this from an input file
    P = df

    # Model Params #
    m.Params.MIPGap = 0.005
    #todo: pull these parameters from an input file for easier control
    #      create a function to handle these intializations
    #      need prices
    
    
    ##########################
    ### Decision Variables ###
    ##########################

    state = m.addVars(t_horizon,states ,vtype=GRB.BINARY,name = 'state')
    # state[1,'c'] = 0/1
    # states = ['i','c','d','spinr','supr','regu','regd']

    SoC = m.addVars(t_horizon, vtype = GRB.CONTINUOUS, name = 'SoC',
                    lb = soc_min,
                    ub = soc_cap)

    ##########################
    ### Objective Function ###
    ##########################
    m.setObjective(gp.quicksum(P[j][i]*state[i,j] for i in range(t_horizon) for j in states), 
                   GRB.MAXIMIZE)

    ##########################
    ###### Constraints #######
    ##########################

    # battery can only be in 1 state in a given timeframe
    m.addConstrs(
        (state.sum(t,'*') == 1 for t in range(t_horizon)), 
        name = 'single_assignment'
    )

    #intial state of charge
    m.addConstr(
        SoC[0] == init_SOC + (gp.quicksum(J[k]*state[0,k] for k in states)),
        name = 'initial_state_of_charge'
    )

    #state of charge in time t is the state of charge in the previous time period + charge added/subtracted by state in time 
    m.addConstrs(
        (SoC[t] == SoC[t-1] + gp.quicksum(J[k]*state[t,k] for k in states) for t in range(1,t_horizon)),
        name = 'state_of_charge'
    )

    #only one full cycle per 24 hours
    for day in df.hour.unique():
        expr = gp.LinExpr()
        for t in range(t_horizon):
            if model_t_to_day[t] == day:
                expr.addTerms(J['c'],state[t,'c'])
                expr.addTerms(J['regd'],state[t,'regd'])
        m.addConstr(
            expr <= 1, #change to 1, can only charge 100% total in a day
            name = 'day %d cycle limit' % day
        )
    
    #exp1 = J['c']state[0,'c'] + J['regd'],state[0,'regd'] ... + J['c']state[286,'c'] + J['regd'],state[286,'regd'] <= 1
    #exp2 = J['c']state[287,'c'] + J['regd'],state[0,'regd'] ... + J['c']state[286,'c'] + J['regd'],state[600,'regd'] <= 1


    #need to create for discharging as well

    m.write('out.lp')
    m.optimize()

    for v in m.getVars():
        if(abs(v.X)>0):
            print('%s %g' % (v.VarName, v.X))

    # graphical representation of the state of charge
    # ggplot in python



except gp.GurobiError as e:
    print('Error code ' + str(e.errno) + ": " + str(e))

except AttributeError:
    print('Encountered an attribute error')