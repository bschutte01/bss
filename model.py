import gurobipy as gp
from gurobipy import GRB
import pandas as pd
import numpy as np
from pathlib import Path
import os

file_loc = os.path.dirname(os.path.realpath(__file__))
file_name = 'nodal_input.xls'
path_to_file = Path(file_loc)/file_name

df = pd.read_excel(path_to_file)
#what to do about blank values?

df['day'] = df['date_time'].dt.date
df['hour'] = df['date_time'].dt.hour

try:
    m = gp.Model('main')

    ##########################
    ####### Parameters #######
    ##########################

    # Time params #
    t_delta = 5 #length of a timeframe in minutes
    #days = 1 #num of days to consider
    t_horizon = max(df.index)+1

    #model_t_to_day = {[t for t in range(t_horizon)][i]:np.array([day for day in range(days)]).repeat(24 * 60/t_delta)[i] for i in range(t_horizon)}
    model_t_to_day = {[t for t in list(df.index)][i]:df['day'][i] for i in list(df.index)}
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
    DA_states = ['DAi','DAc','DAd','DAspinr','DAsupr','DAregu','DAregd']

    # Charging params #
    init_SOC = .25      #state of charge prior to running model
    cd = {states[i]:[0,1*rt_eff,-1,-1,-1,-1,1*rt_eff][i] for i in range(len(states))}    #control if state is charge or discharge
    TP_eff = {states[i]:[0,1,1,0.1,0,0.2,0.2][i] for i in range(len(states))}    #TP efficiency for each state
    J = {s:TP_eff[s]*cd[s]*(t_delta/duration) for s in states}
    DAJ = {f'DA{k}': v for k,v in J.items()} #amount of energy added/removed for each state

    #is 20% TP accurate for ancillary market rates? how sensitive is market participation to this TP?


    # Prices #
    #just generate some data randomly, need to get this from an input file
    P = df
    DAP = pd.DataFrame(np.random.normal(loc = 1.0, scale =15,
                                        size = (df.shape[0],len(DA_states))),
                                        columns=DA_states)
    P = pd.concat([P,DAP], axis = 1)


    # Model Params #
    m.Params.MIPGap = 0.005
    #todo: pull these parameters from an input file for easier control
    #      create a function to handle these intializations
    
    ##########################
    ### Decision Variables ###
    ##########################

    state = m.addVars(t_horizon,states ,vtype=GRB.BINARY,name = 'state')
    # state[1,'c'] = 0/1
    # states = ['i','c','d','spinr','supr','regu','regd']

    DA_state = m.addVars(t_horizon,DA_states ,vtype=GRB.BINARY,name = 'DA_state')

    SoC = m.addVars(t_horizon, vtype = GRB.CONTINUOUS, name = 'SoC',
                    lb = soc_min,
                    ub = soc_cap)

    ##########################
    ### Objective Function ###
    ##########################
    m.setObjective(gp.quicksum(P[j][i]*state[i,j] for i in range(t_horizon) for j in states)
                   + gp.quicksum(P[j][i]*DA_state[i,j] for i in range(t_horizon) for j in DA_states), 
                   GRB.MAXIMIZE)

    ##########################
    ###### Constraints #######
    ##########################

    # battery can only be in 1 state in a given timeframe
    m.addConstrs(
        (state.sum(t,'*') + DA_state.sum(t,'*') == 1 for t in range(t_horizon)), 
        name = 'single_assignment'
    )

    #intial state of charge
    m.addConstr(
        SoC[0] == init_SOC + (gp.quicksum(J[k]*state[0,k] for k in states)
                              +gp.quicksum(DAJ[k]*DA_state[0,k] for k in DA_states)),
        name = 'initial_state_of_charge'
    )

    #state of charge in time t is the state of charge in the previous time period + charge added/subtracted by state in time 
    m.addConstrs(
        (SoC[t] == SoC[t-1] + gp.quicksum(J[k]*state[t,k] for k in states)
         + gp.quicksum(DAJ[k]*DA_state[t,k] for k in DA_states) for t in range(1,t_horizon)
         ),
        name = 'state_of_charge'
    )

    #only one full cycle per 24 hours
    for day in df.day.unique():
        expr_charge = gp.LinExpr()
        expr_discharge = gp.LinExpr()

        for t in range(t_horizon):
            if model_t_to_day[t] == day:
                expr_charge.addTerms(J['c'],state[t,'c'])
                expr_charge.addTerms(J['regd'],state[t,'regd'])
                expr_charge.addTerms(DAJ['DAc'],DA_state[t,'DAc'])
                expr_charge.addTerms(DAJ['DAregd'],DA_state[t,'DAregd'])

                expr_discharge.addTerms(J['d'],state[t,'d'])
                expr_discharge.addTerms(J['spinr'],state[t,'spinr'])
                expr_discharge.addTerms(J['supr'],state[t,'supr'])
                expr_discharge.addTerms(J['regu'],state[t,'regu'])
                expr_discharge.addTerms(DAJ['DAd'],DA_state[t,'DAd'])
                expr_discharge.addTerms(DAJ['DAspinr'],DA_state[t,'DAspinr'])
                expr_discharge.addTerms(DAJ['DAsupr'],DA_state[t,'DAsupr'])
                expr_discharge.addTerms(DAJ['DAregu'],DA_state[t,'DAregu'])


        m.addConstr(
            expr_charge <= 1,
            name = 'day %s charge cycle limit' % day
        )

        m.addConstr(
            expr_discharge >= -1,
            name = 'day %s discharge cycle limit' % day
        )

        #add day ahead states to cycle limit
        #150% potential in the future, depends on evaluation period. 150% might depend on how much time were solving for
    
    #if we participate in the day ahead, we must participate in the full hour
    for day in df.day.unique():
        for hour in df.hour.unique():
            expr = gp.LinExpr()
            temp = df.index[(df['hour']==hour) & (df['day'] == day)]
            #print(temp)
            #print(len(temp))
            if(len(temp)>1):
                m.addConstrs(
                    (
                    gp.quicksum(DA_state[t,k] for t in temp[1:]) == (len(temp)-1)*DA_state[temp[0],k]
                            for k in DA_states),
                            name = 'commit_hour'
                )

    #need to create for discharging as well

    m.write('out.lp')
    m.optimize()

    #for v in m.getVars():
    #    if(abs(v.X)>0):
    #        print('%s %g' % (v.VarName, v.X))


    final_SOC = []
    final_state = []
    final_price = []
    for t in df.index:
        final_SOC.append(SoC[t].X)
        for s in states:
            if state[t,s].X > 0.9:
                final_state.append(s)
                final_price.append(P[s][t])
        for s in DA_states:
            if DA_state[t,s].X > 0.9:
                final_state.append(s)
                final_price.append(P[s][t])

    
    data = {
        'date_time':df['date_time'],
        'SoC':final_SOC,
        'state':final_state,
        'price': final_price
    }
    final_SOC_df = pd.DataFrame(data)
    final_SOC_df.to_csv(Path(file_loc)/Path('output\\output.csv'),
                        index = False)

except gp.GurobiError as e:
    print('Error code ' + str(e.errno) + ": " + str(e))

except AttributeError:
    print('Encountered an attribute error')