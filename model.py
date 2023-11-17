import gurobipy as gp
from gurobipy import GRB
import pandas as pd
import numpy as np
from pathlib import Path
import os
import datetime

file_loc = os.path.dirname(os.path.realpath(__file__))
file_name = '2022_nodal_input.xlsx'
path_to_file = Path(file_loc)/file_name

temp = datetime.datetime.now()
run_id = temp.strftime('%Y%m%d%H%M%S')

print('reading data')
df = pd.read_excel(path_to_file)

print('creating data columns')
df['day'] = df['date_time'].dt.date
df['hour'] = df['date_time'].dt.hour

print('sorting by date')
df= df.sort_values(by = 'date_time', ignore_index= True)
df = df.fillna(0)

try:
    print('Model initialized')
    m = gp.Model('main')

    ##########################
    ####### Parameters #######
    ##########################

    # Time params #
    print('creating parameters')
    t_delta = 5 #length of a timeframe in minutes
    #days = 1 #num of days to consider
    t_horizon = max(df.index)+1

    #model_t_to_day = {[t for t in range(t_horizon)][i]:np.array([day for day in range(days)]).repeat(24 * 60/t_delta)[i] for i in range(t_horizon)}
    #model_t_to_day = {[t for t in list(df.index)][i]:df['day'][i] for i in list(df.index)}
    model_t_to_day = df['day']
    #gives the corresponding day when given a model period t as the parameter
    #ex: model_t_to_day[7] to get the corresponding day of the 7th model period


    # Battery params #
    soc_cap = .95    #max battery charge
    soc_min = .2    #min batter charge
    rt_eff = .85    #round trip efficiency, todo: break this down into components based on Nathan's input
    duration = 360    #duration of battery in minutes, inputs will be in 1,2,4,6,8 hours

    #products the battery can be in
    ### in order they are idle, charge, discharge, spinning reserve,
    ### supplemental reserve, regulation up, regulation down
    products = ['i','c','d','spinr','suppr','regu','regd']
    DA_products = ['DAi','DAc','DAd','DAspinr','DAsuppr','DAregu','DAregd']

    # Charging params #
    init_SOC = .25      #state of charge prior to running model
    cd = {products[i]:[0,1*rt_eff,-1,-1,-1,-1,1*rt_eff][i] for i in range(len(products))}    #control if product is charge or discharge
    TP_eff = {products[i]:[0,1,1,0.1,0.1,0.2,0.2][i] for i in range(len(products))}    #TP efficiency for each product
    J = {s:TP_eff[s]*cd[s]*(t_delta/duration) for s in products}
    DAJ = {f'DA{k}': v for k,v in J.items()} #amount of energy added/removed for each product

    #is 20% TP accurate for ancillary market rates? how sensitive is market participation to this TP?


    # Prices #
    #just generate some data randomly, need to get this from an input file
    P = df
    np.random.seed(2023)
    #DAP = pd.DataFrame(np.random.normal(loc = 1.0, scale =15,
    #                                    size = (df.shape[0],len(DA_products))),
    #                                    columns=DA_products)
    #P = pd.concat([P,DAP], axis = 1)


    # Model Params #
    m.Params.MIPGap = 0.005
    #todo: pull these parameters from an input file for easier control
    #      create a function to handle these intializations
    
    ##########################
    ### Decision Variables ###
    ##########################
    print('creating decision variables')
    product = m.addVars(t_horizon,products ,vtype=GRB.BINARY,name = 'product')
    # product[1,'c'] = 0/1
    # products = ['i','c','d','spinr','suppr','regu','regd']

    DA_product = m.addVars(t_horizon,DA_products ,vtype=GRB.BINARY,name = 'DA_product')

    SoC = m.addVars(t_horizon, vtype = GRB.CONTINUOUS, name = 'SoC',
                    lb = soc_min,
                    ub = soc_cap)

    ##########################
    ### Objective Function ###
    ##########################
    print('setting objective function')
    m.setObjective(gp.quicksum(P[j][i]*product[i,j] for i in range(t_horizon) for j in products)
                   + gp.quicksum(P[j][i]*DA_product[i,j] for i in range(t_horizon) for j in DA_products), 
                   GRB.MAXIMIZE)

    ##########################
    ###### Constraints #######
    ##########################
    print('building constraints')
    # battery can only be in 1 product in a given timeframe
    print('...single_assignment')
    m.addConstrs(
        (product.sum(t,'*') + DA_product.sum(t,'*') == 1 for t in range(t_horizon)), 
        name = 'single_assignment'
    )

    #intial state of charge
    print('...initial_state_of_charge')
    m.addConstr(
        SoC[0] == init_SOC + (gp.quicksum(J[k]*product[0,k] for k in products)
                              +gp.quicksum(DAJ[k]*DA_product[0,k] for k in DA_products)),
        name = 'initial_state_of_charge'
    )

    #state of charge in time t is the state of charge in the previous time period + charge added/subtracted by product in time 
    print('...state_of_charge')
    m.addConstrs(
        (SoC[t] == SoC[t-1] + gp.quicksum(J[k]*product[t,k] for k in products)
         + gp.quicksum(DAJ[k]*DA_product[t,k] for k in DA_products) for t in range(1,t_horizon)
         ),
        name = 'state_of_charge'
    )

    #only one full cycle per 24 hours
    print('...charge cycle limit')
    for day in df.day.unique():
        expr_charge = gp.LinExpr()
        expr_discharge = gp.LinExpr()

        for t in range(t_horizon):
            if model_t_to_day[t] == day:
                expr_charge.addTerms(J['c'],product[t,'c'])
                expr_charge.addTerms(J['regd'],product[t,'regd'])
                expr_charge.addTerms(DAJ['DAc'],DA_product[t,'DAc'])
                expr_charge.addTerms(DAJ['DAregd'],DA_product[t,'DAregd'])

                expr_discharge.addTerms(J['d'],product[t,'d'])
                expr_discharge.addTerms(J['spinr'],product[t,'spinr'])
                expr_discharge.addTerms(J['suppr'],product[t,'suppr'])
                expr_discharge.addTerms(J['regu'],product[t,'regu'])
                expr_discharge.addTerms(DAJ['DAd'],DA_product[t,'DAd'])
                expr_discharge.addTerms(DAJ['DAspinr'],DA_product[t,'DAspinr'])
                expr_discharge.addTerms(DAJ['DAsuppr'],DA_product[t,'DAsuppr'])
                expr_discharge.addTerms(DAJ['DAregu'],DA_product[t,'DAregu'])


        m.addConstr(
            expr_charge <= 1,
            name = 'day %s charge cycle limit' % day
        )

        m.addConstr(
            expr_discharge >= -1,
            name = 'day %s discharge cycle limit' % day
        )

        #add day ahead products to cycle limit
        #150% potential in the future, depends on evaluation period. 150% might depend on how much time were solving for
    
    #if we participate in the day ahead, we must participate in the full hour
    print('...commit_hour for day ahead')
    for day in df.day.unique():
        for hour in df.hour.unique():
            expr = gp.LinExpr()
            temp = df.index[(df['hour']==hour) & (df['day'] == day)]
            #print(temp)
            #print(len(temp))
            if(len(temp)>1):
                m.addConstrs(
                    (
                    gp.quicksum(DA_product[t,k] for t in temp[1:]) == (len(temp)-1)*DA_product[temp[0],k]
                            for k in DA_products),
                            name = 'commit_hour'
                )

    #need to create for discharging as well
    print('all constraints complete')
    m.write(run_id + '_out.lp')
    print('beginning optimization')
    m.optimize()

    #for v in m.getVars():
    #    if(abs(v.X)>0):
    #        print('%s %g' % (v.VarName, v.X))

    print('outputting results')
    final_SOC = []
    final_product = []
    final_price = []
    for t in df.index:
        final_SOC.append(SoC[t].X)
        for s in products:
            if product[t,s].X > 0.9:
                final_product.append(s)
                final_price.append(P[s][t])
        for s in DA_products:
            if DA_product[t,s].X > 0.9:
                final_product.append(s)
                final_price.append(P[s][t])

    
    data = {
        'date_time':df['date_time'],
        'SoC':final_SOC,
        'product':final_product,
        'price': final_price
    }
    final_SOC_df = pd.DataFrame(data)
    final_SOC_df.to_csv(Path(file_loc)/Path('output\\'+ run_id +'_output.csv'),
                        index = False)

except gp.GurobiError as e:
    print('Error code ' + str(e.errno) + ": " + str(e))

except AttributeError:
    print('Encountered an attribute error')