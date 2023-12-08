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

df = pd.read_excel(path_to_file)

df['day'] = df['date_time'].dt.date
df['hour'] = df['date_time'].dt.hour

try:
    m = gp.Model('main')

    ##########################
    ####### Parameters #######
    ##########################

    # Time params #
    t_delta = 60 #length of a timeframe in minutes
    #days = 1 #num of days to consider
    t_horizon = max(df.index)+1

    #model_t_to_day = {[t for t in range(t_horizon)][i]:np.array([day for day in range(days)]).repeat(24 * 60/t_delta)[i] for i in range(t_horizon)}
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
    cd = {products[i]:[0,1*rt_eff,-1,-1,-1,-1,1*rt_eff][i] for i in range(len(products))}  
    cd2 = {products[i]:[0,1,-1,-1,-1,-1,1][i] for i in range(len(products))}   #control if product is charge or discharge
    cd3 = {DA_products[i]:[0,1,-1,-1,-1,-1,1][i] for i in range(len(products))} 
    TP_eff = {products[i]:[0,1,1,0.1,0.1,0.2,0.2][i] for i in range(len(products))}    #TP efficiency for each product
    J = {s:TP_eff[s]*cd[s]*(t_delta/duration) for s in products}
    DAJ = {f'DA{k}': v for k,v in J.items()} #amount of energy added/removed for each product

    #is 20% TP accurate for ancillary market rates? how sensitive is market participation to this TP?


    # Prices #
    #just generate some data randomly, need to get this from an input file
    avgs = df.groupby(['day','hour'], as_index = False)[['i','c','d','spinr','suppr','regu','regd','DAi','DAc','DAd','DAspinr','DAsuppr','DAregu','DAregd']].aggregate('sum')
    avgs['date_time'] = pd.to_datetime(avgs.day) + avgs['hour'].apply(lambda x: pd.Timedelta(x,'hour'))
    t_horizon = max(avgs.index)+1
    P = avgs
    print(P.head)
    #DAP = pd.DataFrame(np.random.normal(loc = 1.0, scale =15,
    #                                    size = (avgs.shape[0],len(DA_products))),
    #                                    columns=DA_products)
    
    #P = pd.concat([P,DAP], axis = 1)


    # Model Params #
    m.Params.MIPGap = 0.005
    #todo: pull these parameters from an input file for easier control
    #      create a function to handle these intializations
    
    ##########################
    ### Decision Variables ###
    ##########################

    product = m.addVars(t_horizon,products ,vtype=GRB.BINARY,name = 'product')
    # product[1,'c'] = 0/1
    # products = ['i','c','d','spinr','suppr','regu','regd']

    DA_product = m.addVars(t_horizon,DA_products ,vtype=GRB.BINARY,name = 'DA_product')

    SoC = m.addVars(t_horizon, vtype = GRB.CONTINUOUS, name = 'SoC',
                    lb = soc_min,
                    ub = soc_cap)

    charge_amt = m.addVars(t_horizon,products ,vtype=GRB.CONTINUOUS,name = 'charge_amt')
    DA_charge_amt = m.addVars(t_horizon,DA_products ,vtype=GRB.CONTINUOUS,name = 'DA_charge_amt')

    ##########################
    ### Objective Function ###
    ##########################
    
    m.setObjective(gp.quicksum(P[j][i]*charge_amt[i,j] for i in range(t_horizon) for j in products)
                   + gp.quicksum(P[j][i]*DA_charge_amt[i,j] for i in range(t_horizon) for j in DA_products), 
                   GRB.MAXIMIZE)

    ##########################
    ###### Constraints #######
    ##########################

    # battery can only be in 1 product in a given timeframe
    m.addConstrs(
        (product.sum(t,'*') + DA_product.sum(t,'*') == 1 for t in range(t_horizon)), 
        name = 'single_assignment'
    )
    print('...charging only when product is active')
    m.addConstrs(
        (charge_amt[i,j] <= cd2[j]*J[j]*product[i,j] for i in range(t_horizon) for j in products),
        name = 'charge_when_product_up'
    )
    
    m.addConstrs(
        (DA_charge_amt[i,j] <= cd3[j]*DAJ[j]*DA_product[i,j] for i in range(t_horizon) for j in DA_products),
        name = 'DA_charge_when_product_up'
    )
    m.addConstr(
        SoC[0] == init_SOC + (gp.quicksum(cd2[k]*charge_amt[0,k] for k in products)
                            +gp.quicksum(cd3[k]*DA_charge_amt[0,k] for k in DA_products)),
        name = 'initial_state_of_charge'
    )
    #state of charge in time t is the state of charge in the previous time period + charge added/subtracted by product in time 
    print('...state_of_charge')
    m.addConstrs(
        (SoC[t] == SoC[t-1] + gp.quicksum(cd2[k]*charge_amt[t,k] for k in products)
        + gp.quicksum(cd3[k]*DA_charge_amt[t,k] for k in DA_products) for t in range(1,t_horizon)
        ),
        name = 'state_of_charge'
    )

    #only one full cycle per 24 hours
    for day in df.day.unique():
        expr_charge = gp.LinExpr()
        expr_discharge = gp.LinExpr()

        for t in range(t_horizon):
            if model_t_to_day[t] == day:
                expr_charge.addTerms(cd2['c'],charge_amt[t,'c'])
                expr_charge.addTerms(cd2['regd'],charge_amt[t,'regd'])
                expr_charge.addTerms(cd3['DAc'],DA_charge_amt[t,'DAc'])
                expr_charge.addTerms(cd3['DAregd'],DA_charge_amt[t,'DAregd'])

                expr_discharge.addTerms(cd2['d'],charge_amt[t,'d'])
                expr_discharge.addTerms(cd2['spinr'],charge_amt[t,'spinr'])
                expr_discharge.addTerms(cd2['suppr'],charge_amt[t,'suppr'])
                expr_discharge.addTerms(cd2['regu'],charge_amt[t,'regu'])
                expr_discharge.addTerms(cd3['DAd'],DA_charge_amt[t,'DAd'])
                expr_discharge.addTerms(cd3['DAspinr'],DA_charge_amt[t,'DAspinr'])
                expr_discharge.addTerms(cd3['DAsuppr'],DA_charge_amt[t,'DAsuppr'])
                expr_discharge.addTerms(cd3['DAregu'],DA_charge_amt[t,'DAregu'])


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
    #for day in df.day.unique():
    #    for hour in df.hour.unique():
    #        expr = gp.LinExpr()
    #        temp = df.index[(df['hour']==hour) & (df['day'] == day)]
    #        #print(temp)
    #        #print(len(temp))
    #        if(len(temp)>1):
    #            m.addConstrs(
    #                (
    #                gp.quicksum(DA_product[t,k] for t in temp[1:]) == (len(temp)-1)*DA_product[temp[0],k]
    #                        for k in DA_products),
    #                        name = 'commit_hour'
    #            )

    #need to create for discharging as well

    m.write(run_id + '_out.lp')
    m.optimize()

    #for v in m.getVars():
    #    if(abs(v.X)>0):
    #        print('%s %g' % (v.VarName, v.X))


    final_SOC = []
    final_product = []
    final_price = []
    final_charge_amt = []
    for t in avgs.index:
        final_SOC.append(SoC[t].X)
        for s in products:
            if product[t,s].X > 0.9:
                final_product.append(s)
                final_price.append(P[s][t])
                final_charge_amt.append(cd2[s]*charge_amt[t,s].X)
        for s in DA_products:
            if DA_product[t,s].X > 0.9:
                final_product.append(s)
                final_price.append(P[s][t])
                final_charge_amt.append(cd3[s]*DA_charge_amt[t,s].X)

    
    data = {
        'date_time':avgs['date_time'],
        'SoC':final_SOC,
        'product':final_product,
        'price': final_price,
        'charge_amt':final_charge_amt
    }
    final_SOC_df = pd.DataFrame(data)
    final_SOC_df.to_csv(Path(file_loc)/Path('output\\'+ run_id +'_output.csv'),
                        index = False)

except gp.GurobiError as e:
    print('Error code ' + str(e.errno) + ": " + str(e))

except AttributeError:
    print('Encountered an attribute error')