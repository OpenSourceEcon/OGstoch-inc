# -*- coding: utf-8 -*-
"""
Created on Mon Jul 10 11:12:52 2017

@author: jb97
"""

import matplotlib.pyplot as plt
import numpy as np # contains interpolation as "interp"
from scipy.optimize import minimize # for optimization

def solve_equilibrium():
    
    capital_share = 0.35
    depreciation  = 0.05
    update_factor = 0.35
    
    capital_to_labor_ratio = 5.55954
    tolerance              = 10e-5
    error                  = tolerance+10
    iteration              = 0
    
    while error>tolerance:
        iteration += 1
        
        rate = capital_share*((capital_to_labor_ratio)**(capital_share-1)) - depreciation
        wage = (1-capital_share)*(capital_to_labor_ratio**capital_share)
        
        rates = rate*np.ones(80)
        wages = wage*np.ones(80)
        [aggregate_capital, aggregate_effective_labor, aggregate_consumption, aggregate_labor, aggregete_productivity] = solve_cohort(0, rates, wages)
        
        total_capital          = sum(aggregate_capital)
        total_effective_labor  = sum(aggregate_effective_labor  )
        k_lpr                  = total_capital/total_effective_labor
        error                  = abs(k_lpr - capital_to_labor_ratio)
        capital_to_labor_ratio = update_factor*k_lpr + (1-update_factor)*capital_to_labor_ratio
        
        total_output           = (total_capital**capital_share)*((total_effective_labor)**(1-capital_share))
        total_consumption      = sum(aggregate_consumption)
        goods_market_error     = total_output - total_consumption - depreciation*total_capital
         
        print('The good market error is '     , goods_market_error     )
        print('The current interest rate is ' , rate                   )
        print('The current wage is '          , wage                   )
        print('The error is '                 , error                  )
        print('The iteration is '             , iteration              )
        print('The capital to labor ratio is ', capital_to_labor_ratio )
    
    print('Equilibrium aggregate capital is '    , total_capital         )
    print('Equilibrium aggregate labor is '      , total_effective_labor )
    print('Equilibrium aggregate consumption is ', total_consumption     )
    print('Equilibrium aggregate output is '     , total_output          )
    print('Equilibrium interest rate is '        , rate                  )
    print('Equilibrium wage is '                 , wage                  )
    plt.plot(aggregate_capital)

def solve_cohort( birth_year, rates, wages ):
    
    # Unloading the global parameter structure
    n_assets            = 20
    asset_lower_bound   = .001
    asset_upper_bound   = 2000
    assets              = np.logspace(np.log(asset_lower_bound)/np.log(10), np.log(asset_upper_bound)/np.log(10), num=n_assets, base=10.0)
    productivity_shocks = np.array([.25, 1.25, 2])
    n_prodshocks        = len(productivity_shocks)
    trans_probs         = np.matrix([[.75,.20,.05],[.20,.60,.20],[.05,.20,.75]])
    max_age             = 80
    survival_probabilities = np.ones(max_age)

    
    # Determining effective region of price vectors
    wages = wages[ birth_year : birth_year + max_age ]
    rates = rates[ birth_year : birth_year + max_age ]
    
    # Solving agent optimization
    [value, savings, labors, consumptions] = solve_agent( wages, rates, assets, n_assets, productivity_shocks,
                                                        n_prodshocks, trans_probs, max_age, survival_probabilities)
    
    # Calculating aggregates
    [aggregate_capital, aggregate_effective_labor, aggregate_labor, aggregate_consumption, aggregate_productivity, population] = solve_distribution( savings, labors, consumptions, assets, n_assets, productivity_shocks,
                                                        n_prodshocks, trans_probs, max_age, survival_probabilities )
    
    # Outputting cohort aggregates    
    return (aggregate_capital, aggregate_effective_labor, aggregate_consumption, aggregate_labor, aggregate_productivity)


def solve_agent( wages, rates, assets, n_assets, productivity_shocks,
                n_prodshocks, trans_probs, max_age, survival_probabilities):
    
    # Defining parameters
    crra                = 3
    consumption_share   = 0.7
    discount_factor     = 0.90
    
    # Preallocating output arrays
    value         = np.zeros( [ n_assets, n_prodshocks,  max_age + 1 ] )
    savings       = np.zeros( [ n_assets, n_prodshocks,  max_age     ] )
    labors        = np.zeros( [ n_assets, n_prodshocks,  max_age     ] )
    consumptions  = np.zeros( [ n_assets, n_prodshocks,  max_age     ] )
    
    # Solving the individual optimization problem over the state space
    for age in range( max_age-1, -1, -1 ):
        for ip in range( 0, n_prodshocks ):
            for ia in range( 0, n_assets ):
                
                # Calculating continuation value
                continuation_value = np.dot( trans_probs[ ip, : ], np.transpose( value[:, :, age+1] ) )
                continuation_value = np.squeeze(np.asarray(continuation_value))    # removes the  square brackets from continuation_value

                # Solving the value function
                initial_guess = [assets[ia], .5]    # initial guess of savings and labor supply
                result = minimize(solve_value, initial_guess, args=(rates[age], wages[age], crra, consumption_share, discount_factor, assets, continuation_value,
                                                              assets[ia], productivity_shocks[ip], survival_probabilities[age]),
                                                              method='Nelder-Mead')
                # Saving the value and policy functions
                value       [ia,ip,age] = -1*result.fun
                savings     [ia,ip,age] = result.x[0]
                labors      [ia,ip,age] = result.x[1]
                consumptions[ia,ip,age] = ( (1+rates[age])*assets[ia] + wages[age]*productivity_shocks[ip]*result.x[1]
                                            - result.x[0] )
                
    
    # Outputting solution
    return (value, savings, labors, consumptions)


def solve_value( optimizers, rate, wage, crra, consumption_share, discount_factor, assets, continuation_value,
                 asset, productivity_shock, survival_probability ):
    
    # Renaming optimizers
    saving = optimizers[0]
    labor  = optimizers[1]
    
    # Testing constraints
    if (saving<assets[0]) or (saving>assets[-1]) or (labor<0) or (labor>1):
        value = np.inf
        return value
    
    
    # Solving consumption
    consumption = (1 +rate)*asset + wage*productivity_shock*labor - saving
    
    if consumption<0:
        value = np.inf
        return value
    
    value = ( ( 1/(1-crra) )*((( consumption**consumption_share )*( (1-labor)**(1-consumption_share) ))**(1-crra)) + 
              survival_probability*discount_factor*np.interp( saving, assets, continuation_value ) )
    
    return -1*value


def solve_distribution( savings, labors, consumptions, assets, n_assets, productivity_shocks,
                        n_prodshocks, trans_probs, max_age, survival_probabilities ):
    
    # Defining parameters
    initial_productivity_distribution = np.array([.25,.50,.25])
    
    # Preallocating the  distribution array
    distribution = np.zeros( [ n_assets, n_prodshocks,  max_age ] )
    
    # Populating the initial year
    distribution[0,:,0] = initial_productivity_distribution # sets initial capital  to lowest gridpoint
    
    # Transitioning probabilities throughout lifetime
    for age in range(0,max_age-1):
        if age==1:
            print('the distribution is ', distribution[:,:,0])
        for ia in range(0,n_assets):
            for ip1 in range(0,n_prodshocks):
                upper_bounds = assets<=savings[ia,ip1,age]
                upper_bound  = np.argmin(upper_bounds)      # index of upper bound
                
                lower_bound  = max([0,upper_bound - 1])     # index of lower bound
                if (upper_bound>lower_bound):
                    weight_upper = ( (savings[ia,ip1,age] - assets[lower_bound]) / 
                                     (assets[upper_bound] - assets[lower_bound]) )
                    weight_lower = 1 - weight_upper
                else:
                    weight_upper = 1.0
                    weight_lower = 0.0
                    
                for ip2 in range(0,n_prodshocks):
                    distribution[upper_bound,ip2,age+1] = distribution[upper_bound,ip2,age+1] + weight_upper*survival_probabilities[age]*trans_probs[ip1,ip2]*distribution[ia,ip1,age]
                    distribution[lower_bound,ip2,age+1] = distribution[lower_bound,ip2,age+1] + weight_lower*survival_probabilities[age]*trans_probs[ip1,ip2]*distribution[ia,ip1,age]

    
    
    # Calculating aggregates
    asset_array               = np.tile( assets[:,np.newaxis,np.newaxis], (1, n_prodshocks, max_age) )
    aggregate_capital         = np.sum(np.sum(np.multiply( asset_array       , distribution),axis=0),axis=0)
    
    aggregate_consumption     = np.sum(np.sum(np.multiply( consumptions      , distribution),axis=0),axis=0)
    
    aggregate_labor           = np.sum(np.sum(np.multiply( labors            , distribution),axis=0),axis=0)
    productivity_array        = np.transpose( np.tile( productivity_shocks[:,np.newaxis,np.newaxis], (1, n_assets, max_age) ), (1, 0, 2))
    effective_labor           = np.multiply(labors, productivity_array)
    aggregate_effective_labor = np.sum(np.sum(np.multiply( effective_labor   , distribution),axis=0),axis=0)
    aggregate_productivity    = np.sum(np.sum(np.multiply( productivity_array, distribution),axis=0),axis=0)
    population                = np.sum(np.sum(                                 distribution ,axis=0),axis=0)
    
    # Outputting the results
    return (aggregate_capital, aggregate_effective_labor, aggregate_labor, aggregate_consumption, aggregate_productivity, population) 