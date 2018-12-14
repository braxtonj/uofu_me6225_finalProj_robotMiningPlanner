# -*- coding: utf-8 -*-
'''
vi-pi.py - functions relating to policy iteration
For use in the Automated Robot Mining project

12 Dec 2018 - Johnston, Germer, and Stucki
'''
import copy, time, traceback, sys, os
import numpy as np
import common as cmn
from matplotlib import cm

# Braxton Johnston
def dispNeighborPolicies(r,c,valGrid,policyGrid,fn=None,figTitle=None,show=True):
    neighborPolicies = np.full(shape=(3,3),fill_value='',dtype=np.object)
    neighborVals = np.full(shape=(3,3),fill_value=np.nan,dtype=np.float)
    for nR in range(-1,2):
        for nC in range(-1,2):
            neighborR = r+nR
            neighborC = c+nC
            if neighborR < 0 or neighborR >= policyGrid.shape[0] or neighborC < 0 or neighborC >= policyGrid.shape[1]:
                continue
            else:
                neighborPolicies[(nR,nC)] = policyGrid[(neighborR,neighborC)]
                neighborVals[(nR,nC)]     = valGrid[(neighborR,neighborC)]
    cmn.printPolicyMap(neighborVals, neighborPolicies, start=None, goal=None, newTicks={'rVal':r,'cVal':c}, fn=fn, figTitle=figTitle, show=show)


# Braxton Johnston
def checkIfAbsorbing( r, c, valGrid, policyGrid, iter=-1, saveImgs=False ):
    dbg = True
    # check all actionable neighbors to see if they are pointing to rc.
    directedAtRC = []
    for act in cmn._actions:
        ((r1, c1), (r2, c2)) = cmn.getRCAfterAction(r, c, act)
        if r2 < 0 or r2 >= policyGrid.shape[0] or c2 < 0 or c2 >= policyGrid.shape[1]:
            directedAtRC.append(True)
        else:
            if   policyGrid[(r2,c2)] == 's'  and (c == c2 and r2 < r):  directedAtRC.append(True)
            elif policyGrid[(r2,c2)] == 'sw' and (c < c2 and r2 < r):   directedAtRC.append(True)
            elif policyGrid[(r2,c2)] == 'w'  and (c < c2 and r2 == r):  directedAtRC.append(True)
            elif policyGrid[(r2,c2)] == 'nw' and (c < c2 and r2 > r):   directedAtRC.append(True)
            elif policyGrid[(r2,c2)] == 'n'  and (c == c2 and r2 > r):  directedAtRC.append(True)
            elif policyGrid[(r2,c2)] == 'ne' and (c > c2 and r2 > r):   directedAtRC.append(True)
            elif policyGrid[(r2,c2)] == 'e'  and (c > c2 and r2 == r):  directedAtRC.append(True)
            elif policyGrid[(r2,c2)] == 'se' and (c > c2 and r2 < r):   directedAtRC.append(True)
            else:                                                       directedAtRC.append(False)
    directedAtRC = np.array(directedAtRC)
    if dbg and False:
        print '({},{}) results in '.format(r,c), directedAtRC
        #for actNdx in range(len(cmn._actions)): print '\t{}: {}'.format(cmn._actions[actNdx],directedAtRC[actNdx])
    if np.all(directedAtRC) or len(directedAtRC[directedAtRC==True])>cmn._PI_ABSORBING_MIN_DIRECTED:
        if dbg: print '({},{}) is absrobing'.format(r,c)
        if saveImgs:
            dispNeighborPolicies(copy.copy(r), copy.copy(c), valGrid.copy(), policyGrid.copy(), fn=os.path.join(cmn._IMG_FLDR,'absorbing_i{}_r{}_c{}.svg'.format(iter,r,c)), figTitle='Abosorbing Behavior at ({},{}) for Policy Iteration {}'.format(r,c,iter), show=False)
        return True
    else:
        return False

# Braxton Johnston
def getStaticRewardsFromPolicy(rewards,policy):
    rwrd = np.zeros(shape=policy.shape)
    for r in range(policy.shape[0]):
        for c in range(policy.shape[1]):
            rwrd[(r,c)] = rewards[policy[(r,c)]][(r,c)]
    return rwrd

def policy_iteration(gridMap, actions, start, goal, timeout=None, constActChangeCountMax=None, idxs=None, legIdx=None):
    '''
    Perform Policy Iteration on a grid with the allowed actions

    gridMap - array which indicates borders, obstacles, and supplies the 
              transition function
    actions - the allowable action to be used

    returns - 
        iterations - How many loop iterations until convergence
        val_ref    - The settled values for each state
        dir_grid   - The settle policy for each state
    '''
    if not constActChangeCountMax is None:
        priorActionChangeNum = -1
        staticActionChangeCount = 0
    if not timeout is None:
        startTime = time.time()

    printMapsForEachIteration = True
    printMapsForEachIterationFreq = 42

    #Create array to store the last iteration and current working iteration
    oldStart = copy.copy(gridMap.start)
    oldGoal = copy.copy(gridMap.goal)
    gridMap.start = copy.copy(start)
    gridMap.goal = copy.copy(goal)

    gridMap.calcRatesOfChange(n=gridMap.rateOrder,dbg=False)
    gridMap.calcRewardFunction(goal=cmn._PI_GOAL_REWARD,start=cmn._PI_START_REWARD,obstacle=cmn._PI_OBSTACLE_REWARD,edge=cmn._PI_EDGE_REWARD, vrs=cmn._CALC_REWARD_VRS, legVal=cmn._PI_LEG_REWARD, legIdx=legIdx )

    #Create an array to store actions and init to an arbitrary policy
    dir_ref = np.random.choice(cmn._actions,size=gridMap.height_map.shape)

    rwrds = getStaticRewardsFromPolicy(gridMap.rewards,dir_ref)
    init_reward_map = rwrds.copy()
    with open(os.path.join(cmn._OUT_FLDR,'base_rewards.txt'), 'w') as f:
        f.write(cmn.prettyStr2DMatrix(rwrds))

    val_ref =             rwrds.copy()
    val_working_ref     = rwrds.copy()
    val_working_current = rwrds.copy()

    dir_current = dir_ref.copy() #Keep revising assigned values until convergence is attained
    iterations    = 0     #Count iterations for performance evaluation
    while True:
        iterations += 1
        print '\n', iterations,; sys.stdout.flush();

        tmpRwrds = getStaticRewardsFromPolicy(gridMap.rewards, dir_current) #Base reward function is dependent on direction of travel
        end_reward_map = tmpRwrds.copy()

        #Keep revising assigned values until convergence is attained
        viIter = 0
        breakMe = False
        while not breakMe:
            viIter += 1
            print '.',

            val_working_current_prior = val_working_current.copy()

            #Modify rewards if absorbing
            ''' SOOOOOOO SLOW
            for r in range(gridMap.rows):
                for c in range(gridMap.cols):
                    if checkIfAbsorbing( copy.copy(r), copy.copy(c), valGrid=val_working_ref.copy(), policyGrid=dir_ref.copy(), iter=iterations, saveImgs=True ):
                        tmpRwrds[(r,c)] = cmn._PI_ABSORBING_REWARD
            '''
            # Calculate every grid entry
            for r in range(gridMap.rows):
                for c in range(gridMap.cols):
                    #Get a list of reachable states and probability of arriving
                    options = gridMap.transition((r,c), dir_ref[r,c])
                    #print 'OPTIONS\n{}\n'.format(options)
                    #Sum the probability times the previous value at that state
                    total = 0
                    for o in options:
                        total = total + o[0] * val_working_ref[o[1][0],o[1][1]]
                    #Finish evaluating current state by adding discount factor
                    val_working_current[r,c] = tmpRwrds[r,c] + cmn._PI_LAMBDA * total
                    if not timeout is None and time.time()-startTime>timeout:
                        val_working_ref = val_working_current.copy()
                        break
            #Check for convergence (difference less than epsilon)
            if (abs(val_working_current - val_working_ref) < cmn._PI_EPSILON).all():
                breakMe = True
            else:
                breakMe = False

            if printMapsForEachIteration and (viIter==1 or viIter%printMapsForEachIterationFreq==0) or breakMe:
                if idxs is None:
                    tmpFN = 'policygrid_'
                    tmpTitle = 'Policy Iteration\nDig_nan_leg_nan'
                else:
                    tmpFN = 'policygrid_d{}_{}_'.format(idxs[0],idxs[1])
                    tmpTitle = 'Policy Iteration\nDig_{}_leg_{}'.format(idxs[0],idxs[1])
                tmpTitle = '{}{}'.format(tmpTitle,'  PI{} VI{}  DEL {}\nStart: {}   Goal: {}'.format(iterations,viIter,np.average(np.abs(val_working_current-val_working_ref)),gridMap.start,gridMap.goal))
                tmpFNARROWS = '{}pi{}_vi{}_arr.svg'.format(tmpFN,iterations,viIter)
                tmpFNSANARR = '{}pi{}_vi{}_sanarr.svg'.format(tmpFN,iterations,viIter)
                cmn.printPolicyMap(val_grid=val_working_current, dir_grid=dir_current, start=gridMap.start, goal=gridMap.goal,
                                   fn=os.path.join(cmn._IMG_FLDR, tmpFNARROWS), show=False, figTitle=tmpTitle, figSizeScale=1)
                cmn.printPolicyMap(val_grid=val_working_current, dir_grid=dir_current, start=gridMap.start, goal=gridMap.goal,
                                   fn=os.path.join(cmn._IMG_FLDR, tmpFNSANARR), show=False, figTitle=tmpTitle, figSizeScale=1, drawArrows=False)


            #Update the reference copy and reiterate, if necessary
            val_working_ref = val_working_current.copy()
            
        #Once the values have converged, perform the policy update
        #Loop again to determine direction changes based on updated values
        best_vals = val_working_current.copy()
        for r in range(gridMap.rows):
            for c in range(gridMap.cols):
                
                action_info = [] #Info about the current best action
                
                #Loop through possible actions checking for policy improvement
                for a in actions:
                    
                    #Get state where this action leads
                    s_prime = gridMap.test_new_pos((r,c), a)
                    
                    #Ignore obstacles and walls, o/w add to potentials list
                    if (s_prime != (r,c)):
                        action_info.append(
                                (val_working_current[s_prime[0],s_prime[1]], a, s_prime))
                        
                #Evaluate potentials for best candidate
                if action_info != []:
                    just_vals = [x[0] for x in action_info]
                    best_action = action_info[np.argmax(just_vals)][1]
                    best_state  = action_info[np.argmax(just_vals)][2]
                    best_val = action_info[np.argmax(just_vals)][0]
                    old_dir = gridMap.test_new_pos((r,c), dir_ref[r,c])
                    
                    #If taking a different course leads to better value, switch 
                    if val_working_current[best_state[0], best_state[1]] > \
                                   val_ref[old_dir[0],old_dir[1]]:
                        dir_current[r,c] = best_action
                        best_vals[(r,c)] = best_val
        val_ref = val_working_ref.copy()
        currActionChangeNum = np.count_nonzero(dir_ref!=dir_current)
        if not constActChangeCountMax is None:
            if currActionChangeNum == priorActionChangeNum:
                staticActionChangeCount += 1
            else:
                staticActionChangeCount = 0
            priorActionChangeNum = currActionChangeNum
        #print '\n\tPRIOR VAL_REF GENERATED {} ACTION CHANGES:\n{}\n\n'.format(currActionChangeNum,cmn.prettyStr2DMatrix(val_ref,'\t\t',5))
        if not timeout is None:
            dt = time.time() - startTime

        if printMapsForEachIteration:
            if idxs is None:
                tmpFN = 'policygrid_'
                tmpTitle = 'Policy Iteration\nDig_nan_leg_nan'
            else:
                tmpFN = 'policygrid_d{}_{}_'.format(idxs[0], idxs[1])
                tmpTitle = 'Policy Iteration\nDig_{}_leg_{}'.format(idxs[0], idxs[1])
            tmpTitle = '{}{}'.format(tmpTitle,
                                     '  PI{} VI{}\nStart: {}   Goal: {}'.format(iterations, 0, gridMap.start,
                                                                                gridMap.goal))
            tmpFNARROWS = '{}pi{}_vi{}_arr.svg'.format(tmpFN, iterations, np.nan)
            tmpFNSANARR = '{}pi{}_vi{}_sanarr.svg'.format(tmpFN, iterations, np.nan)
            cmn.printPolicyMap(val_grid=val_working_current, dir_grid=dir_current, start=gridMap.start,
                               goal=gridMap.goal,
                               fn=os.path.join(cmn._IMG_FLDR, tmpFNARROWS), show=False, figTitle=tmpTitle,
                               figSizeScale=1)
            cmn.printPolicyMap(val_grid=val_working_current, dir_grid=dir_current, start=gridMap.start,
                               goal=gridMap.goal,
                               fn=os.path.join(cmn._IMG_FLDR, tmpFNSANARR), show=False, figTitle=tmpTitle,
                               figSizeScale=1, drawArrows=False)

        if (dir_ref == dir_current).all():
            break
        elif not timeout is None and dt > timeout:
            print '\n\t\tEARLY BREAK DUE TO TIMEOUT OF {} s'.format(dt)
            break
        elif not constActChangeCountMax is None and staticActionChangeCount > constActChangeCountMax:
            print '\n\t\tEARLY BREAK DUE TO {} CONSECUTIVE ACTION CHANGES OF {}'.format(staticActionChangeCount,priorActionChangeNum)
            break
        else:
            print '\t\tNORM DIFF: {:1.5g}     ACTION CHANGES: {} --> {:1.2g}%'.format(np.linalg.norm(best_vals-val_ref),currActionChangeNum,100.0*currActionChangeNum/np.prod(val_ref.shape))
        dir_ref = dir_current.copy()

    gridMap.start = oldStart
    gridMap.goal = oldGoal
    return (iterations, val_ref, dir_current, init_reward_map, end_reward_map)
