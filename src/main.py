# -*- coding: utf-8 -*-
"""
main.py - Automated Robot Mining project
Connects to V-Rep, loads the terrain file, 
Computes the slopes across the terrain,
calculates the order of dig site visitation,
runs policy iteration to route plan, and 
controls the V-REP rover in a seperate thread.

Written by Justin Stucki, modified and expanded by Braxton Johston
12 Dec 2018 - Johnston, Germer, and Stucki
"""

#!/usr/bin/python

import thread, time, timeit, os, sys, copy, shutil
import simInterface as sim
import numpy as np
import graph_search
import vi_pi as vipi
import common as cmn
from matplotlib import cm

_SANS_V_REP = False

# Main Method
if __name__ == '__main__':
    print 'using map {} with scale {}'.format(cmn._MAP_FILES[cmn._MAP_CHOICE][0],cmn._MAP_FILES[cmn._MAP_CHOICE][1])

    #Setup folders for plot output and other saving
    reuseFldrs = False
    if os.path.isdir(cmn._IMG_FLDR) or os.path.isdir(cmn._OUT_FLDR):
        try:
            if np.all([not os.path.listdir(cmn._IMG_FLDR),not os.path.listdir(cmn._OUT_FLDR)]):
                reuseFldrs = True
        except: pass
        if not reuseFldrs:
            i = 1
            while True:
                newImgFldr = '{}_{}'.format(cmn._IMG_FLDR,i)
                newOutFldr = '{}_{}'.format(cmn._OUT_FLDR,i)
                i+=1
                if not os.path.isdir(newImgFldr) and not os.path.isdir(newOutFldr): break
            if os.path.isdir(cmn._IMG_FLDR):    shutil.move(cmn._IMG_FLDR,newImgFldr)
            if os.path.isdir(cmn._OUT_FLDR):shutil.move(cmn._OUT_FLDR,newOutFldr)

    os.makedirs(cmn._IMG_FLDR)
    os.makedirs(cmn._OUT_FLDR)
    try:
        if os.path.isdir(newOutFldr) and cmn._GRAB_PICKLE:
            try:
                shutil.copy(os.path.join(newOutFldr,'genMap.txt'),os.path.join(cmn._OUT_FLDR,'genMap.txt'))
            except:
                print 'could not copy genMap from {} to {}'.format(newOutFldr,cmn._OUT_FLDR)
        if os.path.isdir(newImgFldr) and cmn._GRAB_PICKLE:
            try:
                shutil.copy(os.path.join(newImgFldr,'contourMap.png'),os.path.join(cmn._IMG_FLDR,'contourMap.png'))
            except:
                print 'could not copy contourMap from {} to {}'.format(newOutFldr,cmn._IMG_FLDR)

            try:
                shutil.copy(os.path.join(newImgFldr, 'dig_site_ordering_r100_c100_d3_INIT.svg'), os.path.join(cmn._IMG_FLDR, 'dig_site_ordering_r100_c100_d3_INIT.svg'))
            except:
                print 'could not copy dig_site_ordering_r100_c100_d3_INIT from {} to {}'.format(newOutFldr, cmn._IMG_FLDR)

            try:
                shutil.copy(os.path.join(newImgFldr, 'discreteObsMap_digPlanner_post.svg'), os.path.join(cmn._IMG_FLDR, 'discreteObsMap_digPlanner_post.svg'))
            except:
                print 'could not copy discreteObsMap_digPlanner_post from {} to {}'.format(newOutFldr, cmn._IMG_FLDR)

            try:
                shutil.copy(os.path.join(newImgFldr,'discreteObsMap_digPlanner_pre.svg'),os.path.join(cmn._IMG_FLDR,'discreteObsMap_digPlanner_pre.svg'))
            except:
                print 'could not copy discreteObsMap_digPlanner_pre from {} to {}'.format(newOutFldr,cmn._IMG_FLDR)
    except:
        pass
     #Setup and start VREP interface
    
    s = sim.Sim()
    s.sim_connect()
    s.sim_start()
    s.sim_get_handles()
    
    #If using V_REP, start second thread for rover control 
    if not _SANS_V_REP:
        #Start the interface thread to send frame instructions
        try:
            thread.start_new_thread( s.sim_drive_rover, ())
        except:
            print "Error: unable to start thread"

    #Load terrain image file and compute slope map
    print "Loading", cmn._map_file
    g = graph_search.GridMap(s, cmn._map_file)
    g.calcRatesOfChange(n=3, ret=False,dbg=False)

    #Create plotes for each action direction and save to disk
    cmap = cm.Spectral;
    cmap.set_bad(color='k')
    cmn.createArrImg(g.height_map, cmap=cmap, plotTitle='Height Map', fn=os.path.join(cmn._IMG_FLDR, 'heightmap.svg'),
                     show=False)
    for a in cmn._actions:
        cmn.createArrImg(g.rates[0][a], cmap=cmap, plotTitle='Rate Map: {}'.format(a),
                         fn=os.path.join(cmn._IMG_FLDR, 'ratemap_{}.svg'.format(a)), show=False)
        cmn.overlayArrImgs(arr1=g.height_map, arr2=g.rates[0][a], plotTitle='Overlayed Rate Map: {}'.format(a),
                           fn=os.path.join(cmn._IMG_FLDR, 'ratemap_ovr_{}.svg'.format(a)), show=False)

    rover_pos = s.sim_get_xy(s.rover_handle)
    changed_rover_pos = cmn.convert_VREPXY(rover_pos)

    print '\n{} ---> {}'.format(rover_pos,changed_rover_pos)
    
    #Plan dig site visitation order
    g.calcDigSiteOrder(start=changed_rover_pos,savePickle=cmn._SAVE_PICKLE,grabPickle=cmn._GRAB_PICKLE)
    print '\n\tusing dig order {}\n'.format(g.digs)
    
    #For each dig site...
    for i in range(len(g.digs)):
        for j in range(2):
            #Plan route to dig site
            if j is 0:
                g.goal = g.digs[i]
                print "Destination: dig(" + str(i) + ")", g.goal
            #Or back to the dump collection bin
            else:
                g.goal = g.dump
                print "Destination: Dump site", g.goal
            #Track the running time for policy iteration to complete
            start = timeit.default_timer()
            rover_pos = s.sim_get_xy(s.rover_handle)
            iterations, val_grid, dir_grid, init_reward_map, end_reward_map = vipi.policy_iteration(gridMap=g, actions=cmn._actions, start=rover_pos, goal=g.goal, constActChangeCountMax=cmn._PI_ConstActChangeCountMax,idxs=(i,j), legIdx=i*2+j)
            print "Converged after", iterations, "iterations."
            stop = timeit.default_timer()
            print "Time(" , i, "): ", stop - start, "\n\n"

            #print 'Val Grid:\n{}\n'.format(cmn.prettyStr2DMatrix(val_grid,'\t'))
            tmpFn = 'policygrid_d{}_{}.svg'.format(i,j)
            tmpTitle = 'Policy Iteration\nDig_{}_leg_{}\nStart: ({}, {})   Goal: ({}, {})'.format(i,j,rover_pos[cmn._X],rover_pos[cmn._Y],g.goal[cmn._X],g.goal[cmn._Y])
            cmn.printPolicyMap(val_grid=val_grid, dir_grid=dir_grid, start=rover_pos, goal=g.goal, fn=os.path.join(cmn._IMG_FLDR,tmpFn), figTitle=tmpTitle)

            tmpFn = 'policygrid_d{}_{}_init_reward.svg'.format(i, j)
            tmpTitle = 'Policy Iteration INIT REWARD\nDig_{}_leg_{}\nStart: ({}, {})   Goal: ({}, {})'.format(i, j,
                                                                                                  rover_pos[cmn._X],
                                                                                                  rover_pos[cmn._Y],
                                                                                                  g.goal[cmn._X],
                                                                                                  g.goal[cmn._Y])
            cmn.printPolicyMap(val_grid=init_reward_map, dir_grid=dir_grid, start=rover_pos, goal=g.goal,
                               fn=os.path.join(cmn._IMG_FLDR, tmpFn), figTitle=tmpTitle, drawArrows=False)

            tmpFn = 'policygrid_d{}_{}_end_reward.svg'.format(i, j)
            tmpTitle = 'Policy Iteration END REWARD\nDig_{}_leg_{}\nStart: ({}, {})   Goal: ({}, {})'.format(i, j,
                                                                                                  rover_pos[cmn._X],
                                                                                                  rover_pos[cmn._Y],
                                                                                                  g.goal[cmn._X],
                                                                                                  g.goal[cmn._Y])
            cmn.printPolicyMap(val_grid=end_reward_map, dir_grid=dir_grid, start=rover_pos, goal=g.goal,
                               fn=os.path.join(cmn._IMG_FLDR, tmpFn), figTitle=tmpTitle, drawArrows=False)

            #Follow policy to give driving commands to rover
            done = False
            s.vel = cmn._VEL
            rover_head  = s.sim_get_orient(s.rover_handle)[2]
            last_head = rover_head
            #While not at the destination
            while rover_pos != g.goal and done is False:
                policy_head = cmn._headings[dir_grid[rover_pos[cmn._X], rover_pos[cmn._Y]]]
                last_pos = None
                #Orient the rover to the heading listed in the policy for current state
                while abs(policy_head-rover_head) > 4.0 and done is False:  # TURNING FIRST
                    print "T", rover_pos, "->", g.goal, "   Heading:", rover_head, "   via action ", dir_grid[rover_pos[cmn._X], rover_pos[cmn._Y]]
                    if abs(rover_head-last_head) > 5: #REMOVE OUTLIERS.  MAX MOVEMENT PER "ITERATION"
                        continue
                    rover_head  = s.sim_get_orient(s.rover_handle)[2] # ALPHA->ROLL, BETA->PITCH, GAMMA->YAW
                    #If turning +1deg narrows the difference of headings: turn LEFT
                    diff = policy_head - rover_head
                    if diff > 0.0:
                        s.dir = "R"
                    else:
                        s.dir = "L"
                    #time.sleep(0.05)
                    last_head = rover_head
                    rover_pos = s.sim_get_xy(s.rover_handle)
                    if g.is_goal(rover_pos):
                        done = True
                    last_pos  = rover_pos
                #Once oriented, move forward to a different state
                while rover_pos == last_pos and done is False: #NOT NEAR GOAL
                    print "S", rover_pos, "->", g.goal, "   Heading:", rover_head
                    s.dir = "F"
                    rover_pos   = s.sim_get_xy(s.rover_handle)
                    if g.is_goal(rover_pos):
                        done = True
                    #time.sleep(0.05)

                rover_pos = s.sim_get_xy(s.rover_handle)
            #Arrived at destination
            s.vel = 0

    #End VREP simulation
    s.sim_stop()
