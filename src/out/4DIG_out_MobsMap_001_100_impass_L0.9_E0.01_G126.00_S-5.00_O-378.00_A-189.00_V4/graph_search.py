#!/usr/bin/env python
'''
graph_search.py - Helper classes and functions 
for performing graph search operations for planning.
Adapted and expanded from Project 1: Graph Search
For use in the Automated Robot Mining project

12 Dec 2018 - Smith, Germer, and Stucki
'''
import numpy as np
#import heapq
import matplotlib.pyplot as plt
from matplotlib import cm
#from scipy.spatial import distance
import sklearn as skl
from PIL import Image
import numpy as np
import digSiteGraph as _dsg
import sys, copy, os, shutil
import common as cmn
import simInterface as sim
from scipy.stats import skewnorm
import math

#Grid map object to contain objects and terrain information
class GridMap:
    def __init__(self, sim_class, map_path=None):
        self.goal = None
        self.start = None
        self.rows = None
        self.cols = None
        self.digs = None #x,y coordinates of dig sites 0-9
        self.dump = None #x,y coordinates of the dump 
        self.bin  = None #x,y coordinates of the bin to avoid
        self.height_map = None
        self.map_path = map_path
        self.rover_buffer = 0.55 #m from center to furthest perimeter point
        self.rateOrder = 3
        if map_path is not None:
            self.read_map(map_path)
            #self.calcRatesOfChange(n=self.rateOrder,ret=False,dbg=False)
        if sim_class.clientID is not -1:
            if sim_class.dig_handles is not None:
                self.digs = [cmn.convert_VREPXY(sim_class.sim_get_xy(handle))
                    for handle in sim_class.dig_handles]
            if sim_class.dump_handle is not None:
                self.dump = cmn.convert_VREPXY(sim_class.sim_get_xy(sim_class.dump_handle))
            if sim_class.bin_handle is not None:
                self.bin = cmn.convert_VREPXY(sim_class.sim_get_xy(sim_class.bin_handle))

    #Convert and scale height map from image file
    # Justin Stucki
    def read_map(self, map_path):
        im = Image.open(map_path)
        #sim = cmn.convert_grayscale(im)
        png = np.array(im)
        png = np.rot90(png, 2)

        np.set_printoptions(threshold=np.nan)
        print '\n\nPNG:\n{}\n\n'.format(png.shape)
        np.set_printoptions(threshold=np.nan)

        (self.rows, self.cols, is_rgb) = png.shape
        print "r", self.rows, "c", self.cols
        self.height_map = np.zeros((self.rows, self.cols), dtype=np.float)
        if is_rgb == 3:
            maxRGB = cmn.rgb2hex((255,255,255)) * cmn._MAP_SCALE
            print "Converting PNG to scaled height map..."
            for r in range(self.rows):
                for c in range(self.cols):
                    self.height_map[r,self.cols-c-1] = float(cmn.rgb2hex(png[r,c])) / float(maxRGB)
        else:
            for r in range(self.rows):
                for c in range(self.cols):
                    self.height_map[r,self.cols-c-1] = float(png[(r,c,2)])/255
        im.close()

        #Set to True for plot of height map and console printout
        if False:
            for r in range(self.rows):
                for c in range(self.cols):
                    self.height_map[r,c]  = round(self.height_map[r,c] , 2)
            fig, ax = plt.subplots()
            im = ax.imshow(self.height_map, 'Spectral')
            im.set_interpolation('nearest')
            plt.show()
            np.set_printoptions(threshold=np.nan, linewidth=2000) 
            print self.height_map

    #Test if a state is the goal
    def is_goal(self,s):
        return (s[cmn._X] == self.goal[cmn._X] and s[cmn._Y] == self.goal[cmn._Y])

    #Test if a state is the start
    def is_start(self,s):
        return (s[cmn._X] == self.start[cmn._X] and s[cmn._Y] == self.start[cmn._Y])

    #Test if a new state can be traveled to
    #Adapted from Project 1
    def test_new_pos(self, s, a):
        new_pos = list(s[:])
        if a == 's':
            if s[cmn._Y] > 0:
                new_pos[cmn._Y] -= 1
        elif a == 'n':
            if s[cmn._Y] < self.rows - 1:
                new_pos[cmn._Y] += 1
        elif a == 'w':
            if s[cmn._X] > 0:
                new_pos[cmn._X] -= 1
        elif a == 'e':
            if s[cmn._X] < self.cols - 1:
                new_pos[cmn._X] += 1
        # Added to allow movement up and to the left all at once
        elif a == 'sw':
            if s[cmn._Y] > 0 and s[cmn._X] > 0:
                new_pos[cmn._Y] -= 1
                new_pos[cmn._X] -= 1
        # Added to allow movement up and to the right all at once
        elif a == 'se':
            if s[cmn._Y] > 0 and s[cmn._X] < self.cols - 1:
                new_pos[cmn._Y] -= 1
                new_pos[cmn._X] += 1
        # Added to allow movement down and to the right all at once
        elif a == 'ne':
            if s[cmn._Y] < self.rows - 1 and s[cmn._X] < self.cols - 1:
                new_pos[cmn._Y] += 1
                new_pos[cmn._X] += 1
        # Added to allow movement down and to the left all at once
        elif a == 'nw':
            if s[cmn._Y] < self.rows - 1 and s[cmn._X] > 0:
                new_pos[cmn._Y] += 1
                new_pos[cmn._X] -= 1
        else:
            print 'Unknown action:', str(a)

        s_prime = tuple(new_pos)
        return s_prime

    #Compute the action probability distribution for a state and action
    # Justin Stucki
    def transition(self, s, a):
        #divide probabilities across desired direction and four side directions
        bins = 5
        #Calibrated to start sliding at about 30 degrees
        calibration_factor = 4
        #Compute the rover heading vector from heading angle
        heading = [np.cos(np.radians(cmn._headings[a])), 
                   np.sin(np.radians(cmn._headings[a])), 0]
        #Estimate slope vector based on neighboring slopes
        slope = [self.rates[0]['e'][s],self.rates[0]['n'][s],0]
        #Cross produces a resulting magnitude and sirection
        cross = np.cross(heading, slope)
        #Ressulting magnitude is used to skew the normal distribution
        skew = calibration_factor * np.linalg.norm(cross)
        #Sign of the cross produce -> left vs right skew
        if cross[2] > 0.0:
            skew = -1 * skew
        skewSamples = 100
        r = skewnorm.rvs(skew, size=skewSamples)
        #Divide into bins to distribute to actions
        x = np.histogram(r, bins)
        #Normalize to ensure 100% probability overall
        total = np.sum(x[0])
        dist = np.zeros((1,bins), dtype=float)
        for i in range(len(x[0])):
            dist[0][i] = float(x[0][i]) / float(total)
        
        #Assign distribution
        dir8 = ['n','ne','e','se','s','sw','w','nw'] #Circular action order
        idx = dir8.index(a)
        ccw2 = (dist[0][0], self.test_new_pos(s, dir8[idx - 2]))
        ccw1 = (dist[0][1], self.test_new_pos(s, dir8[idx - 1]))
        fwd  = (dist[0][2], self.test_new_pos(s, a))
        cw1  = (dist[0][3],  self.test_new_pos(s, dir8[(idx + 1) % 8]))
        cw2  = (dist[0][4],  self.test_new_pos(s, dir8[(idx + 2) % 8]))
        return (ccw2, ccw1, fwd, cw1, cw2)

    #Determine the order that digs to be visited to prevent occlusion
    # Braxton Johnston
    def calcDigSiteOrder(self, start,savePickle=True,grabPickle=False):
        dbg = True
        if dbg: print 'generating connected graph...'
        self.dsg = _dsg.DigSiteGraph( g=self, env=self.map_path, start=start )
        self.dsg.build(grabPickle=grabPickle,savePickle=savePickle)
        tmpDigs = copy.copy(self.dsg.shortestDigOrder)
        self.digs = []
        for dig in tmpDigs:
            self.digs.append(cmn.convert_VREPXY(dig))


    #Compute the slope at each state going in each of the eight directions
    # Braxton Johnston / Jay Dee Germer
    def calcRatesOfChange(self, n=1, ret=False, dbg=False):
        '''
            Calculates nth "rate of change" of the map in all action directions.  Could be thought of as nth gradients using the actions for a base
                ie
                 TYPES:     LIST        DICT      ndarray
                    self.rates[ord-1][action_str][(r,c)]  ---> for a given order, ord, and action, action_str ('n','ne',..) this is the calculated rate at (r,c)
                     self.rates[0] =   {a: d/da     --->  n:d/dn, ne:d/dne, e:d/de, ... }
                            .      =              .
                            .      =              .
                            .      =              .
                    self.rates[n-1] = {a: d^n/da^n  --->  n:d^n/dn^n, ne:d^n/dne^n, ...}     Double use of n is confusing.  o well :)

        :param n: order of the gradient, int
        :return:
        '''
        printVals = True
        printFinal = False
        badVal = np
        self.rates = []
        for ord in range(n):
            if dbg and not printVals: print 'finding {}th order: '.format(ord+1),; sys.stdout.flush()
            if dbg and printVals:     print 'finding {}th order:'.format(ord+1)
            self.rates.append({})
            for actNdx in range(len(cmn._actions)):
                if dbg and not printVals: print ' {},'.format(cmn._actions[actNdx]),; sys.stdout.flush()
                if dbg and printVals:     print '\t{}'.format(cmn._actions[actNdx])
                self.rates[ord].update({cmn._actions[actNdx]:np.zeros(shape=(self.rows,self.cols),dtype=np.float)})
                if ord == 0:
                    v = self.height_map.copy() # Values, v, to use in computation.  height_map to start
                else:
                    v = self.rates[ord-1][cmn._actions[actNdx]].copy()
                for r in range(self.rows):
                    for c in range(self.cols):
                        if r == 0 or r == self.rows-1 or c == 0 or c == self.cols-1:
                            d_da = np.nan
                        else:
                            ((r1, c1), (r2, c2)) = cmn.getRCAfterAction(r, c, cmn._actions[actNdx])

                            if len(cmn._actions[actNdx]) > 1:
                                fact = 2*cmn._sim_res*np.sqrt(2) # The diag traverse a longer distance.  1^2 + 1^2 = 2^2.  duh.
                            else:
                                fact = 2*cmn._sim_res
                            d_da = (v[(r2,c2)] - v[(r1,c1)])/fact
                        self.rates[ord][cmn._actions[actNdx]][(r, c)] = d_da

                absMax = np.nanmax([np.abs(np.nanmax(self.rates[ord][cmn._actions[actNdx]])), np.abs(np.nanmin(self.rates[ord][cmn._actions[actNdx]]))])
                self.rates[ord][cmn._actions[actNdx]][np.isnan(self.rates[ord][cmn._actions[actNdx]])] = absMax*5.0

            if dbg and printVals:
                    print '\nORIG'
                    print cmn.prettyStr2DMatrix(v,'\t')
                    print '\nCALC'
                    print cmn.prettyStr2DMatrix(self.rates[ord][cmn._actions[actNdx]],'\t')
                    print '\n'
                    #raw_input('cont..')


            if dbg and not printVals:
                print 'd^{ORD}/dn^{ORD}:\n\tMin:  {MIN}\n\tMax:  {MAX}\n\tMean: {MEAN}\n\tStd:  {STD}\n'.format(
                                    ORD= ord+1,
                                    MIN= np.nanmin( self.rates[ord]['n']),
                                    MAX= np.nanmax( self.rates[ord]['n']),
                                    MEAN=np.nanmean(self.rates[ord]['n']),
                                    STD= np.nanstd( self.rates[ord]['n'])
                )


        if dbg and printFinal:
            print 'Final'
            for actNdx in range(len(cmn._actions)):
                print 'd^{n}/d{a}^{n}'.format(n=ord+1,a=cmn._actions[actNdx])
                print cmn.prettyStr2DMatrix(self.rates[ord][cmn._actions[actNdx]],'\t')
                print '\n'

        if ret: return copy.deepcopy(self.rates)

    #Compute the reward distribution for policy iteration
    # Braxton Johnston
    def calcRewardFunction(self, goal=42.0, start=0.0, obstacle=-7.0, edge=0, ret=False, vrs=-1, legVal=0.0, legIdx=None ):
        if vrs < 1: vrs = cmn._CALC_REWARD_VRS
        self.rewards = {}
        obsRateThresh = 1.0
        critRate = np.tan(np.pi/20) #IE if the slops is less than 9 deg
        negCritRate = np.tan(np.pi/15)

        startPos = np.array(self.start)
        goalPos = np.array(self.goal)

        #For use later
        goalRwrds = np.zeros(shape=(self.rows,self.cols))
        maxRad = (self.rows+self.cols)/np.sqrt(2)
        fid = cmn._sim_res * 5
        nns = cmn.neighbors(mat=goalRwrds, rad=maxRad, r=self.goal[0], c=self.goal[1])
        for nn in zip(list(nns[0]), list(nns[1])):
            tmpR = np.linalg.norm([nn[0]-goalPos[0],nn[1]-goalPos[1]])
            if np.isnan(tmpR) or tmpR < fid:
                tmpR = fid # EPS
            goalRwrds[nn] = goal*((1.0/maxRad)*np.abs(tmpR-maxRad)+1.0/np.power(tmpR,1.0/2))
        #goalRwrds[self.goal] = goal

        if legIdx is None:
            legRwrds = np.zeros(shape=(self.rows,self.cols))
        else:
            # SHOULD CONSIDER USING GAUSSIAN BLUR ON THE PATH TO SMOOTH IT..  RIGHT NOW THERE ARE PEAKS THE PLANNER CAN GET LOST POINTING TOWARDS
            print 'calculating leg reward using {}'.format(self.dsg.shortestRoverPathNodes[legIdx])
            legRwrds = np.zeros(shape=(self.rows,self.cols))
            tmpPath = copy.copy(self.dsg.shortestRoverPathNodes[legIdx])
            tmpPath = [cmn.convert_VREPXY(tmpNode) for tmpNode in tmpPath]
            for tmpNode in tmpPath:
                maxRad = 5.0
                nns = cmn.neighbors(mat=legRwrds, rad=maxRad, r=tmpNode[0], c=tmpNode[1])
                for nn in zip(list(nns[0]), list(nns[1])):
                    tmpR = np.linalg.norm([nn[0] - tmpNode[0], nn[1] - tmpNode[1]])
                    legRwrds[nn] += legVal*(np.abs(tmpR-maxRad)/maxRad)

            cmap = cm.Spectral;
            cmap.set_bad(color='k')
            cmn.createArrImg(legRwrds, cmap=cmap,
                             plotTitle='Base State Rewards for Leg {}'.format(legIdx),
                             fn=os.path.join(cmn._IMG_FLDR,
                                             'base_rewards_leg{}.svg'.format(legIdx)), show=False)

        maxEdge = np.max([self.rows,self.cols])
        for i in range(len(cmn._actions)):
            tmpRewardMap = np.full(shape=(self.rows, self.cols), fill_value=0.0, dtype=np.float)

            obsRwrds = np.zeros(shape=tmpRewardMap.shape)
            # Rewards based on self.rates first.
            for r in range(self.rows):
                for c in range(self.cols):
                    if vrs == 1:
                        if np.abs(self.rates[0][cmn._actions[i]][(r, c)].item()) > obsRateThresh:
                            obsRwrds[(r, c)] = obstacle
                    elif vrs == 2:
                        if self.rates[0][cmn._actions[i]][(r, c)] <= 0:
                            obsRwrds[(r,c)] = 0.0
                        else:
                            obsRwrds[(r,c)] = obstacle*np.power(self.rates[0][cmn._actions[i]][(r, c)].item(),2)
                    elif vrs == 3:
                        if np.abs(self.rates[0][cmn._actions[i]][(r,c)]) <= critRate:
                            obsRwrds[(r, c)] = 2.0
                        else:
                            obsRwrds[(r, c)] = obstacle*(np.exp(np.abs(self.rates[0][cmn._actions[i]][(r,c)]))-np.exp(critRate))
                    elif vrs == 4:
                        absRate = np.abs(self.rates[0][cmn._actions[i]][(r,c)])
                        tmpCritRate = critRate
                        if self.rates[0][cmn._actions[i]][(r,c)] < 0:
                            tmpCritRate = negCritRate
                        if absRate > tmpCritRate:
                            obsRwrds[(r, c)] = obstacle-np.abs(absRate-tmpCritRate)
                        else:
                            obsRwrds[(r, c)] = obstacle*cmn.sigmoid(cmn.scaleRange([0, float(tmpRewardMap[(r,c)])/tmpCritRate, 1], -6, 6)[1])

            maxReward = np.nanmax(obsRwrds)
            minReward = np.nanmin(obsRwrds)
            maxAbReward = np.nanmax([np.abs(maxReward), np.abs(minReward)])


            '''
            goalSlopeRad = (maxEdge / 2.0) * np.sqrt(maxEdge)
            nns = cmn.neighbors(mat=tmpRewardMap, rad=goalSlopeRad, r=self.goal[0], c=self.goal[1])
            for nn in zip(list(nns[0]), list(nns[1])):
                npNN = np.array(nn)
                goalRwrds[nn] = (1 / 5.0) * goal * (np.linalg.norm(npNN - self.goal) - goalSlopeRad)
            maxRad = 5
            fid = 0.4
            for rad in np.arange(maxRad, 0, -fid):
                nns = cmn.neighbors(mat=tmpRewardMap, rad=rad, r=self.goal[0], c=self.goal[1])
                for nn in zip(list(nns[0]), list(nns[1])):
                    if obsRwrds[nn] < 0:
                        priorInfluence = -1 * cmn.sigmoid(
                            cmn.scaleRange([0, float(obsRwrds[nn] / minReward), 1], -6, 6)[1])
                    else:
                        priorInfluence = cmn.sigmoid(
                            cmn.scaleRange([0, float(obsRwrds[nn] / maxReward), 1], -6, 6)[1])
                    goalRwrds[nn] = goal * (
                            priorInfluence + cmn.sigmoid(cmn.scaleRange([0, maxRad - rad, maxRad], -6, 6)[1]))
            goalRwrds[self.goal] = goal
            '''

            tmpRewardMap = np.add(goalRwrds,obsRwrds)
            tmpRewardMap = np.add(legRwrds,tmpRewardMap)

            #EDGES
            [tmpRewardMap[0,:], tmpRewardMap[-1,:], tmpRewardMap[:,0], tmpRewardMap[:,-1]] = [edge]*4

            cmap = cm.Spectral;
            cmap.set_bad(color='k')
            cmn.createArrImg(obsRwrds, cmap=cmap, plotTitle='Base State Rewards for Leg {} via {}'.format(legIdx,cmn._actions[i]),
                             fn=os.path.join(cmn._IMG_FLDR, 'base_rewards_l{}_{}_obs.svg'.format(legIdx,cmn._actions[i])), show=False)

            cmn.overlayArrImgs(arr1=self.height_map, arr2=obsRwrds, cmap1=cm.gray, cmap2=cmap, alpha1=1, alpha2=.8,
                               arr2Masked=False,
                               plotTitle='Base State Rewards for Leg {} via {} (with heightmap)'.format(legIdx,cmn._actions[i]),
                               fn=os.path.join(cmn._IMG_FLDR, 'base_rewards_l{}_{}_obs_ovr.svg'.format(legIdx,cmn._actions[i])),
                               show=False)

            cmn.createArrImg(goalRwrds, cmap=cmap, plotTitle='Base State Rewards for Leg {} via {}'.format(legIdx,cmn._actions[i]),
                             fn=os.path.join(cmn._IMG_FLDR, 'base_rewards_l{}_{}_goal.svg'.format(legIdx,cmn._actions[i])), show=False)

            cmn.overlayArrImgs(arr1=self.height_map, arr2=goalRwrds, cmap1=cm.gray, cmap2=cmap, alpha1=1, alpha2=.8,
                               arr2Masked=False,
                               plotTitle='Base State Rewards for Leg {} via {} (with heightmap)'.format(legIdx,cmn._actions[i]),
                               fn=os.path.join(cmn._IMG_FLDR, 'base_rewards_l{}_{}_goal_ovr.svg'.format(legIdx,cmn._actions[i])),
                               show=False)

            #GOAL AND START
            maxRad = 5.0
            fid = 0.4
            for rad in np.arange(maxRad,0,-fid):
                tmpRewardMap[cmn.neighbors(mat=tmpRewardMap, rad=rad, r=self.start[0], c=self.start[1])] = start*cmn.sigmoid(cmn.scaleRange([0,maxRad-rad,maxRad],-6,6)[1])


            cmap = cm.Spectral; cmap.set_bad(color='k')
            cmn.createArrImg(tmpRewardMap, cmap=cmap, plotTitle='Base State Rewards for Leg {} via {}'.format(legIdx,cmn._actions[i]),
                             fn=os.path.join(cmn._IMG_FLDR, 'base_rewards_l{}_{}.svg'.format(legIdx,cmn._actions[i])), show=False)

            cmn.overlayArrImgs(arr1=self.height_map, arr2=tmpRewardMap, cmap1=cm.gray, cmap2=cmap, alpha1=1, alpha2=.8,
                               arr2Masked=False,
                               plotTitle='Base State Rewards for Leg {} via {} (with heightmap)'.format(legIdx,cmn._actions[i]),
                               fn=os.path.join(cmn._IMG_FLDR, 'base_rewards_l{}_{}_ovr.svg'.format(legIdx,cmn._actions[i])),
                               show=False)

            self.rewards.update({cmn._actions[i]: tmpRewardMap.copy()})

        shutil.copy('graph_search.py',os.path.join(cmn._OUT_FLDR,'graph_search.py'))
        if ret: return self.rewards.copy()
