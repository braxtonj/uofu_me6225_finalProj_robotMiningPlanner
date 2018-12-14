#!/usr/bin/env python

'''
Converts the environment to a simple polygonal environment and runs A* search on the start, dig and dump locations to find the optimal ordering.

Also, the shortest path is used in the reward function later on.
'''

import numpy as np
from scipy import stats
import matplotlib.pyplot as plotter
import scipy.spatial.ckdtree as ckdt
import matplotlib.colors as mcl
from math import pi
from PIL import Image
from collisions import PolygonEnvironment
import time, heapq, traceback, sys, pickle, os, shutil
import copy
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import cm
import common as cmn
import skimage.morphology as skimrph
import skimage.filters as skifltr
import cv2 as ocv
import networkx as nx
import itertools, pickle

np.warnings.filterwarnings('ignore')

_DEBUG = True
_ALL_DEBUG = True
_DEBUG_LIL = True


freeNodeSize   = .1
freeNodeColor  = 'k'
digNodeSize    = 200
digNodeColor   = 'seagreen'
obsNodeSize    = 10
obsNodeColor   = 'r'
startNodeSize  = 200
startNodeColor = 'mediumvioletred'
dumpNodeSize   = 200
dumpNodeColor  = 'gold'
pathNodeSize   = 130
pathEdgeWidth  = 1
obsEdgeColor   = 'b'
obsEdgeWidth   = .05
othrEdgeColor  = 'dimgrey'
othrEdgeWidth  = .05
imgBGColor     = 'ghostwhite'
pathColormap   = 'Blues'
figWidth       = 12


def genFullyConGrid(maxr, maxc, rdim):
    tmpGrid = nx.grid_2d_graph(maxr, maxc)
    nx.set_edge_attributes(tmpGrid, rdim, 'r')
    print '\tadding diags...'
    diagActions = ['ne', 'se', 'sw', 'nw']
    diagEdges = []
    # ADD DIAG EDGES
    for r in range(maxr):
        for c in range(maxc):
            for diag in diagActions:
                ((r1, c1), (r2, c2)) = cmn.getRCAfterAction(r, c, diag)  # SANS ROW FLIP IN CMN
                if r2 >= 0 and r2 < r and c2 >= 0 and c2 < c:
                    diagEdges.append(((r, c), (r2, c2)))
    tmpGrid.add_edges_from(diagEdges, r=np.sqrt(2) * cmn._sim_res)
    return tmpGrid.copy()



class DigSiteGraph:
    def __init__(self, g, env, start):
        '''
        Initialize a PRM planning instance
        '''
        self.start = start
        self.envFN = env
        self.rows = g.rows
        self.cols = g.cols
        self.rateOrder = g.rateOrder
        self.heightMap = g.height_map.copy()
        self.rates = g.calcRatesOfChange(n=self.rateOrder, ret=True, dbg=False)
        self.digSites = g.digs
        self.dumpSites = [g.dump]
        self.n = len(g.digs)
        self.dim = len(g.digs[0])

        # Set later.
        self.limits = None
        self.map_limits = None
        self.ranges = None

        self.badRatioMin = 0
        self.conProb = 0.05

    def genMap(self, env):
        _PENALTY_SCALE = 5
        dbg = True
        if dbg: print 'generating map...'
        mapFN = os.path.join(cmn._OUT_FLDR,'genMap.txt')
        mapFileLines = ['Bounds: {XMIN} {XMAX} {YMIN} {YMAX}'.format(XMIN=0,XMAX=self.cols-1,YMIN=0,YMAX=self.rows-1),'RectRobot: {ROBO_H} {ROBO_W}'.format(ROBO_H=cmn._ROBOT_2D_SIZE[0],ROBO_W=cmn._ROBOT_2D_SIZE[1])]
        im = Image.open(env)
        png = np.array(im)
        png = np.rot90(png, 2)
        (self.rows, self.cols, is_rgb) = png.shape
        print "r", self.rows, "c", self.cols
        self.discreteObsMap = np.zeros((self.rows, self.cols), dtype=np.float)  #Occupancy grid for obstacles
        numActions = len(cmn._actions)
        thresh = np.arctan(np.pi/6)
        print 'thresh: {}'.format(thresh)
        ord = 1 # Just look at d/dx in each action direction
        print "Converting IMG to discreteObsMap (occupancy grid)..."
        for r in range(self.rows):
            for c in range(self.cols):
                if any([np.abs(self.rates[ord-1][act][(r,c)]) > thresh for act in cmn._actions]):
                    self.discreteObsMap[(r,c)] = 1
                else:
                    self.discreteObsMap[(r,c)] = 0
        im.close()
        cmap2 = mcl.LinearSegmentedColormap.from_list(name='MaskedColorMap',
                                                      colors=[cm.Spectral(256),cm.Spectral(256)],
                                                      N=2)
        tmpTitle = 'Discrete Obstacle Map \n Dig Site Ordering Planner - PRE'
        cmn.overlayArrImgs(arr1=self.heightMap, arr2=np.ma.masked_where(self.discreteObsMap==0,self.discreteObsMap,copy=True),
                           plotTitle=tmpTitle, cmap1=cm.gray, cmap2=cmap2, alpha1=1, alpha2=.7, arr2Masked=True,
                           fn=os.path.join(cmn._IMG_FLDR, 'discreteObsMap_digPlanner_pre.svg'), show=False)

        self.discreteObsMap = skimrph.erosion(self.discreteObsMap,skimrph.disk(.75))
        #self.discreteObsMap = skifltr.gaussian(self.discreteObsMap,sigma=1)
        self.discreteObsMap = skimrph.dilation(self.discreteObsMap,skimrph.disk(1.5))

        tmpTitle = 'Discrete Obstacle Map \n Dig Site Ordering Planner - POST'
        cmn.overlayArrImgs(arr1=self.heightMap, arr2=np.ma.masked_where(self.discreteObsMap==0,self.discreteObsMap,copy=True),
                           plotTitle=tmpTitle, cmap1=cm.gray, cmap2=cmap2, alpha1=1, alpha2=.7, arr2Masked=True,
                           fn=os.path.join(cmn._IMG_FLDR, 'discreteObsMap_digPlanner_post.svg'), show=False)

        cvMap = self.discreteObsMap.astype(np.uint8)
        contours,hierarchy = ocv.findContours(cvMap.copy(),ocv.RETR_TREE,ocv.CHAIN_APPROX_SIMPLE)
        #Only care about outter shapes.
        outterNdx = []
        for i in range(len(hierarchy[0])):
            if hierarchy[0][i][3] == -1: outterNdx.append(i)
        contours = [contours[i] for i in outterNdx]
        for obstacle in contours:
            tmpObstacle = 'Obstacle: '
            #print obstacle
            #raw_input('cont..')
            for vrtx in obstacle:
                tmpObstacle += '{} {} '.format(vrtx[0][0],vrtx[0][1])
            mapFileLines.append( tmpObstacle )

        normedHeightMap = (255.0*(self.heightMap - np.min(self.heightMap))/np.ptp(self.heightMap)).astype(np.uint8)

        heightMapImg = np.stack((normedHeightMap.copy(),)*3,-1)
        heightMapImg = heightMapImg.astype(np.uint8)
        heightMapImg = ocv.applyColorMap(heightMapImg,ocv.COLORMAP_AUTUMN)
        ocv.drawContours(image=heightMapImg,contours=contours,contourIdx=-1,color=(255,194,31),thickness=1)
        ocv.namedWindow('contouredMaps', ocv.WINDOW_AUTOSIZE)
        figSize = cmn.calcNewFigSize(orig=heightMapImg,maxSize=cmn._OCV_MaxImgSize, integers=True)
        resizedHeightMapImg = ocv.resize(heightMapImg, figSize, interpolation=ocv.INTER_CUBIC)
        ocv.imshow("contouredMaps", resizedHeightMapImg)
        ocv.imwrite(os.path.join(cmn._IMG_FLDR,"contourMap.png"), resizedHeightMapImg)
        #ocv.waitKey()
        ocv.destroyAllWindows()

        if dbg: print 'writng map...'
        with open(mapFN, 'w') as f:
            f.write('\n'.join(mapFileLines))

        return mapFN

    def build(self,savePickle=True,grabPickle=False):
        startTime = time.time()
        print 'building dig site graph'
        a_star_heuristic = lambda a, b: np.linalg.norm(np.array(a) - np.array(b))

        if not grabPickle:
            self.genMapFN = self.genMap(self.envFN)
        else:
            input = open('data_001_genMapFN.pkl', 'rb')
            self.genMapFN = pickle.load(input)
            input.close()
        if savePickle:
            output = open('data_001_genMapFN.pkl', 'wb')
            pickle.dump(self.genMapFN,output,pickle.HIGHEST_PROTOCOL)
            output.close()

        self.pe = PolygonEnvironment()
        self.pe.read_env(self.genMapFN)

        if not grabPickle:
            self.limits = self.pe.lims
            self.map_limits = [[self.pe.x_min, self.pe.x_max], [self.pe.y_min, self.pe.y_max]]
            self.ranges = self.limits[:, 1] - self.limits[:, 0]

            print '\tgenerating dig occupancy grid'
            self.occupancyGrid = np.zeros(shape=(self.rows,self.cols), dtype=np.uint)
            self.obsNodes = []
            self.startNodes = []
            self.digNodes = []
            self.dumpNodes = []
            self.freeNodes = []
            self.nodeTypes = {}
            digOccStartTime = time.time()
            for r in range(self.rows):
                print '\t\trow {} of {}'.format(r,self.rows)  #SOOOOOO SLOW
                for c in range(self.cols):
                    if self.pe.test_collisions((r,c)) or (r==0 or r==self.rows-1 or c==0 or c==self.cols-1): #REPLACE WITH BULLET
                        self.occupancyGrid[(r, c)] = cmn._OG_OBS    #OR EDGE
                        self.obsNodes.append((r,c))
                    elif (r,c) == self.start:
                        self.occupancyGrid[(r, c)] = cmn._OG_START
                        self.startNodes.append((r,c))
                        self.freeNodes.append((r,c))
                    elif (r,c) in self.digSites:
                        self.occupancyGrid[(r, c)] = cmn._OG_DIG
                        self.digNodes.append((r,c))
                        self.freeNodes.append((r,c))
                    elif (r,c) in self.dumpSites:
                        self.occupancyGrid[(r, c)] = cmn._OG_DUMP
                        self.dumpNodes.append((r,c))
                        self.freeNodes.append((r,c))
                    else:
                        self.occupancyGrid[(r, c)] = cmn._OG_FREE
                        self.freeNodes.append((r,c))
                    self.nodeTypes.update({(r,c):self.occupancyGrid[(r, c)]})
            print '\t\toccupancy grid took {} s'.format(time.time()-digOccStartTime)
            print '\tgenerating dig nx grid'

            self.origBaseGrid = genFullyConGrid(maxr=self.rows, maxc=self.cols, rdim=cmn._sim_res)
            nx.set_node_attributes(self.origBaseGrid, self.nodeTypes, 'type')
            self.baseGrid = self.origBaseGrid.copy()

            self.nodeSizes = []
            self.nodeColors = []
            self.edgeColors = []
            self.edgeWidths = []
            for node in self.origBaseGrid.nodes(data=True):
                if node[1]['type'] ==cmn._OG_FREE:  # FREE SPACE
                    self.nodeSizes.append(freeNodeSize)
                    self.nodeColors.append(freeNodeColor)
                if node[1]['type'] == cmn._OG_DIG:  # DIG
                    self.nodeSizes.append(digNodeSize)
                    self.nodeColors.append(digNodeColor)
                if node[1]['type'] == cmn._OG_OBS:  # OBS
                    self.nodeSizes.append(obsNodeSize)
                    self.nodeColors.append(obsNodeColor)
                if node[1]['type'] == cmn._OG_START:  # START
                    self.nodeSizes.append(startNodeSize)
                    self.nodeColors.append(startNodeColor)
                if node[1]['type'] == cmn._OG_DUMP:  # Dump
                    self.nodeSizes.append(dumpNodeSize)
                    self.nodeColors.append(dumpNodeColor)

            rmvEdgs = []
            for edg in self.baseGrid.edges(data=True):
                if (edg[0] in self.freeNodes and edg[1] in self.obsNodes) or \
                        (edg[1] in self.freeNodes and edg[0] in self.obsNodes):
                    rmvEdgs.append(edg)
            self.baseGrid.remove_edges_from(rmvEdgs)
            self.origBaseGrid.remove_edges_from(rmvEdgs) # BOTH SHOULD BE SEPARATED

            # COULD MERGE OBS NODES INTO ONE BUT WOULD
            #   REQUIRE DIFFERENT COLLISION CHECKING LOGIC BELOW

            # The above does not find "cross-over" connections..

            print '\tgenerating obstacle nx grid...'
            obsGridStartTime = time.time()
            self.obsGrid = self.baseGrid.subgraph(self.obsNodes).copy()
            rmvEdgs = []
            ndx = 0
            obsSize = len(list(self.obsGrid.edges()))
            edgeCheckSubStart = time.time()

            # Skip crossover check for less accurate but MUCH faster execution.  This just removes free connections that cross the line of an obstacle connection
            checkCrossOver = False
            if checkCrossOver:
                for edg in self.obsGrid.edges(data=True):
                    if ndx%(int(np.ceil(obsSize*.01)))==0:
                        print '\t\t{} of {} obs edges checked in {} S'.format(ndx,obsSize,time.time()-edgeCheckSubStart)
                        edgeCheckSubStart = time.time()

                    #np.any([(edg[0]==(r2,c2) or edg[1]==(r2,c2)) for ((r1,c1),(r2,c2)) in [cmn.getRCAfterAction(edg[0][0], edg[0][1], tmpAct) for tmpAct in cmn._actions]])
                    ''' THIS IS SOOOO MUCH SLOWER..  THOUGHT I could look around node of interest rather than every node
                                    for edgBase in [tmpedg for tmpedg in self.baseGrid.edges(data=True) if np.any([(edg[0]==(r2,c2) or edg[1]==(r2,c2)) for ((_,_),(r2,c2)) in [cmn.getRCAfterAction(edg[0][0], edg[0][1], tmpAct) for tmpAct in cmn._actions]])]:
                    '''
                    for edgBase in self.baseGrid.edges():
                        if not edgBase in self.obsGrid.edges():
                            if self.pe.line_line_collision(edg, edgBase):
                                rmvEdgs.append(edgBase)
                    ndx += 1
                self.baseGrid.remove_edges_from(rmvEdgs)
            
            for edg in self.origBaseGrid.edges():
                if edg[0] in self.obsNodes or edg[1] in self.obsNodes:
                    self.edgeColors.append(obsEdgeColor)
                    self.edgeWidths.append(obsEdgeWidth)
                else:
                    self.edgeColors.append(othrEdgeColor)
                    self.edgeWidths.append(othrEdgeWidth)

            print '\t\tobstacle grid took {} s'.format(time.time()-obsGridStartTime)
            print '\tsaving initial base grid...'
            #print self.origBaseGrid.nodes()
            nx.draw_networkx(self.origBaseGrid,
                             pos={k:np.array(k) for k in self.origBaseGrid.nodes()},
                             node_size=self.nodeSizes,
                             node_color=self.nodeColors,
                             edge_color=self.edgeColors,
                             linewidths=self.edgeWidths,
                             with_labels=False)
            plt.xlim((-1, self.cols))
            plt.ylim((-1, self.rows))
            fig = plt.gcf()
            rc_ratio = float(self.rows)/self.cols
            fig.set_size_inches(11, int(np.ceil(11 * rc_ratio)))
            ax = plt.gca()
            ax.set_facecolor('ghostwhite')
            bbox_props = dict(boxstyle="square", fc="w", ec="0.7", alpha=0.5)
            for digIdx in range(len(self.digNodes)):
                tmpTxtPos = list(self.digNodes[digIdx])
                tmpTxtPos[1] -= 1.5
                ax.text(tmpTxtPos[0], tmpTxtPos[1], 'Dig'.format(digIdx + 1), ha="center", va="center", size=20,
                        bbox=bbox_props)
            for startIdx in range(len(self.startNodes)):
                tmpTxtPos = list(self.startNodes[startIdx])
                tmpTxtPos[1] -= 1.5
                ax.text(tmpTxtPos[0], tmpTxtPos[1], 'Start', ha="center", va="center", size=20,
                        bbox=bbox_props)
            for dumpIdx in range(len(self.dumpNodes)):
                tmpTxtPos = list(self.dumpNodes[dumpIdx])
                tmpTxtPos[1] -= 1.5
                ax.text(tmpTxtPos[0], tmpTxtPos[1], 'Dump', ha="center", va="center", size=20,
                        bbox=bbox_props)
            plt.savefig(os.path.join(cmn._IMG_FLDR, 'dig_site_ordering_r{}_c{}_d{}_INIT.svg'.format(
                self.rows,self.cols, len(self.digNodes))))
            plt.draw()
            plt.cla()
            plt.clf()
            plt.close()

            print '\tgenerating free nx grid...'
            self.freeGrid = self.baseGrid.subgraph(self.freeNodes).copy()
            self.digGrid = self.baseGrid.subgraph(self.digNodes).copy()
        else:
            print '\topening pickles... '; sys.stdout.flush();
            self.freeGrid = nx.read_gpickle('data_001_freeGrid.pkl')
            self.digGrid = nx.read_gpickle('data_001_digGrid.pkl')
            self.baseGrid = nx.read_gpickle('data_001_baseGrid.pkl')
            self.origBaseGrid = nx.read_gpickle('data_001_origBaseGrid.pkl')

            input = open('data_001.pkl', 'rb')
            self.genMapFN = pickle.load(input)
            self.limits = pickle.load(input)
            self.map_limits = pickle.load(input)
            self.ranges = pickle.load(input)
            self.discreteObsMap = pickle.load(input)
            self.rows = pickle.load(input)
            self.cols = pickle.load(input)
            self.nodeTypes = pickle.load(input)
            self.occupancyGrid = pickle.load(input)
            self.nodeSizes = pickle.load(input)
            self.nodeColors = pickle.load(input)
            self.edgeColors = pickle.load(input)
            self.edgeWidths = pickle.load(input)
            self.obsNodes = pickle.load(input)
            self.startNodes = pickle.load(input)
            self.digNodes = pickle.load(input)
            self.dumpNodes = pickle.load(input)
            self.freeNodes = pickle.load(input)
            input.close()

        if savePickle:
            print '\tsaving pickles...'; sys.stdout.flush();
            nx.write_gpickle(self.freeGrid, 'data_001_freeGrid.pkl')
            nx.write_gpickle(self.digGrid, 'data_001_digGrid.pkl')
            nx.write_gpickle(self.baseGrid, 'data_001_baseGrid.pkl')
            nx.write_gpickle(self.origBaseGrid, 'data_001_origBaseGrid.pkl')

            output = open('data_001.pkl', 'wb')
            pickle.dump(self.genMapFN,output,pickle.HIGHEST_PROTOCOL)
            pickle.dump(self.limits,output,pickle.HIGHEST_PROTOCOL)
            pickle.dump(self.map_limits,output,pickle.HIGHEST_PROTOCOL)
            pickle.dump(self.ranges,output,pickle.HIGHEST_PROTOCOL)
            pickle.dump(self.discreteObsMap,output,pickle.HIGHEST_PROTOCOL)
            pickle.dump(self.rows,output,pickle.HIGHEST_PROTOCOL)
            pickle.dump(self.cols,output,pickle.HIGHEST_PROTOCOL)
            pickle.dump(self.nodeTypes,output,pickle.HIGHEST_PROTOCOL)
            pickle.dump(self.occupancyGrid,output,pickle.HIGHEST_PROTOCOL)
            pickle.dump(self.nodeSizes,output,pickle.HIGHEST_PROTOCOL)
            pickle.dump(self.nodeColors,output,pickle.HIGHEST_PROTOCOL)
            pickle.dump(self.edgeColors,output,pickle.HIGHEST_PROTOCOL)
            pickle.dump(self.edgeWidths,output,pickle.HIGHEST_PROTOCOL)
            pickle.dump(self.obsNodes,output,pickle.HIGHEST_PROTOCOL)
            pickle.dump(self.startNodes,output,pickle.HIGHEST_PROTOCOL)
            pickle.dump(self.digNodes,output,pickle.HIGHEST_PROTOCOL)
            pickle.dump(self.dumpNodes,output,pickle.HIGHEST_PROTOCOL)
            pickle.dump(self.freeNodes,output,pickle.HIGHEST_PROTOCOL)
            output.close()
        print '\tgrid generation took {} s...\n'.format(time.time() - startTime)

        digPathPlanningStartTime = time.time()
        self.digOrderings = list(itertools.permutations(self.digNodes))
        #self.digOrderings =[self.digNodes]   # JUST ONCE

        self.shortestRoverPathNodes = None
        self.shortestRoverPathEdges = None
        self.shortestRoverPathLength = 0.0
        self.pathLengths = []
        self.shortestDigOrder = []
        allBroken = True
        while allBroken:
            print '\n\tFind dig paths for {} dig locations'.format(len(self.digNodes))
            for digOrderIdx in range(len(self.digOrderings)):
                digOrder = self.digOrderings[digOrderIdx]
                # print digOrder
                print '\n\t\tChecking out combo {} of {}'.format(digOrderIdx + 1, len(self.digOrderings))
                startPos = self.startNodes[0]
                currPos = startPos
                dumpPos = self.dumpNodes[0]
                roverPathNodes = []
                roverPathEdges = []
                MutatedBaseGrid = self.baseGrid.copy()
                totalLength = 0.0
                print '\t\tCalculating dig ', ;
                sys.stdout.flush()
                broken = False
                for digIdx in range(len(digOrder)):
                    print '{} at {}..  '.format(digIdx + 1, digOrder[digIdx]), ;
                    sys.stdout.flush()
                    try:
                        tmpPath = nx.astar_path(MutatedBaseGrid,
                                                source=currPos,
                                                target=digOrder[digIdx],
                                                heuristic=a_star_heuristic,
                                                weight='r')
                        edgs = list(zip(tmpPath, tmpPath[1:]))
                        tmpLen = 0.0
                        for edg in edgs:
                            try:
                                tmpLen += nx.get_edge_attributes(MutatedBaseGrid, 'r')[edg]
                            except:
                                pass
                        roverPathNodes.append(tmpPath)
                        roverPathEdges.append(edgs)
                        totalLength += tmpLen
                        currPos = digOrder[digIdx]
                    except:
                        # print '\t\tfailed for dig {} on {} --> {}\n{}\n'.format(digIdx+1,currPos,digOrder[digIdx],traceback.format_exc())
                        broken = True
                        break
                    try:
                        tmpPath = nx.astar_path(MutatedBaseGrid,
                                                source=currPos,
                                                target=dumpPos,
                                                heuristic=a_star_heuristic,
                                                weight='r')
                        edgs = list(zip(tmpPath, tmpPath[1:]))
                        tmpLen = 0.0
                        for edg in edgs:
                            try:
                                tmpLen += nx.get_edge_attributes(MutatedBaseGrid, 'r')[edg]
                            except:
                                pass
                        roverPathNodes.append(tmpPath)
                        roverPathEdges.append(edgs)
                        totalLength += tmpLen
                        currPos = dumpPos
                    except:
                        # print '\t\tfailed for dump {} on {} --> {}\n{}\n'.format(digIdx+1,currPos,dumpPos,traceback.format_exc())
                        broken = True
                        break
                    MutatedBaseGrid.remove_node(digOrder[digIdx])
                if broken: continue
                self.pathLengths.append(totalLength)
                if self.shortestRoverPathNodes is None:
                    self.shortestRoverPathNodes = copy.copy(roverPathNodes)
                    self.shortestRoverPathEdges = copy.copy(roverPathEdges)
                    self.shortestDigOrder = copy.copy(digOrder)
                    self.shortestRoverPathLength = totalLength
                else:
                    if totalLength < self.shortestRoverPathLength:
                        self.shortestRoverPathNodes = copy.copy(roverPathNodes)
                        self.shortestRoverPathEdges = copy.copy(roverPathEdges)
                        self.shortestDigOrder = copy.copy(digOrder)
                        self.shortestRoverPathLength = totalLength

            if self.shortestRoverPathNodes is None:
                allBroken = True
                print '\t\tFAILURE: Removing the furthest dig site... ',; sys.stdout.flush();
                tmpLens = []
                tmpDigs  = []
                for digPos in self.digOrderings[0]:
                    tmpPath = nx.astar_path(self.baseGrid,
                                            source=self.startNodes[0],
                                            target=digPos,
                                            heuristic=a_star_heuristic,
                                            weight='r')
                    edgs = list(zip(tmpPath, tmpPath[1:]))
                    tmpLen = 0.0
                    for edg in edgs:
                        try:
                            tmpLen += nx.get_edge_attributes(self.baseGrid, 'r')[edg]
                        except:
                            pass
                    tmpLens.append(tmpLen)
                    tmpDigs.append(digPos)

                maxIdx = np.nanargmax(tmpLens)
                rmvNode = copy.copy(tmpDigs[maxIdx])

                self.basegrid.remove_node(rmvNode)
                self.digGrid.remove_node(rmvNode)
                self.freeGrid.remove_node(rmvNode)
                self.occupancyGrid[rmvNode] = cmn._OG_RMVD
                self.digNodes.remove(rmvNode)
                print 'removed {}'.format(rmvNode); sys.stdout.flush();
            else:
                allBroken = False

                print '\n\tPATH LENGTH STATS\n\n{}\n\n'.format(stats.describe(self.pathLengths))

                print '\tdig ordering took {} s...'.format(time.time()-digPathPlanningStartTime)

                print '\tfound dig order to be {}'.format(self.shortestDigOrder)

                print '\tplotting total path of {}...'.format(self.shortestRoverPathLength)

                nx.draw_networkx(self.origBaseGrid,
                             pos={k:np.array(k) for k in self.origBaseGrid.nodes()},
                             node_size=self.nodeSizes,
                             node_color=self.nodeColors,
                             edge_color=self.edgeColors,
                             linewidths=self.edgeWidths,
                             with_labels=False)

                for i in range(len(self.shortestRoverPathNodes)):
                    try:
                        nx.draw_networkx_nodes(self.origBaseGrid,
                                               pos={k:np.array(k) for k in self.shortestRoverPathNodes[i]},
                                               nodelist=self.shortestRoverPathNodes[i],
                                               node_color=[i]*len(self.shortestRoverPathNodes[i]),
                                               vmin=-5, vmax=len(self.shortestRoverPathNodes)+1,
                                               cmap=pathColormap,
                                               node_size=pathNodeSize)
                    except:
                        print 'skipping node {} in path due to\n{}\n'.format(self.shortestRoverPathNodes[i],traceback.format_exc())

                for i in range(len(self.shortestRoverPathEdges)):
                    try:
                        nx.draw_networkx_edges(self.origBaseGrid,
                                           pos={k:np.array(k) for k in self.shortestRoverPathNodes[i]},
                                           edgelist=self.shortestRoverPathEdges[i],
                                           width=pathEdgeWidth)
                    except:
                        print 'skipping edge {} in path due to\n{}\n'.format(self.shortestRoverPathEdges[i],traceback.format_exc())

                nx.draw_networkx_nodes(self.origBaseGrid,
                                       pos={k:np.array(k) for k in self.digNodes},
                                       nodelist=self.digNodes,
                                       node_color=digNodeColor,
                                       node_size=digNodeSize)
                nx.draw_networkx_nodes(self.origBaseGrid,
                                       pos={k:np.array(k) for k in self.startNodes},
                                       nodelist=self.startNodes,
                                       node_color=startNodeColor,
                                       node_size=startNodeSize)
                nx.draw_networkx_nodes(self.origBaseGrid,
                                       pos={k:np.array(k) for k in self.dumpNodes},
                                       nodelist=self.dumpNodes,
                                       node_color=dumpNodeColor,
                                       node_size=dumpNodeSize)

                plt.xlim((-1,self.rows))
                plt.ylim((-1,self.cols))
                fig = plt.gcf()
                rc_ratio = float(self.rows)/self.cols
                fig.set_size_inches(figWidth,int(np.ceil(figWidth*rc_ratio)))
                ax = plt.gca()
                ax.set_facecolor(imgBGColor)

                bbox_props = dict(boxstyle="square", fc="w", ec="0.7", alpha=0.5)
                for digIdx in range(len(self.shortestDigOrder)):
                    tmpTxtPos = list(self.shortestDigOrder[digIdx])
                    tmpTxtPos[1] -= 1.5
                    ax.text(tmpTxtPos[0], tmpTxtPos[1], 'Dig {}'.format(digIdx+1), ha="center", va="center", size=20,
                        bbox=bbox_props)
                for startIdx in range(len(self.startNodes)):
                    tmpTxtPos = list(self.startNodes[startIdx])
                    tmpTxtPos[1] -= 1.5
                    ax.text(tmpTxtPos[0], tmpTxtPos[1], 'Start', ha="center", va="center", size=20,
                        bbox=bbox_props)
                for dumpIdx in range(len(self.dumpNodes)):
                    tmpTxtPos = list(self.dumpNodes[dumpIdx])
                    tmpTxtPos[1] -= 1.5
                    ax.text(tmpTxtPos[0], tmpTxtPos[1], 'Dump', ha="center", va="center", size=20,
                        bbox=bbox_props)

                totTime = time.time()-startTime
                print '\ttotal execution time: {} s\n'.format(totTime)
                plt.savefig(os.path.join(cmn._IMG_FLDR,'dig_site_ordering_r{}_c{}_d{}.svg'.format(self.rows,self.cols,len(self.digNodes))))
                plt.draw()
                #plt.show()
