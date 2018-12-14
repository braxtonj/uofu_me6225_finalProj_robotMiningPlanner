"""
common.py - Global reference variables and helper functions

Written by Braxton Johnston, modified and expanded by Justin Stucki
12 Dec 2018 - Johnston, Germer, and Stucki
"""
import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt
import matplotlib.patches as pch
from matplotlib.legend_handler import HandlerPatch
from mpl_toolkits.axes_grid1 import make_axes_locatable
import sys, copy, traceback, os

from PIL import Image

'''     GLOBAL VARIABLES     '''
_actions =  ['n',   'ne',   'e',  'se',    's',    'sw',     'w',    'nw'] #Circular action order
_headings = {'n':90,'ne':45,'e':0,'se':-45,'s':-90,'sw':-135,'w':180,'nw':135}
_X = 1
_Y = 0
_VEL = 4
_MAP_CHOICE = -2
_MAP_SCALE  = 2
_MAP_FILES = [
 ('HeightMap50.png',  0.040),           #0.040m -   7.38m x  5.76m
 ('HeightMap25.png',  0.080),           #0.080m -   7.38m x  5.76m
 ('HeightMap12.png',  0.160),           #0.160m -   7.38m x  5.76m
 ('spiralUp50.png',   0.040),           #0.040m -     20m x    20m
 ('spiralUp8.png',    0.250),           #0.250m -     20m x    20m
 ('spiralDown50.png', 0.040),           #0.040m -     20m x    20m
 ('spiralDown8.png',  0.250),           #0.250m -     20m x    20m
 ('rand_terrain.png', 0.164),           #0.164m -  14.76m x 11.52m large
 ('rand_terrain.png', 0.082),           #0.082m -   7.38m x  5.76m competition
 ('obsMap_001_400.png',   0.040),       #0.040m -     16m x    16m
 ('obsMap_001_100_impass.png', 0.160),       #0.040m - 16m x    16m
 ('obsMap_001_100_pass.png',   0.160)]  #0.040m - 16m x    16m
(_map_file, _sim_res) = _MAP_FILES[_MAP_CHOICE]

_DIG_obsThresh = 1
_CALC_REWARD_VRS = 4            # 1: Obstacle via threshold     2: obs*(Rate[0][ACTION][(i,j)]^2)       3: 2 if RATE<thresh, obs*(e^(|x|)-e^thresh)     4: scaled using sigmoid        5: obs*cissoidOfDiocles(RATE)
_PI_GOAL_REWARD      = 42.0*3     # Reward value of goal
_PI_START_REWARD     = -5      # Start
_PI_OBSTACLE_REWARD  = -3*_PI_GOAL_REWARD      # Obstacle
_PI_EDGE_REWARD = _PI_OBSTACLE_REWARD*1.3      # Edge
_PI_ABSORBING_REWARD = _PI_OBSTACLE_REWARD*.5      # Absorbing
_PI_LEG_REWARD = _PI_GOAL_REWARD/27.0
_PI_ConstActChangeCountMax = 42 # Number of consecutive action change counts before considering PI complete.  Setting to None eliminates this constraint
_PI_LAMBDA  = 0.900             # Discount factor
_PI_EPSILON = .01              # Set convergence metric
_PI_ABSORBING_MIN_DIRECTED = np.ceil(.8*len(_actions)).astype(np.uint8)
_PI_DrawArrows = True
_PI_MaxFigDimIn = 9   #Inches
_OCV_MaxImgSize = 1200 #Pixels

# Occupancy grid vals
_OG_RMVD  = 7
_OG_OBS   = 4
_OG_START = 3
_OG_DIG   = 2
_OG_DUMP  = 1
_OG_FREE  = 0

# BEWARE PICKLE USE.  IT MESSES WITH PLOTS FROM NETWORKX.  EVERYTHING ELSE WORKS THOUGH.
_GRAB_PICKLE = False
_SAVE_PICKLE = True

# ROBOT VALUES IN MM:  722 l x 613 w CLEARANCE 172 WHEEL 235
#       no planning on clearance yet.
_ROBOT_2D_SIZE = (.722/_sim_res,.613/_sim_res)
_FLDR_APPEND = '_M{MAP}_L{LAMBDA}_E{EPSILON}_G{GOAL:.2f}_S{START:.2f}_O{OBSTACLE:.2f}_A{ABSORBING:.2f}_V{REWARDVRS}'.format(
    MAP=_map_file.split('.png')[0],
    GOAL=_PI_GOAL_REWARD,START=_PI_START_REWARD,OBSTACLE=_PI_OBSTACLE_REWARD,ABSORBING=_PI_ABSORBING_REWARD,
    LAMBDA=_PI_LAMBDA,EPSILON=_PI_EPSILON,REWARDVRS=_CALC_REWARD_VRS
)
if not os.path.exists('imgs'):
    os.makedirs('imgs')
if not os.path.exists('out'):
    os.makedirs('out')
_IMG_FLDR = os.path.join('imgs','imgs{}'.format(_FLDR_APPEND))
_OUT_FLDR = os.path.join('out','out{}'.format(_FLDR_APPEND))
'''   END GLOBAL VARIABLES   '''


def convert_VREPXY( pos ):
    changedPos = list(copy.copy(pos))
    changedPos = changedPos[::-1]
    changedPos = tuple(changedPos)
    return changedPos

class HandlerEllipse(HandlerPatch):
    def create_artists(self, legend, orig_handle,
                       xdescent, ydescent, width, height, fontsize, trans):
        center = 0.5 * width - 0.5 * xdescent, 0.5 * height - 0.5 * ydescent
        p = pch.Circle(xy=center, radius=3 )
        self.update_prop(p, orig_handle, legend)
        p.set_transform(trans)
        return [p]

def getNextAction(act,dIdx):
    # dIdx > 0:  CLOCKWISE         MOVEMENT  IE N --> NE
    # dIdx < 0:  COUNTER-CLOCKWISE MOVEMENT  IE N --> NW
    return _actions[(_actions.index(act)+dIdx)%len(_actions)]

def prettyStr2DMatrix(matrix,rowDelim='',gSigFigs=-1):
    # thank you https://stackoverflow.com/questions/13214809/pretty-print-2d-python-list
    gSigFigs = int(gSigFigs)
    if gSigFigs > 0:
        frmtStr = '{{:1.{}g}}'.format(gSigFigs)
        s = [[frmtStr.format(e) for e in row] for row in matrix]
    else:
        s = [[str(e) for e in row] for row in matrix]
    lens = [max(map(len, col)) for col in zip(*s)]
    fmt = '\t'.join('{{:{}}}'.format(x) for x in lens)
    table = [fmt.format(*row) for row in s]
    tmpStr = '\n{}'.format(rowDelim).join(table)
    tmpStr = '{}{}'.format(rowDelim,tmpStr)
    return tmpStr

def generateSlopedHeightmap(maxX,maxY,maxz,angle):
    '''
    x by y grid of heights z.
    starting at (0,0) increase in the direction of angle till you get to (maxX, maxY) pos

    :param maxX:   maximum x
    :param maxY:   maximum y
    :param angle:  angle to increase [angle_xTOy_dir, angle_xTOz_dir]
    :return:
    '''
    x = np.arange(0,maxX,1)
    y = np.arange(0,maxY,1)
    xx,yy = np.meshgrid(x,y)
    z = xx*np.sin(angle[0])*yy*np.sin(angle[1])
    plt.imshow(z,cmap='binary')
    plt.show()
    plt.savefig(os.path.join(_IMG_FLDR,'genMap.svg'))

def rgb2hex((r,g,b)):
    rw =  r/16
    rr = int((r/16.0 - rw)*16)
    gw =  g/16
    gr = int((g/16.0 - gw)*16)
    bw =  b/16
    br = int((b/16.0 - bw)*16)
    return ((rw << 20)+(rr << 16)+(gw << 12)+(gr << 8)+(bw << 4)+(br << 0))

def scaleRange( r, min, max ):
    r += -(np.min(r))
    r /= np.max(r)/(max-min)
    r += min
    return r

def sigmoid(x, derivative=False):
    sigm = 1. / (1. + np.exp(-x))
    if derivative:
        return sigm * (1. - sigm)
    return sigm

def cissoidOfDiocles( x, a, posFact=1.0, negFact=1.0 ):
    ''' Treats x as x>=0 Fails at x=2*a '''
    v = np.abs(x)
    if x>=0: v*posFact
    if x<0:  v*negFact
    return np.sqrt(np.power(v,3)/(2*a-v))

#Thank ou StackOverflow
#https://stackoverflow.com/questions/22550302/find-neighbors-in-a-matrix
def neighbors(mat, rad, r, c):
    ''' Grabs indexes of neighbors within some radius of (r,c).  Note that rad must be greater than 1 since we are dealing with indexes '''
    tmpR = np.arange(0,mat.shape[0])
    tmpC = np.arange(0,mat.shape[1])
    mask = np.power(tmpR[:,np.newaxis]-r,2)+np.power(tmpC[np.newaxis,:]-c,2) < np.power(rad,2)
    return np.where(mask)

def printPolicyMap( val_grid, dir_grid, start=None, goal=None, newTicks=None, fn=None, figTitle=None, show=True, figSizeScale=1.0, dbg=False, drawArrows=True ):
    if dbg: print 'generating figure...'; sys.stdout.flush()
    fig, ax = plt.subplots()
    #ax.set_position([0,.2,1,1])
    masked_array = np.ma.masked_where(False, val_grid.copy())
    #print prettyStr2DMatrix(val_grid,'\t\t\t')
    #print '\n{}'.format(prettyStr2DMatrix(masked_array,'\t\t\t'))
    cmap = cm.Spectral
    cmap.set_bad(color='k')
    im = ax.imshow(masked_array, cmap)
    im.set_interpolation('nearest')
    ax.invert_yaxis()
    if (not dir_grid is None) and _PI_DrawArrows and drawArrows:
        for r in range(val_grid.shape[0]):
            for c in range(val_grid.shape[1]):
                if (not np.isnan(val_grid[(r,c)])) and dir_grid[(r,c)]!='':
                    arrowEnd = [c - .5, r - .5]
                    arrowStart = [c - .5, r - .5]
                    if dir_grid[(r, c)] == 'n':  # Note that "up" is "down".  weird
                        arrowStart[0] += .5
                        arrowStart[1] += .25
                        arrowEnd[0] += .5
                        arrowEnd[1] += .75
                    elif dir_grid[(r, c)] == 'ne':
                        arrowStart[0] += .31
                        arrowStart[1] += .31
                        arrowEnd[0] += .69
                        arrowEnd[1] += .69
                    elif dir_grid[(r, c)] == 'e':
                        arrowStart[0] += .25
                        arrowStart[1] += .5
                        arrowEnd[0] += .75
                        arrowEnd[1] += .5
                    elif dir_grid[(r, c)] == 'se':
                        arrowStart[0] += .31
                        arrowStart[1] += .69
                        arrowEnd[0] += .69
                        arrowEnd[1] += .31
                    elif dir_grid[(r, c)] == 's':
                        arrowStart[0] += .5
                        arrowStart[1] += .75
                        arrowEnd[0] += .5
                        arrowEnd[1] += .25
                    elif dir_grid[(r, c)] == 'sw':
                        arrowStart[0] += .69
                        arrowStart[1] += .69
                        arrowEnd[0] += .31
                        arrowEnd[1] += .31
                    elif dir_grid[(r, c)] == 'w':
                        arrowStart[0] += .75
                        arrowStart[1] += .5
                        arrowEnd[0] += .25
                        arrowEnd[1] += .5
                    elif dir_grid[(r, c)] == 'nw':
                        arrowStart[0] += .69
                        arrowStart[1] += .31
                        arrowEnd[0] += .31
                        arrowEnd[1] += .69
                    tmpal = .7
                    im.axes.annotate("",
                                     xy=(arrowEnd[0], arrowEnd[1]), xycoords='data',
                                     xytext=(arrowStart[0], arrowStart[1]), textcoords='data',
                                     arrowprops=dict(arrowstyle="->, head_length={}, head_width={}".format(0.4*figSizeScale,0.2*figSizeScale),
                                                     lw=.3,
                                                     connectionstyle="arc3",
                                                     alpha=tmpal,
                                                     edgecolor=(0,0,0,tmpal),
                                                     facecolor=(0,0,0,tmpal)),
                                     )

    if not figTitle is None:
        plt.title(figTitle)
    if not start is None:
        startFaceColor = np.array([0, 197, 255, 255*.7])/255.0
        startEdgeColor = 1.1*startFaceColor.copy(); startEdgeColor[3]=.8
        for i in range(startEdgeColor.size):
            if   startEdgeColor[i] > 1.0: startEdgeColor[i] = 1.0
            elif startEdgeColor[i] < 0.0: startEdgeColor[i] = 0.0
        startCircle = plt.Circle((start[_X],start[_Y]),radius=1,edgecolor=tuple(startEdgeColor),facecolor=tuple(startFaceColor),label='Start')
        ax.add_artist(startCircle)
    if not goal is None:
        goalFaceColor = np.array([203, 68, 255, 255*.7])/255.0
        goalEdgeColor = 1.1*goalFaceColor.copy(); goalEdgeColor[3]=.8
        for i in range(goalEdgeColor.size):
            if   goalEdgeColor[i] > 1.0: goalEdgeColor[i] = 1.0
            elif goalEdgeColor[i] < 0.0: goalEdgeColor[i] = 0.0
        goalCircle = plt.Circle((goal[_X],goal[_Y]),radius=1,edgecolor=tuple(goalEdgeColor),facecolor=tuple(goalFaceColor),label='Goal')
        ax.add_artist(goalCircle)
    if not newTicks is None:
        print 'XTICKLABELS: {}\nYTICKLABELS: {}'.format([v.get_text() for v in ax.get_xticklabels()],
                                                        [v.get_text() for v in ax.get_yticklabels()])
        lblsX = [v.get_text() for v in ax.get_xticklabels()]
        lblsY = [v.get_text() for v in ax.get_yticklabels()]
        xVal = newTicks['cVal']-1
        for i in range(len(lblsX)):
            if (i+1)%2 == 0:
                lblsX[i] = str(xVal)
                xVal += 1
            else:
                lblsX[i] = ''
        yVal = newTicks['rVal'] - 1
        for i in range(len(lblsY)):
            if (i+1)%2 == 0:
                lblsY[i] = str(yVal)
                yVal += 1
            else:
                lblsY[i] = ''
        ax.set_xticklabels(lblsX)
        ax.set_yticklabels(lblsY)

    figSize = calcNewFigSize(orig=val_grid,maxSize=_PI_MaxFigDimIn*figSizeScale)
    oldFigSize = copy.copy(plt.rcParams["figure.figsize"])
    plt.rcParams['figure.figsize'] = figSize
    fig.set_figheight(figSize[_X])
    fig.set_figwidth(figSize[_Y])
    cbar_kw = {}
    divider = make_axes_locatable(im.axes)
    cax1 = divider.append_axes("right", size="5%", pad=0.05)
    cbar = im.axes.figure.colorbar(im, cax=cax1, **cbar_kw)
    cbar.ax.set_ylabel('', rotation=-90, va="bottom")

    if not start is None and not goal is None:
        leg = ax.legend([startCircle,goalCircle],["Start","Goal"], handler_map={pch.Circle: HandlerEllipse()}, loc='upper right')
        leg.get_frame().set_alpha(.5)
    elif not start is None:
        leg = ax.legend([startCircle],["Start"], handler_map={pch.Circle: HandlerEllipse()}, loc='upper right')
        leg.get_frame().set_alpha(.5)
    elif not goal is None:
        leg = ax.legend([goalCircle],["Goal"], handler_map={pch.Circle: HandlerEllipse()}, loc='upper right')
    if not start is None or not goal is None:
        leg.get_frame().set_alpha(.5)

    if not fn is None:
        if os.path.isfile(fn): os.remove(fn)
        plt.savefig(fn, bbox_inches="tight", figsize=figSize)
    if show:
        plt.show()
    else:
        plt.close()
    plt.rcParams['figure.figsize'] = oldFigSize

def createArrImg(arr,cmap=None,plotTitle='',fn=None,show=False):
    figSize = calcNewFigSize(orig=arr, maxSize=_PI_MaxFigDimIn)
    if cmap is None:
        cmap = cm.Spectral
        cmap.set_bad(color='k')
    fig, ax = plt.subplots()
    im2 = ax.imshow(arr, cmap=cmap, interpolation='nearest')
    ax.invert_yaxis()
    plt.title(plotTitle)
    cbar_kw = {}
    divider = make_axes_locatable(ax)
    cax1 = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(im2, cax=cax1, **cbar_kw)
    cbar.ax.set_ylabel('', rotation=-90, va="bottom")
    if not fn is None:
        plt.savefig(fn,figsize=figSize)
    if show:
        plt.show()
    else:
        plt.close(fig)

def overlayArrImgs(arr1,arr2,cmap1=None,cmap2=None,alpha1=.7,alpha2=.8,arr2Masked=False,plotTitle='',fn=None,show=False):
    figSize = calcNewFigSize(orig=arr1, maxSize=_PI_MaxFigDimIn)
    if cmap1 is None:
        cmap1 = cm.gray
        cmap1.set_bad(color='k')
    if cmap2 is None:
        cmap2 = cm.Spectral
        cmap2.set_bad(color='w')
    fig, ax = plt.subplots()
    fig.suptitle(plotTitle)
    im1 = ax.imshow(arr1, cmap=cmap1, alpha=alpha1, interpolation='nearest')
    if arr2Masked: im2 = ax.imshow(arr2, cmap=cmap2, alpha=alpha2, interpolation='none')
    else:          im2 = ax.imshow(arr2, cmap=cmap2, alpha=alpha2, interpolation='nearest')
    ax.invert_yaxis()
    cbar_kw = {}
    divider = make_axes_locatable(ax)
    cax1 = divider.append_axes("right", size="5%", pad=0.05)
    if not arr2Masked: cbar = fig.colorbar(im2, cax=cax1, **cbar_kw)
    else:              cbar = fig.colorbar(im1, cax=cax1, **cbar_kw)
    cbar.ax.set_ylabel('', rotation=-90, va="bottom")
    if not fn is None:
        plt.savefig(fn,figsize=figSize)
    if show:
        plt.show()
    else:
        plt.close(fig)

def calcNewFigSize(orig, maxSize=None, integers=False):
    figSize = np.array((np.float(orig.shape[_X]) / (np.max(orig.shape)), np.float(orig.shape[_Y]) / (np.max(orig.shape))))
    if integers:
        figSize = np.ceil(figSize).astype(np.uint8)
    if not maxSize is None:
        figSize = tuple(figSize*maxSize)
    else:
        figSize = tuple(figSize*_PI_MaxFigDimIn)
    return figSize

def getRCAfterAction(r,c,act):

    # FOR USE OF (R2,C2)-(R1,C1) FOR DIST IN RATE CALC

    r1 = copy.copy(r)
    c1 = copy.copy(c)
    r2 = copy.copy(r)
    c2 = copy.copy(c)
    if   act == 'n':
        r2 -= 1
        r1 += 1
    elif act == 'ne':
        r2 -= 1
        r1 += 1
        c2 += 1
        c1 -= 1
    elif act == 'e':
        c2 += 1
        c1 -= 1
    elif act == 'se':
        r2 += 1
        r1 -= 1
        c2 += 1
        c1 -= 1
    elif act == 's':
        r2 += 1
        r1 -= 1
    elif act == 'sw':
        r2 += 1
        r1 -= 1
        c2 -= 1
        c1 += 1
    elif act == 'w':
        c2 -= 1
        c1 += 1
    elif act == 'nw':
        r2 -= 1
        r1 += 1
        c2 -= 1
        c1 += 1
    return ((r1,c1), (r2,c2))


# Create a Grayscale version of the image
def convert_grayscale(image):
  # Get size
  width, height = image.size

  # Create new Image and a Pixel Map
  new = Image.create_image(width, height)
  pixels = new.load()

  # Transform to grayscale
  for i in range(width):
    for j in range(height):
      # Get Pixel
      pixel = Image.get_pixel(image, i, j)

      # Get R, G, B values (This are int from 0 to 255)
      red =   pixel[0]
      green = pixel[1]
      blue =  pixel[2]

      # Transform to grayscale
      gray = (red * 0.299) + (green * 0.587) + (blue * 0.114)

      # Set Pixel in new image
      pixels[i, j] = (int(gray), int(gray), int(gray))

    # Return new image
    return new


def merge_nodes(G, nodes, new_node, attr_dict=None, **attr):
    """
    Merges the selected `nodes` of the graph G into one `new_node`,
    meaning that all the edges that pointed to or from one of these
    `nodes` will point to or from the `new_node`.
    attr_dict and **attr are defined as in `G.add_node`.

    courtesy of https://gist.github.com/Zulko/7629206
    """

    G.add_node(new_node, attr_dict, **attr)  # Add the 'merged' node

    for n1, n2, data in G.edges(data=True):
        # For all edges related to one of the nodes to merge,
        # make an edge going to or coming from the `new gene`.
        if n1 in nodes:
            G.add_edge(new_node, n2, data)
        elif n2 in nodes:
            G.add_edge(n1, new_node, data)

    for n in nodes:  # remove the merged nodes
        G.remove_node(n)

