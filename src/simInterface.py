'''
simInterface.py - Interface with Robotnik rover in V-REP
Adapted for use in the Automated Robot Mining project
from U of U - CS 6730 - Project 4: Trajectory Optimization
helper functions were expanded for the needs of this project.
12 Dec 2018 - Johnston, Germer, and Stucki
'''

import sys, time
import numpy as np
import common as cmn
try:
    import vrep
except:
    print("Import of vrep failed. Make sure the 'vrep.py' file is in this directory.")
    sys.exit(1)

    
class Sim:
    def __init__(self):    
        self.clientID = -1
        self.rover_name = "Robotnik_Summit_XL"
        self.wheels = { "FR" : ( 0, "joint_front_right_wheel" ),
                        "BR" : ( 1, "joint_back_right_wheel"  ),
                        "BL" : ( 2, "joint_back_left_wheel"   ),
                        "FL" : ( 3, "joint_front_left_wheel"  )}
        self.dig_names = ['Dig{}'.format(i) for i in range(0,10)]
        self.dump_name = 'Dump'
        self.bin_name = 'Bin'
        self.terrain_name = 'TerrainMap'

        self.rover_handle  = None
        self.wheel_handles = None
        self.dig_handles   = None
        self.dump_handle   = None
        self.bin_handle    = None
        self.dir = "F"
        self.vel = 0

    def sim_connect(self):
        vrep.simxFinish(-1)
        self.clientID = vrep.simxStart('127.0.0.1', 19997, True, True, 5000, 5)
        if self.clientID != -1:
            vrep.simxSynchronous(self.clientID, True)
        else:
            print("Failed to connect to remote API server.")
            
    def sim_load(self, scenePathAndName, options, operationMode):
        vrep.simxLoadScene(self.clientID, scenePathAndName, options, operationMode)
        
    def sim_start(self):
        if self.clientID != -1:
            vrep.simxStartSimulation(self.clientID, vrep.simx_opmode_blocking)
        else:
            print "Unable to start sim: Invalic ClientID"
            
    def sim_stop(self):
        if self.clientID != -1:
            print("Simulation complete.")
            vrep.simxStopSimulation(self.clientID, vrep.simx_opmode_blocking)
            vrep.simxGetPingTime(self.clientID)
            vrep.simxFinish(self.clientID)
        else:
            print("Failed connecting to remote API server")
    
    
    def sim_get_handles(self):
        if self.clientID != -1:
            self.rover_handle = vrep.simxGetObjectHandle(self.clientID, self.rover_name, vrep.simx_opmode_blocking)[1]
    
            wheel_names = [self.wheels["FR"][1], self.wheels["BR"][1], 
                           self.wheels["BL"][1], self.wheels["FL"][1]]    
            self.wheel_handles = [
                    vrep.simxGetObjectHandle(self.clientID, wheel, vrep.simx_opmode_blocking)[1]
                    for wheel in wheel_names ]
            
            self.dig_handles = [
                    vrep.simxGetObjectHandle(self.clientID, dig, vrep.simx_opmode_blocking)[1]
                    for dig in self.dig_names]

            self.dump_handle = vrep.simxGetObjectHandle(self.clientID, self.dump_name, vrep.simx_opmode_blocking)[1]
            self.bin_handle  = vrep.simxGetObjectHandle(self.clientID, self.bin_name,  vrep.simx_opmode_blocking)[1]

            self.terrain_handle = vrep.simxGetObjectHandle(self.clientID, self.terrain_name, vrep.simx_opmode_blocking)[1]
    
    def sim_drive_rover(self):
        wheel_dirs = {"F" : [-1, -1,  1,  1], "B" : [ 1,  1, -1, -1],
                      "L" : [-1, -1, -1, -1], "R" : [ 1,  1,  1,  1]}
        #wheel_dirs = {"F": [-1, -1, 1, 1],   "B": [1, 1, -1, -1],
        #              "R": [-1, -1, -1, -1], "R": [1, 1, 1, 1]}
        while 1:
            for j in range(4):
                vrep.simxSetJointTargetVelocity(
                        self.clientID, 
                        self.wheel_handles[j],  
                        wheel_dirs[self.dir][j]*self.vel, 
                        vrep.simx_opmode_oneshot)
            vrep.simxSynchronousTrigger(self.clientID)
            time.sleep(0.001)

    def sim_get_vel(self, handle):
        return vrep.simxGetObjectVelocity(self.clientID, handle, vrep.simx_opmode_blocking)

    def sim_get_pos(self, handle, ref=-1):
        return vrep.simxGetObjectPosition(self.clientID, handle, ref, vrep.simx_opmode_blocking)[1]

    def sim_get_xy(self, handle, ref=-1):
        valid = False
        pos = self.sim_get_pos(handle, ref)
        noticeCount = 11
        cnt = 0
        while valid is False:
            cnt += 1
            if cnt >= noticeCount:
				print '\tsim_get_xy taking {} iterations'.format(cnt)
            valid = True
            for i in range(len(pos)):
                if np.isnan(pos[i]):
                    pos = self.sim_get_pos(handle, ref)
                    valid = False
        return (int(pos[1] / cmn._sim_res),int(pos[0] / cmn._sim_res))
        
    def sim_get_orient(self, handle, ref = -1):
        #Returns alpha (roll), beta (pitch), gamma (heading)
        orient = vrep.simxGetObjectOrientation(self.clientID, handle, ref, vrep.simx_opmode_blocking)[1]
        return [orient[i] * 180.0 / np.pi for i in range(len(orient))]

    #def sim_set_shape_texture(self, handle):
    #    vrep.simxCallScriptFunction( self.clientID, )