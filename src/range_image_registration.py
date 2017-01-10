# -*- coding: utf-8 -*-
"""
Created on Mon Dec 19 15:28:23 2016

@author: bokorn
"""

import openravepy
import numpy as np
import math
import cv2

class RangeImageRegistration(object):
    def __init__(self, model_xml_filename,
                 fx = 529, fy = 525,
                 cx = 328, cy =  267,
                 near = 0.01, far = 10,
                 image_dim = (640, 480)):
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.near = near
        self.far = far
        self.image_dim = image_dim
        self.env = openravepy.Environment()
        self.env.Load(model_xml_filename)
        self.model = self.env.GetBodies()[-1]
        self.sensor = openravepy.RaveCreateSensor(self.env, 'offscreen_render_camera')
        self.sensor.SendCommand('setintrinsic ' + str(fx) + ' ' + str(fy) \
            + ' ' + str(cx) + ' ' + str(cy) + ' ' + str(near) + ' ' + str(far))
        self.sensor.SendCommand('setdims ' + str(image_dim[0]) + ' ' + str(image_dim[1]))
        self.sensor.Configure(openravepy.Sensor.ConfigureCommand.PowerOn)

        self.sensor.SendCommand('addbody ' + self.model.GetName() + ' 0 255 0')
       
        self.model.SetTransform(np.eye(4))
        self.sensor.SetTransform(np.eye(4))

    def __del__(self):
        print 'Start del'
        self.sensor.SendCommand('clearbodies')
        self.sensor.Configure(openravepy.Sensor.ConfigureCommand.PowerOff)
        self.sensor.Configure(openravepy.Sensor.ConfigureCommand.RenderDataOff)
        self.env.Destroy()
    
    def __exit__(self):
        print 'Start exit'
        self.sensor.SendCommand('clearbodies')
        self.sensor.Configure(openravepy.Sensor.ConfigureCommand.PowerOff)
        self.sensor.Configure(openravepy.Sensor.ConfigureCommand.RenderDataOff)
        self.env.Destroy()
    
    def A3_reg(self, sensor_img):
        self.sensor.SimulationStep(0.01)        
        data = self.sensor.GetSensorData(openravepy.Sensor.Type.Laser)
        depth_data = data.intensity
        depth_image = np.reshape(depth_data, [self.image_dim[1],self.image_dim[0],3])

        range_image = np.sqrt(np.sum(np.square(depth_image), axis=2))
        sensor_range_img = np.sqrt(np.sum(np.square(sensor_img), axis=2))
        
        z_image = depth_image[:,:,2]
        z_image[range_image == 0] = np.NaN
        range_image[range_image == 0] = np.NaN
        depth_image[:,:,2] = z_image

        
        dxdi = cv2.Sobel(depth_image[:,:,0],cv2.CV_64F,1,0,ksize=5)
        dydj = cv2.Sobel(depth_image[:,:,1],cv2.CV_64F,0,1,ksize=5)        

        dzdi = cv2.Sobel(depth_image[:,:,2],cv2.CV_64F,1,0,ksize=5)
        dzdj = cv2.Sobel(depth_image[:,:,2],cv2.CV_64F,0,1,ksize=5)  
        
        drdi = cv2.Sobel(range_image,cv2.CV_64F,1,0,ksize=5)
        drdj = cv2.Sobel(range_image,cv2.CV_64F,0,1,ksize=5)

        r_hat_image = depth_image
        r_hat_image[:,:,0] = np.divide(r_hat_image[:,:,0], range_image)
        r_hat_image[:,:,1] = np.divide(r_hat_image[:,:,1], range_image)
        r_hat_image[:,:,2] = np.divide(r_hat_image[:,:,2], range_image)

        n_d_image = np.dstack((-dzdi,-dzdj,depth_image[:,:,2]))
        #n_d_image = np.dstack((drdi,drdj,range_image))
        n_hat_image = n_d_image
        n_mag = np.sqrt(np.sum(np.square(n_d_image), axis=2))
        n_hat_image[:,:,0] = np.divide(n_hat_image[:,:,0], n_mag)
        n_hat_image[:,:,1] = np.divide(n_hat_image[:,:,1], n_mag)
        n_hat_image[:,:,2] = np.divide(n_hat_image[:,:,2], n_mag)

        dr = range_image - sensor_range_img
        
        #nnT = np.zeros((3,3))
        #ndT = np.zeros((3,3))
        #dnT = np.zeros((3,3))
        #ddT = np.zeros((3,3))
        #Rrnn = np.zeros((3,1))
        #Rrnd = np.zeros((3,1))
        A = np.zeros((6,6))
        b = np.zeros((6,1))
        for j in range(self.image_dim[1]):
            for k in range(self.image_dim[0]):
                pt = np.reshape(depth_image[j,k,:], (3,1))
                n_hat = np.reshape(n_hat_image[j,k,:], (3,1))
                r_hat = np.reshape(r_hat_image[j,k,:], (3,1))
                if(not (any(np.isnan(pt)) or any(np.isnan(n_hat)) or any(np.isnan(r_hat)))):
                    d = np.cross(pt,n_hat, axis=0)
                    #nnT = nnT + np.dot(n_hat, n_hat.T)
                    A[0:3,0:3] += np.dot(n_hat, n_hat.T)
                    #ndT = ndT + np.dot(n_hat, d.T)
                    A[0:3,3:6] += np.dot(n_hat, d.T)
                    #dnT = dnT + np.dot(d, n_hat.T)
                    A[3:6,0:3] += np.dot(d, n_hat.T)
                    #ddT = ddT + np.dot(d, d.T)
                    A[3:6,3:6] += np.dot(d, d.T)
                    #Rrnn = Rrnn + dr[j,k]*(np.dot(r_hat.T, n_hat))*n_hat
                    b[0:3] += dr[j,k]*(np.dot(r_hat.T, n_hat))*n_hat
                    #Rrnd = Rrnd + dr[j,k]*(np.dot(r_hat.T, n_hat))*d
                    b[3:6] += dr[j,k]*(np.dot(r_hat.T, n_hat))*d
        
        tw = np.dot(np.linalg.inv(A),-b)
        print tw
        return tw
#        axis = np.array([1, 0, 0])
#        angle = -np.pi/2
#        T_model = openravepy.matrixFromAxisAngle(axis*angle)
#        T_model[0:3,3] = np.array([0,-.1,0.5]);
#        self.model.SetTransform(np.dot(T_model, self.model.GetTransform()))
#        self.sensor.SimulationStep(0.01)
#        
    def __call__(self, sensor_img):
        self.sensor.SimulationStep(0.01)        
        data = self.sensor.GetSensorData(openravepy.Sensor.Type.Laser)
        depth_data = data.intensity
        depth_image = np.reshape(depth_data, [self.image_dim[1],self.image_dim[0],3])

        range_image = np.sqrt(np.sum(np.square(depth_image), axis=2))
        
        z_image = depth_image[:,:,2]
        z_image[range_image == 0] = np.NaN
        range_image[range_image == 0] = np.NaN
        depth_image[:,:,2] = z_image

        dxdi = cv2.Sobel(depth_image[:,:,0],cv2.CV_64F,1,0,ksize=5)
        dydj = cv2.Sobel(depth_image[:,:,1],cv2.CV_64F,0,1,ksize=5)        

        dzdi = cv2.Sobel(depth_image[:,:,2],cv2.CV_64F,1,0,ksize=5)
        dzdj = cv2.Sobel(depth_image[:,:,2],cv2.CV_64F,0,1,ksize=5)  

        drdi = cv2.Sobel(range_image,cv2.CV_64F,1,0,ksize=5)
        drdj = cv2.Sobel(range_image,cv2.CV_64F,0,1,ksize=5)

        p_image = np.divide(dzdi, dxdi)
        q_image = np.divide(dzdj, dydj)

        #p = np.divide(drdi, dxdi)
        #q = np.divide(drdi, dydi)

        dZ_image = sensor_img[:,:,2] - depth_image[:,:,2]
        
        A = np.zeros((6,6))
        b = np.zeros((6,1))
        for j in range(self.image_dim[1]):
            for k in range(self.image_dim[0]):
                pt = np.reshape(depth_image[j,k,:], (3,1))
                X = pt[0]
                Y = pt[1]
                Z = pt[2]
                p = p_image[j,k]
                q = q_image[j,k]
                dZ = dZ_image[j,k]
                
                if(not (np.isnan(p) or np.isnan(q))):
                    r = -Y - q*Z
                    s = X + p*Z
                    t = q*X - p*Y
                    A[0,0] += p*p
                    A[0,1] += p*q
                    A[0,2] += -p
                    A[0,3] += p*r
                    A[0,4] += p*s
                    A[0,5] += p*t
                    
                    A[1,1] += q*q
                    A[1,2] += -q
                    A[1,3] += q*r
                    A[1,4] += q*s
                    A[1,5] += q*t

                    A[2,2] += 1
                    A[2,3] += -r
                    A[2,4] += -s
                    A[2,5] += -t

                    A[3,3] += r*r
                    A[3,4] += r*s
                    A[3,5] += r*t

                    A[4,4] += s*s
                    A[4,5] += s*t

                    A[5,5] += t*t
                    
                    b[0] += p*dZ
                    b[1] += q*dZ
                    b[2] += -dZ
                    b[3] += r*dZ
                    b[4] += s*dZ
                    b[5] += t*dZ
                    
        for j in range(6):
            for k in range(j+1,6):
                A[k,j] = A[j,k]

        tw = np.dot(np.linalg.inv(A),b)
        print tw
        return tw
#        axis = np.array([1, 0, 0])
#        angle = -np.pi/2
#        T_model = openravepy.matrixFromAxisAngle(axis*angle)
#        T_model[0:3,3] = np.array([0,-.1,0.5]);
#        self.model.SetTransform(np.dot(T_model, self.model.GetTransform()))
#        self.sensor.SimulationStep(0.01)
#        

    def getDepthImage(self, T=None):
        if(T is not None):
            self.model.SetTransform(np.dot(T, self.model.GetTransform()))
            self.sensor.SimulationStep(0.01)        
        data = self.sensor.GetSensorData(openravepy.Sensor.Type.Laser)
        depth_data = data.intensity
        depth_image = np.reshape(depth_data, [self.image_dim[1],self.image_dim[0],3])
        range_image = np.sqrt(np.sum(np.square(depth_image), axis=2))
        if(T is not None):
            self.model.SetTransform(np.dot(np.linalg.inv(T), self.model.GetTransform()))
            self.sensor.SimulationStep(0.01)                
        #return range_image
        return depth_image

    def setTransform(self, T):
        self.model.SetTransform(np.dot(T, self.model.GetTransform()))
        self.sensor.SimulationStep(0.01)        
    
    
def format_coord_gen(img):
    numrows = img.shape[0]
    numcols = img.shape[1]

    def format_coord(x, y):      
        col = int(x+0.5)
        row = int(y+0.5)

        if col>=0 and col<numcols and row>=0 and row<numrows:
            val = img[row,col]
            return 'x=%1.4f, y=%1.4f, [%1.4f]'%(x, y, val)
        else:
            return 'x=%1.4f, y=%1.4f'%(x, y)
    
    return format_coord

if __name__ == "__main__":
    from matplotlib import pyplot as plt
    try:
        range_reg = RangeImageRegistration('../../pr-ordata/data/objects/fuze_bottle.kinbody.xml')
        
        axis = np.array([1, 0, 0])
        angle = -np.pi/3
        T_model = np.eye(4)

        axis = np.array([1, 0, 0])
        angle = -np.pi/2
        T_model = openravepy.matrixFromAxisAngle(axis*angle)
        T_model[0:3,3] = np.array([0,-.1,0.5])
        
        range_reg.setTransform(T_model)
            
        axis = np.array([0, 0, 1])
        angle = -np.pi/20
        T_model = openravepy.matrixFromAxisAngle(axis*angle)
        T_model = np.eye(4)
        T_model[0:3,3] = np.array([0,-.05,0])

        depth_img_orig = range_reg.getDepthImage()
        depth_img_rot = range_reg.getDepthImage(T_model)
            
        fig = plt.figure()    
        ax = fig.add_subplot(211)
        ax.imshow(depth_img_orig[:,:,2])
        ax = fig.add_subplot(212)
        ax.imshow(depth_img_rot[:,:,2])
        plt.show()
    
        range_reg(depth_img_rot)
                
    finally:
        print 'Destroy'
        del range_reg
        openravepy.RaveDestroy() 