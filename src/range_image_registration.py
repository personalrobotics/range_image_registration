# -*- coding: utf-8 -*-
"""
Created on Mon Dec 19 15:28:23 2016

@author: bokorn
"""

import openravepy
import numpy as np
import math
import cv2

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

    def section2Equation(self, sensor_img):
        self.sensor.SimulationStep(0.01)        
        data = self.sensor.GetSensorData(openravepy.Sensor.Type.Laser)
        depth_data = data.intensity
        
        # Depth image is [h,w,3] with the channels being X,Y,Z
        depth_image = np.reshape(depth_data, [self.image_dim[1],self.image_dim[0],3])

        # Actual range image using norm of points
        range_image = np.sqrt(np.sum(np.square(depth_image), axis=2))

        # Setting all background points to NaN for filtering purposes
        z_image = depth_image[:,:,2]
        z_image[range_image == 0] = np.NaN
        range_image[range_image == 0] = np.NaN
        depth_image[:,:,2] = z_image

        # X-Y Image derivatives
        dxdi = cv2.Sobel(depth_image[:,:,0],cv2.CV_64F,1,0,ksize=5)
        dydj = cv2.Sobel(depth_image[:,:,1],cv2.CV_64F,0,1,ksize=5)        

        # Z Image derivatives
        dzdi = cv2.Sobel(depth_image[:,:,2],cv2.CV_64F,1,0,ksize=5)
        dzdj = cv2.Sobel(depth_image[:,:,2],cv2.CV_64F,0,1,ksize=5)  

        # Approximations of dZ/dX and dZ/dY
        #### Possible source of error ####
        p_image = np.divide(dzdi, dxdi)
        q_image = np.divide(dzdj, dydj)

        # dZ/dt
        #### Possible source of error ####
        dZ_image = sensor_img[:,:,2] - depth_image[:,:,2]
        
        
        A = np.zeros((6,6))
        b = np.zeros((6,1))      
        for j in range(self.image_dim[1]):
            for k in range(self.image_dim[0]):
                # R in math
                pt = np.reshape(depth_image[j,k,:], (3,1))
                X = pt[0]
                Y = pt[1]
                Z = pt[2]
                
                # Individual componants of p_i, q_i, and (Zt)_i [page 6]
                p = p_image[j,k]
                q = q_image[j,k]
                dZ = dZ_image[j,k]
                
                if(not (np.isnan(p) or np.isnan(q))):
                    # Described on page 5                    
                    r = -Y - q*Z
                    s = X + p*Z
                    t = q*X - p*Y
                    
                    # Described on page 6
                    c = np.reshape(np.array([p, q, -1, r, s, t]),(6,1))
                    A += np.dot(c, c.T)
                    b += dZ*c

        # t = [U,V,W], omega = [A,B,C]
        # tw = [U,V,W,A,B,C] = [t_x, t_y, t_z, r_x, r_y, r_z]
        #### I think this maps to [y,x,z,...] ####
        #### Possible source of error ####
        tw = np.linalg.solve(A,b)
        return tw

    def section3Equation(self, sensor_img):
        self.sensor.SimulationStep(0.01)        
        data = self.sensor.GetSensorData(openravepy.Sensor.Type.Laser)
        depth_data = data.intensity
    
        # Depth image is [h,w,3] with the channels being X,Y,Z
        depth_image = np.reshape(depth_data, [self.image_dim[1],self.image_dim[0],3])
        [h,w,d] = depth_image.shape
        
        # Actual range image using norm of points
        range_image = np.sqrt(np.sum(np.square(depth_image), axis=2))
        sensor_range_img = np.sqrt(np.sum(np.square(sensor_img), axis=2))

        # Setting all background points to NaN for filtering purposes
        z_image = depth_image[:,:,2]
        z_image[range_image == 0] = np.NaN
        range_image[range_image == 0] = np.NaN
        depth_image[:,:,2] = z_image

        # Normalized range vectors described in page 9
        r_hat_image = depth_image
        r_hat_image[:,:,0] = np.divide(r_hat_image[:,:,0], range_image)
        r_hat_image[:,:,1] = np.divide(r_hat_image[:,:,1], range_image)
        r_hat_image[:,:,2] = np.divide(r_hat_image[:,:,2], range_image)

        # X-Y Image derivatives
        dxdi = cv2.Sobel(depth_image[:,:,0],cv2.CV_64F,1,0,ksize=5)
        dydj = cv2.Sobel(depth_image[:,:,1],cv2.CV_64F,0,1,ksize=5)        

        # Z Image derivatives
        dzdi = cv2.Sobel(depth_image[:,:,2],cv2.CV_64F,1,0,ksize=5)
        dzdj = cv2.Sobel(depth_image[:,:,2],cv2.CV_64F,0,1,ksize=5)  

        # Approximations of dZ/dX and dZ/dY
        #### Possible source of error ####
        # dXdZ = np.divide(dxdi, dzdi)
        # dYdZ = np.divide(dydj, dzdj)

        # Surface normal according to A.7 [page 12]
        #### Doesn't seem right to have z component as Z ####
        #### Possible source of error ####        
        # n_d_image = np.dstack((-dzdi,-dzdj,-depth_image[:,:,2]))
        # Surface normal according http://stackoverflow.com/questions/34644101/calculate-surface-normals-from-depth-image-using-neighboring-pixels-cross-produc      
        n_d_image = np.dstack((-dzdi,-dzdj,np.ones([h,w,1])))
        # Surface normal using dX/dZ and dY/dZ
        # n_d_image = np.dstack((dXdZ,dYdZ,np.ones([h,w,1])))
        # Normalized surface normal
        n_hat_image = n_d_image
        n_mag = np.sqrt(np.sum(np.square(n_d_image), axis=2))
        n_hat_image[:,:,0] = np.divide(n_hat_image[:,:,0], n_mag)
        n_hat_image[:,:,1] = np.divide(n_hat_image[:,:,1], n_mag)
        n_hat_image[:,:,2] = np.divide(n_hat_image[:,:,2], n_mag)

        # d range/dt
        #### Possible source of error ####
        dr = sensor_range_img - range_image
        
        A = np.zeros((6,6))
        b = np.zeros((6,1))
        for j in range(self.image_dim[1]):
            for k in range(self.image_dim[0]):
                # R in math
                pt = np.reshape(depth_image[j,k,:], (3,1))
                # n_hat in the math
                n_hat = np.reshape(n_hat_image[j,k,:], (3,1))
                # r_hat in the math
                r_hat = np.reshape(r_hat_image[j,k,:], (3,1))
                
                if(not (any(np.isnan(pt)) or any(np.isnan(n_hat)) or any(np.isnan(r_hat)))):
                    # d from page 10             
                    d = np.cross(pt,n_hat, axis=0)
                    # Reorginization of integrals on page 10 into matrix form
                    #### Possible source of error ####
                    A[0:3,0:3] += np.dot(n_hat, n_hat.T)
                    A[0:3,3:6] += np.dot(n_hat, d.T)
                    A[3:6,0:3] += np.dot(d, n_hat.T)
                    A[3:6,3:6] += np.dot(d, d.T)
                    b[0:3] += -dr[j,k]*(np.dot(r_hat.T, n_hat))*n_hat
                    b[3:6] += -dr[j,k]*(np.dot(r_hat.T, n_hat))*d        
        
        # t = [U,V,W], omega = [A,B,C]
        # tw = [U,V,W,A,B,C] = [t_x, t_y, t_z, r_x, r_y, r_z]
        #### I think this maps to [y,x,z,...] ####
        #### Possible source of error ####
        tw = np.linalg.solve(A,b)
        return tw

    def vector2Trans(self, tw):
        t = tw[0:3]
        R, _ = cv2.Rodrigues(tw[3:6])
        trans = np.eye(4)
        trans[0:3, 0:3] = R
        trans[0:3,3] = t.flatten()
        return trans
        
    def displayResults(self, sensor_img, tw):
        model_image = self.getDepthImage()
        # Converts instantaneous velocities to transform
        trans = self.vector2Trans(tw)    
        # Generates new image using tranform to match sensor image
        trans_image = self.getDepthImage(trans)
        # Extract depth data        
        model_z = model_image[:,:,2]
        sensor_z = sensor_img[:,:,2]
        trans_z = trans_image[:,:,2]
        # Compare transformed image and sensor image
        err = sensor_z - trans_z
        err_mask = np.logical_and(sensor_z != 0, trans_z != 0)
        offset_mask = np.logical_xor(sensor_z != 0, model_z != 0)
        
        print 'Mean Error:', np.sum(err[err_mask].flatten())/np.sum(err_mask.flatten())
        
        fig = plt.figure()
        ax = fig.add_subplot(321)
        ax.imshow(sensor_z)
        ax.set_title('Sensor Data')
        ax.axis('off')
        ax = fig.add_subplot(322)
        ax.imshow(model_z)
        ax.set_title('Model Data')
        ax.axis('off')
        ax = fig.add_subplot(323)
        ax.imshow(trans_z)
        ax.set_title('Transformed Data')
        ax.axis('off')
        ax = fig.add_subplot(324)
        ax.imshow(offset_mask)
        ax.set_title('Sensor Model Offset')
        ax.axis('off')
        ax = fig.add_subplot(325)
        ax.imshow(err)
        ax.set_title('Transform Error')
        ax.axis('off')
        #ax = fig.add_subplot(326)
        #ax.imshow(dZ_image)
        plt.show()
        
    def __call__(self, sensor_img):
        tw =  self.section3Equation(sensor_img)
        trans = self.vector2Trans(tw)
        return trans
    

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
    
    

if __name__ == "__main__":
    from matplotlib import pyplot as plt
    try:
        # Initialize Range Image Registration with Fuze bottle        
        range_reg = RangeImageRegistration('../../pr-ordata/data/objects/fuze_bottle.kinbody.xml')
        
        # Set fuse bottle transform
        axis = np.array([1, 0, 0])
        angle = -np.pi/2
        T_model = openravepy.matrixFromAxisAngle(axis*angle)
        T_model[0:3,3] = np.array([0,-.1,0.5])
        range_reg.setTransform(T_model)

        # Create small translation for sensor image        
        axis = np.array([0, 0, 1])
        angle = -np.pi/20
        #T_model = openravepy.matrixFromAxisAngle(axis*angle)
        T_model = np.eye(4)
        #t_model = np.array([0,-.002,0])
        t_model = np.array([-.001,0,0])
        T_model[0:3,3] = t_model

        # Generate synthetic range image using local transform
        sensor_img = range_reg.getDepthImage(T_model)
            
        tw_2 = range_reg.section2Equation(sensor_img)
        range_reg.displayResults(sensor_img, tw_2)
        
        tw_3 = range_reg.section3Equation(sensor_img)
        range_reg.displayResults(sensor_img, tw_3)
        
                
    finally:
        print 'Destroy'
        del range_reg
        openravepy.RaveDestroy() 