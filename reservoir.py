import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import matlab.engine
import matlab
from scipy.signal import argrelextrema

def gen_r_matrix(size,density):
    #Generation of Random Reservoir
    density_matrix = nx.to_numpy_array(nx.gnp_random_graph(size,density))
    rand_vector = (2 * np.random.rand(size)) - 1

    #Scaled According to the Max Eigenvalue
    eigvalues,_ = np.linalg.eig(density_matrix*rand_vector)
    max_len = np.absolute(np.amax(eigvalues))
    return (density_matrix*rand_vector)/max_len

def gen_lorenz_data(init_pos,sigma,rho,beta,tf,dt,skip,split):
    #Creation of Training Trajectory Using Matlab's ode45 Function (Runge Kutta 4th/5th Order)
    traj = np.asarray(eng.gen_lorenz(init_pos, sigma, rho, beta, tf, dt))

    skip_steps = int(skip/dt)
    traj = traj[skip_steps:]

    split_num = int(split*traj.shape[0])
    print(split_num)

    #Splits the Data at a Certain Point - After that Point To Be Predicted
    return traj[:split_num], traj[split_num:]

def lin_reg(R, U):
    #Optimization of W_Out According to Gathered R_States and Training Data
    R_T = np.transpose(R)

    return np.dot(np.dot(np.transpose(U), R_T),
                  np.linalg.inv(np.dot(R, R_T) +
                  0.0001 * np.identity(R.shape[0])))


class RC():
    def __init__(self, dim, r_size, sigma, density):
        # r_state - vector of size (r_size, 1), initially all 0s, that contains the momentary (kth) predicted reservoir data,
        #           based on the previous r_state and u (modified by r_matrix + w_in)
        # r_matrix - random reservoir matrix w size (r_size, r_size), multiplied with r_state in advance method
        # w_in - random (-0.1 < val < 0.1) matrix w size (r_size,dim) - weights for training set
        # w_out - matrix of size (dim,r_size) (initially all 0s)

        self.r_state = np.zeros(r_size)
        self.r_matrix = gen_r_matrix(r_size,density)
        self.w_in = (2 * sigma * np.random.rand(r_size,dim)) - sigma
        self.w_out = np.zeros([dim,r_size])
        self.r_size = r_size

    def advance(self,u):
        #u(t) - actual data (training set) as a 3x1 vector, representing the dimensions
        #       that will be compared to the predictions v(t), v1(t) = u2(t)

        #r_state - "advanced" using given eqn
        self.r_state = 1/(1+np.exp(
                       -(np.dot(self.r_matrix,self.r_state) + np.dot(self.w_in,u))))

    def readout(self):
        #v(t) prediction data, dimx1 vector based on r_state
        v = np.dot(self.w_out, self.r_state)
        return v

    def train(self,u_arr):
        #R - 300xr_size matrix initially filled w 0s, traj.shape[0] = tf/dt
        r_state_arr = np.zeros([self.r_size,u_arr.shape[0]])

        #Sets the first n columns of R to r_state (300x1) which is advanced - generated r_states given current and input u
        for i in range(u_arr.shape[0]):
            r_state_arr[:,i] = self.r_state
            self.advance(u_arr[i])

        #Optimizing W_Out Based on Training Data from U Array
        self.w_out = lin_reg(r_state_arr, u_arr)

    def predict(self, t_steps):
        v_arr = np.zeros([t_steps,3])

        # Sets each ith row of predicted to the ith prediction made (3 dimensions), r_state advanced after each setting
        for i in range(t_steps):
            v_arr[i] = self.readout()
            self.advance(v_arr[i])

        return v_arr

if __name__ == '__main__':
    #Training Data
    eng = matlab.engine.start_matlab()
    dt, tf = 0.02, 250.0
    train_data, val_data = gen_lorenz_data(matlab.double([1,1,1]), 10.0, 8/3, 28.0, tf, dt, 25.0, 0.8)
    t_range = np.arange(0, tf, dt)

    #Reservoir Predictions
    network = RC(3, 300, 0.1, 0.05)
    network.train(train_data)
    prediction = network.predict(val_data.shape[0])

    #x,y,z Comparison Plots of Predicted (V) and Actual (U) Data
    plt.figure(1)
    plt.subplot(311)
    plt.plot(t_range[-prediction.shape[0]:], prediction[:,0])
    plt.plot(t_range[-val_data.shape[0]:], val_data[:,0])
    plt.xlabel('Time (t)')
    plt.ylabel('x')
    plt.subplot(312)
    plt.plot(t_range[-prediction.shape[0]:], prediction[:,1])
    plt.plot(t_range[-val_data.shape[0]:], val_data[:,1])
    plt.xlabel('Time (t)')
    plt.ylabel('y')
    plt.subplot(313)
    plt.plot(t_range[-prediction.shape[0]:], prediction[:,2])
    plt.plot(t_range[-val_data.shape[0]:], val_data[:,2])
    plt.xlabel('Time (t)')
    plt.ylabel('z')

    #3D Representation of Predicted (Blue) and Actual (Orange) Data
    plt.figure(2)
    ax = plt.axes(projection='3d')
    ax.plot3D(prediction[:,0], prediction[:,1], prediction[:,2])
    ax.plot3D(val_data[:, 0], val_data[:, 1], val_data[:, 2])
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    #Poincare Plots w/ Z_max
    plt.figure(3)
    local_z_max = prediction[argrelextrema(prediction[:,2], np.greater)[0],2]
    plt.scatter(local_z_max[:-1], local_z_max[1:])
    plt.xlabel('Z_{max,n}')
    plt.ylabel('Z_{max,n+1}')
    plt.show()

    eng.quit()
