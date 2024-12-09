import numpy as np
#from scipy.io import loadmat  
from copy import copy, deepcopy  
import torch 
import torch.nn as nn    


# def filter_function(input_value=None, previous_input_value=None, a=None, b=None):        
#     filter_value = input_value    
#     self.x_L[1:3] = self.x_L[0:2]    
#     self.x_L[0] = self.qHr_L   
#     self.y_L[1:3] = self.y_L[0:2]    
#     self.y_L[0] = np.sum(np.dot(self.x_L, self.b)) - np.sum(np.dot(self.y_L[2:0:-1], self.a[2:0:-1]))
#     self.qHr_L = self.y_L[0] * 0.1   
#     return filter_value   


class Network(nn.Module):
    def __init__(self, n_input=18, n_first=128, n_second=64, n_third=2):
        super(Network,self).__init__()    
        self.fc1 = nn.Linear(n_input, n_first)     
        self.fc2 = nn.Linear(n_first, n_second)   
        self.fc3 = nn.Linear(n_second, n_third)    
        
        self.fc1_weight = np.load("f1_weight.npy")   
        self.fc2_weight = np.load("f2_weight.npy")   
        self.fc3_weight = np.load("f3_weight.npy")   
        
        self.fc1_bias   = np.load("f1_bias.npy")    
        self.fc2_bias   = np.load("f2_bias.npy")    
        self.fc3_bias   = np.load("f3_bias.npy")    
        
        self.fc1.weight.data = torch.tensor(self.fc1_weight, dtype=torch.float32)
        self.fc1.bias.data   = torch.tensor(self.fc1_bias, dtype=torch.float32)

        self.fc2.weight.data = torch.tensor(self.fc2_weight, dtype=torch.float32)
        self.fc1.bias.data   = torch.tensor(self.fc2_bias, dtype=torch.float32)

        self.fc3.weight.data = torch.tensor(self.fc3_weight, dtype=torch.float32)  
        self.fc3.bias.data   = torch.tensor(self.fc3_bias, dtype=torch.float32)  
    
    def forward(self, x):  
        x = torch.relu(self.fc1(x))   
        x = torch.relu(self.fc2(x))   
        x = self.fc3(x)    
        return x    
    

class DNN:  
    def __init__(self, n_input, n_first, n_second, n_third) -> None:
        self.n_input  = n_input
        self.n_first  = n_first
        self.n_second = n_second
        self.n_third  = n_third
        
        self.input_dim  = 4  
        self.output_dim = 2  
        
        self.b = np.array([0.0730, 0, -0.0730])   
        self.a = np.array([1.0000, -1.8486, 0.8541])    
        
        #self.b = np.array([0.0336,    0.0671,    0.0336])
        #self.a = np.array([1.0000,   -1.4190,    0.5533])
        
        # for filter function 
        self.x_L = np.zeros(3)
        self.y_L = np.zeros(3)
        self.x_R = np.zeros(3)
        self.y_R = np.zeros(3)   
        
        self.in_2       = np.ones(self.input_dim)    
        self.in_1       = np.ones(self.input_dim)    
        self.in_0       = np.ones(self.input_dim)   
           
        self.out_3      = np.ones(self.output_dim)     
        self.out_2      = np.ones(self.output_dim)    
        self.out_1      = np.ones(self.output_dim)    
        
        self.input_data = np.zeros(self.n_input)   
        
        self.q_T_L = 0
        self.q_T_R = 0
        self.dq_T_L = 0
        self.dq_T_R = 0
        
        self.qHr_L = 0
        self.qHr_R = 0
        
        self.kd_L = 14.142
        self.kp_L = 50.0
        self.kp_R = 50.0
        self.kd_R = 14.142  
        
        self.LTx   = 0  
        self.RTx   = 0
        self.LTAVx = 0
        self.RTAVx = 0  
        
        self.dqTd_history_L = np.zeros(3)
        self.dqTd_filtered_history_L = np.zeros(3)
        self.dqTd_filtered_L = 0
        
        self.dqTd_history_R = np.zeros(3)
        self.dqTd_filtered_history_R = np.zeros(3)
        self.dqTd_filtered_R = 0
        
        self.hip_torque_L = 0
        self.hip_torque_R = 0  
        
        self.network = Network(n_input=n_input, n_first=n_first, n_second=n_second, n_third=n_third)   
        
    def generate_assistance(self, LTx, RTx, LTAVx, RTAVx, kp, kd):
        self.LTx = LTx
        self.RTx = RTx
        self.LTAVx = LTAVx
        self.RTAVx = RTAVx  

        self.kp_L = kp
        self.kp_R = kp
        self.kd_L = kd  
        self.kd_R = kd  
 
        self.q_T_L  = self.LTx * np.pi / 180.0
        self.q_T_R  = self.RTx * np.pi / 180.0 
        self.dq_T_L = self.LTAVx * np.pi / 180.0  
        self.dq_T_R = self.RTAVx * np.pi / 180.0    
        
        self.in_0   = np.array([self.q_T_L, self.q_T_R, self.dq_T_L, self.dq_T_R])   

        self.input_data = np.concatenate((self.in_2, self.in_1, self.in_0, self.out_3, self.out_2, self.out_1), axis=None)  
        
        self.in_2  = np.copy(self.in_1)  
        self.in_1  = np.copy(self.in_0)  
        self.out_3 = np.copy(self.out_2)    
        self.out_2 = np.copy(self.out_1)    
        
        input_data_tensor = torch.tensor(self.input_data, dtype=torch.float32)  
        output_tensor = self.network(input_data_tensor)   
        output_data = output_tensor.detach().numpy()  
        self.qHr_L, self.qHr_R = output_data  
        
        self.out_1 = np.array([self.qHr_L, self.qHr_R])  

        # filter process 
        self.x_L[1:3] = self.x_L[0:2]
        self.x_L[0] = self.qHr_L  
        self.y_L[1:3] = self.y_L[0:2]  
        self.y_L[0] = np.sum(np.dot(self.x_L, self.b)) - np.sum(np.dot(self.y_L[2:0:-1], self.a[2:0:-1]))  
        self.qHr_L = self.y_L[0] * 0.1  

        self.x_R[1:3] = self.x_R[0:2]  
        self.x_R[0] = self.qHr_R
        self.y_R[1:3] = self.y_R[0:2]  
        self.y_R[0] = np.sum(np.dot(self.x_R, self.b)) - np.sum(np.dot(self.y_R[2:0:-1], self.a[2:0:-1]))
        self.qHr_R = self.y_R[0] * 0.1   

        # filter dqTd_L
        self.dqTd_history_L[1:3] = self.dqTd_history_L[0:2]
        self.dqTd_history_L[0] = self.LTAVx  
        self.dqTd_filtered_history_L[1:3] = self.dqTd_filtered_history_L[0:2]  
        self.dqTd_filtered_history_L[0] = np.sum(np.dot(self.dqTd_history_L, self.b)) - np.sum(np.dot(self.dqTd_filtered_history_L[2:0:-1], self.a[2:0:-1]))
        self.dqTd_filtered_L = self.dqTd_filtered_history_L[0]
        
        # filter dqTd_R  
        self.dqTd_history_R[1:3] = self.dqTd_history_R[0:2]  
        self.dqTd_history_R[0] = self.RTAVx 
        self.dqTd_filtered_history_R[1:3] = self.dqTd_filtered_history_R[0:2]  
        self.dqTd_filtered_history_R[0] = np.sum(np.dot(self.dqTd_history_R, self.b)) - np.sum(np.dot(self.dqTd_filtered_history_R[2:0:-1], self.a[2:0:-1]))
        self.dqTd_filtered_R = self.dqTd_filtered_history_R[0]   
        
        # interactive controller  
        # self.hip_torque_L = (self.qHr_L * self.kp_L + self.dqTd_filtered_L * self.kd_L * (-1.0)) * 0.008  
        # self.hip_torque_R = (self.qHr_R * self.kp_R + self.dqTd_filtered_R * self.kd_R * (-1.0)) * 0.008  

        self.hip_torque_L = ((self.qHr_L - self.q_T_L) * self.kp_L + self.dq_T_L * self.kd_L * (-1.0)) * 0.008
        self.hip_torque_R = ((self.qHr_R - self.q_T_R) * self.kp_R + self.dq_T_R * self.kd_R * (-1.0)) * 0.008    

def main():  
    dnn = DNN(18, 128, 64, 2)  
    
    np.save("f1_weight.npy", dnn.fc1_weight)   
    np.save("f2_weight.npy", dnn.fc2_weight)    
    np.save("f3_weight.npy", dnn.fc3_weight)      
    
    np.save("f1_bias.npy", dnn.fc1_bias)   
    np.save("f2_bias.npy", dnn.fc2_bias)    
    np.save("f3_bias.npy", dnn.fc3_bias)  


if __name__ == "__main__":
    from scipy.signal import butter, filtfilt  
    
    # 设计一个低通滤波器
    # order = 2              # 滤波器的阶数
    # sample_freq = 45 
    # cutoff_freq = 4.0      # 截止频率（相对于采样率的归一化值）
    # input_factor = cutoff_freq/sample_freq  
    # b, a = butter(order, input_factor, btype='low')    
    
    # print("b", b) 
    # print("a", a)   
    
    # main()  
    
    f1_weight = np.load("f1_weight.npy")     
    f2_weight = np.load("f2_weight.npy")     
    f3_weight = np.load("f3_weight.npy")     
    
    f1_bias = np.load("f1_bias.npy")    
    f2_bias = np.load("f2_bias.npy")   
    f3_bias = np.load("f3_bias.npy")      
    
    print(f1_weight.shape, f2_weight.shape, f3_weight.shape, f1_bias.shape, f2_bias.shape, f3_bias.shape)    