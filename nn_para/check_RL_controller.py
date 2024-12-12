import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

class Network(nn.Module):
    def __init__(self,saved_policy_path) -> None:
        super(Network,self).__init__()
        self.fc1 = nn.Linear(18,128)
        self.fc2 = nn.Linear(128,64)
        self.fc3 = nn.Linear(64,2)
        
        # fc1_weight = []

        # fc2_weight = []

        # fc3_weight = []

        # fc1_bias = []
        
        # fc2_bias = []
        
        # fc3_bias = []
        
        # fc1_weight_tensor = torch.tensor(fc1_weight, dtype=torch.float32)
        # fc1_bias_tensor = torch.tensor(fc1_bias, dtype=torch.float32)

        # fc2_weight_tensor = torch.tensor(fc2_weight, dtype=torch.float32)
        # fc2_bias_tensor = torch.tensor(fc2_bias, dtype=torch.float32)

        # fc3_weight_tensor = torch.tensor(fc3_weight, dtype=torch.float32)
        # fc3_bias_tensor = torch.tensor(fc3_bias, dtype=torch.float32)

        # # Load the converted tensors into the model
        # self.fc1.weight.data = fc1_weight_tensor
        # self.fc1.bias.data = fc1_bias_tensor

        # self.fc2.weight.data = fc2_weight_tensor
        # self.fc2.bias.data = fc2_bias_tensor

        # self.fc3.weight.data = fc3_weight_tensor
        # self.fc3.bias.data = fc3_bias_tensor
    
    def forward(self,x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
    def load_saved_policy(self,state_dict):
        self.fc1.weight.data = state_dict['p_fc1.weight']
        self.fc1.bias.data = state_dict['p_fc1.bias']
        self.fc2.weight.data = state_dict['p_fc2.weight']
        self.fc2.bias.data = state_dict['p_fc2.bias']
        self.fc3.weight.data = state_dict['p_fc3.weight']
        self.fc3.bias.data = state_dict['p_fc3.bias']


class DNN:
    def __init__(self, n_input, n_first, n_second, n_third, saved_policy_path) -> None:
        self.n_input = n_input
        self.n_first = n_first
        self.n_second = n_second
        self.n_third = n_third
        self.b = np.array([0.0730, 0, -0.0730])
        self.a = np.array([1.0000, -1.8486, 0.8541])
        #self.b = np.array([0.0336,    0.0671,    0.0336])
        #self.a = np.array([1.0000,   -1.4190,    0.5533])
        self.x_L = np.zeros(3)
        self.y_L = np.zeros(3)
        self.x_R = np.zeros(3)
        self.y_R = np.zeros(3)

        self.in_2 = np.ones(4)
        self.in_1 = np.ones(4)
        self.out_3 = np.ones(2)
        self.out_2 = np.ones(2)
        self.out_1 = np.ones(2)
        self.input_data = np.zeros(self.n_input)
        self.qTd_L = 10
        self.qTd_R = 10
        self.dqTd_L = 0
        self.dqTd_R = 0
        self.out_first = np.zeros(self.n_first)
        self.out_second = np.zeros(self.n_second)
        self.out_third = np.zeros(self.n_third)
        self.qHr_L = 0
        self.qHr_R = 0
        self.kd2 = 14.142
        self.kp2 = 50.0
        self.kp3 = 50.0
        self.kd3 = 14.142
        self.dqTd_history_L = np.zeros(3)
        self.dqTd_filtered_history_L = np.zeros(3)
        self.dqTd_filtered_L = 0
        self.dqTd_history_R = np.zeros(3)
        self.dqTd_filtered_history_R = np.zeros(3)
        self.dqTd_filtered_R = 0
        self.hip_torque_L = 0
        self.hip_torque_R = 0
        self.LTx = 0
        self.RTx = 0
        self.LTAVx = 0
        self.RTAVx = 0
        
        self.saved_policy_path = saved_policy_path
        self.network = Network(self.saved_policy_path)
        self.network.load_saved_policy(torch.load(self.saved_policy_path))
        print(f"Loaded policy from {self.saved_policy_path}")
        
    def generate_assistance(self, LTx, RTx, LTAVx, RTAVx, kp, kd):
        self.LTx = LTx
        self.RTx = RTx
        self.LTAVx = LTAVx
        self.RTAVx = RTAVx

        self.kp2 = kp
        self.kp3 = kp
        self.kd2 = kd
        self.kd3 = kd

        self.qTd_L = LTx * 3.1415926 / 180.0
        self.qTd_R = RTx * 3.1415926 / 180.0
        self.dqTd_L = LTAVx * 3.1415926 / 180.0
        self.dqTd_R = RTAVx * 3.1415926 / 180.0

        self.input_data = np.concatenate((self.in_2, self.in_1, self.qTd_L, self.qTd_R, self.dqTd_L, self.dqTd_R, self.out_3, self.out_2, self.out_1), axis=None)
        self.in_2 = np.copy(self.in_1)
        self.in_1 = np.array([self.qTd_L, self.qTd_R, self.dqTd_L, self.dqTd_R])
        self.out_3 = np.copy(self.out_2)
        self.out_2 = np.copy(self.out_1)

        self.out_first[:] = 0
        self.out_second[:] = 0
        self.out_third[:] = 0
        
        input_data_tensor = torch.tensor(self.input_data, dtype=torch.float32)
        output_tensor = self.network(input_data_tensor)
        output_data = output_tensor.detach().numpy()
        self.qHr_L, self.qHr_R = output_data
        self.out_1 = np.array([self.qHr_L, self.qHr_R])

        # for i in range(self.n_first):
            # self.out_first[i] = np.dot(self.input_data, self.fc1_weight[i,:]) + self.fc1_bias[i]
            # self.out_first[i] = np.maximum(0, self.out_first[i])
        
        # for i in range(self.n_second):
            # self.out_second[i] = np.dot(self.out_first, self.fc2_weight[i,:]) + self.fc2_bias[i]
            # self.out_second[i] = np.maximum(0, self.out_second[i])
        
        # for i in range(self.n_third):
            # self.out_third[i] = np.dot(self.out_second, self.fc3_weight[i,:]) + self.fc3_bias[i]
            # #self.out_third[i] = np.maximum(0, self.out_third[i])
        
        # self.qHr_L, self.qHr_R = self.out_third
        # self.out_1 = np.array([self.qHr_L, self.qHr_R])

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
        #
        self.hip_torque_L = (self.qHr_L * self.kp2 + self.dqTd_filtered_L * self.kd2 * (-1.0)) * 0.008
        self.hip_torque_R = (self.qHr_R * self.kp3 + self.dqTd_filtered_R * self.kd3 * (-1.0)) * 0.008

        #self.hip_torque_L = ((self.qHr_L-self.qTd_L) * self.kp2 + self.dqTd_L * self.kd2 * (-1.0)) * 0.008
        #self.hip_torque_R = ((self.qHr_R-self.qTd_R) * self.kp3 + self.dqTd_R * self.kd3 * (-1.0)) * 0.008
        
        #self.hip_torque_R = self.qHr_R-self.qTd_R
        # print(f"qHr_L={self.qHr_L}, qHr_R={self.qHr_R}, hip_torque_L={self.hip_torque_L}, hip_torque_R={self.hip_torque_R}")
        return self.hip_torque_L, self.hip_torque_R, self.qHr_L* self.kp2, self.dqTd_filtered_L* self.kd2, self.qHr_R* self.kp2, self.dqTd_filtered_R* self.kd2


def main():
    saved_policy_path = "C:/Users/jzhu35/Dropbox/BiomechatronicsLab/Projects/Reinforcement Learning Control/main_base/nn_08022024_exov13/max.pt"
    saved_kinematics_path = "C:/Users/jzhu35/Dropbox/BiomechatronicsLab/Personal Members Folders/Junxi/Code/Assistance timing issue/Controller_100Hz_Gain_0x2_20240830_2/20240830-172817-Junxi-Walk-S1x25-Trial01.csv"
    state_dict = torch.load(saved_policy_path, map_location='cpu')
    print(state_dict.keys())

    for key in state_dict.keys():
        print(f"key={key}, shape={state_dict[key].shape}")   
    
    dnn = DNN(18,128,64,2,saved_policy_path=saved_policy_path)  
    
    # Load the Excel file
    kinematics_data = np.genfromtxt(saved_kinematics_path, delimiter=',', skip_header=1)

    # Display the first few rows of the data
    print(kinematics_data[0:5,:])
    dataLength = kinematics_data.shape[0]
    hip_torque_L_sim_list = np.zeros((dataLength,1))
    hip_torque_R_sim_list = np.zeros((dataLength,1))
    qHr_L_sim_list = np.zeros((dataLength,1))
    qHr_R_sim_list = np.zeros((dataLength,1))
    dqTd_filtered_L_sim_list = np.zeros((dataLength,1))
    dqTd_filtered_R_sim_list = np.zeros((dataLength,1))
    hip_torque_L_rec_list = kinematics_data[:, 4] * 0.3 / 20
    hip_torque_R_rec_list = kinematics_data[:, 5] * 0.3 / 20
    for idx in range(dataLength):
        LTx = kinematics_data[idx, 0]
        RTx = kinematics_data[idx, 1]
        LTAVx = kinematics_data[idx, 2]
        RTAVx = kinematics_data[idx, 3]
        kp = 400
        kd = 0.1*np.sqrt(400)
        hip_torque_L, hip_torque_R, qHr_L, dqTd_filtered_L, qHr_R, dqTd_filtered_R = dnn.generate_assistance(LTx, RTx, LTAVx, RTAVx, kp, kd)
        hip_torque_L_sim_list[idx] = -hip_torque_L
        hip_torque_R_sim_list[idx] = hip_torque_R
        qHr_L_sim_list[idx] = qHr_L
        qHr_R_sim_list[idx] = qHr_R
        dqTd_filtered_L_sim_list[idx] = dqTd_filtered_L
        dqTd_filtered_R_sim_list[idx] = dqTd_filtered_R
    
    

    fig, axs = plt.subplots(3, 2, figsize=(15, 10), sharex=True)

    # First row: Recorded vs Computed Torque
    axs[0, 0].plot(hip_torque_L_rec_list, label='Recorded Left')
    axs[0, 0].plot(hip_torque_L_sim_list, label='Computed Left')
    axs[0, 0].set_title('Left Leg Torque')
    axs[0, 0].legend()

    axs[0, 1].plot(hip_torque_R_rec_list, label='Recorded Right')
    axs[0, 1].plot(hip_torque_R_sim_list, label='Computed Right')
    axs[0, 1].set_title('Right Leg Torque')
    axs[0, 1].legend()

    # Second row: Thigh Angle and Speed
    ax2_0 = axs[1, 0].twinx()
    axs[1, 0].plot(kinematics_data[:, 0], 'g-', label='Thigh Angle Left')
    ax2_0.plot(kinematics_data[:, 2], 'b-', label='Thigh Speed Left')
    axs[1, 0].set_title('Left Leg Thigh Angle and Speed')
    axs[1, 0].set_ylabel('Thigh Angle (deg)')
    ax2_0.set_ylabel('Thigh Speed (deg/s)')
    axs[1, 0].legend(loc='upper left')
    ax2_0.legend(loc='upper right')

    ax2_1 = axs[1, 1].twinx()
    axs[1, 1].plot(kinematics_data[:, 1], 'g-', label='Thigh Angle Right')
    ax2_1.plot(kinematics_data[:, 3], 'b-', label='Thigh Speed Right')
    axs[1, 1].set_title('Right Leg Thigh Angle and Speed')
    axs[1, 1].set_ylabel('Thigh Angle (deg)')
    ax2_1.set_ylabel('Thigh Speed (deg/s)')
    axs[1, 1].legend(loc='upper left')
    ax2_1.legend(loc='upper right')

    # Third row: 
    ax3_0 = axs[2, 0].twinx()
    axs[2, 0].plot(qHr_L_sim_list, 'g-', label='qHr')
    ax3_0.plot(dqTd_filtered_L_sim_list, 'b-', label='dqTd_filtered')
    axs[2, 0].set_title('')
    axs[2, 0].set_ylabel('qHr (deg)')
    ax3_0.set_ylabel('dqTd_filtered (deg/s)')
    axs[2, 0].legend(loc='upper left')
    ax3_0.legend(loc='upper right')

    ax3_1 = axs[2, 1].twinx()
    axs[2, 1].plot(qHr_R_sim_list, 'g-', label='qHr')
    ax3_1.plot(dqTd_filtered_R_sim_list, 'b-', label='dqTd_filtered')
    axs[2, 1].set_title('')
    axs[2, 1].set_ylabel('qHr (deg)')
    ax3_1.set_ylabel('dqTd_filtered (deg/s)')
    axs[2, 1].legend(loc='upper left')
    ax3_1.legend(loc='upper right')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
   main()