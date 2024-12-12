import numpy as np
from math import pi, tan, sin, cos
import matplotlib.pyplot as plt  
import csv  
import torch
import torch.nn as nn   
from DNN_torch import DNN  
from utils import *  
import scipy.signal as signal  


def butterworth_filter(data, cutoff, fs, order=4, filter_type='low'):
    """
    实现 Butterworth 滤波器
    :param data: 输入信号
    :param cutoff: 截止频率（Hz）
    :param fs: 采样率（Hz）
    :param order: 滤波器阶数
    :param filter_type: 滤波器类型，'low' 为低通，'high' 为高通
    :return: 滤波后的信号
    """
    # 归一化截止频率
    nyquist = 0.5 * fs
    normalized_cutoff = cutoff / nyquist

    # 预包络变换 (Pre-warping)
    omega = tan(pi * normalized_cutoff)

    # 计算滤波器系数 (Butterworth 特性)
    poles = []
    for k in range(order):
        angle = (2 * k + 1) * pi / (2 * order)
        poles.append(complex(-omega * sin(angle), omega * cos(angle)))

    # 系数计算 (差分方程)
    a = [1.0]
    for p in poles:
        a = np.convolve(a, [1.0, -2.0 * p.real, abs(p)**2])

    b = [omega**order]  # 滤波器增益

    # 数字化滤波器（双线性变换）
    if filter_type == 'high':
        a = a[::-1]
        b = [-b[0] if i % 2 else b[0] for i in range(len(a))]

    a = np.real(a).tolist()
    b = np.real(b).tolist()

    # 滤波计算（IIR Filter）
    y = np.zeros_like(data)
    x = np.array(data)

    for i in range(len(data)):
        y[i] = sum(b[j] * x[i - j] for j in range(len(b)) if i - j >= 0) \
             - sum(a[j] * y[i - j] for j in range(1, len(a)) if i - j >= 0)
    return y


def butterworth_filter(data, cutoff, fs, order=4, filter_type='low'):
    """
        Implement Butterworth filter using scipy
        :param data: Input signal
        :param cutoff: Cutoff frequency in Hz
        :param fs: Sampling frequency in Hz
        :param order: Order of the filter
        :param filter_type: 'low' for low-pass, 'high' for high-pass
        :return: Filtered signal
    """
    # Normalized cutoff frequency
    nyquist = 0.5 * fs
    normalized_cutoff = cutoff / nyquist

    # Design the Butterworth filter using scipy.signal.butter
    b, a = signal.butter(order, normalized_cutoff, btype=filter_type, analog=False)

    # Apply the filter to the data using scipy.signal.filtfilt (zero-phase filtering)
    filtered_signal = signal.filtfilt(b, a, data)
    return filtered_signal



# 测试代码
if __name__ == "__main__":  

    # # 模拟输入信号
    # fs = 1000  # 采样率
    # t = np.linspace(0, 1, fs, endpoint=False)  # 时间
    # input_signal = np.sin(2 * pi * 5 * t) + 0.5 * np.sin(2 * pi * 50 * t)  # 5 Hz + 50 Hz

    # # Butterworth 滤波器应用
    # cutoff = 50 # 截止频率 (Hz)
    # filtered_signal = butterworth_filter(input_signal, cutoff, fs, order=4, filter_type='low')

    # # 绘图
    # plt.figure(figsize=(10, 6))
    # plt.plot(t, input_signal, label='Original Signal')
    # plt.plot(t, filtered_signal, label='Filtered Signal', color='red')
    # plt.xlabel('Time (s)')
    # plt.ylabel('Amplitude')
    # plt.legend()
    # plt.title('Butterworth Low-pass Filter')
    # plt.grid()
    # plt.show()  
    
    ##### hip ann #####
    # network setup  
    hip_dnn = DNN(18, 128, 64, 2)        
    # network setup   

    # Load CSV data manually
    # file_path = 'RL_Controller_torch/20240929-130752.csv'  # Replace with your actual CSV file path
    # file_path = 'IMU_reading/imu_Serial_Python/imu_data.csv' 
    # file_path = 'RL_Controller_torch/imu_data.csv' 
    # file_path = 'data/imu_comparison.csv'  
    file_path = '../data/Leo/Without_IMU/Zhimin/20241211-120727-Zhimin-Walk-S0x75-Trail02.csv'  


    # Apply Butterworth Low-pass filter with cutoff frequency of 10 Hz
    cutoff = 5  # Cutoff frequency (Hz)
    
    # Initialize lists for each column
    time_list = []
    L_torque = []
    R_torque = []
    L_cmd = []
    R_cmd = []
    R_IMU_angle = []
    L_IMU_angle = []
    L_IMU_Vel = []  
    R_IMU_Vel = []  
    L_Ref_Ang = []   
    R_Ref_Ang = []   
    L_Ref_Vel = []   
    R_Ref_Vel = []   
    
    L_IMU_Vel_filter = []  
    R_IMU_Vel_filter = []  
    
    Peak  = []   

    # Read the CSV file
    with open(file_path, 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        
        # Skip header row
        next(csvreader)
        
        # Extract data row by row
        for row in csvreader:
            print("length :", len(row))
            time_list.append(float(row[11]))    
            L_IMU_angle.append(float(row[0])) 
            R_IMU_angle.append(float(row[1]))   
            L_IMU_Vel.append(float(row[2])) 
            R_IMU_Vel.append(float(row[3])) 
            # L_cmd.append(-1 * float(row[4]))       
            # R_cmd.append(float(row[5]))     
            # L_Ref_Ang.append(float(row[6]))
            # R_Ref_Ang.append(float(row[7]))
            # L_Ref_Vel.append(float(row[8]))
            # R_Ref_Vel.append(float(row[9]))    
       
    for i in range(len(L_IMU_angle)):     
        print(f"Time when running NN = {time_list[i]:^8.3f}")     
        hip_dnn.generate_assistance(L_IMU_angle[i], R_IMU_angle[i], L_IMU_Vel[i], R_IMU_Vel[i], kp, kd)   
        
        L_Ref_Ang.append(float(hip_dnn.qHr_L)) 
        L_Ref_Vel.append(float(hip_dnn.qHr_L_ori))  
        
        R_Ref_Ang.append(float(hip_dnn.qHr_R))     
        R_Ref_Vel.append(float(hip_dnn.qHr_R_ori))   
        
        # L_IMU_Vel_filter.append(butterworth_filter([L_IMU_Vel[i]], cutoff, 100, order=4, filter_type='low'))   
        # R_IMU_Vel_filter.append(butterworth_filter([R_IMU_Vel[i]], cutoff, 100, order=4, filter_type='low'))   
        
        # L_IMU_Vel_filter = butterworth_filter(L_IMU_Vel, cutoff, 100, order=4, filter_type='low')  
        # R_IMU_Vel_filter = butterworth_filter(R_IMU_Vel, cutoff, 100, order=4, filter_type='low')   

    
    fig, axs = plt.subplots(3, 2, figsize=(10, 8))

    axs[1, 0].plot(time_list, L_IMU_angle, label='Left Position Actual', color='red')  
    axs[1, 0].plot(time_list, L_Ref_Ang, label='Left Position Reference', color='blue') 
    axs[1, 0].plot(time_list, L_Ref_Vel, label='Left Position Reference', color='green')   
    # axs[1, 0].set_title('Angular Position (IMU)')
    axs[1, 0].legend(loc='upper center', bbox_to_anchor=(0.5, 1.20), ncol=2) 
    axs[1, 0].set_xlabel('Time (s)')  
    axs[1, 0].set_ylabel('IMU Angular Position (Deg)')   

    axs[1, 1].plot(time_list, R_IMU_angle, label='Right Position Reference', color='red')
    axs[1, 1].plot(time_list, R_Ref_Ang, label='Right Position Reference', color='blue')  
    axs[1, 1].plot(time_list, R_Ref_Vel, label='Right Position Reference', color='green')
    # axs[1, 1].set_title('Angular Position (encoder)') 
    # axs[1, 1].set_title('Angular Position (IMU)')
    axs[1, 1].legend(loc='upper center', bbox_to_anchor=(0.5, 1.20), ncol=2) 
    axs[1, 1].set_xlabel('Time (s)')   
    axs[1, 1].set_ylabel('Encoder Angular Position (Deg)')       

    axs[2, 0].plot(time_list, L_IMU_Vel, label='Left Velocity Actual', color='red')
    axs[2, 0].plot(time_list, L_IMU_Vel_filter, label='Left Velocity Reference', color='blue')  
    # axs[2, 0].set_title('Angular Velocity (IMU)') 
    axs[2, 0].legend(loc='upper center', bbox_to_anchor=(0.5, 1.20), ncol=2) 
    axs[2, 0].set_xlabel('Time (s)')  
    axs[2, 0].set_ylabel('IMU Angular Velocity (Deg/s)')      

    axs[2, 1].plot(time_list, R_IMU_Vel, label='Left Velocity Reference', color='red')
    axs[2, 1].plot(time_list, R_IMU_Vel_filter, label='Right Velocity Reference', color='blue')
    # axs[2, 1].set_title('Angular Velocity (encoder)')
    # axs[2, 1].legend() 
    axs[2, 1].legend(loc='upper center', bbox_to_anchor=(0.5, 1.20), ncol=2) 
    axs[2, 1].set_xlabel('Time (s)')  
    axs[2, 1].set_ylabel('Encoder Angular Velocity (Deg/s)')   

    # # Formatting the plots
    # for ax in axs.flat:
    #     ax.set(xlabel='Time (s)', ylabel='Value')
    #     ax.grid(True)

    # Adjust layout for better spacing
    plt.tight_layout()  

    # plt.savefig('figures/test_per_leo.png')

    # Display the plot
    plt.show()