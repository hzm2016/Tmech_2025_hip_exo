import csv
import matplotlib.pyplot as plt
from math import *


# Load CSV data manually
# file_path = 'RL_Controller_torch/20240929-130752.csv'  # Replace with your actual CSV file path
# file_path = 'IMU_reading/imu_Serial_Python/imu_data.csv' 
# file_path = 'RL_Controller_torch/imu_data.csv' 
# file_path = 'data/imu_comparison.csv'  
file_path = 'data/Leo/Without_IMU/Zhimin/20241211-120727-Zhimin-Walk-S0x75-Trail02.csv'  

# Initialize lists for each column
time = []
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
Peak  = []

# Read the CSV file
with open(file_path, 'r') as csvfile:
    csvreader = csv.reader(csvfile)
    
    # Skip header row
    next(csvreader)
    
    # Extract data row by row
    for row in csvreader:
        print("length :", len(row))
        time.append(float(row[11]))    
        L_IMU_angle.append(float(row[0])) 
        R_IMU_angle.append(float(row[1]))   
        L_IMU_Vel.append(float(row[2])) 
        R_IMU_Vel.append(float(row[3])) 
        L_cmd.append(-1 * float(row[4]))     
        R_cmd.append(float(row[5]))
        L_Ref_Ang.append(float(row[6]))
        R_Ref_Ang.append(float(row[7]))
        L_Ref_Vel.append(float(row[8]))
        R_Ref_Vel.append(float(row[9]))    

# Create subplots
fig, axs = plt.subplots(3, 2, figsize=(10, 8))

#Peak = 0.35*6.4*9.81*sin(R_Ref_Ang)

# # Plotting the data
# axs[0, 0].plot(time, L_cmd, label='Left Motor Command', color='red') 
# axs[0, 0].plot(time, R_cmd, label='Right Motor Command', color='blue') 
# # axs[0, 0].set_title('Left Limb')    
# axs[0, 0].legend(loc='upper center', bbox_to_anchor=(0.5, 1.20), ncol=2) 
# axs[0, 0].set_xlabel('Time (s)')  
# axs[0, 0].set_ylabel('Torque (Nm)')  

# # axs[0, 1].plot(time, R_cm, label='Right Motor Command', color='red')
# axs[0, 1].plot(time, L_cmd, label='Left Torque', color='red')
# axs[0, 1].plot(time, R_cmd, label='Right Torque', color='blue')
# # axs[0, 1].set_title('Right Limb')     
# axs[0, 1].legend(loc='upper center', bbox_to_anchor=(0.5, 1.20), ncol=2) 
# axs[0, 1].set_xlabel('Time (s)')  
# axs[0, 1].set_ylabel('Torque (Nm)')  

axs[1, 0].plot(time, L_IMU_angle, label='Left Position Actual', color='red')  
axs[1, 0].plot(time, L_Ref_Ang, label='Left Position Reference', color='blue')
# axs[1, 0].set_title('Angular Position (IMU)')
axs[1, 0].legend(loc='upper center', bbox_to_anchor=(0.5, 1.20), ncol=2) 
axs[1, 0].set_xlabel('Time (s)')  
axs[1, 0].set_ylabel('IMU Angular Position (Deg)')   

axs[1, 1].plot(time, R_IMU_angle, label='Right Position Reference', color='red')
axs[1, 1].plot(time, R_Ref_Ang, label='Right Position Reference', color='blue')
# axs[1, 1].set_title('Angular Position (encoder)') 
# axs[1, 1].set_title('Angular Position (IMU)')
axs[1, 1].legend(loc='upper center', bbox_to_anchor=(0.5, 1.20), ncol=2) 
axs[1, 1].set_xlabel('Time (s)')   
axs[1, 1].set_ylabel('Encoder Angular Position (Deg)')       

axs[2, 0].plot(time, L_IMU_Vel, label='Left Velocity Actual', color='red')
axs[2, 0].plot(time, L_Ref_Vel, label='Left Velocity Reference', color='blue')  
# axs[2, 0].set_title('Angular Velocity (IMU)') 
axs[2, 0].legend(loc='upper center', bbox_to_anchor=(0.5, 1.20), ncol=2) 
axs[2, 0].set_xlabel('Time (s)')  
axs[2, 0].set_ylabel('IMU Angular Velocity (Deg/s)')      

axs[2, 1].plot(time, R_IMU_Vel, label='Left Velocity Reference', color='red')
axs[2, 1].plot(time, R_Ref_Vel, label='Right Velocity Reference', color='blue')
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

plt.savefig('figures/test_per_leo.png')

# Display the plot
plt.show()