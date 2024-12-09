import numpy as np  
import time  
from math import *    
from DNN_torch import DNN      

L_Ctl        = 1   
R_Ctl        = 1   
Cmd_scale    = 20.0            
ctl_mode     = 0             # 1 for with IMU 0 for without IMU   
nn_mode      = 1  
kcontrol     = 0.25          # 1.5 para running. 2 para climbing.  
max_cmd      = 8.0         
torque_scale = 40.0   

L_Cmd = 0
R_Cmd = 0
pk    = 0  
kp    = 10  
kd    = 400    
duration = 60   

flag = 'Trail02' 
speed_name_list = ['Walk-S0x75', 'Walk-S1x25', 'Walk-S1x75', 'Run-S2x00']   

control_mode = 'Without_IMU'  
robot_name   = 'Leo'          # Hip13;; Hip15 
subject_name = 'Zhimin'      
speed_name   = speed_name_list[0]      # 0: 0.75, 1: 1.25, 2: 1.75, 3: 2.00     

# control_mode = 'Without_IMU'  
# Subject_name = 'Ivan'    
# speed_name   = '2.0'     # 0.75, 1.25, 1.75, 2.0     

# control_mode = 'Without_IMU' 
# Subject_name = 'quiun'    
# speed_name   = '2.0'     # 0.75, 1.25, 1.75, 2.0   

save_data    = 0    
plot_data    = 0  
start_ctl    = 1  

##### hip ann #####
# network setup  
hip_dnn = DNN(18, 128, 64, 2)        
# network setup   

date  = time.localtime(time.time())    
date_year      = date.tm_year     
date_month     = date.tm_mon    
date_day       = date.tm_mday  
date_hour      = date.tm_hour
date_minute    = date.tm_min
date_second    = date.tm_sec   

date_string = f"{date_year:04}{date_month:02}{date_day:02}-{date_hour:02}{date_minute:02}{date_second:02}"
root_dir  = '../data/' + robot_name + '/' + control_mode + '/' + subject_name + '/' 
file_name = root_dir + date_string + '-' + subject_name + '-' + speed_name + '-' + flag       

def rad2deg(rad):
    deg = rad*180.0/pi
    return deg

def deg2rad(deg):
    rad = deg*pi/180.0
    return rad

def saturate(Cmd,sat):
    if Cmd>sat:  
        Cmd=sat
    if Cmd<-sat: 
        Cmd=-sat
    return Cmd   

def impedance_control(
    Kp=0.0, 
    Kd=0.0, 
    ref_pos=0.0, ref_vel=0.0, 
    real_pos=0.0, real_vel=0.0,  
    tau_ff=0.0  
):  
    Cmd_tau = 0.0  
    Cmd_tau = (ref_pos - real_pos) * Kp + (ref_vel - real_vel) * Kd + tau_ff  

    return Cmd_tau      

def smooth(old_value=None, value=None):  
    smoothed_value = 0.7 * old_value + 0.3 * value   
    return smoothed_value   

def write_csv(file_name,V1,V2,V3,V4,V5,V6,V7,V8,V9,V10,V11,V12,V13):  
    with open(file_name + ".csv", "a") as log:  
        log.write("{0},{1},{2},{3},{4},{5},{6},{7},{8},{9},{10},{11},{12}\n".format(str(V1),str(V2),str(V3),str(V4),str(V5),str(V6),str(V7),str(V8),str(V9),str(V10),str(V11),str(V12),str(V13))) 

# def write_csv(V1,V2,V3,V4,V5,V6,V7,V8,V9,V10,V11,V12,V13):  
#     with open(root_dir + str(dateh)+str(datem)+str(dates)+".csv", "a") as log:
#         log.write("{0},{1},{2},{3},{4},{5},{6},{7},{8},{9},{10},{11},{12}\n".format(str(V1),str(V2),str(V3),str(V4),str(V5),str(V6),str(V7),str(V8),str(V9),str(V10),str(V11),str(V12),str(V13)))  


# date  = time.localtime(time.time())  
# date_year  = date.tm_year 
# date_month = date.tm_mon  
# date_day   = date.tm_mday  
# dateh      = date.tm_hour  
# datem      = date.tm_min  
# dates      = date.tm_sec    
# root_dir = '../data/Leo/' + control_mode + '/' + Subject_name + '-' + speed_name + '-'   
# def write_csv(V1,V2,V3,V4,V5,V6,V7,V8,V9,V10,V11,V12,V13):  
#     with open(root_dir + str(dateh)+str(datem)+str(dates)+".csv", "a") as log:
#         log.write("{0},{1},{2},{3},{4},{5},{6},{7},{8},{9},{10},{11},{12}\n".format(str(V1),str(V2),str(V3),str(V4),str(V5),str(V6),str(V7),str(V8),str(V9),str(V10),str(V11),str(V12),str(V13)))