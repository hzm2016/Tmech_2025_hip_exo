import ReadIMU as ReadIMU 
import time  
from DNN_torch import DNN  
import datetime  
import numpy as np   
import csv                   
from utils import *        


ComPort = '/dev/serial0'     
imu = ReadIMU.READIMU(ComPort)      


# date = time.localtime(time.time())  
# date_year = date.tm_year  
# date_month = date.tm_mon  
# date_day = date.tm_mday  
# date_hour = date.tm_hour  
# date_minute = date.tm_min  
# date_second = date.tm_sec   

# root_path = "../data/Ours/With_IMU/Lily/s0x75"  # "s1x25, s1x75, sx20"
# # Create filename with format {Year}{Month}{Day}-{Hour}{Minute}{Second}.csv
# csv_filename = root_path + f"{date_year:04}{date_month:02}{date_day:02}-{date_hour:02}{date_minute:02}{date_second:02}.csv"


if save_data:   
    with open(file_name + ".csv", "a") as log:  
        log.write("{0},{1},{2},{3},{4},{5},{6},{7},{8},{9},{10},{11},{12}\n".format(
            "t", 
            "L_IMU_angle", "R_IMU_angle", 
            "L_IMU_vel", "R_IMU_vel", 
            "L_encoder", "R_encoder", 
            "L_encoder_vel", "R_encoder_vel",  
            "L_Cmd", "R_Cmd",   
            "L_tau", "R_tau", 
            "L_ref_pos", "R_ref_pos",   
            "L_ref_vel", "R_ref_vel"   
        ))  
        
with open(file_name, 'a', newline='') as csvfile:   
    variablenames = ['L_IMU_Ang', 'R_IMU_Ang', 
                     'L_IMU_Vel', 'R_IMU_Vel', 
                     'L_Cmd',     'R_Cmd', 
                     'L_Ref_Ang', 'R_Ref_Ang', 
                     'L_Ref_Vel', 'R_Ref_Vel',   
                     'Peak', 'Time']
    writer = csv.DictWriter(csvfile, fieldnames=variablenames)   

    csvfile.seek(0, 2)   
    if csvfile.tell() == 0:    
        writer.writeheader()    

    start = time.time()  
    while True:  
        now = (time.time() - start)   
        
        imu.read()    
        imu.decode()     
        
        L_IMU_angle = imu.XIMUL 
        R_IMU_angle = imu.XIMUR   
        L_IMU_vel = imu.XVIMUL   
        R_IMU_vel = imu.XVIMUR    
        
        #### filtering the input position and velocity ####    
        
        #### filtering the input position and velocity ####   
        
        print(f"Time when running NN = {now:^8.3f}")  
        hip_dnn.generate_assistance(L_IMU_angle, R_IMU_angle, L_IMU_vel, R_IMU_vel, kp, kd)  

        # calculate assistive torque in different ways    
        L_Cmd = L_Ctl * hip_dnn.hip_torque_L * kcontrol * 1.0/Cmd_scale      
        R_Cmd = R_Ctl * hip_dnn.hip_torque_R * kcontrol * 1.0/Cmd_scale         
        
        # L_Cmd =  cal_assistive_force(kp=kp, kd=kd, ref_pos=hip_dnn.qHr_L, real_vel=hip_dnn.dqTd_filtered_L)    
        # R_Cmd =  cal_assistive_force(kp=kp, kd=kd, ref_pos=hip_dnn.qHr_R, real_vel=hip_dnn.dqTd_filtered_R)    

        if (L_Cmd > pk or R_Cmd > pk):  
            if (R_Cmd > L_Cmd): 
                pk = R_Cmd
            if (L_Cmd > R_Cmd):  
                pk = L_Cmd    
        
        if (ref_type == 1):  
            ## second reference motion to Teensy   
            R_P_L_int16 = int(imu.ToUint(hip_dnn.qHr_L, -1 * position_scale, position_scale, 16))       
            R_P_R_int16 = int(imu.ToUint(hip_dnn.qHr_R, -1 * position_scale, position_scale, 16))         
            R_V_L_int16 = int(imu.ToUint(hip_dnn.dqTd_filtered_L, -1 * velocity_scale, velocity_scale, 16))       
            R_V_R_int16 = int(imu.ToUint(hip_dnn.dqTd_filtered_R, -1 * velocity_scale, velocity_scale, 16))       
            
            b1 = (R_P_L_int16 >> 8 & 0x00ff)
            b2 = (R_P_L_int16 & 0x00FF)  
            b3 = (R_P_R_int16 >> 8 & 0x00ff)  
            b4 = (R_P_R_int16 & 0x00FF)   
            b5 = (R_V_L_int16 >> 8 & 0x00ff)
            b6 = (R_V_L_int16 & 0x00FF)  
            b7 = (R_V_R_int16 >> 8 & 0x00ff)  
            b8 = (R_V_R_int16 & 0x00FF)     
            
            imu.send_reference(b1, b2, b3, b4,b5, b6, b7, b8)   
        else: 
            ## send torque command back    
            B1_int16 = int(imu.ToUint(L_Cmd, -1 * torque_scale, torque_scale, 16))     
            B2_int16 = int(imu.ToUint(R_Cmd, -1 * torque_scale, torque_scale, 16))        
            b1 = (B1_int16 >> 8 & 0x00ff)  
            b2 = (B1_int16 & 0x00FF)  
            b3 = (B2_int16 >> 8 & 0x00ff)  
            b4 = (B2_int16 & 0x00FF)   
            
            imu.send(b1, b2, b3, b4)   
        
        data = {
            'L_IMU_Ang': L_IMU_angle,
            'R_IMU_Ang': R_IMU_angle,
            'L_IMU_Vel': L_IMU_vel,
            'R_IMU_Vel': R_IMU_vel, 
            'L_Cmd'    : L_Cmd,  
            'R_Cmd'    : R_Cmd,  
            'L_Ref_Ang': hip_dnn.qHr_L,    
            'R_Ref_Ang': hip_dnn.qHr_R,    
            'L_Ref_Vel': hip_dnn.dqTd_filtered_L,   
            'R_Ref_Vel': hip_dnn.dqTd_filtered_R,      
            'Peak'     : pk,  
            'Time'     : now
        }   
        writer.writerow(data) 
        csvfile.flush()    
        print(f"| now: {now:^8.3f} | L_IMU_Ang: {L_IMU_angle:^8.3f} | R_IMU_Ang: {R_IMU_angle:^8.3f} | L_IMU_Vel: {L_IMU_vel:^8.3f} | R_IMU_Vel: {R_IMU_vel:^8.3f} | L_Cmd: {L_Cmd:^8.3f} | R_Cmd: {R_Cmd:^8.3f} | Peak: {pk:^8.3f} |")