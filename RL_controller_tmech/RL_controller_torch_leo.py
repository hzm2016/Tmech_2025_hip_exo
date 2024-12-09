import time   
from math import *   
import kqExoskeletonIO as kqio   
import ReadIMU as ReadIMU   
from utils import *   

def SendCmdTorque(cmd1, cmd2):  
    Ant.Cmd.Loop_L  = kqio.TOR_LOOP
    Ant.Cmd.Loop_R  = kqio.TOR_LOOP    
    Ant.Cmd.Value_L = cmd1    
    Ant.Cmd.Value_R = cmd2    
    
def SendCmdPos(cmd1, cmd2):  
    Ant.Cmd.Loop_L  = kqio.PLACE_LOOP
    Ant.Cmd.Loop_R  = kqio.PLACE_LOOP   
    Ant.Cmd.Value_L = cmd1    
    Ant.Cmd.Value_R = cmd2    

if save_data:   
    with open(file_name + ".csv", "a") as log:  
        log.write("{0},{1},{2},{3},{4},{5},{6},{7},{8},{9},{10},{11},{12}\n".format(
            "t", 
            "L_IMU_angle", "R_IMU_angle", 
            "L_IMU_vel", "R_IMU_vel", 
            "L_encoder", "R_encoder", 
            "L_encoder_vel", "R_encoder_vel", 
            "L_Cmd", "R_Cmd",   
            "L_tau", "R_tau"   
        ))  

if plot_data:   
    imu_buffer = np.memmap("imu_data.dat", dtype='float32', mode='r+', shape=(4,))    

print("Initializing the comunication with the Exoskeleton")   
GetSec           = kqio.GetSec    
Ant              = kqio.AntCH("/dev/ttyAMA0")                # This is the comport that connects the Raspberry Pi 4 to the LEO
# Ant              = kqio.AntCH("/dev/serial0")  
Ant.Cmd.CmdMode  = kqio.CMD_SERVO_OVERRIDE  
StartSec         = GetSec()   
UpdateSec        = StartSec  
UpdateState      = Ant.ComState  
UpdateSuccessSec = StartSec  
AntConnected     = (Ant.ComState == 1)   

ComTotalCnt = 1   
ComErrorCnt = 0   
print("Succesful initialization")   

# input("Getting initial angular position values for encoders and press enter")  
# The follwing sign definition follow the rigth hand side rule assuming that the rotation axis is pointing outside of the exoskeleton (for each motor)
StartHipAngle_L = rad2deg(Ant.Data.HipAngle_L)     
StartHipAngle_R = rad2deg(Ant.Data.HipAngle_R)     

start = time.time()    

while(AntConnected):  
    if UpdateState == 1:  
        UpdateSuccessSec = GetSec()    
        now = time.time() - start    
        
        L_tau = Ant.Data.HipTor_L    
        R_tau = Ant.Data.HipTor_R     
        
        L_encoder     = rad2deg(Ant.Data.HipAngle_L) - StartHipAngle_L
        R_encoder     = rad2deg(Ant.Data.HipAngle_R) - StartHipAngle_R
        L_encoder_vel = rad2deg(Ant.Data.HipSpeed_L)    
        R_encoder_vel = rad2deg(Ant.Data.HipSpeed_R)   

        L_IMU_angle   = -1 * L_encoder    
        R_IMU_angle   = -1 * R_encoder      
        L_IMU_vel     = -1 * L_encoder_vel      
        R_IMU_vel     = -1 * R_encoder_vel       
        
        if plot_data:   
            imu_buffer[0] = L_IMU_angle
            imu_buffer[1] = R_IMU_angle 
            imu_buffer[2] = L_IMU_vel  
            imu_buffer[3] = R_IMU_vel   
            imu_buffer.flush()   
        
        hip_dnn.generate_assistance(L_IMU_angle, R_IMU_angle, L_IMU_vel, R_IMU_vel, kp, kd)    
        
        L_Cmd     = L_Ctl * hip_dnn.hip_torque_L/torque_scale * kcontrol  
        R_Cmd     = R_Ctl * hip_dnn.hip_torque_R/torque_scale * kcontrol    

        L_Cmd_sat = saturate(L_Cmd, max_cmd)     
        R_Cmd_sat = saturate(R_Cmd, max_cmd)       
        
        # send torque cmd  
        if start_ctl:   
            SendCmdTorque(L_Cmd_sat, R_Cmd_sat)    
        else:  
            SendCmdTorque(0.0, 0.0)     
        
        # SendCmdTorque(00.5, 00.5)    
        
        
        if save_data:  
            write_csv(
                file_name,  
                f"{now:^8.5f}", 
                f"{L_IMU_angle:^8.5f}",     
                f"{R_IMU_angle:^8.3f}", 
                f"{L_IMU_vel:^8.3f}",  
                f"{R_IMU_vel:^8.3f}",   
                f"{L_encoder:^8.3f}",  
                f"{R_encoder:^8.3f}",  
                f"{L_encoder_vel:^8.3f}",  
                f"{R_encoder_vel:^8.3f}",  
                f"{L_Cmd:^8.3f}", 
                f"{R_Cmd:^8.3f}",
                f"{L_tau:^8.3f}", 
                f"{R_tau:^8.3f}" 
            )   
            
        print(f" Time: {now:^8.3f}, L_IMU: {L_IMU_angle:^8.3f} | R_IMU: {R_IMU_angle:^8.3f} | L_IMU_vel: {L_IMU_vel:^8.3f} | R_IMU_vel: {R_IMU_vel:^8.3f} | L_CMD: {L_Cmd:^8.3f} | R_CMD: {R_Cmd:^8.3f} | Peak: {pk:^8.3f} ")
            
    if now > duration:   
        Ant.Disconnect()  
        print('Run Finish')  
        break  
    elif UpdateState == -1:  
        print('Com Error')
        break 
    else: 
        ComErrorCnt += 1 
        if GetSec() - UpdateSuccessSec > 0.3: 
            print('Error: Com Lost') 
            Ant.Cmd.CmdMode = kqio.CMD_OBS_ONLY 
            Ant.Disconnect() 
            break 

    UpdateState = Ant.Update()   
    ComTotalCnt += 1    