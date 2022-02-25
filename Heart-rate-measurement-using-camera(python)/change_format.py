from unittest import result
import pandas as pd
import numpy as np

f = open("result/seungcheol.txt","r")
dic = {"time":[],"bpm":[]}

##0.0초 때의 시간을 입력받음
# initial_time = input()
initial_time = "18:12:32:024"
# 소요시간은 시간 데이터 형식으로 변경
def result_time(addtime,initial_time = initial_time):
    ini_time_split = initial_time.split(":")
    int_ini_time_list = [int(x) for x in ini_time_split]
    addtime_split = addtime.split(".")
    int_addtime_list = [int(x) for x in addtime_split]
    int_ini_time_list[-1] += int_addtime_list[-1]
    int_ini_time_list[-2] += int_addtime_list[0]
    if int_ini_time_list[-1] >=1000:
        int_ini_time_list[-2] += 1
        int_ini_time_list[-1] -= 1000
    if int_ini_time_list[-2] >=60:
        int_ini_time_list[-3] += 1
        int_ini_time_list[-2] -= 60
    if int_ini_time_list[-3] >=60:
        int_ini_time_list[-4] += 1
        int_ini_time_list[-3] -= 60
    int_ini_time_list = [str(x) for x in int_ini_time_list]
    return ":".join(int_ini_time_list)



# 소요시간 데이터를 읽어 dic에 저장
while(True):
    lines = f.readline()
    if not lines:
        break
    lines = lines.split(",")
    lines[-1] = lines[-1].strip().split(".")[0]
    lines[-1] = lines[-1][:-3]+"."+lines[-1][-3:]
    time = result_time(lines[-1])
    dic["time"].append(time)
    dic["bpm"].append(lines[1])


## 0.5초 간격으로 평균 구하기   
def mk_avg_result(initial_result):
    result = {"time":[],"bpm":[]}
    temp_time = initial_result["time"][0][:9] + ("0" if int(initial_result["time"][0][9]) < 5 else "5")
    temp_bpm = []
    for i in range(len(initial_result["time"])):
        initial_millisec = int(initial_result["time"][i].split(":")[3][0])
        # 0.5초를 기준으로 변경될 때 마다 result dict 에 저장
        if initial_millisec < 5:
            if temp_time != initial_result["time"][i][:9] + "0" and len(temp_bpm)!=0:
                temp_bpm = np.array(temp_bpm)
                bpm_avg = np.mean(temp_bpm)
                result["time"].append(temp_time)
                result["bpm"].append(bpm_avg)
                temp_time = initial_result["time"][i][:9] + "0"
                temp_bpm = []
            else:
                temp_bpm.append(float(initial_result["bpm"][i]))
        else:
            if temp_time != initial_result["time"][i][:9] + "5" and len(temp_bpm)!=0:
                temp_bpm = np.array(temp_bpm)
                bpm_avg = np.mean(temp_bpm)
                result["time"].append(temp_time)
                result["bpm"].append(bpm_avg)
                temp_time = initial_result["time"][i][:9] + "5"
                temp_bpm = []
            else:
                temp_bpm.append(float(initial_result["bpm"][i]))
        
       
    return result
    
   

result_pd = pd.DataFrame(mk_avg_result(dic))
result_pd.to_csv("chage_format.csv",encoding="cp949")
