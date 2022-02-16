import cv2
import numpy as np
from paddleocr import PaddleOCR

class Acin_OCR():
    def __init__(self):
        self.ocr = PaddleOCR(use_angle_cls=True, lang='en') # need to run only once to download and load model into memory


    ## 이미지 window에 띄우기
    def showimg(self, img):
        cv2.imshow('img',img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    
    ## 초기 셋팅
    def initial_setting(self,img):
        self.height, self.width, self.channel = img.shape
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return self.gaussian_N_threshold(gray)


    ## 가우시안 블러 처리후 2진화
    def gaussian_N_threshold(self, gray):
        img_blurred = cv2.GaussianBlur(gray, ksize=(3, 3), sigmaX=0)

        self.img_thresh = cv2.adaptiveThreshold(
            img_blurred, 
            maxValue=255.0, 
            adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            thresholdType=cv2.THRESH_BINARY_INV, 
            blockSize=19, 
            C=9
        )
        return self.find_contours(self.img_thresh)


    ## 윤곽선 찾기
    def find_contours(self, img_thresh):
        contours, _ = cv2.findContours(
            img_thresh, 
            mode=cv2.RETR_LIST, 
            method=cv2.CHAIN_APPROX_SIMPLE
        )
        
        temp_result = np.zeros((self.height, self.width, self.channel), dtype=np.uint8)

        cv2.drawContours(temp_result, contours=contours, contourIdx=-1, color=(255, 255, 255))
        return self.contour_to_ractangle(contours)


    ## 찾은 윤곽선을 사각형으로 변경(준비단계)
    def contour_to_ractangle(self,contours):
        temp_result = np.zeros((self.height, self.width, self.channel), dtype=np.uint8)

        contours_dict = []

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(temp_result, pt1=(x, y), pt2=(x+w, y+h), color=(255, 255, 255), thickness=2)
            
            # insert to dict
            contours_dict.append({
                'contour': contour,
                'x': x,
                'y': y,
                'w': w,
                'h': h,
                'cx': x + (w / 2),
                'cy': y + (h / 2)
            })
        return contours_dict

    ## BPM 데이터 후보군 찾기
    def find_all_BPM_contours(self,contours_dict):
        ## BPM 데이터 후보군 찾기
        MIN_AREA = 1000    #최소 넓이
        MIN_WIDTH, MIN_HEIGHT = 100, 80    #최소 길이, 최소 높이
        MIN_RATIO, MAX_RATIO = 1.4, 1.7    #가로 대비 세로 비율의 최소, 최대

        bpm_possible_contours = []

        cnt = 0
        for d in contours_dict:
            area = d['w'] * d['h']
            ratio = d['w'] / d['h']
            
            if area > MIN_AREA \
            and d['w'] > MIN_WIDTH and d['h'] > MIN_HEIGHT \
            and MIN_RATIO < ratio < MAX_RATIO:
                d['idx'] = cnt
                cnt += 1
                bpm_possible_contours.append(d)
        return self.find_likely_BPM_contour(bpm_possible_contours)
    
    ## bpm 측정기 후보군 찾기
    def find_likely_BPM_contour(self,bpm_possible_contours):
        one_third_X = round(self.height/3*2)
        one_third_Y = round(self.width/3*1)

        bpm_contour_result = []

        for contour in bpm_possible_contours:
            if contour["x"] < one_third_X and contour["y"] > one_third_Y:
                bpm_contour_result.append(contour)

        ## 좌표를 통해 이미지 자르기
        plate_imgs = []
        for result in bpm_contour_result:
            img_cropped = cv2.getRectSubPix(
                self.img_thresh, 
                patchSize=(int(result["w"]), int(result["h"])), 
                center=(int(result["cx"]), int(result["cy"]))
                )
            plate_imgs.append(img_cropped)
        return self.img_to_string(plate_imgs,division="bpm")
    


    ## 시계 데이터 후보군 찾기
    def find_all_time_contours(self,contours_dict):
        MIN_AREA = 200    #최소 넓이
        MIN_WIDTH, MIN_HEIGHT = 2, 9    #최소 길이, 최소 높이
        MIN_RATIO, MAX_RATIO = 0.2, 1.0    #가로 대비 세로 비율의 최소, 최대

        time_data_possible_contours = []

        cnt = 0
        for d in contours_dict:
            area = d['w'] * d['h']
            ratio = d['w'] / d['h']
            
            if area > MIN_AREA \
            and d['w'] > MIN_WIDTH and d['h'] > MIN_HEIGHT \
            and MIN_RATIO < ratio < MAX_RATIO:
                d['idx'] = cnt
                cnt += 1
                time_data_possible_contours.append(d)
                
        # visualize possible contours
        temp_result = np.zeros((self.height, self.width, self.channel), dtype=np.uint8)

        for d in time_data_possible_contours:
        #     cv2.drawContours(temp_result, d['contour'], -1, (255, 255, 255))
            cv2.rectangle(temp_result, pt1=(d['x'], d['y']), pt2=(d['x']+d['w'], d['y']+d['h']), color=(255, 255, 255), thickness=2)
        return self.find_likely_time_contour(time_data_possible_contours)


    ## 배열 특징을 이용해 가능성이 높은 컨투어 파악
    def find_chars(self,contour_list):
        MAX_DIAG_MULTIPLYER = 6.5 # 5    #컨투어의 대각선길이와 두 컨투어 사이의 중심 좌표의 배수
        MAX_ANGLE_DIFF = 5.0 # 12.0    #두 컨투의 사이의 세타θ 값
        MAX_AREA_DIFF = 0.7 # 0.5    #컨투어의 면적 차이
        MAX_WIDTH_DIFF = 1.0    #컨투어의 가로길이 차이
        MAX_HEIGHT_DIFF = 0.4     #컨투어의 세로 길이 차이
        MIN_N_MATCHED = 3 # 3    #위의 조건들을 만족하는 컨투어의 개수
        matched_result_idx = []
        
        for d1 in contour_list:
            matched_contours_idx = []
            for d2 in contour_list:
                if d1['idx'] == d2['idx']:
                    continue

                dx = abs(d1['cx'] - d2['cx'])
                dy = abs(d1['cy'] - d2['cy'])

                diagonal_length1 = np.sqrt(d1['w'] ** 2 + d1['h'] ** 2)

                distance = np.linalg.norm(np.array([d1['cx'], d1['cy']]) - np.array([d2['cx'], d2['cy']]))
                if dx == 0:
                    angle_diff = 90
                else:
                    angle_diff = np.degrees(np.arctan(dy / dx))
                area_diff = abs(d1['w'] * d1['h'] - d2['w'] * d2['h']) / (d1['w'] * d1['h'])
                width_diff = abs(d1['w'] - d2['w']) / d1['w']
                height_diff = abs(d1['h'] - d2['h']) / d1['h']

                if distance < diagonal_length1 * MAX_DIAG_MULTIPLYER \
                and angle_diff < MAX_ANGLE_DIFF and area_diff < MAX_AREA_DIFF \
                and width_diff < MAX_WIDTH_DIFF and height_diff < MAX_HEIGHT_DIFF:
                    matched_contours_idx.append(d2['idx'])

            # append this contour
            matched_contours_idx.append(d1['idx'])

            if len(matched_contours_idx) < MIN_N_MATCHED:
                continue

            matched_result_idx.append(matched_contours_idx)

            # unmatched_contour_idx = []
            # for d4 in contour_list:
            #     if d4['idx'] not in matched_contours_idx:
            #         unmatched_contour_idx.append(d4['idx'])

            # unmatched_contour = np.take(possible_contours, unmatched_contour_idx)
            
            # # recursive
            # recursive_contour_list = find_chars(unmatched_contour)
            
            # for idx in recursive_contour_list:
            #     matched_result_idx.append(idx)

            # break

        return matched_result_idx


    ## 특징이 매치되는 컨투어 저장
    def find_likely_time_contour(self,time_data_possible_contours):
        result_idx = self.find_chars(time_data_possible_contours)

        time_matched_result = []
        for idx_list in result_idx:
            time_matched_result.append(np.take(time_data_possible_contours, idx_list))
        return self.time_rotate_img(time_matched_result)


    ## 기울기 맞추기
    def time_rotate_img(self,time_matched_result):
        PLATE_WIDTH_PADDING = 3 # 1.3
        PLATE_HEIGHT_PADDING = 1.5 # 1.5
        MIN_PLATE_RATIO = 2
        MAX_PLATE_RATIO = 15

        plate_imgs = []
        plate_infos = []

        for i, matched_chars in enumerate(time_matched_result):
            sorted_chars = sorted(matched_chars, key=lambda x: x['cx'])

            plate_cx = (sorted_chars[0]['cx'] + sorted_chars[-1]['cx']) / 2
            plate_cy = (sorted_chars[0]['cy'] + sorted_chars[-1]['cy']) / 2
            
            plate_width = (sorted_chars[-1]['x'] + sorted_chars[-1]['w'] - sorted_chars[0]['x']) * PLATE_WIDTH_PADDING
            
            sum_height = 0
            for d in sorted_chars:
                sum_height += d['h']

            plate_height = int(sum_height / len(sorted_chars) * PLATE_HEIGHT_PADDING)
            
            triangle_height = sorted_chars[-1]['cy'] - sorted_chars[0]['cy']
            triangle_hypotenus = np.linalg.norm(
                np.array([sorted_chars[0]['cx'], sorted_chars[0]['cy']]) - 
                np.array([sorted_chars[-1]['cx'], sorted_chars[-1]['cy']])
            )
            
            angle = np.degrees(np.arcsin(triangle_height / triangle_hypotenus))
            
            rotation_matrix = cv2.getRotationMatrix2D(center=(plate_cx, plate_cy), angle=angle, scale=1.0)
            
            img_rotated = cv2.warpAffine(self.img_thresh, M=rotation_matrix, dsize=(self.width, self.height))
            
            img_cropped = cv2.getRectSubPix(
                img_rotated, 
                patchSize=(int(plate_width), int(plate_height)), 
                center=(int(plate_cx), int(plate_cy))
            )
            
            if img_cropped.shape[1] / img_cropped.shape[0] < MIN_PLATE_RATIO or img_cropped.shape[1] / img_cropped.shape[0] < MIN_PLATE_RATIO > MAX_PLATE_RATIO:
                continue
            
            plate_imgs.append(img_cropped)
            plate_infos.append({
                'x': int(plate_cx - plate_width / 2),
                'y': int(plate_cy - plate_height / 2),
                'w': int(plate_width),
                'h': int(plate_height)
            })
        return self.img_to_string(plate_imgs)


    ## 이미지 인식
    def img_to_string(self,img_list,division = "time"):
        if division == "time":
            time_result_string = ""
            for img in img_list:
                result = self.ocr.ocr(img, cls=True)
                try:
                    result = result[0][-1][0]   #이미지 인식이 불가능 하면 공백을 반환함 따라서 범위 오류가 남
                except IndexError : 
                    result = "" 
                if len(result) == 12 and result[-3:].isdigit():
                    time_result_string = result
                    break
            return time_result_string
        elif division == "bpm":
            bpm_reslut_text = ""
            for img in img_list:
                result = self.ocr.ocr(img, cls=True)
                try:
                    result = result[0][-1][0]   #이미지 인식이 불가능 하면 공백을 반환함 따라서 범위 오류가 남
                except IndexError : 
                    result = "" 
                if len(result) >= 2  and result.isdigit():
                    bpm_reslut_text = result
                    break
            return bpm_reslut_text


##동영상
acin = Acin_OCR()
path = "./dataset1/first_env9.mp4"
cap = cv2.VideoCapture(path)
result = {"time":[],"bpm":[]}
if cap.isOpened():
    while True:
        ret, img = cap.read()
        if ret:
            cv2.imshow("video_file", img)
            contours_dic = acin.initial_setting(img)
            time_result_string = acin.find_all_time_contours(contours_dic)
            result["time"].append(time_result_string)
            bpm_result_string = acin.find_all_BPM_contours(contours_dic)

            ## 인식한 숫자와 전에 인식한 수의 차이가 5이상 일때는 인식 오류로 봄
            prepare_bpm = 0
            for bpm in result['bpm'][-1::-1]:
                if bpm.isdigit():
                    prepare_bpm = int(bpm)
                    break
            if len(result['bpm']) > 0 and bpm_result_string != "" and (prepare_bpm - int(bpm_result_string) > 4 or prepare_bpm - int(bpm_result_string) < -4):    
                bpm_result_string = ""
                
            result['bpm'].append(bpm_result_string)
            cv2.waitKey(100)
        else:
            break
else:
    print("error")
cap.release()                       
cv2.destroyAllWindows()
print(result)
nodata = []
for i in range(len(result["bpm"])):
    if result["bpm"][i] == "":
        nodata.append(i)
    if result["time"][i] == "":
        nodata.append(i)

nodata = list(set(nodata))
nodata.sort(reverse=True)

for i in nodata:
    del result["bpm"][i]
    del result["time"][i]

for i in range(len(result["time"])):
    print("시간: {}   ||    BPM: {}".format(result["time"][i][3:],result['bpm'][i]))




# #사진
# acin = Acin_OCR()
# path = "./dataset1/111.jpg"
# img_ori = cv2.imread(path)
# contours_dic = acin.initial_setting(img_ori)
# string = acin.find_all_time_contours(contours_dic)
# print(string)
