from geopy.geocoders import Nominatim
import json
import time
import pandas as pd
import numpy as np
from dotenv import dotenv_values
import os


def config_load(keys):
    config = dotenv_values('.env')
    result = []
    for key in keys:
        result.append(config[key])
    return result


def geo_finder(locations, sav_name, user_agent="South Korea", succ_msg=False, fail_msg=True):
    # Geolocator 초기화
    geolocator = Nominatim(user_agent=user_agent)
    result = {}

    # 좌표 찾기
    for location in locations:
        [name, location1, location2] = location
        loc = geolocator.geocode(location1)
        if loc:
            la = loc.latitude
            lo = loc.longitude
            result[name] = [la, lo]
            if succ_msg:
                print(f"{name}: {la}, {lo}")
        else:
            loc = geolocator.geocode(location2)
            if loc:
                la = loc.latitude
                lo = loc.longitude
                result[name] = [loc.latitude, loc.longitude]
                if succ_msg:
                    print(f"{name}: {loc.latitude}, {loc.longitude}")
            else:
                result[name] = ['nodata', 'nodata']
                if fail_msg:
                    print(f"{name}의 좌표를 찾을 수 없습니다.")
        time.sleep(2)
        
    with open(sav_name, 'w', encoding="utf-8") as f:
        print("\nJSON 파일로 내보내는 중입니다.")
        json.dump(result, f, indent=4, ensure_ascii=False)
        print("파일 생성 완료")

    return result


try:
    # 환경 변수 로드
    keys = ['SCHOOL_DATA', 'SAV_PATH']
    [SCHOOL_DATA, SAV_PATH] = config_load(keys)

    elschool = pd.read_csv(SCHOOL_DATA, encoding='cp949')
    elschool_cols = ['학교명', '도로명주소']
    elschool_crop = pd.DataFrame(elschool['학교명'])
    elschool_crop['학교명2'] = "서울특별시 " + elschool_crop['학교명']
    elschool_crop['도로명주소'] = elschool['도로명주소']
    print('데이터프레임 변환')
    elschool_arr = np.array(elschool_crop).tolist()
    print(f'elschool_arr_type: {type(elschool_arr)}, {type(elschool_arr[0])}')
    print(f'elschool_arr_0: {elschool_arr[0]}')

    print('저장 경로 설정')
    sav_name = f'{SAV_PATH}geo_json.json'
    geo_dict = geo_finder(locations=elschool_arr, sav_name=sav_name, succ_msg=True, fail_msg=True)

except Exception as e:
    print(f'Error: {e}')
    os.system('pause')