from geopy.distance import geodesic
import pandas as pd
import pandas as pd
import json
from dotenv import dotenv_values
import osmnx as ox
import networkx as nx
import sys
from datetime import datetime
import joblib


# 환경 변수 로드 함수
def config_load(keys):
    config = dotenv_values('.env')
    result = []
    for key in keys:
        result.append(config[key])

    return result


# 최단 대피소 탐색 함수
def distance_calculator(location, df, latitude, longitude):
    shelters = df.copy()

    def get_distance(row):
        dt = geodesic(location, (row[latitude], row[longitude])).km
        return dt

    shelters['distance'] = shelters.apply(get_distance, axis=1)
    min_distance = shelters['distance'].min()
    nearest = shelters.loc[shelters['distance'] == min_distance].iloc[0]

    return nearest



# GPS 좌표를 통해 최단 노드 탐색 함수
def find_nearest_node(row, G, col1, col2):
    row_col1 = row[col1]
    row_col2 = row[col2]
    node = None
    if row_col1 and row_col2:
        node = ox.nearest_nodes(G, row_col2, row_col1)
    
    return node


# 노드 ID를 통한 최단 경로 탐색 함수
def shortest_path(row, G):
    try:
        if (not row['학교_노드']) or (not row['대피소_노드']):
            result = None
        elif row['학교_노드'] == row['대피소_노드']:
            result = [row['학교_노드']]
        else:
            result = nx.shortest_path(G, row['학교_노드'], row['대피소_노드'])

    # 경로가 없는 경우 처리
    except nx.NetworkXNoPath as path_error:
        print(path_error)
        result = None
    
    except nx.NodeNotFound as node_error:
        print(node_error)
        result = None
    
    except Exception as e:
        print(e)
        sys.exit()
    
    return result


# 컬럼 별 None 값 확인 함수
def isna_some_cnt(cols, df, is_iterable=False):
    print()

    # 카운팅 함수(None인 경우 0을 반환)
    def len_row(row):
        if row:
            return len(row)
        else:
            return 0

    for col in cols:
        # 자료형이 배열(2차원) 형태인 경우
        if is_iterable:
            df_copy = df.copy()
            df_copy['cnt'] = df_copy[col].apply(len_row)
            cnt = df_copy.loc[df_copy['cnt']==0, 'cnt'].count()
            print(f'{col} 없는 개수: {cnt}')

        # 자료형이 스칼라 형태인 경우
        else:
            cnt = df[col].isna().sum()
            print(f'{col} 없는 개수: {cnt}')


# 전체 경로 거리 계산 함수
def whole_distance(route, G):
    result = None
    try:
        if route:
            if len(route) == 1:
                result = 0
            elif len(route) > 1:
                edge_lengths = ox.routing.route_to_gdf(G, route)
                result = edge_lengths['length'].sum()
    except Exception as e:
        print(f'error: {e}')
        result = None
    
    return result


# 현재 시간 표시 함수
def datetime_now():
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')


# 메인 함수
def main():
    # 환경 변수 로드
    keys = ['SAV_PATH', 'SHELTER_DATA']
    [SAV_PATH, SHELTER_DATA] = config_load(keys)

    # JSON 형식의 초등학교 GPS 좌표 정보 파일 로드
    geo_dict = None
    with open(SAV_PATH + 'geo_json.json', 'r', encoding='utf-8') as f:
        geo_dict = json.load(f)
    
    geo_keys = geo_dict.keys()

    # 대피소 정보 파일 로드
    shelter_df = pd.read_excel(SHELTER_DATA).dropna().reset_index(drop=True)
    shelter_df = shelter_df[shelter_df['도로명전체주소'].str.contains('서울특별시')].reset_index(drop=True)

    # 컬럼명 변수 선언
    place_col = '시설명'
    latitude_col = '위도(EPSG4326)'
    longitude_col = '경도(EPSG4326)'
    
    # 초등학교 GPS 정보 딕셔너리를 데이터프레임으로 변환
    school_df = pd.DataFrame(geo_dict).T.reset_index().dropna()
    school_df.columns = ['학교명', '학교_위도', '학교_경도']

    # 초등학교 GPS 정보 데이터프레임에 추가 컬럼 생성
    school_df['대피소명'] = ''
    school_df['대피소_위도'] = None
    school_df['대피소_경도'] = None
    isna_some_cnt(['학교_위도', '학교_경도'], school_df)


    # 초등학교 별 최단 거리 대피소 탐색
    print(f'\n대피소 탐색 중 - {datetime_now()}\n')
    fornum = 1
    for location in geo_keys:
        if fornum % 100 == 1:
            print(f'{fornum}번째 학교의 대피소 탐색 중')
        
        school_point = geo_dict[location]

        # 가장 가까운 대피소 탐색
        nearest = distance_calculator(school_point, shelter_df, latitude_col, longitude_col)

        # 반환 받은 시리즈에서 대피소 정보 파싱
        place = nearest[place_col]
        latitude = nearest[latitude_col]
        longitude = nearest[longitude_col]

        # 파싱한 대피소 정보 할당
        school_df.loc[school_df['학교명']==location, '대피소명'] = place
        school_df.loc[school_df['학교명']==location, '대피소_위도'] = latitude
        school_df.loc[school_df['학교명']==location, '대피소_경도'] = longitude
        fornum += 1

    print(f'전체 학교({fornum}개) 대피소 탐색 완료')
    isna_some_cnt(['대피소_위도', '대피소_경도'], school_df)


    # 서울시 그래프 맵 로드
    print(f'\n서울시 그래프 맵 로드 - {datetime_now()}\n')
    G = joblib.load('seoul_graph.pkl')

    # 지점 별 최단 노드 탐색
    print(f'\n대피 경로 탐색 중 - {datetime_now()}\n')
    school_df['학교_노드'] = school_df.apply(find_nearest_node, G=G, col1='학교_위도', col2='학교_경도', axis=1)
    school_df['대피소_노드'] = school_df.apply(find_nearest_node, G=G, col1='대피소_위도', col2='대피소_경도', axis=1)
    isna_some_cnt(['학교_노드', '대피소_노드'], school_df)

    # 노드 간 최단 경로 탐색
    school_df['최단_대피_경로'] = school_df.apply(shortest_path, G=G, axis=1)
    isna_some_cnt(['최단_대피_경로',], school_df, is_iterable=True)

    # 최단 경로 대피 거리 및 소요 시간 계산
    print(f'\n대피 시간 계산 중 - {datetime_now()}\n')
    school_df['총_대피_거리'] = None
    school_df['예상_소요_시간(분)'] = None
    school_df['총_대피_거리'] = school_df['최단_대피_경로'].apply(whole_distance, G=G)
    school_df['예상_소요_시간(분)'] = round((school_df['총_대피_거리'] / (56.5)), 2)
    school_df.to_csv(SAV_PATH + 'school_evacuation.csv', index=False, encoding='utf-8-sig')
    

    # 산출된 데이터프레임 파일로 저장
    school_df_dropna = school_df.dropna()
    school_df_asc = school_df_dropna.sort_values(by='예상_소요_시간(분)', ascending=True)
    school_df_desc = school_df_dropna.sort_values(by='예상_소요_시간(분)', ascending=False)
    school_df_asc.head(10).to_csv(SAV_PATH + 'top_school_evacuation.csv', index=False, encoding='utf-8-sig')
    school_df_desc.head(10).to_csv(SAV_PATH + 'bottom_school_evacuation.csv', index=False, encoding='utf-8-sig')


    # 소요시간 분석
    mean_escape_time = school_df_dropna['예상_소요_시간(분)'].mean()
    top_escape_time1 = school_df_asc.head(10)['예상_소요_시간(분)'].mean()
    top_escape_df = school_df_asc.loc[school_df_asc['예상_소요_시간(분)'] > 10]
    top_escape_time2 = top_escape_df.head(10)['예상_소요_시간(분)'].mean()
    bottom_escape_time = school_df_desc.head(10)['예상_소요_시간(분)'].mean()
    time_diff1 = bottom_escape_time - top_escape_time1
    time_diff2 = bottom_escape_time - top_escape_time2

    # 소요시간 관련 분석 결과 출력
    mean_txt = f"전체 학교 대피시간 평균: {mean_escape_time}분\n"
    mean_txt += f"상위 10개 학교 대피시간 평균1(0 포함): {top_escape_time1}분\n"
    mean_txt += f"상위 10개 학교 대피시간 평균2(0 제외): {top_escape_time2}분\n"
    mean_txt += f"하위 10개 학교 대피시간 평균: {bottom_escape_time}분\n"
    mean_txt += f"상하위 학교 간 대피시간 평균 차이1(0 포함): {time_diff1}분\n"
    mean_txt += f"상하위 학교 간 대피시간 평균 차이2(0 제외): {time_diff2}분\n"
    print('\n', mean_txt)

    with open(SAV_PATH + 'escape_time.txt', 'w', encoding='utf-8') as f:
        f.write(mean_txt)


if __name__ == "__main__":
    print(f'start time: {datetime_now()}\n')
    main()
    print(f'\nend time: {datetime_now()}')
    print('\n프로그램 종료')

