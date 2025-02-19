import joblib
import pandas as pd
from dotenv import dotenv_values
import osmnx as ox


# 환경 변수 로드 함수
def config_load(keys):
    config = dotenv_values('.env')
    result = []
    for key in keys:
        result.append(config[key])
    return result


# 환경 변수 로드
keys = ['SAV_PATH',]
[SAV_PATH,] = config_load(keys)
G = ox.graph_from_place('서울시, 대한민국', network_type='all')

joblib.dump(G, 'seoul_graph.pkl')