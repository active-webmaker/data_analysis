from dotenv import dotenv_values
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from xgboost import XGBClassifier
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping


# 이상치 대체
def outlier_replace(df, x_cols):
    for xc in x_cols:
        try:
            # 이진 데이터의 경우 이상치 처리에서 제외
            if df[xc].nunique() <= 2:
                continue

            Q1 = df[xc].quantile(0.25)
            Q3 = df[xc].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            # 컬럼 자료형이 정수형인 경우 대체값을 정수형으로 형변환.
            if df[xc].dtype.kind == 'i':
                lower_bound = int(lower_bound)
                upper_bound = int(upper_bound)
            df.loc[df[xc] < lower_bound, xc] = lower_bound
            df.loc[df[xc] > upper_bound, xc] = upper_bound

        except:
            print(f"except: {xc}")
    
    return df



# 다중공선성 제거를 위한 '컬럼 삭제'와 'PCA'의 기준 리스트를 각각 반환
def remove_cols(corr, percent):
    # 상관 행렬의 컬럼명을 추출(반복문의 기준 리스트)
    corr_cols = list(corr.columns)

    # corr_cols을 복제(반환 값의 기준 리스트)
    new_corr_cols = corr_cols.copy()

    # 컬럼 삭제 방식 기준 리스트
    remove_list = []

    # PCA 방식 기준 리스트
    pca_list = []

    for col in corr_cols:
        if col in new_corr_cols:
            # 컬럼명을 기준으로 특정 퍼센트 값 이상의 열을 필터링
            filter_corr = corr.loc[abs(corr[col]) > percent, :]

            # 필터링한 열의 이름을 추출
            corr_index = list(filter_corr.index)
            corr_bool1 = col in corr_index

            # 1. PCA 방식 파트(PCA 방식은 상관관계 별로 그룹화가 필요)

            # 필터링한 값(corr_index) 중 반환 값 기준 리스트(new_corr_cols)에 존재하는 값만 추출
            exist_list = [cidx for cidx in corr_index if cidx in new_corr_cols]

            # 처리한 값(exist_list)을 PCA 기준 리스트에 추가.
            pca_list.append(exist_list)

            # 위에서 처리한 값(exist_list)을 중복 처리하면 다중공선성이 오히려 증가할 수 있음.
            # 때문에 반환 값 기준 리스트(new_corr_cols)에서 제거
            new_corr_cols = [ncol for ncol in new_corr_cols if ncol not in exist_list]


            # 2. 컬럼 삭제 방식 파트(컬럼 삭제 방식은 그룹화가 필요 없음)
            # 삭제 기준 리스트에 필러링한 이름 추가.
            try:
                corr_index.remove(col)
                remove_list += corr_index
            except:
                remove_list += col


    # 컬럼 삭제 방식 기준리스트 중복제거
    remove_list = list(set(remove_list))

    # 반환 리스트 생성
    result = (remove_list, pca_list)
    return result



# 리스트 별 PCA 변환 함수
def pca_process(df, pca_list):
    df_list = []

    # 상관관계가 각 높은 변수들로 묶인 리스트 별로 PCA를 진행
    for pca_item in pca_list:
        if len(pca_item) == 1:
            df_list.append(df[pca_item])
            continue
     
        pca_df = df[pca_item]
        pca = PCA(n_components=1)
        pca_data = pca.fit_transform(pca_df)
        new_pca_df = pd.DataFrame(pca_data)
        df_list.append(new_pca_df)
    
    # 반환 데이터프레임 생성
    df_cnt = len(df_list)
    df_concat = pd.concat(df_list, axis=1)
    return (df_cnt, df_concat)


# 목적(훈련, 검증, 테스트)에 따른 데이터 분리 및 오버샘플링 함수
def data_split(x, y, test_per, val_per=0, sampling='SMOTE'):
    result = []

    # 훈련/테스트 데이터 분리
    (X_train, X_test, y_train, y_test) = train_test_split(
        x, y, test_size=test_per, random_state=42, stratify=y
    )

    if sampling == 'SMOTE':
        # 훈련 데이터 오버샘플링
        smote = SMOTE(random_state=42)
        x_resampled, y_resampled = smote.fit_resample(X_train, y_train)
        result.extend([x_resampled, y_resampled])

    elif sampling == 'udersampling':
        # 훈련 데이터 언더샘플링
        udersampling = RandomUnderSampler(random_state=42)
        x_resampled, y_resampled = udersampling.fit_resample(X_train, y_train)
        result.extend([x_resampled, y_resampled])

    # 검증/테스트 데이터 분리
    if val_per != 0:
        (X_val, X_test, y_val, y_test) = train_test_split(
            X_test, y_test, test_size=val_per, random_state=42, stratify=y_test
        )
        result.extend([X_val, y_val])
    
    result.extend([X_test, y_test])
    return result


# 성능지표 출력 및 저장 함수
def report_save(y_test, y_pred, y_prob, sav_name):
    report_dic = classification_report(y_test, y_pred, output_dict=True)
    report = classification_report(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)
    
    pd.DataFrame(report_dic).to_csv(f'{sav_name}_report.csv', index=True, encoding='utf-8-sig')
    with open(f'{sav_name}_report.txt', mode='w', encoding='utf-8-sig') as f:
        f.write(str(report) + '\nROC-AUC Score: ' + str(roc_auc))



# XGBoost 학습 및 예측 함수
def ml_model(split_data, sav_name, model=None, proba=True):
    [X_train, y_train, X_test, y_test] = split_data

    if model is None:
        model = XGBClassifier(tree_method='gpu_hist', gpu_id=0)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    if proba:
        y_prob = model.predict_proba(X_test)[:, 1]
    else:
        y_prob = model.decision_function(X_test)

    report_save(y_test, y_pred, y_prob, sav_name)



# Keras DNN 학습 및 예측 함수
def dl_model(split_data, input_num, sav_name, model=None, epochs=1000, batch_size=32, patience=10, early_stopping=None):
    [X_train, y_train, X_val, y_val, X_test, y_test] = split_data

    if model is None:
        model = Sequential([
            Dense(128, activation='relu', input_dim=input_num),
            Dropout(0.3),
            Dense(64, activation='relu'),
            Dropout(0.3),
            Dense(32, activation='relu'),
            Dense(1, activation='sigmoid')
        ])

    # 컴파일 및 학습
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    if early_stopping is None:
        early_stopping = EarlyStopping(monitor='val_loss', patience=patience)
        
    model.fit(
        X_train, y_train, 
        epochs=epochs, 
        batch_size=batch_size, 
        validation_data=(X_val, y_val), 
        callbacks=[early_stopping]
    )

    # 모델 저장
    model.save(f'{sav_name}.h5')

    # 모델 예측
    y_pred = model.predict(X_test)
    y_prob = y_pred.copy()
    y_pred[y_pred >= 0.5] = 1
    y_pred[y_pred < 0.5] = 0

    # 평가
    report_save(y_test, y_pred, y_prob, sav_name)




# dnn 모델 레이어 구성 함수
def dnn_model_build(input_num, unit_list, drop_rate=0):
    dnn_model = Sequential([
        Dense(unit_list[0], activation='relu', input_dim=input_num),
    ])
    for item in unit_list[1:]:
        if drop_rate > 0:
            drop = Dropout(drop_rate)
            dnn_model.add(drop)
        dense = Dense(item, activation='relu')
        dnn_model.add(dense)

    dnn_model.add(Dense(1, activation='sigmoid'))
    
    return dnn_model



# XGBoost 모델 파라미터 딕셔너리
xgb_param = {
    'n_estimators': np.linspace(50, 500, 6).astype(int),
    'max_depth': np.linspace(3, 8, 5).astype(int),
    'learning_rate': np.logspace(-2, 0, 5),
    'subsample': np.linspace(0.5, 1.0, 6),
    'colsample_bytree': np.linspace(0.3, 1.0, 8),
    'gamma': np.linspace(0, 5.0, 10),
}

# DNN 모델 파라미터 딕셔너리
dnn_parm = {
    'epochs': [20, 30, 50, 100], 
    'batch_size': [2, 4, 8, 16, 32, 64], 
    'callbacks': [EarlyStopping(monitor='val_loss', patience=20), EarlyStopping(monitor='val_loss', patience=30)]
}


# 모델 학습 함수
def model_learning(data_list, df_y, sampling="SMOTE"):
    # GPU 메모리 증가 허용 설정
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
    
    # GPU 모델 학습
    with tf.device('/GPU:0'):
        # 1차 기본 튜닝 값 시도
        for [data, sav_name, input_num] in data_list:
            print("1차 기본 값 모델 학습 시도")
            print('data_cnt:', data.shape)
            print('df_y:', df_y.shape)
            print('sav_name:', sav_name)
            print('input_num:', input_num, '\n')
            sav_name = f"{sampling}_1st_tuning_{sav_name}"

            split_data = data_split(data, df_y, 0.2, sampling=sampling)
            ml_model(split_data, sav_name + '_ml')

            split_data = data_split(data, df_y, 0.4, 0.5, sampling=sampling)
            dl_model(split_data, input_num, sav_name + '_dl')

    xgb_hyper = {
        'n_estimators': 500,
        'max_depth': 10,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.6,
        'gamma': 3.0,
    }

    # GPU 모델 학습
    with tf.device('/GPU:0'):
        # 2차 튜닝 값 모델 시도
        for [data, sav_name, input_num] in data_list:
            print("2차 튜닝 값 모델 학습 시도")
            sav_name = f"{sampling}_2nd_tuning_{sav_name}"
            split_data1 = data_split(data, df_y, 0.2, sampling=sampling)
            split_data2 = data_split(data, df_y, 0.3, 0.5, sampling=sampling)

            xgb_model = XGBClassifier(tree_method='gpu_hist', gpu_id=0, **xgb_hyper)
            dnn_model = dnn_model_build(input_num, [256, 128, 64, 32, 16, 8, 4], 0.5)

            ml_model(split_data1, sav_name + '_ml', model=xgb_model)
            dl_model(split_data2, input_num, sav_name + '_dl', model=dnn_model, epochs=300, batch_size=4, patience=10)


def main():
    # 환경변수 로드
    config = dotenv_values('.env')
    DATA_PATH =  config['DATA_PATH']

    # 데이터셋 로드
    df = pd.read_csv(DATA_PATH, encoding='utf-8-sig')
    print(df.info(), '\n')
    print(df.describe(), '\n')
    print(df.head(), '\n')

    df = df.drop(columns=[' Net Income Flag'])

    # 상관행렬 분석
    corr = df.corr()
    corr.to_csv('corr.csv', index=True, encoding='utf-8')
    corr.to_csv('corr_sig.csv', index=True, encoding='utf-8-sig')

    # 이상치 처리
    x_cols = list(df.columns)
    x_cols.remove('Bankrupt?')
    df = outlier_replace(df, x_cols)

    # 데이터 표준화(정규화)
    df_scaler = StandardScaler()
    
    df_x = df.drop(columns=['Bankrupt?'])
    df_scaler_x = df_scaler.fit_transform(df_x)
    df_x = pd.DataFrame(df_scaler_x, columns=df_x.columns)
    df_y = df['Bankrupt?']

    # 다중공선성 제거를 위한 컬럼 삭제 또는 PCA 변환
    corr = df_x.corr()
    (remove_list, pca_list) = remove_cols(corr, 0.7)
    df_droped = df_x.drop(columns=remove_list)
    (df_cnt, df_pca1) = pca_process(df_x, pca_list)

    pca_num = 30
    pca = PCA(n_components=pca_num)
    df_pca2 = pca.fit_transform(df_x)

    # PCA 적용
    pca = PCA()
    pca.fit(df_x)

    # 설명 분산 비율 누적합 계산
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
    n_components_optimal = np.argmax(cumulative_variance >= 0.90) + 1

    pca = PCA(n_components=n_components_optimal)
    df_pca3 = pca.fit_transform(df_x)
    
    df_droped_cols = len(df_droped.columns)
    df_droped = df_droped.to_numpy()
    df_pca1 = df_pca1.to_numpy()
    df_y = df_y.to_numpy().astype('int')

    # 다중공선성을 제거한 데이터셋 리스트
    data_list = [
        [df_droped, 'not_pca', df_droped_cols],
        [df_pca1, 'pca1', df_cnt],
        [df_pca2, 'pca2', pca_num],
        [df_pca3, 'pca3', n_components_optimal]
    ]

    # 오버샘플링 방식 모델학습
    model_learning(data_list, df_y)

    # 언더샘플링 방식 모델학습
    model_learning(data_list, df_y, sampling='udersampling')


if __name__ == '__main__':
    main()
    print('프로그램 종료')
