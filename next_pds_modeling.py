from modules import Point
from modules import CheckDistance
# from modules import Biharmonic
# from modules import Gauss
from scipy.interpolate import Rbf

import time
import numpy as np
import random
import csv

# define
MAXIMUM_NUMBER_OF_SEARCHES = 10000 # 点が連続でN回生成できなかったら終了
MAXIMUM_NUMBER_OF_POINTS = 441 # 最大生成点数(内部点+表面点)
ALLOWABLE_STRESS = 186 #[MPa] # チタン合金．降伏強さ930MPa．安全率5
INTERVAL = 2.5 #物体表面の点群間隔
# 点の生成範囲の設定
X_MAX = 50
X_MIN = 0
Y_MAX = 50
Y_MIN = 0


# Outputファイル
input_path = 'Output/4_trass/trass4/trass4_stress.csv'
result_ply_path = 'Output/6_next_pds/pds4_re.ply'
result_csv_path = 'Output/6_next_pds/pds4_re.csv'

def main():
    points_obj_list = []  # Pointオブジェクトを保持
    fixed_points = [] # 確定点格納用

    # ANSYS上の点群を取得し座標値を取得
    points = np.loadtxt(input_path, delimiter=',', skiprows=1) # 1行目はラベルのためスキップ
    points = np.insert(points, 2, 0, axis=1)
    print('points = ',points[0:3])
    
    # FEMの節点の読み込み
    for i in range(len(points)):
        point = Point.Point(points[i]) # クラスに格納
        point.system_guid_obj_to_coordinate()
        points_obj_list.append(point)
    
    # Rbfクラスのインスタンス化
    rbf = Rbf(points[:,0], points[:,1], points[:,3], function='multiquadric')  # functionは基底関数の種類を指定できます（デフォルトは'multiquadric'）

    # # Gauss
    # gauss = Gauss.Gauss(points)
    # _, lambdas, cs = gauss.gauss()
    # print('(lambdas, cs):    ', lambdas, cs)

    # # Biharmonic
    # biharmonic = Biharmonic.Biharmonic(points, cs, lambdas)

    # PDS用
    CD = CheckDistance.CheckDistance(ALLOWABLE_STRESS)

    #物体表面に点群生成
    alt_INTERVAL = int(INTERVAL * 10) # INTERVALが小数の場合の対策として10倍している
    for y in range(Y_MIN*10, Y_MAX*10+1, alt_INTERVAL): # INTERVALが小数の場合の対策として10倍している
        y /= 10
        fixed_points.append([X_MIN, y, 0])
        fixed_points.append([X_MAX, y, 0])
    for x in range(X_MIN*10, X_MAX*10+1, alt_INTERVAL): # INTERVALが小数の場合の対策として10倍している
        x /= 10
        fixed_points.append([x, Y_MIN, 0])
        fixed_points.append([x, Y_MAX, 0])
    
    #重複した座標を削除
    fixed_points, _ = np.unique(fixed_points, return_index=True, axis=0)
    surface_fixed_points = len(fixed_points)
    fixed_points = fixed_points.tolist() # append用?


    num = 0
    while num < MAXIMUM_NUMBER_OF_SEARCHES:
        if len(fixed_points) >= MAXIMUM_NUMBER_OF_POINTS:
            break
        
        # ランダムな点を生成
        pds_x = random.uniform(X_MIN, X_MAX)
        pds_y = random.uniform(Y_MIN, Y_MAX)
        pds_point = [pds_x, pds_y, 0]
        #print("pds_point : ",pds_point)

        # 生成点の座標情報をPointに格納し候補点にする
        candidate_point = Point.Point(pds_point)
        candidate_point.pds_coordinate()
        # 点間距離内に他の点が含まれているか否かを判定
        # new_stress = biharmonic.cal(candidate_point.x, candidate_point.y, candidate_point.z)
        new_stress = rbf(candidate_point.x, candidate_point.y)  # 補間値の計算
        #print('new_stress:   ', new_stress)
        flg = CD.check_distance(fixed_points, candidate_point, new_stress)

        # 点間距離内に他の点が存在しないとき候補点を確定点に追加
        if flg:
            fixed_points.append(pds_point)
            num = 0
            print('num : ', num, 'fixed_points : ', len(fixed_points))
        # 点間距離内に他の点が存在するとき
        else :
            num = num + 1
            print('num : ', num)        

    #重複した座標を削除
    fixed_points, _ = np.unique(fixed_points, return_index=True, axis=0)

    print("surface_fixed_points  = ", surface_fixed_points, "個")

    # ply にPDSの結果出力
    print("fixed_points  = ", len(fixed_points), "個")
    with open(result_ply_path, 'w', newline="") as f:
        f.write('ply\n')
        f.write('format ascii 1.0\n')
        f.write('element vertex '+str(len(fixed_points))+'\n')
        f.write('property double x\n')
        f.write('property double y\n')
        f.write('property double z\n')
        f.write('end_header\n')
        for ele in fixed_points:
            f.write(str(ele[0])+' '+str(ele[1])+' '+str(ele[2])+'\n')

    # csv にPDSの結果出力
    with open(result_csv_path, 'w', newline="") as f:
        writer = csv.writer(f)
        writer.writerows(fixed_points)


if __name__ == '__main__':
    start = time.time()  # 時間計測用
    print("START")
    main() # 点群生成
    
    # 時間計測結果の表示
    elapsed_time = time.time() - start 
    print ("elapsed_time:{0}".format(elapsed_time) + "[sec]")
