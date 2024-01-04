# Delaunay分割後、メッシュ生成

import numpy as np
from scipy.spatial import Delaunay
import time
import csv

# define
LOAD = -250000

start = time.time()  # 時間計測用
edges = [] # PLYファイルのedge用


# Input_file = 'Output/result_main/result_112.csv' # Inputファイル
Input_file = 'Output/1_pds/pds2.csv' # Inputファイル
# mesh_data = 'Input/Mesh_Data/cube_50x50mm_mesh.txt' # 物体の表面形状データ。
Output_file = 'Output/2_Delaunay/delaunay2.ply' # Outputファイル
result_csv_path = 'Output/2_Delaunay/delaunay2.csv'

# ファイルの読み込み。物体の頂点を定義する
vertices_3d = np.loadtxt(Input_file, delimiter=',')
print('vertices_3d', vertices_3d[:10])
# 最後の列を削除してxy平面に変換
vertices_2d = np.delete(vertices_3d, -1, axis=1)
print('vertices_2d', vertices_2d[:10])

# Delaunay分割の作成
tri = Delaunay(vertices_2d)
print('simplices：', tri.simplices)

print('len(tri.simplice)：', len(tri.simplices))
i = 0

# edgeノードの作成
for simplice in tri.simplices:
    # Trueのとき，edge生成
    edges.append([simplice[0], simplice[1]])
    edges.append([simplice[0], simplice[2]])
    edges.append([simplice[1], simplice[2]])

#重複したedge nodeを削除
edges, _ = np.unique(edges, return_index=True, axis=0)
print('重複削除後  edges.shape:   ', edges.shape)


# plyで保存
with open(Output_file, 'w', newline="") as f:
    f.write('ply\n')
    f.write('format ascii 1.0\n')
    f.write('comment VCGLIB generated\n')
    f.write('element vertex '+str(len(tri.points))+'\n')
    f.write('property float x\n')
    f.write('property float y\n')
    f.write('property float z\n')
    f.write('element face '+str(len(tri.simplices))+'\n')
    f.write('property list uchar int vertex_indices\n')
    f.write('element edge '+str(len(edges))+'\n')
    f.write('property int vertex1\n')
    f.write('property int vertex2\n')
    f.write('end_header\n')
    for ele in tri.points: # 点群の座標値入力
        f.write(str(ele[0])+' '+str(ele[1])+' '+str(0)+' '+'\n')
    for ele in tri.simplices: # 三角形を構成する点群のノード入力
        f.write('3 '+str(ele[0])+' '+str(ele[1])+' '+str(ele[2])+'\n')
    for ele in edges: # 三角形を構成する点群のノード入力
        f.write(str(ele[0])+' '+str(ele[1])+'\n')

# 追加する列を作成
ones_column1 = np.ones((tri.simplices.shape[0], 1))
nodes = tri.simplices+1
nodes = np.hstack((nodes, ones_column1))

ones_column2 = np.zeros((tri.points.shape[0], 1))
points = np.hstack((tri.points, ones_column2))
print('points:   ', points[0:5])

# 拘束条件作成
constraint_node = []
i = 0
constraint_count = 0 # 拘束条件節点数
for point in tri.points:
    if point[1] == 0:
        constraint_node.append([i, 0, 1, 0, 0])
        constraint_count+=1
        print('i:  ', i)
    i+=1
print('constraint_count:   ', constraint_count)

# 荷重条件作成
load_node = []
i = 0
load_count = 0 # 荷重条件節点数
for point in tri.points:
    if point[1]==50 and 0<=point[0] and point[0]<=25:
        load_node.append([i, 0, LOAD])
        load_count+=1
        print('i:  ', i)
    i+=1
print('load_count:   ', load_count)


# csv にPDSの結果出力
with open(result_csv_path, 'w', newline="") as f:
    writer = csv.writer(f)
    # 節点数 要素数, 材料特性数, 拘束節点数, 載荷節点数, 応力状態（平面歪: 0，平面応力: 1）
    writer.writerow([len(tri.points), len(tri.simplices), 1, constraint_count, load_count, 1])
    # 板厚, 弾性係数[MPa], ポアソン比, 線膨張係数[/K], 単位体積重量，水平及び鉛直方向加速度（gの比）
    writer.writerow([1.0, 110000, 0.28, 0.0000088, 43.458, 0.0, -1.0])
    # 節点1，節点2，節点3，材料特性番号
    writer.writerows(nodes)
    # 節点x座標，節点y座標，節点温度変化
    writer.writerows(points)
    # 拘束節点番号，x及びy方向拘束の有無（拘束: 1，自由: 0）, x及びy方向変位（無拘束でも0を入力）
    writer.writerows(constraint_node)
    # 載荷重節点番号，x方向荷重，y方向荷重
    writer.writerows(load_node)

# 計測結果
elapsed_time = time.time() - start
print ("elapsed_time:{0}".format(elapsed_time) + "[sec]")

