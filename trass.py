# -*- coding: utf-8 -*-
'''平面トラス解析プログラム Ver.1.04 (2021.2.11) by Snow Tree in June
  
平面トラスの節点変位、部材応力、反力の算定
'''
import numpy as np
import trimesh
import csv

# mesh_data = 'Output/2_Delaunay/delaunay2/delaunay2.ply'
mesh_data = 'Output/2_Delaunay/delaunay2.ply'
# txt_file_path = 'Output/2_Delaunay/delaunay2/edge_node.txt'
result_displacement_csv_path = 'Output/4_trass/trass_displacement.csv'
result_stress_csv_path = 'Output/4_trass/trass_stress.csv'

# 2つの数字からなる組が複数存在するとき，同じペアを一つにする関数
# ex.)[[1, 2], [3, 4], [5, 6], [2, 1], [7, 8], [4, 3]]→[[5, 6], [7, 8], [1, 2], [3, 4]]
# target_listは重複削除後のペアを入れること
# ex.)[[1, 2], [3, 4], [5, 6], [1, 2], [2, 1]] は，[1, 2]が重複しているためNG
def remove_pair(target_list):
    # 同一ペア削除後のリスト
    new_list = []
    target_list = target_list.tolist() # list型に変換
    # x座標とy座標を入れ替える
    swapped_list = [[y, x] for x, y in target_list]

    i_list = []
    j_list = []
    i=0
    for pare1 in target_list:
        flg = True
        j = 0
        for pare2 in swapped_list:
            if pare1 == pare2:
                i_list.append(i)
                j_list.append(j)
                flg = False
                break
            j+=1
        if flg:
            new_list.append(target_list[i])
        i+=1
        
    # 同じペアの添字を-1に変換
    for k in range(len(i_list)):
        target = i_list[k]
        if target in j_list:
            j_list[k] = -1

    # 重複削除後のlist作成
    for j in j_list:
        if j == -1: 
            continue
        new_list.append(target_list[j])

    return new_list

def input_data_3():
  #関数からの出力
  global siz_nod, var_max
  global nod_tn, mem_tn, bc_tn, nl_tn
  global mem_ym
  global nod_coo, mem_are, mem_nn, bc_nn, bc_ff, nl_nn, nl_mag
  # 3Dメッシュデータを読み込む
  mesh = trimesh.load(mesh_data)
  # 最後の列を削除してxy平面に変換
  vertices = mesh.vertices
  vertices = np.delete(vertices, -1, axis=1)
  print('vertices:   ', vertices)
  # edgeデータの作成
  edges = mesh.edges
  #重複したedge nodeを削除
  edges, _ = np.unique(edges, return_index=True, axis=0)
  # print('重複削除後  edges.shape:   ', edges.shape)
  edges = remove_pair(edges)
  edges = np.array(edges)
  # print('remove_pair後  edges.shape:   ', edges.shape)
  # print('edges:   ', edges)

  # siz_nod(=2):節点自由度。x、y方向変位の2種類
  siz_nod = 2
  # var_max(=65535):部材数などで指定した変数の型が取り得る最大値
  var_max = 65535
  # nod_tn:総節点数
  nod_tn = len(vertices)
  # mem_tn:総部材数
  mem_tn = len(edges)
  # print('mem_tn:   ', mem_tn)
  # print('len(edges):   ', len(edges))

  # 拘束条件作成
  constraint_coord = []
  constraint_node = []
  i = 0
  for vertice in vertices:
    if vertice[1] == 0:
      constraint_coord.append(vertice)
      constraint_node.append(i)
    i+=1
  # bc_tn:境界条件を与える節点の数
  bc_tn = len(constraint_coord)

  # 荷重条件作成
  load_coord = []
  load_node = []
  i = 0
  for vertice in vertices:
    if vertice[1]==50 and 0<=vertice[0] and vertice[0]<=25:
      load_coord.append(vertice)
      load_node.append(i)
    i+=1
  # nl_tn:外力が作用する節点の数
  nl_tn = len(load_coord)

  # mem_ym   :ヤング係数
  mem_ym = 110000 #[MPa]

  #適用範囲の確認
  if (nod_tn * siz_nod) > (var_max - 1):
    print('error:節点数が適用範囲を超えている。nod_tn * siz_nod > var_max -1 ') 
  if mem_tn > var_max:
    print('error:部材数が適用範囲を超えている。mem_tn > var_max')

  #'uint16' 0~65535
  #numpy配列の宣言
  # nod_coo = np.zeros((nod_tn, siz_nod), dtype = 'float64')
  mem_are = np.zeros(mem_tn, dtype = 'float64')
  mem_nn = np.zeros((mem_tn, siz_nod), dtype = 'uint16')
  # bc_nn = np.zeros(bc_tn, dtype = 'uint16')
  bc_ff = np.zeros((bc_tn, siz_nod), dtype = 'bool')
  # nl_nn = np.zeros(nl_tn, dtype = 'uint16')
  nl_mag = np.zeros((nl_tn, siz_nod), dtype = 'float64')

  # nod_coo[i, j]:i節点の座標(j=0:x方向, j=1:y方向)
#   nod_coo[:] = [[0., 0.], [100., 100.], [200., 0.], [300., 100.], [400., 0.]]
  nod_coo = vertices
  # mem_are[i]   :i部材の断面積
  mem_are[:] = 2. #[mm]
  # mem_nn[i, j]:i部材を構成する節点番号(j=0, 1)
  mem_nn = edges

  # bc_nn[i]:境界条件を与えられている節点番号
  bc_nn = constraint_node
  # bc_ff[i, j]:fixの方向:0, freeの方向:1(j=0:x方向, J=1:y方向)
  bc_ff[:] = [[0, 0]]
  # nl_nn[i]:外力が作用する節点番号
  nl_nn = load_node
  # nl_mag[i, j]:外力の大きさ(j=0:x方向, J=1:y方向)
  nl_mag[:] = [[0., -250.]] #[MPa]


def output_data():
  '''入力データの出力
  
  output_data: [OUTPUT][DATA]
  '''
  #関数への入力
  global siz_nod, mem_ym, nod_tn, nod_coo
  global mem_tn, mem_nn, bc_tn, bc_nn, bc_ff
  global nl_tn, nl_nn, nl_mag
  
  print('========================================')
  print('１．入力データ')
  print('(1) 制御データ')
  print(' siz_nod, var_max：[{}, {}]'.format(siz_nod, var_max))
  print('(2) 共通データ')
  print(' E：[{}]'.format(mem_ym))
  print('(3) 節点座標')
  print(' 節点数：{}'.format(nod_tn))
  for i_nn in range(nod_tn):
    print('{:4d}: [x:{:12.5f}, y:{:12.5f}]'\
      .format(i_nn, nod_coo[i_nn, 0], nod_coo[i_nn, 1]))
  print('(4) 部材')
  print(' 部材数：{}'.format(mem_tn))
  for i_mn in range(mem_tn):
    print('{:4d}: (node:{:4d} -- {:4d}), [A:{:12.5f}]'\
      .format(i_mn, mem_nn[i_mn, 0], mem_nn[i_mn, 1], mem_are[i_mn]))
  print('(5) 境界条件')
  print(' 境界条件を与える節点の数：{}'.format(bc_tn))
  print('   [境界条件：fix:0, free:1]')
  for i_bcn in range(bc_tn):
    print('{:4d}: (node:{:4d}), [{:2d}, {:2d}]'\
      .format(i_bcn, bc_nn[i_bcn], bc_ff[i_bcn, 0], bc_ff[i_bcn, 1]))
  print('(6) 外力')
  print(' 外力が作用する節点の数：{}'.format(nl_tn))
  for i_nln in range(nl_tn):
    print('{:4d}: (node:{:4d}), [x:{:12.5f}, y:{:12.5f}]'\
      .format(i_nln, nl_nn[i_nln], nl_mag[i_nln, 0], nl_mag[i_nln, 1]))


def bc_index(siz_nod, var_max, nod_tn, bc_tn, bc_nn, bc_ff):
  '''計算準備データの生成
  
  bc_index: [Boundary Condition][INDEX]
  
  (1) global変数として結果を受け渡し
  siz_tot:節点の全自由度の総数　(= 節点数(nod_tn) * 自由度(siz_nod))
  siz_fix:全自由度の中で境界条件がfixの部分の数
  siz_fre:全自由度の中で境界条件がfreeの部分の数
  ind_fre[siz_tot → siz_fre]:free部分の通し番号と節点番号の対応表
  ind_fix[siz_fix]:fix部分の通し番号と節点番号の対応表
  '''
  #関数からの出力
  global siz_tot, siz_fre, siz_fix, ind_fre, ind_fix
  
  siz_tot = nod_tn * siz_nod
  if siz_tot >= var_max:
    print('節点数が制限値を超えている')
  siz_fix = 0
  for i_bcn in range(bc_tn):
    for j_dof in range(siz_nod):
      if bc_ff[i_bcn, j_dof] == 0:
        siz_fix += 1
  siz_fre = siz_tot - siz_fix
  ind_fre = np.arange(siz_tot, dtype = 'uint16')
  ind_fix = np.zeros(siz_fix, dtype = 'uint16')
  #fixの部分をvar_maxに入れ替え
  for i_bcn in range(bc_tn):
    ind_1st = bc_nn[i_bcn] * siz_nod
    for j_dof in range(siz_nod):
      if bc_ff[i_bcn, j_dof] == 0:
        ind_fre[ind_1st + j_dof] = var_max
  #fixの部分を除き、前につめる。[0～mm]の範囲が必要なデータ
  tmp_fix = 0
  tmp_fre = 0
  for i in range(siz_tot):
    if ind_fre[i] == var_max:
      ind_fix[tmp_fix] = i
      tmp_fix += 1
    else:
      ind_fre[tmp_fre] = ind_fre[i]
      tmp_fre += 1


def ek_global(i_g_mn, mem_nn, nod_coo, mem_ym, mem_are):
  '''i_g_mn番目の要素の全体座標系要素剛性マトリクスの生成
  
  ek_global: [Element K(stiffness matrix)][GLObal coordinate system]
  
  (1) return
  ek_glo[4, 4]:全体座標系要素剛性マトリクス, dtype = 'float64'
  '''
  ek_loc = np.zeros((2, 2), dtype = 'float64')
  mem_len, mem_rot = mem_len_rotation(i_g_mn, mem_nn, nod_coo)
  #部材座標系要素剛性マトリクス
  ek_loc[0, 0] = mem_ym * mem_are[i_g_mn] / mem_len
  ek_loc[0, 1] = - ek_loc[0, 0]
  ek_loc[1, 0] = ek_loc[0, 1]
  ek_loc[1, 1] = ek_loc[0, 0]
  #全体座標系要素剛性マトリクス
  ek_glo = np.dot(mem_rot.T, np.dot(ek_loc, mem_rot))
  return ek_glo


def mem_len_rotation(i_g_mn, mem_nn, nod_coo):
  '''mem_nn[i_g_mn]の部材長、[[c,s,0,0],[0,0,c,s]]の算定
  
  mem_len_rotation: [member][LENgth][ROTATION]
  
  (1) return
  mem_len: 部材長
  mem_rot: 部材座標系に変換する回転行列
       [[cos(θ), sin(θ), 0, 0],
       [0, 0, cos(θ), sin(θ)]]
      （全体座標系における部材の傾きをθとする。反時計回りを正。）
  '''
  mem_rot = np.zeros((2,4), dtype ='float64') 
  mem_0_nn = mem_nn[i_g_mn, 0]
  mem_1_nn = mem_nn[i_g_mn, 1]
  dif_x = nod_coo[mem_1_nn, 0] - nod_coo[mem_0_nn, 0]
  dif_y = nod_coo[mem_1_nn, 1] - nod_coo[mem_0_nn, 1]
  mem_len = np.sqrt(dif_x * dif_x + dif_y * dif_y)
  c = dif_x / mem_len
  s = dif_y / mem_len
  #mem_rot[:] = [[c, s, 0., 0.], [0., 0., c, s]]
  mem_rot[0, 0] = c
  mem_rot[1, 2] = c
  mem_rot[0, 1] = s
  mem_rot[1, 3] = s
  return mem_len, mem_rot


def set_gk(i_g_mn, siz_nod, ek_glo, mem_nn, gk):
  '''要素剛性マトリクスから全体剛性マトリクスの生成（fix,freeを含む）
  
  set_gk: [SET][Globally assembled K(stiffness matrix)]
  
  (1) return
  gk[siz_tot, siz_tot]:全体剛性マトリクス（fix,freeを含む）
  '''
  be_tn = 2 
  for i_ben in range(be_tn):
    gk_row_1st = mem_nn[i_g_mn, i_ben] * siz_nod
    ek_row_1st = i_ben * siz_nod
    for j_ben in range(be_tn):
      gk_col_1st = siz_nod * mem_nn[i_g_mn, j_ben]
      ek_col_1st = siz_nod * j_ben
      for k_dof in range(siz_nod):
        gk_row = gk_row_1st + k_dof
        ek_row = ek_row_1st + k_dof
        for l_dof in range(siz_nod):
          gk_col = gk_col_1st + l_dof
          ek_col = ek_col_1st + l_dof
          gk[gk_row, gk_col] += ek_glo[ek_row, ek_col]
  return gk


def set_nodal_load(siz_nod, siz_tot, nl_tn, nl_nn, nl_mag):
  '''荷重ベクトルの生成（fix,freeを含む）

  set_nodal_load: [SET][NODAL][LOAD]
  
  (1) return
  nl_v[siz_tot]:荷重ベクトル
  '''
  nl_v = np.zeros(siz_tot, dtype = 'float64')
  if nl_tn != 0:
    for i_nln in range(nl_tn):
      nl_1st = siz_nod * nl_nn[i_nln]
      for j_dof in range(siz_nod):
        nl_v[nl_1st + j_dof] += nl_mag[i_nln, j_dof]
  return nl_v


def gk_reaction_force(siz_tot, siz_fre, siz_fix, ind_fre, ind_fix, gk):
  '''反力算定用の剛性マトリクス
  
  gk_reaction_force:[Globally assembled K(stiffness matrix)][REACTION][FORCE]
  全体剛性マトリクスgkのfix行、free列を抜き出し、gk_rfに代入（反力算定用）
    
  (1) return
  gk_rf[siz_fix, siz_fre]:
    反力計算要に全体剛性マトリクスgk[siz_tot, siz_tot]から必要部分を抜き出す。
    行方向はfix部分のみ抜き出し、列方向はfree部分のみ抜き出す。
  '''
  gk_rf = np.zeros((siz_fix, siz_fre), dtype = 'float64')
  for i_fix in range(siz_fix):
    gk_row_fix = ind_fix[i_fix]
    for j_fre in range(siz_fre):
      gk_col_fre = ind_fre[j_fre]
      gk_rf[i_fix, j_fre] = gk[gk_row_fix, gk_col_fre]
  return gk_rf


def del_fix_gk_nl(siz_tot, siz_fre, ind_fre ,nl_v, gk):
  '''gk[,], nl_v[]のfix部分を削除し、free部分を前に寄せる。
  
  del_fix_gk_nl: [DELete][FIX][Globally assembled K(stiffness matrix)]
           [Nodal Load]
           
  (1) return
  nl_v[siz_tot → siz_fre]:荷重ベクトル(fix部削除)
  gk[siz_tot → siz_fre, siz_tot → siz_fre]:
    　　　　全体座標系剛性マトリクス(fix部削除)
  ※注意：fix部分のデータを削除しfree部分のデータを前に寄せているので、
    siz_freより後のデータは不要。
    行列計算をするときはスライスでfree部分のみを抜き出す必要あり。
  '''
  for i_fre in range(siz_fre):
    gk_row_fre = ind_fre[i_fre]
    nl_v[i_fre] = nl_v[gk_row_fre]
    for j_fre in range(siz_fre):
      gk_col_fre = ind_fre[j_fre]
      gk[i_fre, j_fre] = gk[gk_row_fre, gk_col_fre]
  return nl_v, gk


def node_displacement(siz_nod, nod_tn, siz_fre, ind_fre, dis_fre_v):
  '''ｘ, y方向変位から変位の大きさを求め、nod_dis[i, j]に代入
  
  node_displacement: [NODE][DISPLACEMENT]
  
  (1) return
  nod_dis[i, j]:i節点のj方向の変位
  　　　　(j=0:x方向変位, j=1:y方向変位, j=2:x,y方向を考慮した変位の大きさ)
  '''
  nod_dis = np.zeros((nod_tn, siz_nod + 1), dtype = 'float64')
  for i_fre in range(siz_fre):
    nn = ind_fre[i_fre] // siz_nod
    dof = ind_fre[i_fre] % siz_nod
    nod_dis[nn, dof] = dis_fre_v[i_fre]
  for i_nn in range(nod_tn):
    nod_dis_x = nod_dis[i_nn, 0]
    nod_dis_y = nod_dis[i_nn, 1]
    nod_dis[i_nn, 2] = np.sqrt(nod_dis_x ** 2 + nod_dis_y ** 2)
  return nod_dis


def stress(mem_tn, siz_nod, mem_nn, nod_coo, nod_dis):
  '''部材軸力の計算
  
  stress: [STRESS]
  
  (1) return
  mem_str[i]:i部材の軸力（引張側正）
  '''
  be_tn = 2
  dis_be_glo = np.zeros(be_tn * siz_nod, dtype = 'float64')
  mem_str = np.zeros(mem_tn, dtype = 'float64')
  for i_mn in range(mem_tn):
    mem_len, mem_rot = mem_len_rotation(i_mn, mem_nn, nod_coo)
    for j_ben in range(be_tn):
      dis_1st = j_ben * siz_nod
      for k_dof in range(siz_nod):
        dis_be_glo[dis_1st + k_dof] = nod_dis[mem_nn[i_mn, j_ben], k_dof]
    dis_be_loc = np.dot(mem_rot, dis_be_glo)
    mem_str[i_mn] = mem_ym * mem_are[i_mn] / mem_len \
      * (dis_be_loc[1] - dis_be_loc[0]) 
  return mem_str


def reaction_force(siz_fix, ind_fix, gk_rf, dis_fre_v):
  '''反力の計算
  
  reaction_force: [REACTION][FORCE]
  
  (1) return
  rf_v[siz_fix]:反力 (i:反力の通し番号)
  rf_nn[siz_fix]:反力の通し番号に対応する反力の節点番号 ※リスト
  rf_der[siz_fix]:反力の通し番号に対応する反力の方向の記号 ※リスト
  '''
  rf_v = np.dot(gk_rf, dis_fre_v)
  rf_nn = []
  rf_der = []
  for i_fix in range(siz_fix):
    rf_nn.append(ind_fix[i_fix] // siz_nod)
    if (ind_fix[i_fix] % siz_nod) == 0:
      rf_der.append('x')
    else:
      rf_der.append('y')
  return rf_v, rf_nn, rf_der


def output_result():
  '''計算結果の出力
  
  output_result: [OUTPUT][RESULT]
  誤差確認のためfloatの出力は書式指定せず
  '''
  #関数への入力
  global nod_tn, nod_dis, mem_tn, mem_nn, mem_str
  global siz_fix, rf_nn, rf_der, rf_v
  
  print('========================================')
  print('２．計算結果')
  print('(1) 節点変位')
  print(' 節点: 変位[x:], [y:], [xy:]')
  for i_nn in range(nod_tn):
    print('{:4d}: [x:{}], [y:{}], [xy:{}]'\
      .format(i_nn, nod_dis[i_nn, 0], nod_dis[i_nn, 1], nod_dis[i_nn, 2]))
  print('(2) 部材応力')
  print(' 部材: (節点0--節点1), 部材応力')
  for i_mn in range(mem_tn):
    print('{:4d}: ({:4d} -- {:4d}), {}'\
      .format(i_mn, mem_nn[i_mn, 0], mem_nn[i_mn, 1], mem_str[i_mn]))
  print('(3) 反力')
  print(' No.:(節点-方向), 反力')
  for i_fix in range(siz_fix):
    print('{:4d}:({:4d} - {}), {}'\
      .format(i_fix, rf_nn[i_fix], rf_der[i_fix], rf_v[i_fix]))
    
  # csv に節点変位の結果出力
  with open(result_displacement_csv_path, 'w', newline="") as f1:
    writer = csv.writer(f1)
    writer.writerow(['x座標', 'y座標', 'x方向変位', 'y方向変位', '総変位量'])
    for i in range(nod_tn):
      writer.writerow([nod_coo[i, 0], nod_coo[i, 1], nod_dis[i, 0], nod_dis[i, 1], nod_dis[i, 2]])
  # csv に部材応力の結果出力．ただし，座標値は部材の中点とする．
  with open(result_stress_csv_path, 'w', newline="") as f2:
    writer = csv.writer(f2)
    writer.writerow(['x座標', 'y座標', '部材応力'])
    for i in range(mem_tn):
      target_coo = (nod_coo[mem_nn[i, 0]] + nod_coo[mem_nn[i, 1]]) / 2
      writer.writerow([target_coo[0], target_coo[1], mem_str[i]])


if __name__ == '__main__':
  '''メインプログラム

  (1) 入力、計算準備データ
   input_data(), bc_index() にて
  (2) 主な計算用データ
  gk[siz_tot→nfree]:全体剛性マトリクス  ※def del_fix_gk_nl()前後で使用範囲が変わる
  nl_v[siz_tot→nfree]:荷重ベクトル  ※def del_fix_gk_nl()前後で使用範囲が変わる
  ek_loc[2, 2]:要素剛性マトリクス（部材座標系）
  ek_glo[4, 4]:要素剛性マトリクス（全体座標系）
  gk_rf[siz_fix, siz_fre]:反力算定用の剛性マトリクス（fix行、free列の抜き出し）
  dis_fre_v[siz_fre]:節点変位のベクトル（freeの部分）
  (3) 計算結果
  nod_dis[nod_tn]:節点変位のベクトル（free,fix）
  mem_str[mem_tn]:部材応力
  rf_v[siz_fix]:反力
  rf_nn_der[siz_fix][i]:反力の節点番号と方向(i=0:節点番号、i=1:方向) ※リスト
  '''
  #データ入力
  input_data_3()
  #入力データの出力
  output_data()
  #計算準備データ
  #global siz_tot, siz_fre, siz_fix, ind_fre, ind_fix
  bc_index(siz_nod, var_max, nod_tn, bc_tn, bc_nn, bc_ff)
  gk = np.zeros((siz_tot, siz_tot), dtype = 'float64')
  for i_g_mn in range(mem_tn):
    #全体座標系要素剛性マトリクスの生成
    ek_glo = ek_global(i_g_mn, mem_nn, nod_coo, mem_ym, mem_are)
    #要素剛性マトリクスより全体剛性マトリクスを生成
    gk = set_gk(i_g_mn, siz_nod, ek_glo, mem_nn, gk)
  #荷重ベクトルの生成
  nl_v = set_nodal_load(siz_nod, siz_tot, nl_tn, nl_nn, nl_mag)
  #fixされているところの剛性マトリクスのみを抽出（反力算定用）
  gk_rf = gk_reaction_force(siz_tot, siz_fre, siz_fix, ind_fre, ind_fix, gk)
  #gk[,], nl_v[]のfix部分を削除し、free部分を前に寄せる。
  nl_v, gk = del_fix_gk_nl(siz_tot, siz_fre, ind_fre, nl_v, gk)
  #連立一次方程式を解き、拘束条件のない部分の変位を求める
  dis_fre_v = np.linalg.solve(gk[0:siz_fre, 0:siz_fre], nl_v[0:siz_fre])
  #節点変位の計算
  nod_dis = node_displacement(siz_nod, nod_tn, siz_fre, ind_fre, dis_fre_v)
  #部材応力の計算
  mem_str = stress(mem_tn, siz_nod, mem_nn, nod_coo, nod_dis)
  #反力の計算
  rf_v, rf_nn, rf_der =reaction_force(siz_fix, ind_fix, gk_rf, dis_fre_v)
  #計算結果の出力
  output_result()


