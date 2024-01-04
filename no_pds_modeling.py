import time
import csv

# define
INTERVAL = 5 #点群間隔
# 点の生成範囲の設定
X_MAX = 50
X_MIN = 0
Y_MAX = 50
Y_MIN = 0

# Outputファイル
result_ply_path = 'Output/0_no_pds/pds.ply'
result_csv_path = 'Output/0_no_pds/pds.csv'

def main():
    fixed_points = [] # 確定点格納用

    for y in range(Y_MIN, Y_MAX+1, INTERVAL):
        for x in range(X_MIN, X_MAX+1, INTERVAL):
            fixed_points.append([x, y, 0])
    
    print('fixed_points:   ', fixed_points)
    print('len(fixed_points):    ', len(fixed_points))

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
            f.write(str(ele[0])+' '+str(ele[1])+' '+str(ele[2])+ '\n')

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