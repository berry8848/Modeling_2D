### 散布図をグラデーションにする
import matplotlib.pyplot as plt
# import matplotlib.cm as cm
import numpy as np

input_path1 = 'Output/4_trass/trass2/trass2_stress.csv'
input_path2 = 'Output/4_trass/trass2/trass2_displacement.csv'
save_fig_path = 'Output/5_plt/plt.png' # 図の保存

# ファイルの読み込み
input_list1 = np.loadtxt(input_path1, delimiter=',', skiprows=1) # 1行目はラベルのためスキップ
data1_x = input_list1[:, 0]
data1_y = input_list1[:, 1]
stress_list = input_list1[:, 2]
stress_list = [abs(i) for i in stress_list] # 絶対値に変換

# ファイルの読み込み
input_list2 = np.loadtxt(input_path2, delimiter=',', skiprows=1) # 1行目はラベルのためスキップ
data2_x = input_list2[:, 0]
data2_y = input_list2[:, 1]
displacement_list = input_list2[:, 4]

# # 乱数を生成
# x = np.random.rand(100)
# y = np.random.rand(100)
# value = np.random.rand(100)
# サブプロットを作成 (1行2列の2つのサブプロット)
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))

# 1つ目のサブプロットに散布図1をプロット
axes[0].scatter(data1_x, data1_y, c=stress_list, cmap='Blues')
axes[0].set_title('Scatter Plot 1')
axes[0].set_xlabel('X軸',fontname="Meiryo")  #「fontname="Meiryo"」は日本語表示用
axes[0].set_ylabel('Y軸',fontname="Meiryo")
axes[0].legend()
axes[0].grid(True)

# 2つ目のサブプロットに散布図2をプロット
axes[1].scatter(data2_x, data2_y, c=displacement_list, cmap='Blues')
axes[1].set_title('Scatter Plot 2')
axes[1].set_xlabel('X軸',fontname="Meiryo")
axes[1].set_ylabel('Y軸',fontname="Meiryo")
axes[1].legend()
axes[1].grid(True)

# タイトルを設定
plt.suptitle('2つの散布図',fontname="Meiryo")

# レイアウトを調整
plt.tight_layout()

# 図の保存
plt.savefig(save_fig_path)

# 図を表示
plt.show()

# fig = plt.figure(figsize = (10,10), facecolor='lightblue')
# plt.xlabel('X')
# plt.ylabel('Y')

# # 散布図を表示
# plt.scatter(x_list, y_list, s=50, c=stress_list, cmap='Blues')
 
# # カラーバーを表示
# plt.colorbar(ticks=np.arange(0, 1, 0.1))

# plt.show()
