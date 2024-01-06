### 散布図をグラデーションにする
import matplotlib.pyplot as plt
# import matplotlib.cm as cm
import numpy as np

input_path1 = 'Output/4_trass/trass4/trass4_stress.csv'
input_path2 = 'Output/4_trass/trass4/trass4_displacement.csv'
save_fig_path1 = 'Output/5_plt/plt4_scat.png' # 図の保存  next_pdsの結果はreを付け足す
save_fig_path2 = 'Output/5_plt/plt4_hist.png' # 図の保存  next_pdsの結果はreを付け足す
save_fig_path3 = 'Output/5_plt/plt4_displacement.png' # 図の保存  next_pdsの結果はreを付け足す


# ファイルの読み込み
input_list1 = np.loadtxt(input_path1, delimiter=',', skiprows=1) # 1行目はラベルのためスキップ
data1_x = input_list1[:, 0]
data1_y = input_list1[:, 1]
stress_list = input_list1[:, 2]
print('生データ:   ', len(stress_list))
print('生データ:   ', stress_list)
print('最大応力:   ', np.amax(stress_list), '    最小応力:   ', np.amin(stress_list))
# stress_list = [abs(i) for i in stress_list] # 絶対値に変換
# print('absデータ:   ', len(stress_list))

# ファイルの読み込み
input_list2 = np.loadtxt(input_path2, delimiter=',', skiprows=1) # 1行目はラベルのためスキップ
data2_x = input_list2[:, 0]
data2_y = input_list2[:, 1]
displacement_list = input_list2[:, 4]
print('最大変位:   ', np.amax(displacement_list), '    最小変位:   ', np.amin(displacement_list))



# # サブプロットを作成 (1行2列の2つのサブプロット)
# fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))

# # 1つ目のサブプロットに散布図1をプロット
# axes[0].scatter(data1_x, data1_y, c=stress_list, cmap='Reds')
# axes[0].set_title('Scatter Plot 1')
# axes[0].set_xlabel('X軸',fontname="Meiryo")  #「fontname="Meiryo"」は日本語表示用
# axes[0].set_ylabel('Y軸',fontname="Meiryo")
# axes[0].legend()
# axes[0].grid(True)

# # 2つ目のサブプロットに散布図2をプロット
# axes[1].scatter(data2_x, data2_y, c=displacement_list, cmap='Reds')
# axes[1].set_title('Scatter Plot 2')
# axes[1].set_xlabel('X軸',fontname="Meiryo")
# axes[1].set_ylabel('Y軸',fontname="Meiryo")
# axes[1].legend()
# axes[1].grid(True)

# # タイトルを設定
# plt.suptitle('2つの散布図',fontname="Meiryo")

# # レイアウトを調整
# plt.tight_layout()

# # 図の保存
# fig.savefig(save_fig_path1)

# # 図を表示
# plt.show()


 
# # 応力ヒストグラムの描画
# fig = plt.figure(2)
# plt.xlim(-200, 200) # x軸の表示範囲
# plt.ylim(0,400) # y軸の表示範囲
# plt.xlabel('Stress', fontsize=20)
# plt.ylabel('Frequency', fontsize=20)
# plt.hist(stress_list, range=(-200,200), histtype="bar",edgecolor="blue",bins=20)  #(2)ヒストグラムの描画
# plt.show()
# # 図の保存
# fig.savefig(save_fig_path2)

# 変位ヒストグラムの描画
fig = plt.figure(2)
# plt.xlim(-20, 20) # x軸の表示範囲
# plt.ylim(0,400) # y軸の表示範囲
plt.xlabel('Displacement', fontsize=20)
plt.ylabel('Frequency', fontsize=20)
plt.hist(displacement_list, histtype="bar",edgecolor="blue")  #(2)ヒストグラムの描画
plt.show()
# 図の保存
fig.savefig(save_fig_path3)
