import h5py
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap

# 遍历每个文件
for i in range(1, 37):
    # 生成文件名
    file = f'/public/home/pengfeiliu/Tide/snapshots/snapshot_s{i}.h5'

    # 打开h5文件
    with h5py.File(file, 'r') as f:
        height = f['tasks']['height'][:]  # 读取'height'数据集

    # 创建一个Basemap实例
    m = Basemap(projection='ortho', lat_0=0, lon_0=0)

    # 创建一个表示经度和纬度的网格
    lon = np.linspace(-180, 180, height.shape[2])
    lat = np.linspace(-90, 90, height.shape[1])
    Lon, Lat = np.meshgrid(lon, lat)

    # 将经度和纬度转换为地图投影坐标
    x, y = m(Lon, Lat)

    # 遍历每个时间步长
    for j in range(len(height)):
        # 创建一个新的figure
        fig = plt.figure()

        # 绘制数据
        contour = m.contourf(x, y, height[j], cmap='RdYlBu')

        # 保存图像
        plt.savefig(f'snapshot_s{i}_height_{j}.png')

        # 关闭figure，释放内存
        plt.close(fig)
