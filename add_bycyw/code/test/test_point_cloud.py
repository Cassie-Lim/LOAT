'''
可视化点云图

此代码由官方示例和问chatGPT生成

官方示例： http://www.open3d.org/docs/release/tutorial/visualization/visualization.html
其中pcd的加载问chatGPT，其中需要注意：pcd.points = o3d.utility.Vector3dVector(point_cloud)中point_cloud的格式为（point_num,3），和原本存储格式不一致（150，150，3）,因此需要reshape一下

如果出现warning:** DISPLAY**问题，在vscode里面设置环境变量如： export DISPLAY=:10.0即可

prompter环境已经改变，没有open3d，如果需要可视化，用open3d环境
'''
import pickle
import open3d as o3d
import numpy as np

with open("debug_data/point_cloud.pkl",'rb') as f:
    point_cloud_data = pickle.load(f)

point_cloud = point_cloud_data[0].detach().cpu().numpy()
point_cloud = point_cloud.reshape(-1,3)
# pcd = o3d.io.read_point_cloud("./PointClouds/0120.pcd") # 读取pcd文件

# print(pcd) #只是简单的打印信息：PointCloud with 113662 points.

# #显示，zoom等信息是一些可选项
# o3d.visualization.draw_geometries([pcd])
# # o3d.visualization.draw_geometries([pcd], zoom=0.3412,
# #                                   front=[0.4257, -0.2125, -0.8795],
# #                                   lookat=[2.6172, 2.0475, 1.532],
# #                                   up=[-0.0694, -0.9768, 0.2024])

# # 在同级目录下写入 copy_of_fragment.pcd文件
# o3d.io.write_point_cloud("copy_of_fragment.pcd", pcd)

# 创建一个简单的点云数据
# point_cloud = np.array([[1.0, 2.0, 3.0],
#                        [4.0, 5.0, 6.0],
#                        [7.0, 8.0, 9.0]])

# 创建 Open3D 中的点云对象
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(point_cloud)

o3d.visualization.draw_geometries([pcd],
                                  zoom=0.3412,
                                  front=[0.4257, -0.2125, -0.8795],
                                  lookat=[2.6172, 2.0475, 1.532],
                                  up=[-0.0694, -0.9768, 0.2024])

# 创建一个可视化窗口并添加点云数据
# vis = o3d.visualization.Visualizer()
# vis.create_window()
# vis.add_geometry(pcd)

# # # 设置相机视角
# # vis.get_render_option().point_size = 10
# vis.get_view_control().rotate(500.0, 250.0)

# # 截取渲染窗口中的图像
# image = vis.capture_screen_float_buffer()

# # 将图像保存为文件
# o3d.io.write_image("point_cloud.png", image)

# # 销毁可视化窗口
# vis.destroy_window()


# print("Load a ply point cloud, print it, and render it")
# sample_ply_data = o3d.data.PLYPointCloud()
# pcd = o3d.io.read_point_cloud(sample_ply_data.path)
# o3d.visualization.draw_geometries([pcd],
#                                   zoom=0.3412,
#                                   front=[0.4257, -0.2125, -0.8795],
#                                   lookat=[2.6172, 2.0475, 1.532],
#                                   up=[-0.0694, -0.9768, 0.2024])

print("over")