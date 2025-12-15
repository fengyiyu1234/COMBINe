import napari
import tifffile
import os

# ----------------------------------------------------
# --- 你的文件路径 --- (请修改这部分)
# ----------------------------------------------------

# 1. 你的 annotation_file 文件路径 (它是一个标签图)
#    (这是那个被你的脚本裁剪和旋转过的 atlas)
atlas_file = '/data/hdd12tb-1/fengyi/COMBINe/clearmap/fw2/orientation03012_2/ABA_25um_annotation__-3_-1_2__slice_None_None_None__slice_None_None_None__slice_None_None_None__.tif'

# 2. 你的 reference_file 文件路径 (可选的, 灰度背景)
#    (这是灰度的参考大脑，视觉效果比 annotation_file 好)
reference_file = '/home/fyu7/COMBINe/ClearMap_ryan/ClearMap/Resources/Atlas/ABA_25um_reference.tif' # 如果你没有，就忽略它

# 3. 你的 cell_registration 文件夹的根路径
#    (它应该包含 '0', '1', '2' ... '5' 等子文件夹)
base_density_path = '/data/hdd12tb-1/fengyi/COMBINe/clearmap/fw2/orientation03012_2/cell_registration' 


viewer = napari.Viewer(ndisplay=3)

# 2. 添加图谱背景 (二选一)

# 选项 A: 添加 Annotation 标签图 (推荐，因为有脑区 ID)
# 我们使用 `add_labels`，因为它是一个标签图 (每个 ID 一种颜色)
if os.path.exists(atlas_file):
    atlas_data = tifffile.imread(atlas_file)
    atlas_data = atlas_data.astype(int)
    viewer.add_labels(atlas_data, name='Annotation (Atlas)')
    print(f"成功加载图谱: {atlas_file}")
else:
    print(f"警告: 未找到图谱文件: {atlas_file}")

# 选项 B: 添加 Reference 灰度图 (可选, 视觉背景)
# 如果你两个都想加，也可以。
# if os.path.exists(reference_file):
#     ref_data = tifffile.imread(reference_file)
#     viewer.add_image(
#         ref_data,
#         name='Reference (Gray)',
#         colormap='gray', # 灰度
#         blending='translucent', # 半透明
#         contrast_limits=[ref_data.min(), ref_data.max()]
#     )


# 3. 循环添加 6 个细胞密度图 (0-5)
density_files = [
    os.path.join(base_density_path, str(i), 'density_counts.tif') for i in range(6) # 自动创建 0-5 的路径
]

# 为每个通道定义颜色
colors = ['red', 'green', 'blue', 'magenta', 'cyan', 'yellow']

for i, file_path in enumerate(density_files):
    if os.path.exists(file_path):
        density_data = tifffile.imread(file_path)
        
        # 使用 `add_image` 将密度图作为图像层添加
        viewer.add_image(
            density_data,
            name=f'Class {i}',  # 标记为 Class 0, Class 1, ...
            colormap=colors[i],  # 分配颜色
            blending='additive', # <-- 这是关键！让颜色叠加，黑色=透明
            contrast_limits=[density_data.min(), density_data.max()] # 自动调整对比度
        )
        print(f"成功加载密度图: {file_path}")
    else:
        print(f"警告: 未找到密度图: {file_path}")


# 4. 启动 Napari
print("启动 Napari 查看器...")
napari.run()