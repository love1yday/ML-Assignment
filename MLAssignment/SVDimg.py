import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import datetime

# 个人信息
student_id = "2010040116"
name = "zcg" 

# 获取系统时间
current_time = datetime.datetime.now()

# 读取图片
img = Image.open("2.jpg")
img_array = np.array(img)
print('origin_image shape:', img_array.shape)  # 打印图片形状

k = 5
svd_image = []

# 对每个通道进行奇异值分解和压缩
for ch in range(3):
    im_ch = img_array[:, :, ch]
    
    # 进行奇异值分解
    U, S, Vt = np.linalg.svd(im_ch, full_matrices=False)
    
    # 保留前 k 个奇异值对应的分量
    compressed_U = U[:, :k]
    compressed_S = np.diag(S[:k])
    compressed_Vt = Vt[:k, :]
    
    # 重构近似图像
    compressed_image = np.dot(compressed_U, np.dot(compressed_S, compressed_Vt))
    
    # 转为8位整数
    compressed_image = np.uint8(compressed_image)
    
    # 将每个通道的压缩图像添加到列表
    svd_image.append(compressed_image.astype('uint8'))

# 将压缩后的图像重新组合
img = np.stack((svd_image[0], svd_image[1], svd_image[2]), 2)

# 显示压缩后的图像
plt.imshow(img, cmap='gray')
plt.title(f'Compressed Image (k={k})')
plt.axis('off')

# 输出个人信息和系统时间
print(f"Student ID: {student_id}")
print(f"Name: {name}")
print(f"Current Time: {current_time}")

# 保存压缩后的图像
plt.show()
Image.fromarray(img).save(f"compressed_image2_k{k}.png")
