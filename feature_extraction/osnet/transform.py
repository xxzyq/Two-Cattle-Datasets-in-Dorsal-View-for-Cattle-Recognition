from torchvision import transforms
from PIL import ImageDraw, Image, ImageEnhance, ImageFilter
import random
import torch


class AddLabelImageTransform:
    def __init__(self, label_image_path, probability=0.5, position='topleft', scale_range=(0.2, 0.5), rotation_range=(-30, 30)):
        self.label_image = Image.open(label_image_path)
        self.probability = probability
        self.position = position
        self.scale_range = scale_range
        self.rotation_range = rotation_range  # 新增旋转角度范围

    def __call__(self, img):
        if random.random() < self.probability:
            # 随机选择一个缩放比例
            scale_factor = random.uniform(*self.scale_range)
            # 缩放标签图像
            new_size = (int(self.label_image.width * scale_factor), int(self.label_image.height * scale_factor))
            # scaled_label_image = self.label_image.resize(new_size, Image.ANTIALIAS)
            scaled_label_image = self.label_image.resize(new_size, Image.Resampling.LANCZOS)
            
            # 随机选择一个旋转角度
            rotation_angle = random.uniform(*self.rotation_range)
            # 旋转图像
            rotated_label_image = scaled_label_image.rotate(rotation_angle, expand=True)
            # 根据position确定叠加的具体位置
            base_position = (0, 0)
            if self.position == 'top1':
                base_position = (0, 0)
            elif self.position == 'top2':
                base_position = (0, int(img.height / 6))
            # 计算水平移动的限制
            left_limit = min(0, img.width - rotated_label_image.width)  # 考虑旋转后的宽度
            right_limit = max(0, img.width - rotated_label_image.width)
            # 避免范围错误
            if left_limit > right_limit:
                left_limit, right_limit = right_limit, left_limit
            horizontal_shift = random.randint(left_limit, right_limit)
            position = (base_position[0] + horizontal_shift, base_position[1])
            # 创建一个同样大小的图像并叠加标签图像
            img_with_label = img.copy()
            img_with_label.paste(rotated_label_image, position, rotated_label_image)
            return img_with_label
        return img
    
# def add_half_ellipse_to_image(image_path, output_path):
#     # 打开原始图像
#     img = Image.open(image_path)
#     width, height = img.size
#     # 创建一个同样大小的透明图层
#     ellipse_layer = Image.new('RGBA', img.size, (255, 255, 255, 0))
#     # 椭圆尺寸的随机设置
#     ellipse_height = random.randint(int(height * 0.7), int(height * 1.5))  # 长边
#     ellipse_width = random.randint(int(width * 0.3), int(width * 0.7))    # 短边
#     # 椭圆圆心的位置决定
#     side = random.choice(['left', 'right'])
#     if side == 'left':
#         x0 = -ellipse_width // 2  # 使圆心位于左侧边缘
#     else:
#         x0 = width - ellipse_width // 2  # 使圆心位于右侧边缘
#     # 圆心在垂直方向上的浮动范围
#     vertical_offset = random.randint(-height // 5, height // 5)  # 圆心可以在边缘上下10%的范围内浮动
#     y0 = (height - ellipse_height) // 2 + vertical_offset
#     # 随机灰度和透明度
#     gray_level = random.randint(0, 100)  # 随机灰色级别
#     alpha = random.randint(50, 150)      # 随机透明度
#     color = (gray_level, gray_level, gray_level, alpha)
#     # 在透明图层上绘制椭圆
#     draw = ImageDraw.Draw(ellipse_layer)
#     draw.ellipse([x0, y0, x0 + ellipse_width, y0 + ellipse_height], fill=color)
#     # 将透明图层合并到原始图像上
#     combined = Image.alpha_composite(img.convert('RGBA'), ellipse_layer)
#     # # 保存合成后的图像
#     # combined.save(output_path, 'PNG')
#     return combined
class AddHalfEllipseTransform(object):
    """
    在图像上以 0.5 的概率添加一个半椭圆形状的 Transform。
    """
    def __call__(self, img):
        """
        Args:
            img (PIL Image): 待处理的图像。

        Returns:
            PIL Image: 根据概率，可能添加了半椭圆形状的图像。
        """
        if random.random() > 0.5:
            return img  # 50% 的概率不进行任何变换

        width, height = img.size
        # 创建一个同样大小的透明图层
        ellipse_layer = Image.new('RGBA', (width, height), (255, 255, 255, 0))
        # 设置椭圆尺寸
        ellipse_height = random.randint(int(height * 0.7), int(height * 1.5))
        ellipse_width = random.randint(int(width * 0.3), int(width * 0.7))
        # 决定椭圆的水平位置
        side = random.choice(['left', 'right'])
        if side == 'left':
            x0 = -ellipse_width // 2
        else:
            x0 = width - ellipse_width // 2
        # 设置椭圆的垂直位置
        vertical_offset = random.randint(-height // 5, height // 5)
        y0 = (height - ellipse_height) // 2 + vertical_offset
        # 设置椭圆的颜色和透明度
        gray_level = random.randint(0, 100)
        alpha = random.randint(50, 150)
        color = (gray_level, gray_level, gray_level, alpha)
        # 在透明图层上绘制椭圆
        draw = ImageDraw.Draw(ellipse_layer)
        draw.ellipse([x0, y0, x0 + ellipse_width, y0 + ellipse_height], fill=color)
        # 将透明图层合并到原始图像上
        img = Image.alpha_composite(img.convert('RGBA'), ellipse_layer).convert('RGB')
        
        return img
    
class RandomLocalExposure(torch.nn.Module):
    def __init__(self, probability=0.5, min_area_ratio=0.5, max_area_ratio=1.5, brightness_factor_range=(0.5, 2.0)):
        """
        初始化函数。
        参数:
        - probability: 应用变换的概率。
        - min_area_ratio: 最小受影响区域占整个图像的比例（以面积比计算）。
        - max_area_ratio: 最大受影响区域占整个图像的比例。
        - brightness_factor_range: 亮度调整因子的范围，(最小, 最大)。
        """
        super().__init__()
        self.probability = probability
        self.min_area_ratio = min_area_ratio
        self.max_area_ratio = max_area_ratio
        self.brightness_factor_range = brightness_factor_range

    def forward(self, img):
        """
        应用变换的方法。
        参数:
        - img: PIL图像对象。
        返回:
        - 变换后的图像。
        """
        if random.random() < self.probability:
            width, height = img.size
            # 随机确定椭圆大小
            area_ratio = random.uniform(self.min_area_ratio, self.max_area_ratio)
            area_width = int(width * area_ratio ** 0.5)
            area_height = int(height * area_ratio ** 0.5)
            # 确定椭圆中心，允许椭圆中心在图像边缘外
            center_x = random.randint(-area_width // 2, width + area_width // 2)
            center_y = random.randint(-area_height // 2, height + area_height // 2)

            # 创建椭圆形遮罩
            mask = Image.new("L", (width, height), 0)
            draw = ImageDraw.Draw(mask)
            # 绘制椭圆
            ellipse_box = [
                center_x - area_width // 2, center_y - area_height // 2,
                center_x + area_width // 2, center_y + area_height // 2
            ]
            draw.ellipse(ellipse_box, fill=255)
            mask = mask.filter(ImageFilter.GaussianBlur(radius=20))  # 对遮罩进行高斯模糊

            # 调整图像的亮度
            enhancer = ImageEnhance.Brightness(img)
            brightness_factor = random.uniform(*self.brightness_factor_range)
            img = enhancer.enhance(brightness_factor)

            # 应用遮罩以恢复边缘的原图
            img.putalpha(mask)
            original_img = img.copy().convert("RGBA")
            img = Image.composite(img, original_img, mask)
            img = img.convert("RGB")

        return img
    
# 数据处理
transform_val = transforms.Compose([
    transforms.Resize((300, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.41924568, 0.41247257, 0.41211343], std=[0.27484905, 0.28503192, 0.2850544])
])

transform_train = transforms.Compose([
    transforms.Resize((300, 128)),
    AddHalfEllipseTransform(),  # 在尺寸调整和标准化之前添加椭圆
    AddLabelImageTransform(label_image_path='handrail.png', probability=0.5, position='top2'),
    transforms.ColorJitter(brightness=(0.8, 1.2), contrast=(0.8, 1.3), saturation=(0.8, 1.3), hue=(-0.2, 0.2)),
    RandomLocalExposure(probability=0.5, min_area_ratio=0.5, max_area_ratio=1.5, brightness_factor_range=(0.6, 1.5)),
    AddLabelImageTransform(label_image_path='label.png', probability=0.5, position='top1'),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.49052152,0.45171499,0.43817976], std=[0.29596688,0.29971192,0.30096675]),
])
