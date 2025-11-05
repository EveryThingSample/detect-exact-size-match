#!/usr/bin/env python3

import os
import sys
from io import BytesIO
import requests
from PIL import Image

import cv2
import numpy as np


def peel_gray_border(input, output_path=None, tolerance=5, max_layers=10):
    if isinstance(input, str):
        if input.startswith("http://") or input.startswith("https://"):
            # 网络地址
            response = requests.get(input)
            image = Image.open(BytesIO(response.content)).convert("RGBA")
        elif os.path.exists(input):
            # 本地路径
            image = Image.open(input).convert("RGBA")
        else:
            raise ValueError(f"❌ 无法识别路径或 URL: {input}")
    elif isinstance(input, BytesIO):
        image = Image.open(input).convert("RGBA")
    else:
        raise TypeError("❌ 输入必须是字符串路径、URL 或 BytesIO 对象")

    # 加载图像并转换为 RGBA
    pil_img = image  #Image.open(input_path).convert("RGBA")
    rgba = np.array(pil_img)
    h, w = rgba.shape[:2]

    # 提取 alpha 通道作为图形掩码
    alpha = rgba[:, :, 3]
    mask = alpha > 0

    # 创建灰色掩码：r ≈ g ≈ b 且 alpha > 0
    r, g, b = rgba[:, :, 0], rgba[:, :, 1], rgba[:, :, 2]
    gray_mask = (
        (np.abs(r - g) <= tolerance) &
        (np.abs(r - b) <= tolerance) &
        (np.abs(g - b) <= tolerance) &
        (alpha > 0)
    )

    # 初始图形区域
    current_mask = mask.copy()

    for layer in range(max_layers):
        # 查找轮廓
        contours, _ = cv2.findContours(current_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            break

        # 创建一层边缘掩码
        edge_mask = np.zeros_like(current_mask, dtype=bool)
        for contour in contours:
            for point in contour:
                x, y = point[0]
                edge_mask[y, x] = True

        # 找出边缘中是灰色的像素
        peel_mask = edge_mask & gray_mask

        # 如果没有灰色边缘了，停止剥离
        if not np.any(peel_mask):
            break

        # 从图形中剥离灰色边缘
        current_mask[peel_mask] = False

    # 获取最终图形区域的边界框
    coords = np.argwhere(current_mask)
    if coords.size == 0:
        print("⚠️ 剥离后没有剩余图形，可能整张图都是灰色边缘")
        return None


    rgba[~current_mask] = [0, 0, 0, 0]  # 剥离区域标红
    if output_path:
        Image.fromarray(rgba).save(output_path)
        print(f"🔍 已保存图像到: {output_path}")
    return rgba



def darken_template(template_rgba, factor=0.7, save_path=None):
    # 复制模板
    template_dark = template_rgba.copy()

    # 提取 alpha 通道作为图形区域掩码
    alpha = template_rgba[:, :, 3]
    mask = alpha > 0

    # 仅暗化图形区域的 RGB 通道
    for c in range(3):  # R, G, B
        channel = template_dark[:, :, c]
        channel[mask] = (channel[mask] * factor).astype(np.uint8)

    # 保存暗化后的模板图
    if save_path:
        cv2.imwrite(save_path, cv2.cvtColor(template_dark, cv2.COLOR_RGBA2BGRA))
        print(f"🖼️ 已保存暗化模板图到: {save_path}")

    return template_dark


def fast_match_rgb_with_alpha(template_rgba, image_path, output_path=None, threshold=0.8, weights=(0.3, 0.59, 0.11)):
    # 1. 提取模板的 RGB 通道和 Alpha 掩码
    template_rgb = template_rgba[:, :, :3]
    alpha_mask = (template_rgba[:, :, 3] > 0).astype(np.uint8)  # 0 或 1

    # 2. 分离 R/G/B 通道模板
    template_channels = cv2.split(template_rgb)

    # 3. 读取目标图像并转为 RGB
    image_bgr = load_image(image_path)

    if image_bgr is None:
        print(f"❌ 无法读取图像: {image_path}")
        return
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_channels = cv2.split(image_rgb)

    # 4. 对每个通道进行灰度匹配（使用 Alpha 掩码）
    score_maps = []
    for i in range(3):
        result = cv2.matchTemplate(image_channels[i], template_channels[i], cv2.TM_CCOEFF_NORMED, mask=alpha_mask)
        score_maps.append(result)

    # 5. 加权融合三个通道的匹配得分
    fused_score = (
        weights[0] * score_maps[0] +
        weights[1] * score_maps[1] +
        weights[2] * score_maps[2]
    )

    # 6. 获取最佳匹配位置
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(fused_score)
    h, w = template_rgb.shape[:2]
    top_left = max_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)

    print(f"匹配位置：{top_left} → {bottom_right}")
    print(f"匹配置信度（融合得分）：{max_val:.4f}")

    if max_val < threshold:
        print("❌ 匹配置信度不足")
        return

    # 7. 可视化匹配结果
    if output_path:
        matched = image_bgr.copy()
        cv2.rectangle(matched, top_left, bottom_right, (0, 0, 255), 2)
        cv2.imwrite(output_path, matched)
        print(f"✅ 已保存匹配图像到: {output_path}")
    return [top_left,bottom_right]

def load_image(image):
    if isinstance(image, np.ndarray):
        return image  # 已是图像数组
    elif isinstance(image, str):
        if image.startswith("http://") or image.startswith("https://"):
            # 网络地址
            response = requests.get(image)
            image_array = np.asarray(bytearray(response.content), dtype=np.uint8)
            return cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        else:
            # 本地路径
            return cv2.imread(image, cv2.IMREAD_COLOR)
    else:
        raise ValueError("image 参数必须是 NumPy 图像数组、本地路径或网络 URL")
def fast_match_rgb_with_dynamic_weights(template_rgba, image, output_path=None, threshold=0.8):
    # 1. 提取模板 RGB 和 Alpha 掩码
    template_rgb = template_rgba[:, :, :3]
    alpha_mask = (template_rgba[:, :, 3] > 0).astype(np.uint8)

    # 2. 分离模板通道
    tpl_R, tpl_G, tpl_B = cv2.split(template_rgb)

    # 3. 读取目标图像并转为 RGB


    image_bgr = load_image(image)

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    img_R, img_G, img_B = cv2.split(image_rgb)

    # 4. 对每个通道进行匹配（带掩码）
    result_R = cv2.matchTemplate(img_R, tpl_R, cv2.TM_CCOEFF_NORMED, mask=alpha_mask)
    result_G = cv2.matchTemplate(img_G, tpl_G, cv2.TM_CCOEFF_NORMED, mask=alpha_mask)
    result_B = cv2.matchTemplate(img_B, tpl_B, cv2.TM_CCOEFF_NORMED, mask=alpha_mask)

    # 5. 获取每个通道的最大置信度
    _, max_R, _, _ = cv2.minMaxLoc(result_R)
    _, max_G, _, _ = cv2.minMaxLoc(result_G)
    _, max_B, _, _ = cv2.minMaxLoc(result_B)

    # 6. 归一化为权重
    total = max_R + max_G + max_B
    if total == 0:
        print("❌ 所有通道匹配置信度为 0，无法归一化")
        return None
    wR, wG, wB = max_R / total, max_G / total, max_B / total
    print(f"通道权重：R={wR:.2f}, G={wG:.2f}, B={wB:.2f}")

    # 7. 加权融合匹配得分
    fused_score = wR * result_R + wG * result_G + wB * result_B

    # 8. 获取最终匹配位置
    _, max_val, _, max_loc = cv2.minMaxLoc(fused_score)
    h, w = tpl_R.shape
    top_left = max_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)

    print(f"匹配位置：{top_left} → {bottom_right}")
    print(f"融合匹配置信度：{max_val:.4f}")

    if max_val < threshold:
        print("❌ 匹配置信度不足")
        return None

    # 9. 可视化匹配结果
    if output_path:
        matched = image_bgr.copy()
        cv2.rectangle(matched, top_left, bottom_right, (0, 0, 255), 2)
        cv2.imwrite(output_path, matched)
        print(f"✅ 已保存匹配图像到: {output_path}")
    return [top_left, bottom_right]
# ------------------ 主流程 ------------------

def load_image_from_url(url):
    response = requests.get(url)
    image_array = np.asarray(bytearray(response.content), dtype=np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_UNCHANGED)  # 支持 RGBA
    return image

def get_slider_x(tpl_url_path, search_url,image_path_to_save=None):
    # 1. 去掉灰边，得到模板
    template = peel_gray_border(tpl_url_path, tolerance=1, max_layers=32)

    # 2. 在大图中搜索并画出结果
    template = darken_template(template, factor=0.55)
    res = fast_match_rgb_with_alpha(template, search_url,image_path_to_save,threshold=0.25)
    print(f"result:{res}")
    return res[0][0]

def Test():
    url1 = "https://p9-catpcha.byteimg.com/tos-cn-i-188rlo5p4y/5f6801c38ad94e5988104df3374ae6f1~tplv-188rlo5p4y-1.png"
    url2 = "https://p9-catpcha.byteimg.com/tos-cn-i-188rlo5p4y/4820bfeaa7ef4779b44139a5c97fa567~tplv-188rlo5p4y-2.jpeg"

    x = get_slider_x(url1,url2)
    print(f"x:{x}")
if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python detect_exact_size_match.py template.png search.jpg out_debug.png")
        sys.exit(1)
     tpl_path = sys.argv[1]
    search_path = sys.argv[2]
    out_debug = sys.argv[3]

    # 1. 去掉灰边，得到模板
    template = peel_gray_border(tpl_path, "template.png",tolerance=1, max_layers=32)

    # 2. 在大图中搜索并画出结果
    template =darken_template(template,factor=0.55, save_path=f"{out_debug}_template_dark.png")
    fast_match_rgb_with_dynamic_weights(template, search_path, out_debug,threshold=0.25)

