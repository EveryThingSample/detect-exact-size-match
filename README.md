# Detect Exact Size Match

## 📖 Project Description

This is a **strict non-scaling image template matching tool**, mainly designed for **Douyin (TikTok China) slider captchas** and similar scenarios.  
It can locate the position in the search image that has the **exact same size** as the template, with support for gray-border removal, template darkening, and alpha-channel matching.  
The tool also outputs a series of intermediate debug files to help developers visualize and verify the matching process.

---

## ✨ Features
- Optimized for Douyin slider captcha matching
- Automatically removes gray borders from templates (multi-layer peeling supported)
- Supports RGBA templates and alpha-channel matching
- Dynamically weighted RGB channel fusion for robust matching
- Outputs intermediate debug files (template crop, masks, candidate patches, etc.)

---

## 📦 Dependencies
```bash
pip install opencv-python numpy pillow requests
