# 📸 Image Captioning Projects

Welcome to my collection of **Image Captioning** projects! 🖼️✨  
These projects use **Hugging Face's BLIP (Bootstrapping Language-Image Pretraining) model** to generate captions for images. Each project has a specific use case, from a **web app** to **automated URL captioning** and **local image organization**.

---

## 🚀 Projects Overview

### 1️⃣ **Image Captioning Web App (`image_captioning_app`)**
A **Gradio-powered web app** that allows users to upload images and receive AI-generated captions.

#### 🛠️ **Technologies Used**
- `gradio` for the web interface
- `transformers` for deep learning models
- `PIL` and `numpy` for image handling

#### 🔹 **How It Works**
- The user uploads an image.
- The BLIP model generates a caption.
- The caption is displayed as output.

#### ▶️ **How to Run**
```bash
pip install -r requirements.txt
python image_captioning_app.py

### 2️⃣ **Automated URL Captioner (`automate_url_captioner`)**

A script that **scrapes images from a webpage** and generates captions for them.  
I used this for an article I was writing about **Granada**, extracting and captioning images from Wikipedia.

#### 🛠️ **Technologies Used**
- `requests` for downloading web pages
- `BeautifulSoup` for web scraping
- `PIL` and `transformers` for image processing

#### 🔹 **How It Works**
- The script **scrapes all images** from a given URL.
- It **filters out irrelevant images** (e.g., very small or SVG files).
- It **downloads and processes the images** to generate captions.
- Captions are saved in a text file (`captions.txt`).

#### ▶️ **How to Run**
```bash
pip install -r requirements.txt
python automate_url_captioner.py

### 3️⃣ **Local Image Captioning (`image_captioning_local_files`)**

A script that **automatically generates captions** for a local folder of images.  
I used this to organize images from a **music concert** where I played as a **pianist**. 🎹🎶

#### 🛠️ **Technologies Used**
- `os` and `glob` for handling local files
- `transformers` for AI-powered caption generation
- `PIL` for image processing

#### 🔹 **How It Works**
- The script **scans a specified folder** for images.
- It **processes each image** and generates a caption.
- The captions are saved in a text file (`captions2.txt`) alongside the image names.

#### ▶️ **How to Run**
```bash
pip install -r requirements.txt
python image_captioning_local_files.py
