import os
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
from PIL import Image
from predict import main as detect_main

# 初始化 Flask 应用
app = Flask(__name__)

# 设置上传文件的存储路径
UPLOAD_FOLDER = 'static/images'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# 允许上传的文件类型
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}


# 检查文件类型是否允许
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# 调整图像大小
def resize_image(image_path, output_path, size=(1000, 680)):
    with Image.open(image_path) as img:
        img = img.resize(size, Image.ANTIALIAS)
        img.save(output_path)


# 路由：主页面
@app.route('/')
def index():
    # 初始图像显示test.png，显示默认图片
    image_url = url_for('static', filename='images/background.png')
    return render_template('index.html', image_url=image_url, category='', confidence='', description='')


# 路由：上传图像
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)

    file = request.files['file']

    if file and allowed_file(file.filename):
        filename = secure_filename('test.png')  # 将文件命名为 'test.png'
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)  # 保存原始文件
        resize_image(file_path, file_path)  # 调整图像大小为 1000x680
        image_url = url_for('static', filename='images/test.png')  # 获取上传后的图片路径
        return render_template('index.html', image_url=image_url, category='', confidence='',
                               description='')


# 路由：开始检测（更新识别结果）
@app.route('/detect', methods=['POST'])
def detect():
    # 模拟检测结果，更新为向日葵、90%、是一种植物
    image_url = url_for('static', filename='images/test.png')
    res,num,inf=detect_main('static/images/test.png')
    return render_template('index.html', image_url=image_url, category=res, confidence=num, description=inf)


if __name__ == '__main__':
    # 创建 images 文件夹以存储上传的图像
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True)
