from flask import Flask, request
import os

app = Flask(__name__)

# 设置保存文件的路径
UPLOAD_FOLDER = './uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return "没有文件部分", 400

    file = request.files['file']

    if file.filename == '':
        return "没有选择文件", 400

    # 保存文件
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(file_path)

    return f"文件成功上传到 {file_path}", 200


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
