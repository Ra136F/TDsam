from flask import Flask, request, jsonify
import os

app = Flask(__name__)

# 设置上传文件的保存路径
UPLOAD_FOLDER = 'uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# 确保上传目录存在
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "没有文件部分"}), 400
    file = request.files['file']

    # 如果没有选择文件
    if file.filename == '':
        return jsonify({"error": "没有选择文件"}), 400

    # 保存文件
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(file_path)

    return jsonify({"message": f"文件成功上传到 {file_path}"}), 200


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
