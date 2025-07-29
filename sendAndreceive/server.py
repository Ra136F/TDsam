from flask import Flask, request
import os

app = Flask(__name__)

# ���ñ����ļ���·��
UPLOAD_FOLDER = './uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return "û���ļ�����", 400

    file = request.files['file']

    if file.filename == '':
        return "û��ѡ���ļ�", 400

    # �����ļ�
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(file_path)

    return f"�ļ��ɹ��ϴ��� {file_path}", 200


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
