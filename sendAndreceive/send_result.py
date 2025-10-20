import requests


def upload_zip_to_server(zip_file_path, upload_url):
    # 打开要上传的zip文件
    with open(zip_file_path, 'rb') as zip_file:
        # 使用requests发送POST请求
        files = {'file': (zip_file_path, zip_file, 'application/zip')}
        try:
            response = requests.post(upload_url, files=files)

            # 检查上传是否成功
            if response.status_code == 200:
                print(f"文件成功上传到 {upload_url}")
            else:
                print(f"上传失败, 状态码: {response.status_code}")
        except requests.exceptions.RequestException as e:
            print(f"上传过程中发生错误: {e}")


# 使用示例
zip_file_path = '../result.zip'  # 本地zip文件路径
upload_url = 'http://192.168.31.59:5000/upload'  # 远程服务器的上传API URL

upload_zip_to_server(zip_file_path, upload_url)
