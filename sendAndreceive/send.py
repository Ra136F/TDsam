import requests

def send_csv_to_server(csv_file_path, server_url):
    try:
        with open(csv_file_path, 'rb') as f:
            files = {'file': (csv_file_path, f, 'application/csv')}
            response = requests.post(server_url, files=files)

            if response.status_code == 200:
                print("文件上传成功")
            else:
                print(f"文件上传失败，状态码: {response.status_code}")
    except Exception as e:
        print(f"上传过程中发生错误: {e}")

# 示例：将文件发送到服务端
csv_file_path = '../data/household.csv'  # 需要上传的 CSV 文件路径
server_url = 'http://192.168.31.59:5000/upload'  # 服务器地址
send_csv_to_server(csv_file_path, server_url)
