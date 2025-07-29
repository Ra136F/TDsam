import requests

def send_csv_to_server(csv_file_path, server_url):
    try:
        with open(csv_file_path, 'rb') as f:
            files = {'file': (csv_file_path, f, 'application/csv')}
            response = requests.post(server_url, files=files)

            if response.status_code == 200:
                print("�ļ��ϴ��ɹ�")
            else:
                print(f"�ļ��ϴ�ʧ�ܣ�״̬��: {response.status_code}")
    except Exception as e:
        print(f"�ϴ������з�������: {e}")

# ʾ�������ļ����͵������
csv_file_path = '../data/household.csv'  # ��Ҫ�ϴ��� CSV �ļ�·��
server_url = 'http://192.168.31.59:5000/upload'  # ��������ַ
send_csv_to_server(csv_file_path, server_url)
