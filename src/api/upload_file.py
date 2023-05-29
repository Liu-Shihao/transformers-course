from flask import Flask, request

app = Flask(__name__)

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return 'No file part in the request'

    file = request.files['file']

    if file.filename == '':
        return 'No selected file'

    if file:
        # 保存文件到指定目录
        file.save('../../data/'+file.filename)
        return 'File uploaded successfully'

if __name__ == '__main__':
    app.run()
