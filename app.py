from flask import Flask
from flask import render_template, request, send_from_directory
import app122
import csv
from concurrent.futures import ThreadPoolExecutor
executor = ThreadPoolExecutor(1)
app = Flask(__name__)



@app.route('/upload_seq', methods=['POST'])
def upload_seq():
    file_path = 'static/upload/new_test.txt'
    Seq_cadd = request.form.get('Seq_cadd')
    sequences = Seq_cadd.strip().splitlines()
    final_output = ""

    for sequence in sequences:
        spaced_seq = ' '.join(list(sequence))  
        final_output += f"{spaced_seq}\t1\n" 

    with open(file_path, 'w+') as f:
        f.write(final_output)
    
    try:
        result = app122.predict() 
        csv_file = './output/final_predictions.csv'
        
        table_data = []
        with open(csv_file, 'r') as f:
            reader = csv.reader(f)
            table_data = [row for row in reader]
        
        data = {
            'table_data': table_data  
        }
        
        return render_template('back.html', **data)
    except Exception as e:
        print(e)
        return render_template("exception.html")




@app.route('/', methods=['POST', 'GET'])
def Pse_home():
    return render_template('AMP_home.html')


@app.route('/Pse_example', methods=['POST', 'GET'])
def Pse_example():
    return render_template('AMP_example.html')


@app.route('/Pse_help', methods=['POST', 'GET'])
def Pse_help():
    return render_template('AMP_help.html')


@app.route('/Pse_backhome', methods=['POST', 'GET'])
def Pse_backhome():
    return render_template('AMP_backhome.html')

@app.errorhandler(404)
def Pse_page_not_found(e):
    return render_template('AMP_404.html')


if __name__ == "__main__":
    app.run(debug=True)
