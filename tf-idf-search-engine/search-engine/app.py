from flask import Flask, render_template, request, redirect
import module

app = Flask(__name__)

@app.route('/result', methods=['POST', 'GET'])
def search_result():
    if request.method == 'POST':
        # ambil query dari form di html
        query = request.form.get('search', '')

        # panggil functionnya dari module
        result_df = module.cosine_similarity_T(10, query)

        # hasilnya dimasukkin ke dictionary
        result_json = result_df.to_dict(orient='records')

        # baca konten setiap file
        for result in result_json:
            filepath = result.get('filepath', '')
            if filepath:  
                # handling saat buka file nya
                try:
                    with open(filepath, 'r', encoding='utf-8', errors='ignore') as file:
                        file_contents = file.read()
                        result['file_contents'] = file_contents
                except FileNotFoundError:
                    print(f"File not found: {filepath}")
            else:
                print("Invalid filepath in result:", result)
        
        # render result.html (data + query)
        return render_template('result.html', result_json=result_json, query=query)

    return redirect('/')


@app.route('/', methods=['GET'])
def index():
    # render html template
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)