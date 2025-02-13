# Real-Time-Translation-using-Machine-learning
Real Time Translation using Machine Learning, Google API, Flask.

<h2>📌 Overview</h2>

This is a Flask-based web application that translates text from English to Hindi, Marathi, and Kannada using Hugging Face models for Hindi/Marathi and Google Translate API for Kannada. The application also supports voice input for convenience.


<h2>🛠️ Installation</h2>

<h3>1️⃣ Clone the Repository</h3>

```
git clone https://github.com/AbhishekKameri/Real-Time-Translation-using-Machine-learning.<br>
git cd Real-Time-Translation-using-Machine-learning
```

<h3>2️⃣ Create a Virtual Environment (Recommended)</h3>

```
python -m venv venv
source venv/bin/activate  # On macOS/Linux
venv\Scripts\activate    # On Windows
```

<h3>3️⃣ Install Dependencies</h3>

```
pip install -r requirements.txt
```

<h3>4️⃣ Install Additional Dependencies (if needed)</h3>

```
pip install transformers torch flask googletrans==4.0.0-rc1 sentencepiece
```

<h3>🚀 Running the Application</h3>

<h3>1️⃣ Start Flask Development Server</h3>

```
python app.py
```
* Open http://127.0.0.1:5000/ in your browser.<br><br><br>
  



<h2>🛠️ Manual Execution Process</h2>

<h4>1. Ensure Python is Installed: Verify that Python is installed by running:</h4>

```
python --version
```

<h4>2. Navigate to Project Directory: Open terminal/command prompt and move to the project folder:</h4>

```
cd path/to/Real-Time-Translation-using-Machine-learning
```

<h4>3. Activate Virtual Environment:</h4>

* On macOS/Linux:


```
source venv/bin/activate
```

* On Windows:
```
venv\Scripts\activate
```

<h4>4. Install Required Dependencies:</h4>

```
pip install -r requirements.txt
```

<h4>5. Run the Flask Application:</h4>

```
python app.py
```

<h4>6. Access the Web App:</h4>

* Open a web browser and go to:


```
http://127.0.0.1:5000/
```
* Enter text, select a language, and click 'Translate'<br>
  

<br><h2>🌍 How It Works</h2>

* The app uses Hugging Face's MarianMT models for Hindi and Marathi.
* For Kannada, it uses Google Translate API (via googletrans library).
* Users can enter text in English and select a target language for translation.
* Voice input support allows users to speak instead of typing.
  


<br><h2>📂 Project Structure</h2>

```
Real-Time-Translation-using-Machine-learning/
│── static/
│   ├── style.css  <-- (CSS file)
│── templates/
│   ├── index.html  <-- (HTML file)
│── app.py
│── requirements.txt
│── README.md

```



<br><h2>🛠️ Technologies Used</h2>

* Flask - Web framework
* Hugging Face Transformers - For translation models
* Google Translate API - For Kannada translation
* HTML, CSS, JavaScript - Frontend UI
* Web Speech API - For voice input support



<br><h2>📧 Contact & Support</h2>

If you encounter any issues, feel free to open an issue on GitHub or reach out via email at abhishekkameri47@gmail.com.

Code with fun! 🚀

<h2>Screen Shots</h2>


<img src="https://github.com/user-attachments/assets/a1404ba4-6947-498b-b831-64ac407a0e0b" width="500">















