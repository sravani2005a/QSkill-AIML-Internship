# QSkill AI & Machine Learning Internship — Two Projects

This package contains two projects from the QSkill internship task sheet.

## Project 1 — Iris Flower Classification
Classifies an iris flower into Setosa, Versicolor, or Virginica using the four sepal/petal measurements.

### Run
```bash
cd Project_1_Iris_Flower_Classification
python -m pip install -r requirements.txt
python iris_classification.py
```
The scikit-learn Iris dataset is built in, so no separate dataset download is required. Charts are saved in the `outputs` folder.

## Project 2 — Spam Mail Detector
Classifies a text message as spam or not spam using TF-IDF features and Multinomial Naive Bayes.

### Run
```bash
cd Project_2_Spam_Mail_Detector
python -m pip install -r requirements.txt
python spam_mail_detector.py
```
The script attempts to download a public SMS spam dataset. If the download is blocked, place the UCI SMS Spam Collection file named `SMSSpamCollection.txt` beside `spam_mail_detector.py`. It should contain tab-separated label and message columns.

## VS Code instructions
1. Extract the ZIP file.
2. Open the extracted `QSkill_AIML_Projects` folder in VS Code.
3. Open the terminal: **Terminal → New Terminal**.
4. Install the libraries using `python -m pip install -r requirements.txt` from the relevant project folder.
5. Run the Python file using `python iris_classification.py` or `python spam_mail_detector.py`.

If `python` does not work on Windows, try `py` instead.

## Submission checklist
- Keep the Python source files.
- Include screenshots of each project running and its evaluation output.
- Include generated charts from the `outputs` folders.
- Add your own name, internship details, and screenshots to a report if your coordinator requests one.
- Run both projects on your own computer before submitting. Accuracy varies with the train/test split and dataset.

## Important
The Spam Mail Detector is trained on an SMS dataset, which is a common beginner NLP dataset. In your report, call it a **Spam Message Detector** unless your internship coordinator specifically accepts SMS data as an email dataset. The task sheet says “Spam Mail Detector” and suggests public SMS Spam Collection or Enron data.
