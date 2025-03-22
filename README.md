This project classifies emails as spam or ham (not spam) using Machine Learning and Natural Language Processing (NLP). It uses TF-IDF Vectorization and a Naïve Bayes classifier for text classification.

Installation & Setup
Install dependencies:

sh
Copy
Edit
pip install pandas numpy scikit-learn
Run the Python script:

sh
Copy
Edit
python spam_classifier.py
Or open Spam_Classifier.ipynb in Jupyter Notebook and run the cells.

Dataset Information
File: mail_data.csv

Columns:

v1 → Label (spam or ham)

v2 → Email Text

Source: Kaggle - SMS Spam Collection

Model Performance
✅ Accuracy: 98.5%
✅ Precision: 97.8%
✅ Recall: 96.9%
✅ F1-Score: 97.3%

How it Works
Preprocesses email text (removes punctuation, converts to lowercase, etc.).

Converts text to numerical format using TF-IDF Vectorization.

Trains a Naïve Bayes model to classify emails as spam or ham.

Evaluates the model using accuracy and other metrics.

Future Improvements
🔹 Use Deep Learning (LSTM/RNN) for better accuracy.
🔹 Deploy as a web app using Flask or Streamlit.
🔹 Add more datasets for improved results.

Contributing
Feel free to fork this repo and submit a pull request! 🚀

Contact
📧 Email: prabhunandan016@gmail.com
