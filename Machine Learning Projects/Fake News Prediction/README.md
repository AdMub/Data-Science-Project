# Fake News Detection

![Fake News Banner](https://github.com/user-attachments/assets/63a3dbff-45fd-4b39-ad7a-db214e7ab89a)
</br>


![Fake News Banner](https://github.com/user-attachments/assets/40dec40b-75d5-4cb0-9a4d-43ddd94385fe)
</br>

## Build a System to Identify Unreliable News Articles

## Project Work Flow

![Project Work Flow](https://github.com/user-attachments/assets/4fc41bcc-e5bd-4af6-8e09-552553ce6caf)
</br>


---

## Overview
This project aims to create a system that classifies news articles as **reliable** or **unreliable** using machine learning techniques. A Logistic Regression model was trained on a dataset containing article information, including titles, authors, and text, to predict their reliability.

---

## Dataset Description
The dataset used for this project is structured as follows:

- **train.csv**:
  - `id`: Unique ID for each news article.
  - `title`: The title of the news article.
  - `author`: The author of the article.
  - `text`: The body of the article (may be incomplete).
  - `label`: Binary label indicating article reliability:
    - `1`: Unreliable
    - `0`: Reliable

---

## Tools & Libraries Used

- Python
- Libraries:
  - `numpy`
  - `pandas`
  - `nltk`
  - `scikit-learn`
  - `gradio`

Install Gradio using the following command:
```bash
pip install gradio
```

---

## Data Preprocessing
1. **Loading Data**:
   - The dataset is loaded using `pandas.read_csv`.
2. **Handling Missing Values**:
   - Missing values in `author` and `title` columns are filled with an empty string.
3. **Feature Engineering**:
   - Combined `author` and `title` columns into a new column, `content`.
4. **Text Cleaning & Stemming**:
   - Applied stemming using NLTK's `PorterStemmer`.
   - Removed stopwords.

---

## Model Training
1. **Feature Vectorization**:
   - Used `TfidfVectorizer` to convert textual data into numerical format.
2. **Splitting Dataset**:
   - Divided the data into training (80%) and testing (20%) sets.
3. **Classifier**:
   - Logistic Regression was trained on the vectorized data.

---

## Evaluation Metrics
- **Training Data Accuracy**: 97.8%
- **Test Data Accuracy**: 94.2%
- **Classification Report**:
  - Precision, Recall, and F1-score were calculated for both classes.

---

## Predictive System
A predictive function was implemented to classify input text as `Real` or `Fake` using the trained Logistic Regression model. The function preprocesses input text, vectorizes it, and predicts reliability.

---

## Deployment with Gradio
The predictive system was deployed using Gradio to create an interactive user interface.

### Features:
- **Input**: Text field for news article content.
- **Output**: Text displaying the prediction (`Real` or `Fake`).

### How to Use:
Run the following script to start the Gradio interface:
```python
import gradio as gr

# Define the Gradio interface
interface = gr.Interface(
    fn=predict_news_reliability,
    inputs="text",
    outputs="text",
    title="News Reliability Predictive System",
    description=(
        "This system predicts whether a news article is real or fake. "
        "Input the article text to get the prediction."
    )
)

# Launch the interface
interface.launch()
```

---

## Sample Screenshots
### User Interface
![Gradio Interface](https://github.com/user-attachments/assets/894bea9a-abf8-4879-b70b-9c3b30d3dec9)
</br>

### Example Predictions
![Predictions Example](https://github.com/user-attachments/assets/d9d40b53-3a13-41e1-b275-d79d4f34e4ab)


---

## How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/AdMub/Data-Science-Project.git
   ```
2. Navigate to the project directory:
   ```bash
   cd fake-news-prediction
   ```
3. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the Gradio app:
   ```bash
   python app.py
   ```

---

## Acknowledgements
- [Kaggle](https://www.kaggle.com/) for the dataset.
- [Gradio](https://gradio.app/) for creating user-friendly interfaces.

---

## Future Work
- Add more sophisticated models such as Transformers (e.g., BERT).
- Expand dataset for better generalization.
- Include more interpretability features in the UI.

---

## Contributing
Feel free to fork this repository, make feature improvements, or raise issues. Contributions are welcome!

---

## License
This project is licensed under the MIT License. See `LICENSE` for details.

