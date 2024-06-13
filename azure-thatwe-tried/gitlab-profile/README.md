# Deep Learning Project (Group 02)

---

## Kaggle Competition: [Twitter Emergency Tweets Classification](https://www.kaggle.com/competitions/nlp-getting-started/overview)

### Problem Statement
Twitter has become a crucial communication channel during emergencies, enabling real-time reporting of incidents. However, distinguishing between tweets related to actual disasters and those that aren't can be challenging. This competition aims to develop models capable of accurately classifying tweets as either indicating a real disaster or not.

---

### Dataset
The dataset provided for this competition includes a collection of tweets along with labels indicating whether they pertain to a real disaster or not. Participants are tasked with training models on the labeled data to predict the classification of unseen tweets accurately.

---

### Approach
**Model Selection:**  
For this competition, we used Recurrent Neural Network (RNN) architectures like many to one, and also Long Short-Term Memory (LSTM) networks. RNNs are well-suited for sequence data like text, making them an ideal choice for natural language processing tasks such as tweet classification.

**Feature Engineering:**  
We conducted extensive feature engineering to extract meaningful representations from the tweet text. This involved techniques such as tokenization, vectorization, and possibly the incorporation of external resources such as pre-trained word embeddings.

**Training and Evaluation:**  
The training process involved optimizing the chosen RNN and LSTM models on the training dataset while monitoring performance using appropriate evaluation metrics such as accuracy, precision, recall, and F1-score. We also utilized techniques like cross-validation to ensure the generalization of our models.

---

### Repository Structure
- **CSV/:** Contains the competition datasets.
    - `predictionsLSTM.csv`: LSTM predictions
    - `predictionsRNN.csv`: RNN predictions
    - `train.csv`: Training dataset.
    - `test.csv`: Test dataset for prediction.
- `main.ipynb`: Jupyter Notebook containing RNN and LSTM model implementations.
- `Pipfile`: List of required Python packages.

---

### Results
Our models achieved competitive performance on the provided test dataset, demonstrating the effectiveness of RNN and LSTM architectures for classifying emergency tweets accurately.
