# Sentiment Analysis Project

This project implements a **Sentiment Analysis** model using **TensorFlow** and **Keras** for deep learning, with a **Flask** backend to serve predictions through a web interface. The model is fine-tuned using **GridSearchCV** for hyperparameter optimization to achieve the best performance.

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Model Details](#model-details)
- [API Endpoints](#api-endpoints)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Overview

This project aims to classify text as **positive** or **negative** based on sentiment. It uses a deep learning model built with **TensorFlow** and **Keras**, which is optimized using **GridSearch** for the best hyperparameter combination. The trained model is connected to a web interface via a **Flask** backend, allowing users to input text and receive sentiment predictions in real-time.

## Project Structure

```

├── main.py # Flask application entry point
├── models/
│ ├── best_sentiement_model.h5 # Trained Keras model
| ├── tokenizer.pkl # fitted tokenizer
├── templates/
│ └── index.html # Frontend HTML page
├── Dataset/
│ └── reviews_test.csv # Dataset for training
| └── reviews_train.csv
├── Notebooks/
| └── Sentiment_analysis_DL.ipynb # Jupyter notebook of trained model
├── requirements.txt # Python dependencies
└── README.md # Project documentation
```

## Installation

### Prerequisites

- Python 3.7 or later
- pip

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/VaibhavVermaa16/Sentiment-analysis-using-DL
   cd Sentiment-analysis-using-DL
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the Flask application:
   ```bash
   python main.py
   ```

4. Open a browser and navigate to:
   ```
   http://127.0.0.1:5000
   ```

## Usage

The web interface accepts user input (text) and returns a sentiment classification as either **Positive** or **Negative**.

### Programmatic Access:

You can also use the API directly:
```bash
curl -X POST -d "text=I love this product!" http://127.0.0.1:5000/predict
```

## Model Details

- **Architecture**: A deep learning model using **TensorFlow** and **Keras**.
- **Optimization**: **GridSearchCV** is used to find the best combination of hyperparameters such as batch size, number of epochs, learning rate, and optimizers.
- **Dataset**: The model is trained on a dataset of labeled text reviews (e.g., product reviews).

### Hyperparameters Tuned

- Batch size
- Number of epochs
- Optimizer
- Learning rate

## API Endpoints

- `GET /`: Renders the web frontend.
- `POST /predict`: Accepts text input and returns a JSON response with the sentiment (positive/negative).

### Example Response:

```json
{
"text": "I love this!",
"sentiment": "Positive"
}
```

## Results

The model achieved an accuracy of **92%** & **71%** on the test dataset after hyperparameter tuning using **GridSearch**.

## Contributing

Feel free to open issues and submit pull requests to improve this project.
