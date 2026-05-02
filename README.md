# 🎬 Movie Recommendation System

A complete Movie Recommendation System built with **Streamlit**, featuring multiple recommendation models including Content-Based Filtering and Popularity-based approaches.

## ✨ Features

- **Personalized Recommendations:** Discover movies tailored to specific tastes using content-based filtering.
- **Trending & Popular:** Get recommendations based on overall popularity and highest-rated movies.
- **Search Capabilities:** A powerful search engine to quickly find your favorite movies by title or genre.
- **Interactive UI:** A highly interactive and aesthetic user interface built with Streamlit and custom CSS.

## 🗂️ Project Structure

```text
Recommendation System/
├── app/
│   ├── main.py                    # Main Streamlit application
│   ├── requirements.txt           # Project dependencies
│   ├── assets/
│   │   └── style.css              # Custom UI styling
│   ├── data/                      # Dataset folder (movies, ratings, tags)
│   ├── models/
│   │   ├── content_based.py       # Content-based filtering logic
│   │   ├── popularity.py          # Popularity-based filtering logic
│   │   └── search.py              # Search functionality
│   └── notebooks/                 # Jupyter notebooks for data analysis & model prototyping
└── README.md                      # Project documentation
```

## 🚀 Getting Started

### Prerequisites
Make sure you have Python 3.8+ installed.

### Installation

1. **Clone the repository** (if applicable) or download the files.
2. **Navigate to the app directory:**
   ```bash
   cd "Recommendation System/app"
   ```
3. **Install the required dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   *Note: Core dependencies include `streamlit`, `pandas`, `numpy`, and `scikit-learn`.*

### Running the App

To start the Streamlit application, run the following command from within the `app` directory:

```bash
streamlit run main.py
```

The application will start and open automatically in your default web browser (usually at `http://localhost:8501`).

## 🧠 Models Explained

### 1. Popularity-Based Recommender
Recommends movies that are globally popular and highly rated. It calculates a weighted score based on the average rating and the number of votes a movie has received, minimizing the bias of a few 5-star ratings.

### 2. Content-Based Recommender
Recommends movies similar to a movie the user likes. It utilizes **TF-IDF (Term Frequency-Inverse Document Frequency)** and **Cosine Similarity** on features such as movie genres and tags to find and suggest closely related movies.

## 📊 Dataset
This project uses the famous **MovieLens** dataset. Make sure your `app/data/` folder contains the following files to ensure the app works smoothly:
- `movie.csv`
- `rating.csv`
- `tag.csv`

*(Note: Data is sampled down in the app for performance optimization during live inference).*

## 🤝 Contributing
Contributions, issues, and feature requests are welcome! Feel free to check the issues page if you want to contribute.

## 📝 License
This project is open-source and available under the MIT License.
