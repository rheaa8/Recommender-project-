# Comparative Analysis of Recommender Systems

An experimental study of collaborative filtering, content-based recommendation, matrix factorization, and hybrid recommendation techniques using the **MovieTweetings dataset**.

## Overview

This project implements and evaluates multiple recommendation algorithms to understand their performance across rating prediction and Top-N recommendation tasks.

The objective is to compare different recommendation approaches using a consistent experimental framework, optimize model hyperparameters, and analyze how recommendation quality varies across different user activity levels.

## Key Features

- Implementation of multiple recommendation algorithms from scratch.
- Comparative evaluation of collaborative filtering and content-based approaches.
- Hyperparameter tuning using a separate validation set.
- Rating prediction and Top-N recommendation evaluation.
- Hybrid recommendation through weighted model combinations.
- User-segment analysis based on activity levels.
- Statistical comparison and confidence interval estimation.
- Automated generation of performance visualizations and experiment results.

## Algorithms Implemented

| Category | Algorithms |
|---|---|
| Baselines | Random, Popularity |
| Collaborative Filtering | User-KNN, Item-KNN |
| Implicit Feedback | Item-KNN (Implicit) |
| Content-Based | Genre and Decade-based Recommendation |
| Matrix Factorization | SVD, PureSVD, ALS |
| Hybrid | PureSVD + Item-KNN + Content-Based |

## Evaluation Metrics

The models are evaluated using two categories of metrics:

**1. Rating Prediction**
- RMSE (Root Mean Squared Error)
- MAE (Mean Absolute Error)

**2. Top-N Recommendation**
- Precision@K
- Recall@K
- NDCG@K (Normalized Discounted Cumulative Gain)
- Hit Rate@K
- Catalog Coverage
- Recommendation Novelty

The evaluation also includes 95% confidence intervals and paired statistical comparisons for NDCG@10.

## Experimental Methodology

The experimental pipeline follows these stages:

1. **Data Preprocessing:** Load ratings and movie metadata, remove duplicate interactions, and apply iterative 20-core filtering.
2. **Data Splitting:** Divide the data into training (70%), validation (10%), and testing (20%) sets.
3. **Hyperparameter Optimization:** Tune model parameters using validation data.
4. **Model Training:** Train the selected configurations on the full training split.
5. **Evaluation:** Compare models on held-out test data.
6. **Segment Analysis:** Evaluate recommendation performance for light, medium, and heavy users.
7. **Visualization:** Generate comparative plots and save experiment results.

A fixed random seed (42) is used for reproducibility.

## Project Structure

```text
Recommender_Systems_Code/
│
├── data/                  # Dataset directory
│
├── results/
│   ├── figs/              # Generated visualizations
│   ├── results.json       # Evaluation results
│   └── dims.json          # Experiment dimensions
│
├── recsys_lib.py          # Recommendation algorithms
├── run_experiments.py     # Main experimental pipeline
├── build_report.js        # Automated report generation
├── requirements.txt       # Python dependencies
└── README.md
```

## Dataset

This project uses the **MovieTweetings dataset**, which contains movie ratings and associated movie metadata.

Dataset source: [MovieTweetings – GitHub](https://github.com/sidooms/MovieTweetings)

Download the following files:
- `ratings.dat`
- `movies.dat`

Place them in a `data/` directory one level above the project folder, as currently expected by `run_experiments.py`.

## Installation and Execution

**1. Clone the repository**

```bash
git clone https://github.com/rheaa8/Recommender-project-.git
cd Recommender-project-
```

**2. Install dependencies**

```bash
pip install -r requirements.txt
```

**3. Download and prepare the dataset**

Download `ratings.dat` and `movies.dat` from the dataset source and place them in the expected data directory.

**4. Run experiments**

```bash
python run_experiments.py
```

The pipeline generates evaluation results, model comparisons, and visualizations in the `results/` directory.

## Results and Visualizations

The repository includes experimental outputs and visualizations covering:

- Exploratory Data Analysis (EDA)
- Rating prediction accuracy
- Ranking performance
- Hyperparameter sensitivity
- User-segment performance
- Model trade-offs

See the `results/figs/` directory for the generated plots and `results/results.json` for detailed numerical results.

## Technologies Used

- **Programming Language:** Python
- **Numerical Computing:** NumPy, SciPy
- **Data Processing:** Pandas
- **Visualization:** Matplotlib
- **Report Generation:** JavaScript (Node.js)

## Author

**Rhea Ahuja**

B.Tech – Computer Science and Engineering

[GitHub Profile](https://github.com/rheaa8)

---

*This project explores the design, implementation, and empirical evaluation of recommendation algorithms, with an emphasis on comparative performance, reproducibility, and recommendation quality.*
