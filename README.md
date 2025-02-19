# Movie Portfolio Project (see all the codes used with result by clicking on the pynb. file)

## Overview

This project is a comprehensive analysis of a dataset containing information about movies. The dataset includes various attributes such as budget, gross earnings, genre, director, runtime, ratings, and more. The primary goal of this project is to explore the dataset, perform data cleaning, and derive meaningful insights that can help in understanding the factors that contribute to a movie's success.

## Dataset

The dataset used in this project is named `movies.csv` and contains 6820 rows and 15 columns. The columns include:

- **budget**: The budget of the movie.
- **company**: The production company.
- **country**: The country where the movie was produced.
- **director**: The director of the movie.
- **genre**: The genre of the movie.
- **gross**: The gross earnings of the movie.
- **name**: The name of the movie.
- **rating**: The rating of the movie.
- **released**: The release date of the movie.
- **runtime**: The runtime of the movie in minutes.
- **score**: The IMDb score of the movie.
- **star**: The main actor/actress in the movie.
- **votes**: The number of votes the movie received on IMDb.
- **writer**: The writer of the movie.
- **year**: The year the movie was released.

## Project Steps

### 1. Importing Libraries and Loading Data

The project begins by importing essential Python libraries such as `pandas`, `numpy`, `seaborn`, and `matplotlib`. These libraries are used for data manipulation, analysis, and visualization. The dataset is then loaded into a pandas DataFrame for further analysis.

```python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import matplotlib
plt.style.use('ggplot')
from matplotlib.pyplot import figure
%matplotlib inline
matplotlib.rcParams['figure.figsize'] = (12,8)
pd.options.mode.chained_assignment = None

df = pd.read_csv(r'C:\\Users\\alexf\\Downloads\\movies.csv')
```

### 2. Initial Data Exploration

The dataset is initially explored to understand its structure and content. This includes checking the first few rows of the dataset, understanding the data types of each column, and identifying any missing values.

```python
df.head()
df.dtypes
```

### 3. Data Cleaning

The dataset is checked for missing values, and any necessary cleaning is performed. This includes handling missing data, removing duplicates, and ensuring that the data types are appropriate for analysis.

```python
for col in df.columns:
    pct_missing = np.mean(df[col].isnull())
    print('{} - {}%'.format(col, round(pct_missing*100)))

df.drop_duplicates()
```

### 4. Data Visualization

Various visualizations are created to understand the distribution of different variables and their relationships. This includes box plots, scatter plots, and correlation matrices.

```python
df.boxplot(column=['gross'])
```

### 5. Sorting and Outlier Detection

The dataset is sorted based on gross earnings to identify the highest-grossing movies. Outliers in the gross earnings are also detected using box plots.

```python
df.sort_values(by=['gross'], inplace=False, ascending=False)
```

### 6. Correlation Analysis

A correlation matrix is created to understand the relationships between different numerical variables, such as budget, gross earnings, and IMDb scores.

```python
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True)
plt.title('Correlation Matrix for Numeric Features')
plt.xlabel('Movie Features')
plt.ylabel('Movie Features')
plt.show()
```

### 7. Conclusion

The analysis reveals several key insights:

1. **Budget and Gross Earnings**: There is a positive correlation between a movie's budget and its gross earnings. Higher-budget movies tend to earn more at the box office.

2. **Runtime and IMDb Score**: Movies with longer runtimes tend to have higher IMDb scores, indicating that audiences may prefer longer, more detailed films.

3. **Genre and Success**: Certain genres, such as Action and Adventure, tend to have higher gross earnings compared to others. This suggests that genre plays a significant role in a movie's financial success.

4. **Director and Star Power**: The presence of well-known directors and stars can significantly impact a movie's success, both in terms of gross earnings and IMDb scores.

5. **Year of Release**: Movies released in recent years tend to have higher budgets and gross earnings, possibly due to inflation and advancements in technology.

### 8. Future Work

- **Further Analysis**: Explore the impact of other variables such as country of production and production company on a movie's success.
- **Predictive Modeling**: Develop a predictive model to estimate a movie's gross earnings based on its attributes.
- **Sentiment Analysis**: Perform sentiment analysis on movie reviews to understand audience reception and its impact on box office performance.

## Conclusion

This project provides a detailed analysis of the factors that contribute to a movie's success. By understanding these factors, stakeholders in the film industry can make more informed decisions regarding movie production, marketing, and distribution. The insights derived from this analysis can also be used to predict the potential success of future movies, thereby optimizing resource allocation and maximizing returns.

## Repository Structure

- **Movie Portfolio Project.ipynb**: The Jupyter Notebook containing the complete code and analysis.
- **movies.csv**: The dataset used for the analysis.
- **README.md**: This file, providing an overview of the project and its findings.

## How to Use

1. Clone the repository to your local machine.
2. Open the Jupyter Notebook `Movie Portfolio Project.ipynb` to view the analysis.
3. Run the cells in the notebook to reproduce the results or modify the code for further analysis.
4. see all the codes used with result by clicking on the pynb. file
## Dependencies

- Python 3.x
- pandas
- numpy
- seaborn
- matplotlib


## Acknowledgments

- The dataset used in this project is sourced from [Kaggle](https://www.kaggle.com).
- Special thanks to the open-source community for providing the libraries and tools used in this analysis.

