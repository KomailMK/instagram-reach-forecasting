# Instagram Reach Forecasting

This repository contains code for forecasting Instagram reach using historical data and a SARIMA model. The project includes both a Python script and a Jupyter notebook for analysis and model training.

## Repository Structure
instagram-reach-forecasting/   
│   
├── data/   
│ └── Instagram-Reach.csv # Your dataset   
├── scripts/   
│ ├── forecast_reach.py # Python script   
│ └── forecast_reach.ipynb # Jupyter notebook   
├── .gitignore # Git ignore file   
├── LICENSE # License file (optional)   
└── README.md # README file with instructions   



## Setup Instructions

### Prerequisites

Ensure you have the following installed:
- Python 3.6 or higher
- pip (Python package installer)
- Jupyter Notebook (for running the notebook file)

### Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/your-username/instagram-reach-forecasting.git
    cd instagram-reach-forecasting
    ```

2. Install the required Python packages:
    ```bash
    pip install -r requirements.txt
    ```

### Running the Python Script

1. Ensure your dataset (`Instagram-Reach.csv`) is in the `data` directory.
2. Run the script:
    ```bash
    python scripts/forecast_reach.py
    ```

### Running the Jupyter Notebook

1. Ensure your dataset (`Instagram-Reach.csv`) is in the `data` directory.
2. Open the Jupyter Notebook:
    ```bash
    jupyter notebook scripts/forecast_reach.ipynb
    ```

3. Run the cells in the notebook to perform the analysis and train the model.

## Project Description

This project performs the following steps:
1. **Data Import and Preprocessing**: Loading the dataset and checking for missing values, column information, and descriptive statistics.
2. **Data Analysis and Visualization**: Analyzing trends, distributions, and patterns in Instagram reach data using line charts, bar charts, and box plots.
3. **Feature Engineering**: Creating a day column and analyzing reach based on the days of the week.
4. **Model Training**: Training a SARIMA model to forecast Instagram reach.
5. **Forecasting**: Making predictions for future Instagram reach and visualizing the results.
6. **Saving and Loading the Model**: Saving the trained model and loading it for future use.

## Example Output

Below is an example of the forecasted Instagram reach for the next 30 days:

Date   
2024-07-01 54000    
2024-07-02 55000    
2024-07-03 56000    
...   
2024-07-30 61000   


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## Contact

If you have any questions, feel free to reach out to me at komikhalils091@gmail.com.



