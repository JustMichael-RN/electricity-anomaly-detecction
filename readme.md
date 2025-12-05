# ⚡ Electricity Anomaly Detection System

An intelligent system for detecting unusual patterns in household electricity consumption using machine learning.

## 🎯 Features

- Real-time anomaly detection in electricity consumption
- Interactive web interface built with Streamlit
- Batch analysis for multiple records
- Isolation Forest algorithm for accurate detection
- Visual insights and anomaly scoring

## 📊 Dataset

This project uses the **Individual Household Electric Power Consumption Dataset** available on Kaggle:
- [Download Dataset](https://www.kaggle.com/datasets/uciml/electric-power-consumption-data-set)
- Size: ~20MB (2 million measurements)
- Features: Active/reactive power, voltage, intensity, and sub-metering data

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- pip

### Installation

1. Clone the repository:
```bash
git clone https://github.com/JustMichael-RN/electricity-anomaly-detection.git
cd electricity-anomaly-detection
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download the dataset:
   - Go to [Kaggle Dataset](https://www.kaggle.com/datasets/uciml/electric-power-consumption-data-set)
   - Download `household_power_consumption.txt`
   - Create a `data/` folder and place the file there

### Training the Model

Run the training script to create your anomaly detection model:
```bash
python train_model.py
```

This will:
- Load and preprocess the data
- Train an Isolation Forest model
- Save the model to `model/anomaly_model.pkl`

### Running the App

Start the Streamlit application:
```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## 📁 Project Structure
```
electricity-anomaly-detection/
├── train_model.py          # Training script
├── app.py                  # Streamlit application
├── requirements.txt        # Python dependencies
├── .gitignore             # Git ignore rules
├── README.md              # This file
├── model/                 # Saved model directory
│   └── anomaly_model.pkl
└── data/                  # Dataset directory (not in git)
    └── household_power_consumption.txt
```

## 🔧 How It Works

1. **Data Processing**: The system loads electricity consumption data and extracts key features
2. **Feature Engineering**: Creates time-based features (hour, day, month) for better detection
3. **Model Training**: Uses Isolation Forest to learn normal consumption patterns
4. **Anomaly Detection**: Identifies unusual patterns that deviate from the norm
5. **Visualization**: Provides an intuitive interface for analysis

## 📈 Usage

### Single Prediction
- Enter consumption values manually
- Get instant anomaly detection results
- View anomaly scores

### Batch Analysis
- Upload a CSV file with multiple records
- Analyze all records at once
- Download results with anomaly flags

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is open source and available under the MIT License.

## 👤 Author

**Michael RN**
- GitHub: [@JustMichael-RN](https://github.com/JustMichael-RN)

## 🙏 Acknowledgments

- Dataset from UCI Machine Learning Repository
- Built with Streamlit, scikit-learn, and Python