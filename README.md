# 📊 Strategic Sales Forecasting System
**AI-Powered Time Series Forecasting with Facebook Prophet**

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Prophet](https://img.shields.io/badge/Prophet-1.1%2B-green)
![Flask](https://img.shields.io/badge/Flask-2.3%2B-red)
![License](https://img.shields.io/badge/License-MIT-yellow)

**A production-ready sales forecasting application with modular pipeline architecture and interactive web interface**

[Overview](#-project-overview) • [Key Findings](#-key-findings--eda-insights) • [Architecture](#-what-makes-this-project-different) • [Installation](#-installation--setup) • [Usage](#-how-to-run-the-application)

---

## 🎯 Project Overview

The **Strategic Sales Forecasting System** is an end-to-end machine learning application that predicts restaurant sales up to 6 months in advance using Facebook Prophet. Unlike traditional forecasting projects, this application follows **industry-standard practices** with a **modular pipeline architecture**, making it production-ready, scalable, and easily maintainable.

### 🏢 Business Problem

Restaurant chains face critical challenges:
- 📉 **Unpredictable Demand**: Difficulty in anticipating sales fluctuations
- 💰 **Inventory Management**: Overstocking or understocking issues
- 👥 **Staff Scheduling**: Inefficient workforce allocation
- 📊 **Revenue Planning**: Inaccurate financial forecasting
- 🎯 **Strategic Planning**: Lack of data-driven insights for menu optimization

### 💡 Our Solution

This project delivers:
- ✅ **Accurate Predictions**: 75.5% accuracy using advanced Prophet algorithm
- ✅ **Interactive Dashboard**: Beautiful dark-themed web interface with real-time forecasting
- ✅ **Modular Pipeline**: Separate components for training and prediction
- ✅ **Production Ready**: Automated workflows with proper error handling and logging
- ✅ **Visual Analytics**: Prophet-generated forecast plots with seasonality decomposition

---
# EDA Findings: Restaurant Sales Analysis

This file summarizes the key problems addressed and findings discovered during the Exploratory Data Analysis (EDA) of the restaurant sales dataset (from `1.EDA_restaurant_sales_data.ipynb`).

---

## Key Problems & Findings

1.  **Problem:** Are promotions primarily driving sales volume or overall profit?
    **Finding:** Promotions are a volume strategy. They boost average quantity sold by 92% (from 251 to 482 units) but only increase average profit by 19% (from $2,418 to $2,889).

2.  **Problem:** How do special events impact sales differently than regular promotions?
    **Finding:** Special events are a profit strategy. They increase average profit by 35% (vs. 19% for promotions), suggesting a focus on higher-margin items on those days.

3.  **Problem:** Which meal type is the most profitable, and which is the most volume-driven?
    **Finding:** Lunch generates the highest absolute profit. Breakfast, however, is the most volume-driven (highest quantity sold) but does not yield the highest profit.

4.  **Problem:** How does customer behavior change between weekdays and weekends?
    **Finding:** Demand for all meal types increases on weekends, but Breakfast sees the most significant surge, with average units sold jumping from ~405 on weekdays to ~596 on weekends.

5.  **Problem:** Does adverse weather (Rainy/Cloudy) negatively impact all restaurant types?
    **Finding:** No. While Casual and Fine Dining see the highest sales on Sunny days, Kopitiams show a reverse trend, performing best during Cloudy and Rainy weather.

6.  **Problem:** Which restaurant segments are volume-driven versus price-driven?
    **Finding:** Kopitiams and Food Stalls are volume-driven, with the lowest average prices (around $11) and the highest average quantities sold (~332 and ~312 units, respectively).

7.  **Problem:** Which segment is the premium, high-margin category?
    **Finding:** Fine Dining is the premium segment, commanding the highest average price (approx. $39) but selling the lowest average quantity (approx. 200 units).

8.  **Problem:** Are all menu items equally profitable?
    **Finding:** The menu is clearly split between "Profit Drivers" (Western/Premium items like Mushroom Soup and Chicken Chop with >81% margins) and "Volume Drivers" (Traditional items like Laksa and Roti Canai with <70% margins).

9.  **Problem:** Is sales performance consistent across all 50 restaurants?
    **Finding:** No, there is extreme profit disparity. Top performers (e.g., R006) generated over $1.1 million in profit, while bottom performers (e.g., R017) earned less than $250,000.

10. **Problem:** Do different restaurant types specialize in certain menu items?
    **Finding:** Yes, there is clear item exclusivity. "Beef Rendang" is sold almost exclusively in Fine Dining, while items like "Laksa" and "Char Kway Teow" are staples of Food Stalls and Kopitiams.

---

## ✨ Key Features

### 🎯 Machine Learning Pipeline
- ✅ **Data Ingestion**: Automated CSV loading and date parsing
- ✅ **Model Training**: Prophet with optimized parameters
- ✅ **Cross-Validation**: 3-fold time series validation
- ✅ **Prediction Pipeline**: Inference with proper preprocessing

### 🎨 Modern Web Interface
- ✅ **Dark Theme**: Vibrant design with animated gradients
- ✅ **Interactive Forms**: Quick select buttons (7 days, 1 month, 3 months, 6 months)
- ✅ **Visual Forecasts**: Prophet forecast + components plots
- ✅ **Summary Cards**: Key metrics with hover effects
- ✅ **Responsive Design**: Mobile-friendly layout

### 📊 Prophet Visualizations
- ✅ **Forecast Plot**: Historical data + future predictions with confidence intervals
- ✅ **Components Plot**: 
  - Overall Trend
  - Weekly Seasonality
  - Yearly Seasonality
- ✅ **Key Insights**: Automated analysis of predictions

---

## 📁 Project Structure

```
Strategic Sales Forecasting/
│
├── 📄 app.py                          # Flask application entry point
├── 📄 requirements.txt                # Python dependencies
├── 📄 setup.py                        # Package installation script
├── 📄 README.md                       # This file
│
├── 📂 src/                            # Source code directory
│   ├── 📄 __init__.py
│   ├── 📄 exception.py                # Custom exception handling
│   ├── 📄 logger.py                   # Logging configuration
│   ├── 📄 utils.py                    # Utility functions
│   │
│   ├── 📂 components/                 # ML pipeline components
│   │   ├── 📄 __init__.py
│   │   ├── 📄 data_ingestion.py       # Data loading & processing
│   │   ├── 📄 data_transformation.py  # Feature engineering
│   │   └── 📄 model_trainer.py        # Prophet model training
│   │
│   └── 📂 pipeline/                   # Training & prediction pipelines
│       ├── 📄 __init__.py
│       ├── 📄 predict_pipeline.py     # Prediction pipeline for inference
│       └── 📄 train_pipeline.py       # Automated training pipeline
│
├── 📂 templates/                      # HTML templates for Flask
│   ├── 📄 index.html                  # Landing page (dark theme)
│   └── 📄 home.html                   # Prediction form & results
│
├── 📂 artifacts/                      
│   ├── 📄 model.pkl                   
│   ├── 📄 train_cleaned.csv           
│   └── 📄 train.csv
│   └── 📄 test.csv
│   └── 📄 test_clened.csv
│   └── 📄 data.csv
│
│
├── 📂 notebooks/                      # Jupyter notebooks
│   └── 📓 1.EDA_restaurant_sales_data.ipynb  # EDA & training notebook
│   └── 📓 model_training_all_restaurent.ipynb # experiments
│   └── 📓    model_training_by_restaurent_type.ipynb # experiments
│   ├── 📂 dataset/                        # Raw data files
│       └── 📄 restaurant_sales_data.csv   # Historical sales data
│
└── 📂 logs/                           # Application logs
    └── 📄 app_*.log                   # Timestamped log files
```

### 📌 Key Directories Explained

| Directory | Purpose |
|-----------|---------|
| `src/components/` | Core ML pipeline components (ingestion, transformation, training) |
| `src/pipeline/` | End-to-end pipelines for training and prediction |
| `artifacts/` | Stores trained models and cleaned datasets |
| `models/` | Versioned Prophet models with timestamps |
| `results/` | Forecast outputs in CSV format |
| `templates/` | Dark-themed HTML files for web interface |
| `notebooks/` | Jupyter notebooks for EDA and experimentation |
| `dataset/` | Raw restaurant sales data |
| `logs/` | Application logs for debugging |

---

## 🚀 Installation & Setup

### Prerequisites
- Python 3.8 or higher
- pip package manager
- Virtual environment (recommended)

### Step 1: Clone the Repository
```bash
# Clone the project
git clone https://github.com/yourusername/strategic-sales-forecasting.git

# Navigate to project directory
cd strategic-sales-forecasting
```

### Step 2: Create Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
# Upgrade pip
pip install --upgrade pip

# Install all required packages
pip install -r requirements.txt

```

---

## 🎓 How to Train the Model

### Using Training Pipeline 
```bash
# Run the complete automated training pipeline
python -m src.pipeline.train_pipeline

```
### Verify Training Success
```bash
# Windows
dir artifacts
dir models

# macOS/Linux
ls artifacts/
ls models/
```

## 🌐 How to Run the Application

### Step 1: Ensure Model is Trained
```bash
# Verify model exists
# Windows
if exist artifacts\model.pkl (echo Model found!) else (echo Train model first!)

# macOS/Linux
[ -f artifacts/model.pkl ] && echo "Model found!" || echo "Train model first!"
```

### Step 2: Start the Flask Application
```bash
# Run the Flask app
python app.py
```

**Expected Output:**
```
 * Serving Flask app 'app'
 * Debug mode: on
 * Running on all addresses (0.0.0.0)
 * Running on http://127.0.0.1:5000
 * Running on http://192.168.x.x:5000
Press CTRL+C to quit
```

### Step 3: Access the Application
Open your browser at: **http://localhost:5000**

### Step 4: Use the Application

1. **🏠 Landing Page**
   - View model information (75.5% accuracy, 180 days max)
   - See feature highlights
   - Click "🚀 Get Started"

2. **📊 Prediction Form**
   - Enter forecast period (1-180 days)
   - Use quick select: 7 Days, 2 Weeks, 1 Month, 2 Months, 3 Months, 6 Months
   - Click "🔮 Generate Forecast"

3. **📈 Results Dashboard**
   - **Summary Cards**: Total sales, average daily, peak/low days
   - **Forecast Plot**: Historical + future predictions with red forecast line
   - **Components Plot**: Trend, weekly, and yearly patterns
   - **Key Insights**: Automated analysis

### Stop the Application
Press `CTRL + C` in the terminal

---


## 📝 Quick Start Commands

```bash
# Complete setup and run (copy-paste friendly)

# 1. Setup Environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # macOS/Linux
pip install -r requirements.txt


# 2. Train Model
python -m src.pipeline.train_pipeline

# 3. Run Application
python app.py

# 4. Open browser
# Navigate to: http://localhost:5000

# 5. Stop application
# Press CTRL + C
```
---

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

---

## 👨‍💻 Author

**Dipanjan Santra**

- 🌐 GitHub: [@Dipanjan](https://github.com/Dipanjan777777)
- 💼 LinkedIn: [Dipanjan](https://www.linkedin.com/in/dipanjan-santra/)
- 📧 Email: dipanjansantra2019@gmail.com

---

<div align="center">

**Made with ❤️ using Python, Prophet, and Flask**

⭐ **If you find this project useful, please give it a star!** ⭐

</div>
