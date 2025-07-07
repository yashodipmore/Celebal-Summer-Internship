# 📚 Student Exam Score Predictor Web Application

A comprehensive Streamlit web application that predicts student exam scores based on study hours, previous exam performance, and attendance using Machine Learning.

## 🌟 Features

- **🔮 Score Prediction**: Input student data and get instant exam score predictions
- **📊 Data Analysis**: Interactive visualizations of the dataset
- **📈 Model Insights**: Understanding model performance and feature importance
- **🎯 User-Friendly Interface**: Modern, responsive design with intuitive navigation

## 🚀 Quick Start

### Prerequisites

- Python 3.7 or higher
- pip (Python package manager)

### Installation

1. **Clone or download this repository**
   ```bash
   git clone <repository-url>
   cd "Student Exam Score Prediction"
   ```

2. **Install required packages**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Streamlit application**
   ```bash
   streamlit run app.py
   ```

4. **Open your browser**
   - The application will automatically open at `http://localhost:8501`
   - If it doesn't open automatically, navigate to the URL manually

## 📁 Project Structure

```
Student Exam Score Prediction/
├── app.py                     # Main Streamlit application
├── requirements.txt           # Python dependencies
├── README.md                 # This file
├── student_exam_data.csv     # Training dataset
├── exam_score_predictor.pkl  # Trained ML model
├── scaler.pkl               # Feature scaler
├── Untitled1.ipynb         # Training notebook
└── images/                  # (Optional) Screenshots
```

## 🎯 How to Use

### 1. Home Page 🏠
- Overview of the project and features
- Navigation guide

### 2. Prediction Page 🔮
- Enter student information:
  - **Hours Studied**: Number of hours dedicated to studying (0-20)
  - **Previous Exam Score**: Student's previous exam performance (0-100)
  - **Attendance**: Percentage of classes attended (0-100%)
- Click "Predict Exam Score" to get instant results
- View prediction with performance interpretation

### 3. Data Analysis Page 📊
- Explore the training dataset
- View statistical summaries
- Interactive visualizations:
  - Score distribution
  - Feature correlations
  - Scatter plots showing relationships
  - Pairwise feature analysis

### 4. Model Insights Page 📈
- Understanding model coefficients
- Feature importance analysis
- Sample predictions for different student types
- Interactive sensitivity analysis

## 🔧 Technical Details

### Model Information
- **Algorithm**: Linear Regression
- **Features**: Hours Studied, Previous Exam Score, Attendance
- **Target**: Exam Score (0-100)
- **Preprocessing**: StandardScaler for feature normalization

### Performance Metrics
The model achieves good performance on the test dataset with appropriate R² score and low error rates.

## 📊 Dataset Description

The dataset contains student information with the following features:

| Feature | Description | Range |
|---------|-------------|-------|
| Hours_Studied | Number of hours spent studying | 0-20 hours |
| Previous_Exam_Score | Score from previous examination | 0-100 |
| Attendance | Percentage of classes attended | 0-100% |
| Exam_Score | Target variable - Current exam score | 0-100 |

## 🛠️ Customization

### Adding New Features
1. Update the model training notebook
2. Retrain and save the model
3. Modify the prediction form in `app.py`
4. Update the scaler with new features

### Styling
- Modify the CSS in the `st.markdown()` sections
- Change colors, fonts, and layout in the style definitions
- Add new themes or color schemes

## 🚨 Troubleshooting

### Common Issues

1. **ModuleNotFoundError**
   ```bash
   pip install -r requirements.txt
   ```

2. **Model file not found**
   - Ensure `exam_score_predictor.pkl` and `scaler.pkl` are in the same directory
   - Run the training notebook to regenerate models

3. **CSV file not found**
   - Ensure `student_exam_data.csv` is in the same directory
   - Check file name spelling and path

4. **Port already in use**
   ```bash
   streamlit run app.py --server.port 8502
   ```

## 📝 Future Enhancements

- [ ] Add more ML algorithms (Random Forest, XGBoost)
- [ ] Model comparison functionality
- [ ] Batch prediction from CSV upload
- [ ] Export predictions to CSV
- [ ] Advanced visualization options
- [ ] User authentication
- [ ] Database integration
- [ ] Model retraining interface

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📞 Support

If you encounter any issues or have questions:
1. Check the troubleshooting section
2. Review the requirements.txt for dependency versions
3. Ensure all files are in the correct directory

## 📄 License

This project is open source and available under the MIT License.

---

**Happy Predicting! 🎓📈**
