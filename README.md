# -Salary-Prediction-using-Ensemble-Learning-


### 1. Importing Tools  
You bring in Python libraries:
- **pandas** → to read and handle data.  
- **train_test_split** → to split data into training and testing parts.  
- **RandomForestRegressor** → the machine learning model used to predict salaries.  
- **mean_absolute_error, r2_score** → to check how good the predictions are.  

**2.Reading the Data** 

data = pd.read_csv("Salary_Data.csv")

This loads your salary dataset from your computer.


### 3. Choosing Features and Target  

X = data[["YearsExperience", "Age", "EducationLevel", "JobLevel"]]
y = data["Salary"]
- **X (features):** things that affect salary → experience, age, education, job level.  
- **y (target):** the actual salary you want to predict.  

Education and JobLevel are already given as numbers (Diploma=1, Bachelor=2, etc.).



### 4. Splitting Data  

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

- 80% of data is used to **train** the model.  
- 20% is used to **test** how well it works.  



### 5. Training the Model  
```python
model = RandomForestRegressor(n_estimators=100)
model.fit(X_train, y_train)
```
This builds a **Random Forest model** with 100 decision trees to learn the relationship between features and salary.

---

### 6. Making Predictions 
predictions = model.predict(X_test)
The model predicts salaries for the test data.

### 7. Checking Accuracy  
```python
print("MAE:", mean_absolute_error(y_test, predictions))
print("R2 Score:", r2_score(y_test, predictions))
```
- **MAE (Mean Absolute Error):** average difference between predicted and actual salaries.  
- **R² Score:** how well the model explains salary variation (closer to 1 = better).  

---

### 8. Predicting for a New Employee  
new_employee = pd.DataFrame([[5, 30, 3, 3]], 
    columns=["YearsExperience", "Age", "EducationLevel", "JobLevel"])
model.predict(new_employee)
This predicts salary for a new employee with:
- 5 years experience  
- Age 30  
- Master’s degree (3)  
- Senior level job (3)  

The model will output the **expected salary** for this profile.

```
