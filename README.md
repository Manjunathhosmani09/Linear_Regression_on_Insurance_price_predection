# Linear_Regression_on_Insurance_price_predection
Based on the `Linear_Regression _Insurance_Price.ipynb` notebook you provided, here is the structured description of your project.

### **Project Title: Medical Insurance Cost Prediction using Linear Regression**

This project aims to build a predictive model to estimate the medical insurance costs for individuals based on their demographic and health factors (age, BMI, children, smoking status, and region). The project utilizes statistical analysis to select the most significant features and machine learning to build the final estimator.

-----

### **1. Code Line Descriptions**

Here are the explanations for the critical segments of your code.

**A. Data Loading and Categorical Encoding**

```python
df = pd.read_csv("med1.csv")
df = pd.get_dummies(df, columns=['sex','smoker','region'], drop_first=True)
```

  * **Description:** The dataset is loaded into a pandas DataFrame. Since Linear Regression requires numerical input, categorical variables (`sex`, `smoker`, `region`) are converted into numerical binary flags (0 or 1) using **One-Hot Encoding**. `drop_first=True` is used to avoid the "Dummy Variable Trap" (multicollinearity) by dropping one column from each category.

**B. Outlier Treatment (Winsorization)**

```python
def remove_outlier(col):
    Q1, Q3 = np.percentile(col, [25, 75])
    IQR = Q3 - Q1
    lower_range = Q1 - (1.5 * IQR)
    upper_range = Q3 + (1.5 * IQR)
    return lower_range, upper_range
```

  * **Description:** This custom function calculates the Interquartile Range (IQR). It defines boundaries for outliers. In subsequent lines, values falling outside these boundaries are capped (replaced with the upper/lower limit) rather than removed, preserving the data size while reducing the skewing effect of extreme values.

**C. Correlation Analysis**

```python
sns.heatmap(df.iloc[:, 0:6].corr(), annot=True)
```

  * **Description:** A heatmap is generated to visualize the linear relationships between variables. This highlights that `smoker_yes` has a strong positive correlation with `charges`, indicating it is a primary driver of cost.

**D. Statistical Modeling (OLS)**

```python
import statsmodels.api as sm
X_train = sm.add_constant(X_train)
model = sm.OLS(y_train, X_train).fit()
print(model.summary())
```

  * **Description:** Before using Scikit-Learn, the `statsmodels` library is used to perform Ordinary Least Squares (OLS) regression.
      * `add_constant`: Adds an intercept ($b_0$) to the equation.
      * `summary()`: Generates a detailed statistical report containing P-values, t-statistics, and $R^2$, which are used to determine which features are actually statistically significant.

**E. Feature Selection via P-Values**

```python
X_train4 = X_train.drop(["sex_male", "region_northwest"], axis=1)
```

  * **Description:** Based on the OLS summary, variables with P-values $> 0.05$ (specifically `sex_male` and `region_northwest`) were found to be statistically insignificant. These are dropped to simplify the model without sacrificing accuracy.

**F. Residual Analysis**

```python
stats.shapiro(df_pred["Residuals"])
sms.het_goldfeldquandt(df_pred["Residuals"], X_train4)
```

  * **Description:** Post-modeling diagnostics.
      * `shapiro`: Tests if the residuals (errors) are normally distributed.
      * `het_goldfeldquandt`: Tests for Homoscedasticity (checking if the error variance is constant across all levels of independent variables).

-----

### **2. Structured Project Description**

#### **I. Introduction and Objective**

The cost of medical insurance varies significantly depending on lifestyle and personal attributes. The objective of this project is to develop a **Multiple Linear Regression** model to predict individual medical charges based on features such as age, body mass index (BMI), number of children, and smoking habits.

#### **II. Data Preprocessing and Cleaning**

  * **Encoding:** Categorical variables were transformed into dummy variables.
  * **Memory Optimization:** Data types were converted to `uint8` to optimize memory usage.
  * **Duplicate Removal:** Duplicate records were identified and removed to ensure data integrity.
  * **Outlier Handling:** Boxplots revealed outliers in the `bmi` and `charges` columns. The IQR method was applied to cap these outliers, preventing them from disproportionately influencing the regression line.

#### **III. Exploratory Data Analysis (EDA)**

Correlation analysis provided early insights:

  * **Smoking** had the highest correlation with medical charges, suggesting smokers pay significantly higher premiums.
  * **Age** and **BMI** showed moderate positive correlations.
  * **Region** and **Gender** appeared to have minimal impact on the costs.

#### **IV. Model Building and Feature Selection**

The project utilized a "Backward Elimination" approach using `statsmodels`:

1.  **Initial Model:** Built using all available features. The resulting $R^2$ was **0.752**.
2.  **Statistical pruning:** The summary table showed that `sex_male` (p=0.070) and `region_northwest` (p=0.373) were not statistically significant ($\alpha = 0.05$).
3.  **VIF Check:** Variance Inflation Factor was calculated to ensure no severe multicollinearity existed between predictors.
4.  **Refined Model:** A second model was built excluding the insignificant features. The $R^2$ remained **0.751**, proving that the dropped features added complexity but no predictive value.

#### **V. Model Diagnostics**

To validate the reliability of the Linear Regression, assumptions were tested:

  * **Residual Plot:** Showed a pattern, suggesting non-linearity or heteroscedasticity (variance of errors varies with the value of the independent variable).
  * **Shapiro-Wilk Test:** Confirmed that residuals are not perfectly normally distributed, which is common in financial data involving costs.

### **Structured Interpretation of the Regression Model**

#### **1. The Formula**
$$\text{Cost} = -8392.36 + 222.79(\text{Age}) + 259.66(\text{BMI}) + 516.25(\text{Children}) + 19978.40(\text{Smoker}) - 904.92(\text{SE}) - 723.91(\text{SW})$$

---

#### **2. Coefficient Analysis (The "Price Tags")**

* **Intercept ($-8392.36$):**
    * **Meaning:** This is the mathematical starting point of the prediction line when all other values are 0.
    * **Interpretation:** Since insurance cost cannot be negative, this indicates that the linear model relies heavily on the features (like Age and BMI) to pull the price up to a realistic positive number.

* **Age ($+222.79$):**
    * **Impact:** Moderate Increase.
    * **Interpretation:** For every **1 year** increase in age, the insurance cost increases by approximately **\$222.79**, assuming all other factors stay the same. Older individuals pay more.

* **BMI ($+259.66$):**
    * **Impact:** Moderate Increase.
    * **Interpretation:** For every **1 point** increase in BMI, the cost rises by **\$259.66**. This confirms that higher body mass is associated with higher medical costs.

* **Children ($+516.25$):**
    * **Impact:** Significant Increase.
    * **Interpretation:** Each additional dependent/child adds roughly **\$516.25** to the insurance premium.

* **Smoker\_Yes ($+19978.40$):**
    * **Impact:** **Massive Increase (Primary Driver)**.
    * **Interpretation:** This is the most critical factor. If a person is a smoker (value = 1), their insurance cost jumps by nearly **\$20,000** compared to a non-smoker. This single variable essentially doubles or triples the cost for many individuals.

* **Region_Southeast ($-904.92$) & Region_Southwest ($-723.91$):**
    * **Impact:** Slight Decrease.
    * **Interpretation:** Living in the Southeast or Southwest is associated with slightly lower costs (reductions of \$905 and \$724 respectively) compared to the baseline region (likely Northeast, which was dropped during dummy encoding).

---

#### **3. Business Insights & Conclusion**

1.  **Smoking is the Deal-Breaker:** The \$19,978 premium for smokers dwarfs all other variables. A 20-year-old smoker will likely pay more than a 50-year-old non-smoker.
2.  **Age and Health Matter:** While not as explosive as smoking, the steady accumulation of costs from Age (\$222/year) and BMI (\$260/point) creates a significant price gap between young/healthy and older/at-risk individuals.
3.  **Regional Differences are Minor:** While statistically relevant, the location discounts (< \$1000) are relatively small compared to lifestyle choices.

This equation provides a clear, quantifiable method for the insurance company to assess risk and set premiums.
