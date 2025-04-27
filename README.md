# AeroFit Treadmill Customer Analysis

## Business Problem:
AeroFit wants to identify customer characteristics that influence the purchase of its three treadmill models (KP281, KP481, KP781). The goal is to profile customers based on demographics, fitness habits, and income to improve product recommendations and targeted marketing.

## Objective:
Perform exploratory data analysis, detect patterns through probability analysis, and create customer profiles for each treadmill type to drive better sales strategies and customer satisfaction.

Dataset Link : https://github.com/Jyotiprakash01/Aerofit---Descriptive-Statistics-Probability/blob/main/aerofit_treadmill.csv

```python
import pandas as pd

df = pd.read_csv('aerofit_treadmill.csv')
display(df.head())
```
![image](https://github.com/user-attachments/assets/85050344-a608-4ce8-8499-2fcaa4be1b02)

```python
# Check data types of all columns
print("Data types of all columns:\n", df.dtypes)

# Find the number of rows and columns
print("\nNumber of rows and columns:", df.shape)

# Check for missing values and find the number of missing values in each column
print("\nMissing values per column:\n", df.isnull().sum())

df.info()

df.describe(include="all")
```
![image](https://github.com/user-attachments/assets/8b4ccb86-12c0-4717-8668-8e2b859c985e)
![image](https://github.com/user-attachments/assets/52d584ae-8cc6-43e7-b951-97f696279202)

Observations:
- There are no missing values in the data.
- There are 3 unique products in the dataset.
- KP281 is the most frequently purchased product.
- Minimum and maximum age are 18 and 50 years, respectively, with a mean age of 28.79 years.
- 75% of the people are aged 33 years or younger.
- 75% of the people have 16 years of education or less.
- Out of 180 customers, 104 are Male and 76 are Female.
- Most people are partnered (107 out of 180).
- Average treadmill usage is around 3–4 times per week.
- Average fitness level is slightly above 3 on a 5-point scale.
- Mean income is around 53,720, but with a wide range (29,562 to 104,581).
- Standard deviation for Income and Miles is quite high, indicating potential outliers.
- Miles per week also shows large variation (21 to 360 miles), suggesting a few extreme cases.

Implications:
- No missing values allow for smooth analysis without needing data cleaning.
- Focus on KP281 buyers since it is the most popular product — marketing can target similar profiles.
- Younger customers (majority under 33 years) are a key target audience for treadmill marketing.
- Most customers are well-educated (around 16 years of education), so messaging can assume basic health and fitness awareness.
- Marketing should be slightly more male-focused but not ignore the significant female segment (42% are female).
- Since most customers are partnered, promotions like "family fitness packages" or "couple discounts" could be attractive.
- Wide income and miles variation suggests the need to offer products for different budget segments and fitness needs.
- High standard deviation in Income and Miles hints at the importance of offering customized recommendations based on customer profiles (e.g., casual vs. heavy users).

## Univariate Analysis

```python
import matplotlib.pyplot as plt

# Identify continuous variables
continuous_variables = ['Age', 'Education', 'Usage', 'Fitness', 'Income', 'Miles']

# Create boxplots for each continuous variable
plt.figure(figsize=(15, 10))
for i, variable in enumerate(continuous_variables):
    plt.subplot(2, 3, i + 1)
    plt.boxplot(df[variable])
    plt.title(f'Boxplot of {variable}')
plt.tight_layout()
plt.show()

# Describe the continuous variables and check for outliers
for variable in continuous_variables:
    print(f"\nDescriptive statistics for {variable}:\n{df[variable].describe()}")
    print(f"Difference between mean and median for {variable}: {df[variable].mean() - df[variable].median()}")

# Remove outliers by clipping data between 5th and 95th percentiles
df_cleaned = df.copy()
for variable in continuous_variables:
    lower_bound = df[variable].quantile(0.05)
    upper_bound = df[variable].quantile(0.95)
    df_cleaned[variable] = df_cleaned[variable].clip(lower=lower_bound, upper=upper_bound)

display(df_cleaned.head())
```

![image](https://github.com/user-attachments/assets/2b1b5e8b-946f-4f30-b4e7-0cfdaf897a7c)
![image](https://github.com/user-attachments/assets/cdc57548-f0d7-4530-a531-db6912e33cd6)
![image](https://github.com/user-attachments/assets/0731bbfa-2299-4a4b-8350-963e79ca4776)
![image](https://github.com/user-attachments/assets/daa147ea-9044-41de-85b5-8fdfbe145330)
![image](https://github.com/user-attachments/assets/60eec3d0-45b1-4ccd-8d5f-1ce80df90805)


Analyzing the relationship between categorical features (marital status, gender) and the product purchased using countplots.
Then, analyzing the relationship between continuous features (age, education, usage, fitness, income, miles) and the product purchased using boxplots and scatter plots.
Finally, calculating marginal and conditional probabilities.

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Analyze relationship between categorical features and product purchased
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
sns.countplot(x='Product', hue='MaritalStatus', data=df_cleaned)
plt.title('Product Purchase by Marital Status')
plt.subplot(1, 2, 2)
sns.countplot(x='Product', hue='Gender', data=df_cleaned)
plt.title('Product Purchase by Gender')
plt.tight_layout()
plt.show()

# Analyze relationship between continuous features and product purchased
continuous_features = ['Age', 'Education', 'Usage', 'Fitness', 'Income', 'Miles']
plt.figure(figsize=(15, 10))
for i, feature in enumerate(continuous_features):
    plt.subplot(2, 3, i + 1)
    sns.boxplot(x='Product', y=feature, data=df_cleaned)
    plt.title(f'{feature} vs. Product')
plt.tight_layout()
plt.show()

plt.figure(figsize=(15, 10))
for i, feature in enumerate(continuous_features):
    plt.subplot(2, 3, i + 1)
    sns.scatterplot(x=feature, y='Product', data=df_cleaned, hue='Gender')
    plt.title(f'{feature} vs. Product')
plt.tight_layout()
plt.show()

# Calculate marginal probability of each product being purchased
product_counts = pd.crosstab(index=df_cleaned['Product'], columns='count')
product_probabilities = product_counts / product_counts.sum()
print("\nMarginal Probability of Product Purchase:\n", product_probabilities)

# Calculate conditional probabilities
# Example: Probability of a female customer purchasing KP481
female_customers = df_cleaned[df_cleaned['Gender'] == 'Female']
kp481_female = female_customers[female_customers['Product'] == 'KP481'].shape[0]
conditional_probability = kp481_female / female_customers.shape[0]
print(f"\nConditional Probability of Female buying KP481: {conditional_probability:.2f}")

# Calculate other conditional probabilities as needed
```
![image](https://github.com/user-attachments/assets/306d4b8c-a08b-4be7-9d62-f9292464d978)
![image](https://github.com/user-attachments/assets/7e3f47d0-223c-423d-a3ee-6bd034180890)
![image](https://github.com/user-attachments/assets/d5a00321-0669-464c-a99a-636fb7fc9d8a)
![image](https://github.com/user-attachments/assets/31ff465d-891d-4c53-b1b1-166f7c15abf3)
![image](https://github.com/user-attachments/assets/37ec0bed-c54e-4842-94c7-402a39f972d7)
![image](https://github.com/user-attachments/assets/3143daa1-fc83-432d-8d42-48f536e16762)

Calculating the correlation matrix and visualizing it using a heatmap.

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Calculate the correlation matrix
correlation_matrix = df_cleaned[['Age', 'Education', 'Usage', 'Fitness', 'Income', 'Miles']].corr()

# Create the heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix of Numerical Features')
plt.show()
```
![image](https://github.com/user-attachments/assets/91884dd0-350a-4d84-aa39-296b593c80b8)

Calculating the probability of a male customer buying a KP781 treadmill and creating customer profiles for each product.

```python
# Calculate the probability of a male customer buying a KP781 treadmill
male_customers = df_cleaned[df_cleaned['Gender'] == 'Male']
kp781_male = male_customers[male_customers['Product'] == 'KP781'].shape[0]
probability_male_kp781 = kp781_male / male_customers.shape[0]
print(f"Probability of a male customer buying a KP781 treadmill: {probability_male_kp781:.2f}")

# Create customer profiles for each product
products = ['KP281', 'KP481', 'KP781']
for product in products:
    product_customers = df_cleaned[df_cleaned['Product'] == product]
    print(f"\nCustomer Profile for {product}:")
    print(product_customers.describe())
    print("\nKey characteristics:")
    # Example characteristics, adapt based on your findings
    print(f"  Average Age: {product_customers['Age'].mean():.0f}")
    print(f"  Median Income: {product_customers['Income'].median():.0f}")
    # Add more characteristics as needed (e.g., gender distribution, marital status)
```
![image](https://github.com/user-attachments/assets/408356ad-ee1c-455e-aa3d-6d4dd71c2ebd)
![image](https://github.com/user-attachments/assets/2ff866f8-2717-412b-8f1f-e88b97487ea0)



Key Findings:

- Product Popularity: KP281 is the most frequently purchased treadmill model.
- Customer Demographics:
    The majority of customers are under 33 years old. Most customers have 16 years of education or less. There are slightly more male customers than female customers. Most customers are partnered.
- Fitness Habits:
  Average treadmill usage is 3-4 times per week. Average fitness level is slightly above 3 on a 5-point scale.

- Income:
  Average income is around $53,720, but with a wide range.

- Miles:
  Customers run an average of 103 miles per week, but with significant variation.

- Correlation:
  'Usage', 'Fitness', 'Income', and 'Miles' show moderate to strong positive correlations. This suggests that customers who use the treadmill more often tend to have higher fitness levels, higher incomes, and run more miles per week.

- Customer Profiles:
  Each treadmill model attracts a slightly different customer profile based on demographics, fitness habits, and income.

- Probabilities:
  KP281 has the highest marginal probability of purchase. The probability of a female customer purchasing KP481 is approximately 38%. The probability of a male customer purchasing KP781 is approximately 32%.



Insights from Key Findings:
- Targeted Marketing:
  Focus marketing efforts for KP281 on the general customer base, as it's the most popular model.
  Target younger customers (under 33) with marketing campaigns, as they make up a significant portion of the customer base.
  Consider tailoring marketing messages for KP781 towards male customers, given the higher purchase probability within this segment.
  Explore promotions like "family fitness packages" or "couple discounts" to appeal to the partnered customer segment.

- Product Recommendations:
  Leverage the identified customer profiles for each treadmill model to provide personalized product recommendations. For example, recommend KP781 to customers with higher incomes and fitness levels who run more miles per week.
  Since KP281 is the entry-level and most purchased product, consider suggesting add-ons or bundled products to increase sales for this segment.

- Product Development:
  The wide variation in income and miles suggests the need to offer products for different budget segments and fitness needs.
  Consider developing more advanced models with features targeted towards heavy users and fitness enthusiasts.
  Explore the development of fitness tracking integrations to appeal to customers with higher fitness levels.

- Inventory Management:
  Optimize inventory levels for KP281 based on its higher purchase probability to meet demand efficiently.
  Adjust inventory levels for other models based on identified customer segments and purchase patterns.














