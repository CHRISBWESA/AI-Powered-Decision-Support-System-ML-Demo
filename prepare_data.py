import pandas as pd # pyright: ignore[reportMissingModuleSource]
from sklearn.model_selection import train_test_split # pyright: ignore[reportMissingModuleSource]

# 1️⃣ Load dataset
df = pd.read_csv("StudentsPerformance.csv")

# 2️⃣ Create average score
df['average_score'] = (df['math score'] + df['reading score'] + df['writing score']) / 3

# 3️⃣ Create Pass/Fail column (>=60 pass)
df['pass_fail'] = df['average_score'].apply(lambda x: 1 if x >= 60 else 0)

# 4️⃣ Encode categorical features
df_encoded = pd.get_dummies(df, columns=[
    'gender', 
    'race/ethnicity', 
    'parental level of education', 
    'lunch', 
    'test preparation course'
])

# 5️⃣ Prepare feature sets
X = df_encoded.drop(['math score','reading score','writing score','average_score','pass_fail'], axis=1)
y_class = df_encoded['pass_fail']      # Classification target
y_reg = df_encoded['average_score']    # Regression target

# 6️⃣ Split datasets for supervised learning
X_train_class, X_test_class, y_train_class, y_test_class = train_test_split(X, y_class, test_size=0.2, random_state=42)
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X, y_reg, test_size=0.2, random_state=42)

print("✅ Data is cleaned, encoded, and ready for ML algorithms!")
print("Features shape:", X.shape)
print("Classification target shape:", y_class.shape)
print("Regression target shape:", y_reg.shape)
