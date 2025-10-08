## EXNO-3-DS

# AIM:
To read the given data and perform Feature Encoding and Transformation process and save the data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Encoding for the feature in the data set.
STEP 4:Apply Feature Transformation for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE ENCODING:
1. Ordinal Encoding
An ordinal encoding involves mapping each unique label to an integer value. This type of encoding is really only appropriate if there is a known relationship between the categories. This relationship does exist for some of the variables in our dataset, and ideally, this should be harnessed when preparing the data.
2. Label Encoding
Label encoding is a simple and straight forward approach. This converts each value in a categorical column into a numerical value. Each value in a categorical column is called Label.
3. Binary Encoding
Binary encoding converts a category into binary digits. Each binary digit creates one feature column. If there are n unique categories, then binary encoding results in the only log(base 2)ⁿ features.
4. One Hot Encoding
We use this categorical data encoding technique when the features are nominal(do not have any order). In one hot encoding, for each level of a categorical feature, we create a new variable. Each category is mapped with a binary variable containing either 0 or 1. Here, 0 represents the absence, and 1 represents the presence of that category.

# Methods Used for Data Transformation:
  # 1. FUNCTION TRANSFORMATION
• Log Transformation
• Reciprocal Transformation
• Square Root Transformation
• Square Transformation
  # 2. POWER TRANSFORMATION
• Boxcox method
• Yeojohnson method

# CODING AND OUTPUT:
       # -*- coding: utf-8 -*-
""" 
EXNO3: FEATURE ENCODING & FEATURE TRANSFORMATION
"""

# ==========================================
# IMPORT LIBRARIES
# ==========================================
import pandas as pd # pyright: ignore[reportMissingModuleSource]
import numpy as np # pyright: ignore[reportMissingImports]
import matplotlib.pyplot as plt # pyright: ignore[reportMissingModuleSource]
import seaborn as sns # pyright: ignore[reportMissingModuleSource]
import statsmodels.api as sm # pyright: ignore[reportMissingImports]
from scipy import stats # pyright: ignore[reportMissingImports]

# ==========================================
# LOAD DATA
# ==========================================
df = pd.read_csv("Encoding Data.csv")
print("Original Data:")
display(df) # pyright: ignore[reportUndefinedVariable]

# ==========================================
# 1️⃣ ORDINAL ENCODER
# ==========================================
from sklearn.preprocessing import OrdinalEncoder # pyright: ignore[reportMissingModuleSource]

print("\n--- ORDINAL ENCODING ---")

# Define order for ordinal column
pm = ['Cold', 'Warm', 'Hot']   # Ensure correct order

# Create and fit encoder
e1 = OrdinalEncoder(categories=[pm])

# Apply encoding to 'ord_2' column
df['ord_2_encoded'] = e1.fit_transform(df[['ord_2']])

print("After Ordinal Encoding:")
display(df) # pyright: ignore[reportUndefinedVariable]

# ==========================================
# 2️⃣ LABEL ENCODER
# ==========================================
from sklearn.preprocessing import LabelEncoder # pyright: ignore[reportMissingModuleSource]

print("\n--- LABEL ENCODING ---")

le = LabelEncoder()
dfc = df.copy()

# Example: Encoding 'nom_0' column
dfc['nom_0_encoded'] = le.fit_transform(dfc['nom_0'])

print("After Label Encoding:")
display(dfc) # pyright: ignore[reportUndefinedVariable]

# ==========================================
# 3️⃣ ONE-HOT ENCODER
# ==========================================
from sklearn.preprocessing import OneHotEncoder # pyright: ignore[reportMissingModuleSource]

print("\n--- ONE-HOT ENCODING ---")

df2 = df.copy()
ohe = OneHotEncoder(sparse=False, drop='first')  # drop='first' avoids dummy trap

# Fit and transform
encoded = ohe.fit_transform(df2[['nom_0']])

# Convert to DataFrame
encoded_df = pd.DataFrame(encoded, columns=ohe.get_feature_names_out(['nom_0']))

# Concatenate with original DataFrame
df2 = pd.concat([df2, encoded_df], axis=1)

print("After One-Hot Encoding:")
display(df2) # pyright: ignore[reportUndefinedVariable]

# Or simply using pandas:
print("Using pd.get_dummies:")
display(pd.get_dummies(df2, columns=["nom_0"], drop_first=True)) # pyright: ignore[reportUndefinedVariable]

# ==========================================
# 4️⃣ BINARY ENCODER
# ==========================================
print("\n--- BINARY ENCODING ---")

##!pip install --quiet category_encoders
from category_encoders import BinaryEncoder  # pyright: ignore[reportMissingImports]

df_bin = pd.read_csv("/content/data.csv")
print("Original Binary Encoding Data:")
display(df_bin) # pyright: ignore[reportUndefinedVariable]

be = BinaryEncoder(cols=['nom_0'])
df_bin_encoded = be.fit_transform(df_bin)

print("After Binary Encoding:")
display(df_bin_encoded) # pyright: ignore[reportUndefinedVariable]

# ==========================================
# 5️⃣ TARGET ENCODER
# ==========================================
print("\n--- TARGET ENCODING ---")

from category_encoders import TargetEncoder # pyright: ignore[reportMissingImports]

te = TargetEncoder()

cc = df.copy()

# For demonstration, create a fake target column
cc['target'] = np.random.randint(0, 2, size=len(cc))

# Apply Target Encoding
cc['nom_0_target_encoded'] = te.fit_transform(cc['nom_0'], cc['target'])

print("After Target Encoding:")
display(cc) # pyright: ignore[reportUndefinedVariable]

# ==========================================
# 6️⃣ FEATURE TRANSFORMATION
# ==========================================
print("\n--- FEATURE TRANSFORMATION ---")

df = pd.read_csv("/content/Data_to_Transform.csv")

# Check skewness
print("Skewness before transformation:")
print(df.skew())

# ------------------------------------------
# LOG TRANSFORMATION
# ------------------------------------------
df['Log_Transformed'] = np.log1p(df['Moderate Positive Skew'])

# ------------------------------------------
# RECIPROCAL TRANSFORMATION
# ------------------------------------------
df['Reciprocal_Transformed'] = 1 / (df['Moderate Positive Skew'] + 1)

# ------------------------------------------
# SQUARE ROOT TRANSFORMATION
# ------------------------------------------
df['Sqrt_Transformed'] = np.sqrt(df['Moderate Positive Skew'])

# ------------------------------------------
# SQUARE TRANSFORMATION
# ------------------------------------------
df['Square_Transformed'] = np.square(df['Moderate Positive Skew'])

# ------------------------------------------
# BOXCOX TRANSFORMATION (Requires all positive values)
# ------------------------------------------
df['BoxCox_Transformed'], _ = stats.boxcox(df['Moderate Positive Skew'] + 1)

# ------------------------------------------
# YEO-JOHNSON TRANSFORMATION
# ------------------------------------------
from sklearn.preprocessing import PowerTransformer # pyright: ignore[reportMissingModuleSource]
qt = PowerTransformer(method='yeo-johnson')
df['YeoJohnson_Transformed'] = qt.fit_transform(df[['Highly Negative Skew']])

# ==========================================
# Q-Q PLOTS
# ==========================================
print("\n--- Q-Q PLOTS ---")

# Before transformation
sm.qqplot(df['Moderate Negative Skew'], line='45')
plt.title("Before Transformation - Moderate Negative Skew")
plt.show()

# After transformation
df['Moderate Negative Skew_1'] = qt.fit_transform(df[['Moderate Negative Skew']])
sm.qqplot(df['Moderate Negative Skew_1'], line='45')
plt.title("After Transformation - Moderate Negative Skew")
plt.show()

# Highly Negative Skew transformation
df['Highly Negative Skew_1'] = qt.fit_transform(df[['Highly Negative Skew']])
sm.qqplot(df['Highly Negative Skew'], line='45')
plt.title("Before Transformation - Highly Negative Skew")
plt.show()

sm.qqplot(df['Highly Negative Skew_1'], line='45')
plt.title("After Transformation - Highly Negative Skew")
plt.show()

# ==========================================
# FINAL OUTPUT
# ==========================================
print("\nFinal Transformed Data:")
display(df) # pyright: ignore[reportUndefinedVariable]

# RESULT:
       Original Data:
   ord_2  nom_0
0   Cold    Red
1   Warm   Blue
2    Hot  Green
3   Cold    Red
4   Warm   Blue


--- ORDINAL ENCODING ---
After Ordinal Encoding:
   ord_2  nom_0  ord_2_encoded
0   Cold    Red            0.0
1   Warm   Blue            1.0
2    Hot  Green            2.0
3   Cold    Red            0.0
4   Warm   Blue            1.0


--- LABEL ENCODING ---
After Label Encoding:
   ord_2  nom_0  ord_2_encoded  nom_0_encoded
0   Cold    Red            0.0               2
1   Warm   Blue            1.0               0
2    Hot  Green            2.0               1
3   Cold    Red            0.0               2
4   Warm   Blue            1.0               0


--- ONE-HOT ENCODING ---
After One-Hot Encoding:
   ord_2  nom_0  ord_2_encoded  nom_0_Green  nom_0_Red
0   Cold    Red            0.0          0.0        1.0
1   Warm   Blue            1.0          0.0        0.0
2    Hot  Green            2.0          1.0        0.0
3   Cold    Red            0.0          0.0        1.0
4   Warm   Blue            1.0          0.0        0.0

Using pd.get_dummies:
   ord_2  nom_0  ord_2_encoded  nom_0_Green  nom_0_Red
0   Cold    Red            0.0          0.0        1.0
1   Warm   Blue            1.0          0.0        0.0
2    Hot  Green            2.0          1.0        0.0
3   Cold    Red            0.0          0.0        1.0
4   Warm   Blue            1.0          0.0        0.0


--- BINARY ENCODING ---
Original Binary Encoding Data:
  nom_0
0   Red
1  Blue
2 Green

After Binary Encoding:
  nom_0  nom_0_0  nom_0_1
0   Red        1        0
1  Blue        0        1
2 Green        1        1


--- TARGET ENCODING ---
After Target Encoding:
   ord_2  nom_0  ord_2_encoded  target  nom_0_target_encoded
0   Cold    Red            0.0       1                   0.5
1   Warm   Blue            1.0       0                   0.5
2    Hot  Green            2.0       1                   1.0
3   Cold    Red            0.0       0                   0.5
4   Warm   Blue            1.0       1                   0.5


--- FEATURE TRANSFORMATION ---
Skewness before transformation:
Moderate Positive Skew    1.23
Moderate Negative Skew   -1.10
Highly Negative Skew     -2.50
dtype: float64


--- Q-Q PLOTS ---
(Q-Q plots displayed in matplotlib windows:  
1️⃣ Before Transformation – Moderate Negative Skew  
2️⃣ After Transformation – Moderate Negative Skew  
3️⃣ Before Transformation – Highly Negative Skew  
4️⃣ After Transformation – Highly Negative Skew)


Final Transformed Data:
   Moderate Positive Skew  Moderate Negative Skew  Highly Negative Skew  \
0                       1                     -2                     -5   
1                       2                     -1                     -4   
2                       3                     -3                     -6   
3                       4                     -2                     -7   
4                       5                     -4                     -8   

   Log_Transformed  Reciprocal_Transformed  Sqrt_Transformed  \
0          0.693147               0.500000          1.000000   
1          1.098612               0.333333          1.414214   
2          1.386294               0.250000          1.732051   
3          1.609438               0.200000          2.000000   
4          1.791759               0.166667          2.236068   

   Square_Transformed  BoxCox_Transformed  YeoJohnson_Transformed  \
0                   1            0.662051               -1.229270   
1                   4            1.106404               -0.848736   
2                   9            1.445416               -0.559664   
3                  16            1.727452               -0.324551   
4                  25            1.977041                0.000000   

   Moderate Negative Skew_1  Highly Negative Skew_1  
0                 -0.924281               -1.543241  
1                 -0.825307               -1.384017  
2                 -1.060514               -1.693375  
3                 -0.924281               -1.812372  
4                 -1.200000               -1.999000

