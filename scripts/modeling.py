import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder

# Skapa instans av label encoder
label_encoder = LabelEncoder()

# Ladda datasetet
df = pd.read_csv('data/processed/cleaned_data.csv')

# Förbered data
# 1. Konvertera kategoriska kolumner till numeriska värden
df['sex'] = df['sex'].map({'Male': 0, 'Female': 1})   # 0 för "Male" och 1 för "Female"
df['smoker'] = df['smoker'].map({'No': 0, 'Yes': 1})   # 0 för "No" och 1 för "Yes"
df['day'] = label_encoder.fit_transform(df['day'])     # Kodar "day" med unika nummer

# 2. Välj features (X) och label (y)
X = df[['total_bill', 'sex', 'smoker', 'day']]
y = df['tip']

# Dela upp data i tränings- och testset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

# Träna modellen
model = LinearRegression()
model.fit(X_train, y_train)

# Gör förutsägelser och utvärdera modellen
y_pred = model.predict(X_test)

mse = metrics.mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5  # Root Mean Squared Error
mae = metrics.mean_absolute_error(y_test, y_pred)
r2 = metrics.r2_score(y_test, y_pred)

print("\n### Model Evaluation ###")
print(f"Mean Squared Error (MSE): {mse:.2f} - Genomsnittlig kvadratisk avvikelse, visar hur långt ifrån de verkliga värdena förutsägelserna är i kvadrat.")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f} - Roten av MSE, lättare att tolka eftersom det är i samma enhet som dricksbeloppet.")
print(f"Mean Absolute Error (MAE): {mae:.2f} - Genomsnittliga absoluta avvikelsen, visar den genomsnittliga skillnaden mellan förutsägelser och verkliga värden i samma enhet som dricks.")
print(f"R-squared (R2): {r2:.2f} - Förklarad varians, visar hur mycket av variationen i dricks som modellen kan förklara (0 till 1).")