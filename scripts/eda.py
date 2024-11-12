import pandas as pd
import matplotlib.pyplot as plt

# Ladda den rensade datan
df = pd.read_csv('data/processed/cleaned_data.csv')

# Exempel på EDA: Histogram
plt.figure(figsize=(10, 6))
df['tip'].hist(bins=30)
plt.title('Fördelning av dricksbelopp')
plt.xlabel('Värde')
plt.ylabel('Frekvens')
plt.savefig('reports/figures/dricks_histogram.png')
plt.show()