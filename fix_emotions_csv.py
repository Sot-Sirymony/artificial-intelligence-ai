import pandas as pd

input_path = 'dataset/emotions.csv'
output_path = 'dataset/emotions.csv'

df = pd.read_csv(input_path)
df['emotion'] = df['emotion'].astype(str).str.strip()
df.to_csv(output_path, index=False)

print('✅ Cleaned emotions.csv and removed any whitespace from emotion labels.') 