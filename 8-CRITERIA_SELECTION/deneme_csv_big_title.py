import pandas as pd

# Original DataFrame
df = pd.DataFrame({
    'A': [1, 2],
    'B': [3, 4],
    'C': [5, 6],
    'D': [7, 8],
    'E': [9, 10]
})

# Define grouping
groups = {
    'A': 'vowel',
    'E': 'vowel',
    'B': 'non_vowel',
    'C': 'non_vowel',
    'D': 'non_vowel'
}

# Create MultiIndex columns
df.columns = pd.MultiIndex.from_tuples([
    (groups[col], col) for col in df.columns
])

print(df)

df.to_csv("output.csv", index=False)