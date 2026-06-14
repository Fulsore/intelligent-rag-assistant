# import pandas as pd

# df = pd.read_csv(
#     "rag/datasets/customer_support.csv"
# )

# print(df.columns)

# df = df.drop_duplicates()

# df = df.dropna()

# documents = []

# for row in df.to_dict(orient="records"):

#     text = ""

#     for key, value in row.items():

#         text += f"{key}: {value}\n"

#     documents.append(text)

# clean_df = pd.DataFrame({
#     "text": documents
# })

# clean_df.to_csv(
#     "rag/cleaned_data/final_cleaned_data.csv",
#     index=False
# )

# print(clean_df.head())

# print("Cleaning completed")