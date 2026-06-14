# from datasets import load_dataset
# import pandas as pd
# import os


# os.makedirs("rag/datasets", exist_ok=True)


# datasets_to_download = [

#     {
#         "hf_name": "MakTek/Customer_support_faqs_dataset",
#         "output_name": "maktek.csv"
#     },

#     {
#         "hf_name": "raushan298744/Customer_support_faqs_dataset",
#         "output_name": "raushan.csv"
#     },

#     {
#         "hf_name": "bitext/Bitext-retail-ecommerce-llm-chatbot-training-dataset",
#         "output_name": "bitext.csv"
#     },

#     {
#         "hf_name": "Phuc-200/Bitext-retail-ecommerce-llm-chatbot-training-dataset",
#         "output_name": "phuc.csv"
#     }
# ]


# all_dataframes = []


# for item in datasets_to_download:

#     try:

#         print(f"\nDownloading: {item['hf_name']}")

#         dataset = load_dataset(item["hf_name"])

#         train_df = pd.DataFrame(dataset["train"])

#         save_path = f"rag/datasets/{item['output_name']}"

#         train_df.to_csv(save_path, index=False)

#         print(f"Saved: {save_path}")

#         train_df["source"] = item["hf_name"]

#         all_dataframes.append(train_df)

#     except Exception as e:

#         print(f"FAILED: {item['hf_name']}")
#         print(e)


# print("\nMerging datasets...")


# merged_df = pd.concat(
#     all_dataframes,
#     ignore_index=True
# )


# merged_save_path = "rag/datasets/final_merged_dataset.csv"

# merged_df.to_csv(
#     merged_save_path,
#     index=False
# )

# print(f"\nFinal merged dataset saved at:")
# print(merged_save_path)

# print("\nDataset Shape:")
# print(merged_df.shape)

# print("\nColumns:")
# print(merged_df.columns)

# print("\nPreview:")
# print(merged_df.head())