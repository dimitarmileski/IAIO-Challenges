import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import GradientBoostingRegressor


train_df = pd.read_csv("train_data.csv")
test_df = pd.read_csv("test_data.csv")

datapoint_ids = test_df["datapointID"]

y_train = train_df["answer"]
train_df = train_df.drop(columns=["answer"])

test_df = test_df.drop(columns=["datapointID"])


def add_word_features(df):
    word = df["word"].astype(str)

    #length
    df["word_length"] = word.str.len()

    # vowel count
    df["vowel_count"] = (
        word.str.lower().str.count("[aeiou]")
    )

    return df


train_df = add_word_features(train_df)
test_df = add_word_features(test_df)



categorical_columns = train_df.select_dtypes(
    include=["object"]
).columns

for col in categorical_columns:
    encoder = LabelEncoder()

    combined = pd.concat(
        [train_df[col], test_df[col]],
        ignore_index=True
    )

    encoder.fit(combined)

    train_df[col] = encoder.transform(train_df[col])
    test_df[col] = encoder.transform(test_df[col])


model = GradientBoostingRegressor(
    random_state=42,
    n_estimators=200,
    learning_rate=0.01,
    max_depth=4,
    min_samples_leaf=5,
    min_samples_split=5,

)

print("train")
model.fit(train_df, y_train)

print("predict")
predictions = model.predict(test_df)
predictions = np.maximum(predictions, 0)

submission = pd.DataFrame({
    "subtaskID": 1,
    "datapointID": datapoint_ids,
    "answer": predictions
})

submission.to_csv(
    "submission.csv",
    index=False
)

print(submission.head())
print("submit")