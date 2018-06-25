import pandas as pd
from neural_network import NeuralNetwork

def load_data():
    # Load CSV file into a dataframe
    df = pd.read_csv("iris.csv").sample(frac = 1)
    # Execute any necessary transformations
    transform_features(df)
    # Split data into 60-40 ratio
    split = round(len(df) * .60)
    return (pd.DataFrame(df.iloc[:split, :]), 
            pd.DataFrame(df.iloc[split:, :]))
    
def transform_features(df):
    # Convert labels from strings to numbers
    classes = list(df["CL"].unique())
    df["Y"] = df["CL"].map(lambda row: classes.index(row))


# Fetch data
train_df, test_df = load_data()

# Initialize network and features
nn = NeuralNetwork(layers=(4, 50, 30, 3), reg=1, alpha=3, iter=250)
features = ["SL", "SW", "PL", "PW"]

# Fit the network
nn.fit(train_df[features], train_df["Y"])

# Make predictions
test_df["P"] = nn.predict(test_df[features])

# Calculate accuracy
accuracy = len(test_df[test_df["P"] == test_df["Y"]]) / len(test_df)
print(accuracy)