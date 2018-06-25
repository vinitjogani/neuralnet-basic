import pandas as pd
from neural_network import NeuralNetwork
from sklearn.preprocessing import StandardScaler

def load_data():
    # Load CSV file into a dataframe
    df = pd.read_csv("iris.csv").sample(frac = 1, random_state=123)
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
nn = NeuralNetwork(layers=(4, 50, 50, 3), reg=3e-2, alpha=3, batch_size=90, epochs=200)
features = ["SL", "SW", "PL", "PW"]

# Scale data
scaler = StandardScaler()
train_data = scaler.fit_transform(train_df[features].as_matrix())

# Fit the network
nn.fit(train_data, list(train_df["Y"]))

# Make predictions
test_df["P"] = nn.predict(scaler.transform(test_df[features]))

print(test_df)

# Calculate accuracy
accuracy = len(test_df[test_df["P"] == test_df["Y"]]) / len(test_df)
print(accuracy)