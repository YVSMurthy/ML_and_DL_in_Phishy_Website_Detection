import pandas as pd
import os

# Get the absolute path to the current directory (project root)
current_dir = os.path.dirname(os.path.abspath(__file__))
print(current_dir)

# Read the original CSV file
original_file = os.path.join(current_dir, 'UCI_dataset1.csv')
data = pd.read_csv(original_file)

# Select the columns 'URL' and 'label'
selected_columns = data[['URL', 'label']]

# Save the selected columns to a new CSV file
new_file = os.path.join(current_dir, 'website_list2.csv')
selected_columns.to_csv(new_file, index=False)

print(f"Selected columns saved to {new_file}")
