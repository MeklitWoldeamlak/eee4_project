import json

# Open the JSON file and load its contents
with open('Epsilon.json', 'r') as f:
    data = json.load(f)

# Modify the contents of the JSON object
data['class1.csv'] = 1.0

# Write the updated JSON object back to the file
with open('Epsilon.json', 'w') as f:
    json.dump(data, f)