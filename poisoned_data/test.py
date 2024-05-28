import json

# Open the auth_right.json file
with open("auth_right.json") as file:
    data = json.load(file)

# Add an "id" variable to each JSON object
for i, record in enumerate(data):
    record["id"] = i

# Print the updated data
print(data)

with open("auth_right2.json", "w") as f:
    json.dump(data, f, indent=4)
