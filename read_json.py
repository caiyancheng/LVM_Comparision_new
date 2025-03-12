import json

json_file_name = r'E:\Rainbow_Page\projects\elaTCSF/web_jndData.json'
with open(json_file_name, 'r') as fp:
    json_data = json.load(fp)

X = 1