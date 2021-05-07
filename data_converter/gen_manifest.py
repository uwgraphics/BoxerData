import pandas as pd
import json

# prequest: save source data in features.csv and save prediction results in results.csv
    # The format of these two files can be found in 'https://graphics.cs.wisc.edu/Vis/Boxer/docs/data_preparation/'

def gen_manifest(path):
    file_features = pd.read_csv(path+'/features.csv') 
    file_results = pd.read_csv(path+'/results.csv') 

    classes = list(set(file_results["actual"].tolist()))
    classifiers = list(file_results.columns)
    removed_list = ['id', 'Unnamed: 0', 'actual']
    for item in removed_list:
        if item in classifiers:
            classifiers.remove(item)
    

    feature_doc = {}
    feature_name = file_features.columns.tolist()
    for item in removed_list[:-1]:
        if item in feature_name:
            feature_name.remove(item)


    categories_num_threshold = 20


    features = file_features
    for feature in feature_name:
        column = features[feature].tolist()
        typeo = "nominal"
        categories = []
        description = feature
        bounded =  "false"
        if type(column[0]) is int or type(column[0]) is float:
            if len(set(column)) < len(column)/2 and len(set(column)) < categories_num_threshold:
                typeo = "categorical"
                categories = list(set(column))
            else:
                typeo = "ratio"
        elif type(column[0]) is str:
            if column[1].isnumeric():
                if len(set(column)) < len(column)/2 and len(set(column)) < categories_num_threshold:
                    typeo = "categorical"
                    categories = list(set(column))
                else:
                    typeo = "ratio"
            else:
                if len(set(column)) < len(column)/2 and len(set(column)) < categories_num_threshold:
                    typeo = "categorical"
                    categories = list(set(column))
                else:
                    typeo = "nominal"

        feature_doc[feature] = {
            "type" : typeo,
            "description" : description
        }
        if typeo == "categorical" :
            feature_doc[feature]["categories"] = categories
        elif typeo == "ratio":
            feature_doc[feature]["bounded"] = bounded

    manifest = {
            "datasetName": "conitnuous_income",
            "classes": classes,
            "classifiers": classifiers,
            "features": feature_doc
    }


    with open(path+'/manifest.json', 'w') as outfile:
        json.dump(manifest, outfile)



