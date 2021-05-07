import pandas as pd
import json
import os

class BoxerDataGen:
    def __init__(self, predictions, feature_data, feature_list):
        self.predictions = predictions
        self.feature_data = feature_data
        self.feature_list = feature_list
        self.path = './gen_boxer_data'
        

    def gen_data(self):
        """Generate results.csv, features.csv and manifest.json files needed by Boxer

        Args:
            predictions: A dictionary where model names are used as keys and the prediction 
                results of each model are used as value 
            feature_data: A 2D numpy.ndarray where each row represents an instance and
                each column represents a feature. The default name of the column that contains
                the predicting target is 'actual_prediction'
            feature_list: A list containing names of features

        Returns:
            None
        """
        
        results = pd.DataFrame.from_dict(self.predictions)
        features = pd.DataFrame(self.feature_data, columns = self.feature_list)
        results['id'] = [i for i in range(len(results))]
        results['actual'] = features['actual_prediction']

        features['id'] = [i for i in range(len(features))]


        
        isFile = os.path.isfile(path)  
        if not isFile:
            os.mkdir(path)

        results.to_csv(path + '/results.csv' )
        features.to_csv(path + '/features.csv' )

        gen_manifest(path)




    def gen_manifest(self):
        """Generate manifest.json file needed by Boxer

        Args:
            path: A string that illustrates the path where store features.csv and results.csv

        Returns:
            None
        """
        file_features = pd.read_csv(self.path+'/features.csv') 
        file_results = pd.read_csv(self.path+'/results.csv') 

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






    


