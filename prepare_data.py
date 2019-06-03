import json


class PreparedData:
    def __init__(self):
        pass

    @staticmethod
    def get_prepared_data():
        filename = "schedule_training_data.json"
        with open(filename, 'r') as f:
            training_set = json.load(f)
        return PreparedData.prepare_training_data(training_set)

    @staticmethod
    def to_number(val):
        try:
            return float(val)
        except (ValueError, TypeError):
            if val is None:
                return 0
            elif val == "off":
                return -1
            elif val is bool:
                if val:
                    return 1
                else:
                    return 0
            return 0

    @staticmethod
    def prepare_features(scenario):
        prepared = []
        for feature in scenario:
            prepared.append(PreparedData.to_number(scenario[feature]))
        return prepared

    @staticmethod
    def prepare_training_data(data):
        train_features = []
        train_labels = []
        for row in data['data']:
            train_features.append(PreparedData.prepare_features(row['scenario']))
            train_labels.append(row['shift'])
        return {'features': train_features, 'labels': train_labels}

