import csv
import numpy as np
from colorama import Fore
from model import STSModel
from train import TrainModel
from sklearn.metrics import accuracy_score

class TestModel(TrainModel):
    def __init__(self) -> None:
        super().__init__()
        self.fields = ['Исходные данные', 'Референс', 'Результат работы классификатора']
        self.test()
        pass

    def load_model(self):
        return STSModel().model()

    def get_data(self, trainset_path: str ,validset_path:str, testset_path: str):
        df_train, df_valid, df_test = self.load_data(trainset_path, validset_path, testset_path)
        _, _, test, _, _ = self.preprocess_data(df_train, df_valid, df_test)
        _, _, test_labels = self.get_labels(df_train, df_valid, df_test)
        return df_test, test, test_labels
    
    def _save(self, path, rows):
        with open(path, 'w') as f:
            write = csv.writer(f)
            write.writerow(self.fields)
            write.writerows(rows)

    def _result(self, test_prediction, test_labels):
        return accuracy_score(np.array([round(pred[0]) for pred in test_prediction]), test_labels)
    
    def _print_result(self, df_test, test_prediction, test_labels):
        output_row = []
        for text_1, text_2, pred_label, real_label in zip(df_test['Исходные данные'], df_test['Референс'], test_prediction, test_labels):
            print(Fore.RED if round(pred_label[0]) != real_label else Fore.GREEN , f'{text_1} - {text_2} - {round(pred_label[0])} - {real_label}')
            output_row.append([text_1, text_2, round(pred_label[0])])
        
        print('-------------------------------------------------')
        print(Fore.GREEN, f'Accuracy Score: {self._result(test_prediction, test_labels)}')
        print('-------------------------------------------------')
        
        path = 'result.csv'
        self._save(path, output_row)
        print('Результат сохранен:{}'. format(path))

    def test(self):
        df_test, test, test_labels = self.get_data('./DataSet.csv', './validset.csv', './testset2.csv')
        model = self.load_model()
        model.summary()
        model.load_weights('saved_models/best_model.h5')
        test_prediction = model.predict(test)
        self._print_result(df_test, test_prediction, test_labels)
    
if __name__ == "__main__":
    TestModel()