import random, csv, sys
from dict import *

class Generator:
    def __init__(self, len_data:int, path:str, variety:int) -> None:
        self.fields = ['Исходные данные', 'Референс', 'Классификатор']

        data = self.generator(len_data=len_data, variety = variety)
        self.save(path, data)
        pass

    def replacer(self, string, position):
        new_string = string[:position] + random.choice(KKS_fromat[1]) + string[position+1:]
        while new_string == string: new_string = string[:position] + random.choice(KKS_fromat[1]) + string[position+1:]
        return new_string

    def createKKScode(self):
        application_identifier = ''.join(random.choice(KKS_fromat[0]) for i in range(2))
        main_group = ''.join(random.choice(KKS_fromat[1]) for i in range(2))
        sub_group = ''.join(random.choice(KKS_fromat[0]) for i in range(4))
        object_identifier = ''.join(random.choice(KKS_fromat[1]) for i in range(4))
        return '-'.join(x for x in [application_identifier, main_group, sub_group, object_identifier])

    def generate_word(self, random_word, p2):
        return random.choice(engineer_words) if random.randint(1,100)>p2 else random_word

    def make_phrase(self, KKS_1, KKS_2, random_word_3,  p1, p2):
        random_word = random.choice(engineer_words)
        KKS_2 = '%s %s %s' % (random_word,KKS_2, random_word_3)
        random_word_2 = self.generate_word(random_word, p2)
        KKS_1 = '%s %s %s' % (random_word_2,KKS_1, random_word_3)

        return KKS_1, KKS_2

    def generator(self, len_data, variety):
        one, zero, rows = [], [], []
        for i in range(len_data):
            KKS_1 =  self.createKKScode()
            random_word_3 = random.choice(engineer_adjectives) if random.randint(1,100)>70 else ''
            seed = random.randint(1,100)

            if seed <= variety: ## Идентичные коды 
                KKS_1, KKS_2 = self.make_phrase (KKS_1, KKS_1, random_word_3, 20, 20)
                one.append(1)
                rows.append([KKS_1, KKS_2, 1])
            
            elif variety < seed <= variety + round((100-variety)/2):  ## Коды отличаются на один знак/цифру 
                KKS_2 = self.replacer (KKS_1, random.randint(0, len(KKS_1)-2))
                KKS_1, KKS_2 = self.make_phrase (KKS_1, KKS_2, random_word_3, 95, 95)
                zero.append(0)
                rows.append([KKS_1, KKS_2, 0])
            
            elif variety + round((100-variety)/2) < seed <= 100:  ## Коды отличаются полностью  -
                KKS_1, KKS_2 = self.make_phrase (KKS_1, self.createKKScode(), random_word_3, 90, 97)
                zero.append(0)
                rows.append([KKS_1, KKS_2, 0])
                        
        print(len(zero), len(one), len(zero)+len(one), len(zero)/(len(zero)+len(one)), len(one)/(len(zero)+len(one)))  
        return rows
    
    def save(self, path, rows):
        with open(path, 'w') as f:
            
            # using csv.writer method from CSV package
            write = csv.writer(f)
            
            write.writerow(self.fields)
            write.writerows(rows)


if __name__ == "__main__":
    operation = str(sys.argv[1]) if len(sys.argv) > 1 else 'all'
    set_number = int(sys.argv[2]) if len(sys.argv) > 2 else 'standart'
    variety = int(sys.argv[3]) if len(sys.argv) > 3 else 50
    path = str(sys.argv[4]) if len(sys.argv) > 4 else 'data.csv'
    
    if operation == 'train':
        Generator(set_number, path, variety=variety)
    elif operation == 'validation':
        Generator(set_number, path, variety=variety)
    elif operation == 'test':
        Generator(set_number, path, variety=variety)
    elif operation == 'all' and str(set_number) == 'standart':
        Generator(100000, 'Dataset.csv', variety=55)
        Generator(5000, 'validset.csv', variety=50)
        Generator(2500, 'testset.csv', variety=50)