import os


class FileRenamer:
    @staticmethod
    def change_filenames_in_folder(path='C:/Users/axawe/Desktop/Projects/CatsRecognizer/Dataset/Cats',
                                   new_path='C:/Users/axawe/Desktop/Projects/CatsRecognizer/Dataset/NewCats'):
        breeds = os.listdir(path)

        for breed in breeds:
            filenames = []
            i = 1

            filenames = os.listdir(path + '/' + breed)

            for filename in filenames:
                os.rename(path + breed + '/' + filename, new_path + breed + '/' + breed + '_' + str(i) + '.jpg')
                i += 1

            print(len(filenames))
