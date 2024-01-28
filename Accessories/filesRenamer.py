import os


class FileRenamer:
    @staticmethod
    def change_filenames_in_folder(path='E:/Desktop/PGa/Semestr_9/PUG/Datasets/Cats/test/',
                                   new_path='E:/Desktop/PGa/Semestr_9/PUG/Datasets/Cats/test_watermark/'):
        breeds = os.listdir(path)

        for breed in breeds:
            filenames = []
            i = 1

            filenames = os.listdir(path + '/' + breed)

            for filename in filenames:
                os.rename(path + breed + '/' + filename, new_path + breed + '/' + breed + '_' + str(i) + '.jpg')
                i += 1

            print(len(filenames))


if __name__ == '__main__':
    fr = FileRenamer()
    fr.change_filenames_in_folder()
