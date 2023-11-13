import os

path = 'E:/Desktop/PGa/Semestr_9/PUG/Datasets/Final_dataset_no_labels/'
newpath = 'E:/Desktop/PGa/Semestr_9/PUG/Datasets/Final_dataset_changed_names_no_labels/'
race = 'Tuxedo'
f = []
i = 1
for (dirpath, dirnames, filenames) in os.walk(path + race):
    f.extend(filenames)
    break

for filename in filenames:
    os.rename(path + race + '/' + filename, newpath + race + '/' + race + '_' + str(i) + '.jpg')
    i += 1

print(filenames)

