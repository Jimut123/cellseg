import pickle
file = open('colours_cluster.txt', 'rb')
itm = pickle.load(file)
print(itm)
