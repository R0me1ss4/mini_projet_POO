#USTHB: 2023-2024
#M1 RTB
#Module: Programmation Orientée Objet
#Projet : Implement KNN from scratch
#Projet proposé par : Professeur M.B.Abidine
#       réalisé par : BENNOUI Romeissa et CHIKER Imad

import csv
import random
import math
import operator

random.seed(42) #pour la reproductibilité des nombres aléatoires

"""1- Handle Data: """
# Lecture et affichage des lignes du fichier 'iris.data.txt': 
with open('iris.data.txt', 'r') as csvfile:
    lines = csv.reader(csvfile)
    for row in lines:
        print(', '.join(row))

# Définition de la fonction pour charger les données et diviser en ensembles d'entraînement et de test
def loadDataset(filename, split, trainingSet=[], testSet=[]):
    with open(filename, 'r') as csvfile:
        lines = csv.reader(csvfile)
        dataset = list(lines)
        for x in range(len(dataset)-1):
            for y in range(4):
                dataset[x][y] = float(dataset[x][y])

            if random.random() < split:
                trainingSet.append(dataset[x])
            else:
                testSet.append(dataset[x])

# Test
trainingSet=[]
testSet=[]
loadDataset('iris.data.txt', 0.66, trainingSet, testSet) #66% Entrainement et 34% test 
print('Train: ' + repr(len(trainingSet)))
print('Test: ' + repr(len(testSet)))

"""2- Similarity: """
# Définition de la fonction pour calculer la distance euclidienne entre deux instances
def euclideanDistance(instance1, instance2, length):
    distance = 0
    for i in range(length):
        distance += (instance1[i] - instance2[i]) ** 2
    return math.sqrt(distance)

# Test
data1 = [2, 2, 2, 'a']
data2 = [4, 4, 4, 'b']
distance = euclideanDistance(data1, data2, 3)
print('Distance:', repr(distance))

"""3- Neighbors: """
# Définition de la fonction pour obtenir les k voisins les plus proches d'une instance de test
def getNeighbors(trainingSet, testInstance, k):
    distances = []
    length = len(testInstance) - 1
    for x in range(len(trainingSet)):
        dist = euclideanDistance(testInstance, trainingSet[x], length)
        distances.append((trainingSet[x], dist))
    distances.sort(key=operator.itemgetter(1))
    neighbors = []
    for x in range(k):
        neighbors.append(distances[x][0])
    return neighbors

# Test:
trainSet = [[2, 2, 2, 'a'], [4, 4, 4, 'b']]
testInstance = [5, 5, 5]
k = 1
neighbors = getNeighbors(trainSet, testInstance, k)
print(neighbors)

"""4- Response: """
# Définition de la fonction pour obtenir la réponse basée sur les voisins
def getResponse(neighbors):
    classVotes = {}
    for x in range(len(neighbors)):
        response = neighbors[x][-1]  # En supposant que la réponse (classe) est extraite en accédant au dernier élément de chaque voisin.
        if response in classVotes:
            classVotes[response] += 1
        else:
            classVotes[response] = 1
    sortedVotes = sorted(classVotes.items(), key=operator.itemgetter(1), reverse=True)
    return sortedVotes[0][0]
#Dans l'apprentissage supervisé, chaque instance de données est généralement associée à une classe ou à une étiquette qui représente la catégorie à laquelle elle appartient. 
#Cette classe est souvent stockée dans la dernière colonne du jeu de données.
# Test :
neighbors = [[1, 1, 1, 'a'], [2, 2, 2, 'a'], [3, 3, 3, 'b']]
response = getResponse(neighbors)
print(response)

"""5- Accuracy: """
# Définition de la fonction pour calculer la précision du modèle
def getAccuracy(testSet, predictions):
    correct = 0
    for x in range(len(testSet)):
        if testSet[x][-1] == predictions[x]:
            correct += 1
    return (correct / float(len(testSet))) * 100.0

# Test :
testSet = [[1, 1, 1, 'a'], [2, 2, 2, 'a'], [3, 3, 3, 'b']]
predictions = ['a', 'a', 'a']
accuracy = getAccuracy(testSet, predictions)
print(accuracy)

"""6- Main: """
# Définition de la fonction principale pour exécuter l'algorithme k-NN
def kNNAlgorithm(trainSet, testSet, k):
    predictions = []
    for x in range(len(testSet)):
        neighbors = getNeighbors(trainSet, testSet[x], k)
        result = getResponse(neighbors)
        predictions.append(result)
    accuracy = getAccuracy(testSet, predictions)
    print(f"\nAccuracy: {accuracy}%")

filename = 'iris.data.txt'
loadDataset(filename, 0.7, trainingSet, testSet) #70% entrainement et 30% test 
# Tester l'algorithme kNN avec k=3 et la distance Euclidienne:
kNNAlgorithm(trainingSet, testSet, k=3)
#Nous prenons une petite valeur de K car notre base d'apprentissage est petite. 

"""7- Another distance metric (minkowski distance) """
# Définition de la fonction pour calculer la distance de Manhattan
def manhattanDistance(instance1, instance2, length):
    distance = 0
    for i in range(length):
        distance += abs(instance1[i] - instance2[i])
    return distance

# Définition de la fonction pour obtenir les k voisins les plus proches avec la distance de Manhattan
def getNeighbors2(trainingSet, testInstance, k):
    distances = []
    length = len(testInstance) - 1
    for x in range(len(trainingSet)):
        dist = manhattanDistance(testInstance, trainingSet[x], length)
        distances.append((trainingSet[x], dist))
    distances.sort(key=operator.itemgetter(1))
    neighbors = []
    for x in range(k):
        neighbors.append(distances[x][0])
    return neighbors

# Définition de la fonction principale pour tester l'algorithme k-NN avec k=3 et la distance de Manhattan
def kNNAlgorithm2(trainSet, testSet, k):
    predictions = []
    for x in range(len(testSet)):
        neighbors = getNeighbors2(trainSet, testSet[x], k)
        result = getResponse(neighbors)
        predictions.append(result)
    accuracy = getAccuracy(testSet, predictions)
    print(f"\nAccuracy with manhattan distance : {accuracy}%")
          
filename = 'iris.data.txt'
loadDataset(filename, 0.7, trainingSet, testSet) #70% entrainement et 30% test
# Tester l'algorithme kNN avec k=3 et la distance de Manhattan:
kNNAlgorithm2(trainingSet, testSet, k=3)

#on obtient : Accuracy: 89.36170212765957% et Accuracy with manhattan distance : 95.0% 
#donc la prédiction en utilisant la distance de manhattan est meilleure. 
