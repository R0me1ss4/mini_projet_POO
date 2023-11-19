# mini_projet_POO
Ce programme implémente un algorithme k-NN (k plus proches voisins) pour la classification des données du jeu de données Iris.
Explication des sections du code :
- Section 1: Handle Data
Cette section lit et affiche chaque ligne du fichier 'iris.data.txt'.
- Section 2: loadDataset Function
Cette fonction prend en entrée un fichier de données, une proportion de division (split), et deux ensembles vides (trainingSet et testSet). 
Elle charge les données du fichier CSV, et les divise en ensembles d'entraînement et de test en fonction de la valeur de split.
Les données sont stockées dans les ensembles correspondants.
- Section 3: Euclidean Distance Function
Cette fonction calcule la distance euclidienne entre deux instances de données de même longueur.
Cela est utilisé pour mesurer la similarité entre deux points dans l'espace des caractéristiques.
-  Section 4: getNeighbors Function
Cette fonction trouve les k voisins les plus proches d'une instance de test en calculant la distance euclidienne entre l'instance de test et chaque instance dans l'ensemble d'entraînement.
Les voisins sont triés par distance, et les k plus proches sont retournés.
- Section 5: getResponse Function
Cette fonction détermine la classe la plus fréquente parmi les voisins. Elle suppose que la classe est stockée à la dernière position de chaque voisin.
Les votes sont triés par fréquence décroissante, et la classe majoritaire est retournée.
- Section 6: getAccuracy Function
Cette fonction calcule la précision du modèle en comparant les classes prédites avec les classes réelles dans l'ensemble de test.
- Section 7: kNNAlgorithm Function
Cette fonction exécute l'algorithme k-NN. Pour chaque instance dans l'ensemble de test, elle trouve les k voisins les plus proches, détermine la classe majoritaire parmi ces voisins, et stocke la prédiction.
Enfin, elle imprime la précision du modèle.
- Section 8: Another distance metric (Manhattan distance)
Cette section ajoute la possibilité d'utiliser la distance de Manhattan comme alternative à la distance euclidienne.
Deux fonctions supplémentaires (manhattanDistance et getNeighbors2) sont définies pour calculer la distance de Manhattan et trouver les voisins en conséquence.
La fonction kNNAlgorithm2 est ensuite utilisée pour tester l'algorithme k-NN avec la distance de Manhattan.
