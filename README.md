# Intelligence artificielle : Réseaux de neurones

### Equipe

Yannis Bacha & Pierre Chat

## Question 1

> Quelle sera la prochaine valeur du poids du neurone gagnant dans le cas où η = 0 ?

Si η est nul, la prochaine valeur du poids du neurone gagnant restera inchangée.

> Quelle sera la prochaine valeur du poids du neurone gagnant dans le cas où η = 1 ?

Si η vaut 1, la prochaine valeur du poids du neurone gagnant sera égale à la valeur tirée. Chaque neurone n'apprend
seulement que de la dernière entrée.

> En déduire géométriquement la prochaine valeur du poids dans le cas normal où η ∈ ]0, 1[.

Plus η tend vers 0, moins le poids du neurone va être modifié. Plus η tend vers 1, plus le poids va être modifié.

> Si σ augmente, est-ce que les neurones vont plus ou moins apprendre l’entrée courante ?

Si σ augmente, les neurones vont plus apprendre de l'entrée courante. Dans la formule, une valeur de σ plus importante
implique que l'exponentielle se rapproche de 1.

> En déduire l’influence que doit avoir σ sur la “grille” de neurone, sera-t-elle plus “lâche” ou plus “serrée”
si σ augmente ?

Si σ augmente, la grille apparaitra alors plus serrée car les neurones voisins seront plus influencés par le neurone
gagnant.

> Prenons le cas d’une carte avec un seul neurone qui reçoit 2 entrées x1 et x2 . Durant l’apprentissage x1
(respectivement x2) est présenté n1 (respectivement n2) fois. Après l’apprentissage où se situera
géométriquement le poids du neurone ?

Si n1 est largement supérieur à n2, le poids du neurone se trouvera nettement plus proche de x1 et inversement.
Si n1 et n2 sont à peu près égaux et suffisemment grands, le poids du neurone se trouvera entre x1 et x2.

## Question 2

```python
def compute(self, x):
    self.y = numpy.sqrt(numpy.sum((x - self.weights) ** 2))
```

```python
def learn(self, eta, sigma, posxbmu, posybmu, x):
    exp = numpy.exp(-(((self.posx - posxbmu) ** 2) + ((self.posy - posybmu) ** 2)) / (2 * (sigma ** 2)))
    self.weights[0] += eta * exp * (x[0] - self.weights[0])
    self.weights[1] += eta * exp * (x[1] - self.weights[1])
```

![GIF](img/giphy.gif)

## Question 3

> Influence des éléments suivants sur le fonctionnement de l’algorithme de Kohonen

#### Référence : η = 0.05 | σ = 1.4 | N = 30000 | 

![GIF](img/giphy.gif)

- taux d’apprentissage η

Il s'agit de la vitesse à laquelle les neurones vont converger vers la solution tirée. Plus η tendra vers 1 plus les poids des
neurones sera modifié et donc convergera plus vite et inversement.

![GIF](img/giphy2.gif)

On voit bien qu'avec η = 0.25 les poids convergent bien plus vite.

- largeur du voisinage σ

Si on augmente σ, les noeuds seront plus éparpillés sur la grille.

![GIF](img/giphy3.gif)

Avec σ = 10 on voit que dès le premier rafraichissement, une partie des neurones convergent vers la solution mais
qu'une grande partie d'entre eux n'ont pas leur poids qui évolue et donc restent sur place.

Après avoir relancé l'algorithme et attendu la fin du traitement, on peut obtenir une carte comme ci-dessous.

![Sigma](img/sigma.png)

On voit bien qu'après 30001 boucles, on se retrouve avec des neurones qui n'ont pas beaucoup appris, voir pas du tout.

- nombre de pas de temps d’apprentissage N


- taille de la carte


- jeu de données. En particulier cr´eez vos propres jeux de donn´ees avec des donn´ees non uniform´ement distribu´ees
pour ´etudier la r´epartition des poids des neurones. Etudiez s´eparemment le jeu de donn´ees MNIST. ´

- la topologie de la carte (Bonus). En particulier au lieu d’utiliser une grille carr´ee, utilisez une grille hexagonale.


