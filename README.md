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
