# Week 2

## Binary classification


- cat vs non cat -> y e {0, 1}
- techniques
    - forward pass / propagation
    - back propagation
- Logistic regression
    - algo
    - binary classification
    - image (64 x 64) x 3 -> feat vector (64 x 64 x 3 = 12288) n = nx = 12288

- single training notation
    - (x, y) x e Rnx, y e {0,1}
    - m training examples : {(x1, y1), (x2, y2), ..., (xm, ym)}
    - m = mtrain | m test = #test examples
    - X.shape == (nx, m)
    - Y.shape == (1, m)

## Logistic regression

- Given x, want yhat = P(y=1 | x) 0 <= yhat <= 1
- parameters: w e Rnx, b e R
- output yhat = σ(w'X + b)  = σ(z) [sigmoid graph]
- σ(z) = 1 / (1+e^-z)
- If z is large σ(z) -> 1/(1+0) = 1
- if z is large negative σ(z) -> 1/(1 + e^-z) -> 1 / (1+BigNumb) -> 0

## Logistic regression cost function


- yhat = σ(w^T.X + B), where σ(z) = 1/(1 + e^-z)
- Given {(x1,y1), ... , (xm,ym)}, want yhati = yi
- loss (error) function which tell us how well out algorithm is doing on a single training example.
    - l(yhat, y) = 1/2 . (yhat - y)^2
        - non convex
        - local minimum
        - gradient decent not work well with squared error.
    - l(yhat, y) = - (y log(yhat) + (1-y) . log(1-yhat))
        - if y = 1: l(yhat, y) = - log(yhat) <- want log(yhat) large, want yhat large.
        - if y = 0: l(yhat, y) = - log(1-yhat) <- want log(1-yhat) large, ... want yhat small
- Cost function: J(w,b) = 1/m Σ l(yhat^i, y^i) = - 1/m Σ [y^i . log(yhat^i + (1-y^i) . log(1-yhat^i)]
    - measures how are you doing on the entire training set.

## Gradient descent

- Want to find w,b that minimize J(w,b)
- J(w)
- Repeat { w := w - α . dJ(w) / dw }
- α : learning rate
- dJ(w)/dw : the update of the change we want to make to the parameters w, alias dw
- Repeat { w := w - α . dw ; b := b - α . db }


## Derivatives

- intuition about derivatives
- in straight line the slope doesn't change

## More derivatives

- in a curved line the slope change


## Computation Graph

Left to right computation

- J(a,b,c) = 3(a+bc)
    - u = bc
        - v = a+u
            - J = 3v

compute the value of J.

Right to left is better to compute the derivatives.

## Derivatives with a computation graph

dJ / dv = ?
dJ / da = ? (chain's rule)
dJ / du = ?
dJ / db = ? -> dJ/du . du/db

- J(a,b,c) = 3(a+bc)
    - u = bc
        - v = a+u
            - J = 3v


## Logistic regression gradient descent

```
z = w'x + b
yhat = a = σ(z)
L(a,y) = -(ylog(a) + (1-y)log(1-a))


x1 --+
     |
w1 --+
     |
x2 --+----> z = w1.x1 + w2.x2 + b --> a = σ(z) --> L(a,y)
     |
w2 --+
     |
b  --+


"da" = dL(a,y) / da
     = - y/a + (1-y)/(1-a)

"dz" = dL/dz = dL(a,y)/dz
     = a - y
     = dL/da . da/dz


"dw1" = dL / dw1
      = x1 . dz

"dw2" = dL / dw2
      = xw . dz

```

## Logistic regression on m examples


```
J = 0; dw1 = 0; dw2 = 0; db = 0

for i = 1 to m:
    zi = w'.xi + b
    ai = σ(zi)
    J += - [yi . log(ai) + (1-yi) . log(1-ai)]
    dzi = ai - yi   # ai == yhat
    dw1 += x1i . dzi
    dw2 += x2i . dzi
    db += dzi

J /= m; dw1 /= m; dw2 /= m ; db /= m

--------------------

dw1 = dJ/dw1


w1 := w1 - α . dw1
w2 := w2 - α . dw2
b  := b - α . db
```


