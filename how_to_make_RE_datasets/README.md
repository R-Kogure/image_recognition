# How to make Random Erasing image datasets

### Random Erasingの実装

引数のprobはREが実行される確率．原論文の中での基本は0.5だった．  
削除される領域が縦横ともに画像内に収まる条件下で実行される．  

```python
def randomerasing(image, prob = 0.5, sl = 0.02, sh = 0.4, rl = 0.3):

    if np.random.rand() > prob:
        return image
    
    h, w, _ = image.shape
    area = h*w

    while True:
        erase_area = np.random.uniform(sl, sh)*area
        aspect_ratio = np.random.uniform(rl, 1/rl)

        h_erase = int(np.sqrt(erase_area * aspect_ratio))
        w_erase = int(np.sqrt(erase_area / aspect_ratio))

        left = np.random.randint(0, w)
        top = np.random.randint(0, h)

        if left + w_erase <= w and top + h_erase <= h:
            image[top:top+h_erase, left:left+w_erase, :] = np.random.rand(h_erase, w_erase, 3) * 255.0
            break
    
    return image

```


### REを適用したデータセットを作る

```python
#何らかのデータセットを持ってくる
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()  


x_train_erased = np.copy(x_train) #erasedがREを適用したデータを入れるところ

for i in range(len(x_train)):
    x_train_erased[i] = randomerasing(x_train[i])

# RE適用率50%のデータセットができる

```

### この後どう使うのか的なところ

適用したい任意のモデルの入力などに合わせてresizeして使ったり...  
例えば
```python
func = lamda x : tf.image.resize(x, (224, 224))
x_train = tf.py_function(func, [x_train], tf.float32)
```
などを実行することで(N, 224, 224, 3)に変形できる．  
