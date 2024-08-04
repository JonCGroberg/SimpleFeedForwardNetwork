# SimpleFeedForwardNetwork

Example Usage:

```haskell
main :: IO ()
main = do
  let x = fromLists [[0, 0], [0, 1], [1, 0], [1, 1]]
  let y            = fromLists [[0], [1], [1], [0]]
  let hiddenSize   = 4
  let hiddenLayers = 1

  network <- createNetwork (x, y) hiddenSize hiddenLayers

  let trainedNetwork = train network 1000

  predict trainedNetwork [[0, 0]]
  predict trainedNetwork [[0, 1]]
  predict trainedNetwork [[1, 0]]
  predict trainedNetwork [[1, 1]]
```
