module Main where

import           Control.Monad
import           Lib
import           Numeric.LinearAlgebra

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


----------------------------------------- NETWORK FUNCTIONS ----------------------------------------

createNetwork :: TrainingData -> Int -> Int -> IO (Weights, TrainingData)
createNetwork (x, y) hiddenSize hiddenLayers = do
  let inputSize  = cols x
  let outputSize = cols y

  ws <- do -- Weights connect (between) each layer ∴ n layers are connected by n-1 weights
    wI <- randn inputSize hiddenSize
    wH <- replicateM (hiddenLayers - 1) (randn hiddenSize hiddenSize)
    wO <- randn hiddenSize outputSize
    return (concat [[wI], wH, [wO]]) -- return a single list of weights

  let network = (ws, (x, y))
  return network


train :: (Weights, TrainingData) -> Int -> (Weights, TrainingData) -- Start learning
train network epochs = iterate updateWeights network !! epochs
 where
  updateWeights :: (Weights, TrainingData) -> (Weights, TrainingData)
  updateWeights (weights, trainingData) = (newWieghts, trainingData)
   where
    zsAndLayers = forwardProp (weights, trainingData) -- We only need our prediction (ŷ) BUT we can resuse the calculations
    newWieghts  = backProp (reverse weights, trainingData) zsAndLayers


predict :: (Weights, TrainingData) -> [[R]] -> IO () -- Test some data on our network
predict (w, (_, y)) testData = do
  let input = fromLists testData
  display . head . fst $ forwardProp (w, (input, y))


--------------------------------------- TRAINING FUNCTIONS -----------------------------------------

-- Run data through the network & get the output
forwardProp :: (Weights, TrainingData) -> (Layers, WeightedInputs)
forwardProp (weights, (input, _)) = zsAndLayers
 where
    -- reverse for backProp and then split Layers and Zs into their own lists
  zsAndLayers = split . reverse $ calcLayers input weights

  calcLayers :: Matrix R -> Weights -> [Matrix R]
  calcLayers layerꜜ list = layerꜜ : case list of
    []     -> []
    w : ws -> z : rest
     where
      z     = layerꜜ • w
      layer = sigmoid z
      rest  = calcLayers layer ws


-- Figure out how to change the weights using calculus
backProp :: (Weights, TrainingData) -> (Layers, WeightedInputs) -> Weights
backProp (ws, (x, y)) (ŷ : layerInputs, z : zs) = newWieghts
 where
  α           = 1 -- learning rate
  loss'       = squareError' y ŷ
  eO          = squareError' y ŷ ⨀ sigmoid' z
  layerErrors = calcLayerErrors eO ws zs

  -- ws' = layerInput • layerErrors (we also map our learning rate over ws')
  ws'         = map (* α) (zipWith (•) (map tr layerInputs) layerErrors)

  newWieghts  = reverse $ zipWith (+) ws ws' -- Return the new weights unreversed


-- Previous error (this layers output) -> this error
calcLayerErrors :: Matrix R -> Weights -> WeightedInputs -> [Matrix R]
calcLayerErrors eꜜ _        []       = [eꜜ]
calcLayerErrors eꜜ (w : ws) (z : zs) = eꜜ : restOfe
 where
  e       = eꜜ • tr w ⨀ sigmoid' z
  restOfe = calcLayerErrors e ws zs
