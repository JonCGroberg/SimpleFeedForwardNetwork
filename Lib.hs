module Lib
    ( Weights
    , TrainingData
    , WeightedInputs
    , Layers
    , (•)
    , (⨀)
    , sigmoid
    , sigmoid'
    , squareError
    , squareError'
    , display
    , prettyMatrix
    , split
    )
where
import           Numeric.LinearAlgebra.Data


---------------------------------- Types ----------------------------------
type Weights = [Matrix R]
type TrainingData = (Matrix R, Matrix R)
type WeightedInputs = [Matrix R]
type Layers = [Matrix R]


----------------------- Matrix Multiply Functions ------------------------
(•) :: Semigroup a => a -> a -> a
(•) x y = x <> y                                         -- make dot product look like a dot

(⨀) :: Num a => a -> a -> a
(⨀) x y = x * y                                         -- make it clear what we are doing


----------------------------- Math Functions -----------------------------
sigmoid :: Matrix R -> Matrix R                         -- maps sigmoid over the array
sigmoid = cmap s
s x = 1.0 / (1 + exp (-x))

sigmoid' :: Matrix R -> Matrix R                        -- maps sigmoid' over the array
sigmoid' = cmap s'
s' x = s x * (1.0 - s x)

squareError :: Matrix R -> Matrix R -> Matrix R
squareError y ŷ = (y - ŷ) ^ 2

squareError' :: Matrix R -> Matrix R -> Matrix R
squareError' y ŷ = 2 * (y - ŷ)


---------------------- Display Functions(IO things) ----------------------
display :: Matrix R -> IO ()
display = prettyMatrix 3

prettyMatrix :: Int -> Matrix R -> IO ()
prettyMatrix i m = if cols m == 1 && rows m == 1        -- if single value print it without brackets
    then dispDots i m
    else putStr
        (concat
            [ ( head
              . map (\x -> "┌ " ++ replicate (length x) ' ' ++ " ┐\n")                  -- ┌     ┐
              . tail
              . lines
              )
                (dispf i m)
            , (concatMap (\x -> "│ " ++ x ++ " │\n") . tail . lines) (dispf i m)        -- │     │
            , ( last
              . map (\x -> "└ " ++ replicate (length x) ' ' ++ " ┘\n")                  -- └     ┘
              . tail
              . lines
              )
                (dispf i m)
            ]
        )


---------------------------- List Functions ----------------------------
split :: [a] -> ([a], [a])
split = foldr (\x (ys, zs) -> (x : zs, ys)) ([], [])
