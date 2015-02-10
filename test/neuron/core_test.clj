(ns neuron.core-test
  (:require [clojure.test :refer :all]
            [neuron.core :refer :all]))



(deftest initialize-weights
(testing "initialize weights"
(is (= (initialize-weights' [2 3 2] non-random-list) [(matrix [[1 1] [1 1] [1 1]]) (matrix [[1 1 1] [1 1 1]])]))))



(deftest activate-layer-n+1-1
	(testing "compute layer n+1 activation given layer n activation"
	(is (=(activate-layer-n+1 (matrix [[1] [2] [3]]) (matrix [[1 1 1] [2 2 2]]) identity) (matrix [[6] [12]])))))

(deftest compute-output-layer-delta-1
  (testing "compute output delta given output layer activations and expected output"
    (is (= (compute-output-deltas (matrix [[1] [2] [3]]) (matrix [[1] [-2] [6]]) identity) (matrix [[0] [8] [-9]])))))



(deftest compute-layer-n-delta-1
  (testing "compute layer n delta, given weight matrix and delta n+1"
    (is (= (compute-layer-n-delta (matrix [[1 2 3]]) (matrix [[1]]) (matrix [1 1 1]) identity) (trans (matrix [[1 2 3]]))))
    


(deftest compute-layer-n-delta-2
  (testing "compute layer n delta, given weight matrix and delta n+1"
    (is (= (compute-layer-n-delta (matrix [[1 2 3] [4 5 6]]) (matrix [[2] [4]]) (matrix [[1] [1] [1]]) identity) (trans (matrix [[18 24 30]]))))
