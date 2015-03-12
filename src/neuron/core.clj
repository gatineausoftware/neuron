(ns neuron.core
  (:use neuron.digits)
  (:use neuron.batch)
  (:use neuron.monitor)
)



(def digits-train (load-directory "trainingDigits/ones_and_zeros/train"))

(def digits-test (load-directory "trainingDigits/ones_and_zeros/test"))

;(def nn (build-network-batch [1024 30 10] 0.01 400 40 sigmoid sigmoid' cross-entropy-output-deltas digits-train ))

;(fp nn (first (nth digits-test 10)) sigmoid)

;(def c (future (train-mf 0.005 2000 40 40 digits-train digits-test))) - produces good results.


;(def m (future (monitor neural-net digits-test sigmoid 40)))


