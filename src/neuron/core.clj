(ns neuron.core
  (:use neuron.digits)
  (:use neuron.batch)
)



(def digits-train (load-directory "trainingDigits/ones_and_zeros/train"))

(def digits-test (load-directory "trainingDigits/ones_and_zeros/test"))

;(def nn (build-network-batch [1024 30 10] 0.01 400 40 sigmoid sigmoid' cross-entropy-output-deltas digits-train ))

;(fp nn (first (nth digits-test 10)) sigmoid)


