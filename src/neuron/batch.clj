(ns neuron.batch
  (:use incanter.core)
  (:use incanter.stats)
  (:use neuron.digits)
)


; (matrix [1 2 3]) produces a 1 x 3 matrix
;  mult does element by element multiplaction
;  mmult does matrix multiplication



;activation functions sigma(z)

(defn sigmoid [z]
(div 1 (plus 1 (exp (minus 0 z)))))


(defn tanh [z] (Math/tanh z))



;derivative of activation functions sigma'(z).  Note that derivatives of both sigmoid and tahn are expressed as function of sigmoid and tahn, i.e., 
;they are functions of a, not z.

(defn sigmoid' [a]
(mult a (minus 1 a))
)



(defn tanh' [a]
(minus 1 (mult a a))
)



;sampling....replace


(defn get-random-list [r n]
(take n (repeatedly #(rand-int r)))
)

(defn get-batch [col size]
(let [l (get-random-list (count col) size)]
  (map #(nth col %) l)
))





; helpful for unit testing

(defn non-random-list [rows cols]
(vec (repeat (* rows cols) 1)))


(defn rand-list "Create a list of random doubles between -epsilon and + epsilon." 
[epsilon rows cols] 
(map (fn [x] (- (rand (* 2 epsilon)) epsilon)) (range 0 (* rows cols))))



(defn add-bias [c]
(bind-rows [1] c))


; layers is vector of network sizes  (inc the cols to add in the bias).
(defn initialize-weights' "Generate random initial weight matrices for given layers. layers must be a vector of the sizes of the layers." 
 [layers rfn] 
 (for [i (range 0 (dec (length layers)))] 
 (let [rows (get layers (inc i))
       cols (inc (get layers i))]
       (matrix (rfn rows cols) cols))))


(defn initialize-weights [layers epsilon]
	(initialize-weights' layers (partial rand-list epsilon)))
	

(defn new-gradient-matrix 
"Create accumulator matrix of gradients with the same structure as the given weight matrix with all elements set to 0." 
[weight-matrix] 
(let [[ rows cols] (dim weight-matrix)] 
(matrix 0 rows cols)))





; cross entropy cost function...assumes using sigmoid activation
; fix hack...don't need the afn'..bad clojure
(defn cross-entropy-output-deltas 
	[output-layer-activations expected-output afn']
        (minus output-layer-activations expected-output))


; quadratic cost function

(defn quadratic-output-deltas
  [output-layer-activations expected-output afn']
  (mult (minus output-layer-activations expected-output) (afn' output-layer-activations)))



;;layer-n-input...column matrix
;;weights - column 1 is weights from neuron 1 to next layer

(defn activate-layer-n+1
[layer-n-input weights-n-to-n+1 afn]
(matrix (map afn (mmult weights-n-to-n+1 layer-n-input))))


;input is column vector
;weights vector of matrices
;returns a vector of column matrices representing activations

(defn propagate-activations 
[input weights afn]
(loop [layer-n-input (add-bias input)
       weights' weights
       activations [(add-bias input)]]
       (if (empty? weights') activations
       (let [a (activate-layer-n+1 layer-n-input (first weights') afn)
       		 b (if (empty? (rest weights')) a (add-bias a))
       		 ]
       (recur b (rest weights') (conj activations b))))))
 
 
 
 
;;weights-n-to-n+1 column 1 is weights from neuron 1 to next layer neurons
;;layer-n+1-delta a column matrix with deltas for each output neuron
;;layer-n-activations a column matrix with activations for each layer n neuron
;;output is column matrix of activations.



(defn compute-layer-n-delta [weights-n-to-n+1 layer-n+1-delta layer-n-activation afn']
  (mult (mmult (trans weights-n-to-n+1) layer-n+1-delta) (matrix (afn' layer-n-activation))))
             
 
 ; output-delta column matrix
 ; weights, vector of matrices
 ; activations vector of matrices
 ; afn' - derivative of activation function
 ; for a 2-3-2 network, output will be deltas for 1 hidden layer and output layer 
 
 ; need to strip out bias from each delta before accumulating...(matrix (rest delta))
            

(defn compute-hidden-layer-deltas
	[output-delta weights activations afn']
	(loop  [w (reverse (rest weights))
		   deltas [output-delta]
	       a (rest (reverse (rest activations)))
	       ]
	     
		(if (empty? w) deltas
		(let [d (compute-layer-n-delta (first w) (first deltas) (first a) afn')]
		(recur (rest w) (cons (matrix (rest d)) deltas) (rest a))))))
		
	


(defn calc-gradients
 [deltas activations] 
 (map #( mmult %1 (trans %2)) deltas activations))
	


; afn: activation function
; afn': derivative of activatino functin
; odf: output delta function (based on cross-entropy + sigmoid, or quadritic)

(defn train-network
[weights input target afn afn' odf]
(let [
	 activations (propagate-activations input weights afn)
	 deltas (compute-hidden-layer-deltas (odf (last activations) target afn') weights activations afn')
	 gradients (calc-gradients deltas activations)]
	 gradients))



(defn train-data
[weights sample-data learning-rate afn afn' odf]
(loop [
	    s sample-data
	    g (new-gradient-matrix weights)
	    ]
	    (if (empty? s)
	    (map minus weights (map #(mult %1 learning-rate) g))
	    (recur (rest s) (map plus g (train-network weights (matrix (first (first s))) (matrix (second (first s))) afn afn' odf))))))
	    
	    


;;learning rate is _not_ divided by batch size and passed to train-data
(defn train-epochs-batch 
[weights sample-data cycles batch-size learning-rate afn afn' odf]
(if (zero? cycles) weights
    (let [batch (get-batch sample-data batch-size)]
      (recur (train-data weights sample-data learning-rate afn afn' odf) sample-data (dec cycles) batch-size learning-rate afn afn' odf))))



(defn train-epochs
  [weights sample-data cycles learning-rate afn afn' odf]
  (if (zero? cycles) weights
      (recur (train-data weights sample-data learning-rate afn afn' odf) sample-data (dec cycles) learning-rate afn afn' odf)))




 (defn fp
    [weights input afn]
    (last (propagate-activations (matrix input) weights afn)))




(defn build-network-batch [size learning-rate cycles batch-size afn afn' odf data]
(train-epochs-batch (initialize-weights size 2) data cycles batch-size learning-rate afn afn' odf))


(defn build-network [size learning-rate cycles afn afn' odf data]
(train-epochs (initialize-weights size 1) data cycles learning-rate afn afn' odf))




;;example usage:
;;(def nn (build-network [2 3 1] 0.1 200 tanh tanh' quadratic-output-deltas sample-data))
;;(fp nn [1 1])


;;todo

;add MSE to jump out of loop.


;;(def nn (build-network [1024 30 10] 2 100 200 digit-data)
 

