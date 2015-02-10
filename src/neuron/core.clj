(ns neuron.core
  (:use incanter.core)
  (:use incanter.stats)
)


; (matrix [1 2 3]) produces a 1 x 3 matrix
; mult does element by element multiplaction
; mmult does matrix multiplication


(def sample-data [[[ 0 0] [0]] [[ 0 1] [1]] [[ 1 0] [1]] [[ 1 1] [0]]])



(defn activation-fn [x] (Math/tanh x))

(defn dactivation-fn [y] (- 1.0 (* y y)))


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



(defn compute-output-deltas 
	[output-layer-activations expected-output dfn]
	(mult (minus output-layer-activations expected-output) (map dfn output-layer-activations)))




;;layer-n-input...column matrix
;;weights - column 1 is weights from neuron 1 to next layer

(defn activate-layer-n+1
[layer-n-input weights-n-to-n+1 afn]
(matrix (map afn (mmult weights-n-to-n+1 layer-n-input))))


;input is column vector
;weights vector of matrices
;returns a vector of column matrices representing activations

(defn propagate-activations 
[input weights]
(loop [layer-n-input (add-bias input)
       weights' weights
       activations [(add-bias input)]]
       (if (empty? weights') activations
       (let [a (activate-layer-n+1 layer-n-input (first weights') activation-fn)
       		 b (if (empty? (rest weights')) a (add-bias a))
       		 ]
       
       (recur 
       		  b
       		   (rest weights')
               (conj activations b))))))
 
 
 
 
;;weights-n-to-n+1 column 1 is weights from neuron 1 to next layer neurons
;;layer-n+1-delta a column matrix with deltas for each output neuron
;;layer-n-activations a column matrix with activations for each layer n neuron
;;output is column matrix of activations.

(defn compute-layer-n-delta [weights-n-to-n+1 layer-n+1-delta layer-n-activation dfn]
(mult (mmult (trans weights-n-to-n+1) layer-n+1-delta) (matrix (map dfn layer-n-activation))))
             
 
 ; output-delta column matrix
 ; weights, vector of matrices
 ; activations vector of matrices
 ; dafn - derivative of activation function
 ;;for a 2-3-2 network, output will be deltas for 1 hidden layer and output layer 
 
 ; need to strip out bias from each delta before accumulating...(matrix (rest delta))
            
(defn compute-hidden-layer-deltas
	[output-delta weights activations dafn]
	(loop  [w (reverse (rest weights))
		   deltas [output-delta]
	       a (rest (reverse (rest activations)))
	       ]
	     
		(if (empty? w) deltas
		(let [d (compute-layer-n-delta (first w) (first deltas) (first a) dafn)]
		(recur (rest w) (cons (matrix (rest d)) deltas) (rest a))))))
		
	
(defn calc-gradients
 [deltas activations] 
 (map #( mmult %1 (trans %2)) deltas activations))
	



(defn train-network
[weights input target]
(let [
	 activations (propagate-activations input weights)
	 deltas (compute-hidden-layer-deltas (compute-output-deltas (last activations) target dactivation-fn) weights activations dactivation-fn)
	 gradients (calc-gradients deltas activations)]
	 gradients))

;computes gradients for all training samples...network weights get adjusted after each epoch.
;note that I am not not dividing gradient by number of training examples....which could lead to dramatic over-shooting as the network improves!!!
(defn train-data
[weights sample-data learning-rate]
(loop [
	    s sample-data
	    g (new-gradient-matrix weights)
	    ]
	    (if (empty? s)
	    (map minus weights (map #(mult %1 learning-rate) g))
	    (recur (rest s) (map plus g (train-network weights (matrix (first (first s))) (matrix (second (first s)))))))))
	    
	    


(defn train-epochs 
[weights sample-data cycles learning-rate]
(if (zero? cycles) weights
(recur (train-data weights sample-data learning-rate) sample-data (dec cycles) learning-rate)))


(defn fp
[weights input]
(last (propagate-activations (matrix input) weights))
)




(defn build-network [size learning-rate cycles data]
(train-epochs (initialize-weights size 2) data cycles learning-rate))





;;example usage:
;;(def nn (build-network [2 3 1] 0.1 200 sample-data))
;;(fp nn [1 1])


;;todo

;add MSE to jump out of loop.
;should I be computing gradient based on all data in dataset? i.e., not adjusting weights based on a single training example?  does it matter?  stochastic gradient descent?


