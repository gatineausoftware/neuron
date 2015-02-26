(ns neuron.batch
  (:use incanter.core)
  (:use incanter.stats)
 
  
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
; fix hack...don't need the afn'..bad clojure...I should get rid of afn' here and in quadratic and call train network with a partial when using quadratic
; or hardcode....
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
 
 
(defn fp
    [weights input afn]
    (last (propagate-activations (matrix input) weights afn)))



 
 
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
		
	



(defn check-prediction [output expected-output]
(if (= (first (apply max-key second (map-indexed vector output))) (.indexOf expected-output 1)) 1 0))


; is there a way to do this as a single ->> ?
(defn check-progress 
[weights test-data afn sample-size]
(let [batch (get-batch test-data sample-size)
      activations (map #(fp weights (first %) afn) batch)
      expected-output (map second batch)
      ]
      (->> (map check-prediction activations expected-output)
           (reduce + 0)
            ((partial #(div %2 %1) sample-size))
            )))


(defn spot-check [weights test-data afn size]
(let [batch (get-batch test-data size)
      activations (map #(fp weights (first %) afn) batch)
      expected-output (map second batch)
      accurate (map check-prediction activations expected-output)]
      (map vector activations expected-output accurate))) 






; afn: activation function
; afn': derivative of activatino functin
; odf: output delta function (based on cross-entropy + sigmoid, or quadritic)

(defn compute-gradient
[weights input target afn afn' odf]
(let   [ activations (propagate-activations input weights afn)
	 deltas (compute-hidden-layer-deltas (odf (last activations) target afn') weights activations afn')]
       (map #(mmult %1 (trans %2)) deltas activations)))



(defn train-network
[weights sample-data learning-rate afn afn' odf]
    (loop [
	    s sample-data
	    g (new-gradient-matrix weights)
	    ]
	    (if (empty? s)
	    (map minus weights (map #(mult %1 learning-rate) g))
	    (recur (rest s) (map plus g (compute-gradient weights (matrix (first (first s))) (matrix (second (first s))) afn afn' odf))))))
	    
	    


;;learning rate is _not_ divided by batch size and passed to train-data
;;also I appear not to be using the batch, but using the whole sample data
(defn train-epochs-batch 
[weights sample-data cycles batch-size learning-rate afn afn' odf]
(if (zero? cycles) weights
    (let [batch (get-batch sample-data batch-size)]
      (recur (train-network weights batch learning-rate afn afn' odf) sample-data (dec cycles) batch-size learning-rate afn afn' odf))))



(defn train-epochs
  [weights sample-data cycles learning-rate afn afn' odf]
  (if (zero? cycles) weights
      (recur (train-network weights sample-data learning-rate afn afn' odf) sample-data (dec cycles) learning-rate afn afn' odf)))



(defn train-with-progress
  [weights training-data test-data cycles batch-size sample-size learning-rate afn afn' odf]
  (loop [w weights
         c cycles]
  (if (zero? c) w
      (do
        (let [batch (get-batch training-data batch-size)
              new-weights (train-network w batch learning-rate afn afn' odf)]
          (println (- cycles c)  ": " (format "%.2f" (double (check-progress new-weights test-data afn sample-size)))) 
        (recur new-weights (dec c))
      )))))



(defn build-network-batch [size learning-rate cycles batch-size afn afn' odf data]
(train-epochs-batch (initialize-weights size 2) data cycles batch-size learning-rate afn afn' odf))


(defn build-network [size learning-rate cycles afn afn' odf data]
(train-epochs (initialize-weights size 1) data cycles learning-rate afn afn' odf))

(defn train-mf [learning-rate cycles batch-size sample-size digits-train digits-test]
(train-with-progress (initialize-weights [1024 30 10] 2) digits-train digits-test cycles batch-size sample-size learning-rate sigmoid sigmoid' cross-entropy-output-deltas)
)





