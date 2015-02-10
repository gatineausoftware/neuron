(ns neuron.digits)
(require '[clojure.java.io :as io])


(defmacro dbg[x] `(let [x# ~x] (println "dbg:" '~x "=" x#) x#))

(def decimal->binary {0 [1 0 0 0 0 0 0 0 0 0 ]
                      1 [0 1 0 0 0 0 0 0 0 0 ]
                      2 [0 0 1 0 0 0 0 0 0 0 ]
                      3 [0 0 0 1 0 0 0 0 0 0 ]
                      4 [0 0 0 0 1 0 0 0 0 0 ]
                      5 [0 0 0 0 0 1 0 0 0 0 ]
                      6 [0 0 0 0 0 0 1 0 0 0 ]
                      7 [0 0 0 0 0 0 0 1 0 0 ]
                      8 [0 0 0 0 0 0 0 0 1 0 ]
                      9 [0 0 0 0 0 0 0 0 0 1]})

(defn string->bitmap [s] (vec (map #(Character/getNumericValue %) (filter #(and (not= % \return) (not= % \newline)) s))))

(defn get-labels [fileList] (map #(->> % .getName first Character/getNumericValue) fileList))

(defn get-inputs [file]
 (let [output (->> file .getName first Character/getNumericValue decimal->binary)
       input (string->bitmap (slurp file))]
  [input output])
 )

;(def sample-data (map get-inputs (.listFiles (io/file "trainingDigits"))))






(defn get-inputs2 [directory]
(reduce (fn [data inputFile]
         (let [output (->> inputFile .getName first Character/getNumericValue)
               input (string->bitmap (slurp inputFile))]
          (assoc data output (conj (data output) input))
          ))
        {} (.listFiles (io/file directory))))


(def data (get-inputs2 "trainingDigits"))









;how to organize epochs?  also may need MSE?





;conj col new-element0