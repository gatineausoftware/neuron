(ns neuron.monitor
 (:use neuron.batch)
 (:use [incanter core stats charts])
)


(def p (xy-plot))


(defn monitor [n test-data afn sample-size]
(view p)
(loop [i 0]
(Thread/sleep 5000)
(if (nil? @n) nil 
(let [x (check-progress @n test-data afn sample-size)]
(println i ": " (format "%.2f" (double x)))
(add-points p i x)))
(recur (inc i))))


(defn get-range [x]
[(apply min x) (apply max x)]
)

(defn test-x [x]
[1 2]
)


(defn init-frame []
(let [frame (java.awt.Frame.)]
  (.setVisible frame true)
  (.setSize frame (java.awt.Dimension. 320 320))
  (.getGraphics frame)))




(defn scale-feature [w]
  (vec  (map #(if (< 0 %) 255 0) w))
)

;just linear scale from 0 to 255


(defn map->range [start end min max x]
  (Math/round (+ start (*  (/ (- end start) (- max min)) (- x min))))
)


(defn scale-feature2 [w]
(vec (map (partial map->range 0 255 (apply min w) (apply max w)) w)))


(defn visualize-neuron [nn layer neuron gfx]
  (let [w (-> (nth @nn (- layer 1))
               (nth neuron)) 
        s (scale-feature2 w)]
     (for [x (range 0 32) y (range 0 32)]
      (let [c (s (+ (* y 32) x))]
        (.setColor gfx (java.awt.Color. c c c ))
        (.fillRect gfx (* x 10) (* y 10) 10 10)))))



;from REPL: 

;(future  (def nn (train-mf 0.0005 4000 40 40 digits-train digits-test)))
;
;(future-cancel m)

