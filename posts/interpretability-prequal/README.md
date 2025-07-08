# High-level overview of Distill/Anthropic’s interpretability thread's first article


source: https://distill.pub/2020/circuits/zoom-in/

A review of Zoom In:

Three claims:  
1) Features correspond to directions.  
   - ex: curve detectors [1], high- and low-frequency detectors [2], pose-invariant body-feature detectors  

2) Features are connected by weights “forming circuits”

   early curves → curves → complex shapes  
   (layer 1)     (layer 2)     (layer 3)

   The first arrow represents a circuit:

   +------------+      +-------------------+      +------------+  
   |            |      |                   |      |            |  
   |  Layer 1   | ---> |  Weights between  | ---> |  Layer 2   |  
   | (features) |      |   (+, –, values)  |      | (features) |  
   |            |      |                   |      |            |  
   +------------+      +-------------------+      +------------+  

   + Weights are neurons that fired when the curve was detected.  
   - Weights represent the rest of the space without that curve component.

                  +------------------------+       +------------------------+  
                  |   Oriented Fur (3b)    |       |   Oriented Fur (3b)    |  
                  |   (Left-Oriented)      |       |   (Right-Oriented)     |  
                  +------------------------+       +------------------------+  
                             |                               |  
                             v                               v  
                  +------------------------+       +------------------------+  
                  |  Oriented Heads (4a)   |       |  Oriented Heads (4a)   |  
                  +------------------------+       +------------------------+  
                             |                               |  
                             +------------+  +--------------+  
                                          v  v  
                                 Union over left and right  
                                          |  
                                          v  
                          +-------------------------------+  
                          | Orientation-Invariant Head    |  
                          |           (4b)                |  
                          +-------------------------------+  
                                          |  
                                          v  
                          +-------------------------------+  
                          | Orientation-Invariant         |  
                          |     Head + Neck (4c)          |  
                          +-------------------------------+

   - Polysemantic neurons respond to multiple features, e.g., ‘car’ + ‘cat’.  
   - In biology, circuit motifs are the building blocks of decision making.  
   - Orientation-invariant neurons respond to dogs facing either way (as shown above).  
   - These pure detectors spread features across many polysemantic neurons; this phenomenon is known as superposition.  
   - Superposition allows the network to save neurons for later and store more features in fewer neurons.

3) Universality.  
   - The first layer of vision models learns Gabor filters.  
   - Different networks develop correlated neurons.  
   - There is not enough study on this, but ideally different neural networks can develop highly correlated neurons and learn similar hidden-layer representations.  
   - Low-level features like [1], [2] can be observed across AlexNet, Inception v1, and VGG.  
