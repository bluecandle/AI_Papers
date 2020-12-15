# EfficientDet: Scalable and Efficient Object Detection

Property: DL, Object_Detection
Status: 1회독완료

[정리 원본](https://www.notion.so/EfficientDet-Scalable-and-Efficient-Object-Detection-b868a111f9bb426883849f91c6a576db)

### 논문

---

[https://arxiv.org/abs/1911.09070](https://arxiv.org/abs/1911.09070)

### code

---

[https://github.com/google/automl/tree/](https://github.com/google/automl/tree/) master/efficientdet

### 이 논문의 한 줄

---

EfficientNet , BiFPN, model scaling 을 통해, 효율적&효과적으로 feature fusing 이 이루어지도록 하여 새로운 detector family 구조를 제안. 

### keywords

---

- Object Detection
- Efficiency
- BiFPN

### 내용정리

---

- 

### 문구

---

- weighted bi-directional feature pyramid network (BiFPN)
- a compound scaling method that uniformly scales the resolution, depth, and width for all backbone, feature network, and box/class pre- diction networks at the same time.
- 4x – 9x smaller and using 13x – 42x fewer FLOPs than previous detectors.
- Given these real-world resource constraints, model efficiency becomes increasingly important for object detection.
    - 흠.... 그치 아무리 SOTA accuracy 가 나오더라도, 모델이 너무 크고 계산량이 많으면, 실제 상황에서 사용하는데 무리가 있다!
- This paper aims to tackle this problem by systematically studying various design choices of detector architectures.
    - Problem: Is it possible to build a scalable detection architecture with both higher accuracy and better efficiency across a wide spectrum of resource constraints (e.g., from 3B to 300B FLOPs)

### challenge 1: Efficient multi-scale feature fusion

기존 FPN 에서 각 layer 의 feature 에 의한 정보를 구분없이 그냥 합치기만 했었는데... 생각해보니 그렇게 하지 않고, 구분 있이 해야하는거 아니냐는 주장!

since these different input features are at different resolutions, we observe they usually contribute to the fused output feature unequally.

⇒ 이 문제를 해결하기 위한 해결책이 BiFPN

⇒ which introduces **learnable weights** to learn the **importance of different input features**, while repeatedly applying top-down and bottom-up multi-scale feature fusion.

### challenge 2: model scaling

we observe that scaling up feature network and box/class prediction network is also critical when taking into account both accuracy and efficiency.

A compound scaling method for object detectors, which jointly scales up the resolution/depth/width for all backbone, feature network, box/class prediction network.

그래서... backbone으로 EfficientNet을 사용하고, BiFPN , compound scaling과 합쳐서 새로운 object detector 를 고안해내었다. 

Combining EfficientNet backbones with our proposed BiFPN and compound scaling, we have developed a new family of object detectors, named EfficientDet, which consistently achieves better accuracy with much fewer parameters and FLOPs than previous object detectors.

### BiFPN (bidirectional feature pyramid network)

our goal is to find a transformation f that can effectively aggregate different features and output a list of new features: $\vec P^{out} = f(\vec P^{in})$.

우선 일반 FPN 은 어떻게 되더라??

The conventional FPN aggregates multi-scale features in a top-down manner:

![EfficientDet%20Scalable%20and%20Efficient%20Object%20Detecti%2001c6aabcc7404e4c9f8829f2cebc8b9c/Untitled.png](EfficientDet%20Scalable%20and%20Efficient%20Object%20Detecti%2001c6aabcc7404e4c9f8829f2cebc8b9c/Untitled.png)

where Resize is usually an upsampling or downsampling op for resolution matching, and Conv is usually a convolutional op for feature processing.

⇒ Conventional top-down FPN is inherently limited by the one-way information flow.

Cross-Scale Conncetions

NAS-FPN 은 필요 연산량이 어마어마하게 많고,찾아낸 network 도 irregular 하고 이해, 변형이 어렵.

PANET 이 꽤 좋은 성능을 보여줬다는 점에서 착안하여, optimization 을 해보려고 했다!

(1) input 이 하나인 node 를 제거

why? feature fusion 없이 input 하나만 들어오는 node 라면, feature fusing 을 목표로 하는 network 에서 기여하는 바가 적을 것이라는 가정 하에! 오호...

(2) original input 을 output node 에 skip connection 하는 extra edge 추가

⇒ cost 를 많이 들이지 않으면서 feature 를 많이 fuse 하기 위해!

(3) we treat each bidirectional (top-down & bottom-up) path as one feature network layer, and repeat the same layer multiple times to enable more high-level feature fusion

⇒ 흠...이건 잘 모르겠네

⇒ 일단 서로 다른 layer 에 있는 node들간에 위 → 아래 , 아래 → 위로 움직이는걸 하나의 feature network layer 로 간주한다는 말인데...

**Weighted Feature Fusion**

since different input features are at different resolutions, they usually contribute to the output feature unequally

⇒ we propose to add an additional weight for each input, and let the network to learn the importance of each input feature

[1] Unbounded fusion

$O = \sum_i w_i · I_i$

$w_i$ : learnable weight

We find a scale can achieve comparable accuracy to other approaches with minimal computational costs. However, since the scalar weight is unbounded, it could potentially cause training instability. Therefore, we resort to weight normalization to bound the value range of each weight.

⇒ weight normalization을 하자는 생각에서 softmax 를 떠올릴 수 있음!

[2] Softmax-based fusion

$O = \sum_i {e^{w_i} \over \sum_j e^{w_j}} \cdot I_i$

각각 모든 weight 에 softmax 를 적용하여 모든 weight 가 range [0,1] 의 probability 가 되도록 하여, 각 input 의 importance 를 대변하도록 한다는 생각임.

⇒ 하지만, 이렇게 하면 GPU hardware 가 느려진다는 문제가 발견되었음.

[3] Fast normalized fusion

그래서, 빠른 normalize fusion 방법을 도입함.

$O = \sum_i {w_i \over \epsilon + \sum_j w_j} \cdot I_i$ .

$w_i >= 0$  가 각각 $w_i$ 이후에 Relu 를 적용하기 때문에 보장됨.

그리고 $\epsilon$ 값은 0.0001임.

결국 softmax 와 같게 range [0,1] 이지만, softmax 과정이 없기 때문에

⇒ but since there is no softmax operation here, it is much more efficient. 

Our final BiFPN integrates both the bidirectional cross-scale connections and the fast normalized fusion.

[예시]

![EfficientDet%20Scalable%20and%20Efficient%20Object%20Detecti%2001c6aabcc7404e4c9f8829f2cebc8b9c/Untitled%201.png](EfficientDet%20Scalable%20and%20Efficient%20Object%20Detecti%2001c6aabcc7404e4c9f8829f2cebc8b9c/Untitled%201.png)

Notably, to further improve the efficiency, we use depthwise separable convolution for feature fusion and add batch normalization and activation after each convolution.

### EfficientDet : 그럼 이제 각 요소들 설명했으니, 어떻게 합쳐지는지 설명할게

**Architecture**

we will discuss the network architecture and a new compound scaling method for EfficientDet.

![EfficientDet%20Scalable%20and%20Efficient%20Object%20Detecti%2001c6aabcc7404e4c9f8829f2cebc8b9c/Untitled%202.png](EfficientDet%20Scalable%20and%20Efficient%20Object%20Detecti%2001c6aabcc7404e4c9f8829f2cebc8b9c/Untitled%202.png)

We employ ImageNet pre-trained EfficientNets as the backbone network.

Our proposed BiFPN serves as the feature network, which takes level 3-7 features {P3, P4, P5, P6, P7} from the backbone network and repeatedly applies top-down and bottom-up bidirectional feature fusion.

the class and box network weights are shared across all levels of features

![EfficientDet%20Scalable%20and%20Efficient%20Object%20Detecti%2001c6aabcc7404e4c9f8829f2cebc8b9c/Untitled%203.png](EfficientDet%20Scalable%20and%20Efficient%20Object%20Detecti%2001c6aabcc7404e4c9f8829f2cebc8b9c/Untitled%203.png)

**Compound Scaling**

develop a family of models that can meet a wide spectrum of resource constraints:

A key challenge here is how to scale up a baseline EfficientDet model.

기존 연구에서는 baseline detector 를 scale up 하기 위해:

bigger backbone network 를 차용하거나, 더 큰 input image 를 사용하거나, 더 많은 FPN layer 를 쌓는다거나 하는 방법으로 진행해왔다.

⇒ 근데, image classification에서 network 의 width, depth, and input resolution 의 차원을 모두 증가시키는 방법으로 성능 향상을 일으킨 것에 영감을 얻어서...

⇒ $\phi$ 라는 compound coefficient 를 사용하여 backbone, BiFPN, class/box network 그리고 resolution 의 모든 dimension을 jointly scaling up 하는 방법을 제안.

(grid search 는 너무 말도 안되게 computationally expensive 하다 object detection 에서는! 고로...)

⇒ we use a heuristic-based scaling approach, but still follow the main idea of jointly scaling up all dimensions.

backbone network, BiFPN network, Box/class prediction network 는 뭐 나중에 필요할 때 보기!

Input image Resolution은...

so we linearly increase resolutions using equation: $R_{input} = 512 + φ · 128$

Notably, our compound scaling is heuristic-based and might not be optimal, but we will show that this simple scaling method can significantly improve efficiency than other single-dimension scaling methods.

### 알고리즘 설명

---

### 기타

---