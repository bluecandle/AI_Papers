# SORT

Date: Dec 10, 2020
Property: DL, Tracker
Status: 1회독완료

[정리원본](https://www.notion.so/SORT-cde9cc89508c4ce48a3efeb99f36563f)

### 논문

---

[https://arxiv.org/pdf/1602.00763.pdf](https://arxiv.org/pdf/1602.00763.pdf)

### 코드

---

[https://github.com/abewley/sort](https://github.com/abewley/sort)

### 이 논문의 한 줄

---

Kalman Filter 와 Hungarian Algorithm 을 활용하여 Estimation, Association 을 반복 수행하는 것으로 target(사람) 을 tracking 한다.

### keywords

---

- Computer Vision
- Multiple Object Tracking (MOT)
- Detection
- Data Association

### 내용정리

---

[https://blog.jesse.kim/post/157](https://blog.jesse.kim/post/157)

즉 SORT는 프레임별로 estimation - association - correction을 반복하며 인식된 물체들을 추적한다.

---

### 문구

---

- Despite only using a rudimentary combination of familiar techniques such as the Kalman Filter and Hungarian algorithm for the tracking components, this approach achieves an accuracy comparable to state-of-the-art online trackers.
- this work is primarily targeted towards online tracking where only detections from the previous and the current frame are presented to the tracker.
- data association problem where the aim is to associate detections across frames in a video sequence.
- this work explores how simple MOT can be and how well it can perform.

- Occam's Razor ... Simple is the best!
    - appearance features beyond the detection component are ignored in tracking and only the bounding box position and size are used for both motion estimation and data association.
    - issues regarding short-term and long-term occlusion are also ignored, as they occur very rarely and ...
        - 오호... occlusion 에 대해 신경쓰지 않았다고 하는데, 그렇다면 occlusion 문제를 해결하기 위해서는 SORT 를 사용하지 않고 다른 Tracker 를 사용하거나, 혹은 Detector 에서 성능의 개선이 더 이루어져야 한다는 의미겠지?
- incorporating complexity in the form of object re-identification adds significant overhead into the tracking framework.
    - 흠... 이 논문 저자는 Re-ID 에 대해 부정적인 견해를 갖고있군. Re-ID 를 하게되면 Overhead 때문에 실제 상황에서 쓰이기 어렵다는 논리인거같은데, Re-ID 관련 논문 읽어보고 생각해볼 필요 있겠다.
- This work instead focuses on efficient and reliable handling of the common frame-to-frame associations.
    - corner case 그런거 신경 안쓰고, 그냥 일단 제일 자명한 것에만 집중하겠다는 의도!
- A pragmatic tracking approach based on the **Kalman filter** and the **Hungarian algorithm** is presented and evaluated on a recent MOT benchmark.
- we utilise the Faster Region CNN (FrRCNN) detec- tion framework ⇒ ㅇㅎ... 하긴 2016년에 나왔으니 Detector로 Faster R-CNN 을 썼구나.
- **detection quality has a significant impact on tracking performance** when comparing the FrRCNN detections to ACF detections.
- **the best detector (FrRCNN(VGG16)) leads to the best tracking accuracy** for both MDP and the proposed method.
    - 결국 Detector 성능이 좋아야 Tracker 성능이 좋아질 수 있다! ⇒ 흠... 그렇다면 Detector 성능을 더 올리는 것에 집중해볼 필요가 있다는 생각!

***Estimation Model***

- We approximate the inter-frame displacements of each object with a linear constant velocity model which is independent of other objects and camera motion. The state of each target is modelled as:
    - $x = [u, v, s, r, u^˙, v^˙, s^˙]^T$
    - 각 객체의 프레임간 차이 (불일치) 를 linear constant velocity model 로 근사(approximate)한다. 그리고 이 linear constant velocity 모델은 다른 객체나 카메라 모션에 대해 독립적임.
    - 각 타겟의 state 는 위의 식으로 Model 된다.
    - u 와 v 는 객체의 수평, 수직 pixel location.
    - s 와 r 은 객체 Bbox 의 scale(영역 크기) 와 aspect ratio 를 나타냄.
        - aspect ratio 는 constant 로 간주된다??
- When a detection is associated to a target, the detected **bounding box is used to update the target state** where the velocity components are solved optimally via a Kalman filter framework.
- If no detection is associated to the target, its state is simply predicted without correction using the linear velocity model.
    - target과 detection이 association 이 되면, detect 된 bbox 가 tartget 의 state 를 업데이트 하는데 사용된다. ( velocity 값이 Kalman filter 를 통해 구해짐)
    - target에 detection association 이 되지 않으면, 그냥 단순히 linear velocity model 을 통해 state 가 예측된다.

***Data Association***

- 각 detection 과 타겟들에 대한 모든 predicted Bbox 간의 IOU Distance를 계산한 행렬을 구하고,  헝가리안 알고리즘을 이용하여 assignment 가 이루어진다.
    - 단, $IOU_{min}$ 값이 있어서 그 값보다 작으면 ㄴㄴ
    - We found that the IOU distance of the bounding boxes **implicitly handles short term occlusion caused by passing targets**.
        - 오호... Bbox 간의 IOU distance 가 짧은 순간의 occlusion 을 적절히 처리해준다.

***Creation and Deletion of Track Identities***

- we consider any detection with an overlap less than $IOU_{min}$ to signify the existence of an untracked object.
- The tracker is initialised using the geometry of the bounding box with the velocity set to zero. ( 그치 처음에 target 생성될 때는 state의 velocity 값이 0으로 되어있겠지 )
    - Since the velocity is unobserved at this point the **covariance of the velocity** component is initialised with large values, reflecting this uncertainty
        - 처음 target 이 생성될 때 state 중 velocity 의 uncertainty 를 처리하기 위해 velocity 의 covariance 를 높게 지정해준다!
    - the new tracker then undergoes a probationary period where the target needs to be associated with detections to accumulate enough evidence in order to prevent tracking of false positives.
        - 오호...그리고 처음에 target 이 감지된 이후, false positive 의 결과를 내지 않기 위해 임시 단계를 갖게한다고 한다.
- Tracks are terminated if they are not detected for $T_{Lost}$ frames
    - 그리고 연구에서는 $T_{Lost}$  값을 1로 두었는데
        - (1) linear constant velocity model 이 실제 현상을 잘 반영하지 못하는 모델이고
        - (2) Re-ID 는 이 연구의 범주에서 벗어나기 때문, 그냥 좀 끊겨도 tracking 만 되면 된다는 생각!
        - 그리고 빠르게 지워내는 것이 효율성이 좋다고 한다! Target 이 다시 등장하면 어차피 다시 새로운 identity 로 tracking 이 다시 재개될 것이라는 점도 있고 !

resurgence _ (활동의)재기

### 알고리즘 설명

---

### 기타

---

원리가 간단 하면서도 연산량에 큰 영향을 주지 않는 트래킹 알고리즘 이며, opensource이기 때문에 누구나 사용 가능한 코드입니다.

요즘에 영상처리에서 Yolo나 SSD 같은 Object Detection알고리즘 결과물에 붙여서 사용 하면 더 Smooth한 결과물을 보게 될겁니다.
