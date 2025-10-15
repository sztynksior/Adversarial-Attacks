# Adversarial-Attacks
This small example project demonstrates how to **train a perturbation mask** used in an adversarial attack against a road-signs image classification model.

## About Adversarial Attacks
Image-recognition neural networks are vulnerable to adversarial attacks. It is possible to cause a trained model to misclassify an image as a chosen target class by adding a trained perturbation (a "mask") to the image. The perturbation is usually constrained so the changes are small and often hard for humans to notice, yet large enough to mislead the model.

## Results
### Model performence before attack
<img width="237" height="207" alt="pedestrians" src="https://github.com/user-attachments/assets/604d489c-5f03-4563-a372-b6ee963ead59" />

Initial prediction: Pedestrians

![stop](https://github.com/user-attachments/assets/a852c145-bfdd-449e-a951-b44b7b0b0713)

Initial prediction: Stop

<img width="150" height="137" alt="yield" src="https://github.com/user-attachments/assets/674607a2-05b3-43c9-9e92-6cde0f26ccf3" />

Initial prediction: Yield

### Model performence after attack
Target class: Pedestrians, Oryginal class: Stop

Step: 158, Prediction: Pedestrians, Loss: -23.124422073364258, Target Class Probability: 0.40884917974472046

<img width="590" height="222" alt="image" src="https://github.com/user-attachments/assets/51773000-b34b-45f9-a4bf-303f73b3e359" />

Target class 27 reached at step 158.

---
Target class: Yield, Oryginal class: Pedestrians

Step: 73, Prediction: Yield, Loss: -63.81779098510742, Target Class Probability: 0.5005481243133545

<img width="590" height="222" alt="image" src="https://github.com/user-attachments/assets/2bdb744d-9e5e-4141-93eb-daebeb1575da" />

Target class 13 reached at step 73.

---

Target class: Pedestrians, Oryginal class: Yield

Step: 57, Prediction: Pedestrians, Loss: -48.407772064208984, Target Class Probability: 0.5284264087677002

<img width="590" height="222" alt="image" src="https://github.com/user-attachments/assets/ec8437e8-9b82-4dfe-95ee-864dcf78b88c" />

Target class 27 reached at step 57.

---

Target class: Stop, Oryginal class: Yield

Step: 425, Prediction: Stop, Loss: -407.3600158691406, Target Class Probability: 0.5093487501144409

<img width="590" height="222" alt="image" src="https://github.com/user-attachments/assets/2dc81c54-217a-4ba0-b85c-642c5c6f3a53" />

Target class 14 reached at step 425.

## Conclusion


